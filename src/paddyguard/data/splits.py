"""Grouped, stratified splits (M2). Fixed seed, never re-split.

S  : train / val / test, grouped by duplicate group, stratified by class.
T1 : seg_train / seg_val / seg_test (after dedup), stratified by class.
T2 : field background -> cal_pool / test; white background -> zero_shot (no calibration pool).
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Mapping
from typing import Any

import pandas as pd
import yaml
from sklearn.model_selection import StratifiedGroupKFold

from paddyguard.data.dataset_card import _table
from paddyguard.data.dedup import DROPPED, DROPPED_DUPLICATE, is_representative
from paddyguard.data.manifest import read_manifest, suffixed, write_manifest
from paddyguard.utils import io, paths
from paddyguard.utils.logging import get_logger
from paddyguard.utils.seed import set_seed

ZERO_SHOT = "zero_shot"
FOLD_TOLERANCE = 1e-6


def fold_counts(ratios: Mapping[str, float], n_folds: int) -> dict[str, int]:
    """How many of the n_folds folds each split gets; ratios must fit the folds exactly."""
    counts = {name: round(r * n_folds) for name, r in ratios.items()}
    exact = all(abs(r * n_folds - counts[name]) < FOLD_TOLERANCE for name, r in ratios.items())
    if not exact or sum(counts.values()) != n_folds:
        raise ValueError(f"ratios {dict(ratios)} do not divide into {n_folds} whole folds")
    return counts


def group_split(
    df: pd.DataFrame, ratios: Mapping[str, float], n_folds: int, seed: int
) -> pd.DataFrame:
    """Assign `split` with StratifiedGroupKFold: whole groups, class balance kept per split."""
    counts = fold_counts(ratios, n_folds)
    fold_to_split = [name for name, k in counts.items() for _ in range(k)]
    out = df.copy()
    kfold = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    assignment = pd.Series(index=out.index, dtype=object)
    for fold, (_, test_idx) in enumerate(kfold.split(out, out["label_raw"], out["group_id"])):
        assignment.iloc[test_idx] = fold_to_split[fold]
    out["split"] = assignment
    return out


def ensure_not_split(df: pd.DataFrame) -> None:
    """Splits are fixed once made: refuse if any kept image already has one."""
    assigned = df["split"].dropna()
    if (~assigned.isin(DROPPED)).any():
        raise RuntimeError("splits already assigned; they are fixed and must not be redone")


def _assign(df: pd.DataFrame, mask: pd.Series, key: str, cfg: Mapping[str, Any]) -> None:
    """Group-split the rows in `mask` with the ratios and folds configured under `key`."""
    if mask.any():
        ratios, folds = cfg["splits"][key], cfg["split_folds"][key.split("_")[0]]
        df.loc[mask, "split"] = group_split(df[mask], ratios, folds, cfg["split_seed"])["split"]


def assign_all_splits(df: pd.DataFrame, cfg: Mapping[str, Any]) -> pd.DataFrame:
    """Fill `split` for every kept image of S, T1 and T2 (dropped rows keep their marker)."""
    ensure_not_split(df)
    out = df.copy()
    open_rows = out["split"].isna()
    _assign(out, open_rows & (out["dataset"] == "S"), "S", cfg)
    _assign(out, open_rows & (out["dataset"] == "T1"), "T1", cfg)
    extra_test_members = (
        (out["dataset"] == "T1") & (out["split"] == "seg_test") & ~is_representative(out)
    )
    out.loc[extra_test_members, "split"] = DROPPED_DUPLICATE  # evaluation keeps one image per group
    t2 = open_rows & (out["dataset"] == "T2")
    _assign(out, t2 & (out["background"] == "field"), "T2", cfg)
    out.loc[t2 & (out["background"] == "white"), "split"] = ZERO_SHOT
    return out


def allowed_cal_sizes(pool: int, candidates: list[int], max_fraction: float) -> list[int]:
    """Calibration sizes that use at most max_fraction of the pool, so draws stay different."""
    return [n for n in candidates if n <= max_fraction * pool]


def shared_raw_labels(labelmap: Mapping[str, Any], dataset: str, shared_key: str) -> set[str]:
    """Raw labels of `dataset` that map to a class shared with Paddy Doctor."""
    shared = set(labelmap["shared"][shared_key])
    return {raw for raw, canon in labelmap["map"][dataset].items() if canon in shared}


def _counts_table(df: pd.DataFrame, title: str) -> str:
    pivot = df.groupby(["label_raw", "split"]).size().unstack(fill_value=0)
    pivot["total"] = pivot.sum(axis=1)
    rows = [[label, *r] for label, r in zip(pivot.index, pivot.to_numpy().tolist(), strict=True)]
    return f"### {title}\n\n" + _table(["class folder", *pivot.columns], rows)


def _t2_section(df: pd.DataFrame, cfg: Mapping[str, Any], labelmap: Mapping[str, Any]) -> str:
    field = df[(df["dataset"] == "T2") & (df["background"] == "field")]
    shared = shared_raw_labels(labelmap, "T2", "S_T2")
    pool = field[(field["split"] == "cal_pool") & field["label_raw"].isin(shared)]
    test = field[(field["split"] == "test") & field["label_raw"].isin(shared)]
    low_n = cfg["low_n_test_per_class"]
    rows = [
        [label, int((pool["label_raw"] == label).sum()), n, "LOW-N" if n < low_n else ""]
        for label, n in test["label_raw"].value_counts().sort_index().items()
    ]
    sizes = allowed_cal_sizes(
        len(pool), cfg["cal_candidates"]["T2"], cfg["t2_cal_max_fraction_of_pool"]
    )
    return (
        "### T2 (field background), shared classes only\n\n"
        + _table(["class folder", "cal_pool images", "test images", f"flag (< {low_n} test)"], rows)
        + f"\n\nCalibration pool (shared classes): **{len(pool)}** images, test: **{len(test)}**. "
        f"Allowed calibration sizes (<= {cfg['t2_cal_max_fraction_of_pool']:.0%} of the pool): "
        f"**{sizes}**. Classes flagged LOW-N are kept; report their numbers with bootstrap CIs and "
        "mark them low-n in every T2 table."
    )


def render_report(df: pd.DataFrame, cfg: Mapping[str, Any], labelmap: Mapping[str, Any]) -> str:
    kept = df[~df["split"].isin(DROPPED)]
    parts = [
        "# Split report",
        "Generated by `python -m paddyguard.data.splits`. Do not edit by hand.",
    ]
    parts.append(
        f"Seed {cfg['split_seed']}, whole duplicate groups per split, stratified by class folder."
    )
    dropped = (
        df[df["split"].isin(DROPPED)].groupby(["dataset", "split"]).size().reset_index(name="n")
    )
    parts.append(
        "## Dropped images\n\n" + _table(["dataset", "reason", "images"], dropped.values.tolist())
    )
    for ds, title in [
        ("S", "S (Paddy Doctor)"),
        ("T1", "T1 (Sethy)"),
        ("T2", "T2 (Dhan-Shomadhan)"),
    ]:
        parts.append(_counts_table(kept[kept["dataset"] == ds], title))
    parts.append(_t2_section(kept, cfg, labelmap))
    t1_pool = int((kept[kept["dataset"] == "T1"]["split"] != "seg_test").sum())
    sizes = allowed_cal_sizes(t1_pool, cfg["cal_candidates"]["T1"], 1.0)
    parts.append(
        f"### T1 calibration pool\n\nT1 images outside seg_test: **{t1_pool}**; sizes: {sizes}."
    )
    return "\n\n".join(parts) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(paths.configs_dir() / "data.yaml"))
    parser.add_argument("--debug", action="store_true", help="work on manifest_debug.csv")
    args = parser.parse_args(argv)
    cfg = io.load_yaml(args.config)
    rel = io.load_yaml(paths.configs_dir() / "reliability.yaml")
    cfg["cal_candidates"] = rel["cal_sizes"]
    labelmap = yaml.safe_load((paths.configs_dir() / "labelmap.yaml").read_text())
    set_seed(cfg["seed"])
    log = get_logger(
        "paddyguard.splits", io.new_run_dir("data", cfg["seed"], config=cfg) / "log.txt"
    )
    suffix = "_debug" if args.debug else ""
    manifest_path = suffixed(paths.data_root() / cfg["manifest_path"], suffix)
    df = read_manifest(manifest_path)
    if df["group_id"].isna().any():
        log.error("group_id is empty: run `python -m paddyguard.data.dedup` first")
        return 1
    try:
        out = assign_all_splits(df, cfg)
    except RuntimeError as err:
        log.error("%s (%s)", err, manifest_path)
        return 1
    write_manifest(out, manifest_path)
    report = suffixed(paths.reports_root() / "splits_report.md", suffix)
    report.write_text(render_report(out, cfg, labelmap))
    log.info("wrote %s and %s", manifest_path, report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
