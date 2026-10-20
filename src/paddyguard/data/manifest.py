"""Build data/manifests/manifest.csv: one row per image, for all datasets (M1)."""

from __future__ import annotations

import argparse
import random
import re
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image

from paddyguard.data.download import Check, list_images, verify_layout
from paddyguard.utils import io, paths
from paddyguard.utils.logging import get_logger
from paddyguard.utils.seed import set_seed

MANIFEST_COLUMNS = [
    "image_id",
    "dataset",
    "path",
    "label_raw",
    "label",
    "variety",
    "age_days",
    "width",
    "height",
    "background",
    "mask_path",
    "phash",
    "group_id",
    "split",
]
MAX_LISTED = 10  # how many offending file names a check detail lists
_INFO_LABEL = re.compile(r"^(?P<cls>.*?)\s*\((?P<bg>[^)]*?)\s*background\)\s*$", re.IGNORECASE)


def _rel(path: Path, data_root: Path) -> str:
    return path.relative_to(data_root).as_posix()


def _listed(items: list[str]) -> str:
    shown = ", ".join(sorted(items)[:MAX_LISTED])
    return shown + (f" (+{len(items) - MAX_LISTED} more)" if len(items) > MAX_LISTED else "")


def _row(dataset: str, path: str, label_raw: str, **extra: Any) -> dict[str, Any]:
    return {"dataset": dataset, "path": path, "label_raw": label_raw, **extra}


def read_paddy_doctor(data_root: Path, cfg: Mapping[str, Any]) -> tuple[list[dict], list[Check]]:
    ds = cfg["datasets"]["S"]
    root = data_root / ds["root"]
    table = pd.read_csv(root / ds["csv"])
    rows, missing = [], []
    for r in table.itertuples(index=False):
        file = root / ds["images_dir"] / str(r.label) / str(r.image_id)
        if not file.is_file():
            missing.append(str(r.image_id))
            continue
        rows.append(
            _row("S", _rel(file, data_root), str(r.label), variety=r.variety, age_days=r.age)
        )
    on_disk = len(list_images(root / ds["images_dir"], cfg["image_extensions"]))
    checks = [
        Check(
            "Paddy Doctor csv rows have an image in their label folder",
            not missing,
            f"{len(missing)} missing: {_listed(missing)}" if missing else f"{len(rows)} rows",
        ),
        Check(
            "Paddy Doctor csv covers every image on disk",
            on_disk == len(table),
            f"{len(table)} csv rows, {on_disk} images on disk",
        ),
    ]
    return rows, checks


def _index_by_stem(files: list[Path]) -> dict[str, list[Path]]:
    index: dict[str, list[Path]] = {}
    for f in files:
        index.setdefault(f.stem.lower(), []).append(f)
    return index


def read_sethy(data_root: Path, cfg: Mapping[str, Any]) -> tuple[list[dict], list[Check]]:
    """Sethy images, each paired with its RiceSeg mask by exact filename stem (extension ignored)."""
    ds, ms = cfg["datasets"]["T1"], cfg["datasets"]["RiceSeg"]
    exts = cfg["image_extensions"]
    images = list_images(data_root / ds["root"] / ds["images_dir"], exts)
    masks = _index_by_stem(list_images(data_root / ms["root"] / ms["masks_dir"], exts))
    rows, no_mask, wrong_class = [], [], []
    for img in images:
        found = masks.get(img.stem.lower(), [])
        mask = found[0] if len(found) == 1 else None
        if mask is None:
            no_mask.append(img.stem)
        elif mask.parent.name != img.parent.name:
            wrong_class.append(img.stem)
        rows.append(
            _row(
                "T1",
                _rel(img, data_root),
                img.parent.name,
                mask_path=_rel(mask, data_root) if mask else None,
            )
        )
    image_stems = {i.stem.lower() for i in images}
    orphans = [m[0].stem for stem, m in masks.items() if stem not in image_stems]
    ambiguous = [stem for stem, m in masks.items() if len(m) > 1]
    problems = {
        "images without a mask": no_mask,
        "masks without an image": orphans,
        "stems with several masks": ambiguous,
        "mask in another class folder": wrong_class,
    }
    detail = "; ".join(f"{k}: {len(v)} ({_listed(v)})" for k, v in problems.items() if v)
    pairing = Check(
        "Sethy-RiceSeg pairing",
        not detail,
        detail or f"{len(images) - len(no_mask)} of {len(images)} images paired by stem",
    )
    return rows, [pairing]


def parse_info_label(text: str, aliases: Mapping[str, str]) -> tuple[str, str] | None:
    """'Rice Blast(Feild Background)' -> ('Rice Blast', 'field'); None if it does not parse."""
    m = _INFO_LABEL.match(text.strip())
    background = aliases.get(m.group("bg").strip().lower()) if m else None
    return (m.group("cls").strip(), background) if m and background else None


def read_dhan(data_root: Path, cfg: Mapping[str, Any]) -> tuple[list[dict], list[Check]]:
    ds = cfg["datasets"]["T2"]
    root = data_root / ds["root"]
    rows = []
    for folder, background in ds["backgrounds"].items():
        for img in list_images(root / folder, cfg["image_extensions"]):
            rows.append(_row("T2", _rel(img, data_root), img.parent.name, background=background))
    return rows, [
        _check_dhan_csv(rows, root / ds["info_csv"], ds["csv_background_aliases"], data_root)
    ]


def _check_dhan_csv(
    rows: list[dict], info_csv: Path, aliases: Mapping[str, str], data_root: Path
) -> Check:
    info = pd.read_csv(info_csv)
    expected = {}
    for name, text in zip(info["pictureName"], info["Diseases"], strict=True):
        expected[str(name)] = parse_info_label(str(text), aliases) or ("?", "?")
    disagree, not_in_csv = [], []
    for r in rows:
        name = Path(r["path"]).name
        if name not in expected:
            not_in_csv.append(name)
        elif expected[name] != (r["label_raw"], r["background"]):
            disagree.append(name)
    on_disk = {Path(r["path"]).name for r in rows}
    csv_only = [n for n in expected if n not in on_disk]
    problems = {
        "files not in the info CSV": not_in_csv,
        "CSV rows without a file": csv_only,
        "CSV class/background differs from folders": disagree,
    }
    detail = "; ".join(f"{k}: {len(v)} ({_listed(v)})" for k, v in problems.items() if v)
    return Check(
        "Dhan info CSV agrees with folders",
        not detail,
        detail or f"{len(rows)} files agree with {len(info)} CSV rows",
    )


def _subsample(rows: list[dict], n: int, rng: random.Random) -> list[dict]:
    return rows if len(rows) <= n else sorted(rng.sample(rows, n), key=lambda r: r["path"])


def _read_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as im:
        return im.size


def _finalize(rows: list[dict], data_root: Path) -> pd.DataFrame:
    df = pd.DataFrame(rows).sort_values(["dataset", "path"], kind="stable").reset_index(drop=True)
    df["image_id"] = [
        f"{d}_{i:06d}" for d, i in zip(df["dataset"], df.groupby("dataset").cumcount())
    ]
    sizes = [_read_size(data_root / p) for p in df["path"]]
    df["width"], df["height"] = [s[0] for s in sizes], [s[1] for s in sizes]
    for col in MANIFEST_COLUMNS:
        if col not in df:
            df[col] = None
    df["age_days"] = df["age_days"].astype("Int64")
    return df[MANIFEST_COLUMNS]


def build_manifest(
    data_root: Path, cfg: Mapping[str, Any], debug: bool = False
) -> tuple[pd.DataFrame, list[Check]]:
    """Read every dataset and return (manifest, checks). `debug` keeps n images per dataset."""
    readers = {"S": read_paddy_doctor, "T1": read_sethy, "T2": read_dhan}
    rng = random.Random(cfg["seed"])
    all_rows, checks = [], []
    for key, reader in readers.items():
        rows, ds_checks = reader(data_root, cfg)
        expected = cfg["datasets"][key]["expected_count"]
        if not debug:
            checks.append(
                Check(
                    f"{key} image count",
                    len(rows) == expected,
                    f"expected {expected}, found {len(rows)}",
                )
            )
        all_rows += _subsample(rows, cfg["debug"]["n_images"], rng) if debug else rows
        checks += ds_checks
    return _finalize(all_rows, data_root), checks


_STR_COLUMNS = [
    "label_raw", "label", "variety", "background", "mask_path", "phash", "group_id", "split",
]  # fmt: skip


def read_manifest(path: Path) -> pd.DataFrame:
    """Read manifest.csv with text columns kept as text (a hex phash can look like a number)."""
    df = pd.read_csv(path, dtype={c: "string" for c in _STR_COLUMNS})
    df[_STR_COLUMNS] = df[_STR_COLUMNS].astype(object).where(df[_STR_COLUMNS].notna(), None)
    df["age_days"] = df["age_days"].astype("Int64")
    return df


def has_m2_columns(path: Path) -> bool:
    """True if an existing manifest already carries hashes, groups or splits (M2 output)."""
    if not path.is_file():
        return False
    df = read_manifest(path)
    return bool(df[["phash", "group_id", "split"]].notna().any().any())


def suffixed(path: Path, suffix: str) -> Path:
    """manifest.csv + '_debug' -> manifest_debug.csv"""
    return path.with_name(path.stem + suffix + path.suffix)


def write_manifest(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def main(argv: list[str] | None = None) -> int:
    from paddyguard.data.dataset_card import render_dataset_card

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(paths.configs_dir() / "data.yaml"))
    parser.add_argument("--debug", action="store_true", help="keep n_images per dataset")
    parser.add_argument(
        "--force", action="store_true", help="overwrite a manifest that already has hashes/splits"
    )
    args = parser.parse_args(argv)
    cfg = io.load_yaml(args.config)
    set_seed(cfg["seed"])
    run_dir = io.new_run_dir("data", cfg["seed"], config=cfg)
    log = get_logger("paddyguard.manifest", run_dir / "log.txt")
    data_root = paths.data_root()
    layout = verify_layout(data_root, cfg)
    for c in layout:
        log.info("%s %s: %s", "OK" if c.ok else "MISSING", c.name, c.detail)
    if not all(c.ok for c in layout):
        return 1
    suffix = "_debug" if args.debug else ""
    manifest_path = suffixed(data_root / cfg["manifest_path"], suffix)
    if has_m2_columns(manifest_path) and not args.force:
        log.error(
            "%s already has hashes/splits; splits are fixed. Use --force to overwrite",
            manifest_path,
        )
        return 1
    df, checks = build_manifest(data_root, cfg, debug=args.debug)
    write_manifest(df, manifest_path)
    card_path = paths.reports_root() / cfg["card_path"]
    card_path = card_path.with_name(card_path.stem + suffix + card_path.suffix)
    card_path.parent.mkdir(parents=True, exist_ok=True)
    card_path.write_text(render_dataset_card(df, checks, cfg))
    for c in checks:
        log.info("%s %s: %s", "OK" if c.ok else "FAIL", c.name, c.detail)
    log.info("wrote %s (%d rows) and %s", manifest_path, len(df), card_path)
    return 0 if all(c.ok for c in checks) else 1


if __name__ == "__main__":
    sys.exit(main())
