"""Near-duplicate detection, duplicate groups and cross-dataset leakage (M2).

Perceptual hashes (64 bit) -> pairs within a Hamming threshold -> union-find groups, per dataset.
Target-dataset (T1, T2) images that duplicate another target image or a Paddy Doctor image are
marked in the `split` column as dropped; Paddy Doctor duplicates are kept but share a group_id so
the splitter keeps them together.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Mapping
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path
from typing import Any

import imagehash
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageOps

from paddyguard.data.dataset_card import _table
from paddyguard.data.manifest import read_manifest, suffixed, write_manifest
from paddyguard.utils import io, paths
from paddyguard.utils.logging import get_logger
from paddyguard.utils.seed import set_seed

DROPPED_DUPLICATE = "dropped_duplicate"
DROPPED_LEAKAGE = "dropped_leakage"
DROPPED = (DROPPED_DUPLICATE, DROPPED_LEAKAGE)
SOURCE = "S"
TARGETS = ("T1", "T2")
DEFAULT_CHUNK_ROWS = 512
HASH_CHUNK = 64  # images per task handed to a worker process
THUMB = 200  # boundary-sheet thumbnail size in pixels
SHEET_COLUMNS = 3


def phash_hex(path: Path, draft_size: int, hash_size: int) -> str:
    """Perceptual hash of one image as hex. EXIF rotation is applied before hashing."""
    with Image.open(path) as im:
        im.draft("RGB", (draft_size, draft_size))  # fast reduced-size JPEG decode; no-op otherwise
        upright = ImageOps.exif_transpose(im)
        return str(imagehash.phash(upright.convert("RGB"), hash_size=hash_size))


def compute_hashes(
    data_root: Path, rel_paths: list[str], draft_size: int, hash_size: int
) -> list[str]:
    fn = partial(phash_hex, draft_size=draft_size, hash_size=hash_size)
    with ProcessPoolExecutor() as pool:
        return list(pool.map(fn, [data_root / p for p in rel_paths], chunksize=HASH_CHUNK))


def hex_to_uint64(hexes: list[str]) -> np.ndarray:
    return np.array([int(h, 16) for h in hexes], dtype=np.uint64)


def popcount(x: np.ndarray) -> np.ndarray:
    return np.bitwise_count(x).astype(np.int64)


def _pairs_below(
    a: np.ndarray, b: np.ndarray, max_dist: int, chunk_rows: int, upper_triangle: bool
) -> np.ndarray:
    out = []
    for start in range(0, len(a), chunk_rows):
        block = a[start : start + chunk_rows]
        dist = popcount(block[:, None] ^ b[None, :])
        rows, cols = np.nonzero(dist <= max_dist)
        keep = cols > rows + start if upper_triangle else np.ones(len(rows), dtype=bool)
        out.append(np.stack([rows[keep] + start, cols[keep], dist[rows[keep], cols[keep]]], axis=1))
    return np.concatenate(out) if out else np.empty((0, 3), dtype=np.int64)


def near_pairs(
    hashes: np.ndarray, max_dist: int, chunk_rows: int = DEFAULT_CHUNK_ROWS
) -> np.ndarray:
    """All pairs (i < j) of one hash array with Hamming distance <= max_dist, as rows (i, j, d)."""
    return _pairs_below(hashes, hashes, max_dist, chunk_rows, upper_triangle=True)


def cross_pairs(
    a: np.ndarray, b: np.ndarray, max_dist: int, chunk_rows: int = DEFAULT_CHUNK_ROWS
) -> np.ndarray:
    """All pairs (i in a, j in b) with Hamming distance <= max_dist, as rows (i, j, d)."""
    return _pairs_below(a, b, max_dist, chunk_rows, upper_triangle=False)


def cluster(n: int, pairs: np.ndarray) -> np.ndarray:
    """Union-find over `pairs` (rows i, j, ...). Returns for each item the smallest index of its cluster."""
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, j in pairs[:, :2].tolist():
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[max(ri, rj)] = min(ri, rj)
    return np.array([find(i) for i in range(n)])


def _histogram(pairs: np.ndarray, hist_max: int) -> list[int]:
    return (
        np.bincount(pairs[:, 2], minlength=hist_max + 1).tolist()
        if len(pairs)
        else [0] * (hist_max + 1)
    )


def _dataset_groups(
    sub: pd.DataFrame, threshold: int, hist_max: int, chunk_rows: int
) -> dict[str, Any]:
    hashes = hex_to_uint64(sub["phash"].tolist())
    pairs = near_pairs(hashes, hist_max, chunk_rows)
    labels = cluster(len(sub), pairs[pairs[:, 2] <= threshold])
    group_id = sub["image_id"].groupby(labels).transform("min").to_numpy()
    sizes = pd.Series(labels).value_counts()
    return {"hashes": hashes, "pairs": pairs, "group_id": group_id, "sizes": sizes}


def is_representative(df: pd.DataFrame) -> pd.Series:
    """True for the one image per duplicate group that evaluation sets keep (the group's id)."""
    return df["image_id"] == df["group_id"]


def assign_groups(
    df: pd.DataFrame,
    thresholds: Mapping[str, int],
    leakage_threshold: int,
    hist_max: int,
    chunk_rows: int = DEFAULT_CHUNK_ROWS,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Set group_id for every image (threshold per dataset) and mark dropped images in `split`.

    Dropped here: T2 images that duplicate another T2 image (T2 has no training use) and target
    images within `leakage_threshold` of a Paddy Doctor image. T1 duplicates are kept as group
    members; the splitter drops the extra members only inside seg_test.
    """
    out = df.copy()
    out["group_id"] = None
    per: dict[str, Any] = {}
    for ds, sub in df.groupby("dataset", sort=False):
        info = _dataset_groups(sub, thresholds[ds], hist_max, chunk_rows)
        out.loc[sub.index, "group_id"] = info["group_id"]
        per[ds] = {**info, "index": sub.index}
    leakage, cross = _mark_drops(out, per, leakage_threshold, hist_max, chunk_rows)
    summary = {
        "thresholds": dict(thresholds),
        "leakage_threshold": leakage_threshold,
        "hist_max": hist_max,
        "datasets": {ds: _dataset_summary(out, p, hist_max) for ds, p in per.items()},
        "leakage": {t: len(leakage.get(t, [])) for t in TARGETS},
        "leakage_by_class": {
            t: out.loc[leakage.get(t, []), "label_raw"].value_counts().to_dict() for t in TARGETS
        },
        "cross_hist": {t: _histogram(c, hist_max) for t, c in cross.items()},
        "cross_pairs": cross,
        "pairs": per,
    }
    return out, summary


def _mark_drops(
    out: pd.DataFrame, per: dict[str, Any], leakage_threshold: int, hist_max: int, chunk_rows: int
) -> tuple[dict[str, list], dict[str, np.ndarray]]:
    leakage: dict[str, list] = {}
    cross: dict[str, np.ndarray] = {}
    if "T2" in per:
        is_dup = out.loc[per["T2"]["index"], "image_id"].to_numpy() != per["T2"]["group_id"]
        out.loc[per["T2"]["index"][is_dup], "split"] = DROPPED_DUPLICATE
    if SOURCE in per:
        for t in (t for t in TARGETS if t in per):
            cross[t] = cross_pairs(per[t]["hashes"], per[SOURCE]["hashes"], hist_max, chunk_rows)
            leaked = np.unique(cross[t][cross[t][:, 2] <= leakage_threshold, 0])
            leakage[t] = list(per[t]["index"][leaked])
            out.loc[leakage[t], "split"] = DROPPED_LEAKAGE
    return leakage, cross


def _dataset_summary(out: pd.DataFrame, p: dict[str, Any], hist_max: int) -> dict[str, Any]:
    sizes = p["sizes"]
    kept = ~out.loc[p["index"], "split"].isin(DROPPED)
    return {
        "images": len(p["index"]),
        "clusters": len(sizes),
        "multi_clusters": int((sizes > 1).sum()),
        "largest_cluster": int(sizes.max()),
        "images_in_multi_clusters": int(sizes[sizes > 1].sum()),
        "kept": int(kept.sum()),
        "histogram": _histogram(p["pairs"], hist_max),
    }


def boundary_pairs(pairs: np.ndarray, threshold: int, n: int, seed: int) -> np.ndarray:
    """n pairs around the threshold: the closest at-or-below it and the closest above it."""
    rng = np.random.default_rng(seed)
    shuffled = pairs[rng.permutation(len(pairs))]
    below = shuffled[shuffled[:, 2] <= threshold]
    above = shuffled[shuffled[:, 2] > threshold]
    below = below[np.argsort(-below[:, 2], kind="stable")][: n // 2]
    above = above[np.argsort(above[:, 2], kind="stable")][: n - n // 2]
    return np.concatenate([below, above])


def _thumb(path: Path) -> Image.Image:
    with Image.open(path) as im:
        im.draft("RGB", (THUMB * 2, THUMB * 2))
        out = ImageOps.exif_transpose(im).convert("RGB")
    out.thumbnail((THUMB, THUMB))
    return out


def save_boundary_sheet(rows: list[dict[str, Any]], data_root: Path, path: Path) -> None:
    """Contact sheet of near-threshold pairs, each captioned with its Hamming distance."""
    cell_w, cell_h = 2 * THUMB + 8, THUMB + 18
    n_rows = -(-len(rows) // SHEET_COLUMNS)
    sheet = Image.new("RGB", (SHEET_COLUMNS * cell_w, max(n_rows, 1) * cell_h), "white")
    draw = ImageDraw.Draw(sheet)
    for k, r in enumerate(rows):
        x, y = (k % SHEET_COLUMNS) * cell_w, (k // SHEET_COLUMNS) * cell_h
        draw.text(
            (x + 2, y + 2), f"d={r['d']} {r['dataset']} {r['id_a']} | {r['id_b']}", fill="black"
        )
        sheet.paste(_thumb(data_root / r["path_a"]), (x, y + 16))
        sheet.paste(_thumb(data_root / r["path_b"]), (x + THUMB + 8, y + 16))
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path, quality=90)


def _pair_row(label: str, d: int, a: pd.Series, b: pd.Series) -> dict[str, Any]:
    return {
        "dataset": label,
        "d": d,
        "id_a": a["image_id"],
        "id_b": b["image_id"],
        "path_a": a["path"],
        "path_b": b["path"],
        "class_a": a["label_raw"],
        "class_b": b["label_raw"],
    }


def boundary_rows(
    df: pd.DataFrame, summary: Mapping[str, Any], n: int, seed: int
) -> list[dict[str, Any]]:
    """Pairs around each threshold, n spread evenly over the datasets and the cross-dataset checks."""
    cross = summary["cross_pairs"]
    per_group = max(n // (len(summary["pairs"]) + len(cross)), 2)
    rows = []
    for ds, p in summary["pairs"].items():
        chosen = boundary_pairs(p["pairs"], summary["thresholds"][ds], per_group, seed)
        sub = df.loc[p["index"]].reset_index(drop=True)
        rows += [_pair_row(ds, d, sub.loc[i], sub.loc[j]) for i, j, d in chosen.tolist()]
    source = df.loc[summary["pairs"][SOURCE]["index"]].reset_index(drop=True)
    for t, pairs in cross.items():
        chosen = boundary_pairs(pairs, summary["leakage_threshold"], per_group, seed)
        sub = df.loc[summary["pairs"][t]["index"]].reset_index(drop=True)
        rows += [
            _pair_row(f"{t} vs S", d, sub.loc[i], source.loc[j]) for i, j, d in chosen.tolist()
        ]
    return rows


def _nonzero_columns(rows: list[list[Any]], hist_max: int) -> tuple[list[str], list[list[Any]]]:
    """Histogram columns that have a pair in some row (odd distances never occur for 64-bit phash)."""
    keep = [d for d in range(hist_max + 1) if any(r[1 + d] for r in rows)]
    return [f"d={d}" for d in keep], [[r[0], *[r[1 + d] for d in keep]] for r in rows]


def render_report(
    summary: Mapping[str, Any], boundary: list[dict[str, Any]], cfg: Mapping[str, Any], sheet: str
) -> str:
    dd, hist_max = cfg["dedup"], summary["hist_max"]
    ds_rows = [
        [ds, s["images"], s["clusters"], s["multi_clusters"], s["largest_cluster"],
         s["images_in_multi_clusters"], s["kept"]]
        for ds, s in summary["datasets"].items()
    ]  # fmt: skip
    hist_rows = [[ds, *s["histogram"]] for ds, s in summary["datasets"].items()]
    hist_rows += [[f"{t} vs S (cross)", *h] for t, h in summary["cross_hist"].items()]
    head, hist_rows = _nonzero_columns(hist_rows, hist_max)
    leak = [[t, n, summary["leakage_by_class"][t]] for t, n in summary["leakage"].items()]
    pair_rows = [
        [r["dataset"], r["d"], r["id_a"], r["id_b"], r["class_a"], r["class_b"]] for r in boundary
    ]
    review = dd.get("review_note") or "NOT YET REVIEWED: no review note in configs/data.yaml."
    thresholds = ", ".join(f"{k} <= {v}" for k, v in summary["thresholds"].items())
    settings = (
        f"## Settings\n\n- Perceptual hash: `imagehash.phash`, hash_size {dd['phash_size']} (64 bit), "
        f"JPEGs decoded at reduced size (draft {dd['draft_size']} px), EXIF rotation applied. "
        "A 64-bit phash always has 32 set bits, so only even distances occur.\n"
        f"- Near-duplicate thresholds (Hamming distance, `dedup.hamming`): {thresholds}\n"
        f"- Cross-dataset leakage threshold (`dedup.leakage_hamming`): <= {summary['leakage_threshold']}\n"
        f"- Review of the boundary pairs: {review}"
    )
    group_head = ["dataset", "images", "groups", "groups with >1 image", "largest group"]
    groups = (
        "## Duplicate groups\n\nUnion-find over pairs within each dataset's threshold. Evaluation "
        "sets keep one representative per group (the image whose id equals the group id). Paddy "
        "Doctor and T1 keep all members, which share a group_id and never straddle a split; T2 "
        "non-representatives are dropped here, and T1 non-representatives are dropped by the "
        "splitter only inside seg_test.\n\n"
        + _table([*group_head, "images in such groups", "kept so far"], ds_rows)
    )
    hist = (
        "## Pairs per Hamming distance\n\nNumber of image pairs at each distance within each "
        "dataset, and between each target and Paddy Doctor (S). Columns with no pairs are omitted.\n\n"
        + _table(["pairs", *head], hist_rows)
    )
    leakage = (
        "## Leakage from Paddy Doctor into the targets\n\nTarget images within the leakage "
        "threshold of a Paddy Doctor image are dropped. A non-zero count is itself a finding.\n\n"
        + _table(["target", "images dropped", "by class folder"], leak)
    )
    around = (
        f"## Pairs around the thresholds\n\nFigure: `{sheet}`. Pairs at or just below a threshold "
        "are treated as duplicates, pairs just above are not.\n\n"
        + _table(["dataset", "distance", "image a", "image b", "class a", "class b"], pair_rows)
    )
    title = "# Duplicate and leakage report\n\nGenerated by `python -m paddyguard.data.dedup`."
    sections = [title, settings, groups, hist, leakage, around]
    return "\n\n".join(sections) + "\n"


def run(df: pd.DataFrame, data_root: Path, cfg: Mapping[str, Any]) -> tuple[pd.DataFrame, dict]:
    dd = cfg["dedup"]
    out = df.copy()
    if out["phash"].isna().any():
        out["phash"] = compute_hashes(
            data_root, out["path"].tolist(), dd["draft_size"], dd["phash_size"]
        )
    return assign_groups(
        out, dd["hamming"], dd["leakage_hamming"], dd["hist_max"], dd["chunk_rows"]
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(paths.configs_dir() / "data.yaml"))
    parser.add_argument("--debug", action="store_true", help="work on manifest_debug.csv")
    args = parser.parse_args(argv)
    cfg = io.load_yaml(args.config)
    set_seed(cfg["seed"])
    log = get_logger(
        "paddyguard.dedup", io.new_run_dir("data", cfg["seed"], config=cfg) / "log.txt"
    )
    suffix = "_debug" if args.debug else ""
    data_root = paths.data_root()
    manifest_path = suffixed(data_root / cfg["manifest_path"], suffix)
    df = read_manifest(manifest_path)
    kept_splits = df["split"].dropna()
    if (~kept_splits.isin(DROPPED)).any():
        log.error("splits already assigned in %s: refusing to re-run dedup", manifest_path)
        return 1
    df["split"] = None
    out, summary = run(df, data_root, cfg)
    write_manifest(out, manifest_path)
    reports = paths.reports_root()
    sheet = f"figures/dedup_boundary_pairs{suffix}.jpg"
    rows = boundary_rows(out, summary, cfg["dedup"]["boundary_pairs"], cfg["seed"])
    save_boundary_sheet(rows, data_root, reports / sheet)
    report = suffixed(reports / "dedup_report.md", suffix)
    report.write_text(render_report(summary, rows, cfg, sheet))
    log.info("wrote %s, %s and %s", manifest_path, report, reports / sheet)
    return 0


if __name__ == "__main__":
    sys.exit(main())
