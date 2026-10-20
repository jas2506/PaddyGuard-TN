"""Cached, resized copies of the kept images (M2). All training and evaluation read these.

S and T2: short side `cache.short_side` (never upscaled), JPEG. T1 (Sethy, ~300 px): native size,
never downscaled, plus the RiceSeg mask as a single-channel PNG with identical geometry.
Output: <data root>/processed/<dataset>/<image_id>.jpg and, for T1, <image_id>_mask.png.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from collections.abc import Mapping
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image, ImageOps

from paddyguard.data.dedup import DROPPED
from paddyguard.data.manifest import read_manifest, suffixed
from paddyguard.utils import io, paths
from paddyguard.utils.logging import get_logger
from paddyguard.utils.seed import set_seed

PROCESSED_DIRNAME = "processed"
CHUNK = 32


def processed_path(
    data_root: Path,
    dataset: str,
    image_id: str,
    suffix: str = ".jpg",
    dirname: str = PROCESSED_DIRNAME,
) -> Path:
    return data_root / dirname / dataset / f"{image_id}{suffix}"


def _save_resized(src: Path, dest: Path, short_side: int, quality: int) -> None:
    with Image.open(src) as im:
        img = ImageOps.exif_transpose(im).convert("RGB")
    scale = short_side / min(img.size)
    if scale < 1:
        img = img.resize((round(img.width * scale), round(img.height * scale)), Image.LANCZOS)
    img.save(dest, format="JPEG", quality=quality)


def _save_native(src: Path, dest: Path, quality: int) -> None:
    """Copy a real JPEG unchanged; re-encode anything else (some Sethy '.jpg' files are PNGs)."""
    with Image.open(src) as im:
        is_jpeg = im.format == "JPEG"
        img = None if is_jpeg else im.convert("RGB")
    if img is None:
        shutil.copyfile(src, dest)
    else:
        img.save(dest, format="JPEG", quality=quality)


def _save_mask(src: Path, dest: Path, size: tuple[int, int]) -> None:
    with Image.open(src) as im:
        mask = im.convert("L")
    if mask.size != size:
        raise ValueError(f"mask {src} is {mask.size}, its photo is {size}")
    mask.save(dest, format="PNG")


def _process_one(row: tuple, data_root: Path, dirname: str, short_side: int, quality: int) -> str:
    dataset, image_id, path, mask_path = row
    dest = processed_path(data_root, dataset, image_id, dirname=dirname)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        if dataset == "T1":
            _save_native(data_root / path, dest, quality)
        else:
            _save_resized(data_root / path, dest, short_side, quality)
    mask_dest = processed_path(data_root, dataset, image_id, "_mask.png", dirname)
    if dataset == "T1" and mask_path and not mask_dest.exists():
        with Image.open(dest) as im:
            _save_mask(data_root / mask_path, mask_dest, im.size)
    return image_id


def build_cache(
    df: pd.DataFrame, data_root: Path, cfg: Mapping[str, Any], dirname: str = PROCESSED_DIRNAME
) -> int:
    """Write processed images for every kept row; existing files are left alone. Returns the count."""
    kept = df[~df["split"].isin(DROPPED)]
    rows = list(
        zip(kept["dataset"], kept["image_id"], kept["path"], kept["mask_path"], strict=True)
    )
    fn = partial(
        _process_one,
        data_root=data_root,
        dirname=dirname,
        short_side=cfg["cache"]["short_side"],
        quality=cfg["cache"]["jpeg_quality"],
    )
    with ProcessPoolExecutor() as pool:
        return sum(1 for _ in pool.map(fn, rows, chunksize=CHUNK))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(paths.configs_dir() / "data.yaml"))
    parser.add_argument("--debug", action="store_true", help="work on manifest_debug.csv")
    args = parser.parse_args(argv)
    cfg = io.load_yaml(args.config)
    set_seed(cfg["seed"])
    log = get_logger(
        "paddyguard.cache", io.new_run_dir("data", cfg["seed"], config=cfg) / "log.txt"
    )
    data_root = paths.data_root()
    suffix = "_debug" if args.debug else ""  # debug ids differ from the real ones: separate folder
    df = read_manifest(suffixed(data_root / cfg["manifest_path"], suffix))
    dirname = PROCESSED_DIRNAME + suffix
    n = build_cache(df, data_root, cfg, dirname)
    log.info("cached %d images under %s", n, data_root / dirname)
    return 0


if __name__ == "__main__":
    sys.exit(main())
