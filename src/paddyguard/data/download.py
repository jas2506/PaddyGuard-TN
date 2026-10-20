"""Verify the raw dataset folders (M1). Never downloads: Mendeley has no stable API."""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from paddyguard.utils import io, paths


@dataclass(frozen=True)
class Check:
    name: str
    ok: bool
    detail: str = ""


def is_image(path: Path, extensions: list[str]) -> bool:
    return path.is_file() and path.suffix.lower() in extensions


def list_images(directory: Path, extensions: list[str]) -> list[Path]:
    """All image files below `directory`, sorted (deterministic order)."""
    return sorted(p for p in directory.rglob("*") if is_image(p, extensions))


def class_counts(images_dir: Path, extensions: list[str]) -> dict[str, int]:
    """Number of image files in each immediate sub-folder of `images_dir`."""
    counts = Counter(
        p.parent.name for p in list_images(images_dir, extensions) if p.parent != images_dir
    )
    return dict(sorted(counts.items()))


def _expected_paths(ds: Mapping[str, Any]) -> list[str]:
    """Paths (relative to the dataset root) that must exist for a dataset to be usable."""
    wanted = [ds[k] for k in ("images_dir", "masks_dir", "csv", "info_csv") if k in ds]
    return wanted + list(ds.get("backgrounds", {}))


def _count_dir(ds: Mapping[str, Any]) -> str | None:
    return ds.get("images_dir") or ds.get("masks_dir")


def verify_layout(data_root: Path, cfg: Mapping[str, Any]) -> list[Check]:
    """One check per dataset: do the expected folders/files exist? If not, says where to get it."""
    checks = []
    for key, ds in cfg["datasets"].items():
        root = data_root / ds["root"]
        missing = [p for p in _expected_paths(ds) if not (root / p).exists()]
        if missing:
            detail = (
                f"missing under {root}: {', '.join(missing)}. Download it manually from "
                f"{ds['source']}, unzip it, and arrange it as described in configs/data.yaml."
            )
        else:
            detail = f"found at {root}"
        checks.append(Check(f"{key} layout", not missing, detail))
    return checks


def print_class_counts(data_root: Path, cfg: Mapping[str, Any]) -> None:
    """Print the number of images per class folder for each dataset that has one."""
    for key, ds in cfg["datasets"].items():
        sub = _count_dir(ds)
        roots = (
            [data_root / ds["root"] / sub]
            if sub
            else [data_root / ds["root"] / b for b in ds.get("backgrounds", {})]
        )
        for r in roots:
            if r.is_dir():
                print(f"{key} {r.name}: {class_counts(r, cfg['image_extensions'])}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(paths.configs_dir() / "data.yaml"))
    args = parser.parse_args(argv)
    cfg = io.load_yaml(args.config)
    data_root = paths.data_root()
    checks = verify_layout(data_root, cfg)
    for c in checks:
        print(f"[{'OK' if c.ok else 'MISSING'}] {c.name}: {c.detail}")
    print_class_counts(data_root, cfg)
    return 0 if all(c.ok for c in checks) else 1


if __name__ == "__main__":
    sys.exit(main())
