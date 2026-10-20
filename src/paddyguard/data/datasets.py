"""PyTorch datasets and loaders over the manifest and the cached images (M3)."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from paddyguard.data.cache import PROCESSED_DIRNAME, processed_path
from paddyguard.data.dedup import DROPPED, is_representative
from paddyguard.data.labelmap import load_labelmap
from paddyguard.utils import paths

# domain -> (manifest dataset, background filter)
DOMAINS: dict[str, tuple[str, str | None]] = {
    "S": ("S", None),
    "T1": ("T1", None),
    "T2": ("T2", "field"),
    "T2-white": ("T2", "white"),
}
TRAIN_DOMAIN, TRAIN_SPLIT = "S", "train"


def select_rows(
    manifest: pd.DataFrame, domain: str, split: str | None, representatives_only: bool = False
) -> pd.DataFrame:
    """Rows of one domain (and split) that were not dropped in M2."""
    if domain not in DOMAINS:
        raise ValueError(f"unknown domain {domain!r}; choose from {sorted(DOMAINS)}")
    dataset, background = DOMAINS[domain]
    rows = manifest[(manifest["dataset"] == dataset) & ~manifest["split"].isin(DROPPED)]
    if background is not None:
        rows = rows[rows["background"] == background]
    if split is not None:
        if split not in set(rows["split"].dropna()):
            raise ValueError(
                f"no split {split!r} in domain {domain}: {sorted(rows['split'].dropna().unique())}"
            )
        rows = rows[rows["split"] == split]
    return rows[is_representative(rows)] if representatives_only else rows


class PaddyDataset(Dataset):
    """Items are (image, label_index, image_id), plus the lesion mask when `with_mask`.

    `classes` restricts the label space and re-indexes labels (default: all ten Paddy Doctor
    classes); images whose label is outside it, or empty, are skipped.
    """

    def __init__(
        self,
        manifest: pd.DataFrame,
        split: str | None,
        domain: str,
        transform: Callable[..., dict[str, Any]],
        classes: Sequence[str] | None = None,
        with_mask: bool = False,
        mask_threshold: int | None = None,
        representatives_only: bool = False,
        data_root: Path | None = None,
        processed_dirname: str = PROCESSED_DIRNAME,
    ) -> None:
        if with_mask and DOMAINS.get(domain, ("",))[0] != "T1":
            raise ValueError("masks exist only for T1 (Sethy)")
        if with_mask and mask_threshold is None:
            raise ValueError(
                "with_mask needs mask_threshold (mask_encoding.threshold in data.yaml)"
            )
        if manifest["label"].isna().all():
            raise ValueError(
                "manifest has no harmonised labels: run python -m paddyguard.data.labelmap"
            )
        self.domain, self.split, self.transform = domain, split, transform
        self.class_names = list(classes) if classes is not None else load_labelmap().source_classes
        rows = select_rows(manifest, domain, split, representatives_only)
        rows = rows[rows["label"].isin(self.class_names)]
        if rows.empty:
            raise ValueError(
                f"no images for domain {domain}, split {split}, classes {self.class_names}"
            )
        self.image_ids = rows["image_id"].tolist()
        self.labels = [self.class_names.index(c) for c in rows["label"]]
        self.with_mask, self.mask_threshold = with_mask, mask_threshold
        self._root = data_root or paths.data_root()
        self._dirname, self._datasets = processed_dirname, rows["dataset"].tolist()

    def __len__(self) -> int:
        return len(self.image_ids)

    def _load(self, i: int, suffix: str, mode: str) -> np.ndarray:
        path = processed_path(
            self._root, self._datasets[i], self.image_ids[i], suffix, self._dirname
        )
        with Image.open(path) as im:
            return np.asarray(im.convert(mode))

    def __getitem__(self, i: int) -> tuple:
        image = self._load(i, ".jpg", "RGB")
        if not self.with_mask:
            out = self.transform(image=image)
            return out["image"], self.labels[i], self.image_ids[i]
        mask = (self._load(i, "_mask.png", "L") >= self.mask_threshold).astype(np.uint8)
        out = self.transform(image=image, mask=mask)
        return out["image"], self.labels[i], self.image_ids[i], out["mask"]


def _seed_worker(worker_id: int, seed: int) -> None:
    info = torch.utils.data.get_worker_info()
    transform = getattr(info.dataset, "transform", None) if info else None
    if hasattr(transform, "set_random_seed"):
        transform.set_random_seed(
            seed + worker_id
        )  # else every worker repeats the same augmentations


def make_loader(
    dataset: PaddyDataset,
    training: bool,
    batch_size: int,
    seed: int,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> DataLoader:
    """DataLoader with a seeded shuffle for training. Training loaders accept only S/train."""
    if training and (dataset.domain, dataset.split) != (TRAIN_DOMAIN, TRAIN_SPLIT):
        raise ValueError(
            f"training loaders accept only {TRAIN_DOMAIN}/{TRAIN_SPLIT}, got {dataset.domain}/{dataset.split}"
        )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=training,
        drop_last=training and len(dataset) > batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        generator=torch.Generator().manual_seed(seed),
        worker_init_fn=partial(_seed_worker, seed=seed),
    )
