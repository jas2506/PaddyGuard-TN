"""Albumentations pipelines (M3): baseline, strong (and strong_mixstyle), eval.

`strong_mixstyle` uses the strong pipeline; MixStyle itself is a model-side change (M5).
All numbers come from configs/transforms.yaml.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import albumentations as A
from albumentations.pytorch import ToTensorV2

from paddyguard.utils import io, paths

TRAIN_NAMES = ("baseline", "strong", "strong_mixstyle")


def load_transform_config(path: str | None = None) -> dict[str, Any]:
    return io.load_yaml(path or paths.configs_dir() / "transforms.yaml")


def _finish(cfg: Mapping[str, Any]) -> list[A.BasicTransform]:
    return [A.Normalize(mean=cfg["mean"], std=cfg["std"]), ToTensorV2()]


def _baseline(cfg: Mapping[str, Any]) -> list[A.BasicTransform]:
    size, b = cfg["img_size"], cfg["baseline"]
    return [
        A.RandomResizedCrop(size=(size, size), scale=tuple(b["crop_scale"])),
        A.HorizontalFlip(p=b["hflip_p"]),
    ]


def _strong_extras(s: Mapping[str, Any]) -> list[A.BasicTransform]:
    cj, bc, hsv = s["color_jitter"], s["brightness_contrast"], s["hue_sat_value"]
    return [
        A.ColorJitter(
            brightness=cj["brightness"], contrast=cj["contrast"],
            saturation=cj["saturation"], hue=cj["hue"], p=cj["p"],
        ),
        A.RandomBrightnessContrast(
            brightness_limit=bc["limit"], contrast_limit=bc["limit"], p=bc["p"]
        ),
        A.HueSaturationValue(
            hue_shift_limit=hsv["hue"], sat_shift_limit=hsv["sat"], val_shift_limit=hsv["val"],
            p=hsv["p"],
        ),
        A.GaussianBlur(blur_limit=tuple(s["gaussian_blur"]["blur_limit"]), p=s["gaussian_blur"]["p"]),
        A.ImageCompression(
            quality_range=tuple(s["jpeg"]["quality"]), p=s["jpeg"]["p"]
        ),
        A.CoarseDropout(
            num_holes_range=tuple(s["coarse_dropout"]["holes"]),
            hole_height_range=tuple(s["coarse_dropout"]["size_fraction"]),
            hole_width_range=tuple(s["coarse_dropout"]["size_fraction"]),
            fill=0, p=s["coarse_dropout"]["p"],
        ),
    ]  # fmt: skip


def build_transform(name: str, cfg: Mapping[str, Any], seed: int | None = None) -> A.Compose:
    """Compose for `name`. Call as t(image=..., [mask=...]) -> dict with torch tensors."""
    size = cfg["img_size"]
    if name == "eval":
        ops = [A.SmallestMaxSize(max_size=cfg["resize"]), A.CenterCrop(height=size, width=size)]
    elif name == "baseline":
        ops = _baseline(cfg)
    elif name in ("strong", "strong_mixstyle"):
        rot = cfg["strong"]["rotation"]
        ops = [
            A.Rotate(limit=rot["limit"], p=rot["p"]),
            *_baseline(cfg),
            *_strong_extras(cfg["strong"]),
        ]
    else:
        raise ValueError(f"unknown transform {name!r}; choose from eval, {', '.join(TRAIN_NAMES)}")
    return A.Compose([*ops, *_finish(cfg)], seed=seed)
