"""Class harmonisation across datasets, and restricting predictions to shared classes (M3)."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from paddyguard.data.manifest import read_manifest, suffixed, write_manifest
from paddyguard.utils import io, paths

SHARED_KEY = {"T1": "S_T1", "T2": "S_T2", "T2-white": "S_T2"}


@dataclass(frozen=True)
class LabelMap:
    canonical: list[str]  # the shared 4-class space; order defines its class indices
    extra: list[str]  # Paddy Doctor-only classes
    map: dict[
        str, dict[str, str | None]
    ]  # dataset -> raw label -> class name (None = no counterpart)
    shared: dict[str, list[str]]  # "S_T1", "S_T2" -> shared classes

    @property
    def source_classes(self) -> list[str]:
        """Every Paddy Doctor class: the classifier's output space and column order."""
        return [*self.canonical, *self.extra]

    def validate(self) -> None:
        known = set(self.source_classes)
        for dataset, mapping in self.map.items():
            bad = {v for v in mapping.values() if v is not None and v not in known}
            if bad:
                raise ValueError(f"labelmap {dataset} maps to unknown classes: {sorted(bad)}")
        for key, classes in self.shared.items():
            bad = [c for c in classes if c not in self.canonical]
            if bad:
                raise ValueError(f"labelmap shared {key} has non-canonical classes: {bad}")


def load_labelmap(path: str | Path | None = None) -> LabelMap:
    raw = io.load_yaml(path or paths.configs_dir() / "labelmap.yaml")
    lm = LabelMap(raw["canonical"], raw["extra_source_classes"], raw["map"], raw["shared"])
    lm.validate()
    return lm


def missing_raw_labels(df: pd.DataFrame, lm: LabelMap) -> dict[str, set[str]]:
    """Raw labels present in the manifest but absent from the label map, per dataset."""
    missing = {}
    for dataset, sub in df.groupby("dataset"):
        unknown = set(sub["label_raw"]) - set(lm.map.get(dataset, {}))
        if unknown:
            missing[dataset] = unknown
    return missing


def harmonise(df: pd.DataFrame, lm: LabelMap) -> pd.DataFrame:
    """Fill `label` from `label_raw`. Raw labels mapped to null keep an empty label."""
    missing = missing_raw_labels(df, lm)
    if missing:
        raise ValueError(f"raw labels not in configs/labelmap.yaml: {missing}")
    out = df.copy()
    out["label"] = None
    for dataset, mapping in lm.map.items():
        rows = out["dataset"] == dataset
        out.loc[rows, "label"] = out.loc[rows, "label_raw"].map(mapping)
    out["label"] = out["label"].astype(object).where(out["label"].notna(), None)
    return out


def shared_classes(lm: LabelMap, domain: str) -> list[str]:
    """Classes a domain shares with Paddy Doctor (all ten classes for Paddy Doctor itself)."""
    if domain == "S":
        return lm.source_classes
    if domain not in SHARED_KEY:
        raise ValueError(f"unknown domain {domain!r}")
    return lm.shared[SHARED_KEY[domain]]


def restrict_logits(logits: Any, class_names: Sequence[str], shared: Sequence[str]) -> Any:
    """Keep only the columns of `shared` classes (in that order). Works on numpy and torch."""
    if logits.shape[-1] != len(class_names):
        raise ValueError(
            f"logits have {logits.shape[-1]} columns, class_names has {len(class_names)}"
        )
    absent = [c for c in shared if c not in class_names]
    if absent:
        raise ValueError(f"model has no output for classes: {absent}")
    return logits[:, [list(class_names).index(c) for c in shared]]


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    z = x - x.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Fill the manifest `label` column from labelmap.yaml."
    )
    parser.add_argument("--config", default=str(paths.configs_dir() / "data.yaml"))
    parser.add_argument("--debug", action="store_true", help="work on manifest_debug.csv")
    args = parser.parse_args(argv)
    cfg = io.load_yaml(args.config)
    path = suffixed(paths.data_root() / cfg["manifest_path"], "_debug" if args.debug else "")
    out = harmonise(read_manifest(path), load_labelmap())
    write_manifest(out, path)
    print(out.groupby("dataset")["label"].agg(labelled="count", total="size").to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
