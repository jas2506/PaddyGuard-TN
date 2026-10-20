"""Synthetic datasets that mimic the layout of the four real downloads (tiny images, CPU only)."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from paddyguard.utils import paths
from paddyguard.utils.io import load_yaml

IMG_SIZE = (32, 24)  # width, height


def _write_photo(path: Path, seed: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.random.default_rng(seed).integers(
        0, 256, (IMG_SIZE[1], IMG_SIZE[0], 3), dtype=np.uint8
    )
    Image.fromarray(arr).save(path, quality=95)


def _write_mask(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.zeros((IMG_SIZE[1], IMG_SIZE[0]), dtype=np.uint8)
    arr[6:14, 8:20] = 255
    Image.fromarray(arr, mode="L").save(path, quality=95)


def make_tree(
    root: Path, *, extra_unmatched: bool = False, dhan_csv_wrong_background: bool = False
) -> None:
    """Paddy Doctor 6 images, Sethy 5 (+5 masks), Dhan 7 (5 field + 2 white)."""
    pd_dir = root / "raw" / "paddy_doctor"
    rows = ["image_id,label,variety,age"]
    for label, n in [("blast", 4), ("hispa", 2)]:
        for i in range(n):
            name = f"{label}_{i}.jpg"
            _write_photo(pd_dir / "train_images" / label / name, i)
            rows.append(f"{name},{label},ADT45,{40 + i}")
    (pd_dir / "train.csv").write_text("\n".join(rows) + "\n")

    for cls, stems in [
        ("Blast", ["BLAST1_001", "BLAST1_002", "BLAST1_003"]),
        ("Tungro", ["TUNGRO1_001", "TUNGRO1_002"]),
    ]:
        for k, stem in enumerate(stems):
            _write_photo(
                root / "raw" / "sethy" / "images" / cls / f"{stem}.JPG", k
            )  # upper-case ext
            _write_mask(root / "raw" / "riceseg" / "masks" / cls / f"{stem}.jpg")
    if extra_unmatched:
        _write_photo(root / "raw" / "sethy" / "images" / "Blast" / "BLAST9_999.jpg", 99)  # no mask
        _write_mask(root / "raw" / "riceseg" / "masks" / "Blast" / "ORPHAN_001.jpg")  # no photo

    dhan = root / "raw" / "dhan"
    info = [",pictureName,Diseases"]
    spec = [
        ("Field Background", "Rice Blast", "bf", 3, "Rice Blast(Feild Background)"),
        ("Field Background", "Browon Spot", "bsf", 2, "Browon Spot(Feild Background)"),
        ("White Background", "Rice Blast", "bw", 2, "Rice Blast(white Background)"),
    ]
    idx = 0
    for background, cls, prefix, n, label in spec:
        for i in range(n):
            name = f"{prefix}_{i}.jpg"
            _write_photo(dhan / background / cls / name, i)
            shown = (
                label.replace("Feild", "white")
                if (dhan_csv_wrong_background and idx == 0)
                else label
            )
            info.append(f"{idx},{name},{shown}")
            idx += 1
    (dhan / "Dhan-Shomadhan_picture_Information.csv").write_text("\n".join(info) + "\n")


@pytest.fixture
def data_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "data"
    make_tree(root)
    monkeypatch.setenv(paths.ENV_DATA, str(root))
    return root


@pytest.fixture
def data_cfg() -> dict[str, Any]:
    """The real configs/data.yaml with expected counts matching the synthetic tree."""
    cfg = copy.deepcopy(load_yaml(paths.configs_dir() / "data.yaml"))
    for key, n in {"S": 6, "T1": 5, "RiceSeg": 5, "T2": 7}.items():
        cfg["datasets"][key]["expected_count"] = n
    return cfg


@pytest.fixture
def tf_cfg() -> dict[str, Any]:
    """The real configs/transforms.yaml, shrunk so tests run on tiny images."""
    cfg = copy.deepcopy(load_yaml(paths.configs_dir() / "transforms.yaml"))
    cfg["img_size"], cfg["resize"] = 16, 18
    return cfg


@pytest.fixture
def prepared(data_root: Path, data_cfg: dict[str, Any]) -> pd.DataFrame:
    """Manifest with hand-made groups/splits, harmonised labels, and a built cache."""
    from paddyguard.data import cache, labelmap
    from paddyguard.data import manifest as mf

    df, _ = mf.build_manifest(data_root, data_cfg)
    df["group_id"] = df["image_id"]
    s, t1, t2 = (df["dataset"] == name for name in ("S", "T1", "T2"))
    df.loc[s, "split"] = (["train"] * 3 + ["val", "test", "test"])[: int(s.sum())]
    df.loc[t1, "split"] = ["seg_train", "seg_train", "seg_val", "seg_test", "seg_test"]
    df.loc[t2 & (df["background"] == "field"), "split"] = [
        "cal_pool",
        "cal_pool",
        "test",
        "test",
        "test",
    ]
    df.loc[t2 & (df["background"] == "white"), "split"] = "zero_shot"
    cache.build_cache(df, data_root, data_cfg)
    return labelmap.harmonise(df, labelmap.load_labelmap())
