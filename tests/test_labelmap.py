from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import torch

from paddyguard.data import labelmap as lm_mod
from paddyguard.data.manifest import read_manifest
from paddyguard.utils import paths


@pytest.fixture
def lm() -> lm_mod.LabelMap:
    return lm_mod.load_labelmap()


def test_real_labelmap_is_consistent(lm: lm_mod.LabelMap) -> None:
    assert lm.canonical == ["bacterial_leaf_blight", "blast", "brown_spot", "tungro"]
    assert len(lm.source_classes) == 10 and lm.source_classes[:4] == lm.canonical
    assert lm.shared["S_T2"] == ["blast", "brown_spot", "tungro"]


def test_validate_rejects_unknown_target_and_bad_shared(lm: lm_mod.LabelMap) -> None:
    bad = lm_mod.LabelMap(lm.canonical, lm.extra, {**lm.map, "T1": {"X": "not_a_class"}}, lm.shared)
    with pytest.raises(ValueError, match="not_a_class"):
        bad.validate()
    bad = lm_mod.LabelMap(lm.canonical, lm.extra, lm.map, {"S_T1": ["hispa"], "S_T2": ["blast"]})
    with pytest.raises(ValueError, match="hispa"):
        bad.validate()


def test_harmonise_fills_labels_and_leaves_unmapped_empty(lm: lm_mod.LabelMap) -> None:
    df = pd.DataFrame(
        {
            "dataset": ["S", "T1", "T2", "T2", "T2"],
            "label_raw": ["hispa", "Brownspot", "Browon Spot", "Rice Turgro", "Leaf Scaled"],
            "label": None,
        }
    )
    out = lm_mod.harmonise(df, lm)
    assert out["label"].tolist()[:4] == ["hispa", "brown_spot", "brown_spot", "tungro"]
    assert pd.isna(out["label"].iloc[4])  # Leaf Scaled has no Paddy Doctor counterpart


def test_missing_raw_labels_are_reported_and_harmonise_refuses(lm: lm_mod.LabelMap) -> None:
    df = pd.DataFrame({"dataset": ["T1"], "label_raw": ["Mystery"], "label": None})
    assert lm_mod.missing_raw_labels(df, lm) == {"T1": {"Mystery"}}
    with pytest.raises(ValueError, match="Mystery"):
        lm_mod.harmonise(df, lm)


def test_label_map_covers_every_raw_label_of_the_real_manifest(lm: lm_mod.LabelMap) -> None:
    path = paths.data_root() / "manifests" / "manifest.csv"
    if not path.is_file():
        pytest.skip("real manifest not available")
    df = read_manifest(path)
    assert lm_mod.missing_raw_labels(df, lm) == {}
    out = lm_mod.harmonise(df, lm)
    unlabelled = out[out["label"].isna()]
    assert set(unlabelled["dataset"]) <= {"T2"}  # only Dhan classes without a counterpart


@pytest.mark.parametrize(
    ("domain", "expected"),
    [("S", None), ("T1", ["bacterial_leaf_blight", "blast", "brown_spot", "tungro"]),
     ("T2", ["blast", "brown_spot", "tungro"]), ("T2-white", ["blast", "brown_spot", "tungro"])],
)  # fmt: skip
def test_shared_classes_by_domain(
    lm: lm_mod.LabelMap, domain: str, expected: list[str] | None
) -> None:
    assert lm_mod.shared_classes(lm, domain) == (expected or lm.source_classes)


def test_restrict_logits_keeps_the_shared_columns_in_the_shared_order(
    lm: lm_mod.LabelMap,
) -> None:
    names = lm.source_classes
    logits = np.arange(2 * len(names), dtype=np.float32).reshape(2, -1)
    out = lm_mod.restrict_logits(logits, names, ["tungro", "blast"])
    assert out.shape == (2, 2)
    assert (out[:, 0] == logits[:, names.index("tungro")]).all()
    assert (out[:, 1] == logits[:, names.index("blast")]).all()


def test_restricted_probabilities_sum_to_one_for_numpy_and_torch(lm: lm_mod.LabelMap) -> None:
    names = lm.source_classes
    logits = np.random.default_rng(0).normal(size=(5, len(names))).astype(np.float32)
    shared = lm.shared["S_T2"]
    p = lm_mod.softmax(lm_mod.restrict_logits(logits, names, shared))
    assert p.shape == (5, 3) and np.allclose(p.sum(axis=1), 1.0, atol=1e-6)
    pt = lm_mod.restrict_logits(torch.from_numpy(logits), names, shared)
    assert isinstance(pt, torch.Tensor) and pt.shape == (5, 3)


def test_restrict_logits_rejects_a_class_the_model_does_not_have(lm: lm_mod.LabelMap) -> None:
    with pytest.raises(ValueError, match="not_a_class"):
        lm_mod.restrict_logits(np.zeros((1, 10)), lm.source_classes, ["blast", "not_a_class"])
    with pytest.raises(ValueError):
        lm_mod.restrict_logits(np.zeros((1, 9)), lm.source_classes, ["blast"])


def test_cli_fills_the_label_column_and_changes_nothing_else(
    data_root: Path, data_cfg: dict[str, Any]
) -> None:
    from paddyguard.data import manifest as mf

    df, _ = mf.build_manifest(data_root, data_cfg)
    df["split"] = "train"  # pretend M2 ran: the CLI must keep these columns intact
    path = data_root / data_cfg["manifest_path"]
    mf.write_manifest(df, path)
    before = read_manifest(path)
    assert lm_mod.main([]) == 0
    out = read_manifest(path)
    assert out["label"].notna().sum() == len(out)  # every synthetic raw label is mapped
    pd.testing.assert_frame_equal(before.drop(columns="label"), out.drop(columns="label"))
