from typing import Any

import pandas as pd
import pytest
import torch

from paddyguard.data import datasets, transforms


@pytest.fixture
def ev(tf_cfg: dict[str, Any]):
    return transforms.build_transform("eval", tf_cfg)


def _ds(prepared: pd.DataFrame, data_root, domain: str, split, ev, **kw):
    return datasets.PaddyDataset(prepared, split, domain, ev, data_root=data_root, **kw)


def test_item_is_image_label_index_and_id(prepared, data_root, ev, tf_cfg) -> None:
    ds = _ds(prepared, data_root, "S", "train", ev)
    image, label, image_id = ds[0]
    assert tuple(image.shape) == (3, tf_cfg["img_size"], tf_cfg["img_size"])
    assert isinstance(label, int) and image_id.startswith("S_")
    assert len(ds) == 3 and ds.class_names[label] in {"blast", "hispa"}


def test_default_classes_are_the_ten_source_classes(prepared, data_root, ev) -> None:
    ds = _ds(prepared, data_root, "S", "train", ev)
    assert len(ds.class_names) == 10
    assert ds.class_names[ds[0][1]] == prepared.set_index("image_id").loc[ds[0][2], "label"]


def test_class_subset_reindexes_and_skips_other_classes(prepared, data_root, ev) -> None:
    ds = _ds(prepared, data_root, "S", None, ev, classes=["tungro", "blast"])
    assert ds.class_names == ["tungro", "blast"]
    assert {ds[i][1] for i in range(len(ds))} == {1}  # only blast images (hispa skipped)
    assert len(ds) == 4  # all four blast images of the synthetic Paddy Doctor set


def test_domains_pick_the_right_rows(prepared, data_root, ev) -> None:
    t2 = _ds(prepared, data_root, "T2", "test", ev, classes=["blast", "brown_spot", "tungro"])
    white = _ds(prepared, data_root, "T2-white", "zero_shot", ev, classes=["blast"])
    assert {i.split("_")[0] for i in t2.image_ids + white.image_ids} == {"T2"}
    assert not set(t2.image_ids) & set(white.image_ids)
    assert len(white) == 2


def test_dropped_rows_are_never_returned(prepared, data_root, ev) -> None:
    prepared.loc[prepared["image_id"] == "S_000000", "split"] = "dropped_duplicate"
    ds = _ds(prepared, data_root, "S", None, ev)
    assert "S_000000" not in ds.image_ids


def test_representatives_only_keeps_one_image_per_group(prepared, data_root, ev) -> None:
    t1 = prepared["dataset"] == "T1"
    prepared.loc[t1, "group_id"] = prepared.loc[t1, "image_id"].iloc[0]  # one big group
    all_t1 = _ds(prepared, data_root, "T1", None, ev)
    reps = _ds(prepared, data_root, "T1", None, ev, representatives_only=True)
    assert len(all_t1) == 5 and len(reps) == 1


def test_with_mask_returns_a_binary_mask_of_the_image_size(prepared, data_root, ev, tf_cfg) -> None:
    ds = _ds(prepared, data_root, "T1", "seg_train", ev, with_mask=True, mask_threshold=128)
    image, _, _, mask = ds[0]
    assert tuple(mask.shape) == tuple(image.shape[1:])
    assert set(mask.unique().tolist()) <= {0, 1} and mask.sum() > 0


def test_with_mask_needs_t1_and_a_threshold(prepared, data_root, ev) -> None:
    with pytest.raises(ValueError, match="threshold"):
        _ds(prepared, data_root, "T1", "seg_train", ev, with_mask=True)
    with pytest.raises(ValueError, match="T1"):
        _ds(prepared, data_root, "S", "train", ev, with_mask=True, mask_threshold=128)


def test_unknown_split_or_domain_is_rejected(prepared, data_root, ev) -> None:
    with pytest.raises(ValueError, match="nope"):
        _ds(prepared, data_root, "S", "nope", ev)
    with pytest.raises(ValueError, match="X9"):
        _ds(prepared, data_root, "X9", None, ev)


def test_training_loader_only_accepts_paddy_doctor_train(prepared, data_root, ev) -> None:
    for domain, split in [("T1", "seg_train"), ("T2", "test"), ("S", "val"), ("S", "test")]:
        ds = _ds(prepared, data_root, domain, split, ev)
        with pytest.raises(ValueError, match="training"):
            datasets.make_loader(ds, training=True, batch_size=2, seed=0)
    ok = _ds(prepared, data_root, "S", "train", ev)
    loader = datasets.make_loader(ok, training=True, batch_size=2, seed=0)
    images, labels, ids = next(iter(loader))
    assert images.shape[0] == 2 and labels.dtype == torch.int64 and len(ids) == 2


def test_loader_order_is_reproducible_per_seed_and_eval_loader_is_ordered(
    prepared, data_root, ev
) -> None:
    ds = _ds(prepared, data_root, "S", None, ev)
    run = lambda seed: [i for _, _, ids in datasets.make_loader(ds, False, 1, seed) for i in ids]
    assert run(0) == ds.image_ids  # evaluation: fixed order
    ds_tr = _ds(prepared, data_root, "S", "train", ev)
    order = lambda seed: [
        i for _, _, ids in datasets.make_loader(ds_tr, True, 1, seed, num_workers=0) for i in ids
    ]
    assert order(5) == order(5)
