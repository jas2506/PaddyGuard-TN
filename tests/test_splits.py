import numpy as np
import pandas as pd
import pytest

from paddyguard.data import dedup, splits


def _synthetic(n: int = 600) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    labels = rng.choice(["a", "b", "c", "d"], n, p=[0.4, 0.3, 0.2, 0.1])
    groups = [f"S_{i // 2:06d}" for i in range(n)]  # pairs of near-duplicates share a group
    return pd.DataFrame(
        {
            "image_id": [f"S_{i:06d}" for i in range(n)],
            "dataset": "S",
            "label_raw": labels,
            "group_id": groups,
            "background": None,
            "split": None,
        }
    )


RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}


def test_group_split_has_requested_ratios_and_no_group_in_two_splits() -> None:
    df = _synthetic()
    out = splits.group_split(df, RATIOS, n_folds=20, seed=42)
    share = out["split"].value_counts(normalize=True)
    for name, r in RATIOS.items():
        assert share[name] == pytest.approx(r, abs=0.04)
    assert (out.groupby("group_id")["split"].nunique() == 1).all()


def test_group_split_is_stratified_and_deterministic() -> None:
    df = _synthetic()
    a = splits.group_split(df, RATIOS, n_folds=20, seed=42)
    b = splits.group_split(df, RATIOS, n_folds=20, seed=42)
    assert a["split"].tolist() == b["split"].tolist()
    assert (a.groupby("label_raw")["split"].nunique() == 3).all()  # every class in every split


def test_ratios_that_do_not_fit_the_folds_are_rejected() -> None:
    with pytest.raises(ValueError):
        splits.group_split(_synthetic(), RATIOS, n_folds=7, seed=0)


def test_assign_all_splits_handles_each_dataset_and_leaves_dropped_rows_alone() -> None:
    s = _synthetic(200)
    t1 = pd.DataFrame(
        {
            "image_id": [f"T1_{i:06d}" for i in range(100)],
            "dataset": "T1",
            "label_raw": ["x", "y"] * 50,
            "group_id": [f"T1_{i:06d}" for i in range(100)],
            "background": None,
            "split": None,
        }
    )
    t1.loc[0, "split"] = dedup.DROPPED_DUPLICATE
    t2 = pd.DataFrame(
        {
            "image_id": [f"T2_{i:06d}" for i in range(100)],
            "dataset": "T2",
            "label_raw": ["p", "q"] * 50,
            "group_id": [f"T2_{i:06d}" for i in range(100)],
            "background": ["field"] * 60 + ["white"] * 40,
            "split": None,
        }
    )
    cfg = {
        "split_seed": 42,
        "split_folds": {"S": 20, "T1": 20, "T2": 5},
        "splits": {
            "S": RATIOS,
            "T1": {"seg_train": 0.7, "seg_val": 0.15, "seg_test": 0.15},
            "T2": {"cal_pool": 0.4, "test": 0.6},
        },
    }
    out = splits.assign_all_splits(pd.concat([s, t1, t2], ignore_index=True), cfg)
    assert out.loc[out.image_id == "T1_000000", "split"].item() == dedup.DROPPED_DUPLICATE
    t1o = out[(out.dataset == "T1") & (out.split != dedup.DROPPED_DUPLICATE)]
    assert set(t1o["split"]) == {"seg_train", "seg_val", "seg_test"}
    t2o = out[out.dataset == "T2"]
    assert set(t2o[t2o.background == "white"]["split"]) == {splits.ZERO_SHOT}
    field = t2o[t2o.background == "field"]
    assert set(field["split"]) == {"cal_pool", "test"}
    assert (field["split"] == "cal_pool").mean() == pytest.approx(0.4, abs=0.1)


def test_resplitting_is_refused() -> None:
    df = splits.group_split(_synthetic(100), RATIOS, n_folds=20, seed=42)
    with pytest.raises(RuntimeError):
        splits.ensure_not_split(df)
    splits.ensure_not_split(_synthetic(100))  # unsplit data is fine


@pytest.mark.parametrize(
    ("pool", "expected"),
    [(80, [25, 50]), (135, [25, 50]), (153, [25, 50]), (154, [25, 50, 100]), (30, [])],
)
def test_allowed_cal_sizes(pool: int, expected: list[int]) -> None:
    assert splits.allowed_cal_sizes(pool, [25, 50, 100], 0.65) == expected


def test_seg_test_keeps_one_image_per_group_but_training_keeps_all_members() -> None:
    n = 400
    df = pd.DataFrame(
        {
            "image_id": [f"T1_{i:06d}" for i in range(n)],
            "dataset": "T1",
            "label_raw": ["x", "y"] * (n // 2),
            "group_id": [f"T1_{(i // 4) * 4:06d}" for i in range(n)],  # groups of four
            "background": None,
            "split": None,
        }
    )
    cfg = {
        "split_seed": 42,
        "split_folds": {"S": 20, "T1": 20, "T2": 5},
        "splits": {"T1": {"seg_train": 0.7, "seg_val": 0.15, "seg_test": 0.15}},
    }
    out = splits.assign_all_splits(df, cfg)
    test = out[out["split"] == "seg_test"]
    assert len(test) > 0 and (test["image_id"] == test["group_id"]).all()
    train = out[out["split"] == "seg_train"]
    assert (train["image_id"] != train["group_id"]).any()  # extra members stay in training
    dropped = out[out["split"] == dedup.DROPPED_DUPLICATE]
    assert len(dropped) > 0 and (dropped["image_id"] != dropped["group_id"]).all()
    assert (
        out.groupby("group_id")["split"].agg(
            lambda s: s.replace(dedup.DROPPED_DUPLICATE, "seg_test").nunique()
        )
        == 1
    ).all()
