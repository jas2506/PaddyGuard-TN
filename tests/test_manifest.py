from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from conftest import IMG_SIZE, make_tree
from paddyguard.data import manifest as mf


@pytest.fixture
def built(data_root: Path, data_cfg: dict[str, Any]) -> tuple[pd.DataFrame, list]:
    return mf.build_manifest(data_root, data_cfg)


def test_columns_and_counts(built: tuple[pd.DataFrame, list]) -> None:
    df, checks = built
    assert list(df.columns) == mf.MANIFEST_COLUMNS
    assert df["dataset"].value_counts().to_dict() == {"T2": 7, "S": 6, "T1": 5}
    assert all(c.ok for c in checks), [c for c in checks if not c.ok]


def test_ids_are_unique_prefixed_and_deterministic(
    built: tuple[pd.DataFrame, list], data_root: Path, data_cfg: dict[str, Any]
) -> None:
    df, _ = built
    assert df["image_id"].is_unique
    assert all(i.startswith(d + "_") for i, d in zip(df["image_id"], df["dataset"], strict=True))
    assert df["image_id"].iloc[0] == "S_000000"
    again, _ = mf.build_manifest(data_root, data_cfg)
    pd.testing.assert_frame_equal(df, again)


def test_paths_are_relative_to_the_data_root_and_exist(
    built: tuple[pd.DataFrame, list], data_root: Path
) -> None:
    df, _ = built
    for rel in list(df["path"]) + list(df["mask_path"].dropna()):
        assert not Path(rel).is_absolute()
        assert (data_root / rel).is_file()


def test_size_columns_match_the_image(built: tuple[pd.DataFrame, list]) -> None:
    df, _ = built
    assert (df["width"] == IMG_SIZE[0]).all() and (df["height"] == IMG_SIZE[1]).all()


def test_paddy_doctor_rows_carry_variety_and_age_but_no_harmonised_label(
    built: tuple[pd.DataFrame, list],
) -> None:
    s = built[0].query("dataset == 'S'")
    assert set(s["variety"]) == {"ADT45"} and s["age_days"].notna().all()
    assert set(s["label_raw"]) == {"blast", "hispa"}
    assert s["label"].isna().all()  # harmonised in M3, not here
    other = built[0].query("dataset != 'S'")
    assert other["variety"].isna().all() and other["age_days"].isna().all()


def test_sethy_masks_pair_by_stem_ignoring_extension_case(built: tuple[pd.DataFrame, list]) -> None:
    t1 = built[0].query("dataset == 'T1'")
    assert t1["mask_path"].notna().all()
    assert (
        t1["path"].str.rsplit(".", n=1).str[0].str.rsplit("/", n=1).str[1]
        == t1["mask_path"].str.rsplit(".", n=1).str[0].str.rsplit("/", n=1).str[1]
    ).all()
    assert built[0].query("dataset != 'T1'")["mask_path"].isna().all()


def test_unmatched_sethy_files_are_reported(tmp_path: Path, data_cfg: dict[str, Any]) -> None:
    root = tmp_path / "d2"
    make_tree(root, extra_unmatched=True)
    data_cfg["datasets"]["T1"]["expected_count"] = 6
    data_cfg["datasets"]["RiceSeg"]["expected_count"] = 6
    df, checks = mf.build_manifest(root, data_cfg)
    pairing = next(c for c in checks if c.name == "Sethy-RiceSeg pairing")
    assert not pairing.ok
    assert "BLAST9_999" in pairing.detail and "ORPHAN_001" in pairing.detail
    assert df.query("dataset == 'T1' and mask_path.isna()").shape[0] == 1


def test_dhan_background_comes_from_the_folder_and_matches_the_csv(
    built: tuple[pd.DataFrame, list],
) -> None:
    t2 = built[0].query("dataset == 'T2'")
    assert t2["background"].value_counts().to_dict() == {"field": 5, "white": 2}
    assert set(t2["label_raw"]) == {"Rice Blast", "Browon Spot"}
    assert built[0].query("dataset != 'T2'")["background"].isna().all()


def test_dhan_csv_disagreement_is_reported(tmp_path: Path, data_cfg: dict[str, Any]) -> None:
    root = tmp_path / "d3"
    make_tree(root, dhan_csv_wrong_background=True)
    _, checks = mf.build_manifest(root, data_cfg)
    assert not next(c for c in checks if c.name == "Dhan info CSV agrees with folders").ok


def test_wrong_expected_count_is_reported(data_root: Path, data_cfg: dict[str, Any]) -> None:
    data_cfg["datasets"]["S"]["expected_count"] = 10407
    _, checks = mf.build_manifest(data_root, data_cfg)
    failed = [c for c in checks if not c.ok]
    assert [c.name for c in failed] == ["S image count"]
    assert "10407" in failed[0].detail and "6" in failed[0].detail


def test_debug_subsamples_each_dataset_deterministically(
    data_root: Path, data_cfg: dict[str, Any]
) -> None:
    data_cfg["debug"]["n_images"] = 3
    df, _ = mf.build_manifest(data_root, data_cfg, debug=True)
    assert df["dataset"].value_counts().to_dict() == {"S": 3, "T1": 3, "T2": 3}
    again, _ = mf.build_manifest(data_root, data_cfg, debug=True)
    assert list(df["path"]) == list(again["path"])


def test_write_manifest_roundtrip(built: tuple[pd.DataFrame, list], tmp_path: Path) -> None:
    out = tmp_path / "m" / "manifest.csv"
    mf.write_manifest(built[0], out)
    back = pd.read_csv(out)
    assert list(back.columns) == mf.MANIFEST_COLUMNS and len(back) == len(built[0])


def test_has_m2_columns_detects_hashes_groups_and_splits(
    built: tuple[pd.DataFrame, list], tmp_path: Path
) -> None:
    path = tmp_path / "manifest.csv"
    assert not mf.has_m2_columns(path)  # no file
    mf.write_manifest(built[0], path)
    assert not mf.has_m2_columns(path)  # fresh M1 output
    df = built[0].copy()
    df["split"] = "train"
    mf.write_manifest(df, path)
    assert mf.has_m2_columns(path)


def test_hex_phash_survives_a_csv_roundtrip_as_text(
    built: tuple[pd.DataFrame, list], tmp_path: Path
) -> None:
    df = built[0].copy()
    df["phash"] = "0000000000001234"  # looks like a number, must stay text with leading zeros
    path = tmp_path / "m.csv"
    mf.write_manifest(df, path)
    assert set(mf.read_manifest(path)["phash"]) == {"0000000000001234"}
