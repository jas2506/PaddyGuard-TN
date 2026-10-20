from pathlib import Path
from typing import Any

from paddyguard.data import manifest as mf
from paddyguard.data.dataset_card import render_dataset_card


def test_card_lists_counts_sizes_checks_and_unchecked_licences(
    data_root: Path, data_cfg: dict[str, Any]
) -> None:
    data_cfg["licences"] = dict.fromkeys(data_cfg["licences"])  # none checked yet
    df, checks = mf.build_manifest(data_root, data_cfg)
    card = render_dataset_card(df, checks, data_cfg)
    assert "Paddy Doctor" in card and "| blast | 4 |" in card
    assert "32×24" in card  # size distribution
    assert "NOT YET CHECKED" in card
    assert "field" in card and "white" in card
    assert "FAIL" not in card


def test_card_copies_a_licence_when_given(data_root: Path, data_cfg: dict[str, Any]) -> None:
    data_cfg["licences"]["T1"] = "CC BY 4.0 (copied from the page)"
    df, checks = mf.build_manifest(data_root, data_cfg)
    assert "CC BY 4.0 (copied from the page)" in render_dataset_card(df, checks, data_cfg)


def test_card_shows_failed_checks(data_root: Path, data_cfg: dict[str, Any]) -> None:
    data_cfg["datasets"]["S"]["expected_count"] = 10407
    df, checks = mf.build_manifest(data_root, data_cfg)
    assert "FAIL" in render_dataset_card(df, checks, data_cfg)
