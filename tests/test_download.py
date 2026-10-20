from pathlib import Path
from typing import Any

from paddyguard.data import download


def test_verify_layout_passes_on_a_complete_tree(data_root: Path, data_cfg: dict[str, Any]) -> None:
    checks = download.verify_layout(data_root, data_cfg)
    assert checks and all(c.ok for c in checks), [c for c in checks if not c.ok]


def test_verify_layout_reports_missing_dataset_with_instructions(
    data_root: Path, data_cfg: dict[str, Any]
) -> None:
    import shutil

    shutil.rmtree(data_root / "raw" / "dhan")
    failed = [c for c in download.verify_layout(data_root, data_cfg) if not c.ok]
    assert failed
    assert any(
        "10.17632/znsxdctwtt.1" in c.detail for c in failed
    )  # tells the user where to get it


def test_class_counts_counts_files_per_class_folder(
    data_root: Path, data_cfg: dict[str, Any]
) -> None:
    counts = download.class_counts(
        data_root / "raw" / "paddy_doctor" / "train_images", data_cfg["image_extensions"]
    )
    assert counts == {"blast": 4, "hispa": 2}
