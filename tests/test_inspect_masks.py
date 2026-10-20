from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image

from paddyguard.data import inspect_masks as im
from paddyguard.data import manifest as mf


def test_mask_stats_on_a_two_level_mask() -> None:
    arr = np.zeros((10, 10), dtype=np.uint8)
    arr[:5] = 255
    s = im.mask_stats(arr, round_to=16, hist_bins=4)
    assert s["mode"] == "L"
    assert s["rounded_values"] == {0: 50, 256: 50}  # 255 rounds to 256 with round_to=16
    assert sum(s["histogram"]) == 100 and s["histogram"][0] == 50 and s["histogram"][-1] == 50


def test_mask_stats_flags_rgb_masks_that_are_really_grey() -> None:
    grey_rgb = np.repeat(np.zeros((4, 4, 1), dtype=np.uint8), 3, axis=2)
    colour = grey_rgb.copy()
    colour[0, 0] = (255, 0, 0)
    assert im.mask_stats(grey_rgb, 16, 4)["channels_equal"] is True
    assert im.mask_stats(colour, 16, 4)["channels_equal"] is False


def test_overlay_is_photo_mask_and_blend_side_by_side() -> None:
    photo = Image.new("RGB", (32, 24), "green")
    mask = Image.new("L", (32, 24), 255)
    out = im.make_overlay(photo, mask)
    assert out.size == (3 * 32, 24)


def test_run_writes_overlays_and_a_note_without_interpreting(
    data_root: Path, data_cfg: dict[str, Any], tmp_path: Path
) -> None:
    df, _ = mf.build_manifest(data_root, data_cfg)
    cfg = data_cfg["mask_inspect"] | {"n_masks": 4, "n_overlays": 2}
    out_dir, note = tmp_path / "ov", tmp_path / "note.md"
    im.run_inspection(df, data_root, cfg, seed=0, overlay_dir=out_dir, note_path=note)
    assert len(list(out_dir.glob("*.png"))) == 2
    text = note.read_text()
    assert "TO BE FILLED" in text  # the questions are for the developer, not the script
    assert "Rounded value" in text


def test_sample_is_in_random_order_so_overlays_mix_classes() -> None:
    ids = [f"T1_{i:06d}" for i in range(80)]
    manifest = pd.DataFrame(
        {
            "image_id": ids,
            "label_raw": ["A"] * 40 + ["B"] * 40,
            "mask_path": [f"m/{i}.jpg" for i in ids],
        }
    )
    first_ten = im._sample_rows(manifest, 40, seed=0).head(10)
    assert set(first_ten["label_raw"]) == {"A", "B"}


def test_note_shows_the_developers_answers_when_the_encoding_is_configured(
    data_root: Path, data_cfg: dict[str, Any], tmp_path: Path
) -> None:
    df, _ = mf.build_manifest(data_root, data_cfg)
    cfg = data_cfg["mask_inspect"] | {"n_masks": 3, "n_overlays": 1}
    note = tmp_path / "note.md"
    im.run_inspection(
        df, data_root, cfg, 0, tmp_path / "ov", note, encoding=data_cfg["mask_encoding"]
    )
    text = note.read_text()
    assert "TO BE FILLED" not in text
    assert "lesions_only" in text and "pixel >= 128" in text and "pixel value 0" in text
