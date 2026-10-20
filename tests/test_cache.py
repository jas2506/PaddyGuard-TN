from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image

from paddyguard.data import cache
from paddyguard.data import manifest as mf


def _processed(data_root: Path, dataset: str, image_id: str, suffix: str = ".jpg") -> Path:
    return data_root / "processed" / dataset / f"{image_id}{suffix}"


def test_cache_resizes_s_and_t2_and_keeps_t1_native_with_masks(
    data_root: Path, data_cfg: dict[str, Any]
) -> None:
    # make one Sethy file a PNG named .jpg (as ~7% of the real ones are) and rotate one Dhan photo
    t1_first = min((data_root / "raw/sethy/images/Blast").iterdir())
    Image.open(t1_first).convert("RGB").save(t1_first, format="PNG")
    t2_first = min((data_root / "raw/dhan/Field Background/Rice Blast").iterdir())
    exif = Image.Exif()
    exif[274] = 6
    Image.open(t2_first).save(t2_first, quality=95, exif=exif)

    df, _ = mf.build_manifest(data_root, data_cfg)
    data_cfg["cache"]["short_side"] = 16
    cache.build_cache(df, data_root, data_cfg)

    s = df[df.dataset == "S"].iloc[0]
    with Image.open(_processed(data_root, "S", s.image_id)) as im:
        assert min(im.size) == 16 and im.format == "JPEG"  # 32x24 -> short side 16
    t1 = df[df.dataset == "T1"]
    for r in t1.itertuples():
        with Image.open(_processed(data_root, "T1", r.image_id)) as im:
            assert im.size == (r.width, r.height) and im.format == "JPEG"  # native, never resized
        with Image.open(_processed(data_root, "T1", r.image_id, "_mask.png")) as m:
            assert m.size == (r.width, r.height) and m.mode == "L" and m.format == "PNG"
    t2 = df[(df.dataset == "T2") & (df.path == t2_first.relative_to(data_root).as_posix())].iloc[0]
    with Image.open(_processed(data_root, "T2", t2.image_id)) as im:
        assert im.size[0] < im.size[1]  # EXIF rotation applied: the 32x24 photo is now portrait


def test_cache_skips_dropped_rows_and_is_idempotent(
    data_root: Path, data_cfg: dict[str, Any]
) -> None:
    df, _ = mf.build_manifest(data_root, data_cfg)
    dropped = df.index[df.dataset == "T1"][0]
    df["split"] = None
    df.loc[dropped, "split"] = "dropped_duplicate"
    cache.build_cache(df, data_root, data_cfg)
    assert not _processed(data_root, "T1", df.loc[dropped, "image_id"]).exists()
    first = _processed(data_root, "S", df[df.dataset == "S"].iloc[0].image_id)
    stamp = first.stat().st_mtime_ns
    cache.build_cache(df, data_root, data_cfg)  # second run does not rewrite existing files
    assert first.stat().st_mtime_ns == stamp
    assert isinstance(df, pd.DataFrame)
