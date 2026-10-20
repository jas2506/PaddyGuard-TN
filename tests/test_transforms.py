from typing import Any

import numpy as np
import pytest
import torch

from paddyguard.data import transforms


def _img(seed: int = 0, size: tuple[int, int] = (60, 40)) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 256, (size[1], size[0], 3), dtype=np.uint8)


@pytest.mark.parametrize("name", ["baseline", "strong", "strong_mixstyle", "eval"])
def test_every_transform_gives_a_normalised_chw_float_tensor(
    tf_cfg: dict[str, Any], name: str
) -> None:
    out = transforms.build_transform(name, tf_cfg, seed=0)(image=_img())["image"]
    assert isinstance(out, torch.Tensor) and out.dtype == torch.float32
    assert tuple(out.shape) == (3, tf_cfg["img_size"], tf_cfg["img_size"])


def test_eval_is_deterministic_and_train_transforms_are_seeded(tf_cfg: dict[str, Any]) -> None:
    ev = transforms.build_transform("eval", tf_cfg)
    assert torch.equal(ev(image=_img())["image"], ev(image=_img())["image"])
    a = transforms.build_transform("strong", tf_cfg, seed=3)(image=_img())["image"]
    b = transforms.build_transform("strong", tf_cfg, seed=3)(image=_img())["image"]
    assert torch.equal(a, b)


def test_strong_mixstyle_uses_the_strong_pipeline(tf_cfg: dict[str, Any]) -> None:
    a = transforms.build_transform("strong", tf_cfg, seed=1)(image=_img())["image"]
    b = transforms.build_transform("strong_mixstyle", tf_cfg, seed=1)(image=_img())["image"]
    assert torch.equal(a, b)


def test_masks_follow_the_image_geometry(tf_cfg: dict[str, Any]) -> None:
    mask = (np.random.default_rng(1).random((40, 60)) > 0.5).astype(np.uint8)
    for name in ("eval", "strong"):
        out = transforms.build_transform(name, tf_cfg, seed=0)(image=_img(), mask=mask)
        assert tuple(out["mask"].shape) == (tf_cfg["img_size"], tf_cfg["img_size"])
        assert set(out["mask"].unique().tolist()) <= {0, 1}  # nearest-neighbour, no blending


def test_unknown_transform_name_is_rejected(tf_cfg: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="nope"):
        transforms.build_transform("nope", tf_cfg)
