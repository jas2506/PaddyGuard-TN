import random

import numpy as np
import torch

from paddyguard.utils.seed import set_seed


def _draw() -> tuple[float, float, float]:
    return random.random(), float(np.random.rand()), float(torch.rand(1).item())


def test_same_seed_gives_same_draws() -> None:
    set_seed(123)
    first = _draw()
    set_seed(123)
    assert _draw() == first


def test_different_seeds_give_different_draws() -> None:
    set_seed(1)
    first = _draw()
    set_seed(2)
    assert _draw() != first
