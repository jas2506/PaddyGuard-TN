"""Classifier backbones (M4). MixStyle hooks are added in M5."""

from __future__ import annotations

import timm
from torch import nn


def create_model(name: str, num_classes: int, pretrained: bool) -> nn.Module:
    """A timm model with a fresh `num_classes` head (EfficientNet-B0 is the main model)."""
    return timm.create_model(name, pretrained=pretrained, num_classes=num_classes)
