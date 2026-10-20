"""Prediction and metric helpers shared by training (M4) and cross-dataset evaluation (M5)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import balanced_accuracy_score, f1_score
from torch import nn
from torch.utils.data import DataLoader


@torch.no_grad()
def predict(
    model: nn.Module, loader: DataLoader, device: torch.device, amp: bool
) -> tuple[list[str], np.ndarray, np.ndarray]:
    """(image ids, true label indices, float32 logits) for every image in `loader`, in order."""
    model.eval()
    ids: list[str] = []
    labels, logits = [], []
    for images, y, batch_ids in loader:
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            out = model(images.to(device, non_blocking=True))
        logits.append(out.float().cpu().numpy())
        labels.append(np.asarray(y))
        ids += list(batch_ids)
    return ids, np.concatenate(labels), np.concatenate(logits).astype(np.float32)


def _recalls(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> list[float | None]:
    """Per-class recall; None for a class that has no true images (not 0)."""
    out: list[float | None] = []
    for c in range(n_classes):
        n = int((y_true == c).sum())
        out.append(float(((y_pred == c) & (y_true == c)).sum() / n) if n else None)
    return out


def classification_metrics(
    y_true: np.ndarray, logits: np.ndarray, class_names: Sequence[str]
) -> dict[str, Any]:
    """Accuracy, macro-F1 (primary), balanced accuracy and per-class recall."""
    y_pred = logits.argmax(axis=1)
    recalls = _recalls(y_true, y_pred, len(class_names))
    return {
        "n_images": len(y_true),
        "accuracy": float((y_true == y_pred).mean()),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "per_class_recall": dict(zip(class_names, recalls, strict=True)),
    }


def per_variety_macro_f1(
    ids: Sequence[str], y_true: np.ndarray, y_pred: np.ndarray, manifest: pd.DataFrame
) -> dict[str, float]:
    """Macro-F1 within each rice variety (Paddy Doctor metadata)."""
    variety = manifest.set_index("image_id")["variety"].reindex(list(ids)).to_numpy()
    out = {}
    for v in sorted({x for x in variety if isinstance(x, str)}):
        mask = variety == v
        out[v] = float(f1_score(y_true[mask], y_pred[mask], average="macro"))
    return out


def preds_frame(
    ids: Sequence[str],
    y_true: np.ndarray,
    logits: np.ndarray,
    class_names: Sequence[str],
    domain: str,
    split: str,
) -> pd.DataFrame:
    """Per-image predictions in the file format of ARCHITECTURE.md section 6."""
    names = np.asarray(class_names)
    frame = pd.DataFrame(
        {"image_id": list(ids), "y_true": names[y_true], "y_pred": names[logits.argmax(axis=1)]}
    )
    for i, c in enumerate(class_names):
        frame[f"logit_{c}"] = logits[:, i].astype(np.float32)
    frame["domain"], frame["split"] = domain, split
    return frame
