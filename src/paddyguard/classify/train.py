"""Train and evaluate the disease classifier on Paddy Doctor (M4).

    python -m paddyguard.classify.train --config configs/cls_effb0.yaml --seed 0 [--debug] [--resume RUN_DIR]

Loop: train epoch -> validate on S/val (macro-F1) -> checkpoint -> early stopping on val macro-F1
-> reload the best weights -> evaluate on S/test. Outputs go to the run directory:
metrics.json, preds_S_test.parquet, ckpt.pt (best weights + class names) and ckpt_last.pt
(everything needed to resume).
"""

from __future__ import annotations

import argparse
import copy
import math
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from paddyguard.classify import evaluate
from paddyguard.classify.models import create_model
from paddyguard.data import transforms
from paddyguard.data.datasets import PaddyDataset, make_loader
from paddyguard.data.labelmap import load_labelmap
from paddyguard.data.manifest import read_manifest
from paddyguard.utils import io, paths
from paddyguard.utils.logging import get_logger
from paddyguard.utils.seed import set_seed

BEST, LAST, METRICS, PREDS_TEST = "ckpt.pt", "ckpt_last.pt", "metrics.json", "preds_S_test.parquet"
SPLITS = ("train", "val", "test")


def lr_factor(step: int, warmup_steps: int, total_steps: int) -> float:
    """Linear warm-up to 1, then cosine decay to 0."""
    if step < warmup_steps:
        return (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))


class EarlyStopper:
    """Tracks the best value of a metric (higher is better) and stops after `patience` misses."""

    def __init__(self, patience: int) -> None:
        self.patience, self.best, self.best_epoch, self.bad_epochs = patience, -math.inf, -1, 0

    def update(self, value: float, epoch: int) -> bool:
        improved = value > self.best
        if improved:
            self.best, self.best_epoch, self.bad_epochs = value, epoch, 0
        else:
            self.bad_epochs += 1
        return improved

    @property
    def should_stop(self) -> bool:
        return self.bad_epochs >= self.patience


def transform_config_for(cfg: Mapping[str, Any]) -> dict[str, Any]:
    """configs/transforms.yaml with the classifier's img_size (resize keeps the 256/224 ratio)."""
    tf = transforms.load_transform_config()
    return {
        **tf,
        "img_size": cfg["img_size"],
        "resize": round(cfg["img_size"] * tf["resize"] / tf["img_size"]),
    }


def debug_subset(manifest: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """Keep at most n random Paddy Doctor images in each of train/val/test."""
    kept = manifest[(manifest["dataset"] == "S") & manifest["split"].isin(SPLITS)]
    parts = [g.sample(n=min(n, len(g)), random_state=seed) for _, g in kept.groupby("split")]
    return pd.concat(parts)


def build_loaders(
    cfg: Mapping[str, Any], manifest: pd.DataFrame, data_root: Path, device: torch.device
) -> dict[str, DataLoader]:
    tf_cfg, seed = transform_config_for(cfg), cfg["seed"]
    train_tf = transforms.build_transform(cfg["augmentation"], tf_cfg, seed=seed)
    eval_tf = transforms.build_transform("eval", tf_cfg)
    loaders = {}
    for split in SPLITS:
        ds = PaddyDataset(
            manifest, split, "S", train_tf if split == "train" else eval_tf, data_root=data_root
        )
        loaders[split] = make_loader(
            ds, training=split == "train", batch_size=cfg["batch_size"], seed=seed,
            num_workers=cfg["num_workers"], pin_memory=device.type == "cuda",
        )  # fmt: skip
    return loaders


def train_one_epoch(model, loader, optimizer, scheduler, scaler, loss_fn, device, amp) -> float:
    model.train()
    total, n = 0.0, 0
    for images, y, _ in loader:
        images, y = images.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            loss = loss_fn(model(images), y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        total, n = total + loss.item() * len(y), n + len(y)
    return total / n


def _acceptance(test: Mapping[str, Any], cfg: Mapping[str, Any]) -> dict[str, Any]:
    acc = cfg["acceptance"]
    recalls = {k: v for k, v in test["per_class_recall"].items() if v is not None}
    worst = min(recalls, key=recalls.get)
    f1_ok = test["macro_f1"] > acc["min_macro_f1"]
    recall_ok = recalls[worst] >= acc["min_class_recall"]
    return {
        "macro_f1_ok": bool(f1_ok), "min_class_recall_ok": bool(recall_ok),
        "lowest_recall_class": worst, "lowest_recall": recalls[worst],
        "passed": bool(f1_ok and recall_ok),
    }  # fmt: skip


def _evaluate_split(model, loader, cfg, device, class_names, split, manifest):
    ids, y, logits = evaluate.predict(model, loader, device, cfg["amp"] and device.type == "cuda")
    metrics = evaluate.classification_metrics(y, logits, class_names)
    if split == "test":
        metrics["per_variety_macro_f1"] = evaluate.per_variety_macro_f1(
            ids, y, logits.argmax(1), manifest
        )
    return metrics, evaluate.preds_frame(ids, y, logits, class_names, "S", split)


def run(
    cfg: Mapping[str, Any],
    manifest: pd.DataFrame,
    data_root: Path,
    run_dir: Path,
    device: torch.device,
    resume: bool = False,
) -> dict[str, Any]:
    """Train, then evaluate the best checkpoint on S/test. Returns the metrics dictionary."""
    run_dir.mkdir(parents=True, exist_ok=True)
    log = get_logger(f"paddyguard.train.{run_dir.name}", run_dir / "log.txt")
    set_seed(cfg["seed"])
    class_names = load_labelmap().source_classes
    loaders = build_loaders(cfg, manifest, data_root, device)
    model = create_model(cfg["model"], len(class_names), cfg["pretrained"]).to(device)
    optimizer = AdamW(
        model.parameters(), lr=cfg["optimizer"]["lr"], weight_decay=cfg["optimizer"]["weight_decay"]
    )
    steps = len(loaders["train"])
    warmup, total = cfg["scheduler"]["warmup_epochs"] * steps, cfg["epochs"] * steps
    scheduler = LambdaLR(optimizer, lambda s: lr_factor(s, warmup, total))
    amp = bool(cfg["amp"]) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=cfg["loss"]["label_smoothing"])
    stopper, history, start = EarlyStopper(cfg["early_stopping"]["patience"]), [], 0
    metric_key = cfg["early_stopping"]["metric"]

    if resume and (run_dir / LAST).is_file():
        state = torch.load(run_dir / LAST, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        stopper.__dict__.update(state["stopper"])
        history, start = state["history"], state["epoch"]
        log.info("resumed from epoch %d", start)

    for epoch in range(start, cfg["epochs"]):
        if stopper.should_stop:
            break
        t0 = time.time()
        loss = train_one_epoch(
            model, loaders["train"], optimizer, scheduler, scaler, loss_fn, device, amp
        )
        val, _ = _evaluate_split(model, loaders["val"], cfg, device, class_names, "val", manifest)
        row = {"epoch": epoch + 1, "train_loss": loss, "val_macro_f1": val["macro_f1"],
               "val_accuracy": val["accuracy"], "lr": scheduler.get_last_lr()[0],
               "seconds": round(time.time() - t0, 1)}  # fmt: skip
        history.append(row)
        if stopper.update(row[metric_key], epoch):
            torch.save({"model": model.state_dict(), "class_names": class_names, "config": dict(cfg),
                        "epoch": epoch + 1, metric_key: row[metric_key]}, run_dir / BEST)  # fmt: skip
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
                    "stopper": copy.deepcopy(stopper.__dict__), "history": history,
                    "epoch": epoch + 1}, run_dir / LAST)  # fmt: skip
        log.info("epoch %d/%d loss %.4f val macro-F1 %.4f acc %.4f", epoch + 1, cfg["epochs"], loss,
                 row["val_macro_f1"], row["val_accuracy"])  # fmt: skip

    model.load_state_dict(
        torch.load(run_dir / BEST, map_location=device, weights_only=False)["model"]
    )
    val, _ = _evaluate_split(model, loaders["val"], cfg, device, class_names, "val", manifest)
    test, preds = _evaluate_split(
        model, loaders["test"], cfg, device, class_names, "test", manifest
    )
    metrics = {
        "run": {"seed": cfg["seed"], "model": cfg["model"], "augmentation": cfg["augmentation"],
                "device": device.type, "git_commit": io.git_commit(), "classes": class_names},
        "best_epoch": stopper.best_epoch + 1, "history": history, "val": val, "test": test,
        "acceptance": _acceptance(test, cfg),
    }  # fmt: skip
    io.save_json(metrics, run_dir / METRICS)
    io.save_parquet(preds, run_dir / PREDS_TEST)
    log.info("test macro-F1 %.4f balanced acc %.4f; acceptance %s", test["macro_f1"],
             test["balanced_accuracy"], metrics["acceptance"])  # fmt: skip
    return metrics


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, help="overrides the seed in the config")
    parser.add_argument("--debug", action="store_true", help="few images, 1 epoch, CPU")
    parser.add_argument(
        "--no-pretrained", action="store_true", help="random init (offline debugging)"
    )
    parser.add_argument(
        "--resume", metavar="RUN_DIR", help="continue an interrupted run in this directory"
    )
    args = parser.parse_args(argv)

    cfg = io.load_yaml(args.config)
    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.no_pretrained:
        cfg["pretrained"] = False
    data_cfg = io.load_yaml(paths.configs_dir() / "data.yaml")
    manifest = read_manifest(paths.data_root() / data_cfg["manifest_path"])
    device = torch.device("cuda" if torch.cuda.is_available() and not args.debug else "cpu")
    if args.debug:
        cfg |= {"epochs": 1, "num_workers": 0}
        manifest = debug_subset(manifest, data_cfg["debug"]["n_images"], cfg["seed"])
    run_dir = (
        Path(args.resume)
        if args.resume
        else io.new_run_dir(Path(args.config).stem, cfg["seed"], config=cfg)
    )
    run(cfg, manifest, paths.data_root(), run_dir, device, resume=bool(args.resume))
    return 0


if __name__ == "__main__":
    sys.exit(main())
