import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from paddyguard.classify import evaluate, train
from paddyguard.data import labelmap
from paddyguard.data import manifest as mf
from paddyguard.utils import io, paths

CLASSES = ["a", "b", "c"]


def test_lr_factor_warms_up_then_decays_to_zero() -> None:
    f = [train.lr_factor(s, warmup_steps=4, total_steps=20) for s in range(20)]
    assert f[0] < f[1] < f[2] < f[3] == pytest.approx(1.0)
    assert all(x >= y for x, y in itertools.pairwise(f[3:]))
    assert f[-1] < 0.01
    assert train.lr_factor(0, warmup_steps=0, total_steps=10) == pytest.approx(1.0)


def test_early_stopper_tracks_the_best_epoch_and_stops_after_patience() -> None:
    es = train.EarlyStopper(patience=2)
    improved = [es.update(v, epoch=i) for i, v in enumerate([0.5, 0.6, 0.55, 0.54])]
    assert improved == [True, True, False, False]
    assert es.best == 0.6 and es.best_epoch == 1 and es.should_stop


def test_classification_metrics_on_known_predictions() -> None:
    y = np.array([0, 0, 1, 1, 2, 2])
    logits = np.eye(3, dtype=np.float32)[[0, 0, 1, 2, 2, 2]]  # one class-1 image predicted as 2
    m = evaluate.classification_metrics(y, logits, CLASSES)
    assert m["accuracy"] == pytest.approx(5 / 6)
    assert m["per_class_recall"] == {"a": 1.0, "b": 0.5, "c": 1.0}
    assert m["balanced_accuracy"] == pytest.approx((1 + 0.5 + 1) / 3)
    assert 0.8 < m["macro_f1"] < 0.9


def test_recall_of_an_absent_class_is_none_not_zero() -> None:
    m = evaluate.classification_metrics(
        np.array([0, 0]), np.eye(3, dtype=np.float32)[[0, 0]], CLASSES
    )
    assert m["per_class_recall"]["c"] is None


def test_preds_frame_follows_the_interface_contract() -> None:
    logits = np.eye(3, dtype=np.float32)[[0, 2]]
    df = evaluate.preds_frame(["i1", "i2"], np.array([0, 1]), logits, CLASSES, "S", "test")
    assert list(df.columns) == [
        "image_id", "y_true", "y_pred", "logit_a", "logit_b", "logit_c", "domain", "split",
    ]  # fmt: skip
    assert df["y_true"].tolist() == ["a", "b"] and df["y_pred"].tolist() == ["a", "c"]
    assert df["logit_a"].dtype == np.float32 and set(df["domain"]) == {"S"}


def test_per_variety_macro_f1() -> None:
    manifest = pd.DataFrame(
        {"image_id": ["i1", "i2", "i3", "i4"], "variety": ["V1", "V1", "V2", "V2"]}
    )
    out = evaluate.per_variety_macro_f1(
        ["i1", "i2", "i3", "i4"], np.array([0, 1, 0, 1]), np.array([0, 1, 1, 1]), manifest
    )
    assert out["V1"] == pytest.approx(1.0) and out["V2"] < 1.0


@pytest.fixture
def tiny_cfg() -> dict[str, Any]:
    cfg = io.load_yaml(paths.configs_dir() / "cls_effb0.yaml")
    cfg |= {"img_size": 32, "epochs": 2, "batch_size": 2, "num_workers": 0, "pretrained": False}
    cfg["scheduler"] = {"name": "cosine", "warmup_epochs": 1}
    return cfg


def _run(cfg, prepared, data_root, run_dir, resume=False) -> dict[str, Any]:
    return train.run(cfg, prepared, data_root, run_dir, device=torch.device("cpu"), resume=resume)


def test_run_trains_and_writes_every_output(tiny_cfg, prepared, data_root, tmp_path: Path) -> None:
    metrics = _run(tiny_cfg, prepared, data_root, tmp_path / "run")
    run = tmp_path / "run"
    for name in ("metrics.json", "preds_S_test.parquet", "ckpt.pt", "ckpt_last.pt", "log.txt"):
        assert (run / name).is_file(), name
    saved = json.loads((run / "metrics.json").read_text())
    assert saved["history"] and len(saved["history"]) == tiny_cfg["epochs"]
    assert {"macro_f1", "balanced_accuracy", "per_class_recall", "per_variety_macro_f1"} <= set(
        saved["test"]
    )
    assert {"macro_f1", "accuracy"} <= set(saved["val"])
    assert set(saved["acceptance"]) >= {"macro_f1_ok", "min_class_recall_ok", "passed"}
    preds = pd.read_parquet(run / "preds_S_test.parquet")
    classes = labelmap.load_labelmap().source_classes
    assert all(f"logit_{c}" in preds.columns for c in classes)  # one logit per training class
    assert set(preds["split"]) == {"test"} and len(preds) == 2
    ckpt = torch.load(run / "ckpt.pt", map_location="cpu", weights_only=False)
    assert ckpt["class_names"] == classes and "model" in ckpt
    assert metrics["test"]["macro_f1"] == saved["test"]["macro_f1"]


def test_resume_continues_instead_of_retraining(
    tiny_cfg, prepared, data_root, tmp_path: Path
) -> None:
    run = tmp_path / "run"
    first = _run({**tiny_cfg, "epochs": 1}, prepared, data_root, run)
    resumed = _run(tiny_cfg, prepared, data_root, run, resume=True)
    assert len(resumed["history"]) == 2
    assert resumed["history"][0] == first["history"][0]  # epoch 1 was not redone


def test_training_is_reproducible_for_a_seed(tiny_cfg, prepared, data_root, tmp_path: Path) -> None:
    a = _run(tiny_cfg, prepared, data_root, tmp_path / "a")
    b = _run(tiny_cfg, prepared, data_root, tmp_path / "b")
    assert [h["train_loss"] for h in a["history"]] == pytest.approx(
        [h["train_loss"] for h in b["history"]]
    )


def test_cli_debug_run_uses_env_paths_and_forces_one_epoch(
    prepared, data_root, tiny_cfg, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(paths.ENV_OUT, str(tmp_path / "out"))
    mf.write_manifest(prepared, data_root / "manifests" / "manifest.csv")
    cfg_path = tmp_path / "cls_tiny.yaml"
    cfg_path.write_text(yaml.safe_dump(tiny_cfg))
    assert train.main(["--config", str(cfg_path), "--debug", "--no-pretrained", "--seed", "3"]) == 0
    (run_dir,) = (tmp_path / "out" / "runs").iterdir()
    assert run_dir.name.startswith("cls_tiny-3-")
    saved = json.loads((run_dir / "metrics.json").read_text())
    assert len(saved["history"]) == 1  # --debug means one epoch
    assert io.load_yaml(run_dir / "config.yaml")["seed"] == 3
    assert (run_dir / "git_commit.txt").is_file()
