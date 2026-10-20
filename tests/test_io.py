import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from paddyguard.utils import io, paths


def test_yaml_roundtrip(tmp_path: Path) -> None:
    f = tmp_path / "c.yaml"
    f.write_text("a: 1\nb: {c: [1, 2]}\n")
    assert io.load_yaml(f) == {"a": 1, "b": {"c": [1, 2]}}


def test_load_yaml_rejects_non_mapping(tmp_path: Path) -> None:
    f = tmp_path / "c.yaml"
    f.write_text("- 1\n- 2\n")
    with pytest.raises(TypeError):
        io.load_yaml(f)


def test_save_json_handles_numpy_and_creates_parents(tmp_path: Path) -> None:
    f = tmp_path / "x" / "y" / "m.json"
    io.save_json({"f1": np.float32(0.5), "n": np.int64(3), "p": tmp_path}, f)
    assert json.loads(f.read_text()) == {"f1": 0.5, "n": 3, "p": str(tmp_path)}


def test_parquet_roundtrip(tmp_path: Path) -> None:
    df = pd.DataFrame({"image_id": ["a", "b"], "logit_x": np.float32([0.1, 0.2])})
    f = tmp_path / "sub" / "p.parquet"
    io.save_parquet(df, f)
    out = pd.read_parquet(f)
    assert list(out["image_id"]) == ["a", "b"]
    assert out["logit_x"].dtype == np.float32


def _init_repo(path: Path) -> None:
    run = lambda *a: subprocess.run(["git", *a], cwd=path, check=True, capture_output=True)
    run("init")
    (path / "f.txt").write_text("x")
    run("add", "f.txt")
    run("-c", "user.name=t", "-c", "user.email=t@example.com", "commit", "-m", "init")


def test_git_commit_in_clean_and_dirty_repo(tmp_path: Path) -> None:
    _init_repo(tmp_path)
    clean = io.git_commit(tmp_path)
    assert len(clean) == 40 and not clean.endswith("-dirty")
    (tmp_path / "f.txt").write_text("changed")
    assert io.git_commit(tmp_path) == clean + "-dirty"


def test_git_commit_outside_a_repo_is_unknown(tmp_path: Path) -> None:
    assert io.git_commit(tmp_path) == "unknown"


def test_new_run_dir_writes_config_seed_and_commit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv(paths.ENV_OUT, str(tmp_path))
    run_dir = io.new_run_dir("cls_test", 7, config={"model": "m", "seed": 0})
    assert run_dir.parent == tmp_path / "runs"
    assert run_dir.name.startswith("cls_test-7-")
    cfg = io.load_yaml(run_dir / "config.yaml")
    assert cfg == {"model": "m", "seed": 7}  # seed argument wins and is recorded
    assert (run_dir / "git_commit.txt").read_text().strip()


def test_new_run_dir_defaults_to_the_named_config_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv(paths.ENV_OUT, str(tmp_path))
    run_dir = io.new_run_dir("cls_effb0", 0)
    assert io.load_yaml(run_dir / "config.yaml")["model"] == "efficientnet_b0"


def test_new_run_dir_unknown_config_fails_loudly(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv(paths.ENV_OUT, str(tmp_path))
    with pytest.raises(FileNotFoundError):
        io.new_run_dir("does_not_exist", 0)
