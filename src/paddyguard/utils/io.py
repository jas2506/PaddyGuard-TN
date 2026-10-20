"""Reading and writing files, and run directories (M0)."""

from __future__ import annotations

import json
import subprocess
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from paddyguard.utils import paths

GIT_TIMEOUT_S = 10
UNKNOWN_COMMIT = "unknown"
RUN_TIMESTAMP_FORMAT = "%Y%m%d-%H%M%S"


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path) as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise TypeError(
            f"{path}: expected a YAML mapping at the top level, got {type(data).__name__}"
        )
    return data


def save_yaml(data: Mapping[str, Any], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(dict(data), f, sort_keys=False)


def _json_default(obj: Any) -> Any:
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def save_json(obj: Any, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=_json_default)


def save_parquet(df: pd.DataFrame, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def _git(args: list[str], cwd: Path) -> str:
    out = subprocess.run(
        ["git", *args], cwd=cwd, capture_output=True, text=True, check=True, timeout=GIT_TIMEOUT_S
    )
    return out.stdout.strip()


def git_commit(cwd: str | Path | None = None) -> str:
    """HEAD commit hash, with '-dirty' if the tree has changes; 'unknown' outside a git repo."""
    cwd = Path(cwd) if cwd is not None else paths.repo_root()
    try:
        commit = _git(["rev-parse", "HEAD"], cwd)
        dirty = bool(_git(["status", "--porcelain"], cwd))
    except (OSError, subprocess.SubprocessError):
        return UNKNOWN_COMMIT
    return commit + "-dirty" if dirty else commit


def _resolve_config(config_name: str, config: Mapping[str, Any] | None) -> dict[str, Any]:
    if config is not None:
        return dict(config)
    return load_yaml(paths.configs_dir() / f"{config_name}.yaml")


def new_run_dir(config_name: str, seed: int, config: Mapping[str, Any] | None = None) -> Path:
    """Create runs/<config_name>-<seed>-<timestamp>/ and record the config, seed and git commit.

    `config` is the resolved config; if omitted, configs/<config_name>.yaml is loaded. The `seed`
    argument overrides any seed in the config so config.yaml always shows the seed actually used.
    """
    resolved = {**_resolve_config(config_name, config), "seed": seed}
    stamp = datetime.now().strftime(RUN_TIMESTAMP_FORMAT)
    run_dir = paths.runs_root() / f"{config_name}-{seed}-{stamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    save_yaml(resolved, run_dir / "config.yaml")
    (run_dir / "git_commit.txt").write_text(git_commit() + "\n")
    return run_dir
