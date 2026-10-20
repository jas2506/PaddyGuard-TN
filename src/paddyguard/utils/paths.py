"""Filesystem roots (M0). Data and outputs are located only through env vars, never hard-coded."""

from __future__ import annotations

import os
from pathlib import Path

ENV_DATA = "PADDYGUARD_DATA"
ENV_OUT = "PADDYGUARD_OUT"
RUNS_DIRNAME = "runs"
CONFIGS_DIRNAME = "configs"
REPORTS_DIRNAME = "reports"


def repo_root() -> Path:
    """Repository root (this file is <repo>/src/paddyguard/utils/paths.py)."""
    return Path(__file__).resolve().parents[3]


def data_root() -> Path:
    """$PADDYGUARD_DATA, default <repo>/data."""
    return Path(os.environ.get(ENV_DATA) or repo_root() / "data")


def out_root() -> Path:
    """$PADDYGUARD_OUT, default <repo>."""
    return Path(os.environ.get(ENV_OUT) or repo_root())


def runs_root() -> Path:
    return out_root() / RUNS_DIRNAME


def configs_dir() -> Path:
    return repo_root() / CONFIGS_DIRNAME


def reports_root() -> Path:
    """Committed small results: <repo>/reports."""
    return repo_root() / REPORTS_DIRNAME
