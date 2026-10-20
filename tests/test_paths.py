from pathlib import Path

import pytest

from paddyguard.utils import paths


def test_env_vars_override_defaults(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv(paths.ENV_DATA, str(tmp_path / "d"))
    monkeypatch.setenv(paths.ENV_OUT, str(tmp_path / "o"))
    assert paths.data_root() == tmp_path / "d"
    assert paths.out_root() == tmp_path / "o"
    assert paths.runs_root() == tmp_path / "o" / "runs"


def test_defaults_are_inside_the_repo(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(paths.ENV_DATA, raising=False)
    monkeypatch.delenv(paths.ENV_OUT, raising=False)
    assert paths.data_root() == paths.repo_root() / "data"
    assert paths.out_root() == paths.repo_root()
    assert (paths.repo_root() / "pyproject.toml").is_file()


def test_reports_root_is_inside_the_repo() -> None:
    assert paths.reports_root() == paths.repo_root() / "reports"
