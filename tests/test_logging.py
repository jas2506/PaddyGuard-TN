from pathlib import Path

from paddyguard.utils.logging import get_logger


def test_logger_writes_to_file(tmp_path: Path) -> None:
    log_file = tmp_path / "run.log"
    logger = get_logger("paddyguard.test_file", log_file)
    logger.info("hello run")
    for h in logger.handlers:
        h.flush()
    assert "hello run" in log_file.read_text()


def test_get_logger_is_idempotent(tmp_path: Path) -> None:
    log_file = tmp_path / "run.log"
    first = get_logger("paddyguard.test_idem", log_file)
    n = len(first.handlers)
    second = get_logger("paddyguard.test_idem", log_file)
    assert second is first and len(second.handlers) == n
