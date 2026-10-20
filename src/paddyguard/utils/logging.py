"""Console + file logger (M0)."""

from __future__ import annotations

import logging
from pathlib import Path

LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


def get_logger(
    name: str, log_file: str | Path | None = None, level: int = logging.INFO
) -> logging.Logger:
    """Logger writing to the console and, if given, `log_file`. Safe to call repeatedly."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False
    formatter = logging.Formatter(LOG_FORMAT)
    if not any(type(h) is logging.StreamHandler for h in logger.handlers):
        console = logging.StreamHandler()
        console.setFormatter(formatter)
        logger.addHandler(console)
    if log_file is not None:
        target = str(Path(log_file).resolve())
        if not any(
            isinstance(h, logging.FileHandler) and h.baseFilename == target for h in logger.handlers
        ):
            Path(log_file).parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
    return logger
