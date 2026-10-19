"""Logging configuration shared by live and offline modes."""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

from data_policy import ensure_private_directory, open_private_append, safe_file_path


LOGGER_NAME = "monitoring"


class PrivateRotatingFileHandler(RotatingFileHandler):
    def _open(self):
        return open_private_append(Path(self.baseFilename))

    def doRollover(self) -> None:
        # Validate every source/destination before the standard rotation mutates any.
        safe_file_path(Path(self.baseFilename))
        for index in range(1, self.backupCount + 1):
            safe_file_path(Path(f"{self.baseFilename}.{index}"))
        super().doRollover()


def configure_logging(level: str = "INFO", log_dir: Path | None = None) -> logging.Logger:
    """Configure one console logger and an optional rotating file log."""
    logger = logging.getLogger(LOGGER_NAME)
    if getattr(logger, "_monitoring_configured", False):
        return logger

    resolved_level = getattr(logging, level.upper(), logging.INFO)
    logger.setLevel(resolved_level)
    logger.propagate = False
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console = logging.StreamHandler()
    console.setLevel(resolved_level)
    console.setFormatter(formatter)
    logger.addHandler(console)

    if log_dir is not None:
        ensure_private_directory(log_dir)
        file_handler = PrivateRotatingFileHandler(
            log_dir / "monitoring.log",
            maxBytes=2_000_000,
            backupCount=3,
            encoding="utf-8",
        )
        file_handler.setLevel(resolved_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    logger._monitoring_configured = True  # type: ignore[attr-defined]
    return logger


def get_logger(component: str) -> logging.Logger:
    """Return a child logger without configuring handlers implicitly."""
    return logging.getLogger(f"{LOGGER_NAME}.{component}")
