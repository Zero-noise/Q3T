#!/usr/bin/env python3
# coding=utf-8
"""
Centralized logging configuration for Qwen3-TTS Jetson deployment.

Usage:
    from log_config import get_logger
    logger = get_logger(__name__)
    logger.info("Model loaded on %s", device)

Features:
- Console output with colored level names
- Rotating file handler (logs/q3t.log, 10MB x 5 backups)
- Configurable via environment variables:
    Q3T_LOG_LEVEL   : root log level   (default: INFO)
    Q3T_LOG_DIR     : log directory     (default: ./logs)
    Q3T_LOG_FILE    : log filename      (default: q3t.log)
    Q3T_LOG_MAX_MB  : max file size MB  (default: 10)
    Q3T_LOG_BACKUPS : backup count      (default: 5)
"""

from __future__ import annotations

import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants & env-driven defaults
# ---------------------------------------------------------------------------
_DEFAULT_LEVEL = "INFO"
_DEFAULT_LOG_DIR = "logs"
_DEFAULT_LOG_FILE = "q3t.log"
_DEFAULT_MAX_BYTES = 10  # MB
_DEFAULT_BACKUP_COUNT = 5

_CONSOLE_FMT = "[%(asctime)s] %(levelname)-7s %(name)s: %(message)s"
_FILE_FMT = "[%(asctime)s] %(levelname)-7s %(name)s (%(filename)s:%(lineno)d): %(message)s"
_DATE_FMT = "%Y-%m-%d %H:%M:%S"

# Sentinel to ensure setup_logging() runs only once
_LOGGING_CONFIGURED = False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def setup_logging(
    level: str | None = None,
    log_dir: str | None = None,
    log_file: str | None = None,
    max_mb: int | None = None,
    backup_count: int | None = None,
) -> None:
    """
    Configure the root logger with console + rotating file handlers.

    Safe to call multiple times — subsequent calls are no-ops.
    Call this once early in main() before any logging happens.
    """
    global _LOGGING_CONFIGURED
    if _LOGGING_CONFIGURED:
        return
    _LOGGING_CONFIGURED = True

    level = (level or os.getenv("Q3T_LOG_LEVEL", _DEFAULT_LEVEL)).upper()
    log_dir = log_dir or os.getenv("Q3T_LOG_DIR", _DEFAULT_LOG_DIR)
    log_file = log_file or os.getenv("Q3T_LOG_FILE", _DEFAULT_LOG_FILE)
    max_mb = max_mb or int(os.getenv("Q3T_LOG_MAX_MB", str(_DEFAULT_MAX_BYTES)))
    backup_count = backup_count or int(
        os.getenv("Q3T_LOG_BACKUPS", str(_DEFAULT_BACKUP_COUNT))
    )

    numeric_level = getattr(logging, level, logging.INFO)

    root = logging.getLogger()
    root.setLevel(numeric_level)

    # Avoid duplicate handlers if something else already added them
    if root.handlers:
        return

    # --- Console handler ---
    console = logging.StreamHandler(sys.stderr)
    console.setLevel(numeric_level)
    console.setFormatter(logging.Formatter(_CONSOLE_FMT, datefmt=_DATE_FMT))
    root.addHandler(console)

    # --- File handler ---
    try:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            filename=str(log_path / log_file),
            maxBytes=max_mb * 1024 * 1024,
            backupCount=backup_count,
            encoding="utf-8",
        )
        file_handler.setLevel(logging.DEBUG)  # file always captures DEBUG+
        file_handler.setFormatter(logging.Formatter(_FILE_FMT, datefmt=_DATE_FMT))
        root.addHandler(file_handler)
    except OSError as exc:
        # If we can't create the log dir/file, warn but don't crash
        console.handle(
            logging.LogRecord(
                name="log_config",
                level=logging.WARNING,
                pathname=__file__,
                lineno=0,
                msg="Failed to set up file logging at %s: %s",
                args=(log_dir, exc),
                exc_info=None,
            )
        )

    # Quiet noisy third-party loggers
    for noisy in ("urllib3", "httpx", "httpcore", "gradio", "uvicorn"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """
    Return a named logger.

    If setup_logging() hasn't been called yet it will be auto-invoked
    with env-driven defaults, so callers at module level are safe.
    """
    if not _LOGGING_CONFIGURED:
        setup_logging()
    return logging.getLogger(name)
