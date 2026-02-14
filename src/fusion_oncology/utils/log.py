"""
Logging configuration for the Fusion Oncology Suite.

Call ``setup_logging()`` once early in the process (e.g. from the CLI
entry-point) to configure root and package-level loggers.
"""

from __future__ import annotations

import logging
import sys


def _build_handlers(log_file: str | None) -> list[logging.Handler]:
    """Build the list of logging handlers.

    Parameters
    ----------
    log_file : str, optional
        If provided, adds a file handler.

    Returns
    -------
    list[logging.Handler]
        Console handler, plus optional file handler.
    """
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stderr)]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    return handlers


def _silence_noisy_loggers() -> None:
    """Set noisy third-party loggers to WARNING level.

    Returns
    -------
    None
    """
    for noisy in ("urllib3", "transformers", "filelock", "huggingface_hub"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def setup_logging(level: str = "INFO", log_file: str | None = None) -> None:
    """
    Configure console (and optional file) logging.

    Parameters
    ----------
    level : str
        Logging level name (``DEBUG``, ``INFO``, ``WARNING``, …).
    log_file : str, optional
        If provided, also write logs to this file.
    """
    fmt = "%(asctime)s │ %(levelname)-7s │ %(name)-30s │ %(message)s"
    handlers = _build_handlers(log_file)
    lvl = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(level=lvl, format=fmt, datefmt="%H:%M:%S", handlers=handlers)
    _silence_noisy_loggers()
