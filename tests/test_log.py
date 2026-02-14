"""Tests for the logging configuration module.

Verifies handler construction, noisy-logger silencing, and
full ``setup_logging`` configuration including file output and
level fallback behaviour.
"""

from __future__ import annotations

import logging

from fusion_oncology.utils.log import (
    _build_handlers,
    _silence_noisy_loggers,
    setup_logging,
)


class TestBuildHandlers:
    """Unit tests for ``_build_handlers``."""

    def test_console_only(self) -> None:
        """Verify that passing ``None`` returns a single StreamHandler.

        Asserts the handler list has exactly one element and that
        it is a ``logging.StreamHandler`` instance.
        """
        handlers = _build_handlers(None)
        assert len(handlers) == 1
        assert isinstance(handlers[0], logging.StreamHandler)

    def test_console_and_file(self, tmp_path) -> None:
        """Verify that passing a file path returns StreamHandler + FileHandler.

        Parameters
        ----------
        tmp_path : pathlib.Path
            Pytest-provided temporary directory for the log file.
        """
        log_file = str(tmp_path / "test.log")
        handlers = _build_handlers(log_file)
        assert len(handlers) == 2
        assert isinstance(handlers[1], logging.FileHandler)
        handlers[1].close()

    def test_empty_string_no_file(self) -> None:
        """Verify that an empty string produces console-only output.

        An empty string is falsy and should yield a single
        ``StreamHandler`` with no ``FileHandler`` appended.
        """
        handlers = _build_handlers("")
        assert len(handlers) == 1


class TestSilenceNoisyLoggers:
    """Unit tests for ``_silence_noisy_loggers``."""

    def test_sets_warning_level(self) -> None:
        """Verify all known noisy loggers are raised to WARNING.

        Iterates through each noisy logger name and asserts
        its level is set to ``logging.WARNING``.
        """
        _silence_noisy_loggers()
        for name in ("urllib3", "transformers", "filelock", "huggingface_hub"):
            assert logging.getLogger(name).level == logging.WARNING


class TestSetupLogging:
    """Integration tests for ``setup_logging``."""

    def _reset_root(self) -> logging.Logger:
        """Clear root handlers so each test starts clean.

        Returns
        -------
        logging.Logger
            The root logger with handlers cleared.
        """
        root = logging.getLogger()
        root.handlers.clear()
        return root

    def _extract_file_handlers(self, root: logging.Logger) -> list:
        """Return all FileHandler instances attached to *root*.

        Parameters
        ----------
        root : logging.Logger
            The root logger to inspect.

        Returns
        -------
        list
            FileHandler instances currently attached.
        """
        return [h for h in root.handlers if isinstance(h, logging.FileHandler)]

    def test_default_level(self) -> None:
        """Verify the default call configures root logger at INFO.

        Calls ``setup_logging`` with no arguments and asserts
        the root logger level equals ``logging.INFO``.
        """
        root = self._reset_root()
        setup_logging()
        assert root.level == logging.INFO

    def test_debug_level(self) -> None:
        """Verify passing ``level='DEBUG'`` sets root to DEBUG.

        Asserts the root logger level equals ``logging.DEBUG``
        after an explicit ``level='DEBUG'`` call.
        """
        root = self._reset_root()
        setup_logging(level="DEBUG")
        assert root.level == logging.DEBUG

    def test_with_log_file(self, tmp_path) -> None:
        """Verify a FileHandler is added when ``log_file`` is given.

        Parameters
        ----------
        tmp_path : pathlib.Path
            Pytest-provided temporary directory for the log file.
        """
        root = self._reset_root()
        setup_logging(log_file=str(tmp_path / "run.log"))
        fh = self._extract_file_handlers(root)
        assert len(fh) >= 1
        for h in fh:
            h.close()

    def test_invalid_level_falls_back(self) -> None:
        """Verify an unrecognised level string defaults to INFO.

        Passes a nonsensical level name and asserts the root
        logger falls back to ``logging.INFO``.
        """
        root = self._reset_root()
        setup_logging(level="NONEXISTENT")
        assert root.level == logging.INFO
