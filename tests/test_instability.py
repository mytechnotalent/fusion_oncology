"""Tests for the mutational instability scoring module.

Verifies point mutation logic, cosine drift computation, scoring,
and detailed report assembly with mocked DNABERT embeddings.
"""

from __future__ import annotations

import logging
from unittest.mock import MagicMock

import numpy as np
import pytest

from fusion_oncology.analysis.instability import InstabilityAnalyzer
from fusion_oncology.config import ProjectConfig


@pytest.fixture()
def cfg(tmp_path):
    """Return a lightweight ProjectConfig for instability tests.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest temporary directory.

    Returns
    -------
    ProjectConfig
        Config with three fuzz iterations.
    """
    return ProjectConfig(
        cache_dir=tmp_path / "cache",
        output_dir=tmp_path / "out",
        fuzz_iterations=3,
    )


@pytest.fixture()
def mock_bert():
    """Return a mock DNABERTEngine with deterministic embeddings.

    Returns
    -------
    MagicMock
        Each ``embed()`` call produces a distinct random 768-d vector.
    """
    bert = MagicMock()
    rng = np.random.default_rng(0)
    bert.embed.side_effect = lambda seq: rng.random(768)
    return bert


@pytest.fixture()
def analyzer(mock_bert, cfg):
    """Build an InstabilityAnalyzer backed by the mock BERT engine.

    Parameters
    ----------
    mock_bert : MagicMock
        Mocked DNABERTEngine.
    cfg : ProjectConfig
        Test-scoped configuration.

    Returns
    -------
    InstabilityAnalyzer
        Ready-to-use analyser instance.
    """
    return InstabilityAnalyzer(bert=mock_bert, config=cfg)


# ── _mutate tests ────────────────────────────────────────────────────────


def _count_diffs(a: str, b: str) -> int:
    """Count character-level differences between two equal-length strings.

    Parameters
    ----------
    a : str
        First string.
    b : str
        Second string.

    Returns
    -------
    int
        Number of positions where *a* and *b* differ.
    """
    return sum(x != y for x, y in zip(a, b))


class TestMutate:
    """Tests for ``InstabilityAnalyzer._mutate``."""

    def test_single_mutation(self, analyzer) -> None:
        """Verify a single mutation changes exactly one base.

        Parameters
        ----------
        analyzer : InstabilityAnalyzer
            Fixture-injected analyzer with mocked BERT.
        """
        original = "ACGTACGTACGT"
        mutated = analyzer._mutate(original, n_mutations=1)
        assert len(mutated) == len(original)
        assert _count_diffs(original, mutated) == 1

    def test_multiple_mutations(self, analyzer) -> None:
        """Verify five mutations changes exactly five bases.

        Parameters
        ----------
        analyzer : InstabilityAnalyzer
            Fixture-injected analyzer with mocked BERT.
        """
        original = "ACGTACGTACGTACGTACGT"
        mutated = analyzer._mutate(original, n_mutations=5)
        assert _count_diffs(original, mutated) == 5

    def test_capped_at_seq_length(self, analyzer) -> None:
        """Verify mutation count is capped at sequence length.

        Parameters
        ----------
        analyzer : InstabilityAnalyzer
            Fixture-injected analyzer with mocked BERT.
        """
        original = "ACGT"
        mutated = analyzer._mutate(original, n_mutations=10)
        assert _count_diffs(original, mutated) == 4

    def test_only_valid_bases(self, analyzer) -> None:
        """Verify all mutated characters are valid nucleotides.

        Parameters
        ----------
        analyzer : InstabilityAnalyzer
            Fixture-injected analyzer with mocked BERT.
        """
        mutated = analyzer._mutate("AAAAAAAAAA", n_mutations=3)
        assert all(c in "ACGT" for c in mutated)


# ── _compute_drifts tests ───────────────────────────────────────────────


class TestComputeDrifts:
    """Tests for ``InstabilityAnalyzer._compute_drifts``."""

    def test_returns_correct_count(self, analyzer) -> None:
        """Verify one drift value is returned per fuzz iteration.

        Parameters
        ----------
        analyzer : InstabilityAnalyzer
            Fixture-injected analyzer configured with three iterations.
        """
        drifts = analyzer._compute_drifts("ACGTACGTACGTACGTACGT")
        assert len(drifts) == 3
        assert all(isinstance(d, float) for d in drifts)

    def test_drifts_non_negative(self, analyzer) -> None:
        """Verify cosine drift values are non-negative.

        Parameters
        ----------
        analyzer : InstabilityAnalyzer
            Fixture-injected analyzer with mocked BERT.
        """
        drifts = analyzer._compute_drifts("ACGTACGTACGTACGTACGT")
        assert all(d >= 0 for d in drifts)


# ── _log_instability tests ──────────────────────────────────────────────


class TestLogInstability:
    """Tests for ``InstabilityAnalyzer._log_instability``."""

    def test_emits_debug_message(self, analyzer, caplog) -> None:
        """Verify the gene name is logged at DEBUG level.

        Parameters
        ----------
        analyzer : InstabilityAnalyzer
            Fixture-injected analyzer with mocked BERT.
        caplog : pytest.LogCaptureFixture
            Pytest fixture for capturing log output.
        """
        with caplog.at_level(logging.DEBUG):
            analyzer._log_instability("EGFR", 0.05, 0.01)
        assert "EGFR" in caplog.text


# ── score tests ──────────────────────────────────────────────────────────


class TestScore:
    """Tests for ``InstabilityAnalyzer.score``."""

    def test_returns_float(self, analyzer) -> None:
        """Verify the score is always a float.

        Parameters
        ----------
        analyzer : InstabilityAnalyzer
            Fixture-injected analyzer with mocked BERT.
        """
        result = analyzer.score("EGFR", "ACGTACGTACGTACGTACGT")
        assert isinstance(result, float)

    def test_short_sequence_returns_zero(self, analyzer) -> None:
        """Verify sequences shorter than 10 bp return 0.0.

        Parameters
        ----------
        analyzer : InstabilityAnalyzer
            Fixture-injected analyzer with mocked BERT.
        """
        assert analyzer.score("TINY", "ACGT") == 0.0

    def test_non_negative(self, analyzer) -> None:
        """Verify the instability score is non-negative.

        Parameters
        ----------
        analyzer : InstabilityAnalyzer
            Fixture-injected analyzer with mocked BERT.
        """
        assert analyzer.score("BRAF", "ACGTACGTACGTACGTACGT") >= 0.0


# ── _empty_report tests ─────────────────────────────────────────────────


class TestEmptyReport:
    """Tests for ``InstabilityAnalyzer._empty_report``."""

    def test_has_zeroed_keys(self, analyzer) -> None:
        """Verify the empty report has all drift values set to zero.

        Parameters
        ----------
        analyzer : InstabilityAnalyzer
            Fixture-injected analyzer with mocked BERT.
        """
        report = analyzer._empty_report("TP53")
        assert report["gene"] == "TP53"
        assert report["mean_drift"] == 0
        assert report["all_drifts"] == []


# ── _build_report tests ─────────────────────────────────────────────────


class TestBuildReport:
    """Tests for ``InstabilityAnalyzer._build_report``."""

    def test_correct_statistics(self, analyzer) -> None:
        """Verify report statistics match the input drift values.

        Parameters
        ----------
        analyzer : InstabilityAnalyzer
            Fixture-injected analyzer with mocked BERT.
        """
        drifts = [0.1, 0.2, 0.3]
        report = analyzer._build_report("EGFR", drifts)
        assert report["gene"] == "EGFR"
        assert abs(report["mean_drift"] - 0.2) < 1e-9
        assert abs(report["max_drift"] - 0.3) < 1e-9
        assert abs(report["min_drift"] - 0.1) < 1e-9


# ── detailed_report tests ───────────────────────────────────────────────


class TestDetailedReport:
    """Tests for ``InstabilityAnalyzer.detailed_report``."""

    def test_short_sequence_returns_empty(self, analyzer) -> None:
        """Verify short sequences produce an empty report.

        Parameters
        ----------
        analyzer : InstabilityAnalyzer
            Fixture-injected analyzer with mocked BERT.
        """
        report = analyzer.detailed_report("TINY", "ACG")
        assert report["mean_drift"] == 0
        assert report["all_drifts"] == []

    def test_normal_sequence_returns_drifts(self, analyzer) -> None:
        """Verify normal sequences produce drift values.

        Parameters
        ----------
        analyzer : InstabilityAnalyzer
            Fixture-injected analyzer with mocked BERT.
        """
        report = analyzer.detailed_report("EGFR", "ACGTACGTACGTACGTACGT")
        assert report["gene"] == "EGFR"
        assert len(report["all_drifts"]) == 3


# ── default config tests ────────────────────────────────────────────────


class TestInitDefault:
    """Tests for ``InstabilityAnalyzer`` with default configuration."""

    def test_default_config_applied(self, mock_bert) -> None:
        """Verify omitting config uses default ProjectConfig.

        Parameters
        ----------
        mock_bert : MagicMock
            Fixture-injected mock DNABERTEngine.
        """
        analyzer = InstabilityAnalyzer(bert=mock_bert)
        assert analyzer.cfg is not None
