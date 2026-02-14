"""
Tests for the synthetic lethality detection module.

Covers partner lookup, expression-based screening, annotation,
and combination therapy suggestion.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fusion_oncology.analysis.synthetic_lethality import (
    KNOWN_SL_PAIRS,
    SyntheticLethalityDetector,
)
from fusion_oncology.config import ProjectConfig


@pytest.fixture()
def cfg(tmp_path: "Path") -> ProjectConfig:
    """Provide a temporary ProjectConfig for testing.

    Parameters
    ----------
    tmp_path : Path
        Pytest-provided temporary directory.

    Returns
    -------
    ProjectConfig
        Configuration with temporary paths.
    """
    return ProjectConfig(cache_dir=tmp_path / "cache", output_dir=tmp_path / "out")


@pytest.fixture()
def detector(cfg: ProjectConfig) -> SyntheticLethalityDetector:
    """Create a SyntheticLethalityDetector instance.

    Parameters
    ----------
    cfg : ProjectConfig
        Test configuration.

    Returns
    -------
    SyntheticLethalityDetector
        Detector instance.
    """
    return SyntheticLethalityDetector(cfg)


class TestKnownPairs:
    """Tests for the curated synthetic lethality pair database."""

    def test_database_is_non_empty(self) -> None:
        """The KNOWN_SL_PAIRS database should contain entries.

        Returns
        -------
        None
        """
        assert len(KNOWN_SL_PAIRS) > 0

    def test_pairs_are_tuples_of_three(self) -> None:
        """Each pair should be a 3-tuple of strings (gene1, gene2, annotation).

        Returns
        -------
        None
        """
        for pair in KNOWN_SL_PAIRS:
            assert len(pair) == 3
            assert isinstance(pair[0], str)
            assert isinstance(pair[1], str)
            assert isinstance(pair[2], str)


class TestSyntheticLethalityDetector:
    """Tests for the SyntheticLethalityDetector class."""

    def test_known_partners_brca1(self, detector: SyntheticLethalityDetector) -> None:
        """BRCA1 should have PARP1 as a known SL partner.

        Parameters
        ----------
        detector : SyntheticLethalityDetector
            Detector instance.
        """
        partners = detector.known_partners("BRCA1")
        partner_names = [p["partner"] for p in partners]
        assert "PARP1" in partner_names

    def test_known_partners_bidirectional(self, detector: SyntheticLethalityDetector) -> None:
        """SL relationship should be bidirectional.

        Parameters
        ----------
        detector : SyntheticLethalityDetector
            Detector instance.
        """
        partners_a = [p["partner"] for p in detector.known_partners("BRCA1")]
        partners_b = [p["partner"] for p in detector.known_partners("PARP1")]
        assert "PARP1" in partners_a
        assert "BRCA1" in partners_b

    def test_unknown_gene_returns_empty(self, detector: SyntheticLethalityDetector) -> None:
        """Unknown gene should return an empty list.

        Parameters
        ----------
        detector : SyntheticLethalityDetector
            Detector instance.
        """
        assert detector.known_partners("NOTAREALGENE") == []

    def test_screen_from_expression(self, detector: SyntheticLethalityDetector) -> None:
        """Expression-based screening should return a list.

        Parameters
        ----------
        detector : SyntheticLethalityDetector
            Detector instance.
        """
        rng = np.random.default_rng(42)
        n_samples, n_genes = 100, 50
        expression = pd.DataFrame(
            rng.standard_normal((n_samples, n_genes)),
            columns=[f"GENE{i}" for i in range(n_genes)],
        )
        results = detector.screen_from_expression(expression, ["GENE0"])
        assert isinstance(results, pd.DataFrame)

    def test_annotate_adds_column(self, detector: SyntheticLethalityDetector) -> None:
        """Annotate should add an SL_Partners column.

        Parameters
        ----------
        detector : SyntheticLethalityDetector
            Detector instance.
        """
        df = pd.DataFrame({"Gene": ["BRCA1", "EGFR", "FAKEGENE"]})
        result = detector.annotate(df)
        assert "SL_Partners" in result.columns

    def test_combination_therapies(self, detector: SyntheticLethalityDetector) -> None:
        """Combination therapy suggestions should return a list.

        Parameters
        ----------
        detector : SyntheticLethalityDetector
            Detector instance.
        """
        combos = detector.combination_therapies("BRCA1")
        assert isinstance(combos, list)
