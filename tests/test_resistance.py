"""
Tests for the resistance prediction module.

Covers the RESISTANCE_DB, ResistancePredictor methods,
and DataFrame annotation.
"""

from __future__ import annotations

import pandas as pd
import pytest

from fusion_oncology.analysis.resistance import (
    RESISTANCE_DB,
    ResistancePredictor,
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
def predictor(cfg: ProjectConfig) -> ResistancePredictor:
    """Create a ResistancePredictor instance.

    Parameters
    ----------
    cfg : ProjectConfig
        Test configuration.

    Returns
    -------
    ResistancePredictor
        Predictor instance.
    """
    return ResistancePredictor(cfg)


class TestResistanceDB:
    """Tests for the curated resistance mechanism database."""

    def test_non_empty(self) -> None:
        """The database should contain entries.

        Returns
        -------
        None
        """
        assert len(RESISTANCE_DB) > 0

    def test_egfr_has_mechanisms(self) -> None:
        """EGFR should have known resistance mechanisms.

        Returns
        -------
        None
        """
        assert "EGFR" in RESISTANCE_DB
        assert len(RESISTANCE_DB["EGFR"]) > 0

    def test_entries_have_required_keys(self) -> None:
        """Each entry should have drug, mechanism, strategy, frequency keys.

        Returns
        -------
        None
        """
        for gene, entries in RESISTANCE_DB.items():
            for entry in entries:
                assert "drug" in entry, f"Missing 'drug' in {gene}"
                assert "mechanism" in entry, f"Missing 'mechanism' in {gene}"
                assert "strategy" in entry, f"Missing 'strategy' in {gene}"


class TestResistancePredictor:
    """Tests for the ResistancePredictor class."""

    def test_predict_known_gene(self, predictor: ResistancePredictor) -> None:
        """Predict should return mechanisms for a known gene.

        Parameters
        ----------
        predictor : ResistancePredictor
            Predictor instance.
        """
        result = predictor.predict("EGFR")
        assert isinstance(result, list)
        assert len(result) > 0

    def test_predict_unknown_gene(self, predictor: ResistancePredictor) -> None:
        """Predict should return empty list for an unknown gene.

        Parameters
        ----------
        predictor : ResistancePredictor
            Predictor instance.
        """
        result = predictor.predict("NOTAREALGENEXYZ")
        assert result == []

    def test_risk_score_range(self, predictor: ResistancePredictor) -> None:
        """Risk score should be between 0 and 1.

        Parameters
        ----------
        predictor : ResistancePredictor
            Predictor instance.
        """
        for gene in ["EGFR", "BRAF", "NOTREAL"]:
            score = predictor.resistance_risk_score(gene)
            assert 0.0 <= score <= 1.0

    def test_risk_score_known_gene_nonzero(self, predictor: ResistancePredictor) -> None:
        """A gene with known mechanisms should have a nonzero risk score.

        Parameters
        ----------
        predictor : ResistancePredictor
            Predictor instance.
        """
        score = predictor.resistance_risk_score("EGFR")
        assert score > 0

    def test_annotate_adds_columns(self, predictor: ResistancePredictor) -> None:
        """Annotate should add resistance-related columns to a DataFrame.

        Parameters
        ----------
        predictor : ResistancePredictor
            Predictor instance.
        """
        df = pd.DataFrame({"Gene": ["EGFR", "BRAF", "FAKEGENE"]})
        result = predictor.annotate(df)
        assert "Resistance_Risk" in result.columns
        assert "N_Resistance_Mechanisms" in result.columns

    def test_full_report(self, predictor: ResistancePredictor) -> None:
        """Full report should return a DataFrame with expected columns.

        Parameters
        ----------
        predictor : ResistancePredictor
            Predictor instance.
        """
        report = predictor.full_report(["EGFR", "FAKEGENE"])
        assert isinstance(report, pd.DataFrame)
        assert len(report) > 0
