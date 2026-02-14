"""
Tests for the clinical evidence aggregation module.

Covers OpenTargets, CIViC, and ClinicalTrials.gov query functions
as well as the composite ClinicalEvidenceAggregator.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from fusion_oncology.analysis.clinical_evidence import (
    ClinicalEvidenceAggregator,
    query_civic,
    query_clinical_trials,
    query_opentargets,
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


class TestQueryOpenTargets:
    """Tests for the ``query_opentargets`` function."""

    def test_returns_dict_on_network_failure(self, cfg: ProjectConfig) -> None:
        """Return a fallback dict when the API is unreachable.

        Parameters
        ----------
        cfg : ProjectConfig
            Test configuration.
        """
        with patch("fusion_oncology.analysis.clinical_evidence.requests.post") as mock:
            mock.side_effect = Exception("Connection refused")
            result = query_opentargets("EGFR", cfg)
        assert isinstance(result, dict)
        assert result.get("overall_score", 0) == 0

    def test_returns_dict_on_success(self, cfg: ProjectConfig) -> None:
        """Return a well-formed dict when the API succeeds.

        Parameters
        ----------
        cfg : ProjectConfig
            Test configuration.
        """
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "data": {
                "search": {"hits": [{"id": "ENSG0001", "name": "EGFR"}]},
            }
        }

        with patch("fusion_oncology.analysis.clinical_evidence.requests.post") as mock:
            mock.return_value = mock_response
            result = query_opentargets("EGFR", cfg)
        assert isinstance(result, dict)


class TestQueryCivic:
    """Tests for the ``query_civic`` function."""

    def test_returns_dict_on_failure(self, cfg: ProjectConfig) -> None:
        """Return empty list when CIViC API is unreachable.

        Parameters
        ----------
        cfg : ProjectConfig
            Test configuration.
        """
        with patch("fusion_oncology.analysis.clinical_evidence.requests.post") as mock:
            mock.side_effect = Exception("timeout")
            result = query_civic("BRAF", cfg)
        assert isinstance(result, list)
        assert len(result) == 0


class TestQueryClinicalTrials:
    """Tests for the ``query_clinical_trials`` function."""

    def test_returns_dict_on_failure(self, cfg: ProjectConfig) -> None:
        """Return fallback list when ClinicalTrials.gov is unreachable.

        Parameters
        ----------
        cfg : ProjectConfig
            Test configuration.
        """
        with patch("fusion_oncology.analysis.clinical_evidence.requests.get") as mock:
            mock.side_effect = Exception("timeout")
            result = query_clinical_trials("KRAS", cfg)
        assert isinstance(result, list)
        assert len(result) == 0


class TestClinicalEvidenceAggregator:
    """Tests for the ``ClinicalEvidenceAggregator`` class."""

    def test_profile_returns_evidence_score(self, cfg: ProjectConfig) -> None:
        """Profile returns a dict with an evidence_score key.

        Parameters
        ----------
        cfg : ProjectConfig
            Test configuration.
        """
        agg = ClinicalEvidenceAggregator(cfg)
        with (
            patch(
                "fusion_oncology.analysis.clinical_evidence.query_opentargets",
                return_value={"overall_score": 0.5},
            ),
            patch(
                "fusion_oncology.analysis.clinical_evidence.query_civic",
                return_value=[],
            ),
            patch(
                "fusion_oncology.analysis.clinical_evidence.query_clinical_trials",
                return_value=[],
            ),
        ):
            result = agg.profile("EGFR")
        assert "evidence_score" in result
        assert isinstance(result["evidence_score"], float)

    def test_annotate_adds_columns(self, cfg: ProjectConfig) -> None:
        """Annotate adds an Evidence_Score column to a DataFrame.

        Parameters
        ----------
        cfg : ProjectConfig
            Test configuration.
        """
        import pandas as pd

        df = pd.DataFrame({"Gene": ["EGFR", "BRAF"]})
        agg = ClinicalEvidenceAggregator(cfg)
        with (
            patch(
                "fusion_oncology.analysis.clinical_evidence.query_opentargets",
                return_value={"overall_score": 0.5},
            ),
            patch(
                "fusion_oncology.analysis.clinical_evidence.query_civic",
                return_value=[],
            ),
            patch(
                "fusion_oncology.analysis.clinical_evidence.query_clinical_trials",
                return_value=[],
            ),
        ):
            result = agg.annotate(df)
        assert "Evidence_Score" in result.columns
