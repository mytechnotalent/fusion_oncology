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
    _build_civic_record,
    _build_ct_query,
    _civic_record_base,
    _civic_record_evidence,
    _collect_civic_evidence,
    _extract_ot_gene_id,
    _fetch_clinical_trials,
    _parse_opentargets_target,
    _parse_trial_details,
    _parse_trial_ids,
    _parse_trial_study,
    _post_civic,
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


# ── OpenTargets internal helpers ─────────────────────────────────────────


class TestExtractOtGeneId:
    """Tests for the ``_extract_ot_gene_id`` helper."""

    def test_extracts_id(self) -> None:
        """Should extract the first hit's ID.

        Returns
        -------
        None
        """
        data = {"search": {"hits": [{"id": "ENSG123"}]}}
        assert _extract_ot_gene_id(data, "EGFR") == "ENSG123"

    def test_no_hits_returns_none(self) -> None:
        """Should return None when no hits are found.

        Returns
        -------
        None
        """
        data = {"search": {"hits": []}}
        assert _extract_ot_gene_id(data, "FAKE") is None


class TestParseOpentargetsTarget:
    """Tests for the ``_parse_opentargets_target`` helper."""

    def test_parses_target_data(self) -> None:
        """Should extract score from associated diseases.

        Returns
        -------
        None
        """
        data = {
            "associatedDiseases": {"rows": [{"disease": {"name": "NSCLC"}, "score": 0.8}]},
            "tractability": [{"label": "Small molecule", "value": True}],
        }
        result = _parse_opentargets_target(data, "ENSG123")
        assert result["overall_score"] == 0.8
        assert result["ensembl_id"] == "ENSG123"


# ── CIViC internal helpers ──────────────────────────────────────────────


class TestCivicRecordBase:
    """Tests for the ``_civic_record_base`` helper."""

    def test_extracts_base_fields(self) -> None:
        """Should extract variant, disease, drugs, and rating.

        Returns
        -------
        None
        """
        ev = {
            "disease": {"name": "NSCLC"},
            "therapies": [{"name": "Erlotinib"}],
            "evidenceRating": 4,
        }
        rec = _civic_record_base("L858R", ev)
        assert rec["variant"] == "L858R"
        assert rec["disease"] == "NSCLC"
        assert "Erlotinib" in rec["drugs"]


class TestCivicRecordEvidence:
    """Tests for the ``_civic_record_evidence`` helper."""

    def test_extracts_evidence_fields(self) -> None:
        """Should extract evidence type, level, direction, significance.

        Returns
        -------
        None
        """
        ev = {
            "evidenceType": "Predictive",
            "evidenceLevel": "A",
            "evidenceDirection": "Supports",
            "significance": "Sensitivity/Response",
        }
        rec = _civic_record_evidence(ev)
        assert rec["evidence_type"] == "Predictive"
        assert rec["evidence_level"] == "A"


class TestBuildCivicRecord:
    """Tests for the ``_build_civic_record`` helper."""

    def test_merges_base_and_evidence(self) -> None:
        """Should combine base and evidence fields into one record.

        Returns
        -------
        None
        """
        ev = {
            "disease": {"name": "Melanoma"},
            "therapies": [],
            "evidenceRating": 3,
            "evidenceType": "Diagnostic",
            "evidenceLevel": "B",
            "evidenceDirection": "Supports",
            "significance": "Positive",
        }
        rec = _build_civic_record("V600E", ev)
        assert rec["variant"] == "V600E"
        assert rec["evidence_type"] == "Diagnostic"


class TestCollectCivicEvidence:
    """Tests for the ``_collect_civic_evidence`` helper."""

    def test_collects_from_nested_structure(self) -> None:
        """Should flatten variant → evidence items into a list.

        Returns
        -------
        None
        """
        genes = [
            {
                "variants": {
                    "nodes": [
                        {
                            "name": "V600E",
                            "evidenceItems": {
                                "nodes": [
                                    {
                                        "disease": {"name": "Melanoma"},
                                        "therapies": [],
                                        "evidenceRating": 3,
                                        "evidenceType": "Predictive",
                                        "evidenceLevel": "A",
                                        "evidenceDirection": "Supports",
                                        "significance": "Sensitivity",
                                    }
                                ]
                            },
                        }
                    ]
                },
            }
        ]
        items = _collect_civic_evidence(genes)
        assert len(items) == 1
        assert items[0]["variant"] == "V600E"


class TestPostCivic:
    """Tests for the ``_post_civic`` helper."""

    def test_success(self) -> None:
        """Should return gene nodes on success.

        Returns
        -------
        None
        """
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"data": {"genes": {"nodes": [{"name": "BRAF"}]}}}
        with patch(
            "fusion_oncology.analysis.clinical_evidence.requests.post", return_value=mock_resp
        ):
            result = _post_civic("BRAF")
        assert result is not None

    def test_no_nodes(self) -> None:
        """Should return None when no gene nodes exist.

        Returns
        -------
        None
        """
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"data": {"genes": {"nodes": []}}}
        with patch(
            "fusion_oncology.analysis.clinical_evidence.requests.post", return_value=mock_resp
        ):
            result = _post_civic("FAKE")
        assert result is None


# ── ClinicalTrials.gov helpers ──────────────────────────────────────────


class TestParseTrialIds:
    """Tests for the ``_parse_trial_ids`` helper."""

    def test_extracts_fields(self) -> None:
        """Should extract NCT ID, title, status, and phase.

        Returns
        -------
        None
        """
        proto = {
            "identificationModule": {"nctId": "NCT001", "briefTitle": "Test Trial"},
            "statusModule": {"overallStatus": "Recruiting"},
            "designModule": {"phases": ["Phase 2"]},
        }
        rec = _parse_trial_ids(proto)
        assert rec["nct_id"] == "NCT001"
        assert rec["status"] == "Recruiting"


class TestParseTrialDetails:
    """Tests for the ``_parse_trial_details`` helper."""

    def test_extracts_conditions_and_interventions(self) -> None:
        """Should extract conditions and intervention names.

        Returns
        -------
        None
        """
        proto = {
            "conditionsModule": {"conditions": ["NSCLC"]},
            "armsInterventionsModule": {"interventions": [{"name": "Drug X"}]},
        }
        rec = _parse_trial_details(proto)
        assert "NSCLC" in rec["conditions"]
        assert "Drug X" in rec["interventions"]


class TestParseTrialStudy:
    """Tests for the ``_parse_trial_study`` helper."""

    def test_combines_ids_and_details(self) -> None:
        """Should merge trial IDs and details into one record.

        Returns
        -------
        None
        """
        study = {
            "protocolSection": {
                "identificationModule": {"nctId": "NCT002", "briefTitle": "T"},
                "statusModule": {"overallStatus": "Active"},
                "designModule": {"phases": []},
                "conditionsModule": {"conditions": ["CRC"]},
                "armsInterventionsModule": {"interventions": []},
            }
        }
        rec = _parse_trial_study(study)
        assert rec["nct_id"] == "NCT002"
        assert "CRC" in rec["conditions"]


class TestBuildCtQuery:
    """Tests for the ``_build_ct_query`` helper."""

    def test_with_cancer_type(self) -> None:
        """Should include both gene and cancer type.

        Returns
        -------
        None
        """
        assert _build_ct_query("EGFR", "NSCLC") == "EGFR NSCLC"

    def test_without_cancer_type(self) -> None:
        """Should default to 'cancer' when no type given.

        Returns
        -------
        None
        """
        assert _build_ct_query("EGFR", None) == "EGFR cancer"


class TestFetchClinicalTrials:
    """Tests for the ``_fetch_clinical_trials`` helper."""

    def test_success(self) -> None:
        """Should return JSON on success.

        Returns
        -------
        None
        """
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"studies": []}
        with patch(
            "fusion_oncology.analysis.clinical_evidence.requests.get", return_value=mock_resp
        ):
            result = _fetch_clinical_trials("EGFR cancer", 10, "EGFR")
        assert result == {"studies": []}

    def test_failure_returns_none(self) -> None:
        """Should return None on network error.

        Returns
        -------
        None
        """
        with patch(
            "fusion_oncology.analysis.clinical_evidence.requests.get", side_effect=Exception("err")
        ):
            result = _fetch_clinical_trials("EGFR", 10, "EGFR")
        assert result is None


class TestQueryCivicSuccess:
    """Tests for ``query_civic`` with successful API response."""

    def test_returns_evidence_list(self, cfg: ProjectConfig) -> None:
        """Should return a list of evidence items.

        Parameters
        ----------
        cfg : ProjectConfig
            Test configuration.
        """
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "data": {
                "genes": {
                    "nodes": [
                        {
                            "variants": {
                                "nodes": [
                                    {
                                        "name": "V600E",
                                        "evidenceItems": {
                                            "nodes": [
                                                {
                                                    "disease": {"name": "Melanoma"},
                                                    "therapies": [{"name": "Vemurafenib"}],
                                                    "evidenceRating": 5,
                                                    "evidenceType": "Predictive",
                                                    "evidenceLevel": "A",
                                                    "evidenceDirection": "Supports",
                                                    "significance": "Sensitivity",
                                                }
                                            ]
                                        },
                                    }
                                ]
                            },
                        }
                    ]
                }
            },
        }
        with patch(
            "fusion_oncology.analysis.clinical_evidence.requests.post", return_value=mock_resp
        ):
            result = query_civic("BRAF", cfg)
        assert len(result) == 1
        assert result[0]["variant"] == "V600E"


class TestQueryClinicalTrialsSuccess:
    """Tests for ``query_clinical_trials`` with successful API response."""

    def test_returns_trials_list(self, cfg: ProjectConfig) -> None:
        """Should return a list of trial dicts on success.

        Parameters
        ----------
        cfg : ProjectConfig
            Test configuration.
        """
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "studies": [
                {
                    "protocolSection": {
                        "identificationModule": {"nctId": "NCT003", "briefTitle": "X"},
                        "statusModule": {"overallStatus": "Recruiting"},
                        "designModule": {"phases": ["Phase 3"]},
                        "conditionsModule": {"conditions": ["Lung Cancer"]},
                        "armsInterventionsModule": {"interventions": [{"name": "DrugA"}]},
                    }
                }
            ]
        }
        with patch(
            "fusion_oncology.analysis.clinical_evidence.requests.get", return_value=mock_resp
        ):
            result = query_clinical_trials("EGFR", config=cfg)
        assert len(result) == 1
        assert result[0]["nct_id"] == "NCT003"
