"""
Tests for the companion diagnostic module.

Covers PatientProfile, CompanionDiagnostic analysis pipeline,
treatment plan ranking, and report generation.
"""

from __future__ import annotations

import pytest

from fusion_oncology.models.companion_dx import (
    CompanionDiagnostic,
    PatientProfile,
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
def sample_patient() -> PatientProfile:
    """Create a sample patient profile for testing.

    Returns
    -------
    PatientProfile
        Patient with EGFR and BRAF mutations.
    """
    return PatientProfile(
        patient_id="TEST_001",
        mutations=[
            {"gene": "EGFR", "variant": "L858R"},
            {"gene": "BRAF", "variant": "V600E"},
            {"gene": "TP53", "variant": "R175H"},
        ],
        expression={"EGFR": 12.5, "BRAF": 8.3, "TP53": 2.1},
        cna={"ERBB2": 1.2, "PTEN": -1.5, "MYC": 0.8},
        cancer_type="NSCLC",
        metadata={"age": 62, "stage": "IIIB"},
    )


class TestPatientProfile:
    """Tests for the PatientProfile class."""

    def test_mutated_genes(self, sample_patient: PatientProfile) -> None:
        """Mutated genes should be sorted and unique.

        Parameters
        ----------
        sample_patient : PatientProfile
            Test patient.
        """
        genes = sample_patient.mutated_genes
        assert "EGFR" in genes
        assert "BRAF" in genes
        assert genes == sorted(genes)

    def test_tmb(self, sample_patient: PatientProfile) -> None:
        """TMB should equal number of mutations.

        Parameters
        ----------
        sample_patient : PatientProfile
            Test patient.
        """
        assert sample_patient.tumour_mutational_burden == 3

    def test_amplified_genes(self, sample_patient: PatientProfile) -> None:
        """Amplified genes filter by CNA threshold.

        Parameters
        ----------
        sample_patient : PatientProfile
            Test patient.
        """
        amp = sample_patient.amplified_genes(threshold=0.5)
        assert "ERBB2" in amp
        assert "MYC" in amp
        assert "PTEN" not in amp

    def test_deleted_genes(self, sample_patient: PatientProfile) -> None:
        """Deleted genes filter by CNA threshold.

        Parameters
        ----------
        sample_patient : PatientProfile
            Test patient.
        """
        deleted = sample_patient.deleted_genes(threshold=-0.5)
        assert "PTEN" in deleted
        assert "ERBB2" not in deleted

    def test_empty_patient(self) -> None:
        """An empty patient should have zero TMB.

        Returns
        -------
        None
        """
        patient = PatientProfile(patient_id="EMPTY", mutations=[])
        assert patient.tumour_mutational_burden == 0
        assert patient.mutated_genes == []


class TestCompanionDiagnostic:
    """Tests for the CompanionDiagnostic analysis engine."""

    def test_analyse_returns_expected_keys(
        self, cfg: ProjectConfig, sample_patient: PatientProfile
    ) -> None:
        """Analysis result should have required top-level keys.

        Parameters
        ----------
        cfg : ProjectConfig
            Test configuration.
        sample_patient : PatientProfile
            Test patient.
        """
        dx = CompanionDiagnostic(cfg)
        result = dx.analyse(sample_patient)
        assert "patient_id" in result
        assert "treatment_plan" in result
        assert "actionable_mutations" in result
        assert "drug_matches" in result
        assert "resistance_alerts" in result
        assert "timestamp" in result

    def test_treatment_plan_ranked(
        self, cfg: ProjectConfig, sample_patient: PatientProfile
    ) -> None:
        """Treatment plan should be ranked by confidence.

        Parameters
        ----------
        cfg : ProjectConfig
            Test configuration.
        sample_patient : PatientProfile
            Test patient.
        """
        dx = CompanionDiagnostic(cfg)
        result = dx.analyse(sample_patient)
        plan = result["treatment_plan"]
        if len(plan) > 1:
            confidences = [r["confidence"] for r in plan]
            assert confidences == sorted(confidences, reverse=True)

    def test_actionable_mutations_tiered(
        self, cfg: ProjectConfig, sample_patient: PatientProfile
    ) -> None:
        """Actionable mutations should be classified into tiers.

        Parameters
        ----------
        cfg : ProjectConfig
            Test configuration.
        sample_patient : PatientProfile
            Test patient.
        """
        dx = CompanionDiagnostic(cfg)
        result = dx.analyse(sample_patient)
        for mut in result["actionable_mutations"]:
            assert "tier" in mut
            assert mut["tier"] in {"I", "II", "III", "IV"}

    def test_generate_report(
        self, cfg: ProjectConfig, sample_patient: PatientProfile
    ) -> None:
        """Generate report should produce a non-empty string.

        Parameters
        ----------
        cfg : ProjectConfig
            Test configuration.
        sample_patient : PatientProfile
            Test patient.
        """
        dx = CompanionDiagnostic(cfg)
        result = dx.analyse(sample_patient)
        report = dx.generate_report(result)
        assert isinstance(report, str)
        assert "COMPANION DIAGNOSTIC REPORT" in report
        assert sample_patient.patient_id in report

    def test_drug_matches_for_egfr_patient(
        self, cfg: ProjectConfig, sample_patient: PatientProfile
    ) -> None:
        """EGFR-mutant patient should have drug matches.

        Parameters
        ----------
        cfg : ProjectConfig
            Test configuration.
        sample_patient : PatientProfile
            Test patient.
        """
        dx = CompanionDiagnostic(cfg)
        result = dx.analyse(sample_patient)
        drugs = [m["drug"] for m in result["drug_matches"]]
        assert len(drugs) > 0
