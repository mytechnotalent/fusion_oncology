"""
Tests for the neoantigen prediction module.

Covers codon translation, peptide generation, MHC binding scoring,
and the NeoantigenPredictor class.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fusion_oncology.analysis.neoantigen import (
    NeoantigenPredictor,
    _empty_neoantigen_df,
    _finalise_candidates,
    _get_protein_sequence,
    _make_candidate_record,
    _parse_hgvsp,
    _priority_label,
    _process_mutation_row,
    _score_and_filter_peptides,
    generate_mutant_peptides,
    score_mhc_binding,
    translate_sequence,
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


class TestTranslateSequence:
    """Tests for the ``translate_sequence`` helper."""

    def test_known_codon(self) -> None:
        """ATG should translate to M (Methionine).

        Returns
        -------
        None
        """
        result = translate_sequence("ATG")
        assert result == "M"

    def test_stop_codon(self) -> None:
        """TAA should translate to empty string (stop codon).

        Returns
        -------
        None
        """
        result = translate_sequence("TAA")
        assert result == ""

    def test_multi_codon(self) -> None:
        """A multi-codon sequence should produce a protein string.

        Returns
        -------
        None
        """
        # ATG GCT = Met Ala
        result = translate_sequence("ATGGCT")
        assert len(result) == 2
        assert result[0] == "M"

    def test_partial_codon_ignored(self) -> None:
        """Trailing bases that don't form a full codon should be ignored.

        Returns
        -------
        None
        """
        result = translate_sequence("ATGG")
        assert result == "M"


class TestGenerateMutantPeptides:
    """Tests for the ``generate_mutant_peptides`` function."""

    def test_returns_list(self) -> None:
        """Should return a list of peptide strings.

        Returns
        -------
        None
        """
        peptides = generate_mutant_peptides("ACDEFGHIK", mutation_position=3, mutant_aa="W")
        assert isinstance(peptides, list)
        assert all(isinstance(p, dict) for p in peptides)

    def test_peptides_contain_mutant(self) -> None:
        """All generated peptides should contain the mutant amino acid.

        Returns
        -------
        None
        """
        peptides = generate_mutant_peptides("ACDEFGHIKLMNPQ", mutation_position=5, mutant_aa="W")
        assert all("W" in p["mut_peptide"] for p in peptides)

    def test_empty_on_short_sequence(self) -> None:
        """A sequence shorter than min_len should produce no peptides.

        Returns
        -------
        None
        """
        peptides = generate_mutant_peptides("ACE", mutation_position=1, mutant_aa="W")
        assert len(peptides) == 0


class TestScoreMhcBinding:
    """Tests for the ``score_mhc_binding`` function."""

    def test_returns_float(self) -> None:
        """Score should be a float.

        Returns
        -------
        None
        """
        score = score_mhc_binding("ACDEFGHIK")
        assert isinstance(score, float)

    def test_score_in_range(self) -> None:
        """Score should be between 0 and 1.

        Returns
        -------
        None
        """
        score = score_mhc_binding("GILGFVFTL")
        assert 0.0 <= score <= 1.0


class TestNeoantigenPredictor:
    """Tests for the ``NeoantigenPredictor`` class."""

    def test_init(self, cfg: ProjectConfig) -> None:
        """Predictor should initialise without error.

        Parameters
        ----------
        cfg : ProjectConfig
            Test configuration.
        """
        pred = NeoantigenPredictor(cfg)
        assert pred is not None

    def test_summarise_empty(self, cfg: ProjectConfig) -> None:
        """Summarise with no predictions should return zero counts.

        Parameters
        ----------
        cfg : ProjectConfig
            Test configuration.
        """
        pred = NeoantigenPredictor(cfg)
        empty_df = pd.DataFrame(
            columns=["Gene", "MHC_Score", "Peptide"],
        )
        summary = pred.summarise(empty_df)
        assert isinstance(summary, dict)
        assert len(summary) == 0


# ── _priority_label ─────────────────────────────────────────────────────


class TestPriorityLabel:
    """Tests for the ``_priority_label`` helper."""

    def test_high(self) -> None:
        """Score >= 0.75 should yield HIGH.

        Returns
        -------
        None
        """
        assert _priority_label(0.80) == "HIGH"

    def test_medium(self) -> None:
        """Score >= 0.5 but < 0.75 should yield MEDIUM.

        Returns
        -------
        None
        """
        assert _priority_label(0.60) == "MEDIUM"

    def test_low(self) -> None:
        """Score < 0.5 should yield LOW.

        Returns
        -------
        None
        """
        assert _priority_label(0.30) == "LOW"


# ── _parse_hgvsp ────────────────────────────────────────────────────────


class TestParseHgvsp:
    """Tests for the ``_parse_hgvsp`` helper."""

    def test_valid_hgvsp(self) -> None:
        """Standard p.V600E notation should parse position and mutant AA.

        Returns
        -------
        None
        """
        result = _parse_hgvsp("p.V600E")
        assert result is not None
        assert result[0] == 599
        assert result[1] == "E"

    def test_short_string_returns_none(self) -> None:
        """Strings too short to parse should return None.

        Returns
        -------
        None
        """
        assert _parse_hgvsp("p.") is None

    def test_empty_returns_none(self) -> None:
        """Empty or None-like input should return None.

        Returns
        -------
        None
        """
        assert _parse_hgvsp("") is None


# ── _get_protein_sequence ───────────────────────────────────────────────


class TestGetProteinSequence:
    """Tests for the ``_get_protein_sequence`` helper."""

    def test_returns_known_sequence(self) -> None:
        """When gene is in gene_sequences dict, return that sequence.

        Returns
        -------
        None
        """
        seqs = {"EGFR": "MRPSGTAGAA"}
        result = _get_protein_sequence("EGFR", seqs, 5)
        assert result == "MRPSGTAGAA"

    def test_generates_synthetic(self) -> None:
        """When gene is absent, generate a synthetic protein sequence.

        Returns
        -------
        None
        """
        result = _get_protein_sequence("FAKE", None, 5)
        assert len(result) >= 55


# ── _make_candidate_record ──────────────────────────────────────────────


class TestMakeCandidateRecord:
    """Tests for the ``_make_candidate_record`` helper."""

    def test_returns_expected_keys(self) -> None:
        """Record should have Gene, Mutation, MHC_Score, Priority keys.

        Returns
        -------
        None
        """
        pep = {"wt_peptide": "ACDEF", "mut_peptide": "ACWEF", "length": 5}
        rec = _make_candidate_record("EGFR", "p.D3W", pep, 0.80)
        assert rec["Gene"] == "EGFR"
        assert rec["Priority"] == "HIGH"
        assert rec["MHC_Score"] == 0.80


# ── _empty_neoantigen_df ────────────────────────────────────────────────


class TestEmptyNeoantigenDf:
    """Tests for the ``_empty_neoantigen_df`` helper."""

    def test_returns_empty_with_columns(self) -> None:
        """Should return a DataFrame with the expected columns but no rows.

        Returns
        -------
        None
        """
        df = _empty_neoantigen_df()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0
        assert "Gene" in df.columns


# ── _finalise_candidates ────────────────────────────────────────────────


class TestFinaliseCandidates:
    """Tests for the ``_finalise_candidates`` helper."""

    def test_sorts_by_mhc_score(self) -> None:
        """Candidates should be sorted by MHC_Score descending.

        Returns
        -------
        None
        """
        cands = [
            {
                "Gene": "A",
                "Mutation": "m1",
                "WT_Peptide": "X",
                "Mut_Peptide": "Y",
                "Length": 9,
                "MHC_Score": 0.3,
                "Priority": "LOW",
            },
            {
                "Gene": "B",
                "Mutation": "m2",
                "WT_Peptide": "X",
                "Mut_Peptide": "Z",
                "Length": 9,
                "MHC_Score": 0.9,
                "Priority": "HIGH",
            },
        ]
        result = _finalise_candidates(cands)
        assert result.iloc[0]["MHC_Score"] == 0.9

    def test_empty_list(self) -> None:
        """Empty candidate list should produce empty DataFrame.

        Returns
        -------
        None
        """
        result = _finalise_candidates([])
        assert len(result) == 0


# ── _process_mutation_row ────────────────────────────────────────────────


class TestProcessMutationRow:
    """Tests for the ``_process_mutation_row`` helper."""

    def test_valid_row(self) -> None:
        """A valid mutation row should produce candidate peptides.

        Returns
        -------
        None
        """
        row = pd.Series({"Hugo_Symbol": "EGFR", "HGVSp_Short": "p.L858R"})
        result = _process_mutation_row(row, None, 0.0)
        assert isinstance(result, list)

    def test_invalid_hgvsp_returns_empty(self) -> None:
        """Invalid HGVSp should return empty list.

        Returns
        -------
        None
        """
        row = pd.Series({"Hugo_Symbol": "EGFR", "HGVSp_Short": ""})
        result = _process_mutation_row(row, None, 0.0)
        assert result == []


# ── _score_and_filter_peptides ──────────────────────────────────────────


class TestScoreAndFilterPeptides:
    """Tests for the ``_score_and_filter_peptides`` helper."""

    def test_filters_below_threshold(self) -> None:
        """Peptides scoring below threshold should be excluded.

        Returns
        -------
        None
        """
        peptides = [{"wt_peptide": "ACDEF", "mut_peptide": "ACWEF", "length": 5}]
        result = _score_and_filter_peptides(peptides, "G", "m", 999.0)
        assert len(result) == 0


# ── NeoantigenPredictor.predict_from_maf ────────────────────────────────


class TestPredictFromMaf:
    """Tests for ``NeoantigenPredictor.predict_from_maf``."""

    def test_returns_dataframe(self, cfg: ProjectConfig) -> None:
        """Should return a DataFrame with expected columns.

        Parameters
        ----------
        cfg : ProjectConfig
            Test configuration.
        """
        pred = NeoantigenPredictor(cfg)
        maf = pd.DataFrame(
            {
                "Hugo_Symbol": ["EGFR", "BRAF"],
                "Variant_Classification": ["Missense_Mutation", "Silent"],
                "HGVSp_Short": ["p.L858R", "p.V600E"],
            }
        )
        result = pred.predict_from_maf(maf)
        assert isinstance(result, pd.DataFrame)

    def test_no_missense_returns_empty(self, cfg: ProjectConfig) -> None:
        """No missense variants should produce an empty result.

        Parameters
        ----------
        cfg : ProjectConfig
            Test configuration.
        """
        pred = NeoantigenPredictor(cfg)
        maf = pd.DataFrame(
            {
                "Hugo_Symbol": ["TP53"],
                "Variant_Classification": ["Silent"],
                "HGVSp_Short": ["p.X1Y"],
            }
        )
        result = pred.predict_from_maf(maf)
        assert len(result) == 0


class TestSummariseWithData:
    """Tests for ``NeoantigenPredictor.summarise`` with real data."""

    def test_groups_by_gene(self, cfg: ProjectConfig) -> None:
        """Summary should group candidates by gene.

        Parameters
        ----------
        cfg : ProjectConfig
            Test configuration.
        """
        pred = NeoantigenPredictor(cfg)
        df = pd.DataFrame(
            {
                "Gene": ["EGFR", "EGFR", "BRAF"],
                "Mutation": ["m1", "m2", "m3"],
                "WT_Peptide": ["A", "B", "C"],
                "Mut_Peptide": ["X", "Y", "Z"],
                "Length": [9, 9, 9],
                "MHC_Score": [0.8, 0.6, 0.9],
                "Priority": ["HIGH", "MEDIUM", "HIGH"],
            }
        )
        summary = pred.summarise(df)
        assert "EGFR" in summary
        assert "BRAF" in summary
        assert summary["EGFR"]["n_candidates"] == 2
