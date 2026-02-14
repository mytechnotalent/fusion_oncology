"""
Tests for the neoantigen prediction module.

Covers codon translation, peptide generation, MHC binding scoring,
and the NeoantigenPredictor class.
"""

from __future__ import annotations

import pandas as pd
import pytest

from fusion_oncology.analysis.neoantigen import (
    NeoantigenPredictor,
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
        peptides = generate_mutant_peptides(
            "ACDEFGHIK", mutation_position=3, mutant_aa="W"
        )
        assert isinstance(peptides, list)
        assert all(isinstance(p, dict) for p in peptides)

    def test_peptides_contain_mutant(self) -> None:
        """All generated peptides should contain the mutant amino acid.

        Returns
        -------
        None
        """
        peptides = generate_mutant_peptides(
            "ACDEFGHIKLMNPQ", mutation_position=5, mutant_aa="W"
        )
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
