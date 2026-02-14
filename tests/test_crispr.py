"""
Tests for the CRISPR guide RNA design module.

Covers guide scanning, on-target scoring, off-target heuristics,
and the CRISPRDesigner class.
"""

from __future__ import annotations

import pandas as pd
import pytest

from fusion_oncology.models.crispr import (
    CRISPRDesigner,
    find_guides_in_sequence,
    off_target_heuristic,
    on_target_score,
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


# A synthetic sequence with known NGG PAM sites
SYNTHETIC_SEQ = (
    "ATGCATGCATGCATGCATGCATGCAGG"  # 27 nt — has NGG at positions 25-26
    "GCTAGCTAGCTAGCTAGCTAGCAGG"  # another NGG site
    "ATATATATATATATATATATATAGGG"  # another site
)


class TestOnTargetScore:
    """Tests for the ``on_target_score`` function."""

    def test_returns_float(self) -> None:
        """Score should be a float.

        Returns
        -------
        None
        """
        score = on_target_score("GCATGCATGCATGCATGCAT")
        assert isinstance(score, float)

    def test_score_in_range(self) -> None:
        """Score should be between 0 and 1.

        Returns
        -------
        None
        """
        score = on_target_score("GCATGCATGCATGCATGCAT")
        assert 0.0 <= score <= 1.0

    def test_wrong_length_returns_zero(self) -> None:
        """A guide of wrong length should score 0.

        Returns
        -------
        None
        """
        assert on_target_score("ACGT") == 0.0

    def test_poly_t_penalised(self) -> None:
        """A guide with poly-T should score lower than one without.

        Returns
        -------
        None
        """
        good = on_target_score("GCATGCATGCATGCATGCAT")
        bad = on_target_score("GCATTTTTTTTTGCATGCAT")
        assert bad < good


class TestOffTargetHeuristic:
    """Tests for the ``off_target_heuristic`` function."""

    def test_returns_float(self) -> None:
        """Risk score should be a float.

        Returns
        -------
        None
        """
        score = off_target_heuristic("GCATGCATGCATGCATGCAT")
        assert isinstance(score, float)

    def test_score_in_range(self) -> None:
        """Risk score should be between 0 and 1.

        Returns
        -------
        None
        """
        score = off_target_heuristic("GCATGCATGCATGCATGCAT")
        assert 0.0 <= score <= 1.0


class TestFindGuidesInSequence:
    """Tests for the ``find_guides_in_sequence`` function."""

    def test_returns_list(self) -> None:
        """Should return a list.

        Returns
        -------
        None
        """
        guides = find_guides_in_sequence(SYNTHETIC_SEQ, gene="TEST")
        assert isinstance(guides, list)

    def test_guides_have_expected_keys(self) -> None:
        """Each guide dict should have required keys.

        Returns
        -------
        None
        """
        guides = find_guides_in_sequence(SYNTHETIC_SEQ, gene="TEST")
        if guides:
            g = guides[0]
            assert "guide_id" in g
            assert "sequence" in g
            assert "on_target" in g
            assert "composite_score" in g

    def test_guides_sorted_by_score(self) -> None:
        """Guides should be sorted by composite_score descending.

        Returns
        -------
        None
        """
        guides = find_guides_in_sequence(SYNTHETIC_SEQ, gene="TEST")
        if len(guides) > 1:
            scores = [g["composite_score"] for g in guides]
            assert scores == sorted(scores, reverse=True)


class TestCRISPRDesigner:
    """Tests for the ``CRISPRDesigner`` class."""

    def test_init(self, cfg: ProjectConfig) -> None:
        """Designer should initialise without error.

        Parameters
        ----------
        cfg : ProjectConfig
            Test configuration.
        """
        designer = CRISPRDesigner(config=cfg)
        assert designer is not None

    def test_design_for_gene(self, cfg: ProjectConfig) -> None:
        """Design should return a list of guide dicts.

        Parameters
        ----------
        cfg : ProjectConfig
            Test configuration.
        """
        designer = CRISPRDesigner(config=cfg)
        guides = designer.design_for_gene("TEST", SYNTHETIC_SEQ)
        assert isinstance(guides, list)

    def test_library_dataframe(self, cfg: ProjectConfig) -> None:
        """Library DataFrame should have expected columns.

        Parameters
        ----------
        cfg : ProjectConfig
            Test configuration.
        """
        designer = CRISPRDesigner(config=cfg)
        designer.design_for_gene("TEST", SYNTHETIC_SEQ)
        df = designer.library_dataframe()
        assert isinstance(df, pd.DataFrame)
        assert "guide_id" in df.columns
        assert "gene" in df.columns

    def test_export_library(self, cfg: ProjectConfig) -> None:
        """Export should write a CSV file.

        Parameters
        ----------
        cfg : ProjectConfig
            Test configuration.
        """
        designer = CRISPRDesigner(config=cfg)
        designer.design_for_gene("TEST", SYNTHETIC_SEQ)
        path = designer.export_library()
        assert path.endswith(".csv") or path.endswith(".csv")

    def test_summary_empty(self, cfg: ProjectConfig) -> None:
        """Summary with no designs should report zero.

        Parameters
        ----------
        cfg : ProjectConfig
            Test configuration.
        """
        designer = CRISPRDesigner(config=cfg)
        summary = designer.summary()
        assert summary["total_guides"] == 0
