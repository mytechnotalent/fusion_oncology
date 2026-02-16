"""
Tests for the cell-line → patient domain adaptation module.

Covers quantile normalisation, ComBat batch correction, feature
alignment, and the end-to-end DomainAdapter pipeline.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fusion_oncology.data.domain_adaptation import (
    DomainAdaptConfig,
    DomainAdapter,
    align_features,
    combat_correct,
    quantile_normalise,
    quantile_normalise_to_reference,
)
from fusion_oncology.config import ProjectConfig


@pytest.fixture()
def cfg(tmp_path):
    """Provide a temporary ProjectConfig.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest-provided temporary directory.

    Returns
    -------
    ProjectConfig
        Config with temp paths.
    """
    return ProjectConfig(cache_dir=tmp_path / "cache", output_dir=tmp_path / "out")


@pytest.fixture()
def cell_line_expr():
    """Generate synthetic cell-line expression data.

    Returns
    -------
    pd.DataFrame
        50 cell lines x 100 genes.
    """
    rng = np.random.default_rng(42)
    genes = [f"GENE_{i}" for i in range(100)]
    return pd.DataFrame(
        rng.lognormal(3.0, 1.5, (50, 100)),
        index=[f"CL_{i}" for i in range(50)],
        columns=genes,
    )


@pytest.fixture()
def patient_expr():
    """Generate synthetic patient expression data.

    Returns
    -------
    pd.DataFrame
        30 patients x 100 genes (different distribution).
    """
    rng = np.random.default_rng(99)
    genes = [f"GENE_{i}" for i in range(100)]
    return pd.DataFrame(
        rng.lognormal(4.0, 1.2, (30, 100)),
        index=[f"PT_{i}" for i in range(30)],
        columns=genes,
    )


class TestQuantileNormalise:
    """Tests for quantile normalisation."""

    def test_output_shape_matches_input(self, cell_line_expr):
        """Output should have same shape as input.

        Returns
        -------
        None
        """
        result = quantile_normalise(cell_line_expr)
        assert result.shape == cell_line_expr.shape

    def test_columns_preserved(self, cell_line_expr):
        """Column names should be preserved.

        Returns
        -------
        None
        """
        result = quantile_normalise(cell_line_expr)
        assert list(result.columns) == list(cell_line_expr.columns)


class TestQuantileNormaliseToReference:
    """Tests for reference-based quantile normalisation."""

    def test_output_shape(self, cell_line_expr, patient_expr):
        """Output should have same rows as source.

        Returns
        -------
        None
        """
        result = quantile_normalise_to_reference(cell_line_expr, patient_expr)
        assert result.shape[0] == cell_line_expr.shape[0]

    def test_no_overlap_returns_unchanged(self):
        """With no common genes, source should return unchanged.

        Returns
        -------
        None
        """
        src = pd.DataFrame({"A": [1, 2, 3]})
        ref = pd.DataFrame({"B": [4, 5, 6]})
        result = quantile_normalise_to_reference(src, ref)
        pd.testing.assert_frame_equal(result, src)


class TestCombatCorrect:
    """Tests for ComBat batch correction."""

    def test_output_shape(self, cell_line_expr, patient_expr):
        """Corrected output should have same shape as input.

        Returns
        -------
        None
        """
        combined = pd.concat([cell_line_expr, patient_expr])
        batch = np.array(["CL"] * 50 + ["PT"] * 30)
        result = combat_correct(combined, batch)
        assert result.shape == combined.shape

    def test_single_batch_returns_copy(self, cell_line_expr):
        """Single batch should return the data unchanged.

        Returns
        -------
        None
        """
        batch = np.array(["A"] * len(cell_line_expr))
        result = combat_correct(cell_line_expr, batch)
        assert result.shape == cell_line_expr.shape

    def test_mismatched_lengths_raises(self, cell_line_expr):
        """Mismatched batch_labels length should raise ValueError.

        Returns
        -------
        None
        """
        batch = np.array(["A", "B"])  # Wrong length
        with pytest.raises(ValueError):
            combat_correct(cell_line_expr, batch)

    def test_parametric_vs_empirical(self, cell_line_expr, patient_expr):
        """Both parametric and empirical modes should run.

        Returns
        -------
        None
        """
        combined = pd.concat([cell_line_expr, patient_expr])
        batch = np.array(["CL"] * 50 + ["PT"] * 30)

        r_param = combat_correct(combined, batch, parametric=True)
        r_emp = combat_correct(combined, batch, parametric=False)
        assert r_param.shape == combined.shape
        assert r_emp.shape == combined.shape

    def test_reduces_batch_mean_difference(self, cell_line_expr, patient_expr):
        """ComBat should reduce mean differences between batches.

        Returns
        -------
        None
        """
        combined = pd.concat([cell_line_expr, patient_expr])
        batch = np.array(["CL"] * 50 + ["PT"] * 30)

        before_diff = abs(cell_line_expr.mean().mean() - patient_expr.mean().mean())

        corrected = combat_correct(combined, batch)
        cl_corrected = corrected.iloc[:50]
        pt_corrected = corrected.iloc[50:]
        after_diff = abs(cl_corrected.mean().mean() - pt_corrected.mean().mean())

        assert after_diff <= before_diff


class TestAlignFeatures:
    """Tests for gene alignment between domains."""

    def test_returns_list_of_strings(self, cell_line_expr, patient_expr):
        """align_features should return a list of gene names.

        Returns
        -------
        None
        """
        genes = align_features(cell_line_expr, patient_expr)
        assert isinstance(genes, list)
        if genes:
            assert isinstance(genes[0], str)

    def test_top_k_limits_output(self, cell_line_expr, patient_expr):
        """top_k should cap the number of genes returned.

        Returns
        -------
        None
        """
        genes = align_features(cell_line_expr, patient_expr, top_k=10)
        assert len(genes) <= 10

    def test_no_overlap_returns_empty(self):
        """No common genes should return an empty list.

        Returns
        -------
        None
        """
        cl = pd.DataFrame(
            np.random.randn(10, 5),
            columns=[f"A{i}" for i in range(5)],
        )
        pt = pd.DataFrame(
            np.random.randn(10, 5),
            columns=[f"B{i}" for i in range(5)],
        )
        assert align_features(cl, pt) == []


class TestDomainAdapter:
    """Tests for the end-to-end adaptation pipeline."""

    def test_adapt_returns_two_dataframes(self, cell_line_expr, patient_expr):
        """adapt should return two adapted DataFrames.

        Returns
        -------
        None
        """
        adapter = DomainAdapter()
        cl_out, pt_out = adapter.adapt(cell_line_expr, patient_expr)
        assert isinstance(cl_out, pd.DataFrame)
        assert isinstance(pt_out, pd.DataFrame)

    def test_adapted_share_columns(self, cell_line_expr, patient_expr):
        """Adapted outputs should have the same columns.

        Returns
        -------
        None
        """
        adapter = DomainAdapter()
        cl_out, pt_out = adapter.adapt(cell_line_expr, patient_expr)
        assert list(cl_out.columns) == list(pt_out.columns)

    def test_adapted_row_counts_preserved(self, cell_line_expr, patient_expr):
        """Row counts should match input.

        Returns
        -------
        None
        """
        adapter = DomainAdapter()
        cl_out, pt_out = adapter.adapt(cell_line_expr, patient_expr)
        assert len(cl_out) == len(cell_line_expr)
        assert len(pt_out) == len(patient_expr)

    def test_aligned_genes_populated(self, cell_line_expr, patient_expr):
        """aligned_genes should be populated after adapt().

        Returns
        -------
        None
        """
        adapter = DomainAdapter()
        adapter.adapt(cell_line_expr, patient_expr)
        assert len(adapter.aligned_genes) > 0

    def test_summary_has_expected_keys(self, cell_line_expr, patient_expr):
        """summary should contain n_aligned_genes and config.

        Returns
        -------
        None
        """
        adapter = DomainAdapter()
        adapter.adapt(cell_line_expr, patient_expr)
        s = adapter.summary()
        assert "n_aligned_genes" in s
        assert "quantile_target" in s
        assert s["n_aligned_genes"] > 0

    def test_custom_config(self, cell_line_expr, patient_expr):
        """Custom DomainAdaptConfig should be respected.

        Returns
        -------
        None
        """
        dc = DomainAdaptConfig(top_k_genes=5)
        adapter = DomainAdapter(config=dc)
        cl_out, pt_out = adapter.adapt(cell_line_expr, patient_expr)
        assert cl_out.shape[1] <= 5
