"""
Tests for the GDSC data loader module.

Covers synthetic fallback generation, GDSCLoader in offline mode,
drug sensitivity queries, resistant / sensitive cell-line lookups,
training matrix construction, and summary output.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fusion_oncology.data.gdsc import (
    GDSCDrugResponse,
    GDSCLoader,
    _generate_synthetic_gdsc,
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
def loader(cfg):
    """Build a GDSCLoader in offline mode.

    Parameters
    ----------
    cfg : ProjectConfig
        Test configuration.

    Returns
    -------
    GDSCLoader
        Loader with synthetic data.
    """
    return GDSCLoader(config=cfg, offline=True)


class TestGDSCDrugResponse:
    """Tests for the GDSCDrugResponse dataclass."""

    def test_default_values(self):
        """Default response should have empty/zero fields.

        Returns
        -------
        None
        """
        r = GDSCDrugResponse()
        assert r.cell_line == ""
        assert r.ln_ic50 == 0.0
        assert r.auc == 0.0


class TestSyntheticGeneration:
    """Tests for the synthetic GDSC data generator."""

    def test_returns_three_dataframes(self):
        """Generator should return dose_response, expression, mutations.

        Returns
        -------
        None
        """
        data = _generate_synthetic_gdsc(n_cell_lines=10, n_drugs=3, n_genes=20)
        assert "dose_response" in data
        assert "expression" in data
        assert "mutations" in data

    def test_dose_response_shape(self):
        """Dose-response should have n_cell_lines * n_drugs rows.

        Returns
        -------
        None
        """
        data = _generate_synthetic_gdsc(n_cell_lines=10, n_drugs=5, n_genes=20)
        assert len(data["dose_response"]) == 50

    def test_expression_shape(self):
        """Expression matrix should be n_cell_lines x n_genes.

        Returns
        -------
        None
        """
        data = _generate_synthetic_gdsc(n_cell_lines=10, n_drugs=3, n_genes=20)
        assert data["expression"].shape == (10, 20)

    def test_mutations_binary(self):
        """Mutation matrix values should be 0 or 1.

        Returns
        -------
        None
        """
        data = _generate_synthetic_gdsc(n_cell_lines=10, n_drugs=3, n_genes=20)
        unique_vals = set(data["mutations"].values.flatten())
        assert unique_vals.issubset({0, 1})

    def test_deterministic_with_seed(self):
        """Same seed should produce identical data.

        Returns
        -------
        None
        """
        d1 = _generate_synthetic_gdsc(seed=123)
        d2 = _generate_synthetic_gdsc(seed=123)
        pd.testing.assert_frame_equal(d1["dose_response"], d2["dose_response"])


class TestGDSCLoaderOffline:
    """Tests for GDSCLoader in offline mode."""

    def test_load_populates_data(self, loader):
        """After load(), dataframes should be populated.

        Returns
        -------
        None
        """
        loader.load()
        assert loader._dose_df is not None
        assert loader._expr_df is not None
        assert loader._mut_df is not None

    def test_dose_response_property(self, loader):
        """dose_response property should return a DataFrame.

        Returns
        -------
        None
        """
        df = loader.dose_response
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

    def test_expression_property(self, loader):
        """expression property should return a DataFrame.

        Returns
        -------
        None
        """
        expr = loader.expression
        assert isinstance(expr, pd.DataFrame)
        assert expr.shape[0] > 0

    def test_mutations_property(self, loader):
        """mutations property should return a DataFrame.

        Returns
        -------
        None
        """
        mut = loader.mutations
        assert isinstance(mut, pd.DataFrame)


class TestDrugSensitivityQueries:
    """Tests for drug-specific queries."""

    def test_drug_sensitivity_filter(self, loader):
        """drug_sensitivity should return filtered rows.

        Returns
        -------
        None
        """
        df = loader.dose_response
        first_drug = df["DRUG_NAME"].iloc[0]
        result = loader.drug_sensitivity(first_drug)
        assert len(result) > 0
        assert (result["DRUG_NAME"] == first_drug).all()

    def test_drug_sensitivity_tissue_filter(self, loader):
        """Tissue filter should further restrict results.

        Returns
        -------
        None
        """
        df = loader.dose_response
        first_drug = df["DRUG_NAME"].iloc[0]
        first_tissue = df["TCGA_DESC"].iloc[0]
        result = loader.drug_sensitivity(first_drug, tissue=first_tissue)
        assert isinstance(result, pd.DataFrame)

    def test_resistant_cell_lines_returns_list(self, loader):
        """resistant_cell_lines should return a list of strings.

        Returns
        -------
        None
        """
        df = loader.dose_response
        first_drug = df["DRUG_NAME"].iloc[0]
        result = loader.resistant_cell_lines(first_drug)
        assert isinstance(result, list)

    def test_sensitive_cell_lines_returns_list(self, loader):
        """sensitive_cell_lines should return a list of strings.

        Returns
        -------
        None
        """
        df = loader.dose_response
        first_drug = df["DRUG_NAME"].iloc[0]
        result = loader.sensitive_cell_lines(first_drug)
        assert isinstance(result, list)


class TestTrainingMatrix:
    """Tests for XGBoost training matrix construction."""

    def test_training_matrix_returns_tuple(self, loader):
        """training_matrix should return (X, y) tuple.

        Returns
        -------
        None
        """
        df = loader.dose_response
        first_drug = df["DRUG_NAME"].iloc[0]
        X, y = loader.training_matrix(first_drug)
        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, np.ndarray)

    def test_training_matrix_labels_binary(self, loader):
        """Labels should be binary (0 or 1).

        Returns
        -------
        None
        """
        df = loader.dose_response
        first_drug = df["DRUG_NAME"].iloc[0]
        X, y = loader.training_matrix(first_drug)
        if len(y) > 0:
            assert set(np.unique(y)).issubset({0, 1})

    def test_training_matrix_x_y_aligned(self, loader):
        """X and y should have the same number of samples.

        Returns
        -------
        None
        """
        df = loader.dose_response
        first_drug = df["DRUG_NAME"].iloc[0]
        X, y = loader.training_matrix(first_drug)
        assert len(X) == len(y)


class TestGDSCSummary:
    """Tests for the summary output."""

    def test_summary_has_expected_keys(self, loader):
        """Summary should report cell lines, drugs, tissues.

        Returns
        -------
        None
        """
        s = loader.summary()
        assert "n_cell_lines" in s
        assert "n_drugs" in s
        assert "n_tissues" in s
        assert "mode" in s

    def test_summary_offline_mode(self, loader):
        """Summary should report offline mode.

        Returns
        -------
        None
        """
        s = loader.summary()
        assert s["mode"] == "offline"
