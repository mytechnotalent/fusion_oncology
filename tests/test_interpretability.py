"""
Tests for the SHAP-based interpretability module.

Covers ShapExplainer construction, SHAP value computation,
gene importance ranking, pathway aggregation, and mechanistic
rationale generation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fusion_oncology.analysis.interpretability import ShapExplainer
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
    return ProjectConfig(
        cache_dir=tmp_path / "cache",
        output_dir=tmp_path / "out",
        xgb_n_estimators=5,
        xgb_max_depth=2,
        top_k_genes=5,
        enable_feature_engineering=False,
    )


@pytest.fixture()
def trained_engine(cfg):
    """Train a minimal XGBoost engine for explainability testing.

    Parameters
    ----------
    cfg : ProjectConfig
        Test configuration.

    Returns
    -------
    XGBoostEngine
        Fitted engine with a tiny synthetic dataset.
    """
    from fusion_oncology.models.xgboost_engine import XGBoostEngine

    rng = np.random.default_rng(42)
    genes = ["EGFR", "BRAF", "TP53", "KRAS", "ALK"]
    X = pd.DataFrame(rng.random((60, 5)), columns=genes)
    y = pd.Series(np.repeat(["LUAD", "BRCA", "KIRC"], 20), name="Class")
    engine = XGBoostEngine(config=cfg)
    engine.fit(X, y)
    return engine


@pytest.fixture()
def explainer(trained_engine):
    """Build a ShapExplainer from a trained XGBoost engine.

    Parameters
    ----------
    trained_engine : XGBoostEngine
        Fitted engine.

    Returns
    -------
    ShapExplainer
        Ready-to-use explainer.
    """
    return ShapExplainer.from_engine(trained_engine)


class TestShapExplainerConstruction:
    """Tests for ShapExplainer construction and validation."""

    def test_from_engine_returns_explainer(self, explainer):
        """ShapExplainer.from_engine should return a valid instance.

        Returns
        -------
        None
        """
        assert isinstance(explainer, ShapExplainer)

    def test_feature_names_stored(self, explainer):
        """Explainer should store feature names from the engine.

        Returns
        -------
        None
        """
        assert len(explainer._feature_names) == 5


class TestShapValues:
    """Tests for SHAP value computation."""

    def test_shap_values_shape(self, explainer, trained_engine):
        """SHAP values should match the input shape.

        Returns
        -------
        None
        """
        rng = np.random.default_rng(0)
        X = pd.DataFrame(
            rng.random((10, 5)),
            columns=["EGFR", "BRAF", "TP53", "KRAS", "ALK"],
        )
        vals = explainer.shap_values(X)
        assert vals.shape[0] == 10
        # Multi-class: shape is (n_samples, n_features, n_classes)
        # or for binary: (n_samples, n_features)
        assert 5 in vals.shape  # n_features should be one of the dims

    def test_shap_values_not_all_zero(self, explainer):
        """SHAP values should be non-trivial (not all zero).

        Returns
        -------
        None
        """
        rng = np.random.default_rng(1)
        X = pd.DataFrame(
            rng.random((10, 5)),
            columns=["EGFR", "BRAF", "TP53", "KRAS", "ALK"],
        )
        vals = explainer.shap_values(X)
        if isinstance(vals, list):
            assert any(np.abs(v).sum() > 0 for v in vals)
        else:
            assert np.abs(vals).sum() > 0


class TestGeneImportance:
    """Tests for gene importance ranking."""

    def test_gene_importance_returns_dataframe(self, explainer):
        """gene_importance should return a DataFrame with gene column.

        Returns
        -------
        None
        """
        rng = np.random.default_rng(2)
        X = pd.DataFrame(
            rng.random((20, 5)),
            columns=["EGFR", "BRAF", "TP53", "KRAS", "ALK"],
        )
        imp = explainer.gene_importance(X)
        assert isinstance(imp, pd.DataFrame)
        assert "gene" in imp.columns
        assert "mean_shap" in imp.columns
        assert len(imp) <= 5

    def test_gene_importance_top_k(self, explainer):
        """top_k parameter should limit results.

        Returns
        -------
        None
        """
        rng = np.random.default_rng(3)
        X = pd.DataFrame(
            rng.random((20, 5)),
            columns=["EGFR", "BRAF", "TP53", "KRAS", "ALK"],
        )
        imp = explainer.gene_importance(X, top_k=2)
        assert len(imp) <= 2


class TestPathwayImportance:
    """Tests for pathway-level SHAP aggregation."""

    def test_pathway_importance_returns_dataframe(self, explainer):
        """pathway_importance should return a DataFrame.

        Returns
        -------
        None
        """
        rng = np.random.default_rng(4)
        X = pd.DataFrame(
            rng.random((20, 5)),
            columns=["EGFR", "BRAF", "TP53", "KRAS", "ALK"],
        )
        pw = explainer.pathway_importance(X)
        assert isinstance(pw, pd.DataFrame)

    def test_pathway_scores_non_negative(self, explainer):
        """Pathway SHAP scores should be non-negative.

        Returns
        -------
        None
        """
        rng = np.random.default_rng(5)
        X = pd.DataFrame(
            rng.random((20, 5)),
            columns=["EGFR", "BRAF", "TP53", "KRAS", "ALK"],
        )
        pw = explainer.pathway_importance(X)
        if "mean_shap" in pw.columns:
            assert (pw["mean_shap"] >= 0).all()


class TestExplainSample:
    """Tests for single-sample explanation."""

    def test_explain_sample_returns_dataframe(self, explainer):
        """explain_sample should return a DataFrame with gene entries.

        Returns
        -------
        None
        """
        rng = np.random.default_rng(6)
        X = pd.DataFrame(
            rng.random((5, 5)),
            columns=["EGFR", "BRAF", "TP53", "KRAS", "ALK"],
        )
        result = explainer.explain_sample(X, sample_idx=0)
        assert isinstance(result, pd.DataFrame)
        assert "gene" in result.columns

    def test_explain_sample_invalid_index_raises(self, explainer):
        """Out-of-range sample_idx should raise an error.

        Returns
        -------
        None
        """
        rng = np.random.default_rng(7)
        X = pd.DataFrame(
            rng.random((5, 5)),
            columns=["EGFR", "BRAF", "TP53", "KRAS", "ALK"],
        )
        with pytest.raises((IndexError, ValueError)):
            explainer.explain_sample(X, sample_idx=999)


class TestFullReport:
    """Tests for the full interpretability report."""

    def test_full_report_has_expected_keys(self, explainer):
        """full_report should contain gene_importance, pathway, sample keys.

        Returns
        -------
        None
        """
        rng = np.random.default_rng(8)
        X = pd.DataFrame(
            rng.random((20, 5)),
            columns=["EGFR", "BRAF", "TP53", "KRAS", "ALK"],
        )
        report = explainer.full_report(X)
        assert isinstance(report, dict)
        assert "gene_importance" in report
        assert "pathway_importance" in report
