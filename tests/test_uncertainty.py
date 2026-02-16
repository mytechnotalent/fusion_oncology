"""Tests for the Bayesian uncertainty quantification module."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fusion_oncology.analysis.uncertainty import (
    BayesianPredictor,
    BootstrapEnsemble,
    UncertaintyConfig,
    _decision_quality,
    calibration_curve,
    expected_calibration_error,
)
from fusion_oncology.config import ProjectConfig

# ── Helpers ──────────────────────────────────────────────────────────────


def _make_classification_data(
    n: int = 200, n_features: int = 10, n_classes: int = 3, seed: int = 42
) -> tuple[pd.DataFrame, pd.Series]:
    """Generate a synthetic multi-class dataset."""
    rng = np.random.RandomState(seed)
    X = pd.DataFrame(
        rng.randn(n, n_features).astype(np.float32),
        columns=[f"feat_{i}" for i in range(n_features)],
    )
    labels = ["BRCA", "LUAD", "COAD"][:n_classes]
    y = pd.Series(rng.choice(labels, size=n))
    return X, y


# ── UncertaintyConfig ───────────────────────────────────────────────────


class TestUncertaintyConfig:
    def test_defaults(self) -> None:
        cfg = UncertaintyConfig()
        assert cfg.n_bootstrap == 20
        assert cfg.confidence_level == 0.95
        assert cfg.entropy_threshold == 0.5

    def test_custom(self) -> None:
        cfg = UncertaintyConfig(n_bootstrap=50, confidence_level=0.9)
        assert cfg.n_bootstrap == 50
        assert cfg.confidence_level == 0.9


# ── Decision quality helper ─────────────────────────────────────────────


class TestDecisionQuality:
    def test_high(self) -> None:
        assert _decision_quality(0.95, 0.85, 0.5) == "HIGH"

    def test_moderate(self) -> None:
        assert _decision_quality(0.75, 0.55, 0.5) == "MODERATE"

    def test_low(self) -> None:
        assert _decision_quality(0.55, 0.35, 0.5) == "LOW"

    def test_uncertain(self) -> None:
        assert _decision_quality(0.30, 0.25, 0.5) == "UNCERTAIN"


# ── BootstrapEnsemble ───────────────────────────────────────────────────


class TestBootstrapEnsemble:
    @pytest.fixture()
    def trained_ensemble(self) -> BootstrapEnsemble:
        X, y = _make_classification_data()
        cfg = UncertaintyConfig(n_bootstrap=5)
        ens = BootstrapEnsemble(uq_config=cfg)
        ens.fit(X, y)
        return ens

    def test_fit_creates_members(self, trained_ensemble: BootstrapEnsemble) -> None:
        assert trained_ensemble.n_models == 5

    def test_predict_proba_shape(self, trained_ensemble: BootstrapEnsemble) -> None:
        X, _ = _make_classification_data()
        probs = trained_ensemble.predict_proba_ensemble(X.iloc[:10])
        # shape: (n_bootstrap, n_samples, n_classes)
        assert probs.shape[0] == 5
        assert probs.shape[1] == 10
        assert probs.shape[2] == 3

    def test_predict_proba_sums_to_one(self, trained_ensemble: BootstrapEnsemble) -> None:
        X, _ = _make_classification_data()
        probs = trained_ensemble.predict_proba_ensemble(X.iloc[:10])
        # Each member / sample row should sum to ~1
        for b in range(probs.shape[0]):
            np.testing.assert_allclose(probs[b].sum(axis=1), 1.0, atol=1e-5)

    def test_predict_with_uncertainty(self, trained_ensemble: BootstrapEnsemble) -> None:
        X, _ = _make_classification_data()
        result = trained_ensemble.predict_with_uncertainty(X.iloc[:10])
        assert "predictions" in result
        assert "confidence" in result
        assert "entropy" in result
        assert result["mean_proba"].shape == (10, 3)


# ── BayesianPredictor ───────────────────────────────────────────────────


class TestBayesianPredictor:
    @pytest.fixture()
    def predictor(self) -> BayesianPredictor:
        X, y = _make_classification_data(n=300, seed=123)
        cfg = UncertaintyConfig(n_bootstrap=5, confidence_level=0.90)
        bp = BayesianPredictor(uq_config=cfg)
        bp.fit(X, y)
        return bp

    def test_predict_returns_dataframe(self, predictor: BayesianPredictor) -> None:
        X, _ = _make_classification_data()
        df = predictor.predict(X.iloc[:20])
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 20

    def test_predict_columns(self, predictor: BayesianPredictor) -> None:
        X, _ = _make_classification_data()
        df = predictor.predict(X.iloc[:5])
        expected = {
            "prediction",
            "confidence",
            "entropy",
            "top_class_prob",
            "top_class_std",
            "ci_lower",
            "ci_upper",
            "decision_quality",
        }
        assert expected.issubset(set(df.columns))

    def test_confidence_range(self, predictor: BayesianPredictor) -> None:
        X, _ = _make_classification_data()
        df = predictor.predict(X.iloc[:30])
        assert df["confidence"].min() >= -0.01
        assert df["confidence"].max() <= 1.01

    def test_entropy_nonnegative(self, predictor: BayesianPredictor) -> None:
        X, _ = _make_classification_data()
        df = predictor.predict(X.iloc[:30])
        assert (df["entropy"] >= -0.01).all()

    def test_decision_quality_values(self, predictor: BayesianPredictor) -> None:
        X, _ = _make_classification_data()
        df = predictor.predict(X.iloc[:50])
        valid = {"HIGH", "MODERATE", "LOW", "UNCERTAIN"}
        assert set(df["decision_quality"].unique()).issubset(valid)

    def test_credible_intervals(self, predictor: BayesianPredictor) -> None:
        X, _ = _make_classification_data()
        df = predictor.credible_intervals(X.iloc[:10])
        assert isinstance(df, pd.DataFrame)
        # Should have columns for each class
        assert any("lower" in c for c in df.columns)
        assert any("upper" in c for c in df.columns)
        assert any("mean" in c for c in df.columns)

    def test_ci_lower_leq_upper(self, predictor: BayesianPredictor) -> None:
        X, _ = _make_classification_data()
        df = predictor.credible_intervals(X.iloc[:10])
        classes = ["BRCA", "COAD", "LUAD"]
        for cls in classes:
            lower = df[f"{cls}_lower"]
            upper = df[f"{cls}_upper"]
            assert (lower <= upper + 1e-8).all()

    def test_summary(self, predictor: BayesianPredictor) -> None:
        s = predictor.summary()
        assert "n_bootstrap" in s
        assert "n_models_trained" in s
        assert s["n_models_trained"] == 5


# ── Calibration functions ────────────────────────────────────────────────


class TestCalibration:
    def test_calibration_curve_returns_dataframe(self) -> None:
        rng = np.random.RandomState(99)
        y_true = rng.randint(0, 2, 200)
        y_prob = rng.random(200)
        df = calibration_curve(y_true, y_prob, n_bins=5)
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        assert "observed_freq" in df.columns
        assert "mean_predicted" in df.columns

    def test_calibration_curve_range(self) -> None:
        rng = np.random.RandomState(99)
        y_true = rng.randint(0, 2, 200)
        y_prob = rng.random(200)
        df = calibration_curve(y_true, y_prob, n_bins=5)
        assert (df["observed_freq"] >= 0).all()
        assert (df["observed_freq"] <= 1).all()
        assert (df["mean_predicted"] >= 0).all()
        assert (df["mean_predicted"] <= 1).all()

    def test_ece_range(self) -> None:
        rng = np.random.RandomState(99)
        y_true = rng.randint(0, 2, 200)
        y_prob = rng.random(200)
        ece = expected_calibration_error(y_true, y_prob)
        assert 0.0 <= ece <= 1.0

    def test_ece_type(self) -> None:
        rng = np.random.RandomState(99)
        y_true = rng.randint(0, 2, 200)
        y_prob = rng.random(200)
        ece = expected_calibration_error(y_true, y_prob)
        assert isinstance(ece, float)
