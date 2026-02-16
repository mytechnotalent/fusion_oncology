"""Tests for the benchmark framework."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fusion_oncology.config import ProjectConfig
from fusion_oncology.validation.benchmark import (
    AblationStudy,
    BenchmarkConfig,
    BenchmarkSuite,
    PipelineModel,
    build_baselines,
    _evaluate_model_cv,
)

# ── Fixtures ─────────────────────────────────────────────────────────────


@pytest.fixture()
def benchmark_data():
    """Small synthetic dataset for benchmark tests."""
    rng = np.random.default_rng(42)
    n, d = 120, 30
    genes = [f"GENE{i}" for i in range(d)]
    X = pd.DataFrame(rng.standard_normal((n, d)), columns=genes)
    # Inject signal
    y_raw = np.repeat(["BRCA", "LUAD", "COAD"], n // 3)
    for i in range(n):
        if y_raw[i] == "BRCA":
            X.iloc[i, :5] += 1.5
        elif y_raw[i] == "LUAD":
            X.iloc[i, 5:10] += 1.5
        else:
            X.iloc[i, 10:15] += 1.5
    y = pd.Series(y_raw, name="Class")
    return X, y


# ── BenchmarkConfig tests ───────────────────────────────────────────────


class TestBenchmarkConfig:
    """Tests for BenchmarkConfig defaults."""

    def test_defaults(self):
        cfg = BenchmarkConfig()
        assert cfg.n_folds == 5
        assert cfg.n_repeats == 3
        assert cfg.significance_level == 0.05
        assert cfg.seed == 42

    def test_custom(self):
        cfg = BenchmarkConfig(n_folds=3, n_repeats=2, seed=99)
        assert cfg.n_folds == 3
        assert cfg.n_repeats == 2


# ── Baseline builder tests ──────────────────────────────────────────────


class TestBaselines:
    """Tests for baseline model construction."""

    def test_build_baselines_returns_four(self):
        baselines = build_baselines()
        assert len(baselines) == 4

    def test_baseline_names(self):
        baselines = build_baselines()
        names = set(baselines.keys())
        assert "Logistic Regression" in names
        assert "Random Forest" in names
        assert "SVM (RBF)" in names
        assert "XGBoost (vanilla)" in names

    def test_baselines_have_fit_predict(self):
        for name, model in build_baselines().items():
            assert hasattr(model, "fit"), f"{name} missing fit"
            assert hasattr(model, "predict"), f"{name} missing predict"


# ── PipelineModel tests ─────────────────────────────────────────────────


class TestPipelineModel:
    """Tests for the PipelineModel sklearn wrapper."""

    def test_fit_predict(self, benchmark_data):
        X, y = benchmark_data
        model = PipelineModel(
            config=ProjectConfig(
                xgb_n_estimators=10,
                xgb_max_depth=2,
                enable_feature_engineering=True,
            )
        )
        X_arr = X.values
        y_arr = np.array([0, 1, 2] * (len(y) // 3))
        model.fit(X_arr, y_arr)
        preds = model.predict(X_arr)
        assert len(preds) == len(y)
        assert set(preds) <= {0, 1, 2}

    def test_predict_proba(self, benchmark_data):
        X, y = benchmark_data
        model = PipelineModel(
            config=ProjectConfig(
                xgb_n_estimators=10,
                xgb_max_depth=2,
                enable_feature_engineering=True,
            )
        )
        X_arr = X.values
        y_arr = np.array([0, 1, 2] * (len(y) // 3))
        model.fit(X_arr, y_arr)
        proba = model.predict_proba(X_arr)
        assert proba.shape == (len(y), 3)
        assert np.allclose(proba.sum(axis=1), 1.0, atol=0.01)

    def test_get_set_params(self):
        model = PipelineModel()
        params = model.get_params()
        assert "config" in params

        new_cfg = ProjectConfig(xgb_n_estimators=42)
        model.set_params(config=new_cfg)
        assert model.config.xgb_n_estimators == 42


# ── CV evaluator tests ───────────────────────────────────────────────────


class TestEvaluateModelCV:
    """Tests for the _evaluate_model_cv function."""

    def test_returns_expected_keys(self, benchmark_data):
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import StratifiedKFold
        from sklearn.preprocessing import LabelEncoder

        X, y = benchmark_data
        le = LabelEncoder()
        y_enc = le.fit_transform(y)
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        model = RandomForestClassifier(n_estimators=10, random_state=42)

        result = _evaluate_model_cv(model, X.values, y_enc, cv, "RF")
        expected_keys = {
            "model",
            "fold_accuracies",
            "fold_f1s",
            "fold_precisions",
            "fold_recalls",
            "fold_aucs",
            "mean_accuracy",
            "std_accuracy",
            "mean_f1",
            "std_f1",
            "mean_precision",
            "std_precision",
            "mean_recall",
            "std_recall",
            "mean_auc",
            "std_auc",
        }
        assert expected_keys <= set(result.keys())

    def test_accuracy_in_valid_range(self, benchmark_data):
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import StratifiedKFold
        from sklearn.preprocessing import LabelEncoder

        X, y = benchmark_data
        le = LabelEncoder()
        y_enc = le.fit_transform(y)
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        model = RandomForestClassifier(n_estimators=10, random_state=42)

        result = _evaluate_model_cv(model, X.values, y_enc, cv, "RF")
        assert 0.0 <= result["mean_accuracy"] <= 1.0
        assert 0.0 <= result["mean_f1"] <= 1.0


# ── Ablation study tests ────────────────────────────────────────────────


class TestAblationStudy:
    """Tests for the AblationStudy class."""

    def test_ablation_returns_dataframe(self, benchmark_data):
        X, y = benchmark_data
        cfg = BenchmarkConfig(n_folds=3, n_repeats=1, seed=42)
        study = AblationStudy(X, y, config=cfg)
        result = study.run()
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 5  # 5 variants
        assert "variant" in result.columns
        assert "mean_accuracy" in result.columns
        assert "delta_f1" in result.columns
        assert "p_value" in result.columns

    def test_full_pipeline_has_zero_delta(self, benchmark_data):
        X, y = benchmark_data
        cfg = BenchmarkConfig(n_folds=3, n_repeats=1, seed=42)
        study = AblationStudy(X, y, config=cfg)
        result = study.run()
        full = result[result["variant"] == "Full Pipeline"].iloc[0]
        assert full["delta_accuracy"] == 0.0
        assert full["delta_f1"] == 0.0


# ── BenchmarkSuite tests ────────────────────────────────────────────────


class TestBenchmarkSuite:
    """Tests for the full BenchmarkSuite."""

    def test_baseline_comparison(self, benchmark_data):
        X, y = benchmark_data
        cfg = BenchmarkConfig(n_folds=3, n_repeats=1, seed=42)
        suite = BenchmarkSuite(X, y, config=cfg)
        comp = suite.baseline_comparison()
        assert isinstance(comp, pd.DataFrame)
        assert len(comp) == 5  # pipeline + 4 baselines
        assert "model" in comp.columns
        assert "mean_accuracy" in comp.columns
        assert "p_value_f1" in comp.columns

    def test_stability_analysis(self, benchmark_data):
        X, y = benchmark_data
        cfg = BenchmarkConfig(n_folds=3, seed=42)
        suite = BenchmarkSuite(X, y, config=cfg)
        stab = suite.stability_analysis(n_subsamples=3, subsample_fraction=0.7)
        assert isinstance(stab, pd.DataFrame)
        assert len(stab) == 3
        assert "mean_accuracy" in stab.columns

    def test_full_benchmark(self, benchmark_data):
        X, y = benchmark_data
        cfg = BenchmarkConfig(n_folds=3, n_repeats=1, seed=42)
        suite = BenchmarkSuite(X, y, config=cfg)
        report = suite.run_full_benchmark()
        assert "comparison_table" in report
        assert "ablation_table" in report
        assert "stability_table" in report
        assert "summary" in report

        s = report["summary"]
        assert 0.0 <= s["pipeline_accuracy"] <= 1.0
        assert isinstance(s["best_baseline"], str)
        assert isinstance(s["n_baselines_beaten_significantly"], int)

    def test_format_report(self, benchmark_data):
        X, y = benchmark_data
        cfg = BenchmarkConfig(n_folds=3, n_repeats=1, seed=42)
        suite = BenchmarkSuite(X, y, config=cfg)
        report = suite.run_full_benchmark()
        md = BenchmarkSuite.format_report(report)
        assert "## Benchmark Results" in md
        assert "Fusion Pipeline" in md
        assert "Logistic Regression" in md
        assert "|" in md  # tables
