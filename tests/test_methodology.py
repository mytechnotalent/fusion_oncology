"""Tests for the methodology formalisation module."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fusion_oncology.config import ProjectConfig
from fusion_oncology.validation.methodology import (
    ARCHITECTURE_SPEC,
    ComponentContribution,
    MethodologyFormaliser,
)

# ── Fixtures ─────────────────────────────────────────────────────────────


@pytest.fixture()
def method_data():
    """Small dataset with injectable signal for methodology tests."""
    rng = np.random.default_rng(42)
    n, d = 120, 30
    genes = [f"GENE{i}" for i in range(d)]
    X = pd.DataFrame(rng.standard_normal((n, d)), columns=genes)
    y_raw = np.repeat(["BRCA", "LUAD", "COAD"], n // 3)
    for i in range(n):
        if y_raw[i] == "BRCA":
            X.iloc[i, :5] += 2.0
        elif y_raw[i] == "LUAD":
            X.iloc[i, 5:10] += 2.0
        else:
            X.iloc[i, 10:15] += 2.0
    y = pd.Series(y_raw, name="Class")
    return X, y


# ── Architecture spec tests ──────────────────────────────────────────────


class TestArchitectureSpec:
    """Tests for the architecture specification constant."""

    def test_spec_keys(self):
        assert "name" in ARCHITECTURE_SPEC
        assert "version" in ARCHITECTURE_SPEC
        assert "layers" in ARCHITECTURE_SPEC
        assert "novelty" in ARCHITECTURE_SPEC

    def test_spec_name(self):
        assert "Fusion Oncology" in ARCHITECTURE_SPEC["name"]

    def test_spec_layers_contain_all_seven(self):
        layers = ARCHITECTURE_SPEC["layers"]
        for i in range(1, 8):
            assert f"Layer {i}" in layers, f"Layer {i} missing"

    def test_spec_novelty_mentions_importance(self):
        assert "importance" in ARCHITECTURE_SPEC["novelty"].lower()


# ── ComponentContribution dataclass ──────────────────────────────────────


class TestComponentContribution:
    """Tests for the ComponentContribution dataclass."""

    def test_creation(self):
        cc = ComponentContribution(
            component="Test",
            marginal_accuracy=0.05,
            marginal_f1=0.03,
            information_gain=0.1,
            feature_utilisation=0.8,
        )
        assert cc.component == "Test"
        assert cc.marginal_accuracy == 0.05

    def test_fields(self):
        import dataclasses

        fields = {f.name for f in dataclasses.fields(ComponentContribution)}
        expected = {
            "component",
            "marginal_accuracy",
            "marginal_f1",
            "information_gain",
            "feature_utilisation",
        }
        assert fields == expected


# ── MethodologyFormaliser tests ──────────────────────────────────────────


class TestMethodologyFormaliser:
    """Tests for the MethodologyFormaliser class."""

    def test_feature_engineering_contribution(self, method_data):
        X, y = method_data
        fm = MethodologyFormaliser(X, y, seed=42)
        cc = fm.feature_engineering_contribution()
        assert isinstance(cc, ComponentContribution)
        assert isinstance(cc.marginal_accuracy, float)
        assert isinstance(cc.marginal_f1, float)
        assert 0.0 <= cc.feature_utilisation <= 1.0

    def test_importance_weighting_analysis(self, method_data):
        X, y = method_data
        fm = MethodologyFormaliser(X, y, seed=42)
        result = fm.importance_weighting_analysis()
        assert "weighted_accuracy" in result
        assert "uniform_accuracy" in result
        assert "random_accuracy" in result
        assert "improvement_over_uniform" in result
        assert 0.0 <= result["weighted_accuracy"] <= 1.0
        assert 0.0 <= result["uniform_accuracy"] <= 1.0

    def test_hyperparameter_sensitivity(self, method_data):
        X, y = method_data
        fm = MethodologyFormaliser(X, y, seed=42)
        df = fm.hyperparameter_sensitivity()
        assert isinstance(df, pd.DataFrame)
        assert "parameter" in df.columns
        assert "value" in df.columns
        assert "accuracy" in df.columns
        assert "f1" in df.columns
        assert len(df) == 12  # 4 depths + 4 estimators + 4 LRs

    def test_synergy_analysis(self, method_data):
        X, y = method_data
        fm = MethodologyFormaliser(X, y, seed=42)
        result = fm.synergy_analysis()
        assert "baseline_accuracy" in result
        assert "improvement_A_feature_eng" in result
        assert "improvement_B_deeper_trees" in result
        assert "synergy_score" in result
        assert "is_synergistic" in result
        assert isinstance(result["is_synergistic"], bool)

    def test_architecture_specification_static(self):
        spec = MethodologyFormaliser.architecture_specification()
        assert spec == ARCHITECTURE_SPEC

    def test_full_analysis(self, method_data):
        X, y = method_data
        fm = MethodologyFormaliser(X, y, seed=42)
        report = fm.full_analysis()
        assert "architecture_specification" in report
        assert "feature_engineering" in report
        assert "importance_weighting" in report
        assert "hyperparameter_sensitivity" in report
        assert "synergy" in report

    def test_format_methodology_report(self, method_data):
        X, y = method_data
        fm = MethodologyFormaliser(X, y, seed=42)
        report = fm.full_analysis()
        md = MethodologyFormaliser.format_methodology_report(report)
        assert "## Methodology Formalisation" in md
        assert "Architecture Specification" in md
        assert "Feature Engineering" in md
        assert "Importance-Weighted" in md
        assert "Synergy" in md
