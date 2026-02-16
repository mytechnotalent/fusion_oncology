"""Tests for the TCGA patient cohort validation module."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fusion_oncology.data.tcga import TCGACohortValidator, TCGALoader

# ── TCGALoader (offline / synthetic) ─────────────────────────────────────


class TestTCGALoader:
    @pytest.fixture()
    def loader(self) -> TCGALoader:
        return TCGALoader(n_patients=80, seed=42)

    def test_expression_shape(self, loader: TCGALoader) -> None:
        expr = loader.expression
        assert isinstance(expr, pd.DataFrame)
        assert expr.shape[0] == 80

    def test_mutations_shape(self, loader: TCGALoader) -> None:
        mut = loader.mutations
        assert isinstance(mut, pd.DataFrame)
        assert mut.shape[0] == 80

    def test_clinical_columns(self, loader: TCGALoader) -> None:
        clin = loader.clinical
        assert isinstance(clin, pd.DataFrame)
        expected = {"OS_MONTHS", "OS_STATUS", "cancer_type", "stage", "age", "TMB"}
        assert expected.issubset(set(clin.columns))

    def test_clinical_os_status_binary(self, loader: TCGALoader) -> None:
        clin = loader.clinical
        assert set(clin["OS_STATUS"].unique()).issubset({0, 1})

    def test_cna_shape(self, loader: TCGALoader) -> None:
        cna = loader.cna
        assert isinstance(cna, pd.DataFrame)
        assert cna.shape[0] == 80

    def test_cancer_types(self, loader: TCGALoader) -> None:
        types = loader.cancer_types
        assert len(types) > 0
        assert isinstance(types, pd.Series)

    def test_patient_subset(self, loader: TCGALoader) -> None:
        sub = loader.patient_subset(cancer_type="LUAD")
        # Returns dict with expression, mutations, clinical, cna keys
        assert isinstance(sub, dict)
        assert "expression" in sub
        assert "mutations" in sub

    def test_training_data(self, loader: TCGALoader) -> None:
        X, y = loader.training_data()
        assert X.shape[0] == y.shape[0]
        assert X.shape[0] == 80
        assert X.shape[1] > 0

    def test_mutation_frequency(self, loader: TCGALoader) -> None:
        result = loader.mutation_frequency("TP53")
        assert isinstance(result, dict)
        # Each value should be a frequency in [0, 1]
        for freq in result.values():
            assert 0 <= freq <= 1

    def test_mutation_frequency_missing_gene(self, loader: TCGALoader) -> None:
        result = loader.mutation_frequency("NOT_A_GENE_XYZ")
        assert result == {}

    def test_summary(self, loader: TCGALoader) -> None:
        s = loader.summary()
        assert isinstance(s, dict)
        assert "n_patients" in s
        assert s["n_patients"] == 80
        assert "n_genes" in s
        assert "n_cancer_types" in s


# ── TCGACohortValidator ──────────────────────────────────────────────────


class TestTCGACohortValidator:
    @pytest.fixture()
    def validator(self) -> TCGACohortValidator:
        loader = TCGALoader(n_patients=120, seed=7)
        return TCGACohortValidator(loader)

    def test_validate_classifier(self, validator: TCGACohortValidator) -> None:
        from xgboost import XGBClassifier

        model = XGBClassifier(
            n_estimators=10,
            max_depth=3,
            verbosity=0,
            use_label_encoder=False,
            eval_metric="mlogloss",
        )
        result = validator.validate_classifier(model)
        assert isinstance(result, dict)
        assert "accuracy" in result
        assert "weighted_f1" in result
        assert 0.0 <= result["accuracy"] <= 1.0
        assert 0.0 <= result["weighted_f1"] <= 1.0

    def test_validate_classifier_has_details(self, validator: TCGACohortValidator) -> None:
        from xgboost import XGBClassifier

        model = XGBClassifier(
            n_estimators=10,
            max_depth=3,
            verbosity=0,
            use_label_encoder=False,
            eval_metric="mlogloss",
        )
        result = validator.validate_classifier(model)
        assert "n_train" in result
        assert "n_test" in result

    def test_mutation_enrichment(self, validator: TCGACohortValidator) -> None:
        df = validator.mutation_enrichment(["TP53", "KRAS"])
        assert isinstance(df, pd.DataFrame)
        assert "gene" in df.columns
        assert "cancer_type" in df.columns
        assert "mutation_freq" in df.columns

    def test_survival_stratification(self, validator: TCGACohortValidator) -> None:
        result = validator.survival_stratification("TP53")
        assert isinstance(result, dict)
        assert "median_survival_mutated" in result
        assert "median_survival_wildtype" in result

    def test_survival_stratification_has_counts(self, validator: TCGACohortValidator) -> None:
        result = validator.survival_stratification("TP53")
        assert "n_mutated" in result
        assert "n_wildtype" in result
        assert result["n_mutated"] + result["n_wildtype"] > 0
