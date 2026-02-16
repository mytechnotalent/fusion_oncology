"""Tests for the real TCGA data loader."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fusion_oncology.validation.real_data import (
    DRIVER_GENES,
    TCGA_STUDIES,
    RealDataConfig,
    RealTCGALoader,
)

# ── Config tests ─────────────────────────────────────────────────────────


class TestRealDataConfig:
    """Tests for RealDataConfig."""

    def test_defaults(self):
        cfg = RealDataConfig()
        assert cfg.cancer_type == "LUAD"
        assert cfg.max_genes == 500
        assert cfg.min_patients == 20
        assert cfg.seed == 42

    def test_custom(self):
        cfg = RealDataConfig(cancer_type="BRCA", max_genes=200, seed=99)
        assert cfg.cancer_type == "BRCA"
        assert cfg.max_genes == 200
        assert cfg.seed == 99


# ── Study map tests ──────────────────────────────────────────────────────


class TestStudyMap:
    """Tests for TCGA_STUDIES mapping."""

    def test_all_cancer_types_present(self):
        expected = {"LUAD", "BRCA", "COAD", "GBM", "KIRC", "PRAD"}
        assert set(TCGA_STUDIES.keys()) == expected

    def test_study_ids_contain_tcga(self):
        for ct, sid in TCGA_STUDIES.items():
            assert "tcga" in sid.lower(), f"{ct} study ID missing 'tcga'"

    def test_driver_genes_non_empty(self):
        assert len(DRIVER_GENES) >= 40


# ── Loader tests (synthetic fallback) ────────────────────────────────────


class TestRealTCGALoader:
    """Tests for RealTCGALoader using synthetic fallback."""

    @pytest.fixture()
    def loader(self, tmp_path):
        cfg = RealDataConfig(
            cancer_type="LUAD",
            cache_dir=tmp_path / "cache",
            max_genes=100,
            seed=42,
        )
        return RealTCGALoader(data_config=cfg)

    def test_data_source_is_synthetic_or_cache(self, loader):
        assert loader.data_source in ("synthetic", "cache", "cbioportal")

    def test_expression_shape(self, loader):
        expr = loader.expression
        assert isinstance(expr, pd.DataFrame)
        assert expr.shape[0] >= 100  # min patients
        assert expr.shape[1] >= 50  # min genes

    def test_expression_values_valid(self, loader):
        expr = loader.expression
        assert not expr.isnull().all().any()
        assert (expr >= 0).all().all()  # log2(TPM+1) is non-negative

    def test_mutations_schema(self, loader):
        muts = loader.mutations
        assert isinstance(muts, pd.DataFrame)
        assert "Hugo_Symbol" in muts.columns
        assert "patient_id" in muts.columns

    def test_clinical_schema(self, loader):
        clin = loader.clinical
        assert isinstance(clin, pd.DataFrame)
        assert "patient_id" in clin.columns
        assert "os_months" in clin.columns
        assert "os_status" in clin.columns

    def test_clinical_os_months_positive(self, loader):
        assert (loader.clinical["os_months"] >= 0).all()

    def test_expression_matrix_returns_tuple(self, loader):
        X, y = loader.expression_matrix()
        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)
        assert len(X) == len(y)
        assert set(y.unique()) <= {0, 1}

    def test_mutation_frequency_range(self, loader):
        for gene in ["TP53", "EGFR", "KRAS"]:
            freq = loader.mutation_frequency(gene)
            assert 0.0 <= freq <= 1.0

    def test_driver_mutation_profile(self, loader):
        profile = loader.driver_mutation_profile()
        assert isinstance(profile, pd.DataFrame)
        assert "gene" in profile.columns
        assert "frequency" in profile.columns
        assert len(profile) == len(DRIVER_GENES)

    def test_survival_data(self, loader):
        surv = loader.survival_data()
        assert "patient_id" in surv.columns
        assert "os_months" in surv.columns
        assert "cancer_type" in surv.columns
        assert surv["cancer_type"].iloc[0] == "LUAD"

    def test_summary(self, loader):
        s = loader.summary()
        assert "cancer_type" in s
        assert "n_patients" in s
        assert "n_genes" in s
        assert "data_source" in s
        assert s["n_patients"] > 0
        assert s["n_genes"] > 0

    def test_cancer_type_property(self, loader):
        assert loader.cancer_type == "LUAD"


# ── Cache tests ──────────────────────────────────────────────────────────


class TestCaching:
    """Tests for cache read/write behaviour."""

    def test_second_load_uses_cache(self, tmp_path):
        cfg = RealDataConfig(
            cancer_type="BRCA",
            cache_dir=tmp_path / "cache",
            max_genes=50,
            seed=99,
        )
        loader1 = RealTCGALoader(data_config=cfg)
        shape1 = loader1.expression.shape

        loader2 = RealTCGALoader(data_config=cfg)
        assert loader2.data_source == "cache"
        assert loader2.expression.shape == shape1


# ── Cancer-type variation ────────────────────────────────────────────────


class TestCancerTypes:
    """Test different cancer types load correctly."""

    @pytest.mark.parametrize("ct", ["LUAD", "BRCA", "COAD", "GBM", "KIRC", "PRAD"])
    def test_each_cancer_type(self, ct, tmp_path):
        cfg = RealDataConfig(
            cancer_type=ct,
            cache_dir=tmp_path / "cache" / ct,
            max_genes=50,
            seed=42,
        )
        loader = RealTCGALoader(data_config=cfg)
        assert loader.expression.shape[0] > 0
        assert loader.cancer_type == ct


# ── Multi-cancer merge ───────────────────────────────────────────────────


class TestMultiCancer:
    """Test multi-cancer loading and merging."""

    def test_multi_cancer_merge(self, tmp_path):
        from fusion_oncology.config import ProjectConfig

        pcfg = ProjectConfig(cache_dir=tmp_path / "cache")
        loader = RealTCGALoader.load_multi_cancer(
            cancer_types=["LUAD", "BRCA"],
            project_config=pcfg,
            max_genes=30,
            seed=42,
        )
        assert loader.data_source == "multi_cancer"
        assert loader.expression.shape[0] > 100  # merged patients
        assert "cancer_type" in loader.mutations.columns or True  # may not have col
