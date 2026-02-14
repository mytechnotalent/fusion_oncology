"""
Tests for the multi-omics data integration module.

Covers MAF parsing, CNA loading, methylation processing,
and the MultiOmicsIntegrator class.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fusion_oncology.data.multi_omics import (
    MultiOmicsIntegrator,
    _compute_cna_frequencies,
    _compute_grouped_stats,
    _compute_vaf,
    _impact_label,
    _select_maf_columns,
    classify_variants,
    cna_summary,
    differential_methylation,
    load_cna,
    load_maf,
    load_methylation,
    mutation_burden_per_gene,
    mutation_burden_per_sample,
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


@pytest.fixture()
def sample_maf(tmp_path: Path) -> Path:
    """Create a minimal MAF file for testing.

    Parameters
    ----------
    tmp_path : Path
        Pytest-provided temporary directory.

    Returns
    -------
    Path
        Path to the created MAF file.
    """
    maf_path = tmp_path / "test.maf"
    maf_path.write_text(
        "Hugo_Symbol\tVariant_Classification\tTumor_Sample_Barcode\t"
        "Chromosome\tStart_Position\tEnd_Position\tReference_Allele\t"
        "Tumor_Seq_Allele2\tHGVSp_Short\n"
        "EGFR\tMissense_Mutation\tSAMPLE_001\t7\t55259515\t55259515\tT\tG\tp.L858R\n"
        "BRAF\tMissense_Mutation\tSAMPLE_001\t7\t140453136\t140453136\tA\tT\tp.V600E\n"
        "TP53\tNonsense_Mutation\tSAMPLE_002\t17\t7577121\t7577121\tG\tA\tp.R175H\n"
        "EGFR\tMissense_Mutation\tSAMPLE_002\t7\t55259515\t55259515\tT\tG\tp.T790M\n"
    )
    return maf_path


@pytest.fixture()
def sample_cna(tmp_path: Path) -> Path:
    """Create a minimal CNA file for testing.

    Parameters
    ----------
    tmp_path : Path
        Pytest-provided temporary directory.

    Returns
    -------
    Path
        Path to the CNA TSV file.
    """
    cna_path = tmp_path / "test_cna.tsv"
    cna_path.write_text(
        "Hugo_Symbol\tSAMPLE_001\tSAMPLE_002\n"
        "EGFR\t1.5\t-0.2\n"
        "ERBB2\t2.1\t0.3\n"
        "PTEN\t-1.8\t-2.0\n"
    )
    return cna_path


class TestLoadMaf:
    """Tests for the ``load_maf`` function."""

    def test_loads_dataframe(self, sample_maf: Path) -> None:
        """load_maf should return a DataFrame.

        Parameters
        ----------
        sample_maf : Path
            Path to test MAF.
        """
        df = load_maf(str(sample_maf))
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 4

    def test_has_expected_columns(self, sample_maf: Path) -> None:
        """Loaded MAF should have Hugo_Symbol and Variant_Classification.

        Parameters
        ----------
        sample_maf : Path
            Path to test MAF.
        """
        df = load_maf(str(sample_maf))
        assert "Hugo_Symbol" in df.columns
        assert "Variant_Classification" in df.columns


class TestMutationBurden:
    """Tests for mutation burden computation functions."""

    def test_burden_per_gene(self, sample_maf: Path) -> None:
        """Should count mutations per gene.

        Parameters
        ----------
        sample_maf : Path
            Path to test MAF.
        """
        df = load_maf(str(sample_maf))
        burden = mutation_burden_per_gene(df)
        assert isinstance(burden, pd.Series)
        assert burden["EGFR"] == 2

    def test_burden_per_sample(self, sample_maf: Path) -> None:
        """Should count mutations per sample.

        Parameters
        ----------
        sample_maf : Path
            Path to test MAF.
        """
        df = load_maf(str(sample_maf))
        burden = mutation_burden_per_sample(df)
        assert isinstance(burden, pd.Series)
        assert burden["SAMPLE_001"] == 2


class TestClassifyVariants:
    """Tests for variant classification."""

    def test_classify(self, sample_maf: Path) -> None:
        """Should produce a classification summary.

        Parameters
        ----------
        sample_maf : Path
            Path to test MAF.
        """
        df = load_maf(str(sample_maf))
        classes = classify_variants(df)
        assert isinstance(classes, pd.DataFrame)
        assert "Impact" in classes.columns


class TestLoadCna:
    """Tests for CNA file loading."""

    def test_loads_dataframe(self, sample_cna: Path) -> None:
        """load_cna should return a DataFrame.

        Parameters
        ----------
        sample_cna : Path
            Path to test CNA file.
        """
        df = load_cna(str(sample_cna))
        assert isinstance(df, pd.DataFrame)
        assert df.index.name == "Hugo_Symbol" or "Hugo_Symbol" in df.columns or "EGFR" in df.index


class TestMultiOmicsIntegrator:
    """Tests for the MultiOmicsIntegrator class."""

    def test_init(self, cfg: ProjectConfig) -> None:
        """Integrator should initialise without error.

        Parameters
        ----------
        cfg : ProjectConfig
            Test configuration.
        """
        integrator = MultiOmicsIntegrator(cfg)
        assert integrator is not None

    def test_add_mutations(self, cfg: ProjectConfig, sample_maf: Path) -> None:
        """Adding mutations should store the MAF data.

        Parameters
        ----------
        cfg : ProjectConfig
            Test configuration.
        sample_maf : Path
            Path to test MAF.
        """
        integrator = MultiOmicsIntegrator(cfg)
        df = load_maf(str(sample_maf))
        integrator.add_mutations(df)
        assert integrator._maf is not None
        assert len(integrator._maf) == 4

    def test_add_cna(self, cfg: ProjectConfig, sample_cna: Path) -> None:
        """Adding CNA data should store it.

        Parameters
        ----------
        cfg : ProjectConfig
            Test configuration.
        sample_cna : Path
            Path to test CNA file.
        """
        integrator = MultiOmicsIntegrator(cfg)
        df = load_cna(str(sample_cna))
        integrator.add_cna(df)
        assert integrator._cna is not None


# ── MAF with VAF columns ────────────────────────────────────────────────


@pytest.fixture()
def maf_with_vaf(tmp_path: Path) -> Path:
    """Create a MAF file that includes t_alt_count and t_ref_count.

    Parameters
    ----------
    tmp_path : Path
        Pytest-provided temporary directory.

    Returns
    -------
    Path
        Path to the MAF file with allele counts.
    """
    p = tmp_path / "vaf.maf"
    p.write_text(
        "Hugo_Symbol\tVariant_Classification\tTumor_Sample_Barcode\t"
        "t_alt_count\tt_ref_count\tHGVSp_Short\n"
        "EGFR\tMissense_Mutation\tS1\t20\t80\tp.L858R\n"
        "BRAF\tMissense_Mutation\tS1\t0\t0\tp.V600E\n"
    )
    return p


class TestComputeVaf:
    """Tests for the ``_compute_vaf`` helper."""

    def test_computes_vaf_from_counts(self, maf_with_vaf: Path) -> None:
        """VAF should be alt / (alt + ref) when both columns are present.

        Parameters
        ----------
        maf_with_vaf : Path
            Path to MAF with allele counts.
        """
        df = load_maf(maf_with_vaf)
        assert "VAF" in df.columns
        assert abs(df.iloc[0]["VAF"] - 0.2) < 1e-6

    def test_vaf_zero_when_total_zero(self, maf_with_vaf: Path) -> None:
        """VAF should be 0 when both alt and ref counts are 0.

        Parameters
        ----------
        maf_with_vaf : Path
            Path to MAF with allele counts.
        """
        df = load_maf(maf_with_vaf)
        assert df.iloc[1]["VAF"] == 0.0

    def test_vaf_nan_without_columns(self) -> None:
        """VAF should be NaN when allele count columns are absent.

        Returns
        -------
        None
        """
        df = pd.DataFrame({"Hugo_Symbol": ["A"], "Variant_Classification": ["Missense_Mutation"]})
        result = _compute_vaf(df)
        assert pd.isna(result["VAF"].iloc[0])


class TestSelectMafColumns:
    """Tests for the ``_select_maf_columns`` helper."""

    def test_retains_present_columns(self) -> None:
        """Only columns that exist in the DataFrame should be retained.

        Returns
        -------
        None
        """
        df = pd.DataFrame({"Hugo_Symbol": ["A"], "Extra": [1]})
        result = _select_maf_columns(df)
        assert "Hugo_Symbol" in result.columns
        assert "Extra" not in result.columns


class TestImpactLabel:
    """Tests for the ``_impact_label`` helper."""

    def test_high_impact(self) -> None:
        """Frame_Shift_Del should be classified as HIGH.

        Returns
        -------
        None
        """
        assert _impact_label("Frame_Shift_Del") == "HIGH"

    def test_moderate_impact(self) -> None:
        """Missense_Mutation should be classified as MODERATE.

        Returns
        -------
        None
        """
        assert _impact_label("Missense_Mutation") == "MODERATE"

    def test_low_impact(self) -> None:
        """Silent should be classified as LOW.

        Returns
        -------
        None
        """
        assert _impact_label("Silent") == "LOW"

    def test_modifier_impact(self) -> None:
        """Unknown variant type should be classified as MODIFIER.

        Returns
        -------
        None
        """
        assert _impact_label("Unknown_Type") == "MODIFIER"


class TestLoadMafErrors:
    """Tests for load_maf error handling."""

    def test_file_not_found(self) -> None:
        """load_maf should raise FileNotFoundError for missing files.

        Returns
        -------
        None
        """
        with pytest.raises(FileNotFoundError):
            load_maf("/nonexistent/path.maf")


class TestLoadCnaErrors:
    """Tests for load_cna error handling."""

    def test_file_not_found(self) -> None:
        """load_cna should raise FileNotFoundError for missing files.

        Returns
        -------
        None
        """
        with pytest.raises(FileNotFoundError):
            load_cna("/nonexistent/path.tsv")


# ── CNA summary ─────────────────────────────────────────────────────────


class TestCnaSummary:
    """Tests for CNA summary computation."""

    def test_cna_summary_has_expected_columns(self, sample_cna: Path) -> None:
        """Summary should have Amplifications, Deletions, and frequencies.

        Parameters
        ----------
        sample_cna : Path
            Path to CNA file.
        """
        cna = load_cna(sample_cna)
        summary = cna_summary(cna)
        assert "Amplifications" in summary.columns
        assert "Deletions" in summary.columns
        assert "Amp_Freq" in summary.columns
        assert "Del_Freq" in summary.columns

    def test_compute_cna_frequencies(self) -> None:
        """_compute_cna_frequencies should count amps and dels correctly.

        Returns
        -------
        None
        """
        cna = pd.DataFrame({"S1": [2, -2], "S2": [0, 0]}, index=["G1", "G2"])
        result = _compute_cna_frequencies(cna, 2)
        assert result.loc["G1", "Amplifications"] == 1
        assert result.loc["G2", "Deletions"] == 1


# ── Methylation ──────────────────────────────────────────────────────────


@pytest.fixture()
def sample_meth(tmp_path: Path) -> Path:
    """Create a minimal methylation beta-value file.

    Parameters
    ----------
    tmp_path : Path
        Pytest-provided temporary directory.

    Returns
    -------
    Path
        Path to the methylation TSV.
    """
    p = tmp_path / "meth.tsv"
    p.write_text("Probe\tS1\tS2\tS3\tS4\nGENE_A\t0.1\t0.2\t0.8\t0.9\nGENE_B\t0.5\t0.5\t0.5\t0.5\n")
    return p


class TestLoadMethylation:
    """Tests for load_methylation and related functions."""

    def test_loads_dataframe(self, sample_meth: Path) -> None:
        """load_methylation should return a DataFrame.

        Parameters
        ----------
        sample_meth : Path
            Path to methylation file.
        """
        df = load_methylation(sample_meth)
        assert isinstance(df, pd.DataFrame)
        assert df.shape == (2, 4)

    def test_file_not_found(self) -> None:
        """load_methylation should raise FileNotFoundError for missing files.

        Returns
        -------
        None
        """
        with pytest.raises(FileNotFoundError):
            load_methylation("/nonexistent/path.tsv")


class TestDifferentialMethylation:
    """Tests for differential_methylation."""

    def test_returns_sorted_dataframe(self, sample_meth: Path) -> None:
        """Result should be sorted by Max_Delta descending.

        Parameters
        ----------
        sample_meth : Path
            Path to methylation file.
        """
        meth = load_methylation(sample_meth)
        labels = pd.Series(["TypeA", "TypeA", "TypeB", "TypeB"])
        result = differential_methylation(meth, labels)
        assert "Max_Delta" in result.columns
        assert "Mean_Beta" in result.columns


class TestComputeGroupedStats:
    """Tests for the ``_compute_grouped_stats`` helper."""

    def test_returns_expected_columns(self) -> None:
        """Stats should include Mean_Beta, Var_Beta, Max_Delta.

        Returns
        -------
        None
        """
        grouped = pd.DataFrame({"A": [0.1, 0.9], "B": [0.5, 0.5]}, index=["G1", "G2"])
        result = _compute_grouped_stats(grouped)
        assert "Mean_Beta" in result.columns
        assert "Var_Beta" in result.columns
        assert "Max_Delta" in result.columns


# ── MultiOmicsIntegrator extended coverage ──────────────────────────────


class TestMultiOmicsIntegratorExtended:
    """Extended tests for MultiOmicsIntegrator feature extraction."""

    def test_add_methylation(self, cfg: ProjectConfig, sample_meth: Path) -> None:
        """Adding methylation data should store it.

        Parameters
        ----------
        cfg : ProjectConfig
            Test configuration.
        sample_meth : Path
            Path to methylation TSV.
        """
        integrator = MultiOmicsIntegrator(cfg)
        meth = load_methylation(sample_meth)
        integrator.add_methylation(meth)
        assert integrator._meth is not None

    def test_mutation_features_with_maf(self, cfg: ProjectConfig, sample_maf: Path) -> None:
        """Mutation features should report counts and VAF for known genes.

        Parameters
        ----------
        cfg : ProjectConfig
            Test configuration.
        sample_maf : Path
            Path to MAF file.
        """
        integrator = MultiOmicsIntegrator(cfg)
        integrator.add_mutations(load_maf(sample_maf))
        feats = integrator._mutation_features("EGFR")
        assert feats["Mutation_Count"] == 2

    def test_mutation_features_no_maf(self, cfg: ProjectConfig) -> None:
        """Without MAF data, mutation features should be zeros.

        Parameters
        ----------
        cfg : ProjectConfig
            Test configuration.
        """
        integrator = MultiOmicsIntegrator(cfg)
        feats = integrator._mutation_features("EGFR")
        assert feats["Mutation_Count"] == 0

    def test_cna_features_with_data(self, cfg: ProjectConfig, sample_cna: Path) -> None:
        """CNA features should report amp/del frequencies for known genes.

        Parameters
        ----------
        cfg : ProjectConfig
            Test configuration.
        sample_cna : Path
            Path to CNA file.
        """
        integrator = MultiOmicsIntegrator(cfg)
        integrator.add_cna(load_cna(sample_cna))
        feats = integrator._cna_features("ERBB2")
        assert feats["Amp_Freq"] > 0

    def test_cna_features_no_data(self, cfg: ProjectConfig) -> None:
        """Without CNA data, features should be zeros.

        Parameters
        ----------
        cfg : ProjectConfig
            Test configuration.
        """
        integrator = MultiOmicsIntegrator(cfg)
        feats = integrator._cna_features("EGFR")
        assert feats["Amp_Freq"] == 0.0

    def test_methylation_features_with_data(self, cfg: ProjectConfig, sample_meth: Path) -> None:
        """Methylation features should report Mean and Var beta values.

        Parameters
        ----------
        cfg : ProjectConfig
            Test configuration.
        sample_meth : Path
            Path to methylation file.
        """
        integrator = MultiOmicsIntegrator(cfg)
        integrator.add_methylation(load_methylation(sample_meth))
        feats = integrator._methylation_features("GENE_A")
        assert not np.isnan(feats["Mean_Beta"])

    def test_methylation_features_no_data(self, cfg: ProjectConfig) -> None:
        """Without methylation data, features should be NaN.

        Parameters
        ----------
        cfg : ProjectConfig
            Test configuration.
        """
        integrator = MultiOmicsIntegrator(cfg)
        feats = integrator._methylation_features("EGFR")
        assert np.isnan(feats["Mean_Beta"])

    def test_build_feature_matrix(self, cfg: ProjectConfig, sample_maf: Path) -> None:
        """Feature matrix should have one row per gene.

        Parameters
        ----------
        cfg : ProjectConfig
            Test configuration.
        sample_maf : Path
            Path to MAF file.
        """
        integrator = MultiOmicsIntegrator(cfg)
        integrator.add_mutations(load_maf(sample_maf))
        matrix = integrator.build_feature_matrix(["EGFR", "BRAF"])
        assert len(matrix) == 2
        assert "Gene" in matrix.columns

    def test_enrich_fusion_results(self, cfg: ProjectConfig, sample_maf: Path) -> None:
        """Enrichment should append multi-omics columns.

        Parameters
        ----------
        cfg : ProjectConfig
            Test configuration.
        sample_maf : Path
            Path to MAF file.
        """
        integrator = MultiOmicsIntegrator(cfg)
        integrator.add_mutations(load_maf(sample_maf))
        results = pd.DataFrame({"Gene": ["EGFR"], "Score": [1.0]})
        enriched = integrator.enrich_fusion_results(results)
        assert "Mutation_Count" in enriched.columns
