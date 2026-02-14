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
    classify_variants,
    load_cna,
    load_maf,
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
