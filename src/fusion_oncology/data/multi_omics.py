"""
Multi-omics data integration.

Parses and harmonises additional genomic data layers beyond RNA-Seq
expression: somatic mutations (MAF format), copy-number alterations,
and DNA methylation beta values.  Each loader returns a standardised
DataFrame keyed on gene symbol so downstream modules can join freely.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from fusion_oncology.config import ProjectConfig

logger = logging.getLogger(__name__)


# ── MAF (Mutation Annotation Format) ────────────────────────────────────

_MAF_KEEP_COLS = [
    "Hugo_Symbol",
    "Variant_Classification",
    "Variant_Type",
    "Tumor_Sample_Barcode",
    "HGVSp_Short",
    "t_alt_count",
    "t_ref_count",
]


def _select_maf_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Retain only clinically relevant MAF columns.

    Parameters
    ----------
    df : pd.DataFrame
        Raw MAF DataFrame.

    Returns
    -------
    pd.DataFrame
        DataFrame restricted to present columns from ``_MAF_KEEP_COLS``.
    """
    present = [c for c in _MAF_KEEP_COLS if c in df.columns]
    return df[present].copy()


def _compute_vaf(df: pd.DataFrame) -> pd.DataFrame:
    """Add a ``VAF`` (variant allele frequency) column.

    Parameters
    ----------
    df : pd.DataFrame
        MAF DataFrame with optional ``t_alt_count`` and ``t_ref_count``.

    Returns
    -------
    pd.DataFrame
        Same DataFrame with a ``VAF`` column appended.
    """
    if "t_alt_count" not in df.columns or "t_ref_count" not in df.columns:
        df["VAF"] = np.nan
        return df
    df["t_alt_count"] = pd.to_numeric(df["t_alt_count"], errors="coerce").fillna(0)
    df["t_ref_count"] = pd.to_numeric(df["t_ref_count"], errors="coerce").fillna(0)
    total = df["t_alt_count"] + df["t_ref_count"]
    df["VAF"] = np.where(total > 0, df["t_alt_count"] / total, 0.0)
    return df


def load_maf(
    path: Path,
    config: ProjectConfig | None = None,
) -> pd.DataFrame:
    """Load and normalise a MAF (Mutation Annotation Format) file.

    Parameters
    ----------
    path : Path
        Filesystem path to the ``.maf`` or ``.maf.gz`` file.
    config : ProjectConfig, optional
        Runtime configuration (reserved for future filtering thresholds).

    Returns
    -------
    pd.DataFrame
        Columns include ``Hugo_Symbol``, ``Variant_Classification``,
        ``VAF``, and other clinically relevant fields.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    """
    if not Path(path).exists():
        raise FileNotFoundError(f"MAF file not found: {path}")
    raw = pd.read_csv(path, sep="\t", comment="#", low_memory=False)
    df = _compute_vaf(_select_maf_columns(raw))
    logger.info("Loaded MAF: %d variants across %d genes", len(df), df["Hugo_Symbol"].nunique())
    return df


def mutation_burden_per_gene(maf: pd.DataFrame) -> pd.Series:
    """Compute the total mutation count per gene from a MAF DataFrame.

    Parameters
    ----------
    maf : pd.DataFrame
        Output of :func:`load_maf` with at least a ``Hugo_Symbol`` column.

    Returns
    -------
    pd.Series
        Mutation count indexed by ``Hugo_Symbol``, sorted descending.
    """
    counts = maf["Hugo_Symbol"].value_counts()
    counts.name = "Mutation_Count"
    return counts


def mutation_burden_per_sample(maf: pd.DataFrame) -> pd.Series:
    """Compute tumor mutation burden (TMB) per sample.

    Parameters
    ----------
    maf : pd.DataFrame
        Output of :func:`load_maf` with a ``Tumor_Sample_Barcode`` column.

    Returns
    -------
    pd.Series
        Mutation count indexed by ``Tumor_Sample_Barcode``, sorted descending.
    """
    counts = maf["Tumor_Sample_Barcode"].value_counts()
    counts.name = "TMB"
    return counts


# ── Variant classification ──────────────────────────────────────────────

_HIGH_IMPACT = {
    "Frame_Shift_Del",
    "Frame_Shift_Ins",
    "Nonsense_Mutation",
    "Splice_Site",
    "Translation_Start_Site",
}

_MODERATE_IMPACT = {
    "In_Frame_Del",
    "In_Frame_Ins",
    "Missense_Mutation",
    "Nonstop_Mutation",
}

_LOW_IMPACT = {"Silent", "Splice_Region"}


def _impact_label(vc: str) -> str:
    """Map a variant classification string to an impact tier.

    Parameters
    ----------
    vc : str
        TCGA variant classification label.

    Returns
    -------
    str
        One of ``HIGH``, ``MODERATE``, ``LOW``, or ``MODIFIER``.
    """
    if vc in _HIGH_IMPACT:
        return "HIGH"
    if vc in _MODERATE_IMPACT:
        return "MODERATE"
    if vc in _LOW_IMPACT:
        return "LOW"
    return "MODIFIER"


def classify_variants(maf: pd.DataFrame) -> pd.DataFrame:
    """Add a simplified ``Impact`` column based on variant classification.

    Maps TCGA-standard ``Variant_Classification`` labels into four
    buckets: ``HIGH``, ``MODERATE``, ``LOW``, ``MODIFIER``.

    Parameters
    ----------
    maf : pd.DataFrame
        Output of :func:`load_maf`.

    Returns
    -------
    pd.DataFrame
        Input DataFrame with an additional ``Impact`` column.
    """
    maf = maf.copy()
    maf["Impact"] = maf["Variant_Classification"].apply(_impact_label)
    return maf


# ── Copy-Number Alterations ─────────────────────────────────────────────


def load_cna(
    path: Path,
    config: ProjectConfig | None = None,
) -> pd.DataFrame:
    """Load a GISTIC-style copy-number alteration matrix.

    Expects a tab-separated file where rows are genes and columns are
    samples.

    Parameters
    ----------
    path : Path
        Filesystem path to the CNA matrix file.
    config : ProjectConfig, optional
        Runtime configuration (reserved for future use).

    Returns
    -------
    pd.DataFrame
        Gene x sample matrix with integer CNA values.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    """
    if not Path(path).exists():
        raise FileNotFoundError(f"CNA file not found: {path}")
    df = pd.read_csv(path, sep="\t", index_col=0)
    logger.info("Loaded CNA: %d genes x %d samples", df.shape[0], df.shape[1])
    return df


def _compute_cna_frequencies(cna: pd.DataFrame, n_samples: int) -> pd.DataFrame:
    """Compute amplification and deletion counts and frequencies.

    Parameters
    ----------
    cna : pd.DataFrame
        Gene x sample CNA matrix.
    n_samples : int
        Total number of samples.

    Returns
    -------
    pd.DataFrame
        Per-gene counts and frequencies.
    """
    summary = pd.DataFrame(
        {"Amplifications": (cna >= 2).sum(axis=1), "Deletions": (cna <= -2).sum(axis=1)}
    )
    summary["Amp_Freq"] = summary["Amplifications"] / n_samples
    summary["Del_Freq"] = summary["Deletions"] / n_samples
    return summary


def cna_summary(cna: pd.DataFrame) -> pd.DataFrame:
    """Summarise copy-number events per gene.

    Parameters
    ----------
    cna : pd.DataFrame
        Gene x sample CNA matrix from :func:`load_cna`.

    Returns
    -------
    pd.DataFrame
        Per-gene summary with columns ``Amplifications``, ``Deletions``,
        ``Amp_Freq``, ``Del_Freq``.
    """
    summary = _compute_cna_frequencies(cna, cna.shape[1])
    return summary.sort_values("Amplifications", ascending=False)


# ── Methylation ─────────────────────────────────────────────────────────


def load_methylation(
    path: Path,
    config: ProjectConfig | None = None,
) -> pd.DataFrame:
    """Load a beta-value methylation matrix.

    Parameters
    ----------
    path : Path
        Filesystem path to the methylation matrix.
    config : ProjectConfig, optional
        Runtime configuration (reserved for future use).

    Returns
    -------
    pd.DataFrame
        Probe x sample matrix of beta values.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    """
    if not Path(path).exists():
        raise FileNotFoundError(f"Methylation file not found: {path}")
    df = pd.read_csv(path, sep="\t", index_col=0)
    logger.info("Loaded methylation: %d probes x %d samples", df.shape[0], df.shape[1])
    return df


def _compute_grouped_stats(grouped: pd.DataFrame) -> pd.DataFrame:
    """Compute per-gene methylation statistics from grouped means.

    Parameters
    ----------
    grouped : pd.DataFrame
        Gene x cancer-type mean beta-value matrix.

    Returns
    -------
    pd.DataFrame
        Summary with ``Mean_Beta``, ``Var_Beta``, ``Max_Delta``.
    """
    return pd.DataFrame(
        {
            "Mean_Beta": grouped.mean(axis=1),
            "Var_Beta": grouped.var(axis=1),
            "Max_Delta": grouped.max(axis=1) - grouped.min(axis=1),
        }
    )


def differential_methylation(
    methylation: pd.DataFrame,
    labels: pd.Series,
    gene_col: str = "Gene_Symbol",
) -> pd.DataFrame:
    """Compute per-gene differential methylation between cancer types.

    Parameters
    ----------
    methylation : pd.DataFrame
        Probe x sample beta-value matrix.
    labels : pd.Series
        Cancer-type labels aligned with sample columns.
    gene_col : str
        Column name mapping probes to gene symbols.

    Returns
    -------
    pd.DataFrame
        Per-gene summary with ``Mean_Beta``, ``Var_Beta``, ``Max_Delta``.
    """
    mt = methylation.T
    mt["Label"] = labels.values[: len(mt)]
    grouped = mt.groupby("Label").mean().T
    result = _compute_grouped_stats(grouped)
    return result.sort_values("Max_Delta", ascending=False)


# ── Multi-omics Integrator ──────────────────────────────────────────────


class MultiOmicsIntegrator:
    """Harmonise multiple genomic data layers for joint analysis.

    Accepts optional MAF, CNA, and methylation data and produces a
    unified per-gene feature matrix that can be appended to the Fusion
    Index for richer target prioritisation.

    Parameters
    ----------
    config : ProjectConfig, optional
        Runtime configuration.
    """

    def __init__(self, config: ProjectConfig | None = None) -> None:
        """Initialise the integrator.

        Parameters
        ----------
        config : ProjectConfig, optional
            Runtime configuration.  Falls back to defaults when ``None``.
        """
        self.cfg = config or ProjectConfig()
        self._maf: pd.DataFrame | None = None
        self._cna: pd.DataFrame | None = None
        self._meth: pd.DataFrame | None = None

    def add_mutations(self, maf: pd.DataFrame) -> None:
        """Register a MAF DataFrame for integration.

        Parameters
        ----------
        maf : pd.DataFrame
            Output of :func:`load_maf`.
        """
        self._maf = classify_variants(maf)
        logger.info("Registered %d mutations for integration", len(self._maf))

    def add_cna(self, cna: pd.DataFrame) -> None:
        """Register a CNA matrix for integration.

        Parameters
        ----------
        cna : pd.DataFrame
            Output of :func:`load_cna`.
        """
        self._cna = cna
        logger.info("Registered CNA data: %d genes", cna.shape[0])

    def add_methylation(self, meth: pd.DataFrame) -> None:
        """Register a methylation matrix for integration.

        Parameters
        ----------
        meth : pd.DataFrame
            Output of :func:`load_methylation`.
        """
        self._meth = meth
        logger.info("Registered methylation data: %d probes", meth.shape[0])

    def _mutation_features(self, gene: str) -> dict[str, Any]:
        """Extract mutation-based features for a single gene.

        Parameters
        ----------
        gene : str
            Gene symbol.

        Returns
        -------
        dict[str, Any]
            Keys: ``Mutation_Count``, ``High_Impact_Count``, ``Mean_VAF``.
        """
        if self._maf is None:
            return {"Mutation_Count": 0, "High_Impact_Count": 0, "Mean_VAF": 0.0}
        gm = self._maf[self._maf["Hugo_Symbol"] == gene]
        hi = int((gm["Impact"] == "HIGH").sum()) if "Impact" in gm.columns else 0
        vaf = float(gm["VAF"].mean()) if len(gm) > 0 else 0.0
        return {"Mutation_Count": len(gm), "High_Impact_Count": hi, "Mean_VAF": vaf}

    def _cna_features(self, gene: str) -> dict[str, float]:
        """Extract CNA-based features for a single gene.

        Parameters
        ----------
        gene : str
            Gene symbol.

        Returns
        -------
        dict[str, float]
            Keys: ``Amp_Freq``, ``Del_Freq``.
        """
        if self._cna is None or gene not in self._cna.index:
            return {"Amp_Freq": 0.0, "Del_Freq": 0.0}
        gc = self._cna.loc[gene]
        n = len(gc)
        return {
            "Amp_Freq": float((gc >= 2).sum() / n),
            "Del_Freq": float((gc <= -2).sum() / n),
        }

    def _methylation_features(self, gene: str) -> dict[str, float]:
        """Extract methylation-based features for a single gene.

        Parameters
        ----------
        gene : str
            Gene symbol.

        Returns
        -------
        dict[str, float]
            Keys: ``Mean_Beta``, ``Var_Beta``.
        """
        if self._meth is None or gene not in self._meth.index:
            return {"Mean_Beta": np.nan, "Var_Beta": np.nan}
        gm = self._meth.loc[gene]
        return {"Mean_Beta": float(gm.mean()), "Var_Beta": float(gm.var())}

    def _gene_features(self, gene: str) -> dict[str, Any]:
        """Build the full feature dict for one gene across all layers.

        Parameters
        ----------
        gene : str
            Gene symbol.

        Returns
        -------
        dict[str, Any]
            Combined mutation, CNA, and methylation features.
        """
        row: dict[str, Any] = {"Gene": gene}
        row.update(self._mutation_features(gene))
        row.update(self._cna_features(gene))
        row.update(self._methylation_features(gene))
        return row

    def build_feature_matrix(self, genes: list[str]) -> pd.DataFrame:
        """Build a unified per-gene feature matrix from all registered layers.

        Parameters
        ----------
        genes : list[str]
            Gene symbols to include (typically the top-K from XGBoost).

        Returns
        -------
        pd.DataFrame
            One row per gene with columns from each registered data
            layer.
        """
        rows = [self._gene_features(gene) for gene in genes]
        result = pd.DataFrame(rows)
        logger.info("Built multi-omics feature matrix: %d genes x %d features", *result.shape)
        return result

    def enrich_fusion_results(
        self,
        results: pd.DataFrame,
        genes: list[str] | None = None,
    ) -> pd.DataFrame:
        """Merge multi-omics features into existing fusion results.

        Parameters
        ----------
        results : pd.DataFrame
            Existing fusion results with a ``Gene`` column.
        genes : list[str], optional
            Gene list to build features for.

        Returns
        -------
        pd.DataFrame
            Input results with multi-omics columns appended.
        """
        if genes is None:
            genes = results["Gene"].tolist()
        features = self.build_feature_matrix(genes)
        return results.merge(features, on="Gene", how="left")
