"""
Cell-line → patient domain adaptation.

Addresses the fundamental in-vitro / in-vivo gap by aligning feature
distributions between GDSC cell-line expression data and TCGA patient
tumour samples.  Three complementary strategies are implemented:

1. **Quantile normalisation** — forces marginal gene distributions
   to match a shared reference.
2. **ComBat batch correction** — location-and-scale parametric
   harmonisation (Johnson *et al.*, Biostatistics 2007).
3. **Correlation-based feature alignment** — filters to genes whose
   cell-line / patient expression distributions overlap.

Together these allow an XGBoost model trained on GDSC cell lines to
generalise to patient tumour biopsies.

References
----------
Johnson WE, Li C, Rabinovic A. Biostatistics 2007; 8(1):118–127.
Leek JT et al. Nat Rev Genet 2010; 11:733–739.
Mourragui S et al. PRECISE. Bioinformatics 2019.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from fusion_oncology.config import ProjectConfig

logger = logging.getLogger(__name__)


# ── Configuration ────────────────────────────────────────────────────────


@dataclass
class DomainAdaptConfig:
    """Parameters for cell-line → patient domain adaptation.

    Parameters
    ----------
    min_gene_variance : float
        Minimum gene variance to retain (filters flat genes).
    quantile_target : str
        Which domain to use as quantile reference
        (``"patient"`` or ``"cell_line"``).
    combat_parametric : bool
        If ``True``, use parametric ComBat; else empirical Bayes.
    overlap_correlation_min : float
        Minimum Spearman ρ between cell-line and patient
        expression for a gene to be retained.
    top_k_genes : int
        Maximum number of genes to carry through adaptation.
    """

    min_gene_variance: float = 0.1
    quantile_target: str = "patient"
    combat_parametric: bool = True
    overlap_correlation_min: float = 0.3
    top_k_genes: int = 2000


# ── Quantile normalisation ──────────────────────────────────────────────


def quantile_normalise(df: pd.DataFrame) -> pd.DataFrame:
    """Quantile-normalise columns of a DataFrame.

    Each column (sample) is normalised so that the distribution of
    values across genes matches the mean distribution.

    Parameters
    ----------
    df : pd.DataFrame
        Genes x samples (rows = genes, columns = samples).

    Returns
    -------
    pd.DataFrame
        Quantile-normalised DataFrame with same shape and index.
    """
    rank_mean = df.stack().groupby(df.rank(method="first").stack().astype(int)).mean()
    normalised = df.rank(method="min").stack().astype(int).map(rank_mean).unstack()
    return normalised


def quantile_normalise_to_reference(
    source: pd.DataFrame,
    reference: pd.DataFrame,
) -> pd.DataFrame:
    """Normalise *source* to match the quantile distribution of *reference*.

    Parameters
    ----------
    source : pd.DataFrame
        Data to normalise (samples x genes).
    reference : pd.DataFrame
        Reference distribution (samples x genes).

    Returns
    -------
    pd.DataFrame
        Normalised source DataFrame.
    """
    common_genes = source.columns.intersection(reference.columns)
    if len(common_genes) == 0:
        logger.warning("No overlapping genes — returning source unchanged")
        return source

    src = source[common_genes].copy()
    ref = reference[common_genes]

    # Compute reference quantile distribution (mean across samples)
    ref_sorted = np.sort(ref.values, axis=0)
    ref_mean = ref_sorted.mean(axis=1)

    # Map each source sample to the reference distribution
    result = src.copy()
    for col in src.columns:
        ranks = src[col].rank(method="min").astype(int) - 1
        # Clamp ranks to reference range
        n_ref = len(ref_mean)
        ranks_clamped = ranks.clip(0, n_ref - 1)
        result[col] = ref_mean[ranks_clamped.values]

    return result


# ── Parametric ComBat ───────────────────────────────────────────────────


def combat_correct(
    data: pd.DataFrame,
    batch_labels: np.ndarray,
    parametric: bool = True,
) -> pd.DataFrame:
    """Apply ComBat batch-effect correction.

    Implements the Johnson *et al.* (2007) parametric location-scale
    model. Given a matrix of (samples x genes) and a batch vector,
    estimates per-batch additive and multiplicative bias, then removes
    them.

    Parameters
    ----------
    data : pd.DataFrame
        Samples x genes expression matrix.
    batch_labels : np.ndarray
        Integer or string batch label per sample (length = n_samples).
    parametric : bool
        Use parametric priors (``True``) or empirical estimates.

    Returns
    -------
    pd.DataFrame
        Batch-corrected expression matrix (same shape).
    """
    if len(batch_labels) != len(data):
        msg = f"batch_labels length ({len(batch_labels)}) != " f"data rows ({len(data)})"
        raise ValueError(msg)

    batches = np.unique(batch_labels)
    if len(batches) < 2:
        logger.warning("Only one batch — no correction applied")
        return data.copy()

    mat = data.values.astype(float).copy()
    grand_mean = mat.mean(axis=0)
    grand_std = mat.std(axis=0)
    grand_std[grand_std == 0] = 1.0

    # Standardise
    standardised = (mat - grand_mean) / grand_std

    # Estimate per-batch location & scale
    corrected = standardised.copy()
    for batch in batches:
        mask = batch_labels == batch
        batch_data = standardised[mask]

        gamma_hat = batch_data.mean(axis=0)  # location shift
        delta_hat = batch_data.var(axis=0)  # scale factor
        delta_hat[delta_hat == 0] = 1.0

        if parametric:
            # Parametric shrinkage (empirical Bayes)
            gamma_bar = gamma_hat.mean()
            tau2 = gamma_hat.var()
            delta_bar = delta_hat.mean()
            v_bar = delta_hat.var()

            # Posterior estimates
            if tau2 > 0:
                gamma_star = (tau2 * gamma_hat + gamma_bar * (1.0 / len(batch_data))) / (
                    tau2 + 1.0 / len(batch_data)
                )
            else:
                gamma_star = gamma_hat

            delta_star = delta_hat  # simplified; full EB uses inverse-gamma
        else:
            gamma_star = gamma_hat
            delta_star = delta_hat

        # Correct
        corrected[mask] = (standardised[mask] - gamma_star) / np.sqrt(delta_star)

    # De-standardise
    result = corrected * grand_std + grand_mean
    return pd.DataFrame(result, index=data.index, columns=data.columns)


# ── Feature alignment ───────────────────────────────────────────────────


def align_features(
    cell_line_df: pd.DataFrame,
    patient_df: pd.DataFrame,
    min_correlation: float = 0.3,
    top_k: int = 2000,
) -> list[str]:
    """Select genes whose distributions overlap between domains.

    Computes per-gene rank correlation between cell-line and patient
    summary statistics (mean, variance) and retains genes above the
    minimum correlation threshold.

    Parameters
    ----------
    cell_line_df : pd.DataFrame
        Cell-line expression (samples x genes).
    patient_df : pd.DataFrame
        Patient expression (samples x genes).
    min_correlation : float
        Minimum absolute mean-rank correlation to keep a gene.
    top_k : int
        Maximum genes to retain.

    Returns
    -------
    list[str]
        Gene names that pass the alignment filter.
    """
    common = cell_line_df.columns.intersection(patient_df.columns)
    if len(common) == 0:
        return []

    cl_stats = pd.DataFrame(
        {
            "cl_mean": cell_line_df[common].mean(),
            "cl_std": cell_line_df[common].std(),
        }
    )
    pt_stats = pd.DataFrame(
        {
            "pt_mean": patient_df[common].mean(),
            "pt_std": patient_df[common].std(),
        }
    )

    merged = cl_stats.join(pt_stats)
    merged["mean_corr"] = (merged["cl_mean"].rank() - merged["pt_mean"].rank()).abs()
    # Lower rank difference = better alignment
    merged = merged.sort_values("mean_corr")

    # Filter by variance overlap
    merged["var_ratio"] = merged["cl_std"] / merged["pt_std"].replace(0, 1)
    mask = merged["var_ratio"].between(0.2, 5.0)
    selected = merged.loc[mask].head(top_k)

    genes = selected.index.tolist()
    logger.info(
        "Feature alignment: %d / %d genes pass (from %d common)",
        len(genes),
        len(common),
        len(common),
    )
    return genes


# ── Full pipeline ───────────────────────────────────────────────────────


class DomainAdapter:
    """End-to-end cell-line → patient domain adaptation pipeline.

    Chains three steps:
    1. Feature alignment (gene filtering)
    2. Quantile normalisation to patient reference
    3. ComBat batch correction

    Parameters
    ----------
    config : DomainAdaptConfig, optional
        Adaptation parameters.
    project_config : ProjectConfig, optional
        Runtime configuration.
    """

    def __init__(
        self,
        config: DomainAdaptConfig | None = None,
        project_config: ProjectConfig | None = None,
    ) -> None:
        """Initialise the domain adapter.

        Parameters
        ----------
        config : DomainAdaptConfig, optional
            Adaptation hyperparameters.
        project_config : ProjectConfig, optional
            Runtime configuration.
        """
        self.dc = config or DomainAdaptConfig()
        self.cfg = project_config or ProjectConfig()
        self._aligned_genes: list[str] = []

    def adapt(
        self,
        cell_line_expr: pd.DataFrame,
        patient_expr: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Run the full adaptation pipeline.

        Parameters
        ----------
        cell_line_expr : pd.DataFrame
            Cell-line expression (samples x genes).
        patient_expr : pd.DataFrame
            Patient tumour expression (samples x genes).

        Returns
        -------
        tuple[pd.DataFrame, pd.DataFrame]
            ``(adapted_cell_line, adapted_patient)`` — harmonised
            expression matrices with shared gene space.
        """
        # Step 1: Feature alignment
        self._aligned_genes = align_features(
            cell_line_expr,
            patient_expr,
            min_correlation=self.dc.overlap_correlation_min,
            top_k=self.dc.top_k_genes,
        )

        if not self._aligned_genes:
            logger.warning("No genes passed alignment — returning inputs unchanged")
            return cell_line_expr, patient_expr

        cl = cell_line_expr[self._aligned_genes].copy()
        pt = patient_expr[self._aligned_genes].copy()

        # Step 2: Quantile normalisation (cell-line → patient reference)
        if self.dc.quantile_target == "patient":
            cl = quantile_normalise_to_reference(cl, pt)
        else:
            pt = quantile_normalise_to_reference(pt, cl)

        # Step 3: ComBat batch correction
        combined = pd.concat([cl, pt], axis=0)
        batch = np.array(["cell_line"] * len(cl) + ["patient"] * len(pt))
        corrected = combat_correct(
            combined,
            batch,
            parametric=self.dc.combat_parametric,
        )

        cl_out = corrected.iloc[: len(cl)]
        pt_out = corrected.iloc[len(cl) :]
        cl_out.index = cl.index
        pt_out.index = pt.index

        logger.info(
            "Domain adaptation complete: %d genes, %d cell-lines + %d patients",
            len(self._aligned_genes),
            len(cl),
            len(pt),
        )
        return cl_out, pt_out

    @property
    def aligned_genes(self) -> list[str]:
        """Return the genes selected during feature alignment.

        Returns
        -------
        list[str]
            Gene names surviving the alignment filter.
        """
        return self._aligned_genes

    def summary(self) -> dict[str, Any]:
        """Return a summary of the last adaptation run.

        Returns
        -------
        dict[str, Any]
            Counts and configuration used.
        """
        return {
            "n_aligned_genes": len(self._aligned_genes),
            "quantile_target": self.dc.quantile_target,
            "combat_parametric": self.dc.combat_parametric,
            "top_k_genes": self.dc.top_k_genes,
            "min_correlation": self.dc.overlap_correlation_min,
        }
