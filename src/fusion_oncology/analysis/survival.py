"""
Survival analysis hooks.

Provides Kaplan–Meier estimation and log-rank testing against gene
expression strata.  Requires that the input dataframe includes
``OS_MONTHS`` (overall survival time) and ``OS_STATUS`` (event flag)
columns alongside gene expression values.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from fusion_oncology.config import ProjectConfig

logger = logging.getLogger(__name__)

try:
    from lifelines import KaplanMeierFitter
    from lifelines.statistics import logrank_test

    _HAS_LIFELINES = True
except ImportError:  # graceful degradation
    _HAS_LIFELINES = False
    logger.info("lifelines not installed – survival analysis disabled")


class SurvivalAnalyzer:
    """
    Kaplan–Meier survival analysis stratified by gene expression.

    Parameters
    ----------
    config : ProjectConfig
        Supplies survival column names.
    """

    def __init__(self, config: ProjectConfig | None = None) -> None:
        """Initialise the survival analyser.

        Parameters
        ----------
        config : ProjectConfig, optional
            Runtime configuration.  Uses defaults when ``None``.

        Raises
        ------
        ImportError
            If the ``lifelines`` package is not installed.
        """
        self.cfg = config or ProjectConfig()
        if not _HAS_LIFELINES:
            raise ImportError(
                "The 'lifelines' package is required for survival analysis.  "
                "Install it with:  pip install lifelines"
            )

    def stratify(
        self,
        clinical: pd.DataFrame,
        gene: str,
        n_groups: int = 2,
    ) -> pd.DataFrame:
        """
        Split patients into high / low expression groups for *gene*.

        Parameters
        ----------
        clinical : pd.DataFrame
            Must contain the gene column plus survival columns.
        gene : str
            Column name for the gene of interest.
        n_groups : int
            Number of quantile-based groups (default 2 = median split).

        Returns
        -------
        pd.DataFrame
            Original frame with an added ``expr_group`` column.
        """
        df = clinical.copy()
        df["expr_group"] = pd.qcut(df[gene], q=n_groups, labels=False)
        return df

    def _validate_columns(
        self,
        clinical: pd.DataFrame,
        gene: str,
    ) -> tuple[str, str]:
        """Check required columns and return time / event names.

        Parameters
        ----------
        clinical : pd.DataFrame
            Clinical dataframe to validate.
        gene : str
            Gene column name that must be present.

        Returns
        -------
        tuple[str, str]
            ``(time_col, event_col)`` from the config.

        Raises
        ------
        KeyError
            If a required column is missing.
        """
        time_col = self.cfg.survival_time_col
        event_col = self.cfg.survival_event_col
        for col in (time_col, event_col, gene):
            if col not in clinical.columns:
                raise KeyError(f"Missing column: {col}")
        return time_col, event_col

    def _split_groups(
        self,
        clinical: pd.DataFrame,
        gene: str,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Stratify patients and return high / low subsets.

        Parameters
        ----------
        clinical : pd.DataFrame
            Clinical dataframe with gene expression values.
        gene : str
            Gene column used for stratification.

        Returns
        -------
        tuple[pd.DataFrame, pd.DataFrame]
            ``(high_df, low_df)`` expression groups.
        """
        df = self.stratify(clinical, gene)
        high = df[df["expr_group"] == 1]
        low = df[df["expr_group"] == 0]
        return high, low

    def _fit_kmf(
        self,
        group_df: pd.DataFrame,
        time_col: str,
        event_col: str,
        label: str,
    ) -> KaplanMeierFitter:
        """Fit a :class:`KaplanMeierFitter` on one group.

        Parameters
        ----------
        group_df : pd.DataFrame
            Subset of patients belonging to one stratum.
        time_col : str
            Column with survival times.
        event_col : str
            Column with event flags.
        label : str
            Display label for the fitted curve.

        Returns
        -------
        KaplanMeierFitter
            Fitted estimator.
        """
        return KaplanMeierFitter().fit(
            group_df[time_col],
            group_df[event_col],
            label=label,
        )

    def _run_logrank(
        self,
        high: pd.DataFrame,
        low: pd.DataFrame,
        time_col: str,
        event_col: str,
    ) -> float:
        """Run the log-rank test between two groups.

        Parameters
        ----------
        high : pd.DataFrame
            High-expression group.
        low : pd.DataFrame
            Low-expression group.
        time_col : str
            Column with survival times.
        event_col : str
            Column with event flags.

        Returns
        -------
        float
            Log-rank test p-value.
        """
        lr = logrank_test(
            high[time_col],
            low[time_col],
            high[event_col],
            low[event_col],
        )
        return float(lr.p_value)

    def _build_km_result(
        self,
        kmf_high: KaplanMeierFitter,
        kmf_low: KaplanMeierFitter,
        p_value: float,
    ) -> dict[str, Any]:
        """Assemble the Kaplan–Meier result dictionary.

        Parameters
        ----------
        kmf_high : KaplanMeierFitter
            Fitted estimator for the high-expression group.
        kmf_low : KaplanMeierFitter
            Fitted estimator for the low-expression group.
        p_value : float
            Log-rank test p-value.

        Returns
        -------
        dict[str, Any]
            Result with fitted estimators, p-value, and medians.
        """
        return {
            "kmf_high": kmf_high,
            "kmf_low": kmf_low,
            "logrank_p": p_value,
            "median_high": float(kmf_high.median_survival_time_),
            "median_low": float(kmf_low.median_survival_time_),
        }

    def _log_km_result(
        self,
        gene: str,
        result: dict[str, Any],
    ) -> None:
        """Log a summary of the Kaplan–Meier analysis.

        Parameters
        ----------
        gene : str
            Gene that was analysed.
        result : dict[str, Any]
            Result dictionary from :meth:`_build_km_result`.
        """
        logger.info(
            "%s survival: log-rank p=%.4e " " median(H)=%.1f  median(L)=%.1f",
            gene,
            result["logrank_p"],
            result["median_high"],
            result["median_low"],
        )

    def kaplan_meier(
        self,
        clinical: pd.DataFrame,
        gene: str,
    ) -> dict[str, Any]:
        """Fit Kaplan–Meier curves for high vs low expression and run
        the log-rank test.

        Parameters
        ----------
        clinical : pd.DataFrame
            Must contain the gene column, the survival-time column
            (``config.survival_time_col``), and the event column
            (``config.survival_event_col``).
        gene : str
            Column name for the gene of interest.

        Returns
        -------
        dict[str, Any]
            Keys: ``kmf_high`` (:class:`KaplanMeierFitter`),
            ``kmf_low`` (:class:`KaplanMeierFitter`),
            ``logrank_p`` (float), ``median_high`` (float),
            ``median_low`` (float).

        Raises
        ------
        KeyError
            If a required column is missing from *clinical*.
        """
        time_col, event_col = self._validate_columns(clinical, gene)
        high, low = self._split_groups(clinical, gene)
        kmf_h = self._fit_kmf(high, time_col, event_col, f"{gene} HIGH")
        kmf_l = self._fit_kmf(low, time_col, event_col, f"{gene} LOW")
        p_val = self._run_logrank(high, low, time_col, event_col)
        result = self._build_km_result(kmf_h, kmf_l, p_val)
        self._log_km_result(gene, result)
        return result
