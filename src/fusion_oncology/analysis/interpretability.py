"""
SHAP-based model interpretability for the drug-sensitivity engine.

Provides gene-level and pathway-level explanations for XGBoost
predictions so that clinicians can ask *"why did the model rank
this target highest?"* and receive a mechanistically grounded
answer backed by Shapley additive explanations.

Layers of explanation:

1. **Gene-level SHAP** — per-feature Shapley values via
   ``shap.TreeExplainer`` (exact, polynomial-time for tree ensembles).
2. **Pathway-level aggregation** — sums absolute SHAP values for
   genes belonging to each cancer signalling pathway.
3. **Mechanistic rationale** — maps top SHAP drivers to the
   drug-target and resistance databases to generate human-readable
   narratives.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
import shap
import xgboost as xgb

from fusion_oncology.analysis.drug_target import DrugTargetMapper
from fusion_oncology.analysis.pathway import CANCER_PATHWAYS, PathwayEnrichment
from fusion_oncology.analysis.resistance import ResistancePredictor
from fusion_oncology.config import ProjectConfig

logger = logging.getLogger(__name__)


# ── SHAP Explainer ───────────────────────────────────────────────────────


class ShapExplainer:
    """SHAP-based interpretability wrapper for XGBoost models.

    Wraps ``shap.TreeExplainer`` to produce gene-level and
    pathway-level feature attributions.  Designed to accept a fitted
    ``XGBoostEngine`` or any ``xgb.XGBClassifier`` / ``Booster``.

    Parameters
    ----------
    model : xgb.XGBClassifier | xgb.Booster
        A fitted XGBoost model.
    feature_names : list[str]
        Column names corresponding to the model's input features.
    config : ProjectConfig, optional
        Runtime configuration.
    """

    def __init__(
        self,
        model: xgb.XGBClassifier | xgb.Booster,
        feature_names: list[str],
        config: ProjectConfig | None = None,
    ) -> None:
        """Initialise the SHAP explainer.

        Parameters
        ----------
        model : xgb.XGBClassifier | xgb.Booster
            Fitted XGBoost model.
        feature_names : list[str]
            Feature names matching model input columns.
        config : ProjectConfig, optional
            Runtime configuration.
        """
        self.cfg = config or ProjectConfig()
        self._feature_names = list(feature_names)
        self._model = model
        # Use get_booster() for XGBClassifier to avoid SHAP/XGBoost
        # shape mismatch in multi-class mode with newer XGBoost (≥2.1).
        if hasattr(model, "get_booster"):
            booster = model.get_booster()
            # Set num_class on booster to help SHAP infer output shape
            n_classes = getattr(model, "n_classes_", None)
            if n_classes and n_classes > 2:
                booster.set_param({"num_class": n_classes})
            self._explainer = shap.TreeExplainer(booster)
        else:
            self._explainer = shap.TreeExplainer(model)
        self._drug_mapper = DrugTargetMapper()
        self._resistance = ResistancePredictor()
        self._pathway = PathwayEnrichment()
        logger.info(
            "ShapExplainer initialised (%d features)",
            len(feature_names),
        )

    # ── from XGBoostEngine convenience constructor ───────────────────

    @classmethod
    def from_engine(
        cls,
        engine: Any,
        config: ProjectConfig | None = None,
    ) -> "ShapExplainer":
        """Create a ShapExplainer directly from an XGBoostEngine.

        Parameters
        ----------
        engine : XGBoostEngine
            A fitted ``XGBoostEngine`` instance.
        config : ProjectConfig, optional
            Runtime configuration.

        Returns
        -------
        ShapExplainer
            Ready-to-use explainer.

        Raises
        ------
        ValueError
            If the engine has not been fitted.
        """
        if engine.model is None:
            raise ValueError("XGBoostEngine must be fitted before explaining.")
        return cls(
            model=engine.model,
            feature_names=engine._feature_names,
            config=config or engine.cfg,
        )

    # ── raw SHAP values ─────────────────────────────────────────────────

    def _compute_shap_values(
        self,
        X: pd.DataFrame,
    ) -> np.ndarray:
        """Compute SHAP values for every sample in *X*.

        Parameters
        ----------
        X : pd.DataFrame
            Input feature matrix (samples x features).

        Returns
        -------
        np.ndarray
            SHAP values.  Shape depends on the number of classes:
            ``(n_samples, n_features)`` for binary,
            ``(n_samples, n_features, n_classes)`` for multi-class.
        """
        X_num = X[self._feature_names] if set(self._feature_names).issubset(X.columns) else X
        return self._explainer.shap_values(X_num)

    def shap_values(self, X: pd.DataFrame) -> np.ndarray:
        """Public API for raw SHAP value computation.

        Parameters
        ----------
        X : pd.DataFrame
            Input feature matrix.

        Returns
        -------
        np.ndarray
            SHAP values array.
        """
        return self._compute_shap_values(X)

    # ── gene-level importance ────────────────────────────────────────────

    def _collapse_multiclass(self, sv: np.ndarray) -> np.ndarray:
        """Collapse multi-class SHAP values to 2-D by averaging.

        Parameters
        ----------
        sv : np.ndarray
            Raw SHAP values — either ``(n, f)`` or ``(n, f, c)`` or
            a list of arrays (one per class).

        Returns
        -------
        np.ndarray
            Shape ``(n, f)`` — class-averaged absolute SHAP values.
        """
        if isinstance(sv, list):
            # shap.TreeExplainer returns list of arrays for multi-class
            stacked = np.stack(sv, axis=-1)  # (n, f, c)
            return np.abs(stacked).mean(axis=-1)
        if sv.ndim == 3:
            return np.abs(sv).mean(axis=-1)
        return np.abs(sv)

    def gene_importance(self, X: pd.DataFrame, top_k: int = 20) -> pd.DataFrame:
        """Rank genes by mean |SHAP| across all samples and classes.

        Parameters
        ----------
        X : pd.DataFrame
            Input feature matrix.
        top_k : int
            Number of top features to return.

        Returns
        -------
        pd.DataFrame
            Columns: ``gene``, ``mean_shap``, ``std_shap``, ``rank``.
            Sorted by descending ``mean_shap``.
        """
        sv = self._compute_shap_values(X)
        abs_sv = self._collapse_multiclass(sv)
        means = abs_sv.mean(axis=0)
        stds = abs_sv.std(axis=0)

        df = pd.DataFrame(
            {
                "gene": self._feature_names,
                "mean_shap": means,
                "std_shap": stds,
            }
        )
        df = df.sort_values("mean_shap", ascending=False).head(top_k)
        df["rank"] = range(1, len(df) + 1)
        df = df.reset_index(drop=True)
        logger.info("Top gene by SHAP: %s (%.6f)", df.iloc[0]["gene"], df.iloc[0]["mean_shap"])
        return df

    # ── pathway-level aggregation ────────────────────────────────────────

    def _aggregate_pathway_shap(
        self,
        gene_shap: dict[str, float],
    ) -> list[dict[str, Any]]:
        """Sum absolute SHAP contributions per cancer pathway.

        Parameters
        ----------
        gene_shap : dict[str, float]
            Mapping of gene name → mean |SHAP|.

        Returns
        -------
        list[dict[str, Any]]
            Each dict: ``pathway``, ``shap_sum``, ``n_genes``,
            ``contributing_genes``.
        """
        rows: list[dict[str, Any]] = []
        for pathway_name, pathway_genes in CANCER_PATHWAYS.items():
            hits = {g: gene_shap[g] for g in pathway_genes if g in gene_shap}
            if hits:
                rows.append(
                    {
                        "pathway": pathway_name,
                        "shap_sum": round(sum(hits.values()), 6),
                        "n_genes": len(hits),
                        "contributing_genes": sorted(hits, key=hits.get, reverse=True),
                    }
                )
        return rows

    def pathway_importance(self, X: pd.DataFrame) -> pd.DataFrame:
        """Aggregate SHAP importance at the pathway level.

        Sums gene-level |SHAP| for every gene that belongs to a
        curated cancer signalling pathway.  This provides clinicians
        with a higher-level explanation: *which biological process
        is driving the prediction?*

        Parameters
        ----------
        X : pd.DataFrame
            Input feature matrix.

        Returns
        -------
        pd.DataFrame
            Columns: ``pathway``, ``shap_sum``, ``n_genes``,
            ``contributing_genes``.  Sorted descending by ``shap_sum``.
        """
        gene_df = self.gene_importance(X, top_k=len(self._feature_names))
        gene_shap = dict(zip(gene_df["gene"], gene_df["mean_shap"]))
        rows = self._aggregate_pathway_shap(gene_shap)

        if not rows:
            return pd.DataFrame(
                columns=["pathway", "shap_sum", "n_genes", "contributing_genes"],
            )
        df = pd.DataFrame(rows).sort_values("shap_sum", ascending=False)
        return df.reset_index(drop=True)

    # ── sample-level explanation ─────────────────────────────────────────

    def explain_sample(
        self,
        X: pd.DataFrame,
        sample_idx: int = 0,
        top_k: int = 10,
    ) -> pd.DataFrame:
        """Explain a single sample's prediction.

        Parameters
        ----------
        X : pd.DataFrame
            Full feature matrix.
        sample_idx : int
            Row index of the sample to explain.
        top_k : int
            Number of top contributing features to return.

        Returns
        -------
        pd.DataFrame
            Columns: ``gene``, ``shap_value``, ``feature_value``,
            ``direction``.  Sorted by absolute SHAP contribution.
        """
        sv = self._compute_shap_values(X)
        abs_sv = self._collapse_multiclass(sv)
        sample_shap = abs_sv[sample_idx]

        # For direction, use the raw (signed) values
        if isinstance(sv, list):
            raw_signed = np.stack(sv, axis=-1)[sample_idx].mean(axis=-1)
        elif sv.ndim == 3:
            raw_signed = sv[sample_idx].mean(axis=-1)
        else:
            raw_signed = sv[sample_idx]

        X_num = X[self._feature_names] if set(self._feature_names).issubset(X.columns) else X
        feature_vals = X_num.iloc[sample_idx].values

        df = pd.DataFrame(
            {
                "gene": self._feature_names,
                "shap_value": raw_signed,
                "abs_shap": sample_shap,
                "feature_value": feature_vals,
            }
        )
        df["direction"] = np.where(df["shap_value"] > 0, "risk ↑", "protective ↓")
        df = df.sort_values("abs_shap", ascending=False).head(top_k)
        df = df.drop(columns=["abs_shap"]).reset_index(drop=True)
        return df

    # ── mechanistic rationale ────────────────────────────────────────────

    def _build_gene_rationale(self, gene: str, shap_val: float) -> dict[str, Any]:
        """Build a mechanistic rationale for a single gene.

        Combines SHAP attribution with drug-target lookups and
        resistance alerts to form a human-readable explanation.

        Parameters
        ----------
        gene : str
            Gene symbol.
        shap_val : float
            Mean absolute SHAP value for this gene.

        Returns
        -------
        dict[str, Any]
            Rationale record with gene, shap, drugs, resistance,
            pathways, and narrative fields.
        """
        drugs = self._drug_mapper.lookup(gene)
        resistance = self._resistance.predict(gene)
        pathway_hits = self._pathway.lookup(gene)

        drug_names = [d["drug"] for d in drugs[:3]] if drugs else []
        resist_mechs = [r["mechanism"] for r in resistance[:2]] if resistance else []
        pw_names = list(pathway_hits) if pathway_hits else []

        # Build narrative
        parts: list[str] = []
        direction = "high importance" if shap_val > 0 else "moderate importance"
        parts.append(f"{gene} shows {direction} (SHAP={shap_val:.4f}).")

        if drug_names:
            parts.append(f"Targetable with {', '.join(drug_names)}.")
        if pw_names:
            parts.append(f"Member of {', '.join(pw_names[:2])} pathway(s).")
        if resist_mechs:
            parts.append(f"Resistance risk: {resist_mechs[0]}.")

        return {
            "gene": gene,
            "mean_shap": round(shap_val, 6),
            "drugs": drug_names,
            "resistance_mechanisms": resist_mechs,
            "pathways": pw_names,
            "narrative": " ".join(parts),
        }

    def mechanistic_rationale(
        self,
        X: pd.DataFrame,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """Generate mechanistic rationales for top SHAP drivers.

        For each of the top-K genes by SHAP importance, queries the
        drug-target database, resistance catalogue, and pathway
        membership to produce a clinician-facing narrative.

        Parameters
        ----------
        X : pd.DataFrame
            Input feature matrix.
        top_k : int
            Number of top genes to explain.

        Returns
        -------
        list[dict[str, Any]]
            One rationale dict per gene.
        """
        gene_df = self.gene_importance(X, top_k=top_k)
        rationales = []
        for _, row in gene_df.iterrows():
            rationale = self._build_gene_rationale(row["gene"], row["mean_shap"])
            rationales.append(rationale)
        return rationales

    # ── full report ──────────────────────────────────────────────────────

    def full_report(self, X: pd.DataFrame, top_k: int = 10) -> dict[str, Any]:
        """Generate a comprehensive interpretability report.

        Combines gene-level SHAP rankings, pathway-level aggregation,
        a single-sample deep dive, and mechanistic rationales into
        a single report dict.

        Parameters
        ----------
        X : pd.DataFrame
            Input feature matrix.
        top_k : int
            Number of top genes / pathways to include.

        Returns
        -------
        dict[str, Any]
            Keys: ``gene_importance``, ``pathway_importance``,
            ``sample_explanation``, ``mechanistic_rationales``,
            ``summary``.
        """
        gene_df = self.gene_importance(X, top_k=top_k)
        pathway_df = self.pathway_importance(X)
        sample_df = self.explain_sample(X, sample_idx=0, top_k=top_k)
        rationales = self.mechanistic_rationale(X, top_k=min(top_k, 5))

        return {
            "gene_importance": gene_df,
            "pathway_importance": pathway_df,
            "sample_explanation": sample_df,
            "mechanistic_rationales": rationales,
            "summary": {
                "top_gene": gene_df.iloc[0]["gene"] if not gene_df.empty else None,
                "top_pathway": (pathway_df.iloc[0]["pathway"] if not pathway_df.empty else None),
                "n_features_explained": len(gene_df),
                "n_pathways_hit": len(pathway_df),
            },
        }
