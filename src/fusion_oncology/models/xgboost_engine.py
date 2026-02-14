"""
XGBoost-based gene importance scoring.

Trains a lightweight gradient-boosted classifier on the TCGA expression
matrix and returns per-gene feature importance scores.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import make_scorer, fbeta_score
from sklearn.model_selection import StratifiedKFold, cross_validate as sklearn_cv
from sklearn.preprocessing import LabelEncoder

from fusion_oncology.config import ProjectConfig

logger = logging.getLogger(__name__)


class XGBoostEngine:
    """
    Wraps XGBoost training and feature-importance extraction.

    Parameters
    ----------
    config : ProjectConfig
        Supplies ``xgb_n_estimators``, ``xgb_max_depth``, and ``top_k_genes``.
    """

    def __init__(self, config: ProjectConfig | None = None) -> None:
        """Initialise the XGBoost engine.

        Parameters
        ----------
        config : ProjectConfig, optional
            Runtime configuration.  Uses defaults when ``None``.
        """
        self.cfg = config or ProjectConfig()
        self.model: xgb.XGBClassifier | None = None
        self.label_encoder = LabelEncoder()
        self._feature_names: list[str] = []

    # ── training helpers ────────────────────────────────────────────────

    def _build_classifier(self) -> xgb.XGBClassifier:
        """Build a full XGBoost classifier with production hyper-params.

        Returns
        -------
        xgb.XGBClassifier
            Un-fitted classifier instance.
        """
        return xgb.XGBClassifier(
            n_estimators=self.cfg.xgb_n_estimators,
            max_depth=self.cfg.xgb_max_depth,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="multi:softprob",
            eval_metric="mlogloss",
            use_label_encoder=False,
            verbosity=0,
            n_jobs=-1,
        )

    # ── training ─────────────────────────────────────────────────────────

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "XGBoostEngine":
        """Train XGBoost on the expression matrix.

        Parameters
        ----------
        X : pd.DataFrame
            Gene-expression features (samples × genes).
        y : pd.Series
            Cancer-type labels.

        Returns
        -------
        self
        """
        X_num = X.select_dtypes(include=[np.number])
        self._feature_names = list(X_num.columns)
        y_enc = self.label_encoder.fit_transform(y)
        self.model = self._build_classifier()
        self.model.fit(X_num, y_enc)
        n, d = self.cfg.xgb_n_estimators, self.cfg.xgb_max_depth
        logger.info("XGBoost fitted  (%d trees, depth %d)", n, d)
        return self

    # ── evaluation helpers ───────────────────────────────────────────────

    def _encode_labels(self, y: pd.Series) -> np.ndarray:
        """Encode labels, reusing the fitted encoder when available.

        Parameters
        ----------
        y : pd.Series
            Raw cancer-type labels.

        Returns
        -------
        np.ndarray
            Integer-encoded label array.
        """
        if hasattr(self.label_encoder, "classes_"):
            return self.label_encoder.transform(y)
        return LabelEncoder().fit_transform(y)

    def _build_cv_classifier(self) -> xgb.XGBClassifier:
        """Build a lightweight XGBoost classifier for CV scoring.

        Returns
        -------
        xgb.XGBClassifier
            Un-fitted classifier with minimal logging.
        """
        return xgb.XGBClassifier(
            n_estimators=self.cfg.xgb_n_estimators,
            max_depth=self.cfg.xgb_max_depth,
            verbosity=0,
            use_label_encoder=False,
            n_jobs=-1,
        )

    @staticmethod
    def _summarise(scores: np.ndarray) -> tuple[float, float]:
        """Return (mean, std) for a per-fold score array."""
        return float(np.mean(scores)), float(np.std(scores))

    def _compute_cv_metrics(
        self,
        cv_results: dict[str, np.ndarray],
    ) -> dict[str, float]:
        """Compute mean ± std for all six evaluation metrics.

        Parameters
        ----------
        cv_results : dict[str, np.ndarray]
            Output of ``sklearn.model_selection.cross_validate``
            containing ``test_<scorer>`` arrays.

        Returns
        -------
        dict[str, float]
            Keys: ``mean_accuracy``, ``std_accuracy``,
            ``mean_precision``, ``std_precision``,
            ``mean_recall``, ``std_recall``,
            ``mean_f1``, ``std_f1``,
            ``mean_f2``, ``std_f2``,
            ``mean_roc_auc``, ``std_roc_auc``.
        """
        result: dict[str, float] = {}
        metric_map = {
            "accuracy": "accuracy",
            "precision": "precision_weighted",
            "recall": "recall_weighted",
            "f1": "f1_weighted",
            "f2": "f2_weighted",
            "roc_auc": "roc_auc_ovr_weighted",
        }
        for short_name, scorer_key in metric_map.items():
            arr = cv_results.get(f"test_{scorer_key}", np.array([np.nan]))
            mean, std = self._summarise(arr)
            result[f"mean_{short_name}"] = mean
            result[f"std_{short_name}"] = std

        logger.info(
            "CV  acc=%.4f  prec=%.4f  rec=%.4f  " "f1=%.4f  f2=%.4f  auc=%.4f",
            result["mean_accuracy"],
            result["mean_precision"],
            result["mean_recall"],
            result["mean_f1"],
            result["mean_f2"],
            result["mean_roc_auc"],
        )
        return result

    # ── evaluation ───────────────────────────────────────────────────────

    def cross_validate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        folds: int = 5,
    ) -> dict[str, Any]:
        """Run stratified k-fold cross-validation.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix (samples × genes).  Only numeric columns
            are used.
        y : pd.Series
            Cancer-type labels aligned with *X* rows.
        folds : int
            Number of stratified CV folds (default 5).

        Returns
        -------
        dict[str, float]
            Keys ``mean_accuracy`` and ``std_accuracy``.
        """
        X_num = X.select_dtypes(include=[np.number])

        # Filter out classes with fewer samples than folds to prevent
        # XGBoost label-mismatch errors when a class is absent from a
        # training fold.
        class_counts = y.value_counts()
        rare = class_counts[class_counts < folds].index
        if len(rare):
            logger.info(
                "Filtering %d rare classes (< %d samples) for CV: %s",
                len(rare),
                folds,
                list(rare)[:5],
            )
            mask = ~y.isin(rare)
            X_num = X_num.loc[mask]
            y = y.loc[mask]

        # Re-encode to contiguous 0..N-1 for the filtered subset
        le = LabelEncoder()
        y_enc = le.fit_transform(y)

        skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
        clf = self._build_cv_classifier()

        f2_scorer = make_scorer(
            fbeta_score,
            beta=2,
            average="weighted",
            zero_division=0,
        )
        scoring = {
            "accuracy": "accuracy",
            "precision_weighted": "precision_weighted",
            "recall_weighted": "recall_weighted",
            "f1_weighted": "f1_weighted",
            "f2_weighted": f2_scorer,
            "roc_auc_ovr_weighted": "roc_auc_ovr_weighted",
        }
        cv_results = sklearn_cv(clf, X_num, y_enc, cv=skf, scoring=scoring)
        return self._compute_cv_metrics(cv_results)

    # ── importance ───────────────────────────────────────────────────────

    def top_genes(self, k: int | None = None) -> dict[str, float]:
        """
        Return the *k* most important genes by XGBoost gain score.

        Parameters
        ----------
        k : int, optional
            Number of top genes. Defaults to ``config.top_k_genes``.

        Returns
        -------
        dict
            Ordered mapping ``{gene_name: importance_score}``.
        """
        if self.model is None:
            raise RuntimeError("Model not fitted yet – call .fit() first")
        k = k or self.cfg.top_k_genes
        importances = self.model.feature_importances_
        indices = np.argsort(importances)[::-1][:k]
        result = {self._feature_names[i]: float(importances[i]) for i in indices}
        logger.info("Top %d genes: %s", k, list(result.keys()))
        return result

    def all_importances(self) -> pd.Series:
        """Return all gene importances as a descending-sorted Series.

        Returns
        -------
        pd.Series
            Index = gene names, values = XGBoost feature importances,
            sorted from most to least important.

        Raises
        ------
        RuntimeError
            If the model has not been fitted yet.
        """
        if self.model is None:
            raise RuntimeError("Model not fitted yet – call .fit() first")
        return pd.Series(self.model.feature_importances_, index=self._feature_names).sort_values(
            ascending=False
        )
