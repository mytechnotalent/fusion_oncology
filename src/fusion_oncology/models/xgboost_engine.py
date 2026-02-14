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
from sklearn.model_selection import StratifiedKFold, cross_val_score
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

    # ── training ─────────────────────────────────────────────────────────

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "XGBoostEngine":
        """
        Train XGBoost on the expression matrix.

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

        self.model = xgb.XGBClassifier(
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
        self.model.fit(X_num, y_enc)
        logger.info(
            "XGBoost fitted  (%d trees, depth %d)",
            self.cfg.xgb_n_estimators,
            self.cfg.xgb_max_depth,
        )
        return self

    # ── evaluation ───────────────────────────────────────────────────────

    def cross_validate(
        self, X: pd.DataFrame, y: pd.Series, folds: int = 5
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
        y_enc = (
            self.label_encoder.transform(y)
            if hasattr(self.label_encoder, "classes_")
            else LabelEncoder().fit_transform(y)
        )
        skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)

        model_cv = xgb.XGBClassifier(
            n_estimators=self.cfg.xgb_n_estimators,
            max_depth=self.cfg.xgb_max_depth,
            verbosity=0,
            use_label_encoder=False,
            n_jobs=-1,
        )
        scores = cross_val_score(model_cv, X_num, y_enc, cv=skf, scoring="accuracy")
        result = {
            "mean_accuracy": float(np.mean(scores)),
            "std_accuracy": float(np.std(scores)),
        }
        logger.info(
            "CV accuracy: %.4f ± %.4f", result["mean_accuracy"], result["std_accuracy"]
        )
        return result

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
        return pd.Series(
            self.model.feature_importances_, index=self._feature_names
        ).sort_values(ascending=False)
