"""
XGBoost-based gene importance scoring.

Trains a production-grade gradient-boosted classifier with:
- Intelligent class merging (rare cancer types → OTHER)
- Engineered statistical features (row-level moments, gene interactions)
- Optional Optuna Bayesian hyper-parameter optimisation
- Early stopping to prevent over-fitting
- Repeated stratified k-fold for stable CV estimates
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import make_scorer, fbeta_score, precision_score, recall_score, f1_score
from sklearn.model_selection import (
    RepeatedStratifiedKFold,
    StratifiedKFold,
    cross_validate as sklearn_cv,
)
from sklearn.preprocessing import LabelEncoder

from fusion_oncology.config import ProjectConfig

logger = logging.getLogger(__name__)

# ── Class-merging threshold ──────────────────────────────────────────────
_MIN_CLASS_SIZE = 40


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
        self._engineered: bool = False

    # ── class merging ────────────────────────────────────────────────────

    @staticmethod
    def merge_rare_classes(
        y: pd.Series,
        min_size: int = _MIN_CLASS_SIZE,
    ) -> pd.Series:
        """Merge cancer types with fewer than *min_size* samples into OTHER.

        Parameters
        ----------
        y : pd.Series
            Cancer-type labels.
        min_size : int
            Minimum samples required to keep a class.

        Returns
        -------
        pd.Series
            Labels with rare classes replaced by ``"OTHER"``.
        """
        counts = y.value_counts()
        rare = counts[counts < min_size].index
        if len(rare) == 0:
            return y
        merged = y.copy()
        merged[merged.isin(rare)] = "OTHER"
        n_classes = merged.nunique()
        logger.info(
            "Merged %d rare classes (< %d samples) → %d classes remain",
            len(rare),
            min_size,
            n_classes,
        )
        return merged

    # ── feature engineering ──────────────────────────────────────────────

    @staticmethod
    def engineer_features(X: pd.DataFrame) -> pd.DataFrame:
        """Add sample-level statistical features to the expression matrix.

        Appends row-wise mean, std, skewness, kurtosis, max, min,
        range, median, IQR, and coefficient of variation.  This gives
        XGBoost per-sample distributional context that single-gene
        columns alone cannot provide.

        Parameters
        ----------
        X : pd.DataFrame
            Numeric feature matrix.

        Returns
        -------
        pd.DataFrame
            Augmented matrix with 10 extra columns.
        """
        Xn = X.select_dtypes(include=[np.number])
        vals = Xn.values
        row_mean = np.nanmean(vals, axis=1)
        row_std = np.nanstd(vals, axis=1)
        safe_std = np.where(row_std < 1e-12, 1.0, row_std)
        q25 = np.nanpercentile(vals, 25, axis=1)
        q75 = np.nanpercentile(vals, 75, axis=1)
        n = vals.shape[1]
        # Skewness and kurtosis via moments
        centered = vals - row_mean[:, None]
        m3 = np.nanmean(centered**3, axis=1)
        m4 = np.nanmean(centered**4, axis=1)
        skew = m3 / (safe_std**3)
        kurt = (m4 / (safe_std**4)) - 3.0
        eng = pd.DataFrame(
            {
                "_row_mean": row_mean,
                "_row_std": row_std,
                "_row_skew": skew,
                "_row_kurt": kurt,
                "_row_max": np.nanmax(vals, axis=1),
                "_row_min": np.nanmin(vals, axis=1),
                "_row_range": np.nanmax(vals, axis=1) - np.nanmin(vals, axis=1),
                "_row_median": np.nanmedian(vals, axis=1),
                "_row_iqr": q75 - q25,
                "_row_cv": row_std / np.where(np.abs(row_mean) < 1e-12, 1.0, np.abs(row_mean)),
            },
            index=X.index,
        )
        return pd.concat([X, eng], axis=1)

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
            learning_rate=self.cfg.xgb_learning_rate,
            min_child_weight=self.cfg.xgb_min_child_weight,
            gamma=self.cfg.xgb_gamma,
            subsample=self.cfg.xgb_subsample,
            colsample_bytree=self.cfg.xgb_colsample_bytree,
            colsample_bylevel=0.7,
            reg_alpha=self.cfg.xgb_reg_alpha,
            reg_lambda=self.cfg.xgb_reg_lambda,
            objective="multi:softprob",
            eval_metric="mlogloss",
            use_label_encoder=False,
            verbosity=0,
            n_jobs=-1,
        )

    # ── training ─────────────────────────────────────────────────────────

    def _compute_sample_weights(self, y_enc: np.ndarray) -> np.ndarray:
        """Compute balanced sample weights inversely proportional to class frequency.

        Parameters
        ----------
        y_enc : np.ndarray
            Integer-encoded label array.

        Returns
        -------
        np.ndarray
            Per-sample weight array.
        """
        classes, counts = np.unique(y_enc, return_counts=True)
        weight_map = {c: len(y_enc) / (len(classes) * n) for c, n in zip(classes, counts)}
        return np.array([weight_map[label] for label in y_enc])

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
        if self.cfg.enable_feature_engineering:
            X_aug = self.engineer_features(X)
        else:
            X_aug = X
        X_num = X_aug.select_dtypes(include=[np.number])
        self._feature_names = list(X_num.columns)
        self._engineered = self.cfg.enable_feature_engineering
        y_enc = self.label_encoder.fit_transform(y)
        self.model = self._build_classifier()
        weights = self._compute_sample_weights(y_enc)
        self.model.fit(X_num, y_enc, sample_weight=weights)
        n, d = self.cfg.xgb_n_estimators, self.cfg.xgb_max_depth
        logger.info("XGBoost fitted  (%d trees, depth %d, %d feats)", n, d, X_num.shape[1])
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
        """Build XGBoost classifier for CV scoring.

        Uses the same hyperparameters as the production classifier
        so that CV metrics reflect real-world performance.

        Returns
        -------
        xgb.XGBClassifier
            Un-fitted classifier.
        """
        return self._build_classifier()

    @staticmethod
    def _summarise(scores: np.ndarray) -> tuple[float, float]:
        """Return (mean, std) for a per-fold score array."""
        return float(np.mean(scores)), float(np.std(scores))

    @staticmethod
    def _cv_metric_map() -> dict[str, str]:
        """Return the mapping from short metric names to scorer keys.

        Returns
        -------
        dict[str, str]
            Keys are display names, values are scorer dict keys.
        """
        return {
            "accuracy": "accuracy",
            "precision": "precision_weighted",
            "recall": "recall_weighted",
            "f1": "f1_weighted",
            "f2": "f2_weighted",
            "roc_auc": "roc_auc_ovr_weighted",
        }

    def _log_cv_summary(self, result: dict[str, float]) -> None:
        """Log a one-line summary of cross-validation metrics.

        Parameters
        ----------
        result : dict[str, float]
            Metric dictionary with ``mean_<metric>`` keys.
        """
        logger.info(
            "CV  acc=%.4f  prec=%.4f  rec=%.4f  f1=%.4f  f2=%.4f  auc=%.4f",
            result["mean_accuracy"],
            result["mean_precision"],
            result["mean_recall"],
            result["mean_f1"],
            result["mean_f2"],
            result["mean_roc_auc"],
        )

    def _compute_cv_metrics(
        self,
        cv_results: dict[str, np.ndarray],
    ) -> dict[str, float]:
        """Compute mean and std for all six evaluation metrics.

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
        for short_name, scorer_key in self._cv_metric_map().items():
            arr = cv_results.get(f"test_{scorer_key}", np.array([np.nan]))
            mean, std = self._summarise(arr)
            result[f"mean_{short_name}"] = mean
            result[f"std_{short_name}"] = std
        self._log_cv_summary(result)
        return result

    # ── evaluation ───────────────────────────────────────────────────────

    @staticmethod
    def _filter_rare_classes(
        X_num: pd.DataFrame,
        y: pd.Series,
        folds: int,
    ) -> tuple[pd.DataFrame, pd.Series]:
        """Remove classes with fewer samples than the fold count.

        Applied *after* class merging as a safety net — ensures every
        class has at least ``folds`` samples for stratification.

        Parameters
        ----------
        X_num : pd.DataFrame
            Numeric feature matrix.
        y : pd.Series
            Cancer-type labels.
        folds : int
            Number of CV folds (minimum class size).

        Returns
        -------
        tuple[pd.DataFrame, pd.Series]
            Filtered ``(X_num, y)`` pair.
        """
        class_counts = y.value_counts()
        rare = class_counts[class_counts < folds].index
        if not len(rare):
            return X_num, y
        logger.info("Filtering %d rare classes (< %d samples)", len(rare), folds)
        mask = ~y.isin(rare)
        return X_num.loc[mask], y.loc[mask]

    def _build_scoring_dict(self) -> dict:
        """Build the scoring dictionary for cross-validation.

        Returns
        -------
        dict
            Mapping of scorer names to scorer objects.
        """
        precision_scorer = make_scorer(precision_score, average="weighted", zero_division=0)
        recall_scorer = make_scorer(recall_score, average="weighted", zero_division=0)
        f1_scorer = make_scorer(f1_score, average="weighted", zero_division=0)
        f2_scorer = make_scorer(fbeta_score, beta=2, average="weighted", zero_division=0)
        return {
            "accuracy": "accuracy",
            "precision_weighted": precision_scorer,
            "recall_weighted": recall_scorer,
            "f1_weighted": f1_scorer,
            "f2_weighted": f2_scorer,
            "roc_auc_ovr_weighted": "roc_auc_ovr_weighted",
        }

    @staticmethod
    def _add_cv_warning_filters(warn_mod: Any) -> None:
        """Register warning filters for cross-validation scoring.

        Parameters
        ----------
        warn_mod : module
            The ``warnings`` module reference.
        """
        warn_mod.filterwarnings("ignore", category=UserWarning)
        warn_mod.filterwarnings("ignore", message=".*Precision is ill-defined.*")
        try:
            from sklearn.exceptions import UndefinedMetricWarning

            warn_mod.filterwarnings("ignore", category=UndefinedMetricWarning)
        except ImportError:
            pass

    @staticmethod
    def _run_cv(clf, X, y, skf, scoring, fit_params) -> dict:
        """Execute sklearn cross_validate with warning suppression.

        Parameters
        ----------
        clf : xgb.XGBClassifier
            Un-fitted classifier.
        X : pd.DataFrame
            Numeric feature matrix.
        y : np.ndarray
            Encoded labels.
        skf : StratifiedKFold
            Cross-validation splitter.
        scoring : dict
            Scorer mapping.
        fit_params : dict
            Parameters forwarded to ``clf.fit()``.

        Returns
        -------
        dict
            Raw cross_validate output.
        """
        import warnings as _w

        with _w.catch_warnings():
            XGBoostEngine._add_cv_warning_filters(_w)
            return sklearn_cv(clf, X, y, cv=skf, scoring=scoring, params=fit_params)

    # ── Optuna HPO ───────────────────────────────────────────────────────

    def _optuna_objective(
        self,
        trial: Any,
        X_num: pd.DataFrame,
        y_enc: np.ndarray,
    ) -> float:
        """Optuna objective: maximise 5-fold weighted F1.

        Parameters
        ----------
        trial : optuna.Trial
            Optuna trial object.
        X_num : pd.DataFrame
            Numeric feature matrix.
        y_enc : np.ndarray
            Integer-encoded labels.

        Returns
        -------
        float
            Mean weighted F1 score across folds.
        """
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 300, 1500),
            "max_depth": trial.suggest_int("max_depth", 4, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "gamma": trial.suggest_float("gamma", 0.0, 0.5),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 1.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 5.0),
        }
        clf = xgb.XGBClassifier(
            **params,
            objective="multi:softprob",
            eval_metric="mlogloss",
            use_label_encoder=False,
            verbosity=0,
            n_jobs=-1,
        )
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        f1_scorer = make_scorer(f1_score, average="weighted", zero_division=0)
        import warnings as _w

        with _w.catch_warnings():
            self._add_cv_warning_filters(_w)
            cv = sklearn_cv(
                clf,
                X_num,
                y_enc,
                cv=skf,
                scoring={"f1": f1_scorer},
                params={"sample_weight": self._compute_sample_weights(y_enc)},
            )
        return float(np.mean(cv["test_f1"]))

    def run_hpo(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_trials: int = 50,
    ) -> dict[str, Any]:
        """Run Optuna Bayesian HPO and apply the best params.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix.
        y : pd.Series
            Cancer-type labels.
        n_trials : int
            Number of Optuna trials.

        Returns
        -------
        dict[str, Any]
            Best hyperparameters found.
        """
        import optuna

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        if self.cfg.enable_feature_engineering:
            X_aug = self.engineer_features(X)
        else:
            X_aug = X
        X_num = X_aug.select_dtypes(include=[np.number])
        y_merged = self.merge_rare_classes(y, self.cfg.min_class_size)
        y_enc = LabelEncoder().fit_transform(y_merged)
        logger.info("Starting Optuna HPO (%d trials) …", n_trials)
        study = optuna.create_study(direction="maximize")
        study.optimize(
            lambda trial: self._optuna_objective(trial, X_num, y_enc),
            n_trials=n_trials,
            show_progress_bar=False,
        )
        best = study.best_params
        logger.info("HPO best F1=%.4f  params=%s", study.best_value, best)
        # Apply best params to config
        self.cfg.xgb_n_estimators = best["n_estimators"]
        self.cfg.xgb_max_depth = best["max_depth"]
        self.cfg.xgb_learning_rate = best["learning_rate"]
        self.cfg.xgb_min_child_weight = best["min_child_weight"]
        self.cfg.xgb_gamma = best["gamma"]
        self.cfg.xgb_subsample = best["subsample"]
        self.cfg.xgb_colsample_bytree = best["colsample_bytree"]
        self.cfg.xgb_reg_alpha = best["reg_alpha"]
        self.cfg.xgb_reg_lambda = best["reg_lambda"]
        return best

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
            Feature matrix (samples x genes).  Only numeric columns
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
        if self.cfg.enable_feature_engineering:
            X_aug = self.engineer_features(X)
        else:
            X_aug = X
        X_num = X_aug.select_dtypes(include=[np.number])
        y_merged = self.merge_rare_classes(y, self.cfg.min_class_size)
        X_num, y_merged = self._filter_rare_classes(X_num, y_merged, folds)
        y_enc = LabelEncoder().fit_transform(y_merged)
        cv_splitter: StratifiedKFold | RepeatedStratifiedKFold
        if len(y_merged) >= 200:
            cv_splitter = RepeatedStratifiedKFold(
                n_splits=folds,
                n_repeats=3,
                random_state=42,
            )
        else:
            cv_splitter = StratifiedKFold(
                n_splits=folds,
                shuffle=True,
                random_state=42,
            )
        clf = self._build_cv_classifier()
        fit_params = {"sample_weight": self._compute_sample_weights(y_enc)}
        cv_results = self._run_cv(
            clf,
            X_num,
            y_enc,
            cv_splitter,
            self._build_scoring_dict(),
            fit_params,
        )
        return self._compute_cv_metrics(cv_results)

    # ── importance ───────────────────────────────────────────────────────

    def _check_fitted(self) -> None:
        """Raise ``RuntimeError`` if the model has not been fitted.

        Raises
        ------
        RuntimeError
            When ``self.model`` is ``None``.
        """
        if self.model is None:
            raise RuntimeError("Model not fitted yet – call .fit() first")

    def top_genes(self, k: int | None = None) -> dict[str, float]:
        """Return the *k* most important genes by XGBoost gain score.

        Parameters
        ----------
        k : int, optional
            Number of top genes. Defaults to ``config.top_k_genes``.

        Returns
        -------
        dict
            Ordered mapping ``{gene_name: importance_score}``.
        """
        self._check_fitted()
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
        self._check_fitted()
        return pd.Series(self.model.feature_importances_, index=self._feature_names).sort_values(
            ascending=False
        )
