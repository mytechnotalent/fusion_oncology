"""
Bayesian uncertainty quantification for prediction confidence.

Provides posterior uncertainty estimates, credible intervals, and
calibrated decision confidence for clinical predictions.  Clinicians
care deeply about *how sure* a model is, not just *what* it predicts.

Key components
--------------
* **BootstrapEnsemble** — trains ``n_boot`` XGBoost replicates on
  bootstrap resamples and aggregates predictions with full
  posterior distributions.
* **BayesianPredictor** — wraps a trained ensemble to produce
  per-sample credible intervals, entropy-based confidence, and
  calibrated decision thresholds.
* **CalibrationCurve** — reliability-diagram data for visual
  calibration assessment.

References
----------
Efron, B. "Bootstrap methods: another look at the jackknife."
*Annals of Statistics* 7 (1979).
Gal, Y. & Ghahramani, Z. "Dropout as a Bayesian Approximation."
*ICML* (2016).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

from fusion_oncology.config import ProjectConfig

logger = logging.getLogger(__name__)


# ── Configuration ────────────────────────────────────────────────────────


@dataclass
class UncertaintyConfig:
    """Parameters for Bayesian uncertainty estimation.

    Parameters
    ----------
    n_bootstrap : int
        Number of bootstrap ensemble members.
    bootstrap_fraction : float
        Fraction of data to sample per bootstrap replicate.
    confidence_level : float
        Credible interval coverage (e.g. 0.95 for 95% CI).
    entropy_threshold : float
        Normalised entropy below which prediction is "confident".
    n_estimators : int
        Boosting rounds per ensemble member.
    max_depth : int
        Tree depth per ensemble member.
    seed : int
        Master random seed.
    """

    n_bootstrap: int = 20
    bootstrap_fraction: float = 0.8
    confidence_level: float = 0.95
    entropy_threshold: float = 0.5
    n_estimators: int = 200
    max_depth: int = 4
    seed: int = 42


# ── Bootstrap ensemble ──────────────────────────────────────────────────


class BootstrapEnsemble:
    """Bootstrap-aggregated XGBoost ensemble for posterior estimation.

    Trains ``n_bootstrap`` classifiers on random resamples of the
    training data.  Prediction variance across ensemble members
    captures epistemic uncertainty.

    Parameters
    ----------
    uq_config : UncertaintyConfig, optional
        Uncertainty estimation parameters.
    project_config : ProjectConfig, optional
        Runtime configuration.
    """

    def __init__(
        self,
        uq_config: UncertaintyConfig | None = None,
        project_config: ProjectConfig | None = None,
    ) -> None:
        self.uq_cfg = uq_config or UncertaintyConfig()
        self.cfg = project_config or ProjectConfig()
        self._models: list[xgb.XGBClassifier] = []
        self._label_encoder = LabelEncoder()
        self._classes: np.ndarray = np.array([])
        self._feature_names: list[str] = []

    @property
    def n_models(self) -> int:
        """Return number of trained ensemble members.

        Returns
        -------
        int
            Ensemble size.
        """
        return len(self._models)

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> None:
        """Train the bootstrap ensemble.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix.
        y : pd.Series
            Class labels.
        """
        self._feature_names = list(X.columns)
        y_enc = self._label_encoder.fit_transform(y)
        self._classes = self._label_encoder.classes_
        n_classes = len(self._classes)
        n = len(X)
        sample_size = max(1, int(n * self.uq_cfg.bootstrap_fraction))
        rng = np.random.default_rng(self.uq_cfg.seed)

        self._models = []
        for i in range(self.uq_cfg.n_bootstrap):
            # Bootstrap resample
            idx = rng.choice(n, size=sample_size, replace=True)
            X_boot = X.iloc[idx].reset_index(drop=True)
            y_boot = y_enc[idx]

            # Ensure all classes represented (add one sample each if missing)
            unique_boot = set(y_boot)
            for cls in range(n_classes):
                if cls not in unique_boot:
                    cls_idx = np.where(y_enc == cls)[0]
                    if len(cls_idx) > 0:
                        pick = rng.choice(cls_idx)
                        X_boot = pd.concat(
                            [X_boot, X.iloc[[pick]].reset_index(drop=True)],
                            ignore_index=True,
                        )
                        y_boot = np.append(y_boot, cls)

            model = xgb.XGBClassifier(
                n_estimators=self.uq_cfg.n_estimators,
                max_depth=self.uq_cfg.max_depth,
                learning_rate=0.1,
                objective="multi:softprob",
                eval_metric="mlogloss",
                random_state=self.uq_cfg.seed + i,
                verbosity=0,
                use_label_encoder=False,
                num_class=n_classes,
            )
            model.fit(X_boot, y_boot)
            self._models.append(model)
            logger.debug("Trained ensemble member %d/%d", i + 1, self.uq_cfg.n_bootstrap)

        logger.info("Bootstrap ensemble trained: %d members", len(self._models))

    def predict_proba_ensemble(
        self,
        X: pd.DataFrame,
    ) -> np.ndarray:
        """Get probability predictions from all ensemble members.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix.

        Returns
        -------
        np.ndarray
            Shape ``(n_models, n_samples, n_classes)``.
        """
        n_classes = len(self._classes)
        all_probs = []
        for model in self._models:
            probs = model.predict_proba(X)
            # Ensure consistent shape
            if probs.shape[1] < n_classes:
                padded = np.zeros((probs.shape[0], n_classes))
                padded[:, : probs.shape[1]] = probs
                probs = padded
            all_probs.append(probs)
        return np.array(all_probs)

    def predict_with_uncertainty(
        self,
        X: pd.DataFrame,
    ) -> dict[str, Any]:
        """Predict with full uncertainty quantification.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix.

        Returns
        -------
        dict[str, Any]
            Keys: ``predictions``, ``mean_proba``, ``std_proba``,
            ``credible_lower``, ``credible_upper``, ``entropy``,
            ``confidence``, ``classes``.
        """
        ensemble_probs = self.predict_proba_ensemble(X)
        # ensemble_probs: (n_models, n_samples, n_classes)

        mean_proba = ensemble_probs.mean(axis=0)  # (n_samples, n_classes)
        std_proba = ensemble_probs.std(axis=0)

        # Credible intervals
        alpha = 1 - self.uq_cfg.confidence_level
        lower = np.percentile(ensemble_probs, 100 * alpha / 2, axis=0)
        upper = np.percentile(ensemble_probs, 100 * (1 - alpha / 2), axis=0)

        # Predictions
        predictions = self._label_encoder.inverse_transform(mean_proba.argmax(axis=1))

        # Normalised entropy
        eps = 1e-10
        entropy = -np.sum(mean_proba * np.log(mean_proba + eps), axis=1)
        max_entropy = np.log(len(self._classes))
        norm_entropy = entropy / max(max_entropy, eps)

        # Confidence = 1 - normalised entropy
        confidence = 1.0 - norm_entropy

        return {
            "predictions": predictions,
            "mean_proba": mean_proba,
            "std_proba": std_proba,
            "credible_lower": lower,
            "credible_upper": upper,
            "entropy": norm_entropy,
            "confidence": confidence,
            "classes": list(self._classes),
        }


# ── Bayesian predictor ──────────────────────────────────────────────────


class BayesianPredictor:
    """High-level Bayesian predictor with decision-support output.

    Wraps :class:`BootstrapEnsemble` and provides clinical-grade
    uncertainty reporting.

    Parameters
    ----------
    uq_config : UncertaintyConfig, optional
        Uncertainty parameters.
    project_config : ProjectConfig, optional
        Runtime configuration.
    """

    def __init__(
        self,
        uq_config: UncertaintyConfig | None = None,
        project_config: ProjectConfig | None = None,
    ) -> None:
        self.uq_cfg = uq_config or UncertaintyConfig()
        self.cfg = project_config or ProjectConfig()
        self.ensemble = BootstrapEnsemble(self.uq_cfg, self.cfg)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Train the Bayesian prediction model.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix.
        y : pd.Series
            Class labels.
        """
        self.ensemble.fit(X, y)

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """Predict with full uncertainty report.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix.

        Returns
        -------
        pd.DataFrame
            Columns: ``prediction``, ``confidence``, ``entropy``,
            ``top_class_prob``, ``top_class_std``, ``ci_lower``,
            ``ci_upper``, ``decision_quality``.
        """
        result = self.ensemble.predict_with_uncertainty(X)

        top_idx = result["mean_proba"].argmax(axis=1)
        n_samples = len(top_idx)

        rows: list[dict[str, Any]] = []
        for i in range(n_samples):
            cls_idx = top_idx[i]
            rows.append(
                {
                    "prediction": result["predictions"][i],
                    "confidence": round(float(result["confidence"][i]), 4),
                    "entropy": round(float(result["entropy"][i]), 4),
                    "top_class_prob": round(float(result["mean_proba"][i, cls_idx]), 4),
                    "top_class_std": round(float(result["std_proba"][i, cls_idx]), 4),
                    "ci_lower": round(float(result["credible_lower"][i, cls_idx]), 4),
                    "ci_upper": round(float(result["credible_upper"][i, cls_idx]), 4),
                    "decision_quality": _decision_quality(
                        float(result["confidence"][i]),
                        float(result["mean_proba"][i, cls_idx]),
                        self.uq_cfg.entropy_threshold,
                    ),
                }
            )

        return pd.DataFrame(rows)

    def credible_intervals(
        self,
        X: pd.DataFrame,
    ) -> pd.DataFrame:
        """Return credible intervals for all classes.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix.

        Returns
        -------
        pd.DataFrame
            Multi-level columns: ``(class, lower/mean/upper)`` per sample.
        """
        result = self.ensemble.predict_with_uncertainty(X)
        records: list[dict[str, float]] = []
        for i in range(len(X)):
            row: dict[str, float] = {}
            for j, cls in enumerate(result["classes"]):
                row[f"{cls}_lower"] = round(float(result["credible_lower"][i, j]), 4)
                row[f"{cls}_mean"] = round(float(result["mean_proba"][i, j]), 4)
                row[f"{cls}_upper"] = round(float(result["credible_upper"][i, j]), 4)
            records.append(row)
        return pd.DataFrame(records)

    def summary(self) -> dict[str, Any]:
        """Return a summary of the Bayesian predictor state.

        Returns
        -------
        dict[str, Any]
            Configuration and ensemble metadata.
        """
        return {
            "n_bootstrap": self.uq_cfg.n_bootstrap,
            "bootstrap_fraction": self.uq_cfg.bootstrap_fraction,
            "confidence_level": self.uq_cfg.confidence_level,
            "entropy_threshold": self.uq_cfg.entropy_threshold,
            "n_models_trained": self.ensemble.n_models,
            "classes": list(self.ensemble._classes),
        }


# ── Calibration ──────────────────────────────────────────────────────────


def calibration_curve(
    y_true: np.ndarray | pd.Series,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> pd.DataFrame:
    """Compute reliability-diagram calibration data.

    Parameters
    ----------
    y_true : array-like
        Binary ground truth (1 = positive).
    y_prob : np.ndarray
        Predicted probabilities for the positive class.
    n_bins : int
        Number of histogram bins.

    Returns
    -------
    pd.DataFrame
        Columns: ``bin_mid``, ``observed_freq``, ``mean_predicted``,
        ``count``.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    bins = np.linspace(0, 1, n_bins + 1)
    rows: list[dict[str, float]] = []

    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (y_prob >= lo) & (y_prob < hi)
        if mask.sum() == 0:
            continue
        rows.append(
            {
                "bin_mid": round((lo + hi) / 2, 3),
                "observed_freq": round(float(y_true[mask].mean()), 4),
                "mean_predicted": round(float(y_prob[mask].mean()), 4),
                "count": int(mask.sum()),
            }
        )

    return pd.DataFrame(rows)


def expected_calibration_error(
    y_true: np.ndarray | pd.Series,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Compute Expected Calibration Error (ECE).

    $$
    ECE = \\sum_{b=1}^{B} \\frac{|B_b|}{N} |\\bar{p}_b - \\bar{y}_b|
    $$

    Parameters
    ----------
    y_true : array-like
        Binary ground truth.
    y_prob : np.ndarray
        Predicted probabilities.
    n_bins : int
        Number of bins.

    Returns
    -------
    float
        ECE value in ``[0, 1]``.
    """
    cal = calibration_curve(y_true, y_prob, n_bins)
    if cal.empty:
        return 0.0
    total = cal["count"].sum()
    ece = sum(
        row["count"] / total * abs(row["observed_freq"] - row["mean_predicted"])
        for _, row in cal.iterrows()
    )
    return round(ece, 4)


# ── Helpers ──────────────────────────────────────────────────────────────


def _decision_quality(
    confidence: float,
    top_prob: float,
    entropy_threshold: float,
) -> str:
    """Classify decision quality for clinical decision support.

    Parameters
    ----------
    confidence : float
        Prediction confidence (1 − normalised entropy).
    top_prob : float
        Probability of the top predicted class.
    entropy_threshold : float
        Threshold beneath which confidence is "uncertain".

    Returns
    -------
    str
        One of ``"HIGH"``, ``"MODERATE"``, ``"LOW"``, ``"UNCERTAIN"``.
    """
    if confidence >= 0.8 and top_prob >= 0.7:
        return "HIGH"
    if confidence >= 0.6 and top_prob >= 0.5:
        return "MODERATE"
    if confidence >= entropy_threshold:
        return "LOW"
    return "UNCERTAIN"
