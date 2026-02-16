"""
Benchmark framework — baseline comparisons and ablation studies.

Provides a rigorous statistical comparison of the Fusion Oncology pipeline
against standard baselines (Random Forest, Logistic Regression, SVM,
plain XGBoost) and systematic ablation of each pipeline component to
quantify individual contributions.

All comparisons use repeated stratified k-fold cross-validation with
paired statistical tests (Wilcoxon signed-rank) for significance.

Example
-------
>>> suite = BenchmarkSuite(X, y, seed=42)
>>> report = suite.run_full_benchmark()
>>> print(report["comparison_table"])
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC

from fusion_oncology.config import ProjectConfig
from fusion_oncology.models.xgboost_engine import XGBoostEngine

logger = logging.getLogger(__name__)


# ── Configuration ────────────────────────────────────────────────────────


@dataclass
class BenchmarkConfig:
    """Configuration for the benchmark suite.

    Parameters
    ----------
    n_folds : int
        Number of CV folds.
    n_repeats : int
        Number of CV repeats (for repeated stratified k-fold).
    significance_level : float
        Alpha for statistical tests.
    seed : int
        Random seed.
    """

    n_folds: int = 5
    n_repeats: int = 3
    significance_level: float = 0.05
    seed: int = 42


# ── Cross-validated model evaluator ──────────────────────────────────────


def _evaluate_model_cv(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    cv: Any,
    model_name: str = "",
) -> dict[str, Any]:
    """Evaluate a sklearn-compatible model via cross-validation.

    Returns per-fold accuracy, F1, precision, recall, and ROC-AUC.

    Parameters
    ----------
    model : sklearn estimator
        Must implement ``fit`` / ``predict`` / ``predict_proba``.
    X : np.ndarray
        Feature matrix.
    y : np.ndarray
        Integer-encoded labels.
    cv : cross-validation splitter
        Scikit-learn CV object.
    model_name : str
        Human-readable model name.

    Returns
    -------
    dict
        ``fold_accuracies``, ``fold_f1s``, ``fold_precisions``,
        ``fold_recalls``, ``fold_aucs``, and aggregate statistics.
    """
    fold_acc: list[float] = []
    fold_f1: list[float] = []
    fold_prec: list[float] = []
    fold_rec: list[float] = []
    fold_auc: list[float] = []

    for train_idx, test_idx in cv.split(X, y):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model_clone = _clone_model(model)
            model_clone.fit(X_tr, y_tr)
            y_pred = model_clone.predict(X_te)

            fold_acc.append(accuracy_score(y_te, y_pred))
            fold_f1.append(f1_score(y_te, y_pred, average="weighted", zero_division=0))
            fold_prec.append(precision_score(y_te, y_pred, average="weighted", zero_division=0))
            fold_rec.append(recall_score(y_te, y_pred, average="weighted", zero_division=0))

            try:
                if hasattr(model_clone, "predict_proba"):
                    y_prob = model_clone.predict_proba(X_te)
                    n_cls = len(np.unique(y))
                    if n_cls == 2:
                        # Binary: use probability of positive class
                        y_score = y_prob[:, 1] if y_prob.ndim == 2 else y_prob
                        auc = roc_auc_score(y_te, y_score)
                    else:
                        auc = roc_auc_score(
                            y_te,
                            y_prob,
                            multi_class="ovr",
                            average="weighted",
                        )
                else:
                    auc = float("nan")
            except (ValueError, TypeError):
                auc = float("nan")
            fold_auc.append(auc)

    return {
        "model": model_name,
        "fold_accuracies": np.array(fold_acc),
        "fold_f1s": np.array(fold_f1),
        "fold_precisions": np.array(fold_prec),
        "fold_recalls": np.array(fold_rec),
        "fold_aucs": np.array(fold_auc),
        "mean_accuracy": float(np.mean(fold_acc)),
        "std_accuracy": float(np.std(fold_acc)),
        "mean_f1": float(np.mean(fold_f1)),
        "std_f1": float(np.std(fold_f1)),
        "mean_precision": float(np.mean(fold_prec)),
        "std_precision": float(np.std(fold_prec)),
        "mean_recall": float(np.mean(fold_rec)),
        "std_recall": float(np.std(fold_rec)),
        "mean_auc": float(np.nanmean(fold_auc)),
        "std_auc": float(np.nanstd(fold_auc)),
    }


def _clone_model(model: Any) -> Any:
    """Clone a model by re-instantiating from its parameters.

    Parameters
    ----------
    model : sklearn estimator

    Returns
    -------
    sklearn estimator
        Fresh instance with same hyperparameters.
    """
    from sklearn.base import clone

    return clone(model)


# ── Baseline model factory ───────────────────────────────────────────────


def build_baselines(seed: int = 42, n_classes: int = 3) -> dict[str, Any]:
    """Build standard baseline models for comparison.

    Parameters
    ----------
    seed : int
        Random seed.
    n_classes : int
        Number of target classes (forwarded to XGBoost builder).

    Returns
    -------
    dict[str, sklearn estimator]
        Named baseline models.
    """
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=2000,
            solver="lbfgs",
            C=1.0,
            random_state=seed,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            min_samples_leaf=5,
            random_state=seed,
            n_jobs=-1,
        ),
        "SVM (RBF)": SVC(
            kernel="rbf",
            probability=True,
            gamma="scale",
            random_state=seed,
        ),
        "XGBoost (vanilla)": _build_vanilla_xgb(seed, n_classes),
    }


def _build_vanilla_xgb(seed: int, n_classes: int = 3) -> Any:
    """Build a plain XGBoost without Fusion Oncology feature engineering.

    Parameters
    ----------
    seed : int
        Random seed.
    n_classes : int
        Number of target classes.  When 2, uses ``binary:logistic``
        to avoid the XGBoost >=3.2 ``num_class`` bug.
    """
    import xgboost as xgb

    if n_classes == 2:
        return xgb.XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="binary:logistic",
            eval_metric="logloss",
            use_label_encoder=False,
            verbosity=0,
            random_state=seed,
            n_jobs=-1,
        )
    return xgb.XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multi:softprob",
        eval_metric="mlogloss",
        use_label_encoder=False,
        verbosity=0,
        random_state=seed,
        n_jobs=-1,
    )


# ── Full Pipeline model (XGBoost + feature engineering) ──────────────────


class PipelineModel:
    """Wraps the Fusion Oncology XGBoost engine as an sklearn estimator.

    This model applies the full preprocessing and feature engineering
    pipeline (row-level statistics, class merging, sample weighting)
    before XGBoost training — representing the complete production
    system for fair comparison against baselines.

    Parameters
    ----------
    config : ProjectConfig | None
        Pipeline configuration.
    """

    def __init__(self, config: ProjectConfig | None = None) -> None:
        self.config = config or ProjectConfig(
            xgb_n_estimators=200,
            xgb_max_depth=6,
            enable_feature_engineering=True,
        )
        self._engine: XGBoostEngine | None = None
        self._le = LabelEncoder()
        self.classes_: np.ndarray = np.array([])

    def fit(self, X: np.ndarray, y: np.ndarray) -> "PipelineModel":
        """Fit the full pipeline.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix.
        y : np.ndarray
            Integer-encoded labels.

        Returns
        -------
        self
        """
        X_df = pd.DataFrame(X, columns=[f"G{i}" for i in range(X.shape[1])])
        y_s = pd.Series(y, name="label")
        self._engine = XGBoostEngine(config=self.config)
        self._engine.fit(X_df, y_s)
        self.classes_ = np.unique(y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict labels.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix.

        Returns
        -------
        np.ndarray
            Predicted labels (integer-encoded, matching input to fit).
        """
        assert self._engine is not None and self._engine.model is not None
        X_df = pd.DataFrame(X, columns=[f"G{i}" for i in range(X.shape[1])])
        if self.config.enable_feature_engineering:
            X_df = XGBoostEngine.engineer_features(X_df)
        X_num = X_df.select_dtypes(include=[np.number])
        raw = self._engine.model.predict(X_num).astype(int)
        return self._engine.label_encoder.inverse_transform(raw)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix.

        Returns
        -------
        np.ndarray
            Probability matrix (n_samples, n_classes).
        """
        assert self._engine is not None and self._engine.model is not None
        X_df = pd.DataFrame(X, columns=[f"G{i}" for i in range(X.shape[1])])
        if self.config.enable_feature_engineering:
            X_df = XGBoostEngine.engineer_features(X_df)
        X_num = X_df.select_dtypes(include=[np.number])
        return self._engine.model.predict_proba(X_num)

    def get_params(self, deep: bool = True) -> dict:
        """Sklearn-compatible get_params."""
        return {"config": self.config}

    def set_params(self, **params: Any) -> "PipelineModel":
        """Sklearn-compatible set_params."""
        if "config" in params:
            self.config = params["config"]
        return self


# ── Ablation study ───────────────────────────────────────────────────────


class AblationStudy:
    """Systematic component removal to quantify individual contributions.

    Compares the full pipeline against ablated variants where one
    component at a time is disabled:

    1. **No feature engineering** — remove row-level statistical features.
    2. **No class merging** — keep all rare classes.
    3. **No sample weighting** — uniform weights.
    4. **Shallow trees** — max_depth=2 instead of production depth.
    5. **Fewer estimators** — 50 trees instead of production count.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series
        Labels.
    config : BenchmarkConfig | None
        Benchmark configuration.
    project_config : ProjectConfig | None
        Project configuration.
    """

    def __init__(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        config: BenchmarkConfig | None = None,
        project_config: ProjectConfig | None = None,
    ) -> None:
        self.X = X
        self.y = y
        self.cfg = config or BenchmarkConfig()
        self.pcfg = project_config or ProjectConfig()
        self._le = LabelEncoder()

    def _prepare_data(self) -> tuple[np.ndarray, np.ndarray]:
        """Encode labels and convert to arrays."""
        X_num = self.X.select_dtypes(include=[np.number]).values
        y_enc = self._le.fit_transform(self.y)
        return X_num, y_enc

    def _build_cv(self) -> RepeatedStratifiedKFold:
        return RepeatedStratifiedKFold(
            n_splits=self.cfg.n_folds,
            n_repeats=self.cfg.n_repeats,
            random_state=self.cfg.seed,
        )

    def run(self) -> pd.DataFrame:
        """Run the ablation study.

        Returns
        -------
        pd.DataFrame
            Columns: ``variant``, ``mean_accuracy``, ``std_accuracy``,
            ``mean_f1``, ``std_f1``, ``delta_accuracy``, ``delta_f1``,
            ``p_value``.
        """
        X_arr, y_arr = self._prepare_data()
        cv = self._build_cv()

        # Full pipeline
        full_cfg = ProjectConfig(
            xgb_n_estimators=200,
            xgb_max_depth=6,
            enable_feature_engineering=True,
            min_class_size=self.pcfg.min_class_size,
        )
        full_model = PipelineModel(config=full_cfg)
        full_result = _evaluate_model_cv(full_model, X_arr, y_arr, cv, "Full Pipeline")

        variants: list[dict[str, Any]] = [
            {
                "name": "Full Pipeline",
                "config": full_cfg,
            },
            {
                "name": "No Feature Engineering",
                "config": ProjectConfig(
                    xgb_n_estimators=200,
                    xgb_max_depth=6,
                    enable_feature_engineering=False,
                    min_class_size=self.pcfg.min_class_size,
                ),
            },
            {
                "name": "Shallow Trees (depth=2)",
                "config": ProjectConfig(
                    xgb_n_estimators=200,
                    xgb_max_depth=2,
                    enable_feature_engineering=True,
                    min_class_size=self.pcfg.min_class_size,
                ),
            },
            {
                "name": "Few Estimators (n=50)",
                "config": ProjectConfig(
                    xgb_n_estimators=50,
                    xgb_max_depth=6,
                    enable_feature_engineering=True,
                    min_class_size=self.pcfg.min_class_size,
                ),
            },
            {
                "name": "Minimal Config (depth=2, n=50, no FE)",
                "config": ProjectConfig(
                    xgb_n_estimators=50,
                    xgb_max_depth=2,
                    enable_feature_engineering=False,
                    min_class_size=self.pcfg.min_class_size,
                ),
            },
        ]

        rows: list[dict[str, Any]] = []
        for v in variants:
            model = PipelineModel(config=v["config"])
            result = _evaluate_model_cv(model, X_arr, y_arr, cv, v["name"])

            # Statistical test vs full pipeline
            if v["name"] == "Full Pipeline":
                p_val = 1.0
            else:
                try:
                    _, p_val = stats.wilcoxon(
                        full_result["fold_f1s"],
                        result["fold_f1s"],
                        alternative="greater",
                    )
                except ValueError:
                    p_val = 1.0

            rows.append(
                {
                    "variant": v["name"],
                    "mean_accuracy": result["mean_accuracy"],
                    "std_accuracy": result["std_accuracy"],
                    "mean_f1": result["mean_f1"],
                    "std_f1": result["std_f1"],
                    "delta_accuracy": result["mean_accuracy"] - full_result["mean_accuracy"],
                    "delta_f1": result["mean_f1"] - full_result["mean_f1"],
                    "p_value": float(p_val),
                }
            )

        df = pd.DataFrame(rows)
        logger.info("Ablation study complete: %d variants evaluated", len(rows))
        return df


# ── Main benchmark suite ────────────────────────────────────────────────


class BenchmarkSuite:
    """Full benchmark suite: baselines + ablation + statistical tests.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series
        Labels.
    config : BenchmarkConfig | None
        Benchmark configuration.
    project_config : ProjectConfig | None
        Pipeline configuration.
    """

    def __init__(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        config: BenchmarkConfig | None = None,
        project_config: ProjectConfig | None = None,
    ) -> None:
        self.X = X
        self.y = y
        self.cfg = config or BenchmarkConfig()
        self.pcfg = project_config or ProjectConfig()
        self._le = LabelEncoder()

    def _prepare(self) -> tuple[np.ndarray, np.ndarray]:
        X_num = self.X.select_dtypes(include=[np.number]).values
        y_enc = self._le.fit_transform(self.y)
        return X_num, y_enc

    def _build_cv(self) -> RepeatedStratifiedKFold:
        return RepeatedStratifiedKFold(
            n_splits=self.cfg.n_folds,
            n_repeats=self.cfg.n_repeats,
            random_state=self.cfg.seed,
        )

    # ── baseline comparison ──────────────────────────────────────────

    def baseline_comparison(self) -> pd.DataFrame:
        """Compare the full pipeline against standard baselines.

        Returns
        -------
        pd.DataFrame
            Comparative performance table with statistical tests.
        """
        X_arr, y_arr = self._prepare()
        cv = self._build_cv()

        # Full pipeline
        full_cfg = ProjectConfig(
            xgb_n_estimators=200,
            xgb_max_depth=6,
            enable_feature_engineering=True,
        )
        pipeline = PipelineModel(config=full_cfg)
        pipeline_result = _evaluate_model_cv(pipeline, X_arr, y_arr, cv, "Fusion Pipeline")

        n_classes = len(np.unique(y_arr))
        baselines = build_baselines(self.cfg.seed, n_classes=n_classes)
        results: list[dict[str, Any]] = []

        # Pipeline first
        results.append(
            {
                "model": "Fusion Pipeline",
                "mean_accuracy": pipeline_result["mean_accuracy"],
                "std_accuracy": pipeline_result["std_accuracy"],
                "mean_f1": pipeline_result["mean_f1"],
                "std_f1": pipeline_result["std_f1"],
                "mean_precision": pipeline_result["mean_precision"],
                "mean_recall": pipeline_result["mean_recall"],
                "mean_auc": pipeline_result["mean_auc"],
                "improvement_accuracy": 0.0,
                "improvement_f1": 0.0,
                "p_value_accuracy": 1.0,
                "p_value_f1": 1.0,
                "significant": False,
            }
        )

        for name, model in baselines.items():
            result = _evaluate_model_cv(model, X_arr, y_arr, cv, name)

            # Paired Wilcoxon test (pipeline > baseline)
            try:
                _, p_acc = stats.wilcoxon(
                    pipeline_result["fold_accuracies"],
                    result["fold_accuracies"],
                    alternative="greater",
                )
            except ValueError:
                p_acc = 1.0

            try:
                _, p_f1 = stats.wilcoxon(
                    pipeline_result["fold_f1s"],
                    result["fold_f1s"],
                    alternative="greater",
                )
            except ValueError:
                p_f1 = 1.0

            imp_acc = pipeline_result["mean_accuracy"] - result["mean_accuracy"]
            imp_f1 = pipeline_result["mean_f1"] - result["mean_f1"]

            results.append(
                {
                    "model": name,
                    "mean_accuracy": result["mean_accuracy"],
                    "std_accuracy": result["std_accuracy"],
                    "mean_f1": result["mean_f1"],
                    "std_f1": result["std_f1"],
                    "mean_precision": result["mean_precision"],
                    "mean_recall": result["mean_recall"],
                    "mean_auc": result["mean_auc"],
                    "improvement_accuracy": imp_acc,
                    "improvement_f1": imp_f1,
                    "p_value_accuracy": float(p_acc),
                    "p_value_f1": float(p_f1),
                    "significant": p_f1 < self.cfg.significance_level,
                }
            )

        df = pd.DataFrame(results)
        logger.info(
            "Baseline comparison: pipeline acc=%.3f vs best baseline acc=%.3f",
            pipeline_result["mean_accuracy"],
            df.iloc[1:]["mean_accuracy"].max(),
        )
        return df

    # ── ablation ─────────────────────────────────────────────────────

    def ablation_study(self) -> pd.DataFrame:
        """Run ablation study (delegates to AblationStudy).

        Returns
        -------
        pd.DataFrame
            Ablation results.
        """
        study = AblationStudy(
            self.X,
            self.y,
            config=self.cfg,
            project_config=self.pcfg,
        )
        return study.run()

    # ── cross-dataset stability ──────────────────────────────────────

    def stability_analysis(
        self,
        n_subsamples: int = 5,
        subsample_fraction: float = 0.7,
    ) -> pd.DataFrame:
        """Assess pipeline stability across random data sub-samples.

        Parameters
        ----------
        n_subsamples : int
            Number of random sub-samples.
        subsample_fraction : float
            Fraction of data per sub-sample.

        Returns
        -------
        pd.DataFrame
            Per-subsample metrics for stability assessment.
        """
        X_arr, y_arr = self._prepare()
        rng = np.random.default_rng(self.cfg.seed)
        cv = StratifiedKFold(n_splits=self.cfg.n_folds, shuffle=True, random_state=self.cfg.seed)
        rows: list[dict[str, float]] = []

        for i in range(n_subsamples):
            n = len(y_arr)
            k = int(n * subsample_fraction)
            idx = rng.choice(n, size=k, replace=False)
            X_sub = X_arr[idx]
            y_sub = y_arr[idx]

            pipeline = PipelineModel(
                config=ProjectConfig(
                    xgb_n_estimators=200,
                    xgb_max_depth=6,
                    enable_feature_engineering=True,
                )
            )
            result = _evaluate_model_cv(pipeline, X_sub, y_sub, cv, f"Subsample-{i}")
            rows.append(
                {
                    "subsample": i,
                    "n_samples": k,
                    "mean_accuracy": result["mean_accuracy"],
                    "mean_f1": result["mean_f1"],
                    "mean_auc": result["mean_auc"],
                }
            )

        df = pd.DataFrame(rows)
        logger.info(
            "Stability: acc CV=%.4f (mean=%.3f)",
            df["mean_accuracy"].std() / max(df["mean_accuracy"].mean(), 1e-10),
            df["mean_accuracy"].mean(),
        )
        return df

    # ── full report ──────────────────────────────────────────────────

    def run_full_benchmark(self) -> dict[str, Any]:
        """Run the complete benchmark suite.

        Returns
        -------
        dict
            ``comparison_table``, ``ablation_table``,
            ``stability_table``, ``summary``.
        """
        logger.info("Starting full benchmark suite …")
        comparison = self.baseline_comparison()
        ablation = self.ablation_study()
        stability = self.stability_analysis()

        # Build summary
        pipeline_row = comparison[comparison["model"] == "Fusion Pipeline"].iloc[0]
        best_baseline_row = (
            comparison.iloc[1:].sort_values("mean_accuracy", ascending=False).iloc[0]
        )

        n_significant = int(comparison.iloc[1:]["significant"].sum())

        summary: dict[str, Any] = {
            "pipeline_accuracy": float(pipeline_row["mean_accuracy"]),
            "pipeline_f1": float(pipeline_row["mean_f1"]),
            "best_baseline": str(best_baseline_row["model"]),
            "best_baseline_accuracy": float(best_baseline_row["mean_accuracy"]),
            "best_baseline_f1": float(best_baseline_row["mean_f1"]),
            "accuracy_improvement": float(
                pipeline_row["mean_accuracy"] - best_baseline_row["mean_accuracy"]
            ),
            "f1_improvement": float(pipeline_row["mean_f1"] - best_baseline_row["mean_f1"]),
            "n_baselines_beaten_significantly": n_significant,
            "stability_cv": float(
                stability["mean_accuracy"].std() / max(stability["mean_accuracy"].mean(), 1e-10)
            ),
            "ablation_most_impactful": str(
                ablation.iloc[1:].sort_values("delta_f1").iloc[0]["variant"]
            ),
        }

        logger.info(
            "Benchmark complete: pipeline acc=%.3f vs best baseline=%s (%.3f)",
            summary["pipeline_accuracy"],
            summary["best_baseline"],
            summary["best_baseline_accuracy"],
        )

        return {
            "comparison_table": comparison,
            "ablation_table": ablation,
            "stability_table": stability,
            "summary": summary,
        }

    # ── formatted output ─────────────────────────────────────────────

    @staticmethod
    def format_report(results: dict[str, Any]) -> str:
        """Format benchmark results as a Markdown report.

        Parameters
        ----------
        results : dict
            Output from ``run_full_benchmark()``.

        Returns
        -------
        str
            Markdown-formatted report.
        """
        lines: list[str] = []
        lines.append("## Benchmark Results\n")

        # Summary
        s = results["summary"]
        lines.append("### Summary\n")
        lines.append(f"- **Pipeline accuracy**: {s['pipeline_accuracy']:.4f}")
        lines.append(f"- **Pipeline F1**: {s['pipeline_f1']:.4f}")
        lines.append(f"- **Best baseline**: {s['best_baseline']}")
        lines.append(f"- **Best baseline accuracy**: {s['best_baseline_accuracy']:.4f}")
        lines.append(f"- **Accuracy improvement**: {s['accuracy_improvement']:+.4f}")
        lines.append(f"- **F1 improvement**: {s['f1_improvement']:+.4f}")
        lines.append(
            f"- **Baselines beaten (p < 0.05)**: " f"{s['n_baselines_beaten_significantly']}/4"
        )
        lines.append(f"- **Stability (CV of accuracy)**: {s['stability_cv']:.4f}")
        lines.append(f"- **Most impactful ablation**: {s['ablation_most_impactful']}")

        # Comparison table
        lines.append("\n### Baseline Comparison\n")
        comp = results["comparison_table"]
        lines.append("| Model | Accuracy | F1 | AUC | Δ Acc | p-value |")
        lines.append("|-------|----------|-----|-----|-------|---------|")
        for _, row in comp.iterrows():
            sig = " *" if row.get("significant", False) else ""
            lines.append(
                f"| {row['model']} | {row['mean_accuracy']:.4f}±{row['std_accuracy']:.4f} "
                f"| {row['mean_f1']:.4f}±{row['std_f1']:.4f} "
                f"| {row['mean_auc']:.4f} "
                f"| {row['improvement_accuracy']:+.4f} "
                f"| {row['p_value_f1']:.4f}{sig} |"
            )

        # Ablation table
        lines.append("\n### Ablation Study\n")
        abl = results["ablation_table"]
        lines.append("| Variant | Accuracy | F1 | Δ F1 | p-value |")
        lines.append("|---------|----------|-----|------|---------|")
        for _, row in abl.iterrows():
            lines.append(
                f"| {row['variant']} | {row['mean_accuracy']:.4f}±{row['std_accuracy']:.4f} "
                f"| {row['mean_f1']:.4f}±{row['std_f1']:.4f} "
                f"| {row['delta_f1']:+.4f} "
                f"| {row['p_value']:.4f} |"
            )

        return "\n".join(lines)
