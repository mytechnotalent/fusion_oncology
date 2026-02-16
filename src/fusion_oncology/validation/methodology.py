"""
Methodology formalisation — architecture specification and analysis.

Provides formal mathematical descriptions and empirical validation of
the Fusion Oncology architecture:

1. **XGBoost-importance-weighted DNABERT-2 embedding fusion** — a novel
   feature-space coupling where gradient-boosted feature importances
   gate transformer embeddings, creating a biologically-informed
   attention mechanism.

2. **Hybrid ML-ODE coupling** — XGBoost predictions parameterise a
   Gompertzian ODE digital twin, enabling personalised treatment
   simulation that bridges data-driven inference and mechanistic
   modelling.

3. **RL-on-digital-twin optimisation** — REINFORCE policy gradient
   operating on the ODE simulator, closing the loop from genomics
   → prediction → simulation → treatment recommendation.

4. **Component contribution analysis** — quantifies the marginal
   information gain of each architectural layer.

Example
-------
>>> formaliser = MethodologyFormaliser(X, y)
>>> report = formaliser.full_analysis()
>>> print(report["architecture_specification"])
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.metrics import mutual_info_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

from fusion_oncology.config import ProjectConfig
from fusion_oncology.models.xgboost_engine import XGBoostEngine

logger = logging.getLogger(__name__)


# ── Architecture specification ───────────────────────────────────────────


ARCHITECTURE_SPEC: dict[str, str] = {
    "name": "Fusion Oncology Multi-Modal Architecture",
    "version": "1.0",
    "layers": """
Layer 1 — Feature Engineering (FE):
  Input:  X ∈ ℝ^{n×d}  (n samples, d genes)
  Output: X̃ ∈ ℝ^{n×(d+10)}
  Operation: X̃ = [X ∥ Φ(X)]
    where Φ(X) appends per-row μ, σ, skew, kurt, max, min, range, median, IQR, CV.
  Complexity: O(n·d)

Layer 2 — Importance-Weighted Classification (IWC):
  Input:  X̃ ∈ ℝ^{n×(d+10)}, y ∈ {1,...,C}^n
  Model:  f_θ = XGBoost(X̃, y; θ)  with balanced sample weights w_i = n/(C·n_c)
  Output: Importance vector ω ∈ ℝ^d  where ω_j = Gain(feature j)
  Cross-validation: RepeatedStratifiedKFold(k=5, r=3) with 12 metrics

Layer 3 — Sequence Embedding (SE):
  Input:  Top-K genes G = argsort(ω)[-K:]
  For each g ∈ G:
    s_g = fetch_refseq(g) ∈ {A,C,G,T}*
    e_g = DNABERT-2(s_g) ∈ ℝ^768
  Output: E = [e_{g1}, ..., e_{gK}] ∈ ℝ^{K×768}

Layer 4 — Fusion Feature Construction (FFC):
  Input:  X ∈ ℝ^{n×d}, E ∈ ℝ^{K×768}, ω ∈ ℝ^d
  For sample i:
    z_i = Σ_{g∈G} X_{i,g}·ω_g·e_g ∈ ℝ^768
  Fusion matrix: Z = [X ∥ z_1, ..., z_n] ∈ ℝ^{n×(d+768)}
  This is an importance-weighted attention over genomic embeddings.

Layer 5 — Fusion Classification (FC):
  Input:  Z ∈ ℝ^{n×(d+768)}, y ∈ {1,...,C}^n
  Model:  g_φ = XGBoost(Z, y; φ)
  Output: ŷ ∈ {1,...,C}^n, Fusion Index = ω·instability·1000

Layer 6 — Digital Twin Parameterisation (DTP):
  Input:  Per-gene importance ω, mutation profile M
  ODE system:
    dN/dt = r·N·ln(K/N) - δ(t)·N·(1 - R/N)
    dR/dt = μ·(N - R) - γ·R
  Where δ(t) = Σ_j D_j(t)·ε_j  (drug-specific kill rate from PK/PD)
  Parameters: r, K from tumour biology; ε_j from ML predictions

Layer 7 — RL Treatment Optimisation (RTO):
  Environment: Digital Twin ODE (Layer 6)
  State:  s_t = [N_t, R_t, drug_conc_t, time_t, response_t]
  Action: a_t ∈ {no-drug, low-dose, standard-dose, high-dose}
  Reward: r_t = -ΔN/N_0 - λ·dose_toxicity
  Policy: π_θ(a|s) via 2-layer MLP trained with REINFORCE
""",
    "novelty": """
The key methodological contribution is the IMPORTANCE-WEIGHTED EMBEDDING
FUSION (Layer 4): rather than simply concatenating XGBoost features with
DNABERT-2 embeddings, the fusion weights each gene's transformer embedding
by its gradient-boosted importance score AND the sample's expression
value for that gene. This creates a biologically-informed attention
mechanism where:
  - XGBoost importance acts as a learned "prior" on gene relevance
  - Expression values provide sample-specific "context"
  - Transformer embeddings capture sequence-level biological information
The result is a 768-dimensional "genomic context vector" per sample that
encodes both statistical and biological information.

The secondary contribution is the ML→ODE→RL closed-loop architecture:
  ML predictions → ODE parameters → RL environment → treatment policy
This bridges the gap between data-driven inference and mechanistic
simulation in a trainable end-to-end system.
""",
}


# ── Component contribution analysis ─────────────────────────────────────


@dataclass
class ComponentContribution:
    """Results from component contribution analysis.

    Attributes
    ----------
    component : str
        Component name.
    marginal_accuracy : float
        Accuracy improvement when adding this component.
    marginal_f1 : float
        F1 improvement when adding this component.
    information_gain : float
        Mutual information gain from this component's features.
    feature_utilisation : float
        Fraction of added features with non-zero importance.
    """

    component: str
    marginal_accuracy: float
    marginal_f1: float
    information_gain: float
    feature_utilisation: float


class MethodologyFormaliser:
    """Formal analysis of the Fusion Oncology architecture.

    Provides empirical evidence for each architectural claim:
    - Feature engineering improves accuracy
    - Importance weighting is better than uniform weighting
    - Component interactions are synergistic
    - Architecture is robust to hyperparameter perturbation

    Parameters
    ----------
    X : pd.DataFrame
        Gene expression features.
    y : pd.Series
        Labels (cancer types or survival).
    project_config : ProjectConfig | None
        Configuration.
    seed : int
        Random seed.
    """

    def __init__(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        project_config: ProjectConfig | None = None,
        seed: int = 42,
    ) -> None:
        self.X = X
        self.y = y
        self.pcfg = project_config or ProjectConfig()
        self.seed = seed
        self._le = LabelEncoder()

    # ── feature engineering contribution ─────────────────────────────

    def feature_engineering_contribution(self) -> ComponentContribution:
        """Quantify the contribution of row-level feature engineering.

        Compares XGBoost with and without the 10 engineered features
        (row mean, std, skew, kurtosis, max, min, range, median, IQR, CV).

        Returns
        -------
        ComponentContribution
        """
        X_raw = self.X.select_dtypes(include=[np.number])
        X_eng = XGBoostEngine.engineer_features(self.X)
        y_enc = self._le.fit_transform(self.y)

        acc_raw, f1_raw = self._quick_cv(X_raw.values, y_enc)
        acc_eng, f1_eng = self._quick_cv(X_eng.select_dtypes(include=[np.number]).values, y_enc)

        # Information gain from engineered features
        eng_cols = [c for c in X_eng.columns if c.startswith("_")]
        ig = self._mutual_info_cols(X_eng[eng_cols], y_enc) if eng_cols else 0.0

        # Feature utilisation
        engine = XGBoostEngine(
            config=ProjectConfig(
                xgb_n_estimators=100,
                xgb_max_depth=4,
                enable_feature_engineering=True,
            )
        )
        engine.fit(self.X, self.y)
        imps = engine.all_importances()
        eng_used = sum(1 for c in eng_cols if c in imps.index and imps[c] > 0)
        utilisation = eng_used / max(len(eng_cols), 1)

        return ComponentContribution(
            component="Feature Engineering (10 row-level statistics)",
            marginal_accuracy=acc_eng - acc_raw,
            marginal_f1=f1_eng - f1_raw,
            information_gain=ig,
            feature_utilisation=utilisation,
        )

    # ── importance weighting analysis ────────────────────────────────

    def importance_weighting_analysis(self) -> dict[str, Any]:
        """Compare importance-weighted vs uniform embedding fusion.

        Creates synthetic fusion features with:
        1. **Importance-weighted**: z_i = Σ (X_ig · ω_g · e_g)
        2. **Uniform-weighted**: z_i = Σ (X_ig · e_g)
        3. **Random-weighted**: z_i = Σ (X_ig · r_g · e_g)

        Returns
        -------
        dict
            ``weighted_accuracy``, ``uniform_accuracy``,
            ``random_accuracy``, ``improvement``.
        """
        rng = np.random.default_rng(self.seed)
        X_num = self.X.select_dtypes(include=[np.number])
        y_enc = self._le.fit_transform(self.y)

        # Train XGBoost for importances
        engine = XGBoostEngine(
            config=ProjectConfig(
                xgb_n_estimators=100,
                xgb_max_depth=4,
                enable_feature_engineering=False,
            )
        )
        engine.fit(self.X, self.y)
        importances = engine.all_importances()

        # Simulate embeddings (768-dim per gene, but use PCA projection
        # of gene expression as a proxy for DNABERT embeddings)
        embed_dim = min(32, X_num.shape[1])
        pca = PCA(n_components=embed_dim, random_state=self.seed)
        gene_embeddings = pca.fit_transform(X_num.T)  # genes × embed_dim

        # Importance-weighted fusion
        omega = np.array([importances.get(c, 0.0) for c in X_num.columns])
        omega = omega / max(omega.sum(), 1e-10)

        X_vals = X_num.values
        z_weighted = X_vals @ np.diag(omega) @ gene_embeddings
        X_fused_w = np.hstack([X_vals, z_weighted])

        # Uniform fusion
        uniform = np.ones(len(omega)) / len(omega)
        z_uniform = X_vals @ np.diag(uniform) @ gene_embeddings
        X_fused_u = np.hstack([X_vals, z_uniform])

        # Random fusion
        rand_w = rng.random(len(omega))
        rand_w /= rand_w.sum()
        z_random = X_vals @ np.diag(rand_w) @ gene_embeddings
        X_fused_r = np.hstack([X_vals, z_random])

        acc_w, f1_w = self._quick_cv(X_fused_w, y_enc)
        acc_u, f1_u = self._quick_cv(X_fused_u, y_enc)
        acc_r, f1_r = self._quick_cv(X_fused_r, y_enc)

        return {
            "weighted_accuracy": acc_w,
            "weighted_f1": f1_w,
            "uniform_accuracy": acc_u,
            "uniform_f1": f1_u,
            "random_accuracy": acc_r,
            "random_f1": f1_r,
            "improvement_over_uniform": acc_w - acc_u,
            "improvement_over_random": acc_w - acc_r,
        }

    # ── hyperparameter sensitivity ───────────────────────────────────

    def hyperparameter_sensitivity(self) -> pd.DataFrame:
        """Assess pipeline sensitivity to key hyperparameters.

        Sweeps XGBoost depth, estimator count, and learning rate
        around the production values to show the architecture is
        robust (not over-tuned to a single configuration).

        Returns
        -------
        pd.DataFrame
            Columns: ``parameter``, ``value``, ``accuracy``, ``f1``.
        """
        X_num = self.X.select_dtypes(include=[np.number]).values
        y_enc = self._le.fit_transform(self.y)
        rows: list[dict[str, Any]] = []

        # Depth sweep
        for depth in [2, 4, 6, 8]:
            cfg = ProjectConfig(
                xgb_n_estimators=200,
                xgb_max_depth=depth,
                enable_feature_engineering=True,
            )
            from fusion_oncology.validation.benchmark import PipelineModel as PM

            acc, f1 = self._quick_cv_model(PM(config=cfg), X_num, y_enc)
            rows.append(
                {
                    "parameter": "max_depth",
                    "value": depth,
                    "accuracy": acc,
                    "f1": f1,
                }
            )

        # Estimator count sweep
        for ne in [50, 100, 200, 500]:
            cfg = ProjectConfig(
                xgb_n_estimators=ne,
                xgb_max_depth=6,
                enable_feature_engineering=True,
            )
            from fusion_oncology.validation.benchmark import PipelineModel as PM

            acc, f1 = self._quick_cv_model(PM(config=cfg), X_num, y_enc)
            rows.append(
                {
                    "parameter": "n_estimators",
                    "value": ne,
                    "accuracy": acc,
                    "f1": f1,
                }
            )

        # Learning rate sweep
        for lr in [0.01, 0.05, 0.1, 0.3]:
            cfg = ProjectConfig(
                xgb_n_estimators=200,
                xgb_max_depth=6,
                xgb_learning_rate=lr,
                enable_feature_engineering=True,
            )
            from fusion_oncology.validation.benchmark import PipelineModel as PM

            acc, f1 = self._quick_cv_model(PM(config=cfg), X_num, y_enc)
            rows.append(
                {
                    "parameter": "learning_rate",
                    "value": lr,
                    "accuracy": acc,
                    "f1": f1,
                }
            )

        df = pd.DataFrame(rows)
        logger.info("Hyperparameter sensitivity: %d configurations tested", len(df))
        return df

    # ── synergy analysis ─────────────────────────────────────────────

    def synergy_analysis(self) -> dict[str, Any]:
        """Test whether pipeline components are synergistic.

        Synergy exists when the improvement from combining components
        exceeds the sum of their individual improvements.

        Tests:
        - A = Feature Engineering alone
        - B = Class Merging alone
        - A+B together vs predicted additive effect

        Returns
        -------
        dict
            ``improvement_A``, ``improvement_B``,
            ``improvement_AB``, ``predicted_additive``,
            ``synergy_score``, ``is_synergistic``.
        """
        X_num = self.X.select_dtypes(include=[np.number]).values
        y_enc = self._le.fit_transform(self.y)

        # Baseline: no features, no engineering
        from fusion_oncology.validation.benchmark import PipelineModel as PM

        base_cfg = ProjectConfig(
            xgb_n_estimators=200,
            xgb_max_depth=6,
            enable_feature_engineering=False,
            min_class_size=1,
        )
        acc_base, _ = self._quick_cv_model(PM(config=base_cfg), X_num, y_enc)

        # A: Feature engineering only
        a_cfg = ProjectConfig(
            xgb_n_estimators=200,
            xgb_max_depth=6,
            enable_feature_engineering=True,
            min_class_size=1,
        )
        acc_a, _ = self._quick_cv_model(PM(config=a_cfg), X_num, y_enc)
        imp_a = acc_a - acc_base

        # B: Deep trees only (depth=8 vs 6)
        b_cfg = ProjectConfig(
            xgb_n_estimators=200,
            xgb_max_depth=8,
            enable_feature_engineering=False,
            min_class_size=1,
        )
        acc_b, _ = self._quick_cv_model(PM(config=b_cfg), X_num, y_enc)
        imp_b = acc_b - acc_base

        # A+B: Both
        ab_cfg = ProjectConfig(
            xgb_n_estimators=200,
            xgb_max_depth=8,
            enable_feature_engineering=True,
            min_class_size=1,
        )
        acc_ab, _ = self._quick_cv_model(PM(config=ab_cfg), X_num, y_enc)
        imp_ab = acc_ab - acc_base

        predicted_additive = imp_a + imp_b
        synergy = imp_ab - predicted_additive

        return {
            "baseline_accuracy": acc_base,
            "improvement_A_feature_eng": imp_a,
            "improvement_B_deeper_trees": imp_b,
            "improvement_AB_combined": imp_ab,
            "predicted_additive": predicted_additive,
            "synergy_score": synergy,
            "is_synergistic": synergy > 0,
        }

    # ── architecture specification ───────────────────────────────────

    @staticmethod
    def architecture_specification() -> dict[str, str]:
        """Return the formal architecture specification.

        Returns
        -------
        dict[str, str]
            Keys: ``name``, ``version``, ``layers``, ``novelty``.
        """
        return ARCHITECTURE_SPEC.copy()

    # ── full analysis ────────────────────────────────────────────────

    def full_analysis(self) -> dict[str, Any]:
        """Run complete methodology formalisation.

        Returns
        -------
        dict
            ``architecture_specification``, ``feature_engineering``,
            ``importance_weighting``, ``hyperparameter_sensitivity``,
            ``synergy``.
        """
        logger.info("Running full methodology analysis …")

        fe = self.feature_engineering_contribution()
        iw = self.importance_weighting_analysis()
        hp = self.hyperparameter_sensitivity()
        syn = self.synergy_analysis()

        return {
            "architecture_specification": self.architecture_specification(),
            "feature_engineering": {
                "component": fe.component,
                "marginal_accuracy": fe.marginal_accuracy,
                "marginal_f1": fe.marginal_f1,
                "information_gain": fe.information_gain,
                "feature_utilisation": fe.feature_utilisation,
            },
            "importance_weighting": iw,
            "hyperparameter_sensitivity": hp,
            "synergy": syn,
        }

    # ── formatting ───────────────────────────────────────────────────

    @staticmethod
    def format_methodology_report(results: dict[str, Any]) -> str:
        """Format methodology analysis as a Markdown report.

        Parameters
        ----------
        results : dict
            Output from ``full_analysis()``.

        Returns
        -------
        str
            Markdown-formatted report.
        """
        lines: list[str] = []
        lines.append("## Methodology Formalisation\n")

        # Architecture
        spec = results["architecture_specification"]
        lines.append("### Architecture Specification\n")
        lines.append(f"**{spec['name']}** v{spec['version']}\n")
        lines.append("```")
        lines.append(spec["layers"].strip())
        lines.append("```\n")
        lines.append("### Novelty Claim\n")
        lines.append(spec["novelty"].strip())

        # Feature engineering
        fe = results["feature_engineering"]
        lines.append("\n### Feature Engineering Contribution\n")
        lines.append(f"- **Component**: {fe['component']}")
        lines.append(f"- **Marginal accuracy**: {fe['marginal_accuracy']:+.4f}")
        lines.append(f"- **Marginal F1**: {fe['marginal_f1']:+.4f}")
        lines.append(f"- **Information gain**: {fe['information_gain']:.4f}")
        lines.append(f"- **Feature utilisation**: {fe['feature_utilisation']:.1%}")

        # Importance weighting
        iw = results["importance_weighting"]
        lines.append("\n### Importance-Weighted vs Uniform Fusion\n")
        lines.append(
            f"| Weighting | Accuracy | F1 |\n"
            f"|-----------|----------|----|\n"
            f"| **Importance** | {iw['weighted_accuracy']:.4f} | {iw['weighted_f1']:.4f} |\n"
            f"| Uniform | {iw['uniform_accuracy']:.4f} | {iw['uniform_f1']:.4f} |\n"
            f"| Random | {iw['random_accuracy']:.4f} | {iw['random_f1']:.4f} |"
        )
        lines.append(
            f"\nImportance weighting improves accuracy by "
            f"{iw['improvement_over_uniform']:+.4f} over uniform."
        )

        # Synergy
        syn = results["synergy"]
        lines.append("\n### Component Synergy\n")
        lines.append(f"- Baseline: {syn['baseline_accuracy']:.4f}")
        lines.append(f"- +Feature Engineering: {syn['improvement_A_feature_eng']:+.4f}")
        lines.append(f"- +Deeper Trees: {syn['improvement_B_deeper_trees']:+.4f}")
        lines.append(f"- Combined: {syn['improvement_AB_combined']:+.4f}")
        lines.append(f"- Predicted additive: {syn['predicted_additive']:+.4f}")
        lines.append(f"- **Synergy score**: {syn['synergy_score']:+.4f}")
        lines.append(f"- Synergistic: {'YES ✓' if syn['is_synergistic'] else 'No (additive)'}")

        return "\n".join(lines)

    # ── internal CV helpers ──────────────────────────────────────────

    def _quick_cv(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_folds: int = 5,
    ) -> tuple[float, float]:
        """Quick 5-fold CV returning (mean_accuracy, mean_f1)."""
        import xgboost as xgb

        n_classes = len(np.unique(y))
        if n_classes == 2:
            model = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.1,
                objective="binary:logistic",
                eval_metric="logloss",
                use_label_encoder=False,
                verbosity=0,
                random_state=self.seed,
            )
        else:
            model = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.1,
                objective="multi:softprob",
                eval_metric="mlogloss",
                use_label_encoder=False,
                verbosity=0,
                random_state=self.seed,
            )
        return self._quick_cv_model(model, X, y, n_folds)

    def _quick_cv_model(
        self,
        model: Any,
        X: np.ndarray,
        y: np.ndarray,
        n_folds: int = 5,
    ) -> tuple[float, float]:
        """Quick k-fold CV for any sklearn-compatible model."""
        from sklearn.base import clone
        from sklearn.metrics import accuracy_score, f1_score

        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=self.seed)
        accs, f1s = [], []
        for tr, te in skf.split(X, y):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                m = clone(model)
                m.fit(X[tr], y[tr])
                pred = m.predict(X[te])
                accs.append(accuracy_score(y[te], pred))
                f1s.append(f1_score(y[te], pred, average="weighted", zero_division=0))
        return float(np.mean(accs)), float(np.mean(f1s))

    def _mutual_info_cols(
        self,
        X_subset: pd.DataFrame,
        y: np.ndarray,
    ) -> float:
        """Mean mutual information between feature columns and labels."""
        mis = []
        for col in X_subset.columns:
            vals = pd.qcut(X_subset[col], q=10, duplicates="drop").cat.codes.values
            mi = mutual_info_score(y, vals)
            mis.append(mi)
        return float(np.mean(mis)) if mis else 0.0
