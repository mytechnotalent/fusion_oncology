"""
Fusion Engine – true multi-modal fusion of XGBoost and DNABERT-2.

This is the central orchestrator: it trains an initial XGBoost for gene
ranking, computes DNABERT-2 embeddings for the top genes, then builds
a *fusion feature matrix* that concatenates drug-sensitivity values with
sensitivity-weighted DNABERT-2 embeddings.  A second XGBoost classifier
is trained on this combined space so that a single model jointly learns
from both modalities — making this a genuine learned fusion, not just
a heuristic product of two independent scores.
"""

from __future__ import annotations

import logging
import re
from typing import Any

import numpy as np
import pandas as pd

from fusion_oncology.analysis.drug_target import DrugTargetMapper
from fusion_oncology.analysis.instability import InstabilityAnalyzer
from fusion_oncology.analysis.network_pharmacology import InteractionNetwork
from fusion_oncology.analysis.pathway import PathwayEnrichment
from fusion_oncology.analysis.resistance import ResistancePredictor
from fusion_oncology.analysis.synthetic_lethality import SyntheticLethalityDetector
from fusion_oncology.config import ProjectConfig
from fusion_oncology.models.dnabert_engine import DNABERTEngine
from fusion_oncology.models.xgboost_engine import XGBoostEngine
from fusion_oncology.utils.bio import fetch_gene_sequence

logger = logging.getLogger(__name__)


class FusionEngine:
    """
    True multi-modal fusion engine combining XGBoost and DNABERT-2.

    Workflow
    --------
    1.  Train *baseline* XGBoost on drug-sensitivity features → top-K genes.
    2.  Fetch RefSeq DNA sequences for the top-K genes from NCBI.
    3.  Compute 768-dim DNABERT-2 embeddings for each gene sequence.
    4.  Build a **fusion feature matrix**: for every cell line, weight each
        gene's embedding by that cell line's sensitivity value, producing
        a 768-dim "genomic context" vector per sample.  Concatenate this
        with the original drug-sensitivity features.
    5.  Train a *fusion* XGBoost on the combined feature space and run
        stratified 5-fold CV — a single model that jointly learns from
        both modalities.
    6.  Compute mutational instability via embedding drift and derive
        the Fusion Index (importance × instability × 1000).
    7.  Enrich against KEGG / Reactome pathways.
    8.  Annotate drug targets and druggability.
    9.  Score resistance risk per gene.
    10. Screen for synthetic lethality partners.
    11. Compute network pharmacology strategic score.

    Parameters
    ----------
    config : ProjectConfig
        Runtime configuration.
    """

    _GENE_RE = re.compile(r"^[A-Z][A-Z0-9/-]*$")

    _CANCER_DRIVERS = frozenset(
        {
            "ABL1",
            "AKT1",
            "ALK",
            "APC",
            "AR",
            "ARID1A",
            "ATM",
            "BRAF",
            "BRCA1",
            "BRCA2",
            "CCND1",
            "CCNE1",
            "CDK4",
            "CDK6",
            "CDKN2A",
            "CREBBP",
            "CTNNB1",
            "DNMT3A",
            "EGFR",
            "EP300",
            "ERBB2",
            "ERG",
            "ESR1",
            "ETV6",
            "EZH2",
            "FGFR1",
            "FGFR2",
            "FGFR3",
            "FLT3",
            "GATA3",
            "HRAS",
            "IDH1",
            "IDH2",
            "JAK2",
            "KIT",
            "KMT2A",
            "KRAS",
            "MAP2K1",
            "MDM2",
            "MET",
            "MTOR",
            "MYC",
            "NF1",
            "NF2",
            "NOTCH1",
            "NPM1",
            "NRAS",
            "PDGFRA",
            "PIK3CA",
            "PTEN",
            "PTCH1",
            "RAF1",
            "RB1",
            "RET",
            "RUNX1",
            "SF3B1",
            "SMAD4",
            "STK11",
            "TERT",
            "TET2",
            "TP53",
            "VHL",
        }
    )

    # ── initialisation helpers ───────────────────────────────────────────

    def _init_analyzers(self) -> None:
        """Create downstream analysis components.

        Instantiates instability, pathway, drug-target, resistance,
        synthetic-lethality, and network-pharmacology analyzers.
        """
        self.instability = InstabilityAnalyzer(self.bert, self.cfg)
        self.pathway = PathwayEnrichment(self.cfg)
        self.drug_mapper = DrugTargetMapper(self.cfg)
        self.resistance = ResistancePredictor(self.cfg)
        self.sl_detector = SyntheticLethalityDetector(self.cfg)
        self.network = InteractionNetwork(self.cfg)

    def __init__(self, config: ProjectConfig | None = None) -> None:
        """Initialise sub-engines and analysis components.

        Parameters
        ----------
        config : ProjectConfig, optional
            Runtime configuration.  Uses defaults when ``None``.
        """
        self.cfg = config or ProjectConfig()
        self.xgb = XGBoostEngine(self.cfg)
        self.bert = DNABERTEngine(self.cfg)
        self._init_analyzers()
        self.results: pd.DataFrame = pd.DataFrame()
        self.cv_metrics: dict[str, Any] = {}
        self.fusion_cv_metrics: dict[str, Any] = {}
        self.dnabert_metrics: dict[str, Any] = {}
        self._gene_embeddings: dict[str, np.ndarray] = {}

    # ── run helpers ──────────────────────────────────────────────────────

    def _train_xgboost(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> dict[str, float]:
        """Train XGBoost and return top-gene importance scores.

        Parameters
        ----------
        X : pd.DataFrame
            Gene-expression feature matrix.
        y : pd.Series
            Cancer-type labels.

        Returns
        -------
        dict[str, float]
            Top-K gene names mapped to importance scores.
        """
        logger.info("Step 1/4: Training XGBoost classifier …")
        y_merged = XGBoostEngine.merge_rare_classes(y, self.cfg.min_class_size)
        if self.cfg.enable_hpo:
            self.xgb.run_hpo(X, y_merged)
        self.xgb.fit(X, y_merged)
        self.cv_metrics = self.xgb.cross_validate(X, y_merged)
        return self.xgb.top_genes()

    def _fetch_sequences(
        self,
        genes: dict[str, float],
    ) -> dict[str, str]:
        """Retrieve DNA sequences from NCBI for each gene.

        Parameters
        ----------
        genes : dict[str, float]
            Gene names (keys) with importance scores (values).

        Returns
        -------
        dict[str, str]
            Gene names mapped to their RefSeq DNA sequences.
        """
        logger.info("Step 2/4: Fetching gene sequences from NCBI …")
        sequences: dict[str, str] = {}
        for gene in genes:
            sequences[gene] = fetch_gene_sequence(gene, self.cfg)
        return sequences

    def _gene_row(
        self,
        gene: str,
        importance: float,
        seq: str,
    ) -> dict[str, Any]:
        """Build a single result row for one gene.

        Parameters
        ----------
        gene : str
            Gene symbol.
        importance : float
            XGBoost feature importance score.
        seq : str
            DNA sequence for the gene.

        Returns
        -------
        dict[str, Any]
            Row dict with gene metrics.
        """
        inst = self.instability.score(gene, seq)
        fi = round(importance * inst * 1000, 4)
        logger.info("  %s imp=%.4f inst=%.4f fi=%.4f", gene, importance, inst, fi)
        return {
            "Gene": gene,
            "XGB_Importance": round(importance, 6),
            "Instability": round(inst, 6),
            "Fusion_Index": fi,
            "Seq_Length": len(seq),
        }

    def _score_instability(
        self,
        top: dict[str, float],
        sequences: dict[str, str],
    ) -> None:
        """Score mutational instability and populate results.

        Parameters
        ----------
        top : dict[str, float]
            Top gene importance scores from XGBoost.
        sequences : dict[str, str]
            Gene-name → DNA-sequence mapping.
        """
        logger.info("Step 3/4: Computing mutational instability …")
        rows = [self._gene_row(g, imp, sequences[g]) for g, imp in top.items()]
        df = pd.DataFrame(rows).sort_values(
            "Fusion_Index",
            ascending=False,
        )
        self.results = df.reset_index(drop=True)

    def _enrich_pathways(self) -> None:
        """Annotate results with KEGG / Reactome pathway enrichment.

        Silently skips enrichment when the pathway database or
        network is unavailable.
        """
        logger.info("Step 4/7: Pathway enrichment …")
        try:
            self.results = self.pathway.annotate(self.results)
        except Exception:
            logger.warning("Pathway enrichment skipped " "(network or DB unavailable)")

    def _annotate_drugs(self) -> None:
        """Annotate results with drug-target mappings.

        Adds druggability and known-target columns to
        ``self.results``.
        """
        logger.info("Step 5/7: Drug-target annotation …")
        self.results = self.drug_mapper.annotate(self.results)

    def _score_resistance(self) -> None:
        """Score resistance risk for each gene in results.

        Adds a resistance-risk column to ``self.results``.
        """
        logger.info("Step 6/7: Resistance risk scoring …")
        self.results = self.resistance.annotate(self.results)

    def _sl_label(self, gene: str) -> str:
        """Format synthetic-lethality partners as a display string.

        Parameters
        ----------
        gene : str
            Gene symbol to look up.

        Returns
        -------
        str
            Comma-separated partner list, or ``"—"`` if none.
        """
        partners = self.sl_detector.known_partners(gene)
        return ", ".join(d["partner"] for d in partners) or "—"

    def _annotate_network(self) -> None:
        """Add network pharmacology and SL partner annotations.

        Populates ``Strategic_Score`` and ``SL_Partners`` columns
        on ``self.results``.
        """
        logger.info("Step 7/7: Network pharmacology & SL screening …")
        genes = self.results["Gene"]
        self.results["Strategic_Score"] = genes.apply(
            self.network.strategic_score,
        )
        self.results["SL_Partners"] = genes.apply(self._sl_label)

    # ── fusion feature helpers ───────────────────────────────────────────

    def _compute_gene_embeddings(
        self,
        sequences: dict[str, str],
    ) -> dict[str, np.ndarray]:
        """Compute DNABERT-2 embeddings for each gene sequence.

        Parameters
        ----------
        sequences : dict[str, str]
            Gene-name → DNA-sequence mapping.

        Returns
        -------
        dict[str, np.ndarray]
            Gene-name → 768-dim embedding vector mapping.
        """
        logger.info("Computing DNABERT-2 embeddings for %d genes …", len(sequences))
        return {gene: self.bert.embed(seq) for gene, seq in sequences.items()}

    @staticmethod
    def _build_embedding_matrix(
        gene_list: list[str],
        embeddings: dict[str, np.ndarray],
    ) -> np.ndarray:
        """Stack gene embeddings into a (K × emb_dim) matrix.

        Parameters
        ----------
        gene_list : list[str]
            Ordered gene names.
        embeddings : dict[str, np.ndarray]
            Gene-name → embedding vector mapping.

        Returns
        -------
        np.ndarray
            Shape ``(K, emb_dim)`` embedding matrix.
        """
        return np.stack([embeddings[g] for g in gene_list])

    @staticmethod
    def _extract_sensitivity_values(
        X: pd.DataFrame,
        gene_list: list[str],
    ) -> np.ndarray:
        """Extract per-cell-line sensitivity values for top genes.

        Parameters
        ----------
        X : pd.DataFrame
            Full drug-sensitivity feature matrix.
        gene_list : list[str]
            Top-K gene names to extract.

        Returns
        -------
        np.ndarray
            Shape ``(n_samples, K)`` sensitivity matrix.
        """
        sensitivity = np.zeros((len(X), len(gene_list)))
        for i, gene in enumerate(gene_list):
            if gene in X.columns:
                sensitivity[:, i] = X[gene].values
        return sensitivity

    @staticmethod
    def _compute_weighted_embeddings(
        sensitivity: np.ndarray,
        emb_matrix: np.ndarray,
    ) -> np.ndarray:
        """Compute sensitivity-weighted gene embedding per cell line.

        Parameters
        ----------
        sensitivity : np.ndarray
            Shape ``(n_samples, K)`` drug-sensitivity values.
        emb_matrix : np.ndarray
            Shape ``(K, emb_dim)`` gene embedding matrix.

        Returns
        -------
        np.ndarray
            Shape ``(n_samples, emb_dim)`` weighted embeddings.
        """
        return sensitivity @ emb_matrix

    @staticmethod
    def _normalise_embeddings(weighted: np.ndarray) -> np.ndarray:
        """L2-normalise each row of the weighted embedding matrix.

        Parameters
        ----------
        weighted : np.ndarray
            Shape ``(n_samples, emb_dim)`` unnormalised embeddings.

        Returns
        -------
        np.ndarray
            L2-normalised embeddings (unit vectors per row).
        """
        norms = np.linalg.norm(weighted, axis=1, keepdims=True)
        norms = np.where(norms < 1e-10, 1.0, norms)
        return weighted / norms

    def _build_fusion_features(
        self,
        X: pd.DataFrame,
        gene_list: list[str],
    ) -> pd.DataFrame:
        """Build the fusion feature matrix combining both modalities.

        Concatenates the original drug-sensitivity features with
        768-dim sensitivity-weighted DNABERT-2 embeddings per sample.

        Parameters
        ----------
        X : pd.DataFrame
            Original drug-sensitivity feature matrix.
        gene_list : list[str]
            Top-K gene names with available embeddings.

        Returns
        -------
        pd.DataFrame
            Combined feature matrix with ``N + 768`` columns.
        """
        emb_matrix = self._build_embedding_matrix(gene_list, self._gene_embeddings)
        sensitivity = self._extract_sensitivity_values(X, gene_list)
        weighted = self._compute_weighted_embeddings(sensitivity, emb_matrix)
        normed = self._normalise_embeddings(weighted)
        emb_dim = emb_matrix.shape[1]
        emb_cols = [f"BERT_emb_{i}" for i in range(emb_dim)]
        emb_df = pd.DataFrame(normed, index=X.index, columns=emb_cols)
        return pd.concat([X, emb_df], axis=1)

    def _train_fusion_xgboost(
        self,
        X_fused: pd.DataFrame,
        y: pd.Series,
    ) -> None:
        """Train XGBoost on the fusion feature matrix and evaluate.

        Creates a fresh ``XGBoostEngine``, fits it on the combined
        drug-sensitivity + DNABERT-2 embedding features, and stores
        the fusion CV metrics.

        Parameters
        ----------
        X_fused : pd.DataFrame
            Fusion feature matrix (original + 768-dim embeddings).
        y : pd.Series
            Cancer-type labels.
        """
        logger.info("Training fusion XGBoost on %d features …", X_fused.shape[1])
        y_merged = XGBoostEngine.merge_rare_classes(y, self.cfg.min_class_size)
        self.fusion_xgb = XGBoostEngine(self.cfg)
        if self.cfg.enable_hpo:
            self.fusion_xgb.run_hpo(X_fused, y_merged)
        self.fusion_xgb.fit(X_fused, y_merged)
        self.fusion_cv_metrics = self.fusion_xgb.cross_validate(X_fused, y_merged)

    # ── metrics helpers ────────────────────────────────────────────────

    @staticmethod
    def _instability_snr(inst: np.ndarray) -> float:
        """Compute signal-to-noise ratio of instability scores.

        Parameters
        ----------
        inst : np.ndarray
            Array of instability scores.

        Returns
        -------
        float
            Mean divided by standard deviation, or ``0.0`` if constant.
        """
        std = float(np.std(inst))
        if std < 1e-12:
            return 0.0
        return float(np.mean(inst) / std)

    def _compute_dnabert_metrics(self) -> None:
        """Compute DNABERT-2 evaluation metrics from scored results.

        Populates ``self.dnabert_metrics`` with instability statistics
        including mean, std, range, and signal-to-noise ratio.
        """
        inst = self.results["Instability"].values
        self.dnabert_metrics["mean_instability"] = float(np.mean(inst))
        self.dnabert_metrics["std_instability"] = float(np.std(inst))
        self.dnabert_metrics["max_instability"] = float(np.max(inst))
        self.dnabert_metrics["min_instability"] = float(np.min(inst))
        self.dnabert_metrics["n_genes_scored"] = len(inst)
        self.dnabert_metrics["instability_range"] = float(np.ptp(inst))
        self.dnabert_metrics["signal_noise_ratio"] = self._instability_snr(inst)

    def _compute_driver_enrichment(self) -> None:
        """Measure enrichment of top genes in known cancer driver list.

        Checks how many of the top-K genes appear in the COSMIC
        Cancer Gene Census reference set.
        """
        genes = set(self.results["Gene"].tolist())
        hits = genes & self._CANCER_DRIVERS
        total = len(genes)
        enrichment = len(hits) / total if total else 0.0
        self.dnabert_metrics["driver_genes_found"] = sorted(hits)
        self.dnabert_metrics["driver_enrichment"] = enrichment
        self.dnabert_metrics["n_driver_hits"] = len(hits)

    def _compute_all_metrics(self) -> None:
        """Compute DNABERT-2 and pipeline evaluation metrics.

        Combines instability statistics and cancer driver enrichment
        into a single evaluation pass.
        """
        self._compute_dnabert_metrics()
        self._compute_driver_enrichment()

    def _run_all_annotations(self) -> None:
        """Run all downstream annotation steps on scored results.

        Executes pathway enrichment, drug-target mapping, resistance
        scoring, and network pharmacology in sequence.
        """
        self._enrich_pathways()
        self._annotate_drugs()
        self._score_resistance()
        self._annotate_network()

    # ── public API ───────────────────────────────────────────────────────

    def _filter_gene_columns(
        self,
        X: pd.DataFrame,
    ) -> pd.DataFrame:
        """Keep only columns whose names resemble gene symbols.

        Filters out non-gene column names (e.g. ``Retinoic acid``,
        ``others``) that cannot be processed by DNABERT-2 sequence
        analysis.

        Parameters
        ----------
        X : pd.DataFrame
            Input feature matrix.

        Returns
        -------
        pd.DataFrame
            Filtered matrix with only gene-symbol columns.
        """
        gene_cols = [c for c in X.columns if self._GENE_RE.match(str(c))]
        dropped = set(X.columns) - set(gene_cols)
        if dropped:
            logger.info(
                "Filtered %d non-gene columns: %s",
                len(dropped),
                sorted(dropped)[:5],
            )
        if not gene_cols:
            logger.warning(
                "No gene-symbol columns found; using all columns",
            )
            return X
        return X[gene_cols]

    def run(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """Execute the full fusion pipeline.

        Parameters
        ----------
        X : pd.DataFrame
            Gene-expression feature matrix.
        y : pd.Series
            Cancer-type labels.

        Returns
        -------
        pd.DataFrame
            Ranked targets with columns:
            ``Gene | XGB_Importance | Instability | Fusion_Index``
        """
        X = self._filter_gene_columns(X)
        top = self._train_xgboost(X, y)
        sequences = self._fetch_sequences(top)
        self._gene_embeddings = self._compute_gene_embeddings(sequences)
        X_fused = self._build_fusion_features(X, list(top.keys()))
        self._train_fusion_xgboost(X_fused, y)
        self._score_instability(top, sequences)
        self._compute_all_metrics()
        self._run_all_annotations()
        return self.results

    # ── summary helpers ──────────────────────────────────────────────────

    def _ranking_row(self, row: pd.Series) -> str:
        """Build a single ranking table row.

        Parameters
        ----------
        row : pd.Series
            Result row with ``Gene`` and ``Fusion_Index``.

        Returns
        -------
        str
            Formatted ASCII-art row.
        """
        fi = row["Fusion_Index"]
        return f"║  {row['Gene']:<12}  Fusion Index: {fi:>8.4f}  ║"

    def _build_ranking_lines(self) -> list[str]:
        """Build the boxed ranking table lines.

        Returns
        -------
        list[str]
            Lines forming the ASCII-art ranking box.
        """
        hdr = [
            "╔══════════════════════════════════════╗",
            "║   FUSION ONCOLOGY – TARGET RANKING   ║",
            "╠══════════════════════════════════════╣",
        ]
        hdr.extend(self._ranking_row(row) for _, row in self.results.iterrows())
        hdr.append("╚══════════════════════════════════════╝")
        return hdr

    def _append_cv_metrics(self, lines: list[str]) -> None:
        """Append cross-validation metrics to the summary lines.

        Parameters
        ----------
        lines : list[str]
            Mutable list of summary lines to extend.
        """
        if not self.cv_metrics:
            return
        m = self.cv_metrics
        lines.append("\nXGBoost Baseline CV metrics (5-fold stratified):")
        for label, key in [
            ("Accuracy", "accuracy"),
            ("Precision", "precision"),
            ("Recall", "recall"),
            ("F1-Score", "f1"),
            ("F2-Score", "f2"),
            ("ROC AUC", "roc_auc"),
        ]:
            mean = m.get(f"mean_{key}", 0)
            std = m.get(f"std_{key}", 0)
            lines.append(f"  {label:>10s}: {mean:.4f} ± {std:.4f}")

    def _append_fusion_cv_metrics(self, lines: list[str]) -> None:
        """Append fusion model CV metrics to the summary lines.

        Parameters
        ----------
        lines : list[str]
            Mutable list of summary lines to extend.
        """
        if not self.fusion_cv_metrics:
            return
        m = self.fusion_cv_metrics
        lines.append("\nFusion Model CV metrics (XGBoost + DNABERT-2 features):")
        for label, key in [
            ("Accuracy", "accuracy"),
            ("Precision", "precision"),
            ("Recall", "recall"),
            ("F1-Score", "f1"),
            ("F2-Score", "f2"),
            ("ROC AUC", "roc_auc"),
        ]:
            mean = m.get(f"mean_{key}", 0)
            std = m.get(f"std_{key}", 0)
            lines.append(f"  {label:>10s}: {mean:.4f} ± {std:.4f}")

    def _append_dnabert_metrics(self, lines: list[str]) -> None:
        """Append DNABERT-2 evaluation metrics to summary lines.

        Parameters
        ----------
        lines : list[str]
            Mutable list of summary lines to extend.
        """
        if not self.dnabert_metrics:
            return
        m = self.dnabert_metrics
        lines.append("\nDNABERT-2 Evaluation Metrics:")
        lines.append(f"  Mean Instability: {m['mean_instability']:.6f}")
        lines.append(f"  Std  Instability: {m['std_instability']:.6f}")
        lines.append(f"  Range: {m['min_instability']:.6f} \u2013 {m['max_instability']:.6f}")
        lines.append(f"  Signal-to-Noise: {m['signal_noise_ratio']:.2f}")

    def _format_driver_line(self) -> str:
        """Format the cancer driver enrichment as a summary string.

        Returns
        -------
        str
            Formatted enrichment line with gene names.
        """
        m = self.dnabert_metrics
        hits = m.get("n_driver_hits", 0)
        total = m.get("n_genes_scored", 0)
        pct = m.get("driver_enrichment", 0) * 100
        drivers = ", ".join(m.get("driver_genes_found", []))
        return f"  Driver Enrichment: {hits}/{total} ({pct:.0f}%) [{drivers}]"

    def _append_pipeline_metrics(self, lines: list[str]) -> None:
        """Append fusion pipeline evaluation metrics to summary lines.

        Parameters
        ----------
        lines : list[str]
            Mutable list of summary lines to extend.
        """
        if not self.dnabert_metrics:
            return
        lines.append("\nFusion Pipeline Evaluation:")
        lines.append(self._format_driver_line())

    # ── convenience ──────────────────────────────────────────────────────

    def summary(self) -> str:
        """Return a human-readable summary of the latest analysis.

        Returns
        -------
        str
            Multi-line string containing a boxed ranking table,
            XGBoost CV metrics, DNABERT-2 metrics, and pipeline
            evaluation (if available).
        """
        if self.results.empty:
            return "No analysis has been run yet."
        lines = self._build_ranking_lines()
        self._append_cv_metrics(lines)
        self._append_fusion_cv_metrics(lines)
        self._append_dnabert_metrics(lines)
        self._append_pipeline_metrics(lines)
        return "\n".join(lines)
