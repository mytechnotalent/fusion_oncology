"""
Fusion Engine – combines XGBoost importance with DNABERT instability.

This is the central orchestrator: it trains XGBoost for dynamic gene
scoring, retrieves DNA sequences from NCBI, computes structural
instability via embedding drift, and produces a ranked fusion index.
Now also integrates clinical evidence, resistance prediction,
synthetic lethality screening, and network pharmacology scoring.
"""

from __future__ import annotations

import logging
from typing import Any

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
    Multi-modal analysis engine fusing XGBoost and DNABERT.

    Workflow
    --------
    1.  Train XGBoost on the expression matrix → top-K gene importance.
    2.  For each top gene, fetch the RefSeq DNA sequence from NCBI.
    3.  Compute DNABERT embedding and measure mutational instability
        (average cosine drift under random single-nucleotide mutations).
    4.  Combine importance × instability into a Fusion Index.
    5.  Enrich against KEGG / Reactome pathways.
    6.  Annotate drug targets and druggability.
    7.  Score resistance risk per gene.
    8.  Screen for synthetic lethality partners.
    9.  Compute network pharmacology strategic score.

    Parameters
    ----------
    config : ProjectConfig
        Runtime configuration.
    """

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
        self.xgb.fit(X, y)
        self.cv_metrics = self.xgb.cross_validate(X, y)
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

    # ── public API ───────────────────────────────────────────────────────

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
        top = self._train_xgboost(X, y)
        self._score_instability(top, self._fetch_sequences(top))
        self._enrich_pathways()
        self._annotate_drugs()
        self._score_resistance()
        self._annotate_network()
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
        mean = self.cv_metrics["mean_accuracy"]
        std = self.cv_metrics["std_accuracy"]
        lines.append(f"\nXGBoost CV accuracy: {mean:.4f} ± {std:.4f}")

    # ── convenience ──────────────────────────────────────────────────────

    def summary(self) -> str:
        """Return a human-readable summary of the latest analysis.

        Returns
        -------
        str
            Multi-line string containing a boxed ranking table and
            XGBoost cross-validation metrics (if available).
        """
        if self.results.empty:
            return "No analysis has been run yet."
        lines = self._build_ranking_lines()
        self._append_cv_metrics(lines)
        return "\n".join(lines)
