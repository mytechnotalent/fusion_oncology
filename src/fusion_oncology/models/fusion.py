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
        self.instability = InstabilityAnalyzer(self.bert, self.cfg)
        self.pathway = PathwayEnrichment(self.cfg)
        self.drug_mapper = DrugTargetMapper(self.cfg)
        self.resistance = ResistancePredictor(self.cfg)
        self.sl_detector = SyntheticLethalityDetector(self.cfg)
        self.network = InteractionNetwork(self.cfg)

        # Populated after analysis
        self.results: pd.DataFrame = pd.DataFrame()
        self.cv_metrics: dict[str, Any] = {}

    # ── public API ───────────────────────────────────────────────────────

    def run(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """
        Execute the full fusion pipeline.

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
        # Step 1 – XGBoost importance
        logger.info("Step 1/4: Training XGBoost classifier …")
        self.xgb.fit(X, y)
        self.cv_metrics = self.xgb.cross_validate(X, y)
        top = self.xgb.top_genes()

        # Step 2 – Fetch sequences
        logger.info("Step 2/4: Fetching gene sequences from NCBI …")
        sequences: dict[str, str] = {}
        for gene in top:
            sequences[gene] = fetch_gene_sequence(gene, self.cfg)

        # Step 3 – Instability scoring
        logger.info("Step 3/4: Computing mutational instability …")
        rows: list[dict[str, Any]] = []
        for gene, importance in top.items():
            seq = sequences[gene]
            inst = self.instability.score(gene, seq)
            fusion_index = importance * inst * 1000
            rows.append(
                {
                    "Gene": gene,
                    "XGB_Importance": round(importance, 6),
                    "Instability": round(inst, 6),
                    "Fusion_Index": round(fusion_index, 4),
                    "Seq_Length": len(seq),
                }
            )
            logger.info(
                "  %s  imp=%.4f  inst=%.4f  fusion=%.4f",
                gene,
                importance,
                inst,
                fusion_index,
            )

        self.results = (
            pd.DataFrame(rows)
            .sort_values("Fusion_Index", ascending=False)
            .reset_index(drop=True)
        )

        # Step 4 – Pathway enrichment
        logger.info("Step 4/7: Pathway enrichment …")
        try:
            self.results = self.pathway.annotate(self.results)
        except Exception:
            logger.warning("Pathway enrichment skipped (network or DB unavailable)")

        # Step 5 – Drug-target annotation
        logger.info("Step 5/7: Drug-target annotation …")
        self.results = self.drug_mapper.annotate(self.results)

        # Step 6 – Resistance risk scoring
        logger.info("Step 6/7: Resistance risk scoring …")
        self.results = self.resistance.annotate(self.results)

        # Step 7 – Network pharmacology & synthetic lethality
        logger.info("Step 7/7: Network pharmacology & SL screening …")
        self.results["Strategic_Score"] = self.results["Gene"].apply(
            self.network.strategic_score
        )
        self.results["SL_Partners"] = self.results["Gene"].apply(
            lambda g: ", ".join(
                d["partner"] for d in self.sl_detector.known_partners(g)
            )
            or "—"
        )

        return self.results

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
        lines = [
            "╔══════════════════════════════════════╗",
            "║   FUSION ONCOLOGY – TARGET RANKING   ║",
            "╠══════════════════════════════════════╣",
        ]
        for _, row in self.results.iterrows():
            lines.append(
                f"║  {row['Gene']:<12}  Fusion Index: {row['Fusion_Index']:>8.4f}  ║"
            )
        lines.append("╚══════════════════════════════════════╝")
        if self.cv_metrics:
            lines.append(
                f"\nXGBoost CV accuracy: {self.cv_metrics['mean_accuracy']:.4f} "
                f"± {self.cv_metrics['std_accuracy']:.4f}"
            )
        return "\n".join(lines)
