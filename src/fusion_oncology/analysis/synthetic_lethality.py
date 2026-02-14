"""
Synthetic lethality detection.

Identifies gene pairs where simultaneous loss of function is lethal to
cancer cells while individual knockouts are tolerated.  This is a key
strategy for targeting tumour-suppressor-loss cancers (e.g. BRCA1/2
deficiency → PARP inhibitor sensitivity).
"""

from __future__ import annotations

import logging
from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from fusion_oncology.config import ProjectConfig

logger = logging.getLogger(__name__)

# ── Curated synthetic-lethal pairs (literature-validated) ────────────────
# Sources: SynLethDB, Decipher, published CRISPR screens

KNOWN_SL_PAIRS: list[tuple[str, str, str]] = [
    ("BRCA1", "PARP1", "Approved: Olaparib (PARP inhibitor)"),
    ("BRCA2", "PARP1", "Approved: Olaparib, Rucaparib"),
    ("BRCA1", "PARP2", "Approved: Niraparib"),
    ("RB1", "AURKB", "Preclinical: Aurora-B inhibitors"),
    ("TP53", "WEE1", "Phase II: Adavosertib (WEE1 inhibitor)"),
    ("TP53", "CHK1", "Phase I/II: Prexasertib"),
    ("KRAS", "STK33", "Preclinical: STK33 dependency in KRAS-mutant"),
    ("MYC", "CDK9", "Phase I: CDK9 inhibitors"),
    ("ARID1A", "EZH2", "Phase II: Tazemetostat (EZH2 inhibitor)"),
    ("PTEN", "PLK4", "Preclinical: PLK4 inhibitors in PTEN-null"),
    ("VHL", "HIF2A", "Approved: Belzutifan (HIF-2α inhibitor, ccRCC)"),
    ("SMAD4", "AURKB", "Preclinical"),
    ("APC", "CTNNB1", "Preclinical: beta-catenin inhibitors"),
    ("CDKN2A", "CDK4", "Approved: Palbociclib in CDKN2A-loss"),
    ("CDKN2A", "CDK6", "Approved: Ribociclib in CDKN2A-loss"),
    ("ATM", "PARP1", "Phase III: PARP inhibitors in ATM-loss"),
    ("ATM", "ATR", "Phase II: ATR inhibitors"),
    ("STK11", "MTOR", "Preclinical: mTOR inhibitors in LKB1-loss"),
    ("SETD2", "WEE1", "Preclinical"),
    ("BAP1", "EZH2", "Phase II: Tazemetostat"),
    ("RAS", "RAF", "Vertical pathway combination therapy"),
    ("EGFR", "MET", "Combination: Osimertinib + Capmatinib"),
    ("BRAF", "MEK1", "Approved: Dabrafenib + Trametinib"),
    ("BRAF", "MEK2", "Approved: Encorafenib + Binimetinib"),
]


class SyntheticLethalityDetector:
    """Detect synthetic lethal interactions for prioritised gene targets.

    Combines a curated knowledge base of known SL pairs with a
    correlation-based computational screen that identifies candidate
    pairs from expression data.

    Parameters
    ----------
    config : ProjectConfig, optional
        Runtime configuration.
    """

    def __init__(self, config: ProjectConfig | None = None) -> None:
        """Initialise the detector with the curated SL database.

        Parameters
        ----------
        config : ProjectConfig, optional
            Runtime configuration.  Falls back to defaults when ``None``.
        """
        self.cfg = config or ProjectConfig()
        self._build_lookup()

    def _build_lookup(self) -> None:
        """Index the curated SL pairs for fast gene-based retrieval.

        Populates ``self._lookup`` as a dict mapping each gene to a list
        of ``(partner, annotation)`` tuples.
        """
        self._lookup: dict[str, list[tuple[str, str]]] = {}
        for g1, g2, annot in KNOWN_SL_PAIRS:
            self._lookup.setdefault(g1, []).append((g2, annot))
            self._lookup.setdefault(g2, []).append((g1, annot))

    def known_partners(self, gene: str) -> list[dict[str, str]]:
        """Return curated synthetic-lethal partners for *gene*.

        Parameters
        ----------
        gene : str
            HGNC gene symbol.

        Returns
        -------
        list[dict[str, str]]
            Each dict has keys ``partner`` and ``annotation``.
        """
        pairs = self._lookup.get(gene, [])
        return [{"partner": p, "annotation": a} for p, a in pairs]

    def _compute_correlation(
        self,
        X: pd.DataFrame,
        gene_a: str,
        gene_b: str,
        method: str,
    ) -> tuple[float, float]:
        """Compute pairwise correlation between two genes.

        Parameters
        ----------
        X : pd.DataFrame
            Expression matrix (samples × genes).
        gene_a : str
            First gene symbol.
        gene_b : str
            Second gene symbol.
        method : str
            ``"spearman"`` or ``"pearson"``.

        Returns
        -------
        tuple[float, float]
            Correlation coefficient and p-value.
        """
        if method == "spearman":
            return stats.spearmanr(X[gene_a], X[gene_b])
        return stats.pearsonr(X[gene_a], X[gene_b])

    def _build_sl_candidate(
        self,
        gene_a: str,
        gene_b: str,
        corr: float,
        pval: float,
    ) -> dict[str, Any]:
        """Build a single SL candidate record.

        Parameters
        ----------
        gene_a : str
            Primary gene symbol.
        gene_b : str
            Candidate partner symbol.
        corr : float
            Correlation coefficient.
        pval : float
            Associated p-value.

        Returns
        -------
        dict[str, Any]
            Record with ``Gene_A``, ``Gene_B``, ``Correlation``,
            ``P_Value``, and ``Known_SL`` keys.
        """
        known = any(p == gene_b for p, _ in self._lookup.get(gene_a, []))
        return {
            "Gene_A": gene_a,
            "Gene_B": gene_b,
            "Correlation": round(float(corr), 4),
            "P_Value": float(pval),
            "Known_SL": known,
        }

    def _finalize_sl_screen(
        self,
        results: list[dict[str, Any]],
        threshold: float,
    ) -> pd.DataFrame:
        """Sort, log, and return the SL screen results DataFrame.

        Parameters
        ----------
        results : list[dict[str, Any]]
            Raw candidate records.
        threshold : float
            Correlation threshold used (for logging).

        Returns
        -------
        pd.DataFrame
            Sorted DataFrame of SL candidates.
        """
        df = pd.DataFrame(results)
        if not df.empty:
            df = df.sort_values("Correlation").reset_index(drop=True)
        logger.info("SL screen: %d candidate pairs (threshold=%.2f)", len(df), threshold)
        return df

    def _screen_one_target(
        self,
        X: pd.DataFrame,
        gene_a: str,
        targets: list[str],
        method: str,
        threshold: float,
    ) -> list[dict[str, Any]]:
        """Screen all genes against a single target for SL candidates.

        Parameters
        ----------
        X : pd.DataFrame
            Expression matrix.
        gene_a : str
            Target gene to screen partners for.
        targets : list[str]
            All target genes (excluded from partner candidates).
        method : str
            Correlation method.
        threshold : float
            Correlation threshold.

        Returns
        -------
        list[dict[str, Any]]
            Candidate records below *threshold*.
        """
        hits: list[dict[str, Any]] = []
        for gene_b in X.columns:
            if gene_a == gene_b or gene_b in targets:
                continue
            corr, pval = self._compute_correlation(X, gene_a, gene_b, method)
            if corr < threshold:
                hits.append(self._build_sl_candidate(gene_a, gene_b, corr, pval))
        return hits

    def screen_from_expression(
        self,
        X: pd.DataFrame,
        target_genes: list[str],
        threshold: float = -0.5,
        method: str = "spearman",
    ) -> pd.DataFrame:
        """Compute candidate SL pairs from expression anti-correlation.

        Genes whose expression is strongly anti-correlated with a target
        gene are candidates for synthetic lethality: when one is high
        the other is low, suggesting mutual exclusivity.

        Parameters
        ----------
        X : pd.DataFrame
            Expression matrix (samples × genes).
        target_genes : list[str]
            Genes to screen partners for (typically the top-K).
        threshold : float
            Correlation threshold below which a pair is flagged as a
            candidate (default ``-0.5``).
        method : str
            Correlation method: ``"spearman"`` (default) or ``"pearson"``.

        Returns
        -------
        pd.DataFrame
            Columns: ``Gene_A``, ``Gene_B``, ``Correlation``, ``P_Value``,
            ``Known_SL``.
        """
        results: list[dict[str, Any]] = []
        for gene_a in (g for g in target_genes if g in X.columns):
            results.extend(self._screen_one_target(X, gene_a, target_genes, method, threshold))
        return self._finalize_sl_screen(results, threshold)

    def _annotate_sl_gene(self, gene: str) -> tuple[str, int]:
        """Return SL partner summary for a single gene.

        Parameters
        ----------
        gene : str
            HGNC gene symbol.

        Returns
        -------
        tuple[str, int]
            Comma-separated partner names and partner count.
        """
        partners = self.known_partners(gene)
        names = ", ".join(p["partner"] for p in partners) if partners else "—"
        return names, len(partners)

    def annotate(self, results: pd.DataFrame) -> pd.DataFrame:
        """Add synthetic-lethality annotations to fusion results.

        Parameters
        ----------
        results : pd.DataFrame
            Must contain a ``Gene`` column.

        Returns
        -------
        pd.DataFrame
            Input with extra columns: ``SL_Partners`` (comma-separated
            partner gene names), ``SL_Count`` (number of known partners).
        """
        annotations = [self._annotate_sl_gene(g) for g in results["Gene"]]
        out = results.copy()
        out["SL_Partners"] = [a[0] for a in annotations]
        out["SL_Count"] = [a[1] for a in annotations]
        return out

    def _build_combo_suggestion(
        self,
        gene: str,
        partner: dict[str, str],
    ) -> dict[str, str]:
        """Build a single combination-therapy suggestion.

        Parameters
        ----------
        gene : str
            Primary target gene.
        partner : dict[str, str]
            Partner record with ``partner`` and ``annotation`` keys.

        Returns
        -------
        dict[str, str]
            Suggestion with ``target_gene``, ``sl_partner``,
            ``strategy``, and ``annotation`` keys.
        """
        return {
            "target_gene": gene,
            "sl_partner": partner["partner"],
            "strategy": f"Target {partner['partner']} in {gene}-deficient tumours",
            "annotation": partner["annotation"],
        }

    def combination_therapies(self, gene: str) -> list[dict[str, str]]:
        """Suggest combination therapies based on SL relationships.

        Parameters
        ----------
        gene : str
            HGNC gene symbol of the primary target.

        Returns
        -------
        list[dict[str, str]]
            Each dict has keys ``target_gene``, ``sl_partner``,
            ``strategy``, ``annotation``.
        """
        partners = self.known_partners(gene)
        return [self._build_combo_suggestion(gene, p) for p in partners]
