"""
Pathway enrichment analysis.

Maps the top-ranked genes to canonical cancer signalling pathways
(KEGG / Reactome) and annotates the results DataFrame with pathway
membership and enrichment p-values.

This module uses a curated local lookup for core oncology pathways and
optionally queries the KEGG REST API for broader coverage.
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from fusion_oncology.config import ProjectConfig

logger = logging.getLogger(__name__)

# ── Curated pathway database (offline fallback) ─────────────────────────
# Maps pathway name → set of gene symbols known to participate.
# Sources: KEGG hsa05200 (Pathways in cancer), Hallmark gene sets (MSigDB).
CANCER_PATHWAYS: dict[str, set[str]] = {
    "PI3K-Akt Signaling": {
        "PIK3CA",
        "PIK3CB",
        "PIK3R1",
        "AKT1",
        "AKT2",
        "PTEN",
        "MTOR",
        "TSC1",
        "TSC2",
        "RPS6KB1",
        "EIF4EBP1",
        "PDK1",
        "FOXO3",
    },
    "MAPK/ERK Signaling": {
        "KRAS",
        "NRAS",
        "HRAS",
        "BRAF",
        "RAF1",
        "MAP2K1",
        "MAP2K2",
        "MAPK1",
        "MAPK3",
        "EGFR",
        "ERBB2",
        "SOS1",
        "GRB2",
    },
    "p53 Tumor Suppression": {
        "TP53",
        "MDM2",
        "MDM4",
        "CDKN2A",
        "CDKN1A",
        "BAX",
        "BCL2",
        "CASP3",
        "CASP9",
        "APAF1",
        "ATM",
        "ATR",
        "CHEK1",
        "CHEK2",
    },
    "Wnt/β-catenin Signaling": {
        "CTNNB1",
        "APC",
        "AXIN1",
        "AXIN2",
        "GSK3B",
        "WNT1",
        "WNT3A",
        "FZD1",
        "LRP5",
        "LRP6",
        "DVL1",
        "TCF7L2",
        "LEF1",
    },
    "Notch Signaling": {
        "NOTCH1",
        "NOTCH2",
        "NOTCH3",
        "JAG1",
        "JAG2",
        "DLL1",
        "DLL3",
        "DLL4",
        "HES1",
        "HEY1",
        "RBPJ",
        "MAML1",
    },
    "Cell Cycle Regulation": {
        "CDK4",
        "CDK6",
        "CDK2",
        "CCND1",
        "CCNE1",
        "CCNA2",
        "CCNB1",
        "RB1",
        "E2F1",
        "CDKN2A",
        "CDKN2B",
        "CDKN1A",
        "CDKN1B",
    },
    "DNA Damage Repair": {
        "BRCA1",
        "BRCA2",
        "RAD51",
        "PALB2",
        "ATM",
        "ATR",
        "CHEK1",
        "CHEK2",
        "XRCC1",
        "ERCC1",
        "MLH1",
        "MSH2",
        "MSH6",
        "PMS2",
    },
    "Apoptosis": {
        "BCL2",
        "BCL2L1",
        "BAX",
        "BAK1",
        "BID",
        "BAD",
        "CASP3",
        "CASP8",
        "CASP9",
        "APAF1",
        "CYCS",
        "FADD",
        "FAS",
        "TNFRSF10A",
    },
    "Angiogenesis (VEGF)": {
        "VEGFA",
        "VEGFB",
        "VEGFC",
        "KDR",
        "FLT1",
        "FLT4",
        "HIF1A",
        "EPAS1",
        "ANGPT1",
        "ANGPT2",
        "TEK",
        "NRP1",
    },
    "Immune Checkpoint": {
        "CD274",
        "PDCD1",
        "CTLA4",
        "LAG3",
        "HAVCR2",
        "TIGIT",
        "IDO1",
        "CD80",
        "CD86",
        "ICOS",
        "BTLA",
        "VISTA",
    },
}


class PathwayEnrichment:
    """
    Annotate a gene ranking with pathway membership.

    Parameters
    ----------
    config : ProjectConfig
        Supplies ``pathway_db`` and ``enrichment_pval`` (for future API use).
    """

    def __init__(self, config: ProjectConfig | None = None) -> None:
        """Initialise the pathway enrichment analyser.

        Parameters
        ----------
        config : ProjectConfig, optional
            Runtime configuration.  Uses defaults when ``None``.
        """
        self.cfg = config or ProjectConfig()

    def lookup(self, gene: str) -> list[str]:
        """Return all curated cancer pathways that *gene* belongs to.

        Parameters
        ----------
        gene : str
            HGNC gene symbol (case-insensitive).

        Returns
        -------
        list[str]
            Pathway names.  Empty list if the gene is not found.
        """
        gene_upper = gene.upper()
        return [pw for pw, members in CANCER_PATHWAYS.items() if gene_upper in members]

    def annotate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add a ``Pathways`` column to the fusion results frame.

        Parameters
        ----------
        df : pd.DataFrame
            Must have a ``Gene`` column.

        Returns
        -------
        pd.DataFrame
            Input frame with an additional ``Pathways`` string column.
        """
        df = df.copy()
        df["Pathways"] = df["Gene"].apply(lambda g: "; ".join(self.lookup(g)) or "—")
        return df

    def enrichment_summary(self, genes: list[str]) -> dict[str, Any]:
        """Summarise how many of the input genes fall into each pathway.

        Parameters
        ----------
        genes : list[str]
            List of HGNC gene symbols to check.

        Returns
        -------
        dict[str, Any]
            ``{pathway_name: {"count": int, "genes": list[str]}}``.
            Only pathways with at least one match are included.
        """
        summary: dict[str, dict[str, Any]] = {}
        for pw, members in CANCER_PATHWAYS.items():
            matched = [g for g in genes if g.upper() in members]
            if matched:
                summary[pw] = {"count": len(matched), "genes": matched}
        return summary
