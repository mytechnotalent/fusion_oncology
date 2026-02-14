"""
Drug-target mapping and druggability annotation.

Cross-references the top gene targets against a curated set of
known druggable targets and approved oncology therapeutics.
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from fusion_oncology.config import ProjectConfig

logger = logging.getLogger(__name__)

# ── Curated drug-target map ──────────────────────────────────────────────
# Real approved / late-stage oncology drugs mapped to their primary
# molecular targets.  Sources: FDA labels, ClinicalTrials.gov, DrugBank.
DRUG_TARGET_DB: dict[str, list[dict[str, str]]] = {
    "EGFR": [
        {"drug": "Erlotinib", "status": "Approved", "indication": "NSCLC, Pancreatic"},
        {"drug": "Gefitinib", "status": "Approved", "indication": "NSCLC"},
        {"drug": "Osimertinib", "status": "Approved", "indication": "NSCLC (T790M)"},
        {"drug": "Cetuximab", "status": "Approved", "indication": "CRC, HNSCC"},
    ],
    "ERBB2": [
        {"drug": "Trastuzumab", "status": "Approved", "indication": "HER2+ Breast"},
        {"drug": "Pertuzumab", "status": "Approved", "indication": "HER2+ Breast"},
        {"drug": "Lapatinib", "status": "Approved", "indication": "HER2+ Breast"},
        {
            "drug": "T-DXd (Enhertu)",
            "status": "Approved",
            "indication": "HER2+ Breast, Gastric",
        },
    ],
    "BRAF": [
        {"drug": "Vemurafenib", "status": "Approved", "indication": "Melanoma (V600E)"},
        {"drug": "Dabrafenib", "status": "Approved", "indication": "Melanoma, NSCLC"},
        {"drug": "Encorafenib", "status": "Approved", "indication": "CRC, Melanoma"},
    ],
    "KRAS": [
        {"drug": "Sotorasib", "status": "Approved", "indication": "NSCLC (G12C)"},
        {"drug": "Adagrasib", "status": "Approved", "indication": "NSCLC (G12C)"},
    ],
    "PIK3CA": [
        {"drug": "Alpelisib", "status": "Approved", "indication": "HR+ Breast"},
    ],
    "MTOR": [
        {"drug": "Everolimus", "status": "Approved", "indication": "RCC, Breast, PNET"},
        {"drug": "Temsirolimus", "status": "Approved", "indication": "RCC"},
    ],
    "CDK4": [
        {"drug": "Palbociclib", "status": "Approved", "indication": "HR+ Breast"},
        {"drug": "Ribociclib", "status": "Approved", "indication": "HR+ Breast"},
    ],
    "CDK6": [
        {"drug": "Abemaciclib", "status": "Approved", "indication": "HR+ Breast"},
    ],
    "BCL2": [
        {"drug": "Venetoclax", "status": "Approved", "indication": "CLL, AML"},
    ],
    "VEGFA": [
        {
            "drug": "Bevacizumab",
            "status": "Approved",
            "indication": "CRC, NSCLC, RCC, GBM",
        },
    ],
    "KDR": [
        {
            "drug": "Ramucirumab",
            "status": "Approved",
            "indication": "Gastric, NSCLC, CRC",
        },
        {"drug": "Sunitinib", "status": "Approved", "indication": "RCC, GIST"},
        {"drug": "Sorafenib", "status": "Approved", "indication": "HCC, RCC, DTC"},
    ],
    "PTEN": [
        {
            "drug": "—",
            "status": "No direct drug",
            "indication": "Tumor suppressor (loss-of-function)",
        },
    ],
    "TP53": [
        {
            "drug": "APR-246 (Eprenetapopt)",
            "status": "Phase III",
            "indication": "MDS, AML (mutant p53)",
        },
    ],
    "BRCA1": [
        {
            "drug": "Olaparib",
            "status": "Approved",
            "indication": "Ovarian, Breast, Prostate",
        },
        {"drug": "Rucaparib", "status": "Approved", "indication": "Ovarian, Prostate"},
    ],
    "BRCA2": [
        {
            "drug": "Olaparib",
            "status": "Approved",
            "indication": "Ovarian, Breast, Prostate",
        },
        {"drug": "Niraparib", "status": "Approved", "indication": "Ovarian"},
    ],
    "CD274": [
        {
            "drug": "Atezolizumab",
            "status": "Approved",
            "indication": "NSCLC, Urothelial, TNBC",
        },
        {"drug": "Durvalumab", "status": "Approved", "indication": "NSCLC, Bladder"},
        {"drug": "Avelumab", "status": "Approved", "indication": "Merkel Cell, RCC"},
    ],
    "PDCD1": [
        {
            "drug": "Pembrolizumab",
            "status": "Approved",
            "indication": "Pan-tumor (MSI-H)",
        },
        {
            "drug": "Nivolumab",
            "status": "Approved",
            "indication": "Melanoma, NSCLC, RCC, +more",
        },
    ],
    "CTLA4": [
        {"drug": "Ipilimumab", "status": "Approved", "indication": "Melanoma, RCC"},
    ],
    "IDO1": [
        {
            "drug": "Epacadostat",
            "status": "Phase III (failed)",
            "indication": "Melanoma",
        },
    ],
    "ALK": [
        {"drug": "Crizotinib", "status": "Approved", "indication": "NSCLC (ALK+)"},
        {"drug": "Alectinib", "status": "Approved", "indication": "NSCLC (ALK+)"},
        {"drug": "Lorlatinib", "status": "Approved", "indication": "NSCLC (ALK+)"},
    ],
    "RET": [
        {
            "drug": "Selpercatinib",
            "status": "Approved",
            "indication": "NSCLC, MTC, Thyroid",
        },
        {"drug": "Pralsetinib", "status": "Approved", "indication": "NSCLC, MTC"},
    ],
    "MET": [
        {"drug": "Capmatinib", "status": "Approved", "indication": "NSCLC (MET ex14)"},
        {"drug": "Tepotinib", "status": "Approved", "indication": "NSCLC (MET ex14)"},
    ],
    "FGFR2": [
        {
            "drug": "Pemigatinib",
            "status": "Approved",
            "indication": "Cholangiocarcinoma",
        },
        {
            "drug": "Futibatinib",
            "status": "Approved",
            "indication": "Cholangiocarcinoma",
        },
    ],
}


class DrugTargetMapper:
    """
    Maps identified gene targets to known oncology drugs.

    Parameters
    ----------
    config : ProjectConfig
        Unused currently; reserved for future DrugBank API integration.
    """

    def __init__(self, config: ProjectConfig | None = None) -> None:
        """Initialise the drug-target mapper.

        Parameters
        ----------
        config : ProjectConfig, optional
            Runtime configuration.  Reserved for future DrugBank API
            integration.
        """
        self.cfg = config or ProjectConfig()

    def lookup(self, gene: str) -> list[dict[str, str]]:
        """Return all drug entries for *gene*.

        Parameters
        ----------
        gene : str
            HGNC gene symbol (case-insensitive).

        Returns
        -------
        list[dict[str, str]]
            Each dict has keys ``drug``, ``status``, and ``indication``.
            Empty list if no entries exist.
        """
        return DRUG_TARGET_DB.get(gene.upper(), [])

    def is_druggable(self, gene: str) -> bool:
        """Check whether an approved or late-stage drug targets *gene*.

        Parameters
        ----------
        gene : str
            HGNC gene symbol (case-insensitive).

        Returns
        -------
        bool
            ``True`` if at least one entry has status starting with
            ``"Approved"``.
        """
        entries = self.lookup(gene)
        return any(e["status"].startswith("Approved") for e in entries)

    def _approved_drugs_label(self, gene: str) -> str:
        """Build a semicolon-separated label of approved drugs for *gene*.

        Parameters
        ----------
        gene : str
            HGNC gene symbol (case-insensitive).

        Returns
        -------
        str
            Semicolon-separated approved drug names, or ``"—"`` when none
            are approved.
        """
        approved = [
            e["drug"] for e in self.lookup(gene) if e["status"].startswith("Approved")
        ]
        return "; ".join(approved) or "—"

    def annotate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add drug-target columns to the fusion results DataFrame.

        Adds:
        - ``Druggable`` (bool)
        - ``Approved_Drugs`` (semicolon-separated string)

        Parameters
        ----------
        df : pd.DataFrame
            Must have a ``Gene`` column.

        Returns
        -------
        pd.DataFrame
            Annotated copy.
        """
        df = df.copy()
        df["Druggable"] = df["Gene"].apply(self.is_druggable)
        df["Approved_Drugs"] = df["Gene"].apply(self._approved_drugs_label)
        return df

    def _gene_drug_rows(self, gene: str) -> list[dict[str, str]]:
        """Produce drug-target rows for a single gene.

        Parameters
        ----------
        gene : str
            HGNC gene symbol.

        Returns
        -------
        list[dict[str, str]]
            One dict per drug entry, or a single placeholder row when no
            known drugs exist.
        """
        entries = self.lookup(gene)
        if entries:
            return [{"Gene": gene, **e} for e in entries]
        return [
            {"Gene": gene, "drug": "—", "status": "No known drug", "indication": "—"}
        ]

    def full_report(self, genes: list[str]) -> pd.DataFrame:
        """Produce a detailed drug-target table for a list of genes.

        Parameters
        ----------
        genes : list[str]
            HGNC gene symbols to look up.

        Returns
        -------
        pd.DataFrame
            Columns: ``Gene | Drug | Status | Indication``.
            Genes with no known drug receive a placeholder row.
        """
        rows: list[dict[str, str]] = []
        for g in genes:
            rows.extend(self._gene_drug_rows(g))
        return pd.DataFrame(rows)
