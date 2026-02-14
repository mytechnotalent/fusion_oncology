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
    # ── Additional GDSC / kinase targets ─────────────────────────────────
    "FGFR1": [
        {"drug": "Erdafitinib", "status": "Approved", "indication": "Urothelial (FGFR+)"},
        {"drug": "Infigratinib", "status": "Approved", "indication": "Cholangiocarcinoma"},
    ],
    "FGFR3": [
        {"drug": "Erdafitinib", "status": "Approved", "indication": "Urothelial (FGFR+)"},
    ],
    "FGFR4": [
        {"drug": "Futibatinib", "status": "Approved", "indication": "Cholangiocarcinoma"},
    ],
    "JAK1": [
        {"drug": "Ruxolitinib", "status": "Approved", "indication": "Myelofibrosis, PV"},
    ],
    "JAK2": [
        {"drug": "Ruxolitinib", "status": "Approved", "indication": "Myelofibrosis, PV"},
        {"drug": "Fedratinib", "status": "Approved", "indication": "Myelofibrosis"},
    ],
    "ABL1": [
        {"drug": "Imatinib", "status": "Approved", "indication": "CML, GIST"},
        {"drug": "Dasatinib", "status": "Approved", "indication": "CML"},
        {"drug": "Nilotinib", "status": "Approved", "indication": "CML"},
        {"drug": "Ponatinib", "status": "Approved", "indication": "CML (T315I)"},
    ],
    "KIT": [
        {"drug": "Imatinib", "status": "Approved", "indication": "GIST"},
        {"drug": "Sunitinib", "status": "Approved", "indication": "GIST (2nd line)"},
        {"drug": "Ripretinib", "status": "Approved", "indication": "GIST (4th line)"},
        {"drug": "Avapritinib", "status": "Approved", "indication": "GIST (PDGFRA D842V)"},
    ],
    "PDGFRA": [
        {"drug": "Imatinib", "status": "Approved", "indication": "GIST"},
        {"drug": "Avapritinib", "status": "Approved", "indication": "GIST (D842V)"},
    ],
    "PDGFRB": [
        {"drug": "Imatinib", "status": "Approved", "indication": "MDS/MPD"},
    ],
    "AKT1": [
        {"drug": "Capivasertib", "status": "Approved", "indication": "HR+ Breast"},
    ],
    "AKT2": [
        {"drug": "Capivasertib", "status": "Approved", "indication": "HR+ Breast"},
    ],
    "MAP2K1": [
        {"drug": "Trametinib", "status": "Approved", "indication": "Melanoma, NSCLC"},
        {"drug": "Cobimetinib", "status": "Approved", "indication": "Melanoma"},
        {"drug": "Binimetinib", "status": "Approved", "indication": "Melanoma"},
    ],
    "MAP2K2": [
        {"drug": "Trametinib", "status": "Approved", "indication": "Melanoma, NSCLC"},
    ],
    "NRAS": [
        {"drug": "Binimetinib", "status": "Approved", "indication": "Melanoma (NRAS)"},
    ],
    "HDAC1": [
        {"drug": "Vorinostat", "status": "Approved", "indication": "CTCL"},
        {"drug": "Romidepsin", "status": "Approved", "indication": "CTCL, PTCL"},
        {"drug": "Panobinostat", "status": "Approved", "indication": "Multiple Myeloma"},
    ],
    "HDAC2": [
        {"drug": "Vorinostat", "status": "Approved", "indication": "CTCL"},
    ],
    "HDAC3": [
        {"drug": "Vorinostat", "status": "Approved", "indication": "CTCL"},
    ],
    "HDAC6": [
        {"drug": "Ricolinostat", "status": "Phase II", "indication": "Multiple Myeloma"},
    ],
    "BRD4": [
        {"drug": "OTX015", "status": "Phase I", "indication": "Hematologic malignancies"},
    ],
    "PARP1": [
        {"drug": "Olaparib", "status": "Approved", "indication": "Ovarian, Breast, Prostate"},
        {"drug": "Niraparib", "status": "Approved", "indication": "Ovarian"},
        {"drug": "Talazoparib", "status": "Approved", "indication": "Breast (BRCA+)"},
    ],
    "PARP2": [
        {"drug": "Olaparib", "status": "Approved", "indication": "Ovarian, Breast"},
    ],
    "RAF1": [
        {"drug": "Sorafenib", "status": "Approved", "indication": "HCC, RCC"},
    ],
    "ROCK1": [
        {"drug": "Fasudil", "status": "Approved (Japan/China)", "indication": "Cerebral vasospasm"},
        {"drug": "Ripasudil", "status": "Approved (Japan)", "indication": "Glaucoma"},
    ],
    "ROCK2": [
        {"drug": "Belumosudil", "status": "Approved", "indication": "cGVHD"},
    ],
    "AURKA": [
        {"drug": "Alisertib", "status": "Phase III", "indication": "PTCL, Breast"},
    ],
    "AURKB": [
        {"drug": "Barasertib", "status": "Phase II", "indication": "AML"},
    ],
    "AURKC": [
        {
            "drug": "Alisertib",
            "status": "Phase III",
            "indication": "Cross-reactive Aurora kinase inhibitor",
        },
    ],
    "PLK1": [
        {"drug": "Volasertib", "status": "Phase III", "indication": "AML"},
    ],
    "WEE1": [
        {"drug": "Adavosertib", "status": "Phase II", "indication": "Ovarian, solid tumors"},
    ],
    "ATR": [
        {"drug": "Ceralasertib", "status": "Phase II", "indication": "Solid tumors"},
    ],
    "ATM": [
        {"drug": "AZD0156", "status": "Phase I", "indication": "Solid tumors"},
    ],
    "CHEK1": [
        {"drug": "Prexasertib", "status": "Phase II", "indication": "Solid tumors"},
    ],
    "CHEK2": [
        {"drug": "Prexasertib", "status": "Phase II", "indication": "Solid tumors"},
    ],
    "MDM2": [
        {"drug": "Idasanutlin", "status": "Phase III", "indication": "AML (TP53 wild-type)"},
    ],
    "NTRK1": [
        {
            "drug": "Larotrectinib",
            "status": "Approved",
            "indication": "NTRK fusion-positive solid tumors",
        },
        {
            "drug": "Entrectinib",
            "status": "Approved",
            "indication": "NTRK fusion-positive solid tumors",
        },
    ],
    "NTRK2": [
        {"drug": "Larotrectinib", "status": "Approved", "indication": "NTRK fusion-positive"},
    ],
    "NTRK3": [
        {"drug": "Larotrectinib", "status": "Approved", "indication": "NTRK fusion-positive"},
    ],
    "ROS1": [
        {"drug": "Crizotinib", "status": "Approved", "indication": "NSCLC (ROS1+)"},
        {"drug": "Entrectinib", "status": "Approved", "indication": "NSCLC (ROS1+)"},
    ],
    "SRC": [
        {"drug": "Dasatinib", "status": "Approved", "indication": "CML (also SRC inhibitor)"},
        {"drug": "Bosutinib", "status": "Approved", "indication": "CML"},
    ],
    "IGF1R": [
        {
            "drug": "Linsitinib",
            "status": "Phase III (failed)",
            "indication": "Adrenocortical carcinoma",
        },
    ],
    "TGFBR1": [
        {"drug": "Galunisertib", "status": "Phase II", "indication": "HCC, Pancreatic, MDS"},
    ],
    "TGFBR2": [
        {"drug": "Galunisertib", "status": "Phase II", "indication": "HCC, Pancreatic"},
    ],
    "ACVR1": [
        {"drug": "Saracatinib", "status": "Phase II", "indication": "Fibrodysplasia ossificans"},
    ],
    "ACVR1C": [
        {
            "drug": "—",
            "status": "No approved drug",
            "indication": "TGF-β/activin signaling (research target)",
        },
    ],
    "ACVR1B": [
        {
            "drug": "—",
            "status": "No approved drug",
            "indication": "TGF-β/activin signaling (research target)",
        },
    ],
    "AR": [
        {"drug": "Enzalutamide", "status": "Approved", "indication": "Prostate cancer"},
        {"drug": "Abiraterone", "status": "Approved", "indication": "Prostate cancer"},
        {"drug": "Apalutamide", "status": "Approved", "indication": "Prostate cancer"},
        {"drug": "Darolutamide", "status": "Approved", "indication": "Prostate cancer"},
    ],
    "ESR1": [
        {"drug": "Tamoxifen", "status": "Approved", "indication": "ER+ Breast"},
        {"drug": "Fulvestrant", "status": "Approved", "indication": "ER+ Breast"},
        {"drug": "Elacestrant", "status": "Approved", "indication": "ER+ Breast (ESR1 mut)"},
    ],
    "SMO": [
        {"drug": "Vismodegib", "status": "Approved", "indication": "Basal cell carcinoma"},
        {"drug": "Sonidegib", "status": "Approved", "indication": "Basal cell carcinoma"},
    ],
    "PTCH1": [
        {"drug": "Vismodegib", "status": "Approved", "indication": "Basal cell carcinoma"},
    ],
    "FLT3": [
        {"drug": "Midostaurin", "status": "Approved", "indication": "AML (FLT3+)"},
        {"drug": "Gilteritinib", "status": "Approved", "indication": "AML (FLT3+)"},
    ],
    "IDH1": [
        {"drug": "Ivosidenib", "status": "Approved", "indication": "AML, Cholangiocarcinoma"},
    ],
    "IDH2": [
        {"drug": "Enasidenib", "status": "Approved", "indication": "AML (IDH2 mutant)"},
    ],
    "EZH2": [
        {"drug": "Tazemetostat", "status": "Approved", "indication": "Epithelioid sarcoma, FL"},
    ],
    "DNMT1": [
        {"drug": "Azacitidine", "status": "Approved", "indication": "MDS, AML"},
        {"drug": "Decitabine", "status": "Approved", "indication": "MDS, AML"},
    ],
    "DNMT3A": [
        {"drug": "Azacitidine", "status": "Approved", "indication": "MDS, AML"},
    ],
    "XPO1": [
        {"drug": "Selinexor", "status": "Approved", "indication": "Multiple Myeloma, DLBCL"},
    ],
    "TUBB": [
        {"drug": "Paclitaxel", "status": "Approved", "indication": "Breast, Ovarian, NSCLC"},
        {"drug": "Vinblastine", "status": "Approved", "indication": "Hodgkin, Testicular"},
        {"drug": "Eribulin", "status": "Approved", "indication": "Breast, Liposarcoma"},
    ],
    "TOP1": [
        {"drug": "Irinotecan", "status": "Approved", "indication": "CRC"},
        {"drug": "Topotecan", "status": "Approved", "indication": "Ovarian, SCLC"},
    ],
    "TOP2A": [
        {"drug": "Doxorubicin", "status": "Approved", "indication": "Breast, Sarcoma, Lymphoma"},
        {"drug": "Etoposide", "status": "Approved", "indication": "SCLC, Testicular, Lymphoma"},
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
        approved = [e["drug"] for e in self.lookup(gene) if e["status"].startswith("Approved")]
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
        return [{"Gene": gene, "drug": "—", "status": "No known drug", "indication": "—"}]

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
