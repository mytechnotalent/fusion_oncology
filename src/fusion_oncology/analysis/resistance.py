"""
Resistance prediction and drug-resistance mechanism modelling.

Predicts how tumours may develop resistance to targeted therapies by
cataloguing known resistance mutations, alternative pathway activations,
and bypass mechanisms.  Integrates with the Fusion Index to flag
targets at high risk of resistance and suggest combination strategies.
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from fusion_oncology.config import ProjectConfig

logger = logging.getLogger(__name__)


# ── Curated resistance mechanism database ────────────────────────────────

RESISTANCE_DB: dict[str, list[dict[str, str]]] = {
    "EGFR": [
        {
            "drug": "Erlotinib/Gefitinib",
            "mechanism": "T790M gatekeeper mutation",
            "strategy": "Switch to Osimertinib (3rd-gen TKI)",
            "frequency": "~60%",
        },
        {
            "drug": "Osimertinib",
            "mechanism": "C797S mutation",
            "strategy": "Combination with allosteric EGFR inhibitor",
            "frequency": "~10-25%",
        },
        {
            "drug": "Osimertinib",
            "mechanism": "MET amplification bypass",
            "strategy": "Add MET inhibitor (Capmatinib/Tepotinib)",
            "frequency": "~15-20%",
        },
        {
            "drug": "Any EGFR TKI",
            "mechanism": "Histological transformation (SCLC)",
            "strategy": "Switch to platinum-etoposide chemotherapy",
            "frequency": "~5-15%",
        },
    ],
    "BRAF": [
        {
            "drug": "Vemurafenib/Dabrafenib",
            "mechanism": "MAPK pathway reactivation (MEK/ERK)",
            "strategy": "Add MEK inhibitor (Trametinib)",
            "frequency": "~70%",
        },
        {
            "drug": "BRAF + MEK inhibitors",
            "mechanism": "BRAF amplification",
            "strategy": "ERK inhibitor (Ulixertinib)",
            "frequency": "~10-15%",
        },
        {
            "drug": "BRAF + MEK inhibitors",
            "mechanism": "RTK upregulation (EGFR, PDGFR)",
            "strategy": "Add RTK inhibitor",
            "frequency": "~10-20%",
        },
    ],
    "KRAS": [
        {
            "drug": "Sotorasib",
            "mechanism": "KRAS G12C secondary mutations (Y96D)",
            "strategy": "Next-generation KRAS inhibitor",
            "frequency": "~10-20%",
        },
        {
            "drug": "Sotorasib/Adagrasib",
            "mechanism": "KRAS amplification",
            "strategy": "Combination with SHP2 inhibitor",
            "frequency": "~5-10%",
        },
        {
            "drug": "Sotorasib/Adagrasib",
            "mechanism": "Upstream RTK activation (EGFR/FGFR)",
            "strategy": "Add upstream RTK inhibitor",
            "frequency": "~10-15%",
        },
        {
            "drug": "Sotorasib/Adagrasib",
            "mechanism": "PI3K pathway activation",
            "strategy": "Add PI3K inhibitor",
            "frequency": "~5-10%",
        },
    ],
    "ALK": [
        {
            "drug": "Crizotinib",
            "mechanism": "ALK secondary mutations (L1196M, G1269A)",
            "strategy": "Switch to Alectinib or Lorlatinib",
            "frequency": "~30-40%",
        },
        {
            "drug": "Alectinib",
            "mechanism": "ALK G1202R solvent-front mutation",
            "strategy": "Switch to Lorlatinib (3rd-gen ALK TKI)",
            "frequency": "~20-30%",
        },
        {
            "drug": "Lorlatinib",
            "mechanism": "ALK compound mutations",
            "strategy": "Combination therapy or clinical trial",
            "frequency": "~15%",
        },
    ],
    "ERBB2": [
        {
            "drug": "Trastuzumab",
            "mechanism": "PI3K/Akt activation (PIK3CA mutation)",
            "strategy": "Add PI3K inhibitor or switch to T-DXd",
            "frequency": "~20-30%",
        },
        {
            "drug": "Trastuzumab",
            "mechanism": "HER2 extracellular domain shedding",
            "strategy": "Margetuximab or T-DXd (ADC)",
            "frequency": "~15%",
        },
    ],
    "PIK3CA": [
        {
            "drug": "Alpelisib",
            "mechanism": "PTEN loss",
            "strategy": "Add AKT inhibitor (Capivasertib)",
            "frequency": "~20%",
        },
        {
            "drug": "Alpelisib",
            "mechanism": "mTOR activation",
            "strategy": "Add mTOR inhibitor",
            "frequency": "~10-15%",
        },
    ],
    "BCL2": [
        {
            "drug": "Venetoclax",
            "mechanism": "BCL2 G101V mutation",
            "strategy": "MCL1 inhibitor combination",
            "frequency": "~15%",
        },
        {
            "drug": "Venetoclax",
            "mechanism": "MCL1 upregulation",
            "strategy": "MCL1 inhibitor (AMG 176, S64315)",
            "frequency": "~30%",
        },
    ],
    "PDCD1": [
        {
            "drug": "Pembrolizumab/Nivolumab",
            "mechanism": "JAK1/JAK2 loss-of-function",
            "strategy": "Combination with CTLA-4 inhibitor",
            "frequency": "~10%",
        },
        {
            "drug": "Pembrolizumab/Nivolumab",
            "mechanism": "Beta-2-microglobulin (B2M) loss",
            "strategy": "Alternative immunotherapy (CTLA-4, LAG-3)",
            "frequency": "~5-10%",
        },
        {
            "drug": "Pembrolizumab/Nivolumab",
            "mechanism": "WNT/β-catenin pathway activation",
            "strategy": "Combination with anti-VEGF or chemotherapy",
            "frequency": "~25%",
        },
    ],
    "BRCA1": [
        {
            "drug": "Olaparib/Rucaparib",
            "mechanism": "BRCA1 reversion mutation (restores HR)",
            "strategy": "Platinum rechallenge or clinical trial",
            "frequency": "~20-25%",
        },
        {
            "drug": "Olaparib",
            "mechanism": "53BP1/SHIELDIN loss (restores HR)",
            "strategy": "ATR inhibitor combination",
            "frequency": "~10%",
        },
    ],
    "BRCA2": [
        {
            "drug": "Olaparib/Niraparib",
            "mechanism": "BRCA2 reversion mutation",
            "strategy": "Platinum rechallenge or clinical trial",
            "frequency": "~20-25%",
        },
    ],
    "RET": [
        {
            "drug": "Selpercatinib",
            "mechanism": "RET solvent-front mutations (G810R/S/C)",
            "strategy": "Next-generation RET inhibitor",
            "frequency": "~10-20%",
        },
    ],
    "MET": [
        {
            "drug": "Capmatinib/Tepotinib",
            "mechanism": "MET secondary mutations (D1228N/H, Y1230C)",
            "strategy": "Type II MET inhibitor",
            "frequency": "~10-20%",
        },
    ],
    # ── Additional GDSC / kinase targets ─────────────────────────────────
    "FGFR1": [
        {
            "drug": "Erdafitinib",
            "mechanism": "FGFR gatekeeper mutations (V561M)",
            "strategy": "Next-gen FGFR inhibitor or combination",
            "frequency": "~10-15%",
        },
    ],
    "FGFR2": [
        {
            "drug": "Pemigatinib/Futibatinib",
            "mechanism": "FGFR2 kinase domain mutations",
            "strategy": "Covalent FGFR inhibitor (Futibatinib)",
            "frequency": "~10-20%",
        },
    ],
    "FGFR3": [
        {
            "drug": "Erdafitinib",
            "mechanism": "FGFR3 gatekeeper or activation loop mutations",
            "strategy": "Pan-FGFR inhibitor or combination",
            "frequency": "~10-15%",
        },
    ],
    "ABL1": [
        {
            "drug": "Imatinib",
            "mechanism": "BCR-ABL T315I gatekeeper mutation",
            "strategy": "Switch to Ponatinib (3rd-gen TKI)",
            "frequency": "~15-20%",
        },
        {
            "drug": "Dasatinib/Nilotinib",
            "mechanism": "Compound mutations in ABL kinase domain",
            "strategy": "Ponatinib or Asciminib (allosteric inhibitor)",
            "frequency": "~10-15%",
        },
    ],
    "KIT": [
        {
            "drug": "Imatinib",
            "mechanism": "Secondary KIT mutations (exon 13/14/17)",
            "strategy": "Switch to Sunitinib or Ripretinib",
            "frequency": "~40-60%",
        },
    ],
    "JAK2": [
        {
            "drug": "Ruxolitinib",
            "mechanism": "JAK2 persistence / clonal evolution",
            "strategy": "Combination with BET inhibitor or BCL-xL inhibitor",
            "frequency": "~30%",
        },
    ],
    "AR": [
        {
            "drug": "Enzalutamide",
            "mechanism": "AR amplification / AR-V7 splice variant",
            "strategy": "AR degrader (PROTAC) or switch to chemotherapy",
            "frequency": "~20-30%",
        },
        {
            "drug": "Abiraterone",
            "mechanism": "Glucocorticoid receptor upregulation",
            "strategy": "Switch to Enzalutamide or add GR antagonist",
            "frequency": "~10-15%",
        },
    ],
    "ESR1": [
        {
            "drug": "Tamoxifen/Aromatase inhibitors",
            "mechanism": "ESR1 ligand-binding domain mutations (Y537S, D538G)",
            "strategy": "Switch to Elacestrant (oral SERD) or Fulvestrant",
            "frequency": "~20-40%",
        },
    ],
    "FLT3": [
        {
            "drug": "Midostaurin/Gilteritinib",
            "mechanism": "FLT3 on-target resistance mutations (F691L)",
            "strategy": "Next-gen FLT3 inhibitor or combination",
            "frequency": "~10-20%",
        },
    ],
    "IDH1": [
        {
            "drug": "Ivosidenib",
            "mechanism": "IDH1 second-site mutations or IDH2 switching",
            "strategy": "Dual IDH1/2 inhibitor or combination",
            "frequency": "~10%",
        },
    ],
    "AKT1": [
        {
            "drug": "Capivasertib",
            "mechanism": "PTEN loss or mTORC1 reactivation",
            "strategy": "AKT + mTOR inhibitor combination",
            "frequency": "~15-20%",
        },
    ],
    "NTRK1": [
        {
            "drug": "Larotrectinib",
            "mechanism": "NTRK solvent-front mutations (G595R, G667C)",
            "strategy": "Switch to Selitrectinib (2nd-gen TRK inhibitor)",
            "frequency": "~10-20%",
        },
    ],
    "ROS1": [
        {
            "drug": "Crizotinib",
            "mechanism": "ROS1 G2032R solvent-front mutation",
            "strategy": "Switch to Lorlatinib or Repotrectinib",
            "frequency": "~30-40%",
        },
    ],
    "SMO": [
        {
            "drug": "Vismodegib",
            "mechanism": "SMO D473H mutation",
            "strategy": "Downstream GLI inhibitor or itraconazole",
            "frequency": "~15-20%",
        },
    ],
    "MAP2K1": [
        {
            "drug": "Trametinib/Cobimetinib",
            "mechanism": "MEK1 mutations or ERK reactivation",
            "strategy": "ERK inhibitor combination",
            "frequency": "~15-25%",
        },
    ],
}


class ResistancePredictor:
    """Predict and catalogue drug resistance mechanisms for gene targets.

    Uses a curated knowledge base of clinically observed resistance
    mechanisms and suggests mitigation strategies including combination
    therapies and next-generation inhibitors.

    Parameters
    ----------
    config : ProjectConfig, optional
        Runtime configuration.
    """

    def __init__(self, config: ProjectConfig | None = None) -> None:
        """Initialise the resistance predictor.

        Parameters
        ----------
        config : ProjectConfig, optional
            Runtime configuration.  Falls back to defaults when ``None``.
        """
        self.cfg = config or ProjectConfig()

    def predict(self, gene: str) -> list[dict[str, str]]:
        """Retrieve known resistance mechanisms for *gene*.

        Parameters
        ----------
        gene : str
            HGNC gene symbol.

        Returns
        -------
        list[dict[str, str]]
            Each dict has keys: ``drug``, ``mechanism``, ``strategy``,
            ``frequency``.
        """
        mechanisms = RESISTANCE_DB.get(gene, [])
        if not mechanisms:
            logger.info("No known resistance mechanisms catalogued for %s", gene)
        return mechanisms

    def resistance_risk_score(self, gene: str) -> float:
        """Compute a resistance risk score for *gene*.

        Higher scores indicate more known resistance mechanisms and
        thus greater clinical risk of treatment failure.

        Parameters
        ----------
        gene : str
            HGNC gene symbol.

        Returns
        -------
        float
            Score in ``[0, 1]`` range.  1.0 = maximum documented
            resistance risk.
        """
        mechanisms = self.predict(gene)
        if not mechanisms:
            return 0.0

        # Weight by number of mechanisms, cap at 1.0
        n = len(mechanisms)
        return min(1.0, n * 0.2)

    def _annotate_gene(
        self,
        gene: str,
    ) -> tuple[float, int, str]:
        """Return resistance annotation fields for a single gene.

        Parameters
        ----------
        gene : str
            HGNC gene symbol.

        Returns
        -------
        tuple[float, int, str]
            ``(risk_score, n_mechanisms, top_mechanism)``.
        """
        mechanisms = self.predict(gene)
        risk = self.resistance_risk_score(gene)
        top = mechanisms[0]["mechanism"] if mechanisms else "\u2014"
        return risk, len(mechanisms), top

    def annotate(self, results: pd.DataFrame) -> pd.DataFrame:
        """Enrich fusion results with resistance risk information.

        Parameters
        ----------
        results : pd.DataFrame
            Must contain a ``Gene`` column.

        Returns
        -------
        pd.DataFrame
            Input with extra columns: ``Resistance_Risk``,
            ``N_Resistance_Mechanisms``, ``Top_Resistance_Mechanism``.
        """
        rows = [self._annotate_gene(g) for g in results["Gene"]]
        out = results.copy()
        out["Resistance_Risk"] = [r[0] for r in rows]
        out["N_Resistance_Mechanisms"] = [r[1] for r in rows]
        out["Top_Resistance_Mechanism"] = [r[2] for r in rows]
        return out

    def _mechanism_row(
        self,
        gene: str,
        m: dict[str, str],
        risk: float,
    ) -> dict[str, Any]:
        """Build a single report row for one resistance mechanism.

        Parameters
        ----------
        gene : str
            HGNC gene symbol.
        m : dict[str, str]
            Single mechanism entry from the curated database.
        risk : float
            Pre-computed resistance risk score.

        Returns
        -------
        dict[str, Any]
            Report row dict.
        """
        return {
            "Gene": gene,
            "Drug": m["drug"],
            "Mechanism": m["mechanism"],
            "Strategy": m["strategy"],
            "Frequency": m["frequency"],
            "Risk_Score": risk,
        }

    def _mechanism_rows(
        self,
        gene: str,
        mechanisms: list[dict[str, str]],
        risk: float,
    ) -> list[dict[str, Any]]:
        """Build report rows for a gene with known resistance mechanisms.

        Parameters
        ----------
        gene : str
            HGNC gene symbol.
        mechanisms : list[dict[str, str]]
            Known resistance mechanisms from the curated database.
        risk : float
            Pre-computed resistance risk score.

        Returns
        -------
        list[dict[str, Any]]
            One row dict per mechanism.
        """
        return [self._mechanism_row(gene, m, risk) for m in mechanisms]

    def _empty_mechanism_row(
        self,
        gene: str,
    ) -> dict[str, Any]:
        """Build a placeholder row for a gene with no known mechanisms.

        Parameters
        ----------
        gene : str
            HGNC gene symbol.

        Returns
        -------
        dict[str, Any]
            Row with placeholder values.
        """
        return {
            "Gene": gene,
            "Drug": "\u2014",
            "Mechanism": "None catalogued",
            "Strategy": "\u2014",
            "Frequency": "\u2014",
            "Risk_Score": 0.0,
        }

    def full_report(self, genes: list[str]) -> pd.DataFrame:
        """Generate a comprehensive resistance report for multiple genes.

        Parameters
        ----------
        genes : list[str]
            HGNC gene symbols.

        Returns
        -------
        pd.DataFrame
            All known resistance mechanisms for the input genes with
            columns: ``Gene``, ``Drug``, ``Mechanism``, ``Strategy``,
            ``Frequency``, ``Risk_Score``.
        """
        rows: list[dict[str, Any]] = []
        for gene in genes:
            mechs = self.predict(gene)
            risk = self.resistance_risk_score(gene)
            ext = self._mechanism_rows(gene, mechs, risk)
            rows.extend(ext or [self._empty_mechanism_row(gene)])
        return pd.DataFrame(rows)
