"""
Companion diagnostic mode — per-patient tumour profiling.

Given a patient's tumour sequencing results (mutation list, expression
profile, copy-number data), this module produces a personalised
treatment recommendation report by integrating all analysis layers.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

from fusion_oncology.analysis.drug_target import DrugTargetMapper
from fusion_oncology.analysis.pathway import PathwayEnrichment
from fusion_oncology.analysis.resistance import ResistancePredictor
from fusion_oncology.analysis.synthetic_lethality import SyntheticLethalityDetector
from fusion_oncology.config import ProjectConfig

logger = logging.getLogger(__name__)


class PatientProfile:
    """Structured representation of a single patient's tumour data.

    Parameters
    ----------
    patient_id : str
        Unique patient identifier (e.g. TCGA barcode).
    mutations : list[dict[str, Any]]
        List of somatic mutations, each with at least ``gene`` and
        ``variant`` keys.
    expression : dict[str, float] | None
        Gene-level expression values (TPM or FPKM).
    cna : dict[str, float] | None
        Gene-level copy-number log₂ ratios.
    cancer_type : str
        Cancer type label (e.g. ``"BRCA"``, ``"LUAD"``).
    metadata : dict[str, Any] | None
        Additional clinical metadata (age, stage, prior treatments).
    """

    def __init__(
        self,
        patient_id: str,
        mutations: list[dict[str, Any]],
        expression: dict[str, float] | None = None,
        cna: dict[str, float] | None = None,
        cancer_type: str = "unknown",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Initialise the patient profile.

        Parameters
        ----------
        patient_id : str
            Unique identifier.
        mutations : list[dict[str, Any]]
            Somatic mutation list.
        expression : dict[str, float] | None
            Gene expression values.
        cna : dict[str, float] | None
            Copy-number log₂ ratios.
        cancer_type : str
            Cancer type label.
        metadata : dict[str, Any] | None
            Additional clinical data.
        """
        self.patient_id = patient_id
        self.mutations = mutations
        self.expression = expression or {}
        self.cna = cna or {}
        self.cancer_type = cancer_type
        self.metadata = metadata or {}

    @property
    def mutated_genes(self) -> list[str]:
        """Return deduplicated list of mutated gene symbols.

        Returns
        -------
        list[str]
            Sorted unique gene symbols.
        """
        return sorted({m.get("gene", "") for m in self.mutations if m.get("gene")})

    @property
    def tumour_mutational_burden(self) -> int:
        """Return total number of somatic mutations.

        Returns
        -------
        int
            Count of mutations.
        """
        return len(self.mutations)

    def amplified_genes(self, threshold: float = 0.5) -> list[str]:
        """Return genes with copy-number amplification above *threshold*.

        Parameters
        ----------
        threshold : float
            Minimum log₂ ratio to call amplification.

        Returns
        -------
        list[str]
            Amplified gene symbols.
        """
        return [g for g, v in self.cna.items() if v >= threshold]

    def deleted_genes(self, threshold: float = -0.5) -> list[str]:
        """Return genes with homozygous deletion below *threshold*.

        Parameters
        ----------
        threshold : float
            Maximum log₂ ratio to call deletion.

        Returns
        -------
        list[str]
            Deleted gene symbols.
        """
        return [g for g, v in self.cna.items() if v <= threshold]


class CompanionDiagnostic:
    """Per-patient treatment recommendation engine.

    Integrates mutation profiling, drug matching, pathway analysis,
    synthetic lethality screening, and resistance prediction to
    produce a ranked list of therapeutic options.

    Parameters
    ----------
    config : ProjectConfig, optional
        Runtime configuration.
    """

    def __init__(self, config: ProjectConfig | None = None) -> None:
        """Initialise analysis components.

        Parameters
        ----------
        config : ProjectConfig, optional
            Runtime configuration.
        """
        self.cfg = config or ProjectConfig()
        self._drug_matcher = DrugTargetMapper(self.cfg)
        self._pathway = PathwayEnrichment(self.cfg)
        self._sl_detector = SyntheticLethalityDetector(self.cfg)
        self._resistance = ResistancePredictor(self.cfg)

    def analyse(self, patient: PatientProfile) -> dict[str, Any]:
        """Run the full companion diagnostic analysis for a patient.

        Parameters
        ----------
        patient : PatientProfile
            Patient tumour data.

        Returns
        -------
        dict[str, Any]
            Comprehensive results with keys:
            - ``patient_id``
            - ``cancer_type``
            - ``tmb`` (tumour mutational burden)
            - ``actionable_mutations`` — druggable mutations
            - ``drug_matches`` — matched therapies
            - ``pathway_hits`` — enriched pathways
            - ``synthetic_lethal`` — SL partner opportunities
            - ``resistance_alerts`` — resistance mechanism warnings
            - ``treatment_plan`` — ranked recommendations
            - ``timestamp``
        """
        logger.info("Running companion diagnostic for %s", patient.patient_id)

        # 1. Drug matching
        drug_matches = self._match_drugs(patient)

        # 2. Pathway analysis
        pathway_hits = self._analyse_pathways(patient)

        # 3. Synthetic lethality screening
        sl_hits = self._screen_synthetic_lethality(patient)

        # 4. Resistance alerts
        resistance_alerts = self._check_resistance(patient)

        # 5. Actionable mutation classification
        actionable = self._classify_actionable(patient)

        # 6. Ranked treatment plan
        treatment_plan = self._build_treatment_plan(
            patient, drug_matches, sl_hits, resistance_alerts
        )

        return {
            "patient_id": patient.patient_id,
            "cancer_type": patient.cancer_type,
            "tmb": patient.tumour_mutational_burden,
            "actionable_mutations": actionable,
            "drug_matches": drug_matches,
            "pathway_hits": pathway_hits,
            "synthetic_lethal": sl_hits,
            "resistance_alerts": resistance_alerts,
            "treatment_plan": treatment_plan,
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        }

    def _match_drugs(self, patient: PatientProfile) -> list[dict[str, Any]]:
        """Match patient mutations to approved/investigational drugs.

        Parameters
        ----------
        patient : PatientProfile
            Patient tumour data.

        Returns
        -------
        list[dict[str, Any]]
            Each dict: ``gene``, ``drug``, ``status``, ``indication``.
        """
        matches: list[dict[str, Any]] = []
        for gene in patient.mutated_genes:
            hits = self._drug_matcher.lookup(gene)
            for drug_info in hits:
                matches.append(
                    {
                        "gene": gene,
                        "drug": drug_info["drug"],
                        "status": drug_info["status"],
                        "indication": drug_info["indication"],
                    }
                )

        # Also check amplified genes
        for gene in patient.amplified_genes():
            hits = self._drug_matcher.lookup(gene)
            for drug_info in hits:
                entry = {
                    "gene": gene,
                    "drug": drug_info["drug"],
                    "status": drug_info["status"],
                    "indication": drug_info["indication"],
                    "alteration": "amplification",
                }
                if entry not in matches:
                    matches.append(entry)

        return matches

    def _analyse_pathways(self, patient: PatientProfile) -> list[dict[str, Any]]:
        """Identify cancer pathways hit by patient mutations.

        Parameters
        ----------
        patient : PatientProfile
            Patient tumour data.

        Returns
        -------
        list[dict[str, Any]]
            Each dict: ``pathway``, ``genes_hit``, ``coverage``.
        """
        all_altered = set(patient.mutated_genes) | set(patient.amplified_genes())
        pathway_results: list[dict[str, Any]] = []

        for gene in all_altered:
            pathways = self._pathway.lookup(gene)
            for pw in pathways:
                pathway_results.append({"pathway": pw, "gene": gene})

        # Aggregate
        from collections import Counter

        pw_genes: dict[str, list[str]] = {}
        for hit in pathway_results:
            pw_genes.setdefault(hit["pathway"], []).append(hit["gene"])

        return [
            {
                "pathway": pw,
                "genes_hit": sorted(set(genes)),
                "n_genes_hit": len(set(genes)),
            }
            for pw, genes in pw_genes.items()
        ]

    def _screen_synthetic_lethality(
        self, patient: PatientProfile
    ) -> list[dict[str, Any]]:
        """Screen for synthetic lethal partner genes.

        Parameters
        ----------
        patient : PatientProfile
            Patient tumour data.

        Returns
        -------
        list[dict[str, Any]]
            SL opportunities with ``lost_gene``, ``sl_partner``,
            ``drug_available``.
        """
        sl_hits: list[dict[str, Any]] = []

        # Check deleted genes for SL partners
        lost_genes = set(patient.deleted_genes()) | set(patient.mutated_genes)
        for gene in lost_genes:
            partner_entries = self._sl_detector.known_partners(gene)
            if partner_entries:
                for entry in partner_entries:
                    partner = entry["partner"]
                    # Check if partner has a drug
                    drugs = self._drug_matcher.lookup(partner)
                    sl_hits.append(
                        {
                            "lost_gene": gene,
                            "sl_partner": partner,
                            "drugs_available": [d["drug"] for d in drugs],
                            "rationale": (
                                f"Loss of {gene} creates dependency on "
                                f"{partner} — targeting {partner} may be "
                                f"selectively lethal to tumour."
                            ),
                        }
                    )

        return sl_hits

    def _check_resistance(self, patient: PatientProfile) -> list[dict[str, Any]]:
        """Check for known resistance mechanisms.

        Parameters
        ----------
        patient : PatientProfile
            Patient tumour data.

        Returns
        -------
        list[dict[str, Any]]
            Resistance alerts.
        """
        alerts: list[dict[str, Any]] = []
        seen_genes: set[str] = set()
        for mutation in patient.mutations:
            gene = mutation.get("gene", "")
            variant = mutation.get("variant", "")
            if gene in seen_genes:
                continue
            seen_genes.add(gene)
            mechanisms = self._resistance.predict(gene)
            for mech in mechanisms:
                alerts.append(
                    {
                        "gene": gene,
                        "variant": variant,
                        "mechanism": mech.get("mechanism", ""),
                        "affected_drugs": [mech.get("drug", "")],
                        "strategy": [mech.get("strategy", "")],
                    }
                )
        return alerts

    def _classify_actionable(self, patient: PatientProfile) -> list[dict[str, Any]]:
        """Classify mutations by clinical actionability tier.

        Uses an AMP/ASCO/CAP-inspired tiering system:
        - Tier I: FDA-approved companion diagnostic
        - Tier II: Investigational / off-label evidence
        - Tier III: Preclinical evidence
        - Tier IV: Variant of unknown significance

        Parameters
        ----------
        patient : PatientProfile
            Patient tumour data.

        Returns
        -------
        list[dict[str, Any]]
            Each dict: ``gene``, ``variant``, ``tier``, ``rationale``.
        """
        tiered: list[dict[str, Any]] = []

        for mutation in patient.mutations:
            gene = mutation.get("gene", "")
            variant = mutation.get("variant", "")

            drugs = self._drug_matcher.lookup(gene)
            has_approved = any(d["status"] == "Approved" for d in drugs)
            has_trial = any(d["status"] == "Clinical trial" for d in drugs)

            if has_approved:
                tier = "I"
                rationale = "FDA-approved therapy available."
            elif has_trial:
                tier = "II"
                rationale = "Investigational therapy in clinical trials."
            elif drugs:
                tier = "III"
                rationale = "Preclinical evidence of druggability."
            else:
                tier = "IV"
                rationale = "Variant of unknown significance."

            tiered.append(
                {
                    "gene": gene,
                    "variant": variant,
                    "tier": tier,
                    "rationale": rationale,
                }
            )

        return sorted(tiered, key=lambda x: x["tier"])

    def _build_treatment_plan(
        self,
        patient: PatientProfile,
        drug_matches: list[dict[str, Any]],
        sl_hits: list[dict[str, Any]],
        resistance_alerts: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Build a ranked treatment recommendation list.

        Considers:
        1. Approved targeted therapies matching mutations.
        2. Combination strategies from synthetic lethality.
        3. Immunotherapy eligibility based on TMB.
        4. Resistance-aware drug selection.

        Parameters
        ----------
        patient : PatientProfile
            Patient tumour data.
        drug_matches : list[dict[str, Any]]
            Matched drugs.
        sl_hits : list[dict[str, Any]]
            Synthetic lethality opportunities.
        resistance_alerts : list[dict[str, Any]]
            Known resistance issues.

        Returns
        -------
        list[dict[str, Any]]
            Ranked recommendations with ``rank``, ``therapy``,
            ``type``, ``rationale``, ``confidence``.
        """
        recommendations: list[dict[str, Any]] = []

        # Resistance-affected drugs
        resisted_drugs: set[str] = set()
        for alert in resistance_alerts:
            resisted_drugs.update(alert.get("affected_drugs", []))

        # 1. Approved targeted therapies
        for match in drug_matches:
            if match["status"] == "Approved":
                confidence = 0.9
                if match["drug"] in resisted_drugs:
                    confidence *= 0.3  # penalise
                recommendations.append(
                    {
                        "therapy": match["drug"],
                        "type": "targeted",
                        "target_gene": match["gene"],
                        "rationale": (
                            f"Approved for {match['indication']} targeting "
                            f"{match['gene']}."
                        ),
                        "confidence": round(confidence, 3),
                        "resistance_concern": match["drug"] in resisted_drugs,
                    }
                )

        # 2. SL-based combinations
        for sl in sl_hits:
            if sl["drugs_available"]:
                for drug in sl["drugs_available"]:
                    if drug == "—":
                        continue
                    recommendations.append(
                        {
                            "therapy": drug,
                            "type": "synthetic_lethality",
                            "target_gene": sl["sl_partner"],
                            "rationale": sl["rationale"],
                            "confidence": 0.65,
                            "resistance_concern": False,
                        }
                    )

        # 3. Immunotherapy (if high TMB)
        if patient.tumour_mutational_burden >= 10:
            recommendations.append(
                {
                    "therapy": "Pembrolizumab (anti-PD-1)",
                    "type": "immunotherapy",
                    "target_gene": "TMB-H",
                    "rationale": (
                        f"TMB = {patient.tumour_mutational_burden} mutations "
                        f"(≥10 = TMB-High). FDA-approved agnostic indication."
                    ),
                    "confidence": 0.80,
                    "resistance_concern": False,
                }
            )

        # 4. Investigational therapies
        for match in drug_matches:
            if match["status"] == "Clinical trial":
                recommendations.append(
                    {
                        "therapy": match["drug"],
                        "type": "investigational",
                        "target_gene": match["gene"],
                        "rationale": (
                            f"In clinical trials for {match['indication']} "
                            f"targeting {match['gene']}."
                        ),
                        "confidence": 0.50,
                        "resistance_concern": False,
                    }
                )

        # Deduplicate and rank
        seen: set[str] = set()
        unique: list[dict[str, Any]] = []
        for rec in sorted(recommendations, key=lambda r: r["confidence"], reverse=True):
            key = f"{rec['therapy']}_{rec['target_gene']}"
            if key not in seen:
                seen.add(key)
                unique.append(rec)

        for i, rec in enumerate(unique, 1):
            rec["rank"] = i

        return unique

    def generate_report(self, results: dict[str, Any]) -> str:
        """Generate a human-readable clinical report.

        Parameters
        ----------
        results : dict[str, Any]
            As returned by ``analyse()``.

        Returns
        -------
        str
            Formatted text report.
        """
        lines = [
            "=" * 72,
            "COMPANION DIAGNOSTIC REPORT",
            "=" * 72,
            f"Patient ID:   {results['patient_id']}",
            f"Cancer Type:  {results['cancer_type']}",
            f"TMB:          {results['tmb']} mutations",
            f"Generated:    {results['timestamp']}",
            "",
            "─" * 72,
            "ACTIONABLE MUTATIONS",
            "─" * 72,
        ]

        for mut in results.get("actionable_mutations", []):
            lines.append(
                f"  [{mut['tier']}] {mut['gene']} {mut.get('variant', '')} "
                f"— {mut['rationale']}"
            )

        lines.extend(
            [
                "",
                "─" * 72,
                "TREATMENT RECOMMENDATIONS",
                "─" * 72,
            ]
        )

        for rec in results.get("treatment_plan", []):
            resist_flag = " ⚠ RESISTANCE" if rec.get("resistance_concern") else ""
            lines.append(
                f"  #{rec['rank']}  {rec['therapy']} ({rec['type']})"
                f"  [{rec['confidence']:.0%} confidence]{resist_flag}"
            )
            lines.append(f"       → {rec['rationale']}")

        if results.get("resistance_alerts"):
            lines.extend(
                [
                    "",
                    "─" * 72,
                    "RESISTANCE ALERTS",
                    "─" * 72,
                ]
            )
            for alert in results["resistance_alerts"]:
                lines.append(
                    f"  ⚠ {alert['gene']} {alert['variant']}: " f"{alert['mechanism']}"
                )
                if alert.get("strategy"):
                    lines.append(f"    Strategy: {', '.join(alert['strategy'])}")

        if results.get("synthetic_lethal"):
            lines.extend(
                [
                    "",
                    "─" * 72,
                    "SYNTHETIC LETHALITY OPPORTUNITIES",
                    "─" * 72,
                ]
            )
            for sl in results["synthetic_lethal"]:
                lines.append(f"  {sl['lost_gene']} → {sl['sl_partner']}")
                lines.append(f"    {sl['rationale']}")

        lines.extend(["", "=" * 72])
        return "\n".join(lines)
