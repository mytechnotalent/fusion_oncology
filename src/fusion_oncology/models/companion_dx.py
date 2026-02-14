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

    # -- helpers for _match_drugs ------------------------------------

    def _match_mutated_drugs(
        self,
        patient: PatientProfile,
    ) -> list[dict[str, Any]]:
        """Collect drug matches for each mutated gene.

        Parameters
        ----------
        patient : PatientProfile
            Patient tumour data.

        Returns
        -------
        list[dict[str, Any]]
            Drug match dicts with ``gene``, ``drug``, ``status``,
            ``indication``.
        """
        matches: list[dict[str, Any]] = []
        for gene in patient.mutated_genes:
            for info in self._drug_matcher.lookup(gene):
                matches.append(
                    {
                        "gene": gene,
                        "drug": info["drug"],
                        "status": info["status"],
                        "indication": info["indication"],
                    }
                )
        return matches

    def _match_amplified_drugs(
        self,
        patient: PatientProfile,
        matches: list[dict[str, Any]],
    ) -> None:
        """Append drug matches for amplified genes in-place.

        Parameters
        ----------
        patient : PatientProfile
            Patient tumour data.
        matches : list[dict[str, Any]]
            Existing match list to extend (mutated in place).
        """
        for gene in patient.amplified_genes():
            for info in self._drug_matcher.lookup(gene):
                entry = {
                    "gene": gene,
                    "drug": info["drug"],
                    "status": info["status"],
                    "indication": info["indication"],
                    "alteration": "amplification",
                }
                if entry not in matches:
                    matches.append(entry)

    def _match_drugs(
        self,
        patient: PatientProfile,
    ) -> list[dict[str, Any]]:
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
        matches = self._match_mutated_drugs(patient)
        self._match_amplified_drugs(patient, matches)
        return matches

    # -- helpers for _analyse_pathways --------------------------------

    def _collect_pathway_hits(
        self,
        patient: PatientProfile,
    ) -> list[dict[str, Any]]:
        """Collect raw pathway hits for every altered gene.

        Parameters
        ----------
        patient : PatientProfile
            Patient tumour data.

        Returns
        -------
        list[dict[str, Any]]
            Raw hits with ``pathway`` and ``gene`` keys.
        """
        altered = set(patient.mutated_genes) | set(patient.amplified_genes())
        hits: list[dict[str, Any]] = []
        for gene in altered:
            for pw in self._pathway.lookup(gene):
                hits.append({"pathway": pw, "gene": gene})
        return hits

    def _aggregate_pathways(
        self,
        hits: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Aggregate raw pathway hits into pathway summaries.

        Parameters
        ----------
        hits : list[dict[str, Any]]
            Raw pathway hits.

        Returns
        -------
        list[dict[str, Any]]
            Each dict: ``pathway``, ``genes_hit``, ``n_genes_hit``.
        """
        pw_genes: dict[str, list[str]] = {}
        for hit in hits:
            pw_genes.setdefault(hit["pathway"], []).append(hit["gene"])
        return [
            {"pathway": pw, "genes_hit": sorted(set(g)), "n_genes_hit": len(set(g))}
            for pw, g in pw_genes.items()
        ]

    def _analyse_pathways(
        self,
        patient: PatientProfile,
    ) -> list[dict[str, Any]]:
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
        hits = self._collect_pathway_hits(patient)
        return self._aggregate_pathways(hits)

    # -- helpers for _screen_synthetic_lethality ---------------------

    def _build_sl_hit(
        self,
        gene: str,
        partner: str,
    ) -> dict[str, Any]:
        """Build a single synthetic-lethality hit record.

        Parameters
        ----------
        gene : str
            Lost gene symbol.
        partner : str
            Synthetic-lethal partner gene.

        Returns
        -------
        dict[str, Any]
            Record with ``lost_gene``, ``sl_partner``,
            ``drugs_available``, ``rationale``.
        """
        drugs = self._drug_matcher.lookup(partner)
        return {
            "lost_gene": gene,
            "sl_partner": partner,
            "drugs_available": [d["drug"] for d in drugs],
            "rationale": (
                f"Loss of {gene} creates dependency on "
                f"{partner} — targeting {partner} may be "
                f"selectively lethal to tumour."
            ),
        }

    def _screen_gene_sl(
        self,
        gene: str,
    ) -> list[dict[str, Any]]:
        """Screen one gene for synthetic-lethal partners.

        Parameters
        ----------
        gene : str
            Gene symbol.

        Returns
        -------
        list[dict[str, Any]]
            SL hit records.
        """
        hits: list[dict[str, Any]] = []
        for entry in self._sl_detector.known_partners(gene):
            hits.append(self._build_sl_hit(gene, entry["partner"]))
        return hits

    def _screen_synthetic_lethality(
        self,
        patient: PatientProfile,
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
        lost = set(patient.deleted_genes()) | set(patient.mutated_genes)
        hits: list[dict[str, Any]] = []
        for gene in lost:
            hits.extend(self._screen_gene_sl(gene))
        return hits

    # -- helpers for _check_resistance --------------------------------

    def _check_gene_resistance(
        self,
        gene: str,
        variant: str,
    ) -> list[dict[str, Any]]:
        """Check a single gene for resistance mechanisms.

        Parameters
        ----------
        gene : str
            Gene symbol.
        variant : str
            Variant identifier.

        Returns
        -------
        list[dict[str, Any]]
            Resistance alert dicts.
        """
        alerts: list[dict[str, Any]] = []
        for mech in self._resistance.predict(gene):
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

    def _check_resistance(
        self,
        patient: PatientProfile,
    ) -> list[dict[str, Any]]:
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
        alerts, seen = [], set()
        for mutation in patient.mutations:
            gene = mutation.get("gene", "")
            if gene not in seen:
                seen.add(gene)
                alerts.extend(
                    self._check_gene_resistance(gene, mutation.get("variant", ""))
                )
        return alerts

    # -- helpers for _classify_actionable -----------------------------

    def _determine_tier(
        self,
        drugs: list[dict[str, Any]],
    ) -> tuple[str, str]:
        """Determine actionability tier from drug lookup results.

        Parameters
        ----------
        drugs : list[dict[str, Any]]
            Drug entries from the mapper.

        Returns
        -------
        tuple[str, str]
            ``(tier, rationale)`` pair.
        """
        if any(d["status"] == "Approved" for d in drugs):
            return "I", "FDA-approved therapy available."
        if any(d["status"] == "Clinical trial" for d in drugs):
            return "II", "Investigational therapy in clinical trials."
        if drugs:
            return "III", "Preclinical evidence of druggability."
        return "IV", "Variant of unknown significance."

    def _classify_mutation(
        self,
        mutation: dict[str, Any],
    ) -> dict[str, Any]:
        """Classify a single mutation by actionability tier.

        Parameters
        ----------
        mutation : dict[str, Any]
            Mutation dict with ``gene`` and ``variant`` keys.

        Returns
        -------
        dict[str, Any]
            Tier record with ``gene``, ``variant``, ``tier``,
            ``rationale``.
        """
        gene = mutation.get("gene", "")
        variant = mutation.get("variant", "")
        drugs = self._drug_matcher.lookup(gene)
        tier, rationale = self._determine_tier(drugs)
        return {"gene": gene, "variant": variant, "tier": tier, "rationale": rationale}

    def _classify_actionable(
        self,
        patient: PatientProfile,
    ) -> list[dict[str, Any]]:
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
        tiered = [self._classify_mutation(m) for m in patient.mutations]
        return sorted(tiered, key=lambda x: x["tier"])

    # -- helpers for _build_treatment_plan ----------------------------

    def _build_rec(
        self,
        therapy: str,
        rtype: str,
        gene: str,
        rationale: str,
        conf: float,
        resist: bool,
    ) -> dict[str, Any]:
        """Build a single treatment recommendation dict.

        Parameters
        ----------
        therapy : str
            Drug or therapy name.
        rtype : str
            Recommendation type (e.g. ``"targeted"``).
        gene : str
            Target gene symbol.
        rationale : str
            Clinical rationale text.
        conf : float
            Confidence score.
        resist : bool
            Whether a resistance concern exists.

        Returns
        -------
        dict[str, Any]
            Recommendation record.
        """
        return {
            "therapy": therapy,
            "type": rtype,
            "target_gene": gene,
            "rationale": rationale,
            "confidence": conf,
            "resistance_concern": resist,
        }

    def _resisted_drug_set(
        self,
        resistance_alerts: list[dict[str, Any]],
    ) -> set[str]:
        """Collect drug names affected by resistance mechanisms.

        Parameters
        ----------
        resistance_alerts : list[dict[str, Any]]
            Resistance alerts.

        Returns
        -------
        set[str]
            Drug names with known resistance.
        """
        resisted: set[str] = set()
        for alert in resistance_alerts:
            resisted.update(alert.get("affected_drugs", []))
        return resisted

    def _approved_recs(
        self,
        drug_matches: list[dict[str, Any]],
        resisted: set[str],
    ) -> list[dict[str, Any]]:
        """Build recommendations for approved targeted therapies.

        Parameters
        ----------
        drug_matches : list[dict[str, Any]]
            All drug matches.
        resisted : set[str]
            Drugs with resistance concern.

        Returns
        -------
        list[dict[str, Any]]
            Approved-therapy recommendations.
        """
        recs: list[dict[str, Any]] = []
        approved = [m for m in drug_matches if m["status"] == "Approved"]
        for m in approved:
            conf = 0.27 if m["drug"] in resisted else 0.9
            rat = f"Approved for {m['indication']}" f" targeting {m['gene']}."
            recs.append(
                self._build_rec(
                    m["drug"], "targeted", m["gene"], rat, conf, m["drug"] in resisted
                )
            )
        return recs

    def _sl_recs(
        self,
        sl_hits: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Build recommendations from synthetic-lethality hits.

        Parameters
        ----------
        sl_hits : list[dict[str, Any]]
            SL opportunities.

        Returns
        -------
        list[dict[str, Any]]
            SL-based recommendations.
        """
        recs: list[dict[str, Any]] = []
        for sl in sl_hits:
            for drug in sl.get("drugs_available", []):
                if drug != "—":
                    recs.append(
                        self._build_rec(
                            drug,
                            "synthetic_lethality",
                            sl["sl_partner"],
                            sl["rationale"],
                            0.65,
                            False,
                        )
                    )
        return recs

    def _immunotherapy_rec(
        self,
        patient: PatientProfile,
    ) -> list[dict[str, Any]]:
        """Build immunotherapy recommendation if TMB is high.

        Parameters
        ----------
        patient : PatientProfile
            Patient tumour data.

        Returns
        -------
        list[dict[str, Any]]
            Zero or one immunotherapy recommendation.
        """
        if patient.tumour_mutational_burden < 10:
            return []
        tmb = patient.tumour_mutational_burden
        rat = (
            f"TMB = {tmb} mutations "
            f"(≥10 = TMB-High). FDA-approved agnostic indication."
        )
        return [
            self._build_rec(
                "Pembrolizumab (anti-PD-1)", "immunotherapy", "TMB-H", rat, 0.80, False
            )
        ]

    def _investigational_recs(
        self,
        drug_matches: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Build recommendations for investigational therapies.

        Parameters
        ----------
        drug_matches : list[dict[str, Any]]
            All drug matches.

        Returns
        -------
        list[dict[str, Any]]
            Investigational-therapy recommendations.
        """
        recs: list[dict[str, Any]] = []
        for m in drug_matches:
            if m["status"] == "Clinical trial":
                rat = (
                    f"In clinical trials for {m['indication']} "
                    f"targeting {m['gene']}."
                )
                recs.append(
                    self._build_rec(
                        m["drug"], "investigational", m["gene"], rat, 0.50, False
                    )
                )
        return recs

    def _deduplicate_and_rank(
        self,
        recs: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Deduplicate recommendations and assign rank numbers.

        Parameters
        ----------
        recs : list[dict[str, Any]]
            Unranked recommendations (may have duplicates).

        Returns
        -------
        list[dict[str, Any]]
            Deduplicated, confidence-sorted, ranked list.
        """
        seen, unique = set(), []
        for rec in sorted(recs, key=lambda r: r["confidence"], reverse=True):
            key = f"{rec['therapy']}_{rec['target_gene']}"
            if key not in seen:
                seen.add(key)
                rec["rank"] = len(unique) + 1
                unique.append(rec)
        return unique

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
        resisted = self._resisted_drug_set(resistance_alerts)
        recs = self._approved_recs(drug_matches, resisted)
        recs.extend(self._sl_recs(sl_hits))
        recs.extend(self._immunotherapy_rec(patient))
        recs.extend(self._investigational_recs(drug_matches))
        return self._deduplicate_and_rank(recs)

    # -- helper for analyse ------------------------------------------

    def _assemble_result(
        self,
        patient: PatientProfile,
        drugs: list[dict[str, Any]],
        paths: list[dict[str, Any]],
        sl: list[dict[str, Any]],
        resist: list[dict[str, Any]],
        action: list[dict[str, Any]],
        plan: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Assemble the final analysis result dictionary.

        Parameters
        ----------
        patient : PatientProfile
            Patient tumour data.
        drugs : list[dict[str, Any]]
            Drug matches.
        paths : list[dict[str, Any]]
            Pathway hits.
        sl : list[dict[str, Any]]
            Synthetic-lethality opportunities.
        resist : list[dict[str, Any]]
            Resistance alerts.
        action : list[dict[str, Any]]
            Actionable mutation classifications.
        plan : list[dict[str, Any]]
            Ranked treatment plan.

        Returns
        -------
        dict[str, Any]
            Complete analysis result.
        """
        return {
            "patient_id": patient.patient_id,
            "cancer_type": patient.cancer_type,
            "tmb": patient.tumour_mutational_burden,
            "actionable_mutations": action,
            "drug_matches": drugs,
            "pathway_hits": paths,
            "synthetic_lethal": sl,
            "resistance_alerts": resist,
            "treatment_plan": plan,
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        }

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
        drugs = self._match_drugs(patient)
        paths = self._analyse_pathways(patient)
        sl = self._screen_synthetic_lethality(patient)
        resist = self._check_resistance(patient)
        action = self._classify_actionable(patient)
        plan = self._build_treatment_plan(patient, drugs, sl, resist)
        return self._assemble_result(patient, drugs, paths, sl, resist, action, plan)

    # -- helpers for generate_report ----------------------------------

    def _report_header(
        self,
        results: dict[str, Any],
    ) -> list[str]:
        """Build report header lines.

        Parameters
        ----------
        results : dict[str, Any]
            Analysis results.

        Returns
        -------
        list[str]
            Header lines.
        """
        return [
            "=" * 72,
            "COMPANION DIAGNOSTIC REPORT",
            "=" * 72,
            f"Patient ID:   {results['patient_id']}",
            f"Cancer Type:  {results['cancer_type']}",
            f"TMB:          {results['tmb']} mutations",
            f"Generated:    {results['timestamp']}",
        ]

    def _report_actionable(
        self,
        results: dict[str, Any],
    ) -> list[str]:
        """Build actionable-mutations section lines.

        Parameters
        ----------
        results : dict[str, Any]
            Analysis results.

        Returns
        -------
        list[str]
            Actionable mutation report lines.
        """
        lines = ["", "─" * 72, "ACTIONABLE MUTATIONS", "─" * 72]
        for mut in results.get("actionable_mutations", []):
            lines.append(
                f"  [{mut['tier']}] {mut['gene']} "
                f"{mut.get('variant', '')} — {mut['rationale']}"
            )
        return lines

    def _report_treatments(
        self,
        results: dict[str, Any],
    ) -> list[str]:
        """Build treatment-recommendations section lines.

        Parameters
        ----------
        results : dict[str, Any]
            Analysis results.

        Returns
        -------
        list[str]
            Treatment recommendation lines.
        """
        lines = ["", "─" * 72, "TREATMENT RECOMMENDATIONS", "─" * 72]
        for rec in results.get("treatment_plan", []):
            flag = " ⚠ RESISTANCE" if rec.get("resistance_concern") else ""
            lines.append(
                f"  #{rec['rank']}  {rec['therapy']} ({rec['type']})"
                f"  [{rec['confidence']:.0%} confidence]{flag}"
            )
            lines.append(f"       → {rec['rationale']}")
        return lines

    def _report_resistance_alerts(
        self,
        results: dict[str, Any],
    ) -> list[str]:
        """Build resistance-alerts section lines.

        Parameters
        ----------
        results : dict[str, Any]
            Analysis results.

        Returns
        -------
        list[str]
            Resistance alert lines, empty list if none.
        """
        if not results.get("resistance_alerts"):
            return []
        lines = ["", "─" * 72, "RESISTANCE ALERTS", "─" * 72]
        for a in results["resistance_alerts"]:
            lines.append(f"  ⚠ {a['gene']} {a['variant']}: " f"{a['mechanism']}")
            if a.get("strategy"):
                lines.append(f"    Strategy: {', '.join(a['strategy'])}")
        return lines

    def _report_sl_opportunities(
        self,
        results: dict[str, Any],
    ) -> list[str]:
        """Build synthetic-lethality opportunities section lines.

        Parameters
        ----------
        results : dict[str, Any]
            Analysis results.

        Returns
        -------
        list[str]
            SL opportunity lines, empty list if none.
        """
        if not results.get("synthetic_lethal"):
            return []
        lines = ["", "─" * 72, "SYNTHETIC LETHALITY OPPORTUNITIES", "─" * 72]
        for sl in results["synthetic_lethal"]:
            lines.append(f"  {sl['lost_gene']} → {sl['sl_partner']}")
            lines.append(f"    {sl['rationale']}")
        return lines

    def _report_footer(self) -> list[str]:
        """Build report footer lines.

        Returns
        -------
        list[str]
            Footer lines.
        """
        return ["", "=" * 72]

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
        lines = self._report_header(results)
        lines.extend(self._report_actionable(results))
        lines.extend(self._report_treatments(results))
        lines.extend(self._report_resistance_alerts(results))
        lines.extend(self._report_sl_opportunities(results))
        lines.extend(self._report_footer())
        return "\n".join(lines)
