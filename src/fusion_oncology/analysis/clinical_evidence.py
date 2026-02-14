"""
Clinical evidence integration.

Queries public APIs - OpenTargets, CIViC (Clinical Interpretations of
Variants in Cancer), and ClinicalTrials.gov - to attach real-world
clinical evidence to computationally prioritised gene targets.
"""

# fmt: off
from __future__ import annotations

import logging
from typing import Any

import requests

from fusion_oncology.config import ProjectConfig

logger = logging.getLogger(__name__)

_OPENTARGETS_URL = "https://api.platform.opentargets.org/api/v4/graphql"
_CIVIC_URL = "https://civicdb.org/api/graphql"
_CLINICALTRIALS_URL = "https://clinicaltrials.gov/api/v2/studies"

_OT_SEARCH_QUERY = """
query SearchTarget($q: String!) {
  search(queryString: $q, entityNames: ["target"], page: {size: 1, index: 0}) {
    hits { id name }
  }
}
"""

_OT_ASSOC_QUERY = """
query TargetAssoc($id: String!) {
  target(ensemblId: $id) {
    approvedSymbol
    tractability { label modality value }
    associatedDiseases(page: {size: 10, index: 0}) {
      rows {
        disease { name }
        score
        datatypeScores { id score }
      }
    }
  }
}
"""

_CIVIC_EVIDENCE_QUERY = """
query GeneEvidence($name: String!) {
  genes(name: $name) {
    nodes {
      variants {
        nodes {
          name
          evidenceItems {
            nodes {
              disease { name }
              therapies { name }
              evidenceType
              evidenceLevel
              evidenceDirection
              significance
              evidenceRating
            }
          }
        }
      }
    }
  }
}
"""


# -- OpenTargets helpers --------------------------------------------------


def _opentargets_empty(ensembl_id=None):
    """Return an empty OpenTargets result skeleton.

    Parameters
    ----------
    ensembl_id : str or None, optional
        Ensembl gene ID to embed in the result.

    Returns
    -------
    dict[str, Any]
        Dict with neutral defaults for all OpenTargets fields.
    """
    return {"ensembl_id": ensembl_id, "overall_score": 0, "diseases": [], "tractability": {}}


def _post_opentargets(query, variables, gene_symbol, context):
    """POST a GraphQL query to the OpenTargets API.

    Parameters
    ----------
    query : str
        GraphQL query string.
    variables : dict
        GraphQL variables mapping.
    gene_symbol : str
        Gene symbol (used only for log messages).
    context : str
        Description of the query context for logging.

    Returns
    -------
    dict or None
        The ``data`` payload from the response, or ``None`` on failure.
    """
    try:
        payload = {"query": query, "variables": variables}
        resp = requests.post(_OPENTARGETS_URL, json=payload, timeout=15)
        resp.raise_for_status()
        return resp.json()["data"]
    except Exception as exc:
        logger.error("OpenTargets %s failed for %s: %s", context, gene_symbol, exc)
        return None


def _extract_ot_gene_id(data, gene_symbol):
    """Extract the Ensembl gene ID from an OpenTargets search response.

    Parameters
    ----------
    data : dict
        The ``data`` payload from the search response.
    gene_symbol : str
        Gene symbol (for the warning log message).

    Returns
    -------
    str or None
        Ensembl gene ID, or ``None`` when no hits.
    """
    hits = data["search"]["hits"]
    if not hits:
        logger.warning("OpenTargets: no target found for %s", gene_symbol)
        return None
    return hits[0]["id"]


def _resolve_opentargets_gene(gene_symbol):
    """Resolve an HGNC gene symbol to an Ensembl gene ID via OpenTargets.

    Parameters
    ----------
    gene_symbol : str
        HGNC gene symbol (e.g. ``"EGFR"``).

    Returns
    -------
    str or None
        Ensembl gene ID, or ``None`` if resolution fails.
    """
    qs = {"q": gene_symbol}
    data = _post_opentargets(_OT_SEARCH_QUERY, qs, gene_symbol, "search")
    if data is None:
        return None
    return _extract_ot_gene_id(data, gene_symbol)


def _parse_opentargets_target(data, ensembl_id):
    """Parse OpenTargets target data into a structured result dict.

    Parameters
    ----------
    data : dict
        The ``target`` object from the OpenTargets GraphQL response.
    ensembl_id : str
        Ensembl gene ID.

    Returns
    -------
    dict[str, Any]
        Keys: ``ensembl_id``, ``overall_score``, ``diseases``,
        ``tractability``.
    """
    rows = data.get("associatedDiseases", {}).get("rows", [])
    diseases = [{"name": r["disease"]["name"], "score": r["score"]} for r in rows]
    overall = max((d["score"] for d in diseases), default=0)
    tract = {t["label"]: t["value"] for t in data.get("tractability", [])}
    result = {"ensembl_id": ensembl_id, "overall_score": overall}
    result.update(diseases=diseases, tractability=tract)
    return result


def _fetch_opentargets_associations(ensembl_id, gene_symbol):
    """Fetch disease associations for an Ensembl gene ID from OpenTargets.

    Parameters
    ----------
    ensembl_id : str
        Ensembl gene ID.
    gene_symbol : str
        HGNC gene symbol (for logging).

    Returns
    -------
    dict[str, Any]
        Parsed association result, or empty skeleton on failure.
    """
    vs = {"id": ensembl_id}
    data = _post_opentargets(_OT_ASSOC_QUERY, vs, gene_symbol, "association query")
    if data is None or "target" not in data:
        return _opentargets_empty(ensembl_id)
    return _parse_opentargets_target(data["target"], ensembl_id)


def query_opentargets(gene_symbol, config=None):
    """Retrieve disease associations for *gene_symbol* from OpenTargets.

    Uses the OpenTargets Platform GraphQL API to pull the top
    disease associations, overall association score, and tractability
    assessment for the target.

    Parameters
    ----------
    gene_symbol : str
        HGNC gene symbol (e.g. ``"EGFR"``).
    config : ProjectConfig, optional
        Runtime configuration (reserved for future API-key support).

    Returns
    -------
    dict[str, Any]
        Keys: ``ensembl_id``, ``overall_score``, ``diseases``
        (list of dicts with ``name`` and ``score``), ``tractability``.
    """
    ensembl_id = _resolve_opentargets_gene(gene_symbol)
    if ensembl_id is None:
        return _opentargets_empty()
    return _fetch_opentargets_associations(ensembl_id, gene_symbol)


# -- CIViC helpers --------------------------------------------------------


def _civic_record_base(variant_name, ev):
    """Build the base fields for a CIViC evidence record.

    Parameters
    ----------
    variant_name : str
        Name of the variant.
    ev : dict
        Raw evidence-item node from the CIViC GraphQL response.

    Returns
    -------
    dict[str, Any]
        Dict with ``variant``, ``disease``, ``drugs``, ``rating``.
    """
    disease = ev.get("disease", {}).get("name", "")
    drugs = [t["name"] for t in ev.get("therapies", [])]
    rating = ev.get("evidenceRating", 0)
    return {"variant": variant_name, "disease": disease, "drugs": drugs, "rating": rating}


def _civic_record_evidence(ev):
    """Extract evidence classification fields from a CIViC node.

    Parameters
    ----------
    ev : dict
        Raw evidence-item node from the CIViC GraphQL response.

    Returns
    -------
    dict[str, str]
        Dict with ``evidence_type``, ``evidence_level``,
        ``evidence_direction``, ``clinical_significance``.
    """
    return {"evidence_type": ev.get("evidenceType", ""),
            "evidence_level": ev.get("evidenceLevel", ""),
            "evidence_direction": ev.get("evidenceDirection", ""),
            "clinical_significance": ev.get("significance", "")}


def _build_civic_record(variant_name, ev):
    """Build a single CIViC evidence record dict.

    Parameters
    ----------
    variant_name : str
        Name of the variant.
    ev : dict
        Raw evidence-item node from the CIViC GraphQL response.

    Returns
    -------
    dict[str, Any]
        Normalised evidence record with keys ``variant``, ``disease``,
        ``drugs``, ``evidence_type``, ``evidence_level``,
        ``evidence_direction``, ``clinical_significance``, ``rating``.
    """
    rec = _civic_record_base(variant_name, ev)
    rec.update(_civic_record_evidence(ev))
    return rec


def _collect_civic_evidence(genes):
    """Flatten CIViC gene nodes into a list of evidence records.

    Parameters
    ----------
    genes : list
        Gene nodes from the CIViC GraphQL response.

    Returns
    -------
    list[dict[str, Any]]
        Flat list of evidence records.
    """
    items = []
    for variant in genes[0].get("variants", {}).get("nodes", []):
        for ev in variant.get("evidenceItems", {}).get("nodes", []):
            items.append(_build_civic_record(variant["name"], ev))
    return items


def _post_civic(gene_symbol):
    """POST the evidence query to the CIViC GraphQL API.

    Parameters
    ----------
    gene_symbol : str
        HGNC gene symbol.

    Returns
    -------
    list or None
        Gene nodes from the response, or ``None`` on failure.
    """
    try:
        pld = {"query": _CIVIC_EVIDENCE_QUERY, "variables": {"name": gene_symbol}}
        resp = requests.post(_CIVIC_URL, json=pld, timeout=15)
        resp.raise_for_status()
        body = resp.json()
        nodes = body.get("data", {}).get("genes", {}).get("nodes", [])
        if not nodes:
            logger.info("CIViC: no gene entry for %s", gene_symbol)
            return None
        return nodes
    except Exception as exc:
        logger.error("CIViC query failed for %s: %s", gene_symbol, exc)
        return None


def query_civic(gene_symbol, config=None):
    """Retrieve clinical evidence items for *gene_symbol* from CIViC.

    Each returned dict represents a curated clinical assertion linking
    a variant of the gene to a drug response, diagnosis, or prognosis.

    Parameters
    ----------
    gene_symbol : str
        HGNC gene symbol (e.g. ``"BRAF"``).
    config : ProjectConfig, optional
        Runtime configuration (reserved for future use).

    Returns
    -------
    list[dict[str, Any]]
        Each dict has keys: ``variant``, ``disease``, ``drugs``,
        ``evidence_type``, ``evidence_level``, ``evidence_direction``,
        ``clinical_significance``, ``rating``.
    """
    genes = _post_civic(gene_symbol)
    if not genes:
        return []
    items = _collect_civic_evidence(genes)
    logger.info("CIViC: %d evidence items for %s", len(items), gene_symbol)
    return items


# -- ClinicalTrials.gov helpers -------------------------------------------


def _parse_trial_ids(proto):
    """Extract identification and status fields from a protocol section.

    Parameters
    ----------
    proto : dict
        The ``protocolSection`` of a ClinicalTrials.gov study.

    Returns
    -------
    dict[str, Any]
        Dict with ``nct_id``, ``title``, ``status``, ``phase``.
    """
    ident = proto.get("identificationModule", {})
    status_mod = proto.get("statusModule", {})
    design = proto.get("designModule", {})
    return {"nct_id": ident.get("nctId", ""), "title": ident.get("briefTitle", ""),
            "status": status_mod.get("overallStatus", ""), "phase": design.get("phases", [])}


def _parse_trial_details(proto):
    """Extract conditions and interventions from a protocol section.

    Parameters
    ----------
    proto : dict
        The ``protocolSection`` of a ClinicalTrials.gov study.

    Returns
    -------
    dict[str, Any]
        Dict with ``conditions`` and ``interventions``.
    """
    conds = proto.get("conditionsModule", {})
    arms = proto.get("armsInterventionsModule", {})
    interventions = [i.get("name", "") for i in arms.get("interventions", [])]
    return {"conditions": conds.get("conditions", []), "interventions": interventions}


def _parse_trial_study(study):
    """Parse a single ClinicalTrials.gov study record.

    Parameters
    ----------
    study : dict
        Raw study object from the ClinicalTrials.gov API.

    Returns
    -------
    dict[str, Any]
        Keys: ``nct_id``, ``title``, ``status``, ``phase``,
        ``conditions``, ``interventions``.
    """
    proto = study.get("protocolSection", {})
    rec = _parse_trial_ids(proto)
    rec.update(_parse_trial_details(proto))
    return rec


def _fetch_clinical_trials(query_term, max_results, gene_symbol):
    """GET trial data from the ClinicalTrials.gov API.

    Parameters
    ----------
    query_term : str
        Free-text search term.
    max_results : int
        Maximum number of studies to request.
    gene_symbol : str
        Gene symbol (for logging).

    Returns
    -------
    dict or None
        Parsed JSON response body, or ``None`` on failure.
    """
    params = {"query.term": query_term, "pageSize": max_results, "format": "json"}
    try:
        resp = requests.get(_CLINICALTRIALS_URL, params=params, timeout=15)
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        logger.error("ClinicalTrials.gov query failed for %s: %s", gene_symbol, exc)
        return None


def _build_ct_query(gene_symbol, cancer_type):
    """Build the ClinicalTrials.gov free-text query term.

    Parameters
    ----------
    gene_symbol : str
        HGNC gene symbol.
    cancer_type : str or None
        Optional cancer-type filter.

    Returns
    -------
    str
        Query string combining gene symbol and cancer context.
    """
    if cancer_type:
        return f"{gene_symbol} {cancer_type}"
    return f"{gene_symbol} cancer"


def query_clinical_trials(gene_symbol, cancer_type=None, max_results=20, config=None):
    """Search ClinicalTrials.gov for trials targeting *gene_symbol*.

    Parameters
    ----------
    gene_symbol : str
        HGNC gene symbol (e.g. ``"KRAS"``).
    cancer_type : str, optional
        Additional cancer-type filter term (e.g. ``"NSCLC"``).
    max_results : int
        Maximum number of trial records to return.
    config : ProjectConfig, optional
        Runtime configuration (reserved for future use).

    Returns
    -------
    list[dict[str, Any]]
        Each dict has keys: ``nct_id``, ``title``, ``status``,
        ``phase``, ``conditions``, ``interventions``.
    """
    query_term = _build_ct_query(gene_symbol, cancer_type)
    data = _fetch_clinical_trials(query_term, max_results, gene_symbol)
    if data is None:
        return []
    trials = [_parse_trial_study(s) for s in data.get("studies", [])]
    logger.info("ClinicalTrials.gov: %d trials for %s", len(trials), gene_symbol)
    return trials


# -- Aggregator helpers ----------------------------------------------------


def _compute_evidence_score(ot, civic, trials):
    """Compute a composite clinical-evidence score.

    Parameters
    ----------
    ot : dict
        OpenTargets result dict.
    civic : list
        List of CIViC evidence records.
    trials : list
        List of clinical-trial records.

    Returns
    -------
    float
        Weighted evidence score in ``[0, 1]``, rounded to four decimals.
    """
    ot_score = ot.get("overall_score", 0)
    civic_score = min(len(civic) / 10, 1.0)
    trial_score = min(len(trials) / 10, 1.0)
    return round(0.4 * ot_score + 0.3 * civic_score + 0.3 * trial_score, 4)


def _build_evidence_row(gene, prof):
    """Build a single evidence-annotation row for a gene.

    Parameters
    ----------
    gene : str
        HGNC gene symbol.
    prof : dict
        Evidence profile returned by ``ClinicalEvidenceAggregator.profile``.

    Returns
    -------
    dict[str, Any]
        Dict with ``Gene``, ``Evidence_Score``, ``Active_Trials``,
        ``CIViC_Entries``, ``OT_Score``.
    """
    ot_s = prof["opentargets"].get("overall_score", 0)
    return {"Gene": gene, "Evidence_Score": prof["evidence_score"],
            "Active_Trials": len(prof["trials"]),
            "CIViC_Entries": len(prof["civic"]), "OT_Score": ot_s}


def _build_profile_result(gene, ot, civic, trials, score):
    """Assemble the profile result dictionary.

    Parameters
    ----------
    gene : str
        HGNC gene symbol.
    ot : dict
        OpenTargets result.
    civic : list
        CIViC evidence items.
    trials : list
        Clinical trial records.
    score : float
        Composite evidence score.

    Returns
    -------
    dict[str, Any]
        Full evidence profile dictionary.
    """
    return {"gene": gene, "opentargets": ot, "civic": civic,
            "trials": trials, "evidence_score": score}


# -- Aggregator ------------------------------------------------------------


class ClinicalEvidenceAggregator:
    """Aggregate clinical evidence from multiple public APIs.

    Pulls data from OpenTargets, CIViC, and ClinicalTrials.gov and
    produces a unified evidence profile for each gene target.

    Parameters
    ----------
    config : ProjectConfig, optional
        Runtime configuration.
    """

    def __init__(self, config=None):
        """Initialise the aggregator.

        Parameters
        ----------
        config : ProjectConfig, optional
            Runtime configuration.  Falls back to defaults when ``None``.
        """
        self.cfg = config or ProjectConfig()

    def profile(self, gene):
        """Build a full clinical evidence profile for *gene*.

        Parameters
        ----------
        gene : str
            HGNC gene symbol.

        Returns
        -------
        dict[str, Any]
            Keys: ``gene``, ``opentargets`` (from ``query_opentargets``),
            ``civic`` (from ``query_civic``), ``trials``
            (from ``query_clinical_trials``), ``evidence_score``.
        """
        ot = query_opentargets(gene, self.cfg)
        civic = query_civic(gene, self.cfg)
        trials = query_clinical_trials(gene, config=self.cfg)
        score = _compute_evidence_score(ot, civic, trials)
        return _build_profile_result(gene, ot, civic, trials, score)

    def annotate(self, results):
        """Enrich a fusion-results DataFrame with clinical evidence scores.

        Parameters
        ----------
        results : pd.DataFrame
            Must contain a ``Gene`` column.

        Returns
        -------
        pd.DataFrame
            Input with extra columns: ``Evidence_Score``,
            ``Active_Trials``, ``CIViC_Entries``, ``OT_Score``.
        """
        import pandas as pd
        scores = [_build_evidence_row(g, self.profile(g)) for g in results["Gene"]]
        ev_df = pd.DataFrame(scores)
        return results.merge(ev_df, on="Gene", how="left")
