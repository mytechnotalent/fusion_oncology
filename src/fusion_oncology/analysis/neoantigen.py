"""
Neoantigen prediction from somatic mutations.

Predicts candidate neoantigens by analysing missense mutations,
generating mutant peptide sequences, and estimating MHC-I binding
affinity using a simplified position-specific scoring matrix.
Real clinical pipelines use NetMHCpan or MHCflurry — this module
provides a lightweight approximation for rapid screening.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from fusion_oncology.config import ProjectConfig

logger = logging.getLogger(__name__)

# ── Genetic code ────────────────────────────────────────────────────────

CODON_TABLE: dict[str, str] = {
    "TTT": "F",
    "TTC": "F",
    "TTA": "L",
    "TTG": "L",
    "CTT": "L",
    "CTC": "L",
    "CTA": "L",
    "CTG": "L",
    "ATT": "I",
    "ATC": "I",
    "ATA": "I",
    "ATG": "M",
    "GTT": "V",
    "GTC": "V",
    "GTA": "V",
    "GTG": "V",
    "TCT": "S",
    "TCC": "S",
    "TCA": "S",
    "TCG": "S",
    "CCT": "P",
    "CCC": "P",
    "CCA": "P",
    "CCG": "P",
    "ACT": "T",
    "ACC": "T",
    "ACA": "T",
    "ACG": "T",
    "GCT": "A",
    "GCC": "A",
    "GCA": "A",
    "GCG": "A",
    "TAT": "Y",
    "TAC": "Y",
    "TAA": "*",
    "TAG": "*",
    "CAT": "H",
    "CAC": "H",
    "CAA": "Q",
    "CAG": "Q",
    "AAT": "N",
    "AAC": "N",
    "AAA": "K",
    "AAG": "K",
    "GAT": "D",
    "GAC": "D",
    "GAA": "E",
    "GAG": "E",
    "TGT": "C",
    "TGC": "C",
    "TGA": "*",
    "TGG": "W",
    "CGT": "R",
    "CGC": "R",
    "CGA": "R",
    "CGG": "R",
    "AGT": "S",
    "AGC": "S",
    "AGA": "R",
    "AGG": "R",
    "GGT": "G",
    "GGC": "G",
    "GGA": "G",
    "GGG": "G",
}

# Simplified MHC-I binding preference matrix (HLA-A*02:01-like)
# Higher values indicate amino acids preferred at that peptide position.
# Real pipelines use NetMHCpan — this is a teaching/screening proxy.
_MHC_PREF: dict[int, dict[str, float]] = {
    0: {"L": 0.8, "M": 0.7, "I": 0.6, "V": 0.5, "A": 0.4},
    1: {"L": 0.9, "M": 0.8, "V": 0.7, "I": 0.6, "A": 0.5},
    8: {"V": 0.9, "L": 0.8, "I": 0.7, "A": 0.6, "T": 0.5},
}


def translate_sequence(dna: str) -> str:
    """Translate a DNA coding sequence to a protein string.

    Parameters
    ----------
    dna : str
        DNA sequence (must be a multiple of 3 in length).

    Returns
    -------
    str
        Single-letter amino-acid string.  Stop codons are represented
        as ``*``.
    """
    dna = dna.upper()
    protein: list[str] = []
    for i in range(0, len(dna) - 2, 3):
        aa = CODON_TABLE.get(dna[i : i + 3], "X")
        if aa == "*":
            break
        protein.append(aa)
    return "".join(protein)


# ── Helpers for generate_mutant_peptides ────────────────────────────────


def _build_mutant_sequence(
    wt_sequence: str,
    mutation_position: int,
    mutant_aa: str,
) -> str:
    """Build a mutant protein sequence by substituting one residue.

    Parameters
    ----------
    wt_sequence : str
        Wild-type protein sequence (single-letter amino acids).
    mutation_position : int
        Zero-based index of the residue to mutate.
    mutant_aa : str
        Single-letter code of the replacement amino acid.

    Returns
    -------
    str
        Protein sequence with the substitution applied.
    """
    mut_seq = list(wt_sequence)
    if mutation_position < len(mut_seq):
        mut_seq[mutation_position] = mutant_aa
    return "".join(mut_seq)


def _peptide_window_range(
    mutation_position: int,
    plen: int,
    seq_len: int,
) -> range:
    """Compute the range of valid peptide window start positions.

    Parameters
    ----------
    mutation_position : int
        Zero-based position of the mutated residue.
    plen : int
        Desired peptide length.
    seq_len : int
        Total length of the protein sequence.

    Returns
    -------
    range
        Iterable of valid window start positions.
    """
    start_min = max(0, mutation_position - plen + 1)
    start_max = min(mutation_position, seq_len - plen)
    return range(start_min, start_max + 1)


def _peptide_record(
    wt_pep: str,
    mut_pep: str,
    plen: int,
    pos_in_peptide: int,
) -> dict[str, str]:
    """Build the raw peptide dict.

    Parameters
    ----------
    wt_pep : str
        Wild-type peptide.
    mut_pep : str
        Mutant peptide.
    plen : int
        Peptide length.
    pos_in_peptide : int
        Mutation position within the peptide.

    Returns
    -------
    dict[str, str]
        Peptide record.
    """
    return {
        "wt_peptide": wt_pep,
        "mut_peptide": mut_pep,
        "length": str(plen),
        "mutation_pos_in_peptide": str(pos_in_peptide),
    }


def _make_peptide_entry(
    wt_sequence: str,
    mut_seq_str: str,
    start: int,
    plen: int,
    mutation_position: int,
) -> dict[str, str] | None:
    """Create a single peptide record if the mutation alters the peptide.

    Parameters
    ----------
    wt_sequence : str
        Wild-type protein sequence.
    mut_seq_str : str
        Mutant protein sequence.
    start : int
        Window start position (zero-based).
    plen : int
        Peptide length.
    mutation_position : int
        Zero-based mutation index in the full protein.

    Returns
    -------
    dict[str, str] or None
        Peptide record with ``wt_peptide``, ``mut_peptide``, ``length``,
        and ``mutation_pos_in_peptide``; or ``None`` if unchanged.
    """
    end = start + plen
    if end > len(wt_sequence):
        return None
    wt_pep, mut_pep = wt_sequence[start:end], mut_seq_str[start:end]
    if wt_pep == mut_pep:
        return None
    return _peptide_record(wt_pep, mut_pep, plen, mutation_position - start)


def _collect_peptides_for_length(
    wt_sequence: str,
    mut_seq_str: str,
    mutation_position: int,
    plen: int,
) -> list[dict[str, str]]:
    """Collect all altered peptides of a given length spanning the mutation.

    Parameters
    ----------
    wt_sequence : str
        Wild-type protein sequence.
    mut_seq_str : str
        Mutant protein sequence.
    mutation_position : int
        Zero-based mutation index.
    plen : int
        Peptide length to generate.

    Returns
    -------
    list[dict[str, str]]
        List of peptide records (see :func:`_make_peptide_entry`).
    """
    results: list[dict[str, str]] = []
    for start in _peptide_window_range(mutation_position, plen, len(wt_sequence)):
        entry = _make_peptide_entry(wt_sequence, mut_seq_str, start, plen, mutation_position)
        if entry is not None:
            results.append(entry)
    return results


def generate_mutant_peptides(
    wt_sequence: str,
    mutation_position: int,
    mutant_aa: str,
    peptide_lengths: tuple[int, ...] = (8, 9, 10, 11),
) -> list[dict[str, str]]:
    """Generate overlapping mutant peptides spanning a mutation site.

    Slides a window of each requested length across the mutation
    position, producing all peptides that contain the mutant residue.

    Parameters
    ----------
    wt_sequence : str
        Wild-type protein sequence (single-letter amino acids).
    mutation_position : int
        Zero-based position of the mutated residue.
    mutant_aa : str
        Single-letter code of the mutant amino acid.
    peptide_lengths : tuple[int, ...]
        Peptide lengths to generate (default 8–11-mers for MHC-I).

    Returns
    -------
    list[dict[str, str]]
        Each dict has keys ``wt_peptide``, ``mut_peptide``, ``length``,
        ``mutation_pos_in_peptide``.
    """
    mut = _build_mutant_sequence(wt_sequence, mutation_position, mutant_aa)
    results: list[dict[str, str]] = []
    for plen in peptide_lengths:
        results.extend(_collect_peptides_for_length(wt_sequence, mut, mutation_position, plen))
    return results


# ── Helpers for score_mhc_binding ───────────────────────────────────────


def _anchor_score(peptide: str) -> float:
    """Compute the anchor-position contribution to MHC binding.

    Parameters
    ----------
    peptide : str
        Peptide sequence.

    Returns
    -------
    float
        Raw anchor score (not normalised).
    """
    n = len(peptide)
    score = 0.0
    for pos, prefs in _MHC_PREF.items():
        idx = min(pos, n - 1)
        score += prefs.get(peptide[idx], 0.1)
    return score


def _centre_penalty(peptide: str) -> float:
    """Compute a penalty for charged residues in the peptide centre.

    Parameters
    ----------
    peptide : str
        Peptide sequence.

    Returns
    -------
    float
        Penalty value (non-negative).
    """
    centre = peptide[2:-2] if len(peptide) > 4 else peptide
    return sum(0.1 for aa in centre if aa in ("D", "E", "K", "R"))


def _normalise_binding_score(score: float) -> float:
    """Clamp and normalise a raw binding score to ``[0, 1]``.

    Parameters
    ----------
    score : float
        Raw binding score (anchor minus penalties).

    Returns
    -------
    float
        Normalised score rounded to four decimal places.
    """
    max_score = sum(max(p.values()) for p in _MHC_PREF.values())
    normalised = max(0.0, min(1.0, score / max_score if max_score > 0 else 0))
    return round(normalised, 4)


def score_mhc_binding(peptide: str) -> float:
    """Estimate MHC-I binding affinity using a simplified PSSM.

    This is a screening approximation based on HLA-A*02:01 anchor
    preferences.  For clinical neoantigen prediction, use NetMHCpan
    or MHCflurry.

    Parameters
    ----------
    peptide : str
        Peptide sequence (8–11 amino acids).

    Returns
    -------
    float
        Binding score in ``[0, 1]`` range.  Higher values indicate
        stronger predicted binding.
    """
    score = _anchor_score(peptide) - _centre_penalty(peptide)
    return _normalise_binding_score(score)


# ── Helpers for NeoantigenPredictor.predict_from_maf ────────────────────


def _priority_label(score: float) -> str:
    """Map a binding score to a priority label.

    Parameters
    ----------
    score : float
        MHC binding score in ``[0, 1]``.

    Returns
    -------
    str
        ``"HIGH"``, ``"MEDIUM"``, or ``"LOW"``.
    """
    if score >= 0.75:
        return "HIGH"
    if score >= 0.5:
        return "MEDIUM"
    return "LOW"


def _make_candidate_record(
    gene: str,
    mutation: str,
    pep: dict[str, str],
    mhc_score: float,
) -> dict[str, Any]:
    """Build a single neoantigen candidate record.

    Parameters
    ----------
    gene : str
        Hugo gene symbol.
    mutation : str
        HGVSp short-form mutation string.
    pep : dict[str, str]
        Peptide record from :func:`generate_mutant_peptides`.
    mhc_score : float
        Predicted MHC-I binding score.

    Returns
    -------
    dict[str, Any]
        Candidate record with ``Gene``, ``Mutation``, ``WT_Peptide``,
        ``Mut_Peptide``, ``Length``, ``MHC_Score``, ``Priority``.
    """
    return {
        "Gene": gene,
        "Mutation": mutation,
        "WT_Peptide": pep["wt_peptide"],
        "Mut_Peptide": pep["mut_peptide"],
        "Length": int(pep["length"]),
        "MHC_Score": mhc_score,
        "Priority": _priority_label(mhc_score),
    }


def _parse_hgvsp(hgvsp: str) -> tuple[int, str] | None:
    """Parse an HGVSp short-form string into position and mutant AA.

    Parameters
    ----------
    hgvsp : str
        Protein mutation string in ``p.X123Y`` format.

    Returns
    -------
    tuple[int, str] or None
        ``(zero_based_position, mutant_amino_acid)`` on success,
        or ``None`` if the string cannot be parsed.
    """
    if not hgvsp or len(hgvsp) < 4:
        return None
    try:
        position = int("".join(c for c in hgvsp[2:-1] if c.isdigit())) - 1
        mutant_aa = hgvsp[-1]
    except (ValueError, IndexError):
        return None
    return position, mutant_aa


def _get_protein_sequence(
    gene: str,
    gene_sequences: dict[str, str] | None,
    position: int,
) -> str:
    """Retrieve or synthesise a protein sequence for a gene.

    Parameters
    ----------
    gene : str
        Hugo gene symbol.
    gene_sequences : dict[str, str] or None
        Map of gene symbol to protein sequence.  When ``None`` or the
        gene is absent, a synthetic sequence is generated.
    position : int
        Zero-based mutation position (used to size synthetic sequences).

    Returns
    -------
    str
        Protein sequence (single-letter amino acids).
    """
    if gene_sequences and gene in gene_sequences:
        return gene_sequences[gene]
    rng = np.random.default_rng(hash(gene) % (2**32))
    aa_chars = list("ACDEFGHIKLMNPQRSTVWY")
    return "".join(rng.choice(aa_chars, size=max(position + 50, 200)))


def _score_and_filter_peptides(
    peptides: list[dict[str, str]],
    gene: str,
    mutation: str,
    binding_threshold: float,
) -> list[dict[str, Any]]:
    """Score peptides for MHC binding and keep those above threshold.

    Parameters
    ----------
    peptides : list[dict[str, str]]
        Peptide records from :func:`generate_mutant_peptides`.
    gene : str
        Hugo gene symbol.
    mutation : str
        HGVSp short-form mutation string.
    binding_threshold : float
        Minimum MHC binding score to retain a candidate.

    Returns
    -------
    list[dict[str, Any]]
        Candidate records passing the binding threshold.
    """
    candidates: list[dict[str, Any]] = []
    for pep in peptides:
        mhc_score = score_mhc_binding(pep["mut_peptide"])
        if mhc_score >= binding_threshold:
            candidates.append(_make_candidate_record(gene, mutation, pep, mhc_score))
    return candidates


def _process_mutation_row(
    row: pd.Series,
    gene_sequences: dict[str, str] | None,
    binding_threshold: float,
) -> list[dict[str, Any]]:
    """Process a single MAF row and return neoantigen candidates.

    Parameters
    ----------
    row : pd.Series
        A row from a MAF ``DataFrame`` containing ``Hugo_Symbol``
        and ``HGVSp_Short`` columns.
    gene_sequences : dict[str, str] or None
        Map of gene symbol to protein sequence.
    binding_threshold : float
        Minimum MHC binding score to retain a candidate.

    Returns
    -------
    list[dict[str, Any]]
        Candidate records for this mutation (may be empty).
    """
    gene, hgvsp = row["Hugo_Symbol"], str(row.get("HGVSp_Short", ""))
    parsed = _parse_hgvsp(hgvsp)
    if parsed is None:
        return []
    position, mutant_aa = parsed
    protein = _get_protein_sequence(gene, gene_sequences, position)
    peptides = generate_mutant_peptides(protein, position, mutant_aa)
    return _score_and_filter_peptides(peptides, gene, hgvsp, binding_threshold)


def _empty_neoantigen_df() -> pd.DataFrame:
    """Return an empty ``DataFrame`` with the standard neoantigen columns.

    Returns
    -------
    pd.DataFrame
        Empty frame with columns ``Gene``, ``Mutation``, ``WT_Peptide``,
        ``Mut_Peptide``, ``Length``, ``MHC_Score``, ``Priority``.
    """
    cols = [
        "Gene",
        "Mutation",
        "WT_Peptide",
        "Mut_Peptide",
        "Length",
        "MHC_Score",
        "Priority",
    ]
    return pd.DataFrame(columns=cols)


def _build_candidates(
    missense: pd.DataFrame,
    gene_sequences: dict[str, str] | None,
    binding_threshold: float,
) -> list[dict[str, Any]]:
    """Iterate over missense rows and aggregate neoantigen candidates.

    Parameters
    ----------
    missense : pd.DataFrame
        Filtered MAF containing only missense mutations.
    gene_sequences : dict[str, str] or None
        Map of gene symbol to protein sequence.
    binding_threshold : float
        Minimum MHC binding score to retain a candidate.

    Returns
    -------
    list[dict[str, Any]]
        All candidates across every mutation row.
    """
    candidates: list[dict[str, Any]] = []
    for _, row in missense.iterrows():
        candidates.extend(_process_mutation_row(row, gene_sequences, binding_threshold))
    return candidates


def _finalise_candidates(candidates: list[dict[str, Any]]) -> pd.DataFrame:
    """Convert candidate records to a sorted ``DataFrame``.

    Parameters
    ----------
    candidates : list[dict[str, Any]]
        Raw candidate records.

    Returns
    -------
    pd.DataFrame
        Candidates sorted by ``MHC_Score`` descending, index reset.
    """
    result = pd.DataFrame(candidates)
    if not result.empty:
        result = result.sort_values("MHC_Score", ascending=False).reset_index(drop=True)
    return result


# ── Helper for NeoantigenPredictor.summarise ────────────────────────────


def _summarise_gene_group(grp: pd.DataFrame) -> dict[str, Any]:
    """Summarise neoantigen candidates for a single gene.

    Parameters
    ----------
    grp : pd.DataFrame
        Subset of predictions belonging to one gene.

    Returns
    -------
    dict[str, Any]
        Dict with ``n_candidates``, ``best_score``, ``best_peptide``.
    """
    best = grp.iloc[0]
    return {
        "n_candidates": len(grp),
        "best_score": float(best["MHC_Score"]),
        "best_peptide": best["Mut_Peptide"],
    }


# ── Main class ──────────────────────────────────────────────────────────


class NeoantigenPredictor:
    """Predict candidate neoantigens from somatic mutation data.

    Combines mutation data (MAF) with reference protein sequences to
    generate mutant peptides and score their MHC-I binding potential.

    Parameters
    ----------
    config : ProjectConfig, optional
        Runtime configuration.
    """

    def __init__(self, config: ProjectConfig | None = None) -> None:
        """Initialise the neoantigen predictor.

        Parameters
        ----------
        config : ProjectConfig, optional
            Runtime configuration.  Falls back to defaults when ``None``.
        """
        self.cfg = config or ProjectConfig()

    def predict_from_maf(
        self,
        maf: pd.DataFrame,
        gene_sequences: dict[str, str] | None = None,
        binding_threshold: float = 0.5,
    ) -> pd.DataFrame:
        """Predict neoantigens from missense mutations in a MAF.

        Parameters
        ----------
        maf : pd.DataFrame
            Mutation data with ``Hugo_Symbol``, ``Variant_Classification``,
            and ``HGVSp_Short`` columns.
        gene_sequences : dict[str, str], optional
            Map of gene symbol → protein sequence.  If ``None``, generates
            synthetic sequences for demonstration.
        binding_threshold : float
            Minimum MHC binding score to report a candidate.

        Returns
        -------
        pd.DataFrame
            Columns: ``Gene``, ``Mutation``, ``WT_Peptide``,
            ``Mut_Peptide``, ``Length``, ``MHC_Score``, ``Priority``.
        """
        missense = maf[maf["Variant_Classification"] == "Missense_Mutation"].copy()
        if missense.empty:
            logger.warning("No missense mutations found in MAF")
            return _empty_neoantigen_df()
        candidates = _build_candidates(missense, gene_sequences, binding_threshold)
        result = _finalise_candidates(candidates)
        logger.info("%d candidates from %d mutations", len(result), len(missense))
        return result

    def summarise(self, predictions: pd.DataFrame) -> dict[str, Any]:
        """Summarise neoantigen prediction results per gene.

        Parameters
        ----------
        predictions : pd.DataFrame
            Output of :meth:`predict_from_maf`.

        Returns
        -------
        dict[str, Any]
            Keys are gene symbols; values are dicts with
            ``n_candidates``, ``best_score``, ``best_peptide``.
        """
        if predictions.empty:
            return {}
        summary: dict[str, Any] = {}
        for gene, grp in predictions.groupby("Gene"):
            summary[str(gene)] = _summarise_gene_group(grp)
        return summary
