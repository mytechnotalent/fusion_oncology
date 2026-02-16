"""
CRISPR guide RNA design for gene knockout screening.

Generates single-guide RNA (sgRNA) candidates for top-ranked
therapeutic targets, scores them for on-target efficiency and
off-target risk, and exports ranked lists ready for library
construction.
"""

from __future__ import annotations

import hashlib
import logging
import re
from typing import Any

import numpy as np
import pandas as pd

from fusion_oncology.config import ProjectConfig

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────

PAM_MOTIF: str = "NGG"
"""Canonical SpCas9 PAM site (5ʹ-NGG-3ʹ)."""

GUIDE_LENGTH: int = 20
"""Standard spacer length in nucleotides."""

# Doench 2016 Rule Set 2 simplified position-weight scoring.
# Keys are 0-indexed positions, values are per-nucleotide weights
# learned from empirical cutting efficiency data.
_POSITION_WEIGHTS: dict[int, dict[str, float]] = {
    0: {"G": 0.10, "A": 0.00, "T": -0.05, "C": 0.05},
    1: {"G": 0.08, "A": -0.02, "T": -0.04, "C": 0.02},
    19: {"G": 0.12, "A": 0.00, "T": -0.08, "C": 0.04},  # 3ʹ end
}

_COMPLEMENTS: dict[str, str] = {
    "A": "T",
    "T": "A",
    "G": "C",
    "C": "G",
    "N": "N",
}
"""Watson-Crick complement lookup table."""


# ── Helper functions ───────────────────────────────────────────────


def _gc_content(seq: str) -> float:
    """Compute GC fraction of a DNA sequence.

    Parameters
    ----------
    seq : str
        Nucleotide string (ACGT).

    Returns
    -------
    float
        Fraction in ``[0, 1]``.
    """
    if not seq:
        return 0.0
    gc = sum(1 for c in seq.upper() if c in ("G", "C"))
    return gc / len(seq)


def _has_polyT(seq: str, threshold: int = 4) -> bool:
    """Check for a poly-T run that terminates Pol III transcription.

    Parameters
    ----------
    seq : str
        Nucleotide string.
    threshold : int
        Minimum run length to flag.

    Returns
    -------
    bool
        ``True`` if a poly-T run >= *threshold* is found.
    """
    return bool(re.search(f"T{{{threshold},}}", seq.upper()))


def _seed_complement_score(seq: str) -> float:
    """Score the seed region (positions 1-12 from PAM) for self-complementarity.

    High self-complementarity in the seed reduces cutting efficiency.

    Parameters
    ----------
    seq : str
        20-nt spacer sequence.

    Returns
    -------
    float
        Penalty score; higher means worse (more self-complementary).
    """
    seed = seq[-12:]  # 12 nt proximal to PAM
    complements = {"A": "T", "T": "A", "G": "C", "C": "G"}
    rev_comp = "".join(complements.get(c, "N") for c in reversed(seed))
    matches = sum(1 for a, b in zip(seed, rev_comp) if a == b)
    return matches / len(seed)


def _unique_id(seq: str) -> str:
    """Generate a short deterministic identifier for a guide sequence.

    Parameters
    ----------
    seq : str
        20-nt guide sequence.

    Returns
    -------
    str
        8-character hex digest.
    """
    return hashlib.md5(seq.encode()).hexdigest()[:8]  # noqa: S324


# ── Scoring helpers ────────────────────────────────────────────────


def _position_score(guide: str) -> float:
    """Compute position-weighted nucleotide preference score.

    Uses simplified Doench Rule Set 2 weights at key spacer
    positions.

    Parameters
    ----------
    guide : str
        20-nt spacer sequence (upper-case).

    Returns
    -------
    float
        Additive score contribution from positional weights.
    """
    total = 0.0
    for pos, weights in _POSITION_WEIGHTS.items():
        total += weights.get(guide[pos], 0.0)
    return total


def _gc_penalty(guide: str) -> float:
    """Compute GC-content bonus or penalty for a guide.

    Optimal GC is 40-70 %; extreme values reduce efficiency.

    Parameters
    ----------
    guide : str
        20-nt spacer sequence (upper-case).

    Returns
    -------
    float
        Score adjustment (positive = bonus, negative = penalty).
    """
    gc = _gc_content(guide)
    if 0.4 <= gc <= 0.7:
        return 0.15
    if gc < 0.3 or gc > 0.8:
        return -0.20
    return 0.0


def _polyt_penalty(guide: str) -> float:
    """Return poly-T terminator penalty for a guide.

    A run of four or more T nucleotides can terminate Pol III
    transcription, reducing sgRNA expression.

    Parameters
    ----------
    guide : str
        20-nt spacer sequence (upper-case).

    Returns
    -------
    float
        ``-0.30`` if poly-T is present, ``0.0`` otherwise.
    """
    if _has_polyT(guide):
        return -0.30
    return 0.0


def _self_comp_penalty(guide: str) -> float:
    """Return self-complementarity penalty for the guide seed region.

    Parameters
    ----------
    guide : str
        20-nt spacer sequence (upper-case).

    Returns
    -------
    float
        Negative score adjustment proportional to seed
        self-complementarity.
    """
    return -(_seed_complement_score(guide) * 0.2)


# ── Scoring ────────────────────────────────────────────────────────


def on_target_score(guide: str) -> float:
    """Estimate on-target cutting efficiency for a 20-nt guide.

    Uses a simplified model inspired by Doench Rule Set 2:
    1. Positional nucleotide preferences.
    2. GC content penalty (optimal ~40-70 %).
    3. Poly-T terminator penalty.
    4. Self-complementarity penalty.

    Parameters
    ----------
    guide : str
        20-nucleotide spacer sequence (5ʹ → 3ʹ, excluding PAM).

    Returns
    -------
    float
        Score in ``[0, 1]``.  Higher is better.
    """
    guide = guide.upper()
    if len(guide) != GUIDE_LENGTH:
        return 0.0
    score = 0.5 + _position_score(guide) + _gc_penalty(guide)
    score += _polyt_penalty(guide) + _self_comp_penalty(guide)
    return float(np.clip(score, 0.0, 1.0))


def _gc_deviation_risk(guide: str, genome_gc: float) -> float:
    """Compute off-target risk component from GC deviation.

    Guides whose GC content deviates strongly from the genome
    background are more likely to bind non-specifically.

    Parameters
    ----------
    guide : str
        20-nt spacer sequence (upper-case).
    genome_gc : float
        Background genome GC fraction.

    Returns
    -------
    float
        Risk contribution from GC deviation.
    """
    gc = _gc_content(guide)
    return abs(gc - genome_gc) * 0.5


def _complexity_risk(guide: str) -> float:
    """Compute off-target risk from low sequence complexity.

    Low-complexity sequences (few unique 3-mers) are more likely
    to match many genomic loci.

    Parameters
    ----------
    guide : str
        20-nt spacer sequence (upper-case).

    Returns
    -------
    float
        ``0.25`` if diversity is below threshold, ``0.0`` otherwise.
    """
    trimers = {guide[i : i + 3] for i in range(len(guide) - 2)}
    diversity = len(trimers) / max(1, len(guide) - 2)
    if diversity < 0.5:
        return 0.25
    return 0.0


def off_target_heuristic(guide: str, genome_gc: float = 0.41) -> float:
    """Estimate off-target risk heuristically.

    A full off-target analysis requires genome alignment
    (e.g. Cas-OFFinder), but this heuristic flags high-risk
    guides based on:
    - Low-complexity / repetitive content
    - Extreme GC bias
    - Seed region uniqueness proxy

    Parameters
    ----------
    guide : str
        20-nucleotide spacer.
    genome_gc : float
        Background genome GC content for comparison.

    Returns
    -------
    float
        Risk score in ``[0, 1]``.  Higher = more off-target risk.
    """
    guide = guide.upper()
    risk = 0.3 + _gc_deviation_risk(guide, genome_gc)
    risk += _complexity_risk(guide)
    return float(np.clip(risk, 0.0, 1.0))


# ── Guide design ───────────────────────────────────────────────────


def _revcomp(s: str) -> str:
    """Return the reverse complement of a DNA string.

    Parameters
    ----------
    s : str
        DNA sequence (upper-case ACGTN).

    Returns
    -------
    str
        Reverse-complement sequence.
    """
    return "".join(_COMPLEMENTS.get(c, "N") for c in reversed(s))


def _score_guide_candidate(
    spacer: str,
) -> tuple[float, float, float]:
    """Score a guide candidate for on/off-target metrics.

    Parameters
    ----------
    spacer : str
        20-nt spacer sequence.

    Returns
    -------
    tuple[float, float, float]
        ``(on_target, off_target_risk, composite_score)`` rounded
        to four decimal places.
    """
    ot = round(on_target_score(spacer), 4)
    otr = round(off_target_heuristic(spacer), 4)
    comp = round(ot * (1 - otr), 4)
    return ot, otr, comp


def _build_guide_record(
    gene: str,
    spacer: str,
    strand: str,
    position: int,
) -> dict[str, Any]:
    """Build a scored guide-RNA record dictionary.

    Parameters
    ----------
    gene : str
        Gene symbol.
    spacer : str
        20-nt spacer sequence.
    strand : str
        Strand label (``"+"`` or ``"-"``).
    position : int
        0-based position in the original sequence.

    Returns
    -------
    dict[str, Any]
        Guide record with scoring columns.
    """
    ot, otr, comp = _score_guide_candidate(spacer)
    rec = {"guide_id": f"sg_{gene}_{_unique_id(spacer)}"}
    rec.update(gene=gene, sequence=spacer, strand=strand, position=position)
    rec.update(gc_content=round(_gc_content(spacer), 3), on_target=ot)
    rec.update(off_target_risk=otr, composite_score=comp)
    return rec


def _scan_strand_for_guides(
    seq: str,
    gene: str,
    strand: str,
) -> list[dict[str, Any]]:
    """Scan one strand of a DNA sequence for SpCas9 guide candidates.

    Identifies every 20-nt spacer upstream of an NGG PAM on the
    given strand.

    Parameters
    ----------
    seq : str
        DNA sequence (already oriented for scanning).
    gene : str
        Gene symbol for annotation.
    strand : str
        Strand label (``"+"`` or ``"-"``).

    Returns
    -------
    list[dict[str, Any]]
        Guide records for all valid PAM sites found.
    """
    guides: list[dict[str, Any]] = []
    for i in range(GUIDE_LENGTH, len(seq) - 2):
        if seq[i + 1 : i + 3] == "GG":
            spacer = seq[i - GUIDE_LENGTH : i]
            if len(spacer) == GUIDE_LENGTH and "N" not in spacer:
                pos = i - GUIDE_LENGTH if strand == "+" else len(seq) - i
                guides.append(_build_guide_record(gene, spacer, strand, pos))
    return guides


def find_guides_in_sequence(
    sequence: str,
    gene: str = "unknown",
    pam: str = PAM_MOTIF,
) -> list[dict[str, Any]]:
    """Scan a DNA sequence for all SpCas9-compatible guides.

    Identifies every 20-nt spacer upstream of an NGG PAM on both
    strands and scores each candidate.

    Parameters
    ----------
    sequence : str
        DNA sequence to scan (typically exonic).
    gene : str
        Gene symbol for annotation.
    pam : str
        PAM motif.  Default ``"NGG"``.

    Returns
    -------
    list[dict[str, Any]]
        Each dict contains ``guide_id``, ``gene``, ``sequence``,
        ``strand``, ``position``, ``gc_content``, ``on_target``,
        ``off_target_risk``, ``composite_score``.
    """
    seq = sequence.upper()
    sense = _scan_strand_for_guides(seq, gene, "+")
    anti = _scan_strand_for_guides(_revcomp(seq), gene, "-")
    return sorted(
        sense + anti,
        key=lambda g: g["composite_score"],
        reverse=True,
    )


# ── Main class ─────────────────────────────────────────────────────


class CRISPRDesigner:
    """Design and rank CRISPR sgRNAs for therapeutic target knockout.

    Given a list of high-priority gene targets (e.g. from FusionEngine
    ranking), this class generates scored guide RNA candidates for each
    gene and exports a library-ready table.

    Parameters
    ----------
    config : ProjectConfig, optional
        Runtime configuration.
    guides_per_gene : int
        Maximum number of top guides to retain per gene.
    min_score : float
        Minimum composite score threshold for inclusion.
    """

    def __init__(
        self,
        config: ProjectConfig | None = None,
        guides_per_gene: int = 5,
        min_score: float = 0.25,
    ) -> None:
        """Initialise the designer.

        Parameters
        ----------
        config : ProjectConfig, optional
            Runtime configuration.
        guides_per_gene : int
            Maximum guides to keep per gene.
        min_score : float
            Minimum composite score to accept a guide.
        """
        self.cfg = config or ProjectConfig()
        self.guides_per_gene = guides_per_gene
        self.min_score = min_score
        self._library: list[dict[str, Any]] = []

    def _filter_and_select(
        self,
        guides: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], int]:
        """Filter guides by minimum score and select top candidates.

        Parameters
        ----------
        guides : list[dict[str, Any]]
            All guide candidates, sorted by composite score.

        Returns
        -------
        tuple[list[dict[str, Any]], int]
            Selected guides and count of guides passing the
            score threshold.
        """
        passing = [g for g in guides if g["composite_score"] >= self.min_score]
        return passing[: self.guides_per_gene], len(passing)

    def design_for_gene(
        self,
        gene: str,
        sequence: str,
    ) -> list[dict[str, Any]]:
        """Design guides for a single gene.

        Parameters
        ----------
        gene : str
            HGNC gene symbol.
        sequence : str
            Coding DNA sequence to scan for PAM sites.

        Returns
        -------
        list[dict[str, Any]]
            Top-ranked guides for this gene.
        """
        all_guides = find_guides_in_sequence(sequence, gene=gene)
        selected, n_pass = self._filter_and_select(all_guides)
        self._library.extend(selected)
        msg = "%s: found %d guides, %d passing threshold, selected %d"
        logger.info(msg, gene, len(all_guides), n_pass, len(selected))
        return selected

    def design_for_targets(
        self,
        targets: list[dict[str, Any]],
        sequences: dict[str, str],
    ) -> pd.DataFrame:
        """Design guides for a ranked list of gene targets.

        Parameters
        ----------
        targets : list[dict[str, Any]]
            As returned by ``FusionEngine.run()`` — must have a
            ``"Gene"`` key.
        sequences : dict[str, str]
            Mapping of gene symbol → coding DNA sequence.

        Returns
        -------
        pd.DataFrame
            Full guide library with scoring columns.
        """
        self._library.clear()
        for target in targets:
            gene = target.get("Gene", target.get("gene", ""))
            if gene in sequences:
                self.design_for_gene(gene, sequences[gene])
            else:
                logger.warning("No sequence for %s — skipping.", gene)
        return self.library_dataframe()

    def _empty_library_df(self) -> pd.DataFrame:
        """Return an empty DataFrame with library column schema.

        Returns
        -------
        pd.DataFrame
            Zero-row frame with all guide-library columns.
        """
        cols = "guide_id gene sequence strand position".split()
        cols += "gc_content on_target off_target_risk composite_score".split()
        return pd.DataFrame(columns=cols)

    def library_dataframe(self) -> pd.DataFrame:
        """Return the current guide library as a DataFrame.

        Returns
        -------
        pd.DataFrame
            All designed guides with columns: ``guide_id``, ``gene``,
            ``sequence``, ``strand``, ``position``, ``gc_content``,
            ``on_target``, ``off_target_risk``, ``composite_score``.
        """
        if not self._library:
            return self._empty_library_df()
        return (
            pd.DataFrame(self._library)
            .sort_values("composite_score", ascending=False)
            .reset_index(drop=True)
        )

    def export_library(self, path: str | None = None) -> str:
        """Export guide library as a CSV file.

        Parameters
        ----------
        path : str, optional
            Output file path.  Defaults to
            ``<output_dir>/crispr_library.csv``.

        Returns
        -------
        str
            Path to the written CSV file.
        """
        if path is None:
            path = str(self.cfg.output_dir / "crispr_library.csv")
        df = self.library_dataframe()
        df.to_csv(path, index=False)
        logger.info("Exported %d guides to %s", len(df), path)
        return path

    def _empty_summary(self) -> dict[str, Any]:
        """Return a zeroed summary dictionary.

        Returns
        -------
        dict[str, Any]
            Summary with all counts at zero and no top guide.
        """
        return {
            "total_guides": 0,
            "genes_covered": 0,
            "mean_on_target": 0.0,
            "mean_off_target_risk": 0.0,
            "top_guide": None,
        }

    def _compute_summary(self) -> dict[str, Any]:
        """Compute summary statistics from the current library.

        Returns
        -------
        dict[str, Any]
            Keys: ``total_guides``, ``genes_covered``,
            ``mean_on_target``, ``mean_off_target_risk``,
            ``top_guide``.
        """
        df = self.library_dataframe()
        top = df.iloc[0].to_dict() if len(df) > 0 else None
        ot = round(df["on_target"].mean(), 4)
        otr = round(df["off_target_risk"].mean(), 4)
        res = {"total_guides": len(df), "genes_covered": df["gene"].nunique()}
        res.update(mean_on_target=ot, mean_off_target_risk=otr, top_guide=top)
        return res

    def summary(self) -> dict[str, Any]:
        """Return a summary of the designed library.

        Returns
        -------
        dict[str, Any]
            Keys: ``total_guides``, ``genes_covered``,
            ``mean_on_target``, ``mean_off_target_risk``,
            ``top_guide``.
        """
        if not self._library:
            return self._empty_summary()
        return self._compute_summary()
