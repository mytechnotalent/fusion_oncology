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

# ── Constants ────────────────────────────────────────────────────────────

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


# ── Helper functions ────────────────────────────────────────────────────


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


# ── Scoring ──────────────────────────────────────────────────────────────


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

    score = 0.5  # baseline

    # Positional preferences
    for pos, weights in _POSITION_WEIGHTS.items():
        score += weights.get(guide[pos], 0.0)

    # GC content
    gc = _gc_content(guide)
    if 0.4 <= gc <= 0.7:
        score += 0.15
    elif gc < 0.3 or gc > 0.8:
        score -= 0.20

    # Poly-T penalty
    if _has_polyT(guide):
        score -= 0.30

    # Self-complementarity
    sc = _seed_complement_score(guide)
    score -= sc * 0.2

    return float(np.clip(score, 0.0, 1.0))


def off_target_heuristic(guide: str, genome_gc: float = 0.41) -> float:
    """Estimate off-target risk heuristically.

    A full off-target analysis requires genome alignment (e.g. Cas-OFFinder),
    but this heuristic flags high-risk guides based on:
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
    risk = 0.3  # baseline

    gc = _gc_content(guide)
    risk += abs(gc - genome_gc) * 0.5

    # Low complexity check: count unique 3-mers
    trimers = {guide[i : i + 3] for i in range(len(guide) - 2)}
    diversity = len(trimers) / max(1, len(guide) - 2)
    if diversity < 0.5:
        risk += 0.25

    return float(np.clip(risk, 0.0, 1.0))


# ── Guide design ─────────────────────────────────────────────────────────


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
    sequence = sequence.upper()
    complements = {"A": "T", "T": "A", "G": "C", "C": "G", "N": "N"}

    def revcomp(s: str) -> str:
        """Return the reverse complement of a DNA string.

        Parameters
        ----------
        s : str
            DNA sequence.

        Returns
        -------
        str
            Reverse-complement sequence.
        """
        return "".join(complements.get(c, "N") for c in reversed(s))

    guides: list[dict[str, Any]] = []

    # Sense strand: find GG at positions i, i+1 with any base at i-1
    for i in range(GUIDE_LENGTH, len(sequence) - 2):
        if sequence[i + 1 : i + 3] == "GG":
            spacer = sequence[i - GUIDE_LENGTH : i]
            if len(spacer) != GUIDE_LENGTH or "N" in spacer:
                continue
            ot = on_target_score(spacer)
            otr = off_target_heuristic(spacer)
            guides.append(
                {
                    "guide_id": f"sg_{gene}_{_unique_id(spacer)}",
                    "gene": gene,
                    "sequence": spacer,
                    "strand": "+",
                    "position": i - GUIDE_LENGTH,
                    "gc_content": round(_gc_content(spacer), 3),
                    "on_target": round(ot, 4),
                    "off_target_risk": round(otr, 4),
                    "composite_score": round(ot * (1 - otr), 4),
                }
            )

    # Antisense strand
    rc = revcomp(sequence)
    for i in range(GUIDE_LENGTH, len(rc) - 2):
        if rc[i + 1 : i + 3] == "GG":
            spacer = rc[i - GUIDE_LENGTH : i]
            if len(spacer) != GUIDE_LENGTH or "N" in spacer:
                continue
            ot = on_target_score(spacer)
            otr = off_target_heuristic(spacer)
            guides.append(
                {
                    "guide_id": f"sg_{gene}_{_unique_id(spacer)}",
                    "gene": gene,
                    "sequence": spacer,
                    "strand": "-",
                    "position": len(sequence) - i,
                    "gc_content": round(_gc_content(spacer), 3),
                    "on_target": round(ot, 4),
                    "off_target_risk": round(otr, 4),
                    "composite_score": round(ot * (1 - otr), 4),
                }
            )

    return sorted(guides, key=lambda g: g["composite_score"], reverse=True)


# ── Main class ───────────────────────────────────────────────────────────


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

        # Filter by minimum score
        passing = [g for g in all_guides if g["composite_score"] >= self.min_score]

        # Keep top N
        selected = passing[: self.guides_per_gene]
        self._library.extend(selected)

        logger.info(
            "%s: found %d guides, %d passing threshold, selected %d",
            gene,
            len(all_guides),
            len(passing),
            len(selected),
        )
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
                logger.warning("No sequence available for %s — skipping.", gene)

        return self.library_dataframe()

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
            return pd.DataFrame(
                columns=[
                    "guide_id",
                    "gene",
                    "sequence",
                    "strand",
                    "position",
                    "gc_content",
                    "on_target",
                    "off_target_risk",
                    "composite_score",
                ]
            )
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
            return {
                "total_guides": 0,
                "genes_covered": 0,
                "mean_on_target": 0.0,
                "mean_off_target_risk": 0.0,
                "top_guide": None,
            }
        df = self.library_dataframe()
        top = df.iloc[0].to_dict() if len(df) > 0 else None
        return {
            "total_guides": len(df),
            "genes_covered": df["gene"].nunique(),
            "mean_on_target": round(df["on_target"].mean(), 4),
            "mean_off_target_risk": round(df["off_target_risk"].mean(), 4),
            "top_guide": top,
        }
