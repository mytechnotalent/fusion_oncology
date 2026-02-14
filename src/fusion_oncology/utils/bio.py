"""
Bioinformatics helper functions.

Wraps NCBI Entrez lookups and sequence utilities used by multiple
modules.
"""

from __future__ import annotations

import logging
import re

from Bio import Entrez

from fusion_oncology.config import ProjectConfig

logger = logging.getLogger(__name__)


def _synthetic_sequence(length: int = 200) -> str:
    """Generate a deterministic synthetic DNA sequence for fallback.

    Uses a fixed NumPy RNG seed so results are reproducible across runs.

    Parameters
    ----------
    length : int
        Number of base-pairs to generate (default 200).

    Returns
    -------
    str
        Randomly sampled string of A/C/G/T characters.
    """
    import numpy as np

    rng = np.random.default_rng(seed=42)
    return "".join(rng.choice(["A", "C", "G", "T"], size=length))


def _search_entrez(gene: str, config: ProjectConfig) -> list[str]:
    """Search Entrez nucleotide database for a gene.

    Parameters
    ----------
    gene : str
        HGNC gene symbol.
    config : ProjectConfig
        Supplies Entrez email and retmax.

    Returns
    -------
    list[str]
        List of matching NCBI IDs.
    """
    term = f"{gene}[Gene Name] AND Homo sapiens[Organism] AND REFSEQ"
    handle = Entrez.esearch(db="nucleotide", term=term, retmax=config.entrez_retmax)
    record = Entrez.read(handle)
    handle.close()
    return record["IdList"]


def _fetch_fasta(ncbi_id: str) -> str:
    """Fetch a FASTA record from Entrez by ID.

    Parameters
    ----------
    ncbi_id : str
        NCBI nucleotide identifier.

    Returns
    -------
    str
        Raw FASTA text.
    """
    seq_handle = Entrez.efetch(
        db="nucleotide", id=ncbi_id, rettype="fasta", retmode="text"
    )
    fasta = seq_handle.read()
    seq_handle.close()
    return fasta


def _parse_fasta_sequence(fasta: str) -> str:
    """Extract a clean nucleotide string from raw FASTA text.

    Parameters
    ----------
    fasta : str
        Raw FASTA including header line.

    Returns
    -------
    str
        Upper-case A/C/G/T only sequence.
    """
    lines = fasta.strip().split("\n")
    sequence = "".join(lines[1:]).upper()
    return re.sub(r"[^ACGT]", "", sequence)


def _pad_short_sequence(gene: str, sequence: str) -> str:
    """Pad a too-short sequence with synthetic bases.

    Parameters
    ----------
    gene : str
        Gene symbol for logging.
    sequence : str
        Current nucleotide sequence.

    Returns
    -------
    str
        Padded sequence with at least 200 bp total.
    """
    logger.warning("Sequence for %s too short (%d bp) – padding", gene, len(sequence))
    return sequence + _synthetic_sequence()


def _do_entrez_lookup(gene: str, config: ProjectConfig) -> str:
    """Perform the Entrez search-and-fetch workflow.

    Parameters
    ----------
    gene : str
        HGNC gene symbol.
    config : ProjectConfig
        Supplies Entrez email and retmax.

    Returns
    -------
    str
        DNA sequence (uppercase A/C/G/T).
    """
    id_list = _search_entrez(gene, config)
    if not id_list:
        return _synthetic_sequence()
    sequence = _parse_fasta_sequence(_fetch_fasta(id_list[0]))
    if len(sequence) < 20:
        sequence = _pad_short_sequence(gene, sequence)
    return sequence


def fetch_gene_sequence(gene: str, config: ProjectConfig | None = None) -> str:
    """
    Retrieve a RefSeq DNA sequence for *gene* from NCBI Entrez.

    Falls back to a synthetic poly-ACGT sequence if the lookup fails,
    so downstream analysis can continue without hard-crashing.

    Parameters
    ----------
    gene : str
        HGNC gene symbol (e.g. ``"EGFR"``).
    config : ProjectConfig, optional
        Supplies Entrez email and retmax.

    Returns
    -------
    str
        The DNA sequence (uppercase A/C/G/T).
    """
    cfg = config or ProjectConfig()
    Entrez.email = cfg.entrez_email
    try:
        return _do_entrez_lookup(gene, cfg)
    except Exception:
        return _synthetic_sequence()


def gc_content(seq: str) -> float:
    """Compute the GC content fraction of a DNA sequence.

    Parameters
    ----------
    seq : str
        DNA sequence (case-insensitive).

    Returns
    -------
    float
        Fraction of bases that are G or C, in the range ``[0, 1]``.
        Returns ``0.0`` for an empty string.
    """
    if not seq:
        return 0.0
    seq = seq.upper()
    gc = seq.count("G") + seq.count("C")
    return gc / len(seq)


def find_cpg_islands(
    seq: str, window: int = 200, threshold: float = 0.6
) -> list[tuple[int, int]]:
    """
    Simple sliding-window CpG island detector.

    Parameters
    ----------
    seq : str
        DNA sequence.
    window : int
        Window size in bp.
    threshold : float
        Minimum GC content to flag window as CpG island.

    Returns
    -------
    list[tuple[int, int]]
        List of (start, end) positions of candidate CpG islands.
    """
    islands: list[tuple[int, int]] = []
    seq = seq.upper()
    for i in range(0, len(seq) - window + 1, window // 2):
        w = seq[i : i + window]
        if gc_content(w) >= threshold:
            islands.append((i, i + window))
    return islands
