"""Tests for bio utilities."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from fusion_oncology.config import ProjectConfig
from fusion_oncology.utils.bio import (
    _do_entrez_lookup,
    _fetch_fasta,
    _pad_short_sequence,
    _parse_fasta_sequence,
    _search_entrez,
    _synthetic_sequence,
    fetch_gene_sequence,
    find_cpg_islands,
    gc_content,
)


# ── gc_content ───────────────────────────────────────────────────────────


def test_gc_content():
    """Validate GC-content calculation across edge cases.

    Asserts
    -------
    - Pure G/C input yields 1.0.
    - Pure A/T input yields 0.0.
    - Equal mix (ACGT) yields 0.5.
    - Empty string yields 0.0 without error.
    """
    assert gc_content("GGCC") == 1.0
    assert gc_content("AATT") == 0.0
    assert abs(gc_content("ACGT") - 0.5) < 1e-9
    assert gc_content("") == 0.0


# ── find_cpg_islands ────────────────────────────────────────────────────


def test_cpg_islands():
    """Verify CpG island detection with extreme sequences.

    Constructs a 400 bp all-G sequence (GC = 1.0) and asserts that
    at least one island window is detected.  Then constructs a 400 bp
    all-A sequence (GC = 0.0) and asserts that zero islands are
    returned.
    """
    seq = "G" * 400
    islands = find_cpg_islands(seq, window=200, threshold=0.6)
    assert len(islands) >= 1
    seq_at = "A" * 400
    assert find_cpg_islands(seq_at, window=200, threshold=0.6) == []


# ── _synthetic_sequence ─────────────────────────────────────────────────


def test_synthetic_sequence_default():
    """Default synthetic sequence uses seed 42 and has correct length.

    Returns
    -------
    None
    """
    seq = _synthetic_sequence(length=100)
    assert len(seq) == 100
    assert set(seq).issubset({"A", "C", "G", "T"})


def test_synthetic_sequence_gene_seed():
    """Gene name should change the seed and produce a different sequence.

    Returns
    -------
    None
    """
    seq_a = _synthetic_sequence(length=50, gene="EGFR")
    seq_b = _synthetic_sequence(length=50, gene="BRAF")
    assert seq_a != seq_b


def test_synthetic_sequence_reproducible():
    """Same gene name produces the same sequence every time.

    Returns
    -------
    None
    """
    s1 = _synthetic_sequence(length=50, gene="TP53")
    s2 = _synthetic_sequence(length=50, gene="TP53")
    assert s1 == s2


# ── _search_entrez ──────────────────────────────────────────────────────


def test_search_entrez():
    """_search_entrez should return a list of IDs from Entrez.

    Returns
    -------
    None
    """
    cfg = ProjectConfig()
    mock_handle = MagicMock()
    mock_record = {"IdList": ["12345"]}
    with patch("fusion_oncology.utils.bio.Entrez.esearch", return_value=mock_handle):
        with patch("fusion_oncology.utils.bio.Entrez.read", return_value=mock_record):
            ids = _search_entrez("EGFR", cfg)
    assert ids == ["12345"]


# ── _fetch_fasta ────────────────────────────────────────────────────────


def test_fetch_fasta():
    """_fetch_fasta should return raw FASTA text.

    Returns
    -------
    None
    """
    mock_handle = MagicMock()
    mock_handle.read.return_value = ">seq1\nACGTACGT\n"
    with patch("fusion_oncology.utils.bio.Entrez.efetch", return_value=mock_handle):
        fasta = _fetch_fasta("12345")
    assert fasta == ">seq1\nACGTACGT\n"


# ── _parse_fasta_sequence ───────────────────────────────────────────────


def test_parse_fasta_sequence():
    """_parse_fasta_sequence strips header and non-ACGT characters.

    Returns
    -------
    None
    """
    raw = ">NM_005228.5\nACGTacgtNNNN\nTTTT"
    result = _parse_fasta_sequence(raw)
    assert result == "ACGTACGTTTTT"


# ── _pad_short_sequence ─────────────────────────────────────────────────


def test_pad_short_sequence():
    """_pad_short_sequence pads short sequences with synthetic bases.

    Returns
    -------
    None
    """
    padded = _pad_short_sequence("TEST", "ACGT")
    assert len(padded) > 4
    assert padded.startswith("ACGT")


# ── _do_entrez_lookup ───────────────────────────────────────────────────


def test_do_entrez_lookup_success():
    """_do_entrez_lookup returns a parsed sequence on success.

    Returns
    -------
    None
    """
    cfg = ProjectConfig()
    with patch("fusion_oncology.utils.bio._search_entrez", return_value=["999"]):
        with patch(
            "fusion_oncology.utils.bio._fetch_fasta", return_value=">s\nACGTACGTACGTACGTACGTACGT\n"
        ):
            result = _do_entrez_lookup("EGFR", cfg)
    assert "ACGT" in result


def test_do_entrez_lookup_no_ids():
    """_do_entrez_lookup returns synthetic when no IDs found.

    Returns
    -------
    None
    """
    cfg = ProjectConfig()
    with patch("fusion_oncology.utils.bio._search_entrez", return_value=[]):
        result = _do_entrez_lookup("FAKE", cfg)
    assert len(result) == 1000
    assert set(result).issubset({"A", "C", "G", "T"})


def test_do_entrez_lookup_short_sequence():
    """_do_entrez_lookup pads a too-short fetched sequence.

    Returns
    -------
    None
    """
    cfg = ProjectConfig()
    with patch("fusion_oncology.utils.bio._search_entrez", return_value=["1"]):
        with patch("fusion_oncology.utils.bio._fetch_fasta", return_value=">s\nACGT\n"):
            result = _do_entrez_lookup("SHORT", cfg)
    assert len(result) > 20


# ── fetch_gene_sequence ─────────────────────────────────────────────────


def test_fetch_gene_sequence_success():
    """fetch_gene_sequence returns sequence on successful lookup.

    Returns
    -------
    None
    """
    with patch("fusion_oncology.utils.bio._do_entrez_lookup", return_value="ACGTACGT"):
        seq = fetch_gene_sequence("EGFR")
    assert seq == "ACGTACGT"


def test_fetch_gene_sequence_fallback():
    """fetch_gene_sequence falls back to synthetic on exception.

    Returns
    -------
    None
    """
    with patch("fusion_oncology.utils.bio._do_entrez_lookup", side_effect=Exception("net err")):
        seq = fetch_gene_sequence("EGFR")
    assert len(seq) == 1000
    assert set(seq).issubset({"A", "C", "G", "T"})
