"""Tests for bio utilities."""

from fusion_oncology.utils.bio import find_cpg_islands, gc_content


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


def test_cpg_islands():
    """Verify CpG island detection with extreme sequences.

    Constructs a 400 bp all-G sequence (GC = 1.0) and asserts that
    at least one island window is detected.  Then constructs a 400 bp
    all-A sequence (GC = 0.0) and asserts that zero islands are
    returned.
    """
    # All G/C → every window should be flagged
    seq = "G" * 400
    islands = find_cpg_islands(seq, window=200, threshold=0.6)
    assert len(islands) >= 1

    # All A/T → no islands
    seq_at = "A" * 400
    assert find_cpg_islands(seq_at, window=200, threshold=0.6) == []
