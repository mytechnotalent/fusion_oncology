"""Tests for pathway enrichment."""

from fusion_oncology.analysis.pathway import PathwayEnrichment


def test_lookup_known_gene():
    """Look up EGFR in the curated pathway database.

    Asserts
    -------
    - The returned list of pathway names includes
      ``"MAPK/ERK Signaling"``.
    """
    pe = PathwayEnrichment()
    paths = pe.lookup("EGFR")
    assert "MAPK/ERK Signaling" in paths


def test_lookup_unknown_gene():
    """Look up a fabricated gene symbol.

    Asserts
    -------
    - An empty list is returned when the gene is not present in
      any curated pathway.
    """
    pe = PathwayEnrichment()
    assert pe.lookup("NONEXISTENT_GENE_XYZ") == []


def test_annotate(sample_results):
    """Annotated DataFrame must gain a ``Pathways`` column with hits.

    Parameters
    ----------
    sample_results : pd.DataFrame
        Fixture providing a small fusion-results table.
    """
    pe = PathwayEnrichment()
    annotated = pe.annotate(sample_results)
    assert "Pathways" in annotated.columns
    # TP53 should be in at least one pathway
    tp53_row = annotated[annotated["Gene"] == "TP53"].iloc[0]
    assert tp53_row["Pathways"] != "—"


def test_enrichment_summary():
    """Compute pathway enrichment for a known gene panel.

    Uses EGFR, KRAS, BRAF, and TP53 — all members of the MAPK/ERK
    pathway.

    Asserts
    -------
    - ``"MAPK/ERK Signaling"`` appears in the summary dict.
    - Its ``count`` value is at least 2.
    """
    pe = PathwayEnrichment()
    summary = pe.enrichment_summary(["EGFR", "KRAS", "BRAF", "TP53"])
    assert "MAPK/ERK Signaling" in summary
    assert summary["MAPK/ERK Signaling"]["count"] >= 2
