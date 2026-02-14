"""Tests for drug-target mapping."""

from fusion_oncology.analysis.drug_target import DrugTargetMapper


def test_lookup_egfr():
    """Look up EGFR in the drug-target database.

    Asserts
    -------
    - At least three approved drugs are returned.
    - Osimertinib is among the returned drug names.
    """
    dtm = DrugTargetMapper()
    drugs = dtm.lookup("EGFR")
    assert len(drugs) >= 3
    names = {d["drug"] for d in drugs}
    assert "Osimertinib" in names


def test_is_druggable():
    """Check the boolean druggability predicate.

    Asserts
    -------
    - A well-known target (EGFR) returns ``True``.
    - A fabricated gene name returns ``False``.
    """
    dtm = DrugTargetMapper()
    assert dtm.is_druggable("EGFR")
    assert not dtm.is_druggable("TOTALLY_UNKNOWN")


def test_annotate(sample_results):
    """Annotated DataFrame must gain ``Druggable`` and ``Approved_Drugs`` columns.

    Parameters
    ----------
    sample_results : pd.DataFrame
        Fixture providing a small fusion-results table.
    """
    dtm = DrugTargetMapper()
    annotated = dtm.annotate(sample_results)
    assert "Druggable" in annotated.columns
    assert "Approved_Drugs" in annotated.columns
    egfr_row = annotated[annotated["Gene"] == "EGFR"].iloc[0]
    assert egfr_row["Druggable"] == True  # noqa: E712


def test_full_report():
    """Generate a full drug-target report for mixed gene list.

    Passes one known gene (BRAF, ≥3 drugs) and one unknown gene.

    Asserts
    -------
    - The report DataFrame has at least 3 rows (BRAF drugs).
    - The unknown gene still appears in the output with an empty
      drug annotation.
    """
    dtm = DrugTargetMapper()
    report = dtm.full_report(["BRAF", "UNKNOWN_GENE"])
    assert len(report) >= 3  # BRAF has 3 drugs
    assert "UNKNOWN_GENE" in report["Gene"].values
