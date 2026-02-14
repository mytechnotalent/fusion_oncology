"""Tests for the HTML report generator."""

from fusion_oncology.viz.report import generate_html_report
from fusion_oncology.viz.plots import fusion_bar

import matplotlib

matplotlib.use("Agg")


def test_generate_html(sample_results, tiny_config):
    """Generated HTML must exist and contain key content strings.

    Parameters
    ----------
    sample_results : pd.DataFrame
        Fixture providing a small fusion-results table.
    tiny_config : ProjectConfig
        Lightweight config fixture with temp output dir.
    """
    fig = fusion_bar(sample_results)
    path = generate_html_report(
        results=sample_results,
        figures={"Test Figure": fig},
        cv_metrics={"mean_accuracy": 0.92, "std_accuracy": 0.03},
        config=tiny_config,
    )
    assert path.exists()
    html = path.read_text()
    assert "Fusion Oncology" in html
    assert "EGFR" in html
    assert "0.92" in html
