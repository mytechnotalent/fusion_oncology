"""
HTML report generator.

Assembles figures, tables, and metadata into a self-contained HTML
file suitable for sharing with collaborators.
"""

from __future__ import annotations

import base64
import logging
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

from fusion_oncology.config import ProjectConfig

logger = logging.getLogger(__name__)


def _fig_to_base64(fig: plt.Figure) -> str:
    """Render a matplotlib figure to a base-64-encoded PNG string.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure to encode.

    Returns
    -------
    str
        Base-64 encoded PNG data (ASCII string).
    """
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def _df_to_html(df: pd.DataFrame) -> str:
    """Convert a DataFrame to an HTML ``<table>`` string.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to render.

    Returns
    -------
    str
        HTML table markup with CSS class ``styled-table``.
    """
    return df.to_html(index=False, classes="styled-table", border=0)


_CSS = """
<style>
  body { font-family: 'Segoe UI', Tahoma, sans-serif; margin: 2rem; background: #f7f7fa; color: #222; }
  h1 { color: #1a1a2e; border-bottom: 3px solid #e94560; padding-bottom: .3em; }
  h2 { color: #0f3460; margin-top: 2em; }
  .styled-table { border-collapse: collapse; width: 100%; margin: 1em 0; font-size: .9rem; }
  .styled-table th { background: #0f3460; color: #fff; padding: .6em 1em; text-align: left; }
  .styled-table td { padding: .5em 1em; border-bottom: 1px solid #ddd; }
  .styled-table tr:hover { background: #e8ecf1; }
  .fig-container { text-align: center; margin: 1.5em 0; }
  .fig-container img { max-width: 100%; border: 1px solid #ccc; border-radius: 6px; }
  .meta { font-size: .8rem; color: #888; }
</style>
"""


def _build_html_header(now: str) -> list[str]:
    """Return the opening HTML sections including head and title.

    Parameters
    ----------
    now : str
        Formatted UTC timestamp string.

    Returns
    -------
    list[str]
        HTML fragments for the page header.
    """
    return [
        "<!DOCTYPE html><html><head><meta charset='utf-8'>",
        f"<title>Fusion Oncology Report — {now}</title>",
        _CSS,
        "</head><body>",
        "<h1>Fusion Oncology — Analysis Report</h1>",
        f"<p class='meta'>Generated {now}</p>",
    ]


def _build_cv_section(
    cv_metrics: dict[str, Any] | None,
) -> list[str]:
    """Build the cross-validation performance section.

    Parameters
    ----------
    cv_metrics : dict[str, Any] or None
        XGBoost cross-validation metrics.

    Returns
    -------
    list[str]
        HTML fragments (empty list when *cv_metrics* is falsy).
    """
    if not cv_metrics:
        return []
    rows = []
    for label, key in [
        ("Accuracy", "accuracy"),
        ("Precision", "precision"),
        ("Recall", "recall"),
        ("F1-Score", "f1"),
        ("F2-Score", "f2"),
        ("ROC AUC", "roc_auc"),
    ]:
        mean = cv_metrics.get(f"mean_{key}", 0)
        std = cv_metrics.get(f"std_{key}", 0)
        rows.append(f"<tr><td>{label}</td>" f"<td>{mean:.4f} &plusmn; {std:.4f}</td></tr>")
    return [
        "<h2>Model Performance (5-fold stratified CV)</h2>",
        "<table><tr><th>Metric</th><th>Score</th></tr>",
        *rows,
        "</table>",
    ]


def _build_results_table(results: pd.DataFrame) -> list[str]:
    """Build the target-ranking table section.

    Parameters
    ----------
    results : pd.DataFrame
        The fusion ranking table.

    Returns
    -------
    list[str]
        HTML fragments for the ranking table.
    """
    return ["<h2>Target Ranking</h2>", _df_to_html(results)]


def _build_figures_section(
    figures: dict[str, plt.Figure],
) -> list[str]:
    """Encode figures as base-64 images and wrap in HTML.

    Parameters
    ----------
    figures : dict[str, matplotlib.figure.Figure]
        Named matplotlib figures to embed.

    Returns
    -------
    list[str]
        HTML fragments for every embedded figure.
    """
    parts: list[str] = []
    for title, fig in figures.items():
        b64 = _fig_to_base64(fig)
        parts.append(f"<h2>{title}</h2>")
        img = f'<img src="data:image/png;base64,{b64}">'
        parts.append(f'<div class="fig-container">{img}</div>')
    return parts


def _assemble_html(sections: list[str], cfg: ProjectConfig) -> Path:
    """Join HTML sections, write to disk, and return the path.

    Parameters
    ----------
    sections : list[str]
        Ordered HTML fragments.
    cfg : ProjectConfig
        Supplies ``output_dir``.

    Returns
    -------
    Path
        Path to the written HTML file.
    """
    out = cfg.output_dir / "fusion_report.html"
    out.write_text("\n".join(sections))
    logger.info("Report saved → %s", out)
    return out


def generate_html_report(
    results: pd.DataFrame,
    figures: dict[str, plt.Figure],
    cv_metrics: dict[str, Any] | None = None,
    config: ProjectConfig | None = None,
) -> Path:
    """Build and save a self-contained HTML report.

    Parameters
    ----------
    results : pd.DataFrame
        The fusion ranking table.
    figures : dict[str, matplotlib.figure.Figure]
        Named matplotlib figures to embed.
    cv_metrics : dict, optional
        XGBoost cross-validation metrics.
    config : ProjectConfig, optional
        Output directory and format settings.

    Returns
    -------
    Path
        Path to the written HTML file.
    """
    cfg = config or ProjectConfig()
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    sections = _build_html_header(now)
    sections += _build_cv_section(cv_metrics)
    sections += _build_results_table(results)
    sections += _build_figures_section(figures)
    sections.append("</body></html>")
    return _assemble_html(sections, cfg)
