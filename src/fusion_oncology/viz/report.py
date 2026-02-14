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


def generate_html_report(
    results: pd.DataFrame,
    figures: dict[str, plt.Figure],
    cv_metrics: dict[str, Any] | None = None,
    config: ProjectConfig | None = None,
) -> Path:
    """
    Build and save a self-contained HTML report.

    Parameters
    ----------
    results : pd.DataFrame
        The fusion ranking table.
    figures : dict[str, Figure]
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

    sections: list[str] = [
        "<!DOCTYPE html><html><head><meta charset='utf-8'>",
        f"<title>Fusion Oncology Report — {now}</title>",
        _CSS,
        "</head><body>",
        "<h1>Fusion Oncology — Analysis Report</h1>",
        f"<p class='meta'>Generated {now}</p>",
    ]

    # Model performance
    if cv_metrics:
        sections.append("<h2>Model Performance</h2>")
        sections.append(
            f"<p>XGBoost {cv_metrics.get('mean_accuracy', 0):.4f} "
            f"&plusmn; {cv_metrics.get('std_accuracy', 0):.4f} accuracy "
            f"(5-fold stratified CV)</p>"
        )

    # Target ranking table
    sections.append("<h2>Target Ranking</h2>")
    sections.append(_df_to_html(results))

    # Figures
    for title, fig in figures.items():
        b64 = _fig_to_base64(fig)
        sections.append(f"<h2>{title}</h2>")
        sections.append(
            f'<div class="fig-container"><img src="data:image/png;base64,{b64}"></div>'
        )

    sections.append("</body></html>")

    out = cfg.output_dir / "fusion_report.html"
    out.write_text("\n".join(sections))
    logger.info("Report saved → %s", out)
    return out
