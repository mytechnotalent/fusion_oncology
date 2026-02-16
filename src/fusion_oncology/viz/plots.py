"""
Publication-quality plots for Fusion Oncology results.

All functions accept standard pandas / numpy inputs and return
``matplotlib.Figure`` objects that can be saved or displayed.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # non-interactive backend for servers / CI

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from fusion_oncology.config import ProjectConfig

logger = logging.getLogger(__name__)

# House style
sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
PALETTE = sns.color_palette("viridis", 10)


def fusion_bar(results: pd.DataFrame, config: ProjectConfig | None = None) -> plt.Figure:
    """Horizontal bar chart of Fusion Index scores.

    Parameters
    ----------
    results : pd.DataFrame
        Must contain ``Gene`` and ``Fusion_Index`` columns.
    config : ProjectConfig, optional
        Runtime configuration (unused currently; reserved for future
        style overrides).

    Returns
    -------
    matplotlib.figure.Figure
        The rendered bar-chart figure.
    """
    fig, ax = plt.subplots(figsize=(8, max(4, len(results) * 0.6)))
    df = results.sort_values("Fusion_Index")
    colors = sns.color_palette("magma", n_colors=len(df))
    ax.barh(df["Gene"], df["Fusion_Index"], color=colors)
    ax.set_xlabel("Fusion Index")
    ax.set_title("Fusion Oncology — Target Ranking")
    fig.tight_layout()
    return fig


def _draw_scatter(ax: Any, results: pd.DataFrame) -> Any:
    """Plot the XGB-importance vs instability scatter.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axes.
    results : pd.DataFrame
        Must contain ``XGB_Importance``, ``Instability``, and
        ``Fusion_Index`` columns.

    Returns
    -------
    matplotlib.collections.PathCollection
        The scatter artist (used for the colour-bar).
    """
    return ax.scatter(
        results["XGB_Importance"],
        results["Instability"],
        s=results["Fusion_Index"] * 200 + 50,
        c=results["Fusion_Index"],
        cmap="magma",
        edgecolors="k",
        alpha=0.85,
    )


def _annotate_genes(ax: Any, results: pd.DataFrame) -> None:
    """Label each point with its gene name.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axes.
    results : pd.DataFrame
        Must contain ``Gene``, ``XGB_Importance``, and
        ``Instability`` columns.
    """
    for _, row in results.iterrows():
        ax.annotate(
            row["Gene"],
            (row["XGB_Importance"], row["Instability"]),
            fontsize=9,
            ha="left",
            va="bottom",
        )


def _style_scatter_axes(ax: Any, sc: Any, fig: plt.Figure) -> None:
    """Apply labels, title, colour-bar, and layout.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axes.
    sc : matplotlib.collections.PathCollection
        Scatter artist for the colour-bar.
    fig : matplotlib.figure.Figure
        Parent figure.
    """
    ax.set_xlabel("XGBoost Importance")
    ax.set_ylabel("Embedding Instability")
    ax.set_title("Importance vs Structural Instability")
    plt.colorbar(sc, label="Fusion Index")
    fig.tight_layout()


def importance_vs_instability(results: pd.DataFrame) -> plt.Figure:
    """Scatter plot of XGBoost importance vs embedding instability.

    Point size scales with the Fusion Index and colour encodes it on
    a ``magma`` colourmap.  Gene labels are annotated beside each dot.

    Parameters
    ----------
    results : pd.DataFrame
        Must contain ``XGB_Importance``, ``Instability``,
        ``Fusion_Index``, and ``Gene`` columns.

    Returns
    -------
    matplotlib.figure.Figure
        The rendered scatter-plot figure.
    """
    fig, ax = plt.subplots(figsize=(7, 6))
    sc = _draw_scatter(ax, results)
    _annotate_genes(ax, results)
    _style_scatter_axes(ax, sc, fig)
    return fig


def _empty_heatmap() -> plt.Figure:
    """Return a placeholder figure when no genes match.

    Returns
    -------
    matplotlib.figure.Figure
        A figure with a centred "no matching genes" message.
    """
    fig, ax = plt.subplots()
    ax.text(
        0.5,
        0.5,
        "No matching genes found in expression matrix",
        ha="center",
        va="center",
    )
    return fig


def _draw_heatmap(X: pd.DataFrame, present: list[str], max_samples: int) -> plt.Figure:
    """Render the seaborn heatmap for selected genes.

    Parameters
    ----------
    X : pd.DataFrame
        Expression matrix (samples x genes).
    present : list[str]
        Gene columns that exist in *X*.
    max_samples : int
        Maximum rows to display.

    Returns
    -------
    matplotlib.figure.Figure
        The rendered heatmap figure.
    """
    sub = X[present].iloc[:max_samples]
    fig, ax = plt.subplots(figsize=(max(6, len(present)), 8))
    sns.heatmap(sub, cmap="vlag", center=0, ax=ax, yticklabels=False)
    ax.set_title("Expression Heatmap — Top Fusion Targets")
    fig.tight_layout()
    return fig


def expression_heatmap(
    X: pd.DataFrame,
    y: pd.Series,
    top_genes: list[str],
    max_samples: int = 200,
) -> plt.Figure:
    """Cluster-map of top gene expression across cancer types.

    Parameters
    ----------
    X : pd.DataFrame
        Expression matrix (samples x genes).
    y : pd.Series
        Cancer-type labels (used for row colouring).
    top_genes : list[str]
        Gene names to include as heatmap columns.
    max_samples : int
        Maximum number of samples to display (default 200).

    Returns
    -------
    matplotlib.figure.Figure
        The rendered heatmap figure.
    """
    present = [g for g in top_genes if g in X.columns]
    if not present:
        return _empty_heatmap()
    return _draw_heatmap(X, present, max_samples)


def _color_boxes(bp: dict[str, Any], n: int) -> None:
    """Apply a magma colour palette to box-plot patches.

    Parameters
    ----------
    bp : dict[str, Any]
        Box-plot artist dict returned by ``ax.boxplot``.
    n : int
        Number of boxes (palette size).
    """
    colors = sns.color_palette("magma", n)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)


def instability_boxplot(drift_reports: list[dict[str, Any]]) -> plt.Figure:
    """Box plot of per-gene drift distributions.

    Parameters
    ----------
    drift_reports : list[dict[str, Any]]
        Each dict must have keys ``gene`` (str) and ``all_drifts``
        (list[float]).

    Returns
    -------
    matplotlib.figure.Figure
        The rendered box-plot figure.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    data = [r["all_drifts"] for r in drift_reports]
    labels = [r["gene"] for r in drift_reports]
    bp = ax.boxplot(data, tick_labels=labels, patch_artist=True)
    _color_boxes(bp, len(data))
    ax.set(ylabel="Cosine Drift", title="Mutational Instability Distribution")
    fig.tight_layout()
    return fig


def _scatter_one_group(
    ax: Any, pca_df: pd.DataFrame, mask: pd.Series, lbl: str, color: Any
) -> None:
    """Scatter one cancer-type group on the PCA axes.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axes.
    pca_df : pd.DataFrame
        DataFrame with ``PC1`` and ``PC2`` columns.
    mask : pd.Series
        Boolean mask for the cancer type.
    lbl : str
        Cancer-type label.
    color : Any
        Matplotlib color.
    """
    ax.scatter(
        pca_df.loc[mask, "PC1"],
        pca_df.loc[mask, "PC2"],
        label=lbl,
        s=15,
        alpha=0.7,
        color=color,
    )


def _draw_pca_groups(ax: Any, pca_df: pd.DataFrame, labels: pd.Series) -> None:
    """Draw per-cancer-type scatter groups on the PCA axes.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axes.
    pca_df : pd.DataFrame
        DataFrame with ``PC1`` and ``PC2`` columns.
    labels : pd.Series
        Cancer-type labels aligned with *pca_df* rows.
    """
    unique = labels.unique()
    palette = sns.color_palette("husl", len(unique))
    for i, lbl in enumerate(unique):
        mask = labels == lbl
        _scatter_one_group(ax, pca_df, mask, lbl, palette[i])


def _style_pca_axes(ax: Any, fig: plt.Figure) -> None:
    """Add legend, axis labels, title, and tighten layout.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axes.
    fig : matplotlib.figure.Figure
        Parent figure.
    """
    ax.legend(fontsize=7, loc="best", ncol=2)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("PCA — Sample Clustering by Cancer Type")
    fig.tight_layout()


def pca_scatter(
    pca_df: pd.DataFrame,
    labels: pd.Series,
) -> plt.Figure:
    """2-D PCA scatter plot coloured by cancer type.

    Parameters
    ----------
    pca_df : pd.DataFrame
        DataFrame with ``PC1`` and ``PC2`` columns.
    labels : pd.Series
        Cancer-type labels aligned with *pca_df* rows.

    Returns
    -------
    matplotlib.figure.Figure
        The rendered scatter-plot figure.
    """
    fig, ax = plt.subplots(figsize=(8, 7))
    _draw_pca_groups(ax, pca_df, labels)
    _style_pca_axes(ax, fig)
    return fig


def save_figure(fig: plt.Figure, name: str, config: ProjectConfig | None = None) -> Path:
    """Persist a matplotlib figure to the output directory as PNG.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure to save.
    name : str
        Base filename (without extension).
    config : ProjectConfig, optional
        Supplies ``output_dir`` and ``figure_dpi``.

    Returns
    -------
    Path
        Absolute path to the saved PNG file.
    """
    cfg = config or ProjectConfig()
    path = cfg.output_dir / f"{name}.png"
    fig.savefig(path, dpi=cfg.figure_dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved figure → %s", path)
    return path
