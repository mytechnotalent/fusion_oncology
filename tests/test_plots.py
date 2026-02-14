"""Tests for the visualization module (smoke tests — no visual assertion)."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fusion_oncology.viz.plots import (
    expression_heatmap,
    fusion_bar,
    importance_vs_instability,
    instability_boxplot,
    pca_scatter,
    save_figure,
)


def test_fusion_bar(sample_results):
    """``fusion_bar`` must return a valid matplotlib Figure.

    Parameters
    ----------
    sample_results : pd.DataFrame
        Fixture providing a small fusion-results table.
    """
    fig = fusion_bar(sample_results)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_importance_vs_instability(sample_results):
    """``importance_vs_instability`` must return a valid matplotlib Figure.

    Parameters
    ----------
    sample_results : pd.DataFrame
        Fixture providing a small fusion-results table.
    """
    fig = importance_vs_instability(sample_results)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_expression_heatmap(synthetic_expression):
    """``expression_heatmap`` must return a valid matplotlib Figure.

    Parameters
    ----------
    synthetic_expression : tuple[pd.DataFrame, pd.Series]
        Fixture providing ``(X, y)``.
    """
    X, y = synthetic_expression
    fig = expression_heatmap(X, y, list(X.columns[:3]))
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_instability_boxplot():
    """Smoke-test the instability box-plot renderer.

    Builds two synthetic drift-report dicts and asserts that
    ``instability_boxplot`` returns a valid ``matplotlib.figure.Figure``
    without raising.
    """
    reports = [
        {"gene": "A", "all_drifts": [0.01, 0.02, 0.015]},
        {"gene": "B", "all_drifts": [0.03, 0.04, 0.035]},
    ]
    fig = instability_boxplot(reports)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_save_figure(tmp_path, sample_results, tiny_config):
    """Saved figure must exist on disk as a ``.png`` file.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest-provided temporary directory.
    sample_results : pd.DataFrame
        Fixture providing a small fusion-results table.
    tiny_config : ProjectConfig
        Lightweight config fixture.
    """
    tiny_config.output_dir = tmp_path
    fig = fusion_bar(sample_results)
    path = save_figure(fig, "test_fig", tiny_config)
    assert path.exists()
    assert path.suffix == ".png"


def test_expression_heatmap_no_matching_genes(synthetic_expression):
    """Heatmap with no matching genes should still return a Figure.

    Parameters
    ----------
    synthetic_expression : tuple[pd.DataFrame, pd.Series]
        Fixture providing ``(X, y)``.
    """
    X, y = synthetic_expression
    fig = expression_heatmap(X, y, ["NONEXISTENT_GENE"])
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_pca_scatter():
    """``pca_scatter`` should return a valid matplotlib Figure.

    Returns
    -------
    None
    """
    rng = np.random.default_rng(42)
    pca_df = pd.DataFrame({"PC1": rng.random(20), "PC2": rng.random(20)})
    labels = pd.Series(["TypeA"] * 10 + ["TypeB"] * 10)
    fig = pca_scatter(pca_df, labels)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_save_figure_default_config(tmp_path, sample_results):
    """save_figure should work with default config when none is given.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest-provided temporary directory.
    sample_results : pd.DataFrame
        Fixture providing a small fusion-results table.
    """
    from fusion_oncology.config import ProjectConfig

    cfg = ProjectConfig(output_dir=tmp_path)
    fig = fusion_bar(sample_results)
    path = save_figure(fig, "default_test", cfg)
    assert path.exists()
    plt.close(fig)
