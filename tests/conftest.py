"""Shared pytest fixtures for the Fusion Oncology test suite."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fusion_oncology.config import ProjectConfig


@pytest.fixture()
def tiny_config(tmp_path):
    """Create a lightweight :class:`ProjectConfig` for fast tests.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest-provided temporary directory.

    Returns
    -------
    ProjectConfig
        Config with minimal tree count, depth, and gene budget so
        tests complete quickly.
    """
    return ProjectConfig(
        cache_dir=tmp_path / "cache",
        output_dir=tmp_path / "results",
        top_k_genes=2,
        fuzz_iterations=2,
        xgb_n_estimators=5,
        xgb_max_depth=2,
        min_class_size=5,
    )


@pytest.fixture()
def synthetic_expression():
    """Build a small synthetic gene-expression matrix with labels.

    Returns
    -------
    tuple[pd.DataFrame, pd.Series]
        ``(X, y)`` where *X* is 60 samples × 50 genes and *y*
        contains three cancer-type labels (BRCA, LUAD, KIRC).
    """
    rng = np.random.default_rng(0)
    n_samples, n_genes = 60, 50
    genes = [f"GENE{i}" for i in range(n_genes)]
    X = pd.DataFrame(rng.random((n_samples, n_genes)), columns=genes)
    y = pd.Series(np.repeat(["BRCA", "LUAD", "KIRC"], n_samples // 3), name="Class")
    return X, y


@pytest.fixture()
def dummy_sequence():
    """Provide a short deterministic DNA sequence for testing.

    Returns
    -------
    str
        52-character ACGT repeat string.
    """
    return "ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT"


@pytest.fixture()
def sample_results():
    """Create a small fusion-results DataFrame for assertion tests.

    Returns
    -------
    pd.DataFrame
        Three-row table with columns ``Gene``, ``XGB_Importance``,
        ``Instability``, ``Fusion_Index``, and ``Seq_Length``.
    """
    return pd.DataFrame(
        {
            "Gene": ["EGFR", "BRAF", "TP53"],
            "XGB_Importance": [0.12, 0.08, 0.06],
            "Instability": [0.003, 0.005, 0.002],
            "Fusion_Index": [0.36, 0.40, 0.12],
            "Seq_Length": [3000, 2500, 4000],
        }
    )
