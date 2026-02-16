"""Shared pytest fixtures for the Fusion Oncology test suite."""

from __future__ import annotations

import os

# ── Global thread-pool and device guards ─────────────────────────────────
# On macOS, XGBoost ships its own libomp while PyTorch ships another.
# When both are loaded in a single pytest session the duplicate OpenMP
# runtimes deadlock on fork / thread-pool init.  Setting these env vars
# *before* any C extension import avoids the conflict.

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")

import numpy as np
import pandas as pd
import pytest

from fusion_oncology.config import ProjectConfig


@pytest.fixture(autouse=True)
def _force_cpu_torch(monkeypatch):
    """Ensure all torch operations use CPU during tests.

    This prevents MPS backend deadlocks that occur when many test
    files share a single pytest session on Apple Silicon.
    """
    try:
        import torch

        monkeypatch.setenv("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
        # Patch at string-lookup level to avoid initialising backends
        monkeypatch.setattr("torch.cuda.is_available", lambda: False)
        monkeypatch.setattr("torch.backends.mps.is_available", lambda: False)
    except (ImportError, AttributeError):
        pass


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
        ``(X, y)`` where *X* is 60 samples x 50 genes and *y*
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
