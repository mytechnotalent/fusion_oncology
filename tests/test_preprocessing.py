"""Tests for data preprocessing utilities."""

import numpy as np
import pandas as pd

from fusion_oncology.data.preprocessing import (
    log_normalise,
    reduce_pca,
    standard_scale,
    variance_filter,
)


def _make_df(rows=50, cols=20, seed=0):
    """Create a random expression-like DataFrame for testing.

    Parameters
    ----------
    rows : int
        Number of sample rows.
    cols : int
        Number of gene columns.
    seed : int
        NumPy RNG seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Random float matrix with columns ``G0 … G{cols-1}``.
    """
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        rng.random((rows, cols)), columns=[f"G{i}" for i in range(cols)]
    )


def test_variance_filter_removes_low_variance():
    """Verify that constant columns are dropped by the variance filter.

    Injects two all-ones columns into a random DataFrame and asserts
    that they are removed and the output has fewer columns than the
    input.
    """
    df = _make_df()
    # Inject two constant columns
    df["G0"] = 1.0
    df["G1"] = 1.0
    filtered = variance_filter(df, quantile=0.1)
    assert "G0" not in filtered.columns
    assert filtered.shape[1] < df.shape[1]


def test_log_normalise_positive():
    """Verify that log-normalised values are non-negative.

    Applies ``log_normalise`` to a random DataFrame and asserts every
    cell is >= 0.
    """
    df = _make_df()
    result = log_normalise(df)
    assert (result >= 0).all().all()


def test_standard_scale_zero_mean():
    """Verify that standard-scaled columns have near-zero mean.

    Applies ``standard_scale`` and checks that every column mean is
    within 1e-10 of zero.
    """
    df = _make_df()
    scaled = standard_scale(df)
    means = scaled.mean()
    assert np.allclose(means, 0, atol=1e-10)


def test_reduce_pca_shape():
    """Verify PCA reduction returns the requested component count.

    Reduces a 30×10 matrix to 2 components and asserts the output
    shape is (30, 2) with columns named ``PC1`` and ``PC2``.
    """
    df = _make_df(rows=30, cols=10)
    pca = reduce_pca(df, n_components=2)
    assert pca.shape == (30, 2)
    assert "PC1" in pca.columns
