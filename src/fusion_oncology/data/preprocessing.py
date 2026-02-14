"""
Feature-level preprocessing for the gene-expression matrix.

Includes variance filtering, log-normalisation, and optional
dimensionality reduction (PCA) for visualisation.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


def _compute_variance_threshold(
    X: pd.DataFrame, quantile: float
) -> tuple[pd.Index, float]:
    """Compute the variance threshold and genes that pass it.

    Parameters
    ----------
    X : pd.DataFrame
        Numeric expression matrix.
    quantile : float
        Fraction of lowest-variance genes to discard.

    Returns
    -------
    tuple[pd.Index, float]
        Index of genes to keep and the computed threshold value.
    """
    variances = X.var()
    threshold = variances.quantile(quantile)
    return variances[variances >= threshold].index, threshold


def variance_filter(X: pd.DataFrame, quantile: float = 0.25) -> pd.DataFrame:
    """
    Keep only genes whose variance exceeds the given quantile threshold.

    Parameters
    ----------
    X : pd.DataFrame
        Raw expression matrix.
    quantile : float
        Fraction of lowest-variance genes to discard (default 25 %).

    Returns
    -------
    pd.DataFrame
        Filtered expression matrix.
    """
    numeric = X.select_dtypes(include=[np.number])
    keep, threshold = _compute_variance_threshold(numeric, quantile)
    msg = "Variance filter: %d → %d genes  (q=%.2f, threshold=%.4f)"
    logger.info(msg, numeric.shape[1], len(keep), quantile, threshold)
    return X[keep]


def log_normalise(X: pd.DataFrame) -> pd.DataFrame:
    """Apply log2(x + 1) normalisation to the expression matrix.

    Parameters
    ----------
    X : pd.DataFrame
        Raw (non-negative) gene-expression matrix.

    Returns
    -------
    pd.DataFrame
        Log-normalised expression matrix (numeric columns only).
    """
    numeric = X.select_dtypes(include=[np.number])
    result = np.log2(numeric + 1)
    logger.info("Log-normalised %d × %d matrix", *result.shape)
    return result


def standard_scale(X: pd.DataFrame) -> pd.DataFrame:
    """Z-score standardise each gene (column) to zero mean, unit variance.

    Parameters
    ----------
    X : pd.DataFrame
        Expression matrix to scale.

    Returns
    -------
    pd.DataFrame
        Scaled matrix with the same column names and index.
    """
    scaler = StandardScaler()
    scaled = scaler.fit_transform(X.select_dtypes(include=[np.number]))
    return pd.DataFrame(
        scaled, columns=X.select_dtypes(include=[np.number]).columns, index=X.index
    )


def reduce_pca(X: pd.DataFrame, n_components: int = 2) -> pd.DataFrame:
    """Project the expression matrix to *n_components* principal components.

    Useful for 2-D / 3-D scatter plots of sample clusters.

    Parameters
    ----------
    X : pd.DataFrame
        Expression matrix (samples × genes).
    n_components : int
        Number of principal components to retain (default 2).

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ``PC1``, ``PC2``, … and the same row index.
    """
    pca = PCA(n_components=n_components)
    coords = pca.fit_transform(X.select_dtypes(include=[np.number]))
    cols = [f"PC{i + 1}" for i in range(n_components)]
    logger.info(
        "PCA: explained variance = %s",
        np.round(pca.explained_variance_ratio_, 4).tolist(),
    )
    return pd.DataFrame(coords, columns=cols, index=X.index)
