"""Tests for XGBoost engine."""

import numpy as np
import pandas as pd

from fusion_oncology.models.xgboost_engine import XGBoostEngine


def test_fit_and_top_genes(synthetic_expression, tiny_config):
    """Fit the engine and verify ``top_genes`` returns the requested count.

    Parameters
    ----------
    synthetic_expression : tuple[pd.DataFrame, pd.Series]
        Fixture providing ``(X, y)``.
    tiny_config : ProjectConfig
        Lightweight config fixture.
    """
    X, y = synthetic_expression
    eng = XGBoostEngine(tiny_config)
    eng.fit(X, y)
    top = eng.top_genes(k=2)
    assert len(top) == 2
    assert all(isinstance(v, float) for v in top.values())


def test_all_importances(synthetic_expression, tiny_config):
    """All feature importances must be non-negative and match gene count.

    Parameters
    ----------
    synthetic_expression : tuple[pd.DataFrame, pd.Series]
        Fixture providing ``(X, y)``.
    tiny_config : ProjectConfig
        Lightweight config fixture.
    """
    X, y = synthetic_expression
    eng = XGBoostEngine(tiny_config)
    eng.fit(X, y)
    imp = eng.all_importances()
    assert isinstance(imp, pd.Series)
    assert len(imp) == X.select_dtypes(include=[np.number]).shape[1]
    # Importances should be non-negative
    assert (imp >= 0).all()


def test_cross_validate(synthetic_expression, tiny_config):
    """Cross-validation must return accuracy between 0 and 1.

    Parameters
    ----------
    synthetic_expression : tuple[pd.DataFrame, pd.Series]
        Fixture providing ``(X, y)``.
    tiny_config : ProjectConfig
        Lightweight config fixture.
    """
    X, y = synthetic_expression
    eng = XGBoostEngine(tiny_config)
    eng.fit(X, y)
    metrics = eng.cross_validate(X, y, folds=3)
    assert "mean_accuracy" in metrics
    assert 0 <= metrics["mean_accuracy"] <= 1


def test_top_genes_before_fit_raises(tiny_config):
    """Calling ``top_genes`` before ``fit`` must raise ``RuntimeError``.

    Parameters
    ----------
    tiny_config : ProjectConfig
        Lightweight config fixture.
    """
    eng = XGBoostEngine(tiny_config)
    try:
        eng.top_genes()
        assert False, "Should have raised RuntimeError"
    except RuntimeError:
        pass
