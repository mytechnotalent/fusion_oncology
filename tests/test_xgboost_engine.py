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
    """Cross-validation must return all six metrics between 0 and 1.

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
    for key in [
        "mean_accuracy",
        "mean_precision",
        "mean_recall",
        "mean_f1",
        "mean_f2",
        "mean_roc_auc",
    ]:
        assert key in metrics, f"missing {key}"
        assert 0 <= metrics[key] <= 1, f"{key}={metrics[key]} out of [0,1]"


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


def test_encode_labels_with_fitted_encoder(synthetic_expression, tiny_config):
    """_encode_labels should use fitted encoder when available.

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
    encoded = eng._encode_labels(y)
    assert len(encoded) == len(y)
    assert encoded.dtype in (np.int64, np.intp, int)


def test_encode_labels_without_fitted_encoder(tiny_config):
    """_encode_labels should create new encoder when no classes fitted.

    Parameters
    ----------
    tiny_config : ProjectConfig
        Lightweight config fixture.
    """
    eng = XGBoostEngine(tiny_config)
    y = pd.Series(["A", "B", "A", "C"])
    encoded = eng._encode_labels(y)
    assert len(encoded) == 4


def test_filter_rare_classes_none_filtered(synthetic_expression, tiny_config):
    """When all classes have enough samples, no filtering should occur.

    Parameters
    ----------
    synthetic_expression : tuple[pd.DataFrame, pd.Series]
        Fixture providing ``(X, y)``.
    tiny_config : ProjectConfig
        Lightweight config fixture.
    """
    X, y = synthetic_expression
    eng = XGBoostEngine(tiny_config)
    X_num = X.select_dtypes(include=[np.number])
    X_out, y_out = eng._filter_rare_classes(X_num, y, folds=2)
    assert len(X_out) == len(X_num)
    assert len(y_out) == len(y)


def test_filter_rare_classes_removes_rare(tiny_config):
    """Classes with fewer samples than folds should be removed.

    Parameters
    ----------
    tiny_config : ProjectConfig
        Lightweight config fixture.
    """
    eng = XGBoostEngine(tiny_config)
    X = pd.DataFrame({"g1": [1, 2, 3, 4, 5, 6]})
    y = pd.Series(["A", "A", "A", "B", "B", "C"])
    X_out, y_out = eng._filter_rare_classes(X, y, folds=2)
    assert "C" not in y_out.values


def test_compute_sample_weights(tiny_config):
    """Sample weights should be inversely proportional to class frequency.

    Parameters
    ----------
    tiny_config : ProjectConfig
        Lightweight config fixture.
    """
    eng = XGBoostEngine(tiny_config)
    y_enc = np.array([0, 0, 0, 1])
    weights = eng._compute_sample_weights(y_enc)
    assert len(weights) == 4
    assert weights[3] > weights[0]


def test_build_scoring_dict(tiny_config):
    """Scoring dict should have all six metric keys.

    Parameters
    ----------
    tiny_config : ProjectConfig
        Lightweight config fixture.
    """
    eng = XGBoostEngine(tiny_config)
    scoring = eng._build_scoring_dict()
    assert "accuracy" in scoring
    assert "precision_weighted" in scoring
    assert "roc_auc_ovr_weighted" in scoring


def test_all_importances_before_fit_raises(tiny_config):
    """Calling all_importances before fit should raise RuntimeError.

    Parameters
    ----------
    tiny_config : ProjectConfig
        Lightweight config fixture.
    """
    eng = XGBoostEngine(tiny_config)
    try:
        eng.all_importances()
        assert False, "Should have raised RuntimeError"
    except RuntimeError:
        pass
