"""Tests for ProjectConfig."""

from pathlib import Path

from fusion_oncology.config import ProjectConfig


def test_defaults():
    """Verify that default configuration values match expectations.

    Asserts
    -------
    - ``top_k_genes`` defaults to 15.
    - ``fuzz_iterations`` defaults to 20.
    - ``xgb_n_estimators`` defaults to 1000.
    - ``min_class_size`` defaults to 40.
    - ``enable_hpo`` defaults to False.
    """
    cfg = ProjectConfig()
    assert cfg.top_k_genes == 15
    assert cfg.fuzz_iterations == 20
    assert cfg.xgb_n_estimators == 1000
    assert cfg.min_class_size == 40
    assert cfg.enable_hpo is False


def test_override(tmp_path):
    """Verify that explicit field overrides are respected.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest-provided temporary directory.
    """
    cfg = ProjectConfig(
        top_k_genes=10,
        cache_dir=tmp_path / "c",
        output_dir=tmp_path / "o",
    )
    assert cfg.top_k_genes == 10
    assert cfg.cache_dir.exists()
    assert cfg.output_dir.exists()


def test_directories_created(tmp_path):
    """Verify that ``__post_init__`` creates cache and output dirs.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest-provided temporary directory.
    """
    cfg = ProjectConfig(
        cache_dir=tmp_path / "deep" / "cache",
        output_dir=tmp_path / "deep" / "out",
    )
    assert cfg.cache_dir.is_dir()
    assert cfg.output_dir.is_dir()
