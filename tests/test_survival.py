"""Tests for the survival analysis module.

Covers stratification, Kaplan–Meier fitting, log-rank testing,
and graceful handling of missing lifelines.
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from fusion_oncology.config import ProjectConfig


@pytest.fixture()
def cfg(tmp_path):
    """Return a minimal ProjectConfig for survival tests.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest temporary directory.

    Returns
    -------
    ProjectConfig
        Config with default survival column names.
    """
    return ProjectConfig(
        cache_dir=tmp_path / "cache",
        output_dir=tmp_path / "out",
    )


def _has_lifelines() -> bool:
    """Check whether the lifelines package is importable.

    Returns
    -------
    bool
        ``True`` if lifelines is installed, ``False`` otherwise.
    """
    try:
        import lifelines  # noqa: F401

        return True
    except ImportError:
        return False


def _make_clinical_df(gene: str = "EGFR") -> pd.DataFrame:
    """Build a small clinical DataFrame for survival tests.

    Parameters
    ----------
    gene : str
        Gene column name (default ``"EGFR"``).

    Returns
    -------
    pd.DataFrame
        Clinical data with survival time, event, and gene expression.
    """
    rng = np.random.default_rng(0)
    n = 40
    return pd.DataFrame(
        {
            "OS_MONTHS": rng.uniform(1, 60, n),
            "OS_STATUS": rng.integers(0, 2, n),
            gene: rng.normal(5, 2, n),
        }
    )


# ── import guard tests ─────────────────────────────────────────────────


class TestImportGuard:
    """Tests for lifelines import guard."""

    def test_import_error_without_lifelines(self, cfg) -> None:
        """Verify ImportError is raised when lifelines is missing.

        Parameters
        ----------
        cfg : ProjectConfig
            Fixture-injected test configuration.
        """
        with patch.dict(
            "sys.modules",
            {"lifelines": None, "lifelines.statistics": None},
        ):
            with patch("fusion_oncology.analysis.survival._HAS_LIFELINES", False):
                from fusion_oncology.analysis.survival import SurvivalAnalyzer

                with pytest.raises(ImportError):
                    SurvivalAnalyzer(cfg)


# ── Tests that require lifelines ─────────────────────────────────────────

_skip_no_lifelines = pytest.mark.skipif(
    not _has_lifelines(),
    reason="lifelines not installed",
)


@_skip_no_lifelines
class TestStratify:
    """Tests for ``SurvivalAnalyzer.stratify``."""

    def test_adds_expr_group_column(self, cfg) -> None:
        """Verify stratification adds an ``expr_group`` column.

        Parameters
        ----------
        cfg : ProjectConfig
            Fixture-injected test configuration.
        """
        from fusion_oncology.analysis.survival import SurvivalAnalyzer

        sa = SurvivalAnalyzer(cfg)
        df = _make_clinical_df()
        result = sa.stratify(df, "EGFR")
        assert "expr_group" in result.columns

    def test_two_groups_by_default(self, cfg) -> None:
        """Verify default stratification produces two groups.

        Parameters
        ----------
        cfg : ProjectConfig
            Fixture-injected test configuration.
        """
        from fusion_oncology.analysis.survival import SurvivalAnalyzer

        sa = SurvivalAnalyzer(cfg)
        result = sa.stratify(_make_clinical_df(), "EGFR")
        assert result["expr_group"].nunique() == 2


@_skip_no_lifelines
class TestValidateColumns:
    """Tests for ``SurvivalAnalyzer._validate_columns``."""

    def test_valid_columns(self, cfg) -> None:
        """Verify (time_col, event_col) is returned for valid data.

        Parameters
        ----------
        cfg : ProjectConfig
            Fixture-injected test configuration.
        """
        from fusion_oncology.analysis.survival import SurvivalAnalyzer

        sa = SurvivalAnalyzer(cfg)
        df = _make_clinical_df()
        time_col, event_col = sa._validate_columns(df, "EGFR")
        assert time_col == "OS_MONTHS"
        assert event_col == "OS_STATUS"

    def test_missing_column_raises(self, cfg) -> None:
        """Verify KeyError is raised when a required column is absent.

        Parameters
        ----------
        cfg : ProjectConfig
            Fixture-injected test configuration.
        """
        from fusion_oncology.analysis.survival import SurvivalAnalyzer

        sa = SurvivalAnalyzer(cfg)
        df = pd.DataFrame({"EGFR": [1, 2]})
        with pytest.raises(KeyError):
            sa._validate_columns(df, "EGFR")


@_skip_no_lifelines
class TestSplitGroups:
    """Tests for ``SurvivalAnalyzer._split_groups``."""

    def test_splits_into_two(self, cfg) -> None:
        """Verify high and low subsets are returned.

        Parameters
        ----------
        cfg : ProjectConfig
            Fixture-injected test configuration.
        """
        from fusion_oncology.analysis.survival import SurvivalAnalyzer

        sa = SurvivalAnalyzer(cfg)
        high, low = sa._split_groups(_make_clinical_df(), "EGFR")
        assert len(high) > 0
        assert len(low) > 0


@_skip_no_lifelines
class TestKaplanMeier:
    """Tests for ``SurvivalAnalyzer.kaplan_meier``."""

    def test_returns_expected_keys(self, cfg) -> None:
        """Verify result dict has kmf objects and p-value.

        Parameters
        ----------
        cfg : ProjectConfig
            Fixture-injected test configuration.
        """
        from fusion_oncology.analysis.survival import SurvivalAnalyzer

        sa = SurvivalAnalyzer(cfg)
        result = sa.kaplan_meier(_make_clinical_df(), "EGFR")
        assert "kmf_high" in result
        assert "kmf_low" in result
        assert "logrank_p" in result

    def test_p_value_in_range(self, cfg) -> None:
        """Verify log-rank p-value is between 0 and 1.

        Parameters
        ----------
        cfg : ProjectConfig
            Fixture-injected test configuration.
        """
        from fusion_oncology.analysis.survival import SurvivalAnalyzer

        sa = SurvivalAnalyzer(cfg)
        result = sa.kaplan_meier(_make_clinical_df(), "EGFR")
        assert 0 <= result["logrank_p"] <= 1

    def test_missing_gene_raises(self, cfg) -> None:
        """Verify missing gene column raises KeyError.

        Parameters
        ----------
        cfg : ProjectConfig
            Fixture-injected test configuration.
        """
        from fusion_oncology.analysis.survival import SurvivalAnalyzer

        sa = SurvivalAnalyzer(cfg)
        with pytest.raises(KeyError):
            sa.kaplan_meier(_make_clinical_df(), "NONEXISTENT")


@_skip_no_lifelines
class TestFitKmf:
    """Tests for ``SurvivalAnalyzer._fit_kmf``."""

    def test_returns_fitter(self, cfg) -> None:
        """Verify a fitted KaplanMeierFitter is returned.

        Parameters
        ----------
        cfg : ProjectConfig
            Fixture-injected test configuration.
        """
        from lifelines import KaplanMeierFitter

        from fusion_oncology.analysis.survival import SurvivalAnalyzer

        sa = SurvivalAnalyzer(cfg)
        df = _make_clinical_df()
        kmf = sa._fit_kmf(df, "OS_MONTHS", "OS_STATUS", "test")
        assert isinstance(kmf, KaplanMeierFitter)


@_skip_no_lifelines
class TestBuildKmResult:
    """Tests for ``SurvivalAnalyzer._build_km_result``."""

    def test_contains_medians(self, cfg) -> None:
        """Verify result includes median survival times.

        Parameters
        ----------
        cfg : ProjectConfig
            Fixture-injected test configuration.
        """
        from fusion_oncology.analysis.survival import SurvivalAnalyzer

        sa = SurvivalAnalyzer(cfg)
        df = _make_clinical_df()
        kmf_h = sa._fit_kmf(df, "OS_MONTHS", "OS_STATUS", "HIGH")
        kmf_l = sa._fit_kmf(df, "OS_MONTHS", "OS_STATUS", "LOW")
        result = sa._build_km_result(kmf_h, kmf_l, 0.05)
        assert "median_high" in result
        assert "median_low" in result


@_skip_no_lifelines
class TestLogKmResult:
    """Tests for ``SurvivalAnalyzer._log_km_result``."""

    def test_logs_gene_name(self, cfg, caplog) -> None:
        """Verify the gene name is logged at INFO level.

        Parameters
        ----------
        cfg : ProjectConfig
            Fixture-injected test configuration.
        caplog : pytest.LogCaptureFixture
            Pytest fixture for capturing log output.
        """
        import logging

        from fusion_oncology.analysis.survival import SurvivalAnalyzer

        sa = SurvivalAnalyzer(cfg)
        result = {"logrank_p": 0.01, "median_high": 30, "median_low": 20}
        with caplog.at_level(logging.INFO):
            sa._log_km_result("EGFR", result)
        assert "EGFR" in caplog.text
