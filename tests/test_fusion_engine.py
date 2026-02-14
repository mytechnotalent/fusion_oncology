"""Tests for the FusionEngine orchestrator.

All sub-engines (XGBoost, DNABERT, pathway, drug-target, etc.)
are mocked so that the orchestration logic is tested in isolation.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from fusion_oncology.config import ProjectConfig


@pytest.fixture()
def cfg(tmp_path):
    """Return a minimal ProjectConfig for fusion tests.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest temporary directory.

    Returns
    -------
    ProjectConfig
        Config with minimal parameters.
    """
    return ProjectConfig(
        cache_dir=tmp_path / "cache",
        output_dir=tmp_path / "out",
        top_k_genes=2,
        fuzz_iterations=2,
        xgb_n_estimators=5,
        xgb_max_depth=2,
    )


# ── mock factory helpers ────────────────────────────────────────────────


def _mock_xgb():
    """Build a mock XGBoostEngine.

    Returns
    -------
    MagicMock
        Mock with ``fit``, ``top_genes``, ``cross_validate`` stubs.
    """
    xgb = MagicMock()
    xgb.fit.return_value = xgb
    xgb.top_genes.return_value = {"EGFR": 0.5, "BRAF": 0.3}
    xgb.cross_validate.return_value = {"mean_accuracy": 0.9}
    return xgb


def _mock_bert():
    """Build a mock DNABERTEngine.

    Returns
    -------
    MagicMock
        Mock whose ``embed`` returns a random 768-d vector.
    """
    bert = MagicMock()
    bert.embed.return_value = np.random.default_rng(0).random(768)
    return bert


def _mock_instability():
    """Build a mock InstabilityAnalyzer.

    Returns
    -------
    MagicMock
        Mock whose ``score`` returns a fixed drift value.
    """
    inst = MagicMock()
    inst.score.return_value = 0.05
    return inst


def _mock_sub_analyzers():
    """Build mocks for pathway, drug, resistance, SL, network.

    Returns
    -------
    tuple
        Five MagicMock objects for downstream analyzers.
    """
    pathway = MagicMock()
    pathway.annotate.side_effect = lambda df: df
    drug = MagicMock()
    drug.annotate.side_effect = lambda df: df
    resistance = MagicMock()
    resistance.annotate.side_effect = lambda df: df
    sl = MagicMock()
    sl.known_partners.return_value = []
    network = MagicMock()
    network.strategic_score = MagicMock(return_value=0.5)
    return pathway, drug, resistance, sl, network


def _build_engine(cfg):
    """Build a FusionEngine with all sub-engines mocked.

    Parameters
    ----------
    cfg : ProjectConfig
        Test configuration.

    Returns
    -------
    FusionEngine
        Fully mocked engine ready for testing.
    """
    from fusion_oncology.models.fusion import FusionEngine

    with patch.object(FusionEngine, "__init__", lambda self, *a, **k: None):
        engine = FusionEngine.__new__(FusionEngine)
    engine.cfg = cfg
    engine.xgb = _mock_xgb()
    engine.bert = _mock_bert()
    engine.instability = _mock_instability()
    pw, dr, res, sl, net = _mock_sub_analyzers()
    engine.pathway = pw
    engine.drug_mapper = dr
    engine.resistance = res
    engine.sl_detector = sl
    engine.network = net
    engine.results = pd.DataFrame()
    engine.cv_metrics = {}
    engine.dnabert_metrics = {}
    return engine


# ── _filter_gene_columns tests ──────────────────────────────────────────


class TestFilterGeneColumns:
    """Tests for ``FusionEngine._filter_gene_columns``."""

    def _make_df(self) -> pd.DataFrame:
        """Return a DataFrame with mixed gene and non-gene columns.

        Returns
        -------
        pd.DataFrame
            Three rows with gene-like and non-gene columns.
        """
        return pd.DataFrame(
            {
                "EGFR": [1, 2, 3],
                "BRAF": [4, 5, 6],
                "Retinoic acid": [7, 8, 9],
                "others": [10, 11, 12],
            }
        )

    def test_keeps_gene_columns(self, cfg) -> None:
        """Verify valid gene-symbol columns are retained.

        Parameters
        ----------
        cfg : ProjectConfig
            Fixture-injected test configuration.
        """
        from fusion_oncology.models.fusion import FusionEngine

        engine = _build_engine(cfg)
        filtered = FusionEngine._filter_gene_columns(engine, self._make_df())
        assert "EGFR" in filtered.columns
        assert "BRAF" in filtered.columns

    def test_removes_non_gene_columns(self, cfg) -> None:
        """Verify non-gene columns are dropped from the DataFrame.

        Parameters
        ----------
        cfg : ProjectConfig
            Fixture-injected test configuration.
        """
        from fusion_oncology.models.fusion import FusionEngine

        engine = _build_engine(cfg)
        filtered = FusionEngine._filter_gene_columns(engine, self._make_df())
        assert "Retinoic acid" not in filtered.columns
        assert "others" not in filtered.columns

    def test_all_nongen_falls_back(self, cfg) -> None:
        """Verify input is returned unchanged when all columns are non-gene.

        Parameters
        ----------
        cfg : ProjectConfig
            Fixture-injected test configuration.
        """
        from fusion_oncology.models.fusion import FusionEngine

        engine = _build_engine(cfg)
        df = pd.DataFrame({"low": [1], "case": [2]})
        result = FusionEngine._filter_gene_columns(engine, df)
        assert list(result.columns) == ["low", "case"]


# ── _train_xgboost tests ────────────────────────────────────────────────


class TestTrainXgboost:
    """Tests for ``FusionEngine._train_xgboost``."""

    def test_returns_top_genes(self, cfg) -> None:
        """Verify the top-gene importance dict is returned.

        Parameters
        ----------
        cfg : ProjectConfig
            Fixture-injected test configuration.
        """
        engine = _build_engine(cfg)
        X = pd.DataFrame({"EGFR": [1, 2], "BRAF": [3, 4]})
        y = pd.Series(["BRCA", "LUAD"])
        result = engine._train_xgboost(X, y)
        assert "EGFR" in result
        assert "BRAF" in result


# ── _fetch_sequences tests ──────────────────────────────────────────────


class TestFetchSequences:
    """Tests for ``FusionEngine._fetch_sequences``."""

    @patch("fusion_oncology.models.fusion.fetch_gene_sequence")
    def test_returns_sequence_mapping(self, mock_fetch, cfg) -> None:
        """Verify each gene is mapped to its fetched sequence.

        Parameters
        ----------
        mock_fetch : MagicMock
            Patch for ``fetch_gene_sequence``.
        cfg : ProjectConfig
            Fixture-injected test configuration.
        """
        mock_fetch.return_value = "ACGTACGT"
        engine = _build_engine(cfg)
        result = engine._fetch_sequences({"EGFR": 0.5})
        assert result["EGFR"] == "ACGTACGT"


# ── _gene_row tests ─────────────────────────────────────────────────────


class TestGeneRow:
    """Tests for ``FusionEngine._gene_row``."""

    def test_contains_expected_keys(self, cfg) -> None:
        """Verify result row has Gene, XGB_Importance, and Instability.

        Parameters
        ----------
        cfg : ProjectConfig
            Fixture-injected test configuration.
        """
        engine = _build_engine(cfg)
        row = engine._gene_row("EGFR", 0.5, "ACGTACGTACGT")
        assert row["Gene"] == "EGFR"
        assert "XGB_Importance" in row
        assert "Instability" in row
        assert "Fusion_Index" in row
        assert "Seq_Length" in row


# ── _score_instability tests ────────────────────────────────────────────


class TestScoreInstability:
    """Tests for ``FusionEngine._score_instability``."""

    def test_populates_results(self, cfg) -> None:
        """Verify results DataFrame is populated after scoring.

        Parameters
        ----------
        cfg : ProjectConfig
            Fixture-injected test configuration.
        """
        engine = _build_engine(cfg)
        top = {"EGFR": 0.5, "BRAF": 0.3}
        seqs = {"EGFR": "ACGTACGT", "BRAF": "TGCATGCA"}
        engine._score_instability(top, seqs)
        assert len(engine.results) == 2
        assert "Fusion_Index" in engine.results.columns


# ── _sl_label tests ──────────────────────────────────────────────────────


class TestSlLabel:
    """Tests for ``FusionEngine._sl_label``."""

    def test_no_partners_returns_dash(self, cfg) -> None:
        """Verify genes with no SL partners return the em-dash.

        Parameters
        ----------
        cfg : ProjectConfig
            Fixture-injected test configuration.
        """
        engine = _build_engine(cfg)
        assert engine._sl_label("FAKEGENE") == "—"

    def test_with_partners_returns_csv(self, cfg) -> None:
        """Verify known partners are returned as comma-separated text.

        Parameters
        ----------
        cfg : ProjectConfig
            Fixture-injected test configuration.
        """
        engine = _build_engine(cfg)
        engine.sl_detector.known_partners.return_value = [
            {"partner": "PARP1"},
            {"partner": "PARP2"},
        ]
        result = engine._sl_label("BRCA1")
        assert "PARP1" in result
        assert "PARP2" in result


# ── summary tests ────────────────────────────────────────────────────────


class TestSummary:
    """Tests for ``FusionEngine.summary``."""

    def test_no_results_message(self, cfg) -> None:
        """Verify a message string is returned when no analysis has run.

        Parameters
        ----------
        cfg : ProjectConfig
            Fixture-injected test configuration.
        """
        engine = _build_engine(cfg)
        assert "No analysis" in engine.summary()

    def _populate_results(self, engine) -> None:
        """Fill engine.results with sample data for summary tests.

        Parameters
        ----------
        engine : FusionEngine
            Engine to populate.
        """
        engine.results = pd.DataFrame(
            {
                "Gene": ["EGFR", "BRAF"],
                "Fusion_Index": [0.5, 0.3],
            }
        )

    def test_with_results(self, cfg) -> None:
        """Verify summary contains the ranking box after analysis.

        Parameters
        ----------
        cfg : ProjectConfig
            Fixture-injected test configuration.
        """
        engine = _build_engine(cfg)
        self._populate_results(engine)
        summary = engine.summary()
        assert "EGFR" in summary
        assert "FUSION ONCOLOGY" in summary

    def test_with_cv_metrics(self, cfg) -> None:
        """Verify CV metrics appear in the summary when available.

        Parameters
        ----------
        cfg : ProjectConfig
            Fixture-injected test configuration.
        """
        engine = _build_engine(cfg)
        self._populate_results(engine)
        engine.cv_metrics = {"mean_accuracy": 0.9, "std_accuracy": 0.02}
        summary = engine.summary()
        assert "Accuracy" in summary


# ── _ranking_row tests ───────────────────────────────────────────────────


class TestRankingRow:
    """Tests for ``FusionEngine._ranking_row``."""

    def test_format_contains_gene(self, cfg) -> None:
        """Verify the formatted row includes the gene name and score.

        Parameters
        ----------
        cfg : ProjectConfig
            Fixture-injected test configuration.
        """
        engine = _build_engine(cfg)
        row = pd.Series({"Gene": "EGFR", "Fusion_Index": 0.1234})
        line = engine._ranking_row(row)
        assert "EGFR" in line
        assert "0.1234" in line


# ── _append_cv_metrics tests ────────────────────────────────────────────


class TestAppendCvMetrics:
    """Tests for ``FusionEngine._append_cv_metrics``."""

    def test_empty_metrics_no_change(self, cfg) -> None:
        """Verify empty cv_metrics leave lines list unchanged.

        Parameters
        ----------
        cfg : ProjectConfig
            Fixture-injected test configuration.
        """
        engine = _build_engine(cfg)
        lines = []
        engine._append_cv_metrics(lines)
        assert len(lines) == 0

    def test_appends_metric_lines(self, cfg) -> None:
        """Verify CV metrics add formatted lines to the list.

        Parameters
        ----------
        cfg : ProjectConfig
            Fixture-injected test configuration.
        """
        engine = _build_engine(cfg)
        engine.cv_metrics = {
            "mean_accuracy": 0.9,
            "std_accuracy": 0.02,
            "mean_f1": 0.85,
            "std_f1": 0.03,
        }
        lines = []
        engine._append_cv_metrics(lines)
        assert any("Accuracy" in ln for ln in lines)


# ── run integration test ─────────────────────────────────────────────────


class TestRun:
    """Integration test for ``FusionEngine.run`` (mocked sub-engines)."""

    @patch("fusion_oncology.models.fusion.fetch_gene_sequence")
    def test_run_returns_dataframe(self, mock_fetch, cfg) -> None:
        """Verify full pipeline returns a non-empty DataFrame.

        Parameters
        ----------
        mock_fetch : MagicMock
            Patch for ``fetch_gene_sequence``.
        cfg : ProjectConfig
            Fixture-injected test configuration.
        """
        mock_fetch.return_value = "ACGTACGTACGTACGTACGT"
        engine = _build_engine(cfg)
        engine.network.strategic_score = lambda gene: 0.5
        X = pd.DataFrame({"EGFR": [1, 2], "BRAF": [3, 4]})
        y = pd.Series(["BRCA", "LUAD"])
        result = engine.run(X, y)
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0
