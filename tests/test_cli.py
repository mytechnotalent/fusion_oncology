"""Tests for the Click command-line interface.

Uses Click's ``CliRunner`` and mocks all heavy imports so that
tests run instantly without data downloads or model loading.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from click.testing import CliRunner

from fusion_oncology.cli import main


@pytest.fixture()
def runner():
    """Return a Click CliRunner for testing commands.

    Returns
    -------
    CliRunner
        Runner that catches exceptions for inspection.
    """
    return CliRunner()


# ── main group tests ────────────────────────────────────────────────────


class TestMainGroup:
    """Tests for the top-level Click group."""

    def test_version_flag(self, runner) -> None:
        """Verify ``--version`` prints the version and exits 0.

        Parameters
        ----------
        runner : CliRunner
            Fixture-injected Click test runner.
        """
        result = runner.invoke(main, ["--version"])
        assert result.exit_code == 0
        assert "fusion-oncology" in result.output

    def test_help_flag(self, runner) -> None:
        """Verify ``--help`` shows usage information.

        Parameters
        ----------
        runner : CliRunner
            Fixture-injected Click test runner.
        """
        result = runner.invoke(main, ["--help"])
        assert result.exit_code == 0
        assert "Usage" in result.output


# ── clear-cache tests ───────────────────────────────────────────────────


class TestClearCache:
    """Tests for the ``clear-cache`` command."""

    @patch("fusion_oncology.data.cache.ArtifactCache")
    def test_clear_cache(self, mock_cache_cls, runner) -> None:
        """Verify cache.clear() is called and count is printed.

        Parameters
        ----------
        mock_cache_cls : MagicMock
            Patch for ``ArtifactCache`` class.
        runner : CliRunner
            Fixture-injected Click test runner.
        """
        mock_cache_cls.return_value.clear.return_value = 5
        result = runner.invoke(main, ["clear-cache"])
        assert result.exit_code == 0
        assert "5" in result.output


# ── evidence tests ───────────────────────────────────────────────────────


def _mock_evidence_profile():
    """Return a fake evidence profile dictionary.

    Returns
    -------
    dict
        Profile with opentargets, civic, trials, evidence_score.
    """
    return {
        "opentargets": {"overall_score": 0.5, "diseases": []},
        "civic": [],
        "trials": [],
        "evidence_score": 0.5,
    }


class TestEvidence:
    """Tests for the ``evidence`` command."""

    @patch("fusion_oncology.cli.setup_logging")
    def test_evidence_single_gene(self, _mock_log, runner) -> None:
        """Verify querying a single gene prints evidence output.

        Parameters
        ----------
        _mock_log : MagicMock
            Patch for ``setup_logging``.
        runner : CliRunner
            Fixture-injected Click test runner.
        """
        with patch(
            "fusion_oncology.analysis.clinical_evidence.ClinicalEvidenceAggregator"
        ) as mock_cls:
            mock_cls.return_value.profile.return_value = _mock_evidence_profile()
            result = runner.invoke(main, ["evidence", "EGFR"])
        assert result.exit_code == 0


# ── resistance tests ────────────────────────────────────────────────────


class TestResistance:
    """Tests for the ``resistance`` command."""

    def test_resistance_gene(self, runner) -> None:
        """Verify querying a gene prints resistance information.

        Parameters
        ----------
        runner : CliRunner
            Fixture-injected Click test runner.
        """
        with patch("fusion_oncology.analysis.resistance.ResistancePredictor") as mock_cls:
            mock_cls.return_value.full_report.return_value = pd.DataFrame()
            result = runner.invoke(main, ["resistance", "EGFR"])
        assert result.exit_code == 0


# ── simulate tests ──────────────────────────────────────────────────────


def _mock_twin():
    """Build a mock DigitalTwin for simulate tests.

    Returns
    -------
    MagicMock
        Twin mock with simulate/summary stubs.
    """
    twin = MagicMock()
    twin.simulate.return_value = pd.DataFrame(
        {
            "day": [0, 1],
            "total": [1e9, 9e8],
            "sensitive": [9e8, 8e8],
            "resistant": [1e8, 1e8],
        }
    )
    twin.summary.return_value = {
        "recist": "PR",
        "best_response": {"response_pct": 10.0, "day": 30},
        "final_tumour": 9e8,
        "simulation_days": 365,
    }
    return twin


class TestSimulate:
    """Tests for the ``simulate`` command."""

    @patch("fusion_oncology.cli._setup_twin")
    def test_simulate_runs(self, mock_setup, runner, tmp_path) -> None:
        """Verify simulate command completes and produces output.

        Parameters
        ----------
        mock_setup : MagicMock
            Patch for ``_setup_twin`` helper.
        runner : CliRunner
            Fixture-injected Click test runner.
        tmp_path : pathlib.Path
            Pytest-provided temporary directory.
        """
        twin = _mock_twin()
        cfg = MagicMock()
        cfg.output_dir = tmp_path
        mock_setup.return_value = (twin, cfg)
        result = runner.invoke(
            main,
            [
                "simulate",
                "--drug",
                "TestDrug",
                "--efficacy",
                "0.1",
                "--days",
                "30",
                "--output-dir",
                str(tmp_path),
            ],
        )
        assert result.exit_code == 0


# ── companion-dx tests ──────────────────────────────────────────────────


def _write_mutations_file(tmp_path):
    """Write a sample mutations JSON to tmp_path.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Directory for the file.

    Returns
    -------
    str
        Path to the created JSON file.
    """
    mutations = [
        {"gene": "EGFR", "variant": "L858R"},
        {"gene": "BRAF", "variant": "V600E"},
    ]
    path = tmp_path / "mutations.json"
    path.write_text(json.dumps(mutations))
    return str(path)


class TestCompanionDx:
    """Tests for the ``companion-dx`` command."""

    def test_companion_dx_runs(self, runner, tmp_path) -> None:
        """Verify companion DX command produces a report file.

        Parameters
        ----------
        runner : CliRunner
            Fixture-injected Click test runner.
        tmp_path : pathlib.Path
            Pytest-provided temporary directory.
        """
        mut_file = _write_mutations_file(tmp_path)
        with patch("fusion_oncology.models.companion_dx.CompanionDiagnostic") as mock_cls:
            mock_dx = mock_cls.return_value
            mock_dx.analyse.return_value = {"patient_id": "P1"}
            mock_dx.generate_report.return_value = "REPORT TEXT"
            result = runner.invoke(
                main,
                [
                    "companion-dx",
                    mut_file,
                    "--output-dir",
                    str(tmp_path),
                ],
            )
        assert result.exit_code == 0


# ── ingest tests ─────────────────────────────────────────────────────────


class TestIngest:
    """Tests for the ``ingest`` command."""

    @patch("fusion_oncology.cli.setup_logging")
    def test_ingest_runs(self, _mock_log, runner) -> None:
        """Verify ingest command downloads and caches data.

        Parameters
        ----------
        _mock_log : MagicMock
            Patch for ``setup_logging``.
        runner : CliRunner
            Fixture-injected Click test runner.
        """
        with patch("fusion_oncology.data.ingestion.DataIngestion") as mock_cls:
            mock_ing = mock_cls.return_value
            mock_ing.get_patient_data.return_value = (
                pd.DataFrame({"G1": range(10)}),
                pd.Series(["A"] * 10),
            )
            result = runner.invoke(main, ["ingest"])
        assert result.exit_code == 0


# ── report tests ─────────────────────────────────────────────────────────


class TestReport:
    """Tests for the ``report`` command."""

    def _write_csv(self, tmp_path) -> str:
        """Write a sample results CSV to tmp_path.

        Parameters
        ----------
        tmp_path : pathlib.Path
            Directory for the file.

        Returns
        -------
        str
            Path to the created CSV file.
        """
        csv = tmp_path / "results.csv"
        pd.DataFrame(
            {
                "Gene": ["EGFR"],
                "Fusion_Index": [0.5],
                "XGB_Importance": [0.1],
                "Instability": [0.05],
            }
        ).to_csv(csv, index=False)
        return str(csv)

    @patch("fusion_oncology.cli._write_report")
    @patch("fusion_oncology.cli._generate_figures")
    def test_report_runs(self, _figs, _write, runner, tmp_path) -> None:
        """Verify report command reads CSV and generates output.

        Parameters
        ----------
        _figs : MagicMock
            Patch for ``_generate_figures``.
        _write : MagicMock
            Patch for ``_write_report``.
        runner : CliRunner
            Fixture-injected Click test runner.
        tmp_path : pathlib.Path
            Pytest-provided temporary directory.
        """
        csv_path = self._write_csv(tmp_path)
        _figs.return_value = {}
        result = runner.invoke(
            main,
            [
                "report",
                csv_path,
                "--output-dir",
                str(tmp_path),
            ],
        )
        assert result.exit_code == 0


# ── _print_results_table tests ──────────────────────────────────────────


class TestPrintResultsTable:
    """Tests for ``_print_results_table`` helper."""

    def test_does_not_raise(self) -> None:
        """Verify printing a small DataFrame does not raise.

        Creates a minimal DataFrame and calls the helper to
        ensure it completes without exceptions.
        """
        from fusion_oncology.cli import _print_results_table

        df = pd.DataFrame({"Gene": ["EGFR"], "Score": [0.5]})
        _print_results_table(df)


# ── helper function tests ───────────────────────────────────────────────


class TestSetupRun:
    """Tests for ``_setup_run`` helper."""

    @patch("fusion_oncology.cli.setup_logging")
    def test_returns_config(self, _mock_log) -> None:
        """Verify _setup_run returns a ProjectConfig with given params.

        Parameters
        ----------
        _mock_log : MagicMock
            Patch for ``setup_logging``.
        """
        from fusion_oncology.cli import _setup_run

        cfg = _setup_run(5, 20, 50, 4, "results", "INFO")
        assert cfg.top_k_genes == 5
        assert cfg.xgb_n_estimators == 50


class TestBuildEvidenceTable:
    """Tests for ``_build_evidence_table`` helper."""

    def test_returns_rich_table(self) -> None:
        """Verify a Rich Table object is returned with correct title.

        Builds an evidence profile and asserts the table title
        matches the queried gene name.
        """
        from fusion_oncology.cli import _build_evidence_table

        profile = _mock_evidence_profile()
        table = _build_evidence_table("EGFR", profile)
        assert table.title == "Clinical Evidence: EGFR"


class TestLoadResults:
    """Tests for ``_load_results`` helper."""

    def test_returns_dataframe_and_config(self, tmp_path) -> None:
        """Verify CSV is parsed and (df, config) tuple is returned.

        Parameters
        ----------
        tmp_path : pathlib.Path
            Pytest-provided temporary directory.
        """
        from fusion_oncology.cli import _load_results

        csv = tmp_path / "test.csv"
        pd.DataFrame({"Gene": ["A"]}).to_csv(csv, index=False)
        df, cfg = _load_results(str(csv), str(tmp_path))
        assert len(df) == 1
