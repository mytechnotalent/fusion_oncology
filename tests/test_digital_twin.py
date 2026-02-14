"""
Tests for the digital twin tumour simulation module.

Covers DrugRegimen scheduling, SimulationConfig, DigitalTwin ODE
integration, RECIST classification, and regimen comparison.
"""

from __future__ import annotations

import pandas as pd
import pytest

from fusion_oncology.models.digital_twin import (
    DigitalTwin,
    DrugRegimen,
    SimulationConfig,
    TumourState,
)
from fusion_oncology.config import ProjectConfig


@pytest.fixture()
def cfg(tmp_path: "Path") -> ProjectConfig:
    """Provide a temporary ProjectConfig for testing.

    Parameters
    ----------
    tmp_path : Path
        Pytest-provided temporary directory.

    Returns
    -------
    ProjectConfig
        Configuration with temporary paths.
    """
    return ProjectConfig(cache_dir=tmp_path / "cache", output_dir=tmp_path / "out")


class TestDrugRegimen:
    """Tests for the DrugRegimen dataclass."""

    def test_is_active_before_start(self) -> None:
        """Drug should not be active before start day.

        Returns
        -------
        None
        """
        reg = DrugRegimen(name="TestDrug", start_day=10)
        assert not reg.is_active(5)

    def test_is_active_on_start_day(self) -> None:
        """Drug should be active on start day.

        Returns
        -------
        None
        """
        reg = DrugRegimen(name="TestDrug", start_day=0)
        assert reg.is_active(0)

    def test_is_active_after_duration(self) -> None:
        """Drug should not be active after duration expires.

        Returns
        -------
        None
        """
        reg = DrugRegimen(name="TestDrug", start_day=0, duration_days=30)
        assert not reg.is_active(31)

    def test_cycling_on_off(self) -> None:
        """Drug should cycle on and off correctly.

        Returns
        -------
        None
        """
        reg = DrugRegimen(
            name="TestDrug",
            start_day=0,
            duration_days=100,
            cycle_on=3,
            cycle_off=4,
        )
        assert reg.is_active(0)  # day 0: cycle position 0 (on)
        assert reg.is_active(2)  # day 2: cycle position 2 (on)
        assert not reg.is_active(3)  # day 3: cycle position 3 (off)
        assert not reg.is_active(6)  # day 6: cycle position 6 (off)
        assert reg.is_active(7)  # day 7: new cycle, position 0 (on)


class TestTumourState:
    """Tests for the TumourState dataclass."""

    def test_total_computed(self) -> None:
        """Total should equal sensitive + resistant.

        Returns
        -------
        None
        """
        state = TumourState(day=0, sensitive=100, resistant=10, immune=5)
        assert state.total == 110

    def test_zero_populations(self) -> None:
        """Zero populations should give zero total.

        Returns
        -------
        None
        """
        state = TumourState(day=0, sensitive=0, resistant=0, immune=0)
        assert state.total == 0


class TestDigitalTwin:
    """Tests for the DigitalTwin simulation engine."""

    def test_simulate_returns_dataframe(self, cfg: ProjectConfig) -> None:
        """Simulate should return a DataFrame with expected columns.

        Parameters
        ----------
        cfg : ProjectConfig
            Test configuration.
        """
        sim_cfg = SimulationConfig(simulation_days=30, dt=1.0)
        twin = DigitalTwin(sim_config=sim_cfg, project_config=cfg)
        df = twin.simulate()
        assert isinstance(df, pd.DataFrame)
        assert "day" in df.columns
        assert "total" in df.columns
        assert "sensitive" in df.columns
        assert "resistant" in df.columns

    def test_simulate_with_drug(self, cfg: ProjectConfig) -> None:
        """Adding a drug should reduce tumour compared to untreated.

        Parameters
        ----------
        cfg : ProjectConfig
            Test configuration.
        """
        sim_cfg = SimulationConfig(
            simulation_days=30,
            dt=0.1,
            initial_tumour_size=1e9,
            resistant_fraction=0.001,
            immune_kill_rate=1e-12,
            immune_exhaustion=0.5,
            immune_recruitment=1e-15,
        )

        # Untreated
        twin_no_drug = DigitalTwin(sim_config=sim_cfg, project_config=cfg)
        df_no = twin_no_drug.simulate()

        # Treated
        twin_drug = DigitalTwin(sim_config=sim_cfg, project_config=cfg)
        twin_drug.add_regimen(
            DrugRegimen(name="TestDrug", efficacy=0.5, duration_days=30)
        )
        df_drug = twin_drug.simulate()

        # Treated tumour should be smaller at day 30
        final_no = df_no["total"].iloc[-1]
        final_drug = df_drug["total"].iloc[-1]
        assert final_drug < final_no

    def test_recist_response_valid(self, cfg: ProjectConfig) -> None:
        """RECIST classification should be one of CR/PR/SD/PD.

        Parameters
        ----------
        cfg : ProjectConfig
            Test configuration.
        """
        sim_cfg = SimulationConfig(simulation_days=30, dt=1.0)
        twin = DigitalTwin(sim_config=sim_cfg, project_config=cfg)
        twin.simulate()
        recist = twin.recist_response()
        assert recist in {"CR", "PR", "SD", "PD"}

    def test_best_response(self, cfg: ProjectConfig) -> None:
        """Best response should have expected keys.

        Parameters
        ----------
        cfg : ProjectConfig
            Test configuration.
        """
        sim_cfg = SimulationConfig(simulation_days=30, dt=1.0)
        twin = DigitalTwin(sim_config=sim_cfg, project_config=cfg)
        twin.simulate()
        best = twin.best_response()
        assert "day" in best
        assert "total" in best
        assert "response_pct" in best

    def test_compare_regimens(self, cfg: ProjectConfig) -> None:
        """Regimen comparison should return a DataFrame.

        Parameters
        ----------
        cfg : ProjectConfig
            Test configuration.
        """
        sim_cfg = SimulationConfig(simulation_days=60, dt=1.0)
        twin = DigitalTwin(sim_config=sim_cfg, project_config=cfg)

        regimen_sets = {
            "Low dose": [DrugRegimen(name="DrugA", efficacy=0.05, duration_days=60)],
            "High dose": [DrugRegimen(name="DrugA", efficacy=0.25, duration_days=60)],
        }
        comparison = twin.compare_regimens(regimen_sets)
        assert isinstance(comparison, pd.DataFrame)
        assert len(comparison) == 2
        assert "recist" in comparison.columns

    def test_summary(self, cfg: ProjectConfig) -> None:
        """Summary should return a dict with expected keys.

        Parameters
        ----------
        cfg : ProjectConfig
            Test configuration.
        """
        sim_cfg = SimulationConfig(simulation_days=30, dt=1.0)
        twin = DigitalTwin(sim_config=sim_cfg, project_config=cfg)
        twin.simulate()
        summary = twin.summary()
        assert "simulation_days" in summary
        assert "recist" in summary
        assert "final_tumour" in summary
