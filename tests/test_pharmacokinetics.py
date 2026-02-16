"""
Tests for the two-compartment PK/PD pharmacokinetics model.

Covers DrugPKParams defaults, library lookups, PK simulation,
Emax PD computation, daily kill rates, and steady-state metrics.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fusion_oncology.models.pharmacokinetics import (
    DRUG_PK_LIBRARY,
    DrugPKParams,
    PKPDModel,
)
from fusion_oncology.config import ProjectConfig


@pytest.fixture()
def cfg(tmp_path):
    """Provide a temporary ProjectConfig.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest-provided temporary directory.

    Returns
    -------
    ProjectConfig
        Config with temp paths.
    """
    return ProjectConfig(cache_dir=tmp_path / "cache", output_dir=tmp_path / "out")


@pytest.fixture()
def osi_model():
    """Build a PKPDModel for Osimertinib.

    Returns
    -------
    PKPDModel
        Configured with Osimertinib PK parameters.
    """
    return PKPDModel(DRUG_PK_LIBRARY["Osimertinib"])


class TestDrugPKParams:
    """Tests for the DrugPKParams dataclass."""

    def test_default_params_valid(self):
        """Default parameters should have positive values.

        Returns
        -------
        None
        """
        p = DrugPKParams()
        assert p.ka > 0
        assert p.volume_central > 0
        assert p.clearance > 0
        assert p.bioavailability > 0
        assert p.emax > 0
        assert p.ec50 > 0

    def test_library_contains_expected_drugs(self):
        """Library should contain standard oncology drugs.

        Returns
        -------
        None
        """
        expected = {"Osimertinib", "Erlotinib", "Sotorasib", "Olaparib"}
        assert expected.issubset(DRUG_PK_LIBRARY.keys())

    def test_library_params_are_dataclass(self):
        """Each library entry should be a DrugPKParams instance.

        Returns
        -------
        None
        """
        for name, params in DRUG_PK_LIBRARY.items():
            assert isinstance(params, DrugPKParams), f"{name} is not DrugPKParams"


class TestPKPDModelSimulation:
    """Tests for PK simulation time-course generation."""

    def test_simulate_returns_dataframe(self, osi_model):
        """simulate() should return a DataFrame.

        Returns
        -------
        None
        """
        df = osi_model.simulate(duration_days=3)
        assert isinstance(df, pd.DataFrame)

    def test_simulate_has_expected_columns(self, osi_model):
        """Output should contain time, gut, plasma columns.

        Returns
        -------
        None
        """
        df = osi_model.simulate(duration_days=2)
        for col in ["time_h", "gut_mg", "plasma_conc", "peripheral_conc"]:
            assert col in df.columns, f"Missing column: {col}"

    def test_plasma_concentration_non_negative(self, osi_model):
        """Plasma concentration should be non-negative.

        Returns
        -------
        None
        """
        df = osi_model.simulate(duration_days=3)
        assert (df["plasma_conc"] >= 0).all()

    def test_gut_starts_with_bolus(self, osi_model):
        """First gut value should reflect the bioavailable dose.

        Returns
        -------
        None
        """
        df = osi_model.simulate(duration_days=1)
        first_gut = df["gut_mg"].iloc[0]
        expected = 80.0 * DRUG_PK_LIBRARY["Osimertinib"].bioavailability
        assert abs(first_gut - expected) < 1.0

    def test_simulation_produces_nonzero_plasma(self, osi_model):
        """Simulation should produce non-zero plasma concentrations.

        Returns
        -------
        None
        """
        df = osi_model.simulate(duration_days=3)
        assert df["plasma_conc"].max() > 0


class TestEmaxEffect:
    """Tests for sigmoidal Emax PD function."""

    def test_zero_concentration_zero_effect(self, osi_model):
        """Effect at zero concentration should be zero.

        Returns
        -------
        None
        """
        effect = osi_model.emax_effect(0.0)
        assert effect == pytest.approx(0.0, abs=1e-10)

    def test_high_concentration_near_emax(self, osi_model):
        """Very high concentration should approach Emax.

        Returns
        -------
        None
        """
        ec50 = osi_model.drug.ec50
        effect = osi_model.emax_effect(ec50 * 1000)
        assert effect > 0.95 * osi_model.drug.emax

    def test_ec50_gives_half_emax(self, osi_model):
        """At EC₅₀, effect should be ~50% of Emax.

        Returns
        -------
        None
        """
        ec50 = osi_model.drug.ec50
        effect = osi_model.emax_effect(ec50)
        expected = osi_model.drug.emax / 2
        assert effect == pytest.approx(expected, rel=0.01)

    def test_effect_monotonically_increases(self, osi_model):
        """Effect should increase with concentration.

        Returns
        -------
        None
        """
        concs = [0.01, 0.1, 1.0, 10.0, 100.0]
        effects = [osi_model.emax_effect(c) for c in concs]
        for i in range(1, len(effects)):
            assert effects[i] >= effects[i - 1]


class TestDailyKillRate:
    """Tests for daily kill rate computation."""

    def test_daily_kill_rate_positive(self, osi_model):
        """Kill rate should be positive for a realistic dose.

        Returns
        -------
        None
        """
        rates = osi_model.daily_kill_rate(duration_days=5)
        assert isinstance(rates, np.ndarray)
        assert rates.max() > 0

    def test_daily_kill_rate_length(self, osi_model):
        """daily_kill_rate should return one value per day.

        Returns
        -------
        None
        """
        rates = osi_model.daily_kill_rate(duration_days=7)
        assert len(rates) == 7


class TestSteadyStateMetrics:
    """Tests for steady-state PK/PD metrics."""

    def test_metrics_returns_dict(self, osi_model):
        """steady_state_metrics should return a dict.

        Returns
        -------
        None
        """
        m = osi_model.steady_state_metrics(duration_days=7)
        assert isinstance(m, dict)

    def test_metrics_has_expected_keys(self, osi_model):
        """Output should contain Cmax, Cmin, AUC keys.

        Returns
        -------
        None
        """
        m = osi_model.steady_state_metrics(duration_days=7)
        for key in ["Cmax_ng_mL", "Cmin_ng_mL", "AUC_24h"]:
            assert key in m, f"Missing key: {key}"

    def test_cmax_greater_than_cmin(self, osi_model):
        """Cmax should exceed Cmin (peak > trough).

        Returns
        -------
        None
        """
        m = osi_model.steady_state_metrics(duration_days=7)
        assert m["Cmax_ng_mL"] >= m["Cmin_ng_mL"]
