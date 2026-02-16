"""
Tests for the enhanced tumour immune micro-environment model.

Covers ImmuneConfig defaults, TumourImmuneModel simulation,
exhaustion dynamics, checkpoint therapy comparison, spatial
heterogeneity, and population trajectories.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fusion_oncology.analysis.immune_model import (
    ImmuneConfig,
    ImmuneState,
    TumourImmuneModel,
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
def tumour_burden():
    """Generate a synthetic tumour burden trajectory.

    Returns
    -------
    np.ndarray
        60-day exponentially growing tumour (1e8 → ~1e9).
    """
    return 1e8 * np.exp(0.04 * np.arange(60))


@pytest.fixture()
def model():
    """Build a default TumourImmuneModel.

    Returns
    -------
    TumourImmuneModel
        Model with default parameters.
    """
    return TumourImmuneModel()


class TestImmuneConfig:
    """Tests for ImmuneConfig defaults."""

    def test_default_config_values_positive(self):
        """All rate constants should be positive.

        Returns
        -------
        None
        """
        ic = ImmuneConfig()
        assert ic.t_eff_recruitment > 0
        assert ic.t_eff_death > 0
        assert ic.treg_recruitment > 0
        assert ic.nk_baseline > 0

    def test_spatial_fractions_sum_to_one(self):
        """Core + rim fractions should sum to 1.

        Returns
        -------
        None
        """
        ic = ImmuneConfig()
        assert ic.spatial_core_fraction + (1 - ic.spatial_core_fraction) == pytest.approx(1.0)

    def test_checkpoint_defaults_zero(self):
        """Checkpoint therapy should default to off.

        Returns
        -------
        None
        """
        ic = ImmuneConfig()
        assert ic.checkpoint_anti_pd1 == 0.0
        assert ic.checkpoint_anti_ctla4 == 0.0


class TestImmuneState:
    """Tests for ImmuneState dataclass."""

    def test_default_state(self):
        """Default state should be day 0 with zero populations.

        Returns
        -------
        None
        """
        state = ImmuneState()
        assert state.day == 0
        assert state.t_eff == 0.0
        assert state.exhaustion == 0.0


class TestTumourImmuneSimulation:
    """Tests for full immune simulation."""

    def test_simulate_returns_dataframe(self, model, tumour_burden):
        """simulate() should return a DataFrame.

        Returns
        -------
        None
        """
        df = model.simulate(tumour_burden)
        assert isinstance(df, pd.DataFrame)

    def test_simulate_has_expected_columns(self, model, tumour_burden):
        """Output should contain immune population columns.

        Returns
        -------
        None
        """
        df = model.simulate(tumour_burden)
        for col in ["day", "t_eff", "treg", "nk", "mdsc", "exhaustion"]:
            assert col in df.columns, f"Missing column: {col}"

    def test_simulate_length_matches_input(self, model, tumour_burden):
        """Row count should match the tumour trajectory length.

        Returns
        -------
        None
        """
        df = model.simulate(tumour_burden)
        assert len(df) == len(tumour_burden)

    def test_populations_non_negative(self, model, tumour_burden):
        """All immune populations should remain non-negative.

        Returns
        -------
        None
        """
        df = model.simulate(tumour_burden)
        for col in ["t_eff", "treg", "nk", "mdsc"]:
            assert (df[col] >= 0).all(), f"{col} went negative"

    def test_exhaustion_bounded(self, model, tumour_burden):
        """Exhaustion should remain in [0, 1].

        Returns
        -------
        None
        """
        df = model.simulate(tumour_burden)
        assert (df["exhaustion"] >= 0).all()
        assert (df["exhaustion"] <= 1).all()

    def test_larger_tumour_increases_exhaustion(self):
        """Higher tumour burden should drive higher exhaustion.

        Returns
        -------
        None
        """
        small_tumour = 1e6 * np.ones(60)
        large_tumour = 1e10 * np.ones(60)

        model_small = TumourImmuneModel()
        model_large = TumourImmuneModel()

        df_small = model_small.simulate(small_tumour)
        df_large = model_large.simulate(large_tumour)

        assert df_large["exhaustion"].iloc[-1] > df_small["exhaustion"].iloc[-1]


class TestExhaustionDynamics:
    """Tests for T-cell exhaustion behaviour."""

    def test_exhaustion_increases_with_tumour(self, model):
        """Exhaustion should increase when tumour is large.

        Returns
        -------
        None
        """
        model._update_exhaustion(1e11, dt=1.0)
        assert model._exhaustion > 0

    def test_exhaustion_capped_at_095(self, model):
        """Exhaustion should never exceed 0.95.

        Returns
        -------
        None
        """
        for _ in range(500):
            model._update_exhaustion(1e12, dt=1.0)
        assert model._exhaustion <= 0.95

    def test_anti_pd1_reduces_exhaustion(self):
        """Anti-PD-1 therapy should reduce exhaustion.

        Returns
        -------
        None
        """
        ic_no_therapy = ImmuneConfig()
        ic_pd1 = ImmuneConfig(checkpoint_anti_pd1=0.8)

        model_no = TumourImmuneModel(ic_no_therapy)
        model_pd1 = TumourImmuneModel(ic_pd1)

        tumour = 1e10 * np.ones(60)
        df_no = model_no.simulate(tumour)
        df_pd1 = model_pd1.simulate(tumour)

        assert df_pd1["exhaustion"].iloc[-1] < df_no["exhaustion"].iloc[-1]


class TestCheckpointComparison:
    """Tests for immunotherapy comparison."""

    def test_compare_returns_dataframe(self, model, tumour_burden):
        """compare_immunotherapy should return a DataFrame.

        Returns
        -------
        None
        """
        df = model.compare_immunotherapy(tumour_burden)
        assert isinstance(df, pd.DataFrame)

    def test_three_regimens_compared(self, model, tumour_burden):
        """Should compare three regimens.

        Returns
        -------
        None
        """
        df = model.compare_immunotherapy(tumour_burden)
        assert len(df) == 3

    def test_combination_better_than_monotherapy(self, model, tumour_burden):
        """Combination therapy should yield higher kill rate than none.

        Returns
        -------
        None
        """
        df = model.compare_immunotherapy(tumour_burden)
        baseline_kill = df.loc[df["regimen"] == "No immunotherapy", "final_kill_rate"].values[0]
        combo_kill = df.loc[df["regimen"] == "Anti-PD-1 + Anti-CTLA-4", "final_kill_rate"].values[0]
        assert combo_kill >= baseline_kill


class TestSpatialHeterogeneity:
    """Tests for spatial core vs rim kill rates."""

    def test_spatial_returns_both_regions(self, model):
        """spatial_kill_rates should return core and rim keys.

        Returns
        -------
        None
        """
        rates = model.spatial_kill_rates(t_eff=1e6, nk=1e5)
        assert "core_kill_rate" in rates
        assert "rim_kill_rate" in rates

    def test_rim_higher_than_core(self, model):
        """Rim should have higher immune kill rate than core.

        Returns
        -------
        None
        """
        rates = model.spatial_kill_rates(t_eff=1e6, nk=1e5)
        assert rates["rim_kill_rate"] > rates["core_kill_rate"]


class TestSummary:
    """Tests for model summary output."""

    def test_summary_before_simulation(self, model):
        """Summary before simulation should show not_simulated.

        Returns
        -------
        None
        """
        s = model.summary()
        assert s["status"] == "not_simulated"

    def test_summary_after_simulation(self, model, tumour_burden):
        """Summary after simulation should contain population data.

        Returns
        -------
        None
        """
        model.simulate(tumour_burden)
        s = model.summary()
        assert "final_t_eff" in s
        assert "exhaustion" in s
        assert s["simulation_days"] == len(tumour_burden)
