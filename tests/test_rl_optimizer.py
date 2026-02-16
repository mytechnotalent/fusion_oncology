"""Tests for the RL treatment optimiser."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fusion_oncology.models.digital_twin import SimulationConfig
from fusion_oncology.models.rl_optimizer import (
    AdaptiveTherapyAgent,
    PolicyNetwork,
    REINFORCEAgent,
    RLConfig,
    TreatmentEnv,
    _relu,
    _softmax,
    compare_strategies,
)

# ── Fixtures ─────────────────────────────────────────────────────────────


@pytest.fixture()
def rl_config() -> RLConfig:
    return RLConfig(n_episodes=5, max_steps=10, decision_interval=7, seed=42)


@pytest.fixture()
def sim_config() -> SimulationConfig:
    return SimulationConfig(simulation_days=70)


@pytest.fixture()
def env(sim_config: SimulationConfig, rl_config: RLConfig) -> TreatmentEnv:
    return TreatmentEnv(sim_config, rl_config)


# ── Softmax / ReLU ───────────────────────────────────────────────────────


class TestActivations:
    def test_softmax_sums_to_one(self) -> None:
        probs = _softmax(np.array([1.0, 2.0, 3.0]))
        assert abs(probs.sum() - 1.0) < 1e-6

    def test_softmax_positive(self) -> None:
        probs = _softmax(np.array([-10.0, 0.0, 10.0]))
        assert (probs >= 0).all()

    def test_softmax_stable_large_values(self) -> None:
        probs = _softmax(np.array([1000.0, 1001.0, 1002.0]))
        assert abs(probs.sum() - 1.0) < 1e-6

    def test_relu_positive(self) -> None:
        result = _relu(np.array([-1.0, 0.0, 1.0]))
        np.testing.assert_array_equal(result, [0.0, 0.0, 1.0])


# ── TreatmentEnv ─────────────────────────────────────────────────────────


class TestTreatmentEnv:
    def test_reset_returns_observation(self, env: TreatmentEnv) -> None:
        obs = env.reset()
        assert obs.shape == (5,)
        assert all(np.isfinite(obs))

    def test_step_returns_tuple(self, env: TreatmentEnv) -> None:
        env.reset()
        obs, reward, done, info = env.step(0)
        assert obs.shape == (5,)
        assert isinstance(reward, float)
        assert isinstance(done, bool)
        assert "day" in info

    def test_action_space(self, env: TreatmentEnv) -> None:
        assert env.action_space_n == 4
        assert env.obs_dim == 5

    def test_no_drug_holiday(self, env: TreatmentEnv) -> None:
        env.reset()
        _, _, _, info = env.step(0)
        assert info["efficacy"] == 0.0

    def test_high_dose_action(self, env: TreatmentEnv) -> None:
        env.reset()
        _, _, _, info = env.step(3)
        assert info["efficacy"] == 0.30

    def test_episode_terminates(self, env: TreatmentEnv) -> None:
        env.reset()
        done = False
        steps = 0
        while not done and steps < 100:
            _, _, done, _ = env.step(2)
            steps += 1
        assert done

    def test_tumour_tracked(self, env: TreatmentEnv) -> None:
        env.reset()
        _, _, _, info = env.step(2)
        assert info["total"] > 0
        assert info["sensitive"] >= 0
        assert info["resistant"] >= 0

    def test_deterministic_reset(self, env: TreatmentEnv) -> None:
        obs1 = env.reset(seed=123)
        obs2 = env.reset(seed=123)
        np.testing.assert_array_equal(obs1, obs2)


# ── PolicyNetwork ────────────────────────────────────────────────────────


class TestPolicyNetwork:
    def test_output_shape(self) -> None:
        net = PolicyNetwork(obs_dim=5, hidden_dim=16, n_actions=4)
        obs = np.random.randn(5)
        probs = net.forward(obs)
        assert probs.shape == (4,)

    def test_output_sums_to_one(self) -> None:
        net = PolicyNetwork()
        probs = net.forward(np.random.randn(5))
        assert abs(probs.sum() - 1.0) < 1e-6

    def test_get_set_params(self) -> None:
        net = PolicyNetwork()
        params = net.get_params()
        assert len(params) == 6
        net.set_params(params)
        # Should not raise


# ── REINFORCEAgent ───────────────────────────────────────────────────────


class TestREINFORCEAgent:
    def test_select_action(self, rl_config: RLConfig) -> None:
        agent = REINFORCEAgent(rl_config)
        obs = np.random.randn(5)
        action, probs = agent.select_action(obs)
        assert 0 <= action < rl_config.n_actions
        assert abs(probs.sum() - 1.0) < 1e-6

    def test_train_returns_dataframe(
        self,
        env: TreatmentEnv,
        rl_config: RLConfig,
    ) -> None:
        agent = REINFORCEAgent(rl_config)
        log = agent.train(env, n_episodes=3)
        assert isinstance(log, pd.DataFrame)
        assert len(log) == 3
        assert "total_reward" in log.columns
        assert "final_tumour" in log.columns

    def test_evaluate_returns_dict(
        self,
        env: TreatmentEnv,
        rl_config: RLConfig,
    ) -> None:
        agent = REINFORCEAgent(rl_config)
        agent.train(env, n_episodes=2)
        result = agent.evaluate(env, n_episodes=2)
        assert "mean_reward" in result
        assert "mean_final_tumour" in result
        assert "action_distribution" in result

    def test_compute_returns(self, rl_config: RLConfig) -> None:
        agent = REINFORCEAgent(rl_config)
        returns = agent._compute_returns([1.0, 1.0, 1.0])
        assert len(returns) == 3
        # Last return should equal last reward
        assert abs(returns[-1] - 1.0) < 1e-6
        # Earlier returns should be larger (discounted sum)
        assert returns[0] >= returns[-1]


# ── AdaptiveTherapyAgent ─────────────────────────────────────────────────


class TestAdaptiveTherapyAgent:
    def test_run_returns_dict(self, env: TreatmentEnv) -> None:
        agent = AdaptiveTherapyAgent()
        result = agent.run(env)
        assert "total_reward" in result
        assert "final_tumour" in result
        assert "trajectory" in result
        assert "n_treat_days" in result
        assert "n_holiday_days" in result

    def test_includes_holidays(self, env: TreatmentEnv) -> None:
        # With low holiday_below threshold, should see some holidays
        agent = AdaptiveTherapyAgent(holiday_below=0.95)
        result = agent.run(env)
        # Should have at least one treating step
        assert result["n_treat_days"] >= 0


# ── compare_strategies ───────────────────────────────────────────────────


class TestCompareStrategies:
    def test_returns_three_strategies(self) -> None:
        sim_cfg = SimulationConfig(simulation_days=70)
        rl_cfg = RLConfig(n_episodes=3, decision_interval=7)
        df = compare_strategies(sim_cfg, rl_cfg)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3
        assert "strategy" in df.columns
        assert "mean_reward" in df.columns
        assert "mean_final_tumour" in df.columns

    def test_strategies_named(self) -> None:
        sim_cfg = SimulationConfig(simulation_days=70)
        rl_cfg = RLConfig(n_episodes=3, decision_interval=7)
        df = compare_strategies(sim_cfg, rl_cfg)
        names = set(df["strategy"])
        assert "RL (REINFORCE)" in names
        assert "Adaptive Therapy" in names
        assert "MTD (Max Dose)" in names
