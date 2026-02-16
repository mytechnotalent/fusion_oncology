"""
Reinforcement learning treatment optimiser.

Uses the :class:`DigitalTwin` tumour simulator as a *gymnasium*-style
environment so that a policy-gradient agent can learn optimal dosing
strategies that maximise tumour reduction while managing resistance
emergence and immune exhaustion.

Key components
--------------
* **TreatmentEnv** — wraps the digital twin ODE system as a
  state → action → reward environment with continuous observation
  space and discrete action space.
* **PolicyNetwork** — lightweight two-hidden-layer MLP that maps
  tumour state observations to action probabilities.
* **REINFORCEAgent** — classic REINFORCE (Williams 1992) policy-gradient
  learner with optional baseline subtraction for variance reduction.
* **AdaptiveTherapyAgent** — rule-based comparator implementing
  Zhang *et al.* (2017) adaptive therapy protocol.

References
----------
Williams, R.J. "Simple Statistical Gradient-Following Algorithms for
Connectionist Reinforcement Learning." *Machine Learning* 8 (1992).
Zhang *et al.* "Integrating evolutionary dynamics into treatment of
metastatic castrate-resistant prostate cancer." *Nat. Commun.* 8 (2017).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from fusion_oncology.config import ProjectConfig
from fusion_oncology.models.digital_twin import (
    DigitalTwin,
    DrugRegimen,
    SimulationConfig,
)

logger = logging.getLogger(__name__)


# ── Configuration ────────────────────────────────────────────────────────


@dataclass
class RLConfig:
    """Hyper-parameters for the RL treatment optimiser.

    Parameters
    ----------
    n_episodes : int
        Training episodes.
    max_steps : int
        Maximum decision steps per episode (each step = ``decision_interval`` days).
    decision_interval : int
        Days between dosing decisions.
    gamma : float
        Discount factor for future rewards.
    learning_rate : float
        Adam-style step size for policy updates.
    hidden_dim : int
        Width of hidden layers in the policy network.
    n_actions : int
        Number of discrete dosing actions.
    baseline_ema : float
        Exponential moving average decay for REINFORCE baseline.
    seed : int
        Random seed for reproducibility.
    """

    n_episodes: int = 200
    max_steps: int = 52
    decision_interval: int = 7
    gamma: float = 0.99
    learning_rate: float = 1e-3
    hidden_dim: int = 64
    n_actions: int = 4
    baseline_ema: float = 0.9
    seed: int = 42


# ── Treatment environment ───────────────────────────────────────────────

_OBS_DIM = 5  # sensitive, resistant, immune, total, day_fraction


@dataclass
class EnvState:
    """Internal environment state snapshot.

    Parameters
    ----------
    sensitive : float
        Sensitive cell count.
    resistant : float
        Resistant cell count.
    immune : float
        Immune cell level.
    day : int
        Current simulation day.
    total_days : int
        Total simulation horizon.
    """

    sensitive: float = 1e9
    resistant: float = 1e7
    immune: float = 1e6
    day: int = 0
    total_days: int = 364


class TreatmentEnv:
    """Gymnasium-style wrapper around the digital twin.

    Observation space (5-dim, log-normalised):
        [log10(sensitive), log10(resistant), log10(immune),
         log10(total), day / total_days]

    Action space (discrete, ``n_actions``):
        0 = no drug (holiday),  1 = low dose (0.05),
        2 = standard dose (0.15),  3 = high dose (0.30)

    Reward: negative change in log tumour burden, with bonuses
    for tumour shrinkage and penalties for excessive resistance.

    Parameters
    ----------
    sim_config : SimulationConfig, optional
        Digital twin simulation parameters.
    rl_config : RLConfig, optional
        RL hyper-parameters (used for ``n_actions``, ``decision_interval``).
    project_config : ProjectConfig, optional
        Runtime configuration.
    """

    # Dosing levels mapped to drug efficacy (day⁻¹)
    ACTION_EFFICACIES: list[float] = [0.0, 0.05, 0.15, 0.30]

    def __init__(
        self,
        sim_config: SimulationConfig | None = None,
        rl_config: RLConfig | None = None,
        project_config: ProjectConfig | None = None,
    ) -> None:
        self.sim_cfg = sim_config or SimulationConfig(simulation_days=364)
        self.rl_cfg = rl_config or RLConfig()
        self.cfg = project_config or ProjectConfig()
        self._state = EnvState(total_days=self.sim_cfg.simulation_days)
        self._rng = np.random.default_rng(self.rl_cfg.seed)
        self.action_space_n = self.rl_cfg.n_actions
        self.obs_dim = _OBS_DIM

    # ── helpers ──────────────────────────────────────────────────────────

    def _log_safe(self, val: float) -> float:
        """Compute log10 clamped to avoid -inf.

        Parameters
        ----------
        val : float
            Input value.

        Returns
        -------
        float
            ``log10(max(val, 1))``.
        """
        return math.log10(max(val, 1.0))

    def _observe(self) -> np.ndarray:
        """Build normalised observation vector from current state.

        Returns
        -------
        np.ndarray
            Shape ``(5,)`` observation.
        """
        s = self._state
        total = s.sensitive + s.resistant
        day_frac = s.day / max(s.total_days, 1)
        return np.array(
            [
                self._log_safe(s.sensitive) / 12.0,
                self._log_safe(s.resistant) / 12.0,
                self._log_safe(s.immune) / 12.0,
                self._log_safe(total) / 12.0,
                day_frac,
            ],
            dtype=np.float64,
        )

    def _step_ode(self, efficacy: float, days: int) -> None:
        """Advance the tumour ODE system by *days* using Euler steps.

        Parameters
        ----------
        efficacy : float
            Drug kill-rate constant (day⁻¹).
        days : int
            Number of simulation days to advance.
        """
        dt = self.sim_cfg.dt
        s = self._state
        for _ in np.arange(0, days, dt):
            # Gompertzian growth
            pop_s = max(s.sensitive, 1.0)
            pop_r = max(s.resistant, 1.0)
            gs = self.sim_cfg.growth_rate * pop_s * math.log(self.sim_cfg.carrying_capacity / pop_s)
            gr = self.sim_cfg.growth_rate * pop_r * math.log(self.sim_cfg.carrying_capacity / pop_r)
            # Drug kill (sensitive only)
            drug_kill = efficacy * s.sensitive
            # Immune kill
            ik = self.sim_cfg.immune_kill_rate
            immune_kill_s = ik * s.immune * s.sensitive
            immune_kill_r = ik * s.immune * s.resistant * 0.3
            # Resistance conversion
            resist = 0.001 * s.sensitive if efficacy > 0 else 0.0
            # Updates
            ds = gs - drug_kill - immune_kill_s - resist
            dr = gr - immune_kill_r + resist
            di = (
                self.sim_cfg.immune_recruitment * (s.sensitive + s.resistant)
                - self.sim_cfg.immune_exhaustion * s.immune
            )
            s.sensitive = max(0.0, s.sensitive + ds * dt)
            s.resistant = max(0.0, s.resistant + dr * dt)
            s.immune = max(0.0, s.immune + di * dt)
        s.day += days

    def _compute_reward(
        self,
        prev_total: float,
        action: int,
    ) -> float:
        """Compute shaped reward.

        Parameters
        ----------
        prev_total : float
            Tumour burden before the step.
        action : int
            Action taken.

        Returns
        -------
        float
            Scalar reward.
        """
        curr_total = self._state.sensitive + self._state.resistant
        # Primary: negative log-ratio change
        log_prev = self._log_safe(prev_total)
        log_curr = self._log_safe(curr_total)
        delta = log_prev - log_curr  # positive = shrinkage

        reward = delta

        # Bonus for significant shrinkage
        if curr_total < prev_total * 0.9:
            reward += 0.5

        # Penalty for resistance fraction
        if curr_total > 0:
            resist_frac = self._state.resistant / curr_total
            if resist_frac > 0.5:
                reward -= 0.3 * resist_frac

        # Small penalty for drug holidays (encourage decisiveness)
        if action == 0 and curr_total > 1e8:
            reward -= 0.1

        return reward

    # ── public API ───────────────────────────────────────────────────────

    def reset(self, seed: int | None = None) -> np.ndarray:
        """Reset environment to initial tumour state.

        Parameters
        ----------
        seed : int, optional
            Override random seed.

        Returns
        -------
        np.ndarray
            Initial observation.
        """
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        init = self.sim_cfg.initial_tumour_size
        frac = self.sim_cfg.resistant_fraction
        self._state = EnvState(
            sensitive=init * (1 - frac),
            resistant=init * frac,
            immune=1e6,
            day=0,
            total_days=self.sim_cfg.simulation_days,
        )
        return self._observe()

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict[str, Any]]:
        """Execute one dosing decision and advance the simulation.

        Parameters
        ----------
        action : int
            Dosing level index.

        Returns
        -------
        tuple[np.ndarray, float, bool, dict]
            ``(observation, reward, done, info)``
        """
        efficacy = self.ACTION_EFFICACIES[min(action, len(self.ACTION_EFFICACIES) - 1)]
        prev_total = self._state.sensitive + self._state.resistant

        self._step_ode(efficacy, self.rl_cfg.decision_interval)

        reward = self._compute_reward(prev_total, action)
        done = self._state.day >= self._state.total_days

        info = {
            "day": self._state.day,
            "sensitive": self._state.sensitive,
            "resistant": self._state.resistant,
            "immune": self._state.immune,
            "total": self._state.sensitive + self._state.resistant,
            "action": action,
            "efficacy": efficacy,
        }
        return self._observe(), reward, done, info


# ── Policy network (numpy-only) ─────────────────────────────────────────


def _softmax(x: np.ndarray) -> np.ndarray:
    """Numerically stable softmax.

    Parameters
    ----------
    x : np.ndarray
        Logits.

    Returns
    -------
    np.ndarray
        Probabilities summing to 1.
    """
    ex = np.exp(x - x.max())
    return ex / ex.sum()


def _relu(x: np.ndarray) -> np.ndarray:
    """Element-wise ReLU activation.

    Parameters
    ----------
    x : np.ndarray
        Input.

    Returns
    -------
    np.ndarray
        ``max(0, x)`` element-wise.
    """
    return np.maximum(0, x)


class PolicyNetwork:
    """Two-layer MLP mapping observations to action probabilities.

    Uses pure NumPy (no PyTorch required at training time for the
    REINFORCE agent).

    Parameters
    ----------
    obs_dim : int
        Observation vector length.
    hidden_dim : int
        Hidden layer width.
    n_actions : int
        Action space size.
    seed : int
        Random seed for weight initialisation.
    """

    def __init__(
        self,
        obs_dim: int = _OBS_DIM,
        hidden_dim: int = 64,
        n_actions: int = 4,
        seed: int = 42,
    ) -> None:
        rng = np.random.default_rng(seed)
        scale1 = np.sqrt(2.0 / obs_dim)
        scale2 = np.sqrt(2.0 / hidden_dim)
        scale3 = np.sqrt(2.0 / hidden_dim)
        self.W1 = rng.normal(0, scale1, (obs_dim, hidden_dim))
        self.b1 = np.zeros(hidden_dim)
        self.W2 = rng.normal(0, scale2, (hidden_dim, hidden_dim))
        self.b2 = np.zeros(hidden_dim)
        self.W3 = rng.normal(0, scale3, (hidden_dim, n_actions))
        self.b3 = np.zeros(n_actions)

    def forward(self, obs: np.ndarray) -> np.ndarray:
        """Compute action probabilities.

        Parameters
        ----------
        obs : np.ndarray
            Observation vector.

        Returns
        -------
        np.ndarray
            Softmax action probabilities.
        """
        h1 = _relu(obs @ self.W1 + self.b1)
        h2 = _relu(h1 @ self.W2 + self.b2)
        logits = h2 @ self.W3 + self.b3
        return _softmax(logits)

    def get_params(self) -> list[np.ndarray]:
        """Return all trainable parameter arrays.

        Returns
        -------
        list[np.ndarray]
            ``[W1, b1, W2, b2, W3, b3]``
        """
        return [self.W1, self.b1, self.W2, self.b2, self.W3, self.b3]

    def set_params(self, params: list[np.ndarray]) -> None:
        """Set trainable parameters from a flat list.

        Parameters
        ----------
        params : list[np.ndarray]
            Six arrays matching ``get_params()`` shapes.
        """
        self.W1, self.b1, self.W2, self.b2, self.W3, self.b3 = params


# ── REINFORCE agent ──────────────────────────────────────────────────────


class REINFORCEAgent:
    """Policy-gradient agent using REINFORCE with baseline.

    Collects full-episode trajectories, computes discounted returns,
    and updates the policy network via the log-probability x
    advantage gradient estimator.

    Parameters
    ----------
    rl_config : RLConfig, optional
        RL hyper-parameters.
    project_config : ProjectConfig, optional
        Runtime configuration.
    """

    def __init__(
        self,
        rl_config: RLConfig | None = None,
        project_config: ProjectConfig | None = None,
    ) -> None:
        self.rl_cfg = rl_config or RLConfig()
        self.cfg = project_config or ProjectConfig()
        self.policy = PolicyNetwork(
            obs_dim=_OBS_DIM,
            hidden_dim=self.rl_cfg.hidden_dim,
            n_actions=self.rl_cfg.n_actions,
            seed=self.rl_cfg.seed,
        )
        self._rng = np.random.default_rng(self.rl_cfg.seed)
        self._baseline = 0.0
        self._episode_rewards: list[float] = []

    def select_action(self, obs: np.ndarray) -> tuple[int, np.ndarray]:
        """Sample an action from the policy.

        Parameters
        ----------
        obs : np.ndarray
            Current observation.

        Returns
        -------
        tuple[int, np.ndarray]
            ``(action, action_probs)``
        """
        probs = self.policy.forward(obs)
        action = int(self._rng.choice(len(probs), p=probs))
        return action, probs

    def _compute_returns(self, rewards: list[float]) -> np.ndarray:
        """Compute discounted cumulative returns.

        Parameters
        ----------
        rewards : list[float]
            Per-step rewards for one episode.

        Returns
        -------
        np.ndarray
            Discounted return at each step.
        """
        T = len(rewards)
        returns = np.zeros(T)
        G = 0.0
        for t in reversed(range(T)):
            G = rewards[t] + self.rl_cfg.gamma * G
            returns[t] = G
        return returns

    def _numerical_gradient(
        self,
        env: TreatmentEnv,
        trajectory: list[dict[str, Any]],
        returns: np.ndarray,
        eps: float = 1e-4,
    ) -> list[np.ndarray]:
        """Estimate policy gradient via finite differences.

        Parameters
        ----------
        env : TreatmentEnv
            Environment instance (unused, kept for API symmetry).
        trajectory : list[dict]
            Recorded trajectory with ``obs``, ``action``, ``probs`` keys.
        returns : np.ndarray
            Discounted returns.
        eps : float
            Finite-difference perturbation.

        Returns
        -------
        list[np.ndarray]
            Gradient arrays matching ``policy.get_params()`` shapes.
        """
        params = self.policy.get_params()
        grads = [np.zeros_like(p) for p in params]

        # Use score-function estimator: ∇ log π(a|s) x (G - baseline)
        for t, step in enumerate(trajectory):
            obs = step["obs"]
            action = step["action"]
            advantage = returns[t] - self._baseline

            # Compute ∇ log π numerically
            for pi, param in enumerate(params):
                flat = param.ravel()
                for idx in range(min(len(flat), 50)):  # cap for speed
                    old_val = flat[idx]
                    flat[idx] = old_val + eps
                    self.policy.set_params(params)
                    probs_plus = self.policy.forward(obs)
                    flat[idx] = old_val - eps
                    self.policy.set_params(params)
                    probs_minus = self.policy.forward(obs)
                    flat[idx] = old_val
                    self.policy.set_params(params)

                    log_grad = (
                        math.log(max(probs_plus[action], 1e-10))
                        - math.log(max(probs_minus[action], 1e-10))
                    ) / (2 * eps)
                    grads[pi].ravel()[idx] += advantage * log_grad

        return grads

    def _update_policy(
        self,
        grads: list[np.ndarray],
    ) -> None:
        """Apply gradient ascent step to policy parameters.

        Parameters
        ----------
        grads : list[np.ndarray]
            Gradient arrays for each parameter.
        """
        params = self.policy.get_params()
        lr = self.rl_cfg.learning_rate
        new_params = [p + lr * g / max(len(grads), 1) for p, g in zip(params, grads)]
        self.policy.set_params(new_params)

    def train(
        self,
        env: TreatmentEnv,
        n_episodes: int | None = None,
        verbose: bool = False,
    ) -> pd.DataFrame:
        """Train the policy via REINFORCE.

        Parameters
        ----------
        env : TreatmentEnv
            Treatment environment.
        n_episodes : int, optional
            Override ``rl_config.n_episodes``.
        verbose : bool
            Print per-episode summaries.

        Returns
        -------
        pd.DataFrame
            Episode-level training log with columns
            ``episode``, ``total_reward``, ``final_tumour``, ``steps``.
        """
        n_ep = n_episodes or self.rl_cfg.n_episodes
        log_rows: list[dict[str, Any]] = []

        for ep in range(n_ep):
            obs = env.reset(seed=self.rl_cfg.seed + ep)
            trajectory: list[dict[str, Any]] = []
            rewards: list[float] = []
            done = False

            while not done:
                action, probs = self.select_action(obs)
                next_obs, reward, done, info = env.step(action)
                trajectory.append({"obs": obs.copy(), "action": action, "probs": probs.copy()})
                rewards.append(reward)
                obs = next_obs

            total_reward = sum(rewards)
            returns = self._compute_returns(rewards)

            # Update baseline (EMA)
            self._baseline = (
                self.rl_cfg.baseline_ema * self._baseline
                + (1 - self.rl_cfg.baseline_ema) * total_reward
            )

            # Policy gradient update
            grads = self._numerical_gradient(env, trajectory, returns)
            self._update_policy(grads)

            self._episode_rewards.append(total_reward)
            final_total = info.get("total", 0.0) if trajectory else 0.0
            log_rows.append(
                {
                    "episode": ep,
                    "total_reward": total_reward,
                    "final_tumour": final_total,
                    "steps": len(trajectory),
                }
            )
            if verbose and ep % max(1, n_ep // 10) == 0:
                logger.info(
                    "Episode %d: reward=%.2f, final_tumour=%.2e",
                    ep,
                    total_reward,
                    final_total,
                )

        return pd.DataFrame(log_rows)

    def evaluate(
        self,
        env: TreatmentEnv,
        n_episodes: int = 5,
    ) -> dict[str, Any]:
        """Evaluate the learned policy (greedy, no exploration).

        Parameters
        ----------
        env : TreatmentEnv
            Treatment environment.
        n_episodes : int
            Number of evaluation episodes.

        Returns
        -------
        dict[str, Any]
            Aggregated evaluation metrics.
        """
        total_rewards = []
        final_tumours = []
        action_histories = []

        for ep in range(n_episodes):
            obs = env.reset(seed=self.rl_cfg.seed + 10000 + ep)
            done = False
            ep_reward = 0.0
            actions = []

            while not done:
                probs = self.policy.forward(obs)
                action = int(np.argmax(probs))  # greedy
                obs, reward, done, info = env.step(action)
                ep_reward += reward
                actions.append(action)

            total_rewards.append(ep_reward)
            final_tumours.append(info.get("total", 0.0))
            action_histories.append(actions)

        return {
            "mean_reward": float(np.mean(total_rewards)),
            "std_reward": float(np.std(total_rewards)),
            "mean_final_tumour": float(np.mean(final_tumours)),
            "action_distribution": dict(
                zip(*np.unique([a for h in action_histories for a in h], return_counts=True))
            ),
            "n_episodes": n_episodes,
        }


# ── Adaptive therapy comparator ──────────────────────────────────────────


class AdaptiveTherapyAgent:
    """Rule-based adaptive therapy protocol (Zhang 2017).

    Applies treatment only when tumour burden exceeds a threshold,
    and pauses when the tumour shrinks below a lower bound.  This
    allows sensitive cells to competitively suppress resistant clones.

    Parameters
    ----------
    treat_above : float
        Fraction of initial burden above which to treat.
    holiday_below : float
        Fraction of initial burden below which to pause treatment.
    efficacy : float
        Drug kill rate when treating.
    """

    def __init__(
        self,
        treat_above: float = 1.0,
        holiday_below: float = 0.5,
        efficacy: float = 0.15,
    ) -> None:
        self.treat_above = treat_above
        self.holiday_below = holiday_below
        self.efficacy = efficacy

    def run(
        self,
        env: TreatmentEnv,
        seed: int = 42,
    ) -> dict[str, Any]:
        """Execute the adaptive therapy protocol.

        Parameters
        ----------
        env : TreatmentEnv
            Treatment environment.
        seed : int
            Random seed.

        Returns
        -------
        dict[str, Any]
            Protocol results including trajectory and metrics.
        """
        obs = env.reset(seed=seed)
        initial_total = env._state.sensitive + env._state.resistant
        threshold_high = initial_total * self.treat_above
        threshold_low = initial_total * self.holiday_below
        treating = True
        done = False
        total_reward = 0.0
        trajectory: list[dict[str, Any]] = []

        while not done:
            current_total = env._state.sensitive + env._state.resistant
            if treating and current_total < threshold_low:
                treating = False
            elif not treating and current_total > threshold_high:
                treating = True

            # Map treating to action index
            action = 2 if treating else 0  # standard dose or holiday
            obs, reward, done, info = env.step(action)
            total_reward += reward
            info["treating"] = treating
            trajectory.append(info)

        return {
            "total_reward": total_reward,
            "final_tumour": trajectory[-1]["total"] if trajectory else 0.0,
            "trajectory": trajectory,
            "n_treat_days": sum(1 for t in trajectory if t.get("treating", False)),
            "n_holiday_days": sum(1 for t in trajectory if not t.get("treating", False)),
        }


# ── Convenience comparison ───────────────────────────────────────────────


def compare_strategies(
    sim_config: SimulationConfig | None = None,
    rl_config: RLConfig | None = None,
    project_config: ProjectConfig | None = None,
) -> pd.DataFrame:
    """Train an RL agent and compare against adaptive therapy and MTD.

    Parameters
    ----------
    sim_config : SimulationConfig, optional
        Simulation parameters.
    rl_config : RLConfig, optional
        RL training hyper-parameters.
    project_config : ProjectConfig, optional
        Runtime configuration.

    Returns
    -------
    pd.DataFrame
        Comparison table with columns ``strategy``, ``mean_reward``,
        ``mean_final_tumour``.
    """
    rl_cfg = rl_config or RLConfig(n_episodes=50, max_steps=52)
    sim_cfg = sim_config or SimulationConfig(simulation_days=364)

    env = TreatmentEnv(sim_cfg, rl_cfg, project_config)

    # 1. Train RL agent
    agent = REINFORCEAgent(rl_cfg, project_config)
    agent.train(env, verbose=False)
    rl_eval = agent.evaluate(env, n_episodes=3)

    # 2. Adaptive therapy baseline
    adaptive = AdaptiveTherapyAgent()
    adapt_result = adaptive.run(env)

    # 3. Maximum tolerated dose (always max dose)
    mtd_rewards = []
    mtd_tumours = []
    for ep in range(3):
        obs = env.reset(seed=42 + ep)
        done = False
        rr = 0.0
        while not done:
            obs, reward, done, info = env.step(3)  # always high dose
            rr += reward
        mtd_rewards.append(rr)
        mtd_tumours.append(info.get("total", 0.0))

    rows = [
        {
            "strategy": "RL (REINFORCE)",
            "mean_reward": rl_eval["mean_reward"],
            "mean_final_tumour": rl_eval["mean_final_tumour"],
        },
        {
            "strategy": "Adaptive Therapy",
            "mean_reward": adapt_result["total_reward"],
            "mean_final_tumour": adapt_result["final_tumour"],
        },
        {
            "strategy": "MTD (Max Dose)",
            "mean_reward": float(np.mean(mtd_rewards)),
            "mean_final_tumour": float(np.mean(mtd_tumours)),
        },
    ]
    return pd.DataFrame(rows)
