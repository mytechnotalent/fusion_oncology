"""
Digital twin tumour simulation.

Implements a simplified ODE-based tumour growth model that simulates
how a tumour responds to different treatment regimens.  Supports
monotherapy, combination therapy, and sequential scheduling to
predict optimal treatment strategies *in silico*.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from fusion_oncology.config import ProjectConfig

logger = logging.getLogger(__name__)


# ── Data classes ─────────────────────────────────────────────────────────


@dataclass
class DrugRegimen:
    """Represents a single drug treatment schedule.

    Parameters
    ----------
    name : str
        Drug name.
    efficacy : float
        Kill rate constant (day⁻¹) — fraction of sensitive cells
        killed per day at therapeutic concentration.
    resistance_rate : float
        Rate at which sensitive cells convert to resistant (day⁻¹).
    start_day : int
        Day of treatment start (0-indexed).
    duration_days : int
        Number of days the drug is administered.
    cycle_on : int
        Days on (in a cycle).  Use same as *duration_days* for
        continuous dosing.
    cycle_off : int
        Days off between cycles.  ``0`` for continuous dosing.
    """

    name: str
    efficacy: float = 0.15
    resistance_rate: float = 0.001
    start_day: int = 0
    duration_days: int = 90
    cycle_on: int = 21
    cycle_off: int = 7

    def _past_duration(self, day: int) -> bool:
        """Check whether elapsed time exceeds drug duration.

        Parameters
        ----------
        day : int
            Simulation day.

        Returns
        -------
        bool
            ``True`` if drug has exceeded its duration.
        """
        elapsed = day - self.start_day
        return elapsed >= self.duration_days

    def _in_on_phase(self, day: int) -> bool:
        """Check whether current day falls in the on-phase of a cycle.

        Parameters
        ----------
        day : int
            Simulation day.

        Returns
        -------
        bool
            ``True`` if drug is in its on-cycle phase.
        """
        elapsed = day - self.start_day
        cycle_length = self.cycle_on + self.cycle_off
        if cycle_length == 0:
            return True
        return (elapsed % cycle_length) < self.cycle_on

    def is_active(self, day: int) -> bool:
        """Return whether the drug is being administered on a given day.

        Accounts for start day, total duration, and cycling schedule.

        Parameters
        ----------
        day : int
            Simulation day.

        Returns
        -------
        bool
            ``True`` if drug is active on *day*.
        """
        if day < self.start_day:
            return False
        if self._past_duration(day):
            return False
        return self._in_on_phase(day)


@dataclass
class TumourState:
    """Snapshot of tumour cell populations at a single time point.

    Parameters
    ----------
    day : int
        Simulation day.
    sensitive : float
        Number of drug-sensitive cells.
    resistant : float
        Number of drug-resistant cells.
    immune : float
        Immune cell infiltration level (arbitrary units).
    total : float
        Total tumour burden (sensitive + resistant).
    """

    day: int
    sensitive: float
    resistant: float
    immune: float
    total: float = 0.0

    def __post_init__(self) -> None:
        """Compute total burden after initialisation.

        Returns
        -------
        None
        """
        self.total = self.sensitive + self.resistant


@dataclass
class SimulationConfig:
    """Configuration for a digital twin simulation run.

    Parameters
    ----------
    initial_tumour_size : float
        Starting number of tumour cells.
    growth_rate : float
        Gompertzian growth rate constant (day⁻¹).
    carrying_capacity : float
        Maximum tumour size (Gompertz plateau).
    immune_kill_rate : float
        Rate of immune-mediated cell killing (day⁻¹).
    immune_recruitment : float
        Rate of immune cell recruitment per tumour cell.
    immune_exhaustion : float
        Rate of immune cell exhaustion (day⁻¹).
    resistant_fraction : float
        Initial fraction of resistant cells.
    simulation_days : int
        Total number of days to simulate.
    dt : float
        Time step for Euler integration (days).
    """

    initial_tumour_size: float = 1e9
    growth_rate: float = 0.02
    carrying_capacity: float = 1e12
    immune_kill_rate: float = 0.001
    immune_recruitment: float = 1e-7
    immune_exhaustion: float = 0.05
    resistant_fraction: float = 0.01
    simulation_days: int = 365
    dt: float = 0.5


# ── Tumour growth model ─────────────────────────────────────────────────


class DigitalTwin:
    """ODE-based digital twin for tumour growth simulation.

    Models three interacting populations:
    1. **Sensitive cells** — grow following Gompertzian kinetics,
       killed by drugs and immune cells.
    2. **Resistant cells** — grow independently, unaffected by
       current drugs, arise from sensitive cells.
    3. **Immune cells** — recruited by tumour burden, exhaust over
       time, kill both populations.

    Parameters
    ----------
    sim_config : SimulationConfig, optional
        Simulation parameters.
    project_config : ProjectConfig, optional
        Runtime configuration.
    """

    def __init__(
        self,
        sim_config: SimulationConfig | None = None,
        project_config: ProjectConfig | None = None,
    ) -> None:
        """Initialise the digital twin.

        Parameters
        ----------
        sim_config : SimulationConfig, optional
            Simulation parameters.  Defaults to ``SimulationConfig()``.
        project_config : ProjectConfig, optional
            Runtime configuration.
        """
        self.sim = sim_config or SimulationConfig()
        self.cfg = project_config or ProjectConfig()
        self._trajectory: list[TumourState] = []
        self._regimens: list[DrugRegimen] = []

    def add_regimen(self, regimen: DrugRegimen) -> None:
        """Add a drug treatment regimen to the simulation.

        Parameters
        ----------
        regimen : DrugRegimen
            Drug schedule to add.
        """
        self._regimens.append(regimen)
        logger.info(
            "Added regimen: %s (efficacy=%.3f, days %d–%d)",
            regimen.name,
            regimen.efficacy,
            regimen.start_day,
            regimen.start_day + regimen.duration_days,
        )

    def _growth_rate(self, population: float) -> float:
        """Compute Gompertzian growth rate.

        dN/dt = r × N × ln(K/N), where r is the growth rate constant,
        K is the carrying capacity, and N is the current population.

        Parameters
        ----------
        population : float
            Current cell count.

        Returns
        -------
        float
            Growth rate (cells/day).
        """
        if population <= 0:
            return 0.0
        ratio = self.sim.carrying_capacity / max(population, 1.0)
        if ratio <= 0:
            return 0.0
        return self.sim.growth_rate * population * np.log(ratio)

    def _drug_kill_rate(self, day: int) -> float:
        """Compute combined drug kill rate at a given day.

        Active regimens contribute additively (optimistic assumption
        for synergistic combinations).

        Parameters
        ----------
        day : int
            Current simulation day.

        Returns
        -------
        float
            Combined kill rate (day⁻¹).
        """
        total = 0.0
        for reg in self._regimens:
            if reg.is_active(day):
                total += reg.efficacy
        return total

    def _resistance_conversion_rate(self, day: int) -> float:
        """Compute rate of sensitive→resistant conversion.

        Parameters
        ----------
        day : int
            Current simulation day.

        Returns
        -------
        float
            Conversion rate (day⁻¹).
        """
        total = 0.0
        for reg in self._regimens:
            if reg.is_active(day):
                total += reg.resistance_rate
        return total

    def _init_populations(
        self,
    ) -> tuple[float, float, float, float]:
        """Initialise cell populations and clear trajectory.

        Returns
        -------
        tuple[float, float, float, float]
            Sensitive, resistant, immune counts and initial total.
        """
        self._trajectory.clear()
        frac = self.sim.resistant_fraction
        s = self.sim.initial_tumour_size * (1 - frac)
        r = self.sim.initial_tumour_size * frac
        return s, r, 1e6, s + r

    def _should_record(self, day: float) -> bool:
        """Check if current time step is close to an integer day.

        Parameters
        ----------
        day : float
            Current simulation time.

        Returns
        -------
        bool
            ``True`` if *day* is close to an integer day.
        """
        return abs(day - round(day)) < self.sim.dt / 2 and round(day) >= 0

    def _record_state(
        self,
        day: float,
        s: float,
        r: float,
        immune: float,
    ) -> None:
        """Append a TumourState snapshot to the trajectory.

        Parameters
        ----------
        day : float
            Current simulation time.
        s : float
            Sensitive cell count.
        r : float
            Resistant cell count.
        immune : float
            Immune cell count.
        """
        state = TumourState(
            day=int(round(day)),
            sensitive=max(0, s),
            resistant=max(0, r),
            immune=max(0, immune),
        )
        self._trajectory.append(state)

    def _step_sensitive(
        self,
        s: float,
        immune: float,
        day: int,
    ) -> float:
        """Advance sensitive cell population by one Euler step.

        Parameters
        ----------
        s : float
            Sensitive cell count.
        immune : float
            Immune cell count.
        day : int
            Current simulation day (integer).

        Returns
        -------
        float
            Updated sensitive cell count.
        """
        kill = self._drug_kill_rate(day)
        resist = self._resistance_conversion_rate(day)
        ik = self.sim.immune_kill_rate
        change = self._growth_rate(s) - kill * s - ik * immune * s - resist * s
        return max(0, s + change * self.sim.dt)

    def _step_resistant(
        self,
        s: float,
        r: float,
        immune: float,
        day: int,
    ) -> float:
        """Advance resistant cell population by one Euler step.

        Parameters
        ----------
        s : float
            Updated sensitive cell count.
        r : float
            Resistant cell count.
        immune : float
            Immune cell count.
        day : int
            Current simulation day (integer).

        Returns
        -------
        float
            Updated resistant cell count.
        """
        resist = self._resistance_conversion_rate(day)
        ik = self.sim.immune_kill_rate
        change = self._growth_rate(r) - ik * immune * r * 0.3 + resist * max(0, s)
        return max(0, r + change * self.sim.dt)

    def _step_immune(
        self,
        s: float,
        r: float,
        immune: float,
    ) -> float:
        """Advance immune cell population by one Euler step.

        Parameters
        ----------
        s : float
            Updated sensitive cell count.
        r : float
            Updated resistant cell count.
        immune : float
            Immune cell count.

        Returns
        -------
        float
            Updated immune cell count.
        """
        recruit = self.sim.immune_recruitment * (s + r)
        exhaust = self.sim.immune_exhaustion * immune
        return max(0, immune + (recruit - exhaust) * self.sim.dt)

    def _simulation_step(
        self,
        s: float,
        r: float,
        immune: float,
        day: float,
    ) -> tuple[float, float, float]:
        """Execute one simulation step: record state and advance ODEs.

        Parameters
        ----------
        s : float
            Sensitive cell count.
        r : float
            Resistant cell count.
        immune : float
            Immune cell count.
        day : float
            Current simulation time.

        Returns
        -------
        tuple[float, float, float]
            Updated (sensitive, resistant, immune) counts.
        """
        if self._should_record(day):
            self._record_state(day, s, r, immune)
        s = self._step_sensitive(s, immune, int(day))
        r = self._step_resistant(s, r, immune, int(day))
        immune = self._step_immune(s, r, immune)
        return s, r, immune

    def _build_trajectory_df(
        self,
        initial_total: float,
    ) -> pd.DataFrame:
        """Convert trajectory to a DataFrame with response percentage.

        Parameters
        ----------
        initial_total : float
            Initial tumour burden for response calculation.

        Returns
        -------
        pd.DataFrame
            Trajectory data with ``response_pct`` column.
        """
        df = pd.DataFrame([vars(st) for st in self._trajectory])
        if not df.empty:
            resp = (initial_total - df["total"]) / initial_total * 100
            df["response_pct"] = resp.round(2)
        return df

    def _log_simulation(self, df: pd.DataFrame) -> None:
        """Log summary of simulation results.

        Parameters
        ----------
        df : pd.DataFrame
            Simulation trajectory DataFrame.
        """
        final = df["total"].iloc[-1] if not df.empty else 0
        resp = df["response_pct"].iloc[-1] if not df.empty else 0
        logger.info(
            "Simulation complete: %d days, final tumour %.2e (%.1f%% response)",
            self.sim.simulation_days,
            final,
            resp,
        )

    def _finalize_simulation(
        self,
        initial_total: float,
    ) -> pd.DataFrame:
        """Build trajectory DataFrame and log results.

        Parameters
        ----------
        initial_total : float
            Initial tumour burden for response calculation.

        Returns
        -------
        pd.DataFrame
            Simulation results.
        """
        df = self._build_trajectory_df(initial_total)
        self._log_simulation(df)
        return df

    def simulate(self) -> pd.DataFrame:
        """Run the tumour growth simulation.

        Uses forward Euler integration of the ODE system:

        .. math::
            \\frac{dS}{dt} = g(S) - k_{drug} \\cdot S
                            - k_{immune} \\cdot I \\cdot S
                            - r_{resist} \\cdot S

        .. math::
            \\frac{dR}{dt} = g(R) - k_{immune} \\cdot I \\cdot R \\cdot 0.3
                            + r_{resist} \\cdot S

        .. math::
            \\frac{dI}{dt} = \\alpha \\cdot (S + R)
                            - \\beta \\cdot I

        Returns
        -------
        pd.DataFrame
            Columns: ``day``, ``sensitive``, ``resistant``, ``immune``,
            ``total``, ``response_pct``.
        """
        s, r, immune, initial_total = self._init_populations()
        n_steps = int(self.sim.simulation_days / self.sim.dt)
        for step in range(n_steps + 1):
            day = step * self.sim.dt
            s, r, immune = self._simulation_step(s, r, immune, day)
        return self._finalize_simulation(initial_total)

    @property
    def trajectory(self) -> list[TumourState]:
        """Return the simulation trajectory.

        Returns
        -------
        list[TumourState]
            List of tumour state snapshots.
        """
        return self._trajectory

    def _empty_best_response(self) -> dict[str, Any]:
        """Return default best-response dict for empty trajectory.

        Returns
        -------
        dict[str, Any]
            Default response with zero values.
        """
        return {"day": 0, "total": 0, "response_pct": 0}

    def _compute_best_response(self) -> dict[str, Any]:
        """Compute best response metrics from trajectory.

        Returns
        -------
        dict[str, Any]
            Best response day, total, percentage, and populations.
        """
        initial = self._trajectory[0].total
        best = min(self._trajectory, key=lambda s: s.total)
        pct = round((initial - best.total) / max(initial, 1) * 100, 2)
        return {
            "day": best.day,
            "total": best.total,
            "response_pct": pct,
            "sensitive": best.sensitive,
            "resistant": best.resistant,
        }

    def best_response(self) -> dict[str, Any]:
        """Find the time point of maximum treatment response.

        Returns
        -------
        dict[str, Any]
            Keys: ``day``, ``total``, ``response_pct``,
            ``sensitive``, ``resistant``.
        """
        if not self._trajectory:
            return self._empty_best_response()
        return self._compute_best_response()

    def _classify_recist(self, change: float) -> str:
        """Classify RECIST category from percentage change.

        Parameters
        ----------
        change : float
            Percentage change in tumour burden from baseline.

        Returns
        -------
        str
            One of ``"CR"``, ``"PR"``, ``"SD"``, ``"PD"``.
        """
        if change <= -99:
            return "CR"
        if change <= -30:
            return "PR"
        if change >= 20:
            return "PD"
        return "SD"

    def recist_response(self) -> str:
        """Classify treatment response using RECIST-like criteria.

        Based on percentage change in tumour burden from baseline:
        - **CR** (Complete Response): ≥ 99 % reduction
        - **PR** (Partial Response): ≥ 30 % reduction
        - **SD** (Stable Disease): < 30 % reduction, < 20 % increase
        - **PD** (Progressive Disease): ≥ 20 % increase

        Returns
        -------
        str
            One of ``"CR"``, ``"PR"``, ``"SD"``, ``"PD"``.
        """
        if not self._trajectory:
            return "SD"
        initial = self._trajectory[0].total
        nadir = min(s.total for s in self._trajectory)
        change = (nadir - initial) / max(initial, 1) * 100
        return self._classify_recist(change)

    def _load_regimens(
        self,
        regimens: list[DrugRegimen],
    ) -> None:
        """Clear current regimens and load a new set.

        Parameters
        ----------
        regimens : list[DrugRegimen]
            Drug regimens to load.
        """
        self._regimens.clear()
        for reg in regimens:
            self.add_regimen(reg)

    def _simulate_regimen_set(
        self,
        label: str,
        regimens: list[DrugRegimen],
    ) -> dict[str, Any]:
        """Simulate a single regimen set and return summary metrics.

        Parameters
        ----------
        label : str
            Name for this regimen set.
        regimens : list[DrugRegimen]
            Drug regimens to simulate.

        Returns
        -------
        dict[str, Any]
            Summary metrics including RECIST classification.
        """
        self._load_regimens(regimens)
        df = self.simulate()
        best = self.best_response()
        final = df["total"].iloc[-1] if not df.empty else 0
        return {
            "regimen": label,
            "best_response_day": best["day"],
            "best_response_pct": best["response_pct"],
            "final_tumour": final,
            "recist": self.recist_response(),
        }

    def _build_comparison_df(
        self,
        results: list[dict[str, Any]],
    ) -> pd.DataFrame:
        """Build sorted comparison DataFrame from regimen results.

        Parameters
        ----------
        results : list[dict[str, Any]]
            List of per-regimen summary dicts.

        Returns
        -------
        pd.DataFrame
            Sorted comparison table.
        """
        return (
            pd.DataFrame(results)
            .sort_values("best_response_pct", ascending=False)
            .reset_index(drop=True)
        )

    def compare_regimens(
        self,
        regimen_sets: dict[str, list[DrugRegimen]],
    ) -> pd.DataFrame:
        """Compare multiple treatment regimens head-to-head.

        Runs a separate simulation for each regimen set and returns
        a summary comparison table.

        Parameters
        ----------
        regimen_sets : dict[str, list[DrugRegimen]]
            Mapping of regimen name → list of ``DrugRegimen`` objects.

        Returns
        -------
        pd.DataFrame
            Columns: ``regimen``, ``best_response_day``,
            ``best_response_pct``, ``final_tumour``, ``recist``.
        """
        results = [
            self._simulate_regimen_set(label, regs)
            for label, regs in regimen_sets.items()
        ]
        return self._build_comparison_df(results)

    def _summary_dict(self) -> dict[str, Any]:
        """Assemble the summary dict for the last simulation.

        Returns
        -------
        dict[str, Any]
            Complete summary with all metrics.
        """
        fin = self._trajectory[-1].total if self._trajectory else 0
        return {
            "simulation_days": self.sim.simulation_days,
            "initial_tumour": self.sim.initial_tumour_size,
            "final_tumour": fin,
            "best_response": self.best_response(),
            "recist": self.recist_response(),
            "n_regimens": len(self._regimens),
            "regimen_names": [r.name for r in self._regimens],
        }

    def summary(self) -> dict[str, Any]:
        """Return a summary of the last simulation run.

        Returns
        -------
        dict[str, Any]
            Keys: ``simulation_days``, ``initial_tumour``,
            ``final_tumour``, ``best_response``, ``recist``,
            ``n_regimens``, ``regimen_names``.
        """
        return self._summary_dict()
