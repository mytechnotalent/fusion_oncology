"""
Enhanced tumour immune micro-environment (TIME) model.

Extends the base ``DigitalTwin`` single-compartment immune cell with
a structured immune landscape comprising:

1. **Effector T-cells** (CD8⁺) — primary anti-tumour cytotoxicity.
2. **Regulatory T-cells** (Treg) — immunosuppressive, dampen effectors.
3. **NK cells** — innate killer cells, MHC-independent.
4. **MDSCs** — myeloid-derived suppressor cells, create
   immunosuppressive niche.
5. **Exhaustion dynamics** — PD-1/LAG-3 driven effector dysfunction.
6. **Checkpoint blockade** — models anti-PD-1 / anti-CTLA-4 therapy
   restoring effector function.
7. **Spatial heterogeneity** — core vs rim tumour sub-regions with
   differential immune infiltration.

The ODE system:

.. math::

    \\frac{dT_{eff}}{dt} = \\alpha_T \\cdot (S + R)
                          - \\beta_T \\cdot T_{eff}
                          - \\kappa_{Treg} \\cdot T_{reg} \\cdot T_{eff}
                          - \\epsilon \\cdot T_{eff}
                          + \\delta_{CPI} \\cdot \\epsilon \\cdot T_{eff}

    \\frac{dT_{reg}}{dt} = \\alpha_{reg} \\cdot (S + R)
                          - \\beta_{reg} \\cdot T_{reg}
                          - \\delta_{CTLA4} \\cdot T_{reg}

    \\frac{dNK}{dt} = \\sigma_{NK} - \\beta_{NK} \\cdot NK
                     + \\gamma_{NK} \\cdot (S + R)

    \\frac{dMDSC}{dt} = \\alpha_{MDSC} \\cdot (S + R)
                       - \\beta_{MDSC} \\cdot MDSC

References
----------
Kirschner & Panetta, J. Math. Biol. 1998 — tumour-immune dynamics.
Ribas & Wolchok, Science 2018 — cancer immunotherapy review.
de Pillis et al., J. Theor. Biol. 2005 — mixed immune-chemo ODE model.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from fusion_oncology.config import ProjectConfig

logger = logging.getLogger(__name__)


# ── Configuration ────────────────────────────────────────────────────────


@dataclass
class ImmuneConfig:
    """Parameters for the structured immune micro-environment model.

    Parameters
    ----------
    t_eff_recruitment : float
        Rate of effector T-cell recruitment per tumour cell (day⁻¹).
    t_eff_exhaustion_base : float
        Baseline exhaustion rate of effector T-cells (day⁻¹).
    t_eff_death : float
        Natural death rate of effector T-cells (day⁻¹).
    t_eff_kill_rate : float
        Kill rate of effector T-cells against sensitive tumour (day⁻¹).
    t_eff_kill_resistant : float
        Kill rate against resistant tumour cells (reduced, day⁻¹).
    treg_recruitment : float
        Treg recruitment rate per tumour cell (day⁻¹).
    treg_death : float
        Treg natural death rate (day⁻¹).
    treg_suppression : float
        Rate at which Tregs suppress effectors (day⁻¹ per Treg).
    nk_baseline : float
        Baseline NK cell count (homeostatic production, cells/day).
    nk_death : float
        NK cell death rate (day⁻¹).
    nk_recruitment : float
        NK recruitment rate per tumour cell (day⁻¹).
    nk_kill_rate : float
        NK kill rate against tumour cells (day⁻¹).
    mdsc_recruitment : float
        MDSC recruitment rate per tumour cell (day⁻¹).
    mdsc_death : float
        MDSC death rate (day⁻¹).
    mdsc_suppression : float
        Rate at which MDSCs suppress effector T-cells (day⁻¹).
    checkpoint_anti_pd1 : float
        Fraction of exhaustion reversed by anti-PD-1 (0–1).
    checkpoint_anti_ctla4 : float
        Rate of Treg depletion by anti-CTLA-4 (day⁻¹).
    spatial_core_fraction : float
        Fraction of tumour in the poorly-infiltrated core.
    spatial_rim_infiltration : float
        Immune infiltration multiplier for the rim (relative to core).
    initial_t_eff : float
        Starting effector T-cell count.
    initial_treg : float
        Starting Treg count.
    initial_nk : float
        Starting NK cell count.
    initial_mdsc : float
        Starting MDSC count.
    """

    # Effector T-cells
    t_eff_recruitment: float = 2e-7
    t_eff_exhaustion_base: float = 0.03
    t_eff_death: float = 0.02
    t_eff_kill_rate: float = 3e-9
    t_eff_kill_resistant: float = 1e-9

    # Regulatory T-cells
    treg_recruitment: float = 5e-8
    treg_death: float = 0.03
    treg_suppression: float = 1e-7

    # NK cells
    nk_baseline: float = 5e4
    nk_death: float = 0.05
    nk_recruitment: float = 1e-8
    nk_kill_rate: float = 2e-9

    # MDSCs
    mdsc_recruitment: float = 3e-8
    mdsc_death: float = 0.04
    mdsc_suppression: float = 5e-8

    # Checkpoint immunotherapy
    checkpoint_anti_pd1: float = 0.0
    checkpoint_anti_ctla4: float = 0.0

    # Spatial heterogeneity
    spatial_core_fraction: float = 0.6
    spatial_rim_infiltration: float = 3.0

    # Initial populations
    initial_t_eff: float = 5e5
    initial_treg: float = 1e5
    initial_nk: float = 2e5
    initial_mdsc: float = 5e4


@dataclass
class ImmuneState:
    """Snapshot of immune population state at a single time point.

    Parameters
    ----------
    day : int
        Simulation day.
    t_eff : float
        Effector T-cell count.
    treg : float
        Regulatory T-cell count.
    nk : float
        NK cell count.
    mdsc : float
        MDSC count.
    exhaustion : float
        Current exhaustion level (0–1).
    total_kill_rate : float
        Combined immune-mediated kill rate at this time point.
    """

    day: int = 0
    t_eff: float = 0.0
    treg: float = 0.0
    nk: float = 0.0
    mdsc: float = 0.0
    exhaustion: float = 0.0
    total_kill_rate: float = 0.0


# ── Enhanced Immune Model ────────────────────────────────────────────────


class TumourImmuneModel:
    """Structured immune micro-environment model.

    Tracks four immune cell populations and their interactions with
    the tumour, supporting checkpoint immunotherapy simulation and
    spatial heterogeneity between tumour core and rim.

    Parameters
    ----------
    immune_config : ImmuneConfig, optional
        Immune model parameters.
    project_config : ProjectConfig, optional
        Runtime configuration.
    """

    def __init__(
        self,
        immune_config: ImmuneConfig | None = None,
        project_config: ProjectConfig | None = None,
    ) -> None:
        """Initialise the immune model.

        Parameters
        ----------
        immune_config : ImmuneConfig, optional
            Immune parameters.  Defaults to ``ImmuneConfig()``.
        project_config : ProjectConfig, optional
            Runtime configuration.
        """
        self.ic = immune_config or ImmuneConfig()
        self.cfg = project_config or ProjectConfig()
        self._trajectory: list[ImmuneState] = []
        self._exhaustion: float = 0.0

    # ── initialisation ───────────────────────────────────────────────────

    def _init_state(self) -> tuple[float, float, float, float]:
        """Return initial immune population counts.

        Returns
        -------
        tuple[float, float, float, float]
            ``(t_eff, treg, nk, mdsc)`` initial counts.
        """
        self._trajectory.clear()
        self._exhaustion = 0.0
        return (
            self.ic.initial_t_eff,
            self.ic.initial_treg,
            self.ic.initial_nk,
            self.ic.initial_mdsc,
        )

    # ── exhaustion dynamics ──────────────────────────────────────────────

    def _update_exhaustion(
        self,
        tumour_burden: float,
        dt: float,
    ) -> float:
        """Update the T-cell exhaustion level.

        Exhaustion increases with tumour burden (chronic antigen
        stimulation) and is partially reversed by anti-PD-1/PD-L1.

        Parameters
        ----------
        tumour_burden : float
            Current total tumour cell count.
        dt : float
            Time step (days).

        Returns
        -------
        float
            Updated exhaustion level in [0, 1].
        """
        burden_factor = min(1.0, tumour_burden / 1e11)
        d_exhaust = (
            self.ic.t_eff_exhaustion_base * burden_factor
            - self.ic.checkpoint_anti_pd1 * self._exhaustion
        ) * dt
        self._exhaustion = np.clip(self._exhaustion + d_exhaust, 0.0, 0.95)
        return self._exhaustion

    # ── ODE step functions ───────────────────────────────────────────────

    def _step_t_eff(
        self,
        t_eff: float,
        treg: float,
        mdsc: float,
        tumour: float,
        dt: float,
    ) -> float:
        """Advance effector T-cell population.

        Parameters
        ----------
        t_eff : float
            Current effector count.
        treg : float
            Regulatory T-cell count.
        mdsc : float
            MDSC count.
        tumour : float
            Total tumour burden.
        dt : float
            Time step (days).

        Returns
        -------
        float
            Updated effector T-cell count.
        """
        recruitment = self.ic.t_eff_recruitment * tumour
        death = self.ic.t_eff_death * t_eff
        treg_supp = self.ic.treg_suppression * treg * t_eff / max(tumour, 1)
        mdsc_supp = self.ic.mdsc_suppression * mdsc * t_eff / max(tumour, 1)
        exhaustion_loss = self._exhaustion * t_eff * 0.01

        d_teff = recruitment - death - treg_supp - mdsc_supp - exhaustion_loss
        return max(0.0, t_eff + d_teff * dt)

    def _step_treg(
        self,
        treg: float,
        tumour: float,
        dt: float,
    ) -> float:
        """Advance regulatory T-cell population.

        Parameters
        ----------
        treg : float
            Current Treg count.
        tumour : float
            Total tumour burden.
        dt : float
            Time step (days).

        Returns
        -------
        float
            Updated Treg count.
        """
        recruitment = self.ic.treg_recruitment * tumour
        death = self.ic.treg_death * treg
        ctla4_depletion = self.ic.checkpoint_anti_ctla4 * treg

        d_treg = recruitment - death - ctla4_depletion
        return max(0.0, treg + d_treg * dt)

    def _step_nk(
        self,
        nk: float,
        tumour: float,
        dt: float,
    ) -> float:
        """Advance NK cell population.

        Parameters
        ----------
        nk : float
            Current NK count.
        tumour : float
            Total tumour burden.
        dt : float
            Time step (days).

        Returns
        -------
        float
            Updated NK count.
        """
        homeostatic = self.ic.nk_baseline
        recruitment = self.ic.nk_recruitment * tumour
        death = self.ic.nk_death * nk

        d_nk = homeostatic + recruitment - death
        return max(0.0, nk + d_nk * dt)

    def _step_mdsc(
        self,
        mdsc: float,
        tumour: float,
        dt: float,
    ) -> float:
        """Advance MDSC population.

        Parameters
        ----------
        mdsc : float
            Current MDSC count.
        tumour : float
            Total tumour burden.
        dt : float
            Time step (days).

        Returns
        -------
        float
            Updated MDSC count.
        """
        recruitment = self.ic.mdsc_recruitment * tumour
        death = self.ic.mdsc_death * mdsc

        d_mdsc = recruitment - death
        return max(0.0, mdsc + d_mdsc * dt)

    # ── kill rate computation ────────────────────────────────────────────

    def _effective_kill_rate(
        self,
        t_eff: float,
        nk: float,
    ) -> float:
        """Compute the combined immune kill rate against sensitive cells.

        Accounts for T-cell exhaustion reducing effector cytotoxicity.

        Parameters
        ----------
        t_eff : float
            Effector T-cell count.
        nk : float
            NK cell count.

        Returns
        -------
        float
            Combined kill rate (day⁻¹).
        """
        eff_function = 1.0 - self._exhaustion
        t_kill = self.ic.t_eff_kill_rate * t_eff * eff_function
        nk_kill = self.ic.nk_kill_rate * nk
        return t_kill + nk_kill

    def _resistant_kill_rate(
        self,
        t_eff: float,
        nk: float,
    ) -> float:
        """Compute immune kill rate against resistant cells.

        Resistant cells have reduced immune susceptibility.

        Parameters
        ----------
        t_eff : float
            Effector count.
        nk : float
            NK count.

        Returns
        -------
        float
            Kill rate against resistant cells (day⁻¹).
        """
        eff_function = 1.0 - self._exhaustion
        return self.ic.t_eff_kill_resistant * t_eff * eff_function + self.ic.nk_kill_rate * nk * 0.5

    # ── spatial heterogeneity ────────────────────────────────────────────

    def spatial_kill_rates(
        self,
        t_eff: float,
        nk: float,
    ) -> dict[str, float]:
        """Compute spatially-resolved kill rates (core vs rim).

        The tumour core is poorly infiltrated, while the rim has
        higher immune access.

        Parameters
        ----------
        t_eff : float
            Effector count.
        nk : float
            NK count.

        Returns
        -------
        dict[str, float]
            ``core_kill_rate`` and ``rim_kill_rate`` (day⁻¹).
        """
        base_kill = self._effective_kill_rate(t_eff, nk)
        core_fraction = self.ic.spatial_core_fraction
        rim_fraction = 1.0 - core_fraction
        rim_multiplier = self.ic.spatial_rim_infiltration

        # Core has baseline infiltration, rim has enhanced
        core_kill = base_kill / rim_multiplier  # reduced
        rim_kill = base_kill * rim_multiplier / (core_fraction + rim_fraction * rim_multiplier)

        return {
            "core_kill_rate": core_kill,
            "rim_kill_rate": rim_kill,
        }

    # ── simulation ───────────────────────────────────────────────────────

    def simulate(
        self,
        tumour_trajectory: np.ndarray,
        dt: float = 0.5,
    ) -> pd.DataFrame:
        """Simulate immune dynamics coupled to a tumour burden curve.

        Parameters
        ----------
        tumour_trajectory : np.ndarray
            Daily total tumour burden (one value per day).
        dt : float
            Time step (days).

        Returns
        -------
        pd.DataFrame
            Columns: ``day``, ``t_eff``, ``treg``, ``nk``, ``mdsc``,
            ``exhaustion``, ``total_kill_rate``, ``spatial_core_kill``,
            ``spatial_rim_kill``.
        """
        t_eff, treg, nk, mdsc = self._init_state()
        n_days = len(tumour_trajectory)

        for day in range(n_days):
            tumour = float(tumour_trajectory[day])

            # Update exhaustion
            self._update_exhaustion(tumour, dt)

            # Compute kill rates
            kill = self._effective_kill_rate(t_eff, nk)
            spatial = self.spatial_kill_rates(t_eff, nk)

            # Record state
            self._trajectory.append(
                ImmuneState(
                    day=day,
                    t_eff=t_eff,
                    treg=treg,
                    nk=nk,
                    mdsc=mdsc,
                    exhaustion=self._exhaustion,
                    total_kill_rate=kill,
                )
            )

            # Step all populations
            t_eff = self._step_t_eff(t_eff, treg, mdsc, tumour, dt)
            treg = self._step_treg(treg, tumour, dt)
            nk = self._step_nk(nk, tumour, dt)
            mdsc = self._step_mdsc(mdsc, tumour, dt)

        df = pd.DataFrame(
            [
                {
                    "day": s.day,
                    "t_eff": s.t_eff,
                    "treg": s.treg,
                    "nk": s.nk,
                    "mdsc": s.mdsc,
                    "exhaustion": round(s.exhaustion, 4),
                    "total_kill_rate": s.total_kill_rate,
                }
                for s in self._trajectory
            ]
        )
        logger.info(
            "Immune simulation: %d days, final T_eff=%.0f, exhaustion=%.2f",
            n_days,
            t_eff,
            self._exhaustion,
        )
        return df

    # ── checkpoint therapy comparison ────────────────────────────────────

    def compare_immunotherapy(
        self,
        tumour_trajectory: np.ndarray,
    ) -> pd.DataFrame:
        """Compare immune dynamics with and without checkpoint therapy.

        Runs three simulations:
        1. No immunotherapy (baseline)
        2. Anti-PD-1 monotherapy
        3. Anti-PD-1 + anti-CTLA-4 combination

        Parameters
        ----------
        tumour_trajectory : np.ndarray
            Daily tumour burden array.

        Returns
        -------
        pd.DataFrame
            Summary comparison with final T_eff, exhaustion, and
            kill rate for each regimen.
        """
        results = []
        scenarios = [
            ("No immunotherapy", 0.0, 0.0),
            ("Anti-PD-1", 0.6, 0.0),
            ("Anti-PD-1 + Anti-CTLA-4", 0.6, 0.02),
        ]

        for name, pd1, ctla4 in scenarios:
            self.ic.checkpoint_anti_pd1 = pd1
            self.ic.checkpoint_anti_ctla4 = ctla4
            df = self.simulate(tumour_trajectory)
            last = df.iloc[-1]
            results.append(
                {
                    "regimen": name,
                    "final_t_eff": last["t_eff"],
                    "final_treg": last["treg"],
                    "final_exhaustion": last["exhaustion"],
                    "final_kill_rate": last["total_kill_rate"],
                    "peak_t_eff": df["t_eff"].max(),
                }
            )

        # Reset to no therapy
        self.ic.checkpoint_anti_pd1 = 0.0
        self.ic.checkpoint_anti_ctla4 = 0.0

        return pd.DataFrame(results)

    def summary(self) -> dict[str, Any]:
        """Return a summary of the last immune simulation.

        Returns
        -------
        dict[str, Any]
            Final state and key metrics.
        """
        if not self._trajectory:
            return {"status": "not_simulated"}

        last = self._trajectory[-1]
        return {
            "simulation_days": last.day + 1,
            "final_t_eff": last.t_eff,
            "final_treg": last.treg,
            "final_nk": last.nk,
            "final_mdsc": last.mdsc,
            "exhaustion": last.exhaustion,
            "kill_rate": last.total_kill_rate,
            "checkpoint_anti_pd1": self.ic.checkpoint_anti_pd1,
            "checkpoint_anti_ctla4": self.ic.checkpoint_anti_ctla4,
        }
