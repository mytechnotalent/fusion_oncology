"""
Compartmental pharmacokinetics / pharmacodynamics (PK/PD) model.

Upgrades the digital twin from a flat ``efficacy`` constant to a
physiologically motivated two-compartment PK model with an
Emax pharmacodynamic layer.  This captures:

* **Absorption** — first-order oral absorption from gut to plasma.
* **Distribution** — two-compartment (central + peripheral) with
  inter-compartmental clearance.
* **Elimination** — first-order hepatic / renal clearance.
* **Drug effect** — sigmoidal Emax relating plasma concentration
  to tumour cell kill rate.

The ODE system is:

.. math::

    \\frac{dA_{gut}}{dt} = -k_a \\cdot A_{gut}

    \\frac{dC_1}{dt} = \\frac{k_a \\cdot A_{gut}}{V_1}
                      - \\frac{CL}{V_1} \\cdot C_1
                      - \\frac{Q}{V_1} \\cdot C_1
                      + \\frac{Q}{V_2} \\cdot C_2

    \\frac{dC_2}{dt} = \\frac{Q}{V_1} \\cdot C_1
                      - \\frac{Q}{V_2} \\cdot C_2

The pharmacodynamic effect is:

.. math::

    E(C) = E_{max} \\cdot \\frac{C^\\gamma}{EC_{50}^\\gamma + C^\\gamma}

References
----------
Rowland & Tozer, "Clinical Pharmacokinetics", 4th ed.
Mould & Upton, CPT:PSP 2013 — population PK modelling tutorial.
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
class DrugPKParams:
    """Pharmacokinetic parameters for a single drug.

    Parameters
    ----------
    name : str
        Drug name (e.g. ``"Osimertinib"``).
    dose_mg : float
        Oral dose per administration (mg).
    dosing_interval_h : float
        Time between doses (hours).  ``24`` for once-daily.
    ka : float
        First-order absorption rate constant (h⁻¹).
    volume_central : float
        Central compartment volume V₁ (L).
    volume_peripheral : float
        Peripheral compartment volume V₂ (L).
    clearance : float
        Systemic clearance CL (L/h).
    intercompartmental_cl : float
        Inter-compartmental clearance Q (L/h).
    bioavailability : float
        Oral bioavailability F (fraction, 0–1).
    emax : float
        Maximum drug effect (kill rate, day⁻¹).
    ec50 : float
        Concentration at half-maximal effect (ng/mL).
    hill : float
        Hill coefficient γ for the Emax equation.
    molecular_weight : float
        Molecular weight (g/mol) for unit conversion.
    """

    name: str = "Osimertinib"
    dose_mg: float = 80.0
    dosing_interval_h: float = 24.0
    ka: float = 0.5
    volume_central: float = 986.0
    volume_peripheral: float = 3240.0
    clearance: float = 14.3
    intercompartmental_cl: float = 5.3
    bioavailability: float = 0.70
    emax: float = 0.25
    ec50: float = 50.0
    hill: float = 1.5
    molecular_weight: float = 499.6


# ── Pre-configured drug PK profiles ─────────────────────────────────────

DRUG_PK_LIBRARY: dict[str, DrugPKParams] = {
    "Osimertinib": DrugPKParams(
        name="Osimertinib",
        dose_mg=80.0,
        ka=0.5,
        volume_central=986.0,
        volume_peripheral=3240.0,
        clearance=14.3,
        intercompartmental_cl=5.3,
        bioavailability=0.70,
        emax=0.25,
        ec50=50.0,
        hill=1.5,
        molecular_weight=499.6,
    ),
    "Erlotinib": DrugPKParams(
        name="Erlotinib",
        dose_mg=150.0,
        ka=0.8,
        volume_central=232.0,
        volume_peripheral=362.0,
        clearance=4.6,
        intercompartmental_cl=2.1,
        bioavailability=0.59,
        emax=0.20,
        ec50=500.0,
        hill=1.2,
        molecular_weight=393.4,
    ),
    "Sotorasib": DrugPKParams(
        name="Sotorasib",
        dose_mg=960.0,
        ka=0.6,
        volume_central=271.0,
        volume_peripheral=430.0,
        clearance=26.2,
        intercompartmental_cl=8.5,
        bioavailability=0.45,
        emax=0.22,
        ec50=200.0,
        hill=1.3,
        molecular_weight=560.6,
    ),
    "Olaparib": DrugPKParams(
        name="Olaparib",
        dose_mg=300.0,
        dosing_interval_h=12.0,
        ka=1.2,
        volume_central=158.0,
        volume_peripheral=196.0,
        clearance=8.6,
        intercompartmental_cl=3.4,
        bioavailability=0.45,
        emax=0.18,
        ec50=350.0,
        hill=1.4,
        molecular_weight=434.5,
    ),
}


# ── PK/PD Compartmental Model ───────────────────────────────────────────


class PKPDModel:
    """Two-compartment PK model with sigmoidal Emax PD layer.

    Simulates plasma drug concentration over time and converts to
    an instantaneous tumour-cell kill rate that can be fed into the
    ``DigitalTwin`` ODE system.

    Parameters
    ----------
    drug : DrugPKParams
        Pharmacokinetic parameters for the drug.
    config : ProjectConfig, optional
        Runtime configuration.
    """

    def __init__(
        self,
        drug: DrugPKParams | None = None,
        config: ProjectConfig | None = None,
    ) -> None:
        """Initialise the PK/PD model.

        Parameters
        ----------
        drug : DrugPKParams, optional
            Drug PK parameters.  Defaults to Osimertinib.
        config : ProjectConfig, optional
            Runtime configuration.
        """
        self.drug = drug or DrugPKParams()
        self.cfg = config or ProjectConfig()
        self._trajectory: list[dict[str, float]] = []
        logger.info("PKPDModel initialised for %s", self.drug.name)

    # ── dosing schedule ──────────────────────────────────────────────────

    def _dose_times_hours(self, duration_days: int) -> np.ndarray:
        """Generate dosing time points in hours.

        Parameters
        ----------
        duration_days : int
            Total treatment duration in days.

        Returns
        -------
        np.ndarray
            Array of dosing times (hours).
        """
        total_h = duration_days * 24.0
        return np.arange(0, total_h, self.drug.dosing_interval_h)

    def _bolus_amount(self) -> float:
        """Compute the absorbed dose entering the gut compartment.

        Returns
        -------
        float
            Dose x bioavailability (mg).
        """
        return self.drug.dose_mg * self.drug.bioavailability

    # ── ODE step functions ───────────────────────────────────────────────

    def _step_gut(self, a_gut: float, dt_h: float) -> float:
        """Advance gut compartment by one Euler step.

        Parameters
        ----------
        a_gut : float
            Current drug amount in gut (mg).
        dt_h : float
            Time step (hours).

        Returns
        -------
        float
            Updated gut amount.
        """
        da = -self.drug.ka * a_gut
        return max(0.0, a_gut + da * dt_h)

    def _step_central(
        self,
        c1: float,
        c2: float,
        a_gut: float,
        dt_h: float,
    ) -> float:
        """Advance central compartment concentration by one step.

        Parameters
        ----------
        c1 : float
            Current central concentration (ng/mL ≈ µg/L).
        c2 : float
            Current peripheral concentration.
        a_gut : float
            Gut amount (mg) — for absorption influx.
        dt_h : float
            Time step (hours).

        Returns
        -------
        float
            Updated central concentration.
        """
        d = self.drug
        absorption = d.ka * a_gut * 1000.0 / d.volume_central  # mg→µg
        elimination = (d.clearance / d.volume_central) * c1
        to_periph = (d.intercompartmental_cl / d.volume_central) * c1
        from_periph = (d.intercompartmental_cl / d.volume_peripheral) * c2
        dc1 = absorption - elimination - to_periph + from_periph
        return max(0.0, c1 + dc1 * dt_h)

    def _step_peripheral(self, c1: float, c2: float, dt_h: float) -> float:
        """Advance peripheral compartment concentration.

        Parameters
        ----------
        c1 : float
            Central concentration.
        c2 : float
            Peripheral concentration.
        dt_h : float
            Time step (hours).

        Returns
        -------
        float
            Updated peripheral concentration.
        """
        d = self.drug
        to_periph = (d.intercompartmental_cl / d.volume_central) * c1
        from_periph = (d.intercompartmental_cl / d.volume_peripheral) * c2
        dc2 = to_periph - from_periph
        return max(0.0, c2 + dc2 * dt_h)

    # ── pharmacodynamic effect ───────────────────────────────────────────

    def emax_effect(self, concentration: float) -> float:
        """Compute the Emax drug effect at a given concentration.

        Implements the sigmoidal Emax equation:

        .. math::
            E(C) = E_{max} \\cdot \\frac{C^\\gamma}{EC_{50}^\\gamma + C^\\gamma}

        Parameters
        ----------
        concentration : float
            Plasma drug concentration (ng/mL).

        Returns
        -------
        float
            Instantaneous kill rate (day⁻¹).
        """
        if concentration <= 0:
            return 0.0
        d = self.drug
        c_gamma = concentration**d.hill
        ec_gamma = d.ec50**d.hill
        return d.emax * c_gamma / (ec_gamma + c_gamma)

    # ── simulation ───────────────────────────────────────────────────────

    def simulate(
        self,
        duration_days: int = 30,
        dt_hours: float = 0.5,
    ) -> pd.DataFrame:
        """Simulate PK/PD over the treatment duration.

        Parameters
        ----------
        duration_days : int
            Number of days to simulate.
        dt_hours : float
            Euler integration time step (hours).

        Returns
        -------
        pd.DataFrame
            Columns: ``time_h``, ``time_day``, ``gut_mg``,
            ``plasma_conc``, ``peripheral_conc``, ``kill_rate``.
        """
        dose_times = set(self._dose_times_hours(duration_days))
        bolus = self._bolus_amount()
        total_h = duration_days * 24.0
        n_steps = int(total_h / dt_hours)

        a_gut = 0.0
        c1 = 0.0
        c2 = 0.0

        self._trajectory.clear()

        for step in range(n_steps + 1):
            t = step * dt_hours

            # Check for dosing event
            if any(abs(t - dt) < dt_hours / 2 for dt in dose_times):
                a_gut += bolus

            kill = self.emax_effect(c1)

            # Record at ~hourly intervals
            if step % max(1, int(1.0 / dt_hours)) == 0:
                self._trajectory.append(
                    {
                        "time_h": round(t, 2),
                        "time_day": round(t / 24.0, 3),
                        "gut_mg": round(a_gut, 4),
                        "plasma_conc": round(c1, 4),
                        "peripheral_conc": round(c2, 4),
                        "kill_rate": round(kill, 6),
                    }
                )

            # Euler step
            a_gut = self._step_gut(a_gut, dt_hours)
            c1 = self._step_central(c1, c2, a_gut, dt_hours)
            c2 = self._step_peripheral(c1, c2, dt_hours)

        df = pd.DataFrame(self._trajectory)
        logger.info(
            "%s PK simulation: %d days, Cmax=%.1f ng/mL, Emax_kill=%.4f day⁻¹",
            self.drug.name,
            duration_days,
            df["plasma_conc"].max(),
            df["kill_rate"].max(),
        )
        return df

    # ── daily kill-rate profile ──────────────────────────────────────────

    def daily_kill_rate(self, duration_days: int = 30) -> np.ndarray:
        """Compute the average daily kill rate per day.

        Useful for coupling to the ``DigitalTwin`` which operates on
        a daily time step.

        Parameters
        ----------
        duration_days : int
            Number of days.

        Returns
        -------
        np.ndarray
            Shape ``(duration_days,)`` — average kill rate per day.
        """
        df = self.simulate(duration_days=duration_days)
        df["day_int"] = df["time_day"].astype(int)
        daily = df.groupby("day_int")["kill_rate"].mean().values
        return daily[:duration_days]

    # ── steady-state metrics ─────────────────────────────────────────────

    def steady_state_metrics(self, duration_days: int = 14) -> dict[str, float]:
        """Compute steady-state PK/PD metrics.

        Parameters
        ----------
        duration_days : int
            Simulation duration (should be long enough to reach
            steady state, typically 5–7 half-lives).

        Returns
        -------
        dict[str, float]
            ``Cmax``, ``Cmin``, ``Cavg``, ``AUC_24h``,
            ``trough_kill_rate``, ``peak_kill_rate``.
        """
        df = self.simulate(duration_days=duration_days)

        # Use last 24 hours as steady-state window
        last_day = df[df["time_day"] >= duration_days - 1]
        if last_day.empty:
            last_day = df.tail(48)

        cmax = float(last_day["plasma_conc"].max())
        cmin = float(last_day["plasma_conc"].min())
        cavg = float(last_day["plasma_conc"].mean())
        auc_24 = cavg * 24.0  # trapezoidal approximation

        return {
            "Cmax_ng_mL": round(cmax, 2),
            "Cmin_ng_mL": round(cmin, 2),
            "Cavg_ng_mL": round(cavg, 2),
            "AUC_24h": round(auc_24, 2),
            "trough_kill_rate": round(self.emax_effect(cmin), 6),
            "peak_kill_rate": round(self.emax_effect(cmax), 6),
        }

    def summary(self) -> dict[str, Any]:
        """Return a summary of the PK/PD parameterisation.

        Returns
        -------
        dict[str, Any]
            Drug name, dose, PK parameters, and PD parameters.
        """
        d = self.drug
        return {
            "drug": d.name,
            "dose_mg": d.dose_mg,
            "dosing_interval_h": d.dosing_interval_h,
            "bioavailability": d.bioavailability,
            "V1_L": d.volume_central,
            "V2_L": d.volume_peripheral,
            "CL_L_h": d.clearance,
            "Q_L_h": d.intercompartmental_cl,
            "ka_h": d.ka,
            "Emax_day": d.emax,
            "EC50_ng_mL": d.ec50,
            "Hill": d.hill,
        }
