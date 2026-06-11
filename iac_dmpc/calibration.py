"""Calibration helpers for thermal and compressor model parameters.

The functions in this module intentionally separate three data sources:

* field or EnergyPlus time-series data, used to fit RC thermal dynamics;
* manufacturer-like performance maps, used to identify COP variation;
* compressor step responses, used to identify electrical time constants.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import csv
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

from .parameters import CompressorParameters, ThermalParameters


@dataclass(frozen=True)
class ThermalCalibrationData:
    time_s: np.ndarray
    indoor_c: np.ndarray
    ambient_c: np.ndarray
    power_kw: np.ndarray
    cooling_capacity_kw: np.ndarray | None = None


@dataclass(frozen=True)
class ThermalCalibrationResult:
    thermal: ThermalParameters
    residual_rmse_c: float
    sample_count: int
    source: str


def calibrate_thermal_from_steady_state(
    ambient_c: float,
    indoor_c: float,
    electrical_power_kw: float,
    cop: float,
    thermal_time_constant_s: float,
) -> ThermalParameters:
    """Infer RC parameters from a steady operating point and time constant.

    At steady state, (Tamb - Tin) / R = COP * P. The first-order envelope time
    constant gives C = tau / R after unit conversion.
    """
    resistance = (ambient_c - indoor_c) / max(cop * electrical_power_kw, 1e-9)
    capacitance = thermal_time_constant_s / (resistance * 3600.0)
    return ThermalParameters(
        ambient_temperature_c=ambient_c,
        setpoint_c=indoor_c,
        resistance_c_per_kw=resistance,
        capacitance_kwh_per_c=capacitance,
        cop=cop,
    )


def load_thermal_calibration_csv(path: str | Path) -> ThermalCalibrationData:
    """Load field or EnergyPlus calibration data from CSV.

    Required columns are ``time_s``, ``indoor_c``, ``ambient_c``, and
    ``power_kw``. An optional ``cooling_capacity_kw`` column can be provided
    when EnergyPlus or manufacturer map post-processing directly reports
    cooling capacity.
    """
    rows: list[dict[str, str]] = []
    with Path(path).open(newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    if not rows:
        raise ValueError(f"Calibration file is empty: {path}")

    required = {"time_s", "indoor_c", "ambient_c", "power_kw"}
    missing = required - set(rows[0])
    if missing:
        raise ValueError(f"Missing calibration columns {sorted(missing)} in {path}")

    def col(name: str) -> np.ndarray:
        return np.array([float(row[name]) for row in rows], dtype=float)

    cooling = col("cooling_capacity_kw") if "cooling_capacity_kw" in rows[0] and rows[0]["cooling_capacity_kw"] != "" else None
    return ThermalCalibrationData(
        time_s=col("time_s"),
        indoor_c=col("indoor_c"),
        ambient_c=col("ambient_c"),
        power_kw=col("power_kw"),
        cooling_capacity_kw=cooling,
    )


def fit_thermal_rc_from_timeseries(
    data: ThermalCalibrationData,
    initial: ThermalParameters | None = None,
    fit_cop: bool = True,
    fit_internal_gain: bool = False,
    source: str = "timeseries",
) -> ThermalCalibrationResult:
    """Fit thermal RC parameters to field or EnergyPlus time-series data.

    The optimizer simulates the first-order room model over the supplied time
    stamps and minimizes indoor-temperature residuals. If cooling capacity is
    supplied, COP is not identifiable from the time series and is kept at the
    initial value.
    """
    if len(data.time_s) < 3:
        raise ValueError("At least three calibration samples are required")
    if not np.all(np.diff(data.time_s) > 0.0):
        raise ValueError("Calibration time_s must be strictly increasing")

    base = ThermalParameters() if initial is None else initial
    fit_cop = fit_cop and data.cooling_capacity_kw is None

    names = ["resistance", "capacitance"]
    x0 = [base.resistance_c_per_kw, base.capacitance_kwh_per_c]
    lower = [0.05, 0.05]
    upper = [20.0, 40.0]
    if fit_cop:
        names.append("cop")
        x0.append(base.cop)
        lower.append(0.5)
        upper.append(8.0)
    if fit_internal_gain:
        names.append("internal_gain")
        x0.append(base.internal_gain_kw)
        lower.append(0.0)
        upper.append(5.0)

    def unpack(theta: np.ndarray) -> ThermalParameters:
        values = dict(zip(names, theta))
        return replace(
            base,
            ambient_temperature_c=float(np.mean(data.ambient_c)),
            setpoint_c=float(data.indoor_c[0]),
            resistance_c_per_kw=float(values["resistance"]),
            capacitance_kwh_per_c=float(values["capacitance"]),
            cop=float(values.get("cop", base.cop)),
            internal_gain_kw=float(values.get("internal_gain", base.internal_gain_kw)),
        )

    def simulate(params: ThermalParameters) -> np.ndarray:
        indoor = np.zeros_like(data.indoor_c)
        indoor[0] = data.indoor_c[0]
        for k in range(len(indoor) - 1):
            dt = data.time_s[k + 1] - data.time_s[k]
            cooling_kw = (
                data.cooling_capacity_kw[k]
                if data.cooling_capacity_kw is not None
                else params.cop * data.power_kw[k]
            )
            envelope_kw = (data.ambient_c[k] - indoor[k]) / params.resistance_c_per_kw
            derivative = (envelope_kw + params.internal_gain_kw - cooling_kw) / (
                params.capacitance_kwh_per_c * 3600.0
            )
            indoor[k + 1] = indoor[k] + dt * derivative
        return indoor

    def residual(theta: np.ndarray) -> np.ndarray:
        return simulate(unpack(theta)) - data.indoor_c

    result = least_squares(residual, np.array(x0), bounds=(np.array(lower), np.array(upper)))
    calibrated = unpack(result.x)
    rmse = float(np.sqrt(np.mean(residual(result.x) ** 2)))
    return ThermalCalibrationResult(calibrated, rmse, len(data.time_s), source)


def calibrate_cop_from_performance_map(
    ambient_c: np.ndarray,
    indoor_c: np.ndarray,
    power_kw: np.ndarray,
    capacity_kw: np.ndarray,
    base: ThermalParameters | None = None,
) -> ThermalParameters:
    """Fit nominal COP and temperature-lift slope from a performance map.

    The map should contain manufacturer-like operating points. The fitted model
    is ``COP = cop * (1 + slope * (lift - nominal_lift))`` and is converted into
    the existing ``ThermalParameters`` representation.
    """
    params = ThermalParameters() if base is None else base
    ambient = np.asarray(ambient_c, dtype=float)
    indoor = np.asarray(indoor_c, dtype=float)
    power = np.asarray(power_kw, dtype=float)
    capacity = np.asarray(capacity_kw, dtype=float)
    if not (ambient.size == indoor.size == power.size == capacity.size):
        raise ValueError("Performance-map arrays must have the same length")
    if ambient.size < 2:
        raise ValueError("At least two performance-map points are required")

    measured_cop = capacity / np.maximum(power, 1e-9)
    lift = ambient - indoor
    nominal_lift = params.ambient_temperature_c - params.setpoint_c
    design = np.column_stack([np.ones_like(lift), lift - nominal_lift])
    coeff, *_ = np.linalg.lstsq(design, measured_cop, rcond=None)
    cop = max(float(coeff[0]), 0.1)
    slope = float(coeff[1] / cop)
    return replace(params, cop=cop, cop_temperature_slope_per_c=slope)


def fit_first_order_compressor_time_constant(time_s: np.ndarray, power_kw: np.ndarray, final_kw: float) -> CompressorParameters:
    """Estimate compressor time constant from a step response."""
    time = np.asarray(time_s, dtype=float)
    power = np.asarray(power_kw, dtype=float)
    initial = power[0]
    target_632 = initial + 0.632 * (final_kw - initial)
    index = int(np.argmin(np.abs(power - target_632)))
    tau = max(float(time[index] - time[0]), 1.0)
    return replace(CompressorParameters(), time_constant_s=tau)
