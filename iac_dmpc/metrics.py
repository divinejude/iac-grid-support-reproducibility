"""Experiment metrics for reproducible IAC grid-support studies."""

from __future__ import annotations

import numpy as np


def summarize_log(
    log: dict[str, np.ndarray],
    nominal_frequency_hz: float = 60.0,
    temperature_setpoint_c: float = 24.0,
    comfort_band_c: float = 1.0,
) -> dict[str, float]:
    time_s = log["time_s"]
    dt_h = _mean_step_hours(time_s)
    temperature = log["temperature_c"]
    lower_comfort = temperature_setpoint_c - comfort_band_c
    upper_comfort = temperature_setpoint_c + comfort_band_c
    comfort_violation_c = np.maximum(lower_comfort - temperature, 0.0) + np.maximum(temperature - upper_comfort, 0.0)

    return {
        "frequency_nadir_hz": float(np.min(log["frequency_hz"])),
        "frequency_overshoot_hz": float(np.max(log["frequency_hz"]) - nominal_frequency_hz),
        "max_abs_frequency_deviation_hz": float(np.max(np.abs(log["frequency_hz"] - nominal_frequency_hz))),
        "min_pcc_voltage_pu": float(np.min(log["pcc_voltage_pu"])),
        "max_pcc_voltage_pu": float(np.max(log["pcc_voltage_pu"])),
        "mean_pcc_voltage_pu": float(np.mean(log["pcc_voltage_pu"])),
        "min_feeder_voltage_pu": float(np.min(log.get("min_feeder_voltage_pu", log["pcc_voltage_pu"]))),
        "max_feeder_voltage_pu": float(np.max(log.get("max_feeder_voltage_pu", log["pcc_voltage_pu"]))),
        "temperature_min_c": float(np.min(temperature)),
        "temperature_max_c": float(np.max(temperature)),
        "comfort_violation_count": float(np.count_nonzero(comfort_violation_c > 1e-9)),
        "comfort_violation_degree_minutes": float(np.sum(comfort_violation_c) * dt_h * 60.0),
        "aggregate_energy_kwh": float(np.sum(log["aggregate_p_kw"]) * dt_h),
        "reactive_support_kvarh": float(np.sum(np.maximum(log["aggregate_q_kvar"], 0.0)) * dt_h),
        "max_unit_apparent_kva": float(np.max(log["apparent_kva"])),
        "max_aggregate_p_kw": float(np.max(log["aggregate_p_kw"])),
        "max_aggregate_q_kvar": float(np.max(log["aggregate_q_kvar"])),
        "mpc_feasibility_rate": float(np.mean(log["dmpc_status_ok"])),
        "mean_compute_time_ms": float(np.mean(log["controller_compute_time_s"]) * 1000.0),
        "max_compute_time_ms": float(np.max(log["controller_compute_time_s"]) * 1000.0),
        "current_saturation_count": float(np.sum(log["current_saturated"])),
        "ramp_limited_count": float(np.sum(log["ramp_limited"])),
    }


def _mean_step_hours(time_s: np.ndarray) -> float:
    if len(time_s) < 2:
        return 0.0
    return float(np.mean(np.diff(time_s)) / 3600.0)
