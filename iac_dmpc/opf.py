"""Distribution voltage-support OPF approximations and comparisons."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import cvxpy as cp

from .capability import InverterCapability
from .feeder import FeederSolution, VoltageSensitivity


@dataclass(frozen=True)
class VoltageSupportOPFResult:
    p_kw: float
    q_kvar: float
    predicted_voltage_pu: float
    status: str
    objective_value: float


@dataclass(frozen=True)
class SensitivityOPFComparison:
    sensitivity: VoltageSensitivity
    opf: VoltageSupportOPFResult
    opendss_solution: FeederSolution
    voltage_prediction_error_pu: float


def solve_voltage_support_opf(
    sensitivity: VoltageSensitivity,
    capability: InverterCapability,
    p_baseline_kw: float,
    q_baseline_kvar: float,
    fleet_size: int,
    voltage_reference_pu: float = 0.95,
    voltage_min_pu: float = 0.88,
    voltage_max_pu: float = 1.08,
    p_tracking_weight: float = 0.02,
    q_effort_weight: float = 0.01,
    voltage_weight: float = 500.0,
) -> VoltageSupportOPFResult:
    """Solve a convex aggregate OPF using OpenDSS-derived voltage sensitivity.

    This is a distribution-OPF comparison model rather than the closed-loop
    controller itself. It asks what aggregate P/Q setpoint a sensitivity-based
    OPF would choose around the current OpenDSS operating point.
    """
    p = cp.Variable()
    q = cp.Variable()
    voltage = (
        sensitivity.base_voltage_pu
        + sensitivity.dv_dp_pu_per_kw * fleet_size * (p - p_baseline_kw)
        + sensitivity.dv_dq_pu_per_kvar * fleet_size * (q - q_baseline_kvar)
    )
    constraints = [
        p >= capability.p_min_kw,
        p <= capability.p_max_kw,
        q >= -capability.q_max_kvar,
        q <= capability.q_max_kvar,
        cp.norm(cp.hstack([p, q]), 2) <= capability.s_rated_kva,
        voltage >= voltage_min_pu,
        voltage <= voltage_max_pu,
    ]
    objective = (
        voltage_weight * cp.square(cp.pos(voltage_reference_pu - voltage))
        + p_tracking_weight * cp.square(p - p_baseline_kw)
        + q_effort_weight * cp.square(q)
    )
    problem = cp.Problem(cp.Minimize(objective), constraints)
    problem.solve(solver=cp.CLARABEL, warm_start=True, verbose=False)

    if problem.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}:
        q_limit = capability.max_reactive_for_p(p_baseline_kw)
        fallback_q = max(-q_limit, min(q_limit, q_baseline_kvar))
        return VoltageSupportOPFResult(
            p_kw=p_baseline_kw,
            q_kvar=fallback_q,
            predicted_voltage_pu=sensitivity.base_voltage_pu,
            status=problem.status,
            objective_value=float("nan"),
        )

    return VoltageSupportOPFResult(
        p_kw=float(p.value),
        q_kvar=float(q.value),
        predicted_voltage_pu=float(voltage.value),
        status=str(problem.status),
        objective_value=float(problem.value),
    )


def compare_sensitivity_opf_to_power_flow(
    feeder: Any,
    source_voltage_pu: float,
    p_baseline_kw: float,
    q_baseline_kvar: float,
    fleet_size: int,
    capability: InverterCapability,
    perturb_kw: float = 25.0,
    perturb_kvar: float = 25.0,
    extra_load_kw: float = 0.0,
    extra_load_kvar: float = 0.0,
    metric: str = "min",
) -> SensitivityOPFComparison:
    """Compare sensitivity OPF prediction against the nonlinear power flow."""
    sensitivity = feeder.estimate_voltage_sensitivity(
        source_voltage_pu,
        p_baseline_kw * fleet_size,
        q_baseline_kvar * fleet_size,
        perturb_kw=perturb_kw,
        perturb_kvar=perturb_kvar,
        metric=metric,
        extra_load_kw=extra_load_kw,
        extra_load_kvar=extra_load_kvar,
    )
    opf = solve_voltage_support_opf(
        sensitivity,
        capability,
        p_baseline_kw,
        q_baseline_kvar,
        fleet_size,
    )
    actual = feeder.solve_pcc_voltage(
        source_voltage_pu,
        opf.p_kw * fleet_size,
        opf.q_kvar * fleet_size,
        extra_load_kw=extra_load_kw,
        extra_load_kvar=extra_load_kvar,
    )
    actual_voltage = actual.min_voltage_pu if metric == "min" else actual.pcc_voltage_pu
    return SensitivityOPFComparison(
        sensitivity=sensitivity,
        opf=opf,
        opendss_solution=actual,
        voltage_prediction_error_pu=float(opf.predicted_voltage_pu - actual_voltage),
    )


def apparent_headroom_kw(s_rated_kva: float, p_kw: float) -> float:
    """Return available reactive magnitude at a given active power."""
    return math.sqrt(max(s_rated_kva**2 - p_kw**2, 0.0))
