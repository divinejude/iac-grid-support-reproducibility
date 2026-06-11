"""Decentralized MPC controller for active IAC power dispatch."""

from __future__ import annotations

from dataclasses import dataclass

import cvxpy as cp
import numpy as np

from .capability import InverterCapability
from .parameters import DMPCParameters, InverterParameters, ThermalParameters
from .plant import IACPlant, PlantState


@dataclass
class DMPCResult:
    p_command_kw: float
    predicted_states: np.ndarray
    predicted_controls_kw: np.ndarray
    status: str
    objective_value: float
    q_command_kvar: float = 0.0
    predicted_reactive_controls_kvar: np.ndarray | None = None


class DMPCController:
    """Quadratic DMPC for frequency support under strict thermal constraints."""

    def __init__(
        self,
        plant: IACPlant,
        inverter_params: InverterParameters,
        thermal_params: ThermalParameters,
        mpc_params: DMPCParameters,
    ):
        self.plant = plant
        self.inverter_params = inverter_params
        self.thermal_params = thermal_params
        self.params = mpc_params
        self.capability = InverterCapability(
            inverter_params.s_rated_kva,
            inverter_params.p_min_kw,
            inverter_params.p_max_kw,
            inverter_params.q_max_kvar,
        )

    def solve(
        self,
        state: PlantState,
        previous_control_kw: float,
        q_forecast_kvar: np.ndarray,
        disturbance_forecast_hz_per_s: np.ndarray,
        frequency_reference_hz: float = 0.0,
        temperature_reference_c: float | None = None,
    ) -> DMPCResult:
        p = self.params
        n = 3
        horizon = p.horizon_steps
        tref = self.thermal_params.setpoint_c if temperature_reference_c is None else temperature_reference_c
        q_forecast = self._as_horizon(q_forecast_kvar, horizon)
        d_forecast = self._as_horizon(disturbance_forecast_hz_per_s, horizon)

        ad, bd, gd, ed = self.plant.discrete_affine_matrices(p.sample_time_s)
        x = cp.Variable((n, horizon + 1))
        u = cp.Variable(horizon)

        constraints = [x[:, 0] == state.as_vector()]
        objective = 0.0

        for j in range(horizon):
            constraints.append(x[:, j + 1] == ad @ x[:, j] + bd[:, 0] * u[j] + gd + ed[:, 0] * d_forecast[j])
            p_low, p_high = self.capability.active_bounds_for_q(float(q_forecast[j]))
            constraints += [
                u[j] >= p_low,
                u[j] <= p_high,
                x[1, j + 1] >= self.thermal_params.min_temperature_c,
                x[1, j + 1] <= self.thermal_params.max_temperature_c,
            ]

            du = u[j] - (previous_control_kw if j == 0 else u[j - 1])
            objective += p.weight_frequency * cp.square(x[0, j + 1] - frequency_reference_hz)
            objective += p.weight_temperature * cp.square(x[1, j + 1] - tref)
            objective += p.weight_control_move * cp.square(du)
            objective += p.weight_power_tracking * cp.square(u[j] - self.inverter_params.p_set_kw)

        problem = cp.Problem(cp.Minimize(objective), constraints)
        problem.solve(solver=cp.OSQP, warm_start=True, verbose=False)

        if problem.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}:
            fallback = self._fallback_control(state, previous_control_kw, float(q_forecast[0]))
            return DMPCResult(
                p_command_kw=fallback,
                predicted_states=np.full((n, horizon + 1), np.nan),
                predicted_controls_kw=np.full(horizon, np.nan),
                status=problem.status,
                objective_value=float("nan"),
            )

        return DMPCResult(
            p_command_kw=float(u.value[0]),
            predicted_states=np.array(x.value, dtype=float),
            predicted_controls_kw=np.array(u.value, dtype=float),
            status=problem.status,
            objective_value=float(problem.value),
        )

    @staticmethod
    def _as_horizon(values: np.ndarray | float, horizon: int) -> np.ndarray:
        arr = np.asarray(values, dtype=float).reshape(-1)
        if arr.size == 1:
            return np.full(horizon, float(arr[0]))
        if arr.size < horizon:
            return np.pad(arr, (0, horizon - arr.size), mode="edge")
        return arr[:horizon]

    def _fallback_control(self, state: PlantState, previous_control_kw: float, q_kvar: float) -> float:
        p_low, p_high = self.capability.active_bounds_for_q(q_kvar)
        if state.indoor_temperature_c > self.thermal_params.setpoint_c:
            return min(p_high, max(p_low, previous_control_kw + 0.1))
        return max(p_low, min(p_high, previous_control_kw - 0.1))


class PQDMPCController(DMPCController):
    """Convex P/Q DMPC with OpenDSS-derived voltage sensitivity."""

    def solve_pq(
        self,
        state: PlantState,
        previous_p_kw: float,
        previous_q_kvar: float,
        disturbance_forecast_hz_per_s: np.ndarray,
        voltage_base_pu: float,
        dv_dp_pu_per_kw: float,
        dv_dq_pu_per_kvar: float,
        p_baseline_kw: float,
        q_baseline_kvar: float,
        fleet_size: int,
        q_floor_kvar: float = 0.0,
        frequency_reference_hz: float = 0.0,
        temperature_reference_c: float | None = None,
    ) -> DMPCResult:
        p = self.params
        n = 3
        horizon = p.horizon_steps
        tref = self.thermal_params.setpoint_c if temperature_reference_c is None else temperature_reference_c
        d_forecast = self._as_horizon(disturbance_forecast_hz_per_s, horizon)

        ad, bd, gd, ed = self.plant.discrete_affine_matrices(p.sample_time_s)
        x = cp.Variable((n, horizon + 1))
        u_p = cp.Variable(horizon)
        u_q = cp.Variable(horizon)

        constraints = [x[:, 0] == state.as_vector()]
        objective = 0.0

        for j in range(horizon):
            constraints.append(x[:, j + 1] == ad @ x[:, j] + bd[:, 0] * u_p[j] + gd + ed[:, 0] * d_forecast[j])
            voltage_pred = (
                voltage_base_pu
                + dv_dp_pu_per_kw * fleet_size * (u_p[j] - p_baseline_kw)
                + dv_dq_pu_per_kvar * fleet_size * (u_q[j] - q_baseline_kvar)
            )
            constraints += [
                u_p[j] >= self.inverter_params.p_min_kw,
                u_p[j] <= self.inverter_params.p_max_kw,
                u_q[j] >= -self.inverter_params.q_max_kvar,
                u_q[j] >= q_floor_kvar,
                u_q[j] <= self.inverter_params.q_max_kvar,
                cp.norm(cp.hstack([u_p[j], u_q[j]]), 2) <= self.inverter_params.s_rated_kva,
                x[1, j + 1] >= self.thermal_params.min_temperature_c,
                x[1, j + 1] <= self.thermal_params.max_temperature_c,
                voltage_pred >= p.voltage_min_pu,
                voltage_pred <= p.voltage_max_pu,
            ]

            du_p = u_p[j] - (previous_p_kw if j == 0 else u_p[j - 1])
            du_q = u_q[j] - (previous_q_kvar if j == 0 else u_q[j - 1])
            objective += p.weight_frequency * cp.square(x[0, j + 1] - frequency_reference_hz)
            objective += p.weight_temperature * cp.square(x[1, j + 1] - tref)
            objective += p.weight_voltage * cp.square(cp.pos(p.voltage_reference_pu - voltage_pred))
            objective += p.weight_control_move * cp.square(du_p)
            objective += p.weight_reactive_move * cp.square(du_q)
            objective += p.weight_power_tracking * cp.square(u_p[j] - self.inverter_params.p_set_kw)
            objective += p.weight_reactive_effort * cp.square(u_q[j])

        problem = cp.Problem(cp.Minimize(objective), constraints)
        problem.solve(solver=cp.CLARABEL, warm_start=True, verbose=False)

        if problem.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}:
            fallback_p = self._fallback_control(state, previous_p_kw, previous_q_kvar)
            return DMPCResult(
                p_command_kw=fallback_p,
                q_command_kvar=previous_q_kvar,
                predicted_states=np.full((n, horizon + 1), np.nan),
                predicted_controls_kw=np.full(horizon, np.nan),
                predicted_reactive_controls_kvar=np.full(horizon, np.nan),
                status=problem.status,
                objective_value=float("nan"),
            )

        return DMPCResult(
            p_command_kw=float(u_p.value[0]),
            q_command_kvar=float(u_q.value[0]),
            predicted_states=np.array(x.value, dtype=float),
            predicted_controls_kw=np.array(u_p.value, dtype=float),
            predicted_reactive_controls_kvar=np.array(u_q.value, dtype=float),
            status=problem.status,
            objective_value=float(problem.value),
        )
