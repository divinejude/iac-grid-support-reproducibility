"""Thermal, compressor, and reduced frequency plant models."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import control
from scipy.integrate import solve_ivp
from scipy.linalg import expm

from .parameters import CompressorParameters, FrequencyPlantParameters, InverterParameters, ThermalParameters


@dataclass
class PlantState:
    frequency_deviation_hz: float
    indoor_temperature_c: float
    compressor_power_kw: float

    def as_vector(self) -> np.ndarray:
        return np.array(
            [self.frequency_deviation_hz, self.indoor_temperature_c, self.compressor_power_kw],
            dtype=float,
        )

    @classmethod
    def from_vector(cls, x: np.ndarray) -> "PlantState":
        return cls(
            frequency_deviation_hz=float(x[0]),
            indoor_temperature_c=float(x[1]),
            compressor_power_kw=float(x[2]),
        )


class ThermalRoomModel:
    """First-order RC room model driven by compressor cooling capacity."""

    def __init__(self, params: ThermalParameters):
        self.params = params

    def derivative(
        self,
        temperature_c: float,
        compressor_power_kw: float,
        ambient_c: float | None = None,
        cooling_capacity_kw: float | None = None,
    ) -> float:
        p = self.params
        ambient = p.ambient_temperature_c if ambient_c is None else ambient_c
        envelope_heat_kw = (ambient - temperature_c) / p.resistance_c_per_kw
        cooling_kw = p.cop * compressor_power_kw if cooling_capacity_kw is None else cooling_capacity_kw
        return (envelope_heat_kw + p.internal_gain_kw - cooling_kw) / (p.capacitance_kwh_per_c * 3600.0)


class CompressorElectricalModel:
    """First-order inverter-compressor active-power response."""

    def __init__(self, params: CompressorParameters):
        self.params = params

    def derivative(self, power_kw: float, setpoint_kw: float, frequency_deviation_hz: float = 0.0) -> float:
        target_kw = (
            setpoint_kw
            + self.params.frequency_watt_gain_kw_per_hz * frequency_deviation_hz
            + self.params.bias_kw
        )
        return (target_kw - power_kw) / self.params.time_constant_s

    def frequency_to_power_transfer_function(self) -> control.TransferFunction:
        """Return Delta P_IAC / Delta f_IAC = Kp / (Tc s + 1)."""
        return control.tf(
            [self.params.frequency_watt_gain_kw_per_hz],
            [self.params.time_constant_s, 1.0],
        )


class IACPlant:
    """Coupled IAC thermal/electrical plant with a local frequency state."""

    def __init__(
        self,
        thermal: ThermalParameters,
        compressor: CompressorParameters,
        inverter: InverterParameters,
        frequency: FrequencyPlantParameters,
    ):
        self.thermal_params = thermal
        self.compressor_params = compressor
        self.inverter_params = inverter
        self.frequency_params = frequency
        self.thermal_model = ThermalRoomModel(thermal)
        self.compressor_model = CompressorElectricalModel(compressor)

    def continuous_matrices(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return xdot = A x + B u + g + E d for x=[df, T, P]."""
        f = self.frequency_params
        th = self.thermal_params
        comp = self.compressor_params
        inv = self.inverter_params

        a = np.zeros((3, 3), dtype=float)
        b = np.zeros((3, 1), dtype=float)
        g = np.zeros(3, dtype=float)
        e = np.zeros((3, 1), dtype=float)

        a[0, 0] = -1.0 / f.time_constant_s
        a[0, 2] = -f.load_relief_gain_hz_per_kw_s
        g[0] = f.load_relief_gain_hz_per_kw_s * inv.p_set_kw
        e[0, 0] = 1.0

        denom = th.capacitance_kwh_per_c * 3600.0
        a[1, 1] = -1.0 / (th.resistance_c_per_kw * denom)
        a[1, 2] = -th.cop / denom
        g[1] = (th.ambient_temperature_c / th.resistance_c_per_kw + th.internal_gain_kw) / denom

        a[2, 2] = -1.0 / comp.time_constant_s
        b[2, 0] = 1.0 / comp.time_constant_s
        return a, b, g, e

    def continuous_state_space(self) -> control.StateSpace:
        """Return the linear plant xdot = A x + B u, y = x as a control.StateSpace."""
        a, b, _, _ = self.continuous_matrices()
        c = np.eye(a.shape[0])
        d = np.zeros((a.shape[0], b.shape[1]))
        return control.ss(a, b, c, d)

    def discrete_affine_matrices(self, sample_time_s: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Exact zero-order-hold discretization of xdot = A x + B u + g + E d."""
        a, b, g, e = self.continuous_matrices()
        n = a.shape[0]
        m = b.shape[1]
        r = e.shape[1]
        aug = np.zeros((n + m + 1 + r, n + m + 1 + r), dtype=float)
        aug[:n, :n] = a
        aug[:n, n : n + m] = b
        aug[:n, n + m] = g
        aug[:n, n + m + 1 :] = e
        exp_aug = expm(aug * sample_time_s)
        ad = exp_aug[:n, :n]
        bd = exp_aug[:n, n : n + m]
        gd = exp_aug[:n, n + m]
        ed = exp_aug[:n, n + m + 1 :]
        return ad, bd, gd, ed

    def rhs(self, _time_s: float, x: np.ndarray, control_kw: float, disturbance_hz_per_s: float) -> np.ndarray:
        a, b, g, e = self.continuous_matrices()
        return a @ x + b[:, 0] * control_kw + g + e[:, 0] * disturbance_hz_per_s

    def step_exact(
        self,
        state: PlantState,
        control_kw: float,
        disturbance_hz_per_s: float,
        sample_time_s: float,
    ) -> PlantState:
        ad, bd, gd, ed = self.discrete_affine_matrices(sample_time_s)
        x_next = ad @ state.as_vector() + bd[:, 0] * control_kw + gd + ed[:, 0] * disturbance_hz_per_s
        return PlantState.from_vector(x_next)

    def step_ode(
        self,
        state: PlantState,
        control_kw: float,
        disturbance_hz_per_s: float,
        sample_time_s: float,
    ) -> PlantState:
        solution = solve_ivp(
            lambda t, x: self.rhs(t, x, control_kw, disturbance_hz_per_s),
            (0.0, sample_time_s),
            state.as_vector(),
            rtol=1e-8,
            atol=1e-10,
        )
        return PlantState.from_vector(solution.y[:, -1])

    def step_thermal_compressor(
        self,
        state: PlantState,
        compressor_power_command_kw: float,
        cooling_capacity_kw: float,
        measured_frequency_deviation_hz: float,
        sample_time_s: float,
        ambient_c: float | None = None,
    ) -> PlantState:
        """Advance only IAC thermal and compressor states.

        This is used when the grid frequency state is supplied by an external
        feeder or swing-equation simulation rather than by the reduced MPC
        prediction model.
        """

        def rhs(_time_s: float, y: np.ndarray) -> np.ndarray:
            temperature, compressor_power = y
            d_temperature = self.thermal_model.derivative(
                temperature,
                compressor_power,
                ambient_c=ambient_c,
                cooling_capacity_kw=cooling_capacity_kw,
            )
            d_power = self.compressor_model.derivative(
                compressor_power,
                compressor_power_command_kw,
                measured_frequency_deviation_hz,
            )
            return np.array([d_temperature, d_power])

        solution = solve_ivp(
            rhs,
            (0.0, sample_time_s),
            np.array([state.indoor_temperature_c, state.compressor_power_kw]),
            rtol=1e-8,
            atol=1e-10,
        )
        return PlantState(
            frequency_deviation_hz=measured_frequency_deviation_hz,
            indoor_temperature_c=float(solution.y[0, -1]),
            compressor_power_kw=float(solution.y[1, -1]),
        )
