"""Compressor operational constraints and manufacturer-like maps."""

from __future__ import annotations

from dataclasses import dataclass, replace
import math

from .parameters import CompressorParameters, InverterParameters, ThermalParameters


@dataclass
class CompressorCommandState:
    last_power_command_kw: float
    last_switch_time_s: float = 0.0
    enabled: bool = True


@dataclass(frozen=True)
class CompressorOperatingPoint:
    power_kw: float
    speed_hz: float
    cop: float
    cooling_capacity_kw: float
    ramp_limited: bool
    dwell_limited: bool


class CompressorConstraintModel:
    """Maps active-power commands to feasible speed/COP/cooling operation."""

    def __init__(self, compressor: CompressorParameters, inverter: InverterParameters, thermal: ThermalParameters):
        self.compressor = compressor
        self.inverter = inverter
        self.thermal = thermal

    def apply(
        self,
        state: CompressorCommandState,
        requested_power_kw: float,
        indoor_temperature_c: float,
        ambient_temperature_c: float,
        sample_time_s: float,
        time_s: float,
    ) -> tuple[CompressorOperatingPoint, CompressorCommandState]:
        dwell_limited = False
        feasible = max(self.inverter.p_min_kw, min(self.inverter.p_max_kw, requested_power_kw))
        max_delta = self.compressor.ramp_rate_kw_per_s * sample_time_s
        delta = feasible - state.last_power_command_kw
        ramp_limited = abs(delta) > max_delta
        power = state.last_power_command_kw + max(-max_delta, min(max_delta, delta))

        if not state.enabled and time_s - state.last_switch_time_s < self.compressor.minimum_dwell_time_s:
            power = self.inverter.p_min_kw
            dwell_limited = True

        speed = self.power_to_speed_hz(power)
        cop = self.cop(indoor_temperature_c, ambient_temperature_c, speed)
        next_state = replace(state, last_power_command_kw=power)
        return (
            CompressorOperatingPoint(
                power_kw=power,
                speed_hz=speed,
                cop=cop,
                cooling_capacity_kw=cop * power,
                ramp_limited=ramp_limited,
                dwell_limited=dwell_limited,
            ),
            next_state,
        )

    def power_to_speed_hz(self, power_kw: float) -> float:
        frac = (power_kw - self.inverter.p_min_kw) / max(self.inverter.p_max_kw - self.inverter.p_min_kw, 1e-9)
        frac = max(0.0, min(1.0, frac))
        # Compressor power is roughly cubic with speed; invert that relationship.
        speed_frac = frac ** (1.0 / 3.0)
        return self.compressor.min_speed_hz + speed_frac * (self.compressor.max_speed_hz - self.compressor.min_speed_hz)

    def cop(self, indoor_temperature_c: float, ambient_temperature_c: float, speed_hz: float) -> float:
        lift_c = max(ambient_temperature_c - indoor_temperature_c, 1.0)
        nominal_lift_c = self.thermal.ambient_temperature_c - self.thermal.setpoint_c
        lift_factor = 1.0 + self.thermal.cop_temperature_slope_per_c * (lift_c - nominal_lift_c)
        speed_ratio = speed_hz / max(self.compressor.rated_speed_hz, 1e-9)
        speed_penalty = 1.0 - 0.08 * (speed_ratio - 1.0) ** 2
        return max(1.0, self.thermal.cop * lift_factor * speed_penalty)
