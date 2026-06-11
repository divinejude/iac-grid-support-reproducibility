"""Autonomous Volt-Var and Frequency-Watt support laws."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class VoltVarDroop:
    """IEEE-style piecewise Volt-Var curve.

    Positive Q means capacitive injection during a voltage sag. Negative Q
    means absorption during over-voltage.
    """

    q_max_kvar: float
    v1_pu: float = 0.94
    v2_pu: float = 0.98
    v3_pu: float = 1.02
    v4_pu: float = 1.06

    def q_command(self, voltage_pu: float) -> float:
        if voltage_pu <= self.v1_pu:
            return self.q_max_kvar
        if voltage_pu < self.v2_pu:
            return self.q_max_kvar * (self.v2_pu - voltage_pu) / (self.v2_pu - self.v1_pu)
        if voltage_pu <= self.v3_pu:
            return 0.0
        if voltage_pu < self.v4_pu:
            return -self.q_max_kvar * (voltage_pu - self.v3_pu) / (self.v4_pu - self.v3_pu)
        return -self.q_max_kvar


@dataclass(frozen=True)
class FrequencyWattFiveRegion:
    """Five-region autonomous active-power modulator for an inverter AC."""

    p_min_kw: float
    p_set_kw: float
    p_max_kw: float
    t_min_c: float
    t_set_c: float
    t_max_c: float
    frequency_deadband_hz: float = 0.03
    full_response_deviation_hz: float = 0.30

    def p_command(self, frequency_deviation_hz: float, indoor_temperature_c: float) -> float:
        if abs(frequency_deviation_hz) <= self.frequency_deadband_hz:
            return self.p_set_kw

        response = min(
            max((abs(frequency_deviation_hz) - self.frequency_deadband_hz), 0.0)
            / max(self.full_response_deviation_hz - self.frequency_deadband_hz, 1e-9),
            1.0,
        )

        if frequency_deviation_hz < 0.0:
            if indoor_temperature_c >= self.t_max_c:
                return self.p_min_kw
            thermal_headroom = min(
                max((indoor_temperature_c - self.t_set_c) / max(self.t_max_c - self.t_set_c, 1e-9), 0.0),
                1.0,
            )
            return self.p_set_kw - response * thermal_headroom * (self.p_set_kw - self.p_min_kw)

        if indoor_temperature_c <= self.t_min_c:
            return self.p_max_kw
        thermal_headroom = min(
            max((self.t_set_c - indoor_temperature_c) / max(self.t_set_c - self.t_min_c, 1e-9), 0.0),
            1.0,
        )
        return self.p_set_kw + response * thermal_headroom * (self.p_max_kw - self.p_set_kw)
