"""Central parameter definitions for the IAC grid-support simulation."""

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class GridParameters:
    nominal_frequency_hz: float = 60.0
    nominal_voltage_rms: float = 240.0
    signal_sample_time_s: float = 1.0 / 2400.0
    sogi_gain: float = math.sqrt(2.0)
    pll_kp: float = 120.0
    pll_ki: float = 2500.0


@dataclass(frozen=True)
class FeederParameters:
    """Single-bus Thevenin feeder parameters in per-unit."""

    source_voltage_pu: float = 1.0
    base_power_kva: float = 10.0
    resistance_pu: float = 0.018
    reactance_pu: float = 0.045
    fixed_load_kw: float = 3.0
    fixed_load_kvar: float = 1.0
    voltage_solver_iterations: int = 12


@dataclass(frozen=True)
class TCLFleetParameters:
    """Aggregate TCL fleet used for distribution-level studies."""

    unit_count: int = 300
    allocation_strategy: str = "load_proportional"
    benchmark_name: str = "IEEE 13-node test feeder"


@dataclass(frozen=True)
class SwingFrequencyParameters:
    """Bulk-grid swing-equation equivalent used by the simulation plant."""

    inertia_constant_s: float = 4.5
    damping_pu_per_hz: float = 1.2
    base_power_kw: float = 100.0


@dataclass(frozen=True)
class InverterParameters:
    s_rated_kva: float = 4.0
    p_min_kw: float = 0.6
    p_set_kw: float = 2.0
    p_max_kw: float = 3.5
    q_max_kvar: float = 2.2


@dataclass(frozen=True)
class InverterInnerLoopParameters:
    current_controller_bandwidth_hz: float = 18.0
    dc_link_voltage_v: float = 390.0
    dc_link_current_limit_a: float = 13.0
    rms_current_limit_a: float = 18.0
    anti_windup_gain: float = 0.35


@dataclass(frozen=True)
class ThermalParameters:
    ambient_temperature_c: float = 32.0
    setpoint_c: float = 24.0
    comfort_band_c: float = 1.0
    resistance_c_per_kw: float = 4.0 / 3.0
    capacitance_kwh_per_c: float = 2.5
    cop: float = 3.0
    cop_temperature_slope_per_c: float = -0.035
    internal_gain_kw: float = 0.25

    @property
    def min_temperature_c(self) -> float:
        return self.setpoint_c - self.comfort_band_c

    @property
    def max_temperature_c(self) -> float:
        return self.setpoint_c + self.comfort_band_c


@dataclass(frozen=True)
class CompressorParameters:
    time_constant_s: float = 30.0
    frequency_watt_gain_kw_per_hz: float = 1.2
    bias_kw: float = 0.0
    ramp_rate_kw_per_s: float = 0.08
    min_speed_hz: float = 25.0
    rated_speed_hz: float = 60.0
    max_speed_hz: float = 90.0
    minimum_dwell_time_s: float = 30.0


@dataclass(frozen=True)
class MeasurementParameters:
    voltage_noise_std_pu: float = 0.002
    frequency_noise_std_hz: float = 0.003
    temperature_noise_std_c: float = 0.03
    power_noise_std_kw: float = 0.015
    delay_steps: int = 1
    random_seed: int = 7


@dataclass(frozen=True)
class UncertaintyParameters:
    resistance_std_fraction: float = 0.15
    capacitance_std_fraction: float = 0.20
    cop_std_fraction: float = 0.10
    compressor_time_constant_std_fraction: float = 0.15


@dataclass(frozen=True)
class FrequencyPlantParameters:
    """Reduced local frequency-response model for one flexible IAC resource.

    The exogenous disturbance is in Hz/s. Without IAC action, a disturbance
    of -0.00875 Hz/s with tau=40 s settles near -0.35 Hz.
    """

    time_constant_s: float = 40.0
    load_relief_gain_hz_per_kw_s: float = 0.005


@dataclass(frozen=True)
class DMPCParameters:
    sample_time_s: float = 5.0
    horizon_steps: int = 12
    weight_frequency: float = 60.0
    weight_temperature: float = 3.0
    weight_voltage: float = 500.0
    weight_control_move: float = 0.04
    weight_reactive_move: float = 0.02
    weight_power_tracking: float = 0.02
    weight_reactive_effort: float = 0.01
    voltage_reference_pu: float = 0.96
    voltage_min_pu: float = 0.88
    voltage_max_pu: float = 1.08
