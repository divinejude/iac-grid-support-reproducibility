"""Measurement noise, communication delay, and parameter uncertainty tools."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, replace

import numpy as np

from .parameters import (
    CompressorParameters,
    MeasurementParameters,
    ThermalParameters,
    UncertaintyParameters,
)


@dataclass(frozen=True)
class Measurement:
    voltage_pu: float
    frequency_deviation_hz: float
    temperature_c: float
    power_kw: float


class MeasurementChannel:
    """Adds configurable Gaussian sensor noise and fixed sample delay."""

    def __init__(self, params: MeasurementParameters):
        self.params = params
        self.rng = np.random.default_rng(params.random_seed)
        self.buffer: deque[Measurement] = deque(maxlen=params.delay_steps + 1)

    def sample(self, measurement: Measurement) -> Measurement:
        noisy = Measurement(
            voltage_pu=measurement.voltage_pu + self.rng.normal(0.0, self.params.voltage_noise_std_pu),
            frequency_deviation_hz=measurement.frequency_deviation_hz
            + self.rng.normal(0.0, self.params.frequency_noise_std_hz),
            temperature_c=measurement.temperature_c + self.rng.normal(0.0, self.params.temperature_noise_std_c),
            power_kw=measurement.power_kw + self.rng.normal(0.0, self.params.power_noise_std_kw),
        )
        self.buffer.append(noisy)
        if len(self.buffer) <= self.params.delay_steps:
            return noisy
        return self.buffer[0]


def sample_uncertain_parameters(
    thermal: ThermalParameters,
    compressor: CompressorParameters,
    uncertainty: UncertaintyParameters,
    seed: int,
) -> tuple[ThermalParameters, CompressorParameters]:
    rng = np.random.default_rng(seed)

    def perturb(value: float, std_fraction: float) -> float:
        return max(1e-9, value * (1.0 + rng.normal(0.0, std_fraction)))

    return (
        replace(
            thermal,
            resistance_c_per_kw=perturb(thermal.resistance_c_per_kw, uncertainty.resistance_std_fraction),
            capacitance_kwh_per_c=perturb(thermal.capacitance_kwh_per_c, uncertainty.capacitance_std_fraction),
            cop=perturb(thermal.cop, uncertainty.cop_std_fraction),
        ),
        replace(
            compressor,
            time_constant_s=perturb(
                compressor.time_constant_s,
                uncertainty.compressor_time_constant_std_fraction,
            ),
        ),
    )
