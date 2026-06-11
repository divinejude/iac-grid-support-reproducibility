"""Dynamic grid co-simulation interfaces.

The simulation can use the built-in dynamic swing backend or a replay backend
that imports trajectories exported by a transmission dynamic simulator. Replay
files may include both a no-control disturbance trajectory and an IAC
step-response kernel, allowing active-power modulation to affect frequency by
linear superposition.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol
import csv

from .feeder import SwingFrequencyModel, SwingFrequencyState
from .parameters import SwingFrequencyParameters


@dataclass(frozen=True)
class GridDynamicInput:
    time_s: float
    iac_power_kw: float
    disturbance_kw: float
    sample_time_s: float


@dataclass(frozen=True)
class GridDynamicOutput:
    frequency_deviation_hz: float
    backend_name: str
    converged: bool = True


class DynamicGridBackend(Protocol):
    """Minimal dynamic-grid co-simulation contract."""

    @property
    def name(self) -> str:
        ...

    def initialize(self, nominal_iac_kw: float) -> GridDynamicOutput:
        ...

    def step(self, input_data: GridDynamicInput) -> GridDynamicOutput:
        ...


class SwingEquationDynamicBackend:
    """Built-in dynamic backend used when no external simulator is available."""

    name = "swing_equation_dynamic_backend"

    def __init__(self, params: SwingFrequencyParameters):
        self.params = params
        self.model: SwingFrequencyModel | None = None
        self.state = SwingFrequencyState(0.0)

    def initialize(self, nominal_iac_kw: float) -> GridDynamicOutput:
        self.model = SwingFrequencyModel(self.params, nominal_iac_kw)
        self.state = SwingFrequencyState(0.0)
        return GridDynamicOutput(self.state.frequency_deviation_hz, self.name)

    def step(self, input_data: GridDynamicInput) -> GridDynamicOutput:
        if self.model is None:
            raise RuntimeError("Dynamic grid backend must be initialized before stepping")
        self.state = self.model.step(
            self.state,
            input_data.iac_power_kw,
            input_data.disturbance_kw,
            input_data.sample_time_s,
        )
        return GridDynamicOutput(self.state.frequency_deviation_hz, self.name)


class CsvReplayDynamicBackend:
    """Replay a frequency trajectory from a dynamic simulator export.

    The CSV must contain ``time_s`` and ``frequency_deviation_hz`` columns. If
    it also contains ``iac_step_response_hz_per_kw``, the backend treats that
    column as the frequency response to a sustained 1 kW IAC load reduction and
    superposes changes in aggregate IAC load onto the replayed no-control
    frequency trajectory.
    """

    name = "csv_replay_dynamic_backend"

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.time_s: list[float] = []
        self.frequency_deviation_hz: list[float] = []
        self.iac_step_response_hz_per_kw: list[float] = []
        self.nominal_iac_kw = 0.0
        self.last_load_relief_kw = 0.0
        self.relief_changes: list[tuple[float, float]] = []

    def initialize(self, nominal_iac_kw: float) -> GridDynamicOutput:
        _ = nominal_iac_kw
        with self.path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            rows = list(reader)
        if not rows:
            raise ValueError(f"Dynamic grid replay file is empty: {self.path}")
        required = {"time_s", "frequency_deviation_hz"}
        missing = required - set(rows[0])
        if missing:
            raise ValueError(f"Missing dynamic replay columns {sorted(missing)} in {self.path}")
        self.time_s = [float(row["time_s"]) for row in rows]
        self.frequency_deviation_hz = [float(row["frequency_deviation_hz"]) for row in rows]
        self.iac_step_response_hz_per_kw = (
            [float(row["iac_step_response_hz_per_kw"]) for row in rows]
            if "iac_step_response_hz_per_kw" in rows[0]
            else []
        )
        self.nominal_iac_kw = nominal_iac_kw
        self.last_load_relief_kw = 0.0
        self.relief_changes = [(0.0, 0.0)]
        return GridDynamicOutput(self.frequency_deviation_hz[0], self.name)

    def _interpolate(self, values: list[float], t: float) -> float:
        if t <= self.time_s[0]:
            return values[0]
        if t >= self.time_s[-1]:
            return values[-1]
        for k in range(len(self.time_s) - 1):
            if self.time_s[k] <= t <= self.time_s[k + 1]:
                fraction = (t - self.time_s[k]) / max(self.time_s[k + 1] - self.time_s[k], 1e-12)
                return values[k] + fraction * (values[k + 1] - values[k])
        return values[-1]

    def step(self, input_data: GridDynamicInput) -> GridDynamicOutput:
        if not self.time_s:
            raise RuntimeError("CSV replay backend must be initialized before stepping")
        t = input_data.time_s + input_data.sample_time_s
        value = self._interpolate(self.frequency_deviation_hz, t)
        if self.iac_step_response_hz_per_kw:
            load_relief_kw = self.nominal_iac_kw - input_data.iac_power_kw
            delta_relief_kw = load_relief_kw - self.last_load_relief_kw
            if abs(delta_relief_kw) > 1e-9:
                self.relief_changes.append((t, delta_relief_kw))
                self.last_load_relief_kw = load_relief_kw
            value += sum(
                delta_kw * self._interpolate(self.iac_step_response_hz_per_kw, max(0.0, t - event_time))
                for event_time, delta_kw in self.relief_changes
            )
        return GridDynamicOutput(float(value), self.name)


class DynamicGridCosimulator:
    """Owns the selected dynamic grid backend for closed-loop simulation."""

    def __init__(self, backend: DynamicGridBackend, nominal_iac_kw: float):
        self.backend = backend
        self.output = backend.initialize(nominal_iac_kw)

    @property
    def frequency_deviation_hz(self) -> float:
        return self.output.frequency_deviation_hz

    def step(
        self,
        time_s: float,
        iac_power_kw: float,
        disturbance_kw: float,
        sample_time_s: float,
    ) -> GridDynamicOutput:
        self.output = self.backend.step(GridDynamicInput(time_s, iac_power_kw, disturbance_kw, sample_time_s))
        return self.output
