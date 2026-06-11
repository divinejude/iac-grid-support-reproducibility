"""Physically grounded grid/feeder models for IAC simulations."""

from __future__ import annotations

from dataclasses import dataclass
import cmath
import math
from pathlib import Path
from typing import Any

from .parameters import FeederParameters, SwingFrequencyParameters


@dataclass(frozen=True)
class FeederSolution:
    pcc_voltage_pu: float
    voltage_angle_rad: float
    current_pu: complex
    apparent_power_pu: complex
    min_voltage_pu: float
    max_voltage_pu: float


@dataclass(frozen=True)
class TCLLoadAllocation:
    name: str
    bus: str
    phases: int
    conn: str
    kv: float
    weight: float


@dataclass(frozen=True)
class VoltageSensitivity:
    base_voltage_pu: float
    dv_dp_pu_per_kw: float
    dv_dq_pu_per_kvar: float


class SingleBusTheveninFeeder:
    """Single-load-bus Thevenin equivalent.

    Sign convention: positive P/Q is net consumption at the PCC. Capacitive
    inverter injection is represented by negative net Q contribution.
    """

    def __init__(self, params: FeederParameters):
        self.params = params
        self.impedance_pu = complex(params.resistance_pu, params.reactance_pu)

    def solve_pcc_voltage(
        self,
        source_voltage_pu: float,
        iac_p_kw: float,
        iac_q_kvar: float,
        extra_load_kw: float = 0.0,
        extra_load_kvar: float = 0.0,
    ) -> FeederSolution:
        p = self.params
        source = complex(source_voltage_pu, 0.0)
        net_kw = p.fixed_load_kw + extra_load_kw + iac_p_kw
        net_kvar = p.fixed_load_kvar + extra_load_kvar - iac_q_kvar
        s_pu = complex(net_kw, net_kvar) / p.base_power_kva
        voltage = source

        for _ in range(p.voltage_solver_iterations):
            current = s_pu.conjugate() / max(abs(voltage), 1e-6)
            voltage = source - self.impedance_pu * current

        current = s_pu.conjugate() / max(abs(voltage), 1e-6)
        return FeederSolution(
            pcc_voltage_pu=abs(voltage),
            voltage_angle_rad=cmath.phase(voltage),
            current_pu=current,
            apparent_power_pu=s_pu,
            min_voltage_pu=abs(voltage),
            max_voltage_pu=abs(voltage),
        )

    def estimate_voltage_sensitivity(
        self,
        source_voltage_pu: float,
        iac_p_kw: float,
        iac_q_kvar: float,
        perturb_kw: float = 25.0,
        perturb_kvar: float = 25.0,
        metric: str = "min",
        extra_load_kw: float = 0.0,
        extra_load_kvar: float = 0.0,
    ) -> VoltageSensitivity:
        base = self.solve_pcc_voltage(source_voltage_pu, iac_p_kw, iac_q_kvar, extra_load_kw, extra_load_kvar)
        p_plus = self.solve_pcc_voltage(
            source_voltage_pu,
            iac_p_kw + perturb_kw,
            iac_q_kvar,
            extra_load_kw,
            extra_load_kvar,
        )
        q_plus = self.solve_pcc_voltage(
            source_voltage_pu,
            iac_p_kw,
            iac_q_kvar + perturb_kvar,
            extra_load_kw,
            extra_load_kvar,
        )
        self.solve_pcc_voltage(source_voltage_pu, iac_p_kw, iac_q_kvar, extra_load_kw, extra_load_kvar)

        def voltage(solution: FeederSolution) -> float:
            if metric == "pcc":
                return solution.pcc_voltage_pu
            if metric == "min":
                return solution.min_voltage_pu
            raise ValueError(f"Unknown voltage sensitivity metric: {metric}")

        return VoltageSensitivity(
            base_voltage_pu=voltage(base),
            dv_dp_pu_per_kw=(voltage(p_plus) - voltage(base)) / perturb_kw,
            dv_dq_pu_per_kvar=(voltage(q_plus) - voltage(base)) / perturb_kvar,
        )


class OpenDSSFeeder:
    """OpenDSS snapshot power-flow co-simulation adapter.

    The IAC is represented as a controllable single-phase load at the PCC.
    Positive IAC reactive command means capacitive support, implemented as a
    negative OpenDSS load kvar.
    """

    def __init__(self, params: FeederParameters, nominal_voltage_rms: float = 240.0):
        try:
            import opendssdirect as dss
        except ImportError as exc:
            raise RuntimeError(
                "OpenDSS co-simulation requires opendssdirect.py. "
                "Install dependencies with `pip install -r requirements.txt`."
            ) from exc

        self.dss: Any = dss
        self.params = params
        self.nominal_voltage_rms = nominal_voltage_rms
        self._build_circuit()

    def _command(self, command: str) -> None:
        self.dss.Text.Command(command)

    def _build_circuit(self) -> None:
        p = self.params
        self._command("Clear")
        self._command(
            "New Circuit.IACFeeder "
            "basekv=0.24 pu=1.0 phases=1 bus1=source.1 frequency=60"
        )
        self._command(
            "Edit Vsource.Source "
            "bus1=source.1 phases=1 basekv=0.24 pu=1.0 angle=0 frequency=60"
        )
        self._command(
            "New Line.Feeder "
            f"bus1=source.1 bus2=pcc.1 phases=1 r1={p.resistance_pu} x1={p.reactance_pu} "
            f"r0={p.resistance_pu} x0={p.reactance_pu} c1=0 c0=0 length=1 units=none"
        )
        self._command(
            "New Load.Fixed "
            f"bus1=pcc.1 phases=1 conn=wye kv=0.24 kw={p.fixed_load_kw} "
            f"kvar={p.fixed_load_kvar} model=1"
        )
        self._command("New Load.IAC bus1=pcc.1 phases=1 conn=wye kv=0.24 kw=0 kvar=0 model=1")
        self._command("Set VoltageBases=[0.24]")
        self._command("CalcVoltageBases")
        self._command("Set MaxControlIter=100")

    def solve_pcc_voltage(
        self,
        source_voltage_pu: float,
        iac_p_kw: float,
        iac_q_kvar: float,
        extra_load_kw: float = 0.0,
        extra_load_kvar: float = 0.0,
    ) -> FeederSolution:
        p = self.params
        fixed_kw = p.fixed_load_kw + extra_load_kw
        fixed_kvar = p.fixed_load_kvar + extra_load_kvar
        load_kvar = -iac_q_kvar

        self._command(f"Edit Vsource.Source pu={source_voltage_pu}")
        self._command(f"Edit Load.Fixed kw={fixed_kw} kvar={fixed_kvar}")
        self._command(f"Edit Load.IAC kw={max(iac_p_kw, 0.0)} kvar={load_kvar}")
        self._command("Solve mode=snap")
        if not self.dss.Solution.Converged():
            raise RuntimeError("OpenDSS power flow did not converge")

        self.dss.Circuit.SetActiveBus("pcc")
        voltage_mag_angle = self.dss.Bus.VMagAngle()
        if len(voltage_mag_angle) < 2:
            raise RuntimeError("OpenDSS did not return PCC voltage")

        voltage_rms = float(voltage_mag_angle[0])
        voltage_angle_rad = math.radians(float(voltage_mag_angle[1]))
        net_kw = fixed_kw + iac_p_kw
        net_kvar = fixed_kvar - iac_q_kvar
        apparent_pu = complex(net_kw, net_kvar) / p.base_power_kva

        self.dss.Circuit.SetActiveElement("Line.Feeder")
        current_mag_angle = self.dss.CktElement.CurrentsMagAng()
        current_pu = 0.0j
        if len(current_mag_angle) >= 2:
            current_base_a = p.base_power_kva * 1000.0 / max(self.nominal_voltage_rms, 1e-6)
            current_pu = cmath.rect(
                current_mag_angle[0] / max(current_base_a, 1e-9),
                math.radians(current_mag_angle[1]),
            )

        return FeederSolution(
            pcc_voltage_pu=voltage_rms / self.nominal_voltage_rms,
            voltage_angle_rad=voltage_angle_rad,
            current_pu=current_pu,
            apparent_power_pu=apparent_pu,
            min_voltage_pu=voltage_rms / self.nominal_voltage_rms,
            max_voltage_pu=voltage_rms / self.nominal_voltage_rms,
        )


class IEEE13OpenDSSFleetFeeder:
    """IEEE 13-node benchmark feeder with load-proportional TCL placement."""

    def __init__(
        self,
        dss_file: str | Path | None = None,
        base_power_kva: float = 5000.0,
        monitored_bus: str | None = None,
        allocation_strategy: str = "load_proportional",
        weak_bus_exponent: float = 2.0,
    ):
        try:
            import opendssdirect as dss
        except ImportError as exc:
            raise RuntimeError(
                "IEEE 13-node OpenDSS co-simulation requires opendssdirect.py. "
                "Install dependencies with `pip install -r requirements.txt`."
            ) from exc

        self.dss: Any = dss
        project_root = Path(__file__).resolve().parents[1]
        self.dss_file = (project_root / "examples/ieee13/IEEE13Nodeckt.dss") if dss_file is None else Path(dss_file).resolve()
        self.base_power_kva = base_power_kva
        self.monitored_bus = monitored_bus
        self.allocation_strategy = allocation_strategy
        self.weak_bus_exponent = weak_bus_exponent
        self.allocations: list[TCLLoadAllocation] = []
        self.stress_allocations: list[TCLLoadAllocation] = []
        self._compile_benchmark()
        self._create_tcl_loads()

    def _command(self, command: str) -> None:
        self.dss.Text.Command(command)

    def _compile_benchmark(self) -> None:
        if not self.dss_file.exists():
            raise FileNotFoundError(f"IEEE 13-node OpenDSS file not found: {self.dss_file}")
        self._command(f"Compile [{self.dss_file}]")
        self._command("Set MaxControlIter=100")
        self._command("Solve mode=snap")
        if not self.dss.Solution.Converged():
            raise RuntimeError("OpenDSS IEEE 13-node benchmark did not converge at initialization")

    def _create_tcl_loads(self) -> None:
        load_records = []
        for load_name in self.dss.Loads.AllNames():
            self.dss.Loads.Name(load_name)
            self.dss.Circuit.SetActiveElement(f"Load.{load_name}")
            kw = max(float(self.dss.Loads.kW()), 0.0)
            if kw <= 0.0:
                continue
            load_records.append(
                {
                    "name": load_name,
                    "bus": self.dss.CktElement.BusNames()[0],
                    "phases": int(self.dss.CktElement.NumPhases()),
                    "conn": "delta" if self.dss.Loads.IsDelta() else "wye",
                    "kv": float(self.dss.Loads.kV()),
                    "kw": kw,
                }
            )

        weights = self._allocation_weights(load_records)
        total_kw = sum(record["kw"] for record in load_records)
        if total_kw <= 0.0:
            raise RuntimeError("IEEE 13-node benchmark has no positive loads for TCL allocation")

        self.allocations = []
        self.stress_allocations = []
        for record, weight in zip(load_records, weights):
            tcl_name = f"TCL_{record['name']}"
            stress_name = f"Stress_{record['name']}"
            self._command(
                f"New Load.{tcl_name} "
                f"bus1={record['bus']} phases={record['phases']} conn={record['conn']} "
                f"kv={record['kv']} kw=0 kvar=0 model=1"
            )
            self._command(
                f"New Load.{stress_name} "
                f"bus1={record['bus']} phases={record['phases']} conn={record['conn']} "
                f"kv={record['kv']} kw=0 kvar=0 model=1"
            )
            self.allocations.append(
                TCLLoadAllocation(
                    name=tcl_name,
                    bus=record["bus"],
                    phases=record["phases"],
                    conn=record["conn"],
                    kv=record["kv"],
                    weight=weight,
                )
            )
            self.stress_allocations.append(
                TCLLoadAllocation(
                    name=stress_name,
                    bus=record["bus"],
                    phases=record["phases"],
                    conn=record["conn"],
                    kv=record["kv"],
                    weight=weight,
                )
            )

    def _allocation_weights(self, load_records: list[dict[str, Any]]) -> list[float]:
        total_kw = sum(record["kw"] for record in load_records)
        if total_kw <= 0.0:
            raise RuntimeError("IEEE 13-node benchmark has no positive loads for TCL allocation")

        if self.allocation_strategy == "load_proportional":
            raw = [record["kw"] for record in load_records]
        elif self.allocation_strategy == "weak_bus_prioritized":
            voltages = []
            for record in load_records:
                bus_name = str(record["bus"]).split(".")[0]
                voltage, _angle = self._bus_phase_voltage(bus_name, str(record["bus"]))
                voltages.append(voltage)
            max_voltage = max(voltages)
            weakness = [max(max_voltage - voltage, 1e-4) ** self.weak_bus_exponent for voltage in voltages]
            raw = [record["kw"] * weak for record, weak in zip(load_records, weakness)]
        else:
            raise ValueError(
                "Unknown TCL allocation strategy "
                f"{self.allocation_strategy!r}. Expected 'load_proportional' or 'weak_bus_prioritized'."
            )

        total = sum(raw)
        return [value / total for value in raw]

    def solve_pcc_voltage(
        self,
        source_voltage_pu: float,
        iac_p_kw: float,
        iac_q_kvar: float,
        extra_load_kw: float = 0.0,
        extra_load_kvar: float = 0.0,
    ) -> FeederSolution:
        self._command(f"Edit Vsource.Source pu={source_voltage_pu}")
        for allocation in self.allocations:
            self._command(
                f"Edit Load.{allocation.name} "
                f"kw={max(iac_p_kw * allocation.weight, 0.0)} "
                f"kvar={-iac_q_kvar * allocation.weight}"
            )
        for allocation in self.stress_allocations:
            self._command(
                f"Edit Load.{allocation.name} "
                f"kw={max(extra_load_kw * allocation.weight, 0.0)} "
                f"kvar={max(extra_load_kvar * allocation.weight, 0.0)}"
            )
        self._command("Solve mode=snap")
        if not self.dss.Solution.Converged():
            raise RuntimeError("OpenDSS IEEE 13-node power flow did not converge")

        voltage_pu, angle_rad = self._fleet_weighted_voltage()
        min_voltage, max_voltage = self._all_bus_voltage_extrema()
        apparent_pu = complex(iac_p_kw, -iac_q_kvar) / self.base_power_kva
        return FeederSolution(
            pcc_voltage_pu=voltage_pu,
            voltage_angle_rad=angle_rad,
            current_pu=0.0j,
            apparent_power_pu=apparent_pu,
            min_voltage_pu=min_voltage,
            max_voltage_pu=max_voltage,
        )

    def estimate_voltage_sensitivity(
        self,
        source_voltage_pu: float,
        iac_p_kw: float,
        iac_q_kvar: float,
        perturb_kw: float = 25.0,
        perturb_kvar: float = 25.0,
        metric: str = "min",
        extra_load_kw: float = 0.0,
        extra_load_kvar: float = 0.0,
    ) -> VoltageSensitivity:
        base = self.solve_pcc_voltage(source_voltage_pu, iac_p_kw, iac_q_kvar, extra_load_kw, extra_load_kvar)
        p_plus = self.solve_pcc_voltage(
            source_voltage_pu,
            iac_p_kw + perturb_kw,
            iac_q_kvar,
            extra_load_kw,
            extra_load_kvar,
        )
        q_plus = self.solve_pcc_voltage(
            source_voltage_pu,
            iac_p_kw,
            iac_q_kvar + perturb_kvar,
            extra_load_kw,
            extra_load_kvar,
        )
        self.solve_pcc_voltage(source_voltage_pu, iac_p_kw, iac_q_kvar, extra_load_kw, extra_load_kvar)

        def voltage(solution: FeederSolution) -> float:
            if metric == "pcc":
                return solution.pcc_voltage_pu
            if metric == "min":
                return solution.min_voltage_pu
            raise ValueError(f"Unknown voltage sensitivity metric: {metric}")

        return VoltageSensitivity(
            base_voltage_pu=voltage(base),
            dv_dp_pu_per_kw=(voltage(p_plus) - voltage(base)) / perturb_kw,
            dv_dq_pu_per_kvar=(voltage(q_plus) - voltage(base)) / perturb_kvar,
        )

    def _fleet_weighted_voltage(self) -> tuple[float, float]:
        if self.monitored_bus:
            return self._bus_average_voltage(self.monitored_bus)

        weighted_voltage = 0.0
        weighted_angle = 0.0
        for allocation in self.allocations:
            bus_name = allocation.bus.split(".")[0]
            voltage, angle = self._bus_phase_voltage(bus_name, allocation.bus)
            weighted_voltage += allocation.weight * voltage
            weighted_angle += allocation.weight * angle
        return weighted_voltage, weighted_angle

    def _bus_average_voltage(self, bus_name: str) -> tuple[float, float]:
        self.dss.Circuit.SetActiveBus(bus_name)
        values = self.dss.Bus.puVmagAngle()
        magnitudes = values[0::2]
        angles = values[1::2]
        if not magnitudes:
            raise RuntimeError(f"OpenDSS bus has no voltage values: {bus_name}")
        return float(sum(magnitudes) / len(magnitudes)), math.radians(float(sum(angles) / len(angles)))

    def _bus_phase_voltage(self, bus_name: str, bus_spec: str) -> tuple[float, float]:
        self.dss.Circuit.SetActiveBus(bus_name)
        nodes = list(self.dss.Bus.Nodes())
        values = self.dss.Bus.puVmagAngle()
        requested_nodes = [int(part) for part in bus_spec.split(".")[1:] if part.isdigit()]
        magnitudes = []
        angles = []
        for node in requested_nodes or nodes:
            if node in nodes:
                index = nodes.index(node)
                magnitudes.append(float(values[2 * index]))
                angles.append(float(values[2 * index + 1]))
        if not magnitudes:
            return self._bus_average_voltage(bus_name)
        return sum(magnitudes) / len(magnitudes), math.radians(sum(angles) / len(angles))

    def _all_bus_voltage_extrema(self) -> tuple[float, float]:
        magnitudes = []
        for bus_name in self.dss.Circuit.AllBusNames():
            self.dss.Circuit.SetActiveBus(bus_name)
            magnitudes.extend(float(value) for value in self.dss.Bus.puVmagAngle()[0::2])
        if not magnitudes:
            raise RuntimeError("OpenDSS returned no bus voltages")
        return min(magnitudes), max(magnitudes)

    def allocation_summary(self) -> list[TCLLoadAllocation]:
        return list(self.allocations)


@dataclass
class SwingFrequencyState:
    frequency_deviation_hz: float = 0.0


class SwingFrequencyModel:
    """Aggregated swing-equation frequency model.

    The disturbance is positive for generation loss or load increase. Increasing
    IAC load worsens under-frequency; reducing IAC load improves it.
    """

    def __init__(self, params: SwingFrequencyParameters, nominal_iac_kw: float):
        self.params = params
        self.nominal_iac_kw = nominal_iac_kw

    def derivative_hz_per_s(
        self,
        frequency_deviation_hz: float,
        iac_power_kw: float,
        disturbance_kw: float,
    ) -> float:
        p = self.params
        net_power_imbalance_kw = disturbance_kw + (iac_power_kw - self.nominal_iac_kw)
        damping_kw = p.damping_pu_per_hz * p.base_power_kw * frequency_deviation_hz
        return -(net_power_imbalance_kw + damping_kw) / (2.0 * p.inertia_constant_s * p.base_power_kw)

    def step(
        self,
        state: SwingFrequencyState,
        iac_power_kw: float,
        disturbance_kw: float,
        sample_time_s: float,
    ) -> SwingFrequencyState:
        def f(df: float) -> float:
            return self.derivative_hz_per_s(df, iac_power_kw, disturbance_kw)

        k1 = f(state.frequency_deviation_hz)
        k2 = f(state.frequency_deviation_hz + 0.5 * sample_time_s * k1)
        k3 = f(state.frequency_deviation_hz + 0.5 * sample_time_s * k2)
        k4 = f(state.frequency_deviation_hz + sample_time_s * k3)
        next_df = state.frequency_deviation_hz + sample_time_s * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
        return SwingFrequencyState(frequency_deviation_hz=next_df)


def disturbance_kw_for_target_nadir(target_hz: float, settling_time_s: float, base_power_kw: float, damping_pu_per_hz: float) -> float:
    """Convenience helper for repeatable frequency-event magnitudes."""
    _ = settling_time_s
    return abs(target_hz) * damping_pu_per_hz * base_power_kw
