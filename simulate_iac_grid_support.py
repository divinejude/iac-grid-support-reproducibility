"""Run a simultaneous Frequency-Watt and Volt-Var IAC support simulation."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import time
from typing import ClassVar

import matplotlib.pyplot as plt
import numpy as np

from iac_dmpc.capability import InverterCapability
from iac_dmpc.compressor import CompressorCommandState, CompressorConstraintModel
from iac_dmpc.dmpc import DMPCController, PQDMPCController
from iac_dmpc.feeder import IEEE13OpenDSSFleetFeeder
from iac_dmpc.grid_cosim import CsvReplayDynamicBackend, DynamicGridCosimulator, SwingEquationDynamicBackend
from iac_dmpc.inverter import InverterInnerLoop, InverterInnerLoopState
from iac_dmpc.parameters import (
    CompressorParameters,
    DMPCParameters,
    FrequencyPlantParameters,
    GridParameters,
    InverterInnerLoopParameters,
    InverterParameters,
    MeasurementParameters,
    SwingFrequencyParameters,
    TCLFleetParameters,
    ThermalParameters,
)
from iac_dmpc.plant import IACPlant, PlantState
from iac_dmpc.rule_based import FrequencyWattFiveRegion, VoltVarDroop
from iac_dmpc.signal_processing import SinglePhaseCurrentDecoupler
from iac_dmpc.uncertainty import Measurement, MeasurementChannel


PROJECT_ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class SimulationScenario:
    ieee13_original_load_kw: ClassVar[float] = 3466.0

    name: str = "impact_ieee13_fleet"
    total_time_s: float = 600.0
    frequency_event_start_s: float = 120.0
    frequency_event_end_s: float = 420.0
    frequency_disturbance_kw: float = 1200.0
    voltage_sag_start_s: float = 180.0
    voltage_sag_end_s: float = 380.0
    voltage_sag_pu: float = 0.92
    voltage_recovery_end_s: float = 460.0
    voltage_recovery_pu: float = 0.985
    voltage_event_type: str = "source_sag"
    feeder_extra_load_kw: float = 0.0
    feeder_extra_load_kvar: float = 0.0
    fleet_size: int = 500
    tcl_share_percent: float | None = None
    allocation_strategy: str = "load_proportional"
    weak_bus_exponent: float = 2.0
    inverter_s_rated_kva: float | None = None
    inverter_q_max_kvar: float | None = None
    rms_current_limit_a: float | None = None
    dc_link_current_limit_a: float | None = None
    comfort_band_c: float = 1.0
    swing_base_power_kw: float = 5000.0
    frequency_backend: str = "csv_replay"
    frequency_replay_csv: str | None = "data/dynamic_traces/lfc_replay_trace.csv"
    random_seed: int = 7
    measurement_delay_steps: int = 0
    voltage_noise_std_pu: float = 0.002
    frequency_noise_std_hz: float = 0.003
    temperature_noise_std_c: float = 0.03
    power_noise_std_kw: float = 0.015
    thermal_resistance_scale: float = 1.0
    thermal_capacitance_scale: float = 1.0
    cop_scale: float = 1.0
    thermal_ambient_temperature_c: float | None = None
    thermal_setpoint_c: float | None = None
    thermal_resistance_c_per_kw: float | None = None
    thermal_capacitance_kwh_per_c: float | None = None
    thermal_cop: float | None = None
    thermal_cop_temperature_slope_per_c: float | None = None
    thermal_internal_gain_kw: float | None = None
    thermal_calibration_source: str | None = None
    compressor_time_constant_scale: float = 1.0

    def resolved_fleet_size(self, p_set_kw: float = 2.0) -> int:
        if self.tcl_share_percent is None:
            return self.fleet_size
        aggregate_kw = self.ieee13_original_load_kw * self.tcl_share_percent / 100.0
        return max(1, int(round(aggregate_kw / p_set_kw)))

    def resolved_tcl_share_percent(self, p_set_kw: float = 2.0) -> float:
        if self.tcl_share_percent is not None:
            return self.tcl_share_percent
        return 100.0 * self.fleet_size * p_set_kw / self.ieee13_original_load_kw


DEFAULT_SCENARIO = SimulationScenario()
CONTROLLER_MODES = ("no_support", "rule_based", "dmpc_active_only", "dmpc_pq", "dmpc_full_pq")


def grid_frequency_disturbance_kw(time_s: float, scenario: SimulationScenario = DEFAULT_SCENARIO) -> float:
    """Positive disturbance means generation loss or load increase."""
    return (
        scenario.frequency_disturbance_kw
        if scenario.frequency_event_start_s <= time_s <= scenario.frequency_event_end_s
        else 0.0
    )


def source_voltage_profile_pu(time_s: float, scenario: SimulationScenario = DEFAULT_SCENARIO) -> float:
    if scenario.voltage_event_type == "feeder_load_step":
        return 1.0
    if scenario.voltage_sag_start_s <= time_s <= scenario.voltage_sag_end_s:
        return scenario.voltage_sag_pu
    if scenario.voltage_sag_end_s < time_s <= scenario.voltage_recovery_end_s:
        return scenario.voltage_recovery_pu
    return 1.0


def feeder_extra_load_profile(
    time_s: float,
    scenario: SimulationScenario = DEFAULT_SCENARIO,
) -> tuple[float, float]:
    if scenario.voltage_event_type != "feeder_load_step":
        return 0.0, 0.0
    if scenario.voltage_sag_start_s <= time_s <= scenario.voltage_sag_end_s:
        return scenario.feeder_extra_load_kw, scenario.feeder_extra_load_kvar
    if scenario.voltage_sag_end_s < time_s <= scenario.voltage_recovery_end_s:
        return 0.5 * scenario.feeder_extra_load_kw, 0.5 * scenario.feeder_extra_load_kvar
    return 0.0, 0.0


def estimate_id_iq_from_single_phase_current(
    decoupler: SinglePhaseCurrentDecoupler,
    p_kw: float,
    q_kvar: float,
    voltage_rms: float,
    frequency_hz: float,
    sample_time_s: float,
    start_phase_rad: float,
    duration_s: float,
) -> tuple[float, float, float]:
    """Synthesize one single-phase current and recover dq components."""
    omega = 2.0 * math.pi * frequency_hz
    samples = max(int(duration_s / sample_time_s), 1)
    theta = start_phase_rad
    dq = None
    for _ in range(samples):
        voltage = math.sqrt(2.0) * voltage_rms * math.cos(theta)
        i_d = p_kw * 1000.0 / voltage_rms
        i_q = q_kvar * 1000.0 / voltage_rms
        current = math.sqrt(2.0) * (i_d * math.cos(theta) - i_q * math.sin(theta))
        dq, _, omega_hat = decoupler.update(current, voltage)
        theta = (theta + omega * sample_time_s) % (2.0 * math.pi)
    assert dq is not None
    return dq.d, dq.q, theta


def run_simulation(
    scenario: SimulationScenario = DEFAULT_SCENARIO,
    controller_mode: str = "dmpc_pq",
) -> dict[str, np.ndarray]:
    if controller_mode not in CONTROLLER_MODES:
        raise ValueError(f"Unknown controller mode {controller_mode!r}. Valid modes: {list(CONTROLLER_MODES)}")

    grid = GridParameters()
    swing_params = SwingFrequencyParameters(base_power_kw=scenario.swing_base_power_kw)
    inverter_nominal = InverterParameters()
    inverter = InverterParameters(
        s_rated_kva=inverter_nominal.s_rated_kva
        if scenario.inverter_s_rated_kva is None
        else scenario.inverter_s_rated_kva,
        p_min_kw=inverter_nominal.p_min_kw,
        p_set_kw=inverter_nominal.p_set_kw,
        p_max_kw=inverter_nominal.p_max_kw,
        q_max_kvar=inverter_nominal.q_max_kvar if scenario.inverter_q_max_kvar is None else scenario.inverter_q_max_kvar,
    )
    fleet = TCLFleetParameters(unit_count=scenario.resolved_fleet_size(inverter.p_set_kw))
    inner_loop_nominal = InverterInnerLoopParameters()
    inner_loop_params = InverterInnerLoopParameters(
        current_controller_bandwidth_hz=inner_loop_nominal.current_controller_bandwidth_hz,
        dc_link_voltage_v=inner_loop_nominal.dc_link_voltage_v,
        dc_link_current_limit_a=inner_loop_nominal.dc_link_current_limit_a
        if scenario.dc_link_current_limit_a is None
        else scenario.dc_link_current_limit_a,
        rms_current_limit_a=inner_loop_nominal.rms_current_limit_a
        if scenario.rms_current_limit_a is None
        else scenario.rms_current_limit_a,
        anti_windup_gain=inner_loop_nominal.anti_windup_gain,
    )
    thermal_nominal = ThermalParameters(comfort_band_c=scenario.comfort_band_c)
    thermal = ThermalParameters(
        ambient_temperature_c=thermal_nominal.ambient_temperature_c
        if scenario.thermal_ambient_temperature_c is None
        else scenario.thermal_ambient_temperature_c,
        setpoint_c=thermal_nominal.setpoint_c if scenario.thermal_setpoint_c is None else scenario.thermal_setpoint_c,
        comfort_band_c=thermal_nominal.comfort_band_c,
        resistance_c_per_kw=thermal_nominal.resistance_c_per_kw * scenario.thermal_resistance_scale
        if scenario.thermal_resistance_c_per_kw is None
        else scenario.thermal_resistance_c_per_kw,
        capacitance_kwh_per_c=thermal_nominal.capacitance_kwh_per_c * scenario.thermal_capacitance_scale
        if scenario.thermal_capacitance_kwh_per_c is None
        else scenario.thermal_capacitance_kwh_per_c,
        cop=thermal_nominal.cop * scenario.cop_scale if scenario.thermal_cop is None else scenario.thermal_cop,
        cop_temperature_slope_per_c=thermal_nominal.cop_temperature_slope_per_c
        if scenario.thermal_cop_temperature_slope_per_c is None
        else scenario.thermal_cop_temperature_slope_per_c,
        internal_gain_kw=thermal_nominal.internal_gain_kw
        if scenario.thermal_internal_gain_kw is None
        else scenario.thermal_internal_gain_kw,
    )
    compressor_nominal = CompressorParameters()
    compressor = CompressorParameters(
        time_constant_s=compressor_nominal.time_constant_s * scenario.compressor_time_constant_scale,
        frequency_watt_gain_kw_per_hz=compressor_nominal.frequency_watt_gain_kw_per_hz,
        bias_kw=compressor_nominal.bias_kw,
        ramp_rate_kw_per_s=compressor_nominal.ramp_rate_kw_per_s,
        min_speed_hz=compressor_nominal.min_speed_hz,
        rated_speed_hz=compressor_nominal.rated_speed_hz,
        max_speed_hz=compressor_nominal.max_speed_hz,
        minimum_dwell_time_s=compressor_nominal.minimum_dwell_time_s,
    )
    frequency = FrequencyPlantParameters()
    mpc = DMPCParameters()
    measurement_params = MeasurementParameters(
        random_seed=scenario.random_seed,
        delay_steps=scenario.measurement_delay_steps,
        voltage_noise_std_pu=scenario.voltage_noise_std_pu,
        frequency_noise_std_hz=scenario.frequency_noise_std_hz,
        temperature_noise_std_c=scenario.temperature_noise_std_c,
        power_noise_std_kw=scenario.power_noise_std_kw,
    )

    plant = IACPlant(thermal, compressor, inverter, frequency)
    dmpc = DMPCController(plant, inverter, thermal, mpc)
    pq_dmpc = PQDMPCController(plant, inverter, thermal, mpc)
    capability = InverterCapability(inverter.s_rated_kva, inverter.p_min_kw, inverter.p_max_kw, inverter.q_max_kvar)
    feeder = IEEE13OpenDSSFleetFeeder(
        allocation_strategy=scenario.allocation_strategy,
        weak_bus_exponent=scenario.weak_bus_exponent,
    )
    if scenario.frequency_backend == "dynamic_swing":
        frequency_backend = SwingEquationDynamicBackend(swing_params)
    elif scenario.frequency_backend == "csv_replay":
        if scenario.frequency_replay_csv is None:
            raise ValueError("frequency_replay_csv must be provided when frequency_backend='csv_replay'")
        frequency_backend = CsvReplayDynamicBackend(PROJECT_ROOT / scenario.frequency_replay_csv)
    else:
        raise ValueError(
            "Unknown frequency backend "
            f"{scenario.frequency_backend!r}. Expected 'dynamic_swing' or 'csv_replay'."
        )
    grid_cosim = DynamicGridCosimulator(frequency_backend, nominal_iac_kw=inverter.p_set_kw * fleet.unit_count)
    inverter_loop = InverterInnerLoop(inverter, inner_loop_params)
    compressor_constraints = CompressorConstraintModel(compressor, inverter, thermal)
    measurement_channel = MeasurementChannel(measurement_params)
    volt_var = VoltVarDroop(q_max_kvar=inverter.q_max_kvar)
    freq_watt = FrequencyWattFiveRegion(
        p_min_kw=inverter.p_min_kw,
        p_set_kw=inverter.p_set_kw,
        p_max_kw=inverter.p_max_kw,
        t_min_c=thermal.min_temperature_c,
        t_set_c=thermal.setpoint_c,
        t_max_c=thermal.max_temperature_c,
    )
    decoupler = SinglePhaseCurrentDecoupler(
        grid.signal_sample_time_s,
        grid.nominal_frequency_hz,
        grid.sogi_gain,
        grid.pll_kp,
        grid.pll_ki,
    )

    steps = int(scenario.total_time_s / mpc.sample_time_s) + 1
    state = PlantState(0.0, thermal.setpoint_c, inverter.p_set_kw)
    inverter_state = InverterInnerLoopState(
        id_a=inverter.p_set_kw * 1000.0 / grid.nominal_voltage_rms,
        iq_a=0.0,
    )
    compressor_command_state = CompressorCommandState(last_power_command_kw=inverter.p_set_kw)
    previous_control = inverter.p_set_kw
    signal_phase = 0.0
    previous_actual_p_kw = inverter.p_set_kw
    previous_actual_q_kvar = 0.0
    previous_voltage_pu = 1.0

    log = {
        "time_s": np.zeros(steps),
        "controller_mode": np.zeros(steps),
        "frequency_hz": np.zeros(steps),
        "frequency_deviation_hz": np.zeros(steps),
        "source_voltage_pu": np.zeros(steps),
        "pcc_voltage_pu": np.zeros(steps),
        "min_feeder_voltage_pu": np.zeros(steps),
        "max_feeder_voltage_pu": np.zeros(steps),
        "measured_voltage_pu": np.zeros(steps),
        "p_command_kw": np.zeros(steps),
        "p_feasible_kw": np.zeros(steps),
        "p_rule_kw": np.zeros(steps),
        "p_actual_kw": np.zeros(steps),
        "aggregate_p_kw": np.zeros(steps),
        "q_command_kvar": np.zeros(steps),
        "q_actual_kvar": np.zeros(steps),
        "aggregate_q_kvar": np.zeros(steps),
        "apparent_kva": np.zeros(steps),
        "temperature_c": np.zeros(steps),
        "measured_temperature_c": np.zeros(steps),
        "compressor_speed_hz": np.zeros(steps),
        "cop": np.zeros(steps),
        "id_a": np.zeros(steps),
        "iq_a": np.zeros(steps),
        "id_ref_a": np.zeros(steps),
        "iq_ref_a": np.zeros(steps),
        "current_saturated": np.zeros(steps),
        "ramp_limited": np.zeros(steps),
        "dmpc_status_ok": np.zeros(steps),
        "controller_compute_time_s": np.zeros(steps),
        "feeder_extra_load_kw": np.zeros(steps),
        "feeder_extra_load_kvar": np.zeros(steps),
    }

    for k in range(steps):
        time_s = k * mpc.sample_time_s
        source_v_pu = source_voltage_profile_pu(time_s, scenario)
        feeder_extra_kw, feeder_extra_kvar = feeder_extra_load_profile(time_s, scenario)
        pre_control_feeder = feeder.solve_pcc_voltage(
            source_v_pu,
            previous_actual_p_kw * fleet.unit_count,
            previous_actual_q_kvar * fleet.unit_count,
            feeder_extra_kw,
            feeder_extra_kvar,
        )
        true_measurement = Measurement(
            voltage_pu=pre_control_feeder.pcc_voltage_pu,
            frequency_deviation_hz=grid_cosim.frequency_deviation_hz,
            temperature_c=state.indoor_temperature_c,
            power_kw=state.compressor_power_kw,
        )
        measured = measurement_channel.sample(true_measurement)
        q_volt_var_local = volt_var.q_command(pre_control_feeder.pcc_voltage_pu)
        q_volt_var_network = volt_var.q_command(pre_control_feeder.min_voltage_pu)
        q_volt_var = q_volt_var_network if controller_mode == "dmpc_full_pq" else q_volt_var_local
        if controller_mode in {"no_support", "dmpc_active_only"}:
            q_request = 0.0
        else:
            q_request = q_volt_var

        disturbance_kw = grid_frequency_disturbance_kw(time_s, scenario)
        d_forecast = np.array(
            [
                -grid_frequency_disturbance_kw(time_s + j * mpc.sample_time_s, scenario)
                / (2.0 * swing_params.inertia_constant_s * swing_params.base_power_kw)
                for j in range(mpc.horizon_steps)
            ]
        )
        q_forecast = np.full(mpc.horizon_steps, q_request)

        measured_state = PlantState(
            measured.frequency_deviation_hz,
            measured.temperature_c,
            measured.power_kw,
        )
        compute_start_s = time.perf_counter()
        p_rule = freq_watt.p_command(measured.frequency_deviation_hz, measured.temperature_c)
        if controller_mode == "no_support":
            p_target = inverter.p_set_kw
            dmpc_status_ok = True
        elif controller_mode == "rule_based":
            p_target = p_rule
            dmpc_status_ok = True
        elif controller_mode == "dmpc_full_pq":
            sensitivity = feeder.estimate_voltage_sensitivity(
                source_v_pu,
                previous_actual_p_kw * fleet.unit_count,
                previous_actual_q_kvar * fleet.unit_count,
                perturb_kw=max(25.0, 0.05 * fleet.unit_count),
                perturb_kvar=max(25.0, 0.05 * fleet.unit_count),
                metric="min",
                extra_load_kw=feeder_extra_kw,
                extra_load_kvar=feeder_extra_kvar,
            )
            result = pq_dmpc.solve_pq(
                measured_state,
                previous_control,
                previous_actual_q_kvar,
                d_forecast,
                sensitivity.base_voltage_pu,
                sensitivity.dv_dp_pu_per_kw,
                sensitivity.dv_dq_pu_per_kvar,
                previous_actual_p_kw,
                previous_actual_q_kvar,
                fleet.unit_count,
                q_floor_kvar=max(q_volt_var, 0.0),
            )
            p_target = result.p_command_kw
            q_request = result.q_command_kvar
            dmpc_status_ok = result.status in {"optimal", "optimal_inaccurate"}
        else:
            result = dmpc.solve(measured_state, previous_control, q_forecast, d_forecast)
            p_target = result.p_command_kw
            dmpc_status_ok = result.status in {"optimal", "optimal_inaccurate"}
        compute_time_s = time.perf_counter() - compute_start_s

        allocated = capability.allocate(p_target, q_request, reactive_priority=True)
        compressor_op, compressor_command_state = compressor_constraints.apply(
            compressor_command_state,
            allocated.p_kw,
            state.indoor_temperature_c,
            thermal.ambient_temperature_c,
            mpc.sample_time_s,
            time_s,
        )
        inverter_result = inverter_loop.step(
            inverter_state,
            compressor_op.power_kw,
            allocated.q_kvar,
            grid.nominal_voltage_rms * max(previous_voltage_pu, 0.2),
            mpc.sample_time_s,
        )
        inverter_state = inverter_result.state
        feeder_solution = feeder.solve_pcc_voltage(
            source_v_pu,
            inverter_result.p_kw * fleet.unit_count,
            inverter_result.q_kvar * fleet.unit_count,
            feeder_extra_kw,
            feeder_extra_kvar,
        )
        previous_voltage_pu = feeder_solution.pcc_voltage_pu

        measured_id, measured_iq, signal_phase = estimate_id_iq_from_single_phase_current(
            decoupler,
            inverter_result.p_kw,
            inverter_result.q_kvar,
            grid.nominal_voltage_rms * feeder_solution.pcc_voltage_pu,
            grid.nominal_frequency_hz + grid_cosim.frequency_deviation_hz,
            grid.signal_sample_time_s,
            signal_phase,
            mpc.sample_time_s,
        )

        log["time_s"][k] = time_s
        log["controller_mode"][k] = float(CONTROLLER_MODES.index(controller_mode))
        log["frequency_deviation_hz"][k] = grid_cosim.frequency_deviation_hz
        log["frequency_hz"][k] = grid.nominal_frequency_hz + grid_cosim.frequency_deviation_hz
        log["source_voltage_pu"][k] = source_v_pu
        log["pcc_voltage_pu"][k] = feeder_solution.pcc_voltage_pu
        log["min_feeder_voltage_pu"][k] = feeder_solution.min_voltage_pu
        log["max_feeder_voltage_pu"][k] = feeder_solution.max_voltage_pu
        log["measured_voltage_pu"][k] = measured.voltage_pu
        log["p_command_kw"][k] = allocated.p_kw
        log["p_feasible_kw"][k] = compressor_op.power_kw
        log["p_rule_kw"][k] = p_rule
        log["p_actual_kw"][k] = inverter_result.p_kw
        log["aggregate_p_kw"][k] = inverter_result.p_kw * fleet.unit_count
        log["q_command_kvar"][k] = allocated.q_kvar
        log["q_actual_kvar"][k] = inverter_result.q_kvar
        log["aggregate_q_kvar"][k] = inverter_result.q_kvar * fleet.unit_count
        log["apparent_kva"][k] = math.hypot(inverter_result.p_kw, inverter_result.q_kvar)
        log["temperature_c"][k] = state.indoor_temperature_c
        log["measured_temperature_c"][k] = measured.temperature_c
        log["compressor_speed_hz"][k] = compressor_op.speed_hz
        log["cop"][k] = compressor_op.cop
        log["id_a"][k] = measured_id
        log["iq_a"][k] = measured_iq
        log["id_ref_a"][k] = inverter_result.id_ref_a
        log["iq_ref_a"][k] = inverter_result.iq_ref_a
        log["current_saturated"][k] = 1.0 if inverter_result.saturated else 0.0
        log["ramp_limited"][k] = 1.0 if compressor_op.ramp_limited else 0.0
        log["dmpc_status_ok"][k] = 1.0 if dmpc_status_ok else 0.0
        log["controller_compute_time_s"][k] = compute_time_s
        log["feeder_extra_load_kw"][k] = feeder_extra_kw
        log["feeder_extra_load_kvar"][k] = feeder_extra_kvar

        if k < steps - 1:
            frequency_output = grid_cosim.step(
                time_s,
                inverter_result.p_kw * fleet.unit_count,
                disturbance_kw,
                mpc.sample_time_s,
            )
            state = plant.step_thermal_compressor(
                state,
                compressor_op.power_kw,
                compressor_op.cop * inverter_result.p_kw,
                frequency_output.frequency_deviation_hz,
                mpc.sample_time_s,
                ambient_c=thermal.ambient_temperature_c,
            )
            previous_control = compressor_op.power_kw
            previous_actual_p_kw = inverter_result.p_kw
            previous_actual_q_kvar = inverter_result.q_kvar

    return log


def save_outputs(
    log: dict[str, np.ndarray],
    output_dir: Path | None = None,
    stem: str = "iac_dmpc_demo",
) -> Path:
    output_dir = PROJECT_ROOT / "outputs" if output_dir is None else output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{stem}.csv"
    png_path = output_dir / f"{stem}.png"

    header = ",".join(log.keys())
    data = np.column_stack([log[key] for key in log.keys()])
    np.savetxt(csv_path, data, delimiter=",", header=header, comments="")

    time_min = log["time_s"] / 60.0
    fig, axes = plt.subplots(4, 1, figsize=(11, 10), sharex=True)
    axes[0].plot(time_min, log["frequency_hz"], label="Grid frequency")
    axes[0].axhline(60.0, color="k", linewidth=0.8, linestyle="--")
    axes[0].set_ylabel("Hz")
    axes[0].legend(loc="best")

    axes[1].plot(time_min, log["p_command_kw"], label="MPC active command")
    axes[1].plot(time_min, log["p_feasible_kw"], label="Ramp/speed feasible P")
    axes[1].plot(time_min, log["p_actual_kw"], label="Compressor active power", linestyle="--")
    axes[1].plot(time_min, log["q_actual_kvar"], label="Reactive power")
    axes[1].plot(time_min, log["apparent_kva"], label="Apparent power")
    axes[1].set_ylabel("kW / kvar / kVA")
    axes[1].legend(loc="best", ncol=2)

    axes[2].plot(time_min, log["temperature_c"], label="Indoor temperature")
    axes[2].axhline(23.0, color="k", linewidth=0.8, linestyle="--")
    axes[2].axhline(25.0, color="k", linewidth=0.8, linestyle="--")
    axes[2].set_ylabel("deg C")
    axes[2].legend(loc="best")

    axes[3].plot(time_min, log["source_voltage_pu"], label="Source voltage")
    axes[3].plot(time_min, log["pcc_voltage_pu"], label="PCC voltage")
    if "min_feeder_voltage_pu" in log:
        axes[3].plot(time_min, log["min_feeder_voltage_pu"], label="Worst feeder voltage")
    axes[3].plot(time_min, log["id_a"], label="Estimated Id")
    axes[3].plot(time_min, log["iq_a"], label="Estimated Iq")
    axes[3].set_ylabel("pu / A")
    axes[3].set_xlabel("Time (min)")
    axes[3].legend(loc="best", ncol=3)

    fig.tight_layout()
    fig.savefig(png_path, dpi=160)
    plt.close(fig)
    return png_path


if __name__ == "__main__":
    simulation_log = run_simulation()
    figure = save_outputs(simulation_log)
    print(f"Saved simulation results to {figure}")
    print(
        "Summary: "
        f"min frequency={simulation_log['frequency_hz'].min():.3f} Hz, "
        f"temperature range=({simulation_log['temperature_c'].min():.3f}, "
        f"{simulation_log['temperature_c'].max():.3f}) deg C, "
        f"max apparent power={simulation_log['apparent_kva'].max():.3f} kVA"
    )
