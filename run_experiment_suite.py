"""Run reproducible controller comparisons for IEEE 13-node TCL studies."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import replace
from pathlib import Path
import time

import matplotlib.pyplot as plt
import numpy as np

from iac_dmpc.metrics import summarize_log
from simulate_iac_grid_support import CONTROLLER_MODES, PROJECT_ROOT, SimulationScenario, run_simulation, save_outputs


ORDERED_CONTROLLERS = ["no_support", "rule_based", "dmpc_active_only", "dmpc_pq", "dmpc_full_pq"]


def scenario_set(name: str, monte_carlo_samples: int = 3) -> list[SimulationScenario]:
    base = SimulationScenario()
    if name == "quick":
        return [base]
    if name == "voltage_sweep":
        return [replace(base, name=f"voltage_sag_{value:.2f}", voltage_sag_pu=value) for value in [0.90, 0.92, 0.94]]
    if name == "fleet_sweep":
        return [replace(base, name=f"fleet_{value}", fleet_size=value) for value in [300, 500, 700]]
    if name == "tcl_share_sweep":
        return [
            replace(
                base,
                name=f"tcl_share_{share:02d}pct",
                tcl_share_percent=float(share),
            )
            for share in [10, 20, 30, 40]
        ]
    if name == "tcl_share_weak_bus_sweep":
        return [
            replace(
                base,
                name=f"weak_bus_tcl_share_{share:02d}pct",
                tcl_share_percent=float(share),
                allocation_strategy="weak_bus_prioritized",
            )
            for share in [10, 20, 30, 40]
        ]
    if name == "placement_sweep":
        return scenario_set("tcl_share_sweep") + scenario_set("tcl_share_weak_bus_sweep")
    if name == "feeder_stress":
        return [
            replace(
                base,
                name=f"feeder_stress_{kw:.0f}kw_{kvar:.0f}kvar",
                voltage_event_type="feeder_load_step",
                feeder_extra_load_kw=float(kw),
                feeder_extra_load_kvar=float(kvar),
                voltage_sag_pu=1.0,
            )
            for kw, kvar in [(600, 300), (900, 450), (1200, 600)]
        ]
    if name == "headroom_sweep":
        return [
            replace(
                base,
                name=f"headroom_{scale:.2f}pu",
                inverter_s_rated_kva=4.0 * scale,
                inverter_q_max_kvar=2.2 * scale,
                rms_current_limit_a=18.0 * scale,
                dc_link_current_limit_a=13.0 * scale,
            )
            for scale in [1.0, 1.25, 1.50]
        ]
    if name == "voltage_support_upgrade":
        return scenario_set("feeder_stress") + scenario_set("headroom_sweep")
    if name == "comfort_sweep":
        return [replace(base, name=f"comfort_{value:.1f}C", comfort_band_c=value) for value in [0.5, 1.0, 2.0]]
    if name == "full":
        return (
            scenario_set("voltage_sweep")
            + scenario_set("feeder_stress")
            + scenario_set("headroom_sweep")
            + scenario_set("fleet_sweep")
            + scenario_set("tcl_share_sweep")
            + scenario_set("tcl_share_weak_bus_sweep")
            + scenario_set("comfort_sweep")
        )
    if name == "monte_carlo":
        rng = np.random.default_rng(2026)
        scenarios = []
        for index in range(monte_carlo_samples):
            scenarios.append(
                replace(
                    base,
                    name=f"mc_{index + 1:03d}",
                    random_seed=1000 + index,
                    measurement_delay_steps=int(rng.integers(0, 3)),
                    voltage_noise_std_pu=float(rng.uniform(0.001, 0.006)),
                    frequency_noise_std_hz=float(rng.uniform(0.001, 0.008)),
                    temperature_noise_std_c=float(rng.uniform(0.02, 0.08)),
                    power_noise_std_kw=float(rng.uniform(0.01, 0.04)),
                    thermal_resistance_scale=float(max(0.65, rng.normal(1.0, 0.15))),
                    thermal_capacitance_scale=float(max(0.55, rng.normal(1.0, 0.20))),
                    cop_scale=float(max(0.70, rng.normal(1.0, 0.10))),
                    compressor_time_constant_scale=float(max(0.60, rng.normal(1.0, 0.15))),
                    fleet_size=int(rng.choice([400, 500, 600])),
                    voltage_sag_pu=float(rng.choice([0.90, 0.92, 0.94])),
                )
            )
        return scenarios
    raise ValueError(f"Unknown scenario set {name!r}")


def apply_thermal_calibration(
    scenarios: list[SimulationScenario],
    calibration_json: Path | None,
) -> list[SimulationScenario]:
    if calibration_json is None:
        return scenarios
    with calibration_json.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    thermal = payload["thermal_parameters"]
    source = payload.get("source", calibration_json.as_posix())
    return [
        replace(
            scenario,
            thermal_ambient_temperature_c=float(thermal["ambient_temperature_c"]),
            thermal_setpoint_c=float(thermal["setpoint_c"]),
            thermal_resistance_c_per_kw=float(thermal["resistance_c_per_kw"]),
            thermal_capacitance_kwh_per_c=float(thermal["capacitance_kwh_per_c"]),
            thermal_cop=float(thermal["cop"]),
            thermal_cop_temperature_slope_per_c=float(thermal["cop_temperature_slope_per_c"]),
            thermal_internal_gain_kw=float(thermal["internal_gain_kw"]),
            thermal_calibration_source=str(source),
        )
        for scenario in scenarios
    ]


def run_suite(scenarios: list[SimulationScenario], output_root: Path) -> list[dict[str, str | float]]:
    rows: list[dict[str, str | float]] = []
    traces: dict[tuple[str, str], dict[str, np.ndarray]] = {}
    output_root.mkdir(parents=True, exist_ok=True)
    total_cases = len(scenarios) * len(ORDERED_CONTROLLERS)
    completed = 0
    suite_start = time.perf_counter()

    for scenario in scenarios:
        for controller in ORDERED_CONTROLLERS:
            case_name = f"{scenario.name}__{controller}"
            case_dir = output_root / "cases" / case_name
            case_start = time.perf_counter()
            remaining_before = total_cases - completed
            print(
                f"[{completed + 1}/{total_cases}] Running {case_name} "
                f"(estimated remaining: {format_duration(estimate_remaining(suite_start, completed, remaining_before))})...",
                flush=True,
            )
            log = run_simulation(scenario=scenario, controller_mode=controller)
            save_outputs(log, output_dir=case_dir, stem=case_name)
            metrics = summarize_log(
                log,
                temperature_setpoint_c=scenario.thermal_setpoint_c
                if scenario.thermal_setpoint_c is not None
                else 24.0,
                comfort_band_c=scenario.comfort_band_c,
            )
            rows.append(
                {
                    "scenario": scenario.name,
                    "controller": controller,
                    "fleet_size": scenario.resolved_fleet_size(),
                    "tcl_share_percent": scenario.resolved_tcl_share_percent(),
                    "allocation_strategy": scenario.allocation_strategy,
                    "voltage_event_type": scenario.voltage_event_type,
                    "voltage_sag_pu": scenario.voltage_sag_pu,
                    "feeder_extra_load_kw": scenario.feeder_extra_load_kw,
                    "feeder_extra_load_kvar": scenario.feeder_extra_load_kvar,
                    "inverter_s_rated_kva": scenario.inverter_s_rated_kva
                    if scenario.inverter_s_rated_kva is not None
                    else 4.0,
                    "inverter_q_max_kvar": scenario.inverter_q_max_kvar
                    if scenario.inverter_q_max_kvar is not None
                    else 2.2,
                    "comfort_band_c": scenario.comfort_band_c,
                    "thermal_calibration_source": scenario.thermal_calibration_source or "nominal",
                    "thermal_ambient_temperature_c": scenario.thermal_ambient_temperature_c
                    if scenario.thermal_ambient_temperature_c is not None
                    else 32.0,
                    "thermal_setpoint_c": scenario.thermal_setpoint_c if scenario.thermal_setpoint_c is not None else 24.0,
                    "thermal_resistance_c_per_kw": scenario.thermal_resistance_c_per_kw
                    if scenario.thermal_resistance_c_per_kw is not None
                    else (4.0 / 3.0) * scenario.thermal_resistance_scale,
                    "thermal_capacitance_kwh_per_c": scenario.thermal_capacitance_kwh_per_c
                    if scenario.thermal_capacitance_kwh_per_c is not None
                    else 2.5 * scenario.thermal_capacitance_scale,
                    "thermal_cop": scenario.thermal_cop if scenario.thermal_cop is not None else 3.0 * scenario.cop_scale,
                    "measurement_delay_steps": scenario.measurement_delay_steps,
                    **metrics,
                }
            )
            traces[(scenario.name, controller)] = log
            completed += 1
            elapsed_case = time.perf_counter() - case_start
            remaining_after = total_cases - completed
            print(
                f"    completed in {format_duration(elapsed_case)}; "
                f"estimated remaining: {format_duration(estimate_remaining(suite_start, completed, remaining_after))}",
                flush=True,
            )

    write_metrics(output_root / "metrics_summary.csv", rows)
    make_comparison_plots(output_root / "figures", traces)
    return rows


def write_metrics(path: Path, rows: list[dict[str, str | float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def estimate_remaining(suite_start: float, completed: int, remaining: int) -> float:
    if completed <= 0:
        return float(remaining * 20.0)
    elapsed = time.perf_counter() - suite_start
    return (elapsed / completed) * remaining


def format_duration(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    minutes, sec = divmod(int(round(seconds)), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes}m {sec}s"
    if minutes:
        return f"{minutes}m {sec}s"
    return f"{sec}s"


def make_comparison_plots(output_dir: Path, traces: dict[tuple[str, str], dict[str, np.ndarray]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    scenarios = sorted({scenario for scenario, _controller in traces})
    for scenario in scenarios:
        fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
        for controller in ORDERED_CONTROLLERS:
            log = traces.get((scenario, controller))
            if log is None:
                continue
            time_min = log["time_s"] / 60.0
            axes[0].plot(time_min, log["frequency_hz"], label=controller)
            voltage_trace = log.get("min_feeder_voltage_pu", log["pcc_voltage_pu"])
            axes[1].plot(time_min, voltage_trace, label=controller)
            axes[2].plot(time_min, log["temperature_c"], label=controller)

        axes[0].axhline(60.0, color="k", linestyle="--", linewidth=0.8)
        axes[0].set_ylabel("Frequency (Hz)")
        axes[1].set_ylabel("Worst feeder voltage (pu)")
        axes[2].set_ylabel("Temperature (deg C)")
        axes[2].set_xlabel("Time (min)")
        axes[2].axhline(23.0, color="k", linestyle="--", linewidth=0.8)
        axes[2].axhline(25.0, color="k", linestyle="--", linewidth=0.8)
        axes[0].legend(loc="best", ncol=2)
        fig.tight_layout()
        fig.savefig(output_dir / f"{scenario}_comparison.png", dpi=160)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scenario-set",
        choices=[
            "quick",
            "voltage_sweep",
            "fleet_sweep",
            "tcl_share_sweep",
            "tcl_share_weak_bus_sweep",
            "placement_sweep",
            "feeder_stress",
            "headroom_sweep",
            "voltage_support_upgrade",
            "comfort_sweep",
            "full",
            "monte_carlo",
        ],
        default="quick",
    )
    parser.add_argument("--monte-carlo-samples", type=int, default=3)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "experiments")
    parser.add_argument(
        "--thermal-calibration-json",
        type=Path,
        default=None,
        help="Apply calibrated thermal parameters from calibrate_from_resstock.py to every scenario.",
    )
    args = parser.parse_args()

    _ = CONTROLLER_MODES
    scenarios = scenario_set(args.scenario_set, monte_carlo_samples=args.monte_carlo_samples)
    scenarios = apply_thermal_calibration(scenarios, args.thermal_calibration_json)
    output_dir = args.output_dir.resolve()
    rows = run_suite(scenarios, output_dir / args.scenario_set)
    print(f"Wrote {len(rows)} cases to {output_dir / args.scenario_set}")


if __name__ == "__main__":
    main()
