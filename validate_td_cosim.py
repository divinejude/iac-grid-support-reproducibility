"""Explicit loose-coupled T&D dynamic co-simulation check using ANDES and OpenDSS.

The check advances the ANDES IEEE 14-bus nonlinear transient simulation in
short macro-steps. At every coupling step, the current ANDES boundary-bus
voltage is sent to the OpenDSS IEEE 13-node feeder; OpenDSS solves the
distribution boundary load with the IAC P/Q trajectory; the resulting feeder
active-power change is returned to ANDES as TGOV1 auxiliary active-power input.

This is explicit loose coupling. It is intended as a reproducible validation
check, not a production-grade T&D co-simulation engine.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from iac_dmpc.plot_style import apply_elsevier_figure_style

from iac_dmpc.feeder import IEEE13OpenDSSFleetFeeder
from validate_andes_transient import CONTROLLER_LABELS, read_controller_trace


ROOT = Path(__file__).resolve().parent
DEFAULT_EXPERIMENT_ROOT = (
    ROOT / "experiments" / "nist_measured_calibrated" / "full"
    if (ROOT / "experiments" / "nist_measured_calibrated" / "full").exists()
    else ROOT / "experiments" / "dynamic_replay_calibrated" / "full"
)
DEFAULT_OUTPUT_DIR = ROOT / "experiments" / "td_cosim"
DEFAULT_PAPER_FIGURE_DIR = ROOT / "paper" / "figures"
DEFAULT_PAPER_TABLE_DIR = ROOT / "paper" / "tables"
CONTROLLERS = ("no_support", "dmpc_active_only", "dmpc_full_pq")


@dataclass(frozen=True)
class TDCosimConfig:
    scenario: str = "voltage_sag_0.92"
    andes_case: str = "ieee14/ieee14_alter.xlsx"
    interface_bus_idx: int = 3
    governor_device: str = "TGOV1_1"
    controller_trace_start_s: float = 175.0
    duration_s: float = 25.0
    coupling_step_s: float = 0.1
    andes_inner_step_s: float = 0.05
    disturbance_time_s: float = 5.0
    disturbance_pu: float = 0.015
    transmission_base_kw: float = 100_000.0
    nominal_frequency_hz: float = 60.0
    nominal_iac_kw: float = 1000.0


def held_value(trace: pd.DataFrame, column: str, time_s: float) -> float:
    times = trace["time_s"].to_numpy(dtype=float)
    values = trace[column].to_numpy(dtype=float)
    idx = int(np.searchsorted(times, time_s, side="right") - 1)
    idx = max(0, min(idx, len(values) - 1))
    return float(values[idx])


def feeder_boundary_power_kw_kvar(feeder: IEEE13OpenDSSFleetFeeder) -> tuple[float, float]:
    kw, kvar = feeder.dss.Circuit.TotalPower()
    return -float(kw), -float(kvar)


def center_of_inertia_frequency_hz(system, inertia_weights: np.ndarray, nominal_frequency_hz: float) -> float:
    omega = np.asarray(system.GENROU.omega.v, dtype=float)
    if omega.size == 0:
        return nominal_frequency_hz
    weights = inertia_weights[: len(omega)]
    if float(weights.sum()) <= 0.0:
        weights = np.ones_like(omega, dtype=float)
    weights = weights / weights.sum()
    return float(nominal_frequency_hz * np.average(omega, weights=weights))


def prepare_andes_system(cfg: TDCosimConfig):
    import andes  # type: ignore

    system = andes.load(andes.get_case(cfg.andes_case), setup=True, no_output=True, default_config=True)
    if system is None:
        raise RuntimeError("ANDES failed to load the IEEE 14-bus dynamic case.")
    if hasattr(system, "Alter") and system.Alter.n:
        system.Alter.u.v[:] = 0
    system.PFlow.run()
    system.TDS.config.tstep = cfg.andes_inner_step_s
    system.TDS.config.tf = 0.0
    gen_m = np.asarray(system.GENROU.M.v, dtype=float)
    gen_m = np.where(gen_m > 0.0, gen_m, 1.0)
    return system, gen_m


def run_one_controller(
    controller: str,
    trace: pd.DataFrame,
    cfg: TDCosimConfig,
) -> pd.DataFrame:
    system, inertia_weights = prepare_andes_system(cfg)
    feeder = IEEE13OpenDSSFleetFeeder()
    nominal_interface_v = float(system.Bus.v.v[system.Bus.idx.v.index(cfg.interface_bus_idx)])

    first_time = cfg.controller_trace_start_s
    first_source = held_value(trace, "source_voltage_pu", first_time)
    first_extra_kw = held_value(trace, "feeder_extra_load_kw", first_time)
    first_extra_kvar = held_value(trace, "feeder_extra_load_kvar", first_time)
    feeder.solve_pcc_voltage(first_source, cfg.nominal_iac_kw, 0.0, first_extra_kw, first_extra_kvar)
    nominal_boundary_kw, nominal_boundary_kvar = feeder_boundary_power_kw_kvar(feeder)

    rows = []
    steps = int(round(cfg.duration_s / cfg.coupling_step_s))
    started = time.time()
    for k in range(steps + 1):
        local_t = k * cfg.coupling_step_s
        source_t = cfg.controller_trace_start_s + local_t
        if k and k % max(1, int(5.0 / cfg.coupling_step_s)) == 0:
            elapsed = time.time() - started
            remaining = elapsed / k * (steps - k)
            print(f"  {controller}: t={local_t:.1f}s, estimated remaining {remaining:.1f}s", flush=True)

        bus_uid = system.Bus.idx.v.index(cfg.interface_bus_idx)
        boundary_v_pu = float(system.Bus.v.v[bus_uid])
        boundary_angle_rad = float(system.Bus.a.v[bus_uid])
        source_voltage_command = held_value(trace, "source_voltage_pu", source_t)
        feeder_source_pu = np.clip(boundary_v_pu / nominal_interface_v * source_voltage_command, 0.70, 1.15)

        aggregate_p_kw = held_value(trace, "aggregate_p_kw", source_t)
        aggregate_q_kvar = held_value(trace, "aggregate_q_kvar", source_t)
        extra_kw = held_value(trace, "feeder_extra_load_kw", source_t)
        extra_kvar = held_value(trace, "feeder_extra_load_kvar", source_t)
        feeder_solution = feeder.solve_pcc_voltage(
            feeder_source_pu,
            aggregate_p_kw,
            aggregate_q_kvar,
            extra_kw,
            extra_kvar,
        )
        boundary_kw, boundary_kvar = feeder_boundary_power_kw_kvar(feeder)
        feeder_relief_pu = (nominal_boundary_kw - boundary_kw) / cfg.transmission_base_kw
        disturbance_pu = cfg.disturbance_pu if local_t >= cfg.disturbance_time_s else 0.0
        paux_pu = feeder_relief_pu - disturbance_pu
        system.TGOV1.set(src="paux0", idx=cfg.governor_device, attr="v", value=paux_pu)

        frequency_hz = center_of_inertia_frequency_hz(system, inertia_weights, cfg.nominal_frequency_hz)
        rows.append(
            {
                "time_s": local_t,
                "controller": controller,
                "controller_label": CONTROLLER_LABELS.get(controller, controller),
                "andes_frequency_hz": frequency_hz,
                "andes_boundary_voltage_pu": boundary_v_pu,
                "andes_boundary_angle_rad": boundary_angle_rad,
                "opendss_source_voltage_pu": feeder_source_pu,
                "opendss_pcc_voltage_pu": feeder_solution.pcc_voltage_pu,
                "opendss_min_voltage_pu": feeder_solution.min_voltage_pu,
                "opendss_max_voltage_pu": feeder_solution.max_voltage_pu,
                "feeder_boundary_kw": boundary_kw,
                "feeder_boundary_kvar": boundary_kvar,
                "nominal_boundary_kw": nominal_boundary_kw,
                "nominal_boundary_kvar": nominal_boundary_kvar,
                "aggregate_iac_p_kw": aggregate_p_kw,
                "aggregate_iac_q_kvar": aggregate_q_kvar,
                "feeder_relief_pu": feeder_relief_pu,
                "disturbance_pu": disturbance_pu,
                "paux_pu": paux_pu,
            }
        )

        if k < steps:
            system.TDS.config.tf = round((k + 1) * cfg.coupling_step_s, 10)
            ok = system.TDS.run(no_summary=True)
            if not ok:
                raise RuntimeError(f"ANDES TDS failed for {controller} at t={local_t:.3f}s")

    return pd.DataFrame(rows)


def compute_metrics(traces: pd.DataFrame, cfg: TDCosimConfig) -> pd.DataFrame:
    rows = []
    for controller, df in traces.groupby("controller", sort=False):
        after = df[df["time_s"] >= cfg.disturbance_time_s]
        rows.append(
            {
                "controller": controller,
                "controller_label": df["controller_label"].iloc[0],
                "frequency_nadir_hz": float(after["andes_frequency_hz"].min()),
                "min_feeder_voltage_pu": float(df["opendss_min_voltage_pu"].min()),
                "min_pcc_voltage_pu": float(df["opendss_pcc_voltage_pu"].min()),
                "max_boundary_load_kw": float(df["feeder_boundary_kw"].max()),
                "min_paux_pu": float(df["paux_pu"].min()),
                "max_paux_pu": float(df["paux_pu"].max()),
            }
        )
    return pd.DataFrame(rows)


def plot_traces(traces: pd.DataFrame, path: Path) -> None:
    apply_elsevier_figure_style(base_size=8.0)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(3, 1, figsize=(7.2, 6.6), sharex=True)
    colors = {"no_support": "#4D4D4D", "dmpc_active_only": "#0072B2", "dmpc_full_pq": "#D55E00"}
    for controller, df in traces.groupby("controller", sort=False):
        label = df["controller_label"].iloc[0]
        color = colors.get(controller, None)
        axes[0].plot(df["time_s"], df["andes_frequency_hz"], label=label, color=color, lw=1.8)
        axes[1].plot(df["time_s"], df["opendss_min_voltage_pu"], label=label, color=color, lw=1.8)
        axes[2].plot(df["time_s"], df["feeder_boundary_kw"], label=label, color=color, lw=1.8)
    axes[0].set_ylabel("COI frequency (Hz)")
    axes[1].set_ylabel("Worst feeder V (pu)")
    axes[2].set_ylabel("Boundary load (kW)")
    axes[2].set_xlabel("Coupled simulation time (s)")
    for ax in axes:
        ax.grid(True, alpha=0.3)
    axes[0].legend(ncol=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=250)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def write_latex_table(metrics: pd.DataFrame, path: Path) -> None:
    rows = []
    for _, row in metrics.iterrows():
        rows.append(
            f"{row['controller_label']} & {row['frequency_nadir_hz']:.3f} & "
            f"{row['min_feeder_voltage_pu']:.4f} & {row['min_pcc_voltage_pu']:.4f} & "
            f"{row['max_boundary_load_kw']:.0f} \\\\"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\\begin{table}[!t]\n"
        "\\centering\n"
        "\\footnotesize\n"
        "\\setlength{\\tabcolsep}{3.5pt}\n"
        "\\caption{Explicit loose-coupled T\\&D consistency check with ANDES IEEE 14-bus and OpenDSS IEEE 13-node systems.}\n"
        "\\label{tab:td_cosim}\n"
        "\\begin{tabular}{lrrrr}\n"
        "\\toprule\n"
        "Controller & Nadir & $V_{\\min}$ & PCC V & Load \\\\\n"
        " & (Hz) & (pu) & (pu) & (kW) \\\\\n"
        "\\midrule\n"
        + "\n".join(rows)
        + "\n\\bottomrule\n"
        "\\end{tabular}\n"
        "\\end{table}\n",
        encoding="utf-8",
    )


def run_cosim(args: argparse.Namespace) -> None:
    cfg = TDCosimConfig(coupling_step_s=args.coupling_step_s, duration_s=args.duration_s)
    experiment_root = Path(args.experiment_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    traces = []
    for idx, controller in enumerate(CONTROLLERS, start=1):
        print(f"[{idx}/{len(CONTROLLERS)}] Running explicit T&D co-sim: {controller}", flush=True)
        trace = read_controller_trace(experiment_root, cfg.scenario, controller)
        traces.append(run_one_controller(controller, trace, cfg))
    all_traces = pd.concat(traces, ignore_index=True)
    metrics = compute_metrics(all_traces, cfg)

    trace_path = output_dir / "td_cosim_traces.csv"
    metrics_path = output_dir / "td_cosim_metrics.csv"
    figure_path = output_dir / "td_cosim_response.png"
    table_path = output_dir / "td_cosim.tex"
    all_traces.to_csv(trace_path, index=False)
    metrics.to_csv(metrics_path, index=False)
    plot_traces(all_traces, figure_path)
    write_latex_table(metrics, table_path)

    paper_figure_dir = Path(args.paper_figure_dir).resolve()
    paper_table_dir = Path(args.paper_table_dir).resolve()
    paper_figure_dir.mkdir(parents=True, exist_ok=True)
    paper_table_dir.mkdir(parents=True, exist_ok=True)
    import shutil

    shutil.copy2(figure_path, paper_figure_dir / "td_cosim_response.png")
    shutil.copy2(figure_path.with_suffix(".pdf"), paper_figure_dir / "td_cosim_response.pdf")
    shutil.copy2(table_path, paper_table_dir / "td_cosim.tex")
    print(metrics.to_string(index=False), flush=True)
    print(f"Wrote explicit loose-coupled T&D co-simulation results to {output_dir}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-root", default=str(DEFAULT_EXPERIMENT_ROOT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--paper-figure-dir", default=str(DEFAULT_PAPER_FIGURE_DIR))
    parser.add_argument("--paper-table-dir", default=str(DEFAULT_PAPER_TABLE_DIR))
    parser.add_argument("--coupling-step-s", type=float, default=0.1)
    parser.add_argument("--duration-s", type=float, default=25.0)
    return parser.parse_args()


if __name__ == "__main__":
    run_cosim(parse_args())
