"""Averaged inverter trackability check for the IAC P/Q command trajectory.

This script consumes the latest full P/Q MPC experiment, expands the slow
controller commands into a sub-millisecond averaged converter simulation, and
exports current-loop, P/Q tracking, PLL, current-limit, and DC-link metrics for
the manuscript.
"""

from __future__ import annotations

import argparse
import math
import shutil
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from iac_dmpc.plot_style import apply_elsevier_figure_style

from iac_dmpc.capability import InverterCapability
from iac_dmpc.parameters import GridParameters, InverterInnerLoopParameters, InverterParameters
from iac_dmpc.signal_processing import DQTransform, SRFPLL


ROOT = Path(__file__).resolve().parent
NIST_TRACE = (
    ROOT
    / "experiments"
    / "nist_measured_calibrated"
    / "full"
    / "cases"
    / "voltage_sag_0.92__dmpc_full_pq"
    / "voltage_sag_0.92__dmpc_full_pq.csv"
)
DEFAULT_TRACE = (
    NIST_TRACE
    if NIST_TRACE.exists()
    else ROOT
    / "experiments"
    / "dynamic_replay_calibrated"
    / "full"
    / "cases"
    / "voltage_sag_0.92__dmpc_full_pq"
    / "voltage_sag_0.92__dmpc_full_pq.csv"
)
DEFAULT_OUTPUT_DIR = ROOT / "experiments" / "converter_validation"
DEFAULT_PAPER_FIGURE_DIR = ROOT / "paper" / "figures"
DEFAULT_PAPER_TABLE_DIR = ROOT / "paper" / "tables"


def require_trace_file(trace_path: Path) -> None:
    if trace_path.exists():
        return
    raise FileNotFoundError(
        "\nMissing full P/Q MPC trace for converter validation:\n"
        f"  {trace_path}\n\n"
        "The Mendeley research-data package should include this representative trace. "
        "If it is absent, regenerate it with:\n"
        "  python generate_dynamic_frequency_trace.py\n"
        "  python run_experiment_suite.py --scenario-set full "
        "--thermal-calibration-json data/nist_nzertf/calibrated_thermal_parameters.json "
        "--output-dir experiments/nist_measured_calibrated\n\n"
        "For a faster smoke test, generate the quick suite and pass its full P/Q trace explicitly, for example:\n"
        "  python run_experiment_suite.py --scenario-set quick "
        "--thermal-calibration-json data/nist_nzertf/calibrated_thermal_parameters.json "
        "--output-dir experiments/reviewer_quick_check\n"
        "  python validate_averaged_converter.py --trace "
        "experiments/reviewer_quick_check/quick/cases/impact_ieee13_fleet__dmpc_full_pq/"
        "impact_ieee13_fleet__dmpc_full_pq.csv\n"
    )


@dataclass(frozen=True)
class AveragedConverterConfig:
    start_time_s: float = 175.0
    duration_s: float = 20.0
    step_s: float = 5.0e-4
    warmup_s: float = 0.5
    filter_inductance_h: float = 2.0e-3
    filter_resistance_ohm: float = 0.08
    dc_link_capacitance_f: float = 2.2e-3
    switching_frequency_hz: float = 10_000.0
    modulation_limit: float = 0.94
    dc_source_tau_s: float = 0.025
    dc_voltage_regulator_a_per_v: float = 0.20
    voltage_notch_start_s: float = 6.0
    voltage_notch_duration_s: float = 0.12
    voltage_notch_pu: float = 0.75
    frequency_pulse_start_s: float = 8.0
    frequency_pulse_duration_s: float = 0.6
    frequency_pulse_hz: float = -0.35
    nominal_voltage_rms: float = 240.0
    nominal_frequency_hz: float = 60.0


@dataclass
class ConverterState:
    id_a: float = 0.0
    iq_a: float = 0.0
    int_d_v: float = 0.0
    int_q_v: float = 0.0
    v_inv_d_v: float = 0.0
    v_inv_q_v: float = 0.0
    vdc_v: float = 390.0
    dc_source_current_a: float = 0.0
    theta_grid_rad: float = 0.0


def zero_order_hold(times: np.ndarray, values: np.ndarray, query: float) -> float:
    idx = int(np.searchsorted(times, query, side="right") - 1)
    idx = max(0, min(idx, len(values) - 1))
    return float(values[idx])


def command_at(trace: pd.DataFrame, absolute_time_s: float) -> dict[str, float]:
    times = trace["time_s"].to_numpy(dtype=float)
    p_kw = zero_order_hold(times, trace["p_command_kw"].to_numpy(dtype=float), absolute_time_s)
    q_kvar = zero_order_hold(times, trace["q_command_kvar"].to_numpy(dtype=float), absolute_time_s)
    voltage_pu = zero_order_hold(times, trace["source_voltage_pu"].to_numpy(dtype=float), absolute_time_s)
    frequency_hz = zero_order_hold(times, trace["frequency_hz"].to_numpy(dtype=float), absolute_time_s)
    return {"p_kw": p_kw, "q_kvar": q_kvar, "voltage_pu": voltage_pu, "frequency_hz": frequency_hz}


def event_profiles(local_time_s: float, voltage_pu: float, frequency_hz: float, cfg: AveragedConverterConfig) -> tuple[float, float]:
    if cfg.voltage_notch_start_s <= local_time_s <= cfg.voltage_notch_start_s + cfg.voltage_notch_duration_s:
        voltage_pu = min(voltage_pu, cfg.voltage_notch_pu)
    if cfg.frequency_pulse_start_s <= local_time_s <= cfg.frequency_pulse_start_s + cfg.frequency_pulse_duration_s:
        frequency_hz += cfg.frequency_pulse_hz
    return voltage_pu, frequency_hz


def limit_current_references(
    p_ref_kw: float,
    q_ref_kvar: float,
    voltage_rms_v: float,
    vdc_v: float,
    inverter: InverterParameters,
    inner: InverterInnerLoopParameters,
) -> tuple[float, float, float, float, bool, bool]:
    capability = InverterCapability(inverter.s_rated_kva, inverter.p_min_kw, inverter.p_max_kw, inverter.q_max_kvar)
    pq = capability.allocate(p_ref_kw, q_ref_kvar, reactive_priority=True)
    id_ref = pq.p_kw * 1000.0 / max(voltage_rms_v, 1e-6)
    iq_ref = pq.q_kvar * 1000.0 / max(voltage_rms_v, 1e-6)

    dc_power_limit_kw = max(vdc_v, 1e-6) * inner.dc_link_current_limit_a / 1000.0
    dc_link_limited = pq.p_kw > dc_power_limit_kw
    if dc_link_limited:
        id_ref = min(id_ref, dc_power_limit_kw * 1000.0 / max(voltage_rms_v, 1e-6))

    current_limit = inner.rms_current_limit_a
    saturated = math.hypot(id_ref, iq_ref) > current_limit
    if saturated:
        iq_limited = math.copysign(min(abs(iq_ref), current_limit), iq_ref)
        id_headroom = math.sqrt(max(current_limit**2 - iq_limited**2, 0.0))
        id_limited = math.copysign(min(abs(id_ref), id_headroom), id_ref)
        id_ref, iq_ref = id_limited, iq_limited

    p_limited_kw = voltage_rms_v * id_ref / 1000.0
    q_limited_kvar = voltage_rms_v * iq_ref / 1000.0
    return id_ref, iq_ref, p_limited_kw, q_limited_kvar, saturated, dc_link_limited


def saturate_voltage(v_d: float, v_q: float, vdc_v: float, cfg: AveragedConverterConfig) -> tuple[float, float, bool, float]:
    limit_rms = cfg.modulation_limit * vdc_v / math.sqrt(2.0)
    magnitude = math.hypot(v_d, v_q)
    if magnitude <= limit_rms:
        return v_d, v_q, False, magnitude / max(limit_rms, 1e-9)
    scale = limit_rms / max(magnitude, 1e-9)
    return v_d * scale, v_q * scale, True, 1.0


def simulate_averaged_converter(
    trace: pd.DataFrame,
    cfg: AveragedConverterConfig,
    inverter: InverterParameters | None = None,
    inner: InverterInnerLoopParameters | None = None,
    grid: GridParameters | None = None,
) -> pd.DataFrame:
    inverter = inverter or InverterParameters()
    inner = inner or InverterInnerLoopParameters()
    grid = grid or GridParameters()

    dt = cfg.step_s
    total_s = cfg.duration_s + cfg.warmup_s
    steps = int(round(total_s / dt)) + 1
    wc = 2.0 * math.pi * inner.current_controller_bandwidth_hz
    kp_i = cfg.filter_inductance_h * wc
    ki_i = cfg.filter_resistance_ohm * wc
    pwm_tau = 1.5 / cfg.switching_frequency_hz
    nominal_omega = 2.0 * math.pi * cfg.nominal_frequency_hz

    state = ConverterState(vdc_v=inner.dc_link_voltage_v)
    first = command_at(trace, cfg.start_time_s)
    state.id_a = first["p_kw"] * 1000.0 / cfg.nominal_voltage_rms
    state.iq_a = first["q_kvar"] * 1000.0 / cfg.nominal_voltage_rms
    state.v_inv_d_v = cfg.nominal_voltage_rms
    state.dc_source_current_a = max(first["p_kw"] * 1000.0 / state.vdc_v, 0.0)

    pll = SRFPLL(dt, nominal_omega, grid.pll_kp, grid.pll_ki)
    pll.reset(0.0)

    rows = []
    for n in range(steps):
        sim_time = n * dt - cfg.warmup_s
        local_time = max(sim_time, 0.0)
        absolute_time = cfg.start_time_s + local_time
        cmd = command_at(trace, absolute_time)
        voltage_pu, grid_frequency_hz = event_profiles(local_time, cmd["voltage_pu"], cmd["frequency_hz"], cfg)
        omega_grid = 2.0 * math.pi * grid_frequency_hz
        state.theta_grid_rad = (state.theta_grid_rad + omega_grid * dt) % (2.0 * math.pi)

        v_rms = cfg.nominal_voltage_rms * voltage_pu
        v_alpha_peak = math.sqrt(2.0) * v_rms * math.cos(state.theta_grid_rad)
        v_beta_peak = math.sqrt(2.0) * v_rms * math.sin(state.theta_grid_rad)
        theta_pll, omega_pll = pll.update(v_alpha_peak, v_beta_peak)
        v_dq = DQTransform.alpha_beta_to_dq(v_alpha_peak / math.sqrt(2.0), v_beta_peak / math.sqrt(2.0), theta_pll)
        v_grid_d = v_dq.d
        v_grid_q = v_dq.q

        id_ref, iq_ref, p_limited_kw, q_limited_kvar, current_sat, dc_limited = limit_current_references(
            cmd["p_kw"], cmd["q_kvar"], max(abs(v_grid_d), 60.0), state.vdc_v, inverter, inner
        )

        err_d = id_ref - state.id_a
        err_q = iq_ref - state.iq_a
        v_ff_d = v_grid_d - pll.omega * cfg.filter_inductance_h * state.iq_a + cfg.filter_resistance_ohm * state.id_a
        v_ff_q = v_grid_q + pll.omega * cfg.filter_inductance_h * state.id_a + cfg.filter_resistance_ohm * state.iq_a
        v_cmd_d = v_ff_d + kp_i * err_d + state.int_d_v
        v_cmd_q = v_ff_q + kp_i * err_q + state.int_q_v
        v_sat_d, v_sat_q, voltage_sat, modulation_index = saturate_voltage(v_cmd_d, v_cmd_q, state.vdc_v, cfg)

        state.int_d_v += dt * (ki_i * err_d + inner.anti_windup_gain * (v_sat_d - v_cmd_d))
        state.int_q_v += dt * (ki_i * err_q + inner.anti_windup_gain * (v_sat_q - v_cmd_q))
        state.v_inv_d_v += dt / max(pwm_tau, dt) * (v_sat_d - state.v_inv_d_v)
        state.v_inv_q_v += dt / max(pwm_tau, dt) * (v_sat_q - state.v_inv_q_v)

        did = (state.v_inv_d_v - v_grid_d - cfg.filter_resistance_ohm * state.id_a + pll.omega * cfg.filter_inductance_h * state.iq_a) / cfg.filter_inductance_h
        diq = (state.v_inv_q_v - v_grid_q - cfg.filter_resistance_ohm * state.iq_a - pll.omega * cfg.filter_inductance_h * state.id_a) / cfg.filter_inductance_h
        state.id_a += dt * did
        state.iq_a += dt * diq

        p_actual_kw = (v_grid_d * state.id_a + v_grid_q * state.iq_a) / 1000.0
        q_actual_kvar = (v_grid_d * state.iq_a - v_grid_q * state.id_a) / 1000.0
        dc_load_current = max(p_actual_kw * 1000.0 / max(state.vdc_v, 80.0), 0.0)
        voltage_regulator_current = cfg.dc_voltage_regulator_a_per_v * (inner.dc_link_voltage_v - state.vdc_v)
        dc_target_current = min(
            max(dc_load_current + voltage_regulator_current, 0.0),
            inner.dc_link_current_limit_a,
        )
        state.dc_source_current_a += dt / cfg.dc_source_tau_s * (dc_target_current - state.dc_source_current_a)
        state.vdc_v += dt / cfg.dc_link_capacitance_f * (state.dc_source_current_a - dc_load_current)
        state.vdc_v = float(np.clip(state.vdc_v, 250.0, 430.0))

        rms_current = math.hypot(state.id_a, state.iq_a)
        if sim_time >= 0.0:
            rows.append(
                {
                    "time_s": local_time,
                    "source_time_s": absolute_time,
                    "p_ref_kw": cmd["p_kw"],
                    "q_ref_kvar": cmd["q_kvar"],
                    "p_limited_kw": p_limited_kw,
                    "q_limited_kvar": q_limited_kvar,
                    "p_actual_kw": p_actual_kw,
                    "q_actual_kvar": q_actual_kvar,
                    "p_error_kw": p_actual_kw - cmd["p_kw"],
                    "q_error_kvar": q_actual_kvar - cmd["q_kvar"],
                    "id_ref_a": id_ref,
                    "iq_ref_a": iq_ref,
                    "id_actual_a": state.id_a,
                    "iq_actual_a": state.iq_a,
                    "rms_current_a": rms_current,
                    "vdc_v": state.vdc_v,
                    "dc_source_current_a": state.dc_source_current_a,
                    "dc_load_current_a": dc_load_current,
                    "grid_voltage_pu": voltage_pu,
                    "grid_frequency_hz": grid_frequency_hz,
                    "pll_frequency_hz": pll.omega / (2.0 * math.pi),
                    "pll_frequency_error_hz": pll.omega / (2.0 * math.pi) - grid_frequency_hz,
                    "modulation_index": modulation_index,
                    "current_saturated": float(current_sat),
                    "dc_link_limited": float(dc_limited),
                    "voltage_saturated": float(voltage_sat),
                }
            )
    return pd.DataFrame(rows)


def compute_metrics(trace: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "max_abs_p_error_kw": float(trace["p_error_kw"].abs().max()),
                "rms_p_error_kw": float(np.sqrt(np.mean(trace["p_error_kw"] ** 2))),
                "max_abs_q_error_kvar": float(trace["q_error_kvar"].abs().max()),
                "rms_q_error_kvar": float(np.sqrt(np.mean(trace["q_error_kvar"] ** 2))),
                "max_rms_current_a": float(trace["rms_current_a"].max()),
                "min_dc_link_v": float(trace["vdc_v"].min()),
                "max_modulation_index": float(trace["modulation_index"].max()),
                "current_saturation_fraction": float(trace["current_saturated"].mean()),
                "dc_limit_fraction": float(trace["dc_link_limited"].mean()),
                "voltage_saturation_fraction": float(trace["voltage_saturated"].mean()),
                "max_abs_pll_frequency_error_hz": float(trace["pll_frequency_error_hz"].abs().max()),
            }
        ]
    )


def plot_validation(trace: pd.DataFrame, path: Path) -> None:
    apply_elsevier_figure_style(base_size=8.0)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(4, 1, figsize=(7.2, 7.6), sharex=True)

    axes[0].plot(trace["time_s"], trace["p_ref_kw"], color="#0072B2", lw=1.8, label="P command")
    axes[0].plot(trace["time_s"], trace["p_actual_kw"], color="#0072B2", lw=1.2, ls="--", label="P actual")
    axes[0].plot(trace["time_s"], trace["q_ref_kvar"], color="#D55E00", lw=1.8, label="Q command")
    axes[0].plot(trace["time_s"], trace["q_actual_kvar"], color="#D55E00", lw=1.2, ls="--", label="Q actual")
    axes[0].set_ylabel("P/Q")
    axes[0].legend(ncol=2, fontsize=8)

    axes[1].plot(trace["time_s"], trace["id_ref_a"], color="#0072B2", lw=1.6, label="$I_d^*$")
    axes[1].plot(trace["time_s"], trace["id_actual_a"], color="#0072B2", lw=1.0, ls="--", label="$I_d$")
    axes[1].plot(trace["time_s"], trace["iq_ref_a"], color="#D55E00", lw=1.6, label="$I_q^*$")
    axes[1].plot(trace["time_s"], trace["iq_actual_a"], color="#D55E00", lw=1.0, ls="--", label="$I_q$")
    axes[1].set_ylabel("Current (A)")
    axes[1].legend(ncol=4, fontsize=8)

    axes[2].plot(trace["time_s"], trace["vdc_v"], color="#009E73", lw=1.5, label="$V_{dc}$")
    axes[2].set_ylabel("$V_{dc}$ (V)")
    ax2b = axes[2].twinx()
    ax2b.plot(trace["time_s"], trace["rms_current_a"], color="#CC79A7", lw=1.2, label="$I_{rms}$")
    ax2b.set_ylabel("$I_{rms}$ (A)")
    axes[2].legend(loc="upper left", fontsize=8)
    ax2b.legend(loc="upper right", fontsize=8)

    axes[3].plot(trace["time_s"], trace["grid_voltage_pu"], color="#4D4D4D", lw=1.3, label="Grid voltage")
    axes[3].plot(trace["time_s"], trace["grid_frequency_hz"] - 60.0, color="#56B4E9", lw=1.3, label="$f-60$")
    axes[3].fill_between(
        trace["time_s"],
        0,
        trace["current_saturated"],
        color="#E69F00",
        alpha=0.25,
        step="pre",
        label="Current saturation",
    )
    axes[3].set_ylabel("pu / Hz")
    axes[3].set_xlabel("Converter validation time (s)")
    axes[3].legend(ncol=3, fontsize=8)

    for ax in axes:
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=250)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def write_latex_table(metrics: pd.DataFrame, path: Path) -> None:
    row = metrics.iloc[0]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\\begin{table}[!t]\n"
        "\\centering\n"
        "\\caption{Averaged converter trackability-check metrics for the full P/Q MPC trajectory.}\n"
        "\\label{tab:converter_validation}\n"
        "\\begin{tabular}{lr}\n"
        "\\toprule\n"
        "Metric & Value \\\\\n"
        "\\midrule\n"
        f"Maximum $|P-P^\\star|$ & {row['max_abs_p_error_kw']:.3f} kW \\\\\n"
        f"RMS $P$ error & {row['rms_p_error_kw']:.3f} kW \\\\\n"
        f"Maximum $|Q-Q^\\star|$ & {row['max_abs_q_error_kvar']:.3f} kvar \\\\\n"
        f"RMS $Q$ error & {row['rms_q_error_kvar']:.3f} kvar \\\\\n"
        f"Maximum RMS current & {row['max_rms_current_a']:.2f} A \\\\\n"
        f"Minimum DC-link voltage & {row['min_dc_link_v']:.1f} V \\\\\n"
        f"Current saturation fraction & {100.0 * row['current_saturation_fraction']:.2f}\\% \\\\\n"
        f"Maximum PLL frequency error & {row['max_abs_pll_frequency_error_hz']:.3f} Hz \\\\\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\end{table}\n",
        encoding="utf-8",
    )


def run_validation(args: argparse.Namespace) -> None:
    cfg = AveragedConverterConfig(
        start_time_s=args.start_time_s,
        duration_s=args.duration_s,
        step_s=args.step_s,
    )
    trace_path = Path(args.trace).resolve()
    require_trace_file(trace_path)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    trace = pd.read_csv(trace_path)
    validation = simulate_averaged_converter(trace, cfg)
    metrics = compute_metrics(validation)

    csv_path = output_dir / "averaged_converter_validation.csv"
    metrics_path = output_dir / "averaged_converter_metrics.csv"
    fig_path = output_dir / "averaged_converter_validation.png"
    table_path = output_dir / "averaged_converter_validation.tex"
    validation.to_csv(csv_path, index=False)
    metrics.to_csv(metrics_path, index=False)
    plot_validation(validation, fig_path)
    write_latex_table(metrics, table_path)

    paper_figures = Path(args.paper_figure_dir).resolve()
    paper_tables = Path(args.paper_table_dir).resolve()
    paper_figures.mkdir(parents=True, exist_ok=True)
    paper_tables.mkdir(parents=True, exist_ok=True)
    shutil.copy2(fig_path, paper_figures / "averaged_converter_validation.png")
    shutil.copy2(fig_path.with_suffix(".pdf"), paper_figures / "averaged_converter_validation.pdf")
    shutil.copy2(table_path, paper_tables / "averaged_converter_validation.tex")

    print(metrics.to_string(index=False))
    print(f"Wrote averaged converter validation to {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", default=str(DEFAULT_TRACE))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--paper-figure-dir", default=str(DEFAULT_PAPER_FIGURE_DIR))
    parser.add_argument("--paper-table-dir", default=str(DEFAULT_PAPER_TABLE_DIR))
    parser.add_argument("--start-time-s", type=float, default=175.0)
    parser.add_argument("--duration-s", type=float, default=20.0)
    parser.add_argument("--step-s", type=float, default=5.0e-4)
    return parser.parse_args()


if __name__ == "__main__":
    run_validation(parse_args())
