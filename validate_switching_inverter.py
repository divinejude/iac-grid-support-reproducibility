"""Switching-level inverter trackability check for the IAC P/Q command trajectory.

The OpenDSS feeder model is intentionally quasi-static. This script adds a
separate fast-timescale validation layer for one representative IAC converter
connected to a Thevenin-equivalent feeder bus. The full P/Q MPC command trace
is held as the slow reference, while the converter is simulated with explicit
bipolar full-bridge PWM, PLL dynamics, dq current PI control, an L-filter plus
Thevenin impedance, DC-link dynamics, current limiting, P/Q priority, and
anti-windup.
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

from iac_dmpc.parameters import GridParameters, InverterInnerLoopParameters, InverterParameters
from iac_dmpc.signal_processing import DQTransform, SOGIFilter, SRFPLL
from validate_averaged_converter import DEFAULT_TRACE, command_at, limit_current_references, require_trace_file, saturate_voltage


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = ROOT / "experiments" / "switching_inverter_validation"
DEFAULT_PAPER_FIGURE_DIR = ROOT / "paper" / "figures"
DEFAULT_PAPER_TABLE_DIR = ROOT / "paper" / "tables"


@dataclass(frozen=True)
class SwitchingInverterConfig:
    start_time_s: float = 175.0
    duration_s: float = 2.0
    step_s: float = 2.0e-5
    warmup_s: float = 0.04
    nominal_voltage_rms: float = 240.0
    nominal_frequency_hz: float = 60.0
    filter_inductance_h: float = 2.0e-3
    filter_resistance_ohm: float = 0.08
    thevenin_resistance_ohm: float = 0.04
    thevenin_inductance_h: float = 0.25e-3
    ripple_damping_ohm: float = 4.0
    dc_link_capacitance_f: float = 2.2e-3
    switching_frequency_hz: float = 10_000.0
    modulation_limit: float = 0.92
    dc_source_tau_s: float = 0.015
    dc_voltage_regulator_a_per_v: float = 0.25
    voltage_sag_start_s: float = 0.45
    voltage_sag_duration_s: float = 0.12
    voltage_sag_pu: float = 0.75
    frequency_step_start_s: float = 1.10
    frequency_step_duration_s: float = 0.45
    frequency_step_hz: float = -0.30
    carrier_initial_phase: float = 0.0
    metric_skip_s: float = 0.08


@dataclass
class SwitchingState:
    id_a: float = 0.0
    iq_a: float = 0.0
    ripple_current_a: float = 0.0
    int_d_v: float = 0.0
    int_q_v: float = 0.0
    vdc_v: float = 390.0
    dc_source_current_a: float = 0.0
    theta_grid_rad: float = 0.0


def event_profiles(local_time_s: float, voltage_pu: float, frequency_hz: float, cfg: SwitchingInverterConfig) -> tuple[float, float]:
    if cfg.voltage_sag_start_s <= local_time_s <= cfg.voltage_sag_start_s + cfg.voltage_sag_duration_s:
        voltage_pu = min(voltage_pu, cfg.voltage_sag_pu)
    if cfg.frequency_step_start_s <= local_time_s <= cfg.frequency_step_start_s + cfg.frequency_step_duration_s:
        frequency_hz += cfg.frequency_step_hz
    return voltage_pu, frequency_hz


def triangular_carrier(time_s: float, switching_frequency_hz: float, initial_phase: float = 0.0) -> float:
    phase = (time_s * switching_frequency_hz + initial_phase) % 1.0
    return 4.0 * abs(phase - 0.5) - 1.0


def dq_to_alpha(d: float, q: float, theta_rad: float) -> float:
    return d * math.cos(theta_rad) - q * math.sin(theta_rad)


def simulate_switching_inverter(
    trace: pd.DataFrame,
    cfg: SwitchingInverterConfig,
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
    nominal_omega = 2.0 * math.pi * cfg.nominal_frequency_hz
    l_total = cfg.filter_inductance_h + cfg.thevenin_inductance_h
    r_total = cfg.filter_resistance_ohm + cfg.thevenin_resistance_ohm
    wc = 2.0 * math.pi * inner.current_controller_bandwidth_hz
    kp_i = cfg.filter_inductance_h * wc
    ki_i = cfg.filter_resistance_ohm * wc

    voltage_sogi = SOGIFilter(dt, nominal_omega, grid.sogi_gain)
    pll = SRFPLL(dt, nominal_omega, grid.pll_kp, grid.pll_ki)
    state = SwitchingState(vdc_v=inner.dc_link_voltage_v)

    first = command_at(trace, cfg.start_time_s)
    state.id_a = first["p_kw"] * 1000.0 / cfg.nominal_voltage_rms
    state.iq_a = first["q_kvar"] * 1000.0 / cfg.nominal_voltage_rms
    state.dc_source_current_a = max(first["p_kw"] * 1000.0 / state.vdc_v, 0.0)

    rows: list[dict[str, float]] = []
    for n in range(steps):
        sim_time = n * dt - cfg.warmup_s
        local_time = max(sim_time, 0.0)
        absolute_time = cfg.start_time_s + local_time
        cmd = command_at(trace, absolute_time)
        voltage_pu, frequency_hz = event_profiles(local_time, cmd["voltage_pu"], cmd["frequency_hz"], cfg)
        omega_grid = 2.0 * math.pi * frequency_hz
        state.theta_grid_rad = (state.theta_grid_rad + omega_grid * dt) % (2.0 * math.pi)

        v_rms = cfg.nominal_voltage_rms * voltage_pu
        v_source = math.sqrt(2.0) * v_rms * math.cos(state.theta_grid_rad)

        v_alpha, v_beta = voltage_sogi.update(v_source, pll.omega)
        theta_pll, omega_pll = pll.update(v_alpha, v_beta)
        v_dq = DQTransform.alpha_beta_to_dq(v_alpha / math.sqrt(2.0), v_beta / math.sqrt(2.0), theta_pll)

        voltage_for_limit = max(abs(v_dq.d), 60.0)
        id_ref, iq_ref, p_limited_kw, q_limited_kvar, current_sat, dc_limited = limit_current_references(
            cmd["p_kw"], cmd["q_kvar"], voltage_for_limit, state.vdc_v, inverter, inner
        )

        err_d = id_ref - state.id_a
        err_q = iq_ref - state.iq_a
        v_ff_d = v_dq.d - omega_pll * cfg.filter_inductance_h * state.iq_a + cfg.filter_resistance_ohm * state.id_a
        v_ff_q = v_dq.q + omega_pll * cfg.filter_inductance_h * state.id_a + cfg.filter_resistance_ohm * state.iq_a
        v_cmd_d = v_ff_d + kp_i * err_d + state.int_d_v
        v_cmd_q = v_ff_q + kp_i * err_q + state.int_q_v
        v_sat_d, v_sat_q, voltage_saturated, modulation_index = saturate_voltage(v_cmd_d, v_cmd_q, state.vdc_v, cfg)

        state.int_d_v += dt * (ki_i * err_d + inner.anti_windup_gain * (v_sat_d - v_cmd_d))
        state.int_q_v += dt * (ki_i * err_q + inner.anti_windup_gain * (v_sat_q - v_cmd_q))

        did = (v_sat_d - v_dq.d - cfg.filter_resistance_ohm * state.id_a + omega_pll * cfg.filter_inductance_h * state.iq_a) / cfg.filter_inductance_h
        diq = (v_sat_q - v_dq.q - cfg.filter_resistance_ohm * state.iq_a - omega_pll * cfg.filter_inductance_h * state.id_a) / cfg.filter_inductance_h
        state.id_a += dt * did
        state.iq_a += dt * diq

        v_sat_alpha_peak = math.sqrt(2.0) * dq_to_alpha(v_sat_d, v_sat_q, theta_pll)
        modulation = float(np.clip(v_sat_alpha_peak / max(state.vdc_v, 1.0), -cfg.modulation_limit, cfg.modulation_limit))
        v_avg_alpha_peak = modulation * state.vdc_v

        carrier = triangular_carrier(local_time, cfg.switching_frequency_hz, cfg.carrier_initial_phase)
        switch_state = 1.0 if modulation >= carrier else -1.0
        v_inverter = switch_state * state.vdc_v

        ripple_resistance = r_total + cfg.ripple_damping_ohm
        dripple_dt = (v_inverter - v_avg_alpha_peak - ripple_resistance * state.ripple_current_a) / l_total
        state.ripple_current_a += dt * dripple_dt
        fundamental_current_a = math.sqrt(2.0) * dq_to_alpha(state.id_a, state.iq_a, theta_pll)
        ac_current_a = fundamental_current_a + state.ripple_current_a

        p_fundamental_kw = (v_dq.d * state.id_a + v_dq.q * state.iq_a) / 1000.0
        q_fundamental_kvar = (v_dq.d * state.iq_a - v_dq.q * state.id_a) / 1000.0
        p_switch_kw = v_source * ac_current_a / 1000.0
        dc_load_current = max(p_fundamental_kw * 1000.0 / max(state.vdc_v, 80.0), 0.0)
        voltage_reg_current = cfg.dc_voltage_regulator_a_per_v * (inner.dc_link_voltage_v - state.vdc_v)
        dc_target_current = min(max(dc_load_current + voltage_reg_current, 0.0), inner.dc_link_current_limit_a)
        state.dc_source_current_a += dt / cfg.dc_source_tau_s * (dc_target_current - state.dc_source_current_a)
        # The AC bridge is switched explicitly, while the compressor/DC side is
        # represented by the fundamental active-power draw. This avoids turning
        # high-frequency capacitor ripple, which would require detailed DC-side
        # parasitics and switching loss models, into an artificial energy sink.
        state.vdc_v += dt / cfg.dc_link_capacitance_f * (state.dc_source_current_a - dc_load_current)
        state.vdc_v = float(np.clip(state.vdc_v, 250.0, 430.0))

        if sim_time >= 0.0:
            rows.append(
                {
                    "time_s": local_time,
                    "source_time_s": absolute_time,
                    "p_ref_kw": cmd["p_kw"],
                    "q_ref_kvar": cmd["q_kvar"],
                    "p_limited_kw": p_limited_kw,
                    "q_limited_kvar": q_limited_kvar,
                    "p_fundamental_kw": p_fundamental_kw,
                    "q_fundamental_kvar": q_fundamental_kvar,
                    "p_switching_instant_kw": p_switch_kw,
                    "p_error_kw": p_fundamental_kw - cmd["p_kw"],
                    "q_error_kvar": q_fundamental_kvar - cmd["q_kvar"],
                    "id_ref_a": id_ref,
                    "iq_ref_a": iq_ref,
                    "id_actual_a": state.id_a,
                    "iq_actual_a": state.iq_a,
                    "ac_current_a": ac_current_a,
                    "ripple_current_a": state.ripple_current_a,
                    "rms_current_a": math.hypot(state.id_a, state.iq_a),
                    "vdc_v": state.vdc_v,
                    "grid_voltage_pu": voltage_pu,
                    "grid_frequency_hz": frequency_hz,
                    "pll_frequency_hz": pll.omega / (2.0 * math.pi),
                    "pll_frequency_error_hz": pll.omega / (2.0 * math.pi) - frequency_hz,
                    "modulation": modulation,
                    "modulation_index": modulation_index,
                    "carrier": carrier,
                    "switch_state": switch_state,
                    "v_inverter_v": v_inverter,
                    "v_source_v": v_source,
                    "current_saturated": float(current_sat),
                    "dc_link_limited": float(dc_limited),
                    "voltage_saturated": float(voltage_saturated),
                }
            )
    return pd.DataFrame(rows)


def compute_metrics(trace: pd.DataFrame, cfg: SwitchingInverterConfig) -> pd.DataFrame:
    eval_trace = trace[trace["time_s"] >= cfg.metric_skip_s]
    if eval_trace.empty:
        eval_trace = trace
    return pd.DataFrame(
        [
            {
                "step_us": cfg.step_s * 1.0e6,
                "switching_frequency_khz": cfg.switching_frequency_hz / 1000.0,
                "max_abs_p_error_kw": float(eval_trace["p_error_kw"].abs().max()),
                "rms_p_error_kw": float(np.sqrt(np.mean(eval_trace["p_error_kw"] ** 2))),
                "max_abs_q_error_kvar": float(eval_trace["q_error_kvar"].abs().max()),
                "rms_q_error_kvar": float(np.sqrt(np.mean(eval_trace["q_error_kvar"] ** 2))),
                "max_rms_current_a": float(eval_trace["rms_current_a"].max()),
                "max_ac_current_abs_a": float(eval_trace["ac_current_a"].abs().max()),
                "min_dc_link_v": float(eval_trace["vdc_v"].min()),
                "max_modulation_index": float(eval_trace["modulation_index"].max()),
                "current_saturation_fraction": float(eval_trace["current_saturated"].mean()),
                "dc_limit_fraction": float(eval_trace["dc_link_limited"].mean()),
                "voltage_saturation_fraction": float(eval_trace["voltage_saturated"].mean()),
                "max_abs_pll_frequency_error_hz": float(eval_trace["pll_frequency_error_hz"].abs().max()),
            }
        ]
    )


def plot_validation(trace: pd.DataFrame, path: Path) -> None:
    apply_elsevier_figure_style(base_size=8.0)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(4, 1, figsize=(7.2, 7.7), sharex=True)

    axes[0].plot(trace["time_s"], trace["p_ref_kw"], color="#0072B2", lw=1.6, label="$P^*$")
    axes[0].plot(trace["time_s"], trace["p_fundamental_kw"], color="#0072B2", lw=1.0, ls="--", label="$P$")
    axes[0].plot(trace["time_s"], trace["q_ref_kvar"], color="#D55E00", lw=1.6, label="$Q^*$")
    axes[0].plot(trace["time_s"], trace["q_fundamental_kvar"], color="#D55E00", lw=1.0, ls="--", label="$Q$")
    axes[0].set_ylabel("P/Q")
    axes[0].legend(ncol=4, fontsize=8)

    axes[1].plot(trace["time_s"], trace["id_ref_a"], color="#0072B2", lw=1.3, label="$I_d^*$")
    axes[1].plot(trace["time_s"], trace["id_actual_a"], color="#0072B2", lw=0.9, ls="--", label="$I_d$")
    axes[1].plot(trace["time_s"], trace["iq_ref_a"], color="#D55E00", lw=1.3, label="$I_q^*$")
    axes[1].plot(trace["time_s"], trace["iq_actual_a"], color="#D55E00", lw=0.9, ls="--", label="$I_q$")
    axes[1].set_ylabel("Current (A)")
    axes[1].legend(ncol=4, fontsize=8)

    axes[2].plot(trace["time_s"], trace["vdc_v"], color="#009E73", lw=1.2, label="$V_{dc}$")
    axes[2].set_ylabel("$V_{dc}$ (V)")
    ax2b = axes[2].twinx()
    ax2b.plot(trace["time_s"], trace["modulation_index"], color="#CC79A7", lw=1.0, label="Mod. index")
    ax2b.set_ylabel("Mod. index")
    axes[2].legend(loc="upper left", fontsize=8)
    ax2b.legend(loc="upper right", fontsize=8)

    stride = max(1, int(len(trace) / 5000))
    axes[3].plot(trace["time_s"].iloc[::stride], trace["ac_current_a"].iloc[::stride], color="#4D4D4D", lw=0.8, label="$i_{ac}$")
    axes[3].plot(trace["time_s"], trace["grid_voltage_pu"] * 20.0, color="#56B4E9", lw=1.0, label="$20V_{grid,pu}$")
    axes[3].fill_between(
        trace["time_s"],
        0,
        trace["voltage_saturated"],
        color="#E69F00",
        alpha=0.25,
        step="pre",
        label="Voltage saturation",
    )
    axes[3].set_ylabel("A / pu flag")
    axes[3].set_xlabel("Switching validation time (s)")
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
        "\\caption{Switching-level inverter trackability-check metrics for the full P/Q MPC trajectory.}\n"
        "\\label{tab:switching_inverter_validation}\n"
        "\\begin{tabular}{lr}\n"
        "\\toprule\n"
        "Metric & Value \\\\\n"
        "\\midrule\n"
        f"Switching step & {row['step_us']:.1f} $\\mu$s \\\\\n"
        f"PWM switching frequency & {row['switching_frequency_khz']:.1f} kHz \\\\\n"
        f"RMS $P$ tracking error & {row['rms_p_error_kw']:.3f} kW \\\\\n"
        f"RMS $Q$ tracking error & {row['rms_q_error_kvar']:.3f} kvar \\\\\n"
        f"Maximum RMS current & {row['max_rms_current_a']:.2f} A \\\\\n"
        f"Peak instantaneous AC current & {row['max_ac_current_abs_a']:.2f} A \\\\\n"
        f"Minimum DC-link voltage & {row['min_dc_link_v']:.1f} V \\\\\n"
        f"Voltage saturation fraction & {100.0 * row['voltage_saturation_fraction']:.2f}\\% \\\\\n"
        f"Maximum PLL frequency error & {row['max_abs_pll_frequency_error_hz']:.3f} Hz \\\\\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\end{table}\n",
        encoding="utf-8",
    )


def run_validation(args: argparse.Namespace) -> None:
    cfg = SwitchingInverterConfig(
        start_time_s=args.start_time_s,
        duration_s=args.duration_s,
        step_s=args.step_s,
    )
    trace_path = Path(args.trace).resolve()
    require_trace_file(trace_path)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    trace = pd.read_csv(trace_path)
    validation = simulate_switching_inverter(trace, cfg)
    metrics = compute_metrics(validation, cfg)

    csv_path = output_dir / "switching_inverter_validation.csv"
    metrics_path = output_dir / "switching_inverter_metrics.csv"
    fig_path = output_dir / "switching_inverter_validation.png"
    table_path = output_dir / "switching_inverter_validation.tex"
    validation.to_csv(csv_path, index=False)
    metrics.to_csv(metrics_path, index=False)
    plot_validation(validation, fig_path)
    write_latex_table(metrics, table_path)

    paper_figures = Path(args.paper_figure_dir).resolve()
    paper_tables = Path(args.paper_table_dir).resolve()
    paper_figures.mkdir(parents=True, exist_ok=True)
    paper_tables.mkdir(parents=True, exist_ok=True)
    shutil.copy2(fig_path, paper_figures / "switching_inverter_validation.png")
    shutil.copy2(fig_path.with_suffix(".pdf"), paper_figures / "switching_inverter_validation.pdf")
    shutil.copy2(table_path, paper_tables / "switching_inverter_validation.tex")

    print(metrics.to_string(index=False))
    print(f"Wrote switching-level inverter validation to {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", default=str(DEFAULT_TRACE))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--paper-figure-dir", default=str(DEFAULT_PAPER_FIGURE_DIR))
    parser.add_argument("--paper-table-dir", default=str(DEFAULT_PAPER_TABLE_DIR))
    parser.add_argument("--start-time-s", type=float, default=175.0)
    parser.add_argument("--duration-s", type=float, default=2.0)
    parser.add_argument("--step-s", type=float, default=2.0e-5)
    return parser.parse_args()


if __name__ == "__main__":
    run_validation(parse_args())
