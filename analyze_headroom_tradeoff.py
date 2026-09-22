"""Stage 01 correction: isolate the shared P/Q headroom tradeoff.

This script runs the existing simultaneous feeder-stress plus frequency event
with the full P/Q controller and varies only the per-IAC apparent-power rating.
It then derives the active-power ceiling implied by the actual reactive command:

    P_ceiling(Q) = min(P_max, sqrt(S_rated^2 - Q^2))

The resulting figure/table are a mechanism ablation, not a new practical
controller proposal.
"""

from __future__ import annotations

import csv
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from iac_dmpc.metrics import summarize_log
from iac_dmpc.parameters import InverterParameters
from run_experiment_suite import apply_thermal_calibration
from simulate_iac_grid_support import PROJECT_ROOT, SimulationScenario, run_simulation


OUTPUT_DIR = PROJECT_ROOT / "experiments" / "stage01_headroom_ablation_A"
CALIBRATION_JSON = PROJECT_ROOT / "data" / "nist_nzertf" / "calibrated_thermal_parameters.json"


CASES = [
    ("finite_2p6kva", 2.6, "finite 2.6 kVA"),
    ("nominal_4p0kva", 4.0, "nominal 4.0 kVA"),
    ("relaxed_6p0kva", 6.0, "relaxed 6.0 kVA"),
]


def active_ceiling_for_q(q_kvar: np.ndarray, s_rated_kva: float, inverter: InverterParameters) -> np.ndarray:
    q_abs = np.minimum(np.abs(q_kvar), s_rated_kva)
    circle_ceiling = np.sqrt(np.maximum(s_rated_kva**2 - q_abs**2, 0.0))
    return np.minimum(inverter.p_max_kw, circle_ceiling)


def reactive_ceiling_for_p(p_kw: np.ndarray, s_rated_kva: float, inverter: InverterParameters) -> np.ndarray:
    p_abs = np.minimum(np.abs(p_kw), s_rated_kva)
    circle_ceiling = np.sqrt(np.maximum(s_rated_kva**2 - p_abs**2, 0.0))
    return np.minimum(inverter.q_max_kvar, circle_ceiling)


def write_log(log: dict[str, np.ndarray], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = list(log.keys())
    data = np.column_stack([log[key] for key in header])
    np.savetxt(path, data, delimiter=",", header=",".join(header), comments="")


def read_log(path: Path) -> dict[str, np.ndarray]:
    data = np.genfromtxt(path, delimiter=",", names=True)
    return {name: np.asarray(data[name], dtype=float) for name in data.dtype.names or []}


def summarize_case(label: str, case_id: str, s_rated_kva: float, log: dict[str, np.ndarray]) -> dict[str, float | str]:
    inverter = InverterParameters(s_rated_kva=s_rated_kva)
    p_ceiling = active_ceiling_for_q(log["q_actual_kvar"], s_rated_kva, inverter)
    q_ceiling = reactive_ceiling_for_p(log["p_actual_kw"], s_rated_kva, inverter)
    utilization = log["apparent_kva"] / s_rated_kva
    event = (log["time_s"] >= 180.0) & (log["time_s"] <= 380.0)
    no_q_active_ceiling = min(inverter.p_max_kw, s_rated_kva)
    ceiling_loss = np.maximum(no_q_active_ceiling - p_ceiling, 0.0)
    binding = utilization >= 0.99
    material = ceiling_loss >= 0.1
    metrics = summarize_log(log, temperature_setpoint_c=float(np.median(log["temperature_c"][:3])))
    row: dict[str, float | str] = {
        "case_id": case_id,
        "label": label,
        "s_rated_kva": s_rated_kva,
        "frequency_nadir_hz": metrics["frequency_nadir_hz"],
        "frequency_overshoot_hz": metrics["frequency_overshoot_hz"],
        "min_feeder_voltage_pu": metrics["min_feeder_voltage_pu"],
        "max_q_kvar": metrics["max_aggregate_q_kvar"],
        "max_unit_apparent_kva": metrics["max_unit_apparent_kva"],
        "max_utilization": float(np.max(utilization)),
        "max_active_ceiling_loss_kw_per_iac": float(np.max(ceiling_loss)),
        "mean_active_ceiling_loss_kw_per_iac_during_event": float(np.mean(ceiling_loss[event])),
        "samples_soc_binding": int(np.count_nonzero(binding)),
        "samples_material_headroom_loss": int(np.count_nonzero(material)),
        "samples_event_material_headroom_loss": int(np.count_nonzero(material & event)),
        "min_p_ceiling_during_event_kw": float(np.min(p_ceiling[event])),
        "min_q_ceiling_during_event_kvar": float(np.min(q_ceiling[event])),
        "comfort_violation_count": metrics["comfort_violation_count"],
    }
    return row


def write_summary_csv(rows: list[dict[str, float | str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_latex_table(rows: list[dict[str, float | str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "\\begin{tabular}{lrrrrrr}",
        "\\toprule",
        "Case & $S$ & $\\min V$ & $f_{\\min}$ & $Q_{\\max}$ & max util. & max $\\Delta P_{\\max}(Q)$ \\\\",
        " & kVA & pu & Hz & kvar & pu & kW/IAC \\\\",
        "\\midrule",
    ]
    for row in rows:
        lines.append(
            f"{row['label']} & "
            f"{float(row['s_rated_kva']):.1f} & "
            f"{float(row['min_feeder_voltage_pu']):.4f} & "
            f"{float(row['frequency_nadir_hz']):.3f} & "
            f"{float(row['max_q_kvar']):.0f} & "
            f"{float(row['max_utilization']):.3f} & "
            f"{float(row['max_active_ceiling_loss_kw_per_iac']):.2f} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def plot_results(logs: dict[str, dict[str, np.ndarray]], rows: list[dict[str, float | str]], pdf_path: Path) -> None:
    inverter = InverterParameters()
    colors = {
        "finite_2p6kva": "#0072B2",
        "nominal_4p0kva": "#D55E00",
        "relaxed_6p0kva": "#009E73",
    }
    labels = {str(row["case_id"]): str(row["label"]) for row in rows}
    fig, axes = plt.subplots(5, 1, figsize=(7.2, 9.4), sharex=True)
    for case_id, log in logs.items():
        s_rated = float(next(row["s_rated_kva"] for row in rows if row["case_id"] == case_id))
        time_min = log["time_s"] / 60.0
        p_ceiling = active_ceiling_for_q(log["q_actual_kvar"], s_rated, InverterParameters(s_rated_kva=s_rated))
        utilization = log["apparent_kva"] / s_rated
        axes[0].plot(time_min, log["p_actual_kw"], color=colors[case_id], label=labels[case_id])
        axes[0].plot(time_min, p_ceiling, color=colors[case_id], linestyle="--", linewidth=1.2)
        axes[1].plot(time_min, log["q_actual_kvar"], color=colors[case_id])
        axes[2].plot(time_min, utilization, color=colors[case_id])
        axes[3].plot(time_min, log["frequency_hz"], color=colors[case_id])
        axes[4].plot(time_min, log["min_feeder_voltage_pu"], color=colors[case_id])

    for ax in axes:
        ax.axvspan(3.0, 6.3333, color="#DDDDDD", alpha=0.28, linewidth=0.0)
        ax.grid(True, color="#E6E6E6", linewidth=0.7)
    axes[0].axhline(inverter.p_max_kw, color="0.25", linestyle=":", linewidth=1.0)
    axes[2].axhline(1.0, color="0.25", linestyle=":", linewidth=1.0)
    axes[3].axhline(60.0, color="0.25", linestyle=":", linewidth=1.0)
    axes[4].axhline(0.95, color="0.25", linestyle=":", linewidth=1.0)

    axes[0].set_ylabel("P and Pmax(Q)\n(kW/IAC)")
    axes[1].set_ylabel("Q\n(kvar/IAC)")
    axes[2].set_ylabel("$\\sqrt{P^2+Q^2}/S$")
    axes[3].set_ylabel("Frequency\n(Hz)")
    axes[4].set_ylabel("Worst V\n(pu)")
    axes[4].set_xlabel("Time (min)")
    axes[0].legend(loc="upper right", frameon=False, ncol=1)
    fig.tight_layout()
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(pdf_path)
    fig.savefig(pdf_path.with_suffix(".png"), dpi=180)
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    scenarios = []
    base = SimulationScenario(
        name="stage01_headroom_ablation",
        voltage_event_type="feeder_load_step",
        feeder_extra_load_kw=1200.0,
        feeder_extra_load_kvar=600.0,
        voltage_sag_pu=1.0,
    )
    for case_id, s_rated, _label in CASES:
        scenarios.append(replace(base, name=case_id, inverter_s_rated_kva=s_rated, inverter_q_max_kvar=2.2))
    scenarios = apply_thermal_calibration(scenarios, CALIBRATION_JSON)

    logs: dict[str, dict[str, np.ndarray]] = {}
    rows: list[dict[str, float | str]] = []
    for scenario, (case_id, s_rated, label) in zip(scenarios, CASES):
        csv_path = OUTPUT_DIR / "cases" / case_id / f"{case_id}__dmpc_full_pq.csv"
        if csv_path.exists():
            print(f"Loading existing {label} full P/Q case...")
            log = read_log(csv_path)
        else:
            print(f"Running {label} full P/Q case...")
            log = run_simulation(scenario, controller_mode="dmpc_full_pq")
            write_log(log, csv_path)
        logs[case_id] = log
        rows.append(summarize_case(label, case_id, s_rated, log))

    write_summary_csv(rows, OUTPUT_DIR / "headroom_ablation_summary.csv")
    write_latex_table(rows, OUTPUT_DIR / "headroom_tradeoff_ablation.tex")
    plot_results(logs, rows, OUTPUT_DIR / "headroom_tradeoff_ablation.pdf")
    print(f"Wrote {OUTPUT_DIR / 'headroom_ablation_summary.csv'}")
    print(f"Wrote {OUTPUT_DIR / 'headroom_tradeoff_ablation.pdf'}")
    print(f"Wrote {OUTPUT_DIR / 'headroom_tradeoff_ablation.tex'}")


if __name__ == "__main__":
    main()
