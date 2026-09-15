"""Generate journal manuscript figures and LaTeX tables from experiment outputs."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import shutil
import sys

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from iac_dmpc.plot_style import apply_elsevier_figure_style


PAPER = ROOT / "paper"
FIGURES = PAPER / "figures"
TABLES = PAPER / "tables"
NIST_EXPERIMENT_ROOT = ROOT / "experiments" / "nist_measured_calibrated"
RESSTOCK_EXPERIMENT_ROOT = ROOT / "experiments" / "dynamic_replay_calibrated"
FULL = NIST_EXPERIMENT_ROOT / "full" if (NIST_EXPERIMENT_ROOT / "full").exists() else RESSTOCK_EXPERIMENT_ROOT / "full"
SUMMARY = (
    ROOT / "paper_outputs_nist_measured_calibrated" / "controller_summary.csv"
    if (ROOT / "paper_outputs_nist_measured_calibrated" / "controller_summary.csv").exists()
    else ROOT / "paper_outputs_dynamic_replay_calibrated" / "controller_summary.csv"
)
CALIBRATION = (
    ROOT / "data" / "nist_nzertf" / "calibrated_thermal_parameters.json"
    if (ROOT / "data" / "nist_nzertf" / "calibrated_thermal_parameters.json").exists()
    else ROOT / "data" / "resstock" / "calibrated_thermal_parameters.json"
)

CONTROLLERS = ["no_support", "rule_based", "dmpc_active_only", "dmpc_pq", "dmpc_full_pq"]
LABELS = {
    "no_support": "No support",
    "rule_based": "Rule based",
    "dmpc_active_only": "MPC-P",
    "dmpc_pq": "MPC-P+VV",
    "dmpc_full_pq": "Full P/Q MPC",
}
COLORS = {
    "no_support": "#4d4d4d",
    "rule_based": "#1f77b4",
    "dmpc_active_only": "#2ca02c",
    "dmpc_pq": "#ff7f0e",
    "dmpc_full_pq": "#d62728",
}


def read_case(scenario: str, controller: str) -> dict[str, np.ndarray]:
    path = FULL / "cases" / f"{scenario}__{controller}" / f"{scenario}__{controller}.csv"
    if not path.exists():
        raise FileNotFoundError(
            "\nMissing representative experiment trace needed to regenerate paper assets:\n"
            f"  {path}\n\n"
            "The Mendeley package includes these representative traces in the expected location. "
            "If you are working from a source checkout without experiment folders, regenerate them with:\n"
            "  python generate_dynamic_frequency_trace.py\n"
            "  python run_experiment_suite.py --scenario-set full "
            "--thermal-calibration-json data/nist_nzertf/calibrated_thermal_parameters.json "
            "--output-dir experiments/nist_measured_calibrated\n"
            "Then rerun:\n"
            "  python paper/generate_paper_assets.py\n"
        )
    data = np.genfromtxt(path, delimiter=",", names=True)
    return {name: np.asarray(data[name]) for name in data.dtype.names}


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def setup_style() -> None:
    apply_elsevier_figure_style(base_size=8.0)


def savefig(fig: plt.Figure, name: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / f"{name}.pdf")
    fig.savefig(FIGURES / f"{name}.png", dpi=350)
    plt.close(fig)


def plot_architecture() -> None:
    fig, ax = plt.subplots(figsize=(7.1, 2.65), constrained_layout=True)
    ax.set_axis_off()
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)

    boxes = {
        "meas": (0.04, 0.58, 0.16, 0.20, "Grid\nmeasurements"),
        "pll": (0.25, 0.58, 0.16, 0.20, "SOGI-PLL"),
        "dq": (0.46, 0.58, 0.16, 0.20, "P/Q\ncurrent split"),
        "ctrl": (0.67, 0.58, 0.18, 0.20, "Rule-based /\nfleet MPC"),
        "feeder": (0.67, 0.17, 0.18, 0.20, "OpenDSS\nIEEE 13 feeder"),
        "thermal": (0.46, 0.17, 0.16, 0.20, "Thermal /\ncompressor state"),
    }
    for x, y, w, h, label in boxes.values():
        ax.add_patch(Rectangle((x, y), w, h, facecolor="white", edgecolor="black", linewidth=0.9))
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center")

    def arrow(start: tuple[float, float], end: tuple[float, float]) -> None:
        ax.add_patch(
            FancyArrowPatch(
                start,
                end,
                arrowstyle="-|>",
                mutation_scale=9,
                linewidth=0.9,
                color="black",
                shrinkA=3,
                shrinkB=3,
            )
        )

    arrow((0.20, 0.68), (0.25, 0.68))
    arrow((0.41, 0.68), (0.46, 0.68))
    arrow((0.62, 0.68), (0.67, 0.68))
    arrow((0.76, 0.58), (0.76, 0.37))
    arrow((0.67, 0.27), (0.62, 0.27))
    arrow((0.54, 0.37), (0.54, 0.58))

    ax.text(0.76, 0.88, "P/Q commands", ha="center", va="bottom")
    ax.text(0.32, 0.88, "Estimated phase and dq currents", ha="center", va="bottom")
    ax.text(0.54, 0.05, "Comfort and compressor limits", ha="center", va="bottom")
    savefig(fig, "architecture")


def plot_time_response(scenario: str, name: str, title: str) -> None:
    logs = {controller: read_case(scenario, controller) for controller in CONTROLLERS}
    fig, axes = plt.subplots(4, 1, figsize=(7.1, 6.35), sharex=True, constrained_layout=True)

    for controller, log in logs.items():
        t = log["time_s"] / 60.0
        axes[0].plot(t, log["frequency_hz"], color=COLORS[controller], label=LABELS[controller])
        axes[1].plot(t, log["min_feeder_voltage_pu"], color=COLORS[controller])
        axes[2].plot(t, log["temperature_c"], color=COLORS[controller])

    full = logs["dmpc_full_pq"]
    t = full["time_s"] / 60.0
    axes[3].plot(t, full["aggregate_p_kw"], color="#2ca02c", label="Aggregate P")
    axes[3].plot(t, full["aggregate_q_kvar"], color="#d62728", label="Aggregate Q")

    axes[0].axhline(60.0, color="black", linestyle="--", linewidth=0.8)
    axes[1].axhline(0.95, color="black", linestyle="--", linewidth=0.8)
    axes[2].axhline(23.0, color="black", linestyle="--", linewidth=0.8)
    axes[2].axhline(25.0, color="black", linestyle="--", linewidth=0.8)
    axes[0].set_ylabel("Frequency (Hz)")
    axes[1].set_ylabel("Worst V (pu)")
    axes[2].set_ylabel("Temp. (deg C)")
    axes[3].set_ylabel("Fleet P/Q (kW/kvar)")
    axes[3].set_xlabel("Time (min)")
    axes[0].set_title(title)
    axes[0].legend(ncol=3, loc="lower right")
    axes[3].legend(ncol=2, loc="best")
    savefig(fig, name)


def plot_controller_summary() -> None:
    rows = {row["controller"]: row for row in read_rows(SUMMARY)}
    metrics = [
        ("frequency_nadir_hz_mean", "Mean frequency nadir (Hz)"),
        ("min_feeder_voltage_pu_mean", "Mean worst voltage (pu)"),
        ("reactive_support_kvarh_mean", "Reactive support (kvarh)"),
        ("mean_compute_time_ms_mean", "Mean solve time (ms)"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(7.1, 5.35), constrained_layout=True)
    x = np.arange(len(CONTROLLERS))
    for index, (axis, (metric, ylabel)) in enumerate(zip(axes.ravel(), metrics)):
        values = [float(rows[controller][metric]) for controller in CONTROLLERS]
        axis.bar(x, values, color=[COLORS[controller] for controller in CONTROLLERS], width=0.72)
        axis.set_ylabel(ylabel)
        axis.set_xticks(x)
        axis.set_xticklabels([LABELS[c] for c in CONTROLLERS], rotation=28, ha="right")
        axis.tick_params(axis="x", pad=1)
        if index < 2:
            axis.tick_params(axis="x", labelbottom=False)
        if metric == "min_feeder_voltage_pu_mean":
            axis.axhline(0.95, color="black", linestyle="--", linewidth=0.8)
    fig.set_constrained_layout_pads(w_pad=0.04, h_pad=0.05, hspace=0.18, wspace=0.10)
    savefig(fig, "controller_summary")


def plot_headroom_sensitivity() -> None:
    rows = read_rows(FULL / "metrics_summary.csv")
    selected = [
        row
        for row in rows
        if row["scenario"].startswith("headroom_") and row["controller"] in {"rule_based", "dmpc_pq", "dmpc_full_pq"}
    ]
    fig, ax = plt.subplots(figsize=(3.5, 2.55))
    for controller in ["rule_based", "dmpc_pq", "dmpc_full_pq"]:
        sub = sorted((row for row in selected if row["controller"] == controller), key=lambda r: float(r["inverter_s_rated_kva"]))
        x = [float(row["inverter_s_rated_kva"]) for row in sub]
        y = [float(row["min_pcc_voltage_pu"]) for row in sub]
        ax.plot(x, y, marker="o", color=COLORS[controller], label=LABELS[controller])
    ax.axhline(0.95, color="black", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Per-unit inverter rating (kVA)")
    ax.set_ylabel("Minimum fleet voltage (pu)")
    ax.set_title("Source-sag headroom sensitivity")
    ax.legend(loc="lower right")
    savefig(fig, "headroom_sensitivity")


def latex_escape(text: str) -> str:
    return text.replace("_", "\\_")


def write_table(path: Path, lines: list[str]) -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_controller_table() -> None:
    rows = {row["controller"]: row for row in read_rows(SUMMARY)}
    case_count = int(float(next(iter(rows.values()))["case_count"]))
    calibrated = json.loads(CALIBRATION.read_text(encoding="utf-8")) if CALIBRATION.exists() else {}
    caption_source = "measured-data-calibrated" if calibration_kind(calibrated) == "nist" else "calibrated"
    lines = [
        "\\begin{table}[!t]",
        f"\\caption{{Aggregate controller performance over {case_count} {caption_source} scenarios}}",
        "\\label{tab:controller_summary}",
        "\\centering",
        "\\begin{tabular}{lccccc}",
        "\\toprule",
        "Controller & $f_{\\min}$ & $V_{\\min}$ & $V_{\\mathrm{PCC}}$ & Comfort & Solve \\\\",
        " & (Hz) & (pu) & (pu) & viol. & (ms) \\\\",
        "\\midrule",
    ]
    for controller in CONTROLLERS:
        row = rows[controller]
        lines.append(
            f"{LABELS[controller]} & "
            f"{float(row['frequency_nadir_hz_mean']):.3f} & "
            f"{float(row['min_feeder_voltage_pu_mean']):.4f} & "
            f"{float(row['min_pcc_voltage_pu_mean']):.4f} & "
            f"{float(row['comfort_violation_count_mean']):.1f} & "
            f"{float(row['mean_compute_time_ms_mean']):.1f} \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    write_table(TABLES / "controller_summary.tex", lines)


def make_selected_case_table() -> None:
    rows = read_rows(FULL / "metrics_summary.csv")
    scenarios = ["voltage_sag_0.92", "feeder_stress_1200kw_600kvar", "headroom_1.50pu", "tcl_share_40pct"]
    lines = [
        "\\begin{table*}[!t]",
        "\\caption{Representative scenario results from the upgraded full experiment suite}",
        "\\label{tab:representative_cases}",
        "\\centering",
        "\\resizebox{\\textwidth}{!}{%",
        "\\begin{tabular}{llcccc}",
        "\\toprule",
        "Scenario & Controller & $f_{\\min}$ (Hz) & $V_{\\min}$ (pu) & $V_{\\mathrm{PCC}}$ (pu) & $Q_{\\max}$ (kvar) \\\\",
        "\\midrule",
    ]
    for scenario in scenarios:
        for controller in CONTROLLERS:
            row = next(row for row in rows if row["scenario"] == scenario and row["controller"] == controller)
            lines.append(
                f"{latex_escape(scenario)} & {LABELS[controller]} & "
                f"{float(row['frequency_nadir_hz']):.3f} & "
                f"{float(row['min_feeder_voltage_pu']):.4f} & "
                f"{float(row['min_pcc_voltage_pu']):.4f} & "
                f"{float(row['max_aggregate_q_kvar']):.1f} \\\\"
            )
        lines.append("\\addlinespace")
    lines += ["\\bottomrule", "\\end{tabular}%", "}", "\\end{table*}"]
    write_table(TABLES / "representative_cases.tex", lines)


def calibration_kind(payload: dict[str, object]) -> str:
    source = str(payload.get("source", "")).lower()
    return "nist" if "nist" in source else "resstock"


def make_parameter_table() -> None:
    calibrated = json.loads(CALIBRATION.read_text(encoding="utf-8")) if CALIBRATION.exists() else {}
    thermal = calibrated.get("thermal_parameters", {})
    caption_source = "NIST measured-data" if calibration_kind(calibrated) == "nist" else "ResStock"
    lines = [
        "\\begin{table}[!t]",
        f"\\caption{{Simulation and controller parameters after {caption_source} calibration}}",
        "\\label{tab:parameters}",
        "\\centering",
        "\\begin{tabular}{lll}",
        "\\toprule",
        "Quantity & Symbol & Value \\\\",
        "\\midrule",
        "Nominal frequency & $f_0$ & 60 Hz \\\\",
        "MPC sample time & $T_s$ & 5 s \\\\",
        "MPC horizon & $M$ & 12 steps \\\\",
        "IAC set power & $P_{set}$ & 2.0 kW/unit \\\\",
        "Compressor range & $P_{min},P_{max}$ & 0.6, 3.5 kW \\\\",
        "Inverter rating & $S_{rated}$ & 4.0 kVA/unit \\\\",
        "Reactive limit & $Q_{max}$ & 2.2 kvar/unit \\\\",
        "Current-loop bandwidth & $f_{ci}$ & 18 Hz \\\\",
        f"Thermal setpoint & $T_{{set}}$ & {float(thermal.get('setpoint_c', 24.0)):.2f} $^\\circ$C \\\\",
        f"Thermal resistance & $R_{{th}}$ & {float(thermal.get('resistance_c_per_kw', 4.0 / 3.0)):.2f} $^\\circ$C/kW \\\\",
        f"Thermal capacitance & $C_{{th}}$ & {float(thermal.get('capacitance_kwh_per_c', 2.5)):.1f} kWh/$^\\circ$C \\\\",
        f"Nominal COP & COP & {float(thermal.get('cop', 3.0)):.2f} \\\\",
        "Comfort band & $\\Delta T$ & $\\pm$1 $^\\circ$C nominal \\\\",
        "Nominal fleet size & $N$ & 500 units \\\\",
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ]
    write_table(TABLES / "parameters.tex", lines)


def make_calibration_table() -> None:
    if not CALIBRATION.exists():
        return
    payload = json.loads(CALIBRATION.read_text(encoding="utf-8"))
    kind = calibration_kind(payload)
    metadata = payload["metadata"]
    fit = payload["fit_quality"]
    thermal = payload["thermal_parameters"]
    if kind == "nist":
        lines = [
            "\\begin{table}[!t]",
            "\\caption{NIST NZERTF measured-data thermal calibration}",
            "\\label{tab:nist_calibration}",
            "\\centering",
            "\\begin{tabular}{ll}",
            "\\toprule",
            "Quantity & Value \\\\",
            "\\midrule",
            f"Measured data year & {payload['year']} \\\\",
            f"Fit window & {payload['fit_window']['start'][:10]}--{payload['fit_window']['end'][:10]} \\\\",
            f"Samples / cooling samples & {fit['sample_count']} / {fit['active_cooling_sample_count']} \\\\",
            f"Indoor-temperature RMSE & {fit['indoor_temperature_rmse_c']:.3f} $^\\circ$C \\\\",
            f"Active-cooling RMSE & {fit['active_cooling_rmse_c']:.3f} $^\\circ$C \\\\",
            f"$R_{{th}}$, $C_{{th}}$ & {thermal['resistance_c_per_kw']:.3f} $^\\circ$C/kW, {thermal['capacitance_kwh_per_c']:.1f} kWh/$^\\circ$C \\\\",
            f"COP and lift slope & {thermal['cop']:.3f}, {thermal['cop_temperature_slope_per_c']:.4f} per $^\\circ$C \\\\",
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table}",
        ]
        write_table(TABLES / "nist_calibration.tex", lines)
        return
    lines = [
        "\\begin{table}[!t]",
        "\\caption{ResStock/EnergyPlus calibration case and fitted parameters}",
        "\\label{tab:resstock_calibration}",
        "\\centering",
        "\\begin{tabular}{ll}",
        "\\toprule",
        "Quantity & Value \\\\",
        "\\midrule",
        f"Dataset building & {payload['state']} {payload['building_id']} \\\\",
        f"Location/weather & {latex_escape(metadata['city'])}; {latex_escape(metadata['weather_file_city'])} \\\\",
        f"Building type & {latex_escape(metadata['building_type'])}, {metadata['sqft']:.0f} ft$^2$ \\\\",
        f"Cooling system & {latex_escape(metadata['cooling_type'])}, {latex_escape(metadata['cooling_efficiency'])} \\\\",
        f"Fit window & {payload['fit_window']['start'][:10]}--{payload['fit_window']['end'][:10]} \\\\",
        f"Indoor-temperature RMSE & {fit['indoor_temperature_rmse_c']:.3f} $^\\circ$C \\\\",
        f"$R_{{th}}$, $C_{{th}}$ & {thermal['resistance_c_per_kw']:.3f} $^\\circ$C/kW, {thermal['capacitance_kwh_per_c']:.1f} kWh/$^\\circ$C \\\\",
        f"COP and lift slope & {thermal['cop']:.3f}, {thermal['cop_temperature_slope_per_c']:.4f} per $^\\circ$C \\\\",
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ]
    write_table(TABLES / "resstock_calibration.tex", lines)


def make_opf_comparison_table() -> None:
    path = ROOT / "experiments" / "opf_comparison" / "opf_comparison.csv"
    if not path.exists():
        return
    rows = read_rows(path)
    lines = [
        "\\begin{table}[!t]",
        "\\caption{Sensitivity OPF prediction versus OpenDSS power-flow check}",
        "\\label{tab:opf_comparison}",
        "\\centering",
        "\\begin{tabular}{lccc}",
        "\\toprule",
        "Scenario & $V_{OPF}$ & $V_{DSS}$ & Error \\\\",
        " & (pu) & (pu) & (pu) \\\\",
        "\\midrule",
    ]
    for row in rows:
        scenario_label = {
            "source_sag_0.92": "Source sag",
            "feeder_stress_1200kw_600kvar": "Feeder stress",
        }.get(row["scenario"], row["scenario"])
        lines.append(
            f"{latex_escape(scenario_label)} & "
            f"{float(row['opf_predicted_voltage_pu']):.4f} & "
            f"{float(row['opendss_actual_voltage_pu']):.4f} & "
            f"{float(row['prediction_error_pu']):+.4f} \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    write_table(TABLES / "opf_comparison.tex", lines)


def main() -> None:
    setup_style()
    plot_architecture()
    if (ROOT / "data" / "nist_nzertf" / "nist_nzertf_calibration_fit.png").exists():
        FIGURES.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(
            ROOT / "data" / "nist_nzertf" / "nist_nzertf_calibration_fit.png",
            FIGURES / "nist_nzertf_calibration_fit.png",
        )
    if (ROOT / "data" / "resstock" / "resstock_calibration_fit.png").exists():
        FIGURES.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / "data" / "resstock" / "resstock_calibration_fit.png", FIGURES / "resstock_calibration_fit.png")
    plot_time_response("voltage_sag_0.92", "source_sag_response", "Upstream source-sag event")
    plot_time_response("feeder_stress_1200kw_600kvar", "feeder_stress_response", "Feeder-local load stress event")
    plot_controller_summary()
    plot_headroom_sensitivity()
    make_controller_table()
    make_selected_case_table()
    make_parameter_table()
    make_calibration_table()
    make_opf_comparison_table()
    print(f"Wrote figures to {FIGURES}")
    print(f"Wrote tables to {TABLES}")


if __name__ == "__main__":
    main()
