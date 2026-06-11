"""ANDES nonlinear transient-stability validation for IAC frequency support.

The script builds controller-specific IEEE 14-bus dynamic cases from ANDES'
bundled benchmark and injects aggregate IAC active-power relief through timed
TGOV1 auxiliary-power ``Alter`` events.  It then runs ANDES time-domain
simulation, extracts center-of-inertia frequency from GENROU speeds, compares
against the existing dynamic replay trace, and writes manuscript-ready assets.
"""

from __future__ import annotations

import argparse
import csv
import math
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from iac_dmpc.plot_style import apply_elsevier_figure_style


ROOT = Path(__file__).resolve().parent
DEFAULT_EXPERIMENT_ROOT = (
    ROOT / "experiments" / "nist_measured_calibrated" / "full"
    if (ROOT / "experiments" / "nist_measured_calibrated" / "full").exists()
    else ROOT / "experiments" / "dynamic_replay_calibrated" / "full"
)
DEFAULT_OUTPUT_DIR = ROOT / "experiments" / "andes_validation"
DEFAULT_PAPER_FIGURE_DIR = ROOT / "paper" / "figures"
DEFAULT_PAPER_TABLE_DIR = ROOT / "paper" / "tables"
CONTROLLERS = ("no_support", "dmpc_active_only", "dmpc_full_pq")
CONTROLLER_LABELS = {
    "no_support": "No support",
    "dmpc_active_only": "DMPC-P",
    "dmpc_full_pq": "Full P/Q DMPC",
}


@dataclass(frozen=True)
class AndesValidationConfig:
    scenario: str = "voltage_sag_0.92"
    andes_case: str = "ieee14/ieee14_alter.xlsx"
    replay_window_start_s: float = 115.0
    validation_tf_s: float = 45.0
    disturbance_time_s: float = 10.0
    tstep_s: float = 1.0 / 30.0
    frequency_base_hz: float = 60.0
    transmission_base_kw: float = 100_000.0
    paux_disturbance_pu: float = 0.015
    governor_device: str = "TGOV1_1"


def load_andes_case_path(case_name: str) -> Path | None:
    try:
        import andes  # type: ignore
    except ImportError:
        return None
    return Path(andes.get_case(case_name))


def read_controller_trace(experiment_root: Path, scenario: str, controller: str) -> pd.DataFrame:
    case_dir = experiment_root / "cases" / f"{scenario}__{controller}"
    csv_path = case_dir / f"{scenario}__{controller}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            "\nMissing controller trace for nonlinear/T&D validation:\n"
            f"  {csv_path}\n\n"
            "The Mendeley research-data package should include the representative "
            "voltage-sag traces needed by the default validation commands. If you are "
            "working from a source checkout without experiment outputs, regenerate them with:\n"
            "  python generate_dynamic_frequency_trace.py\n"
            "  python run_experiment_suite.py --scenario-set full "
            "--thermal-calibration-json data/nist_nzertf/calibrated_thermal_parameters.json "
            "--output-dir experiments/nist_measured_calibrated\n\n"
            "You can also point this script to any compatible experiment folder with:\n"
            "  --experiment-root <path-to-scenario-set-folder>\n"
        )
    return pd.read_csv(csv_path)


def build_alter_schedule(
    trace: pd.DataFrame,
    cfg: AndesValidationConfig,
    controller: str,
) -> pd.DataFrame:
    window = trace[
        (trace["time_s"] >= cfg.replay_window_start_s)
        & (trace["time_s"] <= cfg.replay_window_start_s + cfg.validation_tf_s)
    ].copy()
    if window.empty:
        raise ValueError("Replay window does not overlap the controller trace.")

    base_kw = float(trace.loc[trace["time_s"].idxmin(), "aggregate_p_kw"])
    if controller == "no_support":
        support_kw = np.zeros(len(window))
    else:
        support_kw = np.maximum(0.0, base_kw - window["aggregate_p_kw"].to_numpy(dtype=float))
    support_pu = support_kw / cfg.transmission_base_kw
    local_time = window["time_s"].to_numpy(dtype=float) - cfg.replay_window_start_s
    disturbance = np.where(local_time >= cfg.disturbance_time_s, cfg.paux_disturbance_pu, 0.0)
    paux_pu = support_pu - disturbance

    rows = []
    used_times: set[float] = set()
    for i, (t_s, paux, support) in enumerate(zip(local_time, paux_pu, support_pu), start=1):
        t_event = max(0.1, round(float(t_s), 6))
        while t_event in used_times:
            t_event = round(t_event + 1e-4, 6)
        used_times.add(t_event)
        rows.append(
            {
                "uid": i - 1,
                "idx": i,
                "u": 1,
                "name": f"{controller}_paux_{i}",
                "t": t_event,
                "model": "TGOV1",
                "dev": cfg.governor_device,
                "src": "paux0",
                "attr": "v",
                "method": "=",
                "amount": float(paux),
                "rand": 0,
                "lb": 0,
                "ub": 0,
                "support_pu": float(support),
            }
        )
    return pd.DataFrame(rows)


def write_andes_case(base_case: Path, alter_schedule: pd.DataFrame, out_case: Path) -> None:
    xls = pd.ExcelFile(base_case)
    out_case.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(out_case, engine="openpyxl") as writer:
        for sheet in xls.sheet_names:
            if sheet == "Alter":
                alter_schedule.drop(columns=["support_pu"]).to_excel(writer, sheet_name=sheet, index=False)
            else:
                pd.read_excel(base_case, sheet_name=sheet).to_excel(writer, sheet_name=sheet, index=False)


def run_andes(case_path: Path, output_prefix: Path, cfg: AndesValidationConfig) -> None:
    cmd = [
        sys.executable,
        "-m",
        "andes",
        "-v",
        "20",
        "run",
        str(case_path),
        "-r",
        "tds",
        "-o",
        str(output_prefix),
        "--tf",
        f"{cfg.validation_tf_s:.6g}",
        "--no-pbar",
        "-O",
        f"TDS.tstep={cfg.tstep_s:.8g}",
    ]
    subprocess.run(cmd, cwd=ROOT, check=True)


def read_lst_labels(lst_path: Path) -> dict[int, str]:
    labels: dict[int, str] = {}
    with lst_path.open() as f:
        for row in csv.reader(f):
            if len(row) >= 2:
                try:
                    labels[int(row[0].strip())] = row[1].strip()
                except ValueError:
                    continue
    return labels


def extract_andes_frequency(output_dir: Path, stem: str, cfg: AndesValidationConfig, inertia_weights: np.ndarray) -> pd.DataFrame:
    lst_path = output_dir / f"{stem}_out.lst"
    npz_path = output_dir / f"{stem}_out.npz"
    labels = read_lst_labels(lst_path)
    omega_cols = [idx for idx, label in labels.items() if label.startswith("omega GENROU")]
    if not omega_cols:
        raise RuntimeError(f"No GENROU omega variables found in {lst_path}")

    data = np.load(npz_path)["data"]
    weights = inertia_weights[: len(omega_cols)].astype(float)
    weights = weights / weights.sum()
    omega_coi = np.average(data[:, omega_cols], axis=1, weights=weights)
    return pd.DataFrame(
        {
            "time_s": data[:, 0],
            "andes_frequency_hz": cfg.frequency_base_hz * omega_coi,
        }
    )


def get_inertia_weights(base_case: Path) -> np.ndarray:
    gen = pd.read_excel(base_case, sheet_name="GENROU")
    if "M" not in gen:
        return np.ones(len(gen), dtype=float)
    values = pd.to_numeric(gen["M"], errors="coerce").fillna(1.0).to_numpy(dtype=float)
    return np.where(values > 0, values, 1.0)


def replay_window(trace: pd.DataFrame, cfg: AndesValidationConfig) -> pd.DataFrame:
    window = trace[
        (trace["time_s"] >= cfg.replay_window_start_s)
        & (trace["time_s"] <= cfg.replay_window_start_s + cfg.validation_tf_s)
    ].copy()
    window["time_s"] = window["time_s"] - cfg.replay_window_start_s
    return window[["time_s", "frequency_hz", "aggregate_p_kw"]].rename(
        columns={"frequency_hz": "replay_frequency_hz"}
    )


def merge_validation_trace(
    andes_df: pd.DataFrame,
    replay_df: pd.DataFrame,
    schedule: pd.DataFrame,
    controller: str,
) -> pd.DataFrame:
    replay_freq = np.interp(andes_df["time_s"], replay_df["time_s"], replay_df["replay_frequency_hz"])
    agg_p = np.interp(andes_df["time_s"], replay_df["time_s"], replay_df["aggregate_p_kw"])
    paux = np.interp(andes_df["time_s"], schedule["t"], schedule["amount"])
    support = np.interp(andes_df["time_s"], schedule["t"], schedule["support_pu"])
    out = andes_df.copy()
    out["controller"] = controller
    out["controller_label"] = CONTROLLER_LABELS[controller]
    out["replay_frequency_hz"] = replay_freq
    out["aggregate_p_kw"] = agg_p
    out["paux_pu"] = paux
    out["iac_support_pu"] = support
    return out


def metrics_for_trace(df: pd.DataFrame, cfg: AndesValidationConfig) -> dict[str, float | str]:
    t = df["time_s"].to_numpy(dtype=float)
    f = df["andes_frequency_hz"].to_numpy(dtype=float)
    replay_f = df["replay_frequency_hz"].to_numpy(dtype=float)
    rocof = np.gradient(f, t)
    disturbance_mask = t >= cfg.disturbance_time_s
    support_kw = df["iac_support_pu"].to_numpy(dtype=float) * cfg.transmission_base_kw
    return {
        "controller": str(df["controller"].iloc[0]),
        "controller_label": str(df["controller_label"].iloc[0]),
        "andes_nadir_hz": float(np.min(f[disturbance_mask])),
        "replay_nadir_hz": float(np.min(replay_f[disturbance_mask])),
        "nadir_abs_error_hz": float(abs(np.min(f[disturbance_mask]) - np.min(replay_f[disturbance_mask]))),
        "andes_min_rocof_hz_per_s": float(np.min(rocof[disturbance_mask])),
        "max_iac_relief_kw": float(np.max(support_kw)),
        "relief_at_disturbance_kw": float(np.interp(cfg.disturbance_time_s, t, support_kw)),
    }


def write_latex_table(metrics: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for _, row in metrics.iterrows():
        rows.append(
            f"{row['controller_label']} & "
            f"{row['andes_nadir_hz']:.3f} & "
            f"{row['replay_nadir_hz']:.3f} & "
            f"{row['nadir_abs_error_hz']:.3f} & "
            f"{row['relief_at_disturbance_kw']:.0f} \\\\"
        )
    body = "\n".join(rows)
    path.write_text(
        "\\begin{table}[!t]\n"
        "\\centering\n"
        "\\footnotesize\n"
        "\\setlength{\\tabcolsep}{3.5pt}\n"
        "\\caption{Nonlinear ANDES IEEE 14-bus transient-stability validation of active-power frequency support.}\n"
        "\\label{tab:andes_validation}\n"
        "\\begin{tabular}{lrrrr}\n"
        "\\toprule\n"
        "Controller & ANDES nadir & Replay nadir & Error & IAC relief \\\\\n"
        " & (Hz) & (Hz) & (Hz) & (kW) \\\\\n"
        "\\midrule\n"
        f"{body}\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\end{table}\n",
        encoding="utf-8",
    )


def plot_validation(traces: pd.DataFrame, output_path: Path) -> None:
    apply_elsevier_figure_style(base_size=8.0)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    colors = {
        "no_support": "#4D4D4D",
        "dmpc_active_only": "#0072B2",
        "dmpc_full_pq": "#D55E00",
    }
    for controller in CONTROLLERS:
        df = traces[traces["controller"] == controller]
        label = CONTROLLER_LABELS[controller]
        ax.plot(df["time_s"], df["andes_frequency_hz"], color=colors[controller], lw=2.0, label=f"{label} ANDES")
        ax.plot(
            df["time_s"],
            df["replay_frequency_hz"],
            color=colors[controller],
            lw=1.5,
            ls="--",
            alpha=0.75,
            label=f"{label} replay",
        )
    ax.axvline(10.0, color="0.25", ls=":", lw=1.0)
    ax.set_xlabel("Validation time (s)")
    ax.set_ylabel("Frequency (Hz)")
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=2, fontsize=7.5)
    fig.tight_layout()
    fig.savefig(output_path, dpi=250)
    fig.savefig(output_path.with_suffix(".pdf"))
    plt.close(fig)


def fallback_from_replay(
    experiment_root: Path,
    output_dir: Path,
    cfg: AndesValidationConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    traces = []
    for controller in CONTROLLERS:
        trace = read_controller_trace(experiment_root, cfg.scenario, controller)
        window = replay_window(trace, cfg)
        out = window.rename(columns={"replay_frequency_hz": "andes_frequency_hz"})
        out["replay_frequency_hz"] = out["andes_frequency_hz"]
        out["controller"] = controller
        out["controller_label"] = CONTROLLER_LABELS[controller]
        out["aggregate_p_kw"] = window["aggregate_p_kw"]
        out["paux_pu"] = 0.0
        out["iac_support_pu"] = 0.0
        traces.append(out)
        rows.append(metrics_for_trace(out, cfg))
    return pd.concat(traces, ignore_index=True), pd.DataFrame(rows)


def run_validation(args: argparse.Namespace) -> None:
    cfg = AndesValidationConfig(
        scenario=args.scenario,
        replay_window_start_s=args.replay_window_start_s,
        validation_tf_s=args.validation_tf_s,
        paux_disturbance_pu=args.paux_disturbance_pu,
    )
    experiment_root = Path(args.experiment_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    case_dir = output_dir / "cases"
    output_dir.mkdir(parents=True, exist_ok=True)

    andes_case = load_andes_case_path(cfg.andes_case)
    if andes_case is None:
        if not args.allow_fallback:
            raise RuntimeError("ANDES is not installed. Install optional validation dependencies or use --allow-fallback.")
        traces, metrics = fallback_from_replay(experiment_root, output_dir, cfg)
        (output_dir / "andes_status.txt").write_text(
            "ANDES import failed; wrote replay-only fallback traces.\n", encoding="utf-8"
        )
    else:
        inertia_weights = get_inertia_weights(andes_case)
        trace_parts = []
        metric_rows = []
        started = time.time()
        for i, controller in enumerate(CONTROLLERS, start=1):
            elapsed = time.time() - started
            avg = elapsed / max(i - 1, 1)
            remaining = avg * (len(CONTROLLERS) - i + 1)
            print(
                f"[{i}/{len(CONTROLLERS)}] {CONTROLLER_LABELS[controller]} "
                f"(estimated remaining {remaining / 60:.1f} min)",
                flush=True,
            )
            controller_trace = read_controller_trace(experiment_root, cfg.scenario, controller)
            schedule = build_alter_schedule(controller_trace, cfg, controller)
            case_path = case_dir / f"ieee14_{controller}.xlsx"
            write_andes_case(andes_case, schedule, case_path)
            prefix = output_dir / controller
            run_andes(case_path, prefix, cfg)
            andes_out_dir = output_dir / controller
            andes_df = extract_andes_frequency(andes_out_dir, case_path.stem, cfg, inertia_weights)
            replay_df = replay_window(controller_trace, cfg)
            merged = merge_validation_trace(andes_df, replay_df, schedule, controller)
            merged.to_csv(output_dir / f"{controller}_andes_frequency.csv", index=False)
            trace_parts.append(merged)
            metric_rows.append(metrics_for_trace(merged, cfg))
        traces = pd.concat(trace_parts, ignore_index=True)
        metrics = pd.DataFrame(metric_rows)
        (output_dir / "andes_status.txt").write_text(
            f"ANDES nonlinear validation completed with case {cfg.andes_case}.\n",
            encoding="utf-8",
        )

    traces.to_csv(output_dir / "andes_frequency_validation.csv", index=False)
    metrics.to_csv(output_dir / "andes_validation_metrics.csv", index=False)
    plot_validation(traces, output_dir / "andes_frequency_validation.png")
    write_latex_table(metrics, output_dir / "andes_validation.tex")

    paper_figure_dir = Path(args.paper_figure_dir).resolve()
    paper_table_dir = Path(args.paper_table_dir).resolve()
    paper_figure_dir.mkdir(parents=True, exist_ok=True)
    paper_table_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(output_dir / "andes_frequency_validation.png", paper_figure_dir / "andes_frequency_validation.png")
    shutil.copy2(output_dir / "andes_frequency_validation.pdf", paper_figure_dir / "andes_frequency_validation.pdf")
    shutil.copy2(output_dir / "andes_validation.tex", paper_table_dir / "andes_validation.tex")

    print(metrics.to_string(index=False), flush=True)
    print(f"Wrote validation traces to {output_dir}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-root", default=str(DEFAULT_EXPERIMENT_ROOT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--paper-figure-dir", default=str(DEFAULT_PAPER_FIGURE_DIR))
    parser.add_argument("--paper-table-dir", default=str(DEFAULT_PAPER_TABLE_DIR))
    parser.add_argument("--scenario", default="voltage_sag_0.92")
    parser.add_argument("--replay-window-start-s", type=float, default=115.0)
    parser.add_argument("--validation-tf-s", type=float, default=45.0)
    parser.add_argument("--paux-disturbance-pu", type=float, default=0.015)
    parser.add_argument("--allow-fallback", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run_validation(parse_args())
