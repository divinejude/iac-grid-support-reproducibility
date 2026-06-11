"""Generate paper-ready summary tables and figures from experiment outputs."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from simulate_iac_grid_support import PROJECT_ROOT


KEY_METRICS = [
    "frequency_nadir_hz",
    "min_feeder_voltage_pu",
    "min_pcc_voltage_pu",
    "temperature_max_c",
    "comfort_violation_count",
    "aggregate_energy_kwh",
    "reactive_support_kvarh",
    "mpc_feasibility_rate",
    "mean_compute_time_ms",
]


def load_rows(experiment_dirs: list[Path]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for directory in experiment_dirs:
        metrics_path = directory / "metrics_summary.csv"
        if not metrics_path.exists():
            raise FileNotFoundError(f"Missing metrics file: {metrics_path}")
        with metrics_path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                row["experiment_set"] = directory.name
                rows.append(row)
    return rows


def summarize_by_controller(rows: list[dict[str, str]]) -> list[dict[str, str | float]]:
    controllers = sorted({row["controller"] for row in rows})
    summary = []
    for controller in controllers:
        subset = [row for row in rows if row["controller"] == controller]
        output: dict[str, str | float] = {"controller": controller, "case_count": len(subset)}
        for metric in KEY_METRICS:
            values = np.array([float(row[metric]) for row in subset], dtype=float)
            output[f"{metric}_mean"] = float(np.mean(values))
            output[f"{metric}_min"] = float(np.min(values))
            output[f"{metric}_max"] = float(np.max(values))
        summary.append(output)
    return summary


def write_csv(path: Path, rows: list[dict[str, str | float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()}) if rows else []
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def make_metric_bar(path: Path, rows: list[dict[str, str | float]], metric: str, ylabel: str) -> None:
    controllers = [str(row["controller"]) for row in rows]
    values = [float(row[f"{metric}_mean"]) for row in rows]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(controllers, values, color=["#777777", "#4C78A8", "#F58518", "#54A24B", "#B279A2"][: len(controllers)])
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment-dir",
        action="append",
        type=Path,
        required=True,
        help="Experiment directory containing metrics_summary.csv. May be repeated.",
    )
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "paper_outputs")
    args = parser.parse_args()

    rows = load_rows(args.experiment_dir)
    summary = summarize_by_controller(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "controller_summary.csv", summary)
    write_csv(args.output_dir / "all_case_metrics.csv", rows)
    make_metric_bar(args.output_dir / "frequency_nadir_mean.png", summary, "frequency_nadir_hz", "Mean Frequency Nadir (Hz)")
    make_metric_bar(args.output_dir / "worst_voltage_mean.png", summary, "min_feeder_voltage_pu", "Mean Worst Feeder Voltage (pu)")
    make_metric_bar(args.output_dir / "comfort_violations_mean.png", summary, "comfort_violation_count", "Mean Comfort Violation Count")
    make_metric_bar(args.output_dir / "compute_time_mean.png", summary, "mean_compute_time_ms", "Mean Compute Time (ms)")
    print(f"Wrote paper-ready outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
