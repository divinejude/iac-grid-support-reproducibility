"""Generate a dynamic-simulator-style frequency replay trace.

The trace represents a reduced transmission-area load-frequency-control model
with inertia, damping, turbine-governor lag, and droop. It exports both the
no-IAC disturbance trajectory and the step-response kernel to a sustained
1 kW IAC load reduction. The CSV is consumed by ``CsvReplayDynamicBackend``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp


PROJECT_ROOT = Path(__file__).resolve().parent


def disturbance_kw(t: float, start_s: float, end_s: float, magnitude_kw: float) -> float:
    return magnitude_kw if start_s <= t <= end_s else 0.0


def simulate_lfc(
    time_s: np.ndarray,
    disturbance_magnitude_kw: float,
    iac_step_kw: float,
    event_start_s: float,
    event_end_s: float,
    base_power_kw: float,
    inertia_s: float,
    damping_pu_per_hz: float,
    governor_time_constant_s: float,
    turbine_time_constant_s: float,
    droop_hz_per_pu: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return baseline frequency and IAC step response in Hz."""

    omega0_hz = 60.0

    def rhs(t: float, x: np.ndarray, include_disturbance: bool, step_kw: float) -> np.ndarray:
        df_hz, governor_pu, turbine_pu = x
        disturbance_pu = (
            disturbance_kw(t, event_start_s, event_end_s, disturbance_magnitude_kw) / base_power_kw
            if include_disturbance
            else 0.0
        )
        iac_relief_pu = step_kw / base_power_kw if t >= 0.0 else 0.0
        governor_ref_pu = -df_hz / droop_hz_per_pu
        d_governor = (governor_ref_pu - governor_pu) / governor_time_constant_s
        d_turbine = (governor_pu - turbine_pu) / turbine_time_constant_s
        d_df = omega0_hz / (2.0 * inertia_s) * (turbine_pu + iac_relief_pu - disturbance_pu - damping_pu_per_hz * df_hz)
        return np.array([d_df, d_governor, d_turbine])

    baseline = solve_ivp(
        lambda t, x: rhs(t, x, True, 0.0),
        (float(time_s[0]), float(time_s[-1])),
        np.zeros(3),
        t_eval=time_s,
        rtol=1e-8,
        atol=1e-10,
    )
    step = solve_ivp(
        lambda t, x: rhs(t, x, False, iac_step_kw),
        (float(time_s[0]), float(time_s[-1])),
        np.zeros(3),
        t_eval=time_s,
        rtol=1e-8,
        atol=1e-10,
    )
    if not baseline.success:
        raise RuntimeError(f"Baseline dynamic trace integration failed: {baseline.message}")
    if not step.success:
        raise RuntimeError(f"IAC step-response integration failed: {step.message}")
    return baseline.y[0], step.y[0] / iac_step_kw


def write_trace(path: Path, time_s: np.ndarray, baseline_hz: np.ndarray, step_response: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = np.column_stack([time_s, baseline_hz, step_response])
    header = "time_s,frequency_deviation_hz,iac_step_response_hz_per_kw"
    np.savetxt(path, data, delimiter=",", header=header, comments="", fmt="%.10f")


def plot_trace(path: Path, time_s: np.ndarray, baseline_hz: np.ndarray, step_response: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
    axes[0].plot(time_s / 60.0, 60.0 + baseline_hz)
    axes[0].axhline(60.0, color="black", linestyle="--", linewidth=0.8)
    axes[0].set_ylabel("No-IAC frequency (Hz)")
    axes[1].plot(time_s / 60.0, 1000.0 * step_response)
    axes[1].set_ylabel("Hz per MW relief")
    axes[1].set_xlabel("Time (min)")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-csv", type=Path, default=PROJECT_ROOT / "data" / "dynamic_traces" / "lfc_replay_trace.csv")
    parser.add_argument("--output-figure", type=Path, default=PROJECT_ROOT / "data" / "dynamic_traces" / "lfc_replay_trace.png")
    parser.add_argument("--total-time-s", type=float, default=600.0)
    parser.add_argument("--sample-time-s", type=float, default=5.0)
    parser.add_argument("--event-start-s", type=float, default=120.0)
    parser.add_argument("--event-end-s", type=float, default=420.0)
    parser.add_argument("--disturbance-kw", type=float, default=1200.0)
    parser.add_argument("--base-power-kw", type=float, default=5000.0)
    args = parser.parse_args()

    time_s = np.arange(0.0, args.total_time_s + args.sample_time_s, args.sample_time_s)
    baseline_hz, step_response = simulate_lfc(
        time_s,
        disturbance_magnitude_kw=args.disturbance_kw,
        iac_step_kw=1.0,
        event_start_s=args.event_start_s,
        event_end_s=args.event_end_s,
        base_power_kw=args.base_power_kw,
        inertia_s=4.5,
        damping_pu_per_hz=1.0,
        governor_time_constant_s=0.7,
        turbine_time_constant_s=4.0,
        droop_hz_per_pu=3.0,
    )
    write_trace(args.output_csv, time_s, baseline_hz, step_response)
    plot_trace(args.output_figure, time_s, baseline_hz, step_response)
    print(f"Wrote dynamic replay CSV to {args.output_csv}")
    print(f"Wrote dynamic replay figure to {args.output_figure}")
    print(f"No-IAC frequency nadir: {60.0 + baseline_hz.min():.3f} Hz")
    print(f"Steady 1 MW relief gain: {1000.0 * step_response[-1]:.3f} Hz/MW")


if __name__ == "__main__":
    main()
