"""Calibrate the IAC thermal model from measured NIST NZERTF data.

The script uses the public NIST Net-Zero Energy Residential Test Facility
minute-resolution data files. It downloads the measured HVAC, indoor
environment, and outdoor environment CSV files, extracts a cooling-season
window, and fits an effective whole-building first-order RC model with a
temperature-lift-dependent COP.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path
import shutil
import time
import urllib.request

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from iac_dmpc.calibration import ThermalCalibrationData
from iac_dmpc.parameters import ThermalParameters
from iac_dmpc.plot_style import apply_elsevier_figure_style


PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_ROOT = "https://s3.amazonaws.com/nist-netzero"
DATA_PAGE = "https://pages.nist.gov/netzero/data.html"
TUTORIAL_PAGE = "https://pages.nist.gov/netzero/learn.html"

HVAC_COLUMNS = [
    "Timestamp",
    "HVAC_HVACTempReturnAir",
    "HVAC_HVACTempSupplyAir",
    "HVAC_HeatPumpIndoorUnitPower",
    "HVAC_HeatPumpOutdoorUnitPower",
]
OUTDOOR_COLUMNS = ["Timestamp", "OutEnv_OutdoorAmbTemp"]
OCCUPIED_ROOM_COLUMNS = [
    "IndEnv_RoomTempKitchenTemp",
    "IndEnv_RoomTempDRTemp",
    "IndEnv_RoomTempLRTemp",
    "IndEnv_RoomTempHallLowest",
    "IndEnv_RoomTempHallLowerMid",
    "IndEnv_RoomTempHallMiddle",
    "IndEnv_RoomTempHallUpperMid",
    "IndEnv_RoomTempHallUpper",
    "IndEnv_RoomTempBR4Temp",
    "IndEnv_RoomTempBA1Temp",
    "IndEnv_RoomTempWDTemp",
    "IndEnv_RoomTempMBATemp",
    "IndEnv_RoomTempMBRTemp",
    "IndEnv_RoomTempBR2Temp",
    "IndEnv_RoomTempBR3Temp",
    "IndEnv_RoomTempBA2Temp",
]
INDOOR_COLUMNS = ["Timestamp", *OCCUPIED_ROOM_COLUMNS]


def format_duration(seconds: float) -> str:
    minutes, sec = divmod(int(round(max(seconds, 0.0))), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes}m {sec}s"
    if minutes:
        return f"{minutes}m {sec}s"
    return f"{sec}s"


def download(url: str, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 0:
        print(f"Using cached {path}")
        return path

    print(f"Downloading {url}")
    start = time.perf_counter()
    with urllib.request.urlopen(url, timeout=60) as response, path.open("wb") as handle:
        total = int(response.headers.get("Content-Length", "0") or 0)
        received = 0
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            handle.write(chunk)
            received += len(chunk)
            if total:
                elapsed = max(time.perf_counter() - start, 1e-9)
                rate = received / elapsed
                remaining = (total - received) / max(rate, 1e-9)
                print(
                    f"  {received / 1e6:.1f}/{total / 1e6:.1f} MB "
                    f"(estimated remaining {format_duration(remaining)})",
                    end="\r",
                    flush=True,
                )
    print()
    return path


def load_csv(path: Path, columns: list[str]) -> pd.DataFrame:
    frame = pd.read_csv(path, usecols=columns)
    frame["Timestamp"] = pd.to_datetime(frame["Timestamp"], utc=True).dt.tz_convert(None)
    return frame.rename(columns={"Timestamp": "timestamp"})


def fahrenheit_to_celsius(values: pd.Series) -> pd.Series:
    return (values - 32.0) * 5.0 / 9.0


def build_measurement_frame(args: argparse.Namespace) -> pd.DataFrame:
    raw_dir = args.output_dir / "raw"
    base = f"{DATASET_ROOT}/{args.year}-data-files"
    hvac_path = download(f"{base}/HVAC-minute.csv", raw_dir / f"{args.year}_HVAC-minute.csv")
    indoor_path = download(f"{base}/IndEnv-minute.csv", raw_dir / f"{args.year}_IndEnv-minute.csv")
    outdoor_path = download(f"{base}/OutEnv-minute.csv", raw_dir / f"{args.year}_OutEnv-minute.csv")

    hvac = load_csv(hvac_path, HVAC_COLUMNS)
    indoor = load_csv(indoor_path, INDOOR_COLUMNS)
    outdoor = load_csv(outdoor_path, OUTDOOR_COLUMNS)

    indoor["indoor_c"] = indoor[OCCUPIED_ROOM_COLUMNS].mean(axis=1, skipna=True)
    hvac["return_air_c"] = fahrenheit_to_celsius(hvac["HVAC_HVACTempReturnAir"])
    hvac["supply_air_c"] = fahrenheit_to_celsius(hvac["HVAC_HVACTempSupplyAir"])
    hvac["cooling_electric_kw"] = (
        hvac["HVAC_HeatPumpIndoorUnitPower"].clip(lower=0.0)
        + hvac["HVAC_HeatPumpOutdoorUnitPower"].clip(lower=0.0)
    ) / 1000.0
    hvac["supply_return_delta_c"] = hvac["supply_air_c"] - hvac["return_air_c"]
    outdoor = outdoor.rename(columns={"OutEnv_OutdoorAmbTemp": "outdoor_c"})

    frame = indoor[["timestamp", "indoor_c"]].merge(
        outdoor[["timestamp", "outdoor_c"]], on="timestamp", how="inner"
    )
    frame = frame.merge(
        hvac[["timestamp", "return_air_c", "supply_air_c", "supply_return_delta_c", "cooling_electric_kw"]],
        on="timestamp",
        how="inner",
    )
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna()
    frame = frame.sort_values("timestamp").drop_duplicates("timestamp")
    frame["is_cooling"] = (
        (frame["cooling_electric_kw"] >= args.cooling_power_threshold_kw)
        & (frame["outdoor_c"] >= args.min_cooling_outdoor_c)
        & (frame["supply_return_delta_c"] <= -args.min_supply_return_delta_c)
    )
    frame.loc[~frame["is_cooling"], "cooling_electric_kw"] = 0.0
    return frame


def choose_fit_window(frame: pd.DataFrame, window_days: int) -> pd.DataFrame:
    daily_cooling = frame.set_index("timestamp")["cooling_electric_kw"].resample("D").sum()
    rolling = daily_cooling.rolling(window_days, min_periods=window_days).sum()
    if rolling.dropna().empty:
        return frame
    end_day = rolling.idxmax()
    start = end_day - pd.Timedelta(days=window_days - 1)
    end = end_day + pd.Timedelta(days=1)
    window = frame[(frame["timestamp"] >= start) & (frame["timestamp"] < end)].copy()
    return window if len(window) >= window_days * 24 * 30 else frame


def effective_cop(power_kw: np.ndarray, ambient_c: np.ndarray, indoor_c: np.ndarray, params: ThermalParameters) -> np.ndarray:
    lift = np.maximum(ambient_c - indoor_c, 1.0)
    nominal_lift = params.ambient_temperature_c - params.setpoint_c
    lift_factor = 1.0 + params.cop_temperature_slope_per_c * (lift - nominal_lift)
    return np.clip(params.cop * lift_factor, 0.8, 8.0) * (power_kw > 0.0)


def simulate_temperature(data: ThermalCalibrationData, params: ThermalParameters) -> np.ndarray:
    indoor = np.zeros_like(data.indoor_c)
    indoor[0] = data.indoor_c[0]
    for index in range(len(indoor) - 1):
        dt = data.time_s[index + 1] - data.time_s[index]
        cop = effective_cop(
            np.array([data.power_kw[index]]),
            np.array([data.ambient_c[index]]),
            np.array([indoor[index]]),
            params,
        )[0]
        cooling_kw = cop * data.power_kw[index]
        envelope_kw = (data.ambient_c[index] - indoor[index]) / params.resistance_c_per_kw
        derivative = (envelope_kw + params.internal_gain_kw - cooling_kw) / (
            params.capacitance_kwh_per_c * 3600.0
        )
        indoor[index + 1] = indoor[index] + dt * derivative
    return indoor


def fit_measured_thermal_model(data: ThermalCalibrationData, initial: ThermalParameters) -> tuple[ThermalParameters, np.ndarray]:
    names = ["resistance", "capacitance", "cop", "cop_slope", "internal_gain"]
    x0 = np.array(
        [
            initial.resistance_c_per_kw,
            initial.capacitance_kwh_per_c,
            initial.cop,
            initial.cop_temperature_slope_per_c,
            initial.internal_gain_kw,
        ],
        dtype=float,
    )
    lower = np.array([0.05, 0.5, 1.0, -0.12, 0.0])
    upper = np.array([20.0, 80.0, 8.0, 0.03, 5.0])

    def unpack(theta: np.ndarray) -> ThermalParameters:
        values = dict(zip(names, theta))
        return replace(
            initial,
            resistance_c_per_kw=float(values["resistance"]),
            capacitance_kwh_per_c=float(values["capacitance"]),
            cop=float(values["cop"]),
            cop_temperature_slope_per_c=float(values["cop_slope"]),
            internal_gain_kw=float(values["internal_gain"]),
        )

    # Downweight night/off intervals mildly so the fit still learns the envelope
    # but is driven by cooling-season behavior.
    cooling_weight = np.where(data.power_kw > 0.1, 1.0, 0.35)

    def residual(theta: np.ndarray) -> np.ndarray:
        predicted = simulate_temperature(data, unpack(theta))
        return (predicted - data.indoor_c) * cooling_weight

    result = least_squares(
        residual,
        np.clip(x0, lower, upper),
        bounds=(lower, upper),
        max_nfev=250,
        x_scale=[2.0, 20.0, 3.0, 0.04, 1.0],
    )
    calibrated = unpack(result.x)
    return calibrated, simulate_temperature(data, calibrated)


def make_calibration_data(frame: pd.DataFrame) -> ThermalCalibrationData:
    time_s = (frame["timestamp"] - frame["timestamp"].iloc[0]).dt.total_seconds().to_numpy(dtype=float)
    return ThermalCalibrationData(
        time_s=time_s,
        indoor_c=frame["indoor_c"].to_numpy(dtype=float),
        ambient_c=frame["outdoor_c"].to_numpy(dtype=float),
        power_kw=frame["cooling_electric_kw"].to_numpy(dtype=float),
        cooling_capacity_kw=None,
    )


def compute_fit_metrics(frame: pd.DataFrame, predicted_c: np.ndarray, thermal: ThermalParameters) -> dict[str, float]:
    residual = predicted_c - frame["indoor_c"].to_numpy(dtype=float)
    power = frame["cooling_electric_kw"].to_numpy(dtype=float)
    cop = effective_cop(
        power,
        frame["outdoor_c"].to_numpy(dtype=float),
        frame["indoor_c"].to_numpy(dtype=float),
        thermal,
    )
    cooling_capacity_kw = cop * power
    active = power > 0.1
    return {
        "indoor_temperature_rmse_c": float(np.sqrt(np.mean(residual**2))),
        "indoor_temperature_mae_c": float(np.mean(np.abs(residual))),
        "active_cooling_rmse_c": float(np.sqrt(np.mean(residual[active] ** 2))) if np.any(active) else float("nan"),
        "sample_count": int(len(frame)),
        "active_cooling_sample_count": int(np.count_nonzero(active)),
        "mean_active_cooling_power_kw": float(np.mean(power[active])) if np.any(active) else 0.0,
        "median_effective_cop_active": float(np.median(cop[active])) if np.any(active) else float(thermal.cop),
        "mean_effective_cooling_capacity_kw": float(np.mean(cooling_capacity_kw[active])) if np.any(active) else 0.0,
    }


def plot_calibration(frame: pd.DataFrame, predicted_c: np.ndarray, thermal: ThermalParameters, path: Path) -> None:
    apply_elsevier_figure_style(base_size=8.0)
    path.parent.mkdir(parents=True, exist_ok=True)
    time_days = (frame["timestamp"] - frame["timestamp"].iloc[0]).dt.total_seconds() / 86400.0
    power = frame["cooling_electric_kw"].to_numpy(dtype=float)
    cop = effective_cop(
        power,
        frame["outdoor_c"].to_numpy(dtype=float),
        frame["indoor_c"].to_numpy(dtype=float),
        thermal,
    )
    capacity = cop * power

    fig, axes = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
    axes[0].plot(time_days, frame["indoor_c"], label="Measured indoor")
    axes[0].plot(time_days, predicted_c, label="Fitted RC model", linewidth=1.1)
    axes[0].plot(time_days, frame["outdoor_c"], label="Outdoor", alpha=0.65)
    axes[0].axhline(thermal.setpoint_c, color="black", linestyle="--", linewidth=0.8, label="Setpoint")
    axes[0].set_ylabel("Temp. (deg C)")
    axes[0].legend(loc="best", ncol=2)

    axes[1].plot(time_days, power, label="Measured heat-pump electric")
    axes[1].set_ylabel("Power (kW)")
    axes[1].legend(loc="best")

    axes[2].plot(time_days, capacity, label="Estimated cooling capacity")
    axes[2].set_ylabel("Cooling (kW)")
    axes[2].legend(loc="best")

    active = power > 0.1
    axes[3].plot(time_days[active], cop[active], ".", markersize=2.0, label="Effective COP")
    axes[3].set_ylabel("COP")
    axes[3].set_xlabel("Fit-window time (days)")
    axes[3].set_ylim(0, 8)
    axes[3].legend(loc="best")

    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_latex_table(payload: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fit = payload["fit_quality"]
    thermal = payload["thermal_parameters"]
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
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def calibrate(args: argparse.Namespace) -> dict[str, object]:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame = build_measurement_frame(args)
    fit_frame = choose_fit_window(frame, args.fit_window_days)
    setpoint_c = float(np.median(fit_frame["indoor_c"]))
    initial = ThermalParameters(
        ambient_temperature_c=float(np.median(fit_frame["outdoor_c"])),
        setpoint_c=setpoint_c,
        comfort_band_c=1.0,
        resistance_c_per_kw=ThermalParameters().resistance_c_per_kw,
        capacitance_kwh_per_c=25.0,
        cop=3.2,
        cop_temperature_slope_per_c=-0.03,
        internal_gain_kw=ThermalParameters().internal_gain_kw,
    )
    data = make_calibration_data(fit_frame)
    thermal, predicted_c = fit_measured_thermal_model(data, initial)
    thermal = replace(
        thermal,
        ambient_temperature_c=float(np.median(fit_frame["outdoor_c"])),
        setpoint_c=setpoint_c,
    )
    metrics = compute_fit_metrics(fit_frame, predicted_c, thermal)

    power = fit_frame["cooling_electric_kw"].to_numpy(dtype=float)
    cop = effective_cop(
        power,
        fit_frame["outdoor_c"].to_numpy(dtype=float),
        fit_frame["indoor_c"].to_numpy(dtype=float),
        thermal,
    )
    export = fit_frame[
        [
            "timestamp",
            "indoor_c",
            "outdoor_c",
            "return_air_c",
            "supply_air_c",
            "supply_return_delta_c",
            "cooling_electric_kw",
            "is_cooling",
        ]
    ].copy()
    export.insert(0, "time_s", data.time_s)
    export["predicted_indoor_c"] = predicted_c
    export["effective_cop"] = cop
    export["estimated_cooling_capacity_kw"] = cop * power
    csv_path = args.output_dir / "nist_nzertf_calibration_timeseries.csv"
    export.to_csv(csv_path, index=False)

    payload: dict[str, object] = {
        "source": "NIST Net-Zero Energy Residential Test Facility measured minute-resolution data",
        "dataset_url": DATA_PAGE,
        "documentation_url": TUTORIAL_PAGE,
        "year": int(args.year),
        "metadata": {
            "facility": "NIST Net-Zero Energy Residential Test Facility",
            "data_type": "measured residential test-facility data",
            "indoor_temperature": "mean of occupied-zone room temperature sensors",
            "outdoor_temperature": "OutEnv_OutdoorAmbTemp",
            "hvac_power": "HVAC_HeatPumpIndoorUnitPower + HVAC_HeatPumpOutdoorUnitPower",
            "cooling_detection": (
                f"power >= {args.cooling_power_threshold_kw} kW, outdoor >= "
                f"{args.min_cooling_outdoor_c} deg C, supply-return delta <= "
                f"-{args.min_supply_return_delta_c} deg C"
            ),
        },
        "fit_window": {
            "start": str(fit_frame["timestamp"].iloc[0]),
            "end": str(fit_frame["timestamp"].iloc[-1]),
            "samples": int(len(fit_frame)),
            "window_days": int(args.fit_window_days),
        },
        "fit_quality": metrics,
        "thermal_parameters": asdict(thermal),
        "notes": [
            "NIST HVAC supply and return air temperatures are used only to classify cooling operation.",
            "Cooling capacity is inferred as effective COP times measured heat-pump electric power because the public minute HVAC file does not directly report delivered cooling load.",
            "The fitted parameters are effective whole-building RC/COP parameters for the NZERTF test house, not room-by-room physical envelope constants.",
        ],
    }
    json_path = args.output_dir / "calibrated_thermal_parameters.json"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    plot_path = args.output_dir / "nist_nzertf_calibration_fit.png"
    plot_calibration(fit_frame, predicted_c, thermal, plot_path)
    write_latex_table(payload, PROJECT_ROOT / "paper" / "tables" / "nist_calibration.tex")
    paper_fig = PROJECT_ROOT / "paper" / "figures" / "nist_nzertf_calibration_fit.png"
    paper_fig.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(plot_path, paper_fig)

    print(json.dumps(payload["thermal_parameters"], indent=2))
    print(f"Wrote {json_path}")
    print(f"Wrote {csv_path}")
    print(f"Wrote {plot_path}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", type=int, default=2015)
    parser.add_argument("--fit-window-days", type=int, default=14)
    parser.add_argument("--cooling-power-threshold-kw", type=float, default=0.5)
    parser.add_argument("--min-cooling-outdoor-c", type=float, default=18.0)
    parser.add_argument("--min-supply-return-delta-c", type=float, default=1.5)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "data" / "nist_nzertf")
    args = parser.parse_args()
    calibrate(args)


if __name__ == "__main__":
    main()
