"""Calibrate the IAC thermal model from public ResStock/EnergyPlus outputs.

The default case uses the NREL End-Use Load Profiles public S3 bucket,
ResStock AMY2018 2024 release 2, baseline upgrade 0, and a Texas central-AC
single-family building. The script downloads only the selected building
timeseries and the matching state metadata parquet.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path
import re
import urllib.request

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from iac_dmpc.calibration import ThermalCalibrationData, fit_thermal_rc_from_timeseries
from iac_dmpc.parameters import ThermalParameters
from iac_dmpc.plot_style import apply_elsevier_figure_style


PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_ROOT = (
    "https://oedi-data-lake.s3.amazonaws.com/"
    "nrel-pds-building-stock/end-use-load-profiles-for-us-building-stock/"
    "2024/resstock_amy2018_release_2"
)
KBTU_TO_KWH = 0.29307107017
INTERVAL_HOURS = 0.25


def download(url: str, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        print(f"Downloading {url}")
        urllib.request.urlretrieve(url, path)
    return path


def parse_seer(efficiency: str, fallback: float = 14.0) -> float:
    match = re.search(r"SEER\s*([0-9]+(?:\.[0-9]+)?)", str(efficiency), flags=re.IGNORECASE)
    return float(match.group(1)) if match else fallback


def parse_fahrenheit(value: str, fallback_f: float = 75.0) -> float:
    match = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*F", str(value), flags=re.IGNORECASE)
    return float(match.group(1)) if match else fallback_f


def fahrenheit_to_celsius(value_f: float) -> float:
    return (value_f - 32.0) * 5.0 / 9.0


def choose_fit_window(frame: pd.DataFrame, window_days: int) -> pd.DataFrame:
    daily = frame.set_index("timestamp")["cooling_electric_kw"].resample("D").sum()
    rolling = daily.rolling(window_days, min_periods=window_days).sum()
    if rolling.dropna().empty:
        return frame
    end_day = rolling.idxmax()
    start = end_day - pd.Timedelta(days=window_days - 1)
    end = end_day + pd.Timedelta(days=1)
    window = frame[(frame["timestamp"] >= start) & (frame["timestamp"] < end)].copy()
    return window if len(window) >= 96 else frame


def fit_cop_and_slope(frame: pd.DataFrame, fallback_cop: float, setpoint_c: float) -> tuple[float, float]:
    active = frame[
        (frame["cooling_electric_kw"] > 0.10)
        & (frame["cooling_delivered_kw"] > 0.10)
        & np.isfinite(frame["cooling_electric_kw"])
        & np.isfinite(frame["cooling_delivered_kw"])
    ].copy()
    if len(active) < 20:
        return fallback_cop, ThermalParameters().cop_temperature_slope_per_c

    active["cop_observed"] = active["cooling_delivered_kw"] / active["cooling_electric_kw"].clip(lower=1e-6)
    active = active[(active["cop_observed"] > 0.5) & (active["cop_observed"] < 8.0)]
    if len(active) < 20:
        return fallback_cop, ThermalParameters().cop_temperature_slope_per_c

    cop = float(np.median(active["cop_observed"]))
    lift = active["outdoor_c"].to_numpy() - active["indoor_c"].to_numpy()
    nominal_lift = float(np.median(active["outdoor_c"]) - setpoint_c)
    x = lift - nominal_lift
    y = active["cop_observed"].to_numpy()
    design = np.column_stack([np.ones_like(x), x])
    coeff, *_ = np.linalg.lstsq(design, y, rcond=None)
    nominal_cop = float(np.clip(coeff[0], 0.5, 8.0))
    slope = float(np.clip(coeff[1] / max(nominal_cop, 1e-6), -0.12, 0.02))
    return nominal_cop if np.isfinite(nominal_cop) else cop, slope


def build_calibration_dataset(timeseries: pd.DataFrame) -> pd.DataFrame:
    column_map = {
        "timestamp": "timestamp",
        "out.zone_mean_air_temp.conditioned_space.c": "indoor_c",
        "out.outdoor_air_dryblub_temp.c": "outdoor_c",
        "out.electricity.cooling.energy_consumption": "cooling_electric_kwh",
        "out.electricity.cooling_fans_pumps.energy_consumption": "cooling_fans_pumps_kwh",
        "out.load.cooling.energy_delivered.kbtu": "cooling_delivered_kbtu",
    }
    missing = [name for name in column_map if name not in timeseries.columns]
    if missing:
        raise ValueError(f"Selected ResStock timeseries is missing required columns: {missing}")
    frame = timeseries[list(column_map)].rename(columns=column_map).copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    frame["cooling_electric_kw"] = frame["cooling_electric_kwh"] / INTERVAL_HOURS
    frame["cooling_fans_pumps_kw"] = frame["cooling_fans_pumps_kwh"] / INTERVAL_HOURS
    frame["cooling_delivered_kw"] = frame["cooling_delivered_kbtu"] * KBTU_TO_KWH / INTERVAL_HOURS
    return frame.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["timestamp", "indoor_c", "outdoor_c", "cooling_electric_kw", "cooling_delivered_kw"]
    )


def calibrate(args: argparse.Namespace) -> dict[str, object]:
    raw_dir = args.output_dir / "raw"
    ts_url = (
        f"{DATASET_ROOT}/timeseries_individual_buildings/by_state/"
        f"upgrade=0/state={args.state}/{args.building_id}-0.parquet"
    )
    metadata_url = (
        f"{DATASET_ROOT}/metadata_and_annual_results/by_state/state={args.state}/parquet/"
        f"{args.state}_baseline_metadata_and_annual_results.parquet"
    )
    citation_url = f"{DATASET_ROOT}/resstock_documentation_2024_release_2.pdf"

    ts_path = download(ts_url, raw_dir / f"2024_{args.building_id}-0.parquet")
    metadata_path = download(metadata_url, raw_dir / f"2024_{args.state}_baseline_metadata_and_annual_results.parquet")

    timeseries = pd.read_parquet(ts_path)
    metadata = pd.read_parquet(metadata_path)
    if args.building_id not in metadata.index:
        raise ValueError(f"Building {args.building_id} not found in {metadata_path}")
    meta = metadata.loc[args.building_id]

    frame = build_calibration_dataset(timeseries)
    fit_frame = choose_fit_window(frame, args.fit_window_days)
    setpoint_c = fahrenheit_to_celsius(parse_fahrenheit(meta.get("in.cooling_setpoint", "75F")))
    fallback_cop = parse_seer(meta.get("in.hvac_cooling_efficiency", ""), fallback=14.0) / 3.412
    cop, cop_slope = fit_cop_and_slope(fit_frame, fallback_cop, setpoint_c)

    initial = ThermalParameters(
        ambient_temperature_c=float(np.median(fit_frame["outdoor_c"])),
        setpoint_c=setpoint_c,
        comfort_band_c=1.0,
        resistance_c_per_kw=ThermalParameters().resistance_c_per_kw,
        capacitance_kwh_per_c=ThermalParameters().capacitance_kwh_per_c * float(meta.get("in.sqft", 1500.0)) / 1500.0,
        cop=cop,
        cop_temperature_slope_per_c=cop_slope,
        internal_gain_kw=ThermalParameters().internal_gain_kw,
    )
    time_s = (fit_frame["timestamp"] - fit_frame["timestamp"].iloc[0]).dt.total_seconds().to_numpy()
    calibration_data = ThermalCalibrationData(
        time_s=time_s,
        indoor_c=fit_frame["indoor_c"].to_numpy(dtype=float),
        ambient_c=fit_frame["outdoor_c"].to_numpy(dtype=float),
        power_kw=fit_frame["cooling_electric_kw"].to_numpy(dtype=float),
        cooling_capacity_kw=fit_frame["cooling_delivered_kw"].to_numpy(dtype=float),
    )
    fit = fit_thermal_rc_from_timeseries(
        calibration_data,
        initial=initial,
        fit_cop=False,
        fit_internal_gain=True,
        source="NREL ResStock AMY2018 2024 release 2 / EnergyPlus building output",
    )
    thermal = replace(
        fit.thermal,
        ambient_temperature_c=float(np.median(fit_frame["outdoor_c"])),
        setpoint_c=setpoint_c,
        cop=cop,
        cop_temperature_slope_per_c=cop_slope,
    )

    calibration_csv = args.output_dir / "resstock_calibration_timeseries.csv"
    export = fit_frame[
        [
            "timestamp",
            "indoor_c",
            "outdoor_c",
            "cooling_electric_kw",
            "cooling_fans_pumps_kw",
            "cooling_delivered_kw",
        ]
    ].copy()
    export.insert(0, "time_s", time_s)
    export.to_csv(calibration_csv, index=False)

    payload = {
        "source": "NREL ResStock AMY2018 2024 release 2, baseline upgrade 0, EnergyPlus outputs",
        "dataset_url": DATASET_ROOT,
        "documentation_url": citation_url,
        "building_id": args.building_id,
        "state": args.state,
        "metadata": {
            "sqft": float(meta.get("in.sqft", np.nan)),
            "county": str(meta.get("in.county", "")),
            "county_name": str(meta.get("in.county_name", "")),
            "city": str(meta.get("in.city", "")),
            "weather_file_city": str(meta.get("in.weather_file_city", "")),
            "cooling_setpoint": str(meta.get("in.cooling_setpoint", "")),
            "cooling_efficiency": str(meta.get("in.hvac_cooling_efficiency", "")),
            "cooling_type": str(meta.get("in.hvac_cooling_type", "")),
            "building_type": str(meta.get("in.geometry_building_type_recs", "")),
            "cooling_system_size_kbtu_h": float(meta.get("out.params.size_cooling_system_primary_k_btu_h", np.nan)),
            "annual_cooling_electric_kwh": float(meta.get("out.electricity.cooling.energy_consumption.kwh", np.nan)),
            "annual_cooling_delivered_kbtu": float(meta.get("out.load.cooling.energy_delivered.kbtu", np.nan)),
        },
        "fit_window": {
            "start": str(fit_frame["timestamp"].iloc[0]),
            "end": str(fit_frame["timestamp"].iloc[-1]),
            "samples": int(len(fit_frame)),
            "window_days": int(args.fit_window_days),
        },
        "fit_quality": {
            "indoor_temperature_rmse_c": fit.residual_rmse_c,
            "sample_count": fit.sample_count,
            "median_observed_cop": cop,
        },
        "thermal_parameters": asdict(thermal),
        "notes": [
            "Indoor temperature is out.zone_mean_air_temp.conditioned_space.c from EnergyPlus/ResStock timeseries.",
            "Cooling electric power is interval cooling electricity divided by 0.25 h.",
            "Cooling capacity is EnergyPlus delivered cooling load converted from kBtu per interval to kW.",
            "The RC model is first order and is fitted on the highest-cooling contiguous window to avoid heating-season mismatch.",
        ],
    }
    json_path = args.output_dir / "calibrated_thermal_parameters.json"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    plot_calibration(fit_frame, thermal, args.output_dir / "resstock_calibration_fit.png")
    print(json.dumps(payload["thermal_parameters"], indent=2))
    print(f"Wrote {json_path}")
    print(f"Wrote {calibration_csv}")
    return payload


def plot_calibration(frame: pd.DataFrame, thermal: ThermalParameters, path: Path) -> None:
    apply_elsevier_figure_style(base_size=8.0)
    path.parent.mkdir(parents=True, exist_ok=True)
    time_days = (frame["timestamp"] - frame["timestamp"].iloc[0]).dt.total_seconds() / 86400.0
    fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
    axes[0].plot(time_days, frame["indoor_c"], label="Indoor")
    axes[0].plot(time_days, frame["outdoor_c"], label="Outdoor", alpha=0.8)
    axes[0].axhline(thermal.setpoint_c, color="black", linestyle="--", linewidth=0.8, label="Setpoint")
    axes[0].set_ylabel("Temperature (deg C)")
    axes[0].legend(loc="best")
    axes[1].plot(time_days, frame["cooling_electric_kw"], label="Electric")
    axes[1].plot(time_days, frame["cooling_delivered_kw"], label="Delivered cooling", alpha=0.8)
    axes[1].set_ylabel("Power (kW)")
    axes[1].legend(loc="best")
    axes[2].plot(
        time_days,
        frame["cooling_delivered_kw"] / frame["cooling_electric_kw"].clip(lower=1e-6),
        label="Observed COP",
    )
    axes[2].set_ylim(0, 8)
    axes[2].set_ylabel("COP")
    axes[2].set_xlabel("Fit-window time (days)")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", default="TX")
    parser.add_argument("--building-id", type=int, default=100025)
    parser.add_argument("--fit-window-days", type=int, default=14)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "data" / "resstock")
    args = parser.parse_args()
    calibrate(args)


if __name__ == "__main__":
    main()
