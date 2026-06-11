import unittest

import numpy as np
import pandas as pd

from calibrate_from_nist_nzertf import choose_fit_window, effective_cop
from iac_dmpc.metrics import summarize_log
from iac_dmpc.parameters import ThermalParameters


class NistCalibrationTests(unittest.TestCase):
    def test_effective_cop_is_zero_when_cooling_is_off(self) -> None:
        params = ThermalParameters(
            ambient_temperature_c=30.0,
            setpoint_c=24.0,
            cop=3.0,
            cop_temperature_slope_per_c=-0.03,
        )
        cop = effective_cop(
            np.array([0.0, 1.0]),
            np.array([30.0, 30.0]),
            np.array([24.0, 24.0]),
            params,
        )
        self.assertAlmostEqual(cop[0], 0.0)
        self.assertGreater(cop[1], 0.0)

    def test_choose_fit_window_selects_highest_cooling_days(self) -> None:
        timestamps = pd.date_range("2026-07-01", periods=5 * 24 * 60, freq="min")
        frame = pd.DataFrame({"timestamp": timestamps, "cooling_electric_kw": 0.0})
        frame.loc[2 * 24 * 60 : 4 * 24 * 60 - 1, "cooling_electric_kw"] = 2.0
        window = choose_fit_window(frame, window_days=2)
        self.assertEqual(str(window["timestamp"].iloc[0])[:10], "2026-07-03")

    def test_comfort_metrics_use_supplied_calibrated_setpoint(self) -> None:
        log = {
            "time_s": np.array([0.0, 5.0]),
            "frequency_hz": np.array([60.0, 60.0]),
            "pcc_voltage_pu": np.array([1.0, 1.0]),
            "temperature_c": np.array([23.42, 23.42]),
            "aggregate_p_kw": np.array([1.0, 1.0]),
            "aggregate_q_kvar": np.array([0.0, 0.0]),
            "apparent_kva": np.array([1.0, 1.0]),
            "dmpc_status_ok": np.array([1.0, 1.0]),
            "controller_compute_time_s": np.array([0.0, 0.0]),
            "current_saturated": np.array([0.0, 0.0]),
            "ramp_limited": np.array([0.0, 0.0]),
        }
        default_metrics = summarize_log(log, comfort_band_c=0.5)
        calibrated_metrics = summarize_log(log, temperature_setpoint_c=23.42, comfort_band_c=0.5)
        self.assertGreater(default_metrics["comfort_violation_count"], 0.0)
        self.assertEqual(calibrated_metrics["comfort_violation_count"], 0.0)


if __name__ == "__main__":
    unittest.main()
