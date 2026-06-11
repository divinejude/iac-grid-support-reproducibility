import unittest

import numpy as np

from iac_dmpc.metrics import summarize_log


class MetricsTests(unittest.TestCase):
    def test_summary_metrics_are_computed_from_log(self) -> None:
        log = {
            "time_s": np.array([0.0, 5.0, 10.0]),
            "frequency_hz": np.array([60.0, 59.8, 59.9]),
            "pcc_voltage_pu": np.array([1.0, 0.95, 0.98]),
            "min_feeder_voltage_pu": np.array([0.99, 0.92, 0.95]),
            "max_feeder_voltage_pu": np.array([1.01, 1.02, 1.00]),
            "temperature_c": np.array([24.0, 25.2, 24.5]),
            "aggregate_p_kw": np.array([100.0, 120.0, 110.0]),
            "aggregate_q_kvar": np.array([0.0, 30.0, -10.0]),
            "apparent_kva": np.array([2.0, 2.5, 2.1]),
            "dmpc_status_ok": np.array([1.0, 1.0, 0.0]),
            "controller_compute_time_s": np.array([0.001, 0.002, 0.003]),
            "current_saturated": np.array([0.0, 1.0, 0.0]),
            "ramp_limited": np.array([0.0, 0.0, 1.0]),
        }
        metrics = summarize_log(log)
        self.assertAlmostEqual(metrics["frequency_nadir_hz"], 59.8)
        self.assertAlmostEqual(metrics["min_pcc_voltage_pu"], 0.95)
        self.assertAlmostEqual(metrics["min_feeder_voltage_pu"], 0.92)
        self.assertEqual(metrics["comfort_violation_count"], 1.0)
        self.assertAlmostEqual(metrics["mpc_feasibility_rate"], 2.0 / 3.0)


if __name__ == "__main__":
    unittest.main()
