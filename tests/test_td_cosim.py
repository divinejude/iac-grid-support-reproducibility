import unittest

import pandas as pd

from validate_td_cosim import TDCosimConfig, compute_metrics, held_value


class TDCosimTests(unittest.TestCase):
    def test_held_value_uses_zero_order_hold(self):
        trace = pd.DataFrame({"time_s": [10.0, 20.0, 30.0], "p": [1.0, 2.0, 3.0]})

        self.assertEqual(held_value(trace, "p", 9.0), 1.0)
        self.assertEqual(held_value(trace, "p", 20.0), 2.0)
        self.assertEqual(held_value(trace, "p", 29.9), 2.0)
        self.assertEqual(held_value(trace, "p", 99.0), 3.0)

    def test_metrics_use_post_disturbance_frequency_nadir(self):
        traces = pd.DataFrame(
            {
                "time_s": [0.0, 5.0, 10.0],
                "controller": ["no_support"] * 3,
                "controller_label": ["No support"] * 3,
                "andes_frequency_hz": [59.0, 59.8, 59.7],
                "opendss_min_voltage_pu": [0.95, 0.94, 0.93],
                "opendss_pcc_voltage_pu": [0.98, 0.97, 0.96],
                "feeder_boundary_kw": [1000.0, 1100.0, 1200.0],
                "paux_pu": [0.0, -0.01, -0.02],
            }
        )

        metrics = compute_metrics(traces, TDCosimConfig(disturbance_time_s=5.0))

        self.assertAlmostEqual(metrics["frequency_nadir_hz"].iloc[0], 59.7)
        self.assertAlmostEqual(metrics["min_feeder_voltage_pu"].iloc[0], 0.93)
        self.assertAlmostEqual(metrics["max_boundary_load_kw"].iloc[0], 1200.0)


if __name__ == "__main__":
    unittest.main()
