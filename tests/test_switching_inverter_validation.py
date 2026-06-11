import unittest

import pandas as pd

from validate_switching_inverter import (
    SwitchingInverterConfig,
    compute_metrics,
    simulate_switching_inverter,
    triangular_carrier,
)


class SwitchingInverterValidationTests(unittest.TestCase):
    def test_triangular_carrier_bounds_and_midpoint(self):
        self.assertAlmostEqual(triangular_carrier(0.0, 10_000.0), 1.0)
        self.assertAlmostEqual(triangular_carrier(25.0e-6, 10_000.0), 0.0)
        self.assertAlmostEqual(triangular_carrier(50.0e-6, 10_000.0), -1.0)

    def test_short_switching_run_tracks_fundamental_commands(self):
        trace = pd.DataFrame(
            {
                "time_s": [175.0, 180.0, 185.0],
                "p_command_kw": [0.8, 0.6, 0.6],
                "q_command_kvar": [0.5, 1.5, 1.5],
                "source_voltage_pu": [1.0, 0.92, 0.92],
                "frequency_hz": [60.0, 59.95, 59.95],
            }
        )
        cfg = SwitchingInverterConfig(duration_s=0.08, step_s=5.0e-5, warmup_s=0.02, metric_skip_s=0.02)

        result = simulate_switching_inverter(trace, cfg)
        metrics = compute_metrics(result, cfg).iloc[0]

        self.assertFalse(result.empty)
        self.assertLess(metrics["rms_p_error_kw"], 0.08)
        self.assertLess(metrics["rms_q_error_kvar"], 0.08)
        self.assertGreater(metrics["min_dc_link_v"], 380.0)
        self.assertLess(metrics["voltage_saturation_fraction"], 0.05)


if __name__ == "__main__":
    unittest.main()
