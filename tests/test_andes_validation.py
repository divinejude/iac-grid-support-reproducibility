import unittest

import pandas as pd

from validate_andes_transient import AndesValidationConfig, build_alter_schedule


class AndesValidationScheduleTests(unittest.TestCase):
    def test_iac_relief_reduces_net_disturbance_after_event(self):
        cfg = AndesValidationConfig(
            replay_window_start_s=100.0,
            validation_tf_s=15.0,
            disturbance_time_s=10.0,
            paux_disturbance_pu=0.015,
            transmission_base_kw=100_000.0,
        )
        trace = pd.DataFrame(
            {
                "time_s": [100.0, 105.0, 110.0, 115.0],
                "aggregate_p_kw": [1000.0, 700.0, 300.0, 300.0],
            }
        )

        schedule = build_alter_schedule(trace, cfg, "dmpc_active_only")
        before_event = schedule.loc[schedule["t"] == 5.0, "amount"].iloc[0]
        after_event = schedule.loc[schedule["t"] == 10.0, "amount"].iloc[0]

        self.assertAlmostEqual(before_event, 0.003)
        self.assertAlmostEqual(after_event, -0.008)
        self.assertLess(abs(after_event), cfg.paux_disturbance_pu)

    def test_no_support_keeps_full_disturbance(self):
        cfg = AndesValidationConfig(
            replay_window_start_s=100.0,
            validation_tf_s=10.0,
            disturbance_time_s=10.0,
            paux_disturbance_pu=0.015,
        )
        trace = pd.DataFrame(
            {
                "time_s": [100.0, 105.0, 110.0],
                "aggregate_p_kw": [1000.0, 1000.0, 1000.0],
            }
        )

        schedule = build_alter_schedule(trace, cfg, "no_support")

        self.assertAlmostEqual(schedule.loc[schedule["t"] == 10.0, "amount"].iloc[0], -0.015)


if __name__ == "__main__":
    unittest.main()
