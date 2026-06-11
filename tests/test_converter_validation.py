import unittest

import pandas as pd

from iac_dmpc.parameters import InverterInnerLoopParameters, InverterParameters
from validate_averaged_converter import (
    AveragedConverterConfig,
    limit_current_references,
    simulate_averaged_converter,
)


class AveragedConverterValidationTests(unittest.TestCase):
    def test_current_reference_limiter_preserves_reactive_priority(self):
        id_ref, iq_ref, _, _, saturated, _ = limit_current_references(
            p_ref_kw=3.5,
            q_ref_kvar=2.2,
            voltage_rms_v=240.0,
            vdc_v=390.0,
            inverter=InverterParameters(),
            inner=InverterInnerLoopParameters(rms_current_limit_a=10.0, dc_link_current_limit_a=100.0),
        )

        self.assertTrue(saturated)
        self.assertLessEqual((id_ref**2 + iq_ref**2) ** 0.5, 10.0 + 1e-9)
        self.assertAlmostEqual(iq_ref, 2.2 * 1000.0 / 240.0)

    def test_short_averaged_converter_run_tracks_without_voltage_saturation(self):
        trace = pd.DataFrame(
            {
                "time_s": [175.0, 180.0, 185.0],
                "p_command_kw": [2.0, 1.0, 1.0],
                "q_command_kvar": [0.0, 1.0, 1.0],
                "source_voltage_pu": [1.0, 0.92, 0.92],
                "frequency_hz": [60.0, 59.9, 59.9],
            }
        )
        cfg = AveragedConverterConfig(duration_s=0.1, step_s=1.0e-3, warmup_s=0.02)

        result = simulate_averaged_converter(trace, cfg)

        self.assertFalse(result.empty)
        self.assertLess(result["voltage_saturated"].mean(), 0.05)
        self.assertGreater(result["vdc_v"].min(), 360.0)


if __name__ == "__main__":
    unittest.main()
