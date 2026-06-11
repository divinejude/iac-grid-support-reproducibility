import unittest
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from calibrate_from_resstock import build_calibration_dataset, fahrenheit_to_celsius, parse_seer

from iac_dmpc.calibration import (
    ThermalCalibrationData,
    calibrate_cop_from_performance_map,
    fit_thermal_rc_from_timeseries,
)
from iac_dmpc.capability import InverterCapability
from iac_dmpc.compressor import CompressorCommandState, CompressorConstraintModel
from iac_dmpc.feeder import IEEE13OpenDSSFleetFeeder, OpenDSSFeeder, SingleBusTheveninFeeder
from iac_dmpc.grid_cosim import CsvReplayDynamicBackend, DynamicGridCosimulator, SwingEquationDynamicBackend
from iac_dmpc.inverter import InverterInnerLoop, InverterInnerLoopState
from iac_dmpc.opf import compare_sensitivity_opf_to_power_flow, solve_voltage_support_opf
from iac_dmpc.parameters import (
    CompressorParameters,
    FeederParameters,
    InverterInnerLoopParameters,
    InverterParameters,
    MeasurementParameters,
    SwingFrequencyParameters,
    ThermalParameters,
)
from iac_dmpc.uncertainty import Measurement, MeasurementChannel


class PhysicalModelTests(unittest.TestCase):
    def test_capacitive_q_injection_supports_voltage(self) -> None:
        feeder = SingleBusTheveninFeeder(FeederParameters())
        no_support = feeder.solve_pcc_voltage(0.94, iac_p_kw=2.0, iac_q_kvar=0.0)
        with_support = feeder.solve_pcc_voltage(0.94, iac_p_kw=2.0, iac_q_kvar=2.0)
        self.assertGreater(with_support.pcc_voltage_pu, no_support.pcc_voltage_pu)

    def test_opendss_capacitive_q_injection_supports_voltage(self) -> None:
        feeder = OpenDSSFeeder(FeederParameters())
        no_support = feeder.solve_pcc_voltage(0.94, iac_p_kw=2.0, iac_q_kvar=0.0)
        with_support = feeder.solve_pcc_voltage(0.94, iac_p_kw=2.0, iac_q_kvar=2.0)
        self.assertGreater(with_support.pcc_voltage_pu, no_support.pcc_voltage_pu)

    def test_ieee13_tcl_allocations_are_load_proportional(self) -> None:
        feeder = IEEE13OpenDSSFleetFeeder()
        allocations = feeder.allocation_summary()
        self.assertEqual(len(allocations), 15)
        self.assertAlmostEqual(sum(item.weight for item in allocations), 1.0)
        no_support = feeder.solve_pcc_voltage(0.94, iac_p_kw=600.0, iac_q_kvar=0.0)
        with_support = feeder.solve_pcc_voltage(0.94, iac_p_kw=600.0, iac_q_kvar=600.0)
        self.assertGreater(with_support.pcc_voltage_pu, no_support.pcc_voltage_pu)

    def test_ieee13_voltage_sensitivity_signs_are_physical(self) -> None:
        feeder = IEEE13OpenDSSFleetFeeder()
        sensitivity = feeder.estimate_voltage_sensitivity(0.92, iac_p_kw=1000.0, iac_q_kvar=0.0)
        self.assertLess(sensitivity.dv_dp_pu_per_kw, 0.0)
        self.assertGreater(sensitivity.dv_dq_pu_per_kvar, 0.0)

    def test_ieee13_feeder_load_stress_is_local_and_reactive_support_helps(self) -> None:
        feeder = IEEE13OpenDSSFleetFeeder()
        unstressed = feeder.solve_pcc_voltage(1.0, iac_p_kw=1000.0, iac_q_kvar=0.0)
        stressed = feeder.solve_pcc_voltage(
            1.0,
            iac_p_kw=1000.0,
            iac_q_kvar=0.0,
            extra_load_kw=900.0,
            extra_load_kvar=450.0,
        )
        supported = feeder.solve_pcc_voltage(
            1.0,
            iac_p_kw=1000.0,
            iac_q_kvar=900.0,
            extra_load_kw=900.0,
            extra_load_kvar=450.0,
        )
        self.assertLess(stressed.min_voltage_pu, unstressed.min_voltage_pu)
        self.assertGreater(supported.min_voltage_pu, stressed.min_voltage_pu)

    def test_weak_bus_allocation_differs_from_load_proportional(self) -> None:
        load_proportional = IEEE13OpenDSSFleetFeeder(allocation_strategy="load_proportional")
        weak_bus = IEEE13OpenDSSFleetFeeder(allocation_strategy="weak_bus_prioritized")
        lp_weights = {item.name: item.weight for item in load_proportional.allocation_summary()}
        wb_weights = {item.name: item.weight for item in weak_bus.allocation_summary()}
        self.assertAlmostEqual(sum(lp_weights.values()), 1.0)
        self.assertAlmostEqual(sum(wb_weights.values()), 1.0)
        self.assertGreater(max(abs(wb_weights[name] - lp_weights[name]) for name in lp_weights), 0.05)

    def test_inverter_current_saturation_limits_current(self) -> None:
        inverter = InverterInnerLoop(
            InverterParameters(),
            InverterInnerLoopParameters(rms_current_limit_a=5.0, dc_link_current_limit_a=100.0),
        )
        result = inverter.step(InverterInnerLoopState(), 3.5, 2.2, 240.0, 1.0)
        self.assertTrue(result.saturated)
        self.assertLessEqual((result.state.id_a**2 + result.state.iq_a**2) ** 0.5, 5.0 + 1e-9)

    def test_compressor_ramp_limit_binds_large_step(self) -> None:
        compressor = CompressorConstraintModel(CompressorParameters(ramp_rate_kw_per_s=0.01), InverterParameters(), ThermalParameters())
        op, _ = compressor.apply(
            CompressorCommandState(last_power_command_kw=2.0),
            requested_power_kw=3.5,
            indoor_temperature_c=24.0,
            ambient_temperature_c=32.0,
            sample_time_s=5.0,
            time_s=0.0,
        )
        self.assertTrue(op.ramp_limited)
        self.assertAlmostEqual(op.power_kw, 2.05)

    def test_measurement_delay_returns_old_sample(self) -> None:
        channel = MeasurementChannel(
            MeasurementParameters(
                voltage_noise_std_pu=0.0,
                frequency_noise_std_hz=0.0,
                temperature_noise_std_c=0.0,
                power_noise_std_kw=0.0,
                delay_steps=1,
            )
        )
        first = channel.sample(Measurement(1.0, 0.0, 24.0, 2.0))
        second = channel.sample(Measurement(0.9, -0.1, 25.0, 1.0))
        self.assertEqual(first, second)

    def test_thermal_timeseries_calibration_recovers_physical_scale(self) -> None:
        true = ThermalParameters(resistance_c_per_kw=1.5, capacitance_kwh_per_c=2.0, cop=3.2, internal_gain_kw=0.2)
        time_s = np.arange(0.0, 3600.0 + 60.0, 60.0)
        ambient = np.full_like(time_s, 32.0)
        power = np.full_like(time_s, 2.0)
        indoor = np.zeros_like(time_s)
        indoor[0] = 25.0
        for k in range(len(time_s) - 1):
            dt = time_s[k + 1] - time_s[k]
            dtemp = ((ambient[k] - indoor[k]) / true.resistance_c_per_kw + true.internal_gain_kw - true.cop * power[k]) / (
                true.capacitance_kwh_per_c * 3600.0
            )
            indoor[k + 1] = indoor[k] + dt * dtemp

        result = fit_thermal_rc_from_timeseries(
            ThermalCalibrationData(time_s, indoor, ambient, power),
            initial=ThermalParameters(resistance_c_per_kw=1.0, capacitance_kwh_per_c=1.0, cop=3.0),
            fit_cop=False,
        )
        self.assertLess(result.residual_rmse_c, 0.05)
        self.assertGreater(result.thermal.resistance_c_per_kw, 0.5)
        self.assertGreater(result.thermal.capacitance_kwh_per_c, 0.5)

    def test_manufacturer_map_calibrates_cop_slope(self) -> None:
        calibrated = calibrate_cop_from_performance_map(
            ambient_c=np.array([30.0, 35.0, 40.0]),
            indoor_c=np.array([24.0, 24.0, 24.0]),
            power_kw=np.array([2.0, 2.0, 2.0]),
            capacity_kw=np.array([6.4, 6.0, 5.6]),
        )
        self.assertGreater(calibrated.cop, 2.0)
        self.assertLess(calibrated.cop_temperature_slope_per_c, 0.0)

    def test_resstock_extraction_converts_interval_energy_to_power(self) -> None:
        frame = build_calibration_dataset(
            pd.DataFrame(
                {
                    "timestamp": ["2018-07-18 00:00:00"],
                    "out.zone_mean_air_temp.conditioned_space.c": [24.0],
                    "out.outdoor_air_dryblub_temp.c": [32.0],
                    "out.electricity.cooling.energy_consumption": [0.5],
                    "out.electricity.cooling_fans_pumps.energy_consumption": [0.1],
                    "out.load.cooling.energy_delivered.kbtu": [10.0],
                }
            )
        )
        self.assertAlmostEqual(frame["cooling_electric_kw"].iloc[0], 2.0)
        self.assertAlmostEqual(frame["cooling_fans_pumps_kw"].iloc[0], 0.4)
        self.assertAlmostEqual(frame["cooling_delivered_kw"].iloc[0], 11.7228428068)
        self.assertAlmostEqual(fahrenheit_to_celsius(75.0), 23.8888888889)
        self.assertAlmostEqual(parse_seer("AC, SEER 15"), 15.0)

    def test_dynamic_grid_cosim_backend_steps_frequency(self) -> None:
        backend = SwingEquationDynamicBackend(SwingFrequencyParameters(base_power_kw=5000.0))
        cosim = DynamicGridCosimulator(backend, nominal_iac_kw=1000.0)
        output = cosim.step(0.0, iac_power_kw=1000.0, disturbance_kw=1000.0, sample_time_s=5.0)
        self.assertLess(output.frequency_deviation_hz, 0.0)
        self.assertTrue(output.converged)

    def test_csv_replay_backend_superposes_iac_step_response(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.csv"
            path.write_text(
                "time_s,frequency_deviation_hz,iac_step_response_hz_per_kw\n"
                "0,0.0,0.0\n"
                "5,-0.1,0.001\n"
                "10,-0.2,0.002\n",
                encoding="utf-8",
            )
            backend = CsvReplayDynamicBackend(path)
            cosim = DynamicGridCosimulator(backend, nominal_iac_kw=1000.0)
            no_relief = cosim.step(0.0, iac_power_kw=1000.0, disturbance_kw=0.0, sample_time_s=5.0)
            _ = cosim.step(5.0, iac_power_kw=900.0, disturbance_kw=0.0, sample_time_s=5.0)
            with_relief = cosim.step(10.0, iac_power_kw=900.0, disturbance_kw=0.0, sample_time_s=5.0)
            self.assertAlmostEqual(no_relief.frequency_deviation_hz, -0.1)
            self.assertGreater(with_relief.frequency_deviation_hz, -0.2)

    def test_voltage_support_opf_improves_sensitivity_voltage(self) -> None:
        feeder = SingleBusTheveninFeeder(FeederParameters(base_power_kva=5000.0, fixed_load_kw=900.0, fixed_load_kvar=450.0))
        sensitivity = feeder.estimate_voltage_sensitivity(1.0, iac_p_kw=1000.0, iac_q_kvar=0.0)
        result = solve_voltage_support_opf(
            sensitivity,
            InverterCapability(4.0, 0.6, 3.5, 2.2),
            p_baseline_kw=2.0,
            q_baseline_kvar=0.0,
            fleet_size=500,
        )
        self.assertGreaterEqual(result.q_kvar, -1e-6)
        self.assertGreaterEqual(result.predicted_voltage_pu, sensitivity.base_voltage_pu - 1e-6)

    def test_opf_comparison_reports_small_prediction_error_for_thevenin(self) -> None:
        feeder = SingleBusTheveninFeeder(FeederParameters(base_power_kva=5000.0, fixed_load_kw=900.0, fixed_load_kvar=450.0))
        comparison = compare_sensitivity_opf_to_power_flow(
            feeder,
            source_voltage_pu=1.0,
            p_baseline_kw=2.0,
            q_baseline_kvar=0.0,
            fleet_size=500,
            capability=InverterCapability(4.0, 0.6, 3.5, 2.2),
        )
        self.assertLess(abs(comparison.voltage_prediction_error_pu), 0.02)


if __name__ == "__main__":
    unittest.main()
