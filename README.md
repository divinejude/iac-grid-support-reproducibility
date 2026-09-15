# Inverter-Based TCL Grid-Support Simulation

This repository contains a modular Python framework for an inverter-based air conditioner (IAC) that provides simultaneous active and reactive grid support:

- SOGI-PLL single-phase current decomposition into `Id` and `Iq`
- Apparent-power and compressor mechanical constraints
- OpenDSS co-simulation on the IEEE 13-node benchmark feeder and swing-equation bulk-frequency model
- Dynamic frequency replay trace with governor-turbine dynamics and an IAC load-relief response kernel
- Optional ANDES IEEE 14-bus nonlinear multi-machine transient-stability check
- Explicit ANDES/OpenDSS T&D dynamic co-simulation with 100 ms explicit coupling
- Inverter current-loop bandwidth, current saturation, DC-link limit, and anti-windup
- Averaged inverter trackability check with SRF-PLL, dq PI current control, L-filter dynamics, PWM delay, and DC-link capacitance
- Continuous-time thermal, compressor, and nominal prediction plant models
- Compressor ramp-rate, speed-map, dwell-time, and temperature-dependent COP constraints
- Measurement noise, fixed communication delay, and parameter uncertainty utilities
- Rule-based Volt-Var and five-region Frequency-Watt logic
- Fleet MPC active-power dispatch with strict thermal comfort constraints
- Network-aware full P/Q fleet MPC using OpenDSS finite-difference voltage sensitivity
- Thermal calibration utilities for field/EnergyPlus CSV data and manufacturer performance maps
- Dynamic grid co-simulation backend interface for replacing the built-in swing model
- Convex voltage-support OPF comparison against nonlinear OpenDSS power-flow results
- Public NIST NZERTF measured-data calibration workflow for thermal RC/COP parameters
- Public ResStock/EnergyPlus calibration workflow for comparison thermal RC/COP parameters

## Install

```powershell
pip install -r requirements.txt
```

For the optional nonlinear transient-stability validation:

```powershell
pip install -r requirements-validation.txt
```

If the normal ANDES install hits a Windows long-path error while copying optional Jupyter widget assets, enable Windows long paths or use the lean runtime path documented in `requirements-validation.txt`.

## Calibrate Thermal Parameters From Public Measured NIST NZERTF Data

```powershell
python calibrate_from_nist_nzertf.py
```

The measured-data calibration script downloads public NIST Net-Zero Energy Residential Test Facility minute-resolution files, extracts:

- occupied-zone indoor air temperature
- outdoor ambient temperature
- heat-pump indoor- and outdoor-unit electric power
- supply/return air temperatures for cooling-interval detection

It writes:

- `data/nist_nzertf/calibrated_thermal_parameters.json`
- `data/nist_nzertf/nist_nzertf_calibration_timeseries.csv`
- `data/nist_nzertf/nist_nzertf_calibration_fit.png`

Use the measured-data calibrated parameters in any experiment suite with:

```powershell
python run_experiment_suite.py --scenario-set full --thermal-calibration-json data/nist_nzertf/calibrated_thermal_parameters.json --output-dir experiments/nist_measured_calibrated
```

The script infers delivered cooling capacity through the fitted effective COP model because the public minute HVAC file does not directly report delivered cooling load. The resulting parameters are effective whole-building parameters for the instrumented NZERTF test house.

## Calibrate Thermal Parameters From Public ResStock/EnergyPlus Data

```powershell
python calibrate_from_resstock.py
```

The calibration script downloads a selected public NREL ResStock AMY2018 2024.2 individual-building timeseries and state metadata file from OEDI, extracts:

- conditioned-space indoor air temperature
- outdoor dry-bulb temperature
- cooling electric power
- delivered cooling load

It writes:

- `data/resstock/calibrated_thermal_parameters.json`
- `data/resstock/resstock_calibration_timeseries.csv`
- `data/resstock/resstock_calibration_fit.png`

Use the calibrated parameters in any experiment suite with:

```powershell
python run_experiment_suite.py --scenario-set full --thermal-calibration-json data/resstock/calibrated_thermal_parameters.json
```

## Run The Demo

```powershell
python simulate_iac_grid_support.py
```

The script applies a frequency drop from 120 s to 420 s and a voltage sag from 180 s to 380 s. It writes:

- `outputs/iac_dmpc_demo.csv`
- `outputs/iac_dmpc_demo.png`

The expected behavior is that the controller reduces active compressor power during under-frequency, injects reactive power during the voltage sag, respects the inverter apparent-power circle, and keeps indoor temperature inside the configured comfort band.

The main simulation uses `IEEE13OpenDSSFleetFeeder` from `iac_dmpc/feeder.py`. It compiles the IEEE 13-node feeder from `examples/ieee13/IEEE13Nodeckt.dss`, adds controllable TCL loads at the original benchmark load buses/phases, and distributes aggregate TCL P/Q in proportion to each original load's kW.

See `docs/modeling_assumptions.md` for the main physical assumptions and how to extend the OpenDSS feeder to IEEE 13-node or 34-node studies.

## Run Reproducible Experiments

Generate or refresh the bundled dynamic frequency replay trace:

```powershell
python generate_dynamic_frequency_trace.py
```

```powershell
python run_experiment_suite.py --scenario-set quick
```

Available scenario sets:

- `quick`: one higher-impact IEEE 13-node scenario across five controllers
- `voltage_sweep`: voltage sag depth sweep
- `feeder_stress`: feeder-local load step sweep for voltage-support studies
- `headroom_sweep`: inverter kVA/Q/current headroom sweep
- `voltage_support_upgrade`: focused voltage-improvement suite
- `fleet_sweep`: TCL fleet-size sweep
- `tcl_share_sweep`: TCL penetration sweep at 10%, 20%, 30%, and 40% of IEEE 13-node original load
- `tcl_share_weak_bus_sweep`: TCL penetration sweep with weak-bus-prioritized placement
- `placement_sweep`: load-proportional and weak-bus placement comparison
- `comfort_sweep`: comfort-band sweep
- `full`: all of the above
- `monte_carlo`: limited stochastic diagnostic cases for noise, delay, compressor-time, sampled fleet size, and sag settings; with fixed NIST thermal calibration, this is not statistical Monte Carlo robustness evidence

The suite writes per-case CSV/PNG files, a `metrics_summary.csv` table, and comparison figures under `experiments/<scenario-set>/`. The deposited research-data package includes the derived calibration files, dynamic replay trace, paper-ready summary outputs, and a compact representative subset of full experiment traces used by the default validation and paper-asset commands. The complete full-suite per-case folders can be regenerated by the commands below. The default quick case uses a 500-unit TCL fleet, a 0.92 pu voltage sag, fast local Volt-Var response, full P/Q MPC, and reports both fleet-weighted and worst feeder bus-phase voltage.

For a fast reviewer smoke test after extracting the research-data package:

```powershell
python -m pytest tests
python generate_dynamic_frequency_trace.py
python run_experiment_suite.py --scenario-set quick --thermal-calibration-json data/nist_nzertf/calibrated_thermal_parameters.json --output-dir experiments/reviewer_quick_check
python summarize_experiments.py --experiment-dir experiments/reviewer_quick_check/quick --output-dir paper_outputs_reviewer_quick
```

Generate paper summary tables and figures:

```powershell
python run_experiment_suite.py --scenario-set full --thermal-calibration-json data/nist_nzertf/calibrated_thermal_parameters.json --output-dir experiments/nist_measured_calibrated
python run_experiment_suite.py --scenario-set monte_carlo --thermal-calibration-json data/nist_nzertf/calibrated_thermal_parameters.json --output-dir experiments/nist_measured_calibrated
python summarize_experiments.py --experiment-dir experiments/nist_measured_calibrated/full --experiment-dir experiments/nist_measured_calibrated/monte_carlo --output-dir paper_outputs_nist_measured_calibrated
python summarize_experiments.py --experiment-dir experiments/dynamic_replay_calibrated/full --experiment-dir experiments/dynamic_replay_calibrated/monte_carlo --output-dir paper_outputs_dynamic_replay_calibrated
python paper/generate_paper_assets.py
python scripts/sync_manuscript_assets.py
```

The authoritative EPSR manuscript tree is `submission_epsr/manuscript/`.
The `paper/` directory is an intermediate generation workspace; final
manuscript-facing figures and tables are synchronized into
`submission_epsr/manuscript/figures/` and `submission_epsr/manuscript/tables/`
with hash verification by `scripts/sync_manuscript_assets.py`.

The default `monte_carlo` suite name is retained for command-line reproducibility, but the bundled three-case NIST-calibrated output should be read as a limited stochastic diagnostic rather than a statistical robustness study.

Run the regression tests:

```powershell
python -m pytest tests
```

On some Windows/OpenDSS installations, `opendssdirect` may print a backend shutdown stack trace after the tests have already reported success. The run is considered successful when pytest reports all tests passed and returns exit code 0.

Before uploading local EPSR submission or public reproducibility artifacts, run:

```powershell
python scripts/package_preflight.py
```

The preflight checks required files, public DOI/GitHub references, stale
controller terminology, internal coordination files, caches, build logs, and
absolute local paths.

Compare the OpenDSS sensitivity approximation with a convex voltage-support OPF:

```powershell
python compare_opf_sensitivity.py
```

Run nonlinear transient-stability validation with ANDES:

```powershell
python validate_andes_transient.py
```

This builds controller-specific IEEE 14-bus dynamic cases from the ANDES bundled benchmark, injects aggregate IAC active-power relief as timed TGOV1 auxiliary-power events, runs nonlinear TDS, and writes:

- `experiments/andes_validation/andes_frequency_validation.csv`
- `experiments/andes_validation/andes_validation_metrics.csv`
- `paper/figures/andes_frequency_validation.pdf`
- `paper/tables/andes_validation.tex`

Run the explicit iterative T&D dynamic co-simulation:

```powershell
python validate_td_cosim.py --coupling-step-s 0.1
```

This advances the ANDES IEEE 14-bus dynamic case in short TDS segments, sends the current interface-bus voltage to the OpenDSS IEEE 13-node feeder at every coupling step, solves the feeder with the current IAC P/Q trajectory, and returns the feeder boundary active-power change to ANDES. It writes:

- `experiments/td_cosim/td_cosim_traces.csv`
- `experiments/td_cosim/td_cosim_metrics.csv`
- `paper/figures/td_cosim_response.pdf`
- `paper/tables/td_cosim.tex`

Run averaged inverter/converter validation:

```powershell
python validate_averaged_converter.py
```

This consumes the latest full P/Q MPC source-sag trajectory, simulates an averaged single-phase inverter over the stressed transition, injects an additional fast voltage notch and frequency pulse, and writes:

- `experiments/converter_validation/averaged_converter_validation.csv`
- `experiments/converter_validation/averaged_converter_metrics.csv`
- `paper/figures/averaged_converter_validation.pdf`
- `paper/tables/averaged_converter_validation.tex`

Run switching-level inverter validation:

```powershell
python validate_switching_inverter.py
```

The validation commands above use the bundled representative `voltage_sag_0.92` full P/Q trace by default. If those traces have been moved or removed, either regenerate the full experiment suite or pass an explicit quick-suite trace with `--trace`.
