# Reproducible Experiment Suite

Run:

```powershell
python run_experiment_suite.py --scenario-set quick
```

The experiment runner evaluates five controller modes:

- `no_support`: fixed active power, no reactive support
- `rule_based`: five-region Frequency-Watt plus Volt-Var
- `dmpc_active_only`: DMPC active power with reactive support disabled
- `dmpc_pq`: DMPC active power plus separate Volt-Var reactive support
- `dmpc_full_pq`: full P/Q DMPC with OpenDSS voltage sensitivity and inverter circle constraints

Each case writes:

- Time-series CSV
- Case plot PNG
- Summary metrics row

The top-level `metrics_summary.csv` includes:

- Frequency nadir and maximum absolute frequency deviation
- Minimum, maximum, and mean fleet-weighted IEEE 13-node voltage
- Minimum and maximum bus-phase voltage across the IEEE 13-node feeder
- Temperature limits and comfort violation metrics
- Aggregate energy and reactive support
- Apparent-power utilization
- MPC feasibility and compute time

Scenario sets:

- `quick`: one higher-impact IEEE 13-node scenario with 500 TCLs and a 0.92 pu sag
- `voltage_sweep`: sag depths 0.90, 0.92, 0.94 pu
- `feeder_stress`: local feeder load steps of 600/300, 900/450, and 1200/600 kW/kvar
- `headroom_sweep`: physically explicit inverter kVA/Q/current headroom at 1.00, 1.25, and 1.50 pu
- `voltage_support_upgrade`: focused voltage-improvement suite combining feeder stress and headroom sweeps
- `fleet_sweep`: 300, 500, 700 TCLs
- `tcl_share_sweep`: TCL aggregate participation at 10%, 20%, 30%, and 40% of original IEEE 13-node feeder load
- `tcl_share_weak_bus_sweep`: TCL participation sweep with weak-bus-prioritized placement
- `placement_sweep`: load-proportional and weak-bus TCL placement sweeps
- `comfort_sweep`: +/-0.5, +/-1.0, +/-2.0 C
- `full`: combined sweep set
- `monte_carlo`: randomized physical uncertainty, sensor noise, delay, feeder sag, and fleet-size samples

Paper summaries:

```powershell
python summarize_experiments.py --experiment-dir experiments/full --experiment-dir experiments/monte_carlo --experiment-dir experiments/voltage_support_upgrade
```
