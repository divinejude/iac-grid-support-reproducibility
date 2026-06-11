"""Compare OpenDSS finite-difference sensitivity with convex voltage OPF."""

from __future__ import annotations

import csv
from pathlib import Path

from iac_dmpc.capability import InverterCapability
from iac_dmpc.feeder import IEEE13OpenDSSFleetFeeder
from iac_dmpc.opf import compare_sensitivity_opf_to_power_flow
from iac_dmpc.parameters import InverterParameters
from simulate_iac_grid_support import PROJECT_ROOT, SimulationScenario


def run_comparison(output_dir: Path | None = None) -> Path:
    output = PROJECT_ROOT / "experiments" / "opf_comparison" if output_dir is None else output_dir
    output.mkdir(parents=True, exist_ok=True)
    path = output / "opf_comparison.csv"

    inverter = InverterParameters()
    capability = InverterCapability(
        inverter.s_rated_kva,
        inverter.p_min_kw,
        inverter.p_max_kw,
        inverter.q_max_kvar,
    )
    scenarios = [
        SimulationScenario(name="source_sag_0.92", voltage_sag_pu=0.92),
        SimulationScenario(
            name="feeder_stress_1200kw_600kvar",
            voltage_event_type="feeder_load_step",
            voltage_sag_pu=1.0,
            feeder_extra_load_kw=1200.0,
            feeder_extra_load_kvar=600.0,
        ),
    ]

    rows: list[dict[str, float | str]] = []
    for scenario in scenarios:
        feeder = IEEE13OpenDSSFleetFeeder(allocation_strategy=scenario.allocation_strategy)
        fleet_size = scenario.resolved_fleet_size(inverter.p_set_kw)
        source_voltage_pu = 1.0 if scenario.voltage_event_type == "feeder_load_step" else scenario.voltage_sag_pu
        comparison = compare_sensitivity_opf_to_power_flow(
            feeder=feeder,
            source_voltage_pu=source_voltage_pu,
            p_baseline_kw=inverter.p_set_kw,
            q_baseline_kvar=0.0,
            fleet_size=fleet_size,
            capability=capability,
            extra_load_kw=scenario.feeder_extra_load_kw,
            extra_load_kvar=scenario.feeder_extra_load_kvar,
            metric="min",
        )
        rows.append(
            {
                "scenario": scenario.name,
                "fleet_size": fleet_size,
                "base_voltage_pu": comparison.sensitivity.base_voltage_pu,
                "dv_dp_pu_per_kw": comparison.sensitivity.dv_dp_pu_per_kw,
                "dv_dq_pu_per_kvar": comparison.sensitivity.dv_dq_pu_per_kvar,
                "opf_p_kw_per_unit": comparison.opf.p_kw,
                "opf_q_kvar_per_unit": comparison.opf.q_kvar,
                "opf_predicted_voltage_pu": comparison.opf.predicted_voltage_pu,
                "opendss_actual_voltage_pu": comparison.opendss_solution.min_voltage_pu,
                "prediction_error_pu": comparison.voltage_prediction_error_pu,
                "opf_status": comparison.opf.status,
            }
        )

    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


if __name__ == "__main__":
    print(f"Wrote OPF comparison to {run_comparison()}")
