import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vqpu.chesso.experiments import (
    compare_runtime_modes,
    make_bell_workload,
    make_expand_compress_workload,
    make_noise_stress_workload,
    run_objective_ablation,
    run_standard_suite,
    run_workload,
)


def main() -> None:
    bell = run_workload(make_bell_workload())
    assert bell.execution_kind.value == "compiler"
    assert bell.fidelity is not None and bell.fidelity > 0.999999
    assert bell.active_sector_count == 2

    suite = run_standard_suite()
    assert len(suite.results) == 5
    names = {item.workload_name for item in suite.results}
    assert "bell_compiler" in names
    assert "ghz_3q_compiler" in names
    assert "w_3q_prepared" in names
    assert suite.comparison is not None

    comparison = compare_runtime_modes(make_noise_stress_workload())
    assert comparison.runtime_result.execution_kind.value == "runtime"
    assert comparison.scheduled_result.execution_kind.value == "scheduled"
    assert comparison.runtime_result.metadata["runtime_reports"] == 2
    assert comparison.scheduled_result.metadata["runtime_reports"] == 2

    expand_case = run_workload(make_expand_compress_workload())
    assert expand_case.dynamic_depth >= 1
    assert expand_case.active_sector_count >= 2

    ablation = run_objective_ablation(make_noise_stress_workload())
    assert len(ablation.entries) >= 4
    labels = {entry.label for entry in ablation.entries}
    assert "default" in labels and "noise_averse" in labels
    default_entry = next(entry for entry in ablation.entries if entry.label == "default")
    no_ent = next(entry for entry in ablation.entries if entry.label == "no_entanglement_reward")
    assert default_entry.total_score != no_ent.total_score

    print("Section 13 smoke test passed")


if __name__ == "__main__":
    main()
