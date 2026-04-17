import sys
from pathlib import Path
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from control.runtime import run_chesso_runtime
from experiments import compare_runtime_modes, make_noise_stress_workload, run_objective_ablation
from viz import (
    ablation_rows,
    benchmark_rows,
    export_ablation_summary,
    export_benchmark_summary,
    export_runtime_telemetry,
    render_experiment_report,
    runtime_telemetry,
)


def main() -> None:
    workload = make_noise_stress_workload().materialize()
    report = run_chesso_runtime(workload.bundle, graph=workload.graph, target=workload.target, steps=2)
    telemetry = runtime_telemetry(report)
    assert telemetry.mode == "runtime"
    assert len(telemetry.points) == 2
    assert telemetry.points[-1].active_sectors >= 2

    comparison = compare_runtime_modes(make_noise_stress_workload())
    bench_rows = benchmark_rows(comparison)
    assert len(bench_rows) == 2
    assert {row.execution_kind for row in bench_rows} == {"runtime", "scheduled"}

    ablation = run_objective_ablation(make_noise_stress_workload())
    abl_rows = ablation_rows(ablation)
    assert len(abl_rows) >= 4
    assert any(row.label == "default" for row in abl_rows)

    with TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        runtime_export = export_runtime_telemetry(report, tmp_path, prefix="sec14")
        bench_export = export_benchmark_summary(comparison, tmp_path, prefix="sec14")
        abl_export = export_ablation_summary(ablation, tmp_path, prefix="sec14")
        plot_files = render_experiment_report(telemetry, comparison, ablation, tmp_path, prefix="sec14")

        for bundle in (runtime_export, bench_export, abl_export):
            for path in bundle.files.values():
                assert path.exists() and path.stat().st_size > 0

        for path in plot_files.values():
            assert path.exists() and path.stat().st_size > 0

    print("Section 14 smoke test passed")


if __name__ == "__main__":
    main()
