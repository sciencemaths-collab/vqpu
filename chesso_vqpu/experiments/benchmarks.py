from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, Dict, Iterable, List, Tuple

from compiler import execute_plan, lower_program
from control import evaluate_objective, run_chesso_runtime, run_scheduled_runtime
from core import EntanglementHypergraph, HilbertBundleState

from .workloads import MaterializedWorkload, WorkloadExecutionKind, WorkloadSpec, standard_workloads


@dataclass(slots=True)
class BenchmarkResult:
    """Normalized result for one executed workload."""

    workload_name: str
    execution_kind: WorkloadExecutionKind
    elapsed_seconds: float
    fidelity: float | None
    objective_score: float
    entanglement_score: float
    coherence_score: float
    noise_penalty: float
    active_sector_count: int
    hyperedge_count: int
    measurements_used: int
    dynamic_depth: int
    succeeded: bool = True
    final_bundle: HilbertBundleState | None = None
    final_graph: EntanglementHypergraph | None = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RuntimeModeComparison:
    """Side-by-side runtime and scheduled benchmark outputs."""

    workload_name: str
    runtime_result: BenchmarkResult
    scheduled_result: BenchmarkResult
    objective_gap: float
    fidelity_gap: float | None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class BenchmarkSuiteResult:
    """Collection of benchmark runs plus a compact summary."""

    results: Tuple[BenchmarkResult, ...]
    comparison: RuntimeModeComparison | None = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class _ExecutionOutcome:
    bundle: HilbertBundleState
    graph: EntanglementHypergraph | None
    metadata: Dict[str, Any] = field(default_factory=dict)


def _compatible_target(bundle: HilbertBundleState, target: Any) -> Any:
    if isinstance(target, HilbertBundleState) and target.dims != bundle.dims:
        return None
    if hasattr(target, "shape"):
        try:
            dim = int(getattr(target, "shape")[0])
            if bundle.total_dimension != dim:
                return None
        except Exception:
            pass
    return target


def _materialize(spec_or_workload: WorkloadSpec | MaterializedWorkload) -> MaterializedWorkload:
    return spec_or_workload.materialize() if isinstance(spec_or_workload, WorkloadSpec) else spec_or_workload


def _run_materialized(workload: MaterializedWorkload, execution_kind: WorkloadExecutionKind) -> _ExecutionOutcome:
    if execution_kind == WorkloadExecutionKind.COMPILER:
        if workload.program is None:
            raise ValueError(f"Compiler workload {workload.name!r} is missing a program")
        plan = lower_program(workload.program, config=workload.config)
        result = execute_plan(plan, config=workload.config)
        return _ExecutionOutcome(
            bundle=result.bundle,
            graph=result.graph,
            metadata={
                "call_count": len(plan.call_specs),
                "runtime_reports": len(result.runtime_reports),
                "program_name": workload.program.name,
            },
        )

    if execution_kind == WorkloadExecutionKind.PREPARED:
        if workload.bundle is None:
            raise ValueError(f"Prepared workload {workload.name!r} is missing a bundle")
        return _ExecutionOutcome(bundle=workload.bundle, graph=workload.graph, metadata={"prepared": True})

    if workload.bundle is None:
        raise ValueError(f"Runtime workload {workload.name!r} is missing an initial bundle")

    if execution_kind == WorkloadExecutionKind.SCHEDULED:
        report = run_scheduled_runtime(
            workload.bundle,
            graph=workload.graph,
            target=workload.target,
            steps=workload.steps,
            stop_fidelity=None,
            max_entanglers=workload.max_entanglers,
            in_place=False,
        )
        return _ExecutionOutcome(
            bundle=report.final_bundle,
            graph=report.final_graph,
            metadata={
                "runtime_reports": len(report.steps),
                "scheduled": True,
                "stopped_early": report.metadata.get("stopped_early", False),
            },
        )

    report = run_chesso_runtime(
        workload.bundle,
        graph=workload.graph,
        target=workload.target,
        steps=workload.steps,
        stop_fidelity=None,
        max_entanglers=workload.max_entanglers,
        in_place=False,
    )
    return _ExecutionOutcome(
        bundle=report.final_bundle,
        graph=report.final_graph,
        metadata={
            "runtime_reports": len(report.steps),
            "scheduled": False,
            "stopped_early": report.metadata.get("stopped_early", False),
        },
    )


def run_workload(
    spec_or_workload: WorkloadSpec | MaterializedWorkload,
    *,
    execution_kind: WorkloadExecutionKind | None = None,
) -> BenchmarkResult:
    """Execute one workload and return normalized benchmark metrics."""
    workload = _materialize(spec_or_workload)
    kind = workload.execution_kind if execution_kind is None else WorkloadExecutionKind(execution_kind)

    t0 = perf_counter()
    outcome = _run_materialized(workload, kind)
    elapsed = perf_counter() - t0

    use_target = _compatible_target(outcome.bundle, workload.target)
    objective = evaluate_objective(outcome.bundle, graph=outcome.graph, target=use_target)
    metric = objective.metric_report
    return BenchmarkResult(
        workload_name=workload.name,
        execution_kind=kind,
        elapsed_seconds=float(elapsed),
        fidelity=metric.fidelity_to_target,
        objective_score=float(objective.total_score),
        entanglement_score=float(objective.entanglement_score),
        coherence_score=float(objective.coherence_score),
        noise_penalty=float(objective.noise_penalty),
        active_sector_count=outcome.bundle.active_sector_count,
        hyperedge_count=0 if outcome.graph is None else len(outcome.graph.hyperedges),
        measurements_used=int(outcome.bundle.stats.measurements_used),
        dynamic_depth=int(outcome.bundle.stats.dynamic_depth),
        succeeded=True,
        final_bundle=outcome.bundle,
        final_graph=outcome.graph,
        metadata={
            "description": workload.description,
            "tags": workload.tags,
            "steps": int(workload.steps),
            "target_compatible": use_target is not None,
            **outcome.metadata,
        },
    )


def compare_runtime_modes(spec_or_workload: WorkloadSpec | MaterializedWorkload) -> RuntimeModeComparison:
    """Run the same workload through normal runtime and scheduled runtime."""
    workload = _materialize(spec_or_workload)
    if workload.bundle is None:
        raise ValueError("Runtime mode comparison requires a bundle-backed workload")

    runtime_result = run_workload(workload, execution_kind=WorkloadExecutionKind.RUNTIME)
    scheduled_input = _materialize(spec_or_workload)
    scheduled_result = run_workload(scheduled_input, execution_kind=WorkloadExecutionKind.SCHEDULED)
    fidelity_gap = None
    if runtime_result.fidelity is not None and scheduled_result.fidelity is not None:
        fidelity_gap = float(scheduled_result.fidelity - runtime_result.fidelity)
    return RuntimeModeComparison(
        workload_name=workload.name,
        runtime_result=runtime_result,
        scheduled_result=scheduled_result,
        objective_gap=float(scheduled_result.objective_score - runtime_result.objective_score),
        fidelity_gap=fidelity_gap,
        metadata={
            "steps": int(workload.steps),
            "max_entanglers": int(workload.max_entanglers),
        },
    )


def run_standard_suite() -> BenchmarkSuiteResult:
    """Execute the canonical Section 13 workload suite."""
    specs = standard_workloads()
    results: List[BenchmarkResult] = [run_workload(spec) for spec in specs]
    comparison = compare_runtime_modes(specs[3])
    best = max(results, key=lambda item: item.objective_score)
    return BenchmarkSuiteResult(
        results=tuple(results),
        comparison=comparison,
        metadata={
            "workload_count": len(results),
            "best_workload": best.workload_name,
            "best_objective": float(best.objective_score),
        },
    )


__all__ = [
    "BenchmarkResult",
    "BenchmarkSuiteResult",
    "RuntimeModeComparison",
    "compare_runtime_modes",
    "run_standard_suite",
    "run_workload",
]
