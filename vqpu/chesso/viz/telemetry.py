from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

from vqpu.chesso.control.runtime import RuntimeRunReport, RuntimeStepReport
from vqpu.chesso.control.scheduler import ScheduledRuntimeRunReport, ScheduledRuntimeStepReport
from vqpu.chesso.experiments.ablations import ObjectiveAblationEntry, ObjectiveAblationResult
from vqpu.chesso.experiments.benchmarks import BenchmarkResult, BenchmarkSuiteResult, RuntimeModeComparison


@dataclass(slots=True)
class TelemetryPoint:
    """Compact one-step runtime telemetry snapshot."""

    step: int
    objective_score: float
    fidelity: float
    entanglement: float
    coherence: float
    noise_penalty: float
    resource_penalty: float
    measurement_strength: float
    active_sectors: int
    hyperedges: int
    measurements_used: int
    dynamic_depth: int
    discarded_trace_mass: float
    expansion_count: int
    entangler_count: int
    prune_count: int
    route_count: int
    representation: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RuntimeTelemetry:
    """Time-series telemetry extracted from a runtime report."""

    mode: str
    points: Tuple[TelemetryPoint, ...]
    initial_snapshot: Dict[str, Any]
    final_snapshot: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class BenchmarkSummaryRow:
    """Flat benchmark row suitable for export and plotting."""

    workload_name: str
    execution_kind: str
    objective_score: float
    fidelity: float | None
    entanglement_score: float
    coherence_score: float
    noise_penalty: float
    active_sector_count: int
    hyperedge_count: int
    measurements_used: int
    dynamic_depth: int
    elapsed_seconds: float
    succeeded: bool
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class AblationSummaryRow:
    """Flat objective ablation row suitable for export and plotting."""

    label: str
    total_score: float
    fidelity_score: float
    entanglement_score: float
    coherence_score: float
    noise_penalty: float
    resource_penalty: float
    measurement_strength: float
    expansion_count: int
    prune_accept_count: int
    route_count: int
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ExportBundle:
    """Paths emitted by a telemetry export helper."""

    directory: Path
    files: Dict[str, Path] = field(default_factory=dict)


TelemetryReport = RuntimeRunReport | ScheduledRuntimeRunReport


def _snapshot_to_dict(snapshot: Any) -> Dict[str, Any]:
    raw = asdict(snapshot)
    rep = raw.get("representation")
    if rep is not None:
        raw["representation"] = getattr(rep, "value", str(rep))
    return raw


def _step_from_any(step: RuntimeStepReport | ScheduledRuntimeStepReport) -> RuntimeStepReport:
    return step.runtime if isinstance(step, ScheduledRuntimeStepReport) else step


def runtime_telemetry(report: TelemetryReport) -> RuntimeTelemetry:
    """Extract a normalized runtime trace from either runtime mode."""
    if isinstance(report, ScheduledRuntimeRunReport):
        mode = "scheduled"
        raw_steps = tuple(report.steps)
    else:
        mode = "runtime"
        raw_steps = tuple(report.steps)

    points: List[TelemetryPoint] = []
    for raw in raw_steps:
        step = _step_from_any(raw)
        objective = step.decision.objective
        point = TelemetryPoint(
            step=int(step.step_index),
            objective_score=float(objective.total_score),
            fidelity=float(objective.fidelity_score),
            entanglement=float(objective.entanglement_score),
            coherence=float(objective.coherence_score),
            noise_penalty=float(objective.noise_penalty),
            resource_penalty=float(objective.resource_penalty),
            measurement_strength=float(step.decision.measurement_strength),
            active_sectors=int(step.bundle.active_sector_count),
            hyperedges=0 if step.graph is None else len(step.graph.hyperedges),
            measurements_used=int(step.bundle.stats.measurements_used),
            dynamic_depth=int(step.bundle.stats.dynamic_depth),
            discarded_trace_mass=float(step.bundle.stats.discarded_trace_mass),
            expansion_count=len(step.action_summary.expansions),
            entangler_count=len(step.action_summary.entanglers),
            prune_count=len(step.action_summary.prunes),
            route_count=len(step.action_summary.routes),
            representation=str(step.bundle.quantum_state.representation.value),
            metadata={
                "measurement_records": len(step.measurement_records),
                "graph_summary": None if step.graph_summary is None else asdict(step.graph_summary),
                **dict(step.metadata),
            },
        )
        points.append(point)

    return RuntimeTelemetry(
        mode=mode,
        points=tuple(points),
        initial_snapshot=_snapshot_to_dict(report.initial_snapshot),
        final_snapshot=_snapshot_to_dict(report.final_snapshot),
        metadata=dict(getattr(report, "metadata", {})),
    )


def benchmark_rows(suite_or_results: BenchmarkSuiteResult | RuntimeModeComparison | Sequence[BenchmarkResult]) -> Tuple[BenchmarkSummaryRow, ...]:
    """Normalize benchmark outputs into flat rows."""
    if isinstance(suite_or_results, BenchmarkSuiteResult):
        results = tuple(suite_or_results.results)
        extra_meta = dict(suite_or_results.metadata)
    elif isinstance(suite_or_results, RuntimeModeComparison):
        results = (suite_or_results.runtime_result, suite_or_results.scheduled_result)
        extra_meta = dict(suite_or_results.metadata)
    else:
        results = tuple(suite_or_results)
        extra_meta = {}

    rows = []
    for result in results:
        rows.append(
            BenchmarkSummaryRow(
                workload_name=result.workload_name,
                execution_kind=result.execution_kind.value,
                objective_score=float(result.objective_score),
                fidelity=None if result.fidelity is None else float(result.fidelity),
                entanglement_score=float(result.entanglement_score),
                coherence_score=float(result.coherence_score),
                noise_penalty=float(result.noise_penalty),
                active_sector_count=int(result.active_sector_count),
                hyperedge_count=int(result.hyperedge_count),
                measurements_used=int(result.measurements_used),
                dynamic_depth=int(result.dynamic_depth),
                elapsed_seconds=float(result.elapsed_seconds),
                succeeded=bool(result.succeeded),
                metadata={**extra_meta, **dict(result.metadata)},
            )
        )
    return tuple(rows)


def ablation_rows(result: ObjectiveAblationResult) -> Tuple[AblationSummaryRow, ...]:
    """Flatten objective ablation output for report/export use."""
    rows = []
    for entry in result.entries:
        rows.append(
            AblationSummaryRow(
                label=entry.label,
                total_score=float(entry.total_score),
                fidelity_score=float(entry.fidelity_score),
                entanglement_score=float(entry.entanglement_score),
                coherence_score=float(entry.coherence_score),
                noise_penalty=float(entry.noise_penalty),
                resource_penalty=float(entry.resource_penalty),
                measurement_strength=float(entry.measurement_strength),
                expansion_count=int(entry.expansion_count),
                prune_accept_count=int(entry.prune_accept_count),
                route_count=int(entry.route_count),
                metadata={**dict(result.metadata), **dict(entry.metadata)},
            )
        )
    return tuple(rows)


def _write_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True, default=str)
    return path


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with path.open("w", encoding="utf-8", newline="") as fh:
            fh.write("")
        return path
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(row))
    return path


def export_runtime_telemetry(report: TelemetryReport, directory: str | Path, *, prefix: str = "runtime") -> ExportBundle:
    """Export runtime telemetry as JSON and CSV."""
    out_dir = Path(directory)
    trace = runtime_telemetry(report)
    json_payload = {
        "mode": trace.mode,
        "initial_snapshot": trace.initial_snapshot,
        "final_snapshot": trace.final_snapshot,
        "metadata": trace.metadata,
        "points": [asdict(point) for point in trace.points],
    }
    csv_rows = [
        {
            k: v
            for k, v in asdict(point).items()
            if k != "metadata"
        }
        for point in trace.points
    ]
    files = {
        "runtime_json": _write_json(out_dir / f"{prefix}_telemetry.json", json_payload),
        "runtime_csv": _write_csv(out_dir / f"{prefix}_telemetry.csv", csv_rows),
    }
    return ExportBundle(directory=out_dir, files=files)


def export_benchmark_summary(
    suite_or_results: BenchmarkSuiteResult | RuntimeModeComparison | Sequence[BenchmarkResult],
    directory: str | Path,
    *,
    prefix: str = "benchmarks",
) -> ExportBundle:
    """Export benchmark rows as JSON and CSV."""
    out_dir = Path(directory)
    rows = benchmark_rows(suite_or_results)
    json_payload = [asdict(row) for row in rows]
    csv_payload = [
        {
            k: v
            for k, v in asdict(row).items()
            if k != "metadata"
        }
        for row in rows
    ]
    files = {
        "benchmarks_json": _write_json(out_dir / f"{prefix}_summary.json", json_payload),
        "benchmarks_csv": _write_csv(out_dir / f"{prefix}_summary.csv", csv_payload),
    }
    return ExportBundle(directory=out_dir, files=files)


def export_ablation_summary(result: ObjectiveAblationResult, directory: str | Path, *, prefix: str = "ablation") -> ExportBundle:
    """Export objective ablation rows as JSON and CSV."""
    out_dir = Path(directory)
    rows = ablation_rows(result)
    json_payload = [asdict(row) for row in rows]
    csv_payload = [
        {
            k: v
            for k, v in asdict(row).items()
            if k != "metadata"
        }
        for row in rows
    ]
    files = {
        "ablation_json": _write_json(out_dir / f"{prefix}_summary.json", json_payload),
        "ablation_csv": _write_csv(out_dir / f"{prefix}_summary.csv", csv_payload),
    }
    return ExportBundle(directory=out_dir, files=files)


__all__ = [
    "TelemetryPoint",
    "RuntimeTelemetry",
    "BenchmarkSummaryRow",
    "AblationSummaryRow",
    "ExportBundle",
    "runtime_telemetry",
    "benchmark_rows",
    "ablation_rows",
    "export_runtime_telemetry",
    "export_benchmark_summary",
    "export_ablation_summary",
]
