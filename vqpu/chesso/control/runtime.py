from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np

from vqpu.chesso.core import EntanglementHypergraph, HilbertBundleState, HypergraphSummary, MeasurementRecord, StateBundleSnapshot
from vqpu.chesso.ops import (
    ExpansionCandidate,
    adaptive_expand_bundle,
    apply_default_noise,
    apply_hyperedge_entangler,
    compress_to_top_k_branches,
    flush_lazy_measurements,
    prune_low_population_branches,
    queue_lazy_basis_measurement,
    single_qubit_readout_confusion,
)

from .metrics import BundleMetricReport, basis_probabilities, bundle_metric_report
from .objective import PolicyDecision
from .optimizer import ControlHyperparameters, ControlMemoryState, OptimizerStepReport, optimizer_step, seed_control_memory


@dataclass(slots=True)
class RuntimeActionSummary:
    """Concrete actions applied during one runtime step."""

    expansions: Tuple[str, ...] = ()
    prunes: Tuple[int, ...] = ()
    entanglers: Tuple[str, ...] = ()
    routes: Tuple[str, ...] = ()
    measurements: Tuple[str, ...] = ()


@dataclass(slots=True)
class RuntimeStepReport:
    """Full report for one Section 10 runtime step."""

    step_index: int
    before_snapshot: StateBundleSnapshot
    after_snapshot: StateBundleSnapshot
    decision: PolicyDecision
    optimizer_report: OptimizerStepReport
    memory: ControlMemoryState
    bundle: HilbertBundleState
    graph: EntanglementHypergraph | None
    action_summary: RuntimeActionSummary
    measurement_records: Tuple[MeasurementRecord, ...]
    metrics: BundleMetricReport
    graph_summary: HypergraphSummary | None = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RuntimeRunReport:
    """Packaged result of a multi-step CHESSO runtime execution."""

    initial_snapshot: StateBundleSnapshot
    final_snapshot: StateBundleSnapshot
    steps: Tuple[RuntimeStepReport, ...]
    final_bundle: HilbertBundleState
    final_graph: EntanglementHypergraph | None
    final_memory: ControlMemoryState
    metadata: Dict[str, Any] = field(default_factory=dict)


def _resolve_graph(
    bundle: HilbertBundleState,
    graph: EntanglementHypergraph | None,
    *,
    in_place: bool,
) -> EntanglementHypergraph | None:
    if graph is None:
        return None
    work = graph if in_place else graph.copy()
    work.sync_from_topology(bundle.topology, drop_missing=True)
    return work


def _selected_expansion_candidates(
    decision: PolicyDecision,
    selected_names: Iterable[str],
) -> Tuple[ExpansionCandidate, ...]:
    selected = set(str(name) for name in selected_names)
    if not selected:
        return ()
    chosen = [candidate for candidate in decision.expansion_candidates if candidate.name in selected]
    chosen.sort(key=lambda item: (-float(item.score), item.name))
    return tuple(chosen)


def _rank_hyperedges_for_runtime(
    decision: PolicyDecision,
    graph: EntanglementHypergraph | None,
    selected_routes: Sequence[str],
    *,
    max_edges: int,
) -> Tuple[str, ...]:
    if graph is None or not graph.hyperedges or max_edges <= 0:
        return ()

    ranked: List[str] = []
    seen: set[str] = set()

    for route_id in selected_routes:
        route = graph.routes.get(route_id)
        if route is None:
            continue
        for edge_id in route.edge_path:
            key = str(edge_id)
            if key in graph.hyperedges and key not in seen:
                ranked.append(key)
                seen.add(key)
                if len(ranked) >= max_edges:
                    return tuple(ranked)

    for signal in decision.hyperedge_signals:
        if signal.edge_id in graph.hyperedges and signal.edge_id not in seen:
            ranked.append(signal.edge_id)
            seen.add(signal.edge_id)
            if len(ranked) >= max_edges:
                break
    return tuple(ranked)


def _choose_measurement_targets(decision: PolicyDecision) -> Tuple[int, ...]:
    if not decision.sector_signals:
        return ()
    ranked = sorted(
        decision.sector_signals,
        key=lambda sig: (
            -(0.45 * float(sig.target_mismatch) + 0.35 * float(sig.utility) + 0.2 * float(sig.decoherence_risk)),
            sig.index,
        ),
    )
    return (int(ranked[0].index),)


def _apply_runtime_expansion(
    bundle: HilbertBundleState,
    graph: EntanglementHypergraph | None,
    decision: PolicyDecision,
    optimizer_report: OptimizerStepReport,
) -> Tuple[Tuple[str, ...], EntanglementHypergraph | None]:
    chosen = _selected_expansion_candidates(decision, optimizer_report.selected_expansions)
    if not chosen:
        return (), graph
    adaptive_expand_bundle(bundle, chosen, max_new=len(chosen), min_score=-np.inf, in_place=True)
    if graph is not None:
        graph.sync_from_topology(bundle.topology)
    return tuple(candidate.name for candidate in chosen), graph


def _apply_runtime_entanglement(
    bundle: HilbertBundleState,
    graph: EntanglementHypergraph | None,
    decision: PolicyDecision,
    optimizer_report: OptimizerStepReport,
    *,
    max_entanglers: int,
) -> Tuple[Tuple[str, ...], EntanglementHypergraph | None]:
    if graph is None:
        return (), None
    edge_signal_map = {signal.edge_id: signal for signal in decision.hyperedge_signals}
    chosen_edges = _rank_hyperedges_for_runtime(
        decision,
        graph,
        optimizer_report.selected_routes,
        max_edges=max_entanglers,
    )
    applied: List[str] = []
    for edge_id in chosen_edges:
        signal = edge_signal_map.get(edge_id)
        if signal is None:
            continue
        strength = float(np.clip(0.25 + 0.75 * max(0.0, signal.utility), 0.05, 1.0))
        apply_hyperedge_entangler(bundle, graph, edge_id, strength=strength, in_place=True)
        members = graph.hyperedges[edge_id].members
        apply_default_noise(bundle, members, in_place=True, label_prefix=f"runtime:{edge_id}")
        applied.append(edge_id)
    return tuple(applied), graph


def _apply_runtime_measurement(
    bundle: HilbertBundleState,
    decision: PolicyDecision,
    optimizer_report: OptimizerStepReport,
    *,
    step_index: int,
) -> Tuple[MeasurementRecord, ...]:
    targets = _choose_measurement_targets(decision)
    if not targets:
        return ()

    target_index = int(targets[0])
    dim = int(bundle.dims[target_index])
    readout_confusion = None
    if dim == 2 and bundle.config.noise.readout_error > 0.0:
        readout_confusion = single_qubit_readout_confusion(bundle.config.noise.readout_error)

    queue_lazy_basis_measurement(
        bundle,
        [target_index],
        strength=float(optimizer_report.memory.measurement_strength),
        selective=True,
        sample=False,
        label=f"runtime:peek:{step_index}",
        store_key=f"peek.step_{step_index}",
        readout_confusion=readout_confusion,
        metadata={
            "runtime_step": int(step_index),
            "selected_sector_index": target_index,
            "selected_sector_id": decision.sector_signals[0].sector_id if decision.sector_signals else None,
        },
    )
    records = flush_lazy_measurements(bundle, in_place=True)
    return tuple(records)


def _apply_runtime_compression(
    bundle: HilbertBundleState,
    optimizer_report: OptimizerStepReport,
) -> Tuple[int, ...]:
    applied = tuple(int(idx) for idx in optimizer_report.selected_prunes)
    if applied:
        prune_low_population_branches(
            bundle,
            threshold=float(optimizer_report.memory.prune_threshold),
            in_place=True,
        )
    probs = basis_probabilities(bundle)
    active_branches = int(np.count_nonzero(probs > bundle.config.numerical_tolerance))
    if active_branches > bundle.config.budget.max_branches:
        compress_to_top_k_branches(bundle, k=int(bundle.config.budget.max_branches), in_place=True)
    return applied


def chesso_runtime_step(
    bundle: HilbertBundleState,
    *,
    graph: EntanglementHypergraph | None = None,
    target: HilbertBundleState | np.ndarray | Sequence[complex] | None = None,
    memory: ControlMemoryState | None = None,
    hyperparameters: ControlHyperparameters | None = None,
    step_index: int | None = None,
    max_entanglers: int = 2,
    in_place: bool = False,
) -> RuntimeStepReport:
    """Run one integrated Section 10 CHESSO step."""
    work_bundle = bundle if in_place else bundle.copy()
    work_graph = _resolve_graph(work_bundle, graph, in_place=in_place)
    before = work_bundle.snapshot()

    base_memory = memory or seed_control_memory(work_bundle)
    opt_report = optimizer_step(
        work_bundle,
        graph=work_graph,
        target=target,
        memory=base_memory,
        hyperparameters=hyperparameters,
        in_place_graph=True,
    )
    if opt_report.updated_graph is not None:
        work_graph = opt_report.updated_graph

    applied_expansions, work_graph = _apply_runtime_expansion(
        work_bundle,
        work_graph,
        opt_report.decision,
        opt_report,
    )
    applied_entanglers, work_graph = _apply_runtime_entanglement(
        work_bundle,
        work_graph,
        opt_report.decision,
        opt_report,
        max_entanglers=max_entanglers,
    )
    measurement_records = _apply_runtime_measurement(
        work_bundle,
        opt_report.decision,
        opt_report,
        step_index=int(work_bundle.stats.step if step_index is None else step_index),
    )
    applied_prunes = _apply_runtime_compression(work_bundle, opt_report)

    if work_graph is not None:
        work_graph.sync_from_topology(work_bundle.topology, drop_missing=True)

    after = work_bundle.snapshot()
    metrics_target = target
    if isinstance(target, HilbertBundleState) and target.dims != work_bundle.dims:
        metrics_target = None
    metrics = bundle_metric_report(work_bundle, graph=work_graph, target=metrics_target)
    graph_summary = None if work_graph is None else work_graph.summary()
    action_summary = RuntimeActionSummary(
        expansions=tuple(applied_expansions),
        prunes=tuple(applied_prunes),
        entanglers=tuple(applied_entanglers),
        routes=tuple(opt_report.selected_routes),
        measurements=tuple(record.label for record in measurement_records),
    )

    work_bundle.metadata["last_runtime_step"] = int(work_bundle.stats.step if step_index is None else step_index)
    work_bundle.metadata["last_runtime_actions"] = {
        "expansions": action_summary.expansions,
        "prunes": action_summary.prunes,
        "entanglers": action_summary.entanglers,
        "routes": action_summary.routes,
        "measurements": action_summary.measurements,
    }

    return RuntimeStepReport(
        step_index=int(work_bundle.stats.step if step_index is None else step_index),
        before_snapshot=before,
        after_snapshot=after,
        decision=opt_report.decision,
        optimizer_report=opt_report,
        memory=opt_report.memory,
        bundle=work_bundle,
        graph=work_graph,
        action_summary=action_summary,
        measurement_records=tuple(measurement_records),
        metrics=metrics,
        graph_summary=graph_summary,
        metadata={
            "max_entanglers": int(max_entanglers),
            "target_present": target is not None,
            "graph_present": work_graph is not None,
        },
    )


def run_chesso_runtime(
    bundle: HilbertBundleState,
    *,
    graph: EntanglementHypergraph | None = None,
    target: HilbertBundleState | np.ndarray | Sequence[complex] | None = None,
    memory: ControlMemoryState | None = None,
    hyperparameters: ControlHyperparameters | None = None,
    steps: int | None = None,
    stop_fidelity: float | None = 0.999,
    max_entanglers: int = 2,
    in_place: bool = False,
) -> RuntimeRunReport:
    """Run the integrated CHESSO control loop for multiple steps."""
    work_bundle = bundle if in_place else bundle.copy()
    work_graph = _resolve_graph(work_bundle, graph, in_place=in_place)
    use_memory = memory or seed_control_memory(work_bundle)

    initial_snapshot = work_bundle.snapshot()
    step_reports: List[RuntimeStepReport] = []
    limit = int(work_bundle.config.budget.max_steps if steps is None else steps)
    if limit <= 0:
        raise ValueError("steps must be positive")

    for idx in range(1, limit + 1):
        step_target = target
        if isinstance(target, HilbertBundleState) and target.dims != work_bundle.dims:
            step_target = None
        report = chesso_runtime_step(
            work_bundle,
            graph=work_graph,
            target=step_target,
            memory=use_memory,
            hyperparameters=hyperparameters,
            step_index=idx,
            max_entanglers=max_entanglers,
            in_place=True,
        )
        work_bundle = report.bundle
        work_graph = report.graph
        use_memory = report.memory
        step_reports.append(report)

        if stop_fidelity is not None and report.decision.objective.fidelity_score >= float(stop_fidelity):
            break

    final_snapshot = work_bundle.snapshot()
    return RuntimeRunReport(
        initial_snapshot=initial_snapshot,
        final_snapshot=final_snapshot,
        steps=tuple(step_reports),
        final_bundle=work_bundle,
        final_graph=work_graph,
        final_memory=use_memory,
        metadata={
            "requested_steps": limit,
            "executed_steps": len(step_reports),
            "stopped_early": len(step_reports) < limit,
            "stop_fidelity": stop_fidelity,
        },
    )


__all__ = [
    "RuntimeActionSummary",
    "RuntimeStepReport",
    "RuntimeRunReport",
    "chesso_runtime_step",
    "run_chesso_runtime",
]
