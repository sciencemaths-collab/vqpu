from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np

from core import EntanglementHypergraph, HilbertBundleState, MeasurementRecord, StateBundleSnapshot

from .metrics import basis_probabilities, bundle_metric_report
from .objective import PolicyDecision
from .optimizer import ControlHyperparameters, ControlMemoryState, OptimizerStepReport, optimizer_step, seed_control_memory
from .runtime import (
    RuntimeActionSummary,
    RuntimeStepReport,
    _apply_runtime_compression,
    _apply_runtime_entanglement,
    _apply_runtime_expansion,
    _apply_runtime_measurement,
    _resolve_graph,
)


class ActionPhase(str, Enum):
    """High-level execution phases for one scheduled runtime step."""

    PREPARE = "prepare"
    EXPAND = "expand"
    ENTANGLE = "entangle"
    MEASURE = "measure"
    COMPRESS = "compress"


class ScheduledActionKind(str, Enum):
    """Kinds of actions the Section 11 scheduler can order."""

    EXPANSION = "expansion"
    ENTANGLER = "entangler"
    MEASUREMENT = "measurement"
    COMPRESSION = "compression"


@dataclass(slots=True)
class ScheduledAction:
    """One queued runtime action with priority and resource estimates."""

    action_id: str
    kind: ScheduledActionKind
    phase: ActionPhase
    priority: float
    estimated_cost: float = 1.0
    payload: Dict[str, Any] = field(default_factory=dict)
    deferred_reason: str | None = None


@dataclass(slots=True)
class ActionQueue:
    """Ordered actions for one execution phase."""

    phase: ActionPhase
    actions: Tuple[ScheduledAction, ...] = ()


@dataclass(slots=True)
class ExecutionSchedule:
    """Execution plan emitted by the Section 11 scheduler."""

    phase_order: Tuple[ActionPhase, ...]
    queues: Dict[ActionPhase, ActionQueue]
    route_ranking: Tuple[str, ...] = ()
    branch_pressure: float = 0.0
    measurement_phase: ActionPhase = ActionPhase.MEASURE
    executed_action_ids: Tuple[str, ...] = ()
    deferred_action_ids: Tuple[str, ...] = ()
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ScheduledRuntimeStepReport:
    """Scheduled wrapper around the core runtime step report."""

    runtime: RuntimeStepReport
    schedule: ExecutionSchedule
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ScheduledRuntimeRunReport:
    """Multi-step scheduled runtime execution report."""

    initial_snapshot: StateBundleSnapshot
    final_snapshot: StateBundleSnapshot
    steps: Tuple[ScheduledRuntimeStepReport, ...]
    final_bundle: HilbertBundleState
    final_graph: EntanglementHypergraph | None
    final_memory: ControlMemoryState
    metadata: Dict[str, Any] = field(default_factory=dict)


def _route_merit(graph: EntanglementHypergraph, route_id: str) -> float:
    route = graph.routes[route_id]
    return float(route.score + 0.25 * route.bandwidth - 0.1 * route.latency)


def _rank_routes(graph: EntanglementHypergraph | None, preferred_routes: Sequence[str]) -> Tuple[str, ...]:
    if graph is None:
        return tuple(str(route_id) for route_id in preferred_routes)

    candidates = set(str(route_id) for route_id in preferred_routes if str(route_id) in graph.routes)
    if not candidates:
        candidates = set(graph.routes.keys())
    ranked = sorted(candidates, key=lambda route_id: (-_route_merit(graph, route_id), route_id))
    return tuple(ranked)


def _branch_pressure(bundle: HilbertBundleState) -> float:
    probs = basis_probabilities(bundle)
    tol = float(bundle.config.numerical_tolerance)
    active = int(np.count_nonzero(probs > tol))
    capacity = max(1, int(bundle.config.budget.max_branches))
    return float(active / capacity)


def _should_measure_early(decision: PolicyDecision) -> bool:
    if not decision.sector_signals:
        return False
    mean_decoherence = float(np.mean([sig.decoherence_risk for sig in decision.sector_signals]))
    mean_mismatch = float(np.mean([sig.target_mismatch for sig in decision.sector_signals]))
    return bool(
        mean_decoherence >= 0.45
        or decision.objective.noise_penalty >= 0.45
        or decision.measurement_strength >= 0.45
        or mean_mismatch >= 0.75
    )


def _compression_should_precede_entanglement(bundle: HilbertBundleState, decision: PolicyDecision) -> bool:
    return bool(
        _branch_pressure(bundle) > 0.85
        or decision.objective.resource_penalty >= 0.5
        or decision.objective.noise_penalty >= 0.7
    )


def _sort_actions(actions: Iterable[ScheduledAction]) -> Tuple[ScheduledAction, ...]:
    return tuple(sorted(actions, key=lambda item: (-float(item.priority), item.action_id)))


def build_execution_schedule(
    bundle: HilbertBundleState,
    decision: PolicyDecision,
    optimizer_report: OptimizerStepReport,
    *,
    graph: EntanglementHypergraph | None = None,
    max_entanglers: int = 2,
) -> ExecutionSchedule:
    """Create a coherence-aware action schedule from policy and optimizer state."""
    if max_entanglers < 0:
        raise ValueError("max_entanglers must be non-negative")

    phase_order = (
        ActionPhase.PREPARE,
        ActionPhase.EXPAND,
        ActionPhase.ENTANGLE,
        ActionPhase.MEASURE,
        ActionPhase.COMPRESS,
    )
    queues: Dict[ActionPhase, List[ScheduledAction]] = {phase: [] for phase in phase_order}
    deferred: List[str] = []
    executed: List[str] = []

    branch_pressure = _branch_pressure(bundle)
    measurement_phase = ActionPhase.PREPARE if _should_measure_early(decision) else ActionPhase.MEASURE
    route_ranking = _rank_routes(graph, optimizer_report.selected_routes)

    remaining_sector_slots = max(0, int(bundle.config.budget.max_active_qubits) - int(bundle.active_sector_count))
    selected_expansions = list(optimizer_report.selected_expansions)
    allowed_expansions = selected_expansions[:remaining_sector_slots]
    for name in selected_expansions[remaining_sector_slots:]:
        deferred.append(f"expand:{name}")

    if allowed_expansions:
        exp_priority = float(0.5 + 0.35 * (1.0 - decision.objective.fidelity_score) + 0.15 * max(0.0, 1.0 - branch_pressure))
        queues[ActionPhase.EXPAND].append(
            ScheduledAction(
                action_id="expand",
                kind=ScheduledActionKind.EXPANSION,
                phase=ActionPhase.EXPAND,
                priority=exp_priority,
                estimated_cost=float(len(allowed_expansions)),
                payload={"names": tuple(allowed_expansions)},
            )
        )

    selected_routes = list(route_ranking[: max(0, int(max_entanglers))])
    for route_id in route_ranking[max(0, int(max_entanglers)) :]:
        deferred.append(f"route:{route_id}")
    if selected_routes and max_entanglers > 0:
        ent_priority = float(0.55 + 0.25 * decision.objective.entanglement_score + 0.2 * (1.0 - decision.objective.noise_penalty))
        queues[ActionPhase.ENTANGLE].append(
            ScheduledAction(
                action_id="entangle",
                kind=ScheduledActionKind.ENTANGLER,
                phase=ActionPhase.ENTANGLE,
                priority=ent_priority,
                estimated_cost=float(len(selected_routes)),
                payload={"routes": tuple(selected_routes), "max_entanglers": len(selected_routes)},
            )
        )

    measurements_remaining = max(0, int(bundle.config.budget.max_measurements) - int(bundle.stats.measurements_used))
    if decision.sector_signals and measurements_remaining > 0 and decision.measurement_strength > bundle.config.numerical_tolerance:
        meas_priority = float(0.45 + 0.3 * decision.objective.noise_penalty + 0.25 * decision.measurement_strength)
        queues[measurement_phase].append(
            ScheduledAction(
                action_id="measure",
                kind=ScheduledActionKind.MEASUREMENT,
                phase=measurement_phase,
                priority=meas_priority,
                estimated_cost=1.0,
                payload={"count": 1},
            )
        )
    elif decision.sector_signals:
        deferred.append("measure")

    do_precompress = _compression_should_precede_entanglement(bundle, decision)
    prune_phase = ActionPhase.PREPARE if do_precompress else ActionPhase.COMPRESS
    prune_priority = float(0.4 + 0.4 * branch_pressure + 0.2 * decision.objective.resource_penalty)
    if optimizer_report.selected_prunes or branch_pressure > 0.8 or decision.objective.resource_penalty > 0.4:
        queues[prune_phase].append(
            ScheduledAction(
                action_id="compress",
                kind=ScheduledActionKind.COMPRESSION,
                phase=prune_phase,
                priority=prune_priority,
                estimated_cost=1.0,
                payload={"selected_prunes": tuple(int(idx) for idx in optimizer_report.selected_prunes)},
            )
        )

    finalized: Dict[ActionPhase, ActionQueue] = {}
    for phase in phase_order:
        ordered = _sort_actions(queues[phase])
        finalized[phase] = ActionQueue(phase=phase, actions=ordered)
        executed.extend(action.action_id for action in ordered)

    return ExecutionSchedule(
        phase_order=phase_order,
        queues=finalized,
        route_ranking=tuple(route_ranking),
        branch_pressure=float(branch_pressure),
        measurement_phase=measurement_phase,
        executed_action_ids=tuple(executed),
        deferred_action_ids=tuple(deferred),
        metadata={
            "remaining_sector_slots": remaining_sector_slots,
            "max_entanglers": int(max_entanglers),
            "measurements_remaining": measurements_remaining,
            "precompress": bool(do_precompress),
        },
    )


def execute_schedule(
    bundle: HilbertBundleState,
    schedule: ExecutionSchedule,
    decision: PolicyDecision,
    optimizer_report: OptimizerStepReport,
    *,
    graph: EntanglementHypergraph | None = None,
    step_index: int,
) -> Tuple[HilbertBundleState, EntanglementHypergraph | None, RuntimeActionSummary, Tuple[MeasurementRecord, ...]]:
    """Execute one Section 11 schedule on a working bundle and graph."""
    work_bundle = bundle
    work_graph = graph
    applied_expansions: Tuple[str, ...] = ()
    applied_entanglers: Tuple[str, ...] = ()
    applied_prunes: Tuple[int, ...] = ()
    measurement_records: Tuple[MeasurementRecord, ...] = ()
    route_summary: Tuple[str, ...] = ()

    for phase in schedule.phase_order:
        for action in schedule.queues[phase].actions:
            if action.kind == ScheduledActionKind.EXPANSION:
                custom_report = OptimizerStepReport(
                    decision=optimizer_report.decision,
                    memory=optimizer_report.memory,
                    threshold_update=optimizer_report.threshold_update,
                    hyperedge_updates=optimizer_report.hyperedge_updates,
                    route_updates=optimizer_report.route_updates,
                    selected_expansions=tuple(action.payload.get("names", ())),
                    selected_prunes=optimizer_report.selected_prunes,
                    selected_routes=optimizer_report.selected_routes,
                    updated_graph=optimizer_report.updated_graph,
                    metadata=dict(optimizer_report.metadata),
                )
                applied_expansions, work_graph = _apply_runtime_expansion(work_bundle, work_graph, decision, custom_report)
            elif action.kind == ScheduledActionKind.ENTANGLER:
                selected_routes = tuple(str(route_id) for route_id in action.payload.get("routes", ()))
                route_summary = selected_routes
                custom_report = OptimizerStepReport(
                    decision=optimizer_report.decision,
                    memory=optimizer_report.memory,
                    threshold_update=optimizer_report.threshold_update,
                    hyperedge_updates=optimizer_report.hyperedge_updates,
                    route_updates=optimizer_report.route_updates,
                    selected_expansions=optimizer_report.selected_expansions,
                    selected_prunes=optimizer_report.selected_prunes,
                    selected_routes=selected_routes,
                    updated_graph=optimizer_report.updated_graph,
                    metadata=dict(optimizer_report.metadata),
                )
                applied_entanglers, work_graph = _apply_runtime_entanglement(
                    work_bundle,
                    work_graph,
                    decision,
                    custom_report,
                    max_entanglers=int(action.payload.get("max_entanglers", 0)),
                )
            elif action.kind == ScheduledActionKind.MEASUREMENT:
                measurement_records = _apply_runtime_measurement(
                    work_bundle,
                    decision,
                    optimizer_report,
                    step_index=int(step_index),
                )
            elif action.kind == ScheduledActionKind.COMPRESSION:
                applied_prunes = _apply_runtime_compression(work_bundle, optimizer_report)

    action_summary = RuntimeActionSummary(
        expansions=tuple(applied_expansions),
        prunes=tuple(applied_prunes),
        entanglers=tuple(applied_entanglers),
        routes=tuple(route_summary),
        measurements=tuple(record.label for record in measurement_records),
    )
    return work_bundle, work_graph, action_summary, measurement_records


def scheduled_runtime_step(
    bundle: HilbertBundleState,
    *,
    graph: EntanglementHypergraph | None = None,
    target: HilbertBundleState | np.ndarray | Sequence[complex] | None = None,
    memory: ControlMemoryState | None = None,
    hyperparameters: ControlHyperparameters | None = None,
    step_index: int | None = None,
    max_entanglers: int = 2,
    in_place: bool = False,
) -> ScheduledRuntimeStepReport:
    """Run one Section 11 scheduled runtime step."""
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

    schedule = build_execution_schedule(
        work_bundle,
        opt_report.decision,
        opt_report,
        graph=work_graph,
        max_entanglers=max_entanglers,
    )
    effective_step_index = int(work_bundle.stats.step if step_index is None else step_index)
    work_bundle, work_graph, action_summary, measurement_records = execute_schedule(
        work_bundle,
        schedule,
        opt_report.decision,
        opt_report,
        graph=work_graph,
        step_index=effective_step_index,
    )

    if work_graph is not None:
        work_graph.sync_from_topology(work_bundle.topology, drop_missing=True)

    after = work_bundle.snapshot()
    metrics_target = target
    if isinstance(target, HilbertBundleState) and target.dims != work_bundle.dims:
        metrics_target = None
    metrics = bundle_metric_report(work_bundle, graph=work_graph, target=metrics_target)
    graph_summary = None if work_graph is None else work_graph.summary()

    work_bundle.metadata["last_runtime_step"] = effective_step_index
    work_bundle.metadata["last_schedule"] = {
        "phase_order": tuple(phase.value for phase in schedule.phase_order),
        "measurement_phase": schedule.measurement_phase.value,
        "branch_pressure": schedule.branch_pressure,
    }

    runtime_report = RuntimeStepReport(
        step_index=effective_step_index,
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
            "scheduled": True,
            "max_entanglers": int(max_entanglers),
            "target_present": target is not None,
            "graph_present": work_graph is not None,
        },
    )
    return ScheduledRuntimeStepReport(
        runtime=runtime_report,
        schedule=schedule,
        metadata={
            "target_present": target is not None,
            "graph_present": work_graph is not None,
        },
    )


def run_scheduled_runtime(
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
) -> ScheduledRuntimeRunReport:
    """Run the Section 11 scheduler across multiple CHESSO steps."""
    work_bundle = bundle if in_place else bundle.copy()
    work_graph = _resolve_graph(work_bundle, graph, in_place=in_place)
    use_memory = memory or seed_control_memory(work_bundle)

    initial_snapshot = work_bundle.snapshot()
    step_reports: List[ScheduledRuntimeStepReport] = []
    limit = int(work_bundle.config.budget.max_steps if steps is None else steps)
    if limit <= 0:
        raise ValueError("steps must be positive")

    for idx in range(1, limit + 1):
        step_target = target
        if isinstance(target, HilbertBundleState) and target.dims != work_bundle.dims:
            step_target = None
        report = scheduled_runtime_step(
            work_bundle,
            graph=work_graph,
            target=step_target,
            memory=use_memory,
            hyperparameters=hyperparameters,
            step_index=idx,
            max_entanglers=max_entanglers,
            in_place=True,
        )
        work_bundle = report.runtime.bundle
        work_graph = report.runtime.graph
        use_memory = report.runtime.memory
        step_reports.append(report)

        if stop_fidelity is not None and report.runtime.decision.objective.fidelity_score >= float(stop_fidelity):
            break

    final_snapshot = work_bundle.snapshot()
    return ScheduledRuntimeRunReport(
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
    "ActionPhase",
    "ScheduledActionKind",
    "ScheduledAction",
    "ActionQueue",
    "ExecutionSchedule",
    "ScheduledRuntimeStepReport",
    "ScheduledRuntimeRunReport",
    "build_execution_schedule",
    "execute_schedule",
    "scheduled_runtime_step",
    "run_scheduled_runtime",
]
