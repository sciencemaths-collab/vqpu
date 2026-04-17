from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np

from vqpu.chesso.core import EntanglementHypergraph, HilbertBundleState

from .objective import (
    HyperedgePolicySignal,
    PolicyDecision,
    chesso_policy_decision,
)


@dataclass(slots=True)
class ControlHyperparameters:
    """Tunable gains for Section 9 adaptive control updates."""

    objective_ema_decay: float = 0.8
    fidelity_ema_decay: float = 0.8
    entanglement_ema_decay: float = 0.8
    noise_ema_decay: float = 0.8
    sector_utility_ema_decay: float = 0.75
    edge_utility_ema_decay: float = 0.75

    edge_weight_lr: float = 0.35
    phase_bias_lr: float = 0.3
    route_score_lr: float = 0.25
    expansion_threshold_lr: float = 0.2
    prune_threshold_lr: float = 0.15
    measurement_strength_lr: float = 0.25

    min_expansion_threshold: float = 0.05
    max_expansion_threshold: float = 0.95
    min_prune_threshold: float = 1e-6
    max_prune_threshold: float = 0.2
    min_measurement_strength: float = 0.01
    max_measurement_strength: float = 0.95

    max_edge_weight: float = 4.0
    max_route_score: float = 4.0

    def validate(self) -> None:
        for name in (
            "objective_ema_decay",
            "fidelity_ema_decay",
            "entanglement_ema_decay",
            "noise_ema_decay",
            "sector_utility_ema_decay",
            "edge_utility_ema_decay",
        ):
            value = float(getattr(self, name))
            if not 0.0 <= value < 1.0:
                raise ValueError(f"{name} must be in [0, 1), got {value!r}")
        for name in (
            "edge_weight_lr",
            "phase_bias_lr",
            "route_score_lr",
            "expansion_threshold_lr",
            "prune_threshold_lr",
            "measurement_strength_lr",
            "min_expansion_threshold",
            "max_expansion_threshold",
            "min_prune_threshold",
            "max_prune_threshold",
            "min_measurement_strength",
            "max_measurement_strength",
            "max_edge_weight",
            "max_route_score",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative, got {value!r}")
        if self.min_expansion_threshold > self.max_expansion_threshold:
            raise ValueError("Expansion threshold bounds are inverted")
        if self.min_prune_threshold > self.max_prune_threshold:
            raise ValueError("Prune threshold bounds are inverted")
        if self.min_measurement_strength > self.max_measurement_strength:
            raise ValueError("Measurement strength bounds are inverted")


@dataclass(slots=True)
class ControlMemoryState:
    """Persistent adaptive memory carried from one optimizer step to the next."""

    step_count: int = 0
    objective_ema: float = 0.0
    fidelity_ema: float = 0.0
    entanglement_ema: float = 0.0
    noise_ema: float = 0.0
    mean_sector_utility_ema: float = 0.0
    mean_edge_utility_ema: float = 0.0
    expansion_threshold: float = 0.2
    prune_threshold: float = 1e-3
    measurement_strength: float = 0.1
    preferred_route_history: Tuple[str, ...] = ()
    last_selected_expansions: Tuple[str, ...] = ()
    last_selected_prunes: Tuple[int, ...] = ()
    last_notes: Tuple[str, ...] = ()
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ThresholdUpdate:
    """How the optimizer moved runtime thresholds in one step."""

    old_expansion_threshold: float
    new_expansion_threshold: float
    old_prune_threshold: float
    new_prune_threshold: float
    old_measurement_strength: float
    new_measurement_strength: float
    expansion_pressure: float
    prune_pressure: float
    measurement_pressure: float


@dataclass(slots=True)
class HyperedgeUpdate:
    """Weight and phase adaptation applied to one hyperedge."""

    edge_id: str
    old_weight: float
    new_weight: float
    old_phase_bias: float
    new_phase_bias: float
    utility_drive: float
    reward_drive: float
    route_drive: float


@dataclass(slots=True)
class RouteUpdate:
    """Route-score adaptation for one entanglement route."""

    route_id: str
    old_score: float
    new_score: float
    merit: float


@dataclass(slots=True)
class OptimizerStepReport:
    """Packaged result of one Section 9 optimizer/control step."""

    decision: PolicyDecision
    memory: ControlMemoryState
    threshold_update: ThresholdUpdate
    hyperedge_updates: Tuple[HyperedgeUpdate, ...]
    route_updates: Tuple[RouteUpdate, ...]
    selected_expansions: Tuple[str, ...]
    selected_prunes: Tuple[int, ...]
    selected_routes: Tuple[str, ...]
    updated_graph: EntanglementHypergraph | None
    metadata: Dict[str, Any] = field(default_factory=dict)


def _clip(value: float, lo: float, hi: float) -> float:
    return float(np.clip(float(value), float(lo), float(hi)))


def _ema(previous: float, current: float, decay: float, *, seeded: bool) -> float:
    if not seeded:
        return float(current)
    return float(decay * previous + (1.0 - decay) * current)


def _wrap_phase(angle: float) -> float:
    return float((float(angle) + np.pi) % (2.0 * np.pi) - np.pi)


def _mean_or_zero(values: Iterable[float]) -> float:
    vals = [float(v) for v in values]
    if not vals:
        return 0.0
    return float(np.mean(vals))


def seed_control_memory(
    bundle: HilbertBundleState,
    *,
    decision: PolicyDecision | None = None,
    expansion_threshold: float | None = None,
    prune_threshold: float | None = None,
    measurement_strength: float | None = None,
) -> ControlMemoryState:
    """Create initial control memory aligned with the current runtime state."""
    tol = float(bundle.config.numerical_tolerance)
    default_prune = max(8.0 * tol, min(0.02, 0.5 * float(bundle.config.budget.max_prune_loss)))
    memory = ControlMemoryState(
        expansion_threshold=float(0.2 if expansion_threshold is None else expansion_threshold),
        prune_threshold=float(default_prune if prune_threshold is None else prune_threshold),
        measurement_strength=float(
            bundle.config.default_measurement_strength if measurement_strength is None else measurement_strength
        ),
    )
    if decision is not None:
        return update_control_memory(memory, decision)
    return memory


def update_control_memory(
    memory: ControlMemoryState,
    decision: PolicyDecision,
    *,
    hyperparameters: ControlHyperparameters | None = None,
) -> ControlMemoryState:
    """Update exponential-memory statistics from the latest policy decision."""
    params = hyperparameters or ControlHyperparameters()
    params.validate()

    seeded = memory.step_count > 0
    mean_sector_utility = _mean_or_zero(sig.utility for sig in decision.sector_signals)
    mean_edge_utility = _mean_or_zero(sig.utility for sig in decision.hyperedge_signals)

    return ControlMemoryState(
        step_count=memory.step_count + 1,
        objective_ema=_ema(memory.objective_ema, decision.objective.total_score, params.objective_ema_decay, seeded=seeded),
        fidelity_ema=_ema(memory.fidelity_ema, decision.objective.fidelity_score, params.fidelity_ema_decay, seeded=seeded),
        entanglement_ema=_ema(
            memory.entanglement_ema,
            decision.objective.entanglement_score,
            params.entanglement_ema_decay,
            seeded=seeded,
        ),
        noise_ema=_ema(memory.noise_ema, decision.objective.noise_penalty, params.noise_ema_decay, seeded=seeded),
        mean_sector_utility_ema=_ema(
            memory.mean_sector_utility_ema,
            mean_sector_utility,
            params.sector_utility_ema_decay,
            seeded=seeded,
        ),
        mean_edge_utility_ema=_ema(
            memory.mean_edge_utility_ema,
            mean_edge_utility,
            params.edge_utility_ema_decay,
            seeded=seeded,
        ),
        expansion_threshold=float(memory.expansion_threshold),
        prune_threshold=float(memory.prune_threshold),
        measurement_strength=float(memory.measurement_strength),
        preferred_route_history=tuple(decision.preferred_routes),
        last_selected_expansions=tuple(memory.last_selected_expansions),
        last_selected_prunes=tuple(memory.last_selected_prunes),
        last_notes=tuple(decision.notes),
        metadata={
            **memory.metadata,
            "last_total_score": float(decision.objective.total_score),
            "last_fidelity_score": float(decision.objective.fidelity_score),
            "last_entanglement_score": float(decision.objective.entanglement_score),
            "last_noise_penalty": float(decision.objective.noise_penalty),
        },
    )


def adapt_thresholds(
    memory: ControlMemoryState,
    decision: PolicyDecision,
    *,
    hyperparameters: ControlHyperparameters | None = None,
) -> ThresholdUpdate:
    """Adapt expansion, pruning, and measurement thresholds from policy signals."""
    params = hyperparameters or ControlHyperparameters()
    params.validate()

    mean_sector_utility = _mean_or_zero(sig.utility for sig in decision.sector_signals)
    mean_decoherence = _mean_or_zero(sig.decoherence_risk for sig in decision.sector_signals)
    mean_coherence = _mean_or_zero(sig.local_coherence_norm for sig in decision.sector_signals)
    mean_mismatch = _mean_or_zero(sig.target_mismatch for sig in decision.sector_signals)
    target_gap = 1.0 - float(decision.objective.fidelity_score)

    expansion_pressure = _clip(
        0.45 * max(0.0, mean_sector_utility)
        + 0.35 * target_gap
        + 0.15 * mean_mismatch
        + 0.05 * (1.0 - decision.objective.resource_penalty),
        0.0,
        1.0,
    )
    prune_pressure = _clip(
        0.45 * decision.objective.resource_penalty
        + 0.3 * decision.objective.noise_penalty
        + 0.15 * mean_decoherence
        + 0.1 * (1.0 - mean_coherence),
        0.0,
        1.0,
    )
    measurement_pressure = _clip(
        0.5 * float(decision.measurement_strength)
        + 0.25 * mean_decoherence
        + 0.15 * target_gap
        + 0.1 * decision.objective.noise_penalty,
        0.0,
        1.0,
    )

    new_expansion_threshold = _clip(
        memory.expansion_threshold + params.expansion_threshold_lr * (0.5 - expansion_pressure),
        params.min_expansion_threshold,
        params.max_expansion_threshold,
    )
    new_prune_threshold = _clip(
        memory.prune_threshold + params.prune_threshold_lr * (prune_pressure - 0.35),
        params.min_prune_threshold,
        params.max_prune_threshold,
    )
    new_measurement_strength = _clip(
        memory.measurement_strength + params.measurement_strength_lr * (measurement_pressure - memory.measurement_strength),
        params.min_measurement_strength,
        params.max_measurement_strength,
    )

    return ThresholdUpdate(
        old_expansion_threshold=float(memory.expansion_threshold),
        new_expansion_threshold=float(new_expansion_threshold),
        old_prune_threshold=float(memory.prune_threshold),
        new_prune_threshold=float(new_prune_threshold),
        old_measurement_strength=float(memory.measurement_strength),
        new_measurement_strength=float(new_measurement_strength),
        expansion_pressure=float(expansion_pressure),
        prune_pressure=float(prune_pressure),
        measurement_pressure=float(measurement_pressure),
    )


def adapt_hypergraph(
    graph: EntanglementHypergraph,
    decision: PolicyDecision,
    *,
    memory: ControlMemoryState | None = None,
    hyperparameters: ControlHyperparameters | None = None,
    in_place: bool = False,
) -> Tuple[EntanglementHypergraph, Tuple[HyperedgeUpdate, ...], Tuple[RouteUpdate, ...]]:
    """Update hyperedge weights, phase biases, and route scores."""
    params = hyperparameters or ControlHyperparameters()
    params.validate()

    work = graph if in_place else graph.copy()
    mean_edge_utility = 0.0 if memory is None else float(memory.mean_edge_utility_ema)
    hyperedge_updates: List[HyperedgeUpdate] = []

    edge_signal_map = {signal.edge_id: signal for signal in decision.hyperedge_signals}
    for edge_id, signal in edge_signal_map.items():
        if edge_id not in work.hyperedges:
            continue
        edge = work.hyperedges[edge_id]
        old_weight = float(edge.weight)
        old_phase = float(edge.phase_bias)
        utility_drive = float(signal.utility - mean_edge_utility)
        reward_drive = float(signal.entanglement_reward)
        route_drive = float(signal.route_support)

        weight_delta = params.edge_weight_lr * (0.55 * utility_drive + 0.3 * reward_drive + 0.15 * route_drive)
        new_weight = _clip(old_weight + weight_delta, 0.0, params.max_edge_weight)

        damping = params.phase_bias_lr * (0.5 + 0.5 * _clip(signal.utility, 0.0, 1.0))
        new_phase = _wrap_phase(old_phase * (1.0 - damping))

        edge.weight = float(new_weight)
        edge.phase_bias = float(new_phase)
        hyperedge_updates.append(
            HyperedgeUpdate(
                edge_id=edge_id,
                old_weight=old_weight,
                new_weight=float(new_weight),
                old_phase_bias=old_phase,
                new_phase_bias=float(new_phase),
                utility_drive=float(utility_drive),
                reward_drive=float(reward_drive),
                route_drive=float(route_drive),
            )
        )

    route_updates: List[RouteUpdate] = []
    preferred = set(decision.preferred_routes)
    for route_id, route in work.routes.items():
        old_score = float(route.score)
        covered_signals = [edge_signal_map[str(edge_id)] for edge_id in route.edge_path if str(edge_id) in edge_signal_map]
        edge_merit = _mean_or_zero(sig.utility for sig in covered_signals)
        bandwidth_term = float(route.bandwidth) / (1.0 + float(route.latency))
        preferred_bonus = 0.2 if route_id in preferred else 0.0
        merit = _clip(0.55 * edge_merit + 0.25 * bandwidth_term + 0.2 * preferred_bonus, 0.0, 1.5)
        new_score = _clip(old_score + params.route_score_lr * (merit - old_score), 0.0, params.max_route_score)
        route.score = float(new_score)
        route_updates.append(
            RouteUpdate(
                route_id=route_id,
                old_score=old_score,
                new_score=float(new_score),
                merit=float(merit),
            )
        )

    work.validate()
    hyperedge_updates.sort(key=lambda item: item.edge_id)
    route_updates.sort(key=lambda item: item.route_id)
    return work, tuple(hyperedge_updates), tuple(route_updates)


def select_control_actions(
    decision: PolicyDecision,
    threshold_update: ThresholdUpdate,
) -> Tuple[Tuple[str, ...], Tuple[int, ...], Tuple[str, ...]]:
    """Select expansion, pruning, and route actions from the adapted thresholds."""
    expansions = tuple(
        candidate.name
        for candidate in decision.expansion_candidates
        if float(candidate.score) >= threshold_update.new_expansion_threshold
    )
    prunes = tuple(
        item.basis_index
        for item in decision.prune_suggestions
        if item.accepted and float(item.probability) <= threshold_update.new_prune_threshold + 1e-15
    )
    return expansions, prunes, tuple(decision.preferred_routes)


def optimizer_step(
    bundle: HilbertBundleState,
    *,
    graph: EntanglementHypergraph | None = None,
    target: HilbertBundleState | np.ndarray | Sequence[complex] | None = None,
    decision: PolicyDecision | None = None,
    memory: ControlMemoryState | None = None,
    hyperparameters: ControlHyperparameters | None = None,
    in_place_graph: bool = False,
) -> OptimizerStepReport:
    """Run one Section 9 optimizer step and return the updated control state."""
    params = hyperparameters or ControlHyperparameters()
    params.validate()

    base_memory = memory or seed_control_memory(bundle)
    resolved_decision = decision or chesso_policy_decision(
        bundle,
        graph=graph,
        target=target,
        prune_threshold=base_memory.prune_threshold,
    )
    updated_memory = update_control_memory(base_memory, resolved_decision, hyperparameters=params)
    threshold_update = adapt_thresholds(updated_memory, resolved_decision, hyperparameters=params)

    updated_graph = None
    hyperedge_updates: Tuple[HyperedgeUpdate, ...] = ()
    route_updates: Tuple[RouteUpdate, ...] = ()
    if graph is not None:
        updated_graph, hyperedge_updates, route_updates = adapt_hypergraph(
            graph,
            resolved_decision,
            memory=updated_memory,
            hyperparameters=params,
            in_place=in_place_graph,
        )

    selected_expansions, selected_prunes, selected_routes = select_control_actions(
        resolved_decision,
        threshold_update,
    )

    final_memory = ControlMemoryState(
        step_count=updated_memory.step_count,
        objective_ema=updated_memory.objective_ema,
        fidelity_ema=updated_memory.fidelity_ema,
        entanglement_ema=updated_memory.entanglement_ema,
        noise_ema=updated_memory.noise_ema,
        mean_sector_utility_ema=updated_memory.mean_sector_utility_ema,
        mean_edge_utility_ema=updated_memory.mean_edge_utility_ema,
        expansion_threshold=float(threshold_update.new_expansion_threshold),
        prune_threshold=float(threshold_update.new_prune_threshold),
        measurement_strength=float(threshold_update.new_measurement_strength),
        preferred_route_history=tuple(selected_routes),
        last_selected_expansions=tuple(selected_expansions),
        last_selected_prunes=tuple(selected_prunes),
        last_notes=tuple(resolved_decision.notes),
        metadata={
            **updated_memory.metadata,
            "last_route_updates": len(route_updates),
            "last_hyperedge_updates": len(hyperedge_updates),
        },
    )

    return OptimizerStepReport(
        decision=resolved_decision,
        memory=final_memory,
        threshold_update=threshold_update,
        hyperedge_updates=tuple(hyperedge_updates),
        route_updates=tuple(route_updates),
        selected_expansions=tuple(selected_expansions),
        selected_prunes=tuple(selected_prunes),
        selected_routes=tuple(selected_routes),
        updated_graph=updated_graph,
        metadata={
            "graph_present": graph is not None,
            "target_present": target is not None,
        },
    )


__all__ = [
    "ControlHyperparameters",
    "ControlMemoryState",
    "ThresholdUpdate",
    "HyperedgeUpdate",
    "RouteUpdate",
    "OptimizerStepReport",
    "seed_control_memory",
    "update_control_memory",
    "adapt_thresholds",
    "adapt_hypergraph",
    "select_control_actions",
    "optimizer_step",
]
