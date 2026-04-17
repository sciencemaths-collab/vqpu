from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np

from core import EntanglementHypergraph, HilbertBundleState, SectorId
from ops.expansion import ExpansionCandidate

from .metrics import (
    BundleMetricReport,
    basis_entropy,
    basis_probabilities,
    bundle_metric_report,
    hypergraph_reward,
    l1_coherence,
    reduced_density_matrix,
    state_fidelity,
    von_neumann_entropy,
)


@dataclass(slots=True)
class ObjectiveWeights:
    """Weights for the Section 8 CHESSO objective functional.

    The defaults intentionally favor task fidelity while still rewarding useful
    superposition, entanglement, and coherence. Penalties remain moderate so the
    early simulator explores rather than freezing into timid behavior.
    """

    fidelity: float = 3.0
    superposition: float = 1.0
    entanglement: float = 1.25
    coherence: float = 0.75
    noise_penalty: float = 1.0
    resource_penalty: float = 0.5

    def validate(self) -> None:
        for name, value in (
            ("fidelity", self.fidelity),
            ("superposition", self.superposition),
            ("entanglement", self.entanglement),
            ("coherence", self.coherence),
            ("noise_penalty", self.noise_penalty),
            ("resource_penalty", self.resource_penalty),
        ):
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"Objective weight {name} must be finite and non-negative, got {value!r}")


@dataclass(slots=True)
class SectorPolicySignal:
    """Policy features for one active sector."""

    sector_id: str
    index: int
    dimension: int
    local_entropy_norm: float
    local_coherence_norm: float
    local_purity: float
    population_peak: float
    graph_centrality: float
    decoherence_risk: float
    target_mismatch: float
    utility: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class HyperedgePolicySignal:
    """Policy features for one hyperedge."""

    edge_id: str
    members: Tuple[str, ...]
    order: int
    phase_alignment: float
    entanglement_reward: float
    route_support: float
    utility: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class PruneSuggestion:
    """Suggestion to prune a weak computational-basis branch."""

    basis_index: int
    probability: float
    cumulative_loss: float
    accepted: bool


@dataclass(slots=True)
class ObjectiveReport:
    """Normalized decomposition of the CHESSO objective."""

    total_score: float
    fidelity_score: float
    superposition_score: float
    entanglement_score: float
    coherence_score: float
    noise_penalty: float
    resource_penalty: float
    weights: ObjectiveWeights
    metric_report: BundleMetricReport
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class PolicyDecision:
    """One bundled policy snapshot for the runtime loop."""

    objective: ObjectiveReport
    sector_signals: Tuple[SectorPolicySignal, ...]
    hyperedge_signals: Tuple[HyperedgePolicySignal, ...]
    expansion_candidates: Tuple[ExpansionCandidate, ...]
    prune_suggestions: Tuple[PruneSuggestion, ...]
    preferred_routes: Tuple[str, ...]
    measurement_strength: float
    notes: Tuple[str, ...] = ()
    metadata: Dict[str, Any] = field(default_factory=dict)


# -----------------------------------------------------------------------------
# Linear algebra helpers
# -----------------------------------------------------------------------------


def _safe_log2_dim(dim: int) -> float:
    return float(np.log2(max(2, int(dim))))



def _normalize_unit_interval(value: float) -> float:
    if not np.isfinite(value):
        return 0.0
    return float(np.clip(value, 0.0, 1.0))



def _matrix_sqrt_psd(rho: np.ndarray, *, tol: float = 1e-12) -> np.ndarray:
    vals, vecs = np.linalg.eigh(np.asarray(rho, dtype=np.complex128))
    vals = np.real_if_close(vals).astype(np.float64)
    vals[vals < tol] = 0.0
    return vecs @ np.diag(np.sqrt(vals).astype(np.complex128)) @ vecs.conj().T



def _density_fidelity(rho: np.ndarray, sigma: np.ndarray) -> float:
    rho_t = np.asarray(rho, dtype=np.complex128)
    sigma_t = np.asarray(sigma, dtype=np.complex128)
    if rho_t.shape != sigma_t.shape:
        raise ValueError(f"Density operators must have matching shapes, got {rho_t.shape} and {sigma_t.shape}")
    sqrt_rho = _matrix_sqrt_psd(rho_t)
    middle = sqrt_rho @ sigma_t @ sqrt_rho
    root = _matrix_sqrt_psd(middle)
    val = float(np.real_if_close(np.trace(root)) ** 2)
    return _normalize_unit_interval(val)



def _coherence_normalizer(dim: int) -> float:
    return float(max(1, int(dim) - 1))



def _sector_target_mismatch(
    bundle: HilbertBundleState,
    sector_index: int,
    target: HilbertBundleState | np.ndarray | Sequence[complex] | None,
) -> float:
    if target is None:
        return 0.0
    if not isinstance(target, HilbertBundleState):
        return 0.0
    if bundle.active_sector_count != target.active_sector_count:
        return 0.0
    if bundle.dims[sector_index] != target.dims[sector_index]:
        return 0.0
    rho = reduced_density_matrix(bundle, [sector_index])
    sigma = reduced_density_matrix(target, [sector_index])
    return 1.0 - _density_fidelity(rho, sigma)



def _global_noise_risk(bundle: HilbertBundleState) -> float:
    noise = bundle.config.noise
    raw = (
        1.2 * float(noise.depolarizing_rate)
        + 1.0 * float(noise.amplitude_damping)
        + 0.9 * float(noise.phase_damping)
        + 0.6 * float(noise.readout_error)
        + 0.4 * float(noise.coherent_overrotation)
    )
    return _normalize_unit_interval(raw)



def _resource_penalty(bundle: HilbertBundleState) -> float:
    budget = bundle.config.budget
    active = bundle.active_sector_count / max(1, budget.max_active_qubits)
    depth = bundle.stats.dynamic_depth / max(1, budget.max_dynamic_depth)
    meas = bundle.stats.measurements_used / max(1, budget.max_measurements)
    branches = bundle.stats.branches_used / max(1, budget.max_branches)
    prune = bundle.stats.discarded_trace_mass / max(bundle.config.numerical_tolerance, budget.max_prune_loss)
    val = 0.35 * active + 0.2 * depth + 0.15 * meas + 0.2 * branches + 0.1 * prune
    return _normalize_unit_interval(val)


# -----------------------------------------------------------------------------
# Objective layer
# -----------------------------------------------------------------------------


def evaluate_objective(
    bundle: HilbertBundleState,
    *,
    graph: EntanglementHypergraph | None = None,
    target: HilbertBundleState | np.ndarray | Sequence[complex] | None = None,
    weights: ObjectiveWeights | None = None,
    top_k: int = 5,
) -> ObjectiveReport:
    """Evaluate the Section 8 CHESSO objective on the current bundle state."""
    use_weights = weights or ObjectiveWeights()
    use_weights.validate()

    metrics = bundle_metric_report(bundle, graph=graph, target=target, top_k=top_k)
    total_dim = max(2, bundle.total_dimension)
    entropy_norm = basis_entropy(bundle) / np.log2(total_dim)
    coherence_norm = l1_coherence(bundle) / _coherence_normalizer(total_dim)
    ent_reward = 0.0
    if graph is not None:
        ent_report = hypergraph_reward(bundle, graph)
        ent_reward = ent_report.total_reward / max(1.0, float(len(graph.hyperedges)))
    fidelity = 0.0 if target is None else state_fidelity(bundle, target)

    base_noise = _global_noise_risk(bundle)
    noise_penalty = _normalize_unit_interval(
        base_noise * (0.5 + 0.35 * entropy_norm + 0.15 * coherence_norm)
    )
    resource_penalty = _resource_penalty(bundle)

    total = (
        use_weights.fidelity * fidelity
        + use_weights.superposition * entropy_norm
        + use_weights.entanglement * ent_reward
        + use_weights.coherence * coherence_norm
        - use_weights.noise_penalty * noise_penalty
        - use_weights.resource_penalty * resource_penalty
    )
    return ObjectiveReport(
        total_score=float(total),
        fidelity_score=float(fidelity),
        superposition_score=float(entropy_norm),
        entanglement_score=float(ent_reward),
        coherence_score=float(coherence_norm),
        noise_penalty=float(noise_penalty),
        resource_penalty=float(resource_penalty),
        weights=use_weights,
        metric_report=metrics,
        metadata={
            "global_noise_risk": float(base_noise),
            "active_sector_count": bundle.active_sector_count,
            "hyperedge_count": 0 if graph is None else len(graph.hyperedges),
        },
    )


# -----------------------------------------------------------------------------
# Policy signals
# -----------------------------------------------------------------------------


def sector_policy_signals(
    bundle: HilbertBundleState,
    *,
    graph: EntanglementHypergraph | None = None,
    target: HilbertBundleState | np.ndarray | Sequence[complex] | None = None,
) -> Tuple[SectorPolicySignal, ...]:
    """Compute sector-local CHESSO policy features."""
    if graph is not None:
        graph.sync_from_topology(bundle.topology)
    max_degree = 1.0
    if graph is not None and graph.vertices:
        max_degree = max(1.0, max(graph.weighted_degree(name) for name in graph.vertices))

    global_noise = _global_noise_risk(bundle)
    signals: List[SectorPolicySignal] = []
    for idx, spec in enumerate(bundle.topology.sectors):
        dim = int(spec.dimension)
        local_entropy = von_neumann_entropy(bundle, [idx]) / _safe_log2_dim(dim)
        local_coherence = l1_coherence(bundle, [idx]) / _coherence_normalizer(dim)
        rho = reduced_density_matrix(bundle, [idx])
        local_purity = float(np.real_if_close(np.trace(rho @ rho)))
        peak = float(np.max(np.real_if_close(np.diag(rho)))) if rho.size else 0.0
        centrality = 0.0
        if graph is not None and str(spec.sector_id) in graph.vertices:
            centrality = graph.weighted_degree(str(spec.sector_id)) / max_degree
        target_mismatch = _sector_target_mismatch(bundle, idx, target)
        decoherence_risk = _normalize_unit_interval(
            0.55 * global_noise + 0.25 * local_entropy + 0.15 * local_coherence + 0.05 * (1.0 - local_purity)
        )
        utility = (
            0.28 * local_entropy
            + 0.22 * local_coherence
            + 0.2 * centrality
            + 0.2 * target_mismatch
            + 0.1 * peak
            - 0.25 * decoherence_risk
        )
        signals.append(
            SectorPolicySignal(
                sector_id=str(spec.sector_id),
                index=idx,
                dimension=dim,
                local_entropy_norm=float(local_entropy),
                local_coherence_norm=float(local_coherence),
                local_purity=float(local_purity),
                population_peak=float(peak),
                graph_centrality=float(centrality),
                decoherence_risk=float(decoherence_risk),
                target_mismatch=float(target_mismatch),
                utility=float(utility),
                metadata={"kind": spec.kind.value, "tags": tuple(spec.tags)},
            )
        )
    signals.sort(key=lambda item: (-item.utility, item.index))
    return tuple(signals)



def hyperedge_policy_signals(
    bundle: HilbertBundleState,
    graph: EntanglementHypergraph,
) -> Tuple[HyperedgePolicySignal, ...]:
    """Score hyperedges for routing and entanglement priority."""
    graph.sync_from_topology(bundle.topology)
    reward_report = hypergraph_reward(bundle, graph)
    route_counts: Dict[str, float] = {edge_id: 0.0 for edge_id in graph.hyperedges}
    for route in graph.routes.values():
        strength = max(0.0, float(route.bandwidth)) / (1.0 + max(0.0, float(route.latency)))
        for edge_id in route.edge_path:
            key = str(edge_id)
            if key in route_counts:
                route_counts[key] += strength

    max_route_support = max([1.0, *route_counts.values()])
    signals: List[HyperedgePolicySignal] = []
    for edge_id, edge in graph.hyperedges.items():
        phase_alignment = _normalize_unit_interval((np.cos(edge.phase_bias) + 1.0) / 2.0)
        reward = float(reward_report.edge_rewards.get(edge_id, 0.0))
        route_support = float(route_counts.get(edge_id, 0.0) / max_route_support)
        utility = 0.6 * reward + 0.25 * phase_alignment + 0.15 * route_support
        signals.append(
            HyperedgePolicySignal(
                edge_id=edge_id,
                members=tuple(str(member) for member in edge.members),
                order=edge.order,
                phase_alignment=float(phase_alignment),
                entanglement_reward=reward,
                route_support=float(route_support),
                utility=float(utility),
                metadata={"weight": edge.weight, "capacity": edge.capacity},
            )
        )
    signals.sort(key=lambda item: (-item.utility, item.edge_id))
    return tuple(signals)


# -----------------------------------------------------------------------------
# Decision helpers
# -----------------------------------------------------------------------------


def suggest_expansion_candidates(
    bundle: HilbertBundleState,
    sector_signals: Sequence[SectorPolicySignal],
    *,
    max_new: int | None = None,
    candidate_dimension: int = 2,
    prefix: str = "aux",
    min_utility: float = 0.15,
) -> Tuple[ExpansionCandidate, ...]:
    """Propose new ancilla sectors near high-utility anchors."""
    remaining = bundle.config.budget.max_active_qubits - bundle.active_sector_count
    if remaining <= 0:
        return ()
    limit = remaining if max_new is None else min(max(0, int(max_new)), remaining)
    if limit <= 0:
        return ()

    suggestions: List[ExpansionCandidate] = []
    used_names = {str(sec.sector_id) for sec in bundle.topology.sectors}
    candidate_counter = 0
    for signal in sector_signals:
        if signal.utility < min_utility:
            continue
        name = f"{prefix}_{signal.sector_id}_{candidate_counter}"
        while name in used_names:
            candidate_counter += 1
            name = f"{prefix}_{signal.sector_id}_{candidate_counter}"
        used_names.add(name)
        score = max(0.0, 0.65 * signal.utility + 0.2 * signal.target_mismatch + 0.15 * signal.graph_centrality)
        suggestions.append(
            ExpansionCandidate(
                name=name,
                dimension=int(candidate_dimension),
                score=float(score),
                metadata={
                    "anchor_sector": signal.sector_id,
                    "anchor_index": signal.index,
                    "anchor_utility": signal.utility,
                },
            )
        )
        candidate_counter += 1
        if len(suggestions) >= limit:
            break
    return tuple(suggestions)



def suggest_pruning(
    bundle: HilbertBundleState,
    *,
    threshold: float | None = None,
    max_suggestions: int = 8,
) -> Tuple[PruneSuggestion, ...]:
    """Suggest weak basis branches for pruning within the current loss budget."""
    probs = basis_probabilities(bundle)
    remaining_budget = max(0.0, bundle.config.budget.max_prune_loss - bundle.stats.discarded_trace_mass)
    if remaining_budget <= bundle.config.numerical_tolerance:
        return ()

    dynamic_threshold = threshold
    if dynamic_threshold is None:
        dim_floor = 0.5 / max(1, bundle.total_dimension)
        budget_floor = 0.5 * remaining_budget
        dynamic_threshold = min(budget_floor, max(dim_floor, 10.0 * bundle.config.numerical_tolerance))
    dynamic_threshold = max(0.0, float(dynamic_threshold))

    order = np.argsort(probs)
    suggestions: List[PruneSuggestion] = []
    cumulative = 0.0
    for idx in order:
        p = float(probs[idx])
        if p <= 0.0:
            continue
        accepted = p <= dynamic_threshold and cumulative + p <= remaining_budget + bundle.config.numerical_tolerance
        if accepted:
            cumulative += p
        suggestions.append(
            PruneSuggestion(
                basis_index=int(idx),
                probability=p,
                cumulative_loss=float(cumulative),
                accepted=bool(accepted),
            )
        )
        if len(suggestions) >= max(1, int(max_suggestions)):
            break
    return tuple(suggestions)



def recommend_measurement_strength(
    objective: ObjectiveReport,
    sector_signals: Sequence[SectorPolicySignal],
    *,
    base_strength: float = 0.1,
) -> float:
    """Suggest a soft-measurement strength for the next runtime step."""
    if not sector_signals:
        return _normalize_unit_interval(base_strength)
    avg_risk = float(np.mean([sig.decoherence_risk for sig in sector_signals]))
    avg_coherence = float(np.mean([sig.local_coherence_norm for sig in sector_signals]))
    avg_mismatch = float(np.mean([sig.target_mismatch for sig in sector_signals]))
    raw = base_strength + 0.25 * avg_risk + 0.2 * avg_mismatch - 0.15 * avg_coherence
    raw += 0.1 * objective.noise_penalty
    return _normalize_unit_interval(raw)



def preferred_routes(graph: EntanglementHypergraph | None, *, top_k: int = 3) -> Tuple[str, ...]:
    if graph is None or not graph.routes:
        return ()
    ranked = sorted(
        graph.routes.values(),
        key=lambda route: (
            -(float(route.score) + float(route.bandwidth)) / (1.0 + float(route.latency)),
            route.route_id,
        ),
    )
    return tuple(route.route_id for route in ranked[: max(1, int(top_k))])



def chesso_policy_decision(
    bundle: HilbertBundleState,
    *,
    graph: EntanglementHypergraph | None = None,
    target: HilbertBundleState | np.ndarray | Sequence[complex] | None = None,
    weights: ObjectiveWeights | None = None,
    max_expansions: int | None = None,
    prune_threshold: float | None = None,
    top_k_routes: int = 3,
) -> PolicyDecision:
    """Assemble one coherent Section 8 decision bundle."""
    objective = evaluate_objective(bundle, graph=graph, target=target, weights=weights)
    sectors = sector_policy_signals(bundle, graph=graph, target=target)
    edges = hyperedge_policy_signals(bundle, graph) if graph is not None else ()
    expansions = suggest_expansion_candidates(bundle, sectors, max_new=max_expansions)
    prunes = suggest_pruning(bundle, threshold=prune_threshold)
    measure_strength = recommend_measurement_strength(
        objective,
        sectors,
        base_strength=bundle.config.default_measurement_strength,
    )
    routes = preferred_routes(graph, top_k=top_k_routes)

    notes: List[str] = []
    if objective.fidelity_score < 0.75 and target is not None:
        notes.append("Target fidelity is below 0.75; emphasize corrective evolution.")
    if expansions:
        notes.append(f"Expansion pressure is active around {expansions[0].metadata.get('anchor_sector')}.")
    if any(item.accepted for item in prunes):
        notes.append("Pruning candidates fit inside the current discarded-mass budget.")
    if objective.noise_penalty > 0.5:
        notes.append("Noise penalty is elevated; prefer shorter coherent trajectories.")
    if edges:
        notes.append(f"Top entanglement route leans on {edges[0].edge_id}.")

    return PolicyDecision(
        objective=objective,
        sector_signals=tuple(sectors),
        hyperedge_signals=tuple(edges),
        expansion_candidates=tuple(expansions),
        prune_suggestions=tuple(prunes),
        preferred_routes=tuple(routes),
        measurement_strength=float(measure_strength),
        notes=tuple(notes),
        metadata={
            "target_present": target is not None,
            "graph_present": graph is not None,
        },
    )


__all__ = [
    "ObjectiveWeights",
    "SectorPolicySignal",
    "HyperedgePolicySignal",
    "PruneSuggestion",
    "ObjectiveReport",
    "PolicyDecision",
    "evaluate_objective",
    "sector_policy_signals",
    "hyperedge_policy_signals",
    "suggest_expansion_candidates",
    "suggest_pruning",
    "recommend_measurement_strength",
    "preferred_routes",
    "chesso_policy_decision",
]
