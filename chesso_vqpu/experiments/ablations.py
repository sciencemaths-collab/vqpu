from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Sequence, Tuple

from control import ObjectiveWeights, chesso_policy_decision
from ops import apply_hyperedge_entangler

from .workloads import MaterializedWorkload, WorkloadSpec, make_noise_stress_workload


@dataclass(slots=True)
class ObjectiveAblationCase:
    """One objective-weight variant to evaluate."""

    label: str
    weights: ObjectiveWeights
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ObjectiveAblationEntry:
    """One evaluated ablation outcome."""

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
class ObjectiveAblationResult:
    """Grouped objective ablation outputs for a single workload."""

    workload_name: str
    entries: Tuple[ObjectiveAblationEntry, ...]
    metadata: Dict[str, Any] = field(default_factory=dict)


_DEFAULT_CASES = (
    ObjectiveAblationCase("default", ObjectiveWeights()),
    ObjectiveAblationCase(
        "no_entanglement_reward",
        ObjectiveWeights(fidelity=3.0, superposition=1.0, entanglement=0.0, coherence=0.75, noise_penalty=1.0, resource_penalty=0.5),
    ),
    ObjectiveAblationCase(
        "superposition_heavy",
        ObjectiveWeights(fidelity=2.2, superposition=2.4, entanglement=1.0, coherence=0.9, noise_penalty=1.0, resource_penalty=0.5),
    ),
    ObjectiveAblationCase(
        "noise_averse",
        ObjectiveWeights(fidelity=2.8, superposition=0.8, entanglement=1.0, coherence=0.75, noise_penalty=2.0, resource_penalty=0.8),
    ),
)


def _materialize(spec_or_workload: WorkloadSpec | MaterializedWorkload) -> MaterializedWorkload:
    return spec_or_workload.materialize() if isinstance(spec_or_workload, WorkloadSpec) else spec_or_workload


def default_ablation_cases() -> Tuple[ObjectiveAblationCase, ...]:
    return tuple(_DEFAULT_CASES)


def run_objective_ablation(
    spec_or_workload: WorkloadSpec | MaterializedWorkload | None = None,
    *,
    cases: Sequence[ObjectiveAblationCase] | None = None,
) -> ObjectiveAblationResult:
    """Evaluate how policy outputs change under different objective weights."""
    workload = _materialize(spec_or_workload or make_noise_stress_workload())
    if workload.bundle is None:
        raise ValueError("Objective ablation requires a bundle-backed workload")

    use_cases = tuple(cases or _DEFAULT_CASES)
    entries = []
    for case in use_cases:
        bundle = workload.bundle.copy()
        graph = None if workload.graph is None else workload.graph.copy()
        if graph is not None and graph.hyperedges:
            first_edge = next(iter(graph.hyperedges))
            apply_hyperedge_entangler(bundle, graph, first_edge, strength=0.5, in_place=True)
        decision = chesso_policy_decision(
            bundle,
            graph=graph,
            target=workload.target,
            weights=case.weights,
        )
        entries.append(
            ObjectiveAblationEntry(
                label=case.label,
                total_score=float(decision.objective.total_score),
                fidelity_score=float(decision.objective.fidelity_score),
                entanglement_score=float(decision.objective.entanglement_score),
                coherence_score=float(decision.objective.coherence_score),
                noise_penalty=float(decision.objective.noise_penalty),
                resource_penalty=float(decision.objective.resource_penalty),
                measurement_strength=float(decision.measurement_strength),
                expansion_count=len(decision.expansion_candidates),
                prune_accept_count=sum(1 for item in decision.prune_suggestions if item.accepted),
                route_count=len(decision.preferred_routes),
                metadata={
                    **case.metadata,
                    "notes": tuple(decision.notes),
                },
            )
        )
    return ObjectiveAblationResult(
        workload_name=workload.name,
        entries=tuple(entries),
        metadata={
            "case_count": len(entries),
            "default_label": use_cases[0].label if use_cases else None,
        },
    )


__all__ = [
    "ObjectiveAblationCase",
    "ObjectiveAblationEntry",
    "ObjectiveAblationResult",
    "default_ablation_cases",
    "run_objective_ablation",
]
