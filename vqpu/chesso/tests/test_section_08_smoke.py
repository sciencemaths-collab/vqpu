import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vqpu.chesso.control import (
    ObjectiveWeights,
    chesso_policy_decision,
    evaluate_objective,
    hyperedge_policy_signals,
    preferred_routes,
    sector_policy_signals,
    suggest_expansion_candidates,
    suggest_pruning,
)
from vqpu.chesso.core import EntanglementHypergraph, HilbertBundleState, RuntimeConfig
from vqpu.chesso.ops import apply_local_operator, controlled_x, hadamard


def main() -> None:
    cfg = RuntimeConfig.for_statevector(max_active_qubits=6, seed=37)
    cfg.budget.max_prune_loss = 0.02
    cfg.noise.depolarizing_rate = 0.03
    cfg.noise.phase_damping = 0.01

    # 1) Build an entangled state and a routing graph.
    bell = HilbertBundleState.initialize(sector_dims=[2, 2], config=cfg)
    apply_local_operator(bell, hadamard(), [0])
    apply_local_operator(bell, controlled_x(), [0, 1])

    graph = EntanglementHypergraph.from_topology(bell.topology, max_order=3)
    edge = graph.add_hyperedge(["q0", "q1"], weight=1.3, phase_bias=0.0, coherence_score=0.95, capacity=1.1)
    graph.add_route("q0", "q1", [str(edge.edge_id)], score=0.8, bandwidth=1.2, latency=0.1)

    # 2) Objective should reward the correct target more than a mismatched one.
    same_target = bell.copy()
    wrong_target = HilbertBundleState.initialize(sector_dims=[2, 2], config=cfg)
    obj_same = evaluate_objective(bell, graph=graph, target=same_target, weights=ObjectiveWeights())
    obj_wrong = evaluate_objective(bell, graph=graph, target=wrong_target, weights=ObjectiveWeights())
    assert abs(obj_same.fidelity_score - 1.0) < 1e-12
    assert obj_same.total_score > obj_wrong.total_score
    assert obj_same.entanglement_score > 0.0

    # 3) Symmetric Bell state should give matched sector utilities and positive graph centrality.
    sector_signals = sector_policy_signals(bell, graph=graph, target=same_target)
    assert len(sector_signals) == 2
    assert abs(sector_signals[0].utility - sector_signals[1].utility) < 1e-12
    assert all(sig.graph_centrality > 0.0 for sig in sector_signals)

    # 4) Hyperedge policy should prioritize the only available entanglement edge and route.
    edge_signals = hyperedge_policy_signals(bell, graph)
    assert len(edge_signals) == 1
    assert edge_signals[0].edge_id == "edge:q0|q1"
    assert edge_signals[0].utility > 0.0
    assert preferred_routes(graph) == (next(iter(graph.routes.keys())),)

    # 5) Expansion suggestions should respect the active-sector budget and anchor on high-utility sectors.
    expansions = suggest_expansion_candidates(bell, sector_signals, max_new=2)
    assert 1 <= len(expansions) <= 2
    assert all(candidate.metadata["anchor_sector"] in {"q0", "q1"} for candidate in expansions)
    assert all(candidate.score > 0.0 for candidate in expansions)

    # 6) Pruning should accept only weak branches that fit inside the prune-loss budget.
    skew = HilbertBundleState.initialize(sector_dims=[2, 2], config=cfg)
    skew.quantum_state.data = np.array([
        np.sqrt(0.998),
        np.sqrt(0.001),
        np.sqrt(0.001),
        0.0,
    ], dtype=np.complex128)
    prune = suggest_pruning(skew)
    accepted = [item for item in prune if item.accepted]
    assert len(accepted) == 2
    assert sum(item.probability for item in accepted) <= cfg.budget.max_prune_loss + 1e-12

    # 7) Full policy bundle should be coherent and bounded.
    decision = chesso_policy_decision(bell, graph=graph, target=same_target, max_expansions=1)
    assert decision.objective.total_score == obj_same.total_score
    assert len(decision.sector_signals) == 2
    assert len(decision.hyperedge_signals) == 1
    assert len(decision.expansion_candidates) == 1
    assert 0.0 <= decision.measurement_strength <= 1.0
    assert decision.preferred_routes == preferred_routes(graph)
    assert len(decision.notes) >= 1

    print("Section 08 smoke test passed")


if __name__ == "__main__":
    main()
