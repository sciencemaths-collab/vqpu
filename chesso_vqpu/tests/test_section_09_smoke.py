import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from control import (
    ControlHyperparameters,
    adapt_thresholds,
    optimizer_step,
    seed_control_memory,
    select_control_actions,
    update_control_memory,
)
from core import EntanglementHypergraph, HilbertBundleState, RuntimeConfig
from ops import apply_local_operator, controlled_x, hadamard


def main() -> None:
    cfg = RuntimeConfig.for_statevector(max_active_qubits=6, seed=41)
    cfg.noise.depolarizing_rate = 0.08
    cfg.noise.phase_damping = 0.05
    cfg.budget.max_prune_loss = 0.01

    # 1) Build a Bell state with a deliberately misaligned hyperedge.
    bell = HilbertBundleState.initialize(sector_dims=[2, 2], config=cfg)
    apply_local_operator(bell, hadamard(), [0])
    apply_local_operator(bell, controlled_x(), [0, 1])

    graph = EntanglementHypergraph.from_topology(bell.topology, max_order=3)
    edge = graph.add_hyperedge(["q0", "q1"], weight=1.0, phase_bias=1.2, coherence_score=0.9, capacity=1.0)
    route = graph.add_route("q0", "q1", [str(edge.edge_id)], score=0.1, bandwidth=1.2, latency=0.2)

    wrong_target = HilbertBundleState.initialize(sector_dims=[2, 2], config=cfg)

    memory0 = seed_control_memory(bell)
    report = optimizer_step(
        bell,
        graph=graph,
        target=wrong_target,
        memory=memory0,
        hyperparameters=ControlHyperparameters(),
    )

    # 2) Control memory should advance and absorb the current decision.
    assert report.memory.step_count == 1
    assert report.memory.objective_ema == report.decision.objective.total_score
    assert report.memory.fidelity_ema == report.decision.objective.fidelity_score
    assert report.memory.mean_sector_utility_ema > 0.0

    # 3) Threshold adaptation should remain bounded and lower expansion threshold under mismatch pressure.
    th = report.threshold_update
    assert 0.0 <= th.new_expansion_threshold <= 1.0
    assert 0.0 <= th.new_measurement_strength <= 1.0
    assert th.new_expansion_threshold < th.old_expansion_threshold
    assert th.new_measurement_strength > th.old_measurement_strength

    # 4) Hyperedge adaptation should increase useful edge weight and damp phase bias toward zero.
    assert report.updated_graph is not None
    updated_edge = report.updated_graph.hyperedges[str(edge.edge_id)]
    assert updated_edge.weight > edge.weight
    assert abs(updated_edge.phase_bias) < abs(edge.phase_bias)

    # 5) Route score should improve for the preferred path.
    updated_route = report.updated_graph.routes[route.route_id]
    assert updated_route.score > route.score
    assert report.selected_routes == (route.route_id,)

    # 6) Action selection should accept at least one expansion in this high-mismatch setting.
    assert len(report.selected_expansions) >= 1
    assert all(name.startswith("aux_") for name in report.selected_expansions)

    # 7) Low-probability branches should be discoverable and selectable when the threshold allows it.
    skew = HilbertBundleState.initialize(sector_dims=[2, 2], config=cfg)
    skew.quantum_state.data = np.array([
        np.sqrt(0.998),
        np.sqrt(0.001),
        np.sqrt(0.001),
        0.0,
    ], dtype=np.complex128)
    skew_memory = seed_control_memory(skew, prune_threshold=0.002)
    skew_report = optimizer_step(skew, memory=skew_memory)
    accepted = {item.basis_index for item in skew_report.decision.prune_suggestions if item.accepted}
    assert accepted == {1, 2}
    forced_threshold = replace(skew_report.threshold_update, new_prune_threshold=0.002)
    assert set(select_control_actions(skew_report.decision, forced_threshold)[1]) == {1, 2}

    # 8) Lower-level helpers should stay internally consistent.
    replay = update_control_memory(memory0, report.decision)
    replay_threshold = adapt_thresholds(replay, report.decision)
    replay_actions = select_control_actions(report.decision, replay_threshold)
    assert replay.step_count == 1
    assert replay_actions[2] == report.decision.preferred_routes

    print("Section 09 smoke test passed")


if __name__ == "__main__":
    main()
