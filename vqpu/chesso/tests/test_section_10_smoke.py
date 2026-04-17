import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vqpu.chesso.control import chesso_runtime_step, run_chesso_runtime
from vqpu.chesso.core import EntanglementHypergraph, HilbertBundleState, RuntimeConfig
from vqpu.chesso.ops import apply_local_operator, controlled_x, hadamard


def build_bell_bundle(cfg: RuntimeConfig) -> HilbertBundleState:
    bundle = HilbertBundleState.initialize(sector_dims=[2, 2], config=cfg)
    apply_local_operator(bundle, hadamard(), [0])
    apply_local_operator(bundle, controlled_x(), [0, 1])
    return bundle


def main() -> None:
    cfg = RuntimeConfig.for_statevector(max_active_qubits=6, seed=47)
    cfg.noise.depolarizing_rate = 0.03
    cfg.noise.phase_damping = 0.02
    cfg.noise.readout_error = 0.01
    cfg.budget.max_prune_loss = 0.02
    cfg.budget.max_steps = 4

    bell = build_bell_bundle(cfg)
    graph = EntanglementHypergraph.from_topology(bell.topology, max_order=3)
    edge = graph.add_hyperedge(["q0", "q1"], weight=1.2, phase_bias=0.8, coherence_score=0.9, capacity=1.0)
    route = graph.add_route("q0", "q1", [str(edge.edge_id)], score=0.2, bandwidth=1.1, latency=0.2)
    wrong_target = HilbertBundleState.initialize(sector_dims=[2, 2], config=cfg)

    # 1) One runtime step should stitch policy, optimizer, actions, and measurement together.
    step = chesso_runtime_step(
        bell,
        graph=graph,
        target=wrong_target,
        step_index=1,
        max_entanglers=1,
    )
    assert step.step_index == 1
    assert step.memory.step_count == 1
    assert step.optimizer_report.selected_routes == (route.route_id,)
    assert step.action_summary.routes == (route.route_id,)
    assert len(step.action_summary.entanglers) == 1
    assert step.action_summary.entanglers[0] == str(edge.edge_id)
    assert len(step.measurement_records) == 1
    assert step.measurement_records[0].label.startswith("runtime:peek")
    assert len(step.bundle.measurements) >= 1
    assert len(step.action_summary.expansions) >= 1
    assert step.after_snapshot.trace > 0.999999
    assert step.graph is not None
    assert step.graph_summary is not None
    assert step.graph_summary.vertex_count == step.bundle.active_sector_count
    assert 0.0 <= step.decision.objective.fidelity_score <= 1.0

    # 2) Multi-step run should keep state, graph, and control memory coherent across steps.
    bell2 = build_bell_bundle(cfg)
    graph2 = EntanglementHypergraph.from_topology(bell2.topology, max_order=3)
    edge2 = graph2.add_hyperedge(["q0", "q1"], weight=1.2, phase_bias=0.8, coherence_score=0.9, capacity=1.0)
    graph2.add_route("q0", "q1", [str(edge2.edge_id)], score=0.2, bandwidth=1.1, latency=0.2)

    run = run_chesso_runtime(
        bell2,
        graph=graph2,
        target=wrong_target,
        steps=3,
        max_entanglers=1,
        stop_fidelity=None,
    )
    assert len(run.steps) == 3
    assert run.final_memory.step_count == 3
    assert run.final_graph is not None
    assert run.final_snapshot.trace > 0.999999
    assert run.final_graph.summary().vertex_count == run.final_bundle.active_sector_count
    assert len(run.final_bundle.measurements) >= 3
    assert any(len(report.action_summary.expansions) >= 1 for report in run.steps)

    print("Section 10 smoke test passed")


if __name__ == "__main__":
    main()
