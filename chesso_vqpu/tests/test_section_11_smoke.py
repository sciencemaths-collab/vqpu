import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from control import (
    ActionPhase,
    build_execution_schedule,
    optimizer_step,
    run_scheduled_runtime,
    scheduled_runtime_step,
    seed_control_memory,
)
from core import EntanglementHypergraph, HilbertBundleState, RuntimeConfig
from ops import apply_local_operator, hadamard


def build_uniform_two_qubit_bundle(cfg: RuntimeConfig) -> HilbertBundleState:
    bundle = HilbertBundleState.initialize(sector_dims=[2, 2], config=cfg)
    apply_local_operator(bundle, hadamard(), [0])
    apply_local_operator(bundle, hadamard(), [1])
    return bundle


def main() -> None:
    cfg = RuntimeConfig.for_statevector(max_active_qubits=6, seed=59)
    cfg.noise.depolarizing_rate = 0.35
    cfg.noise.phase_damping = 0.25
    cfg.noise.amplitude_damping = 0.1
    cfg.noise.readout_error = 0.02
    cfg.budget.max_branches = 4
    cfg.budget.max_steps = 3
    cfg.budget.max_prune_loss = 0.05

    bundle = build_uniform_two_qubit_bundle(cfg)
    graph = EntanglementHypergraph.from_topology(bundle.topology, max_order=3)
    edge = graph.add_hyperedge(["q0", "q1"], weight=1.1, phase_bias=0.7, coherence_score=0.9, capacity=1.0)
    route = graph.add_route("q0", "q1", [str(edge.edge_id)], score=0.3, bandwidth=1.2, latency=0.1)
    wrong_target = HilbertBundleState.initialize(sector_dims=[2, 2], config=cfg)

    # 1) Build an explicit schedule and verify the queue logic.
    memory = seed_control_memory(bundle)
    opt = optimizer_step(bundle, graph=graph, target=wrong_target, memory=memory, in_place_graph=False)
    schedule = build_execution_schedule(bundle, opt.decision, opt, graph=graph, max_entanglers=1)
    assert schedule.route_ranking == (route.route_id,)
    assert schedule.branch_pressure >= 1.0
    assert schedule.metadata["precompress"] is True
    assert schedule.queues[ActionPhase.PREPARE].actions
    assert schedule.queues[ActionPhase.ENTANGLE].actions
    assert schedule.queues[schedule.measurement_phase].actions
    assert "entangle" in schedule.executed_action_ids

    # 2) One scheduled runtime step should execute with the planned order and keep state coherent.
    step = scheduled_runtime_step(
        bundle,
        graph=graph,
        target=wrong_target,
        step_index=1,
        max_entanglers=1,
    )
    assert step.runtime.step_index == 1
    assert step.runtime.metadata["scheduled"] is True
    assert step.schedule.route_ranking == (route.route_id,)
    assert step.runtime.action_summary.routes == (route.route_id,)
    assert len(step.runtime.action_summary.entanglers) == 1
    assert len(step.runtime.measurement_records) == 1
    assert step.runtime.after_snapshot.trace > 0.999999
    assert step.runtime.graph is not None
    assert step.runtime.graph_summary is not None
    assert step.runtime.graph_summary.vertex_count == step.runtime.bundle.active_sector_count

    # 3) Multi-step scheduled execution should preserve memory and graph evolution.
    bundle2 = build_uniform_two_qubit_bundle(cfg)
    graph2 = EntanglementHypergraph.from_topology(bundle2.topology, max_order=3)
    edge2 = graph2.add_hyperedge(["q0", "q1"], weight=1.1, phase_bias=0.7, coherence_score=0.9, capacity=1.0)
    graph2.add_route("q0", "q1", [str(edge2.edge_id)], score=0.3, bandwidth=1.2, latency=0.1)

    run = run_scheduled_runtime(
        bundle2,
        graph=graph2,
        target=wrong_target,
        steps=2,
        stop_fidelity=None,
        max_entanglers=1,
    )
    assert len(run.steps) == 2
    assert run.final_memory.step_count == 2
    assert run.final_graph is not None
    assert run.final_snapshot.trace > 0.999999
    assert all(report.runtime.metadata["scheduled"] is True for report in run.steps)
    assert any(report.schedule.branch_pressure >= 1.0 for report in run.steps)

    print("Section 11 smoke test passed")


if __name__ == "__main__":
    main()
