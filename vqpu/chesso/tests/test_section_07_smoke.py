import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vqpu.chesso.control import (
    basis_entropy,
    branch_statistics,
    bundle_metric_report,
    hypergraph_reward,
    l1_coherence,
    mutual_information,
    purity,
    reduced_density_matrix,
    state_fidelity,
    von_neumann_entropy,
)
from vqpu.chesso.core import EntanglementHypergraph, HilbertBundleState, RuntimeConfig, StateRepresentation
from vqpu.chesso.ops import apply_hyperedge_entangler, apply_local_operator, controlled_x, hadamard


def main() -> None:
    cfg_sv = RuntimeConfig.for_statevector(max_active_qubits=6, seed=29)

    # 1) Bell-state metrics: pure globally, maximally mixed locally, MI = 2 bits.
    bell = HilbertBundleState.initialize(sector_dims=[2, 2], config=cfg_sv)
    apply_local_operator(bell, hadamard(), [0])
    apply_local_operator(bell, controlled_x(), [0, 1])

    assert abs(purity(bell) - 1.0) < 1e-12
    reduced = reduced_density_matrix(bell, [0])
    assert np.allclose(reduced, 0.5 * np.eye(2, dtype=np.complex128), atol=1e-12)
    assert abs(von_neumann_entropy(bell, [0]) - 1.0) < 1e-12
    assert abs(mutual_information(bell, [0], [1]) - 2.0) < 1e-12

    bell_vec = np.zeros(4, dtype=np.complex128)
    bell_vec[0] = 1 / np.sqrt(2)
    bell_vec[3] = 1 / np.sqrt(2)
    assert abs(state_fidelity(bell, bell_vec) - 1.0) < 1e-12

    # 2) Plus-state coherence and basis entropy.
    plus = HilbertBundleState.initialize(sector_dims=[2], config=cfg_sv)
    apply_local_operator(plus, hadamard(), [0])
    assert abs(l1_coherence(plus) - 1.0) < 1e-12
    assert abs(basis_entropy(plus) - 1.0) < 1e-12

    # 3) Branch statistics should reflect skewed superposition support.
    skew = HilbertBundleState.initialize(sector_dims=[2, 2], config=cfg_sv)
    custom = np.array([np.sqrt(0.7), np.sqrt(0.2), np.sqrt(0.1), 0.0], dtype=np.complex128)
    skew.quantum_state.data = custom / np.linalg.norm(custom)
    stats = branch_statistics(skew, top_k=2)
    assert stats.effective_support == 3
    assert stats.top_outcomes[0][0] == 0 and abs(stats.top_outcomes[0][1] - 0.7) < 1e-12

    # 4) Hypergraph reward proxy should be positive for an entangled edge.
    ghz = HilbertBundleState.initialize(sector_dims=[2, 2, 2], config=cfg_sv)
    apply_local_operator(ghz, hadamard(), [0])
    apply_local_operator(ghz, controlled_x(), [0, 1])
    apply_local_operator(ghz, controlled_x(), [0, 2])
    graph = EntanglementHypergraph.from_topology(ghz.topology, max_order=3)
    graph.add_hyperedge(["q0", "q1", "q2"], weight=1.2, phase_bias=0.0, coherence_score=0.9, capacity=1.1)
    reward = hypergraph_reward(ghz, graph)
    assert reward.total_reward > 0.0
    assert reward.edge_rewards["edge:q0|q1|q2"] > 0.0

    # 5) Bundle report should package everything together.
    report = bundle_metric_report(ghz, graph=graph, target=ghz)
    assert abs(report.global_purity - 1.0) < 1e-12
    assert report.hypergraph_reward is not None and report.hypergraph_reward.total_reward > 0.0
    assert report.fidelity_to_target is not None and abs(report.fidelity_to_target - 1.0) < 1e-12

    print("Section 07 smoke test passed")


if __name__ == "__main__":
    main()
