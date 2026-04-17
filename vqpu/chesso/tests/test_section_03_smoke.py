import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vqpu.chesso.core import EntanglementHypergraph, HilbertBundleState, RuntimeConfig, StateRepresentation
from vqpu.chesso.ops import apply_hyperedge_entangler, apply_local_operator, controlled_x, hadamard, make_hyperedge_phase_entangler, pauli_x



def main() -> None:
    # Statevector bell-state build with nontrivial target ordering support.
    cfg_sv = RuntimeConfig.for_statevector(max_active_qubits=8, seed=19)
    bundle_sv = HilbertBundleState.initialize(sector_dims=[2, 2], config=cfg_sv)
    apply_local_operator(bundle_sv, hadamard(), ["q0"], label="H_q0")
    apply_local_operator(bundle_sv, controlled_x(), ["q0", "q1"], label="CX_q0_q1")

    expected = np.zeros(4, dtype=np.complex128)
    expected[0] = 1 / np.sqrt(2)
    expected[3] = 1 / np.sqrt(2)
    assert np.allclose(bundle_sv.quantum_state.data, expected, atol=1e-12)
    assert bundle_sv.stats.step == 2

    # Check target permutation by hitting q1 only.
    bundle_perm = HilbertBundleState.initialize(sector_dims=[2, 2], config=cfg_sv)
    apply_local_operator(bundle_perm, pauli_x(), ["q1"], label="X_q1")
    expected_perm = np.zeros(4, dtype=np.complex128)
    expected_perm[1] = 1.0
    assert np.allclose(bundle_perm.quantum_state.data, expected_perm, atol=1e-12)

    # Density-matrix path should preserve purity for unitary evolution.
    cfg_dm = RuntimeConfig.for_density_matrix(max_active_qubits=8, seed=23)
    bundle_dm = HilbertBundleState.initialize(sector_dims=[2, 2], config=cfg_dm)
    apply_local_operator(bundle_dm, hadamard(), ["q0"])
    apply_local_operator(bundle_dm, controlled_x(), ["q0", "q1"])
    bell_dm = np.outer(expected, expected.conj())
    assert bundle_dm.quantum_state.representation == StateRepresentation.DENSITY_MATRIX
    assert np.allclose(bundle_dm.quantum_state.data, bell_dm, atol=1e-12)
    assert abs(bundle_dm.quantum_state.purity - 1.0) < 1e-12

    # Hyperedge entangler should work on multipartite edge and remain unitary-safe.
    bundle_h = HilbertBundleState.initialize(sector_dims=[2, 2, 2], config=cfg_sv)
    graph = EntanglementHypergraph.from_topology(bundle_h.topology, max_order=3)
    graph.add_hyperedge(["q0", "q1", "q2"], weight=0.7, phase_bias=0.15, coherence_score=0.9)
    apply_local_operator(bundle_h, hadamard(), ["q0"])
    apply_hyperedge_entangler(bundle_h, graph, "edge:q0|q1|q2")
    entangler = make_hyperedge_phase_entangler((2, 2, 2), theta=0.3, phase_bias=0.1)
    assert entangler.shape == (8, 8)
    assert np.allclose(entangler.conj().T @ entangler, np.eye(8), atol=1e-12)
    assert abs(np.linalg.norm(bundle_h.quantum_state.data) - 1.0) < 1e-12

    print("Section 03 smoke test passed")


if __name__ == "__main__":
    main()
