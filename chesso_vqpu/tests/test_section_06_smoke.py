import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core import HilbertBundleState, QuantumState, RuntimeConfig, SectorId, SectorKind, SectorSpec, BundleTopology, StateRepresentation
from ops import (
    ExpansionCandidate,
    adaptive_expand_bundle,
    attach_ground_sector,
    apply_local_operator,
    basis_populations,
    compress_deterministic_sector_to_memory,
    controlled_x,
    detach_basis_sector,
    hadamard,
    pauli_x,
    prune_low_population_branches,
    trace_out_sectors,
)


def main() -> None:
    # 1) Attach and detach a ground ancilla without disturbing the logical state.
    cfg_sv = RuntimeConfig.for_statevector(max_active_qubits=4, seed=19)
    cfg_sv.budget.max_prune_loss = 0.2
    bundle = HilbertBundleState.initialize(sector_dims=[2], config=cfg_sv)
    apply_local_operator(bundle, hadamard(), [0])
    original = np.array(bundle.quantum_state.data, copy=True)

    attach_ground_sector(bundle, name="anc0", dimension=2)
    assert bundle.dims == (2, 2)
    expected = np.kron(original, np.array([1.0, 0.0], dtype=np.complex128))
    assert np.allclose(bundle.quantum_state.data, expected, atol=1e-12)

    detach_basis_sector(bundle, ["anc0"], basis_index=0)
    assert bundle.dims == (2,)
    assert bundle.quantum_state.representation == StateRepresentation.STATEVECTOR
    assert np.allclose(bundle.quantum_state.data, original, atol=1e-12)

    # 2) Adaptive expansion should prefer higher-scoring candidates and respect the active-sector budget.
    cfg_budget = RuntimeConfig.for_statevector(max_active_qubits=2, seed=5)
    cfg_budget.budget.max_prune_loss = 0.2
    budget_bundle = HilbertBundleState.initialize(sector_dims=[2], config=cfg_budget)
    adaptive_expand_bundle(
        budget_bundle,
        [
            ExpansionCandidate(name="high", score=3.0),
            ExpansionCandidate(name="mid", score=2.0),
            ExpansionCandidate(name="low", score=1.0),
        ],
        max_new=3,
    )
    names = [str(sec.sector_id) for sec in budget_bundle.topology.sectors]
    assert names == ["q0", "high"]
    assert budget_bundle.metadata["last_expansion"] == ("high",)

    # 3) Pruning should remove low-probability branches and account for discarded mass.
    topology = BundleTopology(
        sectors=[
            SectorSpec(SectorId("q0"), 2, SectorKind.LOGICAL),
            SectorSpec(SectorId("q1"), 2, SectorKind.LOGICAL),
        ]
    )
    vec = np.array([
        np.sqrt(0.97),
        np.sqrt(0.02),
        np.sqrt(0.01),
        0.0,
    ], dtype=np.complex128)
    prune_bundle = HilbertBundleState(
        topology=topology,
        quantum_state=QuantumState(vec, StateRepresentation.STATEVECTOR, topology.dims, cfg_sv.numerical_tolerance),
        config=cfg_sv,
    )
    prune_low_population_branches(prune_bundle, threshold=0.015)
    probs = basis_populations(prune_bundle)
    assert np.allclose(probs, np.array([0.97 / 0.99, 0.02 / 0.99, 0.0, 0.0]), atol=1e-12)
    assert abs(prune_bundle.stats.discarded_trace_mass - 0.01) < 1e-12

    # 4) Tracing out half of a Bell pair should produce the maximally mixed reduced state.
    bell = HilbertBundleState.initialize(sector_dims=[2, 2], config=cfg_sv)
    apply_local_operator(bell, hadamard(), [0])
    apply_local_operator(bell, controlled_x(), [0, 1])
    trace_out_sectors(bell, [1])
    assert bell.dims == (2,)
    assert bell.quantum_state.representation == StateRepresentation.DENSITY_MATRIX
    assert np.allclose(bell.quantum_state.data, 0.5 * np.eye(2, dtype=np.complex128), atol=1e-12)

    # 5) Deterministic compression should store the outcome and remove the sector.
    det = HilbertBundleState.initialize(sector_dims=[2, 2], config=cfg_sv)
    apply_local_operator(det, pauli_x(), [1])
    compress_deterministic_sector_to_memory(det, [1], confidence_threshold=0.999, store_key="ancilla_value")
    assert det.dims == (2,)
    assert det.classical_memory.get("ancilla_value") == 1
    assert np.allclose(det.quantum_state.data, np.array([1.0, 0.0], dtype=np.complex128), atol=1e-12)

    print("Section 06 smoke test passed")


if __name__ == "__main__":
    main()
