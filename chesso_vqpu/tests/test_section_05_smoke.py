import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core import HilbertBundleState, RuntimeConfig, StateRepresentation
from ops import (
    apply_nonselective_measurement,
    evaluate_measurement_probabilities,
    hadamard,
    make_computational_basis_instrument,
    measure_computational_basis,
    queue_lazy_basis_measurement,
    apply_local_operator,
    flush_lazy_measurements,
)


def main() -> None:
    # 1) Weak selective measurement should only partially collapse |+>.
    cfg_sv = RuntimeConfig.for_statevector(max_active_qubits=4, seed=37)
    bundle_sv = HilbertBundleState.initialize(sector_dims=[2], config=cfg_sv)
    apply_local_operator(bundle_sv, hadamard(), [0], label="prep_plus")
    record_weak = measure_computational_basis(
        bundle_sv,
        [0],
        strength=0.2,
        outcome=0,
        selective=True,
        label="weak_z",
        store_key="weak_outcome",
    )
    probs = np.asarray(record_weak.metadata["ideal_probabilities"], dtype=np.float64)
    assert np.allclose(probs, np.array([0.5, 0.5]), atol=1e-12)
    assert record_weak.outcome == 0
    assert bundle_sv.classical_memory.get("weak_outcome") == 0
    assert bundle_sv.quantum_state.representation == StateRepresentation.STATEVECTOR
    amps = bundle_sv.quantum_state.data
    assert np.abs(amps[0]) > np.abs(amps[1]) > 0.0
    assert bundle_sv.stats.measurements_used == 1

    # 2) Strong projective measurement should collapse onto the chosen basis state.
    bundle_strong = HilbertBundleState.initialize(sector_dims=[2], config=cfg_sv)
    apply_local_operator(bundle_strong, hadamard(), [0])
    record_strong = measure_computational_basis(bundle_strong, [0], strength=1.0, outcome=1, selective=True)
    expected_one = np.array([0.0, 1.0], dtype=np.complex128)
    assert record_strong.outcome == 1
    assert np.allclose(bundle_strong.quantum_state.data, expected_one, atol=1e-12)

    # 3) Nonselective measurement should promote a statevector to a density matrix and dephase coherence.
    bundle_mix = HilbertBundleState.initialize(sector_dims=[2], config=cfg_sv)
    apply_local_operator(bundle_mix, hadamard(), [0])
    instrument = make_computational_basis_instrument((2,), strength=1.0)
    eval_probs = evaluate_measurement_probabilities(bundle_mix, instrument, [0])
    assert np.allclose(eval_probs, np.array([0.5, 0.5]), atol=1e-12)
    record_mix = apply_nonselective_measurement(bundle_mix, instrument, [0], store_key="mix_probs")
    expected_mix = 0.5 * np.eye(2, dtype=np.complex128)
    assert record_mix.outcome is None
    assert bundle_mix.quantum_state.representation == StateRepresentation.DENSITY_MATRIX
    assert np.allclose(bundle_mix.quantum_state.data, expected_mix, atol=1e-12)
    assert np.allclose(bundle_mix.classical_memory.get("mix_probs"), np.array([0.5, 0.5]), atol=1e-12)
    assert abs(bundle_mix.quantum_state.purity - 0.5) < 1e-12

    # 4) Lazy measurement queue should flush in order and write to classical memory.
    bundle_lazy = HilbertBundleState.initialize(sector_dims=[2, 2], config=cfg_sv)
    apply_local_operator(bundle_lazy, hadamard(), [0])
    queue_lazy_basis_measurement(bundle_lazy, [0], strength=1.0, outcome=0, store_key="m0")
    queue_lazy_basis_measurement(bundle_lazy, [1], strength=1.0, outcome=0, store_key="m1")
    records = flush_lazy_measurements(bundle_lazy)
    assert len(records) == 2
    assert bundle_lazy.classical_memory.get("m0") == 0
    assert bundle_lazy.classical_memory.get("m1") == 0
    assert bundle_lazy.metadata.get("pending_measurements") == []
    assert bundle_lazy.stats.measurements_used == 2

    print("Section 05 smoke test passed")


if __name__ == "__main__":
    main()
