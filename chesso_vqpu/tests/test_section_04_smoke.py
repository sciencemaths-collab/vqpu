import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core import HilbertBundleState, RuntimeConfig, StateRepresentation
from ops import (
    amplitude_damping_channel,
    apply_channel,
    apply_noisy_local_operator,
    apply_readout_confusion,
    compose_channels,
    depolarizing_channel,
    hadamard,
    phase_damping_channel,
    single_qubit_readout_confusion,
)


def main() -> None:
    # 1) Amplitude damping should relax |1><1| to |0><0| when gamma=1.
    cfg_dm = RuntimeConfig.for_density_matrix(max_active_qubits=4, seed=29)
    bundle_dm = HilbertBundleState.initialize(sector_dims=[2], config=cfg_dm)
    bundle_dm.quantum_state.data = np.array([[0.0, 0.0], [0.0, 1.0]], dtype=np.complex128)
    apply_channel(bundle_dm, amplitude_damping_channel(1.0), ["q0"], label="amp_full")
    expected_ground = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.complex128)
    assert np.allclose(bundle_dm.quantum_state.data, expected_ground, atol=1e-12)
    assert abs(bundle_dm.quantum_state.trace - 1.0) < 1e-12

    # 2) Channel composition should match sequential application.
    rho_plus = 0.5 * np.array([[1.0, 1.0], [1.0, 1.0]], dtype=np.complex128)
    seq_bundle = HilbertBundleState.initialize(sector_dims=[2], config=cfg_dm)
    seq_bundle.quantum_state.data = rho_plus.copy()
    apply_channel(seq_bundle, phase_damping_channel(0.25), [0])
    apply_channel(seq_bundle, depolarizing_channel(0.15), [0])

    cmp_bundle = HilbertBundleState.initialize(sector_dims=[2], config=cfg_dm)
    cmp_bundle.quantum_state.data = rho_plus.copy()
    composed = compose_channels(phase_damping_channel(0.25), depolarizing_channel(0.15), name="phase_then_dep")
    apply_channel(cmp_bundle, composed, [0])
    assert np.allclose(seq_bundle.quantum_state.data, cmp_bundle.quantum_state.data, atol=1e-12)

    # 3) Applying noise to a statevector should promote it to a density matrix.
    cfg_sv = RuntimeConfig.for_statevector(max_active_qubits=4, seed=31)
    cfg_sv.noise.phase_damping = 0.4
    bundle_sv = HilbertBundleState.initialize(sector_dims=[2], config=cfg_sv)
    apply_noisy_local_operator(bundle_sv, hadamard(), ["q0"], label="H_with_noise")
    assert bundle_sv.quantum_state.representation == StateRepresentation.DENSITY_MATRIX
    assert bundle_sv.quantum_state.purity < 1.0
    assert bundle_sv.metadata.get("state_promoted_to_density_matrix") is True

    # 4) Readout hook should perturb ideal probabilities as expected.
    confusion = single_qubit_readout_confusion(0.2)
    observed = apply_readout_confusion(np.array([1.0, 0.0]), confusion)
    assert np.allclose(confusion, np.array([[0.8, 0.2], [0.2, 0.8]]), atol=1e-12)
    assert np.allclose(observed, np.array([0.8, 0.2]), atol=1e-12)

    print("Section 04 smoke test passed")


if __name__ == "__main__":
    main()
