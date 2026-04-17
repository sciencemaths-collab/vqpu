"""Algorithm 2 — VQE on H2.

Hypothesis
──────────
A 2-qubit parity-reduced H2 Hamiltonian is expressive enough to support a
meaningful variational quantum eigensolver demonstration. A 4-parameter
RY ansatz with one CNOT layer should converge to FCI ground state within
chemical accuracy (~1 mHa) in a few dozen parameter-shift gradient steps.

What this exercises in vqpu
───────────────────────────
  • repeated quantum-circuit evaluations (classical optimizer in the loop)
  • Pauli-string expectation values via statevector
  • the baseline simulator (need the exact |ψ⟩ for ⟨H⟩ readout)
  • the Phantom backend to inspect the ansatz's subsystem structure

Target energy (FCI, STO-3G, R=0.735 Å): −1.137306 Ha
HF energy (reference):                   −1.117  Ha
Chemical accuracy band:                  ±0.001593 Ha of FCI
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from vqpu import (  # noqa: E402
    ClassicalSimulatorBackend,
    PhantomSimulatorBackend,
    QuantumCircuit,
)

CHEMICAL_ACCURACY = 1.6e-3  # Ha — convergence tolerance against this H's E₀.

# Parity-reduced 2-qubit H2 STO-3G Hamiltonian at R=0.735 Å.
# Canonical form: H = c0 I + c1 Z0 + c2 Z1 + c3 Z0Z1 + c4 (X0X1 + Y0Y1)
# (The earlier version here was missing YY — XX alone gives a different
# matrix whose ground state is not the H2 FCI energy.)
H2_HAMILTONIAN = [
    ("II", -1.052373),
    ("IZ",  0.397936),
    ("ZI", -0.397936),
    ("ZZ", -0.011280),
    ("XX",  0.180931),
    ("YY",  0.180931),
]


_PAULI = {
    "I": np.eye(2, dtype=complex),
    "X": np.array([[0, 1], [1, 0]], dtype=complex),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "Z": np.array([[1, 0], [0, -1]], dtype=complex),
}


def pauli_string_matrix(name: str) -> np.ndarray:
    result = np.array([[1]], dtype=complex)
    for c in name:
        result = np.kron(result, _PAULI[c])
    return result


def hamiltonian_matrix() -> np.ndarray:
    n = len(H2_HAMILTONIAN[0][0])
    dim = 2 ** n
    H = np.zeros((dim, dim), dtype=complex)
    for name, coeff in H2_HAMILTONIAN:
        H += coeff * pauli_string_matrix(name)
    return H


def expectation(statevector: np.ndarray, H: np.ndarray) -> float:
    return float(np.real(np.vdot(statevector, H @ statevector)))


def ry_ansatz(theta: np.ndarray) -> QuantumCircuit:
    """4-parameter hardware-efficient ansatz: Ry Ry CNOT Ry Ry."""
    c = QuantumCircuit(2, "ry_ansatz")
    c.ry(0, float(theta[0]))
    c.ry(1, float(theta[1]))
    c.cnot(0, 1)
    c.ry(0, float(theta[2]))
    c.ry(1, float(theta[3]))
    return c


def energy_at(theta: np.ndarray, H_matrix: np.ndarray) -> float:
    circuit = ry_ansatz(theta)
    # execute returns a populated statevector for the baseline backend.
    result = ClassicalSimulatorBackend(seed=0).execute(circuit, shots=1)
    return expectation(result.statevector, H_matrix)


def parameter_shift_gradient(
    theta: np.ndarray, H_matrix: np.ndarray, shift: float = np.pi / 2
) -> np.ndarray:
    grad = np.zeros_like(theta)
    for i in range(len(theta)):
        plus = theta.copy(); plus[i] += shift
        minus = theta.copy(); minus[i] -= shift
        grad[i] = (energy_at(plus, H_matrix) - energy_at(minus, H_matrix)) / 2.0
    return grad


def main() -> None:
    H_matrix = hamiltonian_matrix()
    eigenvalues = np.linalg.eigvalsh(H_matrix)
    print(f"  Hamiltonian spectrum: {sorted(eigenvalues.real)}")
    print(f"  Ground state E₀:      {eigenvalues[0]:.6f} Ha "
          f"(from diagonalization of H)")
    print(f"  VQE tolerance:        ±{CHEMICAL_ACCURACY:.4f} Ha of E₀")
    print("")

    # Warm start — a few random restarts help avoid local minima.
    rng = np.random.default_rng(20260415)
    best_final: tuple[float, np.ndarray] | None = None
    for restart in range(4):
        theta = rng.uniform(-np.pi, np.pi, size=4)
        history = []
        lr = 0.3
        for step in range(80):
            e = energy_at(theta, H_matrix)
            history.append(e)
            grad = parameter_shift_gradient(theta, H_matrix)
            theta = theta - lr * grad
            # Simple adaptive step: halve lr if we haven't improved in 10 steps.
            if step > 10 and history[-1] >= history[-10]:
                lr = max(lr * 0.7, 0.01)
        final = energy_at(theta, H_matrix)
        gap_fci = final - float(eigenvalues[0])
        marker = "✓" if gap_fci <= CHEMICAL_ACCURACY else "…"
        print(f"  restart {restart}:  E₀ = {history[0]:+.6f}  →  "
              f"E_final = {final:+.6f} Ha  "
              f"(gap-to-FCI = {gap_fci:+.3e}) {marker}")
        if best_final is None or final < best_final[0]:
            best_final = (final, theta)

    assert best_final is not None
    print("")
    print(f"  Best final energy:    {best_final[0]:+.6f} Ha")
    print(f"  FCI target:           {float(eigenvalues[0]):+.6f} Ha")
    print(f"  Residual:             {best_final[0] - float(eigenvalues[0]):+.3e} Ha")
    if best_final[0] - float(eigenvalues[0]) <= CHEMICAL_ACCURACY:
        print("  ✓ Chemical accuracy achieved.")
    else:
        print("  ✗ Did not reach chemical accuracy — try more restarts or steps.")

    # Phantom view of the final ansatz circuit.
    final_circuit = ry_ansatz(best_final[1])
    phantom = PhantomSimulatorBackend(seed=1).execute(final_circuit, shots=4096)
    meta = phantom.execution_metadata
    print("\n  Phantom view of the optimal ansatz circuit:")
    for s in meta["subsystems"]:
        info = s.get("mps_info")
        rep = s["representation"]
        if info:
            print(f"    qubits={s['qubits']}  rep={rep}  "
                  f"bond_dims={info['bond_dims']}  χ_max={info['max_bond_dim']}")
        else:
            print(f"    qubits={s['qubits']}  rep={rep}  "
                  f"active={s['active_states_final']}")


if __name__ == "__main__":
    main()
