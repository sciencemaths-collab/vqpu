"""Algorithm 8 — String-chain entanglement as a metrology probe.

Goal
────
Turn the string-chain entangled state from algo 7 into something that does
work. Use it as the input state of a Ramsey-style phase-estimation
protocol and show that its entanglement improves the sensitivity to a
global phase φ beyond the shot-noise limit.

The protocol
────────────
1. Prepare a probe state |ψ₀⟩ on N qubits.
2. Evolve under an unknown global phase φ · J_z  with  J_z = Σ Z_i / 2.
   In circuit form: Rz(φ) on every qubit.
3. Rotate the measurement basis (Ry(π/2) — equivalent to a π/2 Ramsey
   pulse) so that J_z precession becomes amplitude modulation.
4. Measure in the computational basis. The resulting counts give an
   estimator φ̂ whose variance is bounded by the Cramér–Rao inequality:

        Var(φ̂) ≥ 1 / (F_Q · shots)

   where F_Q = 4·Var_|ψ₀⟩(J_z) is the Quantum Fisher Information for this
   particular generator.

The three probes we compare
───────────────────────────
   product     |+⟩^N               →  F_Q = N                (SQL)
   GHZ         (|0^N⟩+|1^N⟩)/√2    →  F_Q = N²              (Heisenberg)
   string-chain the algo-7 state    →  F_Q = ?  (measured)

Claim under test
────────────────
The string-chain state is genuinely non-trivially entangled on every
bipartition (we proved this in algo 7). If this entanglement is
collinear with J_z — i.e. the state has large variance of J_z — it
should push F_Q above N (the product-state bound).

Whether it reaches the Heisenberg N² is another question; that requires
the state to be concentrated on the J_z = ±N/2 eigenstates, which is a
much stronger structural requirement than "entangled across every cut."
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from vqpu import ClassicalSimulatorBackend, QuantumCircuit  # noqa: E402
from examples.algorithms.string_chain import (  # noqa: E402
    string_chain_circuit,
)


# ─────────────── Probe-state preparation circuits ───────────────


def product_probe(n: int) -> QuantumCircuit:
    """|+⟩^N — the SQL baseline."""
    c = QuantumCircuit(n, "product_plus")
    for q in range(n):
        c.h(q)
    return c


def ghz_probe(n: int) -> QuantumCircuit:
    """(|0^N⟩ + |1^N⟩)/√2 — the Heisenberg-limited probe."""
    c = QuantumCircuit(n, "ghz")
    c.h(0)
    for q in range(1, n):
        c.cnot(0, q)
    return c


def string_probe(n: int, layers: int, theta: float, nudge: float) -> QuantumCircuit:
    """Our string-chain entangled state from algo 7."""
    return string_chain_circuit(n, layers, theta, nudge, seed=1)


# ─────────────── Quantum Fisher Information ───────────────


_PAULI_Z = np.array([[1, 0], [0, -1]], dtype=complex)
_PAULI_X = np.array([[0, 1], [1, 0]], dtype=complex)
_PAULI_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)


def _collective_generator(axis: str, n: int) -> np.ndarray:
    """J_axis = (1/2) Σ_i σ_axis acting on n qubits. Returns 2^n × 2^n matrix."""
    sigma = {"x": _PAULI_X, "y": _PAULI_Y, "z": _PAULI_Z}[axis]
    dim = 2 ** n
    op = np.zeros((dim, dim), dtype=complex)
    for q in range(n):
        term = np.array([[1]], dtype=complex)
        for k in range(n):
            term = np.kron(term, sigma if k == q else np.eye(2, dtype=complex))
        op += term
    return 0.5 * op


def variance(statevector: np.ndarray, operator: np.ndarray) -> float:
    """Var_|ψ⟩(O) = ⟨O²⟩ − ⟨O⟩² for Hermitian O on a pure state."""
    Op = operator @ statevector
    mean = float(np.real(np.vdot(statevector, Op)))
    mean_sq = float(np.real(np.vdot(Op, Op)))
    return mean_sq - mean * mean


def quantum_fisher_information_axis(
    statevector: np.ndarray, n: int, axis: str
) -> float:
    """F_Q(φ) for the unitary e^{−iφJ_axis} on a pure state = 4·Var(J_axis)."""
    J = _collective_generator(axis, n)
    return 4.0 * variance(statevector, J)


def quantum_fisher_information_global_phase(
    statevector: np.ndarray, n: int
) -> float:
    """Backward-compatible alias — uses J_z."""
    return quantum_fisher_information_axis(statevector, n, "z")


# ─────────────── Simulated Ramsey phase estimation ───────────────


def ramsey_circuit(probe: QuantumCircuit, phi: float) -> QuantumCircuit:
    """probe state → Rz(φ) on every qubit → Ry(−π/2) readout."""
    n = probe.n_qubits
    c = QuantumCircuit(n, f"{probe.name}_ramsey")
    for op in probe.ops:
        c.ops.append(op)
    for q in range(n):
        c.rz(q, phi)
    for q in range(n):
        c.ry(q, -math.pi / 2)
    return c


def estimate_phi_via_jz_expectation(
    counts: dict, n: int, shots: int
) -> float:
    """From measurement counts, estimate ⟨J_z⟩ after Ramsey. For the
    product-state and GHZ protocols this reads out φ directly at small φ.
    Returns the raw ⟨J_z⟩ post-Ramsey — the caller converts to φ̂."""
    mean_jz = 0.0
    for bits, n_hits in counts.items():
        ones = bits.count("1")
        jz = 0.5 * ((n - ones) - ones)
        mean_jz += n_hits * jz
    return mean_jz / shots


def signal_slope_jz_vs_phi(
    probe: QuantumCircuit, phis: np.ndarray, shots: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Run Ramsey at a grid of φ values, return (phis, ⟨J_z⟩(φ))."""
    n = probe.n_qubits
    jz_values = np.zeros_like(phis)
    for i, phi in enumerate(phis):
        circuit = ramsey_circuit(probe, float(phi))
        r = ClassicalSimulatorBackend(seed=seed).execute(circuit, shots=shots)
        jz_values[i] = estimate_phi_via_jz_expectation(r.counts, n, shots)
    return phis, jz_values


@dataclass
class ProbeReport:
    label: str
    n: int
    fq_x: float
    fq_y: float
    fq_z: float
    fq_best: float
    best_axis: str

    def as_row(self) -> str:
        return (
            f"  {self.label:<14s}  n={self.n:>2d}  "
            f"F_Q(x)={self.fq_x:>7.2f}  "
            f"F_Q(y)={self.fq_y:>7.2f}  "
            f"F_Q(z)={self.fq_z:>7.2f}  "
            f"best={self.fq_best:>7.2f} on J_{self.best_axis}  "
            f"best/N={self.fq_best/max(self.n,1):>5.2f}×  "
            f"best/N²={self.fq_best/max(self.n*self.n,1):>5.2f}×"
        )


def analyze_probe(label: str, probe: QuantumCircuit) -> ProbeReport:
    n = probe.n_qubits
    sv = ClassicalSimulatorBackend(seed=1).execute(probe, shots=1).statevector
    fq = {
        axis: quantum_fisher_information_axis(sv, n, axis)
        for axis in ("x", "y", "z")
    }
    best_axis = max(fq, key=lambda a: fq[a])
    return ProbeReport(
        label=label,
        n=n,
        fq_x=fq["x"], fq_y=fq["y"], fq_z=fq["z"],
        fq_best=fq[best_axis], best_axis=best_axis,
    )


def main() -> None:
    print("  Phase-estimation probe comparison")
    print("  Generators tested: e^{−iφ·J_axis}  for  axis ∈ {x, y, z}")
    print("  (we pick whichever axis gives the highest F_Q per probe)")
    print("  " + "─" * 80)

    for n in (4, 6, 8, 10):
        print("")
        print(f"  N = {n} qubits")
        print(analyze_probe("product |+⟩^N", product_probe(n)).as_row())
        print(analyze_probe("GHZ", ghz_probe(n)).as_row())
        print(analyze_probe(
            "string-chain L=4",
            string_probe(n, layers=4, theta=0.45, nudge=0.18),
        ).as_row())
        print(analyze_probe(
            "string-chain L=N",
            string_probe(n, layers=n, theta=0.45, nudge=0.18),
        ).as_row())
        print(analyze_probe(
            "string-chain L=32",
            string_probe(n, layers=32, theta=0.45, nudge=0.18),
        ).as_row())

    print("")
    print("  Reading the table")
    print("  ─────────────────")
    print("  best/N  > 1 : beats shot-noise limit (sensitivity > classical)")
    print("  best/N² = 1 : Heisenberg-limited (entanglement fully exploited)")
    print("  A state can be 'fully entangled' at every cut (algo 7 result) and")
    print("  still have low F_Q on some axes: entanglement is necessary, not")
    print("  sufficient, for metrological advantage.")


if __name__ == "__main__":
    main()
