"""Algorithm 10 — Two-axis twisting + permutation-symmetric probes.

Why
───
Algo 9 confirmed OAT on |+⟩^N saturates Heisenberg (F_Q = N²) at χt = π/2,
and found that string-chain pre-entanglement actively hurts. The natural
next question: is OAT the *fastest* way to reach Heisenberg, or does
another symmetric twist (TAT) get there at smaller χt? And is there any
input state that beats |+⟩^N as the OAT starting point?

TAT in one paragraph
────────────────────
Two-axis twisting Hamiltonian H = χ(J_x² − J_y²) is known to reach
Heisenberg scaling without the cat-formation requirement of OAT. We
Trotterize one step as

  exp(−iε(J_x² − J_y²)) ≈ Ry(−π/2)·U_OAT(+ε)·Ry(+π/2)
                        · Rx(+π/2)·U_OAT(−ε)·Rx(−π/2)

where U_OAT(α) = exp(−iα·J_z²) is our all-pairs ZZ circuit from algo 9.
The first block rotates J_x onto J_z, twists, rotates back. The second
does the same for J_y with the opposite sign, giving the J_x²−J_y²
combination to leading order in ε.

Probes compared
───────────────
  • product |+⟩^N  +  {nothing | OAT | TAT}
  • W_N (single-excitation symmetric state)  +  {nothing | OAT | TAT}
  • string-chain L=4  +  {nothing | OAT | TAT}

For each, we scan χt over a fine grid and report the peak F_Q.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Callable

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from vqpu import ClassicalSimulatorBackend, QuantumCircuit  # noqa: E402
from examples.algorithms.oat_squeezed_probe import apply_oat  # noqa: E402
from examples.algorithms.string_chain import string_chain_circuit  # noqa: E402
from examples.algorithms.string_chain_metrology import (  # noqa: E402
    product_probe, quantum_fisher_information_axis,
)


# ─────────────── Squeezing circuit primitives ───────────────


def apply_tat_step(circuit: QuantumCircuit, epsilon: float) -> None:
    """One first-order Trotter step of TAT with strength ε.

    exp(−iε(J_x² − J_y²))  ≈  exp(−iε J_x²) · exp(+iε J_y²).

    Each half uses collective rotations to conjugate OAT (which is J_z²)
    onto the J_x or J_y axis:
      J_x = Ry(π/2)·J_z·Ry(−π/2)   ⇒   J_x² twist = Ry(π/2) U_OAT(ε) Ry(−π/2)
      J_y = Rx(−π/2)·J_z·Rx(π/2)   ⇒   J_y² twist = Rx(−π/2) U_OAT(ε) Rx(π/2)
    """
    n = circuit.n_qubits

    # +J_x² twist: conjugate OAT with Ry(±π/2) on each qubit (collective Y rotation).
    for i in range(n):
        circuit.ry(i, math.pi / 2)
    apply_oat(circuit, chi_t=epsilon)
    for i in range(n):
        circuit.ry(i, -math.pi / 2)

    # −J_y² twist: conjugate OAT with Rx(∓π/2); sign of ε flipped.
    for i in range(n):
        circuit.rx(i, -math.pi / 2)
    apply_oat(circuit, chi_t=-epsilon)
    for i in range(n):
        circuit.rx(i, math.pi / 2)


# ─────────────── Probe base states ───────────────


def w_state_probe(n: int) -> QuantumCircuit:
    """W_n = (1/√n) Σ_k |e_k⟩  — single-excitation symmetric state.
    Uses the standard linear-circuit construction (Stegmann et al.)."""
    c = QuantumCircuit(n, f"W_{n}")
    # Put all the amplitude on qubit 0 first.
    c.x(0)
    # Then redistribute: angle sequence so the probability of being at
    # site k is equal (1/n) after N-1 controlled rotations.
    for k in range(n - 1):
        # Rotate qubit (k+1) by angle such that amplitude on 0..k is kept
        # at sqrt((k+1)/n), on (k+1) gains sqrt(1/n).  Angle θ_k satisfies
        # cos(θ_k/2) = sqrt((n-k-1)/(n-k)).
        remaining = n - k
        theta = 2 * math.acos(math.sqrt((remaining - 1) / remaining))
        # G-gate: controlled Ry, activated on qubit k being 1
        # Simpler construction: Ry on (k+1) then CNOT(k+1, k) to condition.
        # Here we use: Ry_{k+1}(θ), CNOT(k+1, k), Ry_{k+1}(−θ).
        c.ry(k + 1, theta)
        c.cnot(k + 1, k)
        c.ry(k + 1, -theta)
        # Conditional flip to propagate the excitation.
        c.cnot(k + 1, k)
    return c


def string_chain_probe(n: int, layers: int = 4) -> QuantumCircuit:
    return string_chain_circuit(n, layers, theta=0.45, nudge=0.18, seed=1)


# ─────────────── Analysis ───────────────


def best_qfi(statevector: np.ndarray, n: int) -> tuple[float, str]:
    best = -1.0
    axis = "x"
    for a in ("x", "y", "z"):
        v = quantum_fisher_information_axis(statevector, n, a)
        if v > best:
            best = v
            axis = a
    return best, axis


def scan_chi_t(
    build_probe: Callable[[int], QuantumCircuit],
    n: int,
    chi_t_values: np.ndarray,
    squeezer: str,
) -> dict:
    sim = ClassicalSimulatorBackend(seed=1)
    peak = -1.0
    peak_axis = "x"
    peak_chi_t = 0.0
    for chi_t in chi_t_values:
        base = build_probe(n)
        # Materialize a fresh circuit so we can append gates.
        c = QuantumCircuit(n, f"{base.name}_{squeezer}_chi_t={chi_t:.3f}")
        for op in base.ops:
            c.ops.append(op)
        if squeezer == "oat":
            apply_oat(c, chi_t=float(chi_t))
        elif squeezer == "tat":
            apply_tat_step(c, epsilon=float(chi_t))
        elif squeezer == "none":
            pass
        else:
            raise ValueError(squeezer)
        sv = sim.execute(c, shots=1).statevector
        fq, axis = best_qfi(sv, n)
        if fq > peak:
            peak = fq
            peak_axis = axis
            peak_chi_t = float(chi_t)
    return {
        "peak_F_Q": peak,
        "best_axis": peak_axis,
        "best_chi_t": peak_chi_t,
    }


def run_probe(label: str, build_probe, n: int, chi_t_values: np.ndarray) -> None:
    rows = []
    for squeezer in ("none", "oat", "tat"):
        values = (np.array([0.0]) if squeezer == "none"
                  else chi_t_values)
        r = scan_chi_t(build_probe, n, values, squeezer)
        rows.append((squeezer, r))
    for squeezer, r in rows:
        fq = r["peak_F_Q"]
        ratio_n = fq / max(n, 1)
        ratio_n2 = fq / max(n * n, 1)
        chi_display = (f"{r['best_chi_t']:.3f}"
                       if squeezer != "none" else "  —  ")
        print(f"  {label:<18s} + {squeezer:<4s} :  "
              f"F_Q = {fq:>7.2f}  "
              f"F_Q/N = {ratio_n:>5.2f}×  "
              f"F_Q/N² = {ratio_n2:>5.2f}×  "
              f"axis=J_{r['best_axis']}  χt*={chi_display}")


def main() -> None:
    chi_t_values = np.linspace(0.0, math.pi, 40)
    print("  Permutation-symmetric probe + squeezer landscape")
    print(f"  χt grid: 40 points over [0, π]")
    print("  " + "─" * 86)

    for n in (4, 6, 8, 10):
        print(f"\n  N = {n}")
        print("  " + "─" * 86)
        run_probe("product  |+⟩^N", product_probe, n, chi_t_values)
        run_probe("W_N (Hamming 1)", w_state_probe, n, chi_t_values)
        run_probe(
            "string-chain L=4",
            lambda k=n: string_chain_probe(k, layers=4),
            n, chi_t_values,
        )

    print("")
    print("  Reading the table")
    print("  ─────────────────")
    print("  F_Q/N  > 1 — beats shot-noise limit")
    print("  F_Q/N² = 1 — Heisenberg-limited (information-theoretic max for")
    print("                the collective-spin phase generator family)")
    print("  χt* — twist strength that produced this peak; lower = faster")
    print("         (fewer physical gates in the squeezing step)")


if __name__ == "__main__":
    main()
