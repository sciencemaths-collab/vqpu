"""Algorithm 9 — One-axis twisting (OAT) squeezer for metrology.

Why
───
Algo 8 showed that entanglement across every bipartition is *necessary*
but not sufficient for Heisenberg-limited metrology. The string-chain
state is entangled everywhere yet sits at F_Q ≈ N. To push past SQL we
need the entanglement to *concentrate* the state's variance along the
measurement axis — that's spin squeezing, and the canonical recipe is
OAT (Kitagawa–Ueda 1993):

        H_OAT = χ · J_z²   ⇒   U(t) = exp(−i·χt·J_z²)

Expanded: J_z² = N/4·I + (1/2)·Σ_{i<j} Z_i Z_j  (up to a global phase),
so OAT is a product of ZZ rotations across every pair. In circuit form
each ZZ(2θ) = CNOT · Rz(2θ) on target · CNOT.

What this script does
─────────────────────
  1. Prepare a base probe state (|+⟩^N or the string-chain state).
  2. Apply OAT for a range of χt values.
  3. Measure F_Q on J_x, J_y, J_z for the squeezed state.
  4. Plot the best F_Q / N vs χt and report the peak.

Target
──────
For |+⟩^N + OAT, literature says F_Q_peak ≈ N^(5/3) (so best/N ~ N^(2/3))
— strictly above SQL but not Heisenberg. This is an honest,
well-understood protocol, not a contrived win. Whether the string-chain
state +OAT does better, same, or worse is empirical.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Callable

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from vqpu import ClassicalSimulatorBackend, QuantumCircuit  # noqa: E402
from examples.algorithms.string_chain import string_chain_circuit  # noqa: E402
from examples.algorithms.string_chain_metrology import (  # noqa: E402
    product_probe, quantum_fisher_information_axis,
)


def apply_oat(circuit: QuantumCircuit, chi_t: float) -> None:
    """Append exp(−i·χt·J_z²) to circuit via all-pairs ZZ rotations.

    Identity:
      J_z² = N/4·I + (1/2)·Σ_{i<j} Z_i Z_j   (Z_i² = I, cross-terms doubled)

    Global phase discarded. Each ZZ rotation decomposes as
      exp(−iθ Z_i Z_j) = CNOT(i,j) · Rz_j(2θ) · CNOT(i,j).
    Per-pair rotation angle for U(χt) is θ_pair = χt/2.
    """
    n = circuit.n_qubits
    theta_pair = chi_t / 2.0
    for i in range(n):
        for j in range(i + 1, n):
            circuit.cnot(i, j)
            circuit.rz(j, 2 * theta_pair)
            circuit.cnot(i, j)


def probe_with_oat(base_builder: Callable[[int], QuantumCircuit],
                   n: int, chi_t: float, label: str) -> QuantumCircuit:
    """Build: base probe, then OAT(χt)."""
    base = base_builder(n)
    c = QuantumCircuit(n, f"{label}_oat_chi_t={chi_t:.3f}")
    for op in base.ops:
        c.ops.append(op)
    apply_oat(c, chi_t)
    return c


def best_qfi(statevector: np.ndarray, n: int) -> tuple[float, str]:
    """Max F_Q across {J_x, J_y, J_z}; returns (value, best_axis)."""
    best_val = -1.0
    best_axis = "x"
    for axis in ("x", "y", "z"):
        val = quantum_fisher_information_axis(statevector, n, axis)
        if val > best_val:
            best_val = val
            best_axis = axis
    return best_val, best_axis


def scan_oat(base_builder: Callable[[int], QuantumCircuit],
             label: str, n: int,
             chi_t_values: np.ndarray) -> dict:
    best_fq = -1.0
    best_axis = "x"
    best_chi_t = 0.0
    sim = ClassicalSimulatorBackend(seed=1)
    for chi_t in chi_t_values:
        circuit = probe_with_oat(base_builder, n, float(chi_t), label)
        sv = sim.execute(circuit, shots=1).statevector
        fq, axis = best_qfi(sv, n)
        if fq > best_fq:
            best_fq = fq
            best_axis = axis
            best_chi_t = float(chi_t)
    return {
        "peak_F_Q": best_fq,
        "best_axis": best_axis,
        "best_chi_t": best_chi_t,
        "best_over_N": best_fq / max(n, 1),
        "best_over_N_sq": best_fq / max(n * n, 1),
    }


def string_chain_builder(layers: int, theta: float, nudge: float):
    def build(n: int) -> QuantumCircuit:
        return string_chain_circuit(n, layers, theta, nudge, seed=1)
    return build


def main() -> None:
    chi_t_values = np.linspace(0.0, math.pi / 2, 30)
    print("  OAT-squeezed probe scan")
    print(f"  χt swept over {len(chi_t_values)} values in [0, π/2]")
    print("  " + "─" * 80)
    print(f"  {'probe':<30s} {'N':>3s} {'peak F_Q':>10s} {'best/N':>8s} "
          f"{'best/N²':>9s} {'best axis':>9s} {'χt*':>7s}")
    print("  " + "─" * 80)

    for n in (4, 6, 8, 10):
        # Baseline — |+⟩^N alone (no OAT). Should be SQL: F_Q = N.
        sv0 = ClassicalSimulatorBackend(seed=1).execute(
            product_probe(n), shots=1
        ).statevector
        fq0, axis0 = best_qfi(sv0, n)
        print(f"  {'product |+⟩^N (no OAT)':<30s} {n:>3d} "
              f"{fq0:>10.3f} {fq0 / n:>7.3f}× {fq0 / (n*n):>8.3f}× "
              f"{axis0:>9s} {'—':>7s}")

        # Product + OAT.
        report = scan_oat(product_probe, "product", n, chi_t_values)
        print(f"  {'product |+⟩^N + OAT':<30s} {n:>3d} "
              f"{report['peak_F_Q']:>10.3f} "
              f"{report['best_over_N']:>7.3f}× "
              f"{report['best_over_N_sq']:>8.3f}× "
              f"{report['best_axis']:>9s} {report['best_chi_t']:>7.3f}")

        # String-chain base + OAT — the headline comparison.
        builder = string_chain_builder(layers=4, theta=0.45, nudge=0.18)
        report = scan_oat(builder, "string_chain_L4", n, chi_t_values)
        print(f"  {'string-chain L=4 + OAT':<30s} {n:>3d} "
              f"{report['peak_F_Q']:>10.3f} "
              f"{report['best_over_N']:>7.3f}× "
              f"{report['best_over_N_sq']:>8.3f}× "
              f"{report['best_axis']:>9s} {report['best_chi_t']:>7.3f}")

        builder = string_chain_builder(layers=n, theta=0.45, nudge=0.18)
        report = scan_oat(builder, "string_chain_LN", n, chi_t_values)
        print(f"  {'string-chain L=N + OAT':<30s} {n:>3d} "
              f"{report['peak_F_Q']:>10.3f} "
              f"{report['best_over_N']:>7.3f}× "
              f"{report['best_over_N_sq']:>8.3f}× "
              f"{report['best_axis']:>9s} {report['best_chi_t']:>7.3f}")

        print("  " + "─" * 80)


if __name__ == "__main__":
    main()
