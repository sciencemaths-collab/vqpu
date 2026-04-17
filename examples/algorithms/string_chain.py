"""Algorithm 7 — "String-network entanglement."

Physical picture (user description)
───────────────────────────────────
Each qubit is a node. Neighbors are linked by a flexible "string" that
carries frequency information — XX+YY coupling, which is exactly the
nearest-neighbor hopping term that lets a single excitation delocalize
across the chain. Periodically we "negate" (Z on staggered sites, sign
flip) and "nudge" (small Ry rotation). The goal is: start with one
excitation localized at site 0, watch the state become genuinely
entangled with every other qubit in the chain — "intrinsically linked,
regardless of distance."

What this algorithm actually is
───────────────────────────────
  • Start: |100…0⟩ (one W-basis state with the excitation at site 0).
  • Layer:
      1. XX+YY hop on every neighbor pair  (the "string" — unitary
         exp(−i θ·(X⊗X+Y⊗Y)/2) per pair; iSWAP-family gate that
         preserves total excitation number).
      2. Z on every even site               (staggered "negate").
      3. Ry(φ) on every qubit                (weak "nudge" away from
         strict U(1)-conservation so the state explores beyond the
         single-excitation sector).
  • Repeated L times, reporting entanglement growth.

What we measure
───────────────
  • Bipartite von Neumann entropy S(A|B) at every cut 1..n−1.
  • Maximum bond dimension the state actually requires as an MPS.
  • Connected correlator ⟨Z_i Z_j⟩ − ⟨Z_i⟩⟨Z_j⟩ as a function of |i−j|.

Claim under test
────────────────
After enough layers, every cut reports non-zero entropy — i.e. no
qubit in the chain factors out of the rest. That is "intrinsically
linked." Whether it gets all the way to the Page/maximum entropy
depends on θ and the number of layers.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from vqpu import (  # noqa: E402
    ClassicalSimulatorBackend,
    GateOp,
    PhantomSimulatorBackend,
    QuantumCircuit,
)


def xx_plus_yy_unitary(theta: float) -> np.ndarray:
    """exp(−i θ · (X⊗X + Y⊗Y) / 2). This is the "string."

    Acts trivially on |00⟩, |11⟩ and rotates within {|01⟩, |10⟩}. It
    preserves total excitation number — a single quantum of excitation
    hops along the chain like a particle on a 1D lattice."""
    c = math.cos(theta)
    s = math.sin(theta)
    return np.array([
        [1, 0,        0,        0],
        [0, c,       -1j * s,   0],
        [0, -1j * s,  c,        0],
        [0, 0,        0,        1],
    ], dtype=complex)


def string_chain_circuit(
    n_qubits: int,
    layers: int,
    theta: float,
    nudge: float,
    seed: int = 0,
) -> QuantumCircuit:
    rng = np.random.default_rng(seed)
    c = QuantumCircuit(n_qubits, f"string_chain_n{n_qubits}_L{layers}")

    # Single-excitation initial state: |100…0⟩ (one of the W-basis kets).
    c.x(0)

    hop = xx_plus_yy_unitary(theta)
    for layer in range(layers):
        # String couplings — alternate even/odd bond pairs to get parallelism.
        for i in range(0, n_qubits - 1, 2):
            c.ops.append(GateOp("Hop", hop, [i, i + 1], is_two_qubit=True))
        for i in range(1, n_qubits - 1, 2):
            c.ops.append(GateOp("Hop", hop, [i, i + 1], is_two_qubit=True))
        # Negate on even sites. Keeps it unitary, flips amplitudes.
        for i in range(0, n_qubits, 2):
            c.z(i)
        # Small nudge — break U(1) slightly so we leave the single-excitation
        # subspace and generate genuine superposition structure.
        for i in range(n_qubits):
            jitter = float(rng.normal(0.0, 0.05))
            c.ry(i, nudge + jitter)
    return c


# ──────────────────────────────────────────────────────────────────────
# Measurements
# ──────────────────────────────────────────────────────────────────────


def bipartite_entropy(statevector: np.ndarray, cut: int, n_qubits: int) -> float:
    """von Neumann entropy across the bipartition {0..cut−1} | {cut..n−1}."""
    state = np.asarray(statevector).reshape([2] * n_qubits)
    # Flatten left qubits into one axis, right into another.
    left_dim = 2 ** cut
    right_dim = 2 ** (n_qubits - cut)
    mat = state.reshape(left_dim, right_dim)
    singular = np.linalg.svd(mat, compute_uv=False)
    singular = singular[singular > 1e-14]
    if singular.size == 0:
        return 0.0
    p = singular ** 2
    p = p / p.sum()
    return float(-np.sum(p * np.log2(p + 1e-300)))


def schmidt_rank(statevector: np.ndarray, cut: int, n_qubits: int,
                 tol: float = 1e-10) -> int:
    state = np.asarray(statevector).reshape([2] * n_qubits)
    mat = state.reshape(2 ** cut, 2 ** (n_qubits - cut))
    singular = np.linalg.svd(mat, compute_uv=False)
    if singular.size == 0:
        return 0
    cutoff = tol * singular[0] if singular[0] > 0 else 0
    return int(np.sum(singular > cutoff))


def z_expectation(statevector: np.ndarray, qubit: int, n_qubits: int) -> float:
    """⟨Z_qubit⟩ computed from the dense statevector."""
    dim = 2 ** n_qubits
    sign = np.array([
        1.0 if ((k >> (n_qubits - 1 - qubit)) & 1) == 0 else -1.0
        for k in range(dim)
    ])
    probs = np.abs(statevector) ** 2
    return float(np.sum(sign * probs))


def zz_correlator(statevector: np.ndarray, i: int, j: int, n_qubits: int) -> float:
    """⟨Z_i Z_j⟩."""
    dim = 2 ** n_qubits
    sign = np.array([
        (1.0 if ((k >> (n_qubits - 1 - i)) & 1) == 0 else -1.0)
        * (1.0 if ((k >> (n_qubits - 1 - j)) & 1) == 0 else -1.0)
        for k in range(dim)
    ])
    probs = np.abs(statevector) ** 2
    return float(np.sum(sign * probs))


# ──────────────────────────────────────────────────────────────────────
# Test harness
# ──────────────────────────────────────────────────────────────────────


def main() -> None:
    n = 10
    theta = 0.45
    nudge = 0.18
    snapshots = [0, 1, 2, 4, 8, 16, 32]

    print(f"  n={n}  θ={theta}  nudge={nudge}  layer snapshots={snapshots}")
    print("")
    print(f"  {'L':>3}  {'max S':>7}  {'max χ':>5}  {'entropy vs cut':>20}  "
          f"{'⟨Z₀Z_j⟩ vs |j|':>26}")
    print("  " + "─" * 78)

    max_layers = max(snapshots)
    # Use a fresh simulator per layer count so seeds line up deterministically.
    for L in snapshots:
        circuit = string_chain_circuit(n, L, theta, nudge, seed=1)
        result = ClassicalSimulatorBackend(seed=1).execute(circuit, shots=1)
        sv = result.statevector

        entropies = [bipartite_entropy(sv, cut, n) for cut in range(1, n)]
        ranks = [schmidt_rank(sv, cut, n) for cut in range(1, n)]
        max_S = max(entropies) if entropies else 0.0
        max_chi = max(ranks) if ranks else 1

        # Short entropy-profile rendering: one digit per cut.
        s_bar = "".join(
            f"{s:.1f} " for s in entropies
        )

        # ⟨Z₀Z_j⟩ vs j — pick a few distances.
        z0 = z_expectation(sv, 0, n)
        corr_str = ""
        for j in (1, 3, 5, 7, 9):
            zj = z_expectation(sv, j, n)
            zz = zz_correlator(sv, 0, j, n)
            connected = zz - z0 * zj
            corr_str += f"{connected:+.2f} "

        print(f"  {L:>3}  {max_S:>7.3f}  {max_chi:>5d}  "
              f"{s_bar.strip():<20s}  {corr_str.strip()}")

    # Run the last snapshot through PHANTOM too — what representation
    # does the partitioner land on for this state?
    print("")
    circuit_final = string_chain_circuit(n, max_layers, theta, nudge, seed=1)
    phantom = PhantomSimulatorBackend(seed=1).execute(circuit_final, shots=1024)
    meta = phantom.execution_metadata
    print(f"  PHANTOM view of the final state (after L={max_layers}):")
    for s in meta["subsystems"]:
        info = s.get("mps_info")
        label = f"qubits={s['qubits']}  rep={s['representation']}"
        if info:
            label += (f"  bond_dims={info['bond_dims']}  "
                      f"χ_max={info['max_bond_dim']}")
        else:
            label += f"  active={s['active_states_final']}"
        print(f"    {label}")

    # Sanity: the chain should be fully entangled — no cut should factor.
    sv_final = ClassicalSimulatorBackend(seed=1).execute(
        circuit_final, shots=1
    ).statevector
    cuts_fully_entangled = all(
        bipartite_entropy(sv_final, c, n) > 1e-6
        for c in range(1, n)
    )
    print("")
    if cuts_fully_entangled:
        print(f"  ✓ No bipartition of the {n}-qubit chain factors — every "
              "cut is entangled.")
    else:
        print(f"  ✗ Some cut factorizes. The state is not fully multipartite.")


if __name__ == "__main__":
    main()
