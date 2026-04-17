"""IonQ-Aria-spec Pauli noise model for vqpu gate sequences.

Published fidelities (IonQ Aria, 2024):
    - 1-qubit native gate (GPI / GPI2):   ~99.98%  →  ε_1Q ≈ 2e-4
    - 2-qubit native gate (MS):           ~99.40%  →  ε_2Q ≈ 6e-3
    - SPAM (state prep + measurement):    ~99.70%  →  ε_SPAM ≈ 3e-3 per qubit

This is a Monte Carlo Pauli-channel approximation, not a full Lindblad
simulator. After every logical gate we roll a depolarizing error on each
affected qubit; after measurement we roll a bit-flip per qubit. It is
calibrated to IonQ's public spec and is what Aria would look like if its
error profile were purely incoherent Pauli — which is the charitable
assumption. Real hardware has coherent miscalibrations and correlated
errors that this model omits.

Use for preflight checks and to bracket expected Hellinger-to-ideal before
burning shots on the real machine. For the real machine, submit via
``QPUCloudPlugin("ionq")``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from vqpu.chesso.experiments.aegis_ion import GateSeq, GateTuple, _targets


@dataclass(slots=True, frozen=True)
class IonQNoiseSpec:
    """IonQ Aria defaults. Override per-field to model Forte or a custom run."""

    e_1q: float = 2.0e-4    # single-qubit depolarizing
    e_2q: float = 6.0e-3    # two-qubit depolarizing
    e_nq: float = 1.2e-2    # n-qubit (FULL_UNITARY) — scales with decomposition
    e_spam: float = 3.0e-3  # measurement bit-flip per qubit

    @classmethod
    def aria(cls) -> "IonQNoiseSpec":
        return cls()

    @classmethod
    def forte(cls) -> "IonQNoiseSpec":
        # Forte Enterprise — tighter 2Q fidelity, same order of magnitude.
        return cls(e_1q=1.5e-4, e_2q=3.5e-3, e_nq=7.0e-3, e_spam=2.0e-3)

    @classmethod
    def noiseless(cls) -> "IonQNoiseSpec":
        return cls(e_1q=0.0, e_2q=0.0, e_nq=0.0, e_spam=0.0)


# ───────────────────────── Pauli application kernel ─────────────────────────

_PAULI_NAMES = ("X", "Y", "Z")


def _apply_pauli(statevec: np.ndarray, n_qubits: int, qubit: int, which: str) -> np.ndarray:
    """Apply a single Pauli on `qubit` of an n-qubit big-endian statevector."""
    dim = 1 << n_qubits
    mask = 1 << (n_qubits - 1 - qubit)
    out = np.empty_like(statevec)
    if which == "X":
        for idx in range(dim):
            out[idx] = statevec[idx ^ mask]
        return out
    if which == "Z":
        out = statevec.copy()
        for idx in range(dim):
            if idx & mask:
                out[idx] = -out[idx]
        return out
    if which == "Y":
        out = np.empty_like(statevec)
        for idx in range(dim):
            flipped = idx ^ mask
            sign = 1j if (idx & mask) else -1j
            out[idx] = sign * statevec[flipped]
        return out
    raise ValueError(which)


def _sample_paulis(rng: np.random.Generator, k: int) -> List[str]:
    """Sample k independent Paulis uniformly from {X,Y,Z}."""
    picks = rng.integers(0, 3, size=k)
    return [_PAULI_NAMES[int(p)] for p in picks]


# ────────────────────────── ideal gate application ──────────────────────────

def _ideal_statevec(n_qubits: int, gate_sequence: GateSeq) -> np.ndarray:
    from vqpu.universal import CPUPlugin
    return CPUPlugin().execute_statevector(n_qubits, gate_sequence)


# ──────────────────────── noisy sampling (Monte Carlo) ──────────────────────

def _apply_gate_to_sv(sv: np.ndarray, n_qubits: int, gate: GateTuple) -> np.ndarray:
    """Apply one logical gate via the CPU plugin's kernel (single-shot, no noise)."""
    from vqpu.universal import CPUPlugin
    return CPUPlugin().execute_statevector(n_qubits, [gate], initial_state=sv)


def _per_gate_error_rate(gate: GateTuple, spec: IonQNoiseSpec) -> Tuple[float, int]:
    """Return (error_rate_per_affected_qubit, n_affected_qubits)."""
    targets = _targets(gate)
    k = len(targets)
    name = gate[0]
    if k == 1:
        return spec.e_1q, 1
    if name in {"CNOT", "CZ", "SWAP"}:
        return spec.e_2q, 2
    if name == "FULL_UNITARY":
        # Charge a per-qubit error proportional to decomposition size.
        return spec.e_nq * max(1, k - 1), k
    return spec.e_1q, k


def sample_with_ionq_noise(
    n_qubits: int,
    gate_sequence: Sequence[GateTuple],
    shots: int,
    *,
    spec: IonQNoiseSpec | None = None,
    seed: int | None = None,
) -> Dict[str, int]:
    """Monte Carlo: for each shot, re-run the circuit with per-gate Pauli errors."""
    spec = spec or IonQNoiseSpec.aria()
    rng = np.random.default_rng(seed)
    counts: Dict[str, int] = {}
    gates = list(gate_sequence)

    for _ in range(shots):
        sv = np.zeros(1 << n_qubits, dtype=complex)
        sv[0] = 1.0
        for gate in gates:
            sv = _apply_gate_to_sv(sv, n_qubits, gate)
            err_rate, _ = _per_gate_error_rate(gate, spec)
            if err_rate <= 0.0:
                continue
            for q in _targets(gate):
                if rng.random() < err_rate:
                    which = _PAULI_NAMES[int(rng.integers(0, 3))]
                    sv = _apply_pauli(sv, n_qubits, int(q), which)
        # Born-rule sample the noisy final state.
        probs = np.abs(sv) ** 2
        probs = probs / probs.sum()
        idx = int(rng.choice(1 << n_qubits, p=probs))
        # SPAM bit-flip per qubit at readout.
        if spec.e_spam > 0.0:
            for q in range(n_qubits):
                if rng.random() < spec.e_spam:
                    idx ^= 1 << (n_qubits - 1 - q)
        bits = format(idx, f"0{n_qubits}b")
        counts[bits] = counts.get(bits, 0) + 1
    return counts


# ─────────────────────────── ideal reference counts ─────────────────────────

def ideal_counts(
    n_qubits: int,
    gate_sequence: Sequence[GateTuple],
    shots: int,
    *,
    seed: int | None = None,
) -> Dict[str, int]:
    rng = np.random.default_rng(seed)
    sv = _ideal_statevec(n_qubits, list(gate_sequence))
    probs = np.abs(sv) ** 2
    probs = probs / probs.sum()
    indices = rng.choice(1 << n_qubits, size=shots, p=probs)
    counts: Dict[str, int] = {}
    for idx in indices:
        bits = format(int(idx), f"0{n_qubits}b")
        counts[bits] = counts.get(bits, 0) + 1
    return counts


def expected_circuit_fidelity(
    gate_sequence: Sequence[GateTuple], spec: IonQNoiseSpec | None = None
) -> float:
    """Rough expected whole-circuit fidelity:  ∏_g (1 − ε_g)^k_g  · SPAM term."""
    spec = spec or IonQNoiseSpec.aria()
    f = 1.0
    qubits_touched = set()
    for g in gate_sequence:
        err, _ = _per_gate_error_rate(g, spec)
        targets = _targets(g)
        f *= (1.0 - err) ** len(targets)
        for q in targets:
            qubits_touched.add(int(q))
    f *= (1.0 - spec.e_spam) ** len(qubits_touched)
    return float(f)


__all__ = [
    "IonQNoiseSpec",
    "expected_circuit_fidelity",
    "ideal_counts",
    "sample_with_ionq_noise",
]
