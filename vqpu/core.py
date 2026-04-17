"""
Virtual Quantum Processing Unit (vQPU)
======================================
A quantum-native architecture running on classical hardware.

The design principle: every component thinks in quantum terms
(amplitudes, unitaries, entanglement, measurement) but executes
on whatever backend is available — CPU, GPU, or real QPU.

The architecture has 5 layers:
  1. Amplitude Core    — State lives as complex amplitude vectors, not bits
  2. Gate Engine       — All transforms are unitary matrices (reversible)
  3. Entanglement Mesh — Tensor network tracks which qubits are correlated
  4. Measurement Tap   — Born rule sampling collapses state to classical
  5. Backend Adapter   — Swappable: CPU now, QPU later, same API

Key insight: For N < ~28 qubits, exact classical simulation is feasible.
The architecture doesn't pretend to have quantum speedup — it provides
quantum-correct computation with a clean migration path to real hardware.
"""

import numpy as np
from typing import List, Tuple, Optional, Callable, Dict, Any
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import time
import json


# ═══════════════════════════════════════════════════════════
#  LAYER 0: QUANTUM TYPES
#  Everything is amplitudes, not classical bits
# ═══════════════════════════════════════════════════════════

@dataclass
class QubitState:
    """A single qubit: α|0⟩ + β|1⟩ where |α|² + |β|² = 1"""
    alpha: complex
    beta: complex

    @staticmethod
    def zero():
        return QubitState(1.0 + 0j, 0.0 + 0j)

    @staticmethod
    def one():
        return QubitState(0.0 + 0j, 1.0 + 0j)

    @staticmethod
    def plus():
        s = 1 / np.sqrt(2)
        return QubitState(s + 0j, s + 0j)

    @staticmethod
    def minus():
        s = 1 / np.sqrt(2)
        return QubitState(s + 0j, -s + 0j)

    def probabilities(self):
        return abs(self.alpha)**2, abs(self.beta)**2

    def to_vector(self):
        return np.array([self.alpha, self.beta], dtype=complex)


@dataclass
class QuantumRegister:
    """
    N-qubit register: 2^N complex amplitudes.
    This IS the quantum state — not a description of it,
    but the actual probability amplitude vector.
    """
    n_qubits: int
    amplitudes: np.ndarray  # shape: (2^n,) complex128
    _entanglement_map: Dict[int, set] = field(default_factory=dict)

    @staticmethod
    def from_classical(bits: List[int]) -> 'QuantumRegister':
        """Initialize from classical bit string |bits⟩"""
        n = len(bits)
        amps = np.zeros(2**n, dtype=complex)
        index = int(''.join(str(b) for b in bits), 2)
        amps[index] = 1.0
        return QuantumRegister(n_qubits=n, amplitudes=amps)

    @staticmethod
    def zeros(n: int) -> 'QuantumRegister':
        """Initialize |00...0⟩"""
        return QuantumRegister.from_classical([0] * n)

    def probability_distribution(self) -> np.ndarray:
        """Born rule: P(x) = |⟨x|ψ⟩|²"""
        return np.abs(self.amplitudes) ** 2

    def is_entangled(self, qubit_a: int, qubit_b: int) -> bool:
        """Check if two qubits are entangled (tracked by gate applications)"""
        return qubit_b in self._entanglement_map.get(qubit_a, set())

    def entanglement_pairs(self) -> List[Tuple[int, int]]:
        """Return all entangled qubit pairs"""
        pairs = set()
        for q, partners in self._entanglement_map.items():
            for p in partners:
                pairs.add((min(q, p), max(q, p)))
        return sorted(pairs)

    def mark_entangled(self, qubit_a: int, qubit_b: int):
        """Record that a 2-qubit gate created entanglement"""
        if qubit_a not in self._entanglement_map:
            self._entanglement_map[qubit_a] = set()
        if qubit_b not in self._entanglement_map:
            self._entanglement_map[qubit_b] = set()
        self._entanglement_map[qubit_a].add(qubit_b)
        self._entanglement_map[qubit_b].add(qubit_a)

    def fidelity(self, other: 'QuantumRegister') -> float:
        """Quantum fidelity: |⟨ψ|φ⟩|²"""
        return float(abs(np.vdot(self.amplitudes, other.amplitudes)) ** 2)

    def entropy(self) -> float:
        """Von Neumann entropy of the probability distribution"""
        probs = self.probability_distribution()
        probs = probs[probs > 1e-15]
        return float(-np.sum(probs * np.log2(probs)))

    def __repr__(self):
        n = self.n_qubits
        nonzero = [(i, self.amplitudes[i]) for i in range(2**n) if abs(self.amplitudes[i]) > 1e-10]
        terms = []
        for idx, amp in nonzero[:8]:
            bits = format(idx, f'0{n}b')
            if abs(amp.imag) < 1e-10:
                terms.append(f"{amp.real:.4f}|{bits}⟩")
            else:
                terms.append(f"({amp:.4f})|{bits}⟩")
        s = " + ".join(terms)
        if len(nonzero) > 8:
            s += f" + ... ({len(nonzero)} terms)"
        return f"QuantumRegister({n} qubits): {s}"


# ═══════════════════════════════════════════════════════════
#  LAYER 1: GATE ENGINE
#  All operations are unitary matrices — reversible by design
# ═══════════════════════════════════════════════════════════

class GateLibrary:
    """
    Quantum gates as unitary matrices.
    Every gate G satisfies G†G = I (reversible).
    """

    # --- Single-qubit gates ---
    I = np.eye(2, dtype=complex)
    X = np.array([[0, 1], [1, 0]], dtype=complex)         # NOT / Pauli-X
    Y = np.array([[0, -1j], [1j, 0]], dtype=complex)      # Pauli-Y
    Z = np.array([[1, 0], [0, -1]], dtype=complex)         # Pauli-Z / Phase flip
    H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)  # Hadamard
    S = np.array([[1, 0], [0, 1j]], dtype=complex)         # Phase gate
    T = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=complex)  # π/8 gate

    @staticmethod
    def Rx(theta: float) -> np.ndarray:
        """Rotation around X axis"""
        c, s = np.cos(theta / 2), np.sin(theta / 2)
        return np.array([[c, -1j * s], [-1j * s, c]], dtype=complex)

    @staticmethod
    def Ry(theta: float) -> np.ndarray:
        """Rotation around Y axis"""
        c, s = np.cos(theta / 2), np.sin(theta / 2)
        return np.array([[c, -s], [s, c]], dtype=complex)

    @staticmethod
    def Rz(theta: float) -> np.ndarray:
        """Rotation around Z axis"""
        return np.array([
            [np.exp(-1j * theta / 2), 0],
            [0, np.exp(1j * theta / 2)]
        ], dtype=complex)

    @staticmethod
    def Phase(phi: float) -> np.ndarray:
        """General phase gate"""
        return np.array([[1, 0], [0, np.exp(1j * phi)]], dtype=complex)

    # --- Two-qubit gates ---
    CNOT = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 0]
    ], dtype=complex)

    CZ = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, -1]
    ], dtype=complex)

    SWAP = np.array([
        [1, 0, 0, 0],
        [0, 0, 1, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1]
    ], dtype=complex)

    @staticmethod
    def verify_unitary(gate: np.ndarray, name: str = "gate") -> bool:
        """Verify G†G = I (quantum gates must be reversible)"""
        product = gate.conj().T @ gate
        identity = np.eye(gate.shape[0], dtype=complex)
        is_unitary = np.allclose(product, identity, atol=1e-10)
        if not is_unitary:
            raise ValueError(f"{name} is NOT unitary — violates quantum mechanics")
        return True


class GateEngine:
    """
    Applies quantum gates to registers.
    All operations modify the amplitude vector via matrix multiplication.
    """

    def __init__(self):
        self.gate_count = 0
        self.gate_log: List[dict] = []
        self.lib = GateLibrary()

    def apply_single(self, reg: QuantumRegister, gate: np.ndarray,
                     target: int, label: str = "") -> QuantumRegister:
        """Apply single-qubit gate to target qubit without building full matrix."""
        n = reg.n_qubits
        dim = 2 ** n
        new_amps = np.zeros(dim, dtype=complex)
        step = 2 ** (n - target - 1)

        for i in range(dim):
            bit = (i // step) % 2
            partner = i ^ step  # flip target bit
            if bit == 0:
                new_amps[i] += gate[0, 0] * reg.amplitudes[i] + gate[0, 1] * reg.amplitudes[partner]
                new_amps[partner] += gate[1, 0] * reg.amplitudes[i] + gate[1, 1] * reg.amplitudes[partner]

        reg.amplitudes = new_amps
        self.gate_count += 1
        self.gate_log.append({
            "type": "single", "gate": label or "U",
            "target": target, "time": self.gate_count
        })
        return reg

    def apply_two_qubit(self, reg: QuantumRegister, gate: np.ndarray,
                        control: int, target: int, label: str = "") -> QuantumRegister:
        """Apply two-qubit gate (control, target) to register."""
        n = reg.n_qubits
        dim = 2 ** n
        new_amps = np.zeros(dim, dtype=complex)

        for i in range(dim):
            bits = list(format(i, f'0{n}b'))
            c_bit = int(bits[control])
            t_bit = int(bits[target])

            # Extract the 2-qubit sub-state index
            sub_idx = c_bit * 2 + t_bit

            for j in range(4):
                if abs(gate[j, sub_idx]) < 1e-15:
                    continue
                new_bits = bits.copy()
                new_bits[control] = str(j // 2)
                new_bits[target] = str(j % 2)
                new_idx = int(''.join(new_bits), 2)
                new_amps[new_idx] += gate[j, sub_idx] * reg.amplitudes[i]

        reg.amplitudes = new_amps
        reg.mark_entangled(control, target)
        self.gate_count += 1
        self.gate_log.append({
            "type": "two_qubit", "gate": label or "U2",
            "control": control, "target": target, "time": self.gate_count
        })
        return reg

    def apply_multi(self, reg: QuantumRegister, gate: np.ndarray,
                    targets: List[int], label: str = "") -> QuantumRegister:
        """Apply an arbitrary n-qubit gate (full matrix) to specified qubits."""
        n = reg.n_qubits
        dim = 2 ** n
        n_gate = len(targets)

        if gate.shape[0] == dim:
            # Gate is already the full 2^n operator
            reg.amplitudes = gate @ reg.amplitudes
        else:
            # Need to embed the sub-gate into the full space
            new_amps = np.zeros(dim, dtype=complex)
            for i in range(dim):
                bits = list(format(i, f'0{n}b'))
                sub_idx = 0
                for k, t in enumerate(targets):
                    sub_idx = sub_idx * 2 + int(bits[t])

                for j in range(2**n_gate):
                    if abs(gate[j, sub_idx]) < 1e-15:
                        continue
                    new_bits = bits.copy()
                    for k, t in enumerate(targets):
                        new_bits[t] = str((j >> (n_gate - 1 - k)) & 1)
                    new_idx = int(''.join(new_bits), 2)
                    new_amps[new_idx] += gate[j, sub_idx] * reg.amplitudes[i]
            reg.amplitudes = new_amps

        # Mark entanglement between all targets
        for a in targets:
            for b in targets:
                if a != b:
                    reg.mark_entangled(a, b)

        self.gate_count += 1
        self.gate_log.append({
            "type": "multi", "gate": label or "Un",
            "targets": targets, "time": self.gate_count
        })
        return reg

    def apply_controlled(self, reg: QuantumRegister, gate: np.ndarray,
                         control: int, target: int, label: str = "") -> QuantumRegister:
        """Apply controlled-U gate: apply gate to target only if control is |1⟩"""
        n = reg.n_qubits
        dim = 2 ** n
        new_amps = reg.amplitudes.copy()

        for i in range(dim):
            bits = format(i, f'0{n}b')
            if bits[control] == '1':
                t_bit = int(bits[target])
                for new_t in range(2):
                    if abs(gate[new_t, t_bit]) < 1e-15:
                        continue
                    new_bits = list(bits)
                    new_bits[target] = str(new_t)
                    new_idx = int(''.join(new_bits), 2)
                    if new_t != t_bit:
                        new_amps[new_idx] += gate[new_t, t_bit] * reg.amplitudes[i]
                        new_amps[i] -= reg.amplitudes[i]

        # Simpler: rebuild from scratch
        new_amps2 = np.zeros(dim, dtype=complex)
        for i in range(dim):
            bits = list(format(i, f'0{n}b'))
            c_val = int(bits[control])
            if c_val == 0:
                new_amps2[i] += reg.amplitudes[i]
            else:
                t_val = int(bits[target])
                for new_t in range(2):
                    new_bits = bits.copy()
                    new_bits[target] = str(new_t)
                    new_idx = int(''.join(new_bits), 2)
                    new_amps2[new_idx] += gate[new_t, t_val] * reg.amplitudes[i]

        reg.amplitudes = new_amps2
        reg.mark_entangled(control, target)
        self.gate_count += 1
        self.gate_log.append({
            "type": "controlled", "gate": label or "CU",
            "control": control, "target": target, "time": self.gate_count
        })
        return reg


# ═══════════════════════════════════════════════════════════
#  LAYER 2: MEASUREMENT TAP
#  Born rule sampling — collapses quantum to classical
# ═══════════════════════════════════════════════════════════

class MeasurementTap:
    """
    Quantum measurement via Born rule: P(x) = |⟨x|ψ⟩|²
    
    Key property: measurement is DESTRUCTIVE — it collapses
    the superposition. This is faithfully simulated.
    """

    def __init__(self, seed: Optional[int] = None):
        self.rng = np.random.default_rng(seed)
        self.measurement_log: List[dict] = []

    def measure_all(self, reg: QuantumRegister, shots: int = 1) -> List[str]:
        """
        Measure all qubits. Returns classical bit strings.
        Collapses the state after measurement.
        """
        probs = reg.probability_distribution()
        n = reg.n_qubits
        indices = self.rng.choice(2**n, size=shots, p=probs)
        results = [format(idx, f'0{n}b') for idx in indices]

        if shots >= 1:
            # Collapse the register to the first observed outcome so callers
            # see destructive measurement semantics even when sampling more
            # than once from the pre-measurement distribution.
            collapsed = np.zeros_like(reg.amplitudes)
            collapsed[indices[0]] = 1.0
            reg.amplitudes = collapsed

        self.measurement_log.append({
            "type": "full", "shots": shots, "results": results[:10]
        })
        return results

    def measure_qubit(self, reg: QuantumRegister, qubit: int) -> int:
        """
        Measure a single qubit. Collapses that qubit's state
        but preserves superposition of unmeasured qubits.
        """
        n = reg.n_qubits
        probs = reg.probability_distribution()

        # Probability of measuring |0⟩ on target qubit
        p0 = 0.0
        for i in range(2**n):
            bits = format(i, f'0{n}b')
            if bits[qubit] == '0':
                p0 += probs[i]

        result = 0 if self.rng.random() < p0 else 1

        # Collapse: zero out amplitudes incompatible with result
        for i in range(2**n):
            bits = format(i, f'0{n}b')
            if int(bits[qubit]) != result:
                reg.amplitudes[i] = 0.0

        # Renormalize
        norm = np.linalg.norm(reg.amplitudes)
        if norm > 1e-15:
            reg.amplitudes /= norm

        self.measurement_log.append({
            "type": "single", "qubit": qubit, "result": result
        })
        return result

    def expectation(self, reg: QuantumRegister, observable: np.ndarray) -> float:
        """Compute ⟨ψ|O|ψ⟩ without collapsing state"""
        return float(np.real(reg.amplitudes.conj() @ observable @ reg.amplitudes))

    def sample_distribution(self, reg: QuantumRegister, shots: int = 1024) -> dict:
        """Sample the probability distribution over multiple shots"""
        results = self.measure_all(
            QuantumRegister(reg.n_qubits, reg.amplitudes.copy()), shots
        )
        counts = {}
        for r in results:
            counts[r] = counts.get(r, 0) + 1
        return dict(sorted(counts.items(), key=lambda x: -x[1]))


# ═══════════════════════════════════════════════════════════
#  LAYER 3: CIRCUIT COMPOSER
#  Build quantum circuits as sequences of gates
# ═══════════════════════════════════════════════════════════

@dataclass
class GateOp:
    """Single operation in a quantum circuit"""
    gate_name: str
    gate_matrix: np.ndarray
    targets: List[int]
    params: Optional[List[float]] = None
    is_two_qubit: bool = False


@dataclass
class SymmetryDescriptor:
    """Declarative measurement-time symmetry constraint."""
    name: str
    description: str
    validator: Callable[[str], bool]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def validate(self, bitstring: str) -> bool:
        return bool(self.validator(bitstring))

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "metadata": self.metadata,
        }

    @staticmethod
    def _extract_bits(bitstring: str, qubits: Optional[List[int]]) -> str:
        if qubits is None:
            return bitstring
        return ''.join(bitstring[q] for q in qubits)

    @classmethod
    def fixed_hamming_weight(
        cls,
        weight: int,
        qubits: Optional[List[int]] = None,
        name: Optional[str] = None,
    ) -> 'SymmetryDescriptor':
        scope = list(qubits) if qubits is not None else None
        label = name or f"hamming_weight={weight}"
        description = (
            f"Keep only outcomes with Hamming weight {weight}"
            + (f" on qubits {scope}" if scope is not None else "")
        )

        def validator(bitstring: str) -> bool:
            bits = cls._extract_bits(bitstring, scope)
            return sum(int(bit) for bit in bits) == weight

        return cls(
            name=label,
            description=description,
            validator=validator,
            metadata={"kind": "hamming_weight", "weight": weight, "qubits": scope},
        )

    @classmethod
    def parity(
        cls,
        parity: str,
        qubits: Optional[List[int]] = None,
        name: Optional[str] = None,
    ) -> 'SymmetryDescriptor':
        normalized = parity.lower()
        if normalized not in {"even", "odd"}:
            raise ValueError("parity must be 'even' or 'odd'")

        scope = list(qubits) if qubits is not None else None
        label = name or f"{normalized}_parity"
        description = (
            f"Keep only outcomes with {normalized} parity"
            + (f" on qubits {scope}" if scope is not None else "")
        )

        def validator(bitstring: str) -> bool:
            bits = cls._extract_bits(bitstring, scope)
            ones = sum(int(bit) for bit in bits)
            return (ones % 2 == 0) if normalized == "even" else (ones % 2 == 1)

        return cls(
            name=label,
            description=description,
            validator=validator,
            metadata={"kind": "parity", "parity": normalized, "qubits": scope},
        )

    @classmethod
    def allowed_bitstrings(
        cls,
        bitstrings: List[str],
        qubits: Optional[List[int]] = None,
        name: Optional[str] = None,
    ) -> 'SymmetryDescriptor':
        allowed = sorted(set(bitstrings))
        scope = list(qubits) if qubits is not None else None
        label = name or "allowed_bitstrings"
        description = (
            "Keep only outcomes matching an allowed bitstring set"
            + (f" on qubits {scope}" if scope is not None else "")
        )

        def validator(bitstring: str) -> bool:
            bits = cls._extract_bits(bitstring, scope)
            return bits in allowed

        return cls(
            name=label,
            description=description,
            validator=validator,
            metadata={"kind": "allowed_bitstrings", "bitstrings": allowed, "qubits": scope},
        )

    @classmethod
    def custom(
        cls,
        name: str,
        validator: Callable[[str], bool],
        description: str = "Custom symmetry filter",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> 'SymmetryDescriptor':
        return cls(
            name=name,
            description=description,
            validator=validator,
            metadata=metadata or {"kind": "custom"},
        )


class SymmetryFilter:
    """Apply symmetry constraints to sampled measurement counts."""

    @staticmethod
    def apply(
        counts: Dict[str, int],
        symmetries: List[SymmetryDescriptor],
    ) -> Tuple[Dict[str, int], Optional[dict]]:
        if not symmetries:
            return counts, None

        filtered: Dict[str, int] = {}
        rejected: Dict[str, int] = {}
        rule_rejections = {symmetry.name: 0 for symmetry in symmetries}
        rejected_examples = []

        for bitstring, count in counts.items():
            violated = [
                symmetry.name
                for symmetry in symmetries
                if not symmetry.validate(bitstring)
            ]
            if violated:
                rejected[bitstring] = rejected.get(bitstring, 0) + count
                for name in violated:
                    rule_rejections[name] += count
                if len(rejected_examples) < 5:
                    rejected_examples.append({
                        "bitstring": bitstring,
                        "count": count,
                        "violated": violated,
                    })
                continue
            filtered[bitstring] = filtered.get(bitstring, 0) + count

        kept_shots = sum(filtered.values())
        rejected_shots = sum(rejected.values())
        if kept_shots == 0:
            raise RuntimeError(
                "All measurement shots violated the circuit's symmetry constraints."
            )

        report = {
            "applied": [symmetry.to_dict() for symmetry in symmetries],
            "input_shots": kept_shots + rejected_shots,
            "kept_shots": kept_shots,
            "rejected_shots": rejected_shots,
            "rejection_rate": rejected_shots / (kept_shots + rejected_shots),
            "rule_rejections": rule_rejections,
            "rejected_examples": rejected_examples,
        }

        filtered = dict(sorted(filtered.items(), key=lambda item: -item[1]))
        return filtered, report


class QuantumCircuit:
    """
    Quantum circuit: an ordered sequence of gate operations.
    This is the program that runs on the vQPU.
    """

    def __init__(self, n_qubits: int, name: str = "circuit"):
        self.n_qubits = n_qubits
        self.name = name
        self.ops: List[GateOp] = []
        self.symmetries: List[SymmetryDescriptor] = []
        self.lib = GateLibrary()

    # --- Single-qubit gate shortcuts ---
    def h(self, target: int):
        self.ops.append(GateOp("H", self.lib.H, [target]))
        return self

    def x(self, target: int):
        self.ops.append(GateOp("X", self.lib.X, [target]))
        return self

    def y(self, target: int):
        self.ops.append(GateOp("Y", self.lib.Y, [target]))
        return self

    def z(self, target: int):
        self.ops.append(GateOp("Z", self.lib.Z, [target]))
        return self

    def s(self, target: int):
        self.ops.append(GateOp("S", self.lib.S, [target]))
        return self

    def t(self, target: int):
        self.ops.append(GateOp("T", self.lib.T, [target]))
        return self

    def rx(self, target: int, theta: float):
        self.ops.append(GateOp(f"Rx({theta:.2f})", self.lib.Rx(theta), [target], [theta]))
        return self

    def ry(self, target: int, theta: float):
        self.ops.append(GateOp(f"Ry({theta:.2f})", self.lib.Ry(theta), [target], [theta]))
        return self

    def rz(self, target: int, theta: float):
        self.ops.append(GateOp(f"Rz({theta:.2f})", self.lib.Rz(theta), [target], [theta]))
        return self

    # --- Two-qubit gate shortcuts ---
    def cnot(self, control: int, target: int):
        self.ops.append(GateOp("CNOT", self.lib.CNOT, [control, target], is_two_qubit=True))
        return self

    def cz(self, control: int, target: int):
        self.ops.append(GateOp("CZ", self.lib.CZ, [control, target], is_two_qubit=True))
        return self

    def swap(self, q1: int, q2: int):
        self.ops.append(GateOp("SWAP", self.lib.SWAP, [q1, q2], is_two_qubit=True))
        return self

    # --- Symmetry annotations ---
    def with_symmetry(self, symmetry: SymmetryDescriptor):
        self.symmetries.append(symmetry)
        return self

    def require_hamming_weight(
        self,
        weight: int,
        qubits: Optional[List[int]] = None,
        name: Optional[str] = None,
    ):
        return self.with_symmetry(
            SymmetryDescriptor.fixed_hamming_weight(weight, qubits=qubits, name=name)
        )

    def require_parity(
        self,
        parity: str,
        qubits: Optional[List[int]] = None,
        name: Optional[str] = None,
    ):
        return self.with_symmetry(
            SymmetryDescriptor.parity(parity, qubits=qubits, name=name)
        )

    def require_allowed_bitstrings(
        self,
        bitstrings: List[str],
        qubits: Optional[List[int]] = None,
        name: Optional[str] = None,
    ):
        return self.with_symmetry(
            SymmetryDescriptor.allowed_bitstrings(bitstrings, qubits=qubits, name=name)
        )

    def symmetry_descriptors(self) -> List[dict]:
        return [symmetry.to_dict() for symmetry in self.symmetries]

    # --- Circuit properties ---
    def depth(self) -> int:
        """Circuit depth (longest path through the circuit)"""
        layers = [0] * self.n_qubits
        for op in self.ops:
            max_layer = max(layers[t] for t in op.targets) + 1
            for t in op.targets:
                layers[t] = max_layer
        return max(layers) if layers else 0

    def gate_count(self) -> dict:
        counts = {}
        for op in self.ops:
            counts[op.gate_name] = counts.get(op.gate_name, 0) + 1
        return counts

    def __repr__(self):
        symmetry_part = f", {len(self.symmetries)} symmetries" if self.symmetries else ""
        return (f"QuantumCircuit('{self.name}', {self.n_qubits} qubits, "
                f"{len(self.ops)} gates, depth {self.depth()}{symmetry_part})")


# ═══════════════════════════════════════════════════════════
#  LAYER 4: BACKEND ADAPTER
#  Swappable execution backend — classical now, QPU later
# ═══════════════════════════════════════════════════════════

class Backend(ABC):
    """Abstract backend — same interface for classical sim and real QPU"""

    @abstractmethod
    def execute(self, circuit: QuantumCircuit,
                initial_state: Optional[QuantumRegister] = None,
                shots: int = 1024) -> 'ExecutionResult':
        pass

    @abstractmethod
    def name(self) -> str:
        pass


@dataclass
class ExecutionResult:
    """Result of executing a quantum circuit"""
    counts: dict                    # Measurement outcomes → counts
    statevector: Optional[np.ndarray]  # Final state (simulator only)
    execution_time: float
    backend_name: str
    circuit_name: str
    n_qubits: int
    gate_count: int
    circuit_depth: int
    entanglement_pairs: List[Tuple[int, int]]
    entropy: float
    symmetry_report: Optional[dict] = None
    execution_metadata: Optional[dict] = None

    def most_probable(self) -> str:
        return max(self.counts, key=self.counts.get)

    def probabilities(self) -> dict:
        total = sum(self.counts.values())
        return {k: v / total for k, v in self.counts.items()}


class ClassicalSimulatorBackend(Backend):
    """
    Exact statevector simulator on classical CPU.
    Faithfully simulates quantum mechanics — no approximation.
    Limited to ~28 qubits by memory (2^28 complex128 = 4 GB).
    """

    MAX_QUBITS = 28

    def __init__(self, seed: Optional[int] = None):
        self.seed = seed

    def name(self) -> str:
        return "vQPU::ClassicalSimulator"

    def execute(self, circuit: QuantumCircuit,
                initial_state: Optional[QuantumRegister] = None,
                shots: int = 1024) -> ExecutionResult:

        if circuit.n_qubits > self.MAX_QUBITS:
            raise ValueError(
                f"Classical backend limited to {self.MAX_QUBITS} qubits. "
                f"Circuit requires {circuit.n_qubits}. Use QPU backend."
            )

        t0 = time.time()

        # Initialize register
        if initial_state is not None:
            reg = QuantumRegister(
                initial_state.n_qubits,
                initial_state.amplitudes.copy(),
                dict(initial_state._entanglement_map)
            )
        else:
            reg = QuantumRegister.zeros(circuit.n_qubits)

        # Execute gates
        engine = GateEngine()
        for op in circuit.ops:
            if len(op.targets) > 2:
                engine.apply_multi(reg, op.gate_matrix,
                                   op.targets, op.gate_name)
            elif op.is_two_qubit:
                engine.apply_two_qubit(reg, op.gate_matrix,
                                       op.targets[0], op.targets[1], op.gate_name)
            else:
                engine.apply_single(reg, op.gate_matrix,
                                    op.targets[0], op.gate_name)

        # Measure
        tap = MeasurementTap(seed=self.seed)
        counts = tap.sample_distribution(reg, shots=shots)
        counts, symmetry_report = SymmetryFilter.apply(counts, circuit.symmetries)

        dt = time.time() - t0

        return ExecutionResult(
            counts=counts,
            statevector=reg.amplitudes.copy(),
            execution_time=dt,
            backend_name=self.name(),
            circuit_name=circuit.name,
            n_qubits=circuit.n_qubits,
            gate_count=len(circuit.ops),
            circuit_depth=circuit.depth(),
            entanglement_pairs=reg.entanglement_pairs(),
            entropy=reg.entropy(),
            symmetry_report=symmetry_report,
        )


class QPUBackendStub(Backend):
    """
    Placeholder for real quantum hardware.
    Same API — swap in IBM Quantum, IonQ, etc.
    """

    def __init__(self, provider: str = "ibm_quantum"):
        self.provider = provider

    def name(self) -> str:
        return f"vQPU::QPU({self.provider})"

    def execute(self, circuit: QuantumCircuit,
                initial_state: Optional[QuantumRegister] = None,
                shots: int = 1024) -> ExecutionResult:
        raise NotImplementedError(
            f"Real QPU backend ({self.provider}) not connected. "
            f"Circuit is ready — {circuit.n_qubits} qubits, "
            f"{len(circuit.ops)} gates, depth {circuit.depth()}. "
            f"Swap in your hardware credentials to execute."
        )


# ═══════════════════════════════════════════════════════════
#  LAYER 5: vQPU — THE UNIFIED INTERFACE
#  One object to build, execute, and analyze quantum programs
# ═══════════════════════════════════════════════════════════

class vQPU:
    """
    Virtual Quantum Processing Unit.
    
    Usage:
        qpu = vQPU(backend="simulator")
        circuit = qpu.circuit(4, "bell_test")
        circuit.h(0).cnot(0, 1).cnot(0, 2).cnot(0, 3)
        result = qpu.run(circuit, shots=1024)
    """

    def __init__(self, backend: str = "simulator", seed: Optional[int] = None):
        if backend == "simulator":
            self._backend = ClassicalSimulatorBackend(seed=seed)
        elif backend == "phantom":
            from .phantom import PhantomSimulatorBackend
            self._backend = PhantomSimulatorBackend(seed=seed)
        elif backend.startswith("qpu:"):
            provider = backend.split(":", 1)[1]
            self._backend = QPUBackendStub(provider)
        else:
            raise ValueError(f"Unknown backend: {backend}")

        self._results_history: List[ExecutionResult] = []

    @property
    def backend_name(self) -> str:
        return self._backend.name()

    def circuit(self, n_qubits: int, name: str = "circuit") -> QuantumCircuit:
        """Create a new quantum circuit"""
        return QuantumCircuit(n_qubits, name)

    def run(self, circuit: QuantumCircuit,
            initial_state: Optional[QuantumRegister] = None,
            shots: int = 1024) -> ExecutionResult:
        """Execute a circuit and return results"""
        result = self._backend.execute(circuit, initial_state, shots)
        self._results_history.append(result)
        return result

    def state(self, n_qubits: int, bits: Optional[List[int]] = None) -> QuantumRegister:
        """Create a quantum register"""
        if bits:
            return QuantumRegister.from_classical(bits)
        return QuantumRegister.zeros(n_qubits)

    def history(self) -> List[ExecutionResult]:
        return self._results_history


# ═══════════════════════════════════════════════════════════
#  BUILT-IN QUANTUM ALGORITHMS
#  Pre-built circuits for common quantum algorithms
# ═══════════════════════════════════════════════════════════

class QuantumAlgorithms:
    """Library of standard quantum algorithms"""

    @staticmethod
    def bell_pair(qpu: vQPU) -> QuantumCircuit:
        """Create Bell state: (|00⟩ + |11⟩) / √2"""
        c = qpu.circuit(2, "bell_pair")
        c.h(0).cnot(0, 1)
        return c

    @staticmethod
    def ghz_state(qpu: vQPU, n: int) -> QuantumCircuit:
        """Create GHZ state: (|00...0⟩ + |11...1⟩) / √2"""
        c = qpu.circuit(n, f"ghz_{n}")
        c.h(0)
        for i in range(1, n):
            c.cnot(0, i)
        return c

    @staticmethod
    def quantum_fourier_transform(qpu: vQPU, n: int) -> QuantumCircuit:
        """Quantum Fourier Transform on n qubits"""
        c = qpu.circuit(n, f"qft_{n}")
        for i in range(n):
            c.h(i)
            for j in range(i + 1, n):
                angle = np.pi / (2 ** (j - i))
                # Controlled phase rotation
                c.ops.append(GateOp(
                    f"CP({angle:.3f})",
                    np.array([
                        [1, 0, 0, 0],
                        [0, 1, 0, 0],
                        [0, 0, 1, 0],
                        [0, 0, 0, np.exp(1j * angle)]
                    ], dtype=complex),
                    [j, i], [angle], is_two_qubit=True
                ))
        # Swap qubits for bit-reversal
        for i in range(n // 2):
            c.swap(i, n - 1 - i)
        return c

    @staticmethod
    def grovers_search(qpu: vQPU, n: int, target: int) -> QuantumCircuit:
        """
        Grover's algorithm: search for |target⟩ among 2^n states.
        Uses direct oracle matrix (exact for small n).
        Optimal iterations: ~π/4 × √(2^n)
        """
        c = qpu.circuit(n, f"grover_search_{target}")
        iterations = max(1, int(np.pi / 4 * np.sqrt(2**n)))
        dim = 2 ** n

        # Build oracle matrix: I - 2|target⟩⟨target|
        oracle = np.eye(dim, dtype=complex)
        oracle[target, target] = -1.0

        # Build diffusion matrix: 2|s⟩⟨s| - I where |s⟩ = H^n|0⟩
        diffusion = np.full((dim, dim), 2.0 / dim, dtype=complex) - np.eye(dim, dtype=complex)

        # Initial superposition
        for i in range(n):
            c.h(i)

        # Grover iterations (as full unitary ops)
        for _ in range(iterations):
            c.ops.append(GateOp("Oracle", oracle, list(range(n)), is_two_qubit=(n > 1)))
            c.ops.append(GateOp("Diffuse", diffusion, list(range(n)), is_two_qubit=(n > 1)))

        return c

    @staticmethod
    def variational_ansatz(qpu: vQPU, n: int, params: List[float],
                           layers: int = 2) -> QuantumCircuit:
        """
        Parameterized variational circuit for VQE/QAOA.
        Hardware-efficient ansatz with Ry rotations + CNOT entanglement.
        """
        c = qpu.circuit(n, f"variational_{layers}L")
        p_idx = 0
        for layer in range(layers):
            # Rotation layer
            for i in range(n):
                if p_idx < len(params):
                    c.ry(i, params[p_idx])
                    p_idx += 1
            # Entanglement layer (linear connectivity)
            for i in range(n - 1):
                c.cnot(i, i + 1)
        return c


# ═══════════════════════════════════════════════════════════
#  TEST SUITE — Prove the architecture works
# ═══════════════════════════════════════════════════════════

def run_tests():
    """Comprehensive tests of the vQPU architecture"""

    results = {}
    qpu = vQPU(backend="simulator", seed=42)

    print("╔" + "═"*58 + "╗")
    print("║  vQPU — Virtual Quantum Processing Unit                 ║")
    print("║  Architecture Validation Suite                          ║")
    print("╚" + "═"*58 + "╝")
    print(f"\n  Backend: {qpu.backend_name}")

    # ── TEST 1: Bell State ──
    print("\n" + "─"*60)
    print("  TEST 1: Bell state — Entanglement")
    print("─"*60)
    bell = QuantumAlgorithms.bell_pair(qpu)
    r = qpu.run(bell, shots=4096)
    print(f"  Circuit: {bell}")
    print(f"  Results: {r.counts}")
    print(f"  Entangled pairs: {r.entanglement_pairs}")
    print(f"  Entropy: {r.entropy:.4f} bits")

    # Verify: should get ~50% |00⟩ and ~50% |11⟩
    probs = r.probabilities()
    p00 = probs.get("00", 0)
    p11 = probs.get("11", 0)
    p01 = probs.get("01", 0)
    p10 = probs.get("10", 0)
    bell_ok = (abs(p00 - 0.5) < 0.05 and abs(p11 - 0.5) < 0.05
               and p01 < 0.02 and p10 < 0.02)
    print(f"  P(00)={p00:.3f} P(11)={p11:.3f} P(01)={p01:.3f} P(10)={p10:.3f}")
    print(f"  Bell state valid: {'PASS' if bell_ok else 'FAIL'}")
    results["bell_state"] = {"pass": bell_ok, "p00": p00, "p11": p11, "entropy": r.entropy}

    # ── TEST 2: GHZ State ──
    print("\n" + "─"*60)
    print("  TEST 2: GHZ state — Multi-qubit entanglement")
    print("─"*60)
    for n in [3, 5, 7]:
        ghz = QuantumAlgorithms.ghz_state(qpu, n)
        r = qpu.run(ghz, shots=4096)
        probs = r.probabilities()
        p_all0 = probs.get("0" * n, 0)
        p_all1 = probs.get("1" * n, 0)
        noise = 1.0 - p_all0 - p_all1
        ghz_ok = abs(p_all0 - 0.5) < 0.05 and abs(p_all1 - 0.5) < 0.05
        print(f"  GHZ-{n}: P(|{'0'*n}⟩)={p_all0:.3f} P(|{'1'*n}⟩)={p_all1:.3f} "
              f"noise={noise:.4f} entangled_pairs={len(r.entanglement_pairs)} "
              f"{'PASS' if ghz_ok else 'FAIL'}")
    results["ghz_state"] = {"pass": ghz_ok, "sizes_tested": [3, 5, 7]}

    # ── TEST 3: Quantum Fourier Transform ──
    print("\n" + "─"*60)
    print("  TEST 3: Quantum Fourier Transform")
    print("─"*60)
    qft = QuantumAlgorithms.quantum_fourier_transform(qpu, 4)
    # Feed in |1⟩ state (binary: 0001)
    init = QuantumRegister.from_classical([0, 0, 0, 1])
    r = qpu.run(qft, initial_state=init, shots=4096)
    print(f"  Circuit: {qft}")
    print(f"  Input: |0001⟩")
    print(f"  Output distribution (top 5):")
    for state, count in list(r.counts.items())[:5]:
        print(f"    |{state}⟩: {count/4096:.3f}")
    # QFT of |1⟩ should give roughly uniform distribution with phases
    n_nonzero = sum(1 for c in r.counts.values() if c > 10)
    qft_ok = n_nonzero >= 12  # Most of 16 states should have probability
    print(f"  States with P>0.01: {n_nonzero}/16 {'PASS' if qft_ok else 'FAIL'}")
    results["qft"] = {"pass": qft_ok, "states_populated": n_nonzero}

    # ── TEST 4: Grover's Search ──
    print("\n" + "─"*60)
    print("  TEST 4: Grover's search — Quantum speedup")
    print("─"*60)
    target = 5  # Search for |101⟩ among 8 states
    grover = QuantumAlgorithms.grovers_search(qpu, 3, target)
    r = qpu.run(grover, shots=4096)
    target_bits = format(target, '03b')
    target_prob = r.counts.get(target_bits, 0) / 4096
    print(f"  Searching for |{target_bits}⟩ in 2³=8 states")
    print(f"  Circuit: {grover}")
    print(f"  P(target) = {target_prob:.3f} (classical random: {1/8:.3f})")
    print(f"  Top results:")
    for state, count in list(r.counts.items())[:4]:
        print(f"    |{state}⟩: {count/4096:.3f}")
    grover_ok = target_prob > 0.3  # Should be amplified well above 1/8
    print(f"  Amplification: {target_prob / (1/8):.1f}x {'PASS' if grover_ok else 'FAIL'}")
    results["grover"] = {"pass": grover_ok, "target_prob": target_prob, "amplification": target_prob / (1/8)}

    # ── TEST 5: Unitarity Verification ──
    print("\n" + "─"*60)
    print("  TEST 5: Unitarity — All gates are reversible")
    print("─"*60)
    lib = GateLibrary()
    gates_to_test = {
        "H": lib.H, "X": lib.X, "Y": lib.Y, "Z": lib.Z,
        "S": lib.S, "T": lib.T, "Rx(π/4)": lib.Rx(np.pi/4),
        "Ry(π/3)": lib.Ry(np.pi/3), "Rz(π/6)": lib.Rz(np.pi/6),
        "CNOT": lib.CNOT, "CZ": lib.CZ, "SWAP": lib.SWAP,
    }
    all_unitary = True
    for name, gate in gates_to_test.items():
        try:
            lib.verify_unitary(gate, name)
            print(f"  {name:12s} — G†G = I  ✓")
        except ValueError:
            print(f"  {name:12s} — NOT UNITARY  ✗")
            all_unitary = False
    results["unitarity"] = {"pass": all_unitary}

    # ── TEST 6: Scaling benchmark ──
    print("\n" + "─"*60)
    print("  TEST 6: Scaling — How big can we go?")
    print("─"*60)
    for n in [4, 8, 10, 12, 14]:
        ghz = QuantumAlgorithms.ghz_state(qpu, n)
        r = qpu.run(ghz, shots=1024)
        mem_mb = (2**n * 16) / (1024 * 1024)  # complex128 = 16 bytes
        print(f"  {n:2d} qubits | 2^{n:2d} = {2**n:>8d} amplitudes | "
              f"{mem_mb:>8.2f} MB | {r.execution_time*1000:>8.1f} ms | "
              f"depth {r.circuit_depth}")
    results["scaling"] = {"max_tested": 14}

    # ── Summary ──
    print("\n" + "═"*60)
    all_pass = all(v.get("pass", True) for v in results.values())
    print(f"  ALL TESTS {'PASSED' if all_pass else 'SOME FAILED'}")
    print("═"*60)

    return results


if __name__ == "__main__":
    results = run_tests()

    # Save results
    serializable = {}
    for k, v in results.items():
        serializable[k] = {kk: (float(vv) if isinstance(vv, (np.floating, float)) else vv)
                           for kk, vv in v.items()}

    with open("/home/claude/vqpu_test_results.json", "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\n  Results saved to /home/claude/vqpu_test_results.json")
