"""
Phantom Step 3 + Step 4.

This module adds an initial sparse/pruned simulator and an exact factorized
partitioner for disconnected entanglement regions. The classical region is
represented as bond-dimension-1 product states, which is the simplest useful
special case of an MPS-style region.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import time

import numpy as np

from .core import (
    Backend,
    ExecutionResult,
    GateEngine,
    QuantumCircuit,
    QuantumRegister,
    SymmetryFilter,
)
from .universal import EntanglementScanResult, EntanglementScanner


@dataclass
class PhantomPruningConfig:
    """Configuration for active-set pruning inside the quantum core."""
    amplitude_threshold: float = 1e-8
    max_active_states: Optional[int] = None
    numerical_tolerance: float = 1e-15
    bond_dim: int = 32

    def to_dict(self) -> dict:
        return {
            "amplitude_threshold": self.amplitude_threshold,
            "max_active_states": self.max_active_states,
            "numerical_tolerance": self.numerical_tolerance,
            "bond_dim": self.bond_dim,
        }


@dataclass
class PhantomSubsystemPlan:
    """One subsystem in the Phantom partition."""
    qubits: List[int]
    role: str
    representation: str
    gate_count: int
    estimated_dense_bytes: int
    estimated_factorized_bytes: int

    def to_dict(self) -> dict:
        return {
            "qubits": self.qubits,
            "role": self.role,
            "representation": self.representation,
            "gate_count": self.gate_count,
            "estimated_dense_bytes": self.estimated_dense_bytes,
            "estimated_factorized_bytes": self.estimated_factorized_bytes,
        }


@dataclass
class PhantomBridgeTransfer:
    """Candidate bridge transfer around an articulation point."""
    bridge_qubits: List[int]
    fragment_qubits: List[int]
    bond_dim: int
    transfer_complex_values: int
    transfer_bytes: int
    description: str

    def to_dict(self) -> dict:
        return {
            "bridge_qubits": self.bridge_qubits,
            "fragment_qubits": self.fragment_qubits,
            "bond_dim": self.bond_dim,
            "transfer_complex_values": self.transfer_complex_values,
            "transfer_bytes": self.transfer_bytes,
            "description": self.description,
        }


@dataclass
class PhantomPartition:
    """Initial Phantom partition built from the entanglement scan."""
    scan_result: EntanglementScanResult
    pruning: PhantomPruningConfig
    core_subsystems: List[PhantomSubsystemPlan]
    classical_subsystems: List[PhantomSubsystemPlan]
    candidate_bridges: List[PhantomBridgeTransfer]
    estimated_dense_bytes: int
    estimated_factorized_bytes: int

    def to_dict(self) -> dict:
        return {
            "estimated_dense_bytes": self.estimated_dense_bytes,
            "estimated_factorized_bytes": self.estimated_factorized_bytes,
            "estimated_compression_ratio": (
                self.estimated_dense_bytes / self.estimated_factorized_bytes
                if self.estimated_factorized_bytes else float("inf")
            ),
            "pruning": self.pruning.to_dict(),
            "core_subsystems": [subsystem.to_dict() for subsystem in self.core_subsystems],
            "classical_subsystems": [
                subsystem.to_dict() for subsystem in self.classical_subsystems
            ],
            "candidate_bridges": [
                bridge.to_dict() for bridge in self.candidate_bridges
            ],
            "scan": self.scan_result.to_dict(),
        }


def _gate_count_for_qubits(circuit: QuantumCircuit, qubits: List[int]) -> int:
    qubit_set = set(qubits)
    return sum(1 for op in circuit.ops if set(op.targets).issubset(qubit_set))


def _subgraph_fragments(
    component_qubits: List[int],
    removed_qubits: List[int],
    scan_result: EntanglementScanResult,
) -> List[List[int]]:
    removed = set(removed_qubits)
    component_set = set(component_qubits)
    adjacency = {
        qubit: set()
        for qubit in component_qubits
        if qubit not in removed
    }
    for edge in scan_result.edges:
        a, b = edge.qubits
        if (
            a in component_set
            and b in component_set
            and a not in removed
            and b not in removed
        ):
            adjacency[a].add(b)
            adjacency[b].add(a)

    fragments = []
    visited = set()
    for root in sorted(adjacency):
        if root in visited:
            continue
        stack = [root]
        fragment = []
        while stack:
            node = stack.pop()
            if node in visited:
                continue
            visited.add(node)
            fragment.append(node)
            stack.extend(neighbor for neighbor in adjacency[node] if neighbor not in visited)
        fragments.append(sorted(fragment))

    return fragments


def _build_candidate_bridges(
    scan_result: EntanglementScanResult,
    bond_dim: int,
) -> List[PhantomBridgeTransfer]:
    adjacency = {qubit: set() for qubit in range(scan_result.n_qubits)}
    for edge in scan_result.edges:
        a, b = edge.qubits
        adjacency[a].add(b)
        adjacency[b].add(a)

    bridges: List[PhantomBridgeTransfer] = []
    for component in scan_result.components:
        component_set = set(component.qubits)
        local_bridge_qubits = [
            qubit for qubit in scan_result.bridge_qubits if qubit in component_set
        ]
        for bridge_qubit in local_bridge_qubits:
            visited = set()
            for root in sorted(component_set - {bridge_qubit}):
                if root in visited:
                    continue
                stack = [root]
                fragment = []
                while stack:
                    node = stack.pop()
                    if node in visited or node == bridge_qubit:
                        continue
                    visited.add(node)
                    fragment.append(node)
                    stack.extend(
                        neighbor
                        for neighbor in adjacency[node]
                        if neighbor in component_set
                        and neighbor != bridge_qubit
                        and neighbor not in visited
                    )
                if not fragment:
                    continue

                transfer_complex_values = (2 ** 1) * bond_dim
                bridges.append(
                    PhantomBridgeTransfer(
                        bridge_qubits=[bridge_qubit],
                        fragment_qubits=sorted(fragment),
                        bond_dim=bond_dim,
                        transfer_complex_values=transfer_complex_values,
                        transfer_bytes=transfer_complex_values * 16,
                        description=(
                            f"Candidate Schmidt bridge from fragment {sorted(fragment)} "
                            f"through articulation qubit {bridge_qubit}"
                        ),
                    )
                )

    return bridges


def build_phantom_partition(
    circuit: QuantumCircuit,
    scan_result: Optional[EntanglementScanResult] = None,
    pruning: Optional[PhantomPruningConfig] = None,
) -> PhantomPartition:
    """Build the initial Phantom factorization plan for a circuit."""
    pruning = pruning or PhantomPruningConfig()
    scanner = EntanglementScanner()
    scan_result = scan_result or scanner.scan(circuit)

    core_subsystems: List[PhantomSubsystemPlan] = []
    classical_subsystems: List[PhantomSubsystemPlan] = []

    for component in scan_result.components:
        local_bridge_qubits = sorted(
            qubit for qubit in scan_result.bridge_qubits if qubit in component.qubits
        )
        if not local_bridge_qubits:
            fragment = list(component.qubits)
            dense_bytes = (2 ** len(fragment)) * 16
            if len(fragment) >= 2:
                # Multi-qubit component with no articulation points — prefer
                # MPS with bond_dim > 1. The break-even against dense depends
                # on chi; for small n dense is cheaper, but we route to MPS
                # anyway because the user explicitly asked for bond_dim > 1
                # representation here. Runtime SVDs honor the chi budget and
                # truncate gracefully if the real state needs more.
                chi_cap = max(1, min(pruning.bond_dim, 2 ** (len(fragment) // 2)))
                mps_bytes = 16 * (
                    4 * chi_cap + 2 * chi_cap * chi_cap * max(0, len(fragment) - 2)
                )
                classical_subsystems.append(
                    PhantomSubsystemPlan(
                        qubits=fragment,
                        role="classical_region",
                        representation="mps",
                        gate_count=_gate_count_for_qubits(circuit, fragment),
                        estimated_dense_bytes=dense_bytes,
                        estimated_factorized_bytes=mps_bytes,
                    )
                )
                continue
            core_subsystems.append(
                PhantomSubsystemPlan(
                    qubits=fragment,
                    role="quantum_core",
                    representation="sparse_statevector",
                    gate_count=_gate_count_for_qubits(circuit, fragment),
                    estimated_dense_bytes=dense_bytes,
                    estimated_factorized_bytes=dense_bytes,
                )
            )
            continue

        for bridge_qubit in local_bridge_qubits:
            core_subsystems.append(
                PhantomSubsystemPlan(
                    qubits=[bridge_qubit],
                    role="bridge",
                    representation="sparse_statevector",
                    gate_count=_gate_count_for_qubits(circuit, [bridge_qubit]),
                    estimated_dense_bytes=32,
                    estimated_factorized_bytes=32,
                )
            )

        for fragment in _subgraph_fragments(
            component.qubits,
            local_bridge_qubits,
            scan_result,
        ):
            if len(fragment) == 1:
                classical_subsystems.append(
                    PhantomSubsystemPlan(
                        qubits=fragment,
                        role="classical_region",
                        representation="product_state_mps",
                        gate_count=_gate_count_for_qubits(circuit, fragment),
                        estimated_dense_bytes=32,
                        estimated_factorized_bytes=32,
                    )
                )
            else:
                # Multi-qubit fragment: route to MPS with bond_dim > 1.
                # Bond budget: min(user cap, maximum possible for chain
                # length = 2^⌊k/2⌋).
                dense_bytes = (2 ** len(fragment)) * 16
                chi_cap = max(1, min(pruning.bond_dim, 2 ** (len(fragment) // 2)))
                mps_bytes = 16 * (
                    4 * chi_cap
                    + 2 * chi_cap * chi_cap * max(0, len(fragment) - 2)
                )
                classical_subsystems.append(
                    PhantomSubsystemPlan(
                        qubits=fragment,
                        role="classical_region",
                        representation="mps",
                        gate_count=_gate_count_for_qubits(circuit, fragment),
                        estimated_dense_bytes=dense_bytes,
                        estimated_factorized_bytes=mps_bytes,
                    )
                )

    covered_qubits = {
        qubit
        for subsystem in core_subsystems + classical_subsystems
        for qubit in subsystem.qubits
    }
    for qubit in sorted(set(range(circuit.n_qubits)) - covered_qubits):
        classical_subsystems.append(
            PhantomSubsystemPlan(
                qubits=[qubit],
                role="classical_region",
                representation="product_state_mps",
                gate_count=_gate_count_for_qubits(circuit, [qubit]),
                estimated_dense_bytes=32,
                estimated_factorized_bytes=32,
            )
        )

    dense_bytes = (2 ** circuit.n_qubits) * 16
    factorized_bytes = (
        sum(subsystem.estimated_factorized_bytes for subsystem in core_subsystems)
        + sum(subsystem.estimated_factorized_bytes for subsystem in classical_subsystems)
    )
    candidate_bridges = _build_candidate_bridges(scan_result, pruning.bond_dim)

    return PhantomPartition(
        scan_result=scan_result,
        pruning=pruning,
        core_subsystems=core_subsystems,
        classical_subsystems=classical_subsystems,
        candidate_bridges=candidate_bridges,
        estimated_dense_bytes=dense_bytes,
        estimated_factorized_bytes=factorized_bytes,
    )


@dataclass
class _SparseAmplitudeState:
    """Sparse active-set statevector for one quantum-core subsystem."""
    n_qubits: int
    amplitudes: Dict[int, complex]
    peak_active_states: int = 0
    pruned_probability: float = 0.0
    fidelity_lower_bound: float = 1.0
    pruning_events: int = 0

    @classmethod
    def zero(cls, n_qubits: int) -> "_SparseAmplitudeState":
        return cls(n_qubits=n_qubits, amplitudes={0: 1.0 + 0j}, peak_active_states=1)

    @classmethod
    def from_dense(
        cls,
        amplitudes: np.ndarray,
        numerical_tolerance: float = 1e-15,
    ) -> "_SparseAmplitudeState":
        sparse = {
            idx: complex(amp)
            for idx, amp in enumerate(amplitudes)
            if abs(amp) > numerical_tolerance
        }
        if not sparse:
            sparse = {0: 1.0 + 0j}
        return cls(
            n_qubits=int(np.log2(len(amplitudes))),
            amplitudes=sparse,
            peak_active_states=len(sparse),
        )

    def to_dense(self) -> np.ndarray:
        dense = np.zeros(2 ** self.n_qubits, dtype=complex)
        for idx, amp in self.amplitudes.items():
            dense[idx] = amp
        return dense

    def probabilities(self) -> Dict[int, float]:
        return {
            idx: float(abs(amp) ** 2)
            for idx, amp in self.amplitudes.items()
        }

    def entropy(self) -> float:
        probs = np.array(list(self.probabilities().values()), dtype=float)
        probs = probs[probs > 1e-15]
        if len(probs) == 0:
            return 0.0
        return float(-np.sum(probs * np.log2(probs)))

    def _record_peak(self) -> None:
        self.peak_active_states = max(self.peak_active_states, len(self.amplitudes))

    def prune(self, config: PhantomPruningConfig) -> None:
        original_items = list(self.amplitudes.items())
        if not original_items:
            self.amplitudes = {0: 1.0 + 0j}
            self._record_peak()
            return

        kept = {
            idx: amp
            for idx, amp in original_items
            if abs(amp) >= config.amplitude_threshold
        }
        dropped_probability = float(sum(
            abs(amp) ** 2
            for idx, amp in original_items
            if idx not in kept
        ))

        if config.max_active_states is not None and len(kept) > config.max_active_states:
            sorted_items = sorted(
                kept.items(),
                key=lambda item: abs(item[1]),
                reverse=True,
            )
            kept = dict(sorted_items[:config.max_active_states])
            dropped_probability = float(
                1.0 - sum(abs(amp) ** 2 for amp in kept.values())
            )

        if not kept:
            idx, amp = max(original_items, key=lambda item: abs(item[1]))
            kept = {idx: amp}
            dropped_probability = float(
                sum(abs(other_amp) ** 2 for other_idx, other_amp in original_items if other_idx != idx)
            )

        norm = np.sqrt(sum(abs(amp) ** 2 for amp in kept.values()))
        if norm <= config.numerical_tolerance:
            kept = {0: 1.0 + 0j}
            norm = 1.0

        self.amplitudes = {
            idx: amp / norm
            for idx, amp in kept.items()
            if abs(amp) > config.numerical_tolerance
        }
        if not self.amplitudes:
            self.amplitudes = {0: 1.0 + 0j}

        if dropped_probability > config.numerical_tolerance:
            self.pruning_events += 1
            self.pruned_probability += dropped_probability
            self.fidelity_lower_bound *= max(0.0, 1.0 - dropped_probability)
        self._record_peak()

    def apply_single(
        self,
        gate: np.ndarray,
        target: int,
        config: PhantomPruningConfig,
    ) -> None:
        mask = 1 << (self.n_qubits - target - 1)
        pair_bases = {idx & ~mask for idx in self.amplitudes}
        new_amplitudes: Dict[int, complex] = {}

        for base in pair_bases:
            idx0 = base
            idx1 = base | mask
            a0 = self.amplitudes.get(idx0, 0.0 + 0j)
            a1 = self.amplitudes.get(idx1, 0.0 + 0j)
            out0 = gate[0, 0] * a0 + gate[0, 1] * a1
            out1 = gate[1, 0] * a0 + gate[1, 1] * a1
            if abs(out0) > config.numerical_tolerance:
                new_amplitudes[idx0] = out0
            if abs(out1) > config.numerical_tolerance:
                new_amplitudes[idx1] = out1

        self.amplitudes = new_amplitudes
        self._record_peak()
        self.prune(config)

    def apply_two_qubit(
        self,
        gate: np.ndarray,
        control: int,
        target: int,
        config: PhantomPruningConfig,
    ) -> None:
        control_mask = 1 << (self.n_qubits - control - 1)
        target_mask = 1 << (self.n_qubits - target - 1)
        quad_bases = {
            idx & ~control_mask & ~target_mask
            for idx in self.amplitudes
        }
        new_amplitudes: Dict[int, complex] = {}

        for base in quad_bases:
            idxs = [
                base,
                base | target_mask,
                base | control_mask,
                base | control_mask | target_mask,
            ]
            vec = np.array(
                [self.amplitudes.get(idx, 0.0 + 0j) for idx in idxs],
                dtype=complex,
            )
            out = gate @ vec
            for idx, amp in zip(idxs, out):
                if abs(amp) > config.numerical_tolerance:
                    new_amplitudes[idx] = complex(amp)

        self.amplitudes = new_amplitudes
        self._record_peak()
        self.prune(config)

    def apply_generic(
        self,
        gate: np.ndarray,
        targets: List[int],
        config: PhantomPruningConfig,
    ) -> None:
        register = QuantumRegister(self.n_qubits, self.to_dense())
        engine = GateEngine()
        if len(targets) == 1:
            engine.apply_single(register, gate, targets[0], "U")
        elif len(targets) == 2 and gate.shape == (4, 4):
            engine.apply_two_qubit(register, gate, targets[0], targets[1], "U2")
        else:
            engine.apply_multi(register, gate, targets, "Un")

        self.amplitudes = {
            idx: complex(amp)
            for idx, amp in enumerate(register.amplitudes)
            if abs(amp) > config.numerical_tolerance
        }
        self._record_peak()
        self.prune(config)

    def sample_bitstrings(self, shots: int, rng: np.random.Generator) -> List[str]:
        items = sorted(self.amplitudes.items())
        indices = np.array([idx for idx, _ in items], dtype=int)
        probs = np.array([abs(amp) ** 2 for _, amp in items], dtype=float)
        probs = probs / probs.sum()
        sampled = rng.choice(len(indices), size=shots, p=probs)
        return [
            format(int(indices[idx]), f"0{self.n_qubits}b")
            for idx in sampled
        ]


@dataclass
class _ProductStateQubit:
    """Single-qubit product-state classical region (MPS bond dimension 1)."""
    qubit: int
    vector: np.ndarray = field(
        default_factory=lambda: np.array([1.0 + 0j, 0.0 + 0j], dtype=complex)
    )
    gate_count: int = 0

    def apply(self, gate: np.ndarray) -> None:
        self.vector = gate @ self.vector
        norm = np.linalg.norm(self.vector)
        if norm > 1e-15:
            self.vector = self.vector / norm
        self.gate_count += 1

    def entropy(self) -> float:
        probs = np.abs(self.vector) ** 2
        probs = probs[probs > 1e-15]
        if len(probs) == 0:
            return 0.0
        return float(-np.sum(probs * np.log2(probs)))

    def sample_bits(self, shots: int, rng: np.random.Generator) -> List[str]:
        probs = np.abs(self.vector) ** 2
        probs = probs / probs.sum()
        bits = rng.choice(2, size=shots, p=probs)
        return [str(int(bit)) for bit in bits]


# ─────────────────────────────────────────────────────────────────────
#  Matrix Product State — true MPS with bond_dim > 1.
#
#  Represents a k-qubit state as a chain of tensors T[i] of shape
#  (χ_{i-1}, 2, χ_i) with χ_0 = χ_k = 1. Bond dimensions grow via
#  SVDs on two-qubit gates and are capped at `chi_max`; the retained
#  Schmidt weight gives an exact fidelity lower bound.
#
#  This is what a "classical region" upgrades to once we go beyond
#  bond-dim 1 product states. Multi-qubit fragments in the partition
#  land here instead of in the sparse statevector core, with memory
#  O(k · χ² · 2) instead of O(2^k).
# ─────────────────────────────────────────────────────────────────────


def _svd_truncate(
    matrix: np.ndarray,
    chi_max: int,
    numerical_tolerance: float = 1e-14,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """SVD with truncation to `chi_max` singular values, dropping near-zeros.
    Returns (U, S, Vh, retained_fraction). When truncation occurs, S is
    rescaled so Σ s_i² equals the pre-truncation Frobenius total — this
    preserves the state norm across the truncation step. When no truncation
    is needed, S is returned untouched (blind renormalization to ||s||=1
    is wrong unless the MPS is in mixed-canonical form at this bond, which
    we do not maintain in general)."""
    u, s, vh = np.linalg.svd(matrix, full_matrices=False)
    if s.size == 0:
        return u, s, vh, 1.0
    cutoff = max(numerical_tolerance, s[0] * numerical_tolerance)
    mask = s > cutoff
    s = s[mask]
    u = u[:, mask]
    vh = vh[mask, :]
    total = float(np.sum(s * s))
    if len(s) > chi_max and total > 0:
        s = s[:chi_max]
        u = u[:, :chi_max]
        vh = vh[:chi_max, :]
        kept = float(np.sum(s * s))
        if kept > 0:
            # Preserve pre-truncation Frobenius mass — prevents state-norm
            # drift across truncations.
            s = s * np.sqrt(total / kept)
        retained_fraction = kept / total
    else:
        retained_fraction = 1.0
    return u, s, vh, retained_fraction


def _swap_gate_matrix() -> np.ndarray:
    return np.array(
        [[1, 0, 0, 0],
         [0, 0, 1, 0],
         [0, 1, 0, 0],
         [0, 0, 0, 1]],
        dtype=complex,
    )


@dataclass
class _MPSState:
    """Left-canonical Matrix Product State for a multi-qubit classical region.

    `tensors[i]` has shape `(chi_left, 2, chi_right)`. Qubits in `qubit_order`
    are chain sites in the same order — position i in the chain corresponds
    to physical qubit `qubit_order[i]`. Bonds are capped at `chi_max`; every
    truncation multiplies `fidelity_lower_bound` by the retained Schmidt
    weight so we can report an exact fidelity floor after the whole run.
    """
    tensors: List[np.ndarray]
    qubit_order: List[int]
    chi_max: int = 32
    fidelity_lower_bound: float = 1.0
    truncation_events: int = 0
    peak_bond_dim: int = 1
    gate_count: int = 0

    def __post_init__(self) -> None:
        assert len(self.tensors) == len(self.qubit_order)
        assert len(self.tensors) >= 1
        assert self.tensors[0].shape[0] == 1, (
            f"MPS left boundary must be dim 1, got {self.tensors[0].shape}"
        )
        assert self.tensors[-1].shape[2] == 1, (
            f"MPS right boundary must be dim 1, got {self.tensors[-1].shape}"
        )
        for i, t in enumerate(self.tensors):
            assert t.ndim == 3 and t.shape[1] == 2, (
                f"MPS tensor {i} has bad shape {t.shape}"
            )
            if i > 0:
                assert self.tensors[i - 1].shape[2] == t.shape[0], (
                    f"MPS bond mismatch at site {i}"
                )
        self.peak_bond_dim = max(self.peak_bond_dim, self.max_bond_dim())

    # ---- constructors --------------------------------------------------

    @classmethod
    def zeros(cls, qubit_order: List[int], chi_max: int = 32) -> "_MPSState":
        """Ground state |0…0⟩ as an MPS with every bond χ=1."""
        tensors: List[np.ndarray] = []
        for _ in qubit_order:
            t = np.zeros((1, 2, 1), dtype=complex)
            t[0, 0, 0] = 1.0
            tensors.append(t)
        return cls(tensors=tensors, qubit_order=list(qubit_order), chi_max=chi_max)

    @classmethod
    def from_product_qubits(
        cls,
        qubit_vectors: List[Tuple[int, np.ndarray]],
        chi_max: int = 32,
    ) -> "_MPSState":
        """Build an MPS from a list of (qubit_index, 2-vector) pairs, sorted
        ascending by qubit index so chain order = sorted qubit order."""
        ordered = sorted(qubit_vectors, key=lambda pair: pair[0])
        tensors = [np.asarray(v, dtype=complex).reshape(1, 2, 1) for _, v in ordered]
        qubit_order = [q for q, _ in ordered]
        return cls(tensors=tensors, qubit_order=qubit_order, chi_max=chi_max)

    @classmethod
    def from_dense(
        cls,
        dense: np.ndarray,
        qubit_order: List[int],
        chi_max: int = 32,
    ) -> "_MPSState":
        """Factorize a (2^k,) complex vector into an MPS via successive SVDs.
        Truncation at each bond may lose fidelity; tracked in the return."""
        k = len(qubit_order)
        assert dense.shape == (2 ** k,)
        # Normalize defensively.
        norm = float(np.linalg.norm(dense))
        if norm > 0:
            dense = dense / norm
        tensors: List[np.ndarray] = []
        fidelity = 1.0
        truncations = 0
        remainder = dense.reshape(1, 2 ** k)
        chi_left = 1
        for site in range(k - 1):
            # Reshape remainder as (chi_left * 2, 2^(k-site-1)).
            matrix = remainder.reshape(chi_left * 2, 2 ** (k - site - 1))
            u, s, vh, retained = _svd_truncate(matrix, chi_max)
            fidelity *= retained
            if retained < 1.0:
                truncations += 1
            chi_new = len(s)
            tensors.append(u.reshape(chi_left, 2, chi_new))
            remainder = (s[:, np.newaxis] * vh)
            chi_left = chi_new
        # Final tensor: shape (chi_left, 2, 1)
        tensors.append(remainder.reshape(chi_left, 2, 1))
        state = cls(tensors=tensors, qubit_order=list(qubit_order), chi_max=chi_max)
        state.fidelity_lower_bound = fidelity
        state.truncation_events = truncations
        return state

    # ---- introspection -------------------------------------------------

    @property
    def n_qubits(self) -> int:
        return len(self.tensors)

    def bond_dims(self) -> List[int]:
        return [self.tensors[i].shape[2] for i in range(self.n_qubits - 1)]

    def max_bond_dim(self) -> int:
        bonds = self.bond_dims()
        return max(bonds) if bonds else 1

    def memory_bytes(self) -> int:
        return sum(t.nbytes for t in self.tensors)

    def _position_of(self, qubit: int) -> int:
        return self.qubit_order.index(qubit)

    # ---- conversions ---------------------------------------------------

    def to_dense(self) -> np.ndarray:
        """Contract all tensors to produce a (2^n,) complex vector. The
        output is indexed in chain order (MSB = first site)."""
        psi = self.tensors[0]
        for t in self.tensors[1:]:
            psi = np.tensordot(psi, t, axes=([-1], [0]))
        return psi.reshape(-1)

    def probabilities(self) -> Dict[int, float]:
        dense = self.to_dense()
        return {
            int(i): float(abs(a) ** 2)
            for i, a in enumerate(dense)
            if abs(a) ** 2 > 1e-15
        }

    def entropy(self) -> float:
        probs = np.array(list(self.probabilities().values()), dtype=float)
        probs = probs[probs > 1e-15]
        if len(probs) == 0:
            return 0.0
        return float(-np.sum(probs * np.log2(probs)))

    def sample_bitstrings(self, shots: int, rng: np.random.Generator) -> List[str]:
        """Sample from |ψ|² via the materialized dense vector. For k ≤ ~20
        this is fine; larger sizes would want a proper right-canonical
        conditional-sampling sweep, not implemented in this pass."""
        probs = self.probabilities()
        if not probs:
            return ["0" * self.n_qubits] * shots
        keys = list(probs.keys())
        weights = np.array([probs[k] for k in keys], dtype=float)
        total = weights.sum()
        if total <= 0:
            return ["0" * self.n_qubits] * shots
        weights = weights / total
        sampled = rng.choice(len(keys), size=shots, p=weights)
        return [format(keys[i], f"0{self.n_qubits}b") for i in sampled]

    # ---- gate application ---------------------------------------------

    def apply_single(self, gate: np.ndarray, qubit: int) -> None:
        """Apply a 2×2 gate to the given qubit in-place. No bond change."""
        pos = self._position_of(qubit)
        t = self.tensors[pos]
        self.tensors[pos] = np.einsum("ij,ajb->aib", gate, t)
        self.gate_count += 1

    def apply_two_qubit(self, gate: np.ndarray, qubit_a: int, qubit_b: int) -> None:
        """Apply a 4×4 gate to two qubits. If they are adjacent in the chain,
        do one SVD. Otherwise, SWAP one qubit to adjacency, apply, swap back.
        The gate is assumed to be in the convention used elsewhere in this
        module: row/column index = 2 × bit(first qubit) + bit(second qubit).
        """
        pos_a = self._position_of(qubit_a)
        pos_b = self._position_of(qubit_b)
        assert pos_a != pos_b
        if pos_a > pos_b:
            # Reorient the gate so qubit_a is the 'first' qubit in the
            # 4-tensor: swap the two-qubit indices in both input and output.
            g4 = gate.reshape(2, 2, 2, 2).transpose(1, 0, 3, 2)
            gate = g4.reshape(4, 4)
            pos_a, pos_b = pos_b, pos_a
            qubit_a, qubit_b = qubit_b, qubit_a  # noqa: F841 (kept for clarity)

        if pos_b == pos_a + 1:
            self._apply_two_qubit_adjacent(gate, pos_a)
        else:
            swap = _swap_gate_matrix()
            # SWAP is a unitary acting on the state; tensors change but the
            # chain's qubit_order labeling is preserved — the closing SWAPs
            # undo the physical movement. So we apply the bare adjacent
            # two-qubit primitive for every SWAP and never touch qubit_order.
            for p in range(pos_a, pos_b - 1):
                self._apply_two_qubit_adjacent(swap, p)
            self._apply_two_qubit_adjacent(gate, pos_b - 1)
            for p in range(pos_b - 2, pos_a - 1, -1):
                self._apply_two_qubit_adjacent(swap, p)
        self.gate_count += 1

    def _apply_two_qubit_adjacent(self, gate: np.ndarray, position: int) -> None:
        """Apply a 4×4 gate to sites `position` and `position+1` with
        SVD truncation. The left site becomes left-canonical; the right
        site absorbs the singular values."""
        assert 0 <= position < self.n_qubits - 1
        t_left = self.tensors[position]
        t_right = self.tensors[position + 1]
        chi_L = t_left.shape[0]
        chi_R = t_right.shape[2]
        theta = np.tensordot(t_left, t_right, axes=([2], [0]))  # (χL, 2, 2, χR)
        g4 = gate.reshape(2, 2, 2, 2)
        theta = np.einsum("ijkl,aklb->aijb", g4, theta)
        matrix = theta.reshape(chi_L * 2, 2 * chi_R)
        u, s, vh, retained = _svd_truncate(matrix, self.chi_max)
        if retained < 1.0:
            self.truncation_events += 1
        self.fidelity_lower_bound *= float(retained)
        chi_new = len(s)
        new_left = u.reshape(chi_L, 2, chi_new)
        new_right = (s[:, np.newaxis] * vh).reshape(chi_new, 2, chi_R)
        self.tensors[position] = new_left
        self.tensors[position + 1] = new_right
        self.peak_bond_dim = max(self.peak_bond_dim, chi_new)

    # ---- split detection ----------------------------------------------

    def try_split_at_bond(
        self,
        bond_index: int,
        tolerance: float,
    ) -> Optional[Tuple["_MPSState", "_MPSState"]]:
        """If the Schmidt rank at `bond_index` is effectively 1 (largest
        singular value ≥ 1 − tolerance), factor the MPS into two smaller
        MPSs meeting at that bond. Returns (left_mps, right_mps) or None."""
        assert 0 <= bond_index < self.n_qubits - 1
        # Build the Schmidt matrix at that bond by left-canonicalizing the
        # left half: sweep SVDs from site 0 to site `bond_index`.
        left_tensors = [t.copy() for t in self.tensors[: bond_index + 1]]
        right_tensors = [t.copy() for t in self.tensors[bond_index + 1:]]
        # The bond between left_tensors[-1] and right_tensors[0] is at index
        # bond_index. Contract the left chain fully, then SVD at the seam.
        # To get Schmidt values cleanly: contract left into a single matrix
        # of shape (2^(bond_index+1), chi_bond), do QR/SVD on the right side
        # similarly, then combine. Easier path: reshape left chain to a
        # matrix and SVD it.
        left_matrix = left_tensors[0].reshape(1, 2, -1)
        for t in left_tensors[1:]:
            merged = np.tensordot(left_matrix, t, axes=([-1], [0]))
            left_matrix = merged.reshape(
                1, -1, t.shape[2]
            ) if False else merged.reshape(merged.shape[0],
                                           -1,
                                           merged.shape[-1])
        # left_matrix shape: (1, 2^(bond_index+1), chi_bond)
        lm = left_matrix.reshape(left_matrix.shape[1], left_matrix.shape[2])
        # Ditto on the right: contract right_tensors into shape (chi_bond, 2^...)
        right_matrix = right_tensors[-1]
        for t in reversed(right_tensors[:-1]):
            merged = np.tensordot(t, right_matrix, axes=([-1], [0]))
            right_matrix = merged.reshape(merged.shape[0], -1, merged.shape[-1])
        rm = right_matrix.reshape(right_matrix.shape[0], -1)
        # Full matrix M = lm @ rm, shape (2^left, 2^right). Its SVD gives
        # the Schmidt spectrum across this bond.
        full = lm @ rm
        u, s, vh = np.linalg.svd(full, full_matrices=False)
        if s.size == 0:
            return None
        purity = float(s[0] ** 2 / (s * s).sum()) if s.sum() > 0 else 0.0
        if purity < 1.0 - tolerance:
            return None
        # Rank-1 factorization: |ψ⟩ = s[0] · u[:,0] ⊗ vh[0,:].
        left_vec = u[:, 0] * s[0]
        right_vec = vh[0, :]
        # Renormalize each factor independently to unit norm.
        ln = float(np.linalg.norm(left_vec))
        rn = float(np.linalg.norm(right_vec))
        if ln <= 0 or rn <= 0:
            return None
        left_dense = left_vec / ln
        right_dense = right_vec / rn
        left_mps = _MPSState.from_dense(
            left_dense, self.qubit_order[: bond_index + 1], chi_max=self.chi_max
        )
        right_mps = _MPSState.from_dense(
            right_dense,
            self.qubit_order[bond_index + 1:],
            chi_max=self.chi_max,
        )
        return left_mps, right_mps


# ─────────────────────────────────────────────────────────────────────
#  Dynamic re-splitting — Schmidt-rank-1 factorization of qubits out of
#  a sparse subsystem. After every gate, each qubit in the subsystem
#  is tested: compute its reduced density matrix ρ, check purity
#  tr(ρ²) ≥ 1 − ε. If separable, factor it out as a product-state qubit
#  and keep the residual as a smaller sparse core.
# ─────────────────────────────────────────────────────────────────────

def _remove_bit_at(index: int, position_from_msb: int, n_bits_total: int) -> int:
    """Remove the bit at `position_from_msb` (0 = MSB) from an
    `n_bits_total`-wide integer. Returns the (n-1)-bit result."""
    bit_from_lsb = n_bits_total - 1 - position_from_msb
    upper = (index >> (bit_from_lsb + 1)) << bit_from_lsb
    lower = index & ((1 << bit_from_lsb) - 1)
    return upper | lower


def _reduced_density_single(
    amplitudes: Dict[int, complex],
    n_qubits: int,
    position: int,
) -> np.ndarray:
    """2×2 reduced density matrix of the qubit at big-endian `position`.
    Cost: O(|active amplitudes|). Does not materialize the full state."""
    mask = 1 << (n_qubits - 1 - position)
    rho = np.zeros((2, 2), dtype=complex)
    for idx, amp in amplitudes.items():
        i = 1 if idx & mask else 0
        rho[i, i] += (amp.conjugate() * amp).real
        partner_amp = amplitudes.get(idx ^ mask)
        if partner_amp is not None:
            rho[i, 1 - i] += amp.conjugate() * partner_amp
    return rho


def _try_factor_qubit(
    amplitudes: Dict[int, complex],
    n_qubits: int,
    position: int,
    purity_tolerance: float,
) -> Optional[Tuple[np.ndarray, Dict[int, complex], float]]:
    """If qubit at `position` is Schmidt-rank-1 (up to `purity_tolerance`),
    return (child_vector, residual_amplitudes, purity). Else None.

    The residual is renormalized to ⟨ψ|ψ⟩ = 1. Reindexing removes the
    factored-out bit from every basis-state key.
    """
    rho = _reduced_density_single(amplitudes, n_qubits, position)
    purity = float(
        rho[0, 0].real ** 2
        + rho[1, 1].real ** 2
        + 2 * (abs(rho[0, 1]) ** 2)
    )
    if purity < 1.0 - purity_tolerance:
        return None

    # Pull the dominant eigenvector of ρ = |φ⟩⟨φ|.
    _eigvals, eigvecs = np.linalg.eigh(rho)
    phi = eigvecs[:, -1]  # largest eigenvalue sits last in eigh's output
    # Renormalize defensively; eigh returns unit vectors but be safe.
    phi_norm = float(np.linalg.norm(phi))
    if phi_norm <= 0.0:
        return None
    phi = phi / phi_norm

    # Project |ψ⟩ onto |φ⟩ to extract |ξ_rest⟩:  ξ_r = Σ_i φ*_i ψ(q=i, rest=r)
    mask = 1 << (n_qubits - 1 - position)
    residual: Dict[int, complex] = {}
    for idx, amp in amplitudes.items():
        i = 1 if idx & mask else 0
        new_idx = _remove_bit_at(idx, position, n_qubits)
        residual[new_idx] = residual.get(new_idx, 0.0 + 0j) + phi[i].conjugate() * amp

    residual_norm_sq = sum((a.conjugate() * a).real for a in residual.values())
    if residual_norm_sq <= 0.0:
        return None
    scale = 1.0 / np.sqrt(residual_norm_sq)
    residual = {idx: a * scale for idx, a in residual.items()}
    return phi, residual, purity


def _sparse_max_schmidt_rank(
    amplitudes: Dict[int, complex],
    n_qubits: int,
    tolerance: float,
) -> int:
    """Largest Schmidt rank across consecutive bipartitions of a sparse state.
    Used to decide whether a sparse core will fit into a bond-dim-capped MPS.

    Costs O(n × 2^n) for the dense reshape — guarded by callers before use."""
    if n_qubits < 2:
        return 1
    dense = np.zeros(2 ** n_qubits, dtype=complex)
    for idx, amp in amplitudes.items():
        dense[idx] = amp
    max_rank = 1
    for bond in range(n_qubits - 1):
        matrix = dense.reshape(2 ** (bond + 1), 2 ** (n_qubits - bond - 1))
        singular = np.linalg.svd(matrix, compute_uv=False)
        if singular.size == 0:
            continue
        cutoff = tolerance * float(singular[0]) if singular[0] > 0 else 0.0
        rank = int(np.sum(singular > cutoff))
        if rank > max_rank:
            max_rank = rank
    return max_rank


@dataclass
class _PhantomLiveSubsystem:
    """Mutable live subsystem used by the Phantom runtime."""
    qubits: List[int]
    role: str
    representation: str
    sparse_state: Optional[_SparseAmplitudeState] = None
    product_state: Optional[_ProductStateQubit] = None
    mps_state: Optional[_MPSState] = None
    gate_count: int = 0
    merge_count: int = 0
    split_count: int = 0
    # Lifetime peak active-state count, preserved across sparse→MPS demotions
    # and splits so we can honestly report how big the subsystem ever got.
    lifetime_peak_active: int = 0
    source_subsystems: List[List[int]] = field(default_factory=list)

    def sparse_amplitudes(self) -> Dict[int, complex]:
        if self.sparse_state is not None:
            return dict(self.sparse_state.amplitudes)
        if self.mps_state is not None:
            dense = self.mps_state.to_dense()
            return {
                int(idx): complex(amp)
                for idx, amp in enumerate(dense)
                if abs(amp) > 1e-15
            }
        if self.product_state is not None:
            return {
                idx: complex(amp)
                for idx, amp in enumerate(self.product_state.vector)
                if abs(amp) > 1e-15
            }
        return {0: 1.0 + 0j}

    def active_states_final(self) -> int:
        if self.sparse_state is not None:
            return len(self.sparse_state.amplitudes)
        if self.mps_state is not None:
            return len(self.mps_state.probabilities())
        return 2

    def peak_active_states(self) -> int:
        sparse_peak = (
            self.sparse_state.peak_active_states
            if self.sparse_state is not None
            else 0
        )
        return max(sparse_peak, self.lifetime_peak_active, 1)

    def pruned_probability(self) -> float:
        if self.sparse_state is not None:
            return self.sparse_state.pruned_probability
        return 0.0

    def fidelity_lower_bound(self) -> float:
        if self.sparse_state is not None:
            return self.sparse_state.fidelity_lower_bound
        if self.mps_state is not None:
            return self.mps_state.fidelity_lower_bound
        return 1.0

    def pruning_events(self) -> int:
        if self.sparse_state is not None:
            return self.sparse_state.pruning_events
        return 0

    def entropy(self) -> float:
        if self.sparse_state is not None:
            return self.sparse_state.entropy()
        if self.mps_state is not None:
            return self.mps_state.entropy()
        if self.product_state is not None:
            return self.product_state.entropy()
        return 0.0

    def max_bond_dim(self) -> int:
        if self.mps_state is not None:
            return self.mps_state.max_bond_dim()
        if self.product_state is not None:
            return 1
        if self.sparse_state is not None:
            # Worst case — sparse core has no explicit bond structure.
            return 2 ** len(self.qubits)
        return 1

    def memory_bytes(self) -> int:
        if self.mps_state is not None:
            return self.mps_state.memory_bytes()
        if self.product_state is not None:
            return self.product_state.vector.nbytes
        if self.sparse_state is not None:
            return len(self.sparse_state.amplitudes) * 24  # int+complex ≈ 24B
        return 0


class PhantomSimulatorBackend(Backend):
    """
    Initial Phantom backend.

    It executes disconnected entanglement components independently, uses sparse
    active-set simulation on the quantum cores, and keeps isolated qubits as
    exact product states.
    """

    def __init__(
        self,
        seed: Optional[int] = None,
        pruning: Optional[PhantomPruningConfig] = None,
    ):
        self.seed = seed
        self.pruning = pruning or PhantomPruningConfig()
        self.scanner = EntanglementScanner()

    def name(self) -> str:
        return "vQPU::PhantomSimulator"

    def build_partition(
        self,
        circuit: QuantumCircuit,
        initial_state: Optional[QuantumRegister] = None,
    ) -> PhantomPartition:
        if initial_state is None:
            return build_phantom_partition(circuit, pruning=self.pruning)

        # Arbitrary initial states may already be entangled, so fall back to
        # a single sparse quantum-core subsystem spanning the full register.
        scan_result = EntanglementScanResult(
            n_qubits=circuit.n_qubits,
            n_entangling_ops=len([op for op in circuit.ops if len(op.targets) > 1]),
            edges=[],
            weighted_degree={qubit: 0 for qubit in range(circuit.n_qubits)},
            components=[],
            quantum_core_qubits=list(range(circuit.n_qubits)),
            bridge_qubits=[],
            classical_qubits=[],
            isolated_qubits=[],
            heuristic=(
                "Initial state supplied: using a single full-register quantum core "
                "to preserve arbitrary pre-existing entanglement."
            ),
        )
        core = PhantomSubsystemPlan(
            qubits=list(range(circuit.n_qubits)),
            role="quantum_core",
            representation="sparse_statevector",
            gate_count=len(circuit.ops),
            estimated_dense_bytes=(2 ** circuit.n_qubits) * 16,
            estimated_factorized_bytes=(2 ** circuit.n_qubits) * 16,
        )
        return PhantomPartition(
            scan_result=scan_result,
            pruning=self.pruning,
            core_subsystems=[core],
            classical_subsystems=[],
            candidate_bridges=[],
            estimated_dense_bytes=(2 ** circuit.n_qubits) * 16,
            estimated_factorized_bytes=(2 ** circuit.n_qubits) * 16,
        )

    def execute(
        self,
        circuit: QuantumCircuit,
        initial_state: Optional[QuantumRegister] = None,
        shots: int = 1024,
    ) -> ExecutionResult:
        t0 = time.time()
        rng = np.random.default_rng(self.seed)
        partition = self.build_partition(circuit, initial_state=initial_state)

        live_subsystems: List[Optional[_PhantomLiveSubsystem]] = []
        qubit_location: Dict[int, int] = {}
        merge_events = []
        split_events: List[Dict[str, Any]] = []
        # Tolerance for declaring a qubit Schmidt-rank-1. Scaled off the
        # pruning threshold so an aggressive amplitude cutoff doesn't
        # produce false splits. Floor at 1e-12 so float64 noise alone
        # doesn't cross it.
        split_purity_tolerance = max(
            1e-12,
            10.0 * (self.pruning.amplitude_threshold ** 2),
        )

        def register_subsystem(subsystem: _PhantomLiveSubsystem) -> int:
            subsystem_id = len(live_subsystems)
            live_subsystems.append(subsystem)
            for qubit in subsystem.qubits:
                qubit_location[qubit] = subsystem_id
            return subsystem_id

        for subsystem in partition.core_subsystems:
            if (
                initial_state is not None
                and len(partition.core_subsystems) == 1
                and subsystem.qubits == list(range(circuit.n_qubits))
            ):
                state = _SparseAmplitudeState.from_dense(
                    initial_state.amplitudes,
                    numerical_tolerance=self.pruning.numerical_tolerance,
                )
            else:
                state = _SparseAmplitudeState.zero(len(subsystem.qubits))
            register_subsystem(
                _PhantomLiveSubsystem(
                    qubits=list(subsystem.qubits),
                    role=subsystem.role,
                    representation=subsystem.representation,
                    sparse_state=state,
                    gate_count=subsystem.gate_count,
                    source_subsystems=[list(subsystem.qubits)],
                )
            )

        for subsystem in partition.classical_subsystems:
            if subsystem.representation == "mps":
                # Multi-qubit classical region backed by a true MPS.
                live = _PhantomLiveSubsystem(
                    qubits=list(subsystem.qubits),
                    role=subsystem.role,
                    representation=subsystem.representation,
                    mps_state=_MPSState.zeros(
                        list(subsystem.qubits),
                        chi_max=max(1, self.pruning.bond_dim),
                    ),
                    gate_count=subsystem.gate_count,
                    source_subsystems=[list(subsystem.qubits)],
                )
            else:
                # Single-qubit product state (bond-dim 1).
                live = _PhantomLiveSubsystem(
                    qubits=list(subsystem.qubits),
                    role=subsystem.role,
                    representation=subsystem.representation,
                    product_state=_ProductStateQubit(qubit=subsystem.qubits[0]),
                    gate_count=subsystem.gate_count,
                    source_subsystems=[list(subsystem.qubits)],
                )
            register_subsystem(live)

        def merge_sparse_amplitudes(
            subsystem_ids: List[int],
        ) -> Tuple[List[int], Dict[int, complex], List[List[int]], int, int]:
            subsystems = [
                live_subsystems[subsystem_id]
                for subsystem_id in sorted(set(subsystem_ids))
                if live_subsystems[subsystem_id] is not None
            ]
            assert all(subsystem is not None for subsystem in subsystems)
            subsystems = [subsystem for subsystem in subsystems if subsystem is not None]

            merged_qubits = sorted(
                qubit for subsystem in subsystems for qubit in subsystem.qubits
            )
            union_pos = {qubit: idx for idx, qubit in enumerate(merged_qubits)}
            combined = {0: 1.0 + 0j}
            merge_count = 0
            gate_count = 0
            source_subsystems = []

            for subsystem in sorted(subsystems, key=lambda item: item.qubits):
                source_subsystems.extend(subsystem.source_subsystems)
                merge_count += subsystem.merge_count
                gate_count += subsystem.gate_count
                sub_qubits = subsystem.qubits
                sub_items = subsystem.sparse_amplitudes().items()
                local_positions = [union_pos[qubit] for qubit in sub_qubits]
                updated: Dict[int, complex] = {}
                for prefix_idx, prefix_amp in combined.items():
                    for local_idx, local_amp in sub_items:
                        merged_idx = prefix_idx
                        for local_position, union_position in enumerate(local_positions):
                            bit = (
                                local_idx
                                >> (len(sub_qubits) - 1 - local_position)
                            ) & 1
                            bit_mask = 1 << (len(merged_qubits) - 1 - union_position)
                            if bit:
                                merged_idx |= bit_mask
                            else:
                                merged_idx &= ~bit_mask
                        updated[merged_idx] = updated.get(merged_idx, 0.0 + 0j) + (
                            prefix_amp * local_amp
                        )
                combined = updated

            return merged_qubits, combined, source_subsystems, merge_count, gate_count

        def merge_subsystems(subsystem_ids: List[int], gate_name: str) -> int:
            merged_qubits, amplitudes, source_subsystems, prior_merges, prior_gate_count = (
                merge_sparse_amplitudes(subsystem_ids)
            )
            merged_state = _SparseAmplitudeState(
                n_qubits=len(merged_qubits),
                amplitudes=amplitudes,
                peak_active_states=len(amplitudes),
            )
            merged_state.prune(self.pruning)
            new_subsystem = _PhantomLiveSubsystem(
                qubits=merged_qubits,
                role="quantum_core",
                representation="sparse_statevector",
                sparse_state=merged_state,
                gate_count=prior_gate_count,
                merge_count=prior_merges + 1,
                source_subsystems=source_subsystems or [merged_qubits],
            )
            new_id = register_subsystem(new_subsystem)

            for subsystem_id in sorted(set(subsystem_ids)):
                old = live_subsystems[subsystem_id]
                if old is None:
                    continue
                for qubit in old.qubits:
                    qubit_location[qubit] = new_id
                live_subsystems[subsystem_id] = None

            merge_events.append({
                "gate": gate_name,
                "merged_subsystems": [
                    live_subsystems_id
                    for live_subsystems_id in sorted(set(subsystem_ids))
                ],
                "merged_qubits": merged_qubits,
                "active_states_after_merge": len(new_subsystem.sparse_state.amplitudes),
            })
            return new_id

        def try_split_subsystem(subsystem_idx: int, gate_name: str) -> None:
            """After a gate runs, attempt to factor every separable qubit
            out of this subsystem. Repeats until no more splits are found."""
            subsystem = live_subsystems[subsystem_idx]
            if subsystem is None or subsystem.sparse_state is None:
                return
            if len(subsystem.qubits) < 2:
                return

            while True:
                state = subsystem.sparse_state
                if state is None or len(subsystem.qubits) < 2:
                    return
                factored_at = None
                for local_pos in range(len(subsystem.qubits)):
                    factored = _try_factor_qubit(
                        state.amplitudes,
                        state.n_qubits,
                        local_pos,
                        split_purity_tolerance,
                    )
                    if factored is not None:
                        factored_at = (local_pos, factored)
                        break
                if factored_at is None:
                    return
                local_pos, (phi, residual_amps, purity) = factored_at

                child_qubit = subsystem.qubits[local_pos]
                remaining_qubits = [
                    q for q in subsystem.qubits if q != child_qubit
                ]

                # Child: single-qubit product state with vector |φ⟩.
                phi_vec = np.asarray(phi, dtype=complex)
                phi_norm = float(np.linalg.norm(phi_vec))
                if phi_norm > 0:
                    phi_vec = phi_vec / phi_norm
                child_ps = _ProductStateQubit(
                    qubit=child_qubit,
                    vector=phi_vec,
                )
                child_subsystem = _PhantomLiveSubsystem(
                    qubits=[child_qubit],
                    role="classical_region",
                    representation="product_state_mps",
                    product_state=child_ps,
                    gate_count=0,
                    merge_count=0,
                    split_count=0,
                    source_subsystems=[[child_qubit]],
                )
                child_id = register_subsystem(child_subsystem)

                # Parent: replace sparse state with the residual, or
                # collapse to a product state if only one qubit remains.
                subsystem.qubits = remaining_qubits
                subsystem.split_count += 1
                fidelity_multiplier = max(0.0, min(1.0, purity))

                if len(remaining_qubits) == 0:
                    # Degenerate: nothing left in parent. Retire the slot.
                    live_subsystems[subsystem_idx] = None
                    split_events.append({
                        "gate": gate_name,
                        "parent_subsystem": subsystem_idx,
                        "child_subsystem": child_id,
                        "child_qubit": child_qubit,
                        "remaining_qubits": [],
                        "purity": purity,
                        "parent_became": "retired",
                    })
                    return

                if len(remaining_qubits) == 1:
                    sole = remaining_qubits[0]
                    vec = np.array(
                        [
                            residual_amps.get(0, 0.0 + 0j),
                            residual_amps.get(1, 0.0 + 0j),
                        ],
                        dtype=complex,
                    )
                    vec_norm = float(np.linalg.norm(vec))
                    if vec_norm > 0:
                        vec = vec / vec_norm
                    subsystem.sparse_state = None
                    subsystem.product_state = _ProductStateQubit(
                        qubit=sole, vector=vec
                    )
                    subsystem.representation = "product_state_mps"
                    subsystem.role = "classical_region"
                    split_events.append({
                        "gate": gate_name,
                        "parent_subsystem": subsystem_idx,
                        "child_subsystem": child_id,
                        "child_qubit": child_qubit,
                        "remaining_qubits": remaining_qubits,
                        "purity": purity,
                        "parent_became": "product_state_mps",
                    })
                    return

                # General case: parent stays a sparse core with one fewer qubit.
                residual_state = _SparseAmplitudeState(
                    n_qubits=len(remaining_qubits),
                    amplitudes=residual_amps,
                    peak_active_states=max(
                        state.peak_active_states,
                        len(residual_amps),
                    ),
                    pruned_probability=state.pruned_probability,
                    fidelity_lower_bound=(
                        state.fidelity_lower_bound * fidelity_multiplier
                    ),
                    pruning_events=state.pruning_events,
                )
                residual_state.prune(self.pruning)
                subsystem.sparse_state = residual_state
                split_events.append({
                    "gate": gate_name,
                    "parent_subsystem": subsystem_idx,
                    "child_subsystem": child_id,
                    "child_qubit": child_qubit,
                    "remaining_qubits": remaining_qubits,
                    "purity": purity,
                    "parent_became": "sparse_statevector",
                })
                # Loop continues — the residual may still factor further.

        def try_demote_sparse_to_mps(subsystem_idx: int, gate_name: str) -> None:
            """If a sparse-core subsystem's state is now Schmidt-rank ≤ chi_max
            at every bond, demote it to an MPS. This closes the brick-wall
            static-vs-runtime gap — a weakly entangled dense core collapses
            into the bond-dim-capped MPS representation."""
            subsystem = live_subsystems[subsystem_idx]
            if subsystem is None or subsystem.sparse_state is None:
                return
            if len(subsystem.qubits) < 2:
                return
            chi_max = max(1, self.pruning.bond_dim)
            # Guard against expensive SVD on large subsystems. The cost of the
            # full reshape+SVD sweep is O(n·2^n), so we skip when either the
            # subsystem is wider than 16 qubits or the active set is already
            # dense (no benefit expected).
            n = subsystem.sparse_state.n_qubits
            if n > 16:
                return
            active = len(subsystem.sparse_state.amplitudes)
            if active == 0:
                return
            # Only attempt demotion when the max observable rank fits chi_max.
            max_rank = _sparse_max_schmidt_rank(
                subsystem.sparse_state.amplitudes,
                n,
                tolerance=max(1e-12, 4 * self.pruning.amplitude_threshold ** 2),
            )
            if max_rank > chi_max:
                return
            dense = subsystem.sparse_state.to_dense()
            mps = _MPSState.from_dense(
                dense,
                qubit_order=list(subsystem.qubits),
                chi_max=chi_max,
            )
            # from_dense may truncate if numerical noise lifts the rank past
            # the estimated cap; only install if the fidelity is still ≈ 1.
            if mps.fidelity_lower_bound < 1.0 - 1e-8:
                return
            # Preserve the sparse-phase peak so the subsystem's lifetime-peak
            # telemetry doesn't reset to 0 just because we compacted the state.
            subsystem.lifetime_peak_active = max(
                subsystem.lifetime_peak_active,
                subsystem.sparse_state.peak_active_states,
            )
            subsystem.sparse_state = None
            subsystem.mps_state = mps
            subsystem.representation = "mps"
            subsystem.role = "classical_region"
            split_events.append({
                "gate": gate_name,
                "event": "sparse_to_mps_demotion",
                "subsystem": subsystem_idx,
                "qubits": list(subsystem.qubits),
                "max_bond_dim": mps.max_bond_dim(),
            })

        def try_split_mps_at_bonds(subsystem_idx: int, gate_name: str) -> None:
            """Wider-than-1-qubit bipartition splits. For each MPS bond, check
            if the Schmidt spectrum collapses to rank 1 — if so, factor the
            MPS into two smaller subsystems at that bond. Repeats until no
            more bonds are separable."""
            subsystem = live_subsystems[subsystem_idx]
            if subsystem is None or subsystem.mps_state is None:
                return
            if subsystem.mps_state.n_qubits < 2:
                return
            while True:
                mps = subsystem.mps_state
                if mps is None or mps.n_qubits < 2:
                    return
                split_at = None
                for bond in range(mps.n_qubits - 1):
                    pair = mps.try_split_at_bond(bond, split_purity_tolerance)
                    if pair is not None:
                        split_at = (bond, pair)
                        break
                if split_at is None:
                    return
                bond, (left_mps, right_mps) = split_at
                left_qubits = subsystem.qubits[: bond + 1]
                right_qubits = subsystem.qubits[bond + 1:]

                # Install the right half as a new subsystem; collapse to
                # product state if it's a single qubit.
                right_id = _install_half(
                    right_qubits, right_mps, subsystem
                )
                # Install the left half in-place on the parent.
                _install_left_half(subsystem_idx, left_qubits, left_mps)

                split_events.append({
                    "gate": gate_name,
                    "event": "mps_bond_split",
                    "parent_subsystem": subsystem_idx,
                    "child_subsystem": right_id,
                    "left_qubits": list(left_qubits),
                    "right_qubits": list(right_qubits),
                })
                subsystem = live_subsystems[subsystem_idx]
                if subsystem is None or subsystem.mps_state is None:
                    return

        def _install_half(qubits: List[int], mps: "_MPSState",
                          parent: _PhantomLiveSubsystem) -> int:
            if len(qubits) == 1:
                dense = mps.to_dense()
                vec = np.array(dense, dtype=complex)
                norm = float(np.linalg.norm(vec))
                if norm > 0:
                    vec = vec / norm
                sub = _PhantomLiveSubsystem(
                    qubits=list(qubits),
                    role="classical_region",
                    representation="product_state_mps",
                    product_state=_ProductStateQubit(
                        qubit=qubits[0], vector=vec
                    ),
                    gate_count=0,
                    merge_count=0,
                    split_count=0,
                    source_subsystems=[list(qubits)],
                )
            else:
                sub = _PhantomLiveSubsystem(
                    qubits=list(qubits),
                    role="classical_region",
                    representation="mps",
                    mps_state=mps,
                    gate_count=0,
                    merge_count=0,
                    split_count=0,
                    source_subsystems=[list(qubits)],
                )
            return register_subsystem(sub)

        def _install_left_half(subsystem_idx: int, qubits: List[int],
                               mps: "_MPSState") -> None:
            subsystem = live_subsystems[subsystem_idx]
            subsystem.qubits = list(qubits)
            subsystem.split_count += 1
            if len(qubits) == 1:
                dense = mps.to_dense()
                vec = np.array(dense, dtype=complex)
                norm = float(np.linalg.norm(vec))
                if norm > 0:
                    vec = vec / norm
                subsystem.mps_state = None
                subsystem.product_state = _ProductStateQubit(
                    qubit=qubits[0], vector=vec
                )
                subsystem.representation = "product_state_mps"
                subsystem.role = "classical_region"
            else:
                subsystem.mps_state = mps
                subsystem.representation = "mps"
                subsystem.role = "classical_region"

        for op in circuit.ops:
            if not op.targets:
                continue
            subsystem_ids = [qubit_location[qubit] for qubit in op.targets]
            subsystem_idx = subsystem_ids[0]
            if len(set(subsystem_ids)) > 1:
                subsystem_idx = merge_subsystems(subsystem_ids, op.gate_name)

            subsystem = live_subsystems[subsystem_idx]
            if subsystem is None:
                raise RuntimeError("Merged subsystem unexpectedly missing.")

            if subsystem.product_state is not None:
                subsystem.product_state.apply(op.gate_matrix)
                subsystem.gate_count += 1
                continue

            if subsystem.mps_state is not None:
                if len(op.targets) == 1:
                    subsystem.mps_state.apply_single(
                        op.gate_matrix, op.targets[0]
                    )
                elif len(op.targets) == 2 and op.gate_matrix.shape == (4, 4):
                    subsystem.mps_state.apply_two_qubit(
                        op.gate_matrix, op.targets[0], op.targets[1]
                    )
                else:
                    # Multi-qubit (≥3) gate on MPS: we don't have an MPS-native
                    # path for that yet, so promote to sparse core here rather
                    # than silently losing the gate. The sparse state then
                    # takes over for this subsystem.
                    dense = subsystem.mps_state.to_dense()
                    sparse = _SparseAmplitudeState.from_dense(
                        dense,
                        numerical_tolerance=self.pruning.numerical_tolerance,
                    )
                    subsystem.sparse_state = sparse
                    subsystem.mps_state = None
                    subsystem.representation = "sparse_statevector"
                    local_map = {q: i for i, q in enumerate(subsystem.qubits)}
                    sparse.apply_generic(
                        op.gate_matrix,
                        [local_map[q] for q in op.targets],
                        self.pruning,
                    )
                subsystem.gate_count += 1
                continue

            local_map = {qubit: idx for idx, qubit in enumerate(subsystem.qubits)}
            local_targets = [local_map[qubit] for qubit in op.targets]
            state = subsystem.sparse_state
            if state is None:
                raise RuntimeError("Sparse subsystem missing state data.")

            if len(local_targets) == 1:
                state.apply_single(op.gate_matrix, local_targets[0], self.pruning)
            elif len(local_targets) == 2 and op.gate_matrix.shape == (4, 4):
                state.apply_two_qubit(
                    op.gate_matrix,
                    local_targets[0],
                    local_targets[1],
                    self.pruning,
                )
            else:
                state.apply_generic(op.gate_matrix, local_targets, self.pruning)
            subsystem.gate_count += 1

            # Post-gate passes:
            #  1. Factor any Schmidt-rank-1 qubits out of a sparse core.
            #  2. Demote a low-bond-dim sparse core to a true MPS.
            #  3. Split an MPS at any rank-1 bond (wider bipartition splits).
            try_split_subsystem(subsystem_idx, op.gate_name)
            try_demote_sparse_to_mps(subsystem_idx, op.gate_name)
            try_split_mps_at_bonds(subsystem_idx, op.gate_name)

        active_subsystems = [
            subsystem for subsystem in live_subsystems if subsystem is not None
        ]
        core_samples = {
            tuple(subsystem.qubits): subsystem.sparse_state.sample_bitstrings(shots, rng)
            for subsystem in active_subsystems
            if subsystem.sparse_state is not None
        }
        mps_samples = {
            tuple(subsystem.qubits): subsystem.mps_state.sample_bitstrings(shots, rng)
            for subsystem in active_subsystems
            if subsystem.mps_state is not None
        }
        classical_samples = {
            subsystem.qubits[0]: subsystem.product_state.sample_bits(shots, rng)
            for subsystem in active_subsystems
            if subsystem.product_state is not None
        }

        counts: Dict[str, int] = {}
        for shot_idx in range(shots):
            bits = ["0"] * circuit.n_qubits
            for subsystem in active_subsystems:
                if subsystem.sparse_state is not None:
                    local_bits = core_samples[tuple(subsystem.qubits)][shot_idx]
                    for local_idx, qubit in enumerate(subsystem.qubits):
                        bits[qubit] = local_bits[local_idx]
                elif subsystem.mps_state is not None:
                    local_bits = mps_samples[tuple(subsystem.qubits)][shot_idx]
                    # MPS sample bitstrings are indexed by MPS qubit_order
                    # (which matches subsystem.qubits here since we never
                    # permute qubit_order).
                    for local_idx, qubit in enumerate(subsystem.mps_state.qubit_order):
                        bits[qubit] = local_bits[local_idx]
            for qubit, samples in classical_samples.items():
                bits[qubit] = samples[shot_idx]
            bitstring = "".join(bits)
            counts[bitstring] = counts.get(bitstring, 0) + 1

        counts, symmetry_report = SymmetryFilter.apply(counts, circuit.symmetries)
        dt = time.time() - t0

        subsystem_reports = []
        entropy = 0.0
        for subsystem in active_subsystems:
            entropy += subsystem.entropy()
            estimated_dense_bytes = (2 ** len(subsystem.qubits)) * 16
            if subsystem.mps_state is not None:
                estimated_factorized_bytes = subsystem.mps_state.memory_bytes()
            elif subsystem.sparse_state is not None:
                estimated_factorized_bytes = len(
                    subsystem.sparse_state.amplitudes
                ) * 24  # sparse dict entry ≈ 24B
            else:
                estimated_factorized_bytes = 32
            mps_info: Optional[dict] = None
            if subsystem.mps_state is not None:
                mps_info = {
                    "bond_dims": subsystem.mps_state.bond_dims(),
                    "max_bond_dim": subsystem.mps_state.max_bond_dim(),
                    "peak_bond_dim": subsystem.mps_state.peak_bond_dim,
                    "chi_max": subsystem.mps_state.chi_max,
                    "truncation_events": subsystem.mps_state.truncation_events,
                }
            subsystem_reports.append({
                "qubits": subsystem.qubits,
                "role": subsystem.role,
                "representation": subsystem.representation,
                "gate_count": subsystem.gate_count,
                "estimated_dense_bytes": estimated_dense_bytes,
                "estimated_factorized_bytes": estimated_factorized_bytes,
                "active_states_final": subsystem.active_states_final(),
                "peak_active_states": subsystem.peak_active_states(),
                "pruned_probability": subsystem.pruned_probability(),
                "fidelity_lower_bound": subsystem.fidelity_lower_bound(),
                "pruning_events": subsystem.pruning_events(),
                "merge_count": subsystem.merge_count,
                "split_count": subsystem.split_count,
                "source_subsystems": subsystem.source_subsystems,
                "mps_info": mps_info,
            })

        # Reconstruct the exact joint distribution P(bitstring) by tensoring
        # subsystem marginals. This bypasses shot noise — the benchmark uses
        # it for a fidelity check that distinguishes "Phantom is wrong" from
        # "we didn't take enough shots to tell."
        #
        # Cost: O(product of |active states per subsystem|) ≤ 2^n. For circuits
        # that genuinely factor into product states (post-split chains) it is
        # O(2). Skipped above 28 qubits to keep memory bounded.
        final_probabilities: Dict[int, float] = {}
        if circuit.n_qubits <= 28 and active_subsystems:
            joint: Dict[int, float] = {0: 1.0}
            for subsystem in active_subsystems:
                sub_dist: List[Tuple[int, float]]
                if subsystem.product_state is not None:
                    vec = subsystem.product_state.vector
                    sub_dist = [
                        (0, float(abs(vec[0]) ** 2)),
                        (1, float(abs(vec[1]) ** 2)),
                    ]
                    sub_qubits = subsystem.qubits
                elif subsystem.mps_state is not None:
                    dense = subsystem.mps_state.to_dense()
                    sub_dist = [
                        (i, float(abs(a) ** 2))
                        for i, a in enumerate(dense)
                        if abs(a) ** 2 > 1e-15
                    ]
                    # MPS bitstring index matches qubit_order, not subsystem.qubits,
                    # though they coincide in this implementation since we never
                    # permute qubit_order after construction.
                    sub_qubits = subsystem.mps_state.qubit_order
                elif subsystem.sparse_state is not None:
                    sub_dist = [
                        (idx, float(abs(amp) ** 2))
                        for idx, amp in subsystem.sparse_state.amplitudes.items()
                    ]
                    sub_qubits = subsystem.qubits
                else:
                    continue
                sub_width = len(sub_qubits)
                next_joint: Dict[int, float] = {}
                for current_idx, current_p in joint.items():
                    for local_idx, local_p in sub_dist:
                        if local_p <= 0.0:
                            continue
                        merged_idx = current_idx
                        for pos, qubit in enumerate(sub_qubits):
                            bit = (local_idx >> (sub_width - 1 - pos)) & 1
                            bit_mask = 1 << (circuit.n_qubits - 1 - qubit)
                            if bit:
                                merged_idx |= bit_mask
                            else:
                                merged_idx &= ~bit_mask
                        combined = current_p * local_p
                        next_joint[merged_idx] = (
                            next_joint.get(merged_idx, 0.0) + combined
                        )
                joint = next_joint
            final_probabilities = joint

        execution_metadata = {
            "phantom_partition": partition.to_dict(),
            "subsystems": subsystem_reports,
            "merge_events": merge_events,
            "split_events": split_events,
            "final_probabilities": final_probabilities,
        }

        return ExecutionResult(
            counts=dict(sorted(counts.items(), key=lambda item: -item[1])),
            statevector=None,
            execution_time=dt,
            backend_name=self.name(),
            circuit_name=circuit.name,
            n_qubits=circuit.n_qubits,
            gate_count=len(circuit.ops),
            circuit_depth=circuit.depth(),
            entanglement_pairs=[edge.qubits for edge in partition.scan_result.edges],
            entropy=entropy,
            symmetry_report=symmetry_report,
            execution_metadata=execution_metadata,
        )
