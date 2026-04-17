from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .config import RuntimeConfig, SimulationMode
from .types import (
    BundleTopology,
    ClassicalMemory,
    ComplexArray,
    MeasurementRecord,
    RuntimeStats,
    SectorId,
    SectorKind,
    SectorSpec,
    StateBundleSnapshot,
    StateRepresentation,
)


def _as_complex_array(data: np.ndarray | Sequence[complex]) -> ComplexArray:
    arr = np.asarray(data, dtype=np.complex128)
    if arr.ndim not in (1, 2):
        raise ValueError(f"State data must be rank 1 or 2, got shape {arr.shape}")
    return arr


def _is_hermitian(mat: ComplexArray, tol: float) -> bool:
    return np.allclose(mat, mat.conj().T, atol=tol, rtol=0.0)


def _matrix_trace(mat: ComplexArray) -> float:
    return float(np.real_if_close(np.trace(mat)))


def _normalize_statevector(vec: ComplexArray, tol: float) -> ComplexArray:
    norm = np.linalg.norm(vec)
    if norm <= tol:
        raise ValueError("Cannot normalize a near-zero statevector")
    return vec / norm


def _normalize_density_matrix(rho: ComplexArray, tol: float) -> ComplexArray:
    tr = _matrix_trace(rho)
    if tr <= tol:
        raise ValueError("Cannot normalize a density matrix with near-zero trace")
    rho = rho / tr
    rho = 0.5 * (rho + rho.conj().T)
    return rho


@dataclass(slots=True)
class QuantumState:
    """Quantum state wrapper supporting statevector and density-matrix modes."""

    data: ComplexArray
    representation: StateRepresentation
    dims: Tuple[int, ...]
    tol: float = 1e-10

    def __post_init__(self) -> None:
        self.data = _as_complex_array(self.data)
        expected_dim = int(np.prod(self.dims, dtype=np.int64)) if self.dims else 1
        if self.representation == StateRepresentation.STATEVECTOR:
            if self.data.ndim != 1:
                raise ValueError("Statevector representation requires a rank-1 array")
            if self.data.shape[0] != expected_dim:
                raise ValueError(
                    f"Statevector length {self.data.shape[0]} does not match expected dimension {expected_dim}"
                )
            self.data = _normalize_statevector(self.data, self.tol)
        else:
            if self.data.ndim != 2 or self.data.shape[0] != self.data.shape[1]:
                raise ValueError("Density-matrix representation requires a square rank-2 array")
            if self.data.shape[0] != expected_dim:
                raise ValueError(
                    f"Density-matrix width {self.data.shape[0]} does not match expected dimension {expected_dim}"
                )
            if not _is_hermitian(self.data, self.tol):
                raise ValueError("Density matrix must be Hermitian")
            self.data = _normalize_density_matrix(self.data, self.tol)

    @classmethod
    def zero_state(cls, dims: Tuple[int, ...], representation: StateRepresentation, tol: float = 1e-10) -> "QuantumState":
        total_dim = int(np.prod(dims, dtype=np.int64)) if dims else 1
        if representation == StateRepresentation.STATEVECTOR:
            vec = np.zeros(total_dim, dtype=np.complex128)
            vec[0] = 1.0 + 0.0j
            return cls(vec, representation, dims, tol)
        rho = np.zeros((total_dim, total_dim), dtype=np.complex128)
        rho[0, 0] = 1.0 + 0.0j
        return cls(rho, representation, dims, tol)

    def copy(self) -> "QuantumState":
        return QuantumState(np.array(self.data, copy=True), self.representation, self.dims, self.tol)

    @property
    def total_dimension(self) -> int:
        return int(np.prod(self.dims, dtype=np.int64)) if self.dims else 1

    @property
    def trace(self) -> float:
        if self.representation == StateRepresentation.STATEVECTOR:
            return 1.0
        return _matrix_trace(self.data)

    @property
    def purity(self) -> float:
        if self.representation == StateRepresentation.STATEVECTOR:
            return 1.0
        return float(np.real_if_close(np.trace(self.data @ self.data)))

    def as_density_matrix(self) -> ComplexArray:
        if self.representation == StateRepresentation.DENSITY_MATRIX:
            return np.array(self.data, copy=True)
        vec = self.data.reshape((-1, 1))
        return vec @ vec.conj().T


@dataclass(slots=True)
class HilbertBundleState:
    """State container for Section 1 of the CHESSO simulator.

    This object is intentionally focused on correctness and bookkeeping rather
    than speed. Later sections can add backend-specialized tensor views,
    entanglement hypergraphs, noise channels, and adaptive routing.
    """

    topology: BundleTopology
    quantum_state: QuantumState
    config: RuntimeConfig
    classical_memory: ClassicalMemory = field(default_factory=ClassicalMemory)
    stats: RuntimeStats = field(default_factory=RuntimeStats)
    measurements: List[MeasurementRecord] = field(default_factory=list)
    metadata: Dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.config.validate()
        self.topology.validate()
        if self.quantum_state.dims != self.topology.dims:
            raise ValueError(
                f"State dims {self.quantum_state.dims} do not match topology dims {self.topology.dims}"
            )
        if len(self.topology.sectors) > self.config.budget.max_active_qubits:
            raise ValueError("Active sector count exceeds configured max_active_qubits")

    @classmethod
    def initialize(
        cls,
        *,
        sector_dims: Sequence[int],
        config: RuntimeConfig,
        sector_prefix: str = "q",
        representation: Optional[StateRepresentation] = None,
    ) -> "HilbertBundleState":
        config.validate()
        rep = representation or (
            StateRepresentation.STATEVECTOR
            if config.simulation_mode == SimulationMode.STATEVECTOR
            else StateRepresentation.DENSITY_MATRIX
        )
        sectors = [
            SectorSpec(SectorId(f"{sector_prefix}{i}"), int(dim), SectorKind.LOGICAL)
            for i, dim in enumerate(sector_dims)
        ]
        topology = BundleTopology(sectors)
        topology.validate()
        state = QuantumState.zero_state(topology.dims, rep, config.numerical_tolerance)
        return cls(topology=topology, quantum_state=state, config=config)

    @property
    def dims(self) -> Tuple[int, ...]:
        return self.topology.dims

    @property
    def active_sector_count(self) -> int:
        return len(self.topology.sectors)

    @property
    def total_dimension(self) -> int:
        return self.topology.total_dimension

    def copy(self) -> "HilbertBundleState":
        return HilbertBundleState(
            topology=self.topology.copy(),
            quantum_state=self.quantum_state.copy(),
            config=self.config,
            classical_memory=ClassicalMemory(self.classical_memory.snapshot()),
            stats=RuntimeStats(
                step=self.stats.step,
                dynamic_depth=self.stats.dynamic_depth,
                measurements_used=self.stats.measurements_used,
                branches_used=self.stats.branches_used,
                discarded_trace_mass=self.stats.discarded_trace_mass,
            ),
            measurements=[
                MeasurementRecord(
                    label=m.label,
                    strength=m.strength,
                    outcome=m.outcome,
                    probabilities=None if m.probabilities is None else np.array(m.probabilities, copy=True),
                    metadata=dict(m.metadata),
                )
                for m in self.measurements
            ],
            metadata=dict(self.metadata),
        )

    def snapshot(self) -> StateBundleSnapshot:
        return StateBundleSnapshot(
            representation=self.quantum_state.representation,
            dims=self.dims,
            sector_names=tuple(str(sec.sector_id) for sec in self.topology.sectors),
            trace=self.quantum_state.trace,
            purity=self.quantum_state.purity,
            metadata=dict(self.metadata),
        )

    def add_sector(
        self,
        *,
        name: str,
        dimension: int,
        kind: SectorKind = SectorKind.ANCILLA,
        tags: Sequence[str] = (),
    ) -> None:
        if self.active_sector_count + 1 > self.config.budget.max_active_qubits:
            raise ValueError("Cannot expand bundle beyond max_active_qubits")
        if self.topology.has_sector(name):
            raise ValueError(f"Sector {name!r} already exists")
        if dimension <= 0:
            raise ValueError("dimension must be positive")

        self.topology.sectors.append(
            SectorSpec(
                sector_id=SectorId(name),
                dimension=dimension,
                kind=kind,
                tags=tuple(tags),
            )
        )
        self._tensor_with_ground_sector(dimension)

    def remove_last_sector(self) -> SectorSpec:
        if not self.topology.sectors:
            raise ValueError("Cannot remove a sector from an empty topology")
        removed = self.topology.sectors.pop()
        self._truncate_last_sector(removed.dimension)
        return removed

    def record_measurement(self, record: MeasurementRecord) -> None:
        self.measurements.append(record)
        self.stats.measurements_used += 1

    def advance_step(self, dynamic_depth_increment: int = 0) -> None:
        self.stats.step += 1
        self.stats.dynamic_depth += int(dynamic_depth_increment)

    def _tensor_with_ground_sector(self, new_dim: int) -> None:
        old = self.quantum_state
        if old.representation == StateRepresentation.STATEVECTOR:
            extended = np.zeros(old.total_dimension * new_dim, dtype=np.complex128)
            extended[::new_dim] = old.data
            self.quantum_state = QuantumState(
                extended,
                old.representation,
                self.topology.dims,
                self.config.numerical_tolerance,
            )
            return

        old_rho = old.data
        new_total = old.total_dimension * new_dim
        extended = np.zeros((new_total, new_total), dtype=np.complex128)
        for i in range(old.total_dimension):
            for j in range(old.total_dimension):
                extended[i * new_dim, j * new_dim] = old_rho[i, j]
        self.quantum_state = QuantumState(
            extended,
            old.representation,
            self.topology.dims,
            self.config.numerical_tolerance,
        )

    def _truncate_last_sector(self, removed_dim: int) -> None:
        old = self.quantum_state
        if old.total_dimension % removed_dim != 0:
            raise ValueError("Bundle dimension is not divisible by removed_dim")

        if old.representation == StateRepresentation.STATEVECTOR:
            truncated = old.data[::removed_dim]
            self.quantum_state = QuantumState(
                truncated,
                old.representation,
                self.topology.dims,
                self.config.numerical_tolerance,
            )
            return

        kept_dim = old.total_dimension // removed_dim
        truncated = np.zeros((kept_dim, kept_dim), dtype=np.complex128)
        for i in range(kept_dim):
            for j in range(kept_dim):
                truncated[i, j] = old.data[i * removed_dim, j * removed_dim]
        discarded_mass = max(0.0, 1.0 - _matrix_trace(truncated))
        self.stats.record_prune_loss(discarded_mass)
        self.quantum_state = QuantumState(
            truncated,
            old.representation,
            self.topology.dims,
            self.config.numerical_tolerance,
        )
