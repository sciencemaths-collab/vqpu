from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np

from core.state import HilbertBundleState, QuantumState
from core.types import ComplexArray, SectorId, SectorKind, SectorSpec, StateRepresentation
from .measurement import evaluate_measurement_probabilities, make_computational_basis_instrument
from .unitary_ops import _permutation, sector_indices


@dataclass(slots=True)
class ExpansionCandidate:
    """Candidate sector to add during adaptive expansion."""

    name: str
    dimension: int = 2
    score: float = 0.0
    kind: SectorKind = SectorKind.ANCILLA
    tags: Tuple[str, ...] = ()
    local_state: np.ndarray | Sequence[complex] | None = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if self.dimension <= 0:
            raise ValueError(f"ExpansionCandidate {self.name!r} has invalid dimension {self.dimension}")


def _as_complex_array(data: np.ndarray | Sequence[complex]) -> ComplexArray:
    arr = np.asarray(data, dtype=np.complex128)
    if arr.ndim not in (1, 2):
        raise ValueError(f"Expected rank-1 or rank-2 local state, got shape {arr.shape}")
    return arr


def _ground_state_vector(dimension: int) -> ComplexArray:
    if dimension <= 0:
        raise ValueError("dimension must be positive")
    vec = np.zeros(dimension, dtype=np.complex128)
    vec[0] = 1.0 + 0.0j
    return vec


def _normalize_vector(vec: ComplexArray, tol: float) -> ComplexArray:
    norm = float(np.linalg.norm(vec))
    if norm <= tol:
        raise ValueError("Cannot normalize a near-zero vector")
    return vec / norm


def _normalize_density_matrix(rho: ComplexArray, tol: float) -> ComplexArray:
    if rho.ndim != 2 or rho.shape[0] != rho.shape[1]:
        raise ValueError("Density matrix must be square")
    rho = 0.5 * (rho + rho.conj().T)
    trace = float(np.real_if_close(np.trace(rho)))
    if trace <= tol:
        raise ValueError("Cannot normalize a density matrix with near-zero trace")
    return rho / trace


def _local_density_matrix(
    local_state: np.ndarray | Sequence[complex] | None,
    dimension: int,
    tol: float,
) -> Tuple[ComplexArray, StateRepresentation]:
    if local_state is None:
        vec = _ground_state_vector(dimension)
        return vec, StateRepresentation.STATEVECTOR

    arr = _as_complex_array(local_state)
    if arr.ndim == 1:
        if arr.shape[0] != dimension:
            raise ValueError(
                f"Local state length {arr.shape[0]} does not match dimension {dimension}"
            )
        return _normalize_vector(arr, tol), StateRepresentation.STATEVECTOR

    if arr.shape != (dimension, dimension):
        raise ValueError(
            f"Local density matrix shape {arr.shape} does not match dimension {dimension}"
        )
    return _normalize_density_matrix(arr, tol), StateRepresentation.DENSITY_MATRIX


def _ensure_reshape_budget(bundle: HilbertBundleState) -> None:
    if bundle.active_sector_count > bundle.config.budget.max_active_qubits:
        raise ValueError("Active sector count exceeds max_active_qubits budget")


def _check_prune_budget(bundle: HilbertBundleState, additional_loss: float) -> None:
    projected = bundle.stats.discarded_trace_mass + float(additional_loss)
    if projected - bundle.config.numerical_tolerance > bundle.config.budget.max_prune_loss:
        raise ValueError(
            f"Prune budget exceeded: projected discarded mass {projected:.6g} > "
            f"max_prune_loss {bundle.config.budget.max_prune_loss:.6g}"
        )


def _update_branch_peak(bundle: HilbertBundleState, active_count: int) -> None:
    bundle.stats.branches_used = max(bundle.stats.branches_used, int(active_count))
    if bundle.stats.branches_used > bundle.config.budget.max_branches:
        raise ValueError(
            f"Branch budget exceeded: {bundle.stats.branches_used} > {bundle.config.budget.max_branches}"
        )


def basis_populations(bundle: HilbertBundleState) -> np.ndarray:
    """Return computational-basis populations of the current bundle state."""
    if bundle.quantum_state.representation == StateRepresentation.STATEVECTOR:
        probs = np.abs(bundle.quantum_state.data) ** 2
    else:
        probs = np.real_if_close(np.diag(bundle.quantum_state.data)).astype(np.float64)
    probs = np.asarray(probs, dtype=np.float64)
    probs = np.clip(probs, 0.0, None)
    total = float(np.sum(probs))
    if total > bundle.config.numerical_tolerance:
        probs /= total
    return probs


def attach_local_sector(
    bundle: HilbertBundleState,
    *,
    name: str,
    dimension: int,
    local_state: np.ndarray | Sequence[complex] | None = None,
    kind: SectorKind = SectorKind.ANCILLA,
    tags: Sequence[str] = (),
    metadata: Dict[str, Any] | None = None,
    in_place: bool = True,
) -> HilbertBundleState:
    """Append a new sector initialized in a specified local state."""
    work = bundle if in_place else bundle.copy()
    if work.topology.has_sector(name):
        raise ValueError(f"Sector {name!r} already exists")
    if work.active_sector_count + 1 > work.config.budget.max_active_qubits:
        raise ValueError("Cannot expand bundle beyond max_active_qubits")

    tol = work.config.numerical_tolerance
    local, local_rep = _local_density_matrix(local_state, int(dimension), tol)

    if work.quantum_state.representation == StateRepresentation.STATEVECTOR and local_rep == StateRepresentation.STATEVECTOR:
        new_data = np.kron(work.quantum_state.data, local)
        new_rep = StateRepresentation.STATEVECTOR
    else:
        rho_work = work.quantum_state.as_density_matrix()
        rho_local = local if local_rep == StateRepresentation.DENSITY_MATRIX else np.outer(local, local.conj())
        new_data = np.kron(rho_work, rho_local)
        new_rep = StateRepresentation.DENSITY_MATRIX

    work.topology.sectors.append(
        SectorSpec(
            sector_id=SectorId(name),
            dimension=int(dimension),
            kind=kind,
            tags=tuple(tags),
            metadata=dict(metadata or {}),
        )
    )
    work.quantum_state = QuantumState(
        new_data,
        new_rep,
        work.topology.dims,
        work.config.numerical_tolerance,
    )
    work.advance_step(dynamic_depth_increment=1)
    history = work.metadata.setdefault("reshape_history", [])
    if isinstance(history, list):
        history.append({"action": "attach", "name": name, "dimension": int(dimension)})
    _ensure_reshape_budget(work)
    return work


def attach_ground_sector(
    bundle: HilbertBundleState,
    *,
    name: str,
    dimension: int = 2,
    kind: SectorKind = SectorKind.ANCILLA,
    tags: Sequence[str] = (),
    in_place: bool = True,
) -> HilbertBundleState:
    """Append a new sector initialized in the computational ground state."""
    return attach_local_sector(
        bundle,
        name=name,
        dimension=dimension,
        local_state=None,
        kind=kind,
        tags=tags,
        in_place=in_place,
    )


def adaptive_expand_bundle(
    bundle: HilbertBundleState,
    candidates: Iterable[ExpansionCandidate],
    *,
    max_new: int | None = None,
    min_score: float = 0.0,
    in_place: bool = True,
) -> HilbertBundleState:
    """Attach the highest-scoring candidate sectors until the budget is full."""
    work = bundle if in_place else bundle.copy()
    remaining = work.config.budget.max_active_qubits - work.active_sector_count
    if remaining <= 0:
        work.metadata["last_expansion"] = ()
        return work

    prepared: List[ExpansionCandidate] = []
    for cand in candidates:
        cand.validate()
        if cand.score >= min_score:
            prepared.append(cand)

    prepared.sort(key=lambda c: (-float(c.score), c.name))
    limit = remaining if max_new is None else min(int(max_new), remaining)
    selected = prepared[:limit]
    for cand in selected:
        attach_local_sector(
            work,
            name=cand.name,
            dimension=cand.dimension,
            local_state=cand.local_state,
            kind=cand.kind,
            tags=cand.tags,
            metadata={"score": cand.score, **cand.metadata},
            in_place=True,
        )
    work.metadata["last_expansion"] = tuple(c.name for c in selected)
    if len(prepared) > len(selected):
        work.metadata["expansion_skipped"] = tuple(c.name for c in prepared[len(selected):])
    return work


def _project_statevector_to_basis_block(
    vec: np.ndarray,
    dims: Sequence[int],
    targets: Sequence[int],
    basis_index: int,
) -> Tuple[ComplexArray, float]:
    dims_t = tuple(int(d) for d in dims)
    n = len(dims_t)
    target_t = tuple(int(t) for t in targets)
    target_dims = tuple(dims_t[t] for t in target_t)
    target_dim = int(np.prod(target_dims, dtype=np.int64)) if target_dims else 1
    if not 0 <= basis_index < target_dim:
        raise IndexError(f"basis_index {basis_index} out of range for target_dim {target_dim}")
    rest_dims = tuple(dims_t[i] for i in range(n) if i not in target_t)
    rest_dim = int(np.prod(rest_dims, dtype=np.int64)) if rest_dims else 1

    perm, _ = _permutation(n, target_t)
    tensor = np.asarray(vec, dtype=np.complex128).reshape(dims_t).transpose(perm)
    flat = tensor.reshape(target_dim, rest_dim)
    branch = np.array(flat[basis_index], copy=True)
    probability = float(np.real_if_close(np.vdot(branch, branch)))
    return branch, probability


def _project_density_to_basis_block(
    rho: np.ndarray,
    dims: Sequence[int],
    targets: Sequence[int],
    basis_index: int,
) -> Tuple[ComplexArray, float]:
    dims_t = tuple(int(d) for d in dims)
    n = len(dims_t)
    target_t = tuple(int(t) for t in targets)
    target_dims = tuple(dims_t[t] for t in target_t)
    target_dim = int(np.prod(target_dims, dtype=np.int64)) if target_dims else 1
    if not 0 <= basis_index < target_dim:
        raise IndexError(f"basis_index {basis_index} out of range for target_dim {target_dim}")
    rest_dims = tuple(dims_t[i] for i in range(n) if i not in target_t)
    rest_dim = int(np.prod(rest_dims, dtype=np.int64)) if rest_dims else 1

    bra_perm, _ = _permutation(n, target_t)
    full_perm = bra_perm + tuple(n + idx for idx in bra_perm)

    tensor = np.asarray(rho, dtype=np.complex128).reshape(dims_t + dims_t).transpose(full_perm)
    block = tensor.reshape(target_dim, rest_dim, target_dim, rest_dim)
    selected = np.array(block[basis_index, :, basis_index, :], copy=True)
    probability = float(np.real_if_close(np.trace(selected)))
    return selected.reshape((rest_dim, rest_dim)), probability


def detach_basis_sector(
    bundle: HilbertBundleState,
    targets: Sequence[int | str | SectorId],
    *,
    basis_index: int = 0,
    in_place: bool = True,
) -> HilbertBundleState:
    """Condition on a basis outcome for the target sectors and then remove them."""
    work = bundle if in_place else bundle.copy()
    target_idx = sector_indices(work, targets)
    keep_indices = [i for i in range(work.active_sector_count) if i not in target_idx]
    old_dims = work.dims

    if work.quantum_state.representation == StateRepresentation.STATEVECTOR:
        reduced_vec, kept_probability = _project_statevector_to_basis_block(
            work.quantum_state.data, old_dims, target_idx, basis_index
        )
        _check_prune_budget(work, 1.0 - kept_probability)
        new_data = reduced_vec
        new_rep = StateRepresentation.STATEVECTOR
    else:
        reduced_rho, kept_probability = _project_density_to_basis_block(
            work.quantum_state.data, old_dims, target_idx, basis_index
        )
        _check_prune_budget(work, 1.0 - kept_probability)
        new_data = reduced_rho
        new_rep = StateRepresentation.DENSITY_MATRIX

    removed = [work.topology.sectors[i] for i in target_idx]
    work.topology.sectors = [work.topology.sectors[i] for i in keep_indices]
    work.stats.record_prune_loss(max(0.0, 1.0 - kept_probability))
    work.quantum_state = QuantumState(
        new_data,
        new_rep,
        work.topology.dims,
        work.config.numerical_tolerance,
    )
    work.advance_step(dynamic_depth_increment=1)
    history = work.metadata.setdefault("reshape_history", [])
    if isinstance(history, list):
        history.append(
            {
                "action": "detach_basis",
                "targets": tuple(str(sec.sector_id) for sec in removed),
                "basis_index": int(basis_index),
                "kept_probability": kept_probability,
            }
        )
    return work


def _partial_trace_density_matrix(
    rho: np.ndarray,
    dims: Sequence[int],
    trace_indices: Sequence[int],
) -> ComplexArray:
    dims_curr = list(int(d) for d in dims)
    tensor = np.asarray(rho, dtype=np.complex128).reshape(tuple(dims_curr) + tuple(dims_curr))
    for idx in sorted((int(i) for i in trace_indices), reverse=True):
        n_curr = len(dims_curr)
        tensor = np.trace(tensor, axis1=idx, axis2=idx + n_curr)
        dims_curr.pop(idx)
    if not dims_curr:
        scalar = np.asarray(tensor, dtype=np.complex128).reshape(1, 1)
        return scalar
    kept_dim = int(np.prod(dims_curr, dtype=np.int64))
    return np.asarray(tensor, dtype=np.complex128).reshape(kept_dim, kept_dim)


def _density_to_pure_statevector(rho: np.ndarray, tol: float) -> ComplexArray | None:
    vals, vecs = np.linalg.eigh(np.asarray(rho, dtype=np.complex128))
    idx = int(np.argmax(np.real(vals)))
    lam = float(np.real(vals[idx]))
    if abs(lam - 1.0) > max(tol, 1e-8):
        return None
    vec = np.asarray(vecs[:, idx], dtype=np.complex128)
    phase = np.angle(vec[np.argmax(np.abs(vec))]) if np.any(np.abs(vec) > tol) else 0.0
    vec = vec * np.exp(-1j * phase)
    return vec


def trace_out_sectors(
    bundle: HilbertBundleState,
    targets: Sequence[int | str | SectorId],
    *,
    in_place: bool = True,
    prefer_statevector: bool = True,
) -> HilbertBundleState:
    """Trace out target sectors, keeping the remainder of the bundle."""
    work = bundle if in_place else bundle.copy()
    target_idx = sector_indices(work, targets)
    keep_indices = [i for i in range(work.active_sector_count) if i not in target_idx]
    rho = work.quantum_state.as_density_matrix()
    reduced = _partial_trace_density_matrix(rho, work.dims, target_idx)

    removed = [work.topology.sectors[i] for i in target_idx]
    work.topology.sectors = [work.topology.sectors[i] for i in keep_indices]

    tol = work.config.numerical_tolerance
    if (
        prefer_statevector
        and work.quantum_state.representation == StateRepresentation.STATEVECTOR
        and reduced.shape != (1, 1)
    ):
        pure_vec = _density_to_pure_statevector(reduced, tol)
        if pure_vec is not None:
            work.quantum_state = QuantumState(
                pure_vec,
                StateRepresentation.STATEVECTOR,
                work.topology.dims,
                tol,
            )
        else:
            work.quantum_state = QuantumState(
                reduced,
                StateRepresentation.DENSITY_MATRIX,
                work.topology.dims,
                tol,
            )
    else:
        rep = StateRepresentation.STATEVECTOR if reduced.shape == (1, 1) else StateRepresentation.DENSITY_MATRIX
        data = np.array([1.0 + 0.0j], dtype=np.complex128) if reduced.shape == (1, 1) and rep == StateRepresentation.STATEVECTOR else reduced
        work.quantum_state = QuantumState(data, rep, work.topology.dims, tol)

    work.advance_step(dynamic_depth_increment=1)
    history = work.metadata.setdefault("reshape_history", [])
    if isinstance(history, list):
        history.append(
            {
                "action": "trace_out",
                "targets": tuple(str(sec.sector_id) for sec in removed),
            }
        )
    return work


def prune_low_population_branches(
    bundle: HilbertBundleState,
    *,
    threshold: float,
    in_place: bool = True,
) -> HilbertBundleState:
    """Zero out computational basis branches whose population is below a threshold."""
    if threshold < 0.0:
        raise ValueError("threshold must be non-negative")
    work = bundle if in_place else bundle.copy()
    probs = basis_populations(work)
    keep = probs >= float(threshold)
    if not np.any(keep):
        keep[int(np.argmax(probs))] = True

    discarded = float(np.sum(probs[~keep]))
    _check_prune_budget(work, discarded)

    if work.quantum_state.representation == StateRepresentation.STATEVECTOR:
        new_data = np.array(work.quantum_state.data, copy=True)
        new_data[~keep] = 0.0
        rep = StateRepresentation.STATEVECTOR
    else:
        new_data = np.array(work.quantum_state.data, copy=True)
        removed_idx = np.where(~keep)[0]
        new_data[removed_idx, :] = 0.0
        new_data[:, removed_idx] = 0.0
        rep = StateRepresentation.DENSITY_MATRIX

    work.stats.record_prune_loss(discarded)
    work.quantum_state = QuantumState(new_data, rep, work.dims, work.config.numerical_tolerance)
    _update_branch_peak(work, int(np.sum(keep)))
    work.advance_step(dynamic_depth_increment=1)
    history = work.metadata.setdefault("reshape_history", [])
    if isinstance(history, list):
        history.append({"action": "prune", "threshold": float(threshold), "discarded_mass": discarded})
    return work


def compress_to_top_k_branches(
    bundle: HilbertBundleState,
    *,
    k: int,
    in_place: bool = True,
) -> HilbertBundleState:
    """Keep only the k most-populated computational basis branches."""
    if k <= 0:
        raise ValueError("k must be positive")
    work = bundle if in_place else bundle.copy()
    probs = basis_populations(work)
    if k >= len(probs):
        _update_branch_peak(work, len(probs))
        return work
    if k > work.config.budget.max_branches:
        raise ValueError(
            f"k={k} exceeds configured max_branches={work.config.budget.max_branches}"
        )

    keep_idx = np.argsort(probs)[::-1][:k]
    keep = np.zeros_like(probs, dtype=bool)
    keep[keep_idx] = True
    discarded = float(np.sum(probs[~keep]))
    _check_prune_budget(work, discarded)

    if work.quantum_state.representation == StateRepresentation.STATEVECTOR:
        new_data = np.array(work.quantum_state.data, copy=True)
        new_data[~keep] = 0.0
        rep = StateRepresentation.STATEVECTOR
    else:
        new_data = np.array(work.quantum_state.data, copy=True)
        removed_idx = np.where(~keep)[0]
        new_data[removed_idx, :] = 0.0
        new_data[:, removed_idx] = 0.0
        rep = StateRepresentation.DENSITY_MATRIX

    work.stats.record_prune_loss(discarded)
    work.quantum_state = QuantumState(new_data, rep, work.dims, work.config.numerical_tolerance)
    _update_branch_peak(work, k)
    work.advance_step(dynamic_depth_increment=1)
    history = work.metadata.setdefault("reshape_history", [])
    if isinstance(history, list):
        history.append({"action": "top_k", "k": int(k), "discarded_mass": discarded})
    return work


def compress_deterministic_sector_to_memory(
    bundle: HilbertBundleState,
    targets: Sequence[int | str | SectorId],
    *,
    confidence_threshold: float = 0.999,
    store_key: str | None = None,
    in_place: bool = True,
) -> HilbertBundleState:
    """Store a nearly deterministic target outcome in classical memory and remove the sector."""
    if not 0.0 <= confidence_threshold <= 1.0:
        raise ValueError("confidence_threshold must be in [0, 1]")
    work = bundle if in_place else bundle.copy()
    dims = tuple(work.dims[i] for i in sector_indices(work, targets))
    instrument = make_computational_basis_instrument(dims, strength=1.0, name="deterministic_probe")
    probs = evaluate_measurement_probabilities(work, instrument, targets)
    outcome = int(np.argmax(probs))
    confidence = float(probs[outcome])
    if confidence < confidence_threshold:
        raise ValueError(
            f"Target sectors are not deterministic enough: confidence {confidence:.6g} < "
            f"threshold {confidence_threshold:.6g}"
        )

    detach_basis_sector(work, targets, basis_index=outcome, in_place=True)
    if store_key is not None:
        work.classical_memory.put(store_key, outcome)
    work.metadata["last_deterministic_compression"] = {
        "targets": tuple(str(t) for t in targets),
        "outcome": outcome,
        "confidence": confidence,
    }
    return work


__all__ = [
    "ExpansionCandidate",
    "adaptive_expand_bundle",
    "attach_ground_sector",
    "attach_local_sector",
    "basis_populations",
    "compress_deterministic_sector_to_memory",
    "compress_to_top_k_branches",
    "detach_basis_sector",
    "prune_low_population_branches",
    "trace_out_sectors",
]
