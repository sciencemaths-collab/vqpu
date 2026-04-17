from __future__ import annotations

from typing import Iterable, Sequence, Tuple

import numpy as np

from vqpu.chesso.core.hypergraph import EntanglementHypergraph
from vqpu.chesso.core.state import HilbertBundleState, QuantumState
from vqpu.chesso.core.types import ComplexArray, SectorId, StateRepresentation


# -----------------------------------------------------------------------------
# Basic gate library
# -----------------------------------------------------------------------------


def _complex_matrix(data: Sequence[Sequence[complex]] | np.ndarray) -> ComplexArray:
    arr = np.asarray(data, dtype=np.complex128)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError(f"Operator must be a square matrix, got shape {arr.shape}")
    return arr



def _is_unitary(operator: ComplexArray, tol: float) -> bool:
    ident = np.eye(operator.shape[0], dtype=np.complex128)
    return np.allclose(operator.conj().T @ operator, ident, atol=tol, rtol=0.0)



def tensor_product(*operators: np.ndarray) -> ComplexArray:
    if not operators:
        return np.eye(1, dtype=np.complex128)
    out = _complex_matrix(operators[0])
    for op in operators[1:]:
        out = np.kron(out, _complex_matrix(op))
    return out



def pauli_x() -> ComplexArray:
    return np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)



def pauli_y() -> ComplexArray:
    return np.array([[0.0, -1j], [1j, 0.0]], dtype=np.complex128)



def pauli_z() -> ComplexArray:
    return np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)



def hadamard() -> ComplexArray:
    return (1.0 / np.sqrt(2.0)) * np.array([[1.0, 1.0], [1.0, -1.0]], dtype=np.complex128)



def phase(theta: float) -> ComplexArray:
    return np.array([[1.0, 0.0], [0.0, np.exp(1j * theta)]], dtype=np.complex128)



def rotation_x(theta: float) -> ComplexArray:
    c = np.cos(theta / 2.0)
    s = np.sin(theta / 2.0)
    return np.array([[c, -1j * s], [-1j * s, c]], dtype=np.complex128)



def rotation_y(theta: float) -> ComplexArray:
    c = np.cos(theta / 2.0)
    s = np.sin(theta / 2.0)
    return np.array([[c, -s], [s, c]], dtype=np.complex128)



def rotation_z(theta: float) -> ComplexArray:
    return np.array(
        [[np.exp(-1j * theta / 2.0), 0.0], [0.0, np.exp(1j * theta / 2.0)]],
        dtype=np.complex128,
    )



def controlled_x() -> ComplexArray:
    return np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 0.0],
        ],
        dtype=np.complex128,
    )



def controlled_z() -> ComplexArray:
    return np.diag([1.0, 1.0, 1.0, -1.0]).astype(np.complex128)



def swap() -> ComplexArray:
    return np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.complex128,
    )


# -----------------------------------------------------------------------------
# Indexing and dimension helpers
# -----------------------------------------------------------------------------



def sector_indices(bundle: HilbertBundleState, targets: Sequence[int | str | SectorId]) -> Tuple[int, ...]:
    if not targets:
        raise ValueError("targets must not be empty")
    resolved = []
    names = [str(spec.sector_id) for spec in bundle.topology.sectors]
    for target in targets:
        if isinstance(target, int):
            idx = target
        else:
            name = str(target)
            if name not in names:
                raise KeyError(f"Unknown sector target: {name}")
            idx = names.index(name)
        if idx < 0 or idx >= len(names):
            raise IndexError(f"Target index out of range: {idx}")
        resolved.append(int(idx))
    if len(set(resolved)) != len(resolved):
        raise ValueError(f"Duplicate targets are not allowed: {resolved}")
    return tuple(resolved)



def target_dimensions(bundle: HilbertBundleState, targets: Sequence[int | str | SectorId]) -> Tuple[int, ...]:
    indices = sector_indices(bundle, targets)
    return tuple(bundle.dims[idx] for idx in indices)



def _validate_operator_for_targets(
    bundle: HilbertBundleState,
    operator: np.ndarray,
    targets: Sequence[int | str | SectorId],
) -> Tuple[ComplexArray, Tuple[int, ...]]:
    mat = _complex_matrix(operator)
    dims = target_dimensions(bundle, targets)
    target_dim = int(np.prod(dims, dtype=np.int64)) if dims else 1
    if mat.shape != (target_dim, target_dim):
        raise ValueError(
            f"Operator shape {mat.shape} does not match target dimensions {dims} -> {target_dim}"
        )
    tol = bundle.config.numerical_tolerance
    if not _is_unitary(mat, tol):
        raise ValueError("Section 3 operator application currently requires a unitary operator")
    return mat, dims



def _permutation(n: int, targets: Tuple[int, ...]) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    if len(targets) > n:
        raise ValueError("Too many targets for system rank")
    remainder = tuple(idx for idx in range(n) if idx not in targets)
    perm = tuple(targets) + remainder
    inv = tuple(np.argsort(perm))
    return perm, inv


# -----------------------------------------------------------------------------
# Numerical application kernels
# -----------------------------------------------------------------------------



def apply_operator_to_statevector(
    statevector: np.ndarray,
    dims: Sequence[int],
    operator: np.ndarray,
    targets: Sequence[int],
) -> ComplexArray:
    vec = np.asarray(statevector, dtype=np.complex128)
    dims_t = tuple(int(d) for d in dims)
    n = len(dims_t)
    target_t = tuple(int(t) for t in targets)
    target_dims = tuple(dims_t[t] for t in target_t)
    target_dim = int(np.prod(target_dims, dtype=np.int64)) if target_dims else 1
    rest_dim = int(np.prod([dims_t[i] for i in range(n) if i not in target_t], dtype=np.int64)) if n > len(target_t) else 1

    perm, inv = _permutation(n, target_t)
    tensor = vec.reshape(dims_t).transpose(perm)
    flat = tensor.reshape(target_dim, rest_dim)
    updated = operator @ flat
    restored = updated.reshape(target_dims + tuple(dims_t[i] for i in range(n) if i not in target_t)).transpose(inv)
    return restored.reshape(vec.shape)



def apply_operator_to_density_matrix(
    density_matrix: np.ndarray,
    dims: Sequence[int],
    operator: np.ndarray,
    targets: Sequence[int],
) -> ComplexArray:
    rho = np.asarray(density_matrix, dtype=np.complex128)
    dims_t = tuple(int(d) for d in dims)
    n = len(dims_t)
    target_t = tuple(int(t) for t in targets)
    target_dims = tuple(dims_t[t] for t in target_t)
    rest_dims = tuple(dims_t[i] for i in range(n) if i not in target_t)
    target_dim = int(np.prod(target_dims, dtype=np.int64)) if target_dims else 1
    rest_dim = int(np.prod(rest_dims, dtype=np.int64)) if rest_dims else 1

    bra_perm, bra_inv = _permutation(n, target_t)
    full_perm = bra_perm + tuple(n + idx for idx in bra_perm)
    full_inv = bra_inv + tuple(n + idx for idx in bra_inv)

    tensor = rho.reshape(dims_t + dims_t).transpose(full_perm)
    block = tensor.reshape(target_dim, rest_dim, target_dim, rest_dim)
    updated = np.einsum("ai,irjs,bj->arbs", operator, block, operator.conj(), optimize=True)
    restored = updated.reshape(target_dims + rest_dims + target_dims + rest_dims).transpose(full_inv)
    return restored.reshape(rho.shape)


# -----------------------------------------------------------------------------
# Bundle-facing APIs
# -----------------------------------------------------------------------------



def apply_local_operator(
    bundle: HilbertBundleState,
    operator: np.ndarray,
    targets: Sequence[int | str | SectorId],
    *,
    in_place: bool = True,
    label: str | None = None,
) -> HilbertBundleState:
    mat, _ = _validate_operator_for_targets(bundle, operator, targets)
    target_idx = sector_indices(bundle, targets)
    work = bundle if in_place else bundle.copy()

    if work.quantum_state.representation == StateRepresentation.STATEVECTOR:
        updated = apply_operator_to_statevector(work.quantum_state.data, work.dims, mat, target_idx)
    else:
        updated = apply_operator_to_density_matrix(work.quantum_state.data, work.dims, mat, target_idx)

    work.quantum_state = QuantumState(
        updated,
        work.quantum_state.representation,
        work.quantum_state.dims,
        work.config.numerical_tolerance,
    )
    work.advance_step(dynamic_depth_increment=1)
    if label:
        history = work.metadata.setdefault("operator_history", [])
        if isinstance(history, list):
            history.append(
                {
                    "label": label,
                    "targets": tuple(str(bundle.topology.sectors[i].sector_id) for i in target_idx),
                    "shape": tuple(mat.shape),
                }
            )
    return work



def make_hyperedge_phase_entangler(
    target_dims: Sequence[int],
    theta: float,
    *,
    phase_bias: float = 0.0,
    local_weights: Sequence[float] | None = None,
    profile: str = "occupancy",
) -> ComplexArray:
    """Diagonal n-body phase unitary.

    profile="occupancy"  (default, original CHESSO profile)
        phase(multi) = phase_bias + theta * (weighted_occupancy) ** n
        a smooth nonlinear kernel useful as a generalized spin-coupling term.

    profile="parity"
        phase(multi) = phase_bias - (theta/2) * (-1)^(# excited)
        the textbook parity phase gadget — i.e. exp(-i θ/2 · Z⊗Z⊗…⊗Z) on
        qubit registers. This is what chemistry / QAOA / parity-based
        phase-estimation primitives use.

    profile="count"
        phase(multi) = phase_bias + theta * (# excited) / n
        linear in total excitation, useful for number-conserving ansatze.
    """
    dims = tuple(int(dim) for dim in target_dims)
    if not dims:
        raise ValueError("target_dims must not be empty")
    if any(dim <= 0 for dim in dims):
        raise ValueError(f"All target dimensions must be positive, got {dims}")

    if local_weights is None:
        weights = np.ones(len(dims), dtype=np.float64)
    else:
        weights = np.asarray(local_weights, dtype=np.float64)
        if weights.shape != (len(dims),):
            raise ValueError(
                f"local_weights must have shape {(len(dims),)}, got {weights.shape}"
            )

    profile_key = str(profile).lower()
    if profile_key not in {"occupancy", "parity", "count"}:
        raise ValueError(
            f"unknown hyperedge profile {profile!r}; "
            "expected one of 'occupancy', 'parity', 'count'"
        )

    total_weight = max(float(np.sum(np.abs(weights))), 1e-12)
    total_dim = int(np.prod(dims, dtype=np.int64))
    diag = np.empty(total_dim, dtype=np.complex128)
    n_axes = len(dims)

    for flat_idx, multi in enumerate(np.ndindex(dims)):
        if profile_key == "occupancy":
            occupancy = 0.0
            for idx, dim, weight in zip(multi, dims, weights):
                scaled = 0.0 if dim == 1 else idx / (dim - 1)
                occupancy += weight * scaled
            occupancy /= total_weight
            nonlinear = occupancy ** max(2, n_axes)
            phase = phase_bias + theta * nonlinear
        elif profile_key == "parity":
            # "excited" = any idx > 0; parity is ±1 based on that count.
            # Matches exp(-i θ/2 · Z⊗…⊗Z) on qubits (idx ∈ {0,1}).
            excited = sum(1 for idx in multi if int(idx) > 0)
            parity_sign = 1 if (excited % 2 == 0) else -1
            phase = phase_bias - 0.5 * theta * parity_sign
        else:  # count
            excited = sum(1 for idx in multi if int(idx) > 0)
            phase = phase_bias + theta * excited / max(n_axes, 1)
        diag[flat_idx] = np.exp(1j * phase)
    return np.diag(diag)



def apply_hyperedge_entangler(
    bundle: HilbertBundleState,
    graph: EntanglementHypergraph,
    edge_id: str,
    *,
    strength: float = 1.0,
    in_place: bool = True,
) -> HilbertBundleState:
    if edge_id not in graph.hyperedges:
        raise KeyError(f"Unknown hyperedge: {edge_id}")
    edge = graph.hyperedges[edge_id]
    dims = target_dimensions(bundle, edge.members)
    theta = strength * edge.weight * edge.coherence_score
    operator = make_hyperedge_phase_entangler(dims, theta, phase_bias=edge.phase_bias)
    return apply_local_operator(
        bundle,
        operator,
        edge.members,
        in_place=in_place,
        label=f"hyperedge:{edge_id}",
    )
