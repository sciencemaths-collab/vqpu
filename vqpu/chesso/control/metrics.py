from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np

from vqpu.chesso.core.hypergraph import EntanglementHypergraph
from vqpu.chesso.core.state import HilbertBundleState
from vqpu.chesso.core.types import ComplexArray, SectorId, StateRepresentation
from vqpu.chesso.ops.unitary_ops import sector_indices


@dataclass(slots=True)
class BranchStatistics:
    """Summary statistics over computational-basis branch populations."""

    probabilities: np.ndarray
    effective_support: int
    effective_rank: float
    max_probability: float
    min_nonzero_probability: float
    shannon_entropy_bits: float
    top_outcomes: Tuple[Tuple[int, float], ...] = ()


@dataclass(slots=True)
class ResourceStatistics:
    """Runtime and state-size bookkeeping useful for CHESSO control."""

    active_sector_count: int
    total_dimension: int
    step: int
    dynamic_depth: int
    measurements_used: int
    branches_used: int
    discarded_trace_mass: float


@dataclass(slots=True)
class HypergraphRewardReport:
    """Proxy reward built from multipartite information on hyperedges."""

    total_reward: float
    edge_rewards: Dict[str, float] = field(default_factory=dict)


@dataclass(slots=True)
class BundleMetricReport:
    """High-level metrics bundle used by later CHESSO policy layers."""

    global_entropy_bits: float
    global_purity: float
    basis_entropy_bits: float
    l1_coherence: float
    effective_rank: float
    branch_statistics: BranchStatistics
    resource_statistics: ResourceStatistics
    fidelity_to_target: float | None = None
    hypergraph_reward: HypergraphRewardReport | None = None


# -----------------------------------------------------------------------------
# Internal helpers
# -----------------------------------------------------------------------------


def _density_matrix(bundle: HilbertBundleState) -> ComplexArray:
    return bundle.quantum_state.as_density_matrix()



def _hermitian_eigvals(rho: np.ndarray) -> np.ndarray:
    vals = np.linalg.eigvalsh(np.asarray(rho, dtype=np.complex128))
    vals = np.real_if_close(vals).astype(np.float64)
    vals[vals < 0.0] = 0.0
    total = float(np.sum(vals))
    if total > 0.0:
        vals /= total
    return vals



def _matrix_sqrt_psd(rho: np.ndarray, *, tol: float = 1e-12) -> ComplexArray:
    vals, vecs = np.linalg.eigh(np.asarray(rho, dtype=np.complex128))
    vals = np.real_if_close(vals).astype(np.float64)
    vals[vals < tol] = 0.0
    root = np.diag(np.sqrt(vals).astype(np.complex128))
    return vecs @ root @ vecs.conj().T



def _resolve_target_indices(bundle: HilbertBundleState, targets: Sequence[int | str | SectorId] | None) -> Tuple[int, ...]:
    if targets is None:
        return tuple(range(bundle.active_sector_count))
    return sector_indices(bundle, targets)



def reduced_density_matrix(
    bundle: HilbertBundleState,
    targets: Sequence[int | str | SectorId] | None = None,
) -> ComplexArray:
    """Return the reduced density matrix on the requested sectors.

    The returned subsystem order follows the `targets` order when targets are
    provided. When `targets` is None, the full density matrix is returned.
    """
    idx = _resolve_target_indices(bundle, targets)
    if not idx:
        return np.ones((1, 1), dtype=np.complex128)

    rho = _density_matrix(bundle)
    dims = tuple(int(d) for d in bundle.dims)
    n = len(dims)
    if len(idx) == n:
        return np.array(rho, copy=True)

    target_dims = tuple(dims[i] for i in idx)
    target_dim = int(np.prod(target_dims, dtype=np.int64))
    remainder = tuple(i for i in range(n) if i not in idx)
    rest_dims = tuple(dims[i] for i in remainder)
    rest_dim = int(np.prod(rest_dims, dtype=np.int64)) if rest_dims else 1

    perm = idx + remainder
    tensor = np.asarray(rho, dtype=np.complex128).reshape(dims + dims)
    full_perm = perm + tuple(n + p for p in perm)
    block = tensor.transpose(full_perm).reshape(target_dim, rest_dim, target_dim, rest_dim)
    reduced = np.trace(block, axis1=1, axis2=3)
    return np.asarray(reduced, dtype=np.complex128)


# -----------------------------------------------------------------------------
# Scalar metrics
# -----------------------------------------------------------------------------


def purity(bundle: HilbertBundleState, targets: Sequence[int | str | SectorId] | None = None) -> float:
    rho = reduced_density_matrix(bundle, targets)
    return float(np.real_if_close(np.trace(rho @ rho)))



def von_neumann_entropy(
    bundle: HilbertBundleState,
    targets: Sequence[int | str | SectorId] | None = None,
    *,
    base: float = 2.0,
) -> float:
    vals = _hermitian_eigvals(reduced_density_matrix(bundle, targets))
    nz = vals[vals > 0.0]
    if nz.size == 0:
        return 0.0
    logs = np.log(nz) / np.log(base)
    return float(-np.sum(nz * logs))



def basis_probabilities(bundle: HilbertBundleState) -> np.ndarray:
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



def shannon_entropy(probabilities: Sequence[float], *, base: float = 2.0) -> float:
    p = np.asarray(probabilities, dtype=np.float64)
    p = p[p > 0.0]
    if p.size == 0:
        return 0.0
    logs = np.log(p) / np.log(base)
    return float(-np.sum(p * logs))



def basis_entropy(bundle: HilbertBundleState, *, base: float = 2.0) -> float:
    return shannon_entropy(basis_probabilities(bundle), base=base)



def l1_coherence(bundle: HilbertBundleState, targets: Sequence[int | str | SectorId] | None = None) -> float:
    rho = reduced_density_matrix(bundle, targets)
    offdiag = np.asarray(rho, dtype=np.complex128) - np.diag(np.diag(rho))
    return float(np.sum(np.abs(offdiag)))



def mutual_information(
    bundle: HilbertBundleState,
    a: Sequence[int | str | SectorId],
    b: Sequence[int | str | SectorId],
    *,
    base: float = 2.0,
) -> float:
    idx_a = sector_indices(bundle, a)
    idx_b = sector_indices(bundle, b)
    if set(idx_a) & set(idx_b):
        raise ValueError("mutual_information requires disjoint target sets")
    s_a = von_neumann_entropy(bundle, idx_a, base=base)
    s_b = von_neumann_entropy(bundle, idx_b, base=base)
    s_ab = von_neumann_entropy(bundle, idx_a + idx_b, base=base)
    return float(s_a + s_b - s_ab)



def state_fidelity(
    bundle: HilbertBundleState,
    target: HilbertBundleState | np.ndarray | Sequence[complex],
) -> float:
    rho = _density_matrix(bundle)
    dim = rho.shape[0]

    if isinstance(target, HilbertBundleState):
        sigma = target.quantum_state.as_density_matrix()
    else:
        arr = np.asarray(target, dtype=np.complex128)
        if arr.ndim == 1:
            if arr.shape[0] != dim:
                raise ValueError(f"Target statevector length {arr.shape[0]} does not match dimension {dim}")
            sigma = np.outer(arr, arr.conj())
        elif arr.ndim == 2:
            if arr.shape != (dim, dim):
                raise ValueError(f"Target density shape {arr.shape} does not match {(dim, dim)}")
            sigma = arr
        else:
            raise ValueError("target must be a HilbertBundleState, statevector, or density matrix")

    # Pure-target shortcut.
    vals = _hermitian_eigvals(sigma)
    rank = int(np.count_nonzero(vals > 1e-12))
    if rank == 1:
        eigvals, eigvecs = np.linalg.eigh(np.asarray(sigma, dtype=np.complex128))
        psi = eigvecs[:, int(np.argmax(np.real(eigvals)))]
        fid = np.vdot(psi, rho @ psi)
        return float(np.real_if_close(fid))

    sqrt_rho = _matrix_sqrt_psd(rho)
    middle = sqrt_rho @ np.asarray(sigma, dtype=np.complex128) @ sqrt_rho
    root = _matrix_sqrt_psd(middle)
    return float(np.real_if_close(np.trace(root)) ** 2)


# -----------------------------------------------------------------------------
# CHESSO-specific proxies
# -----------------------------------------------------------------------------


def branch_statistics(bundle: HilbertBundleState, *, top_k: int = 5) -> BranchStatistics:
    probs = basis_probabilities(bundle)
    nz = probs[probs > bundle.config.numerical_tolerance]
    entropy_bits = shannon_entropy(probs, base=2.0)
    effective_rank = float(2.0 ** entropy_bits)
    order = np.argsort(probs)[::-1][: max(1, int(top_k))]
    top = tuple((int(i), float(probs[i])) for i in order if probs[i] > 0.0)
    return BranchStatistics(
        probabilities=np.array(probs, copy=True),
        effective_support=int(np.count_nonzero(probs > bundle.config.numerical_tolerance)),
        effective_rank=effective_rank,
        max_probability=float(np.max(probs)) if probs.size else 0.0,
        min_nonzero_probability=float(np.min(nz)) if nz.size else 0.0,
        shannon_entropy_bits=entropy_bits,
        top_outcomes=top,
    )



def resource_statistics(bundle: HilbertBundleState) -> ResourceStatistics:
    return ResourceStatistics(
        active_sector_count=bundle.active_sector_count,
        total_dimension=bundle.total_dimension,
        step=int(bundle.stats.step),
        dynamic_depth=int(bundle.stats.dynamic_depth),
        measurements_used=int(bundle.stats.measurements_used),
        branches_used=int(bundle.stats.branches_used),
        discarded_trace_mass=float(bundle.stats.discarded_trace_mass),
    )



def hypergraph_reward(
    bundle: HilbertBundleState,
    graph: EntanglementHypergraph,
    *,
    base: float = 2.0,
) -> HypergraphRewardReport:
    edge_scores: Dict[str, float] = {}
    total = 0.0
    for edge_id, edge in graph.hyperedges.items():
        members = tuple(str(member) for member in edge.members)
        order = len(members)
        if order < 2:
            edge_scores[edge_id] = 0.0
            continue

        s_joint = von_neumann_entropy(bundle, members, base=base)
        s_marginals = sum(von_neumann_entropy(bundle, [member], base=base) for member in members)
        total_correlation = max(0.0, s_marginals - s_joint)
        phase_gain = max(0.0, float(np.cos(edge.phase_bias)))
        order_gain = float(order / max(2, graph.max_order))
        reward = float(edge.weight * edge.coherence_score * edge.capacity * phase_gain * order_gain * total_correlation)
        edge_scores[edge_id] = reward
        total += reward
    return HypergraphRewardReport(total_reward=float(total), edge_rewards=edge_scores)



def bundle_metric_report(
    bundle: HilbertBundleState,
    *,
    graph: EntanglementHypergraph | None = None,
    target: HilbertBundleState | np.ndarray | Sequence[complex] | None = None,
    top_k: int = 5,
) -> BundleMetricReport:
    branches = branch_statistics(bundle, top_k=top_k)
    resources = resource_statistics(bundle)
    return BundleMetricReport(
        global_entropy_bits=von_neumann_entropy(bundle),
        global_purity=purity(bundle),
        basis_entropy_bits=basis_entropy(bundle),
        l1_coherence=l1_coherence(bundle),
        effective_rank=branches.effective_rank,
        branch_statistics=branches,
        resource_statistics=resources,
        fidelity_to_target=None if target is None else state_fidelity(bundle, target),
        hypergraph_reward=None if graph is None else hypergraph_reward(bundle, graph),
    )


__all__ = [
    "BranchStatistics",
    "ResourceStatistics",
    "HypergraphRewardReport",
    "BundleMetricReport",
    "reduced_density_matrix",
    "purity",
    "von_neumann_entropy",
    "basis_probabilities",
    "shannon_entropy",
    "basis_entropy",
    "l1_coherence",
    "mutual_information",
    "state_fidelity",
    "branch_statistics",
    "resource_statistics",
    "hypergraph_reward",
    "bundle_metric_report",
]
