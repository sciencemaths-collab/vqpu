from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence, Tuple

import numpy as np

from core import HilbertBundleState, QuantumState, StateRepresentation
from core.config import NoiseConfig
from core.types import ComplexArray, SectorId
from .unitary_ops import _permutation, apply_local_operator, rotation_z, sector_indices, target_dimensions


@dataclass(slots=True, frozen=True)
class KrausChannel:
    """Finite-dimensional CPTP channel represented by Kraus operators."""

    name: str
    kraus_ops: Tuple[ComplexArray, ...]
    target_dim: int
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.kraus_ops:
            raise ValueError("kraus_ops must not be empty")
        if self.target_dim <= 0:
            raise ValueError("target_dim must be positive")

        normalized_ops = []
        for idx, op in enumerate(self.kraus_ops):
            mat = np.asarray(op, dtype=np.complex128)
            if mat.shape != (self.target_dim, self.target_dim):
                raise ValueError(
                    f"Kraus operator {idx} has shape {mat.shape}, expected {(self.target_dim, self.target_dim)}"
                )
            normalized_ops.append(mat)
        object.__setattr__(self, "kraus_ops", tuple(normalized_ops))
        self.validate_trace_preserving()

    def validate_trace_preserving(self, tol: float = 1e-10) -> None:
        accum = np.zeros((self.target_dim, self.target_dim), dtype=np.complex128)
        for op in self.kraus_ops:
            accum += op.conj().T @ op
        if not np.allclose(accum, np.eye(self.target_dim, dtype=np.complex128), atol=tol, rtol=0.0):
            raise ValueError(f"Channel {self.name!r} is not trace preserving within tolerance {tol}")


# -----------------------------------------------------------------------------
# Built-in channel factories
# -----------------------------------------------------------------------------


def identity_channel(dim: int) -> KrausChannel:
    eye = np.eye(int(dim), dtype=np.complex128)
    return KrausChannel(name=f"identity_d{dim}", kraus_ops=(eye,), target_dim=int(dim))



def depolarizing_channel(rate: float) -> KrausChannel:
    if not 0.0 <= rate <= 1.0:
        raise ValueError(f"depolarizing rate must be in [0, 1], got {rate!r}")
    if rate == 0.0:
        return identity_channel(2)

    from .unitary_ops import pauli_x, pauli_y, pauli_z

    p = float(rate)
    k0 = np.sqrt(max(0.0, 1.0 - p)) * np.eye(2, dtype=np.complex128)
    scale = np.sqrt(p / 3.0)
    return KrausChannel(
        name="depolarizing",
        kraus_ops=(k0, scale * pauli_x(), scale * pauli_y(), scale * pauli_z()),
        target_dim=2,
        metadata={"rate": p},
    )



def amplitude_damping_channel(gamma: float) -> KrausChannel:
    if not 0.0 <= gamma <= 1.0:
        raise ValueError(f"amplitude damping gamma must be in [0, 1], got {gamma!r}")
    if gamma == 0.0:
        return identity_channel(2)

    g = float(gamma)
    k0 = np.array([[1.0, 0.0], [0.0, np.sqrt(1.0 - g)]], dtype=np.complex128)
    k1 = np.array([[0.0, np.sqrt(g)], [0.0, 0.0]], dtype=np.complex128)
    return KrausChannel(
        name="amplitude_damping",
        kraus_ops=(k0, k1),
        target_dim=2,
        metadata={"gamma": g},
    )



def phase_damping_channel(gamma: float) -> KrausChannel:
    if not 0.0 <= gamma <= 1.0:
        raise ValueError(f"phase damping gamma must be in [0, 1], got {gamma!r}")
    if gamma == 0.0:
        return identity_channel(2)

    g = float(gamma)
    k0 = np.sqrt(1.0 - g) * np.eye(2, dtype=np.complex128)
    k1 = np.sqrt(g) * np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.complex128)
    k2 = np.sqrt(g) * np.array([[0.0, 0.0], [0.0, 1.0]], dtype=np.complex128)
    return KrausChannel(
        name="phase_damping",
        kraus_ops=(k0, k1, k2),
        target_dim=2,
        metadata={"gamma": g},
    )



def compose_channels(*channels: KrausChannel, name: str | None = None) -> KrausChannel:
    if not channels:
        raise ValueError("compose_channels requires at least one channel")
    target_dim = channels[0].target_dim
    for channel in channels:
        if channel.target_dim != target_dim:
            raise ValueError("All composed channels must have the same target_dim")

    ops = (np.eye(target_dim, dtype=np.complex128),)
    for channel in channels:
        next_ops = []
        for b in channel.kraus_ops:
            for a in ops:
                next_ops.append(b @ a)
        ops = tuple(next_ops)
    return KrausChannel(
        name=name or " o ".join(ch.name for ch in channels),
        kraus_ops=ops,
        target_dim=target_dim,
        metadata={"composed_from": tuple(ch.name for ch in channels)},
    )


# -----------------------------------------------------------------------------
# Numerical application kernels
# -----------------------------------------------------------------------------


def apply_channel_to_density_matrix(
    density_matrix: np.ndarray,
    dims: Sequence[int],
    channel: KrausChannel,
    targets: Sequence[int],
) -> ComplexArray:
    rho = np.asarray(density_matrix, dtype=np.complex128)
    dims_t = tuple(int(d) for d in dims)
    n = len(dims_t)
    target_t = tuple(int(t) for t in targets)
    target_dims = tuple(dims_t[t] for t in target_t)
    target_dim = int(np.prod(target_dims, dtype=np.int64)) if target_dims else 1
    if target_dim != channel.target_dim:
        raise ValueError(
            f"Channel target_dim {channel.target_dim} does not match target dimensions {target_dims} -> {target_dim}"
        )

    rest_dims = tuple(dims_t[i] for i in range(n) if i not in target_t)
    rest_dim = int(np.prod(rest_dims, dtype=np.int64)) if rest_dims else 1

    bra_perm, bra_inv = _permutation(n, target_t)
    full_perm = bra_perm + tuple(n + idx for idx in bra_perm)
    full_inv = bra_inv + tuple(n + idx for idx in bra_inv)

    tensor = rho.reshape(dims_t + dims_t).transpose(full_perm)
    block = tensor.reshape(target_dim, rest_dim, target_dim, rest_dim)
    updated = np.zeros_like(block)
    for op in channel.kraus_ops:
        updated += np.einsum("ai,irjs,bj->arbs", op, block, op.conj(), optimize=True)
    restored = updated.reshape(target_dims + rest_dims + target_dims + rest_dims).transpose(full_inv)
    return restored.reshape(rho.shape)


# -----------------------------------------------------------------------------
# Bundle-facing APIs
# -----------------------------------------------------------------------------


def _promote_statevector_to_density_matrix(bundle: HilbertBundleState) -> None:
    if bundle.quantum_state.representation == StateRepresentation.DENSITY_MATRIX:
        return
    rho = bundle.quantum_state.as_density_matrix()
    bundle.quantum_state = QuantumState(
        rho,
        StateRepresentation.DENSITY_MATRIX,
        bundle.quantum_state.dims,
        bundle.config.numerical_tolerance,
    )
    bundle.metadata["state_promoted_to_density_matrix"] = True



def apply_channel(
    bundle: HilbertBundleState,
    channel: KrausChannel,
    targets: Sequence[int | str | SectorId],
    *,
    in_place: bool = True,
    label: str | None = None,
    promote_statevector: bool = True,
) -> HilbertBundleState:
    target_idx = sector_indices(bundle, targets)
    dims = target_dimensions(bundle, targets)
    expected_dim = int(np.prod(dims, dtype=np.int64)) if dims else 1
    if expected_dim != channel.target_dim:
        raise ValueError(
            f"Channel target_dim {channel.target_dim} does not match target dimensions {dims} -> {expected_dim}"
        )

    work = bundle if in_place else bundle.copy()
    if work.quantum_state.representation == StateRepresentation.STATEVECTOR:
        if not promote_statevector:
            raise ValueError("Noise channels on statevectors require promote_statevector=True")
        _promote_statevector_to_density_matrix(work)

    updated = apply_channel_to_density_matrix(work.quantum_state.data, work.dims, channel, target_idx)
    work.quantum_state = QuantumState(
        updated,
        StateRepresentation.DENSITY_MATRIX,
        work.quantum_state.dims,
        work.config.numerical_tolerance,
    )
    work.advance_step(dynamic_depth_increment=1)

    history = work.metadata.setdefault("channel_history", [])
    if isinstance(history, list):
        history.append(
            {
                "label": label or channel.name,
                "channel": channel.name,
                "targets": tuple(str(work.topology.sectors[i].sector_id) for i in target_idx),
                "kraus_rank": len(channel.kraus_ops),
            }
        )
    return work



def apply_default_noise(
    bundle: HilbertBundleState,
    targets: Sequence[int | str | SectorId],
    *,
    noise: NoiseConfig | None = None,
    in_place: bool = True,
    label_prefix: str = "noise",
) -> HilbertBundleState:
    work = bundle if in_place else bundle.copy()
    cfg = noise or work.config.noise
    target_idx = sector_indices(work, targets)
    target_dims = tuple(work.dims[i] for i in target_idx)

    for idx, dim in zip(target_idx, target_dims):
        if dim != 2:
            continue
        target_name = str(work.topology.sectors[idx].sector_id)
        if cfg.depolarizing_rate > 0.0:
            apply_channel(work, depolarizing_channel(cfg.depolarizing_rate), [idx], label=f"{label_prefix}:depolarizing:{target_name}")
        if cfg.amplitude_damping > 0.0:
            apply_channel(work, amplitude_damping_channel(cfg.amplitude_damping), [idx], label=f"{label_prefix}:amplitude:{target_name}")
        if cfg.phase_damping > 0.0:
            apply_channel(work, phase_damping_channel(cfg.phase_damping), [idx], label=f"{label_prefix}:phase:{target_name}")
    return work



def apply_coherent_overrotation(
    bundle: HilbertBundleState,
    targets: Sequence[int | str | SectorId],
    *,
    strength: float | None = None,
    in_place: bool = True,
    label: str = "coherent_overrotation",
) -> HilbertBundleState:
    work = bundle if in_place else bundle.copy()
    epsilon = float(work.config.noise.coherent_overrotation if strength is None else strength)
    if epsilon <= 0.0:
        return work

    target_idx = sector_indices(work, targets)
    for idx in target_idx:
        if work.dims[idx] != 2:
            continue
        apply_local_operator(work, rotation_z(np.pi * epsilon), [idx], label=f"{label}:{work.topology.sectors[idx].sector_id}")
    return work



def apply_noisy_local_operator(
    bundle: HilbertBundleState,
    operator: np.ndarray,
    targets: Sequence[int | str | SectorId],
    *,
    in_place: bool = True,
    label: str | None = None,
    noise: NoiseConfig | None = None,
) -> HilbertBundleState:
    work = bundle if in_place else bundle.copy()
    apply_local_operator(work, operator, targets, label=label)
    cfg = noise or work.config.noise
    if cfg.coherent_overrotation > 0.0:
        apply_coherent_overrotation(work, targets, strength=cfg.coherent_overrotation, label=f"{label or 'op'}:overrotation")
    if any((cfg.depolarizing_rate, cfg.amplitude_damping, cfg.phase_damping)):
        apply_default_noise(work, targets, noise=cfg, label_prefix=label or "op")
    return work


# -----------------------------------------------------------------------------
# Readout noise hooks
# -----------------------------------------------------------------------------


def single_qubit_readout_confusion(error: float) -> np.ndarray:
    if not 0.0 <= error <= 1.0:
        raise ValueError(f"readout error must be in [0, 1], got {error!r}")
    e = float(error)
    return np.array([[1.0 - e, e], [e, 1.0 - e]], dtype=np.float64)



def apply_readout_confusion(probabilities: Sequence[float], confusion: np.ndarray) -> np.ndarray:
    probs = np.asarray(probabilities, dtype=np.float64)
    if probs.ndim != 1:
        raise ValueError("probabilities must be a rank-1 array")
    if np.any(probs < -1e-12):
        raise ValueError("probabilities must be non-negative")
    probs = np.clip(probs, 0.0, None)
    total = float(np.sum(probs))
    if total <= 0.0:
        raise ValueError("probabilities must sum to a positive value")
    probs /= total

    confusion_t = np.asarray(confusion, dtype=np.float64)
    if confusion_t.shape != (probs.size, probs.size):
        raise ValueError(
            f"confusion matrix shape {confusion_t.shape} does not match probabilities of length {probs.size}"
        )
    observed = confusion_t @ probs
    observed = np.clip(observed, 0.0, None)
    observed /= np.sum(observed)
    return observed



def sample_noisy_readout(
    probabilities: Sequence[float],
    *,
    error: float,
    rng: np.random.Generator | None = None,
) -> int:
    generator = rng or np.random.default_rng()
    confusion = single_qubit_readout_confusion(error)
    observed = apply_readout_confusion(probabilities, confusion)
    return int(generator.choice(len(observed), p=observed))


__all__ = [
    "KrausChannel",
    "identity_channel",
    "depolarizing_channel",
    "amplitude_damping_channel",
    "phase_damping_channel",
    "compose_channels",
    "apply_channel_to_density_matrix",
    "apply_channel",
    "apply_default_noise",
    "apply_coherent_overrotation",
    "apply_noisy_local_operator",
    "single_qubit_readout_confusion",
    "apply_readout_confusion",
    "sample_noisy_readout",
]
