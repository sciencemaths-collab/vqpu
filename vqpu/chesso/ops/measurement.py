from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np

from vqpu.chesso.core.state import HilbertBundleState, QuantumState
from vqpu.chesso.core.types import ComplexArray, MeasurementRecord, SectorId, StateRepresentation
from .noise import apply_readout_confusion
from .unitary_ops import (
    _complex_matrix,
    apply_operator_to_density_matrix,
    apply_operator_to_statevector,
    sector_indices,
    target_dimensions,
)


@dataclass(slots=True)
class POVMInstrument:
    """Finite measurement instrument represented by outcome Kraus operators.

    The instrument is target-local. Each Kraus operator acts on the same target
    Hilbert space. Outcome probabilities are computed from `M_k rho M_k^†`.
    """

    name: str
    kraus_ops: Tuple[ComplexArray, ...]
    target_dim: int
    outcome_labels: Tuple[str, ...] = ()
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.kraus_ops:
            raise ValueError("POVMInstrument requires at least one Kraus operator")
        if self.target_dim <= 0:
            raise ValueError("target_dim must be positive")
        normalized_ops: List[ComplexArray] = []
        for op in self.kraus_ops:
            mat = _complex_matrix(op)
            if mat.shape != (self.target_dim, self.target_dim):
                raise ValueError(
                    f"Kraus operator shape {mat.shape} does not match target_dim {self.target_dim}"
                )
            normalized_ops.append(mat)
        self.kraus_ops = tuple(normalized_ops)
        if self.outcome_labels and len(self.outcome_labels) != len(self.kraus_ops):
            raise ValueError(
                f"outcome_labels has length {len(self.outcome_labels)} but expected {len(self.kraus_ops)}"
            )
        if not self.outcome_labels:
            self.outcome_labels = tuple(str(i) for i in range(len(self.kraus_ops)))

        completeness = np.zeros((self.target_dim, self.target_dim), dtype=np.complex128)
        for op in self.kraus_ops:
            completeness += op.conj().T @ op
        ident = np.eye(self.target_dim, dtype=np.complex128)
        tol = float(self.metadata.get("tolerance", 1e-10))
        if not np.allclose(completeness, ident, atol=tol, rtol=0.0):
            raise ValueError("Kraus operators do not satisfy completeness relation Σ M†M = I")

    @property
    def num_outcomes(self) -> int:
        return len(self.kraus_ops)

    @property
    def effects(self) -> Tuple[ComplexArray, ...]:
        return tuple(op.conj().T @ op for op in self.kraus_ops)


# -----------------------------------------------------------------------------
# Constructors
# -----------------------------------------------------------------------------


def _validate_strength(strength: float) -> float:
    s = float(strength)
    if not 0.0 <= s <= 1.0:
        raise ValueError(f"measurement strength must be in [0, 1], got {strength!r}")
    return s



def _validate_projectors(projectors: Sequence[np.ndarray], *, tol: float = 1e-10) -> Tuple[ComplexArray, ...]:
    if not projectors:
        raise ValueError("projectors must not be empty")
    mats = tuple(_complex_matrix(p) for p in projectors)
    dim = mats[0].shape[0]
    ident = np.eye(dim, dtype=np.complex128)
    accum = np.zeros_like(ident)
    for p in mats:
        if p.shape != (dim, dim):
            raise ValueError("All projectors must have the same shape")
        if not np.allclose(p, p.conj().T, atol=tol, rtol=0.0):
            raise ValueError("Projectors must be Hermitian")
        if not np.allclose(p @ p, p, atol=tol, rtol=0.0):
            raise ValueError("Projectors must be idempotent")
        accum += p
    if not np.allclose(accum, ident, atol=tol, rtol=0.0):
        raise ValueError("Projectors must form a complete resolution of the identity")
    return mats



def make_soft_projective_instrument(
    projectors: Sequence[np.ndarray],
    *,
    strength: float,
    labels: Sequence[str] | None = None,
    name: str = "soft_projective",
    metadata: Dict[str, Any] | None = None,
) -> POVMInstrument:
    """Build a tunable projective-like instrument.

    For a complete projector family {P_y}, define the effect operators:

        E_y = ((1-s)/m) I + s P_y

    where m is the number of outcomes and s ∈ [0,1] is the measurement
    strength. The corresponding Kraus operator is chosen as sqrt(E_y), which is
    closed-form because each P_y is a projector.

    - s = 0 gives a non-informative, minimally disturbing instrument.
    - s = 1 gives the sharp projective measurement.
    """

    s = _validate_strength(strength)
    mats = _validate_projectors(projectors)
    dim = mats[0].shape[0]
    m = len(mats)
    a = (1.0 - s) / m
    ident = np.eye(dim, dtype=np.complex128)
    kraus_ops = []
    for p in mats:
        # sqrt(a I + s P) = sqrt(a)(I-P) + sqrt(a+s)P
        op = np.sqrt(a) * (ident - p) + np.sqrt(a + s) * p
        kraus_ops.append(op)
    return POVMInstrument(
        name=name,
        kraus_ops=tuple(kraus_ops),
        target_dim=dim,
        outcome_labels=tuple(labels) if labels is not None else (),
        metadata={"strength": s, **(metadata or {})},
    )



def computational_basis_projectors(target_dims: Sequence[int]) -> Tuple[ComplexArray, ...]:
    dims = tuple(int(d) for d in target_dims)
    if not dims:
        raise ValueError("target_dims must not be empty")
    if any(d <= 0 for d in dims):
        raise ValueError(f"All target dimensions must be positive, got {dims}")
    total_dim = int(np.prod(dims, dtype=np.int64))
    projectors: List[ComplexArray] = []
    for idx in range(total_dim):
        p = np.zeros((total_dim, total_dim), dtype=np.complex128)
        p[idx, idx] = 1.0
        projectors.append(p)
    return tuple(projectors)



def _basis_labels(target_dims: Sequence[int]) -> Tuple[str, ...]:
    dims = tuple(int(d) for d in target_dims)
    if all(d == 2 for d in dims):
        width = len(dims)
        return tuple(format(i, f"0{width}b") for i in range(2 ** width))
    return tuple(str(multi) for multi in np.ndindex(dims))



def make_computational_basis_instrument(
    target_dims: Sequence[int],
    *,
    strength: float = 1.0,
    name: str = "computational_basis",
    metadata: Dict[str, Any] | None = None,
) -> POVMInstrument:
    projectors = computational_basis_projectors(target_dims)
    return make_soft_projective_instrument(
        projectors,
        strength=strength,
        labels=_basis_labels(target_dims),
        name=name,
        metadata={"target_dims": tuple(int(d) for d in target_dims), **(metadata or {})},
    )


# -----------------------------------------------------------------------------
# Probability evaluation
# -----------------------------------------------------------------------------


def _normalize_probabilities(probabilities: np.ndarray, tol: float) -> np.ndarray:
    probs = np.asarray(probabilities, dtype=np.float64)
    probs = np.clip(np.real_if_close(probs), 0.0, None)
    total = float(np.sum(probs))
    if total <= tol:
        raise ValueError("Measurement probabilities sum to zero")
    probs /= total
    return probs



def evaluate_measurement_probabilities(
    bundle: HilbertBundleState,
    instrument: POVMInstrument,
    targets: Sequence[int | str | SectorId],
) -> np.ndarray:
    target_idx = sector_indices(bundle, targets)
    dims = target_dimensions(bundle, targets)
    expected_dim = int(np.prod(dims, dtype=np.int64)) if dims else 1
    if expected_dim != instrument.target_dim:
        raise ValueError(
            f"Instrument target_dim {instrument.target_dim} does not match target dimensions {dims} -> {expected_dim}"
        )

    probs: List[float] = []
    if bundle.quantum_state.representation == StateRepresentation.STATEVECTOR:
        vec = bundle.quantum_state.data
        for op in instrument.kraus_ops:
            branch = apply_operator_to_statevector(vec, bundle.dims, op, target_idx)
            p = float(np.real_if_close(np.vdot(branch, branch)))
            probs.append(p)
    else:
        rho = bundle.quantum_state.data
        for op in instrument.kraus_ops:
            branch = apply_operator_to_density_matrix(rho, bundle.dims, op, target_idx)
            p = float(np.real_if_close(np.trace(branch)))
            probs.append(p)

    return _normalize_probabilities(np.asarray(probs, dtype=np.float64), bundle.config.numerical_tolerance)


# -----------------------------------------------------------------------------
# Measurement application
# -----------------------------------------------------------------------------


def _store_measurement_value(bundle: HilbertBundleState, key: str | None, value: Any) -> None:
    if key:
        bundle.classical_memory.put(key, value)



def _record_measurement(
    bundle: HilbertBundleState,
    *,
    label: str,
    strength: float,
    outcome: int | None,
    observed_probabilities: np.ndarray,
    ideal_probabilities: np.ndarray,
    targets: Sequence[int],
    instrument: POVMInstrument,
    metadata: Dict[str, Any] | None = None,
) -> MeasurementRecord:
    record = MeasurementRecord(
        label=label,
        strength=strength,
        outcome=outcome,
        probabilities=np.array(observed_probabilities, copy=True),
        metadata={
            "ideal_probabilities": np.array(ideal_probabilities, copy=True),
            "instrument": instrument.name,
            "outcome_labels": tuple(instrument.outcome_labels),
            "targets": tuple(str(bundle.topology.sectors[i].sector_id) for i in targets),
            **(metadata or {}),
        },
    )
    bundle.record_measurement(record)
    return record



def apply_selective_measurement(
    bundle: HilbertBundleState,
    instrument: POVMInstrument,
    targets: Sequence[int | str | SectorId],
    *,
    outcome: int | None = None,
    sample: bool = False,
    in_place: bool = True,
    label: str | None = None,
    rng: np.random.Generator | None = None,
    readout_confusion: np.ndarray | None = None,
    store_key: str | None = None,
    metadata: Dict[str, Any] | None = None,
) -> MeasurementRecord:
    work = bundle if in_place else bundle.copy()
    target_idx = sector_indices(work, targets)
    ideal_probs = evaluate_measurement_probabilities(work, instrument, target_idx)
    observed_probs = ideal_probs if readout_confusion is None else apply_readout_confusion(ideal_probs, readout_confusion)

    if outcome is None:
        if sample:
            generator = rng or np.random.default_rng(work.config.random_seed)
            selected = int(generator.choice(len(observed_probs), p=observed_probs))
        else:
            selected = int(np.argmax(observed_probs))
    else:
        selected = int(outcome)
    if selected < 0 or selected >= instrument.num_outcomes:
        raise IndexError(f"Outcome index {selected} out of range for {instrument.num_outcomes} outcomes")

    op = instrument.kraus_ops[selected]
    if work.quantum_state.representation == StateRepresentation.STATEVECTOR:
        updated = apply_operator_to_statevector(work.quantum_state.data, work.dims, op, target_idx)
        prob = float(np.real_if_close(np.vdot(updated, updated)))
        if prob <= work.config.numerical_tolerance:
            raise ValueError(f"Selected outcome {selected} has near-zero probability")
        updated = updated / np.sqrt(prob)
        work.quantum_state = QuantumState(
            updated,
            StateRepresentation.STATEVECTOR,
            work.quantum_state.dims,
            work.config.numerical_tolerance,
        )
    else:
        branch = apply_operator_to_density_matrix(work.quantum_state.data, work.dims, op, target_idx)
        prob = float(np.real_if_close(np.trace(branch)))
        if prob <= work.config.numerical_tolerance:
            raise ValueError(f"Selected outcome {selected} has near-zero probability")
        branch = branch / prob
        work.quantum_state = QuantumState(
            branch,
            StateRepresentation.DENSITY_MATRIX,
            work.quantum_state.dims,
            work.config.numerical_tolerance,
        )

    work.advance_step(dynamic_depth_increment=1)
    strength = float(instrument.metadata.get("strength", work.config.default_measurement_strength))
    record = _record_measurement(
        work,
        label=label or instrument.name,
        strength=strength,
        outcome=selected,
        observed_probabilities=observed_probs,
        ideal_probabilities=ideal_probs,
        targets=target_idx,
        instrument=instrument,
        metadata={"mode": "selective", **(metadata or {})},
    )
    _store_measurement_value(work, store_key, selected)
    work.metadata["last_measurement_record"] = record
    return record



def apply_nonselective_measurement(
    bundle: HilbertBundleState,
    instrument: POVMInstrument,
    targets: Sequence[int | str | SectorId],
    *,
    in_place: bool = True,
    label: str | None = None,
    store_key: str | None = None,
    metadata: Dict[str, Any] | None = None,
) -> MeasurementRecord:
    work = bundle if in_place else bundle.copy()
    target_idx = sector_indices(work, targets)
    ideal_probs = evaluate_measurement_probabilities(work, instrument, target_idx)

    rho = work.quantum_state.as_density_matrix()
    updated = np.zeros_like(rho)
    for op in instrument.kraus_ops:
        updated += apply_operator_to_density_matrix(rho, work.dims, op, target_idx)
    work.quantum_state = QuantumState(
        updated,
        StateRepresentation.DENSITY_MATRIX,
        work.quantum_state.dims,
        work.config.numerical_tolerance,
    )
    if bundle.quantum_state.representation == StateRepresentation.STATEVECTOR:
        work.metadata["state_promoted_to_density_matrix"] = True

    work.advance_step(dynamic_depth_increment=1)
    strength = float(instrument.metadata.get("strength", work.config.default_measurement_strength))
    record = _record_measurement(
        work,
        label=label or instrument.name,
        strength=strength,
        outcome=None,
        observed_probabilities=ideal_probs,
        ideal_probabilities=ideal_probs,
        targets=target_idx,
        instrument=instrument,
        metadata={"mode": "nonselective", **(metadata or {})},
    )
    _store_measurement_value(work, store_key, np.array(ideal_probs, copy=True))
    work.metadata["last_measurement_record"] = record
    return record



def measure_computational_basis(
    bundle: HilbertBundleState,
    targets: Sequence[int | str | SectorId],
    *,
    strength: float = 1.0,
    outcome: int | None = None,
    sample: bool = False,
    selective: bool = True,
    in_place: bool = True,
    label: str | None = None,
    rng: np.random.Generator | None = None,
    readout_confusion: np.ndarray | None = None,
    store_key: str | None = None,
    metadata: Dict[str, Any] | None = None,
) -> MeasurementRecord:
    dims = target_dimensions(bundle, targets)
    instrument = make_computational_basis_instrument(dims, strength=strength)
    if selective:
        return apply_selective_measurement(
            bundle,
            instrument,
            targets,
            outcome=outcome,
            sample=sample,
            in_place=in_place,
            label=label or instrument.name,
            rng=rng,
            readout_confusion=readout_confusion,
            store_key=store_key,
            metadata=metadata,
        )
    return apply_nonselective_measurement(
        bundle,
        instrument,
        targets,
        in_place=in_place,
        label=label or instrument.name,
        store_key=store_key,
        metadata=metadata,
    )


# -----------------------------------------------------------------------------
# Lazy measurement hooks
# -----------------------------------------------------------------------------


def queue_lazy_measurement(
    bundle: HilbertBundleState,
    instrument: POVMInstrument,
    targets: Sequence[int | str | SectorId],
    *,
    label: str | None = None,
    selective: bool = True,
    outcome: int | None = None,
    sample: bool = False,
    store_key: str | None = None,
    readout_confusion: np.ndarray | None = None,
    metadata: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    pending = bundle.metadata.setdefault("pending_measurements", [])
    if not isinstance(pending, list):
        raise TypeError("bundle.metadata['pending_measurements'] must be a list when present")
    entry = {
        "instrument": instrument,
        "targets": tuple(targets),
        "label": label or instrument.name,
        "selective": bool(selective),
        "outcome": outcome,
        "sample": bool(sample),
        "store_key": store_key,
        "readout_confusion": None if readout_confusion is None else np.array(readout_confusion, copy=True),
        "metadata": dict(metadata or {}),
    }
    pending.append(entry)
    return entry



def queue_lazy_basis_measurement(
    bundle: HilbertBundleState,
    targets: Sequence[int | str | SectorId],
    *,
    strength: float = 1.0,
    label: str | None = None,
    selective: bool = True,
    outcome: int | None = None,
    sample: bool = False,
    store_key: str | None = None,
    readout_confusion: np.ndarray | None = None,
    metadata: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    instrument = make_computational_basis_instrument(target_dimensions(bundle, targets), strength=strength)
    return queue_lazy_measurement(
        bundle,
        instrument,
        targets,
        label=label or instrument.name,
        selective=selective,
        outcome=outcome,
        sample=sample,
        store_key=store_key,
        readout_confusion=readout_confusion,
        metadata=metadata,
    )



def flush_lazy_measurements(
    bundle: HilbertBundleState,
    *,
    in_place: bool = True,
    rng: np.random.Generator | None = None,
) -> List[MeasurementRecord]:
    work = bundle if in_place else bundle.copy()
    pending = work.metadata.get("pending_measurements", [])
    if not pending:
        return []
    if not isinstance(pending, list):
        raise TypeError("bundle.metadata['pending_measurements'] must be a list when present")

    records: List[MeasurementRecord] = []
    while pending:
        entry = pending.pop(0)
        instrument = entry["instrument"]
        targets = entry["targets"]
        if entry["selective"]:
            record = apply_selective_measurement(
                work,
                instrument,
                targets,
                outcome=entry["outcome"],
                sample=entry["sample"],
                in_place=True,
                label=entry["label"],
                rng=rng,
                readout_confusion=entry["readout_confusion"],
                store_key=entry["store_key"],
                metadata={"lazy": True, **entry.get("metadata", {})},
            )
        else:
            record = apply_nonselective_measurement(
                work,
                instrument,
                targets,
                in_place=True,
                label=entry["label"],
                store_key=entry["store_key"],
                metadata={"lazy": True, **entry.get("metadata", {})},
            )
        records.append(record)
    return records


__all__ = [
    "POVMInstrument",
    "make_soft_projective_instrument",
    "computational_basis_projectors",
    "make_computational_basis_instrument",
    "evaluate_measurement_probabilities",
    "apply_selective_measurement",
    "apply_nonselective_measurement",
    "measure_computational_basis",
    "queue_lazy_measurement",
    "queue_lazy_basis_measurement",
    "flush_lazy_measurements",
]
