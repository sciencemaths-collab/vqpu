from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray


ComplexArray = NDArray[np.complex128]
RealArray = NDArray[np.float64]
IndexTuple = Tuple[int, ...]


class SectorKind(str, Enum):
    """Kinds of Hilbert sectors managed by the bundle."""

    LOGICAL = "logical"
    ANCILLA = "ancilla"
    MEMORY = "memory"
    AUXILIARY = "auxiliary"


class StateRepresentation(str, Enum):
    """Numerical representation of the quantum state."""

    STATEVECTOR = "statevector"
    DENSITY_MATRIX = "density_matrix"


@dataclass(slots=True, frozen=True)
class SectorId:
    """Stable identifier for an active Hilbert sector."""

    name: str

    def __str__(self) -> str:
        return self.name


@dataclass(slots=True)
class SectorSpec:
    """Metadata for one active sector in the Hilbert bundle."""

    sector_id: SectorId
    dimension: int
    kind: SectorKind = SectorKind.LOGICAL
    tags: Tuple[str, ...] = ()
    metadata: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if self.dimension <= 0:
            raise ValueError(f"Sector {self.sector_id} has invalid dimension {self.dimension}")


@dataclass(slots=True)
class BundleTopology:
    """Ordered list of sectors that define the tensor-product layout."""

    sectors: List[SectorSpec] = field(default_factory=list)

    def validate(self) -> None:
        names = [str(sec.sector_id) for sec in self.sectors]
        if len(names) != len(set(names)):
            raise ValueError("Sector names must be unique within a topology")
        for sec in self.sectors:
            sec.validate()

    @property
    def dims(self) -> Tuple[int, ...]:
        return tuple(sec.dimension for sec in self.sectors)

    @property
    def total_dimension(self) -> int:
        total = 1
        for dim in self.dims:
            total *= dim
        return total

    def index_of(self, sector_id: str | SectorId) -> int:
        key = str(sector_id)
        for i, sec in enumerate(self.sectors):
            if str(sec.sector_id) == key:
                return i
        raise KeyError(f"Unknown sector_id: {key}")

    def has_sector(self, sector_id: str | SectorId) -> bool:
        try:
            self.index_of(sector_id)
            return True
        except KeyError:
            return False

    def copy(self) -> "BundleTopology":
        return BundleTopology(
            sectors=[
                SectorSpec(
                    sector_id=SectorId(str(sec.sector_id)),
                    dimension=sec.dimension,
                    kind=sec.kind,
                    tags=tuple(sec.tags),
                    metadata=dict(sec.metadata),
                )
                for sec in self.sectors
            ]
        )


@dataclass(slots=True)
class ClassicalMemory:
    """Mutable classical side memory for adaptive control."""

    registers: Dict[str, Any] = field(default_factory=dict)

    def put(self, key: str, value: Any) -> None:
        self.registers[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        return self.registers.get(key, default)

    def snapshot(self) -> Dict[str, Any]:
        return dict(self.registers)


@dataclass(slots=True)
class RuntimeStats:
    """Mutable counters tracked across sections of the runtime."""

    step: int = 0
    dynamic_depth: int = 0
    measurements_used: int = 0
    branches_used: int = 0
    discarded_trace_mass: float = 0.0

    def record_prune_loss(self, mass: float) -> None:
        self.discarded_trace_mass += float(mass)


@dataclass(slots=True)
class MeasurementRecord:
    """Structured record of a soft or hard measurement."""

    label: str
    strength: float
    outcome: Optional[int] = None
    probabilities: Optional[RealArray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class StateBundleSnapshot:
    """Serializable snapshot of the bundle state for logging or testing."""

    representation: StateRepresentation
    dims: Tuple[int, ...]
    sector_names: Tuple[str, ...]
    trace: float
    purity: float
    metadata: Dict[str, Any] = field(default_factory=dict)
