from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Mapping, Sequence, Tuple

from core import BackendKind
from core.types import SectorKind


class QTypeKind(str, Enum):
    """High-level Qλ type families used by the typed IR."""

    QUBIT = "qubit"
    QUDIT = "qudit"
    QREGISTER = "qregister"
    FIELD = "field"
    CLASSICAL = "classical"


@dataclass(slots=True, frozen=True)
class QLambdaType:
    """Typed surface-level object that maps into Hilbert-space dimensions."""

    kind: QTypeKind
    dims: Tuple[int, ...]
    label: str | None = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def total_dimension(self) -> int:
        total = 1
        for dim in self.dims:
            total *= int(dim)
        return total

    def validate(self) -> None:
        if not self.dims:
            raise ValueError("QLambdaType dims must not be empty")
        for dim in self.dims:
            if int(dim) <= 0:
                raise ValueError(f"QLambdaType contains non-positive dimension: {self.dims!r}")


@dataclass(slots=True, frozen=True)
class SectorDeclaration:
    """One declared sector in the typed IR symbol table."""

    name: str
    qtype: QLambdaType
    kind: SectorKind = SectorKind.LOGICAL
    tags: Tuple[str, ...] = ()
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def dimension(self) -> int:
        if len(self.qtype.dims) != 1:
            raise ValueError(f"SectorDeclaration {self.name!r} must map to a single local dimension")
        return int(self.qtype.dims[0])

    def validate(self) -> None:
        if not self.name:
            raise ValueError("SectorDeclaration name must be non-empty")
        self.qtype.validate()
        if len(self.qtype.dims) != 1:
            raise ValueError(f"SectorDeclaration {self.name!r} must have exactly one local dimension")


class InstructionKind(str, Enum):
    """Section 12 typed IR opcodes."""

    GATE = "gate"
    ENTANGLE = "entangle"
    EXPAND = "expand"
    MEASURE = "measure"
    RUN = "run"
    TARGET = "target"
    NOTE = "note"


@dataclass(slots=True, frozen=True)
class IRInstruction:
    """Typed instruction carried through the Section 12 compiler."""

    kind: InstructionKind
    targets: Tuple[str, ...] = ()
    params: Dict[str, Any] = field(default_factory=dict)
    label: str | None = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if self.kind in {InstructionKind.GATE, InstructionKind.ENTANGLE, InstructionKind.MEASURE} and not self.targets:
            raise ValueError(f"Instruction {self.kind.value} requires at least one target")
        if self.kind == InstructionKind.GATE and not self.label:
            raise ValueError("Gate instruction requires a gate label")
        if self.kind == InstructionKind.RUN and "steps" in self.params and int(self.params["steps"]) <= 0:
            raise ValueError("Run instruction steps must be positive")


@dataclass(slots=True, frozen=True)
class IRBlock:
    """Ordered block of IR instructions."""

    name: str
    instructions: Tuple[IRInstruction, ...] = ()
    metadata: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if not self.name:
            raise ValueError("IRBlock name must be non-empty")
        for instruction in self.instructions:
            instruction.validate()


@dataclass(slots=True, frozen=True)
class QLambdaProgram:
    """Typed program container for the Section 12 compiler layer."""

    name: str
    sectors: Tuple[SectorDeclaration, ...]
    blocks: Tuple[IRBlock, ...]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if not self.name:
            raise ValueError("Program name must be non-empty")
        names = [sector.name for sector in self.sectors]
        if len(names) != len(set(names)):
            raise ValueError(f"Duplicate sector declarations detected: {names!r}")
        for sector in self.sectors:
            sector.validate()
        if not self.blocks:
            raise ValueError("Program must contain at least one block")
        for block in self.blocks:
            block.validate()

    @property
    def sector_names(self) -> Tuple[str, ...]:
        return tuple(sector.name for sector in self.sectors)

    @property
    def sector_dims(self) -> Tuple[int, ...]:
        return tuple(sector.dimension for sector in self.sectors)


@dataclass(slots=True, frozen=True)
class RuntimeCallSpec:
    """Backend-ready call emitted by lowering."""

    call_id: str
    call_name: str
    kwargs: Dict[str, Any] = field(default_factory=dict)
    block_name: str = "main"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class CompiledExecutionPlan:
    """Lowered execution plan for the current simulator backend."""

    program: QLambdaProgram
    backend_kind: BackendKind
    call_specs: Tuple[RuntimeCallSpec, ...]
    symbol_table: Mapping[str, SectorDeclaration]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        self.program.validate()
        if not self.call_specs:
            raise ValueError("CompiledExecutionPlan must contain at least one call spec")
        if tuple(self.symbol_table.keys()) != self.program.sector_names:
            raise ValueError("Symbol table order must match program sector order")
        for spec in self.call_specs:
            if not spec.call_id or not spec.call_name:
                raise ValueError("Every RuntimeCallSpec requires call_id and call_name")


@dataclass(slots=True)
class CompilerExecutionResult:
    """Materialized result of executing a compiled plan."""

    plan: CompiledExecutionPlan
    bundle: Any
    graph: Any
    memory: Any = None
    runtime_reports: Tuple[Any, ...] = ()
    call_log: Tuple[RuntimeCallSpec, ...] = ()
    metadata: Dict[str, Any] = field(default_factory=dict)
