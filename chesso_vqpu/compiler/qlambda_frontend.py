from __future__ import annotations

import shlex
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from core.types import SectorKind

from .ir import IRBlock, IRInstruction, InstructionKind, QLambdaProgram, QLambdaType, QTypeKind, SectorDeclaration


def qubit_type(*, label: str | None = None) -> QLambdaType:
    return QLambdaType(kind=QTypeKind.QUBIT, dims=(2,), label=label)


def qudit_type(dimension: int, *, label: str | None = None) -> QLambdaType:
    return QLambdaType(kind=QTypeKind.QUDIT, dims=(int(dimension),), label=label)


def qregister_type(width: int, *, local_dimension: int = 2, label: str | None = None) -> QLambdaType:
    if int(width) <= 0:
        raise ValueError("width must be positive")
    return QLambdaType(kind=QTypeKind.QREGISTER, dims=tuple(int(local_dimension) for _ in range(int(width))), label=label)


def declare_sector(
    name: str,
    *,
    qtype: QLambdaType | None = None,
    dimension: int | None = None,
    kind: SectorKind = SectorKind.LOGICAL,
    tags: Sequence[str] = (),
    metadata: Dict[str, Any] | None = None,
) -> SectorDeclaration:
    if qtype is None:
        qtype = qubit_type(label=name) if dimension is None else qudit_type(int(dimension), label=name)
    return SectorDeclaration(
        name=str(name),
        qtype=qtype,
        kind=kind,
        tags=tuple(str(tag) for tag in tags),
        metadata=dict(metadata or {}),
    )


def gate(gate_name: str, *targets: str, **params: Any) -> IRInstruction:
    return IRInstruction(kind=InstructionKind.GATE, label=str(gate_name), targets=tuple(str(t) for t in targets), params=dict(params))


def entangle(*targets: str, **params: Any) -> IRInstruction:
    return IRInstruction(kind=InstructionKind.ENTANGLE, targets=tuple(str(t) for t in targets), params=dict(params))


def expand(name: str, *, dimension: int = 2, score: float = 0.0, kind: str = "ancilla", **params: Any) -> IRInstruction:
    merged = {"name": str(name), "dimension": int(dimension), "score": float(score), "kind": str(kind), **params}
    return IRInstruction(kind=InstructionKind.EXPAND, label=str(name), params=merged)


def measure(*targets: str, strength: float = 1.0, selective: bool = True, **params: Any) -> IRInstruction:
    merged = {"strength": float(strength), "selective": bool(selective), **params}
    return IRInstruction(kind=InstructionKind.MEASURE, targets=tuple(str(t) for t in targets), params=merged)


def run(*, steps: int = 1, scheduled: bool = False, **params: Any) -> IRInstruction:
    merged = {"steps": int(steps), "scheduled": bool(scheduled), **params}
    return IRInstruction(kind=InstructionKind.RUN, params=merged)


def note(message: str, **params: Any) -> IRInstruction:
    merged = {"message": str(message), **params}
    return IRInstruction(kind=InstructionKind.NOTE, params=merged)


def make_block(name: str, *instructions: IRInstruction, **metadata: Any) -> IRBlock:
    return IRBlock(name=str(name), instructions=tuple(instructions), metadata=dict(metadata))


def make_program(name: str, sectors: Sequence[SectorDeclaration], *blocks: IRBlock, **metadata: Any) -> QLambdaProgram:
    program = QLambdaProgram(name=str(name), sectors=tuple(sectors), blocks=tuple(blocks) or (make_block("main"),), metadata=dict(metadata))
    program.validate()
    return program


def _coerce_value(text: str) -> Any:
    lowered = text.strip().lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    try:
        if any(ch in text for ch in (".", "e", "E")):
            return float(text)
        return int(text)
    except ValueError:
        return text


def _parse_tokens(line: str) -> Tuple[str, List[str], Dict[str, Any]]:
    parts = shlex.split(line, comments=False, posix=True)
    if not parts:
        return "", [], {}
    op = parts[0].lower()
    positionals: List[str] = []
    kwargs: Dict[str, Any] = {}
    for token in parts[1:]:
        if "=" in token:
            key, value = token.split("=", 1)
            kwargs[key.strip()] = _coerce_value(value)
        else:
            positionals.append(token)
    return op, positionals, kwargs


def parse_qlambda_script(script: str | Iterable[str]) -> QLambdaProgram:
    """Parse a very small Qλ-style script into the Section 12 typed IR."""
    lines = script.splitlines() if isinstance(script, str) else list(script)
    program_name = "qlambda_program"
    sectors: List[SectorDeclaration] = []
    blocks: List[IRBlock] = []
    current_name = "main"
    current_instructions: List[IRInstruction] = []

    def flush_block() -> None:
        nonlocal current_instructions, current_name
        blocks.append(IRBlock(name=current_name, instructions=tuple(current_instructions)))
        current_name = "main"
        current_instructions = []

    for raw in lines:
        line = str(raw).strip()
        if not line or line.startswith("#"):
            continue
        op, pos, kwargs = _parse_tokens(line)
        if not op:
            continue
        if op == "program":
            if not pos:
                raise ValueError("program line requires a name")
            program_name = str(pos[0])
            continue
        if op == "block":
            if current_instructions or not blocks:
                flush_block()
                blocks.pop()
            current_name = str(pos[0] if pos else f"block_{len(blocks)}")
            current_instructions = []
            continue
        if op == "end":
            flush_block()
            continue
        if op == "alloc":
            if not pos:
                raise ValueError("alloc requires a sector name")
            name = str(pos[0])
            dim = int(kwargs.pop("dim", kwargs.pop("dimension", 2)))
            kind = SectorKind(str(kwargs.pop("kind", "logical")))
            tags_raw = kwargs.pop("tags", ())
            if isinstance(tags_raw, str):
                tags = tuple(part for part in tags_raw.split(",") if part)
            else:
                tags = tuple(tags_raw)
            sectors.append(declare_sector(name, dimension=dim, kind=kind, tags=tags, metadata=kwargs))
            continue
        if op == "gate":
            if len(pos) < 2:
                raise ValueError("gate requires a gate name followed by one or more targets")
            current_instructions.append(gate(pos[0], *pos[1:], **kwargs))
            continue
        if op == "entangle":
            if len(pos) < 2:
                raise ValueError("entangle requires at least two targets")
            current_instructions.append(entangle(*pos, **kwargs))
            continue
        if op == "expand":
            if not pos:
                raise ValueError("expand requires a sector name")
            current_instructions.append(expand(pos[0], **kwargs))
            continue
        if op == "measure":
            if not pos:
                raise ValueError("measure requires at least one target")
            current_instructions.append(measure(*pos, **kwargs))
            continue
        if op == "run":
            current_instructions.append(run(**kwargs))
            continue
        if op == "note":
            current_instructions.append(note(" ".join(pos), **kwargs))
            continue
        raise ValueError(f"Unsupported Qλ frontend directive: {op!r}")

    if current_instructions or not blocks:
        flush_block()

    program = QLambdaProgram(name=program_name, sectors=tuple(sectors), blocks=tuple(blocks))
    program.validate()
    return program


__all__ = [
    "QLambdaProgram",
    "QLambdaType",
    "QTypeKind",
    "SectorDeclaration",
    "declare_sector",
    "qubit_type",
    "qudit_type",
    "qregister_type",
    "gate",
    "entangle",
    "expand",
    "measure",
    "run",
    "note",
    "make_block",
    "make_program",
    "parse_qlambda_script",
]
