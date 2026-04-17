from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np

from control.runtime import run_chesso_runtime
from control.scheduler import run_scheduled_runtime
from core import BackendKind, EntanglementHypergraph, HilbertBundleState, QuantumState, RuntimeConfig, SimulationMode
from core.types import BundleTopology, SectorId, SectorKind, SectorSpec, StateRepresentation
from ops import apply_hyperedge_entangler, apply_local_operator, attach_local_sector, measure_computational_basis
from ops.unitary_ops import controlled_x, controlled_z, hadamard, pauli_x, pauli_y, pauli_z, phase, rotation_x, rotation_y, rotation_z, swap

from .ir import (
    CompiledExecutionPlan,
    CompilerExecutionResult,
    IRBlock,
    IRInstruction,
    InstructionKind,
    QLambdaProgram,
    RuntimeCallSpec,
    SectorDeclaration,
)


@dataclass(slots=True)
class LoweringOptions:
    """Compilation-time options for the simulator-first backend."""

    backend_kind: BackendKind = BackendKind.NUMPY
    create_graph: bool = True
    auto_route_entanglers: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class _ExecutionContext:
    config: RuntimeConfig
    bundle: HilbertBundleState | None = None
    graph: EntanglementHypergraph | None = None
    memory: Any = None
    runtime_reports: List[Any] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)


_GATE_BUILDERS = {
    "X": lambda params: pauli_x(),
    "Y": lambda params: pauli_y(),
    "Z": lambda params: pauli_z(),
    "H": lambda params: hadamard(),
    "PHASE": lambda params: phase(float(params.get("theta", params.get("phi", 0.0)))),
    "RX": lambda params: rotation_x(float(params.get("theta", 0.0))),
    "RY": lambda params: rotation_y(float(params.get("theta", 0.0))),
    "RZ": lambda params: rotation_z(float(params.get("theta", 0.0))),
    "CX": lambda params: controlled_x(),
    "CNOT": lambda params: controlled_x(),
    "CZ": lambda params: controlled_z(),
    "SWAP": lambda params: swap(),
}


def _make_initial_bundle(sectors: Sequence[SectorDeclaration], config: RuntimeConfig) -> HilbertBundleState:
    topology = BundleTopology(
        [
            SectorSpec(
                sector_id=SectorId(sector.name),
                dimension=sector.dimension,
                kind=sector.kind,
                tags=tuple(sector.tags),
                metadata=dict(sector.metadata),
            )
            for sector in sectors
        ]
    )
    topology.validate()
    representation = (
        StateRepresentation.STATEVECTOR
        if config.simulation_mode == SimulationMode.STATEVECTOR
        else StateRepresentation.DENSITY_MATRIX
    )
    quantum_state = QuantumState.zero_state(topology.dims, representation, config.numerical_tolerance)
    return HilbertBundleState(topology=topology, quantum_state=quantum_state, config=config)


def _ensure_graph(ctx: _ExecutionContext) -> EntanglementHypergraph:
    if ctx.bundle is None:
        raise RuntimeError("Bundle must exist before constructing an entanglement graph")
    if ctx.graph is None:
        ctx.graph = EntanglementHypergraph.from_topology(
            ctx.bundle.topology,
            max_order=int(ctx.config.budget.max_hyperedge_order),
        )
    else:
        ctx.graph.sync_from_topology(ctx.bundle.topology, drop_missing=False)
    return ctx.graph


def _resolve_gate(gate_name: str, params: Mapping[str, Any]) -> np.ndarray:
    key = str(gate_name).upper()
    if key not in _GATE_BUILDERS:
        raise KeyError(f"Unsupported gate in Section 12 lowering: {gate_name!r}")
    return np.asarray(_GATE_BUILDERS[key](params), dtype=np.complex128)


def lower_instruction(instruction: IRInstruction, *, block_name: str, call_index_start: int) -> Tuple[RuntimeCallSpec, ...]:
    instruction.validate()
    specs: List[RuntimeCallSpec] = []
    next_index = int(call_index_start)

    def emit(call_name: str, **kwargs: Any) -> None:
        nonlocal next_index
        specs.append(
            RuntimeCallSpec(
                call_id=f"{block_name}:{next_index:04d}",
                call_name=call_name,
                kwargs=dict(kwargs),
                block_name=block_name,
                metadata={"ir_kind": instruction.kind.value, "label": instruction.label},
            )
        )
        next_index += 1

    if instruction.kind == InstructionKind.GATE:
        emit("apply_gate", gate_name=instruction.label, targets=instruction.targets, params=dict(instruction.params))
    elif instruction.kind == InstructionKind.ENTANGLE:
        payload = dict(instruction.params)
        route_id = payload.pop("route_id", None)
        auto_route = bool(payload.pop("auto_route", True))
        emit("add_hyperedge", members=instruction.targets, params=payload)
        if auto_route and len(instruction.targets) >= 2:
            emit(
                "add_route",
                source=instruction.targets[0],
                target=instruction.targets[-1],
                route_id=route_id,
                edge_members=instruction.targets,
                params={k: payload[k] for k in ("score", "bandwidth", "latency") if k in payload},
            )
        if bool(instruction.params.get("apply", True)):
            emit("apply_entangler", members=instruction.targets, params=dict(instruction.params))
    elif instruction.kind == InstructionKind.EXPAND:
        emit("expand_sector", params=dict(instruction.params))
    elif instruction.kind == InstructionKind.MEASURE:
        emit("measure_basis", targets=instruction.targets, params=dict(instruction.params))
    elif instruction.kind == InstructionKind.RUN:
        emit("run_runtime", params=dict(instruction.params))
    elif instruction.kind == InstructionKind.NOTE:
        emit("note", params=dict(instruction.params))
    elif instruction.kind == InstructionKind.TARGET:
        emit("set_target", params=dict(instruction.params))
    else:
        raise KeyError(f"Unsupported IR instruction kind: {instruction.kind!r}")
    return tuple(specs)


def lower_program(
    program: QLambdaProgram,
    *,
    config: RuntimeConfig | None = None,
    options: LoweringOptions | None = None,
) -> CompiledExecutionPlan:
    """Lower a typed Qλ program into simulator-ready call specs."""
    program.validate()
    cfg = config or RuntimeConfig.for_density_matrix(max_active_qubits=max(4, len(program.sectors) + 2))
    opts = options or LoweringOptions(backend_kind=cfg.backend_kind)

    call_specs: List[RuntimeCallSpec] = [
        RuntimeCallSpec(
            call_id="init:0000",
            call_name="initialize_bundle",
            kwargs={
                "sector_declarations": tuple(program.sectors),
                "create_graph": bool(opts.create_graph),
            },
            block_name="__init__",
            metadata={"backend_kind": opts.backend_kind.value},
        )
    ]

    next_index = 1
    for block in program.blocks:
        lowered = lower_instruction(block.instructions[0], block_name=block.name, call_index_start=next_index) if len(block.instructions) == 1 else ()
        if len(block.instructions) != 1:
            for instruction in block.instructions:
                chunk = lower_instruction(instruction, block_name=block.name, call_index_start=next_index)
                call_specs.extend(chunk)
                next_index += len(chunk)
        else:
            call_specs.extend(lowered)
            next_index += len(lowered)

    plan = CompiledExecutionPlan(
        program=program,
        backend_kind=opts.backend_kind,
        call_specs=tuple(call_specs),
        symbol_table={sector.name: sector for sector in program.sectors},
        metadata={
            "create_graph": bool(opts.create_graph),
            "auto_route_entanglers": bool(opts.auto_route_entanglers),
            **dict(opts.metadata),
        },
    )
    plan.validate()
    return plan


def _dispatch_call(spec: RuntimeCallSpec, ctx: _ExecutionContext) -> None:
    if spec.call_name == "initialize_bundle":
        sectors = spec.kwargs["sector_declarations"]
        ctx.bundle = _make_initial_bundle(sectors, ctx.config)
        if bool(spec.kwargs.get("create_graph", False)):
            ctx.graph = EntanglementHypergraph.from_topology(
                ctx.bundle.topology,
                max_order=int(ctx.config.budget.max_hyperedge_order),
            )
        return

    if ctx.bundle is None:
        raise RuntimeError(f"Call {spec.call_id} requires an initialized bundle")

    if spec.call_name == "apply_gate":
        operator = _resolve_gate(spec.kwargs["gate_name"], spec.kwargs.get("params", {}))
        apply_local_operator(ctx.bundle, operator, spec.kwargs["targets"], in_place=True, label=spec.kwargs["gate_name"])
        return

    if spec.call_name == "add_hyperedge":
        graph = _ensure_graph(ctx)
        params = dict(spec.kwargs.get("params", {}))
        members = tuple(str(member) for member in spec.kwargs["members"])
        edge_id = params.pop("edge_id", None)
        graph.add_hyperedge(
            members,
            weight=float(params.pop("weight", 1.0)),
            phase_bias=float(params.pop("phase_bias", 0.0)),
            coherence_score=float(params.pop("coherence_score", 1.0)),
            capacity=float(params.pop("capacity", 1.0)),
            tags=tuple(params.pop("tags", ())),
            metadata=params,
            edge_id=edge_id,
        )
        return

    if spec.call_name == "add_route":
        graph = _ensure_graph(ctx)
        members = tuple(str(member) for member in spec.kwargs["edge_members"])
        edge_id = graph.make_edge_id(tuple(SectorId(member) for member in members))
        params = dict(spec.kwargs.get("params", {}))
        graph.add_route(
            spec.kwargs["source"],
            spec.kwargs["target"],
            [edge_id],
            score=float(params.pop("score", 0.0)),
            bandwidth=float(params.pop("bandwidth", 1.0)),
            latency=float(params.pop("latency", 0.0)),
            route_id=spec.kwargs.get("route_id"),
            metadata=params,
        )
        return

    if spec.call_name == "apply_entangler":
        graph = _ensure_graph(ctx)
        members = tuple(str(member) for member in spec.kwargs["members"])
        edge_id = graph.make_edge_id(tuple(SectorId(member) for member in members))
        params = dict(spec.kwargs.get("params", {}))
        strength = float(params.get("strength", params.get("weight", 1.0)))
        apply_hyperedge_entangler(ctx.bundle, graph, edge_id, strength=strength, in_place=True)
        return

    if spec.call_name == "expand_sector":
        params = dict(spec.kwargs.get("params", {}))
        name = str(params.pop("name"))
        raw_kind = params.pop("kind", "ancilla")
        kind = raw_kind if isinstance(raw_kind, SectorKind) else SectorKind(str(raw_kind))
        attach_local_sector(
            ctx.bundle,
            name=name,
            dimension=int(params.pop("dimension", 2)),
            local_state=params.pop("local_state", None),
            kind=kind,
            tags=tuple(params.pop("tags", ())),
            metadata=params,
            in_place=True,
        )
        if ctx.graph is not None:
            ctx.graph.sync_from_topology(ctx.bundle.topology, drop_missing=False)
        return

    if spec.call_name == "measure_basis":
        params = dict(spec.kwargs.get("params", {}))
        targets = tuple(str(target) for target in spec.kwargs["targets"])
        record = measure_computational_basis(
            ctx.bundle,
            targets,
            strength=float(params.pop("strength", 1.0)),
            outcome=params.pop("outcome", None),
            sample=bool(params.pop("sample", False)),
            selective=bool(params.pop("selective", True)),
            label=params.pop("label", None),
            store_key=params.pop("store", params.pop("store_key", None)),
            metadata=params,
            in_place=True,
        )
        ctx.notes.append(f"measurement:{record.label}:{record.outcome}")
        return

    if spec.call_name == "run_runtime":
        params = dict(spec.kwargs.get("params", {}))
        scheduled = bool(params.pop("scheduled", False))
        steps = int(params.pop("steps", 1))
        if scheduled:
            report = run_scheduled_runtime(
                ctx.bundle,
                graph=ctx.graph,
                memory=ctx.memory,
                steps=steps,
                stop_fidelity=params.pop("stop_fidelity", 0.999),
                max_entanglers=int(params.pop("max_entanglers", 2)),
                in_place=True,
            )
            ctx.bundle = report.final_bundle
            ctx.graph = report.final_graph
            ctx.memory = report.final_memory
            ctx.runtime_reports.append(report)
        else:
            report = run_chesso_runtime(
                ctx.bundle,
                graph=ctx.graph,
                memory=ctx.memory,
                steps=steps,
                stop_fidelity=params.pop("stop_fidelity", 0.999),
                max_entanglers=int(params.pop("max_entanglers", 2)),
                in_place=True,
            )
            ctx.bundle = report.final_bundle
            ctx.graph = report.final_graph
            ctx.memory = report.final_memory
            ctx.runtime_reports.append(report)
        return

    if spec.call_name == "note":
        params = dict(spec.kwargs.get("params", {}))
        ctx.notes.append(str(params.get("message", "")))
        return

    if spec.call_name == "set_target":
        ctx.notes.append(f"target:{spec.kwargs}")
        return

    raise KeyError(f"Unknown compiled call name: {spec.call_name!r}")


def execute_plan(
    plan: CompiledExecutionPlan,
    *,
    config: RuntimeConfig | None = None,
) -> CompilerExecutionResult:
    """Execute a compiled Section 12 plan against the current simulator backend."""
    plan.validate()
    ctx = _ExecutionContext(config=config or RuntimeConfig.for_density_matrix(max_active_qubits=max(4, len(plan.program.sectors) + 2)))
    call_log: List[RuntimeCallSpec] = []
    for spec in plan.call_specs:
        _dispatch_call(spec, ctx)
        call_log.append(spec)
    if ctx.bundle is None:
        raise RuntimeError("Execution finished without producing a bundle")
    return CompilerExecutionResult(
        plan=plan,
        bundle=ctx.bundle,
        graph=ctx.graph,
        memory=ctx.memory,
        runtime_reports=tuple(ctx.runtime_reports),
        call_log=tuple(call_log),
        metadata={"notes": tuple(ctx.notes)},
    )


__all__ = [
    "LoweringOptions",
    "lower_instruction",
    "lower_program",
    "execute_plan",
]
