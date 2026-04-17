"""CHESSO → vqpu hardware bridge.

Lowers a `CompiledExecutionPlan` from the Qλ compiler into vqpu's flat
`gate_sequence` format and runs it on any vqpu backend plugin — the local
statevector simulator, the phantom backend, or a `QPUCloudPlugin` pointed
at real cloud QPU hardware (IonQ, IBM, Braket, Rigetti).

Why this module exists
──────────────────────
Before this bridge, `execute_plan()` from `vqpu.chesso.compiler` ran only on
CHESSO's internal NumPy runtime. CHESSO's hypergraph entanglers, soft
measurement semantics, and Qλ frontend were cut off from physical qubits.
This module is the missing adapter: the same Qλ program that drove the
simulator now drives IonQ's trapped-ion QPU.

Unsupported ops (raised or skipped with a note)
───────────────────────────────────────────────
- `expand_sector`                — hardware has a fixed qubit count. All
                                   sectors must be declared upfront.
- `measure_basis` (mid-circuit)  — IonQ/most QPUs want end-of-circuit
                                   measurement. Deferred to the implicit
                                   `measure_all()` at submit time.
- `run_runtime` (CHESSO loop)    — no hardware analogue; skipped with
                                   a note so callers know.

Hyperedge entanglers compile to `make_hyperedge_phase_entangler(...)`
matrices and ride vqpu's `FULL_UNITARY` path (which calls
`qc.unitary(...)` in the qiskit-ionq translator, which then transpiles
into the device's native gate set).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from vqpu.chesso.compiler.ir import CompiledExecutionPlan, RuntimeCallSpec
from vqpu.chesso.compiler.lowering import lower_program
from vqpu.chesso.compiler.qlambda_frontend import parse_qlambda_script
from vqpu.chesso.core import RuntimeConfig
from vqpu.chesso.ops.unitary_ops import make_hyperedge_phase_entangler

GateTuple = Tuple[Any, ...]


# ─────────────────────── gate-name translation ─────────────────────────────

_HARDWARE_GATE_NAMES = {
    "X": "X", "Y": "Y", "Z": "Z", "H": "H", "S": "S", "T": "T",
    "CX": "CNOT", "CNOT": "CNOT", "CZ": "CZ", "SWAP": "SWAP",
    "RX": "Rx", "RY": "Ry", "RZ": "Rz", "PHASE": "Phase",
}

_PARAMETRIC_GATES = {"RX", "RY", "RZ", "PHASE"}


def _hardware_gate_name(name: str) -> str:
    key = name.upper()
    try:
        return _HARDWARE_GATE_NAMES[key]
    except KeyError as exc:
        raise KeyError(
            f"CHESSO gate {name!r} has no hardware bridge translation. "
            f"Supported: {sorted(_HARDWARE_GATE_NAMES)}"
        ) from exc


def _param_for_gate(name: str, params: Mapping[str, Any]) -> Optional[float]:
    key = name.upper()
    if key not in _PARAMETRIC_GATES:
        return None
    # CHESSO IR stores rotation angle as `theta` (Phase also accepts `phi`).
    if "theta" in params:
        return float(params["theta"])
    if key == "PHASE" and "phi" in params:
        return float(params["phi"])
    return 0.0


# ─────────── endianness: vqpu q0=MSB, Qiskit q0=LSB ────────────────────────
# The vqpu→qiskit path in QPUCloudPlugin passes FULL_UNITARY straight to
# `qc.unitary(matrix, targets)`. Qiskit reads index 0 as the LSB of the
# matrix; vqpu builds matrices with index 0 as the MSB. Reversing the
# `targets` list makes both conventions agree without touching the matrix.

def _emit_full_unitary(target_indices: Sequence[int], matrix: np.ndarray) -> GateTuple:
    reversed_targets = list(reversed(target_indices))
    return ("FULL_UNITARY", reversed_targets, matrix)


# ───────────────────────── bridge result container ─────────────────────────

@dataclass(slots=True)
class BridgedCircuit:
    """Plan-to-hardware translation output."""

    n_qubits: int
    gate_sequence: List[GateTuple]
    sector_to_qubit: Dict[str, int]
    notes: List[str] = field(default_factory=list)

    def gate_count(self) -> int:
        return len(self.gate_sequence)

    def depth_approx(self) -> int:
        # Layer count assuming every gate is serial; a cheap upper bound.
        return len(self.gate_sequence)


# ─────────────────────────── core lowering ────────────────────────────────

def plan_to_gate_sequence(plan: CompiledExecutionPlan) -> BridgedCircuit:
    """Lower a CHESSO `CompiledExecutionPlan` to a vqpu gate sequence."""
    name_to_qubit: Dict[str, int] = {}
    hyperedges: Dict[Tuple[str, ...], Dict[str, Any]] = {}
    gate_seq: List[GateTuple] = []
    notes: List[str] = []

    def register_sector(name: str, dim: int) -> None:
        if dim != 2:
            raise NotImplementedError(
                f"Sector {name!r} has local dimension {dim}; the hardware "
                "bridge supports qubits only (dim = 2)."
            )
        if name not in name_to_qubit:
            name_to_qubit[name] = len(name_to_qubit)

    for spec in plan.call_specs:
        _lower_spec(spec, name_to_qubit, hyperedges, gate_seq, notes, register_sector)

    return BridgedCircuit(
        n_qubits=len(name_to_qubit),
        gate_sequence=gate_seq,
        sector_to_qubit=dict(name_to_qubit),
        notes=notes,
    )


def _lower_spec(
    spec: RuntimeCallSpec,
    name_to_qubit: Dict[str, int],
    hyperedges: Dict[Tuple[str, ...], Dict[str, Any]],
    gate_seq: List[GateTuple],
    notes: List[str],
    register_sector,
) -> None:
    if spec.call_name == "initialize_bundle":
        for decl in spec.kwargs["sector_declarations"]:
            register_sector(decl.name, decl.dimension)
        return

    if spec.call_name == "apply_gate":
        gate_name = spec.kwargs["gate_name"]
        hw_name = _hardware_gate_name(gate_name)
        targets = [name_to_qubit[str(t)] for t in spec.kwargs["targets"]]
        theta = _param_for_gate(gate_name, spec.kwargs.get("params", {}))
        if theta is None:
            gate_seq.append((hw_name, targets))
        else:
            gate_seq.append((hw_name, targets, theta))
        return

    if spec.call_name == "add_hyperedge":
        members = tuple(str(m) for m in spec.kwargs["members"])
        edge_key = tuple(sorted(members))
        params = spec.kwargs.get("params", {})
        hyperedges[edge_key] = {
            "members": members,
            "weight": float(params.get("weight", 1.0)),
            "phase_bias": float(params.get("phase_bias", 0.0)),
            "coherence_score": float(params.get("coherence_score", 1.0)),
            "profile": str(params.get("profile", "occupancy")),
        }
        return

    if spec.call_name == "apply_entangler":
        members = tuple(str(m) for m in spec.kwargs["members"])
        edge_key = tuple(sorted(members))
        meta = hyperedges.get(edge_key, {
            "members": members, "weight": 1.0,
            "phase_bias": 0.0, "coherence_score": 1.0,
            "profile": "occupancy",
        })
        call_params = dict(spec.kwargs.get("params", {}))
        strength = float(call_params.get("strength", 1.0))
        # Per-call `profile` override wins over the hyperedge-level default.
        profile = str(call_params.get("profile", meta.get("profile", "occupancy")))
        theta = strength * meta["weight"] * meta["coherence_score"]
        target_idx = [name_to_qubit[m] for m in members]
        matrix = make_hyperedge_phase_entangler(
            tuple(2 for _ in members),
            theta,
            phase_bias=meta["phase_bias"],
            profile=profile,
        )
        gate_seq.append(_emit_full_unitary(target_idx, matrix))
        return

    if spec.call_name == "add_route":
        return  # routing metadata for CHESSO's own runtime; no hardware op

    if spec.call_name == "expand_sector":
        raise NotImplementedError(
            "expand_sector is not supported on hardware backends — real QPUs "
            "have a fixed qubit count. Declare every sector upfront with "
            "`alloc` before compiling for hardware."
        )

    if spec.call_name == "measure_basis":
        targets = spec.kwargs.get("targets", ())
        notes.append(
            f"mid-circuit measure on {list(targets)} deferred to "
            "end-of-circuit sampling (hardware bridge has no mid-circuit "
            "measurement path yet)"
        )
        return

    if spec.call_name == "run_runtime":
        notes.append(
            "run_runtime skipped — CHESSO's adaptive runtime loop has no "
            "hardware equivalent. Use the local backend for runtime-driven "
            "experiments."
        )
        return

    if spec.call_name in {"note", "set_target"}:
        return

    raise NotImplementedError(
        f"Bridge has no translation for call spec {spec.call_name!r}"
    )


# ──────────────────────── convenience entry points ────────────────────────

def compile_qlambda_for_hardware(
    source: str,
    *,
    config: Optional[RuntimeConfig] = None,
) -> BridgedCircuit:
    """Parse a Qλ script and lower it straight to a `BridgedCircuit`."""
    program = parse_qlambda_script(source)
    cfg = config or RuntimeConfig.for_statevector(
        max_active_qubits=max(4, len(program.sectors) + 2)
    )
    plan = lower_program(program, config=cfg)
    return plan_to_gate_sequence(plan)


def execute_qlambda_on_backend(
    source: str,
    *,
    backend,
    shots: int = 1024,
    config: Optional[RuntimeConfig] = None,
) -> Dict[str, int]:
    """Parse → lower → run on any backend plugin exposing `execute_sample`.

    `backend` can be a `QPUCloudPlugin("ionq")` instance (live QPU / IonQ
    simulator), a `CPUPlugin`, a `PhantomSimulatorBackend`, or anything
    following the same `execute_sample(n_qubits, gate_sequence, shots)`
    contract.
    """
    circuit = compile_qlambda_for_hardware(source, config=config)
    return backend.execute_sample(
        n_qubits=circuit.n_qubits,
        gate_sequence=circuit.gate_sequence,
        shots=shots,
    )


__all__ = [
    "BridgedCircuit",
    "compile_qlambda_for_hardware",
    "execute_qlambda_on_backend",
    "plan_to_gate_sequence",
]
