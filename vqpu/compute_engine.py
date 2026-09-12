"""Strict local CPU compute boundary; vQPU remains the quantum fabric."""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
from collections.abc import Mapping, Sequence
from typing import Any

from vqpu.core import QuantumCircuit, vQPU

ENGINE_ID = "rad-compute-engine"
ENGINE_VERSION = "0.5.0"
ADAPTER_VERSION = "1.0.0"
CAPABILITY_ID = "rad.compute.local"
_GATES = frozenset({"H", "X", "Y", "Z", "S", "T", "RX", "RY", "RZ", "CNOT", "CZ", "SWAP"})


class ComputeEngineError(ValueError):
    """Safe workload rejection."""


def describe() -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "engine_id": ENGINE_ID,
        "engine_version": ENGINE_VERSION,
        "adapter_version": ADAPTER_VERSION,
        "capability_id": CAPABILITY_ID,
        "operations": ["compute.discover", "compute.plan", "compute.execute", "compute.verify"],
        "backends": ["cpu.quantum_simulator"],
        "network_access": "DENIED",
        "limitations": [
            "local_cpu_only",
            "quantum_simulation_only",
            "maximum_20_qubits",
            "no_qpu_submission",
        ],
    }


def discover() -> dict[str, Any]:
    cores = os.cpu_count() or 1
    return {
        "schema_version": "1.0",
        "engine_id": ENGINE_ID,
        "backends": [
            {
                "backend_id": "cpu.quantum_simulator",
                "compute_class": "CPU",
                "status": "AVAILABLE",
                "logical_cores": cores,
                "machine": platform.machine(),
                "network_required": False,
                "cost_per_run_usd": 0.0,
            }
        ],
    }


def plan(request: Mapping[str, Any]) -> dict[str, Any]:
    workload = _workload(request)
    qubits, shots, gates = workload["qubits"], workload["shots"], workload["gates"]
    unsigned = {
        "schema_version": "1.0",
        "engine_id": ENGINE_ID,
        "engine_version": ENGINE_VERSION,
        "adapter_version": ADAPTER_VERSION,
        "capability_id": CAPABILITY_ID,
        "workload_digest": _digest(workload),
        "backend_id": "cpu.quantum_simulator",
        "estimated_memory_bytes": 16 * 2**qubits,
        "estimated_gate_applications": len(gates),
        "shots": shots,
        "network_required": False,
        "estimated_cost_usd": 0.0,
        "approval_required": True,
    }
    return {**unsigned, "plan_digest": _digest(unsigned)}


def execute(request: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(request, Mapping) or set(request) != {"workload", "plan_digest"}:
        raise ComputeEngineError("execution request is invalid")
    workload = _workload(request["workload"])
    execution_plan = plan(workload)
    if request["plan_digest"] != execution_plan["plan_digest"]:
        raise ComputeEngineError("execution plan digest mismatch")
    qpu = vQPU(backend="simulator", seed=workload["seed"])
    circuit = qpu.circuit(workload["qubits"], "rad-compute")
    _apply(circuit, workload["gates"])
    raw = qpu.run(circuit, shots=workload["shots"])
    unsigned = {
        "schema_version": "1.0",
        "engine_id": ENGINE_ID,
        "engine_version": ENGINE_VERSION,
        "adapter_version": ADAPTER_VERSION,
        "capability_id": CAPABILITY_ID,
        "plan_digest": execution_plan["plan_digest"],
        "workload_digest": execution_plan["workload_digest"],
        "backend_id": "cpu.quantum_simulator",
        "counts": dict(sorted(raw.counts.items())),
        "shots": workload["shots"],
        "seed": workload["seed"],
        "network_used": False,
        "cost_usd": 0.0,
        "limitations": [
            "classical_simulation_not_quantum_hardware",
            "sampling_uncertainty_applies",
        ],
    }
    return {**unsigned, "result_digest": _digest(unsigned)}


def verify(result: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(result, Mapping) or "result_digest" not in result:
        raise ComputeEngineError("compute result is invalid")
    unsigned = {key: value for key, value in result.items() if key != "result_digest"}
    counts, shots = result.get("counts"), result.get("shots")
    valid = (
        result.get("engine_id") == ENGINE_ID
        and result.get("engine_version") == ENGINE_VERSION
        and result.get("backend_id") == "cpu.quantum_simulator"
        and result.get("network_used") is False
        and result.get("cost_usd") == 0.0
        and isinstance(counts, Mapping)
        and isinstance(shots, int)
        and sum(counts.values()) == shots
        and _digest(unsigned) == result["result_digest"]
    )
    return {"verified": valid, "result_digest": result.get("result_digest")}


def _workload(value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) - {
        "workload_type",
        "qubits",
        "shots",
        "seed",
        "gates",
    }:
        raise ComputeEngineError("workload is invalid")
    if value.get("workload_type") != "quantum.simulation":
        raise ComputeEngineError("workload type is unsupported")
    qubits = _integer(value.get("qubits"), 1, 20)
    shots = _integer(value.get("shots", 1024), 1, 100_000)
    seed = _integer(value.get("seed", 7), 0, 2**32 - 1)
    gates = value.get("gates")
    if (
        not isinstance(gates, Sequence)
        or isinstance(gates, (str, bytes))
        or not 1 <= len(gates) <= 10_000
    ):
        raise ComputeEngineError("gate sequence is invalid")
    normalized = [_gate(gate, qubits) for gate in gates]
    return {
        "workload_type": "quantum.simulation",
        "qubits": qubits,
        "shots": shots,
        "seed": seed,
        "gates": normalized,
    }


def _gate(value: object, qubits: int) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) - {"name", "targets", "angle"}:
        raise ComputeEngineError("gate is invalid")
    name, targets = value.get("name"), value.get("targets")
    if name not in _GATES or not isinstance(targets, Sequence) or isinstance(targets, (str, bytes)):
        raise ComputeEngineError("gate is unsupported")
    required = 2 if name in {"CNOT", "CZ", "SWAP"} else 1
    parsed = [_integer(item, 0, qubits - 1) for item in targets]
    if len(parsed) != required or len(set(parsed)) != required:
        raise ComputeEngineError("gate targets are invalid")
    gate: dict[str, Any] = {"name": name, "targets": parsed}
    if name in {"RX", "RY", "RZ"}:
        angle = value.get("angle")
        if (
            isinstance(angle, bool)
            or not isinstance(angle, (int, float))
            or not math.isfinite(angle)
            or abs(angle) > 1e6
        ):
            raise ComputeEngineError("gate angle is invalid")
        gate["angle"] = float(angle)
    elif "angle" in value:
        raise ComputeEngineError("gate angle is invalid")
    return gate


def _apply(circuit: QuantumCircuit, gates: list[dict[str, Any]]) -> None:
    for gate in gates:
        name, targets = gate["name"], gate["targets"]
        method = getattr(circuit, name.lower())
        if name in {"RX", "RY", "RZ"}:
            method(targets[0], gate["angle"])
        else:
            method(*targets)


def _integer(value: object, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise ComputeEngineError("workload is outside bounded limits")
    return value


def _digest(value: Mapping[str, Any]) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return "sha256:" + hashlib.sha256(raw).hexdigest()
