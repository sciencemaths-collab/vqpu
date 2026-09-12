"""Formal qualification of the local CPU compute boundary."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from vqpu.compute_engine import (
    ADAPTER_VERSION,
    CAPABILITY_ID,
    ENGINE_ID,
    ENGINE_VERSION,
    execute,
    plan,
    verify,
)


def qualify() -> dict[str, Any]:
    cases = []
    circuits = (
        (2, [{"name": "H", "targets": [0]}, {"name": "CNOT", "targets": [0, 1]}]),
        (
            3,
            [
                {"name": "H", "targets": [0]},
                {"name": "CNOT", "targets": [0, 1]},
                {"name": "CNOT", "targets": [1, 2]},
            ],
        ),
        (
            2,
            [
                {"name": "X", "targets": [0]},
                {"name": "RY", "targets": [1], "angle": 0.5},
                {"name": "CZ", "targets": [0, 1]},
            ],
        ),
    )
    for seed in (7, 19, 43, 101, 211):
        for index, (qubits, gates) in enumerate(circuits):
            workload = {
                "workload_type": "quantum.simulation",
                "qubits": qubits,
                "shots": 1024,
                "seed": seed,
                "gates": gates,
            }
            execution_plan = plan(workload)
            first = execute({"workload": workload, "plan_digest": execution_plan["plan_digest"]})
            second = execute({"workload": workload, "plan_digest": execution_plan["plan_digest"]})
            passed = first == second and verify(first)["verified"] is True
            cases.append(
                {
                    "case_id": f"cpu-qsim-{seed}-{index}",
                    "backend_id": "cpu.quantum_simulator",
                    "deterministic_replay": first == second,
                    "verified": verify(first)["verified"],
                    "shots": 1024,
                    "outcome": "PASS" if passed else "FAIL",
                }
            )
    passed_count = sum(case["outcome"] == "PASS" for case in cases)
    report: dict[str, Any] = {
        "schema_version": "1.0",
        "engine_id": ENGINE_ID,
        "engine_version": ENGINE_VERSION,
        "adapter_version": ADAPTER_VERSION,
        "capability_id": CAPABILITY_ID,
        "qualification": "QUALIFIED_FOR_LOCAL_CPU_QUANTUM_SIMULATION"
        if passed_count == len(cases)
        else "UNQUALIFIED",
        "case_count": len(cases),
        "passed_count": passed_count,
        "cases": cases,
        "limitations": [
            "local_cpu_only",
            "quantum_simulation_only",
            "maximum_20_qubits",
            "no_qpu_submission",
            "no_quantum_advantage_claim",
        ],
    }
    report["report_digest"] = _digest(report)
    return report


def _digest(value: dict[str, Any]) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description="Qualify RAD Compute Engine local CPU backend")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = qualify()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, sort_keys=True, separators=(",", ":")) + "\n")
    return 0 if report["qualification"] == "QUALIFIED_FOR_LOCAL_CPU_QUANTUM_SIMULATION" else 1


if __name__ == "__main__":
    raise SystemExit(main())
