"""Formal qualification of the Apple Metal MLX compute backend."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from vqpu.accelerator_engine import (
    ADAPTER_VERSION,
    BACKEND_ID,
    CAPABILITY_ID,
    ENGINE_ID,
    ENGINE_VERSION,
    discover,
    execute,
    plan,
    verify,
)


def qualify() -> dict[str, Any]:
    discovery = discover()
    cases: list[dict[str, Any]] = []
    if discovery["status"] == "AVAILABLE":
        for seed in (7, 19, 43, 101, 211):
            for size in (4, 16, 64):
                workload = {
                    "workload_type": "linear_algebra.matmul.float32",
                    "rows": size,
                    "inner": size,
                    "columns": size,
                    "seed": seed,
                    "tolerance": 1e-4,
                }
                execution_plan = plan(workload)
                first = execute(
                    {"workload": workload, "plan_digest": execution_plan["plan_digest"]}
                )
                second = execute(
                    {"workload": workload, "plan_digest": execution_plan["plan_digest"]}
                )
                replay = first["output_bytes_digest"] == second["output_bytes_digest"]
                passed = replay and verify(first)["verified"] is True
                cases.append(
                    {
                        "case_id": f"apple-metal-matmul-{size}-s{seed}",
                        "backend_id": BACKEND_ID,
                        "device_class": first["device_class"],
                        "runtime": first["runtime"],
                        "simulated": False,
                        "deterministic_replay": replay,
                        "reference_verified": first["reference_verified"],
                        "maximum_absolute_reference_error": first[
                            "maximum_absolute_reference_error"
                        ],
                        "outcome": "PASS" if passed else "FAIL",
                    }
                )
    passed_count = sum(case["outcome"] == "PASS" for case in cases)
    qualified = len(cases) == 15 and passed_count == 15
    report: dict[str, Any] = {
        "schema_version": "1.0",
        "engine_id": ENGINE_ID,
        "engine_version": ENGINE_VERSION,
        "adapter_version": ADAPTER_VERSION,
        "capability_id": CAPABILITY_ID,
        "backend_id": BACKEND_ID,
        "qualification": (
            "QUALIFIED_FOR_APPLE_METAL_FLOAT32_MATMUL" if qualified else "UNQUALIFIED"
        ),
        "discovery": discovery,
        "case_count": len(cases),
        "passed_count": passed_count,
        "cases": cases,
        "limitations": [
            "apple_silicon_only",
            "float32_matrix_multiply_only",
            "maximum_dimension_256",
            "local_execution_only",
            "no_cross_device_performance_claim",
        ],
    }
    report["report_digest"] = _digest(report)
    return report


def _digest(value: dict[str, Any]) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description="Qualify the Apple Metal MLX compute backend")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = qualify()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, sort_keys=True, separators=(",", ":")) + "\n")
    return 0 if report["qualification"] == "QUALIFIED_FOR_APPLE_METAL_FLOAT32_MATMUL" else 1


if __name__ == "__main__":
    raise SystemExit(main())
