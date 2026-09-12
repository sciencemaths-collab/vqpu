"""Strict Apple Metal accelerator boundary for RAD Compute Engine."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import platform
import time
from collections.abc import Mapping
from typing import Any, TypeGuard

import numpy as np

ENGINE_ID = "rad-compute-engine"
ENGINE_VERSION = "0.6.0"
ADAPTER_VERSION = "1.0.0"
CAPABILITY_ID = "rad.compute.apple_gpu"
BACKEND_ID = "apple.metal.mlx"


class AcceleratorEngineError(ValueError):
    """Safe accelerator discovery, planning, execution, or verification failure."""


def discover() -> dict[str, Any]:
    """Report Apple Metal availability without unique machine identifiers."""
    machine = platform.machine().lower()
    system = platform.system()
    mlx_version: str | None = None
    runtime_available = False
    device = "UNAVAILABLE"
    if system == "Darwin" and machine == "arm64":
        try:
            import mlx.core as mx

            mlx_version = importlib.metadata.version("mlx")
            prior = mx.default_device()
            mx.set_default_device(mx.gpu)
            probe = mx.array([1.0], dtype=mx.float32) + 1
            mx.eval(probe)
            device = str(mx.default_device()).upper()
            runtime_available = "GPU" in device
            mx.set_default_device(prior)
        except Exception:
            runtime_available = False
            device = "UNAVAILABLE"
    return {
        "schema_version": "1.0",
        "engine_id": ENGINE_ID,
        "engine_version": ENGINE_VERSION,
        "capability_id": CAPABILITY_ID,
        "backend_id": BACKEND_ID,
        "compute_class": "GPU",
        "vendor": "Apple",
        "runtime": "MLX",
        "runtime_version": mlx_version,
        "device": device,
        "status": "AVAILABLE" if runtime_available else "UNAVAILABLE",
        "network_required": False,
        "estimated_cost_usd": 0.0,
        "routable": False,
        "qualification": "REQUIRES_EXACT_ATTESTATION",
    }


def plan(request: Mapping[str, Any]) -> dict[str, Any]:
    """Create a content-addressed plan for a generated matrix multiplication."""
    workload = _workload(request)
    availability = discover()
    if availability["status"] != "AVAILABLE":
        raise AcceleratorEngineError("Apple Metal MLX backend is unavailable")
    rows, inner, columns = workload["rows"], workload["inner"], workload["columns"]
    unsigned = {
        "schema_version": "1.0",
        "engine_id": ENGINE_ID,
        "engine_version": ENGINE_VERSION,
        "adapter_version": ADAPTER_VERSION,
        "capability_id": CAPABILITY_ID,
        "backend_id": BACKEND_ID,
        "workload_digest": _digest(workload),
        "estimated_floating_point_operations": 2 * rows * inner * columns,
        "estimated_memory_bytes": 4 * (rows * inner + inner * columns + rows * columns),
        "network_required": False,
        "estimated_cost_usd": 0.0,
        "approval_required": True,
        "no_fallback": True,
    }
    return {**unsigned, "plan_digest": _digest(unsigned)}


def execute(request: Mapping[str, Any]) -> dict[str, Any]:
    """Execute only on MLX's GPU device and return bounded numerical evidence."""
    if not isinstance(request, Mapping) or set(request) != {"workload", "plan_digest"}:
        raise AcceleratorEngineError("accelerator execution request is invalid")
    workload = _workload(request["workload"])
    execution_plan = plan(workload)
    if request["plan_digest"] != execution_plan["plan_digest"]:
        raise AcceleratorEngineError("accelerator execution plan digest mismatch")
    try:
        import mlx.core as mx

        prior = mx.default_device()
        mx.set_default_device(mx.gpu)
        if "GPU" not in str(mx.default_device()).upper():
            raise AcceleratorEngineError("MLX did not select the Apple GPU")
        left, right = _operands(workload)
        started = time.perf_counter_ns()
        output = mx.matmul(mx.array(left), mx.array(right))
        mx.eval(output)
        elapsed = time.perf_counter_ns() - started
        values = np.asarray(output, dtype=np.float32)
        mx.set_default_device(prior)
    except AcceleratorEngineError:
        raise
    except Exception as exc:
        raise AcceleratorEngineError("Apple Metal execution failed") from exc
    reference = left @ right
    maximum_error = float(np.max(np.abs(values - reference)))
    unsigned = {
        "schema_version": "1.0",
        "engine_id": ENGINE_ID,
        "engine_version": ENGINE_VERSION,
        "adapter_version": ADAPTER_VERSION,
        "capability_id": CAPABILITY_ID,
        "backend_id": BACKEND_ID,
        "plan_digest": execution_plan["plan_digest"],
        "workload_digest": execution_plan["workload_digest"],
        "shape": [workload["rows"], workload["columns"]],
        "dtype": "float32",
        "output": values.tolist(),
        "output_bytes_digest": "sha256:" + hashlib.sha256(values.tobytes()).hexdigest(),
        "maximum_absolute_reference_error": maximum_error,
        "tolerance": workload["tolerance"],
        "reference_verified": maximum_error <= workload["tolerance"],
        "elapsed_nanoseconds": elapsed,
        "device_class": "GPU",
        "runtime": "MLX",
        "network_used": False,
        "cost_usd": 0.0,
        "fallback_used": False,
        "limitations": [
            "apple_silicon_only",
            "float32_matrix_multiply_only",
            "benchmark_performance_is_not_workload_performance",
        ],
    }
    return {**unsigned, "result_digest": _digest(unsigned)}


def verify(result: Mapping[str, Any]) -> dict[str, Any]:
    """Verify identity, numerical acceptance, output bytes, and result digest."""
    if not isinstance(result, Mapping) or "result_digest" not in result:
        raise AcceleratorEngineError("accelerator result is invalid")
    output = result.get("output")
    try:
        values = np.asarray(output, dtype=np.float32)
        bytes_digest = "sha256:" + hashlib.sha256(values.tobytes()).hexdigest()
    except (TypeError, ValueError):
        values = np.asarray([], dtype=np.float32)
        bytes_digest = ""
    unsigned = {key: value for key, value in result.items() if key != "result_digest"}
    valid = (
        result.get("engine_id") == ENGINE_ID
        and result.get("engine_version") == ENGINE_VERSION
        and result.get("adapter_version") == ADAPTER_VERSION
        and result.get("capability_id") == CAPABILITY_ID
        and result.get("backend_id") == BACKEND_ID
        and result.get("device_class") == "GPU"
        and result.get("runtime") == "MLX"
        and result.get("network_used") is False
        and result.get("cost_usd") == 0.0
        and result.get("fallback_used") is False
        and result.get("reference_verified") is True
        and _finite(result.get("maximum_absolute_reference_error"))
        and _finite(result.get("tolerance"))
        and result["maximum_absolute_reference_error"] <= result["tolerance"]
        and isinstance(result.get("shape"), list)
        and values.ndim == 2
        and list(values.shape) == result["shape"]
        and result.get("output_bytes_digest") == bytes_digest
        and _digest(unsigned) == result["result_digest"]
    )
    return {"verified": valid, "result_digest": result.get("result_digest")}


def _workload(value: object) -> dict[str, Any]:
    fields = {"workload_type", "rows", "inner", "columns", "seed", "tolerance"}
    if not isinstance(value, Mapping) or set(value) != fields:
        raise AcceleratorEngineError("accelerator workload is invalid")
    if value.get("workload_type") != "linear_algebra.matmul.float32":
        raise AcceleratorEngineError("accelerator workload type is unsupported")
    return {
        "workload_type": "linear_algebra.matmul.float32",
        "rows": _integer(value.get("rows"), 1, 256),
        "inner": _integer(value.get("inner"), 1, 256),
        "columns": _integer(value.get("columns"), 1, 256),
        "seed": _integer(value.get("seed"), 0, 2**32 - 1),
        "tolerance": _bounded(value.get("tolerance"), 1e-7, 1e-3),
    }


def _operands(workload: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    generator = np.random.default_rng(workload["seed"])
    left = generator.uniform(-1, 1, (workload["rows"], workload["inner"])).astype(np.float32)
    right = generator.uniform(-1, 1, (workload["inner"], workload["columns"])).astype(np.float32)
    return left, right


def _integer(value: object, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise AcceleratorEngineError("accelerator workload is outside bounded limits")
    return value


def _bounded(value: object, minimum: float, maximum: float) -> float:
    if not _finite(value) or not minimum <= value <= maximum:
        raise AcceleratorEngineError("accelerator workload is outside bounded limits")
    return float(value)


def _finite(value: object) -> TypeGuard[int | float]:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _digest(value: Mapping[str, Any]) -> str:
    try:
        raw = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    except (TypeError, ValueError) as exc:
        raise AcceleratorEngineError("accelerator document is not canonical JSON") from exc
    return "sha256:" + hashlib.sha256(raw).hexdigest()
