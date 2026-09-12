"""Fail-closed inventory for heterogeneous RAD Compute Engine backends."""

from __future__ import annotations

import hashlib
import json
import shutil
from collections.abc import Mapping, Sequence
from typing import Any

from vqpu.accelerator_engine import discover as discover_apple
from vqpu.compute_engine import discover as discover_cpu

_REMOTE = {
    "hpc.slurm": ("HPC", "Slurm", "sbatch"),
    "cloud.batch": ("CLOUD", "provider_adapter_required", None),
    "qpu.physical": ("QPU", "provider_adapter_required", None),
}
_APPLE_QUALIFICATION = "sha256:c5a0c47229bf5c9217f00bb5e3d7ef0808e3498c78a1ef454c80795d685c7f68"


class BackendCatalogError(ValueError):
    """Safe backend inventory or qualification rejection."""


def discover_fabric(
    *,
    apple_qualification: Mapping[str, Any] | None = None,
    configured_backends: Sequence[str] = (),
) -> dict[str, Any]:
    """Discover backend families without resolving credentials or granting authority."""
    if (
        not isinstance(configured_backends, Sequence)
        or isinstance(configured_backends, (str, bytes))
        or any(item not in _REMOTE for item in configured_backends)
        or len(set(configured_backends)) != len(configured_backends)
    ):
        raise BackendCatalogError("configured backend identifiers are invalid")
    cpu = discover_cpu()["backends"][0]
    apple = discover_apple()
    apple_qualified = apple_qualification is not None and _attest_apple(apple_qualification)
    backends = [
        {
            "backend_id": cpu["backend_id"],
            "compute_class": "CPU",
            "status": "AVAILABLE",
            "routable": True,
            "qualification": "QUALIFIED_FOR_LOCAL_CPU_QUANTUM_SIMULATION",
            "network_required": False,
            "cost_model": "ZERO_LOCAL_COST",
        },
        {
            "backend_id": apple["backend_id"],
            "compute_class": "GPU",
            "status": apple["status"],
            "routable": apple["status"] == "AVAILABLE" and apple_qualified,
            "qualification": (
                "QUALIFIED_FOR_APPLE_METAL_FLOAT32_MATMUL" if apple_qualified else "UNQUALIFIED"
            ),
            "network_required": False,
            "cost_model": "ZERO_LOCAL_COST",
        },
    ]
    configured = set(configured_backends)
    for backend_id, (compute_class, provider, executable) in _REMOTE.items():
        detected = executable is not None and shutil.which(executable) is not None
        is_configured = backend_id in configured
        backends.append(
            {
                "backend_id": backend_id,
                "compute_class": compute_class,
                "provider": provider,
                "status": "DISCOVERED"
                if detected
                else "CONFIGURED"
                if is_configured
                else "UNAVAILABLE",
                "routable": False,
                "qualification": "UNQUALIFIED",
                "network_required": backend_id != "hpc.slurm",
                "cost_model": "REQUIRED_BEFORE_QUALIFICATION",
                "reason": "backend_requires_separate_adapter_benchmark_and_attestation",
            }
        )
    document = {
        "schema_version": "1.0",
        "engine_id": "rad-compute-engine",
        "engine_version": "0.6.0",
        "fallback_policy": "DENIED",
        "backends": sorted(backends, key=lambda item: item["backend_id"]),
    }
    return {**document, "inventory_digest": _digest(document)}


def _attest_apple(report: Mapping[str, Any]) -> bool:
    fields = {
        "schema_version",
        "engine_id",
        "engine_version",
        "adapter_version",
        "capability_id",
        "backend_id",
        "qualification",
        "discovery",
        "case_count",
        "passed_count",
        "cases",
        "limitations",
        "report_digest",
    }
    if not isinstance(report, Mapping) or set(report) != fields:
        raise BackendCatalogError("Apple GPU qualification report is invalid")
    try:
        parsed = json.loads(json.dumps(report, sort_keys=True, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise BackendCatalogError("Apple GPU qualification report is invalid") from exc
    unsigned = {key: value for key, value in parsed.items() if key != "report_digest"}
    cases = parsed.get("cases")
    expected_ids = {
        f"apple-metal-matmul-{size}-s{seed}"
        for seed in (7, 19, 43, 101, 211)
        for size in (4, 16, 64)
    }
    valid = (
        parsed.get("report_digest") == _APPLE_QUALIFICATION
        and _digest(unsigned) == _APPLE_QUALIFICATION
        and parsed.get("engine_id") == "rad-compute-engine"
        and parsed.get("engine_version") == "0.6.0"
        and parsed.get("adapter_version") == "1.0.0"
        and parsed.get("capability_id") == "rad.compute.apple_gpu"
        and parsed.get("backend_id") == "apple.metal.mlx"
        and parsed.get("qualification") == "QUALIFIED_FOR_APPLE_METAL_FLOAT32_MATMUL"
        and parsed.get("case_count") == parsed.get("passed_count") == 15
        and isinstance(cases, list)
        and {case.get("case_id") for case in cases if isinstance(case, dict)} == expected_ids
        and all(
            isinstance(case, dict)
            and case.get("outcome") == "PASS"
            and case.get("simulated") is False
            and case.get("deterministic_replay") is True
            and case.get("reference_verified") is True
            and case.get("device_class") == "GPU"
            and case.get("runtime") == "MLX"
            for case in cases
        )
    )
    if not valid:
        raise BackendCatalogError("Apple GPU qualification report is not trusted")
    return True


def _digest(value: Mapping[str, Any]) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return "sha256:" + hashlib.sha256(raw).hexdigest()
