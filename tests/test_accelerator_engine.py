from copy import deepcopy

import pytest

from vqpu.accelerator_engine import AcceleratorEngineError, discover, execute, plan, verify


def workload(seed=7, size=8):
    return {
        "workload_type": "linear_algebra.matmul.float32",
        "rows": size,
        "inner": size,
        "columns": size,
        "seed": seed,
        "tolerance": 1e-4,
    }


@pytest.mark.skipif(discover()["status"] != "AVAILABLE", reason="Apple Metal MLX unavailable")
def test_real_apple_gpu_execution_is_bound_verified_and_has_no_fallback():
    item = workload()
    execution_plan = plan(item)
    result = execute({"workload": item, "plan_digest": execution_plan["plan_digest"]})

    assert result["backend_id"] == "apple.metal.mlx"
    assert result["device_class"] == "GPU"
    assert result["runtime"] == "MLX"
    assert result["fallback_used"] is False
    assert result["reference_verified"] is True
    assert verify(result)["verified"] is True


@pytest.mark.skipif(discover()["status"] != "AVAILABLE", reason="Apple Metal MLX unavailable")
def test_real_apple_gpu_output_replays_exactly():
    item = workload(seed=19)
    digest = plan(item)["plan_digest"]
    first = execute({"workload": item, "plan_digest": digest})
    second = execute({"workload": item, "plan_digest": digest})
    assert first == second


@pytest.mark.skipif(discover()["status"] != "AVAILABLE", reason="Apple Metal MLX unavailable")
def test_tampering_and_wrong_plan_fail_closed():
    item = workload()
    with pytest.raises(AcceleratorEngineError, match="digest mismatch"):
        execute({"workload": item, "plan_digest": "sha256:" + "0" * 64})
    result = execute({"workload": item, "plan_digest": plan(item)["plan_digest"]})
    tampered = deepcopy(result)
    tampered["fallback_used"] = True
    assert verify(tampered)["verified"] is False


@pytest.mark.parametrize("dimension", [0, 257])
def test_unbounded_workloads_are_rejected(dimension):
    item = workload()
    item["rows"] = dimension
    with pytest.raises(AcceleratorEngineError, match="bounded"):
        plan(item)
