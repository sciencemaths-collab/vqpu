import pytest

from vqpu.accelerator_engine import discover
from vqpu.accelerator_qualification import qualify


@pytest.mark.skipif(discover()["status"] != "AVAILABLE", reason="Apple Metal MLX unavailable")
def test_formal_apple_metal_qualification_is_complete_and_real():
    report = qualify()
    assert report["qualification"] == "QUALIFIED_FOR_APPLE_METAL_FLOAT32_MATMUL"
    assert report["passed_count"] == report["case_count"] == 15
    assert all(case["simulated"] is False for case in report["cases"])
    assert all(case["deterministic_replay"] for case in report["cases"])
    assert all(case["reference_verified"] for case in report["cases"])
