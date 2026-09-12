from copy import deepcopy

import pytest

from vqpu.accelerator_engine import discover
from vqpu.accelerator_qualification import qualify
from vqpu.backend_catalog import BackendCatalogError, discover_fabric


def qualification():
    return qualify()


def test_remote_backend_families_are_visible_and_unroutable():
    inventory = discover_fabric(configured_backends=("cloud.batch", "qpu.physical"))
    found = {item["backend_id"]: item for item in inventory["backends"]}
    assert inventory["fallback_policy"] == "DENIED"
    for backend_id in ("hpc.slurm", "cloud.batch", "qpu.physical"):
        assert found[backend_id]["routable"] is False
        assert found[backend_id]["qualification"] == "UNQUALIFIED"
    assert "credential" not in str(inventory).lower()


@pytest.mark.skipif(discover()["status"] != "AVAILABLE", reason="Apple Metal MLX unavailable")
def test_exact_apple_qualification_makes_only_apple_gpu_routable():
    inventory = discover_fabric(apple_qualification=qualification())
    found = {item["backend_id"]: item for item in inventory["backends"]}
    assert found["apple.metal.mlx"]["routable"] is True
    assert found["apple.metal.mlx"]["qualification"] == ("QUALIFIED_FOR_APPLE_METAL_FLOAT32_MATMUL")
    assert found["qpu.physical"]["routable"] is False


@pytest.mark.skipif(discover()["status"] != "AVAILABLE", reason="Apple Metal MLX unavailable")
def test_tampered_qualification_cannot_enable_gpu_routing():
    report = deepcopy(qualification())
    report["cases"][0]["simulated"] = True
    with pytest.raises(BackendCatalogError, match="not trusted"):
        discover_fabric(apple_qualification=report)


def test_unknown_or_duplicate_backend_configuration_is_rejected():
    with pytest.raises(BackendCatalogError):
        discover_fabric(configured_backends=("unknown",))
    with pytest.raises(BackendCatalogError):
        discover_fabric(configured_backends=("hpc.slurm", "hpc.slurm"))
