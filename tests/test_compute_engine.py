from copy import deepcopy

import pytest

from vqpu.compute_engine import ComputeEngineError, describe, discover, execute, plan, verify


def bell(seed=7):
    return {
        "workload_type": "quantum.simulation",
        "qubits": 2,
        "shots": 1000,
        "seed": seed,
        "gates": [{"name": "H", "targets": [0]}, {"name": "CNOT", "targets": [0, 1]}],
    }


def test_contract_discovery_is_local_and_non_billing():
    assert describe()["network_access"] == "DENIED"
    backend = discover()["backends"][0]
    assert backend["backend_id"] == "cpu.quantum_simulator"
    assert backend["network_required"] is False
    assert backend["cost_per_run_usd"] == 0


def test_plan_is_deterministic_bounded_and_requires_approval():
    first = plan(bell())
    assert first == plan(bell())
    assert first["approval_required"] is True
    assert first["estimated_memory_bytes"] == 64
    assert first["network_required"] is False


def test_real_cpu_simulation_replays_and_verifies():
    workload = bell()
    first = execute({"workload": workload, "plan_digest": plan(workload)["plan_digest"]})
    second = execute({"workload": workload, "plan_digest": plan(workload)["plan_digest"]})
    assert first == second
    assert sum(first["counts"].values()) == 1000
    assert set(first["counts"]) <= {"00", "11"}
    assert verify(first)["verified"] is True


def test_tampered_plan_and_result_fail_closed():
    workload = bell()
    with pytest.raises(ComputeEngineError, match="digest mismatch"):
        execute({"workload": workload, "plan_digest": "sha256:" + "0" * 64})
    result = execute({"workload": workload, "plan_digest": plan(workload)["plan_digest"]})
    tampered = deepcopy(result)
    tampered["counts"]["00"] += 1
    assert verify(tampered)["verified"] is False


@pytest.mark.parametrize(
    "workload",
    [
        {**bell(), "qubits": 21},
        {**bell(), "shots": 100_001},
        {**bell(), "gates": [{"name": "PYTHON", "targets": [0]}]},
        {**bell(), "gates": [{"name": "CNOT", "targets": [0, 0]}]},
        {**bell(), "gates": [{"name": "RX", "targets": [0], "angle": float("nan")}]},
        {**bell(), "workload_type": "qpu.submit"},
    ],
)
def test_unbounded_unsafe_or_remote_workloads_are_rejected(workload):
    with pytest.raises(ComputeEngineError):
        plan(workload)
