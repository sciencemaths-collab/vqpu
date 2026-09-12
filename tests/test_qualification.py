from vqpu.qualification import qualify


def test_formal_local_cpu_qualification_is_complete_and_reproducible():
    first = qualify()
    assert first == qualify()
    assert first["qualification"] == "QUALIFIED_FOR_LOCAL_CPU_QUANTUM_SIMULATION"
    assert first["passed_count"] == first["case_count"] == 15
    assert all(case["deterministic_replay"] and case["verified"] for case in first["cases"])
