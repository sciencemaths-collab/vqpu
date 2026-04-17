"""Heterogeneous circuit knitting — split a circuit across backends.

Demonstrates cutting a 10-qubit GHZ circuit into two 5-qubit fragments,
executing each independently, and reconstructing the exact full-register
distribution.

For controlled gates (CNOT, CZ), the decomposition is exact: measure
the control in Z, classically forward the result to prepare the
downstream target. Zero sampling overhead.
"""

from vqpu import UniversalvQPU, CutFinder, CircuitKnitter


def demo_ghz_cut():
    """Cut a 10-qubit GHZ at the q4→q5 CNOT boundary."""

    qpu = UniversalvQPU()
    c = qpu.circuit(10, "ghz_10")
    c.h(0)
    for i in range(9):
        c.cnot(i, i + 1)

    print("=== 10-qubit GHZ, cut into 2×5 ===")
    print(f"Circuit: {c.n_qubits} qubits, {len(c.ops)} gates")

    plan = CutFinder.auto_partition(c, max_fragment_qubits=5)
    print(f"Partitions: {[sorted(p) for p in plan.partitions]}")
    print(f"Cuts: {plan.n_cuts}")
    for cut in plan.cuts:
        print(f"  {cut}")

    knitter = CircuitKnitter(plan)
    result = knitter.run(executor=qpu.run, shots=4096)

    print(f"\nReconstructed ({result.n_fragment_executions} executions):")
    for bs in sorted(result.counts, key=result.counts.get, reverse=True)[:5]:
        print(f"  |{bs}⟩  {result.counts[bs]:>5}")
    print(f"Wall time: {result.wall_time_s:.3f}s")

    ref = qpu.run(c, shots=4096)
    print(f"\nReference (full 10-qubit sim):")
    for bs in sorted(ref.counts, key=ref.counts.get, reverse=True)[:5]:
        print(f"  |{bs}⟩  {ref.counts[bs]:>5}")


def demo_manual_partition():
    """Manually partition an 8-qubit circuit at the q3→q4 bridge."""

    qpu = UniversalvQPU()
    c = qpu.circuit(8, "split_chain")

    for i in range(8):
        c.h(i)
    c.cnot(0, 1).cnot(1, 2).cnot(2, 3)
    c.cnot(4, 5).cnot(5, 6).cnot(6, 7)
    c.cnot(3, 4)
    for i in range(8):
        c.rx(i, 0.3)

    print("\n=== 8-qubit, manual partition at q3↔q4 ===")
    plan = CutFinder.partition_manual(c, [{0,1,2,3}, {4,5,6,7}])
    print(f"Cuts: {plan.n_cuts}")

    knitter = CircuitKnitter(plan)
    result = knitter.run(executor=qpu.run, shots=2048)

    print(f"\nTop outcomes ({result.n_fragment_executions} executions):")
    for bs in sorted(result.counts, key=result.counts.get, reverse=True)[:6]:
        print(f"  |{bs}⟩  {result.counts[bs]:>5}")

    er = result.to_execution_result()
    print(f"\nExecutionResult backend: {er.backend_name}")
    print(f"  metadata: {er.execution_metadata}")


def demo_passthrough():
    """No cuts needed — knitter is a transparent passthrough."""

    qpu = UniversalvQPU()
    c = qpu.circuit(4, "small_bell")
    c.h(0).cnot(0, 1).h(2).cnot(2, 3)

    plan = CutFinder.auto_partition(c, max_fragment_qubits=10)
    assert plan.n_cuts == 0

    knitter = CircuitKnitter(plan)
    result = knitter.run(executor=qpu.run, shots=1024)

    print("\n=== No-cut passthrough ===")
    for bs in sorted(result.counts, key=result.counts.get, reverse=True):
        print(f"  |{bs}⟩  {result.counts[bs]:>5}")


if __name__ == "__main__":
    demo_passthrough()
    demo_manual_partition()
    demo_ghz_cut()
