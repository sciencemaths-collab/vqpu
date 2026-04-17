"""Algorithm 1 — Grover's search.

Hypothesis
──────────
Grover amplifies the target amplitude while suppressing the other 2^n − 1
basis states. After k iterations the amplitude on the target grows as
sin((2k+1)θ) with sin²θ = 1/N, so the "significant" amplitudes concentrate
as iterations advance. PHANTOM's sparse active set should track the
narrowing support rather than carrying all 2^n entries.

Test
────
n = 4, 6, 8, 10, 12. Target = the all-ones bitstring. Oracle and diffusion
as dense unitaries. Run through PhantomSimulatorBackend; report:
  • P(target) at end                        — correctness of the algorithm
  • peak_active_states                      — does pruning materialize?
  • exact fidelity vs dense baseline        — no math drift
  • runtime memory vs dense baseline        — apples-to-apples win ratio
"""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from vqpu import (  # noqa: E402
    ClassicalSimulatorBackend,
    GateOp,
    PhantomSimulatorBackend,
    QuantumCircuit,
    build_phantom_partition,
)


def grover_circuit(n: int, target: int) -> QuantumCircuit:
    dim = 2 ** n
    c = QuantumCircuit(n, f"grover_n{n}_t{target}")
    for q in range(n):
        c.h(q)

    oracle = np.eye(dim, dtype=complex)
    oracle[target, target] = -1.0

    diffusion = np.full((dim, dim), 2.0 / dim, dtype=complex) \
        - np.eye(dim, dtype=complex)

    iterations = max(1, int(round(math.pi / 4 * math.sqrt(dim))))
    all_qubits = list(range(n))
    for _ in range(iterations):
        c.ops.append(GateOp("Oracle", oracle, all_qubits, is_two_qubit=True))
        c.ops.append(GateOp("Diffuse", diffusion, all_qubits, is_two_qubit=True))
    return c


def main() -> None:
    print("  Grover search — scaling peak_active_states vs 2^n")
    print("  " + "─" * 70)
    header = f"{'n':>3}  {'dim':>5}  {'iters':>5}  {'P(target)':>9}  " \
             f"{'peak_active':>11}  {'dense':>8}  {'runtime':>8}  " \
             f"{'ratio':>6}  {'BC*':>7}"
    print("  " + header)

    for n in (4, 6, 8, 10, 12):
        target = (1 << n) - 1  # all-ones
        circuit = grover_circuit(n, target)

        # Baseline sim: exact probabilities + counts at 2048 shots.
        t0 = time.perf_counter()
        sim_result = ClassicalSimulatorBackend(seed=1).execute(circuit, shots=2048)
        t_sim = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        phantom_result = PhantomSimulatorBackend(seed=1).execute(circuit, shots=2048)
        t_ph = (time.perf_counter() - t0) * 1000

        # Partition telemetry
        part = build_phantom_partition(circuit)
        dense_bytes = part.estimated_dense_bytes

        # Runtime memory: sum(peak_active per subsystem × 16B)
        subs = (phantom_result.execution_metadata or {}).get("subsystems", [])
        runtime_bytes = max(
            sum(s.get("peak_active_states", 0) * 16 for s in subs),
            1,
        )
        peak_active = max((s.get("peak_active_states", 0) for s in subs),
                          default=0)
        target_bits = format(target, f"0{n}b")
        p_target = sim_result.counts.get(target_bits, 0) / 2048

        # Exact fidelity baseline|phantom
        exact_probs = {
            int(i): float(abs(a) ** 2)
            for i, a in enumerate(sim_result.statevector)
            if abs(a) ** 2 > 1e-15
        }
        phantom_probs = (phantom_result.execution_metadata or {}).get(
            "final_probabilities", {}
        )
        keys = set(exact_probs) | set(phantom_probs)
        bc = sum(
            math.sqrt(exact_probs.get(k, 0.0) * phantom_probs.get(k, 0.0))
            for k in keys
        )

        iters = max(1, int(round(math.pi / 4 * math.sqrt(2 ** n))))
        ratio = dense_bytes / runtime_bytes
        print(f"  {n:>3}  {2**n:>5}  {iters:>5}  {p_target:>9.4f}  "
              f"{peak_active:>11d}  {dense_bytes:>7d}B  "
              f"{runtime_bytes:>7d}B  {ratio:>5.1f}×  {bc:>6.4f}")


if __name__ == "__main__":
    main()
