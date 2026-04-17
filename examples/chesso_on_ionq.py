"""CHESSO → IonQ end-to-end demo.

Parses a Qλ-style program, lowers it through the CHESSO compiler, bridges
the compiled plan into vqpu's gate_sequence, and runs the same sequence
on two substrates:

  • CPUPlugin              — vqpu's local statevector simulator
  • QPUCloudPlugin("ionq") — IonQ's cloud (simulator by default; set
                             IONQ_BACKEND=qpu.forte-1 to hit hardware)

Each program is run on both substrates and the Hellinger distance between
the two count distributions is reported. A correct bridge means the
distance is within sampling noise (≈ 1/√shots).

Usage:
    export IONQ_API_KEY="<your key>"
    python3 examples/chesso_on_ionq.py
"""

from __future__ import annotations

import math
import os
import pathlib
import sys
import time
from typing import Dict

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from vqpu import CPUPlugin, QPUCloudPlugin
from vqpu.chesso import compile_qlambda_for_hardware


# ─────────────────────────────── helpers ──────────────────────────────────

def _renorm(counts: Dict[str, int]) -> Dict[str, float]:
    total = sum(counts.values()) or 1
    return {k: v / total for k, v in counts.items()}


def hellinger(a: Dict[str, int], b: Dict[str, int]) -> float:
    """Hellinger distance between two count distributions.

    0 = identical, 1 = disjoint support. For two samples of the same
    distribution, the expected value scales ~ 1/√shots.
    """
    pa, pb = _renorm(a), _renorm(b)
    keys = set(pa) | set(pb)
    s = sum((math.sqrt(pa.get(k, 0.0)) - math.sqrt(pb.get(k, 0.0))) ** 2 for k in keys)
    return math.sqrt(0.5 * s)


def fmt_counts(counts: Dict[str, int], top: int = 6) -> str:
    if not counts:
        return "(empty)"
    total = sum(counts.values())
    rows = sorted(counts.items(), key=lambda kv: -kv[1])[:top]
    out = []
    for bits, n in rows:
        bar = "█" * int((n / total) * 30)
        out.append(f"       {bits}  {n:>5d}  {n/total:.3f}  {bar}")
    return "\n".join(out)


def header(s: str) -> None:
    print("\n" + "═" * 72)
    print(f"  {s}")
    print("═" * 72)


# ───────────────────────────── Qλ programs ────────────────────────────────

GHZ3 = """
program ghz3
alloc q0
alloc q1
alloc q2
gate H q0
gate CX q0 q1
gate CX q1 q2
"""

# Uses CHESSO's *hypergraph entangler* — a diagonal n-qubit phase unitary
# that no other stack exposes as a first-class primitive. The bridge lowers
# it to vqpu's FULL_UNITARY path, which qiskit-ionq transpiles to native
# Aria gates on submission.
HYPEREDGE_DEMO = """
program hyperedge_2q
alloc q0
alloc q1
gate H q0
gate H q1
entangle q0 q1 weight=3.14159265 apply=true
gate H q0
gate H q1
"""

HYPEREDGE_GHZ_LIKE = """
program hyperedge_3q_phase
alloc q0
alloc q1
alloc q2
gate H q0
gate CX q0 q1
gate CX q1 q2
entangle q0 q1 q2 weight=1.0 apply=true
"""


# ───────────────────────────── driver ─────────────────────────────────────

def run_program(name: str, source: str, shots: int, ionq: QPUCloudPlugin, cpu: CPUPlugin) -> None:
    header(f"Program: {name}")
    circuit = compile_qlambda_for_hardware(source)
    print(f"  Qλ sectors   : {circuit.n_qubits} qubits  "
          f"(mapping: {circuit.sector_to_qubit})")
    print(f"  bridged ops  : {circuit.gate_count()}  "
          f"({', '.join(str(g[0]) for g in circuit.gate_sequence)})")
    for note in circuit.notes:
        print(f"  [bridge note] {note}")

    t0 = time.time()
    local = cpu.execute_sample(
        n_qubits=circuit.n_qubits,
        gate_sequence=circuit.gate_sequence,
        shots=shots,
    )
    local_dt = time.time() - t0

    t0 = time.time()
    remote = ionq.execute_sample(
        n_qubits=circuit.n_qubits,
        gate_sequence=circuit.gate_sequence,
        shots=shots,
    )
    remote_dt = time.time() - t0

    print(f"\n  vqpu CPU sim  ({local_dt:.2f}s):")
    print(fmt_counts(local))
    print(f"\n  IonQ cloud sim  ({remote_dt:.2f}s):")
    print(fmt_counts(remote))

    h = hellinger(local, remote)
    print(f"\n  Hellinger(local, remote) = {h:.4f}  "
          f"(sampling-noise floor ≈ {1/math.sqrt(shots):.4f})")


def main() -> int:
    if not os.environ.get("IONQ_API_KEY"):
        print("[!] IONQ_API_KEY is not set — export it first.")
        return 2

    backend_name = os.environ.get("IONQ_BACKEND", "simulator")
    shots = int(os.environ.get("CHESSO_IONQ_SHOTS",
                               "2000" if backend_name.startswith("simulator") else "512"))

    ionq = QPUCloudPlugin("ionq")
    fp = ionq.probe()
    if fp is None or not fp.is_available:
        print("[!] IonQ bridge failed to come online.")
        return 1

    cpu = CPUPlugin()

    header("CHESSO → vqpu → IonQ bridge demo")
    print(f"  IonQ backend : {fp.name}  ({backend_name})")
    print(f"  Local backend: vqpu CPUPlugin")
    print(f"  Shots        : {shots}")

    run_program("GHZ-3 via CX ladder (baseline)", GHZ3, shots, ionq, cpu)
    run_program("2-qubit hyperedge sandwich  (FULL_UNITARY path)", HYPEREDGE_DEMO, shots, ionq, cpu)
    run_program("GHZ-3 + 3-body hyperedge phase  (CHESSO primitive)",
                HYPEREDGE_GHZ_LIKE, shots, ionq, cpu)

    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
