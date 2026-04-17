"""CHESSO vs baseline head-to-head on IonQ.

Same algorithm, two paths:
  A. Qλ → CHESSO compiler → bridge → vqpu gate_sequence
  B. Hand-written vqpu gate_sequence (what a quantum programmer
     writes today without CHESSO)

Both paths are submitted through the identical IonQ transpile pipeline
(qiskit.transpile with optimization_level=1, then IonQ native gates). We
report three metrics per algorithm:

  1. source_lines    — how many characters of code it took to express it
  2. native_gate_ops — native gate count after IonQ transpilation
                       (this is what actually runs on the ion trap)
  3. hellinger       — distance between the two paths' measured count
                       distributions  (should be at/below sampling noise)

Honest framing
──────────────
For algorithms that use only standard H/CX/Rz gates, CHESSO has no
gate-count advantage — both paths feed the same Qiskit transpiler. What
CHESSO buys you is:
  • expressivity at the Qλ source level,
  • a primitive (hyperedge entangler) no other stack exposes,
  • CHESSO-native telemetry (entanglement hypergraph, bundle topology).

The hypergraph test at the end exhibits the third point: a 3-body
hyperedge entangler lowered to hardware through the bridge. The baseline
path has no natural way to express that unitary without hand-decomposing
the diagonal matrix.

Usage:
    export IONQ_API_KEY="<your key>"
    python3 examples/chesso_vs_baseline.py
"""

from __future__ import annotations

import math
import os
import pathlib
import sys
import time
from typing import Dict, List, Tuple

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from vqpu import QPUCloudPlugin
from vqpu.chesso import compile_qlambda_for_hardware


# ───────────────────── metrics + qiskit transpile helper ──────────────────

def _append_vqpu_to_qiskit(qc, gate_sequence) -> None:
    """Mirror of QPUCloudPlugin._append_qiskit_gate for local counting."""
    for gate in gate_sequence:
        name = gate[0]
        targets = gate[1] if isinstance(gate[1], list) else [gate[1]]
        params = list(gate[2:]) if len(gate) > 2 else []
        if name in ("H", "X", "Y", "Z", "S", "T"):
            getattr(qc, name.lower())(targets[0]); continue
        if name == "CNOT":
            qc.cx(targets[0], targets[1]); continue
        if name == "CZ":
            qc.cz(targets[0], targets[1]); continue
        if name == "SWAP":
            qc.swap(targets[0], targets[1]); continue
        if name in ("Rx", "Ry", "Rz"):
            getattr(qc, name.lower())(float(params[0]), targets[0]); continue
        if name == "Phase":
            qc.p(float(params[0]), targets[0]); continue
        if name == "FULL_UNITARY":
            qc.unitary(params[0], targets, label="U"); continue
        raise ValueError(f"unsupported vqpu gate {name!r}")


def transpile_stats(gate_sequence, n_qubits: int) -> Dict[str, object]:
    """Transpile as IonQ will, then report native gate counts + depth."""
    from qiskit import QuantumCircuit, transpile
    from qiskit_ionq import IonQProvider

    provider = IonQProvider(os.environ["IONQ_API_KEY"])
    backend = provider.get_backend(os.environ.get("IONQ_BACKEND", "simulator"))
    qc = QuantumCircuit(n_qubits)
    _append_vqpu_to_qiskit(qc, gate_sequence)
    qc.measure_all()
    transpiled = transpile(qc, backend=backend, optimization_level=1)
    return {
        "native_ops": dict(transpiled.count_ops()),
        "native_total": transpiled.size(),
        "native_depth": transpiled.depth(),
    }


def hellinger(a: Dict[str, int], b: Dict[str, int]) -> float:
    def norm(d):
        t = sum(d.values()) or 1
        return {k: v / t for k, v in d.items()}
    pa, pb = norm(a), norm(b)
    keys = set(pa) | set(pb)
    s = sum(
        (math.sqrt(pa.get(k, 0.0)) - math.sqrt(pb.get(k, 0.0))) ** 2
        for k in keys
    )
    return math.sqrt(0.5 * s)


# ─────────────────────── paths A and B for GHZ-n ──────────────────────────

def qlambda_ghz_source(n: int) -> str:
    lines = [f"program ghz{n}"]
    for i in range(n):
        lines.append(f"alloc q{i}")
    lines.append("gate H q0")
    for i in range(1, n):
        lines.append(f"gate CX q{i-1} q{i}")
    return "\n".join(lines) + "\n"


def baseline_ghz_gates(n: int) -> List[Tuple]:
    seq: List[Tuple] = [("H", [0])]
    for i in range(1, n):
        seq.append(("CNOT", [i - 1, i]))
    return seq


# ─────────────────── paths A and B for 3-body hyperedge ───────────────────

def qlambda_hyperedge_source() -> str:
    return (
        "program hyper3\n"
        "alloc q0\n"
        "alloc q1\n"
        "alloc q2\n"
        "gate H q0\n"
        "gate CX q0 q1\n"
        "gate CX q1 q2\n"
        "entangle q0 q1 q2 weight=1.0 apply=true\n"
    )


def baseline_hyperedge_gates() -> List[Tuple]:
    """No one writes this by hand — you'd need a diagonal-matrix synthesizer.
    The 'baseline' here is to build the literal matrix and pass FULL_UNITARY.
    That is already borrowing CHESSO's primitive, which is the point.
    """
    from vqpu.chesso.ops.unitary_ops import make_hyperedge_phase_entangler
    m = make_hyperedge_phase_entangler((2, 2, 2), theta=1.0, phase_bias=0.0)
    return [
        ("H", [0]),
        ("CNOT", [0, 1]),
        ("CNOT", [1, 2]),
        ("FULL_UNITARY", [2, 1, 0], m),
    ]


# ───────────────────────────── report helpers ─────────────────────────────

def header(s: str) -> None:
    print("\n" + "═" * 78)
    print(f"  {s}")
    print("═" * 78)


def row(label: str, value: str) -> None:
    print(f"  {label:<34s} {value}")


def short_counts(counts: Dict[str, int], top: int = 4) -> str:
    total = sum(counts.values())
    pairs = sorted(counts.items(), key=lambda kv: -kv[1])[:top]
    return ", ".join(f"{k}:{v/total:.3f}" for k, v in pairs)


# ─────────────────────────────── main ─────────────────────────────────────

def run_pairing(name: str, qlambda_src: str, baseline_gates, shots: int,
                ionq: QPUCloudPlugin) -> Dict:
    header(name)

    # Path A — CHESSO compiler.
    bridged = compile_qlambda_for_hardware(qlambda_src)
    path_a_seq = bridged.gate_sequence
    path_a_nq = bridged.n_qubits
    path_a_src_chars = len(qlambda_src.strip())

    # Path B — hand-written gate sequence.
    path_b_seq = list(baseline_gates)
    path_b_nq = 1 + max(t for g in path_b_seq for t in g[1])
    path_b_src_chars = len(repr(path_b_seq))

    # Pre-transpile both locally to count what IonQ will actually run.
    stats_a = transpile_stats(path_a_seq, path_a_nq)
    stats_b = transpile_stats(path_b_seq, path_b_nq)

    # Submit both to IonQ simulator.
    t0 = time.time()
    counts_a = ionq.execute_sample(
        n_qubits=path_a_nq, gate_sequence=path_a_seq, shots=shots
    )
    dt_a = time.time() - t0

    t0 = time.time()
    counts_b = ionq.execute_sample(
        n_qubits=path_b_nq, gate_sequence=path_b_seq, shots=shots
    )
    dt_b = time.time() - t0

    h = hellinger(counts_a, counts_b)

    row("Qλ source length (chars)",     f"{path_a_src_chars:>6d}")
    row("baseline source length (chars)", f"{path_b_src_chars:>6d}")
    row("",                              "")
    row("A (CHESSO) native gate count",
        f"{stats_a['native_total']:>6d}  depth={stats_a['native_depth']}  "
        f"ops={stats_a['native_ops']}")
    row("B (baseline) native gate count",
        f"{stats_b['native_total']:>6d}  depth={stats_b['native_depth']}  "
        f"ops={stats_b['native_ops']}")
    row("",                              "")
    row("A counts (top 4)",              short_counts(counts_a))
    row("B counts (top 4)",              short_counts(counts_b))
    row("Hellinger(A, B)",               f"{h:.4f}  "
        f"(sampling floor ≈ {1/math.sqrt(shots):.4f}; "
        f"A:{dt_a:.1f}s  B:{dt_b:.1f}s)")

    return {
        "name": name,
        "source_a": path_a_src_chars,
        "source_b": path_b_src_chars,
        "native_a": stats_a["native_total"],
        "native_b": stats_b["native_total"],
        "hellinger": h,
    }


def main() -> int:
    if not os.environ.get("IONQ_API_KEY"):
        print("[!] IONQ_API_KEY is not set.")
        return 2

    backend_name = os.environ.get("IONQ_BACKEND", "simulator")
    shots = int(os.environ.get("CHESSO_IONQ_SHOTS",
                               "2000" if backend_name.startswith("simulator") else "512"))
    ionq = QPUCloudPlugin("ionq")
    fp = ionq.probe()
    if fp is None or not fp.is_available:
        print("[!] IonQ bridge is not live.")
        return 1

    header("CHESSO vs baseline head-to-head  (IonQ)")
    row("Backend",        f"{fp.name}  ({backend_name})")
    row("Shots",          str(shots))

    results = []
    for n in (3, 5, 7):
        r = run_pairing(
            f"GHZ-{n}  (pure H/CX — tests source-ergonomics + equivalence)",
            qlambda_ghz_source(n),
            baseline_ghz_gates(n),
            shots, ionq,
        )
        results.append(r)

    r = run_pairing(
        "3-body hyperedge  (CHESSO primitive — tests expressivity)",
        qlambda_hyperedge_source(),
        baseline_hyperedge_gates(),
        shots, ionq,
    )
    results.append(r)

    header("Scoreboard")
    print(f"  {'test':<42s}  {'srcA':>5s}  {'srcB':>5s}  "
          f"{'natA':>5s}  {'natB':>5s}  {'Hell':>7s}")
    for r in results:
        print(f"  {r['name']:<42s}  {r['source_a']:>5d}  {r['source_b']:>5d}  "
              f"{r['native_a']:>5d}  {r['native_b']:>5d}  {r['hellinger']:>7.4f}")

    print()
    print("  Legend: srcA = Qλ source chars; srcB = hand-written vqpu chars.")
    print("          natA/B = native-gate count after IonQ transpile.")
    print("          Hell   = Hellinger distance between path A/B samples.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
