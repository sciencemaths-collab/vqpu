"""Phantom vs baseline simulator — formal benchmark.

Compares PhantomSimulatorBackend against ClassicalSimulatorBackend
across circuit categories that stress different entanglement regimes.
The structured-vs-unstructured hypothesis is the thing under test:

    Phantom should win on structured circuits (local / sparse
    entanglement) and break even on unstructured ones (dense, random,
    all-to-all). Correctness (output distribution fidelity) must hold
    everywhere — a memory/time win means nothing if the answer changes.

What gets measured per circuit
──────────────────────────────
  • correctness    Bhattacharyya overlap + total-variation distance
                   between Phantom's counts and the baseline's counts.
  • memory         estimated dense statevector bytes vs Phantom's
                   factorized (subsystem × active-states) bytes.
  • wall clock     sim vs phantom `execute()` time.
  • phantom shape  #core subsystems, #classical qubits, merge events,
                   peak active-state count per subsystem, pruned
                   probability mass, fidelity lower bound.

Usage
─────
    python examples/phantom_benchmark.py              # full suite
    python examples/phantom_benchmark.py --quick      # tiny circuits only
    python examples/phantom_benchmark.py --json out.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Callable, Dict, List

import numpy as np

# Path shim so this runs without `pip install -e .`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vqpu import (  # noqa: E402
    ClassicalSimulatorBackend,
    PhantomPruningConfig,
    PhantomSimulatorBackend,
    QuantumCircuit,
    build_phantom_partition,
)


# ────────────────────────────────────────────────────────────────────
# Distribution-distance metrics
# ────────────────────────────────────────────────────────────────────

def bhattacharyya(counts_a: Dict[str, int], counts_b: Dict[str, int]) -> float:
    """Σ √(p_k q_k). 1.0 for identical distributions, 0.0 for disjoint.
    Used as a shot-count-limited metric; see bhattacharyya_exact() for
    the sampling-noise-free version when both sides provide probabilities."""
    ta = sum(counts_a.values()) or 1
    tb = sum(counts_b.values()) or 1
    keys = set(counts_a) | set(counts_b)
    return float(sum(
        math.sqrt((counts_a.get(k, 0) / ta) * (counts_b.get(k, 0) / tb))
        for k in keys
    ))


def tv_distance(counts_a: Dict[str, int], counts_b: Dict[str, int]) -> float:
    """Total-variation distance ∈ [0, 1]. 0 for identical."""
    ta = sum(counts_a.values()) or 1
    tb = sum(counts_b.values()) or 1
    keys = set(counts_a) | set(counts_b)
    return 0.5 * float(sum(
        abs(counts_a.get(k, 0) / ta - counts_b.get(k, 0) / tb)
        for k in keys
    ))


def bhattacharyya_exact(p: Dict[int, float], q: Dict[int, float]) -> float:
    """Bhattacharyya coefficient between two exact probability distributions.
    1.0 iff the distributions match, independent of shot count."""
    keys = set(p) | set(q)
    return float(sum(math.sqrt(p.get(k, 0.0) * q.get(k, 0.0)) for k in keys))


def tv_distance_exact(p: Dict[int, float], q: Dict[int, float]) -> float:
    keys = set(p) | set(q)
    return 0.5 * float(sum(abs(p.get(k, 0.0) - q.get(k, 0.0)) for k in keys))


def baseline_exact_probs(statevector) -> Dict[int, float]:
    """|ψ|² from a dense complex statevector. Guard against None."""
    if statevector is None:
        return {}
    return {
        int(i): float(abs(amp) ** 2)
        for i, amp in enumerate(statevector)
        if abs(amp) ** 2 > 1e-15
    }


# ────────────────────────────────────────────────────────────────────
# Circuit generators — grouped by expected entanglement regime
# ────────────────────────────────────────────────────────────────────

def make_product(n: int) -> QuantumCircuit:
    """No entanglement — every qubit stays factorized."""
    c = QuantumCircuit(n, f"product_{n}")
    for i in range(n):
        (c.h(i) if i % 2 == 0 else c.x(i))
    for i in range(n):
        c.rz(i, np.pi / 5)
    return c


def make_disconnected(n: int, block: int = 3) -> QuantumCircuit:
    """k non-overlapping blocks, each internally entangled; blocks stay separable."""
    c = QuantumCircuit(n, f"disconn_{n}_b{block}")
    for start in range(0, n, block):
        end = min(start + block, n)
        c.h(start)
        for i in range(start + 1, end):
            c.cnot(start, i)
    return c


def make_linear_chain(n: int) -> QuantumCircuit:
    """Nearest-neighbour Hadamard + CNOT chain — textbook MPS-friendly."""
    c = QuantumCircuit(n, f"chain_{n}")
    for i in range(n):
        c.h(i)
    for i in range(n - 1):
        c.cnot(i, i + 1)
    return c


def make_ghz(n: int) -> QuantumCircuit:
    """Star entanglement from qubit 0 — strongly entangled, small active set."""
    c = QuantumCircuit(n, f"ghz_{n}")
    c.h(0)
    for i in range(1, n):
        c.cnot(0, i)
    return c


def make_qft(n: int) -> QuantumCircuit:
    """QFT approximation (H + Rz + CNOT). Dense entanglement growth."""
    c = QuantumCircuit(n, f"qft_{n}")
    for i in range(n):
        c.h(i)
        for j in range(i + 1, n):
            c.rz(j, np.pi / float(2 ** (j - i)))
            c.cnot(i, j)
    return c


def make_all_to_all(n: int) -> QuantumCircuit:
    """Every qubit entangled with every other — Phantom should NOT win here."""
    c = QuantumCircuit(n, f"all2all_{n}")
    for i in range(n):
        c.h(i)
    for i in range(n):
        for j in range(i + 1, n):
            c.cnot(i, j)
    return c


def make_ring(n: int) -> QuantumCircuit:
    """Cyclic nearest-neighbor entangling. No articulation points, so the
    whole component lands as a single multi-qubit classical region — the
    path that routes to MPS with bond_dim > 1."""
    c = QuantumCircuit(n, f"ring_{n}")
    c.h(0)
    for i in range(n - 1):
        c.cnot(i, i + 1)
    c.cnot(n - 1, 0)
    return c


def make_brickwall(n: int, depth: int = 2) -> QuantumCircuit:
    """Shallow brick-wall circuit: alternating even/odd CNOT layers with Ry
    sandwiches. Bond dim should stay bounded for shallow depth — prime
    territory for bond_dim > 1 MPS."""
    c = QuantumCircuit(n, f"brick_{n}_d{depth}")
    for _ in range(depth):
        for i in range(n):
            c.ry(i, np.pi / 4)
        for i in range(0, n - 1, 2):
            c.cnot(i, i + 1)
        for i in range(n):
            c.ry(i, np.pi / 4)
        for i in range(1, n - 1, 2):
            c.cnot(i, i + 1)
    return c


CATEGORIES: Dict[str, Dict] = {
    "PRODUCT (no entanglement)":   {"gen": make_product,      "expected": "WIN"},
    "DISCONNECTED BLOCKS":         {"gen": lambda n: make_disconnected(n, 3),
                                    "expected": "WIN"},
    "LINEAR CHAIN (nn CNOT)":      {"gen": make_linear_chain, "expected": "WIN"},
    "GHZ (star entanglement)":     {"gen": make_ghz,          "expected": "WIN"},
    "QFT (growing entanglement)":  {"gen": make_qft,          "expected": "tie/lose"},
    "ALL-TO-ALL (dense)":          {"gen": make_all_to_all,   "expected": "tie/lose"},
    "RING (no-bridge cycle)":      {"gen": make_ring,         "expected": "WIN (MPS)"},
    "BRICK-WALL (shallow)":        {"gen": lambda n: make_brickwall(n, 2),
                                    "expected": "WIN (MPS)"},
}


# ────────────────────────────────────────────────────────────────────
# Bench harness
# ────────────────────────────────────────────────────────────────────

@dataclass
class Row:
    category: str
    circuit: str
    n_qubits: int
    n_gates: int
    sim_ms: float
    phantom_ms: float
    bhattacharyya: float
    tv: float
    bc_exact: float            # exact-vs-exact, immune to shot noise
    tv_exact: float
    dense_bytes: int
    phantom_bytes: int
    memory_ratio: float        # dense / phantom (static partition estimate)
    runtime_bytes: int         # peak live amplitudes × 16B (what actually got used)
    runtime_ratio: float       # dense / runtime_bytes
    core_subsystems: int
    classical_qubits: int
    merges: int
    splits: int
    peak_active_states: int
    min_fidelity_lower_bound: float
    pruned_mass: float
    max_bond_dim: int          # highest bond dim observed in any MPS subsystem
    truncations: int           # total SVD truncation events

    def as_dict(self) -> dict:
        return asdict(self)


def run_one(category: str, circuit: QuantumCircuit, shots: int, seed: int) -> Row:
    sim = ClassicalSimulatorBackend(seed=seed)
    t0 = time.perf_counter()
    sim_result = sim.execute(circuit, shots=shots)
    sim_ms = (time.perf_counter() - t0) * 1000

    phantom = PhantomSimulatorBackend(
        seed=seed,
        pruning=PhantomPruningConfig(amplitude_threshold=1e-10),
    )
    t0 = time.perf_counter()
    phantom_result = phantom.execute(circuit, shots=shots)
    phantom_ms = (time.perf_counter() - t0) * 1000

    partition = build_phantom_partition(circuit)
    meta = phantom_result.execution_metadata or {}
    subsystems = meta.get("subsystems", [])
    merges = meta.get("merge_events", [])
    splits = meta.get("split_events", [])
    phantom_probs = meta.get("final_probabilities", {}) or {}
    exact_probs = baseline_exact_probs(sim_result.statevector)

    peak_active = max((s.get("peak_active_states", 0) for s in subsystems),
                      default=0)
    min_fid = min((s.get("fidelity_lower_bound", 1.0) for s in subsystems),
                  default=1.0)
    pruned_mass = max((s.get("pruned_probability", 0.0) for s in subsystems),
                      default=0.0)
    mps_infos = [s.get("mps_info") for s in subsystems if s.get("mps_info")]
    max_bond = max((m.get("peak_bond_dim", 1) for m in mps_infos), default=1)
    truncations = sum(m.get("truncation_events", 0) for m in mps_infos)

    dense_bytes = partition.estimated_dense_bytes
    phantom_bytes = max(partition.estimated_factorized_bytes, 1)

    # Runtime memory = sum over subsystems of (peak_active × 16B).
    # This is the ground truth: what the process actually held in memory.
    runtime_bytes = max(
        sum(s.get("peak_active_states", 0) * 16 for s in subsystems),
        1,
    )

    return Row(
        category=category,
        circuit=circuit.name,
        n_qubits=circuit.n_qubits,
        n_gates=len(circuit.ops),
        sim_ms=sim_ms,
        phantom_ms=phantom_ms,
        bhattacharyya=bhattacharyya(sim_result.counts, phantom_result.counts),
        tv=tv_distance(sim_result.counts, phantom_result.counts),
        bc_exact=(bhattacharyya_exact(exact_probs, phantom_probs)
                  if exact_probs and phantom_probs else float("nan")),
        tv_exact=(tv_distance_exact(exact_probs, phantom_probs)
                  if exact_probs and phantom_probs else float("nan")),
        dense_bytes=dense_bytes,
        phantom_bytes=phantom_bytes,
        memory_ratio=dense_bytes / phantom_bytes,
        runtime_bytes=runtime_bytes,
        runtime_ratio=dense_bytes / runtime_bytes,
        core_subsystems=len(partition.core_subsystems),
        classical_qubits=sum(len(s.qubits) for s in partition.classical_subsystems),
        merges=len(merges),
        splits=len(splits),
        peak_active_states=peak_active,
        min_fidelity_lower_bound=min_fid,
        pruned_mass=pruned_mass,
        max_bond_dim=max_bond,
        truncations=truncations,
    )


# ────────────────────────────────────────────────────────────────────
# Rendering
# ────────────────────────────────────────────────────────────────────

def _fmt_bytes(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:>6.1f}{unit}"
        n /= 1024
    return f"{n:>6.1f}TB"


def render_row(r: Row) -> str:
    # Use the exact-vs-exact fidelity when we have it; counts-based is
    # polluted by shot noise on high-entropy distributions.
    have_exact = not math.isnan(r.bc_exact)
    fid_for_gate = r.bc_exact if have_exact else r.bhattacharyya
    fid_mark = "✓" if fid_for_gate > 0.999 else ("~" if fid_for_gate > 0.97 else "✗")
    if have_exact:
        fid_cell = f"BC*={r.bc_exact:.4f} TV*={r.tv_exact:.4f}"
    else:
        fid_cell = f"BC={r.bhattacharyya:.4f} TV={r.tv:.4f}"
    static_cell = f"{r.memory_ratio:>6.1f}×"
    runtime_cell = f"{r.runtime_ratio:>6.1f}×"
    gap = "" if r.memory_ratio / max(r.runtime_ratio, 0.01) < 2.0 else "  ⚠static-inflated"
    return (
        f"  {r.circuit:<22s}  n={r.n_qubits:>2d}  gates={r.n_gates:>3d}  "
        f"{fid_mark} {fid_cell}  "
        f"dense={_fmt_bytes(r.dense_bytes)} "
        f"static={_fmt_bytes(r.phantom_bytes)}({static_cell}) "
        f"runtime={_fmt_bytes(r.runtime_bytes)}({runtime_cell})"
        f"{gap}  "
        f"subs={r.core_subsystems}+{r.classical_qubits}cq  "
        f"merges={r.merges:<2d} splits={r.splits:<3d} "
        f"peak_active={r.peak_active_states:>4d}  "
        f"χ={r.max_bond_dim:<3d} trunc={r.truncations:<2d} "
        f"t:{r.sim_ms:>5.1f}→{r.phantom_ms:>5.1f}ms"
    )


def render_category_summary(category: str, rows: List[Row], expected: str) -> str:
    if not rows:
        return f"  {category}: (no data)"
    avg_static = float(np.mean([r.memory_ratio for r in rows]))
    avg_runtime = float(np.mean([r.runtime_ratio for r in rows]))
    # Prefer exact metric when present (immune to shot noise on uniform
    # distributions); fall back to counts-based for partial runs.
    exact_vals = [r.bc_exact for r in rows if not math.isnan(r.bc_exact)]
    if exact_vals:
        avg_fid = float(np.mean(exact_vals))
        min_fid = min(exact_vals)
        fid_label = "fidelity*"
    else:
        avg_fid = float(np.mean([r.bhattacharyya for r in rows]))
        min_fid = min(r.bhattacharyya for r in rows)
        fid_label = "fidelity "
    # Verdict uses the RUNTIME ratio — it's what actually happened, not what
    # the static partitioner hoped would happen.
    if avg_runtime >= 4.0:
        verdict = "WIN     "
    elif avg_runtime >= 1.5:
        verdict = "edge    "
    elif avg_runtime >= 0.75:
        verdict = "tie     "
    else:
        verdict = "lose    "
    # If static claims a win but runtime didn't deliver, call it out —
    # that's the gap dynamic re-splitting would close.
    regression = ""
    if avg_static >= 4.0 and avg_runtime < 1.5:
        regression = "  ⚠static-overpromised"
    return (
        f"  {verdict}{category:<30s}  "
        f"static={avg_static:>7.1f}×  runtime={avg_runtime:>7.1f}×  "
        f"{fid_label}={avg_fid:.4f} (min {min_fid:.4f})  "
        f"expected={expected}{regression}"
    )


# ────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quick", action="store_true",
                    help="Small circuits only — runs in <5s.")
    ap.add_argument("--shots", type=int, default=4096,
                    help="Shots per backend per circuit (default 4096).")
    ap.add_argument("--seed", type=int, default=20260414,
                    help="RNG seed for reproducibility.")
    ap.add_argument("--json", dest="json_out", type=Path, default=None,
                    help="Write machine-readable results to this file.")
    args = ap.parse_args()

    if args.quick:
        sizes = {"small": [4, 6]}
    else:
        # Cap at 12 qubits so the dense baseline stays well under a second.
        sizes = {"small": [4, 6], "medium": [8, 10], "large": [12]}
    all_sizes = [n for lst in sizes.values() for n in lst]

    print("╔" + "═" * 76 + "╗")
    print("║  Phantom vs Baseline Simulator — Formal Benchmark                           ║")
    print("║  hypothesis: Phantom wins on structured circuits, ties on unstructured.    ║")
    print("╚" + "═" * 76 + "╝")
    print(f"  shots={args.shots}  seed={args.seed}  qubit sizes={all_sizes}")

    rows_by_category: Dict[str, List[Row]] = {}
    for category, cfg in CATEGORIES.items():
        print(f"\n━━━ {category}  (expected: {cfg['expected']})")
        rows: List[Row] = []
        for n in all_sizes:
            try:
                circuit = cfg["gen"](n)
                row = run_one(category, circuit, args.shots, args.seed)
            except Exception as exc:  # pragma: no cover — surfaced to user
                print(f"  [!] {cfg['gen'].__name__}({n}) raised {exc!r}")
                continue
            rows.append(row)
            print(render_row(row))
        rows_by_category[category] = rows

    print("\n" + "═" * 78)
    print("  SUMMARY")
    print("═" * 78)
    print(f"  {'verdict':8s}{'category':<30s}  "
          f"{'avg memory ratio':<22s} {'fidelity':<24s} expected")
    print("  " + "─" * 76)
    for category, rows in rows_by_category.items():
        expected = CATEGORIES[category]["expected"]
        print(render_category_summary(category, rows, expected))

    # Correctness gate — use the exact metric when available (it is immune
    # to shot-noise artifacts on high-entropy distributions). A circuit
    # fails only if its exact-vs-exact fidelity drops below 0.999.
    all_rows = [r for rows in rows_by_category.values() for r in rows]
    def fid_for(r: Row) -> float:
        return r.bc_exact if not math.isnan(r.bc_exact) else r.bhattacharyya
    threshold = 0.999
    failed = [r for r in all_rows if fid_for(r) < threshold]
    print("\n" + "═" * 78)
    if failed:
        print(f"  CORRECTNESS FAIL — {len(failed)} circuits below fidelity {threshold}:")
        for r in failed:
            label = "BC*" if not math.isnan(r.bc_exact) else "BC "
            print(f"    {r.category}/{r.circuit}  {label}={fid_for(r):.6f}  "
                  f"TV={r.tv:.4f}")
        exit_code = 1
    else:
        print(f"  CORRECTNESS OK — all {len(all_rows)} circuits fidelity ≥ "
              f"{threshold} (min={min(fid_for(r) for r in all_rows):.6f}).")
        exit_code = 0

    # Honesty: the static-vs-runtime gap IS the backlog.
    inflated = [r for r in all_rows
                if r.memory_ratio >= 4.0 and r.runtime_ratio < 1.5]
    print("\n  Static-vs-runtime gap (= what dynamic re-splitting would close):")
    if inflated:
        print(f"    {len(inflated)} circuits had optimistic static partitions but")
        print("    filled the full active set at runtime. These are the cases where")
        print("    Phantom merges subsystems during execution and never splits them")
        print("    back apart — so the classical regions collapse into one dense core.")
        for r in sorted(inflated, key=lambda x: -x.memory_ratio)[:5]:
            print(f"      • {r.category}/{r.circuit}: "
                  f"static {r.memory_ratio:.0f}× vs runtime {r.runtime_ratio:.1f}×"
                  f"  (peak_active={r.peak_active_states}/{2**r.n_qubits})")
    else:
        print("    None — runtime memory matched the static estimate on every run.")
    print("\n  Remaining limitations (by design of this implementation):")
    print("    • Merges involving an MPS subsystem fall back to sparse core")
    print("      rather than preserving MPS structure across the merge —")
    print("      conservative for correctness, occasionally wasteful.")
    print("      Sparse→MPS demotion re-collapses it on the next gate.")
    print("    • Demotion uses a dense reshape+SVD sweep, so it is guarded")
    print("      at ≤16 qubits per subsystem. Wider subsystems would need")
    print("      a sparse-native Schmidt decomposition — not wired today.")
    print("    • Phantom's wall-clock can beat the dense simulator even when")
    print("      the active set is full (e.g. QFT), because the sparse-dict")
    print("      code path avoids some of the baseline's tensor overhead.")

    if args.json_out is not None:
        payload = {
            "seed": args.seed,
            "shots": args.shots,
            "sizes": all_sizes,
            "categories": {
                cat: {
                    "expected": CATEGORIES[cat]["expected"],
                    "rows": [r.as_dict() for r in rows],
                }
                for cat, rows in rows_by_category.items()
            },
        }
        args.json_out.write_text(json.dumps(payload, indent=2))
        print(f"\n  JSON results written to {args.json_out}")

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
