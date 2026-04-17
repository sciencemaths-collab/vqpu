"""Push CHESSO vqpu through real IonQ: AEGIS winner vs baseline.

Submits both the CHESSO-compiled baseline gate_sequence and the
AEGIS-Ion-N(12,7,3,1) winner to the IonQ simulator (default) or a live
QPU backend if you override IONQ_BACKEND. Compares each path's Hellinger
distance to the local state-vector ideal.

Environment:
    IONQ_API_KEY       required — read from env, never from a file
    IONQ_BACKEND       default "simulator"  (try "qpu.aria-1", etc.)
    IONQ_NOISE_MODEL   default "aria-1"     (only honored by the simulator;
                                            empty string / "ideal" → noiseless)
    AEGIS_SHOTS        default 1024

This script prints, per workload:
    baseline  2Q count / IonQ Hellinger-to-ideal
    AEGIS     2Q count / IonQ Hellinger-to-ideal
    Δ         improvement (positive = AEGIS is closer to ideal).
"""

from __future__ import annotations

import math
import os
import pathlib
import sys
import time
from typing import Dict, List, Sequence, Tuple

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_ROOT))

from vqpu import QPUCloudPlugin
from vqpu.chesso import compile_qlambda_for_hardware
from vqpu.chesso.experiments import aegis_ion_nested
from vqpu.chesso.experiments.aegis_ion import circuit_depth, count_2q
from vqpu.chesso.experiments.ionq_noise import ideal_counts


# ─────────────────────────── Qλ workloads ──────────────────────────────────

def qlambda_ghz(n: int) -> str:
    lines = [f"program ghz{n}"]
    for i in range(n):
        lines.append(f"alloc q{i}")
    lines.append("gate H q0")
    for i in range(1, n):
        lines.append(f"gate CX q{i-1} q{i}")
    return "\n".join(lines) + "\n"


def qlambda_redundant_ghz(n: int) -> str:
    lines = [f"program redundant_ghz{n}"]
    for i in range(n):
        lines.append(f"alloc q{i}")
    lines.append("gate H q0")
    lines.append("gate H q0")
    lines.append("gate H q0")
    for i in range(1, n):
        lines.append(f"gate CX q{i-1} q{i}")
        lines.append(f"gate CX q{i-1} q{i}")
        lines.append(f"gate CX q{i-1} q{i}")
    return "\n".join(lines) + "\n"


def qlambda_hyperedge() -> str:
    return (
        "program hyper3\n"
        "alloc q0\n"
        "alloc q1\n"
        "alloc q2\n"
        "gate H q0\n"
        "gate H q0\n"
        "gate CX q0 q1\n"
        "gate CX q1 q2\n"
        "gate CX q1 q2\n"
        "gate CX q1 q2\n"
        "entangle q0 q1 q2 weight=1.0 apply=true\n"
    )


# ─────────────────────────── helpers ────────────────────────────────────────

def hellinger(a: Dict[str, int], b: Dict[str, int]) -> float:
    def norm(d: Dict[str, int]) -> Dict[str, float]:
        t = sum(d.values()) or 1
        return {k: v / t for k, v in d.items()}
    pa, pb = norm(a), norm(b)
    keys = set(pa) | set(pb)
    s = sum(
        (math.sqrt(pa.get(k, 0.0)) - math.sqrt(pb.get(k, 0.0))) ** 2 for k in keys
    )
    return math.sqrt(0.5 * s)


def hr(title: str) -> None:
    print("\n" + "═" * 78)
    print(f"  {title}")
    print("═" * 78)


def row(label: str, value: str) -> None:
    print(f"  {label:<34s} {value}")


def top(counts: Dict[str, int], k: int = 4) -> str:
    total = sum(counts.values()) or 1
    pairs = sorted(counts.items(), key=lambda kv: -kv[1])[:k]
    return ", ".join(f"{b}:{v/total:.3f}" for b, v in pairs)


# ─────────────────────────── driver ────────────────────────────────────────

def _submit(ionq: QPUCloudPlugin, n: int, seq: Sequence, shots: int, tag: str) -> Dict[str, int]:
    t0 = time.perf_counter()
    counts = ionq.execute_sample(n_qubits=n, gate_sequence=list(seq), shots=shots)
    dt = time.perf_counter() - t0
    row(f"IonQ submit [{tag}]", f"{dt:.1f}s  → {sum(counts.values())} shots")
    return counts


def run_one(ionq: QPUCloudPlugin, name: str, qlambda_src: str, shots: int) -> Dict:
    hr(name)
    bridged = compile_qlambda_for_hardware(qlambda_src)
    baseline_seq = list(bridged.gate_sequence)
    n = bridged.n_qubits

    t0 = time.perf_counter()
    res = aegis_ion_nested(baseline_seq, n)
    t_aegis = time.perf_counter() - t0
    winner_seq = res.winner.sequence

    b_2q, w_2q = count_2q(baseline_seq), count_2q(winner_seq)
    b_d, w_d = circuit_depth(baseline_seq, n), circuit_depth(winner_seq, n)
    row("qubits",                       f"{n}")
    row("baseline  len / 2Q / depth",   f"{len(baseline_seq):>4d}  {b_2q:>4d}  {b_d:>4d}")
    row("AEGIS     len / 2Q / depth",   f"{len(winner_seq):>4d}  {w_2q:>4d}  {w_d:>4d}")
    row("winner strategy",              res.winner.strategy)
    row("AEGIS search time",            f"{t_aegis*1000:.1f} ms")

    ideal = ideal_counts(n, baseline_seq, shots, seed=17)

    noisy_b = _submit(ionq, n, baseline_seq, shots, "baseline")
    noisy_w = _submit(ionq, n, winner_seq,   shots, "AEGIS   ")

    h_b = hellinger(noisy_b, ideal)
    h_w = hellinger(noisy_w, ideal)
    delta = h_b - h_w

    row("ideal counts (top4)",    top(ideal))
    row("IonQ baseline (top4)",   top(noisy_b))
    row("IonQ AEGIS (top4)",      top(noisy_w))
    row("Hellinger→ideal base",   f"{h_b:.4f}")
    row("Hellinger→ideal AEGIS",  f"{h_w:.4f}")
    row("Δ (positive = AEGIS wins)", f"{delta:+.4f}")

    return {
        "name": name,
        "n": n,
        "baseline_2q": b_2q,
        "winner_2q": w_2q,
        "baseline_depth": b_d,
        "winner_depth": w_d,
        "hell_base": h_b,
        "hell_aegis": h_w,
        "delta": delta,
        "winner_strategy": res.winner.strategy,
    }


def scoreboard(results: List[Dict], backend_name: str, noise_model: str) -> None:
    hr(f"AEGIS-Ion-N vs CHESSO on IonQ — {backend_name} · noise={noise_model or 'none'}")
    print(f"  {'workload':<32s}  {'n':>2s}  {'2Q b→w':>8s}  "
          f"{'H→ideal base':>13s}  {'H→ideal AEGIS':>14s}  {'Δ':>7s}")
    for r in results:
        q2 = f"{r['baseline_2q']}→{r['winner_2q']}"
        print(f"  {r['name']:<32s}  {r['n']:>2d}  {q2:>8s}  "
              f"{r['hell_base']:>13.4f}  {r['hell_aegis']:>14.4f}  {r['delta']:>+7.4f}")
    print()
    print("  H→ideal = Hellinger distance between IonQ counts and the noiseless ideal.")
    print("  Δ > 0    => AEGIS winner lands closer to the ideal than the baseline.")


def main() -> int:
    if not os.environ.get("IONQ_API_KEY"):
        print("[!] IONQ_API_KEY is not set in the environment.")
        print("    Export it for this session only:  export IONQ_API_KEY=<key>")
        return 2

    backend_name = os.environ.get("IONQ_BACKEND", "simulator")
    noise_model = os.environ.get("IONQ_NOISE_MODEL", "aria-1")
    if noise_model.lower() in {"", "ideal", "none"}:
        # User asked for noiseless simulator — unset so the plugin doesn't set it.
        os.environ.pop("IONQ_NOISE_MODEL", None)
        noise_model = ""
    shots = int(os.environ.get("AEGIS_SHOTS", "1024"))

    ionq = QPUCloudPlugin("ionq")
    fp = ionq.probe()
    if fp is None or not fp.is_available:
        print("[!] IonQ bridge is not live. Check the key and network.")
        return 1

    hr("AEGIS-Ion-N(12,7,3,1) pushing CHESSO on IonQ")
    row("backend",        f"{fp.name}  ({backend_name})")
    row("noise model",    noise_model or "none (ideal simulator)")
    row("shots",          str(shots))

    workloads: List[Tuple[str, str]] = [
        ("GHZ-5 (clean)",          qlambda_ghz(5)),
        ("GHZ-5 redundant pairs",  qlambda_redundant_ghz(5)),
        ("GHZ-7 redundant pairs",  qlambda_redundant_ghz(7)),
        ("3-body hyperedge+noise", qlambda_hyperedge()),
    ]

    results: List[Dict] = []
    for name, src in workloads:
        results.append(run_one(ionq, name, src, shots))

    scoreboard(results, fp.name, noise_model)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
