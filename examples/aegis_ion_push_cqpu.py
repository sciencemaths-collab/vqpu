"""Show AEGIS-Ion-N(12,7,3,1) pushing the CHESSO vqpu (cqpu) compiler output.

Pipeline per workload:
    Qλ source  →  chesso bridge  →  baseline gate_sequence
                                 →  AEGIS-Ion-N(12,7,3,1)  →  winner gate_sequence

Metrics reported for each path:
    n_2q       two-qubit gate count (FULL_UNITARY(n) → n(n−1)/2)
    depth      ASAP depth on the vqpu gate graph
    length     total gate count
    hellinger  sampled distance between baseline and AEGIS winner distributions
    fidelity   |⟨ψ_baseline | ψ_aegis⟩|²  from the local state-vector simulator

No IONQ_API_KEY needed — runs on vqpu's CPUPlugin state-vector backend.
"""

from __future__ import annotations

import math
import pathlib
import sys
import time
from typing import Dict, List

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_ROOT))

from vqpu.chesso import compile_qlambda_for_hardware
from vqpu.chesso.experiments import aegis_ion_nested
from vqpu.chesso.experiments.aegis_ion import circuit_depth, count_2q
from vqpu.chesso.experiments.ionq_noise import (
    IonQNoiseSpec,
    expected_circuit_fidelity,
    ideal_counts,
    sample_with_ionq_noise,
)
from vqpu.universal import CPUPlugin


# ─────────────────────────── workload definitions ──────────────────────────

def qlambda_ghz(n: int) -> str:
    lines = [f"program ghz{n}"]
    for i in range(n):
        lines.append(f"alloc q{i}")
    lines.append("gate H q0")
    for i in range(1, n):
        lines.append(f"gate CX q{i-1} q{i}")
    return "\n".join(lines) + "\n"


def qlambda_redundant_ghz(n: int) -> str:
    """GHZ with redundant H·H and CNOT·CNOT pairs sprinkled in.

    A realistic upstream might emit these from naive decomposition or from a
    templated macro expansion. AEGIS should flatten them back out.
    """
    lines = [f"program redundant_ghz{n}"]
    for i in range(n):
        lines.append(f"alloc q{i}")
    lines.append("gate H q0")
    lines.append("gate H q0")  # pair ➀
    lines.append("gate H q0")
    for i in range(1, n):
        lines.append(f"gate CX q{i-1} q{i}")
        lines.append(f"gate CX q{i-1} q{i}")  # pair ➁
        lines.append(f"gate CX q{i-1} q{i}")
    return "\n".join(lines) + "\n"


def qlambda_rotation_cluster(n: int) -> str:
    """Adjacent Rz on the same wire — merges into a single rotation."""
    lines = [f"program rot{n}"]
    for i in range(n):
        lines.append(f"alloc q{i}")
    lines.append("gate H q0")
    for i in range(1, n):
        lines.append(f"gate CX q{i-1} q{i}")
        lines.append(f"gate Rz q{i} theta=0.3")
        lines.append(f"gate Rz q{i} theta=0.7")
        lines.append(f"gate Rz q{i} theta=-1.0")  # exact cancel with 0.3+0.7
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


# ────────────────────────────── measurement ────────────────────────────────

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


# ───────────────────────────── reporting ───────────────────────────────────

def hr(title: str) -> None:
    print("\n" + "═" * 78)
    print(f"  {title}")
    print("═" * 78)


def row(label: str, value: str) -> None:
    print(f"  {label:<32s} {value}")


def short_counts(counts: Dict[str, int], top: int = 4) -> str:
    total = sum(counts.values())
    pairs = sorted(counts.items(), key=lambda kv: -kv[1])[:top]
    return ", ".join(f"{k}:{v/total:.3f}" for k, v in pairs)


# ─────────────────────────────── driver ────────────────────────────────────

def run(
    name: str,
    qlambda_src: str,
    shots: int,
    noise_shots: int,
    cpu: CPUPlugin,
    spec: IonQNoiseSpec,
) -> Dict:
    hr(name)

    bridged = compile_qlambda_for_hardware(qlambda_src)
    baseline_seq = list(bridged.gate_sequence)
    n = bridged.n_qubits

    t0 = time.perf_counter()
    res = aegis_ion_nested(baseline_seq, n)
    t_aegis = time.perf_counter() - t0

    winner_seq = res.winner.sequence

    # Ideal distribution — what a perfect machine would return.
    ideal = ideal_counts(n, baseline_seq, noise_shots, seed=7)

    # Ideal-vs-ideal sanity check on the AEGIS rewrite.
    baseline_counts = cpu.execute_sample(n, baseline_seq, shots)
    aegis_counts = cpu.execute_sample(n, winner_seq, shots)
    h_aegis_vs_base = hellinger(baseline_counts, aegis_counts)

    # Hardware-realistic: each path through IonQ Aria-spec noise.
    noisy_baseline = sample_with_ionq_noise(n, baseline_seq, noise_shots, spec=spec, seed=11)
    noisy_aegis = sample_with_ionq_noise(n, winner_seq, noise_shots, spec=spec, seed=13)
    h_base_vs_ideal = hellinger(noisy_baseline, ideal)
    h_aegis_vs_ideal = hellinger(noisy_aegis, ideal)

    b_2q = count_2q(baseline_seq)
    w_2q = count_2q(winner_seq)
    b_d = circuit_depth(baseline_seq, n)
    w_d = circuit_depth(winner_seq, n)
    f_pred_base = expected_circuit_fidelity(baseline_seq, spec)
    f_pred_winner = expected_circuit_fidelity(winner_seq, spec)

    row("qubits",                f"{n}")
    row("baseline  length / 2Q / depth", f"{len(baseline_seq):>4d}  {b_2q:>4d}  {b_d:>4d}")
    row("AEGIS     length / 2Q / depth", f"{len(winner_seq):>4d}  {w_2q:>4d}  {w_d:>4d}")
    row("winner strategy",       res.winner.strategy)
    row("equiv fidelity (AEGIS ≡ base)", f"{res.winner.metrics.fidelity:.6f}")
    row("Hellinger (AEGIS vs base, noiseless)",
        f"{h_aegis_vs_base:.4f}  (floor≈{1/math.sqrt(shots):.4f})")
    row("expected F (IonQ-spec) base → AEGIS",
        f"{f_pred_base:.4f} → {f_pred_winner:.4f}  "
        f"(Δ={100*(f_pred_winner-f_pred_base):+.2f} pp)")
    row("Hellinger-to-ideal (noisy)",
        f"base={h_base_vs_ideal:.4f}  AEGIS={h_aegis_vs_ideal:.4f}  "
        f"(Δ={h_base_vs_ideal-h_aegis_vs_ideal:+.4f})")
    row("search time",           f"{t_aegis*1000:.1f} ms")
    row("ideal counts (top4)",   short_counts(ideal))
    row("noisy base (top4)",     short_counts(noisy_baseline))
    row("noisy AEGIS (top4)",    short_counts(noisy_aegis))

    return {
        "name": name,
        "n": n,
        "baseline_2q": b_2q,
        "winner_2q": w_2q,
        "baseline_depth": b_d,
        "winner_depth": w_d,
        "baseline_len": len(baseline_seq),
        "winner_len": len(winner_seq),
        "fidelity_equiv": res.winner.metrics.fidelity,
        "hellinger_rewrite": h_aegis_vs_base,
        "f_pred_base": f_pred_base,
        "f_pred_winner": f_pred_winner,
        "hell_base_ideal": h_base_vs_ideal,
        "hell_aegis_ideal": h_aegis_vs_ideal,
        "winner_strategy": res.winner.strategy,
        "search_ms": t_aegis * 1000,
    }


def scoreboard(results: List[Dict]) -> None:
    hr("AEGIS-Ion-N vs CHESSO baseline — hardware-realistic scoreboard (IonQ Aria spec)")
    print(f"  {'workload':<38s}  {'n':>2s}  {'2Q b→w':>8s}  {'F_pred b→w':>14s}  "
          f"{'Hell→ideal b→w':>16s}  {'Δ':>7s}")
    for r in results:
        q2 = f"{r['baseline_2q']}→{r['winner_2q']}"
        fp = f"{r['f_pred_base']:.3f}→{r['f_pred_winner']:.3f}"
        hh = f"{r['hell_base_ideal']:.3f}→{r['hell_aegis_ideal']:.3f}"
        delta = r["hell_base_ideal"] - r["hell_aegis_ideal"]
        print(f"  {r['name']:<38s}  {r['n']:>2d}  {q2:>8s}  {fp:>14s}  {hh:>16s}  "
              f"{delta:>+7.3f}")
    print()
    print("  F_pred    = IonQ-Aria-spec expected circuit fidelity (ε_2Q=6e-3, ε_1Q=2e-4).")
    print("  Hell→ideal = Hellinger distance from noisy sampled counts to the ideal.")
    print("  Δ > 0      => AEGIS winner is closer to the ideal distribution than baseline.")


def main() -> int:
    shots = 2048
    noise_shots = 600  # Monte Carlo is heavier; keep it tractable.
    cpu = CPUPlugin()
    spec = IonQNoiseSpec.aria()
    hr("AEGIS-Ion-N(12,7,3,1) pushing CHESSO vqpu — IonQ Aria noise preview")
    row("ideal backend",    "vqpu CPUPlugin (state-vector)")
    row("noisy backend",    "Pauli-channel Monte Carlo, IonQ Aria spec")
    row("ε_1Q / ε_2Q / SPAM",
        f"{spec.e_1q:.1e} / {spec.e_2q:.1e} / {spec.e_spam:.1e}")
    row("ideal shots / noise shots", f"{shots} / {noise_shots}")

    results: List[Dict] = []
    results.append(run("GHZ-5 (clean)",                 qlambda_ghz(5),              shots, noise_shots, cpu, spec))
    results.append(run("GHZ-7 (clean)",                 qlambda_ghz(7),              shots, noise_shots, cpu, spec))
    results.append(run("GHZ-5 with redundant pairs",    qlambda_redundant_ghz(5),    shots, noise_shots, cpu, spec))
    results.append(run("GHZ-7 with redundant pairs",    qlambda_redundant_ghz(7),    shots, noise_shots, cpu, spec))
    results.append(run("Rz cluster (4q, merge+cancel)", qlambda_rotation_cluster(4), shots, noise_shots, cpu, spec))
    results.append(run("3-body hyperedge + noise",      qlambda_hyperedge(),         shots, noise_shots, cpu, spec))

    scoreboard(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
