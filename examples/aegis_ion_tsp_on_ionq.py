"""Hard 4-city TSP on vqpu via IonQ — AEGIS-Ion-N vs. baseline QAOA.

Problem
───────
Traveling Salesman on 4 cities with distances chosen so that one tour is
strictly shorter than the two alternatives:

    d =  0  10  15  20       optimal tour : 0 → 1 → 3 → 2 → 0   length 80
         10   0  35  25      alt. tours   : length 95   (both)
         15  35   0  30
         20  25  30   0

We fix city 0 at time-slot 0 to break the rotational symmetry; the
remaining 3 cities live in 3 time slots, so the QUBO has 9 binary
variables → 9 qubits. Only (4-1)!/2 = 3 tours are feasible out of 2^9 =
512 configurations. The QUBO is a textbook hard-constraint TSP:

  H = A · Σ_i (1 - Σ_t x_{i,t})²  +  A · Σ_t (1 - Σ_i x_{i,t})²
       ─────────────────────────    ─────────────────────────
       each city visited once       one city per time slot

    + B · Σ d_{i,j} x_{i,t} x_{j,t+1}    (tour length)

After x = (1-z)/2, this becomes an Ising Hamiltonian that we compile to
a QAOA p=1 circuit and run on IonQ.

What we compare
───────────────
    baseline   CHESSO → bridge → QAOA gate_sequence submitted as-is
    AEGIS      same gate_sequence run through AEGIS-Ion-N(12,7,3,1)

Both are submitted to `ionq_simulator` with `noise_model=aria-1` (the same
calibration as a live QPU). We then decode bitstrings → tours and report:
    - 2Q gate count pre-submit
    - P(optimal)            fraction of shots that land on the optimal tour
    - P(feasible)           fraction of shots that land on ANY valid tour
    - <tour length | feas>  mean tour length conditioned on feasibility
    - approx ratio          optimal / <tour length> — higher is better

Environment:
    IONQ_API_KEY   required (jobs:write scope)
    IONQ_BACKEND   default "ionq_simulator"
    AEGIS_SHOTS    default 1024
    QAOA_P         default 1
"""

from __future__ import annotations

import itertools
import math
import os
import pathlib
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_ROOT))

from vqpu import QPUCloudPlugin
from vqpu.chesso.experiments import aegis_ion_nested
from vqpu.chesso.experiments.aegis_ion import circuit_depth, count_2q
from vqpu.universal import CPUPlugin

GateSeq = List[Tuple]

# ─────────────────────────── TSP instance ──────────────────────────────────

DIST = np.array([
    [0, 10, 15, 20],
    [10, 0, 35, 25],
    [15, 35, 0, 30],
    [20, 25, 30, 0],
], dtype=float)

N_CITIES = 4
N_OTHER = N_CITIES - 1       # cities to place (1,2,3)
N_SLOTS = N_CITIES - 1       # time slots to fill (1,2,3); slot 0 fixed to city 0
N_QUBITS = N_OTHER * N_SLOTS  # 9

# Penalty weights: A dominates B so constraint violation is never worth it.
PENALTY_A = 150.0
WEIGHT_B = 1.0


def qubit_index(city: int, slot: int) -> int:
    """city ∈ {1,2,3}, slot ∈ {1,2,3}. Flat index into 9-qubit register."""
    return (city - 1) * N_SLOTS + (slot - 1)


def brute_force_tours() -> List[Tuple[Tuple[int, ...], float]]:
    """All (tour_tuple, tour_length) with city 0 fixed as start/end."""
    tours: List[Tuple[Tuple[int, ...], float]] = []
    for perm in itertools.permutations([1, 2, 3]):
        full = (0,) + perm + (0,)
        length = sum(DIST[full[i], full[i + 1]] for i in range(N_CITIES))
        tours.append((full, float(length)))
    tours.sort(key=lambda p: p[1])
    return tours


# ─────────────────────────── QUBO → Ising ──────────────────────────────────

@dataclass
class Ising:
    """Ising Hamiltonian  H = Σ h[i] Z_i  +  Σ J[(i,j)] Z_i Z_j  +  offset."""

    h: np.ndarray                   # shape (n_qubits,)
    J: Dict[Tuple[int, int], float] # keys (i<j)
    offset: float = 0.0

    def energy(self, bitstring: str) -> float:
        # bitstring: '0' → +1, '1' → −1 under z = 1 - 2x  convention below.
        z = np.array([1 if b == "0" else -1 for b in bitstring], dtype=float)
        e = self.offset
        e += float(np.dot(self.h, z))
        for (i, j), J in self.J.items():
            e += J * z[i] * z[j]
        return e


def build_tsp_ising() -> Ising:
    """Build Ising Hamiltonian for the 4-city TSP QUBO with symmetry reduction."""
    q = np.zeros((N_QUBITS, N_QUBITS))  # QUBO matrix in x-space, x ∈ {0,1}
    linear = np.zeros(N_QUBITS)
    const = 0.0

    # Constraint 1: every "other" city must appear exactly once across slots.
    #   PENALTY_A · Σ_i (1 − Σ_t x_{i,t})²
    #   = PENALTY_A · Σ_i [1 − 2 Σ_t x_{i,t} + (Σ_t x_{i,t})²]
    for city in (1, 2, 3):
        idxs = [qubit_index(city, t) for t in (1, 2, 3)]
        const += PENALTY_A
        for k in idxs:
            linear[k] += -2.0 * PENALTY_A
        for k in idxs:
            q[k, k] += PENALTY_A          # x²_k = x_k when binary
        for a in range(len(idxs)):
            for b in range(a + 1, len(idxs)):
                q[idxs[a], idxs[b]] += 2.0 * PENALTY_A

    # Constraint 2: every time slot must have exactly one city from {1,2,3}.
    # (Slot 0 is fixed to city 0 so it drops out.)
    for slot in (1, 2, 3):
        idxs = [qubit_index(city, slot) for city in (1, 2, 3)]
        const += PENALTY_A
        for k in idxs:
            linear[k] += -2.0 * PENALTY_A
        for k in idxs:
            q[k, k] += PENALTY_A
        for a in range(len(idxs)):
            for b in range(a + 1, len(idxs)):
                q[idxs[a], idxs[b]] += 2.0 * PENALTY_A

    # Distance cost: edges 0→city(1), city(t)→city(t+1), city(3)→0.
    # Transitions from/to city 0 are linear in x; interior transitions are
    # quadratic in x.
    for city in (1, 2, 3):
        linear[qubit_index(city, 1)] += WEIGHT_B * DIST[0, city]   # 0 → slot1
        linear[qubit_index(city, 3)] += WEIGHT_B * DIST[city, 0]   # slot3 → 0
    for t in (1, 2):
        for i in (1, 2, 3):
            for j in (1, 2, 3):
                if i == j:
                    continue
                a, b = qubit_index(i, t), qubit_index(j, t + 1)
                if a < b:
                    q[a, b] += WEIGHT_B * DIST[i, j]
                else:
                    q[b, a] += WEIGHT_B * DIST[i, j]

    # Collapse diagonal QUBO entries into the linear term (x² = x).
    for k in range(N_QUBITS):
        linear[k] += q[k, k]
        q[k, k] = 0.0

    # QUBO → Ising via x = (1 − z) / 2.
    h = np.zeros(N_QUBITS)
    J: Dict[Tuple[int, int], float] = {}
    offset = const
    for k in range(N_QUBITS):
        offset += linear[k] / 2.0
        h[k] += -linear[k] / 2.0
    for a in range(N_QUBITS):
        for b in range(a + 1, N_QUBITS):
            qij = q[a, b]
            if qij == 0.0:
                continue
            offset += qij / 4.0
            h[a] += -qij / 4.0
            h[b] += -qij / 4.0
            J[(a, b)] = qij / 4.0
    return Ising(h=h, J=J, offset=offset)


# ──────────────────────── QAOA gate-sequence builder ───────────────────────

def qaoa_gate_sequence(ising: Ising, gammas: Sequence[float], betas: Sequence[float]) -> GateSeq:
    """Emit a QAOA gate_sequence in the vqpu format."""
    p = len(gammas)
    seq: GateSeq = []
    # Initial superposition.
    for q in range(N_QUBITS):
        seq.append(("H", [q]))
    for layer in range(p):
        gamma = float(gammas[layer])
        beta = float(betas[layer])
        # Cost unitary: single-Z rotations.
        for q in range(N_QUBITS):
            angle = 2.0 * gamma * float(ising.h[q])
            if abs(angle) > 1e-12:
                seq.append(("Rz", [q], angle))
        # Cost unitary: ZZ rotations. Group by qubit pair for AEGIS-friendly
        # cancellation: two adjacent blocks on the same pair fold together.
        for (i, j), Jij in sorted(ising.J.items()):
            angle = 2.0 * gamma * float(Jij)
            if abs(angle) < 1e-12:
                continue
            seq.append(("CNOT", [i, j]))
            seq.append(("Rz", [j], angle))
            seq.append(("CNOT", [i, j]))
        # Mixer: Rx(2β) on every qubit.
        for q in range(N_QUBITS):
            seq.append(("Rx", [q], 2.0 * beta))
    return seq


# ────────────────────────── bitstring → tour ───────────────────────────────

def decode_tour(bitstring: str) -> Tuple[bool, Tuple[int, ...] | None, float | None]:
    """Big-endian qubit 0 first. Returns (feasible, tour, length)."""
    bits = np.array([int(c) for c in bitstring], dtype=int).reshape(N_OTHER, N_SLOTS)
    # Feasibility: each "other" city exactly once across slots, each slot one city.
    if not np.all(bits.sum(axis=1) == 1):
        return False, None, None
    if not np.all(bits.sum(axis=0) == 1):
        return False, None, None
    slot_to_city = {}
    for city_idx in range(N_OTHER):
        for slot_idx in range(N_SLOTS):
            if bits[city_idx, slot_idx] == 1:
                slot_to_city[slot_idx + 1] = city_idx + 1
    tour = (0,) + tuple(slot_to_city[t] for t in (1, 2, 3)) + (0,)
    length = sum(DIST[tour[i], tour[i + 1]] for i in range(N_CITIES))
    return True, tour, float(length)


# ─────────────── local grid search for strong (γ, β) ───────────────────────

def _exact_energy_expectation(seq: GateSeq, ising: Ising, sv_fn) -> float:
    sv = sv_fn(N_QUBITS, seq)
    probs = np.abs(sv) ** 2
    e = 0.0
    for idx, p in enumerate(probs):
        if p == 0.0:
            continue
        bits = format(idx, f"0{N_QUBITS}b")
        e += p * ising.energy(bits)
    return float(e)


def grid_search_angles(ising: Ising, p: int, cpu: CPUPlugin,
                       n_grid: int = 9) -> Tuple[List[float], List[float], float]:
    """Coarse grid then refine once around the minimum. Keeps API calls at zero."""
    sv_fn = lambda n, seq: cpu.execute_statevector(n, seq)
    gammas = np.linspace(0.05, math.pi - 0.05, n_grid)
    betas = np.linspace(0.05, math.pi / 2 - 0.05, n_grid)
    best = (float("inf"), 0.0, 0.0)
    if p == 1:
        for g in gammas:
            for b in betas:
                seq = qaoa_gate_sequence(ising, [g], [b])
                e = _exact_energy_expectation(seq, ising, sv_fn)
                if e < best[0]:
                    best = (e, float(g), float(b))
        # Refine.
        g0, b0 = best[1], best[2]
        fine_gammas = np.linspace(max(0.01, g0 - 0.25), min(math.pi - 0.01, g0 + 0.25), 7)
        fine_betas = np.linspace(max(0.01, b0 - 0.15), min(math.pi / 2 - 0.01, b0 + 0.15), 7)
        for g in fine_gammas:
            for b in fine_betas:
                seq = qaoa_gate_sequence(ising, [g], [b])
                e = _exact_energy_expectation(seq, ising, sv_fn)
                if e < best[0]:
                    best = (e, float(g), float(b))
        return [best[1]], [best[2]], best[0]
    # p=2: start from p=1 winners, small grid around them.
    p1_g, p1_b, _ = grid_search_angles(ising, 1, cpu, n_grid=n_grid)
    g1, b1 = p1_g[0], p1_b[0]
    for g2 in np.linspace(0.05, math.pi - 0.05, 7):
        for b2 in np.linspace(0.05, math.pi / 2 - 0.05, 7):
            seq = qaoa_gate_sequence(ising, [g1, float(g2)], [b1, float(b2)])
            e = _exact_energy_expectation(seq, ising, sv_fn)
            if e < best[0]:
                best = (e, float(g2), float(b2))
    return [g1, best[1]], [b1, best[2]], best[0]


# ───────────────────────────── reporting ──────────────────────────────────

def summarize_counts(counts: Dict[str, int], optimal_length: float) -> Dict:
    total = sum(counts.values()) or 1
    n_opt = 0
    n_feas = 0
    tour_length_sum = 0.0
    tour_hist: Dict[Tuple[int, ...], int] = {}
    for bits, c in counts.items():
        feasible, tour, length = decode_tour(bits)
        if not feasible:
            continue
        n_feas += c
        tour_length_sum += c * length
        if tour is not None:
            tour_hist[tour] = tour_hist.get(tour, 0) + c
        if abs(length - optimal_length) < 1e-9:
            n_opt += c
    p_opt = n_opt / total
    p_feas = n_feas / total
    mean_len = (tour_length_sum / n_feas) if n_feas > 0 else float("nan")
    ar = (optimal_length / mean_len) if n_feas > 0 else 0.0
    return {
        "shots": total,
        "p_optimal": p_opt,
        "p_feasible": p_feas,
        "mean_length": mean_len,
        "approx_ratio": ar,
        "tour_hist": tour_hist,
    }


def hr(s: str) -> None:
    print("\n" + "═" * 78)
    print(f"  {s}")
    print("═" * 78)


def row(label: str, value: str) -> None:
    print(f"  {label:<36s} {value}")


def tour_str(tour: Tuple[int, ...]) -> str:
    return "→".join(str(c) for c in tour)


# ─────────────────────────────── main ──────────────────────────────────────

def main() -> int:
    if not os.environ.get("IONQ_API_KEY"):
        print("[!] IONQ_API_KEY is not set.")
        return 2

    p = int(os.environ.get("QAOA_P", "1"))
    shots = int(os.environ.get("AEGIS_SHOTS", "1024"))
    os.environ.setdefault("IONQ_BACKEND", "ionq_simulator")
    os.environ.setdefault("IONQ_NOISE_MODEL", "aria-1")

    hr("AEGIS-Ion-N on hard 4-city TSP — IonQ simulator + aria-1 noise")
    row("backend",     os.environ["IONQ_BACKEND"])
    row("noise model", os.environ["IONQ_NOISE_MODEL"])
    row("shots",       str(shots))
    row("QAOA p",      str(p))

    # Instance + brute force.
    tours = brute_force_tours()
    optimal_tour, optimal_length = tours[0]
    hr("Instance")
    row("distance matrix rows",
        "  ".join(str(list(int(x) for x in r)) for r in DIST[:1]))
    for r in DIST[1:]:
        print(" " * 38 + "  ".join([str(list(int(x) for x in r))]))
    row("optimal tour",    f"{tour_str(optimal_tour)}  length={optimal_length:.0f}")
    row("alternate tours", ", ".join(f"{tour_str(t)}({L:.0f})" for t, L in tours[1:]))

    # Ising + QAOA.
    ising = build_tsp_ising()
    hr("QAOA construction")
    row("qubits",    f"{N_QUBITS}  (4-city TSP with city 0 fixed at t=0)")
    row("Z terms",   str(int(np.sum(np.abs(ising.h) > 1e-12))))
    row("ZZ terms",  str(len(ising.J)))

    cpu = CPUPlugin()
    t0 = time.perf_counter()
    gammas, betas, best_E = grid_search_angles(ising, p, cpu, n_grid=9)
    t_grid = time.perf_counter() - t0
    row("grid-searched γ", ", ".join(f"{g:.3f}" for g in gammas))
    row("grid-searched β", ", ".join(f"{b:.3f}" for b in betas))
    row("best ⟨H⟩ at chosen angles", f"{best_E:+.3f}")
    row("grid search time",          f"{t_grid:.2f}s")

    baseline_seq = qaoa_gate_sequence(ising, gammas, betas)
    row("baseline seq len / 2Q / depth",
        f"{len(baseline_seq)}  {count_2q(baseline_seq)}  {circuit_depth(baseline_seq, N_QUBITS)}")

    # AEGIS-Ion-N cascade.
    hr("AEGIS-Ion-N(12,7,3,1)")
    t0 = time.perf_counter()
    res = aegis_ion_nested(baseline_seq, N_QUBITS)
    t_aegis = time.perf_counter() - t0
    winner_seq = res.winner.sequence
    row("winner strategy",   res.winner.strategy)
    row("AEGIS seq len / 2Q / depth",
        f"{len(winner_seq)}  {count_2q(winner_seq)}  {circuit_depth(winner_seq, N_QUBITS)}")
    row("equiv fidelity (local state-vec)", f"{res.winner.metrics.fidelity:.6f}")
    row("AEGIS search time",              f"{t_aegis*1000:.1f} ms")

    # Submit both to IonQ.
    ionq = QPUCloudPlugin("ionq")
    fp = ionq.probe()
    if fp is None or not fp.is_available:
        print("[!] IonQ bridge not live.")
        return 1

    hr("Submitting to IonQ")
    t0 = time.perf_counter()
    counts_baseline = ionq.execute_sample(n_qubits=N_QUBITS, gate_sequence=baseline_seq, shots=shots)
    dt_b = time.perf_counter() - t0
    row("baseline submit",  f"{dt_b:.1f}s  ({sum(counts_baseline.values())} shots)")

    t0 = time.perf_counter()
    counts_aegis = ionq.execute_sample(n_qubits=N_QUBITS, gate_sequence=winner_seq, shots=shots)
    dt_a = time.perf_counter() - t0
    row("AEGIS submit",     f"{dt_a:.1f}s  ({sum(counts_aegis.values())} shots)")

    # Decode.
    hr("Decoded tour statistics")
    stats_b = summarize_counts(counts_baseline, optimal_length)
    stats_a = summarize_counts(counts_aegis, optimal_length)

    def show(tag: str, s: Dict) -> None:
        row(f"{tag}  feasible",   f"{100*s['p_feasible']:5.2f}%   ({s['shots']} shots)")
        row(f"{tag}  optimal",    f"{100*s['p_optimal']:5.2f}%")
        row(f"{tag}  ⟨tour|feas⟩", f"{s['mean_length']:.3f}")
        row(f"{tag}  approx ratio", f"{s['approx_ratio']:.4f}")
        hist = ", ".join(
            f"{tour_str(t)}:{c}" for t, c in sorted(s["tour_hist"].items(), key=lambda kv: -kv[1])
        )
        row(f"{tag}  tour hist",   hist or "(no feasible samples)")

    show("baseline", stats_b)
    show("AEGIS   ", stats_a)

    # Scoreboard.
    hr("Scoreboard — AEGIS-Ion-N vs baseline on IonQ (aria-1 noise)")
    print(f"  {'metric':<26s}  {'baseline':>12s}  {'AEGIS':>12s}  {'Δ':>10s}")
    print(f"  {'2Q count':<26s}  {count_2q(baseline_seq):>12d}  "
          f"{count_2q(winner_seq):>12d}  {count_2q(baseline_seq)-count_2q(winner_seq):>+10d}")
    print(f"  {'depth':<26s}  {circuit_depth(baseline_seq, N_QUBITS):>12d}  "
          f"{circuit_depth(winner_seq, N_QUBITS):>12d}  "
          f"{circuit_depth(baseline_seq, N_QUBITS)-circuit_depth(winner_seq, N_QUBITS):>+10d}")
    print(f"  {'P(feasible)':<26s}  {100*stats_b['p_feasible']:>11.2f}%  "
          f"{100*stats_a['p_feasible']:>11.2f}%  "
          f"{100*(stats_a['p_feasible']-stats_b['p_feasible']):>+9.2f}pp")
    print(f"  {'P(optimal)':<26s}  {100*stats_b['p_optimal']:>11.2f}%  "
          f"{100*stats_a['p_optimal']:>11.2f}%  "
          f"{100*(stats_a['p_optimal']-stats_b['p_optimal']):>+9.2f}pp")
    b_mean = stats_b["mean_length"] if not math.isnan(stats_b["mean_length"]) else 0
    a_mean = stats_a["mean_length"] if not math.isnan(stats_a["mean_length"]) else 0
    print(f"  {'<tour|feas>':<26s}  {b_mean:>12.3f}  {a_mean:>12.3f}  "
          f"{b_mean-a_mean:>+10.3f}")
    print(f"  {'approx ratio':<26s}  {stats_b['approx_ratio']:>12.4f}  "
          f"{stats_a['approx_ratio']:>12.4f}  "
          f"{stats_a['approx_ratio']-stats_b['approx_ratio']:>+10.4f}")
    print()
    print(f"  Optimal tour: {tour_str(optimal_tour)} with length {optimal_length:.0f}.")
    print("  Δ > 0 means AEGIS-Ion-N makes the IonQ run find a better answer.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
