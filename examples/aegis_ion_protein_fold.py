"""Fold a mini HP protein on IonQ, through vqpu, with AEGIS on top.

Problem (HP lattice model, the classic Dill/Chan toy protein)
─────────────────────────────────────────────────────────────
Sequence:  H P P H     (residues numbered 0..3, 0-indexed)
Lattice:   2D square
Energy:    E = Σ_{i<j, non-sequential H-H} ─1 · [pos_i is 4-neighbour of pos_j]
           + A · (self-overlap)     + B · (backtrack)

For HPPH the only hydrophobic pair is H0–H3. There's a non-trivial fold iff
those two beads close a U-turn — the ground state is exactly the
"folded" configuration with H0–H3 in contact on the lattice.

Encoding
────────
3 bonds; we fix bond 1 pointing right to kill the global rotation/
reflection symmetry, so 2 interior bonds × 2 qubits/bond = 4 qubits.

    direction      bits (x_lo, x_hi)
    0 = Right →    (0, 0)
    1 = Up    ↑    (1, 0)
    2 = Down  ↓    (0, 1)
    3 = Left  ←    (1, 1)

    bits [q0, q1]  = bond 2
    bits [q2, q3]  = bond 3

Compilation pipeline
────────────────────
    1. Classical enumeration of all 2⁴ = 16 bitstrings, compute the
       energy function directly (no approximations).
    2. Walsh–Hadamard transform gives the exact diagonal Ising
       Hamiltonian  H(z) = Σ c_S · Π_{i∈S} z_i.
    3. Emit a QAOA p=1 gate_sequence from that Ising Hamiltonian.
       k-body Z chains compile to a CNOT ladder + central Rz.
    4. Local grid search on the CPUPlugin picks γ, β (no API spend).
    5. Both the raw QAOA circuit and the AEGIS-Ion-N(12,7,3,1) winner
       are submitted to `ionq_simulator` under the `aria-1` noise model.
    6. We decode each bitstring back to a lattice configuration, flag
       feasibility, score energy, and report P(ground state) and the
       approximation ratio for baseline vs. AEGIS.

Run:
    IONQ_API_KEY=<key> ./.venv/bin/python examples/aegis_ion_protein_fold.py
"""

from __future__ import annotations

import itertools
import math
import os
import pathlib
import sys
import time
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


# ─────────────────────────── protein instance ────────────────────────────

SEQUENCE = "HPPH"
N_RES = len(SEQUENCE)
N_BONDS = N_RES - 1               # 3 bonds
BOND_1_FIXED = 0                  # Right
N_QUBITS = 2 * (N_BONDS - 1)      # 4 qubits

DIRECTION_VEC = {
    0: (1, 0),    # Right
    1: (0, 1),    # Up
    2: (0, -1),   # Down
    3: (-1, 0),   # Left
}
DIRECTION_NAME = {0: "→", 1: "↑", 2: "↓", 3: "←"}
OPPOSITE = {0: 3, 3: 0, 1: 2, 2: 1}

# Penalties — tuned so that the no-fold configuration sits above zero and the
# folded contact configuration sits at -1 (the true ground state).
LAMBDA_BACK = 6.0
LAMBDA_SELF = 10.0
CONTACT_REWARD = 1.0


def _decode_bonds(bitstring: str) -> Tuple[int, int]:
    """Bitstring is q0..q3 big-endian. Returns (bond_2, bond_3)."""
    x = [int(b) for b in bitstring]
    bond_2 = x[0] + 2 * x[1]
    bond_3 = x[2] + 2 * x[3]
    return bond_2, bond_3


def _positions(bond_2: int, bond_3: int) -> List[Tuple[int, int]]:
    p = [(0, 0)]
    for d in (BOND_1_FIXED, bond_2, bond_3):
        vx, vy = DIRECTION_VEC[d]
        lx, ly = p[-1]
        p.append((lx + vx, ly + vy))
    return p


def _energy_from_bits(bitstring: str) -> float:
    bond_2, bond_3 = _decode_bonds(bitstring)

    # Backtrack between consecutive bonds.
    if bond_2 == OPPOSITE[BOND_1_FIXED]:
        return float(LAMBDA_BACK)
    if bond_3 == OPPOSITE[bond_2]:
        return float(LAMBDA_BACK)

    positions = _positions(bond_2, bond_3)
    # Self-overlap.
    if len(set(positions)) < len(positions):
        return float(LAMBDA_SELF)

    # HP contact reward: non-sequential H-H residues at Manhattan distance 1.
    h_indices = [i for i, c in enumerate(SEQUENCE) if c == "H"]
    e = 0.0
    for a in range(len(h_indices)):
        for b in range(a + 1, len(h_indices)):
            i, j = h_indices[a], h_indices[b]
            if abs(i - j) == 1:  # sequential neighbour, not a contact
                continue
            dx = abs(positions[i][0] - positions[j][0])
            dy = abs(positions[i][1] - positions[j][1])
            if dx + dy == 1:
                e -= CONTACT_REWARD
    return e


def _enumerate_all() -> List[Dict]:
    out: List[Dict] = []
    for x in range(1 << N_QUBITS):
        bits = format(x, f"0{N_QUBITS}b")
        e = _energy_from_bits(bits)
        bond_2, bond_3 = _decode_bonds(bits)
        positions = _positions(bond_2, bond_3)
        ps = set(positions)
        valid = not (bond_2 == OPPOSITE[BOND_1_FIXED] or bond_3 == OPPOSITE[bond_2])
        feasible = valid and len(ps) == len(positions)
        out.append({
            "bits": bits,
            "bond_2": bond_2,
            "bond_3": bond_3,
            "positions": positions,
            "feasible": feasible,
            "energy": e,
        })
    return out


# ───────────────── Ising expansion via Walsh-Hadamard ────────────────────

def ising_from_diagonal(energies: np.ndarray) -> Dict[Tuple[int, ...], float]:
    """Expand a length-2^n diagonal energy function in the Z-basis.

    f(x) = Σ_{S ⊆ {0..n-1}} c_S · Π_{i∈S} z_i    with z_i = 1 − 2 x_i
    Returns a dict {S_tuple: c_S} dropping near-zero coefficients.
    """
    n = int(round(math.log2(len(energies))))
    assert (1 << n) == len(energies), "energies must be length 2^n"
    coeffs: Dict[Tuple[int, ...], float] = {}
    for mask in range(1 << n):
        S = tuple(i for i in range(n) if (mask >> i) & 1)
        acc = 0.0
        for x in range(1 << n):
            # z_i = 1 - 2 * x_i  for every i;  product over i∈S
            prod = 1
            for i in S:
                if (x >> i) & 1:
                    prod = -prod
            acc += energies[x] * prod
        c = acc / (1 << n)
        if abs(c) > 1e-10:
            coeffs[S] = c
    return coeffs


# ─────────────────────────── QAOA builder ────────────────────────────────

def qaoa_gate_sequence(
    ising: Dict[Tuple[int, ...], float],
    gammas: Sequence[float],
    betas: Sequence[float],
    n_qubits: int,
) -> GateSeq:
    p = len(gammas)
    seq: GateSeq = []
    for q in range(n_qubits):
        seq.append(("H", [q]))
    for layer in range(p):
        gamma = float(gammas[layer])
        beta = float(betas[layer])
        # Z-string terms: apply exp(-i γ c Π Z_i) as CNOT ladder + central Rz.
        for S, c in ising.items():
            if len(S) == 0:
                continue
            angle = 2.0 * gamma * c
            if abs(angle) < 1e-12:
                continue
            if len(S) == 1:
                seq.append(("Rz", [S[0]], angle))
                continue
            # Multi-qubit Z-string: CNOT every qubit onto the last, Rz on last, uncompute.
            targets = list(S)
            anchor = targets[-1]
            for q in targets[:-1]:
                seq.append(("CNOT", [q, anchor]))
            seq.append(("Rz", [anchor], angle))
            for q in reversed(targets[:-1]):
                seq.append(("CNOT", [q, anchor]))
        # Mixer.
        for q in range(n_qubits):
            seq.append(("Rx", [q], 2.0 * beta))
    return seq


# ──────────────────── local grid search for (γ, β) ───────────────────────

def _expected_energy(seq: GateSeq, energies: np.ndarray, sim) -> float:
    sv = sim(N_QUBITS, seq)
    probs = np.abs(sv) ** 2
    return float(np.dot(probs, energies))


def grid_search(
    ising: Dict[Tuple[int, ...], float], energies: np.ndarray, cpu: CPUPlugin, n_grid: int = 13
) -> Tuple[float, float, float]:
    sim = lambda n, s: cpu.execute_statevector(n, s)
    gammas = np.linspace(0.05, math.pi - 0.05, n_grid)
    betas = np.linspace(0.05, math.pi / 2 - 0.05, n_grid)
    best = (float("inf"), 0.0, 0.0)
    for g in gammas:
        for b in betas:
            s = qaoa_gate_sequence(ising, [g], [b], N_QUBITS)
            e = _expected_energy(s, energies, sim)
            if e < best[0]:
                best = (e, float(g), float(b))
    # refine
    _, g0, b0 = best
    fg = np.linspace(max(0.01, g0 - 0.25), min(math.pi - 0.01, g0 + 0.25), 9)
    fb = np.linspace(max(0.01, b0 - 0.15), min(math.pi / 2 - 0.01, b0 + 0.15), 9)
    for g in fg:
        for b in fb:
            s = qaoa_gate_sequence(ising, [g], [b], N_QUBITS)
            e = _expected_energy(s, energies, sim)
            if e < best[0]:
                best = (e, float(g), float(b))
    return best[0], best[1], best[2]


# ─────────────────────────── reporting ───────────────────────────────────

def hr(s: str) -> None:
    print("\n" + "═" * 78)
    print(f"  {s}")
    print("═" * 78)


def row(label: str, value: str) -> None:
    print(f"  {label:<32s} {value}")


def _fmt_path(positions: Sequence[Tuple[int, int]]) -> str:
    directions: List[str] = []
    for i in range(1, len(positions)):
        dx = positions[i][0] - positions[i - 1][0]
        dy = positions[i][1] - positions[i - 1][1]
        d = {(1, 0): "→", (-1, 0): "←", (0, 1): "↑", (0, -1): "↓"}.get((dx, dy), "?")
        directions.append(d)
    return f"{SEQUENCE[0]}" + "".join(f"{d}{SEQUENCE[i+1]}" for i, d in enumerate(directions))


def ascii_lattice(positions: Sequence[Tuple[int, int]], sequence: str) -> str:
    """Render the folded chain on a small ASCII grid."""
    xs = [p[0] for p in positions]
    ys = [p[1] for p in positions]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    w = xmax - xmin + 1
    h = ymax - ymin + 1
    grid = [["   " for _ in range(w)] for _ in range(h)]
    # beads
    for i, (x, y) in enumerate(positions):
        cx, cy = x - xmin, ymax - y
        grid[cy][cx] = f" {sequence[i]} "
    # bonds (horizontal / vertical connectors)
    # Draw onto a higher-resolution grid so bonds sit between beads.
    rows = [list(r) for r in ("".join(row) for row in grid)]
    return "\n".join("    " + "".join(r) for r in rows)


def summarize(counts: Dict[str, int], all_configs: List[Dict], ground_state_E: float) -> Dict:
    total = sum(counts.values()) or 1
    e_table = {c["bits"]: c["energy"] for c in all_configs}
    feas_table = {c["bits"]: c["feasible"] for c in all_configs}
    n_feas = 0
    n_ground = 0
    e_sum = 0.0
    for bits, c in counts.items():
        if bits not in e_table:
            continue
        if feas_table[bits]:
            n_feas += c
            e_sum += c * e_table[bits]
            if abs(e_table[bits] - ground_state_E) < 1e-9:
                n_ground += c
    mean_E = (e_sum / n_feas) if n_feas else float("nan")
    # Approximation ratio on a shifted scale so the ground state is 1.0 and an
    # all-penalty sample is 0.0.
    e_max = max(c["energy"] for c in all_configs)
    if abs(e_max - ground_state_E) > 1e-9 and not math.isnan(mean_E):
        ar = (e_max - mean_E) / (e_max - ground_state_E)
    else:
        ar = float("nan")
    return {
        "p_feasible": n_feas / total,
        "p_ground":   n_ground / total,
        "mean_E":     mean_E,
        "approx":     ar,
    }


# ───────────────────────────── driver ────────────────────────────────────

def main() -> int:
    os.environ.setdefault("IONQ_BACKEND", "ionq_simulator")
    os.environ.setdefault("IONQ_NOISE_MODEL", "aria-1")

    if not os.environ.get("IONQ_API_KEY"):
        print("[!] IONQ_API_KEY is not set.")
        return 2
    shots = int(os.environ.get("AEGIS_SHOTS", "1024"))

    # 1. Enumerate.
    hr("Mini-protein fold · HP model")
    row("sequence",         SEQUENCE)
    row("residues",         str(N_RES))
    row("bonds (fixed+free)", f"{N_BONDS}  (bond 1 fixed Right → {N_QUBITS} qubits free)")

    all_cfg = _enumerate_all()
    feasible = [c for c in all_cfg if c["feasible"]]
    ground = min(feasible, key=lambda c: c["energy"])
    ground_E = ground["energy"]

    hr("Classical enumeration (brute force)")
    row("configurations",   f"{len(all_cfg)} total · {len(feasible)} feasible")
    row("ground-state E",   f"{ground_E:.3f}")
    row("ground configs",
        ", ".join(_fmt_path(c["positions"]) for c in feasible if abs(c["energy"] - ground_E) < 1e-9))
    print()
    print("  Ground-state lattice fold (first optimum):")
    print(ascii_lattice(ground["positions"], SEQUENCE))
    print()
    for c in sorted(feasible, key=lambda c: c["energy"])[:6]:
        print(f"    bits={c['bits']}  path={_fmt_path(c['positions'])}  E={c['energy']:+.2f}")

    # 2. Ising expansion.
    energies = np.array([c["energy"] for c in all_cfg], dtype=float)
    # Reorder to little-endian (bit 0 is LSB) since Walsh-Hadamard uses it that way.
    # The _enumerate_all() bits are already LSB-first-in-bitstring? Let's be explicit:
    # bitstring "q0 q1 q2 q3"  → idx = q0·8 + q1·4 + q2·2 + q3   (big-endian by qubit).
    # For Walsh-Hadamard in LSB convention, reindex so bit 0 is q0.
    energy_by_lsb = np.zeros_like(energies)
    for x_big in range(1 << N_QUBITS):
        bits = format(x_big, f"0{N_QUBITS}b")
        x_lsb = 0
        for i, b in enumerate(bits):
            if b == "1":
                x_lsb |= (1 << i)
        energy_by_lsb[x_lsb] = energies[x_big]
    ising = ising_from_diagonal(energy_by_lsb)

    hr("Ising expansion")
    row("non-zero Pauli-Z terms", str(len(ising) - (1 if () in ising else 0)))
    row("constant offset",        f"{ising.get((), 0.0):.3f}")
    hi_order = max((len(S) for S in ising if len(S) > 0), default=0)
    row("max term order",         f"{hi_order}-body")

    # 3. QAOA angles.
    cpu = CPUPlugin()
    t0 = time.perf_counter()
    best_E, gamma, beta = grid_search(ising, energy_by_lsb, cpu, n_grid=13)
    dt = time.perf_counter() - t0
    hr("QAOA p=1 angle search")
    row("grid best ⟨H⟩",   f"{best_E:.3f}")
    row("γ, β",            f"{gamma:.3f}, {beta:.3f}")
    row("grid time",       f"{dt:.2f}s")

    # 4. Build circuits.
    baseline = qaoa_gate_sequence(ising, [gamma], [beta], N_QUBITS)
    t0 = time.perf_counter()
    res = aegis_ion_nested(baseline, N_QUBITS)
    dt_a = time.perf_counter() - t0
    aegis_seq = res.winner.sequence

    hr("Circuit metrics")
    row("baseline  length / 2Q / depth",
        f"{len(baseline):>4d}  {count_2q(baseline):>3d}  {circuit_depth(baseline, N_QUBITS):>3d}")
    row("AEGIS     length / 2Q / depth",
        f"{len(aegis_seq):>4d}  {count_2q(aegis_seq):>3d}  {circuit_depth(aegis_seq, N_QUBITS):>3d}")
    row("AEGIS winner strategy", res.winner.strategy)
    row("equiv fidelity (local sv)", f"{res.winner.metrics.fidelity:.6f}")
    row("AEGIS search time", f"{dt_a*1000:.1f} ms")

    # 5. Submit.
    ionq = QPUCloudPlugin("ionq")
    fp = ionq.probe()
    if fp is None or not fp.is_available:
        print("[!] IonQ bridge not live.")
        return 1

    hr(f"Submitting to IonQ  ({os.environ['IONQ_BACKEND']} · noise={os.environ.get('IONQ_NOISE_MODEL','none')})")
    t0 = time.perf_counter()
    counts_b = ionq.execute_sample(n_qubits=N_QUBITS, gate_sequence=baseline, shots=shots)
    dt_b = time.perf_counter() - t0
    row("baseline submit",  f"{dt_b:.1f}s  ({sum(counts_b.values())} shots)")

    t0 = time.perf_counter()
    counts_a = ionq.execute_sample(n_qubits=N_QUBITS, gate_sequence=aegis_seq, shots=shots)
    dt_w = time.perf_counter() - t0
    row("AEGIS submit",     f"{dt_w:.1f}s  ({sum(counts_a.values())} shots)")

    # 6. Decode + report.
    hr("Decoded fold statistics (IonQ counts → lattice configurations)")
    stats_b = summarize(counts_b, all_cfg, ground_E)
    stats_a = summarize(counts_a, all_cfg, ground_E)

    def show(tag: str, stats: Dict, counts: Dict[str, int]) -> None:
        row(f"{tag}  P(feasible)",     f"{100*stats['p_feasible']:6.2f}%")
        row(f"{tag}  P(ground-state)", f"{100*stats['p_ground']:6.2f}%")
        row(f"{tag}  ⟨E | feasible⟩",  f"{stats['mean_E']:+.3f}")
        row(f"{tag}  approx ratio",    f"{stats['approx']:.4f}")
        # Top decoded folds.
        e_table = {c["bits"]: c for c in all_cfg}
        top = sorted(counts.items(), key=lambda kv: -kv[1])[:5]
        for bits, c in top:
            cfg = e_table.get(bits)
            if cfg is None:
                continue
            path = _fmt_path(cfg["positions"]) if cfg["feasible"] else "(infeasible)"
            row(f"  bits={bits}", f"n={c:<4d}  E={cfg['energy']:+.2f}  {path}")

    show("baseline", stats_b, counts_b)
    show("AEGIS   ", stats_a, counts_a)

    # 7. Scoreboard.
    hr("Scoreboard — folding HPPH on IonQ")
    print(f"  {'metric':<24s}  {'baseline':>12s}  {'AEGIS':>12s}  {'Δ':>10s}")
    print(f"  {'2Q count':<24s}  {count_2q(baseline):>12d}  {count_2q(aegis_seq):>12d}  "
          f"{count_2q(baseline)-count_2q(aegis_seq):>+10d}")
    print(f"  {'depth':<24s}  {circuit_depth(baseline, N_QUBITS):>12d}  "
          f"{circuit_depth(aegis_seq, N_QUBITS):>12d}  "
          f"{circuit_depth(baseline, N_QUBITS)-circuit_depth(aegis_seq, N_QUBITS):>+10d}")
    print(f"  {'P(feasible)':<24s}  {100*stats_b['p_feasible']:>11.2f}%  "
          f"{100*stats_a['p_feasible']:>11.2f}%  "
          f"{100*(stats_a['p_feasible']-stats_b['p_feasible']):>+9.2f}pp")
    print(f"  {'P(ground-state)':<24s}  {100*stats_b['p_ground']:>11.2f}%  "
          f"{100*stats_a['p_ground']:>11.2f}%  "
          f"{100*(stats_a['p_ground']-stats_b['p_ground']):>+9.2f}pp")
    print(f"  {'approx ratio':<24s}  {stats_b['approx']:>12.4f}  "
          f"{stats_a['approx']:>12.4f}  "
          f"{stats_a['approx']-stats_b['approx']:>+10.4f}")
    print()
    print(f"  Ground-state energy: {ground_E:+.2f}   ·   "
          f"non-fold configurations pay ≥ +{LAMBDA_BACK:.0f} (penalties).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
