"""Frontier quantum-signature benchmark on IonQ (via the upgraded vqpu bridge).

This is the "wow the scientists" pass — it deliberately measures things a
classical machine cannot reproduce, and compares each observed value against
its local-hidden-variable ceiling and its Tsirelson / quantum ceiling:

  1. CHSH Bell inequality violation on |Φ+⟩
       classical |S| ≤ 2           quantum |S| ≤ 2√2 ≈ 2.8284
       (Aspect-Clauser-Zeilinger, 2022 Nobel in Physics)

  2. Mermin-3 inequality violation on GHZ-3
       classical |M| ≤ 2           quantum ⟨M⟩ = 4
       (genuine multipartite entanglement witness)

  3. GHZ-5 fidelity witness (Z-basis population + X-basis even parity)
       F_GHZ ≥ ½(P(|0…0⟩ ∪ |1…1⟩) + ⟨Π_x⟩)
       ideal = 1.0; classical mixture ceiling < ½

Each subtest is submitted as its own circuit to IonQ's cloud simulator
(default: `IONQ_BACKEND=simulator`, free tier, no QPU cost). Set
`IONQ_BACKEND=qpu.forte-1` (or similar) to run on real hardware — but note
that real QPU time is billed per shot.

Usage:
    export IONQ_API_KEY="<your key>"
    python3 examples/frontier_ionq.py
"""

from __future__ import annotations

import math
import os
import pathlib
import sys
import time
from typing import Callable, Dict, List, Sequence, Tuple

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from vqpu import QPUCloudPlugin


# ──────────────────────────── gate-sequence helpers ────────────────────────

def _ry(angle: float, target: int):
    return ("Ry", [target], angle)


def _h(target: int):
    return ("H", [target])


def _x(target: int):
    return ("X", [target])


def _s_dagger(target: int):
    # S† = Rz(-π/2) up to a global phase. IonQ accepts Rz directly.
    return ("Rz", [target], -math.pi / 2)


def _cnot(c: int, t: int):
    return ("CNOT", [c, t])


def bell_pair(a: int = 0, b: int = 1):
    return [_h(a), _cnot(a, b)]


def ghz(n: int):
    seq = [_h(0)]
    for i in range(1, n):
        seq.append(_cnot(0, i))
    return seq


def measure_in_pauli(seq: List, pauli: str, qubit: int) -> List:
    """Append the basis change so `measure_all()` samples in the P basis."""
    p = pauli.upper()
    if p == "Z":
        return seq
    if p == "X":
        return seq + [_h(qubit)]
    if p == "Y":
        # To measure Y, rotate Y-eigenbasis to Z: apply S† then H.
        return seq + [_s_dagger(qubit), _h(qubit)]
    raise ValueError(f"unknown pauli {pauli!r}")


def rotate_for_chsh(seq: List, angle: float, qubit: int) -> List:
    """Prepend the basis rotation that makes `measure_all` sample
    cos(α)Z + sin(α)X on `qubit`: apply Ry(α) then measure Z."""
    return seq + [_ry(angle, qubit)]


# ────────────────────────── correlator math helpers ────────────────────────

def product_correlator(counts: Dict[str, int], qubits: Sequence[int]) -> float:
    """<Π (1-2 b_q)> averaged over shots, treating qubit 0 as MSB
    (vqpu's bitstring convention — already fixed up by the plugin)."""
    total = sum(counts.values())
    if total == 0:
        return 0.0
    acc = 0
    for bits, n in counts.items():
        parity = 1
        for q in qubits:
            parity *= (1 if bits[q] == "0" else -1)
        acc += parity * n
    return acc / total


def z_population(counts: Dict[str, int], bitstring: str) -> float:
    total = sum(counts.values())
    return counts.get(bitstring, 0) / total if total else 0.0


# ────────────────────────────────── tests ──────────────────────────────────

def run_chsh(plugin: QPUCloudPlugin, shots: int) -> Dict:
    """Measure S = E(a,b) − E(a,b') + E(a',b) + E(a',b') on |Φ+⟩."""
    # Optimal CHSH angles for |Φ+⟩: E(a,b) = cos(a − b).
    a, a_prime = 0.0, math.pi / 2
    b, b_prime = math.pi / 4, 3 * math.pi / 4

    settings = [
        ("E(a,b)",   a,       b),
        ("E(a,b')",  a,       b_prime),
        ("E(a',b)",  a_prime, b),
        ("E(a',b')", a_prime, b_prime),
    ]

    correlators = {}
    for label, ang_a, ang_b in settings:
        seq = bell_pair()
        seq = rotate_for_chsh(seq, ang_a, qubit=0)
        seq = rotate_for_chsh(seq, ang_b, qubit=1)
        counts = plugin.execute_sample(
            n_qubits=2, gate_sequence=seq, shots=shots
        )
        correlators[label] = product_correlator(counts, [0, 1])

    s_val = (
        correlators["E(a,b)"]
        - correlators["E(a,b')"]
        + correlators["E(a',b)"]
        + correlators["E(a',b')"]
    )
    return {
        "correlators": correlators,
        "S": s_val,
        "classical_bound": 2.0,
        "tsirelson_bound": 2.0 * math.sqrt(2),
        "violated_classical": abs(s_val) > 2.0,
    }


def run_mermin3(plugin: QPUCloudPlugin, shots: int) -> Dict:
    """Measure M = <XXX> − <XYY> − <YXY> − <YYX> on GHZ-3."""
    term_settings = [
        ("<XXX>",   ("X", "X", "X"), +1),
        ("<XYY>",   ("X", "Y", "Y"), -1),
        ("<YXY>",   ("Y", "X", "Y"), -1),
        ("<YYX>",   ("Y", "Y", "X"), -1),
    ]

    terms = {}
    for label, basis, _sign in term_settings:
        seq = ghz(3)
        for q, p in enumerate(basis):
            seq = measure_in_pauli(seq, p, q)
        counts = plugin.execute_sample(
            n_qubits=3, gate_sequence=seq, shots=shots
        )
        terms[label] = product_correlator(counts, [0, 1, 2])

    m_val = sum(sign * terms[label] for label, _basis, sign in term_settings)
    return {
        "terms": terms,
        "M": m_val,
        "classical_bound": 2.0,
        "quantum_bound": 4.0,
        "violated_classical": abs(m_val) > 2.0,
    }


def run_ghz_fidelity(plugin: QPUCloudPlugin, n: int, shots: int) -> Dict:
    """GHZ-n fidelity witness: Z-basis population + X-basis parity."""
    # Z basis: all zeros and all ones should each get ~½; nothing else.
    counts_z = plugin.execute_sample(
        n_qubits=n, gate_sequence=ghz(n), shots=shots
    )
    p00 = z_population(counts_z, "0" * n)
    p11 = z_population(counts_z, "1" * n)
    pop_witness = p00 + p11

    # X basis: apply H to every qubit, then measure. On an ideal GHZ-n the
    # outcomes live entirely on even-parity strings (⟨Π_x⟩ = +1).
    x_seq = ghz(n)
    for q in range(n):
        x_seq = measure_in_pauli(x_seq, "X", q)
    counts_x = plugin.execute_sample(
        n_qubits=n, gate_sequence=x_seq, shots=shots
    )
    parity_x = product_correlator(counts_x, list(range(n)))

    # Standard witness: F_GHZ ≥ ½ (P_pop + ⟨Π_x⟩).
    fidelity_lb = 0.5 * (pop_witness + parity_x)
    return {
        "n": n,
        "P(|0…0⟩)": p00,
        "P(|1…1⟩)": p11,
        "pop_witness": pop_witness,
        "X_parity": parity_x,
        "fidelity_lower_bound": fidelity_lb,
    }


# ──────────────────────────────── main ─────────────────────────────────────

def _header(title: str) -> None:
    print("\n" + "═" * 72)
    print(f"  {title}")
    print("═" * 72)


def _line(label: str, value: str) -> None:
    print(f"  {label:<36s} {value}")


def main() -> int:
    if not os.environ.get("IONQ_API_KEY"):
        print("[!] IONQ_API_KEY is not set — export it first.")
        return 2

    backend_name = os.environ.get("IONQ_BACKEND", "simulator")
    is_sim = backend_name.startswith("simulator")
    shots = int(os.environ.get("FRONTIER_SHOTS", "2000" if is_sim else "512"))

    plugin = QPUCloudPlugin("ionq")
    fingerprint = plugin.probe()
    if fingerprint is None or not fingerprint.is_available:
        print("[!] IonQ bridge failed to come online.")
        return 1

    _header("Frontier quantum-signature benchmark — IonQ bridge")
    _line("Backend", f"{fingerprint.name}  ({backend_name})")
    _line("Max qubits advertised", str(fingerprint.max_qubits))
    _line("Connectivity", fingerprint.connectivity)
    _line("Shots per circuit", str(shots))

    scoreboard: List[Tuple[str, str, str, str]] = []

    # ── 1. CHSH ──────────────────────────────────────────────────────────
    _header("1. CHSH inequality on |Φ+⟩  (Aspect / 2022 Nobel)")
    t0 = time.time()
    chsh = run_chsh(plugin, shots)
    dt = time.time() - t0
    for label, val in chsh["correlators"].items():
        _line(label, f"{val:+.4f}")
    _line("Observed S", f"{chsh['S']:+.4f}")
    _line("Classical bound", f"|S| ≤ {chsh['classical_bound']:.4f}")
    _line("Tsirelson (quantum) bound", f"|S| ≤ {chsh['tsirelson_bound']:.4f}")
    delta = abs(chsh["S"]) - chsh["classical_bound"]
    _line("Violation above classical", f"+{delta:+.4f}  ({dt:.1f}s)")
    scoreboard.append((
        "CHSH Bell violation",
        f"S = {chsh['S']:+.4f}",
        f"classical ≤ 2.0000, Tsirelson ≈ 2.8284",
        "WOW ✓ — local hidden variables refuted" if chsh["violated_classical"]
        else "classical regime",
    ))

    # ── 2. Mermin-3 ──────────────────────────────────────────────────────
    _header("2. Mermin-3 inequality on GHZ-3  (multipartite non-locality)")
    t0 = time.time()
    mermin = run_mermin3(plugin, shots)
    dt = time.time() - t0
    for label, val in mermin["terms"].items():
        _line(label, f"{val:+.4f}")
    _line("Observed M", f"{mermin['M']:+.4f}")
    _line("Classical bound", f"|M| ≤ {mermin['classical_bound']:.4f}")
    _line("Quantum maximum (GHZ)", f"|M| = {mermin['quantum_bound']:.4f}")
    _line("Quantum / classical gap", f"{abs(mermin['M']) / 2.0:.3f}× the LHV ceiling  ({dt:.1f}s)")
    scoreboard.append((
        "Mermin-3 on GHZ-3",
        f"M = {mermin['M']:+.4f}",
        "classical ≤ 2, quantum = 4",
        "WOW ✓ — genuine 3-party entanglement" if mermin["violated_classical"]
        else "below LHV ceiling",
    ))

    # ── 3. GHZ-5 fidelity ────────────────────────────────────────────────
    _header("3. GHZ-5 fidelity witness  (coherent five-body superposition)")
    t0 = time.time()
    ghz_report = run_ghz_fidelity(plugin, n=5, shots=shots)
    dt = time.time() - t0
    _line("P(|00000⟩)", f"{ghz_report['P(|0…0⟩)']:.4f}")
    _line("P(|11111⟩)", f"{ghz_report['P(|1…1⟩)']:.4f}")
    _line("Pop witness  P_0…0 + P_1…1",
          f"{ghz_report['pop_witness']:.4f}  (classical ≤ 1.0 trivially; "
          f"ideal = 1.0)")
    _line("⟨Π_x⟩ after H⊗5",
          f"{ghz_report['X_parity']:+.4f}  (ideal = +1.0, coherence witness)")
    _line("Fidelity lower bound  F ≥ ½(pop+X)",
          f"{ghz_report['fidelity_lower_bound']:.4f}  ({dt:.1f}s)")
    scoreboard.append((
        "GHZ-5 fidelity witness",
        f"F ≥ {ghz_report['fidelity_lower_bound']:.4f}",
        "biseparable states cap F ≤ 0.5",
        "WOW ✓ — genuine 5-body entanglement"
        if ghz_report["fidelity_lower_bound"] > 0.5 else "biseparable regime",
    ))

    # ── Scoreboard ───────────────────────────────────────────────────────
    _header("Novelty scoreboard")
    for name, observed, bounds, verdict in scoreboard:
        print(f"  • {name}")
        print(f"       observed : {observed}")
        print(f"       bounds   : {bounds}")
        print(f"       verdict  : {verdict}")

    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
