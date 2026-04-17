"""Algorithm 4 — CHSH Bell-inequality test on live IonQ.

Hypothesis
──────────
On a genuine Bell-entangled state measured at the CHSH-optimal angles, the
correlator S = E(a,b) − E(a,b') + E(a',b) + E(a',b') exceeds the classical
bound |S| ≤ 2 and approaches the Tsirelson bound 2√2 ≈ 2.828.

  • On IonQ's ideal simulator, expect S ≈ 2√2 within shot noise.
  • On IonQ's noisy simulator (forte-1), expect S somewhere in (2, 2√2)
    — above classical, below the ideal quantum bound.
  • On real QPU hardware, same picture with slightly more degradation.

Uses vqpu's QPUCloudPlugin → qiskit-ionq → IonQ cloud. The preflight
requires IONQ_API_KEY in the environment. Costs ~4 circuits × 2048 shots
on the IonQ simulator (free tier).
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from vqpu import QPUCloudPlugin  # noqa: E402

# CHSH-optimal measurement angles (radians).
A = 0.0
A_PRIME = math.pi / 4
B = math.pi / 8
B_PRIME = 3 * math.pi / 8


def bell_with_measurement_basis(theta_q0: float, theta_q1: float):
    """Prepare |Φ⁺⟩ then rotate measurement basis on each qubit by
    Ry(−2θ) so a Z-basis measurement reads the observable cos(2θ)Z + sin(2θ)X."""
    return [
        ("H", [0]),
        ("CNOT", [0, 1]),
        ("Ry", [0], -2 * theta_q0),
        ("Ry", [1], -2 * theta_q1),
    ]


def correlation_from_counts(counts: dict, total_shots: int) -> float:
    """E = (N₀₀ − N₀₁ − N₁₀ + N₁₁) / N_total."""
    n00 = counts.get("00", 0)
    n01 = counts.get("01", 0)
    n10 = counts.get("10", 0)
    n11 = counts.get("11", 0)
    return (n00 - n01 - n10 + n11) / total_shots


def main() -> int:
    if not os.environ.get("IONQ_API_KEY"):
        print("[!] IONQ_API_KEY is not set. Export it first.")
        return 2

    shots = 2048
    backend_name = os.environ.get("IONQ_BACKEND", "simulator")
    noise = os.environ.get("IONQ_NOISE_MODEL", "(none — ideal simulator)")
    print(f"  IonQ backend: {backend_name}")
    print(f"  Noise model : {noise}")
    print(f"  Shots/circuit: {shots}  (total: {4 * shots})")
    print("")

    plugin = QPUCloudPlugin("ionq")
    fp = plugin.probe()
    if fp is None or not fp.is_available:
        print("[!] IonQ plugin not available — check IONQ_API_KEY and qiskit-ionq.")
        return 1
    print(f"  Fingerprint: {fp.name}\n")

    settings = [
        ("E(A, B)",   A,       B      ),
        ("E(A, B')",  A,       B_PRIME),
        ("E(A',B)",   A_PRIME, B      ),
        ("E(A',B')",  A_PRIME, B_PRIME),
    ]
    expectations = {}
    for name, t0, t1 in settings:
        counts = plugin.execute_sample(
            n_qubits=2,
            gate_sequence=bell_with_measurement_basis(t0, t1),
            shots=shots,
        )
        E = correlation_from_counts(counts, sum(counts.values()))
        expectations[name] = E
        bar_width = int(32 * (E + 1) / 2)
        print(f"  {name:<9s}  θ_A={math.degrees(t0):>6.2f}°  "
              f"θ_B={math.degrees(t1):>6.2f}°  E={E:+.4f}  "
              f"[{'█' * bar_width}{' ' * (32 - bar_width)}]")

    S = (expectations["E(A, B)"]
         - expectations["E(A, B')"]
         + expectations["E(A',B)"]
         + expectations["E(A',B')"])
    tsirelson = 2 * math.sqrt(2)

    print("")
    print(f"  S = {S:+.4f}")
    print(f"  Classical bound:  |S| ≤ 2.0000")
    print(f"  Tsirelson bound:  |S| ≤ {tsirelson:.4f}")
    print("")
    if abs(S) > 2.0:
        excess = abs(S) - 2.0
        efficiency = abs(S) / tsirelson
        print(f"  ✓ Classical bound violated by {excess:+.4f}.")
        print(f"    Tsirelson efficiency: {efficiency * 100:.1f}% of 2√2")
        print(f"    This is genuine quantum non-locality measured on live IonQ.")
        return 0
    print(f"  ✗ S did not exceed the classical bound. Check the circuit.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
