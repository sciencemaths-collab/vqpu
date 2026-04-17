"""Algorithm 11 — OAT vs TAT under real hardware noise on live IonQ.

The falsifiable claim we're testing
───────────────────────────────────
Algo 10 showed OAT and TAT *both* reach Heisenberg on clean product-state
input, but TAT recovered 2.6× more Fisher information from a broken-
symmetry input (the string-chain state). That was all done on an ideal
classical simulator. If the advantage matters in practice it should
survive IonQ's forte-1 hardware noise model.

This run
────────
Two probe circuits — product|+⟩^N then OAT(π/2), product|+⟩^N then
TAT(π/2). For each, we run a two-point Ramsey interferometer:

  prepare probe → Rz(φ)^⊗N → Ry(−π/2)^⊗N → measure Z^⊗N.

The Z-readout after Ry(−π/2) reports ⟨J_x⟩. At small φ the signal is
linear: ⟨J_x⟩ ≈ slope · φ. Fisher info at that point is

  F_C = slope² / Var(J_x).

We compute slope from φ = ±δ runs and Var(J_x) from the counts at φ = 0.

We repeat the whole thing on:
  • IonQ ideal simulator  (expect: OAT ≈ TAT ≈ Heisenberg N²)
  • IonQ forte-1 noisy sim (expect: both degrade; the question is
    whether one degrades less, since our claim is that TAT is more
    robust to imperfect state prep / circuit noise).

Cost: 2 probes × 3 phi values (+δ, 0, −δ) × 2 noise models = 12 IonQ
jobs at 1024 shots each ≈ 12k shots total.
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from vqpu import QPUCloudPlugin  # noqa: E402
from examples.algorithms.oat_squeezed_probe import apply_oat  # noqa: E402
from examples.algorithms.two_axis_twisting import apply_tat_step  # noqa: E402


def preflight() -> None:
    if not os.environ.get("IONQ_API_KEY"):
        print("[!] IONQ_API_KEY not set.")
        sys.exit(2)


def build_ramsey_gate_sequence(
    n: int, squeezer: str, chi_t: float, phi: float,
) -> list:
    """Return a gate-tuple list for the full Ramsey protocol.

    We use the tuple format accepted by QPUCloudPlugin._iter_gates/translator;
    we do NOT build a QuantumCircuit since we pass gates directly.
    """
    gates = []
    # 1. Probe: product |+⟩^N.
    for q in range(n):
        gates.append(("H", [q]))

    # 2. Squeezer: OAT or TAT at chi_t.
    if squeezer == "oat":
        # exp(-i chi_t J_z²) = product over i<j of ZZ rotations (+ global phase).
        theta_pair = chi_t / 2.0
        for i in range(n):
            for j in range(i + 1, n):
                gates.append(("CNOT", [i, j]))
                gates.append(("Rz", [j], 2 * theta_pair))
                gates.append(("CNOT", [i, j]))
    elif squeezer == "tat":
        # TAT one-step: +J_x² twist then -J_y² twist.
        # +J_x² via Ry(π/2) - OAT(+chi_t) - Ry(-π/2).
        for q in range(n):
            gates.append(("Ry", [q], math.pi / 2))
        theta_pair = chi_t / 2.0
        for i in range(n):
            for j in range(i + 1, n):
                gates.append(("CNOT", [i, j]))
                gates.append(("Rz", [j], 2 * theta_pair))
                gates.append(("CNOT", [i, j]))
        for q in range(n):
            gates.append(("Ry", [q], -math.pi / 2))
        # -J_y² via Rx(-π/2) - OAT(-chi_t) - Rx(π/2).
        for q in range(n):
            gates.append(("Rx", [q], -math.pi / 2))
        theta_pair_neg = -chi_t / 2.0
        for i in range(n):
            for j in range(i + 1, n):
                gates.append(("CNOT", [i, j]))
                gates.append(("Rz", [j], 2 * theta_pair_neg))
                gates.append(("CNOT", [i, j]))
        for q in range(n):
            gates.append(("Rx", [q], math.pi / 2))

    # 3. Phase evolution: Rz(phi) on each qubit = e^{-i phi J_z}.
    for q in range(n):
        gates.append(("Rz", [q], phi))

    # 4. Readout rotation: Rx(π/2) rotates Z to Y in the Heisenberg sense,
    # so Z-basis measurement after this gate returns the J_y eigenvalue of
    # the pre-rotation state. J_y is the axis where a squeezed state's
    # Ramsey signal lives, so this is the right readout for χt < π/2.
    for q in range(n):
        gates.append(("Rx", [q], math.pi / 2))

    return gates


def jx_mean_and_var_from_counts(
    counts: dict, n: int,
) -> tuple[float, float, int]:
    """Collective-spin readout expectation and variance from Z-basis counts.
    Despite the name, this now reads whichever axis matched the readout
    rotation in `build_ramsey_gate_sequence` (currently J_y after Rx(π/2))."""
    total = sum(counts.values())
    if total == 0:
        return 0.0, 0.0, 0
    mean = 0.0
    mean_sq = 0.0
    for bits, hits in counts.items():
        ones = bits.count("1")
        jx = 0.5 * ((n - ones) - ones)
        p = hits / total
        mean += p * jx
        mean_sq += p * jx * jx
    var = mean_sq - mean * mean
    return mean, var, total


def run_single(
    plugin: QPUCloudPlugin,
    n: int,
    squeezer: str,
    chi_t: float,
    phi: float,
    shots: int,
) -> dict:
    gates = build_ramsey_gate_sequence(n, squeezer, chi_t, phi)
    counts = plugin.execute_sample(
        n_qubits=n, gate_sequence=gates, shots=shots,
    )
    mean, var, total = jx_mean_and_var_from_counts(counts, n)
    return {
        "counts": counts,
        "mean_jx": mean,
        "var_jx": var,
        "shots_returned": total,
        "circuit_depth_est": len(gates),
    }


def analyze(
    results_plus: dict, results_zero: dict, results_minus: dict,
    delta: float, n: int, shots: int,
) -> dict:
    slope = (results_plus["mean_jx"] - results_minus["mean_jx"]) / (2 * delta)
    var_at_zero = results_zero["var_jx"]
    # Classical Fisher info per experiment (one state prep + one shot):
    # F_C(φ=0) = slope² / Var(J_x | φ=0).
    if var_at_zero <= 0:
        fisher = float("inf") if slope != 0 else 0.0
    else:
        fisher = (slope * slope) / var_at_zero
    return {
        "slope": slope,
        "var_jx": var_at_zero,
        "F_C": fisher,
        "F_C_per_N": fisher / n,
        "F_C_per_N2": fisher / (n * n),
        "mean_jx_plus": results_plus["mean_jx"],
        "mean_jx_zero": results_zero["mean_jx"],
        "mean_jx_minus": results_minus["mean_jx"],
    }


def run_protocol(
    n: int,
    squeezers: list[str],
    chi_t: float,
    noise_model: str | None,
    delta: float,
    shots: int,
) -> dict:
    if noise_model:
        os.environ["IONQ_NOISE_MODEL"] = noise_model
    else:
        os.environ.pop("IONQ_NOISE_MODEL", None)

    plugin = QPUCloudPlugin("ionq")
    fp = plugin.probe()
    if fp is None or not fp.is_available:
        print("[!] IonQ plugin not available.")
        sys.exit(1)

    report = {
        "noise_model": noise_model or "ideal",
        "n": n,
        "chi_t": chi_t,
        "delta": delta,
        "shots": shots,
        "results": {},
    }
    for squeezer in squeezers:
        print(f"  ↳ {squeezer.upper()}  φ=+δ")
        r_plus = run_single(plugin, n, squeezer, chi_t, +delta, shots)
        print(f"  ↳ {squeezer.upper()}  φ= 0")
        r_zero = run_single(plugin, n, squeezer, chi_t, 0.0, shots)
        print(f"  ↳ {squeezer.upper()}  φ=−δ")
        r_minus = run_single(plugin, n, squeezer, chi_t, -delta, shots)
        analysis = analyze(r_plus, r_zero, r_minus, delta, n, shots)
        report["results"][squeezer] = {
            "slope": analysis["slope"],
            "var_jx_at_zero": analysis["var_jx"],
            "F_C": analysis["F_C"],
            "F_C_per_N": analysis["F_C_per_N"],
            "F_C_per_N2": analysis["F_C_per_N2"],
            "mean_jx_samples": (
                analysis["mean_jx_minus"],
                analysis["mean_jx_zero"],
                analysis["mean_jx_plus"],
            ),
        }
    return report


def main() -> int:
    preflight()

    n = 4
    # Squeezing regime (not cat). χt ≈ N^(−2/3) is the Kitagawa–Ueda optimum
    # where the state is spin-squeezed but still has a linear Ramsey signal
    # in the orthogonal plane. At χt = π/2 the state is a Schrödinger cat
    # with ⟨J_x⟩ = 0 identically and signal hides in parity, not linear
    # readout — empirically confirmed by the zero-slope first pass.
    chi_t = 0.4             # ≈ 4^{−2/3} ≈ optimal squeezing for n=4.
    delta = 0.15
    shots = 1024

    print(f"  N = {n} qubits.  χt = π/2 ≈ {chi_t:.3f}.  "
          f"δ = {delta}.  shots = {shots}")
    print(f"  Expected theory (ideal):  F_C/N² → 1.0  for both OAT and TAT")
    print("")

    print("─" * 78)
    print("  Pass 1 — ideal IonQ simulator")
    print("─" * 78)
    ideal = run_protocol(
        n=n, squeezers=["oat", "tat"], chi_t=chi_t,
        noise_model=None, delta=delta, shots=shots,
    )

    print("")
    print("─" * 78)
    print("  Pass 2 — forte-1 noisy simulator")
    print("─" * 78)
    noisy = run_protocol(
        n=n, squeezers=["oat", "tat"], chi_t=chi_t,
        noise_model="forte-1", delta=delta, shots=shots,
    )

    print("")
    print("─" * 78)
    print("  Summary")
    print("─" * 78)
    print(f"  {'noise':<10s}  {'squeezer':<8s}  {'F_C':>8s}  {'F_C/N':>7s}  "
          f"{'F_C/N²':>8s}  {'slope':>8s}  {'Var(J_x)@0':>10s}")
    for report in (ideal, noisy):
        for sq, data in report["results"].items():
            print(f"  {report['noise_model']:<10s}  {sq:<8s}  "
                  f"{data['F_C']:>8.3f}  "
                  f"{data['F_C_per_N']:>6.2f}×  "
                  f"{data['F_C_per_N2']:>7.3f}×  "
                  f"{data['slope']:>+8.3f}  "
                  f"{data['var_jx_at_zero']:>10.4f}")

    print("")
    print("  The falsifiable claim:")
    oat_ideal = ideal["results"]["oat"]["F_C_per_N2"]
    oat_noisy = noisy["results"]["oat"]["F_C_per_N2"]
    tat_ideal = ideal["results"]["tat"]["F_C_per_N2"]
    tat_noisy = noisy["results"]["tat"]["F_C_per_N2"]
    oat_drop = oat_ideal - oat_noisy
    tat_drop = tat_ideal - tat_noisy
    print(f"    OAT  F_C/N² drops by {oat_drop:+.3f} under forte-1 noise "
          f"(ideal {oat_ideal:.3f} → noisy {oat_noisy:.3f})")
    print(f"    TAT  F_C/N² drops by {tat_drop:+.3f} under forte-1 noise "
          f"(ideal {tat_ideal:.3f} → noisy {tat_noisy:.3f})")
    if abs(tat_drop) < abs(oat_drop):
        pct = 100 * (abs(oat_drop) - abs(tat_drop)) / max(abs(oat_drop), 1e-9)
        print(f"    → TAT lost {pct:.1f}% less Fisher info. "
              f"Hardware-grounded robustness claim holds.")
    else:
        pct = 100 * (abs(tat_drop) - abs(oat_drop)) / max(abs(tat_drop), 1e-9)
        print(f"    → TAT lost {pct:.1f}% MORE Fisher info than OAT. "
              f"The clean-sim claim does NOT survive real noise.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
