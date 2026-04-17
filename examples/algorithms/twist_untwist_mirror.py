"""Algorithm 12 — Twist–untwist (mirror) protocol on live IonQ.

The move
────────
Standard OAT Ramsey with linear readout can't pull the full Heisenberg
F_Q out of the squeezed state because the signal lives on an axis
rotated away from whatever readout we happen to use. Davis, Bentsen &
Schleier-Smith 2016 and Hosten et al. 2016 showed that if we apply
the squeezer and then *its inverse* around the signal, the
anti-squeezing is refocused — signal becomes linearly readable.

Protocol
────────
  1. |+⟩^N  (coherent spin state prep)
  2. U_twist(+χt)           — forward squeeze
  3. Rz(φ)^⊗N               — unknown signal
  4. U_twist(−χt)           — mirror: reverses the squeeze
  5. Rx(π/2)^⊗N             — readout rotation to J_y
  6. measure Z^⊗N

Why the mirror works: without it, squeezed states have their metrology
signal mixed between J_x and J_y at an angle χt-dependent. Readout on
one axis (say J_y) captures only a projection. The mirror un-does the
squeezing's rotation, so the signal that was "hidden" by anti-squeezing
now lands cleanly on J_y with no alignment penalty — F_C → F_Q.

At χt = π/2 (cat regime), the forward OAT takes |+⟩^N to a Schrödinger
cat. The backward OAT un-makes it, BUT not back to exactly |+⟩^N —
the small Rz(φ) in the middle of the sandwich gets translated into a
linear J_y displacement proportional to N·φ. Measuring that linearly
gives the full F_Q ≈ N² signal.

Comparison to pass 2 of hardware_metrology.py
─────────────────────────────────────────────
  Pass 2 (χt=0.4, no mirror, J_y readout):  F_C ≈ 1.3–2.0 on n=4
  Expected here:                             F_C ≈ N² = 16 on n=4, if ideal
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from vqpu import QPUCloudPlugin  # noqa: E402


def _oat_layer(gates: list, n: int, chi_t: float) -> None:
    """Append exp(-i χt J_z²) as all-pairs ZZ rotations (up to global phase)."""
    theta_pair = chi_t / 2.0
    for i in range(n):
        for j in range(i + 1, n):
            gates.append(("CNOT", [i, j]))
            gates.append(("Rz", [j], 2 * theta_pair))
            gates.append(("CNOT", [i, j]))


def _tat_layer(gates: list, n: int, chi_t: float) -> None:
    """One TAT step: exp(-i χt (J_x² − J_y²)) ≈ exp(-iχt J_x²)·exp(+iχt J_y²).

    Implemented as collective-rotation-conjugated OAT blocks. The inverse
    layer is obtained by calling with −χt."""
    # +J_x² twist: Ry(π/2)·OAT(+χt)·Ry(−π/2)
    for q in range(n):
        gates.append(("Ry", [q], math.pi / 2))
    _oat_layer(gates, n, +chi_t)
    for q in range(n):
        gates.append(("Ry", [q], -math.pi / 2))
    # −J_y² twist: Rx(−π/2)·OAT(−χt)·Rx(+π/2)
    for q in range(n):
        gates.append(("Rx", [q], -math.pi / 2))
    _oat_layer(gates, n, -chi_t)
    for q in range(n):
        gates.append(("Rx", [q], math.pi / 2))


def build_mirror_sequence(
    n: int, squeezer: str, chi_t: float, phi: float, mirror: bool,
) -> list:
    """Full Ramsey protocol with optional mirror (time-reversed squeezer).

    mirror=False: this reproduces the no-mirror Ramsey from the prior
    experiment, for apples-to-apples comparison.
    mirror=True : add the inverse squeezer between the signal and readout,
    so the squeezer's rotation of the signal axis gets unwound.
    """
    gates: list = []
    # 1. Prep |+⟩^N.
    for q in range(n):
        gates.append(("H", [q]))
    # 2. Forward squeezer.
    if squeezer == "oat":
        _oat_layer(gates, n, +chi_t)
    elif squeezer == "tat":
        _tat_layer(gates, n, +chi_t)
    else:
        raise ValueError(squeezer)
    # 3. Signal phase.
    for q in range(n):
        gates.append(("Rz", [q], phi))
    # 4. Mirror = inverse squeezer.
    if mirror:
        if squeezer == "oat":
            _oat_layer(gates, n, -chi_t)
        elif squeezer == "tat":
            _tat_layer(gates, n, -chi_t)
    # 5. Readout rotation onto Z.
    for q in range(n):
        gates.append(("Rx", [q], math.pi / 2))
    return gates


def jy_mean_and_var_from_counts(counts: dict, n: int) -> tuple[float, float]:
    total = sum(counts.values())
    if total == 0:
        return 0.0, 0.0
    mean = 0.0
    mean_sq = 0.0
    for bits, hits in counts.items():
        ones = bits.count("1")
        val = 0.5 * ((n - ones) - ones)
        p = hits / total
        mean += p * val
        mean_sq += p * val * val
    return mean, mean_sq - mean * mean


def run_point(
    plugin: QPUCloudPlugin,
    n: int, squeezer: str, chi_t: float, phi: float,
    mirror: bool, shots: int,
) -> dict:
    gates = build_mirror_sequence(n, squeezer, chi_t, phi, mirror)
    counts = plugin.execute_sample(
        n_qubits=n, gate_sequence=gates, shots=shots,
    )
    mean, var = jy_mean_and_var_from_counts(counts, n)
    return {"mean": mean, "var": var, "shots": sum(counts.values())}


def measure_fc(
    plugin: QPUCloudPlugin,
    n: int, squeezer: str, chi_t: float, mirror: bool,
    delta: float, shots: int,
) -> dict:
    print(f"    ↳ {squeezer.upper()} mirror={mirror}  φ=+δ")
    plus = run_point(plugin, n, squeezer, chi_t, +delta, mirror, shots)
    print(f"    ↳ {squeezer.upper()} mirror={mirror}  φ= 0")
    zero = run_point(plugin, n, squeezer, chi_t, 0.0, mirror, shots)
    print(f"    ↳ {squeezer.upper()} mirror={mirror}  φ=−δ")
    minus = run_point(plugin, n, squeezer, chi_t, -delta, mirror, shots)
    slope = (plus["mean"] - minus["mean"]) / (2 * delta)
    var0 = zero["var"]
    f_c = (slope * slope) / var0 if var0 > 0 else float("inf")
    return {
        "slope": slope, "var0": var0, "F_C": f_c,
        "F_C_per_N": f_c / n, "F_C_per_N2": f_c / (n * n),
        "mean_minus": minus["mean"], "mean_zero": zero["mean"],
        "mean_plus": plus["mean"],
    }


def run_suite(
    n: int, chi_t: float, delta: float, shots: int, noise_model: str | None,
) -> dict:
    if noise_model:
        os.environ["IONQ_NOISE_MODEL"] = noise_model
    else:
        os.environ.pop("IONQ_NOISE_MODEL", None)
    plugin = QPUCloudPlugin("ionq")
    fp = plugin.probe()
    if fp is None or not fp.is_available:
        print("[!] IonQ plugin unavailable.")
        sys.exit(1)
    out = {"noise": noise_model or "ideal"}
    for squeezer in ("oat", "tat"):
        for mirror in (False, True):
            key = f"{squeezer}_{'mirror' if mirror else 'plain'}"
            print(f"\n  [{noise_model or 'ideal'}] {key}")
            out[key] = measure_fc(
                plugin, n, squeezer, chi_t, mirror, delta, shots,
            )
    return out


def main() -> int:
    if not os.environ.get("IONQ_API_KEY"):
        print("[!] IONQ_API_KEY not set.")
        return 2

    n = 4
    chi_t = math.pi / 2           # cat-formation point — where mirror shines.
    delta = 0.15
    shots = 1024
    print(f"  n={n}  χt={chi_t:.3f} (π/2)  δ={delta}  shots={shots}")
    print(f"  Theoretical ceiling:  F_C ≤ F_Q = N² = {n*n}")
    print("─" * 82)
    print("  Pass A — ideal IonQ simulator")
    print("─" * 82)
    ideal = run_suite(n, chi_t, delta, shots, None)
    print("\n" + "─" * 82)
    print("  Pass B — forte-1 noisy IonQ simulator")
    print("─" * 82)
    noisy = run_suite(n, chi_t, delta, shots, "forte-1")

    print("\n" + "═" * 82)
    print("  Results: F_C at cat regime χt=π/2 with J_y readout")
    print("═" * 82)
    print(f"  {'config':<18s}  {'F_C':>8s}  {'F_C/N':>7s}  {'F_C/N²':>8s}  "
          f"{'slope':>8s}  {'Var@0':>8s}")
    print("  " + "─" * 76)
    for noise_key, block in (("ideal", ideal), ("forte-1", noisy)):
        for tag in ("oat_plain", "oat_mirror", "tat_plain", "tat_mirror"):
            d = block[tag]
            print(f"  {noise_key}/{tag:<11s}  "
                  f"{d['F_C']:>8.3f}  "
                  f"{d['F_C_per_N']:>6.2f}×  "
                  f"{d['F_C_per_N2']:>7.3f}×  "
                  f"{d['slope']:>+8.3f}  "
                  f"{d['var0']:>8.3f}")

    print("\n  Key comparisons:")
    for noise_key, block in (("ideal", ideal), ("forte-1", noisy)):
        for sq in ("oat", "tat"):
            plain = block[f"{sq}_plain"]["F_C"]
            mirror = block[f"{sq}_mirror"]["F_C"]
            gain = mirror / max(plain, 1e-9)
            print(f"    {noise_key}/{sq.upper()} mirror vs plain:  "
                  f"{plain:.3f} → {mirror:.3f}  ({gain:.2f}× gain)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
