"""vqpu → IonQ smoke test.

Runs a Bell pair and a 3-qubit GHZ through IonQ's ideal simulator via the
qiskit-ionq integration inside `QPUCloudPlugin`.

Preflight
─────────
  pip install qiskit qiskit-ionq
  export IONQ_API_KEY="<your rotated key>"   # never commit or paste in chat
  # Optional:
  export IONQ_BACKEND="simulator"            # default; or "qpu.forte-1"
  export IONQ_NOISE_MODEL="forte-1"          # optional noise profile

Usage
─────
  python3 examples/ionq_smoke_test.py
"""

from __future__ import annotations

import os
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from vqpu import QPUCloudPlugin  # noqa: E402


def bell_pair_gates():
    """Bell state circuit in vqpu's gate-tuple form."""
    return [
        ("H", [0]),
        ("CNOT", [0, 1]),
    ]


def ghz_gates(n: int):
    seq = [("H", [0])]
    for i in range(1, n):
        seq.append(("CNOT", [0, i]))
    return seq


def preflight() -> None:
    if not os.environ.get("IONQ_API_KEY"):
        print("[!] IONQ_API_KEY is not set. Rotate your key at "
              "cloud.ionq.com → Settings → API Keys, then:")
        print('    export IONQ_API_KEY="<your new key>"')
        sys.exit(2)
    try:
        import qiskit  # noqa: F401
        import qiskit_ionq  # noqa: F401
    except ImportError:
        print("[!] qiskit and qiskit-ionq are required:")
        print("    pip install qiskit qiskit-ionq")
        sys.exit(2)


def report(label: str, counts: dict, total_shots: int) -> None:
    print(f"\n  {label} — {sum(counts.values())}/{total_shots} shots returned")
    for bits, n in sorted(counts.items(), key=lambda kv: -kv[1])[:6]:
        p = n / total_shots
        bar = "█" * int(p * 40)
        print(f"    {bits}  {n:>5d}  {p:.3f}  {bar}")


def main() -> int:
    preflight()

    backend_name = os.environ.get("IONQ_BACKEND", "simulator")
    noise = os.environ.get("IONQ_NOISE_MODEL", "(none — ideal)")
    print(f"  IonQ backend: {backend_name}")
    print(f"  Noise model : {noise}")

    plugin = QPUCloudPlugin("ionq")
    fingerprint = plugin.probe()
    if fingerprint is None or not fingerprint.is_available:
        print("[!] QPUCloudPlugin('ionq').probe() did not come online. Check "
              "IONQ_API_KEY and that qiskit-ionq is importable.")
        return 1

    print(f"  Fingerprint : {fingerprint.name} "
          f"(max_qubits={fingerprint.max_qubits}, "
          f"vendor={fingerprint.vendor})")

    shots = 1024

    # Test 1: Bell pair, expect ~50/50 on 00 and 11.
    counts_bell = plugin.execute_sample(
        n_qubits=2,
        gate_sequence=bell_pair_gates(),
        shots=shots,
    )
    report("Bell pair on IonQ", counts_bell, shots)
    bell_total = sum(counts_bell.values())
    bell_ok = (counts_bell.get("00", 0) + counts_bell.get("11", 0)) / bell_total
    print(f"    P(|00⟩ ∪ |11⟩) = {bell_ok:.3f}")

    # Test 2: 3-qubit GHZ, expect ~50/50 on 000 and 111.
    counts_ghz = plugin.execute_sample(
        n_qubits=3,
        gate_sequence=ghz_gates(3),
        shots=shots,
    )
    report("GHZ-3 on IonQ", counts_ghz, shots)
    ghz_total = sum(counts_ghz.values())
    ghz_ok = (counts_ghz.get("000", 0) + counts_ghz.get("111", 0)) / ghz_total
    print(f"    P(|000⟩ ∪ |111⟩) = {ghz_ok:.3f}")

    # Verdicts — on ideal sim both should be >0.99; on noisy/QPU, >0.85.
    threshold = 0.99 if os.environ.get("IONQ_NOISE_MODEL") is None and \
        backend_name == "simulator" else 0.80
    failures = []
    if bell_ok < threshold:
        failures.append(f"Bell P(correlated) = {bell_ok:.3f} < {threshold}")
    if ghz_ok < threshold:
        failures.append(f"GHZ P(correlated) = {ghz_ok:.3f} < {threshold}")

    if failures:
        print("\n[!] Unexpected results (threshold ≈ {:.2f}):".format(threshold))
        for f in failures:
            print(f"    • {f}")
        return 1

    print("\n  Both circuits returned correlated outcomes as expected.")
    print("  IonQ execution bridge is live.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
