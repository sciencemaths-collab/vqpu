"""`python -m vqpu` — auto-discovery + demo on the fastest backend available."""

import numpy as np

from .universal import UniversalvQPU


def main() -> None:
    print("╔" + "═" * 60 + "╗")
    print("║  vQPU — Universal Virtual Quantum Processing Unit          ║")
    print("║  Probing host for every available compute backend…         ║")
    print("╚" + "═" * 60 + "╝")

    qpu = UniversalvQPU(verbose=True)

    online = [fp for fp in qpu.backends.values() if fp.is_available]
    if not online:
        print("\n[!] No backends came online. Install a runtime or check permissions.")
        return

    # ── Demo 1: Bell pair (tiny smoke test) ───────────────────────────
    print("\n" + "─" * 62)
    print("  Demo 1 — Bell pair (2 qubits)")
    print("─" * 62)
    c = qpu.circuit(2, "bell")
    c.h(0).cnot(0, 1)
    r = c and qpu.run(c, shots=2048)
    print(f"  counts: {r.counts}")
    print(f"  backend: {r.backend_name}   time: {r.execution_time*1000:.1f}ms")

    # ── Demo 2: GHZ-10 (verifies entanglement across many qubits) ─────
    print("\n" + "─" * 62)
    print("  Demo 2 — GHZ state on 10 qubits")
    print("─" * 62)
    c2 = qpu.circuit(10, "ghz_10")
    c2.h(0)
    for i in range(1, 10):
        c2.cnot(0, i)
    r2 = qpu.run(c2, shots=2048)
    p_all_zero = r2.counts.get("0" * 10, 0) / 2048
    p_all_one = r2.counts.get("1" * 10, 0) / 2048
    other = 1.0 - p_all_zero - p_all_one
    print(f"  P(|0⟩^10) = {p_all_zero:.3f}")
    print(f"  P(|1⟩^10) = {p_all_one:.3f}")
    print(f"  P(other)  = {other:.3f}   (should be ≈0 for a clean GHZ)")

    # ── Demo 3: 20-qubit mixed-phase circuit + plan view ──────────────
    print("\n" + "─" * 62)
    print("  Demo 3 — 20-qubit mixed circuit: plan only")
    print("─" * 62)
    c3 = qpu.circuit(20, "showcase_20qb")
    for i in range(20):
        c3.h(i)
    for i in range(19):
        c3.cnot(i, i + 1)
    for i in range(20):
        c3.ry(i, np.pi / 4)
    plan = qpu.plan(c3)
    for phase in plan["phases"]:
        print(f"    {phase['phase']:12s} → {phase['assigned_to']:28s} "
              f"~{phase['est_time_ms']:.2f}ms  ({phase['description']})")
    print(f"\n  hybrid: {plan['is_hybrid']}   "
          f"total est.: {plan['total_est_time_ms']:.1f}ms")

    # ── Backend registry summary ──────────────────────────────────────
    print("\n" + "─" * 62)
    print("  Backend registry")
    print("─" * 62)
    for name, fp in qpu.backends.items():
        status = "ONLINE " if fp.is_available else "offline"
        qb = f"{fp.max_qubits}qb" if fp.max_qubits > 0 else "classical"
        print(f"  [{fp.compute_class.value:5s}] {name:38s} {status}  {qb}")

    print("\nDone. To use in your own code:")
    print("    from vqpu import UniversalvQPU")
    print("    qpu = UniversalvQPU()")
    print("    c = qpu.circuit(n, 'name').h(0).cnot(0, 1)")
    print("    result = qpu.run(c, shots=1024)")


if __name__ == "__main__":
    main()
