"""Cryo-Canonical Basin Weaving on IonQ — live hardware/simulator test.

Runs the CCBW optimizer on a 4-qubit Max-Cut QAOA problem through
IonQ's cloud backend. Works with both the IonQ simulator and real
QPU hardware (Aria-1, Forte-1).

Setup
─────
    export IONQ_API_KEY="your-api-key-here"

    # IonQ simulator (free, fast):
    export IONQ_BACKEND="simulator"

    # IonQ simulator with Aria-1 noise model:
    export IONQ_BACKEND="simulator"
    export IONQ_NOISE_MODEL="aria-1"

    # Real IonQ Aria-1 hardware ($0.01/shot, queued):
    export IONQ_BACKEND="qpu.aria-1"

    python examples/cryo_ionq.py
"""

import os
import sys
import time
import numpy as np

from vqpu import (
    QPUCloudPlugin,
    CryoConfig,
    CryoOptimizer,
    QuantumCircuit,
    ExecutionResult,
)
from vqpu.link import LinkManager, QuantumTask


# ═══════════════════════════════════════════════════════════
#  PROBLEM: 4-qubit weighted Max-Cut
# ═══════════════════════════════════════════════════════════

N_QUBITS = 4
EDGES = [
    (0, 1, 1.0),
    (0, 3, 0.8),
    (1, 2, 1.2),
    (2, 3, 0.9),
    (1, 3, 0.7),
]
P_LAYERS = 2
N_PARAMS = 2 * P_LAYERS


def maxcut_cost(bitstring: str) -> float:
    cost = 0.0
    for i, j, w in EDGES:
        if i < len(bitstring) and j < len(bitstring):
            if bitstring[i] != bitstring[j]:
                cost -= w
    return cost


def brute_force() -> tuple:
    best_cost, best_bs = float("inf"), "0" * N_QUBITS
    for k in range(2 ** N_QUBITS):
        bs = format(k, f"0{N_QUBITS}b")
        c = maxcut_cost(bs)
        if c < best_cost:
            best_cost, best_bs = c, bs
    return best_bs, best_cost


# ═══════════════════════════════════════════════════════════
#  IonQ EXECUTOR — wraps LinkManager into CryoOptimizer API
# ═══════════════════════════════════════════════════════════

class IonQExecutor:
    """Converts QuantumCircuit → gate sequence → IonQ via LinkManager."""

    def __init__(self, link_manager: LinkManager, handle: str = "ionq"):
        self.lm = link_manager
        self.handle = handle
        self.call_count = 0
        self.total_latency_ms = 0.0

    def __call__(self, circuit: QuantumCircuit, shots: int) -> ExecutionResult:
        gate_seq = []
        for op in circuit.ops:
            if op.params:
                gate_seq.append((op.gate_name, op.targets, *op.params))
            else:
                gate_seq.append((op.gate_name, op.targets))

        task = QuantumTask(
            n_qubits=circuit.n_qubits,
            gate_sequence=gate_seq,
            shots=shots,
            tag=f"cryo_{self.call_count}",
        )

        t0 = time.perf_counter()
        counts, link = self.lm.submit(task, prefer=[self.handle])
        lat = (time.perf_counter() - t0) * 1000
        self.total_latency_ms += lat
        self.call_count += 1

        if self.call_count % 50 == 0:
            avg = self.total_latency_ms / self.call_count
            print(f"    [{self.call_count} evals, avg {avg:.0f}ms/eval, "
                  f"via {link.handle}]")

        return ExecutionResult(
            counts=counts,
            statevector=None,
            execution_time=lat / 1000,
            backend_name=link.handle,
            circuit_name=circuit.name,
            n_qubits=circuit.n_qubits,
            gate_count=len(circuit.ops),
            circuit_depth=0,
            entanglement_pairs=[],
            entropy=0.0,
            symmetry_report=None,
            execution_metadata=None,
        )

    def summary(self) -> str:
        avg = self.total_latency_ms / max(1, self.call_count)
        return (f"{self.call_count} evaluations, "
                f"avg {avg:.0f}ms/eval, "
                f"total {self.total_latency_ms/1000:.1f}s on-wire")


# ═══════════════════════════════════════════════════════════
#  QAOA CIRCUIT BUILDER
# ═══════════════════════════════════════════════════════════

def build_qaoa(params: np.ndarray) -> QuantumCircuit:
    gammas = params[:P_LAYERS]
    betas = params[P_LAYERS:]

    circ = QuantumCircuit(N_QUBITS, "cryo_qaoa_ionq")
    for i in range(N_QUBITS):
        circ.h(i)

    for layer in range(P_LAYERS):
        for qi, qj, w in EDGES:
            circ.cnot(qi, qj)
            circ.rz(qj, gammas[layer] * w)
            circ.cnot(qi, qj)
        for i in range(N_QUBITS):
            circ.rx(i, 2 * betas[layer])

    return circ


def cost_from_counts(counts: dict) -> float:
    total = sum(counts.values())
    return sum(maxcut_cost(bs) * c / total for bs, c in counts.items())


# ═══════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════

def main():
    api_key = os.environ.get("IONQ_API_KEY", "")
    backend = os.environ.get("IONQ_BACKEND", "simulator")
    noise = os.environ.get("IONQ_NOISE_MODEL", "")

    if not api_key:
        print("ERROR: Set IONQ_API_KEY environment variable.")
        print("  export IONQ_API_KEY='your-key-here'")
        sys.exit(1)

    print("=" * 60)
    print("  Cryo-Canonical Basin Weaving × IonQ")
    print("=" * 60)
    print(f"  Backend:     {backend}")
    print(f"  Noise model: {noise or 'ideal'}")
    print(f"  Problem:     {N_QUBITS}-qubit Max-Cut, {len(EDGES)} edges")
    print(f"  QAOA layers: {P_LAYERS} (params: {N_PARAMS})")

    gt_bs, gt_cost = brute_force()
    print(f"  Ground truth: |{gt_bs}⟩  cost={gt_cost:.3f}")
    print()

    # ── Connect to IonQ ──────────────────────────────────
    print("Connecting to IonQ...")
    lm = LinkManager()
    try:
        link = lm.forge_ionq(
            handle="ionq",
            api_key=api_key,
            target_backend=backend,
            noise_model=noise or None,
        )
    except Exception as e:
        print(f"ERROR: IonQ handshake failed: {e}")
        sys.exit(1)

    snap = link.snapshot()
    print(f"  Linked: {snap['backend_name']} "
          f"(max {snap['max_qubits']}qb, "
          f"latency {snap['latency_ms']:.0f}ms)")
    print()

    executor = IonQExecutor(lm, handle="ionq")

    # ── Configure CCBW for cloud execution ────────────────
    config = CryoConfig(
        n_force_levels=3,
        epsilon_max=0.3,
        epsilon_min=0.03,
        motif_lambda=1.5,
        n_probe_directions=3,
        max_basins=6,
        refine_steps=15,
        refine_lr=0.05,
        shots=512,
        confidence_threshold=-2.0,
    )

    estimated_evals = (
        8 * config.n_force_levels * (1 + 2 * config.n_probe_directions)
        + 8 * config.n_force_levels * 2 * N_PARAMS
        + 3 * config.refine_steps * 2 * N_PARAMS
    )
    print(f"Estimated evaluations: ~{estimated_evals}")
    print(f"Estimated shots:       ~{estimated_evals * config.shots:,}")
    if "qpu" in backend:
        cost = estimated_evals * config.shots * 0.01
        print(f"Estimated cost:        ~${cost:.2f}")
    print()

    # ── Run CCBW ──────────────────────────────────────────
    print("Running Cryo-Canonical Basin Weaving...")
    t0 = time.perf_counter()

    optimizer = CryoOptimizer(
        build_circuit=build_qaoa,
        cost_from_counts=cost_from_counts,
        executor=executor,
        n_params=N_PARAMS,
        config=config,
        seed=42,
    )
    result = optimizer.run(n_random_starts=8)

    wall = time.perf_counter() - t0
    print()

    # ── Results ───────────────────────────────────────────
    print("=" * 60)
    print("  RESULTS")
    print("=" * 60)
    print(f"  Optimal energy:      {result.optimal_energy:.4f}")
    print(f"  Ground truth:        {gt_cost:.4f}")
    ratio = result.optimal_energy / gt_cost if gt_cost != 0 else 0
    print(f"  Approximation ratio: {ratio:.4f}")
    print(f"  Certified basins:    {len(result.graph.nodes)}")
    print(f"  {executor.summary()}")
    print(f"  Wall time:           {wall:.1f}s")
    print()

    if result.graph.nodes:
        print("  Certified basins (sorted by energy):")
        for i, node in enumerate(sorted(result.graph.nodes, key=lambda n: n.energy)[:5]):
            print(f"    #{i+1}  E={node.energy:.4f}  "
                  f"conf={node.confidence:.3f}  "
                  f"asym={node.asymmetry_norm:.4f}  "
                  f"Q̄={np.mean(node.compliance):.3f}")
        print()

    # ── Final verification circuit ────────────────────────
    print("Final verification (2048 shots)...")
    final_circ = build_qaoa(result.optimal_params)
    final_task = QuantumTask(
        n_qubits=N_QUBITS,
        gate_sequence=[
            (op.gate_name, op.targets, *(op.params or []))
            for op in final_circ.ops
        ],
        shots=2048,
        tag="final_verify",
    )
    final_counts, final_link = lm.submit(final_task, prefer=["ionq"])

    print(f"\n  Top measurement outcomes (via {final_link.handle}):")
    sorted_bs = sorted(final_counts, key=final_counts.get, reverse=True)
    for bs in sorted_bs[:8]:
        cut_val = -maxcut_cost(bs)
        marker = " ← optimal" if cut_val == -gt_cost else ""
        print(f"    |{bs}⟩  count={final_counts[bs]:>4}  "
              f"cut={cut_val:.1f}{marker}")

    # ── Cleanup ───────────────────────────────────────────
    lm.close_all()
    print(f"\n  Link closed. Done.")


if __name__ == "__main__":
    main()
