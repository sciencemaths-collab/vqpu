"""Cryo-Canonical Basin Weaving on Max-Cut QAOA.

Demonstrates the CCBW optimizer finding the optimal cut of a 5-vertex
graph via QAOA, compared against brute-force ground truth.

The optimizer:
1. Probes the γ/β landscape at multiple scales (progressive probing)
2. Builds 3-3+1 motifs to certify stable basins
3. Filters out saddle points and noise-sensitive minima
4. Refines the coldest basin via gradient descent
"""

import numpy as np
from vqpu import UniversalvQPU, CryoConfig, cryo_qaoa


def maxcut_cost(bitstring: str, edges: list) -> float:
    cost = 0.0
    for i, j, w in edges:
        if i < len(bitstring) and j < len(bitstring):
            if bitstring[i] != bitstring[j]:
                cost -= w
    return cost


def brute_force_maxcut(n: int, edges: list) -> tuple:
    best_cost = float("inf")
    best_bs = "0" * n
    for k in range(2 ** n):
        bs = format(k, f"0{n}b")
        c = maxcut_cost(bs, edges)
        if c < best_cost:
            best_cost = c
            best_bs = bs
    return best_bs, best_cost


def main():
    qpu = UniversalvQPU()

    n_qubits = 5
    edges = [
        (0, 1, 1.0), (0, 2, 0.8), (1, 2, 1.2),
        (2, 3, 0.9), (3, 4, 1.1), (1, 4, 0.7),
    ]

    gt_bs, gt_cost = brute_force_maxcut(n_qubits, edges)
    print(f"=== Max-Cut on 5-vertex graph ===")
    print(f"Ground truth: |{gt_bs}⟩  cost={gt_cost:.3f}")

    config = CryoConfig(
        n_force_levels=3,
        epsilon_max=0.4,
        epsilon_min=0.02,
        refine_steps=20,
        refine_lr=0.06,
        shots=1024,
        max_basins=10,
        confidence_threshold=-1.0,
    )

    def cost_fn(bs: str) -> float:
        return maxcut_cost(bs, edges)

    print(f"\nRunning Cryo-QAOA (p=2 layers)...")
    result = cryo_qaoa(
        n_qubits=n_qubits,
        cost_function=cost_fn,
        executor=qpu.run,
        p_layers=2,
        config=config,
        seed=42,
    )

    print(f"\nResults:")
    print(f"  Optimal energy:     {result.optimal_energy:.4f}")
    print(f"  Ground truth:       {gt_cost:.4f}")
    print(f"  Approximation ratio: {result.optimal_energy / gt_cost:.4f}")
    print(f"  Certified basins:   {len(result.graph.nodes)}")
    print(f"  Total evaluations:  {result.n_evaluations}")
    print(f"  Wall time:          {result.wall_time_s:.2f}s")

    print(f"\n  Force schedule: {[f'{e:.4f}' for e in result.force_schedule]}")

    if result.graph.nodes:
        print(f"\n  Top 3 certified basins:")
        for i, node in enumerate(sorted(result.graph.nodes, key=lambda n: n.energy)[:3]):
            print(
                f"    #{i+1}  E={node.energy:.4f}  "
                f"conf={node.confidence:.3f}  "
                f"asym={node.asymmetry_norm:.4f}  "
                f"Q̄={np.mean(node.compliance):.3f}"
            )

    print(f"\n  Convergence (last 10): {[f'{e:.4f}' for e in result.convergence[-10:]]}")

    final_circ = qpu.circuit(n_qubits, "final_check")
    for i in range(n_qubits):
        final_circ.h(i)
    gammas = result.optimal_params[:2]
    betas = result.optimal_params[2:]
    for layer in range(2):
        for i, j, w in edges:
            final_circ.cnot(i, j)
            final_circ.rz(j, gammas[layer])
            final_circ.cnot(i, j)
        for i in range(n_qubits):
            final_circ.rx(i, 2 * betas[layer])

    final_result = qpu.run(final_circ, shots=4096)
    print(f"\n  Final measurement (top 5):")
    for bs in sorted(final_result.counts, key=final_result.counts.get, reverse=True)[:5]:
        cut_val = -maxcut_cost(bs, edges)
        print(f"    |{bs}⟩  count={final_result.counts[bs]:>4}  cut_value={cut_val:.1f}")


if __name__ == "__main__":
    main()
