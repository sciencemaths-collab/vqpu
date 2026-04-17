"""
AQI Application [O] — Combinatorial Optimization on vQPU
=========================================================
Implements QAOA (Quantum Approximate Optimization Algorithm)
on the vQPU architecture for:
  1. Max-Cut — partition a graph to maximize edges cut
  2. Job scheduling — assign jobs to machines minimizing makespan
  3. Portfolio selection — pick assets maximizing return/risk

All verified against classical brute-force.
"""

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from vqpu import vQPU, QuantumCircuit, QuantumRegister, GateLibrary, GateOp
import numpy as np
from itertools import product as iter_product
import time
import json


# ═══════════════════════════════════════════════════════════
#  QAOA ENGINE — The core quantum optimization loop
# ═══════════════════════════════════════════════════════════

class QAOA:
    """
    Quantum Approximate Optimization Algorithm.
    
    Maps combinatorial problems to quantum Hamiltonians,
    then uses alternating cost/mixer layers to search
    for the optimal solution.
    
    Stage 1 (Axiom):    Problem → Cost Hamiltonian H_C
    Stage 2 (Hamilton):  H_C → parameterized circuit
    Stage 3 (Evolve):    Optimize γ, β parameters
    Stage 4 (Measure):   Sample best solution
    """

    def __init__(self, qpu: vQPU, n_qubits: int, cost_function,
                 p_layers: int = 2, name: str = "qaoa"):
        self.qpu = qpu
        self.n_qubits = n_qubits
        self.cost_fn = cost_function
        self.p = p_layers
        self.name = name
        self.lib = GateLibrary()

    def build_circuit(self, gammas: list, betas: list) -> QuantumCircuit:
        """
        Build QAOA circuit:
        |ψ(γ,β)⟩ = U_M(β_p) U_C(γ_p) ... U_M(β_1) U_C(γ_1) |+⟩^n
        """
        c = self.qpu.circuit(self.n_qubits, self.name)

        # Initial superposition: |+⟩^n
        for i in range(self.n_qubits):
            c.h(i)

        # Alternating layers
        for layer in range(self.p):
            # Cost unitary U_C(γ) = e^{-iγH_C}
            # For diagonal Hamiltonians, this is a phase on each basis state
            dim = 2 ** self.n_qubits
            cost_phases = np.zeros(dim, dtype=complex)
            for idx in range(dim):
                bits = [int(b) for b in format(idx, f'0{self.n_qubits}b')]
                cost_val = self.cost_fn(bits)
                cost_phases[idx] = np.exp(-1j * gammas[layer] * cost_val)

            cost_unitary = np.diag(cost_phases)
            c.ops.append(GateOp(
                f"U_C(γ={gammas[layer]:.2f})",
                cost_unitary,
                list(range(self.n_qubits)),
                [gammas[layer]],
                is_two_qubit=(self.n_qubits > 1)
            ))

            # Mixer unitary U_M(β) = e^{-iβΣX_j} = ΠRx(2β)
            for j in range(self.n_qubits):
                c.rx(j, 2 * betas[layer])

        return c

    def evaluate(self, gammas: list, betas: list, shots: int = 2048) -> dict:
        """Run circuit and compute expected cost."""
        circuit = self.build_circuit(gammas, betas)
        result = self.qpu.run(circuit, shots=shots)

        # Compute expected cost from measurement outcomes
        total_cost = 0.0
        total_shots = sum(result.counts.values())
        cost_by_state = {}

        for bitstring, count in result.counts.items():
            bits = [int(b) for b in bitstring]
            cost = self.cost_fn(bits)
            total_cost += cost * count / total_shots
            cost_by_state[bitstring] = cost

        # Find best solution from samples
        best_state = max(cost_by_state, key=cost_by_state.get)
        best_cost = cost_by_state[best_state]

        return {
            "expected_cost": total_cost,
            "best_state": best_state,
            "best_cost": best_cost,
            "counts": result.counts,
            "cost_by_state": cost_by_state,
            "circuit_depth": circuit.depth(),
            "gate_count": len(circuit.ops),
        }

    def optimize(self, iterations: int = 60, lr: float = 0.08,
                 shots: int = 2048, verbose: bool = True) -> dict:
        """
        Optimize QAOA parameters via finite-difference gradient descent.
        This is Stage 3: Quantum Evolution.
        """
        # Initialize parameters
        gammas = np.random.uniform(0, np.pi, self.p)
        betas = np.random.uniform(0, np.pi / 2, self.p)
        epsilon = 0.05

        history = []
        best_overall = {"best_cost": -np.inf}

        if verbose:
            print(f"\n  {'Iter':>5s}  {'E[Cost]':>8s}  {'Best':>8s}  {'State':>12s}")
            print(f"  {'-'*40}")

        for it in range(iterations):
            # Evaluate current parameters
            result = self.evaluate(gammas.tolist(), betas.tolist(), shots)
            history.append({
                "iter": it + 1,
                "expected_cost": result["expected_cost"],
                "best_cost": result["best_cost"],
                "best_state": result["best_state"],
            })

            if result["best_cost"] > best_overall.get("best_cost", -np.inf):
                best_overall = result.copy()
                best_overall["gammas"] = gammas.copy().tolist()
                best_overall["betas"] = betas.copy().tolist()
                best_overall["iter"] = it + 1

            if verbose and (it % 10 == 0 or it == iterations - 1):
                print(f"  {it+1:5d}  {result['expected_cost']:8.3f}  "
                      f"{result['best_cost']:8.3f}  |{result['best_state']}⟩")

            # Gradient estimation (parameter shift)
            grad_gamma = np.zeros(self.p)
            grad_beta = np.zeros(self.p)

            for k in range(self.p):
                # Gamma gradient
                g_plus = gammas.copy(); g_plus[k] += epsilon
                g_minus = gammas.copy(); g_minus[k] -= epsilon
                r_plus = self.evaluate(g_plus.tolist(), betas.tolist(), shots // 2)
                r_minus = self.evaluate(g_minus.tolist(), betas.tolist(), shots // 2)
                grad_gamma[k] = (r_plus["expected_cost"] - r_minus["expected_cost"]) / (2 * epsilon)

                # Beta gradient
                b_plus = betas.copy(); b_plus[k] += epsilon
                b_minus = betas.copy(); b_minus[k] -= epsilon
                r_plus = self.evaluate(gammas.tolist(), b_plus.tolist(), shots // 2)
                r_minus = self.evaluate(gammas.tolist(), b_minus.tolist(), shots // 2)
                grad_beta[k] = (r_plus["expected_cost"] - r_minus["expected_cost"]) / (2 * epsilon)

            # Gradient ascent (maximizing cost)
            gammas += lr * grad_gamma
            betas += lr * grad_beta

        best_overall["history"] = history
        return best_overall


# ═══════════════════════════════════════════════════════════
#  BRUTE FORCE SOLVER — The honest benchmark
# ═══════════════════════════════════════════════════════════

def brute_force(n_bits: int, cost_fn) -> dict:
    """Exhaustively evaluate all 2^n solutions."""
    best_cost = -np.inf
    best_state = None
    all_costs = {}

    for idx in range(2**n_bits):
        bits = [int(b) for b in format(idx, f'0{n_bits}b')]
        cost = cost_fn(bits)
        bitstring = ''.join(str(b) for b in bits)
        all_costs[bitstring] = cost
        if cost > best_cost:
            best_cost = cost
            best_state = bitstring

    return {"best_state": best_state, "best_cost": best_cost, "all_costs": all_costs}


# ═══════════════════════════════════════════════════════════
#  PROBLEM 1: MAX-CUT
#  Partition graph vertices to maximize edges between groups
# ═══════════════════════════════════════════════════════════

def maxcut_cost(bits: list, edges: list, weights: list = None) -> float:
    """
    Cost = sum of weights for edges where endpoints are in different partitions.
    bits[i] = 0 or 1 → partition assignment of vertex i.
    """
    if weights is None:
        weights = [1.0] * len(edges)
    cost = 0.0
    for (u, v), w in zip(edges, weights):
        if bits[u] != bits[v]:
            cost += w
    return cost


def run_maxcut():
    """Test Max-Cut on a 5-vertex weighted graph."""
    print("\n" + "="*60)
    print("  PROBLEM 1: MAX-CUT")
    print("  Partition graph to maximize edges cut")
    print("="*60)

    #   0 --- 1
    #   |\ /| |
    #   | X | |
    #   |/ \| |
    #   3 --- 2
    #    \   /
    #      4
    edges = [
        (0, 1), (0, 2), (0, 3),
        (1, 2), (1, 3),
        (2, 3), (2, 4), (3, 4),
    ]
    weights = [2.0, 1.5, 3.0, 1.0, 2.5, 1.0, 2.0, 1.5]

    n = 5
    cost_fn = lambda bits: maxcut_cost(bits, edges, weights)

    print(f"  Graph: {n} vertices, {len(edges)} weighted edges")
    print(f"  Edges: {edges}")
    print(f"  Weights: {weights}")

    # Brute force
    print("\n  ── Brute force (2^5 = 32 solutions) ──")
    t0 = time.time()
    bf = brute_force(n, cost_fn)
    bf_time = time.time() - t0
    print(f"  Optimal: |{bf['best_state']}⟩ → cost = {bf['best_cost']}")
    print(f"  Time: {bf_time*1000:.1f}ms")

    # Top 5 solutions
    sorted_costs = sorted(bf["all_costs"].items(), key=lambda x: -x[1])
    print(f"  Top 5:")
    for state, cost in sorted_costs[:5]:
        print(f"    |{state}⟩ → {cost:.1f}")

    # QAOA on vQPU
    print(f"\n  ── QAOA on vQPU (p=3 layers) ──")
    qpu = vQPU(backend="simulator", seed=42)
    qaoa = QAOA(qpu, n, cost_fn, p_layers=3, name="maxcut_qaoa")

    t0 = time.time()
    result = qaoa.optimize(iterations=50, lr=0.1, shots=2048, verbose=True)
    qaoa_time = time.time() - t0

    print(f"\n  QAOA best: |{result['best_state']}⟩ → cost = {result['best_cost']}")
    print(f"  Time: {qaoa_time:.2f}s")
    print(f"  Circuit: depth={result['circuit_depth']}, gates={result['gate_count']}")

    # Compare
    match = result["best_cost"] >= bf["best_cost"]
    ratio = result["best_cost"] / bf["best_cost"] * 100
    print(f"\n  ── Comparison ──")
    print(f"  Brute force optimal: {bf['best_cost']}")
    print(f"  QAOA found:          {result['best_cost']}")
    print(f"  Approximation ratio: {ratio:.1f}%")
    print(f"  Found optimal: {'YES' if match else 'NO'}")

    return {
        "problem": "maxcut",
        "n_vertices": n,
        "n_edges": len(edges),
        "brute_force": {"best": bf["best_state"], "cost": bf["best_cost"], "time_ms": bf_time*1000},
        "qaoa": {
            "best": result["best_state"], "cost": result["best_cost"],
            "time_s": qaoa_time, "ratio_pct": ratio,
            "found_optimal": match,
            "p_layers": 3,
        },
        "history": result["history"],
    }


# ═══════════════════════════════════════════════════════════
#  PROBLEM 2: JOB SCHEDULING
#  Assign N jobs to 2 machines, minimize makespan
# ═══════════════════════════════════════════════════════════

def scheduling_cost(bits: list, job_times: list) -> float:
    """
    bits[i] = 0 or 1 → machine assignment.
    Cost = negative makespan (we maximize, so negate).
    We want to MINIMIZE max(machine_0_time, machine_1_time).
    """
    m0 = sum(t for b, t in zip(bits, job_times) if b == 0)
    m1 = sum(t for b, t in zip(bits, job_times) if b == 1)
    makespan = max(m0, m1)
    # Return negative makespan (QAOA maximizes)
    # Add a large offset so costs are positive for better optimization
    total = sum(job_times)
    return total - makespan


def run_scheduling():
    """Test job scheduling on 6 jobs across 2 machines."""
    print("\n" + "="*60)
    print("  PROBLEM 2: JOB SCHEDULING")
    print("  Assign 6 jobs to 2 machines, minimize makespan")
    print("="*60)

    job_times = [3, 5, 7, 2, 4, 6]  # Hours per job
    n = len(job_times)
    total = sum(job_times)  # 27 hours total

    print(f"  Jobs: {n} with times {job_times}")
    print(f"  Total work: {total} hours")
    print(f"  Perfect balance: {total/2} hours each")

    cost_fn = lambda bits: scheduling_cost(bits, job_times)

    # Brute force
    print(f"\n  ── Brute force (2^{n} = {2**n} assignments) ──")
    t0 = time.time()
    bf = brute_force(n, cost_fn)
    bf_time = time.time() - t0

    bf_bits = [int(b) for b in bf["best_state"]]
    m0_jobs = [job_times[i] for i in range(n) if bf_bits[i] == 0]
    m1_jobs = [job_times[i] for i in range(n) if bf_bits[i] == 1]
    bf_makespan = total - bf["best_cost"]

    print(f"  Optimal: |{bf['best_state']}⟩")
    print(f"  Machine 0: jobs {m0_jobs} = {sum(m0_jobs)}h")
    print(f"  Machine 1: jobs {m1_jobs} = {sum(m1_jobs)}h")
    print(f"  Makespan: {bf_makespan}h")

    # QAOA
    print(f"\n  ── QAOA on vQPU (p=3 layers) ──")
    qpu = vQPU(backend="simulator", seed=123)
    qaoa = QAOA(qpu, n, cost_fn, p_layers=3, name="schedule_qaoa")

    t0 = time.time()
    result = qaoa.optimize(iterations=50, lr=0.1, shots=2048, verbose=True)
    qaoa_time = time.time() - t0

    q_bits = [int(b) for b in result["best_state"]]
    q_m0 = [job_times[i] for i in range(n) if q_bits[i] == 0]
    q_m1 = [job_times[i] for i in range(n) if q_bits[i] == 1]
    q_makespan = total - result["best_cost"]

    print(f"\n  QAOA best: |{result['best_state']}⟩")
    print(f"  Machine 0: jobs {q_m0} = {sum(q_m0)}h")
    print(f"  Machine 1: jobs {q_m1} = {sum(q_m1)}h")
    print(f"  Makespan: {q_makespan}h")

    match = result["best_cost"] >= bf["best_cost"]
    print(f"\n  ── Comparison ──")
    print(f"  Optimal makespan:  {bf_makespan}h")
    print(f"  QAOA makespan:     {q_makespan}h")
    print(f"  Found optimal: {'YES' if match else 'NO'}")

    return {
        "problem": "scheduling",
        "n_jobs": n,
        "job_times": job_times,
        "brute_force": {"best": bf["best_state"], "makespan": bf_makespan},
        "qaoa": {
            "best": result["best_state"], "makespan": q_makespan,
            "found_optimal": match, "time_s": qaoa_time,
        },
        "history": result["history"],
    }


# ═══════════════════════════════════════════════════════════
#  PROBLEM 3: PORTFOLIO SELECTION
#  Pick k assets from n to maximize return-to-risk ratio
# ═══════════════════════════════════════════════════════════

def portfolio_cost(bits: list, returns: list, risks: list,
                   correlations: list, budget: int, penalty: float = 5.0) -> float:
    """
    bits[i] = 1 → include asset i.
    Maximize: sum(returns) - λ*risk - penalty*(sum(bits) - budget)^2
    """
    n_selected = sum(bits)
    total_return = sum(r * b for r, b in zip(returns, bits))

    # Portfolio risk with correlations
    total_risk = 0.0
    for i in range(len(bits)):
        for j in range(len(bits)):
            if bits[i] and bits[j]:
                total_risk += risks[i] * risks[j] * correlations[i][j]
    total_risk = np.sqrt(max(0, total_risk))

    # Budget constraint (must pick exactly 'budget' assets)
    budget_penalty = penalty * (n_selected - budget) ** 2

    return total_return - 0.5 * total_risk - budget_penalty


def run_portfolio():
    """Test portfolio selection: pick 3 from 6 assets."""
    print("\n" + "="*60)
    print("  PROBLEM 3: PORTFOLIO SELECTION")
    print("  Pick 3 from 6 assets, maximize risk-adjusted return")
    print("="*60)

    n = 6
    budget = 3
    asset_names = ["TECH", "BANK", "ENRG", "HLTH", "RETL", "UTIL"]
    returns = [0.12, 0.08, 0.15, 0.10, 0.07, 0.05]
    risks =   [0.20, 0.12, 0.25, 0.15, 0.10, 0.08]

    # Correlation matrix
    corr = [
        [1.0, 0.5, 0.3, 0.2, 0.4, 0.1],
        [0.5, 1.0, 0.4, 0.3, 0.5, 0.3],
        [0.3, 0.4, 1.0, 0.2, 0.3, 0.2],
        [0.2, 0.3, 0.2, 1.0, 0.3, 0.4],
        [0.4, 0.5, 0.3, 0.3, 1.0, 0.3],
        [0.1, 0.3, 0.2, 0.4, 0.3, 1.0],
    ]

    print(f"  Assets: {asset_names}")
    print(f"  Returns: {returns}")
    print(f"  Risks:   {risks}")
    print(f"  Budget: pick {budget} of {n}")

    cost_fn = lambda bits: portfolio_cost(bits, returns, risks, corr, budget)

    # Brute force
    print(f"\n  ── Brute force (2^{n} = {2**n} portfolios) ──")
    bf = brute_force(n, cost_fn)
    bf_bits = [int(b) for b in bf["best_state"]]
    selected = [asset_names[i] for i in range(n) if bf_bits[i] == 1]
    print(f"  Optimal: |{bf['best_state']}⟩ → {selected}")
    print(f"  Score: {bf['best_cost']:.4f}")

    # QAOA
    print(f"\n  ── QAOA on vQPU (p=3 layers) ──")
    qpu = vQPU(backend="simulator", seed=77)
    qaoa = QAOA(qpu, n, cost_fn, p_layers=3, name="portfolio_qaoa")

    t0 = time.time()
    result = qaoa.optimize(iterations=50, lr=0.08, shots=2048, verbose=True)
    qaoa_time = time.time() - t0

    q_bits = [int(b) for b in result["best_state"]]
    q_selected = [asset_names[i] for i in range(n) if q_bits[i] == 1]
    print(f"\n  QAOA best: |{result['best_state']}⟩ → {q_selected}")
    print(f"  Score: {result['best_cost']:.4f}")

    match = result["best_cost"] >= bf["best_cost"] - 0.01
    print(f"\n  ── Comparison ──")
    print(f"  Optimal score: {bf['best_cost']:.4f} → {selected}")
    print(f"  QAOA score:    {result['best_cost']:.4f} → {q_selected}")
    print(f"  Found optimal: {'YES' if match else 'NO'}")

    return {
        "problem": "portfolio",
        "n_assets": n,
        "budget": budget,
        "brute_force": {"best": bf["best_state"], "score": bf["best_cost"], "assets": selected},
        "qaoa": {
            "best": result["best_state"], "score": result["best_cost"],
            "assets": q_selected, "found_optimal": match,
        },
        "history": result["history"],
    }


# ═══════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("╔" + "═"*58 + "╗")
    print("║  AQI Application [O] — Combinatorial Optimization       ║")
    print("║  QAOA on vQPU Architecture                              ║")
    print("╚" + "═"*58 + "╝")

    all_results = {}

    all_results["maxcut"] = run_maxcut()
    all_results["scheduling"] = run_scheduling()
    all_results["portfolio"] = run_portfolio()

    # Summary
    print("\n" + "="*60)
    print("  SUMMARY")
    print("="*60)
    for name, res in all_results.items():
        qaoa = res["qaoa"]
        found = qaoa.get("found_optimal", False)
        print(f"  {name:15s} — Found optimal: {'YES' if found else 'NO'}")
    print("="*60)

    with open("/home/claude/optimization_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\n  Results saved to /home/claude/optimization_results.json")
