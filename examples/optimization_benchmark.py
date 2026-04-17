"""
Benchmark a hard QAOA optimization workload on vQPU and plot convergence,
runtime, and backend scaling.
"""

import argparse
import json
import pathlib
import sys
import time

import matplotlib
import networkx as nx
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_ROOT))

from optimize import QAOA, brute_force, maxcut_cost
from vqpu import UniversalvQPU, vQPU


OUTPUT_DIR = REPO_ROOT / "benchmark_results" / "optimization_benchmark"


def build_weighted_maxcut_instance(n_vertices: int, degree: int, seed: int):
    """Create a reproducible weighted regular graph for Max-Cut."""
    graph = nx.random_regular_graph(degree, n_vertices, seed=seed)
    rng = np.random.default_rng(seed)
    edges = sorted(tuple(sorted(edge)) for edge in graph.edges())
    weights = [float(rng.integers(2, 10)) / 2.0 for _ in edges]
    return graph, edges, weights


def partition_from_bits(bitstring: str):
    left = [idx for idx, bit in enumerate(bitstring) if bit == "0"]
    right = [idx for idx, bit in enumerate(bitstring) if bit == "1"]
    return left, right


def timed_evaluate(qaoa: QAOA, gammas: np.ndarray, betas: np.ndarray, shots: int):
    start = time.perf_counter()
    result = qaoa.evaluate(gammas.tolist(), betas.tolist(), shots=shots)
    elapsed = time.perf_counter() - start
    return result, elapsed


def optimize_with_metrics(
    qaoa: QAOA,
    optimum_cost: float,
    iterations: int,
    shots: int,
    lr: float,
    epsilon: float,
    seed: int,
):
    """Run QAOA with timing instrumentation for every iteration."""
    rng = np.random.default_rng(seed)
    gammas = rng.uniform(0, np.pi, qaoa.p)
    betas = rng.uniform(0, np.pi / 2, qaoa.p)

    history = []
    best_overall = {"best_cost": -np.inf}
    total_eval_time = 0.0
    total_evaluations = 0
    total_shots = 0

    for iteration in range(1, iterations + 1):
        iter_start = time.perf_counter()

        result, eval_time = timed_evaluate(qaoa, gammas, betas, shots)
        evals_this_iter = 1
        shots_this_iter = shots
        grad_time = 0.0
        total_eval_time += eval_time
        total_evaluations += 1
        total_shots += shots

        if result["best_cost"] > best_overall["best_cost"]:
            best_overall = {
                **result,
                "gammas": gammas.copy().tolist(),
                "betas": betas.copy().tolist(),
                "iter": iteration,
            }

        grad_gamma = np.zeros(qaoa.p)
        grad_beta = np.zeros(qaoa.p)

        for k in range(qaoa.p):
            g_plus = gammas.copy()
            g_plus[k] += epsilon
            g_minus = gammas.copy()
            g_minus[k] -= epsilon
            r_plus, t_plus = timed_evaluate(qaoa, g_plus, betas, shots // 2)
            r_minus, t_minus = timed_evaluate(qaoa, g_minus, betas, shots // 2)
            grad_gamma[k] = (
                r_plus["expected_cost"] - r_minus["expected_cost"]
            ) / (2 * epsilon)
            grad_time += t_plus + t_minus
            total_eval_time += t_plus + t_minus
            evals_this_iter += 2
            total_evaluations += 2
            shots_this_iter += shots
            total_shots += shots

            b_plus = betas.copy()
            b_plus[k] += epsilon
            b_minus = betas.copy()
            b_minus[k] -= epsilon
            r_plus, t_plus = timed_evaluate(qaoa, gammas, b_plus, shots // 2)
            r_minus, t_minus = timed_evaluate(qaoa, gammas, b_minus, shots // 2)
            grad_beta[k] = (
                r_plus["expected_cost"] - r_minus["expected_cost"]
            ) / (2 * epsilon)
            grad_time += t_plus + t_minus
            total_eval_time += t_plus + t_minus
            evals_this_iter += 2
            total_evaluations += 2
            shots_this_iter += shots
            total_shots += shots

        gammas += lr * grad_gamma
        betas += lr * grad_beta

        iter_time = time.perf_counter() - iter_start
        approximation_ratio = result["best_cost"] / optimum_cost if optimum_cost else 0.0
        history.append(
            {
                "iter": iteration,
                "expected_cost": result["expected_cost"],
                "best_cost": result["best_cost"],
                "best_state": result["best_state"],
                "approximation_ratio": approximation_ratio,
                "iter_time_s": iter_time,
                "eval_time_s": eval_time,
                "grad_time_s": grad_time,
                "evaluations": evals_this_iter,
                "shots": shots_this_iter,
                "circuit_depth": result["circuit_depth"],
                "gate_count": result["gate_count"],
            }
        )

    return {
        "history": history,
        "best": best_overall,
        "total_evaluations": total_evaluations,
        "total_eval_time_s": total_eval_time,
        "total_shots": total_shots,
        "avg_eval_time_s": total_eval_time / max(total_evaluations, 1),
    }


def benchmark_backend_scaling(
    qubits_list,
    p_layers: int,
    shots: int,
    repeats: int,
    seed: int,
):
    """Measure single-evaluation time as the workload scales."""
    scaling = {"vqpu": [], "universal": []}
    rng = np.random.default_rng(seed)

    for n_qubits in qubits_list:
        _, edges, weights = build_weighted_maxcut_instance(
            n_vertices=n_qubits,
            degree=3,
            seed=seed + n_qubits,
        )
        cost_fn = lambda bits, e=edges, w=weights: maxcut_cost(bits, e, w)

        for backend_name, backend in (
            ("vqpu", vQPU(backend="simulator", seed=seed)),
            ("universal", UniversalvQPU(verbose=False)),
        ):
            qaoa = QAOA(
                backend,
                n_qubits,
                cost_fn,
                p_layers=p_layers,
                name=f"scaling_{backend_name}_{n_qubits}q",
            )
            gammas = rng.uniform(0, np.pi, qaoa.p)
            betas = rng.uniform(0, np.pi / 2, qaoa.p)
            circuit = qaoa.build_circuit(gammas.tolist(), betas.tolist())

            times = []
            for _ in range(repeats):
                _, elapsed = timed_evaluate(qaoa, gammas, betas, shots)
                times.append(elapsed)

            scaling[backend_name].append(
                {
                    "n_qubits": n_qubits,
                    "n_edges": len(edges),
                    "gate_count": len(circuit.ops),
                    "depth": circuit.depth(),
                    "median_eval_time_s": float(np.median(times)),
                    "min_eval_time_s": float(np.min(times)),
                    "max_eval_time_s": float(np.max(times)),
                    "shots_per_second": shots / float(np.median(times)),
                }
            )

    return scaling


def plot_instance_graph(graph, weights, best_state: str, output_path: pathlib.Path):
    edge_labels = {
        edge: f"{weight:.1f}"
        for edge, weight in zip(sorted(tuple(sorted(e)) for e in graph.edges()), weights)
    }
    colors = ["#0f766e" if bit == "0" else "#b45309" for bit in best_state]
    pos = nx.spring_layout(graph, seed=11)

    fig, ax = plt.subplots(figsize=(7, 5.5))
    nx.draw_networkx_nodes(graph, pos, node_color=colors, node_size=700, ax=ax)
    nx.draw_networkx_edges(graph, pos, width=2.0, ax=ax)
    nx.draw_networkx_labels(graph, pos, font_color="white", font_weight="bold", ax=ax)
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, font_size=9, ax=ax)
    ax.set_title("Weighted Max-Cut Benchmark Instance")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_convergence(history, optimum_cost: float, output_path: pathlib.Path):
    iterations = [row["iter"] for row in history]
    expected = [row["expected_cost"] for row in history]
    best = [row["best_cost"] for row in history]
    ratio = [row["approximation_ratio"] * 100.0 for row in history]
    iter_time = [row["iter_time_s"] for row in history]

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    axes[0].plot(iterations, expected, marker="o", linewidth=2, label="Expected cost")
    axes[0].plot(iterations, best, marker="s", linewidth=2, label="Best sampled cost")
    axes[0].axhline(optimum_cost, color="#991b1b", linestyle="--", linewidth=1.8, label="Brute-force optimum")
    axes[0].set_ylabel("Cut value")
    axes[0].set_title("QAOA Convergence on a Hard Weighted Max-Cut Instance")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    axes[1].bar(iterations, iter_time, color="#2563eb", alpha=0.85, label="Iteration time")
    ax2 = axes[1].twinx()
    ax2.plot(iterations, ratio, color="#b45309", marker="d", linewidth=2, label="Approximation ratio")
    axes[1].set_xlabel("Iteration")
    axes[1].set_ylabel("Iteration time (s)")
    ax2.set_ylabel("Approximation ratio (%)")
    axes[1].grid(alpha=0.25)

    lines_1, labels_1 = axes[1].get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    axes[1].legend(lines_1 + lines_2, labels_1 + labels_2, loc="lower right")

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_scaling(scaling, output_path: pathlib.Path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))

    for backend_name, color in (("vqpu", "#0f766e"), ("universal", "#7c3aed")):
        rows = scaling[backend_name]
        qubits = [row["n_qubits"] for row in rows]
        eval_time = [row["median_eval_time_s"] for row in rows]
        throughput = [row["shots_per_second"] for row in rows]

        axes[0].plot(qubits, eval_time, marker="o", linewidth=2, color=color, label=backend_name)
        axes[1].plot(qubits, throughput, marker="s", linewidth=2, color=color, label=backend_name)

    axes[0].set_title("Single-Evaluation Scaling")
    axes[0].set_xlabel("Qubits")
    axes[0].set_ylabel("Median evaluation time (s)")
    axes[0].set_yscale("log")
    axes[0].grid(alpha=0.25)

    axes[1].set_title("Sampling Throughput")
    axes[1].set_xlabel("Qubits")
    axes[1].set_ylabel("Shots per second")
    axes[1].grid(alpha=0.25)

    for ax in axes:
        ax.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def write_summary(results, output_path: pathlib.Path):
    summary = {
        "problem": results["problem"],
        "instance": results["instance"],
        "brute_force": results["brute_force"],
        "optimization": results["optimization"],
        "scaling": results["scaling"],
        "artifacts": results["artifacts"],
    }
    output_path.write_text(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=16)
    parser.add_argument("--shots", type=int, default=2048)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=0.08)
    parser.add_argument("--epsilon", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--hard-qubits", type=int, default=10)
    parser.add_argument("--degree", type=int, default=3)
    parser.add_argument("--scaling-qubits", type=int, nargs="+", default=[4, 6, 8, 10])
    parser.add_argument("--scaling-repeats", type=int, default=3)
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    graph, edges, weights = build_weighted_maxcut_instance(
        n_vertices=args.hard_qubits,
        degree=args.degree,
        seed=args.seed,
    )
    cost_fn = lambda bits: maxcut_cost(bits, edges, weights)

    brute_force_start = time.perf_counter()
    brute_force_result = brute_force(args.hard_qubits, cost_fn)
    brute_force_time = time.perf_counter() - brute_force_start

    qpu = vQPU(backend="simulator", seed=args.seed)
    qaoa = QAOA(
        qpu,
        args.hard_qubits,
        cost_fn,
        p_layers=args.layers,
        name="hard_weighted_maxcut",
    )

    optimization_start = time.perf_counter()
    optimization = optimize_with_metrics(
        qaoa=qaoa,
        optimum_cost=brute_force_result["best_cost"],
        iterations=args.iterations,
        shots=args.shots,
        lr=args.learning_rate,
        epsilon=args.epsilon,
        seed=args.seed,
    )
    optimization_time = time.perf_counter() - optimization_start

    best_state = optimization["best"]["best_state"]
    best_cost = optimization["best"]["best_cost"]
    optimum_cost = brute_force_result["best_cost"]
    left, right = partition_from_bits(best_state)
    optimum_left, optimum_right = partition_from_bits(brute_force_result["best_state"])

    scaling = benchmark_backend_scaling(
        qubits_list=args.scaling_qubits,
        p_layers=args.layers,
        shots=args.shots,
        repeats=args.scaling_repeats,
        seed=args.seed,
    )

    artifacts = {
        "instance_graph": str(OUTPUT_DIR / "maxcut_instance.png"),
        "convergence_plot": str(OUTPUT_DIR / "optimization_convergence.png"),
        "scaling_plot": str(OUTPUT_DIR / "backend_scaling.png"),
        "summary_json": str(OUTPUT_DIR / "summary.json"),
    }

    plot_instance_graph(graph, weights, brute_force_result["best_state"], OUTPUT_DIR / "maxcut_instance.png")
    plot_convergence(optimization["history"], optimum_cost, OUTPUT_DIR / "optimization_convergence.png")
    plot_scaling(scaling, OUTPUT_DIR / "backend_scaling.png")

    results = {
        "problem": "weighted_maxcut_qaoa",
        "instance": {
            "n_qubits": args.hard_qubits,
            "degree": args.degree,
            "n_edges": len(edges),
            "edges": edges,
            "weights": weights,
        },
        "brute_force": {
            "best_state": brute_force_result["best_state"],
            "best_cost": brute_force_result["best_cost"],
            "time_s": brute_force_time,
            "partition_left": optimum_left,
            "partition_right": optimum_right,
        },
        "optimization": {
            "best_state": best_state,
            "best_cost": best_cost,
            "approximation_ratio": best_cost / optimum_cost if optimum_cost else 0.0,
            "time_s": optimization_time,
            "iterations": args.iterations,
            "shots": args.shots,
            "layers": args.layers,
            "total_evaluations": optimization["total_evaluations"],
            "total_shots": optimization["total_shots"],
            "avg_eval_time_s": optimization["avg_eval_time_s"],
            "circuit_depth": optimization["best"]["circuit_depth"],
            "gate_count": optimization["best"]["gate_count"],
            "partition_left": left,
            "partition_right": right,
            "history": optimization["history"],
        },
        "scaling": scaling,
        "artifacts": artifacts,
    }

    write_summary(results, OUTPUT_DIR / "summary.json")

    print("Optimization benchmark complete")
    print(f"  Instance: {args.hard_qubits} qubits, {len(edges)} weighted edges")
    print(f"  Brute-force optimum: {optimum_cost:.3f} @ |{brute_force_result['best_state']}|")
    print(f"  Best QAOA sample:   {best_cost:.3f} @ |{best_state}|")
    print(f"  Approx. ratio:      {100.0 * best_cost / optimum_cost:.2f}%")
    print(f"  Total optimize time: {optimization_time:.2f}s")
    print(f"  Total evaluations:   {optimization['total_evaluations']}")
    print(f"  Average eval time:   {optimization['avg_eval_time_s']:.4f}s")
    print("Artifacts:")
    for name, path in artifacts.items():
        print(f"  {name}: {path}")


if __name__ == "__main__":
    main()
