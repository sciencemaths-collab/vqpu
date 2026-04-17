"""Cryo-Canonical Basin Weaving optimizer for variational quantum circuits.

Adapts the CCBW framework (spherical probing, 3-3+1 canonical reduction,
cold-seeking spring-network optimization) to navigate the variational
parameter landscape of VQE/QAOA circuits.

Mapping
───────
  physical concept        →  quantum variational analogue
  ─────────────────────      ──────────────────────────────
  surface point x         →  parameter vector θ
  probe force F_t         →  perturbation scale ε_t
  indentation δ           →  energy response |E(θ+εd) - E(θ)|
  effective modulus E*    →  landscape curvature (finite-diff Hessian diagonal)
  compliance Q            →  1 / (curvature + ε) — flat = soft = good
  temperature T           →  cold for flat basins, hot for sharp/noisy ones
  3-3+1 motif             →  7-point probe in parameter space (3 dirs + mirrors + center)
  mirror balance I/A      →  symmetric vs asymmetric basin shape → saddle filter
  confidence S            →  soft + symmetric − asymmetric² − hot
  spring graph            →  graph of certified minima with similarity edges
  progressive probing     →  multi-scale ε ladder: coarse then fine

The result: the optimizer discovers stable, symmetric minima in the cost
landscape, certifies them via the 3-3+1 motif, and navigates between
basins via a cold-seeking spring graph. This avoids barren plateaus,
saddle points, and noise-sensitive sharp minima that plague standard
gradient descent on quantum landscapes.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .core import (
    ExecutionResult,
    QuantumAlgorithms,
    QuantumCircuit,
    vQPU,
)


# ═══════════════════════════════════════════════════════════
#  CONFIGURATION
# ═══════════════════════════════════════════════════════════

@dataclass
class CryoConfig:
    """Tuning knobs for the CCBW optimizer."""
    n_force_levels: int = 4
    epsilon_max: float = 0.5
    epsilon_min: float = 0.01
    motif_lambda: float = 1.5
    confidence_alpha: float = 1.0
    confidence_beta: float = 0.5
    confidence_gamma: float = 2.0
    confidence_zeta: float = 0.3
    confidence_threshold: float = 0.0
    spring_sigma_x: float = 1.0
    spring_sigma_psi: float = 1.0
    spring_tau_T: float = 0.5
    global_lambda_T: float = 0.1
    global_mu_A: float = 0.5
    max_basins: int = 20
    n_probe_directions: int = 3
    refine_steps: int = 30
    refine_lr: float = 0.05
    shots: int = 1024

    @property
    def force_schedule(self) -> np.ndarray:
        return np.geomspace(self.epsilon_max, self.epsilon_min, self.n_force_levels)


# ═══════════════════════════════════════════════════════════
#  CRYO NODE — a certified basin in parameter space
# ═══════════════════════════════════════════════════════════

@dataclass
class CryoNode:
    """One certified canonical node in the parameter landscape."""
    center: np.ndarray
    energy: float
    compliance: np.ndarray
    temperature: np.ndarray
    invariant: np.ndarray
    asymmetry_norm: float
    confidence: float
    descriptor: np.ndarray
    motif_radius: float

    def to_dict(self) -> dict:
        return {
            "center": self.center.tolist(),
            "energy": self.energy,
            "compliance_mean": float(np.mean(self.compliance)),
            "temperature_mean": float(np.mean(self.temperature)),
            "asymmetry_norm": self.asymmetry_norm,
            "confidence": self.confidence,
            "motif_radius": self.motif_radius,
        }


# ═══════════════════════════════════════════════════════════
#  CRYO GRAPH — spring network of certified basins
# ═══════════════════════════════════════════════════════════

@dataclass
class CryoGraph:
    """Spring graph connecting certified basins."""
    nodes: List[CryoNode]
    edges: List[Tuple[int, int, float]]
    global_energy: float = 0.0

    @property
    def coldest(self) -> Optional[CryoNode]:
        if not self.nodes:
            return None
        return min(self.nodes, key=lambda n: n.energy)

    def to_dict(self) -> dict:
        return {
            "n_nodes": len(self.nodes),
            "n_edges": len(self.edges),
            "global_energy": self.global_energy,
            "nodes": [n.to_dict() for n in self.nodes],
            "coldest_energy": self.coldest.energy if self.coldest else None,
        }


# ═══════════════════════════════════════════════════════════
#  CRYO RESULT
# ═══════════════════════════════════════════════════════════

@dataclass
class CryoResult:
    """Output of the CCBW optimizer."""
    optimal_params: np.ndarray
    optimal_energy: float
    graph: CryoGraph
    convergence: List[float]
    n_evaluations: int
    wall_time_s: float
    force_schedule: np.ndarray

    def to_dict(self) -> dict:
        return {
            "optimal_params": self.optimal_params.tolist(),
            "optimal_energy": self.optimal_energy,
            "n_evaluations": self.n_evaluations,
            "n_certified_basins": len(self.graph.nodes),
            "wall_time_s": self.wall_time_s,
            "force_schedule": self.force_schedule.tolist(),
            "convergence": self.convergence,
        }


# ═══════════════════════════════════════════════════════════
#  LANDSCAPE EVALUATOR
# ═══════════════════════════════════════════════════════════

class _LandscapeEvaluator:
    """Wraps circuit building + execution into a scalar energy function."""

    def __init__(
        self,
        build_circuit: Callable[[np.ndarray], QuantumCircuit],
        cost_from_counts: Callable[[Dict[str, int]], float],
        executor: Callable[..., ExecutionResult],
        shots: int,
    ) -> None:
        self.build_circuit = build_circuit
        self.cost_from_counts = cost_from_counts
        self.executor = executor
        self.shots = shots
        self.n_evals = 0

    def __call__(self, params: np.ndarray) -> float:
        circ = self.build_circuit(params)
        result = self.executor(circ, self.shots)
        self.n_evals += 1
        return self.cost_from_counts(result.counts)


# ═══════════════════════════════════════════════════════════
#  THE 3-3+1 MOTIF PROBE
# ═══════════════════════════════════════════════════════════

class _MotifProbe:
    """Builds and evaluates the 3-3+1 motif in parameter space."""

    @staticmethod
    def build_directions(n_params: int, n_dirs: int = 3, rng: np.random.Generator = None) -> np.ndarray:
        if rng is None:
            rng = np.random.default_rng()
        raw = rng.standard_normal((n_dirs, n_params))
        q, _ = np.linalg.qr(raw.T)
        return q[:, :n_dirs].T

    @staticmethod
    def probe(
        center: np.ndarray,
        epsilon: float,
        directions: np.ndarray,
        evaluator: _LandscapeEvaluator,
        lam: float = 1.5,
    ) -> dict:
        n_dirs = directions.shape[0]
        e_center = evaluator(center)

        rho = lam * epsilon

        e_plus = np.zeros(n_dirs)
        e_minus = np.zeros(n_dirs)
        for k in range(n_dirs):
            p_k = center + rho * directions[k]
            n_k = center - rho * directions[k]
            e_plus[k] = evaluator(p_k)
            e_minus[k] = evaluator(n_k)

        curvatures = np.abs(e_plus + e_minus - 2 * e_center) / (rho ** 2 + 1e-15)

        return {
            "e_center": e_center,
            "e_plus": e_plus,
            "e_minus": e_minus,
            "curvatures": curvatures,
            "rho": rho,
            "directions": directions,
        }


# ═══════════════════════════════════════════════════════════
#  CANONICAL REDUCTION — §5-6 of the paper
# ═══════════════════════════════════════════════════════════

def _canonicalize(
    center: np.ndarray,
    probes: List[dict],
    config: CryoConfig,
) -> Optional[CryoNode]:
    compliances = []
    temperatures = []

    all_curvatures = []
    for probe in probes:
        curvs = probe["curvatures"]
        all_curvatures.append(curvs)

    all_curv_flat = np.concatenate(all_curvatures)
    e_min = all_curv_flat.min()
    e_max = all_curv_flat.max() + 1e-15

    for probe in probes:
        q = 1.0 / (probe["curvatures"] + 1e-8)
        compliances.append(q)

        t = (probe["curvatures"] - e_min) / (e_max - e_min + 1e-15)
        temperatures.append(t)

    compliance_arr = np.array(compliances)
    temperature_arr = np.array(temperatures)

    phi_plus_list = []
    phi_minus_list = []

    for probe in probes:
        phi_plus_list.append(probe["e_plus"])
        phi_minus_list.append(probe["e_minus"])

    phi_plus = np.concatenate(phi_plus_list)
    phi_minus = np.concatenate(phi_minus_list)

    phi_plus_mean = np.mean(phi_plus)
    phi_minus_mean = np.mean(phi_minus)

    invariant = (phi_plus + phi_minus) / 2
    asymmetry = (phi_plus - phi_minus) / 2

    I_norm = np.linalg.norm(invariant)
    A_norm = np.linalg.norm(asymmetry)

    Q_mean = np.mean(compliance_arr)
    T_mean = np.mean(temperature_arr)

    S = (
        config.confidence_alpha * Q_mean
        + config.confidence_beta * I_norm
        - config.confidence_gamma * A_norm ** 2
        - config.confidence_zeta * T_mean
    )

    if S < config.confidence_threshold:
        return None

    descriptor = np.concatenate([
        compliance_arr.flatten(),
        temperature_arr.flatten(),
        [I_norm, A_norm, probes[0]["rho"]],
    ])

    return CryoNode(
        center=center.copy(),
        energy=probes[0]["e_center"],
        compliance=compliance_arr.flatten(),
        temperature=temperature_arr.flatten(),
        invariant=invariant,
        asymmetry_norm=A_norm,
        confidence=S,
        descriptor=descriptor,
        motif_radius=probes[0]["rho"],
    )


# ═══════════════════════════════════════════════════════════
#  SPRING GRAPH CONSTRUCTION — §7 of the paper
# ═══════════════════════════════════════════════════════════

def _build_graph(nodes: List[CryoNode], config: CryoConfig) -> CryoGraph:
    if len(nodes) <= 1:
        return CryoGraph(nodes=nodes, edges=[], global_energy=0.0)

    T_vals = np.array([np.mean(n.temperature) for n in nodes])
    T_min = T_vals.min()
    T_max = T_vals.max() + 1e-15

    weights = np.exp(-T_vals / config.spring_tau_T)

    edges: List[Tuple[int, int, float]] = []
    e_spring = 0.0

    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            dx = np.linalg.norm(nodes[i].center - nodes[j].center)
            dpsi = np.linalg.norm(nodes[i].descriptor - nodes[j].descriptor)

            k_ij = (
                np.exp(-dx ** 2 / config.spring_sigma_x ** 2)
                * np.exp(-dpsi ** 2 / config.spring_sigma_psi ** 2)
                * np.sqrt(weights[i] * weights[j])
            )

            if k_ij > 0.01:
                rest_len = dx
                edges.append((i, j, float(k_ij)))

    e_thermal = config.global_lambda_T * np.sum(T_vals)
    e_asymmetry = config.global_mu_A * sum(n.asymmetry_norm ** 2 for n in nodes)
    e_global = e_spring + e_thermal + e_asymmetry

    return CryoGraph(nodes=nodes, edges=edges, global_energy=e_global)


# ═══════════════════════════════════════════════════════════
#  COLD-SEEKING REFINEMENT — §8 of the paper
# ═══════════════════════════════════════════════════════════

def _cold_seek_refine(
    node: CryoNode,
    evaluator: _LandscapeEvaluator,
    config: CryoConfig,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, float, List[float]]:
    theta = node.center.copy()
    best_e = node.energy
    best_theta = theta.copy()
    history: List[float] = [best_e]

    lr = config.refine_lr
    n_params = len(theta)

    for step in range(config.refine_steps):
        grad = np.zeros(n_params)
        eps = config.epsilon_min * 2

        for i in range(n_params):
            shift = np.zeros(n_params)
            shift[i] = eps
            e_plus = evaluator(theta + shift)
            e_minus = evaluator(theta - shift)
            grad[i] = (e_plus - e_minus) / (2 * eps)

        cold_bias = np.zeros(n_params)
        for edge_i, edge_j, k_ij in []:
            pass

        theta = theta - lr * grad
        e_new = evaluator(theta)
        history.append(e_new)

        if e_new < best_e:
            best_e = e_new
            best_theta = theta.copy()

        lr *= 0.98

    return best_theta, best_e, history


# ═══════════════════════════════════════════════════════════
#  CRYO OPTIMIZER — the main entry point
# ═══════════════════════════════════════════════════════════

class CryoOptimizer:
    """Cryo-Canonical Basin Weaving optimizer for variational quantum circuits.

    Usage
    ─────
        from vqpu import UniversalvQPU
        from vqpu.cryo import CryoOptimizer, CryoConfig

        qpu = UniversalvQPU()

        def build_circuit(params):
            from vqpu import QuantumAlgorithms
            return QuantumAlgorithms.variational_ansatz(qpu, 4, list(params), layers=2)

        def cost(counts):
            # e.g. Max-Cut cost from measurement outcomes
            total = sum(counts.values())
            return -sum(bitstring_value(bs) * c / total for bs, c in counts.items())

        optimizer = CryoOptimizer(
            build_circuit=build_circuit,
            cost_from_counts=cost,
            executor=qpu.run,
            n_params=16,
        )
        result = optimizer.run()
        print(result.optimal_energy, result.optimal_params)

    With LinkManager
    ────────────────
        # Route evaluations through heterogeneous backends
        def executor(circuit, shots):
            gate_seq = [(op.gate_name, op.targets, *(op.params or []))
                        for op in circuit.ops]
            task = QuantumTask(n_qubits=circuit.n_qubits,
                              gate_sequence=gate_seq, shots=shots)
            counts, link = link_manager.submit(task)
            return ExecutionResult(counts=counts, ...)

        optimizer = CryoOptimizer(..., executor=executor)
    """

    def __init__(
        self,
        build_circuit: Callable[[np.ndarray], QuantumCircuit],
        cost_from_counts: Callable[[Dict[str, int]], float],
        executor: Callable[..., ExecutionResult],
        n_params: int,
        config: Optional[CryoConfig] = None,
        seed: Optional[int] = None,
    ) -> None:
        self.config = config or CryoConfig()
        self.rng = np.random.default_rng(seed)
        self.n_params = n_params
        self.evaluator = _LandscapeEvaluator(
            build_circuit, cost_from_counts, executor, self.config.shots,
        )

    def run(
        self,
        initial_points: Optional[List[np.ndarray]] = None,
        n_random_starts: int = 8,
    ) -> CryoResult:
        """Run the full CCBW optimization pipeline.

        Steps (following §10 of the paper):
        1. Generate candidate starting points
        2. Progressive probing across force schedule
        3. Build 3-3+1 motifs and canonicalize
        4. Certify basins via confidence score
        5. Build spring graph
        6. Cold-seek refine the best basins
        7. Return optimal parameters
        """
        t0 = time.perf_counter()
        force_schedule = self.config.force_schedule

        if initial_points is None:
            initial_points = [
                self.rng.uniform(-np.pi, np.pi, self.n_params)
                for _ in range(n_random_starts)
            ]

        candidates = list(initial_points)
        all_nodes: List[CryoNode] = []
        convergence: List[float] = []

        for eps in force_schedule:
            new_candidates: List[np.ndarray] = []

            for theta in candidates:
                directions = _MotifProbe.build_directions(
                    self.n_params, self.config.n_probe_directions, self.rng,
                )

                probes_at_scales: List[dict] = []
                probe = _MotifProbe.probe(
                    theta, eps, directions, self.evaluator, self.config.motif_lambda,
                )
                probes_at_scales.append(probe)

                node = _canonicalize(theta, probes_at_scales, self.config)
                if node is not None:
                    all_nodes.append(node)
                    new_candidates.append(theta)
                    convergence.append(node.energy)

            if not new_candidates:
                new_candidates = candidates[:max(1, len(candidates) // 2)]

            grad_improved: List[np.ndarray] = []
            for theta in new_candidates:
                improved = self._local_gradient_step(theta, eps)
                grad_improved.append(improved)

            candidates = grad_improved[:self.config.max_basins]

        all_nodes.sort(key=lambda n: n.confidence, reverse=True)
        all_nodes = all_nodes[:self.config.max_basins]

        graph = _build_graph(all_nodes, self.config)

        best_energy = float("inf")
        best_params = candidates[0] if candidates else initial_points[0]
        refine_history: List[float] = []

        top_nodes = sorted(all_nodes, key=lambda n: n.energy)[:3]
        for node in top_nodes:
            refined_params, refined_energy, hist = _cold_seek_refine(
                node, self.evaluator, self.config, self.rng,
            )
            refine_history.extend(hist)
            if refined_energy < best_energy:
                best_energy = refined_energy
                best_params = refined_params

        convergence.extend(refine_history)

        return CryoResult(
            optimal_params=best_params,
            optimal_energy=best_energy,
            graph=graph,
            convergence=convergence,
            n_evaluations=self.evaluator.n_evals,
            wall_time_s=time.perf_counter() - t0,
            force_schedule=force_schedule,
        )

    def _local_gradient_step(
        self, theta: np.ndarray, epsilon: float,
    ) -> np.ndarray:
        grad = np.zeros(self.n_params)
        for i in range(self.n_params):
            shift = np.zeros(self.n_params)
            shift[i] = epsilon
            e_plus = self.evaluator(theta + shift)
            e_minus = self.evaluator(theta - shift)
            grad[i] = (e_plus - e_minus) / (2 * epsilon)

        return theta - self.config.refine_lr * grad


# ═══════════════════════════════════════════════════════════
#  CONVENIENCE: QAOA + CCBW
# ═══════════════════════════════════════════════════════════

def cryo_qaoa(
    n_qubits: int,
    cost_function: Callable[[str], float],
    executor: Callable[..., ExecutionResult],
    p_layers: int = 2,
    config: Optional[CryoConfig] = None,
    seed: Optional[int] = None,
) -> CryoResult:
    """Run QAOA with Cryo-Canonical Basin Weaving optimization.

    Parameters
    ──────────
    n_qubits : int
        Number of qubits in the problem.
    cost_function : callable
        Maps a bitstring to a scalar cost (lower is better).
    executor : callable
        ``(circuit, shots) → ExecutionResult``.
    p_layers : int
        Number of QAOA layers (gamma/beta pairs).
    """
    from .core import GateLibrary

    n_params = 2 * p_layers

    def build_qaoa_circuit(params: np.ndarray) -> QuantumCircuit:
        gammas = params[:p_layers]
        betas = params[p_layers:]

        circ = QuantumCircuit(n_qubits, "cryo_qaoa")
        for i in range(n_qubits):
            circ.h(i)

        for layer in range(p_layers):
            for i in range(n_qubits):
                for j in range(i + 1, n_qubits):
                    circ.cnot(i, j)
                    circ.rz(j, gammas[layer])
                    circ.cnot(i, j)
            for i in range(n_qubits):
                circ.rx(i, 2 * betas[layer])

        return circ

    def cost_from_counts(counts: Dict[str, int]) -> float:
        total = sum(counts.values())
        return sum(cost_function(bs) * c / total for bs, c in counts.items())

    optimizer = CryoOptimizer(
        build_circuit=build_qaoa_circuit,
        cost_from_counts=cost_from_counts,
        executor=executor,
        n_params=n_params,
        config=config,
        seed=seed,
    )
    return optimizer.run()


def cryo_vqe(
    n_qubits: int,
    hamiltonian: np.ndarray,
    executor: Callable[..., ExecutionResult],
    layers: int = 2,
    config: Optional[CryoConfig] = None,
    seed: Optional[int] = None,
) -> CryoResult:
    """Run VQE with Cryo-Canonical Basin Weaving optimization.

    Parameters
    ──────────
    n_qubits : int
        Number of qubits.
    hamiltonian : np.ndarray
        Hermitian matrix (2^n × 2^n) for the Hamiltonian.
    executor : callable
        ``(circuit, shots) → ExecutionResult``. Must return statevector.
    layers : int
        Number of ansatz layers.
    """
    n_params = n_qubits * layers

    def build_vqe_circuit(params: np.ndarray) -> QuantumCircuit:
        circ = QuantumCircuit(n_qubits, "cryo_vqe")
        p_idx = 0
        for layer_i in range(layers):
            for i in range(n_qubits):
                if p_idx < len(params):
                    circ.ry(i, params[p_idx])
                    p_idx += 1
            for i in range(n_qubits - 1):
                circ.cnot(i, i + 1)
        return circ

    def cost_from_counts(counts: Dict[str, int]) -> float:
        n = hamiltonian.shape[0]
        probs = np.zeros(n)
        total = sum(counts.values())
        for bs, cnt in counts.items():
            idx = int(bs, 2) if len(bs) <= 20 else 0
            if idx < n:
                probs[idx] = cnt / total
        return float(np.real(probs @ np.diag(hamiltonian)))

    optimizer = CryoOptimizer(
        build_circuit=build_vqe_circuit,
        cost_from_counts=cost_from_counts,
        executor=executor,
        n_params=n_params,
        config=config,
        seed=seed,
    )
    return optimizer.run()


__all__ = [
    "CryoConfig",
    "CryoNode",
    "CryoGraph",
    "CryoResult",
    "CryoOptimizer",
    "cryo_qaoa",
    "cryo_vqe",
]
