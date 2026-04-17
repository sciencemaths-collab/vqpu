from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, Sequence, Tuple

import numpy as np

from compiler import parse_qlambda_script
from compiler.ir import QLambdaProgram
from core import EntanglementHypergraph, HilbertBundleState, QuantumState, RuntimeConfig
from core.types import StateRepresentation
from ops import apply_local_operator, hadamard


TargetLike = HilbertBundleState | np.ndarray | Sequence[complex] | None


class WorkloadExecutionKind(str, Enum):
    """Primary execution path for a benchmark workload."""

    COMPILER = "compiler"
    RUNTIME = "runtime"
    SCHEDULED = "scheduled"
    PREPARED = "prepared"


@dataclass(slots=True)
class MaterializedWorkload:
    """Freshly instantiated workload payload for one benchmark run."""

    name: str
    description: str
    execution_kind: WorkloadExecutionKind
    config: RuntimeConfig
    bundle: HilbertBundleState | None = None
    graph: EntanglementHypergraph | None = None
    target: TargetLike = None
    program: QLambdaProgram | None = None
    steps: int = 1
    max_entanglers: int = 2
    tags: Tuple[str, ...] = ()
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class WorkloadSpec:
    """Reusable workload recipe that can materialize fresh benchmark inputs."""

    name: str
    description: str
    execution_kind: WorkloadExecutionKind
    config_factory: Callable[[], RuntimeConfig]
    bundle_builder: Callable[[RuntimeConfig], HilbertBundleState] | None = None
    graph_builder: Callable[[HilbertBundleState], EntanglementHypergraph | None] | None = None
    target_builder: Callable[[RuntimeConfig], TargetLike] | None = None
    program_builder: Callable[[RuntimeConfig], QLambdaProgram] | None = None
    steps: int = 1
    max_entanglers: int = 2
    tags: Tuple[str, ...] = ()
    metadata: Dict[str, Any] = field(default_factory=dict)

    def materialize(self) -> MaterializedWorkload:
        cfg = self.config_factory()
        bundle = None if self.bundle_builder is None else self.bundle_builder(cfg)
        graph = None
        if bundle is not None and self.graph_builder is not None:
            graph = self.graph_builder(bundle)
        target = None if self.target_builder is None else self.target_builder(cfg)
        program = None if self.program_builder is None else self.program_builder(cfg)
        return MaterializedWorkload(
            name=self.name,
            description=self.description,
            execution_kind=self.execution_kind,
            config=cfg,
            bundle=bundle,
            graph=graph,
            target=target,
            program=program,
            steps=int(self.steps),
            max_entanglers=int(self.max_entanglers),
            tags=tuple(self.tags),
            metadata=dict(self.metadata),
        )


def _statevector_config(*, seed: int, max_active_qubits: int = 6) -> RuntimeConfig:
    cfg = RuntimeConfig.for_statevector(max_active_qubits=max_active_qubits, seed=seed)
    cfg.budget.max_steps = 4
    return cfg


def _density_config(*, seed: int, max_active_qubits: int = 6) -> RuntimeConfig:
    cfg = RuntimeConfig.for_density_matrix(max_active_qubits=max_active_qubits, seed=seed)
    cfg.budget.max_steps = 4
    return cfg


def _bell_program(_: RuntimeConfig) -> QLambdaProgram:
    return parse_qlambda_script(
        """
        program bell_workload
        alloc q0
        alloc q1
        gate H q0
        gate CX q0 q1
        """
    )


def _ghz_program_factory(num_qubits: int) -> Callable[[RuntimeConfig], QLambdaProgram]:
    def _builder(_: RuntimeConfig) -> QLambdaProgram:
        lines = ["program ghz_workload"]
        for i in range(num_qubits):
            lines.append(f"alloc q{i}")
        lines.append("gate H q0")
        for i in range(1, num_qubits):
            lines.append(f"gate CX q0 q{i}")
        return parse_qlambda_script("\n".join(lines))

    return _builder


def _bell_target(_: RuntimeConfig) -> np.ndarray:
    return np.array([1 / np.sqrt(2), 0.0, 0.0, 1 / np.sqrt(2)], dtype=np.complex128)


def _ghz_target_factory(num_qubits: int) -> Callable[[RuntimeConfig], np.ndarray]:
    def _builder(_: RuntimeConfig) -> np.ndarray:
        dim = 2**num_qubits
        vec = np.zeros(dim, dtype=np.complex128)
        vec[0] = 1 / np.sqrt(2)
        vec[-1] = 1 / np.sqrt(2)
        return vec

    return _builder


def _w_bundle(cfg: RuntimeConfig, *, num_qubits: int) -> HilbertBundleState:
    bundle = HilbertBundleState.initialize(sector_dims=[2] * num_qubits, config=cfg)
    vec = np.zeros(2**num_qubits, dtype=np.complex128)
    amp = 1.0 / np.sqrt(num_qubits)
    for i in range(num_qubits):
        basis_index = 1 << (num_qubits - 1 - i)
        vec[basis_index] = amp
    bundle.quantum_state = QuantumState(vec, StateRepresentation.STATEVECTOR, bundle.dims, cfg.numerical_tolerance)
    bundle.metadata["prepared_state"] = "W"
    return bundle


def _w_target_factory(num_qubits: int) -> Callable[[RuntimeConfig], np.ndarray]:
    def _builder(_: RuntimeConfig) -> np.ndarray:
        vec = np.zeros(2**num_qubits, dtype=np.complex128)
        amp = 1.0 / np.sqrt(num_qubits)
        for i in range(num_qubits):
            basis_index = 1 << (num_qubits - 1 - i)
            vec[basis_index] = amp
        return vec

    return _builder


def _plus_plus_bundle(cfg: RuntimeConfig) -> HilbertBundleState:
    bundle = HilbertBundleState.initialize(sector_dims=[2, 2], config=cfg)
    apply_local_operator(bundle, hadamard(), [0])
    apply_local_operator(bundle, hadamard(), [1])
    return bundle


def _zero_target(cfg: RuntimeConfig, *, qubits: int = 2) -> HilbertBundleState:
    return HilbertBundleState.initialize(sector_dims=[2] * qubits, config=cfg)


def _noise_graph(bundle: HilbertBundleState) -> EntanglementHypergraph:
    graph = EntanglementHypergraph.from_topology(bundle.topology, max_order=3)
    edge = graph.add_hyperedge(["q0", "q1"], weight=1.15, phase_bias=0.45, coherence_score=0.85, capacity=1.0)
    graph.add_route("q0", "q1", [str(edge.edge_id)], route_id="r_noise", score=0.55, bandwidth=1.2, latency=0.15)
    return graph


def _noise_config_factory() -> RuntimeConfig:
    cfg = _density_config(seed=23, max_active_qubits=6)
    cfg.noise.depolarizing_rate = 0.18
    cfg.noise.phase_damping = 0.12
    cfg.noise.amplitude_damping = 0.08
    cfg.noise.readout_error = 0.03
    cfg.default_measurement_strength = 0.18
    cfg.budget.max_steps = 3
    return cfg


def _expand_compress_config_factory() -> RuntimeConfig:
    cfg = _statevector_config(seed=31, max_active_qubits=6)
    cfg.noise.depolarizing_rate = 0.05
    cfg.noise.phase_damping = 0.04
    cfg.default_measurement_strength = 0.12
    cfg.budget.max_steps = 2
    cfg.budget.max_branches = 2
    cfg.budget.max_prune_loss = 0.6
    return cfg


def _expand_compress_bundle(cfg: RuntimeConfig) -> HilbertBundleState:
    bundle = _plus_plus_bundle(cfg)
    bundle.metadata["expansion_probe"] = True
    return bundle


def _expand_compress_graph(bundle: HilbertBundleState) -> EntanglementHypergraph:
    graph = EntanglementHypergraph.from_topology(bundle.topology, max_order=4)
    edge = graph.add_hyperedge(["q0", "q1"], weight=1.05, phase_bias=0.3, coherence_score=0.9, capacity=1.1)
    graph.add_route("q0", "q1", [str(edge.edge_id)], route_id="r_expand", score=0.45, bandwidth=1.0, latency=0.05)
    return graph


def make_bell_workload() -> WorkloadSpec:
    return WorkloadSpec(
        name="bell_compiler",
        description="Compile and execute Bell-state preparation through the Qλ IR.",
        execution_kind=WorkloadExecutionKind.COMPILER,
        config_factory=lambda: _statevector_config(seed=11, max_active_qubits=4),
        program_builder=_bell_program,
        target_builder=_bell_target,
        steps=1,
        tags=("bell", "compiler", "baseline"),
    )


def make_ghz_workload(num_qubits: int = 3) -> WorkloadSpec:
    return WorkloadSpec(
        name=f"ghz_{num_qubits}q_compiler",
        description=f"Compile and execute a {num_qubits}-qubit GHZ workload.",
        execution_kind=WorkloadExecutionKind.COMPILER,
        config_factory=lambda: _statevector_config(seed=13, max_active_qubits=max(6, num_qubits + 1)),
        program_builder=_ghz_program_factory(num_qubits),
        target_builder=_ghz_target_factory(num_qubits),
        steps=1,
        tags=("ghz", "compiler", f"{num_qubits}q"),
    )


def make_w_state_workload(num_qubits: int = 3) -> WorkloadSpec:
    return WorkloadSpec(
        name=f"w_{num_qubits}q_prepared",
        description=f"Prepared {num_qubits}-qubit W-state workload for metrics and stability baselines.",
        execution_kind=WorkloadExecutionKind.PREPARED,
        config_factory=lambda: _statevector_config(seed=17, max_active_qubits=max(6, num_qubits + 1)),
        bundle_builder=lambda cfg: _w_bundle(cfg, num_qubits=num_qubits),
        target_builder=_w_target_factory(num_qubits),
        steps=0,
        tags=("wstate", "prepared", f"{num_qubits}q"),
    )


def make_noise_stress_workload() -> WorkloadSpec:
    return WorkloadSpec(
        name="noise_stress_runtime",
        description="Noisy two-qubit workload for runtime-vs-scheduled control comparisons.",
        execution_kind=WorkloadExecutionKind.RUNTIME,
        config_factory=_noise_config_factory,
        bundle_builder=_plus_plus_bundle,
        graph_builder=_noise_graph,
        target_builder=lambda cfg: _zero_target(cfg, qubits=2),
        steps=2,
        max_entanglers=1,
        tags=("noise", "runtime", "comparison"),
    )


def make_expand_compress_workload() -> WorkloadSpec:
    return WorkloadSpec(
        name="expand_compress_runtime",
        description="Branch-pressure workload that exercises adaptive expansion and compression.",
        execution_kind=WorkloadExecutionKind.RUNTIME,
        config_factory=_expand_compress_config_factory,
        bundle_builder=_expand_compress_bundle,
        graph_builder=_expand_compress_graph,
        target_builder=lambda cfg: _zero_target(cfg, qubits=2),
        steps=2,
        max_entanglers=1,
        tags=("expansion", "compression", "runtime"),
    )


def standard_workloads() -> Tuple[WorkloadSpec, ...]:
    return (
        make_bell_workload(),
        make_ghz_workload(3),
        make_w_state_workload(3),
        make_noise_stress_workload(),
        make_expand_compress_workload(),
    )


__all__ = [
    "MaterializedWorkload",
    "WorkloadExecutionKind",
    "WorkloadSpec",
    "make_bell_workload",
    "make_ghz_workload",
    "make_w_state_workload",
    "make_noise_stress_workload",
    "make_expand_compress_workload",
    "standard_workloads",
]
