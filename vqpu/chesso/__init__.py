"""CHESSO vQPU integrated subpackage.

CHESSO (Coherent Hypergraph Entanglement and Superposition Steering Optimizer)
is the simulator-first control stack that rides on top of the vQPU core.
The subsystems are re-exported here so callers can do:

    from vqpu.chesso import RuntimeConfig, run_chesso_runtime
"""

from . import compiler, control, core, experiments, ops, viz
from . import bridge
from .bridge import (
    BridgedCircuit,
    compile_qlambda_for_hardware,
    execute_qlambda_on_backend,
    plan_to_gate_sequence,
)

from .core import (
    BackendKind,
    BundleTopology,
    ClassicalMemory,
    EntanglementCycle,
    EntanglementHyperedge,
    EntanglementHypergraph,
    EntanglementRoute,
    EntanglementVertex,
    HilbertBundleState,
    HyperedgeId,
    HypergraphSummary,
    MeasurementRecord,
    NoiseConfig,
    QuantumState,
    ResourceBudget,
    RuntimeConfig,
    RuntimeStats,
    SectorId,
    SectorKind,
    SectorSpec,
    SimulationMode,
    StateBundleSnapshot,
    StateRepresentation,
)

__all__ = [
    "compiler",
    "control",
    "core",
    "experiments",
    "ops",
    "viz",
    "bridge",
    "BridgedCircuit",
    "compile_qlambda_for_hardware",
    "execute_qlambda_on_backend",
    "plan_to_gate_sequence",
    "BackendKind",
    "BundleTopology",
    "ClassicalMemory",
    "EntanglementCycle",
    "EntanglementHyperedge",
    "EntanglementHypergraph",
    "EntanglementRoute",
    "EntanglementVertex",
    "HilbertBundleState",
    "HyperedgeId",
    "HypergraphSummary",
    "MeasurementRecord",
    "NoiseConfig",
    "QuantumState",
    "ResourceBudget",
    "RuntimeConfig",
    "RuntimeStats",
    "SectorId",
    "SectorKind",
    "SectorSpec",
    "SimulationMode",
    "StateBundleSnapshot",
    "StateRepresentation",
]
