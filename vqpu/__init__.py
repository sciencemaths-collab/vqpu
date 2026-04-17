"""vQPU — Universal Virtual Quantum Processing Unit.

Auto-discovers CPU, GPU (NVIDIA/AMD/Intel/Apple), TPU, and QPU backends
on the host and executes circuits on the best-suited one. No silent CPU
fallback: if a GPU/TPU/QPU plugin claims the work, it runs on that device
or raises loudly.

Typical usage:

    from vqpu import UniversalvQPU

    qpu = UniversalvQPU()           # probes every backend on boot
    c = qpu.circuit(3, "ghz")
    c.h(0).cnot(0, 1).cnot(1, 2)
    result = qpu.run(c, shots=1024)
    print(result.counts)
"""

from .core import (
    QubitState,
    QuantumRegister,
    GateLibrary,
    GateEngine,
    MeasurementTap,
    GateOp,
    SymmetryDescriptor,
    SymmetryFilter,
    QuantumCircuit,
    Backend,
    ClassicalSimulatorBackend,
    QPUBackendStub,
    ExecutionResult,
    vQPU,
    QuantumAlgorithms,
)

from .universal import (
    UniversalvQPU,
    BackendPlugin,
    BackendFingerprint,
    ComputeClass,
    EntanglementEdge,
    EntanglementComponent,
    EntanglementScanResult,
    EntanglementScanner,
    TaskPhase,
    TaskSegment,
    TaskDecomposer,
    HybridRouter,
    CPUPlugin,
    NvidiaGPUPlugin,
    AMDGPUPlugin,
    IntelGPUPlugin,
    AppleSiliconPlugin,
    TPUPlugin,
    QPUCloudPlugin,
)

from .phantom import (
    PhantomPruningConfig,
    PhantomSubsystemPlan,
    PhantomBridgeTransfer,
    PhantomPartition,
    PhantomSimulatorBackend,
    build_phantom_partition,
)

from .knit import (
    WireCut,
    CutPlan,
    CircuitFragment,
    CutFinder,
    FragmentBuilder,
    CircuitKnitter,
    KnitResult,
)

from .cryo import (
    CryoConfig,
    CryoNode,
    CryoGraph,
    CryoResult,
    CryoOptimizer,
    cryo_qaoa,
    cryo_vqe,
)

from . import chesso

__version__ = "0.4.3"

__all__ = [
    # core quantum primitives
    "QubitState", "QuantumRegister", "GateLibrary", "GateEngine",
    "MeasurementTap", "GateOp", "SymmetryDescriptor", "SymmetryFilter",
    "QuantumCircuit", "Backend",
    "ClassicalSimulatorBackend", "QPUBackendStub", "ExecutionResult",
    "vQPU", "QuantumAlgorithms",
    # universal discovery + routing
    "UniversalvQPU", "BackendPlugin", "BackendFingerprint", "ComputeClass",
    "EntanglementEdge", "EntanglementComponent", "EntanglementScanResult",
    "EntanglementScanner",
    "TaskPhase", "TaskSegment", "TaskDecomposer", "HybridRouter",
    # phantom execution + partitioning
    "PhantomPruningConfig", "PhantomSubsystemPlan", "PhantomBridgeTransfer",
    "PhantomPartition", "PhantomSimulatorBackend", "build_phantom_partition",
    # plugins
    "CPUPlugin", "NvidiaGPUPlugin", "AMDGPUPlugin", "IntelGPUPlugin",
    "AppleSiliconPlugin", "TPUPlugin", "QPUCloudPlugin",
    # circuit knitting
    "WireCut", "CutPlan", "CircuitFragment",
    "CutFinder", "FragmentBuilder", "CircuitKnitter", "KnitResult",
    # cryo-canonical basin weaving optimizer
    "CryoConfig", "CryoNode", "CryoGraph", "CryoResult",
    "CryoOptimizer", "cryo_qaoa", "cryo_vqe",
    # CHESSO control stack
    "chesso",
]
