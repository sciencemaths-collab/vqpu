from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional


class SimulationMode(str, Enum):
    """Supported numerical state representations."""

    STATEVECTOR = "statevector"
    DENSITY_MATRIX = "density_matrix"


class BackendKind(str, Enum):
    """High-level backend families for the vQPU runtime."""

    NUMPY = "numpy"
    SIMULATOR = "simulator"
    HYBRID = "hybrid"
    HARDWARE = "hardware"


@dataclass(slots=True)
class NoiseConfig:
    """Simple noise knobs for the earliest simulator prototype.

    These are intentionally lightweight. Later sections can replace or extend
    them with backend-calibrated models and Lindblad or trajectory parameters.
    """

    depolarizing_rate: float = 0.0
    amplitude_damping: float = 0.0
    phase_damping: float = 0.0
    readout_error: float = 0.0
    coherent_overrotation: float = 0.0

    def validate(self) -> None:
        for name in ("depolarizing_rate", "amplitude_damping", "phase_damping", "readout_error", "coherent_overrotation"):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1], got {value!r}")


@dataclass(slots=True)
class ResourceBudget:
    """Hard limits for the runtime."""

    max_active_qubits: int = 32
    max_hyperedge_order: int = 4
    max_dynamic_depth: int = 256
    max_measurements: int = 256
    max_branches: int = 128
    max_steps: int = 100
    max_prune_loss: float = 1e-2

    def validate(self) -> None:
        if self.max_active_qubits <= 0:
            raise ValueError("max_active_qubits must be positive")
        if self.max_hyperedge_order < 2:
            raise ValueError("max_hyperedge_order must be at least 2")
        if self.max_dynamic_depth <= 0:
            raise ValueError("max_dynamic_depth must be positive")
        if self.max_measurements < 0 or self.max_branches < 0:
            raise ValueError("max_measurements and max_branches must be non-negative")
        if self.max_steps <= 0:
            raise ValueError("max_steps must be positive")
        if not 0.0 <= self.max_prune_loss <= 1.0:
            raise ValueError("max_prune_loss must be in [0, 1]")


@dataclass(slots=True)
class RuntimeConfig:
    """Top-level configuration for CHESSO runtime state and execution."""

    simulation_mode: SimulationMode = SimulationMode.DENSITY_MATRIX
    backend_kind: BackendKind = BackendKind.NUMPY
    random_seed: int = 7
    dt: float = 0.05
    numerical_tolerance: float = 1e-10
    default_measurement_strength: float = 0.1
    noise: NoiseConfig = field(default_factory=NoiseConfig)
    budget: ResourceBudget = field(default_factory=ResourceBudget)
    backend_options: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if self.dt <= 0.0:
            raise ValueError("dt must be positive")
        if self.numerical_tolerance <= 0.0:
            raise ValueError("numerical_tolerance must be positive")
        if not 0.0 <= self.default_measurement_strength <= 1.0:
            raise ValueError("default_measurement_strength must be in [0, 1]")
        self.noise.validate()
        self.budget.validate()

    @classmethod
    def for_density_matrix(cls, *, max_active_qubits: int = 16, seed: int = 7) -> "RuntimeConfig":
        cfg = cls(
            simulation_mode=SimulationMode.DENSITY_MATRIX,
            backend_kind=BackendKind.NUMPY,
            random_seed=seed,
        )
        cfg.budget.max_active_qubits = max_active_qubits
        cfg.validate()
        return cfg

    @classmethod
    def for_statevector(cls, *, max_active_qubits: int = 20, seed: int = 7) -> "RuntimeConfig":
        cfg = cls(
            simulation_mode=SimulationMode.STATEVECTOR,
            backend_kind=BackendKind.NUMPY,
            random_seed=seed,
        )
        cfg.budget.max_active_qubits = max_active_qubits
        cfg.validate()
        return cfg


DEFAULT_CONFIG = RuntimeConfig()
DEFAULT_CONFIG.validate()
