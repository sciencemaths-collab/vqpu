"""
Universal Backend Discovery & Hybrid Execution Engine
=====================================================
A truly adaptive system that:

1. DISCOVERS any backend — scans for CPU, GPU (any vendor),
   QPU (any provider), FPGA, TPU, cloud instances, anything
2. PROBES each backend's actual capabilities at runtime
3. SPLITS single circuits across mixed backends intelligently
4. ROUTES sub-tasks to the backend best suited for each part
5. PLUGINS — new backends register themselves, zero core changes

The key insight: quantum algorithms have classical and quantum
phases. The hybrid engine assigns each phase to its optimal
backend, even mixing GPU + QPU in a single execution.
"""

import sys, os, time, json, subprocess, importlib, shutil
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Callable, Any, Tuple
from abc import ABC, abstractmethod
from enum import Enum
from itertools import combinations
import numpy as np

from .core import (QuantumCircuit, QuantumRegister, GateEngine,
                   MeasurementTap, GateLibrary, ExecutionResult,
                   SymmetryFilter,
                   QuantumAlgorithms)


# ═══════════════════════════════════════════════════════════
#  UNIVERSAL CAPABILITY MODEL
#  What can a backend actually do?
# ═══════════════════════════════════════════════════════════

class ComputeClass(Enum):
    CLASSICAL_CPU = "cpu"
    CLASSICAL_GPU = "gpu"
    QUANTUM_SIMULATOR = "qsim"
    QUANTUM_HARDWARE = "qpu"
    FPGA = "fpga"
    TPU = "tpu"
    CLOUD = "cloud"
    UNKNOWN = "unknown"


@dataclass
class BackendFingerprint:
    """Universal description of what a backend can do"""
    name: str
    compute_class: ComputeClass
    vendor: str
    
    # Capacity
    max_qubits: int               # 0 for pure classical
    max_classical_bits: int       # Memory in terms of state vector
    memory_bytes: int
    
    # Speed characteristics
    gate_time_ns: float           # Time per gate operation
    measurement_time_ns: float
    classical_flops: float        # For classical pre/post processing
    
    # Capabilities
    supports_statevector: bool    # Can return full amplitude vector?
    supports_density_matrix: bool # Mixed state simulation?
    supports_noise_model: bool    # Can simulate decoherence?
    native_gates: List[str]       # Which gates run without decomposition?
    connectivity: str             # "all-to-all", "linear", "grid", "heavy-hex"
    
    # Availability
    is_available: bool
    is_local: bool                # vs cloud/remote
    latency_ms: float             # Network latency for remote backends
    cost_per_shot: float          # $ per measurement shot (0 for local)
    
    # For hybrid routing
    best_for: List[str]           # ["state_prep", "unitary_evolution", "measurement", 
                                  #  "classical_opt", "matrix_multiply", "sampling"]

    def __repr__(self):
        avail = "ONLINE" if self.is_available else "OFFLINE"
        return f"{self.name} [{self.compute_class.value}] {avail} ({self.vendor}, {self.max_qubits}qb)"


# ═══════════════════════════════════════════════════════════
#  BACKEND PLUGIN INTERFACE
#  Any backend implements this — CPU, GPU, QPU, anything
# ═══════════════════════════════════════════════════════════

class BackendPlugin(ABC):
    """
    Universal backend plugin interface.
    Implement this to add ANY new compute backend.
    """

    @abstractmethod
    def probe(self) -> Optional[BackendFingerprint]:
        """
        Detect if this backend is available RIGHT NOW.
        Returns fingerprint if available, None if not.
        Called at startup AND periodically to detect hot-plugged hardware.
        """
        pass

    @abstractmethod
    def execute_statevector(self, n_qubits: int, gate_sequence: list,
                           initial_state: Optional[np.ndarray] = None) -> np.ndarray:
        """Execute gates and return final state vector"""
        pass

    @abstractmethod
    def execute_sample(self, n_qubits: int, gate_sequence: list,
                      shots: int, initial_state: Optional[np.ndarray] = None) -> dict:
        """Execute gates and return measurement samples"""
        pass

    def execute_expectation(self, n_qubits: int, gate_sequence: list,
                           observable: np.ndarray,
                           initial_state: Optional[np.ndarray] = None) -> float:
        """Execute and return expectation value (default: via statevector)"""
        sv = self.execute_statevector(n_qubits, gate_sequence, initial_state)
        return float(np.real(sv.conj() @ observable @ sv))

    def benchmark(self, n_qubits: int = 8) -> dict:
        """Run a quick benchmark to measure actual performance"""
        # Default: time a simple circuit
        gates = [("H", [0])] + [("CNOT", [0, i]) for i in range(1, n_qubits)]
        t0 = time.time()
        try:
            self.execute_statevector(n_qubits, gates)
            dt = time.time() - t0
            return {"success": True, "time_s": dt, "n_qubits": n_qubits}
        except Exception as e:
            return {"success": False, "error": str(e)}


# ═══════════════════════════════════════════════════════════
#  DEVICE-NATIVE STATEVECTOR KERNEL
#  A tensor-contraction algorithm that runs on ANY array library
#  (numpy / cupy / torch / jax / mlx). Each GPU/TPU plugin uses
#  this with its own library — no CPU fallback anywhere.
# ═══════════════════════════════════════════════════════════

def _build_gate_matrix(name: str, params: list):
    """Construct the numpy matrix for a named gate (small, stays on host)."""
    if name == "H":
        return (1 / np.sqrt(2)) * np.array([[1, 1], [1, -1]], dtype=complex)
    if name == "X":
        return np.array([[0, 1], [1, 0]], dtype=complex)
    if name == "Y":
        return np.array([[0, -1j], [1j, 0]], dtype=complex)
    if name == "Z":
        return np.array([[1, 0], [0, -1]], dtype=complex)
    if name == "S":
        return np.array([[1, 0], [0, 1j]], dtype=complex)
    if name == "T":
        return np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=complex)
    if name == "CNOT":
        return np.array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]], dtype=complex)
    if name == "CZ":
        return np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,-1]], dtype=complex)
    if name == "SWAP":
        return np.array([[1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]], dtype=complex)
    if name.startswith("Rx"):
        th = params[0] if params else 0.0
        c, s = np.cos(th/2), np.sin(th/2)
        return np.array([[c, -1j*s], [-1j*s, c]], dtype=complex)
    if name.startswith("Ry"):
        th = params[0] if params else 0.0
        c, s = np.cos(th/2), np.sin(th/2)
        return np.array([[c, -s], [s, c]], dtype=complex)
    if name.startswith("Rz"):
        th = params[0] if params else 0.0
        return np.array([[np.exp(-1j*th/2), 0], [0, np.exp(1j*th/2)]], dtype=complex)
    if name == "Phase":
        th = params[0] if params else 0.0
        return np.array([[1, 0], [0, np.exp(1j*th)]], dtype=complex)
    if name == "FULL_UNITARY":
        return params[0] if params else None
    return None


def _normalize_gates(gate_sequence):
    """Yield (numpy_matrix, targets_list) for each executable gate."""
    for gate in gate_sequence:
        if isinstance(gate, dict):
            name = gate["name"]
            targets = gate["targets"]
            params = gate.get("params", [])
        else:
            name = gate[0]
            targets = gate[1] if isinstance(gate[1], list) else [gate[1]]
            params = list(gate[2:]) if len(gate) > 2 else []
        mat = _build_gate_matrix(name, params)
        if mat is None:
            continue
        yield mat, list(targets)


def _perm_after_contract(targets: list, n: int) -> list:
    """After tensordot contracts matrix input-axes with state axes at `targets`,
    output matrix axes come first (positions 0..k-1), then remaining state axes
    in order. Compute the transpose permutation that restores qubit order."""
    k = len(targets)
    remaining = [i for i in range(n) if i not in targets]
    perm = [0] * n
    for i, t in enumerate(targets):
        perm[t] = i
    for i, r in enumerate(remaining):
        perm[r] = k + i
    return perm


def _apply_gates_xp(xp, n_qubits, gate_sequence, initial_state, dtype):
    """Run on any numpy-compatible mutable library (numpy, cupy)."""
    if initial_state is not None:
        state = xp.asarray(initial_state, dtype=dtype)
    else:
        state = xp.zeros(2**n_qubits, dtype=dtype)
        state[0] = 1.0
    state = state.reshape([2] * n_qubits)
    for matrix, targets in _normalize_gates(gate_sequence):
        k = len(targets)
        m = xp.asarray(matrix, dtype=dtype).reshape([2] * (2 * k))
        state = xp.tensordot(m, state, axes=(list(range(k, 2*k)), targets))
        state = xp.transpose(state, _perm_after_contract(targets, n_qubits))
    return state.reshape(-1)


def _apply_gates_jax(jnp, n_qubits, gate_sequence, initial_state, dtype):
    """JAX variant — immutable arrays, basis state via .at[].set()."""
    if initial_state is not None:
        state = jnp.asarray(initial_state, dtype=dtype)
    else:
        state = jnp.zeros(2**n_qubits, dtype=dtype).at[0].set(1.0)
    state = state.reshape([2] * n_qubits)
    for matrix, targets in _normalize_gates(gate_sequence):
        k = len(targets)
        m = jnp.asarray(matrix, dtype=dtype).reshape([2] * (2 * k))
        state = jnp.tensordot(m, state, axes=(list(range(k, 2*k)), targets))
        state = jnp.transpose(state, _perm_after_contract(targets, n_qubits))
    return state.reshape(-1)


def _apply_gates_torch(torch, device, n_qubits, gate_sequence, initial_state, dtype):
    """PyTorch variant — uses .permute() (torch.transpose only swaps two axes)."""
    if initial_state is not None:
        state = torch.tensor(np.asarray(initial_state), dtype=dtype, device=device)
    else:
        state = torch.zeros(2**n_qubits, dtype=dtype, device=device)
        state[0] = 1.0
    state = state.reshape([2] * n_qubits)
    for matrix, targets in _normalize_gates(gate_sequence):
        k = len(targets)
        m = torch.tensor(matrix, dtype=dtype, device=device).reshape([2] * (2 * k))
        state = torch.tensordot(m, state, dims=(list(range(k, 2*k)), targets))
        state = state.permute(*_perm_after_contract(targets, n_qubits))
    return state.reshape(-1)


def _apply_gates_mlx(mx, n_qubits, gate_sequence, initial_state):
    """MLX (Apple Silicon via Metal) variant — complex64 is the native type."""
    if initial_state is not None:
        state = mx.array(np.asarray(initial_state, dtype=np.complex64))
    else:
        init = np.zeros(2**n_qubits, dtype=np.complex64)
        init[0] = 1.0
        state = mx.array(init)
    state = state.reshape([2] * n_qubits)
    for matrix, targets in _normalize_gates(gate_sequence):
        k = len(targets)
        m = mx.array(matrix.astype(np.complex64)).reshape([2] * (2 * k))
        state = mx.tensordot(m, state, axes=(list(range(k, 2*k)), targets))
        state = mx.transpose(state, _perm_after_contract(targets, n_qubits))
    return state.reshape(-1)


def _detect_host_memory_bytes() -> int:
    """Cross-platform available-memory probe. Linux: /proc/meminfo.
    macOS/BSD: sysctl hw.memsize. Falls back to 2GB if unknown."""
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except Exception:
        pass
    try:
        r = subprocess.run(["sysctl", "-n", "hw.memsize"],
                         capture_output=True, text=True, timeout=5)
        if r.returncode == 0:
            return int(r.stdout.strip())
    except Exception:
        pass
    return 2 * 1024**3


def _sample_counts(probs_np: np.ndarray, shots: int, n_qubits: int) -> dict:
    """Draw `shots` samples from a probability vector (absorbs float32 noise)."""
    probs_np = np.maximum(probs_np, 0.0)
    s = probs_np.sum()
    if s <= 0:
        raise RuntimeError("Degenerate probability vector — all zeros.")
    probs_np = probs_np / s
    rng = np.random.default_rng()
    indices = rng.choice(2**n_qubits, size=shots, p=probs_np)
    counts = {}
    for idx in indices:
        bits = format(int(idx), f'0{n_qubits}b')
        counts[bits] = counts.get(bits, 0) + 1
    return counts


# ═══════════════════════════════════════════════════════════
#  BUILT-IN PLUGINS — Auto-detected at startup
# ═══════════════════════════════════════════════════════════

class CPUPlugin(BackendPlugin):
    """Native CPU statevector simulator — always available"""

    def probe(self) -> Optional[BackendFingerprint]:
        cores = os.cpu_count() or 1
        mem_bytes = _detect_host_memory_bytes()
        max_qb = min(28, int(np.log2(mem_bytes / (16 * 3))))

        return BackendFingerprint(
            name=f"CPU({cores}cores)",
            compute_class=ComputeClass.CLASSICAL_CPU,
            vendor="local",
            max_qubits=max_qb,
            max_classical_bits=64,
            memory_bytes=mem_bytes,
            gate_time_ns=50.0 * (2**10),  # Scales with state size
            measurement_time_ns=10.0,
            classical_flops=cores * 2e9,
            supports_statevector=True,
            supports_density_matrix=(max_qb <= 14),
            supports_noise_model=False,
            native_gates=["H", "X", "Y", "Z", "S", "T", "CNOT", "CZ",
                         "SWAP", "Rx", "Ry", "Rz", "Phase"],
            connectivity="all-to-all",
            is_available=True,
            is_local=True,
            latency_ms=0.0,
            cost_per_shot=0.0,
            best_for=["state_prep", "unitary_evolution", "measurement",
                      "classical_opt", "sampling"],
        )

    def execute_statevector(self, n_qubits, gate_sequence, initial_state=None):
        if initial_state is not None:
            amps = initial_state.copy()
        else:
            amps = np.zeros(2**n_qubits, dtype=complex)
            amps[0] = 1.0

        reg = QuantumRegister(n_qubits, amps)
        engine = GateEngine()
        lib = GateLibrary()

        gate_map = {
            "H": lib.H, "X": lib.X, "Y": lib.Y, "Z": lib.Z,
            "S": lib.S, "T": lib.T, "CNOT": lib.CNOT, "CZ": lib.CZ,
            "SWAP": lib.SWAP,
        }

        for gate_name, targets, *params in self._normalize_gates(gate_sequence):
            if gate_name in gate_map:
                mat = gate_map[gate_name]
            elif gate_name.startswith("Rx"):
                mat = lib.Rx(params[0] if params else 0)
            elif gate_name.startswith("Ry"):
                mat = lib.Ry(params[0] if params else 0)
            elif gate_name.startswith("Rz"):
                mat = lib.Rz(params[0] if params else 0)
            elif gate_name == "FULL_UNITARY":
                mat = params[0]
            else:
                continue

            if len(targets) == 1:
                engine.apply_single(reg, mat, targets[0], gate_name)
            elif len(targets) == 2 and mat.shape[0] == 4:
                engine.apply_two_qubit(reg, mat, targets[0], targets[1], gate_name)
            else:
                engine.apply_multi(reg, mat, targets, gate_name)

        return reg.amplitudes

    def execute_sample(self, n_qubits, gate_sequence, shots, initial_state=None):
        sv = self.execute_statevector(n_qubits, gate_sequence, initial_state)
        probs = np.abs(sv) ** 2
        rng = np.random.default_rng()
        indices = rng.choice(2**n_qubits, size=shots, p=probs)
        counts = {}
        for idx in indices:
            bits = format(idx, f'0{n_qubits}b')
            counts[bits] = counts.get(bits, 0) + 1
        return counts

    def _normalize_gates(self, gate_sequence):
        """Normalize various gate formats into (name, targets, *params)"""
        for gate in gate_sequence:
            if isinstance(gate, tuple):
                name = gate[0]
                targets = gate[1] if isinstance(gate[1], list) else [gate[1]]
                params = list(gate[2:]) if len(gate) > 2 else []
                yield name, targets, *params
            elif isinstance(gate, dict):
                yield gate["name"], gate["targets"], *gate.get("params", [])


class NvidiaGPUPlugin(BackendPlugin):
    """NVIDIA GPU via CUDA + cupy. Executes on device — no CPU fallback."""

    _rt_cache = False  # sentinel: not yet probed

    def _runtime(self):
        if self._rt_cache is not False:
            return self._rt_cache
        try:
            import cupy as cp
            cp.cuda.Device(0).use()
            cp.zeros(1, dtype=cp.complex64)  # force allocation on device
            self._rt_cache = cp
        except Exception:
            self._rt_cache = None
        return self._rt_cache

    def probe(self) -> Optional[BackendFingerprint]:
        try:
            r = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total,memory.free",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5
            )
            if r.returncode != 0:
                return None
            parts = r.stdout.strip().split(",")
            gpu_name = parts[0].strip()
            vram_free = float(parts[2].strip()) * 1024 * 1024
            max_qb = min(33, int(np.log2(vram_free / (16 * 3))))
        except Exception:
            return None

        cp = self._runtime()
        tag = "" if cp is not None else "[cupy-missing]"
        return BackendFingerprint(
            name=f"GPU::{gpu_name}{tag}",
            compute_class=ComputeClass.CLASSICAL_GPU,
            vendor="nvidia",
            max_qubits=max_qb,
            max_classical_bits=64,
            memory_bytes=int(vram_free),
            gate_time_ns=5.0 * (2**10),
            measurement_time_ns=1.0,
            classical_flops=10e12,
            supports_statevector=True,
            supports_density_matrix=(max_qb <= 16),
            supports_noise_model=False,
            native_gates=["H","X","Y","Z","S","T","CNOT","CZ","SWAP",
                         "Rx","Ry","Rz"],
            connectivity="all-to-all",
            is_available=(cp is not None),
            is_local=True,
            latency_ms=0.1,
            cost_per_shot=0.0,
            best_for=["unitary_evolution", "matrix_multiply", "sampling"],
        )

    def execute_statevector(self, n_qubits, gate_sequence, initial_state=None):
        cp = self._runtime()
        if cp is None:
            raise RuntimeError(
                "NVIDIA GPU detected but cupy is not importable. "
                "Install cupy-cuda12x (or matching CUDA version). "
                "No CPU fallback — this plugin runs only on GPU."
            )
        sv = _apply_gates_xp(cp, n_qubits, gate_sequence, initial_state, cp.complex64)
        return cp.asnumpy(sv).astype(complex)

    def execute_sample(self, n_qubits, gate_sequence, shots, initial_state=None):
        cp = self._runtime()
        if cp is None:
            raise RuntimeError(
                "NVIDIA GPU detected but cupy is not importable. "
                "Install cupy-cuda12x. No CPU fallback."
            )
        sv = _apply_gates_xp(cp, n_qubits, gate_sequence, initial_state, cp.complex64)
        probs = cp.asnumpy(cp.abs(sv) ** 2).astype(float)
        return _sample_counts(probs, shots, n_qubits)


class AMDGPUPlugin(BackendPlugin):
    """AMD GPU via ROCm. Uses cupy-rocm or torch-hip — no CPU fallback."""

    _rt_cache = False

    def _runtime(self):
        if self._rt_cache is not False:
            return self._rt_cache
        # Try cupy (ROCm build) first — cleanest API
        try:
            import cupy as cp
            if cp.cuda.runtime.getDeviceCount() > 0:
                cp.cuda.Device(0).use()
                cp.zeros(1, dtype=cp.complex64)
                self._rt_cache = ("cupy", cp)
                return self._rt_cache
        except Exception:
            pass
        # Fall forward to torch with HIP backend
        try:
            import torch
            if torch.cuda.is_available() and getattr(torch.version, "hip", None):
                self._rt_cache = ("torch", torch)
                return self._rt_cache
        except Exception:
            pass
        self._rt_cache = (None, None)
        return self._rt_cache

    def probe(self) -> Optional[BackendFingerprint]:
        try:
            r = subprocess.run(["rocm-smi", "--showmeminfo", "vram"],
                             capture_output=True, text=True, timeout=5)
            if r.returncode != 0:
                return None
            lines = r.stdout.strip().split('\n')
            vram_bytes = 4 * 1024**3
            for line in lines:
                if 'Total' in line:
                    for p in line.split():
                        try:
                            vram_bytes = int(p); break
                        except ValueError:
                            continue
            max_qb = min(33, int(np.log2(vram_bytes / (16 * 3))))
        except Exception:
            return None

        kind, _ = self._runtime()
        tag = "" if kind else "[rocm-runtime-missing]"
        return BackendFingerprint(
            name=f"GPU::AMD_ROCm{tag}",
            compute_class=ComputeClass.CLASSICAL_GPU,
            vendor="amd",
            max_qubits=max_qb,
            max_classical_bits=64,
            memory_bytes=vram_bytes,
            gate_time_ns=6.0 * (2**10),
            measurement_time_ns=1.5,
            classical_flops=8e12,
            supports_statevector=True,
            supports_density_matrix=False,
            supports_noise_model=False,
            native_gates=["H","X","Y","Z","CNOT","CZ","Rx","Ry","Rz"],
            connectivity="all-to-all",
            is_available=(kind is not None),
            is_local=True,
            latency_ms=0.1,
            cost_per_shot=0.0,
            best_for=["unitary_evolution", "matrix_multiply"],
        )

    def _missing(self):
        raise RuntimeError(
            "AMD GPU detected but neither cupy-rocm nor torch-hip is importable. "
            "Install one (pip install cupy-rocm-5-0  OR  the ROCm build of torch). "
            "No CPU fallback."
        )

    def execute_statevector(self, n_qubits, gate_sequence, initial_state=None):
        kind, rt = self._runtime()
        if kind == "cupy":
            sv = _apply_gates_xp(rt, n_qubits, gate_sequence, initial_state, rt.complex64)
            return rt.asnumpy(sv).astype(complex)
        if kind == "torch":
            dev = rt.device("cuda")  # HIP exposes as "cuda" in torch
            sv = _apply_gates_torch(rt, dev, n_qubits, gate_sequence,
                                   initial_state, rt.complex64)
            return sv.detach().cpu().numpy().astype(complex)
        self._missing()

    def execute_sample(self, n_qubits, gate_sequence, shots, initial_state=None):
        kind, rt = self._runtime()
        if kind == "cupy":
            sv = _apply_gates_xp(rt, n_qubits, gate_sequence, initial_state, rt.complex64)
            probs = rt.asnumpy(rt.abs(sv) ** 2).astype(float)
            return _sample_counts(probs, shots, n_qubits)
        if kind == "torch":
            dev = rt.device("cuda")
            sv = _apply_gates_torch(rt, dev, n_qubits, gate_sequence,
                                   initial_state, rt.complex64)
            probs = sv.abs().pow(2).detach().cpu().numpy().astype(float)
            return _sample_counts(probs, shots, n_qubits)
        self._missing()


class IntelGPUPlugin(BackendPlugin):
    """Intel GPU/XPU via torch + intel_extension_for_pytorch. No CPU fallback."""

    _rt_cache = False

    def _runtime(self):
        if self._rt_cache is not False:
            return self._rt_cache
        try:
            import torch
            import intel_extension_for_pytorch  # noqa: F401 (registers xpu)
            if hasattr(torch, "xpu") and torch.xpu.is_available():
                self._rt_cache = torch
                return torch
        except Exception:
            pass
        self._rt_cache = None
        return None

    def probe(self) -> Optional[BackendFingerprint]:
        detected = False
        for cmd in [["xpu-smi", "discovery"], ["sycl-ls"]]:
            try:
                r = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
                if r.returncode == 0 and ("Intel" in r.stdout or "GPU" in r.stdout):
                    detected = True
                    break
            except Exception:
                continue
        if not detected:
            return None

        torch = self._runtime()
        tag = "" if torch is not None else "[ipex-missing]"
        return BackendFingerprint(
            name=f"GPU::Intel_XPU{tag}",
            compute_class=ComputeClass.CLASSICAL_GPU,
            vendor="intel",
            max_qubits=28,
            max_classical_bits=64,
            memory_bytes=8 * 1024**3,
            gate_time_ns=8.0 * (2**10),
            measurement_time_ns=2.0,
            classical_flops=5e12,
            supports_statevector=True,
            supports_density_matrix=False,
            supports_noise_model=False,
            native_gates=["H","X","Y","Z","CNOT","Rx","Ry","Rz"],
            connectivity="all-to-all",
            is_available=(torch is not None),
            is_local=True,
            latency_ms=0.2,
            cost_per_shot=0.0,
            best_for=["unitary_evolution", "matrix_multiply"],
        )

    def execute_statevector(self, n_qubits, gate_sequence, initial_state=None):
        torch = self._runtime()
        if torch is None:
            raise RuntimeError(
                "Intel GPU detected but torch + intel_extension_for_pytorch "
                "is not importable. Install both to enable XPU execution. "
                "No CPU fallback."
            )
        dev = torch.device("xpu")
        sv = _apply_gates_torch(torch, dev, n_qubits, gate_sequence,
                               initial_state, torch.complex64)
        return sv.detach().cpu().numpy().astype(complex)

    def execute_sample(self, n_qubits, gate_sequence, shots, initial_state=None):
        torch = self._runtime()
        if torch is None:
            raise RuntimeError(
                "Intel GPU detected but torch + intel_extension_for_pytorch "
                "is not importable. No CPU fallback."
            )
        dev = torch.device("xpu")
        sv = _apply_gates_torch(torch, dev, n_qubits, gate_sequence,
                               initial_state, torch.complex64)
        probs = sv.abs().pow(2).detach().cpu().numpy().astype(float)
        return _sample_counts(probs, shots, n_qubits)


class AppleSiliconPlugin(BackendPlugin):
    """Apple M-series GPU via Metal. Runs on MLX (native) or torch-MPS.
    No CPU fallback — if neither runtime is available, execute raises."""

    _rt_cache = False

    def _runtime(self):
        if self._rt_cache is not False:
            return self._rt_cache
        # Prefer MLX — Apple's native framework, talks to Metal directly
        try:
            import mlx.core as mx
            a = mx.array(np.array([1.0 + 0j], dtype=np.complex64))
            _ = a + a  # smoke test
            self._rt_cache = ("mlx", mx)
            return self._rt_cache
        except Exception:
            pass
        # Fall forward to torch with MPS (complex support is partial on older torch)
        try:
            import torch
            if (hasattr(torch.backends, "mps")
                    and torch.backends.mps.is_available()
                    and torch.backends.mps.is_built()):
                self._rt_cache = ("torch", torch)
                return self._rt_cache
        except Exception:
            pass
        self._rt_cache = (None, None)
        return self._rt_cache

    def probe(self) -> Optional[BackendFingerprint]:
        try:
            r = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                             capture_output=True, text=True, timeout=5)
            if r.returncode != 0 or "Apple" not in r.stdout:
                return None
            r2 = subprocess.run(["sysctl", "-n", "hw.memsize"],
                               capture_output=True, text=True, timeout=5)
            mem = int(r2.stdout.strip()) if r2.returncode == 0 else 8 * 1024**3
            max_qb = min(30, int(np.log2(mem / (16 * 3))))
        except Exception:
            return None

        kind, _ = self._runtime()
        tag = "" if kind else "[mlx-and-torch-missing]"
        return BackendFingerprint(
            name=f"GPU::AppleSilicon{tag}",
            compute_class=ComputeClass.CLASSICAL_GPU,
            vendor="apple",
            max_qubits=max_qb,
            max_classical_bits=64,
            memory_bytes=mem,
            gate_time_ns=6.0 * (2**10),
            measurement_time_ns=1.0,
            classical_flops=7e12,
            supports_statevector=True,
            supports_density_matrix=False,
            supports_noise_model=False,
            native_gates=["H","X","Y","Z","CNOT","CZ","Rx","Ry","Rz"],
            connectivity="all-to-all",
            is_available=(kind is not None),
            is_local=True,
            latency_ms=0.1,
            cost_per_shot=0.0,
            best_for=["unitary_evolution", "matrix_multiply", "sampling"],
        )

    def _missing(self):
        raise RuntimeError(
            "Apple Silicon detected but neither mlx nor torch(MPS) is "
            "importable. Install with `pip install mlx` (preferred) or "
            "`pip install torch`. No CPU fallback."
        )

    def execute_statevector(self, n_qubits, gate_sequence, initial_state=None):
        kind, rt = self._runtime()
        if kind == "mlx":
            sv = _apply_gates_mlx(rt, n_qubits, gate_sequence, initial_state)
            return np.asarray(sv).astype(complex)
        if kind == "torch":
            dev = rt.device("mps")
            sv = _apply_gates_torch(rt, dev, n_qubits, gate_sequence,
                                   initial_state, rt.complex64)
            return sv.detach().cpu().numpy().astype(complex)
        self._missing()

    def execute_sample(self, n_qubits, gate_sequence, shots, initial_state=None):
        kind, rt = self._runtime()
        if kind == "mlx":
            sv = _apply_gates_mlx(rt, n_qubits, gate_sequence, initial_state)
            probs = np.abs(np.asarray(sv)) ** 2
            return _sample_counts(probs.astype(float), shots, n_qubits)
        if kind == "torch":
            dev = rt.device("mps")
            sv = _apply_gates_torch(rt, dev, n_qubits, gate_sequence,
                                   initial_state, rt.complex64)
            probs = sv.abs().pow(2).detach().cpu().numpy().astype(float)
            return _sample_counts(probs, shots, n_qubits)
        self._missing()


class QPUCloudPlugin(BackendPlugin):
    """
    Cloud QPU auto-detection.
    Checks for installed SDK packages and environment credentials.
    """

    PROVIDERS = {
        "ibm_quantum": {
            "packages": ["qiskit_ibm_runtime"],
            "env_keys": ["IBM_QUANTUM_TOKEN", "QISKIT_IBM_TOKEN", "IBMQ_TOKEN"],
            "max_qubits": 127,
            "connectivity": "heavy-hex",
            "gate_time_ns": 50.0,
        },
        "ionq": {
            # qiskit-ionq is the integration path we actually implement;
            # cirq-ionq and amazon-braket would need separate execute bridges.
            "packages": ["qiskit_ionq"],
            "env_keys": ["IONQ_API_KEY"],
            "max_qubits": 36,
            "connectivity": "all-to-all",
            "gate_time_ns": 100000.0,
        },
        "rigetti": {
            # Rigetti's cloud is routed through Braket today; we exercise it
            # via the amazon-braket-sdk Circuit translator.
            "packages": ["braket"],
            "env_keys": ["AWS_ACCESS_KEY_ID"],
            "max_qubits": 84,
            "connectivity": "grid",
            "gate_time_ns": 40.0,
        },
        "google_quantum": {
            "packages": ["cirq_google"],
            "env_keys": ["GOOGLE_CLOUD_PROJECT"],
            "max_qubits": 72,
            "connectivity": "grid",
            "gate_time_ns": 25.0,
        },
        "azure_quantum": {
            "packages": ["azure-quantum"],
            "env_keys": ["AZURE_QUANTUM_WORKSPACE"],
            "max_qubits": 36,
            "connectivity": "all-to-all",
            "gate_time_ns": 100000.0,
        },
        "amazon_braket": {
            "packages": ["braket"],
            "env_keys": ["AWS_ACCESS_KEY_ID"],
            "max_qubits": 79,
            "connectivity": "grid",
            "gate_time_ns": 40.0,
        },
    }

    def __init__(self, provider: str):
        self.provider = provider
        self.config = self.PROVIDERS.get(provider, {})

    def probe(self) -> Optional[BackendFingerprint]:
        if not self.config:
            return None

        # Check 1: Is the SDK installed?
        sdk_found = False
        for pkg in self.config.get("packages", []):
            try:
                importlib.import_module(pkg.replace("-", "_"))
                sdk_found = True
                break
            except ImportError:
                continue

        # Check 2: Are credentials configured?
        creds_found = False
        for key in self.config.get("env_keys", []):
            if os.environ.get(key):
                creds_found = True
                break

        if not sdk_found and not creds_found:
            return None

        return BackendFingerprint(
            name=f"QPU::{self.provider}",
            compute_class=ComputeClass.QUANTUM_HARDWARE,
            vendor=self.provider,
            max_qubits=self.config["max_qubits"],
            max_classical_bits=self.config["max_qubits"],
            memory_bytes=0,
            gate_time_ns=self.config["gate_time_ns"],
            measurement_time_ns=1000.0,
            classical_flops=0,
            supports_statevector=False,
            supports_density_matrix=False,
            supports_noise_model=True,
            native_gates=["Rx", "Ry", "Rz", "CNOT", "CZ"],
            connectivity=self.config["connectivity"],
            is_available=sdk_found and creds_found,
            is_local=False,
            latency_ms=500.0 if creds_found else 0,
            cost_per_shot=0.01,
            best_for=["unitary_evolution", "sampling"],
        )

    def execute_statevector(self, n_qubits, gate_sequence, initial_state=None):
        raise NotImplementedError(
            f"QPU {self.provider}: statevector not available on hardware. "
            f"Use execute_sample instead."
        )

    # Per-provider executor registry. Each entry is a method name on this
    # class; keep dispatch data-driven so adding a provider is one edit here
    # plus the method. Providers absent from this table raise
    # NotImplementedError from execute_sample, which the router respects.
    _EXECUTORS = {
        "ionq": "_execute_ionq_via_qiskit",
        "ibm_quantum": "_execute_ibm_via_qiskit",
        "amazon_braket": "_execute_braket",
        "rigetti": "_execute_braket",
    }

    def execute_sample(self, n_qubits, gate_sequence, shots, initial_state=None):
        handler = self._EXECUTORS.get(self.provider)
        if handler is None:
            raise NotImplementedError(
                f"QPU {self.provider}: no execution bridge wired. Add an "
                f"entry to QPUCloudPlugin._EXECUTORS and the corresponding "
                f"method to support this provider."
            )
        return getattr(self, handler)(
            n_qubits, gate_sequence, shots, initial_state
        )

    def submit_async(self, n_qubits, gate_sequence, shots,
                     initial_state=None) -> str:
        """Submit a circuit and return a provider-specific job id string.
        Used for real QPU queues where blocking on execute_sample is not
        acceptable. Retrieve results later with `retrieve_job(job_id)`."""
        handler = self._EXECUTORS.get(self.provider)
        if handler is None:
            raise NotImplementedError(
                f"QPU {self.provider}: no async submit path wired."
            )
        # Each executor supports an async_submit kwarg; returns the job id.
        return getattr(self, handler)(
            n_qubits, gate_sequence, shots, initial_state,
            async_submit=True,
        )

    def retrieve_job(self, job_id: str) -> Optional[dict]:
        """Poll a previously-submitted job for counts. Returns None if the
        job is still queued or running; returns a counts dict (vqpu
        convention, qubit 0 as MSB) when complete."""
        if self.provider == "ionq":
            return self._retrieve_ionq_job(job_id)
        if self.provider == "ibm_quantum":
            return self._retrieve_ibm_job(job_id)
        if self.provider in ("amazon_braket", "rigetti"):
            return self._retrieve_braket_job(job_id)
        raise NotImplementedError(
            f"QPU {self.provider}: job retrieval not wired."
        )

    # ─────────── IonQ execution bridge (via qiskit-ionq) ───────────

    def _execute_ionq_via_qiskit(self, n_qubits, gate_sequence, shots,
                                 initial_state=None,
                                 async_submit: bool = False):
        """Submit `gate_sequence` to IonQ via the qiskit-ionq provider.

        Reads credentials and routing from the environment:
          IONQ_API_KEY     — required (the plugin.probe also checks this).
          IONQ_BACKEND     — "simulator" (default), "qpu.forte-1", etc.
          IONQ_NOISE_MODEL — optional noise profile for the simulator
                             (e.g. "forte-1", "aria-1").

        Does NOT take the API key as an argument — always reads the env.
        """
        if initial_state is not None:
            raise RuntimeError(
                "IonQ hardware initializes at |0…0⟩; custom initial states "
                "are not supported. Prepend state-prep gates to the circuit."
            )
        try:
            from qiskit import QuantumCircuit as _QC, transpile
            from qiskit_ionq import IonQProvider
        except ImportError as exc:
            raise RuntimeError(
                "qiskit-ionq is not installed. Run: "
                "pip install qiskit qiskit-ionq"
            ) from exc

        api_key = os.environ.get("IONQ_API_KEY")
        if not api_key:
            raise RuntimeError(
                "IONQ_API_KEY is not set in the environment. "
                "Export it first: export IONQ_API_KEY=\"<your key>\""
            )

        provider = IonQProvider(api_key)
        backend_name = os.environ.get("IONQ_BACKEND", "simulator")
        try:
            backend = provider.get_backend(backend_name)
        except Exception as exc:
            raise RuntimeError(
                f"IonQ backend '{backend_name}' could not be created: {exc}. "
                "Valid names include 'simulator', 'qpu.forte-1', "
                "'qpu.forte-enterprise-1'."
            ) from exc

        noise_model = os.environ.get("IONQ_NOISE_MODEL")
        if noise_model:
            backend.set_options(noise_model=noise_model)

        qc = _QC(n_qubits)
        for name, targets, params in self._iter_gates(gate_sequence):
            self._append_qiskit_gate(qc, name, targets, params)
        qc.measure_all()

        # IonQ's "qis" gateset rejects raw `unitary` instructions. Decompose
        # any FULL_UNITARY ops (and anything else non-native) before submit.
        # Use optimization_level=1 per the qiskit-ionq advisory — level 2+
        # re-synthesizes aggressively and blurs the gate-count story.
        qc = transpile(qc, backend=backend, optimization_level=1)

        job = backend.run(qc, shots=shots)
        if async_submit:
            return str(job.job_id()) if hasattr(job, "job_id") else str(job)

        # qiskit-ionq's job exposes get_counts() directly; stock Qiskit
        # backends require .result().get_counts(). Accept either shape so
        # the same code path works in local Aer tests.
        if hasattr(job, "get_counts"):
            raw = job.get_counts()
        else:
            raw = job.result().get_counts()

        # Qiskit bitstring convention puts qubit (n-1) as MSB; the vqpu
        # convention puts qubit 0 as MSB. Reverse each key to match.
        items = raw.items() if hasattr(raw, "items") else list(raw)
        return {str(k)[::-1]: int(v) for k, v in items}

    def _retrieve_ionq_job(self, job_id: str) -> Optional[dict]:
        try:
            from qiskit_ionq import IonQProvider
        except ImportError as exc:
            raise RuntimeError(
                "qiskit-ionq is not installed. Run: pip install qiskit qiskit-ionq"
            ) from exc
        api_key = os.environ.get("IONQ_API_KEY")
        if not api_key:
            raise RuntimeError("IONQ_API_KEY is not set in the environment.")
        provider = IonQProvider(api_key)
        backend_name = os.environ.get("IONQ_BACKEND", "simulator")
        backend = provider.get_backend(backend_name)
        job = backend.retrieve_job(job_id)
        status = str(job.status()).upper() if hasattr(job, "status") else ""
        if "DONE" not in status and "COMPLETED" not in status and "SUCCESS" not in status:
            return None
        if hasattr(job, "get_counts"):
            raw = job.get_counts()
        else:
            raw = job.result().get_counts()
        items = raw.items() if hasattr(raw, "items") else list(raw)
        return {str(k)[::-1]: int(v) for k, v in items}

    # ─────────── IBM Quantum execution bridge (via qiskit-ibm-runtime) ───

    def _execute_ibm_via_qiskit(self, n_qubits, gate_sequence, shots,
                                initial_state=None,
                                async_submit: bool = False):
        """Submit `gate_sequence` to IBM Quantum via qiskit-ibm-runtime.

        Reads from the environment:
          IBM_QUANTUM_TOKEN / QISKIT_IBM_TOKEN / IBMQ_TOKEN — required
          IBM_BACKEND — e.g. "ibm_kyoto", "ibm_brisbane", "ibmq_qasm_simulator"
          IBM_CHANNEL — "ibm_quantum" (default) or "ibm_cloud"
          IBM_INSTANCE — optional hub/group/project path
        """
        if initial_state is not None:
            raise RuntimeError(
                "IBM hardware initializes at |0…0⟩; custom initial states "
                "are not supported. Prepend state-prep gates to the circuit."
            )
        try:
            from qiskit import QuantumCircuit as _QC, transpile
            from qiskit_ibm_runtime import QiskitRuntimeService
            try:
                from qiskit_ibm_runtime import SamplerV2 as Sampler
            except ImportError:
                from qiskit_ibm_runtime import Sampler
        except ImportError as exc:
            raise RuntimeError(
                "qiskit + qiskit-ibm-runtime are required. Run: "
                "pip install qiskit qiskit-ibm-runtime"
            ) from exc

        token = (
            os.environ.get("IBM_QUANTUM_TOKEN")
            or os.environ.get("QISKIT_IBM_TOKEN")
            or os.environ.get("IBMQ_TOKEN")
        )
        if not token:
            raise RuntimeError(
                "No IBM Quantum token in environment. Export one of: "
                "IBM_QUANTUM_TOKEN, QISKIT_IBM_TOKEN, IBMQ_TOKEN."
            )
        channel = os.environ.get("IBM_CHANNEL", "ibm_quantum")
        instance = os.environ.get("IBM_INSTANCE")
        kwargs = {"channel": channel, "token": token}
        if instance:
            kwargs["instance"] = instance
        service = QiskitRuntimeService(**kwargs)

        backend_name = os.environ.get("IBM_BACKEND")
        if backend_name:
            backend = service.backend(backend_name)
        else:
            backend = service.least_busy(operational=True, simulator=False)

        qc = _QC(n_qubits)
        for name, targets, params in self._iter_gates(gate_sequence):
            self._append_qiskit_gate(qc, name, targets, params)
        qc.measure_all()

        # IBM hardware rejects non-native gates without transpilation.
        qc = transpile(qc, backend=backend, optimization_level=1)

        sampler = Sampler(mode=backend)
        job = sampler.run([qc], shots=shots)
        if async_submit:
            return str(job.job_id())

        primitive_result = job.result()
        # SamplerV2 returns a list of pub results; .data.meas is the register
        # populated by measure_all.
        pub = primitive_result[0]
        data = pub.data
        # The register is named "meas" when the circuit used measure_all();
        # be tolerant of either attribute shape.
        counts_src = getattr(data, "meas", None) or next(iter(vars(data).values()))
        raw = counts_src.get_counts()
        items = raw.items() if hasattr(raw, "items") else list(raw)
        return {str(k)[::-1]: int(v) for k, v in items}

    def _retrieve_ibm_job(self, job_id: str) -> Optional[dict]:
        try:
            from qiskit_ibm_runtime import QiskitRuntimeService
        except ImportError as exc:
            raise RuntimeError(
                "qiskit-ibm-runtime is not installed. Run: pip install qiskit-ibm-runtime"
            ) from exc
        token = (
            os.environ.get("IBM_QUANTUM_TOKEN")
            or os.environ.get("QISKIT_IBM_TOKEN")
            or os.environ.get("IBMQ_TOKEN")
        )
        if not token:
            raise RuntimeError("No IBM Quantum token in environment.")
        channel = os.environ.get("IBM_CHANNEL", "ibm_quantum")
        service = QiskitRuntimeService(channel=channel, token=token)
        job = service.job(job_id)
        status = str(job.status()).upper()
        if "DONE" not in status and "COMPLETED" not in status:
            return None
        pub = job.result()[0]
        data = pub.data
        counts_src = getattr(data, "meas", None) or next(iter(vars(data).values()))
        raw = counts_src.get_counts()
        items = raw.items() if hasattr(raw, "items") else list(raw)
        return {str(k)[::-1]: int(v) for k, v in items}

    # ─────────── Amazon Braket execution bridge (multi-provider) ─────────

    def _execute_braket(self, n_qubits, gate_sequence, shots,
                        initial_state=None,
                        async_submit: bool = False):
        """Submit via amazon-braket-sdk. Covers multiple hardware vendors:
        Rigetti (Ankaa), IonQ-via-Braket, OQC Lucy, and the local simulator.

        Env:
          BRAKET_DEVICE_ARN — ARN of the AWS device (defaults to LocalSimulator)
          AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY / AWS_REGION — AWS creds
          AWS_DEFAULT_REGION or AWS_REGION — target region for AWS calls

        Braket stores measurement bitstrings with qubit 0 as MSB, matching
        vqpu's convention — no endian-flip required.
        """
        if initial_state is not None:
            raise RuntimeError(
                "Braket devices initialize at |0…0⟩; custom initial states "
                "are not supported."
            )
        try:
            from braket.circuits import Circuit
        except ImportError as exc:
            raise RuntimeError(
                "amazon-braket-sdk is not installed. Run: pip install amazon-braket-sdk"
            ) from exc

        circuit = Circuit()
        # Braket only measures qubits that appear in the circuit, so we
        # prepend an identity on every qubit index to guarantee the returned
        # bitstring has length n_qubits. Identity is a no-op on the state.
        for q in range(n_qubits):
            circuit.i(q)
        for name, targets, params in self._iter_gates(gate_sequence):
            self._append_braket_gate(circuit, name, targets, params)

        device_arn = os.environ.get("BRAKET_DEVICE_ARN")
        if device_arn:
            from braket.aws import AwsDevice
            device = AwsDevice(device_arn)
        else:
            from braket.devices import LocalSimulator
            device = LocalSimulator()

        task = device.run(circuit, shots=shots)
        if async_submit:
            # LocalSimulator returns synchronous results; treat its id as None
            # but for AwsDevice we get a real task ARN.
            return str(getattr(task, "id", None) or getattr(task, "arn", ""))

        result = task.result()
        counts_raw = result.measurement_counts
        return {str(k): int(v) for k, v in counts_raw.items()}

    def _retrieve_braket_job(self, job_id: str) -> Optional[dict]:
        try:
            from braket.aws import AwsQuantumTask
        except ImportError as exc:
            raise RuntimeError(
                "amazon-braket-sdk is not installed."
            ) from exc
        task = AwsQuantumTask(arn=job_id)
        state = str(task.state()).upper()
        if "COMPLETED" not in state:
            return None
        result = task.result()
        counts_raw = result.measurement_counts
        return {str(k): int(v) for k, v in counts_raw.items()}

    @staticmethod
    def _append_braket_gate(circuit, name: str, targets: list, params: list) -> None:
        if name == "H":
            circuit.h(targets[0]); return
        if name == "X":
            circuit.x(targets[0]); return
        if name == "Y":
            circuit.y(targets[0]); return
        if name == "Z":
            circuit.z(targets[0]); return
        if name == "S":
            circuit.s(targets[0]); return
        if name == "T":
            circuit.t(targets[0]); return
        if name == "CNOT":
            circuit.cnot(targets[0], targets[1]); return
        if name == "CZ":
            circuit.cz(targets[0], targets[1]); return
        if name == "SWAP":
            circuit.swap(targets[0], targets[1]); return
        theta = float(params[0]) if params else 0.0
        if name.startswith("Rx"):
            circuit.rx(targets[0], theta); return
        if name.startswith("Ry"):
            circuit.ry(targets[0], theta); return
        if name.startswith("Rz"):
            circuit.rz(targets[0], theta); return
        if name == "Phase":
            circuit.phaseshift(targets[0], theta); return
        if name == "FULL_UNITARY":
            circuit.unitary(matrix=params[0], targets=list(targets)); return
        raise ValueError(
            f"Gate '{name}' is not supported by the Braket translator."
        )

    @staticmethod
    def _iter_gates(gate_sequence):
        for gate in gate_sequence:
            if isinstance(gate, dict):
                name = gate["name"]
                targets = gate["targets"]
                params = list(gate.get("params", []))
            else:
                name = gate[0]
                targets = gate[1] if isinstance(gate[1], list) else [gate[1]]
                params = list(gate[2:]) if len(gate) > 2 else []
            yield name, list(targets), params

    @staticmethod
    def _append_qiskit_gate(qc, name: str, targets: list, params: list) -> None:
        if name in ("H", "X", "Y", "Z", "S", "T"):
            getattr(qc, name.lower())(targets[0])
            return
        if name == "CNOT":
            qc.cx(targets[0], targets[1]); return
        if name == "CZ":
            qc.cz(targets[0], targets[1]); return
        if name == "SWAP":
            qc.swap(targets[0], targets[1]); return
        if name.startswith("Rx"):
            qc.rx(float(params[0]) if params else 0.0, targets[0]); return
        if name.startswith("Ry"):
            qc.ry(float(params[0]) if params else 0.0, targets[0]); return
        if name.startswith("Rz"):
            qc.rz(float(params[0]) if params else 0.0, targets[0]); return
        if name == "Phase":
            qc.p(float(params[0]) if params else 0.0, targets[0]); return
        if name == "FULL_UNITARY":
            qc.unitary(params[0], targets, label="U"); return
        raise ValueError(
            f"Gate '{name}' is not supported by the IonQ translator."
        )


class TPUPlugin(BackendPlugin):
    """Google TPU via JAX. Pins computation to a TPU device — no CPU fallback."""

    _rt_cache = False

    def _runtime(self):
        if self._rt_cache is not False:
            return self._rt_cache
        try:
            import jax
            import jax.numpy as jnp
            tpu_devs = [d for d in jax.devices()
                        if getattr(d, "platform", "").lower() == "tpu"]
            if tpu_devs:
                self._rt_cache = (jax, jnp, tpu_devs[0])
                return self._rt_cache
        except Exception:
            pass
        self._rt_cache = (None, None, None)
        return self._rt_cache

    def probe(self) -> Optional[BackendFingerprint]:
        jax, _, dev = self._runtime()
        tpu_name = os.environ.get("TPU_NAME")
        if not tpu_name and os.path.exists("/dev/accel0"):
            tpu_name = "local_tpu"
        if not tpu_name and dev is not None:
            tpu_name = str(dev)
        if not tpu_name:
            return None

        tag = "" if dev is not None else "[jax-tpu-missing]"
        return BackendFingerprint(
            name=f"TPU::{tpu_name}{tag}",
            compute_class=ComputeClass.TPU,
            vendor="google",
            max_qubits=26,
            max_classical_bits=64,
            memory_bytes=16 * 1024**3,
            gate_time_ns=3.0 * (2**10),
            measurement_time_ns=1.0,
            classical_flops=50e12,
            supports_statevector=True,
            supports_density_matrix=False,
            supports_noise_model=False,
            native_gates=["H","X","Y","Z","CNOT","Rx","Ry","Rz"],
            connectivity="all-to-all",
            is_available=(dev is not None),
            is_local=not tpu_name.startswith("cloud"),
            latency_ms=50.0,
            cost_per_shot=0.0001,
            best_for=["matrix_multiply", "classical_opt"],
        )

    def execute_statevector(self, n_qubits, gate_sequence, initial_state=None):
        jax, jnp, dev = self._runtime()
        if dev is None:
            raise RuntimeError(
                "TPU environment detected but jax is not importable or no "
                "TPU devices are visible to jax. Install with "
                "`pip install jax[tpu]`. No CPU fallback."
            )
        with jax.default_device(dev):
            sv = _apply_gates_jax(jnp, n_qubits, gate_sequence,
                                 initial_state, jnp.complex64)
            sv = jax.device_get(sv)
        return np.asarray(sv).astype(complex)

    def execute_sample(self, n_qubits, gate_sequence, shots, initial_state=None):
        jax, jnp, dev = self._runtime()
        if dev is None:
            raise RuntimeError(
                "TPU environment detected but jax is not importable or no "
                "TPU devices are visible. Install with `pip install jax[tpu]`. "
                "No CPU fallback."
            )
        with jax.default_device(dev):
            sv = _apply_gates_jax(jnp, n_qubits, gate_sequence,
                                 initial_state, jnp.complex64)
            probs = jax.device_get(jnp.abs(sv) ** 2)
        return _sample_counts(np.asarray(probs).astype(float), shots, n_qubits)


# ═══════════════════════════════════════════════════════════
#  TASK DECOMPOSER — Split work into phases
# ═══════════════════════════════════════════════════════════

class TaskPhase(Enum):
    STATE_PREP = "state_prep"           # Classical: initialize state
    UNITARY_EVOLUTION = "unitary"       # Quantum-native: apply gates
    CLASSICAL_OPTIMIZATION = "opt"      # Classical: parameter updates
    MATRIX_MULTIPLY = "matmul"          # GPU-optimal: large mat-vec
    MEASUREMENT = "measure"             # Either: Born rule sampling
    POST_PROCESSING = "postproc"        # Classical: analyze results


@dataclass
class TaskSegment:
    """One phase of a hybrid computation"""
    phase: TaskPhase
    description: str
    n_qubits: int
    gate_count: int
    memory_bytes: int
    assigned_backend: Optional[str] = None
    estimated_time_ms: float = 0.0


@dataclass
class EntanglementEdge:
    """Weighted interaction edge inferred from the circuit topology."""
    qubits: Tuple[int, int]
    weight: int

    def to_dict(self) -> dict:
        return {
            "qubits": list(self.qubits),
            "weight": self.weight,
        }


@dataclass
class EntanglementComponent:
    """Connected region of qubits linked by entangling gates."""
    qubits: List[int]
    internal_weight: int
    min_single_qubit_cut: int
    max_single_qubit_cut: int
    is_quantum_core: bool

    def to_dict(self) -> dict:
        return {
            "qubits": self.qubits,
            "internal_weight": self.internal_weight,
            "min_single_qubit_cut": self.min_single_qubit_cut,
            "max_single_qubit_cut": self.max_single_qubit_cut,
            "is_quantum_core": self.is_quantum_core,
        }


@dataclass
class EntanglementScanResult:
    """Topology-only Phantom Step 1 scan data for a circuit."""
    n_qubits: int
    n_entangling_ops: int
    edges: List[EntanglementEdge]
    weighted_degree: Dict[int, int]
    components: List[EntanglementComponent]
    quantum_core_qubits: List[int]
    bridge_qubits: List[int]
    classical_qubits: List[int]
    isolated_qubits: List[int]
    heuristic: str = (
        "Topology-only scan: multi-qubit gates build a weighted interaction "
        "graph, connected entangling regions form the candidate quantum core, "
        "and articulation points are flagged as bridge qubits."
    )

    @property
    def largest_core_size(self) -> int:
        core_components = [len(c.qubits) for c in self.components if c.is_quantum_core]
        return max(core_components) if core_components else 0

    @property
    def total_edge_weight(self) -> int:
        return sum(edge.weight for edge in self.edges)

    def to_dict(self) -> dict:
        return {
            "n_qubits": self.n_qubits,
            "n_entangling_ops": self.n_entangling_ops,
            "n_entanglement_edges": len(self.edges),
            "total_edge_weight": self.total_edge_weight,
            "largest_core_size": self.largest_core_size,
            "quantum_core_qubits": self.quantum_core_qubits,
            "bridge_qubits": self.bridge_qubits,
            "classical_qubits": self.classical_qubits,
            "isolated_qubits": self.isolated_qubits,
            "weighted_degree": {
                str(qubit): degree
                for qubit, degree in sorted(self.weighted_degree.items())
            },
            "components": [component.to_dict() for component in self.components],
            "edges": [edge.to_dict() for edge in self.edges],
            "heuristic": self.heuristic,
        }


class EntanglementScanner:
    """
    Phantom Step 1: infer likely entanglement structure from circuit topology.

    This is intentionally topology-driven rather than amplitude-driven so it
    can run before execution and feed later partitioning stages.
    """

    def scan(self, circuit: QuantumCircuit) -> EntanglementScanResult:
        edge_weights: Dict[Tuple[int, int], int] = {}
        n_entangling_ops = 0

        for op in circuit.ops:
            unique_targets = sorted(set(op.targets))
            if len(unique_targets) <= 1:
                continue
            n_entangling_ops += 1
            for a, b in combinations(unique_targets, 2):
                edge_weights[(a, b)] = edge_weights.get((a, b), 0) + 1

        adjacency = {qubit: {} for qubit in range(circuit.n_qubits)}
        for (a, b), weight in edge_weights.items():
            adjacency[a][b] = weight
            adjacency[b][a] = weight

        weighted_degree = {
            qubit: int(sum(neighbors.values()))
            for qubit, neighbors in adjacency.items()
        }

        components = self._components_from_graph(adjacency, edge_weights)
        quantum_core_qubits = sorted(
            qubit
            for component in components
            if component.is_quantum_core
            for qubit in component.qubits
        )
        bridge_qubits = self._articulation_points(adjacency)
        classical_qubits = sorted(
            qubit for qubit in range(circuit.n_qubits) if qubit not in quantum_core_qubits
        )
        isolated_qubits = sorted(
            qubit for qubit, neighbors in adjacency.items() if not neighbors
        )

        return EntanglementScanResult(
            n_qubits=circuit.n_qubits,
            n_entangling_ops=n_entangling_ops,
            edges=[
                EntanglementEdge(qubits=edge, weight=weight)
                for edge, weight in sorted(edge_weights.items())
            ],
            weighted_degree=weighted_degree,
            components=components,
            quantum_core_qubits=quantum_core_qubits,
            bridge_qubits=bridge_qubits,
            classical_qubits=classical_qubits,
            isolated_qubits=isolated_qubits,
        )

    def _components_from_graph(
        self,
        adjacency: Dict[int, Dict[int, int]],
        edge_weights: Dict[Tuple[int, int], int],
    ) -> List[EntanglementComponent]:
        visited = set()
        components = []

        for root in sorted(adjacency):
            if root in visited or not adjacency[root]:
                continue

            stack = [root]
            component_nodes = []
            while stack:
                node = stack.pop()
                if node in visited:
                    continue
                visited.add(node)
                component_nodes.append(node)
                stack.extend(
                    neighbor for neighbor in adjacency[node] if neighbor not in visited
                )

            component_nodes.sort()
            component_set = set(component_nodes)
            component_edges = {
                edge: weight
                for edge, weight in edge_weights.items()
                if edge[0] in component_set and edge[1] in component_set
            }
            internal_weight = int(sum(component_edges.values()))
            cut_values = [
                int(sum(adjacency[node][neighbor] for neighbor in adjacency[node]))
                for node in component_nodes
            ]

            components.append(
                EntanglementComponent(
                    qubits=component_nodes,
                    internal_weight=internal_weight,
                    min_single_qubit_cut=min(cut_values),
                    max_single_qubit_cut=max(cut_values),
                    is_quantum_core=(len(component_nodes) > 1),
                )
            )

        return components

    def _articulation_points(
        self,
        adjacency: Dict[int, Dict[int, int]],
    ) -> List[int]:
        discovery: Dict[int, int] = {}
        low: Dict[int, int] = {}
        parent: Dict[int, Optional[int]] = {}
        articulation = set()
        clock = 0

        def dfs(node: int) -> None:
            nonlocal clock
            clock += 1
            discovery[node] = clock
            low[node] = clock
            child_count = 0

            for neighbor in adjacency[node]:
                if neighbor not in discovery:
                    parent[neighbor] = node
                    child_count += 1
                    dfs(neighbor)
                    low[node] = min(low[node], low[neighbor])

                    if parent.get(node) is None and child_count > 1:
                        articulation.add(node)
                    if (
                        parent.get(node) is not None
                        and low[neighbor] >= discovery[node]
                    ):
                        articulation.add(node)
                elif neighbor != parent.get(node):
                    low[node] = min(low[node], discovery[neighbor])

        for node, neighbors in adjacency.items():
            if not neighbors or node in discovery:
                continue
            parent[node] = None
            dfs(node)

        return sorted(articulation)


class TaskDecomposer:
    """
    Break a quantum computation into phases,
    each assignable to a different backend.
    """

    @staticmethod
    def decompose(
        circuit: QuantumCircuit,
        scan_result: Optional[EntanglementScanResult] = None,
    ) -> List[TaskSegment]:
        """Decompose a circuit into assignable phases."""
        segments = []
        core_size = scan_result.largest_core_size if scan_result is not None else 0
        bridge_count = len(scan_result.bridge_qubits) if scan_result is not None else 0
        state_desc = f"Initialize {circuit.n_qubits}-qubit register"
        if scan_result is not None and scan_result.quantum_core_qubits:
            state_desc += (
                f" ({len(scan_result.quantum_core_qubits)} core, "
                f"{len(scan_result.classical_qubits)} classical"
            )
            if bridge_count:
                state_desc += f", {bridge_count} bridge"
            state_desc += ")"

        # Phase 1: State preparation (classical)
        segments.append(TaskSegment(
            phase=TaskPhase.STATE_PREP,
            description=state_desc,
            n_qubits=circuit.n_qubits,
            gate_count=0,
            memory_bytes=(2 ** circuit.n_qubits) * 16,
        ))

        # Phase 2: Gate application (quantum/GPU)
        # Split into entangling vs single-qubit blocks
        single_gates = [op for op in circuit.ops if len(op.targets) == 1]
        multi_gates = [op for op in circuit.ops if len(op.targets) > 1]

        if single_gates:
            segments.append(TaskSegment(
                phase=TaskPhase.MATRIX_MULTIPLY,
                description=f"{len(single_gates)} single-qubit rotations",
                n_qubits=circuit.n_qubits,
                gate_count=len(single_gates),
                memory_bytes=(2 ** circuit.n_qubits) * 16,
            ))

        if multi_gates:
            entangle_desc = f"{len(multi_gates)} entangling gates"
            if core_size:
                entangle_desc += f" across a {core_size}-qubit core"
            segments.append(TaskSegment(
                phase=TaskPhase.UNITARY_EVOLUTION,
                description=entangle_desc,
                n_qubits=circuit.n_qubits,
                gate_count=len(multi_gates),
                memory_bytes=(2 ** circuit.n_qubits) * 16 * 2,
            ))

        # Phase 3: Measurement (classical)
        segments.append(TaskSegment(
            phase=TaskPhase.MEASUREMENT,
            description=(
                "Born rule sampling"
                + (
                    f" + {len(circuit.symmetries)} symmetry filters"
                    if getattr(circuit, "symmetries", None) else ""
                )
            ),
            n_qubits=circuit.n_qubits,
            gate_count=0,
            memory_bytes=(2 ** circuit.n_qubits) * 8,
        ))

        return segments


# ═══════════════════════════════════════════════════════════
#  HYBRID ROUTER — Assign phases to backends
# ═══════════════════════════════════════════════════════════

class HybridRouter:
    """
    Assigns each task segment to its optimal backend.
    Can mix CPU + GPU + QPU in a single computation.
    """

    def __init__(self, backends: Dict[str, BackendFingerprint]):
        self.backends = backends

    def route(self, segments: List[TaskSegment]) -> List[TaskSegment]:
        """Assign each segment to the best available backend."""
        for seg in segments:
            scores = {}
            for name, fp in self.backends.items():
                if not fp.is_available:
                    continue
                score = self._score_backend(fp, seg)
                if score > 0:
                    scores[name] = score

            if scores:
                best = max(scores, key=scores.get)
                seg.assigned_backend = best
                seg.estimated_time_ms = self._estimate_time(
                    self.backends[best], seg
                )
            else:
                seg.assigned_backend = "NONE_AVAILABLE"

        return segments

    def _score_backend(self, fp: BackendFingerprint, seg: TaskSegment) -> float:
        """Score a backend for a specific task segment."""
        score = 50.0

        # Can it handle the qubit count?
        if seg.n_qubits > fp.max_qubits:
            return -1

        # Some backends (for example remote QPUs) do not expose a meaningful
        # local-memory limit, so only apply this filter when one is known.
        if fp.memory_bytes > 0 and seg.memory_bytes > fp.memory_bytes * 0.8:
            return -1

        # Is this task in the backend's sweet spot?
        phase_key = seg.phase.value
        phase_map = {
            "state_prep": "state_prep",
            "unitary": "unitary_evolution",
            "opt": "classical_opt",
            "matmul": "matrix_multiply",
            "measure": "measurement",
            "postproc": "classical_opt",
        }
        mapped = phase_map.get(phase_key, phase_key)
        if mapped in fp.best_for:
            score += 40

        # Speed bonus
        if fp.compute_class == ComputeClass.CLASSICAL_GPU:
            if seg.phase in (TaskPhase.MATRIX_MULTIPLY, TaskPhase.UNITARY_EVOLUTION):
                score += 30  # GPUs excel at parallel matrix ops
        elif fp.compute_class == ComputeClass.QUANTUM_HARDWARE:
            if seg.phase == TaskPhase.UNITARY_EVOLUTION:
                score += 50  # QPUs are native here
            else:
                score -= 30  # QPUs bad at classical work

        # Latency penalty for remote
        if not fp.is_local:
            score -= fp.latency_ms / 100

        # Cost penalty
        score -= fp.cost_per_shot * 1000

        return score

    def _estimate_time(self, fp: BackendFingerprint, seg: TaskSegment) -> float:
        """Estimate execution time in ms."""
        if seg.gate_count == 0:
            return 0.1
        ops = 2 ** seg.n_qubits * seg.gate_count
        time_ns = ops * fp.gate_time_ns / (2**10)
        return time_ns / 1e6 + fp.latency_ms


# ═══════════════════════════════════════════════════════════
#  UNIVERSAL vQPU — The final orchestrator
# ═══════════════════════════════════════════════════════════

class UniversalvQPU:
    """
    Universal Quantum Processing Unit.
    
    Discovers every available backend, decomposes work into phases,
    routes each phase to its optimal backend, executes, combines.
    
        qpu = UniversalvQPU()   # discovers everything
        result = qpu.run(circuit)  # routes intelligently
    """

    # All known plugin classes — add new ones here
    PLUGIN_CLASSES = [
        CPUPlugin,
        NvidiaGPUPlugin,
        AMDGPUPlugin,
        IntelGPUPlugin,
        AppleSiliconPlugin,
        TPUPlugin,
    ] + [
        lambda p=p: QPUCloudPlugin(p)
        for p in QPUCloudPlugin.PROVIDERS.keys()
    ]

    def __init__(self, extra_plugins: Optional[List[BackendPlugin]] = None,
                 verbose: bool = True):
        self.verbose = verbose
        self.backends: Dict[str, BackendFingerprint] = {}
        self.plugins: Dict[str, BackendPlugin] = {}
        self.scanner = EntanglementScanner()
        self.decomposer = TaskDecomposer()

        # Phase 1: Universal discovery
        if verbose:
            print("\n  [UniversalvQPU] Scanning for backends...")

        all_plugins = []
        for cls in self.PLUGIN_CLASSES:
            try:
                plugin = cls() if isinstance(cls, type) else cls()
                all_plugins.append(plugin)
            except Exception:
                continue

        if extra_plugins:
            all_plugins.extend(extra_plugins)

        for plugin in all_plugins:
            try:
                fp = plugin.probe()
                if fp is not None:
                    self.backends[fp.name] = fp
                    self.plugins[fp.name] = plugin
                    status = "ONLINE" if fp.is_available else "DETECTED (no credentials)"
                    if verbose:
                        qb = f"{fp.max_qubits}qb" if fp.max_qubits > 0 else "classical"
                        print(f"    [{fp.compute_class.value:5s}] {fp.name:30s} "
                              f"{status:25s} {qb}")
            except Exception as e:
                if verbose:
                    print(f"    [error] {type(plugin).__name__}: {e}")

        # Init router
        self.router = HybridRouter(self.backends)

        online = sum(1 for fp in self.backends.values() if fp.is_available)
        detected = len(self.backends) - online
        if verbose:
            print(f"\n  [UniversalvQPU] {online} online, {detected} detected but offline")
            if online == 0:
                print("  [UniversalvQPU] WARNING: No backends available!")

    def circuit(self, n_qubits: int, name: str = "circuit") -> QuantumCircuit:
        return QuantumCircuit(n_qubits, name)

    def scan(self, circuit: QuantumCircuit) -> EntanglementScanResult:
        """Infer entanglement regions before execution."""
        return self.scanner.scan(circuit)

    def phantom_plan(
        self,
        circuit: QuantumCircuit,
        amplitude_threshold: float = 1e-8,
        max_active_states: Optional[int] = None,
        bond_dim: int = 32,
    ) -> dict:
        """Build the initial Phantom Step 3/4 partition plan."""
        from .phantom import PhantomPruningConfig, build_phantom_partition

        pruning = PhantomPruningConfig(
            amplitude_threshold=amplitude_threshold,
            max_active_states=max_active_states,
            bond_dim=bond_dim,
        )
        partition = build_phantom_partition(
            circuit,
            scan_result=self.scan(circuit),
            pruning=pruning,
        )
        return partition.to_dict()

    def plan(self, circuit: QuantumCircuit) -> dict:
        """Show the full execution plan without running."""
        scan_result = self.scan(circuit)
        segments = self.decomposer.decompose(circuit, scan_result)
        routed = self.router.route(segments)

        plan = {
            "circuit": circuit.name,
            "n_qubits": circuit.n_qubits,
            "n_gates": len(circuit.ops),
            "entanglement_scan": scan_result.to_dict(),
            "symmetries": circuit.symmetry_descriptors(),
            "backends_available": {
                name: {"class": fp.compute_class.value, "online": fp.is_available,
                       "max_qubits": fp.max_qubits, "vendor": fp.vendor}
                for name, fp in self.backends.items()
            },
            "phases": [
                {
                    "phase": seg.phase.value,
                    "description": seg.description,
                    "assigned_to": seg.assigned_backend,
                    "est_time_ms": round(seg.estimated_time_ms, 2),
                }
                for seg in routed
            ],
            "total_est_time_ms": sum(seg.estimated_time_ms for seg in routed),
            "is_hybrid": len(set(s.assigned_backend for s in routed)) > 1,
        }
        return plan

    def run_phantom(
        self,
        circuit: QuantumCircuit,
        shots: int = 1024,
        initial_state: Optional[QuantumRegister] = None,
        amplitude_threshold: float = 1e-8,
        max_active_states: Optional[int] = None,
        bond_dim: int = 32,
        seed: Optional[int] = None,
    ) -> ExecutionResult:
        """Execute with the Phantom sparse/factorized backend."""
        from .phantom import PhantomPruningConfig, PhantomSimulatorBackend

        backend = PhantomSimulatorBackend(
            seed=seed,
            pruning=PhantomPruningConfig(
                amplitude_threshold=amplitude_threshold,
                max_active_states=max_active_states,
                bond_dim=bond_dim,
            ),
        )
        return backend.execute(circuit, initial_state=initial_state, shots=shots)

    def run(self, circuit: QuantumCircuit, shots: int = 1024,
            initial_state: Optional[np.ndarray] = None) -> ExecutionResult:
        """Execute with intelligent hybrid routing."""
        t0 = time.time()
        scan_result = self.scan(circuit)

        # Decompose and route
        segments = self.decomposer.decompose(circuit, scan_result)
        routed = self.router.route(segments)
        backends_used = set(s.assigned_backend for s in routed if s.assigned_backend)

        if self.verbose:
            print(f"\n  [UniversalvQPU] Executing: {circuit.name}")
            if scan_result.quantum_core_qubits:
                print(
                    "    scan         → "
                    f"core={scan_result.quantum_core_qubits} "
                    f"bridge={scan_result.bridge_qubits} "
                    f"classical={scan_result.classical_qubits}"
                )
            for seg in routed:
                print(f"    {seg.phase.value:12s} → {seg.assigned_backend or 'N/A':25s} "
                      f"({seg.description})")

        # For now, execute on the backend assigned to the unitary phase
        # (This is where true hybrid would hand off between backends)
        exec_backend_name = None
        for seg in routed:
            if seg.phase in (TaskPhase.UNITARY_EVOLUTION, TaskPhase.MATRIX_MULTIPLY):
                exec_backend_name = seg.assigned_backend
                break
        if exec_backend_name is None:
            exec_backend_name = routed[0].assigned_backend

        if exec_backend_name not in self.plugins:
            raise RuntimeError(f"No plugin for backend: {exec_backend_name}")

        plugin = self.plugins[exec_backend_name]

        # Convert circuit to gate sequence
        gate_seq = []
        for op in circuit.ops:
            entry = (op.gate_name, op.targets)

            serialized_from_name = _build_gate_matrix(
                op.gate_name,
                op.params or [],
            )
            has_explicit_matrix = op.gate_matrix is not None
            needs_full_unitary = has_explicit_matrix and (
                serialized_from_name is None
                or serialized_from_name.shape != op.gate_matrix.shape
                or not np.allclose(serialized_from_name, op.gate_matrix)
            )

            if needs_full_unitary:
                entry = ("FULL_UNITARY", op.targets, op.gate_matrix)
            elif op.params:
                entry = (op.gate_name, op.targets, *op.params)
            gate_seq.append(entry)

        # Execute
        counts = plugin.execute_sample(circuit.n_qubits, gate_seq, shots, initial_state)
        counts, symmetry_report = SymmetryFilter.apply(counts, circuit.symmetries)
        dt = time.time() - t0

        if self.verbose:
            print(f"  [UniversalvQPU] Done in {dt*1000:.1f}ms "
                  f"({len(backends_used)} backend{'s' if len(backends_used)>1 else ''})")

        return ExecutionResult(
            counts=counts,
            statevector=None,
            execution_time=dt,
            backend_name=" + ".join(backends_used),
            circuit_name=circuit.name,
            n_qubits=circuit.n_qubits,
            gate_count=len(circuit.ops),
            circuit_depth=circuit.depth(),
            entanglement_pairs=[edge.qubits for edge in scan_result.edges],
            entropy=0.0,
            symmetry_report=symmetry_report,
        )


# ═══════════════════════════════════════════════════════════
#  TEST: Full universal discovery and hybrid routing
# ═══════════════════════════════════════════════════════════

def run_demo():
    print("╔" + "═"*58 + "╗")
    print("║  Universal vQPU — Backend Discovery & Hybrid Routing    ║")
    print("╚" + "═"*58 + "╝")

    qpu = UniversalvQPU(verbose=True)

    # Test 1: Bell state with full plan
    print("\n" + "━"*60)
    print("  TEST 1: Bell state — execution plan")
    print("━"*60)
    c = qpu.circuit(2, "bell_pair")
    c.h(0).cnot(0, 1)
    plan = qpu.plan(c)
    print(json.dumps(plan, indent=2))
    r = qpu.run(c, shots=2048)
    print(f"  Result: {r.counts}")

    # Test 2: Larger circuit
    print("\n" + "━"*60)
    print("  TEST 2: GHZ-10 — hybrid routing")
    print("━"*60)
    c2 = qpu.circuit(10, "ghz_10")
    c2.h(0)
    for i in range(1, 10):
        c2.cnot(0, i)
    plan2 = qpu.plan(c2)
    print(f"  Hybrid: {plan2['is_hybrid']}")
    for p in plan2["phases"]:
        print(f"    {p['phase']:12s} → {p['assigned_to']:25s} ~{p['est_time_ms']:.1f}ms")
    r2 = qpu.run(c2, shots=2048)
    p0 = r2.counts.get("0"*10, 0) / 2048
    p1 = r2.counts.get("1"*10, 0) / 2048
    print(f"  P(|0⟩^10)={p0:.3f}  P(|1⟩^10)={p1:.3f}")

    # Test 3: What a hypothetical mixed GPU+QPU plan looks like
    print("\n" + "━"*60)
    print("  TEST 3: What hybrid routing WOULD do with GPU+QPU")
    print("━"*60)
    c3 = qpu.circuit(20, "hybrid_demo_20qb")
    for i in range(20):
        c3.h(i)
    for i in range(19):
        c3.cnot(i, i+1)
    for i in range(20):
        c3.ry(i, np.pi/4)
    plan3 = qpu.plan(c3)
    for p in plan3["phases"]:
        print(f"    {p['phase']:12s} → {p['assigned_to']:25s} "
              f"{p['description']}")
    print(f"  Total est: {plan3['total_est_time_ms']:.1f}ms")

    # Summary of all detected backends
    print("\n" + "━"*60)
    print("  BACKEND REGISTRY — Full inventory")
    print("━"*60)
    for name, fp in qpu.backends.items():
        status = "ONLINE" if fp.is_available else "OFFLINE"
        print(f"  {fp.compute_class.value:5s} | {name:30s} | {status:8s} | "
              f"{fp.vendor:12s} | {fp.max_qubits:3d}qb | "
              f"best_for: {', '.join(fp.best_for[:3])}")

    print("\n  To add a new backend, implement BackendPlugin and pass to:")
    print("  UniversalvQPU(extra_plugins=[MyCustomPlugin()])")


if __name__ == "__main__":
    run_demo()
