from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from vqpu.chesso.compiler import execute_plan, lower_program, parse_qlambda_script
from vqpu.chesso.control import state_fidelity
from vqpu.chesso.core import RuntimeConfig


cfg_sv = RuntimeConfig.for_statevector(max_active_qubits=6, seed=7)

bell_program = parse_qlambda_script(
    """
    program bell_demo
    alloc q0
    alloc q1
    gate H q0
    gate CX q0 q1
    """
)
plan = lower_program(bell_program, config=cfg_sv)
result = execute_plan(plan, config=cfg_sv)

bell = np.array([1 / np.sqrt(2), 0.0, 0.0, 1 / np.sqrt(2)], dtype=np.complex128)
assert result.bundle.dims == (2, 2)
assert any(spec.call_name == "apply_gate" for spec in plan.call_specs)
assert state_fidelity(result.bundle, bell) > 0.999999

cfg_compact = RuntimeConfig.for_statevector(max_active_qubits=3, seed=7)

compiled_program = parse_qlambda_script(
    """
    program routed_demo
    alloc q0
    alloc q1
    entangle q0 q1 weight=1.2 phase_bias=0.2 route_id=r01
    expand anc0 dimension=2 score=0.8 kind=ancilla
    run steps=1 scheduled=true max_entanglers=1
    """
)
plan2 = lower_program(compiled_program, config=cfg_compact)
result2 = execute_plan(plan2, config=cfg_compact)

assert result2.bundle.active_sector_count == 3
assert result2.graph is not None
assert len(result2.graph.hyperedges) == 1
assert "r01" in result2.graph.routes
assert len(result2.runtime_reports) == 1
assert any(spec.call_name == "run_runtime" for spec in plan2.call_specs)

print("Section 12 smoke test passed")
