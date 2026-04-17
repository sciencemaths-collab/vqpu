# CHESSO vQPU Implementation Tracker

## Overall plan
- [x] Section 1: Project config, global types, Hilbert bundle state core
- [x] Section 2: Hypergraph entanglement model
- [x] Section 3: Quantum operator library
- [x] Section 4: Noise and channel engine
- [x] Section 5: Soft measurement engine
- [x] Section 6: Expansion and compression engine
- [x] Section 7: Metrics engine
- [x] Section 8: CHESSO objective and policy layer
- [x] Section 9: Optimizer and control updates
- [x] Section 10: Main runtime loop
- [x] Section 11: Scheduler and execution manager
- [x] Section 12: Compiler / IR layer
- [x] Section 13: Benchmarks and workloads
- [x] Section 14: Telemetry and visualization

## Notes
- Current code is simulator-first and NumPy-based.
- Bundle expansion and contraction preserve normalization.
- Hypergraph logic feeds the policy layer beginning in Section 8.
- Package exports were refreshed during final integration so cross-module imports now expose the full runtime surface.
- Section 14 adds report export and plotting for runtime traces, benchmark summaries, and ablation summaries.

## Section 1 deliverables
- `core/config.py`
- `core/types.py`
- `core/state.py`
- `tests/test_section_01_smoke.py`

## Section 2 deliverables
- `core/hypergraph.py`
- `tests/test_section_02_smoke.py`

## Section 3 deliverables
- `ops/__init__.py`
- `ops/unitary_ops.py`
- `tests/test_section_03_smoke.py`

## Section 4 deliverables
- `ops/noise.py`
- `tests/test_section_04_smoke.py`

## Section 5 deliverables
- `ops/measurement.py`
- `tests/test_section_05_smoke.py`

## Section 6 deliverables
- `ops/expansion.py`
- `ops/__init__.py`
- `tests/test_section_06_smoke.py`

## Section 7 deliverables
- `control/__init__.py`
- `control/metrics.py`
- `tests/test_section_07_smoke.py`

## Section 8 deliverables
- `control/objective.py`
- `control/__init__.py`
- `ops/__init__.py`
- `tests/test_section_08_smoke.py`

## Section 9 deliverables
- `control/optimizer.py`
- `control/__init__.py`
- `tests/test_section_09_smoke.py`

## Section 10 deliverables
- `control/runtime.py`
- `control/__init__.py`
- `tests/test_section_10_smoke.py`

## Section 11 deliverables
- `control/scheduler.py`
- `control/__init__.py`
- `tests/test_section_11_smoke.py`

## Section 12 deliverables
- `compiler/__init__.py`
- `compiler/ir.py`
- `compiler/lowering.py`
- `compiler/qlambda_frontend.py`
- `tests/test_section_12_smoke.py`

## Section 13 deliverables
- `experiments/__init__.py`
- `experiments/workloads.py`
- `experiments/benchmarks.py`
- `experiments/ablations.py`
- `tests/test_section_13_smoke.py`

## Section 14 deliverables
- `viz/__init__.py`
- `viz/telemetry.py`
- `viz/plots.py`
- `tests/test_section_14_smoke.py`
