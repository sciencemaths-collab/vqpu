# CHESSO vQPU

CHESSO (Coherent Hypergraph Entanglement and Superposition Steering Optimizer) is a simulator-first virtual quantum processing unit stack.

## Included subsystems
- `core/` dynamic Hilbert-bundle state, config, topology, hypergraph model
- `ops/` operators, noise, soft measurement, expansion, compression
- `control/` metrics, objective, optimizer, runtime, scheduler
- `compiler/` Qλ-style IR, lowering, frontend parser
- `experiments/` workloads, benchmarks, ablations
- `viz/` telemetry, plots, report export
- `tests/` section smoke tests

## Quick start
Create a virtual environment, then install the minimal dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run the section smoke tests:

```bash
python run_all_smoke_tests.py
```

## Project shape
The project uses top-level imports such as `from core import ...` and `from ops import ...`.
Run scripts from the project root after unzipping.

## Suggested next step
Build an end-to-end demo runner that:
1. parses a Qλ-style program,
2. lowers it into runtime actions,
3. executes CHESSO,
4. emits telemetry and plots.
