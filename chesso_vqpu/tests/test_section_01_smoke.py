import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from core import HilbertBundleState, MeasurementRecord, RuntimeConfig, SectorKind


def main() -> None:
    cfg = RuntimeConfig.for_density_matrix(max_active_qubits=8, seed=11)
    bundle = HilbertBundleState.initialize(sector_dims=[2, 2], config=cfg)
    assert bundle.dims == (2, 2)
    assert bundle.total_dimension == 4
    assert abs(bundle.quantum_state.trace - 1.0) < 1e-12

    bundle.add_sector(name="a0", dimension=2, kind=SectorKind.ANCILLA)
    assert bundle.dims == (2, 2, 2)
    assert bundle.total_dimension == 8

    bundle.record_measurement(MeasurementRecord(label="peek_z", strength=0.1, outcome=0))
    bundle.advance_step(dynamic_depth_increment=2)
    snap = bundle.snapshot()

    assert snap.dims == (2, 2, 2)
    assert snap.representation.value == "density_matrix"
    assert bundle.stats.measurements_used == 1
    assert bundle.stats.dynamic_depth == 2

    removed = bundle.remove_last_sector()
    assert removed.kind == SectorKind.ANCILLA
    assert bundle.dims == (2, 2)
    assert abs(bundle.quantum_state.trace - 1.0) < 1e-12

    print("Section 01 smoke test passed")


if __name__ == "__main__":
    main()
