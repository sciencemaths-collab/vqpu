import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core import EntanglementHypergraph, HilbertBundleState, RuntimeConfig, SectorKind


def main() -> None:
    cfg = RuntimeConfig.for_density_matrix(max_active_qubits=8, seed=17)
    bundle = HilbertBundleState.initialize(sector_dims=[2, 2], config=cfg)
    bundle.add_sector(name="a0", dimension=2, kind=SectorKind.ANCILLA)

    graph = EntanglementHypergraph.from_topology(bundle.topology, max_order=3)
    graph.add_hyperedge(["q0", "q1"], weight=1.2, tags=["bell"])
    graph.add_hyperedge(["q1", "a0"], weight=0.8, tags=["bridge"])
    graph.add_hyperedge(["q0", "a0"], weight=0.6, tags=["feedback"])
    tri = graph.add_hyperedge(["q0", "q1", "a0"], weight=2.5, phase_bias=0.3, coherence_score=0.95)

    assert tri.order == 3
    assert graph.neighbors("q1") == {"q0", "a0"}
    assert abs(graph.weighted_degree("q0") - (1.2 + 0.6 + 2.5)) < 1e-12

    graph.add_route("q0", "a0", ["edge:q0|q1", "edge:a0|q1"], score=0.9, latency=0.2)
    cycles = graph.refresh_cycles_from_projection(max_cycle_size=4)
    summary = graph.summary()

    assert len(cycles) >= 1
    assert summary.vertex_count == 3
    assert summary.hyperedge_count == 4
    assert summary.route_count == 1
    assert summary.cycle_count >= 1

    # Ensure topology sync can absorb a new sector.
    bundle.add_sector(name="m0", dimension=2, kind=SectorKind.MEMORY)
    graph.sync_from_topology(bundle.topology)
    assert "m0" in graph.vertices

    print("Section 02 smoke test passed")


if __name__ == "__main__":
    main()
