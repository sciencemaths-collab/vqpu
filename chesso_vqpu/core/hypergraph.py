from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from .types import BundleTopology, SectorId, SectorKind, SectorSpec


@dataclass(slots=True, frozen=True)
class HyperedgeId:
    """Stable identifier for one multipartite entanglement relation."""

    name: str

    def __str__(self) -> str:
        return self.name


@dataclass(slots=True)
class EntanglementVertex:
    """Vertex metadata aligned to a Hilbert sector."""

    sector_id: SectorId
    kind: SectorKind = SectorKind.LOGICAL
    tags: Tuple[str, ...] = ()
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_sector_spec(cls, spec: SectorSpec) -> "EntanglementVertex":
        return cls(
            sector_id=SectorId(str(spec.sector_id)),
            kind=spec.kind,
            tags=tuple(spec.tags),
            metadata=dict(spec.metadata),
        )


@dataclass(slots=True)
class EntanglementHyperedge:
    """Weighted multipartite entanglement relation over 2 or more sectors."""

    edge_id: HyperedgeId
    members: Tuple[SectorId, ...]
    weight: float = 1.0
    phase_bias: float = 0.0
    coherence_score: float = 1.0
    capacity: float = 1.0
    tags: Tuple[str, ...] = ()
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def order(self) -> int:
        return len(self.members)

    def validate(self, *, vertex_names: Set[str], max_order: int) -> None:
        names = [str(member) for member in self.members]
        if len(names) < 2:
            raise ValueError("A hyperedge must contain at least two members")
        if len(names) != len(set(names)):
            raise ValueError(f"Hyperedge {self.edge_id} contains duplicate members: {names}")
        if len(names) > max_order:
            raise ValueError(
                f"Hyperedge {self.edge_id} has order {len(names)} above max_order {max_order}"
            )
        missing = [name for name in names if name not in vertex_names]
        if missing:
            raise ValueError(f"Hyperedge {self.edge_id} references unknown vertices: {missing}")
        for value, label in (
            (self.weight, "weight"),
            (self.phase_bias, "phase_bias"),
            (self.coherence_score, "coherence_score"),
            (self.capacity, "capacity"),
        ):
            if not isfinite(value):
                raise ValueError(f"Hyperedge {self.edge_id} has non-finite {label}: {value!r}")
        if self.weight < 0.0:
            raise ValueError(f"Hyperedge {self.edge_id} weight must be non-negative")
        if self.coherence_score < 0.0:
            raise ValueError(f"Hyperedge {self.edge_id} coherence_score must be non-negative")
        if self.capacity < 0.0:
            raise ValueError(f"Hyperedge {self.edge_id} capacity must be non-negative")


@dataclass(slots=True)
class EntanglementCycle:
    """Closed walk used for topological or holonomy-aware control."""

    cycle_id: str
    vertex_path: Tuple[SectorId, ...]
    holonomy_phase: float = 0.0
    stability: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def validate(self, *, vertex_names: Set[str]) -> None:
        names = [str(v) for v in self.vertex_path]
        if len(names) < 4:
            raise ValueError("A cycle path must contain at least 3 vertices plus the closing vertex")
        if names[0] != names[-1]:
            raise ValueError(f"Cycle {self.cycle_id} is not closed: {names}")
        unique = names[:-1]
        if len(unique) != len(set(unique)):
            raise ValueError(f"Cycle {self.cycle_id} repeats a non-terminal vertex: {names}")
        missing = [name for name in unique if name not in vertex_names]
        if missing:
            raise ValueError(f"Cycle {self.cycle_id} references unknown vertices: {missing}")
        if not isfinite(self.holonomy_phase):
            raise ValueError(f"Cycle {self.cycle_id} holonomy_phase must be finite")
        if not isfinite(self.stability) or self.stability < 0.0:
            raise ValueError(f"Cycle {self.cycle_id} stability must be finite and non-negative")


@dataclass(slots=True)
class EntanglementRoute:
    """Routing metadata for steering entanglement flow through hyperedges."""

    route_id: str
    source: SectorId
    target: SectorId
    edge_path: Tuple[HyperedgeId, ...]
    score: float = 0.0
    bandwidth: float = 1.0
    latency: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def validate(self, *, available_edges: Set[str], vertex_names: Set[str]) -> None:
        if str(self.source) not in vertex_names:
            raise ValueError(f"Route {self.route_id} source {self.source} is not a known vertex")
        if str(self.target) not in vertex_names:
            raise ValueError(f"Route {self.route_id} target {self.target} is not a known vertex")
        if str(self.source) == str(self.target):
            raise ValueError(f"Route {self.route_id} must connect distinct vertices")
        if not self.edge_path:
            raise ValueError(f"Route {self.route_id} must contain at least one hyperedge")
        missing = [str(edge_id) for edge_id in self.edge_path if str(edge_id) not in available_edges]
        if missing:
            raise ValueError(f"Route {self.route_id} references unknown hyperedges: {missing}")
        for value, label in ((self.score, "score"), (self.bandwidth, "bandwidth"), (self.latency, "latency")):
            if not isfinite(value):
                raise ValueError(f"Route {self.route_id} has non-finite {label}: {value!r}")
        if self.bandwidth < 0.0 or self.latency < 0.0:
            raise ValueError(f"Route {self.route_id} bandwidth and latency must be non-negative")


@dataclass(slots=True)
class HypergraphSummary:
    """Small serializable summary used by tests and telemetry later on."""

    vertex_count: int
    hyperedge_count: int
    cycle_count: int
    route_count: int
    max_order: int
    weighted_degrees: Dict[str, float] = field(default_factory=dict)


@dataclass(slots=True)
class EntanglementHypergraph:
    """Section 2 hypergraph model for multipartite entanglement routing.

    The structure is simulator-first and intentionally explicit. Hyperedges can
    connect any number of active sectors up to `max_order`, while cycles and
    routes store higher-level topological hints for later CHESSO policy layers.
    """

    max_order: int = 4
    vertices: Dict[str, EntanglementVertex] = field(default_factory=dict)
    hyperedges: Dict[str, EntanglementHyperedge] = field(default_factory=dict)
    cycles: Dict[str, EntanglementCycle] = field(default_factory=dict)
    routes: Dict[str, EntanglementRoute] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.max_order < 2:
            raise ValueError("max_order must be at least 2")
        self.validate()

    @classmethod
    def from_topology(cls, topology: BundleTopology, *, max_order: int = 4) -> "EntanglementHypergraph":
        graph = cls(max_order=max_order)
        for spec in topology.sectors:
            graph.add_vertex(spec)
        return graph

    @property
    def vertex_names(self) -> Set[str]:
        return set(self.vertices.keys())

    def copy(self) -> "EntanglementHypergraph":
        return EntanglementHypergraph(
            max_order=self.max_order,
            vertices={
                name: EntanglementVertex(
                    sector_id=SectorId(str(vertex.sector_id)),
                    kind=vertex.kind,
                    tags=tuple(vertex.tags),
                    metadata=dict(vertex.metadata),
                )
                for name, vertex in self.vertices.items()
            },
            hyperedges={
                edge_id: EntanglementHyperedge(
                    edge_id=HyperedgeId(str(edge.edge_id)),
                    members=tuple(SectorId(str(member)) for member in edge.members),
                    weight=edge.weight,
                    phase_bias=edge.phase_bias,
                    coherence_score=edge.coherence_score,
                    capacity=edge.capacity,
                    tags=tuple(edge.tags),
                    metadata=dict(edge.metadata),
                )
                for edge_id, edge in self.hyperedges.items()
            },
            cycles={
                cycle_id: EntanglementCycle(
                    cycle_id=cycle.cycle_id,
                    vertex_path=tuple(SectorId(str(vertex)) for vertex in cycle.vertex_path),
                    holonomy_phase=cycle.holonomy_phase,
                    stability=cycle.stability,
                    metadata=dict(cycle.metadata),
                )
                for cycle_id, cycle in self.cycles.items()
            },
            routes={
                route_id: EntanglementRoute(
                    route_id=route.route_id,
                    source=SectorId(str(route.source)),
                    target=SectorId(str(route.target)),
                    edge_path=tuple(HyperedgeId(str(edge_id)) for edge_id in route.edge_path),
                    score=route.score,
                    bandwidth=route.bandwidth,
                    latency=route.latency,
                    metadata=dict(route.metadata),
                )
                for route_id, route in self.routes.items()
            },
            metadata=dict(self.metadata),
        )

    def validate(self) -> None:
        for name, vertex in self.vertices.items():
            if name != str(vertex.sector_id):
                raise ValueError(f"Vertex key {name!r} does not match sector_id {vertex.sector_id}")
        vertex_names = self.vertex_names
        for edge in self.hyperedges.values():
            edge.validate(vertex_names=vertex_names, max_order=self.max_order)
        for cycle in self.cycles.values():
            cycle.validate(vertex_names=vertex_names)
        for route in self.routes.values():
            route.validate(available_edges=set(self.hyperedges.keys()), vertex_names=vertex_names)
            self._validate_route_connectivity(route)

    def summary(self) -> HypergraphSummary:
        return HypergraphSummary(
            vertex_count=len(self.vertices),
            hyperedge_count=len(self.hyperedges),
            cycle_count=len(self.cycles),
            route_count=len(self.routes),
            max_order=self.max_order,
            weighted_degrees={name: self.weighted_degree(name) for name in sorted(self.vertices)},
        )

    def add_vertex(self, vertex: SectorSpec | EntanglementVertex, *, replace: bool = False) -> EntanglementVertex:
        if isinstance(vertex, SectorSpec):
            entry = EntanglementVertex.from_sector_spec(vertex)
        else:
            entry = EntanglementVertex(
                sector_id=SectorId(str(vertex.sector_id)),
                kind=vertex.kind,
                tags=tuple(vertex.tags),
                metadata=dict(vertex.metadata),
            )
        name = str(entry.sector_id)
        if not replace and name in self.vertices:
            raise ValueError(f"Vertex {name!r} already exists")
        self.vertices[name] = entry
        return entry

    def sync_from_topology(self, topology: BundleTopology, *, drop_missing: bool = False) -> None:
        topology.validate()
        topology_names = {str(sec.sector_id) for sec in topology.sectors}
        for spec in topology.sectors:
            self.add_vertex(spec, replace=True)
        if drop_missing:
            for name in list(self.vertices.keys()):
                if name not in topology_names:
                    self.remove_vertex(name, drop_incident_edges=True)
        self.validate()

    def remove_vertex(
        self,
        sector_id: str | SectorId,
        *,
        drop_incident_edges: bool = True,
        drop_routes: bool = True,
        drop_cycles: bool = True,
    ) -> EntanglementVertex:
        name = str(sector_id)
        if name not in self.vertices:
            raise KeyError(f"Unknown vertex: {name}")
        if not drop_incident_edges:
            incident = self.incident_edge_ids(name)
            if incident:
                raise ValueError(f"Vertex {name} still participates in hyperedges: {incident}")
        removed = self.vertices.pop(name)
        if drop_incident_edges:
            for edge_id in list(self.incident_edge_ids(name)):
                self.hyperedges.pop(edge_id, None)
        if drop_routes:
            for route_id, route in list(self.routes.items()):
                if str(route.source) == name or str(route.target) == name:
                    self.routes.pop(route_id, None)
                    continue
                if any(name in {str(member) for member in self.hyperedges[str(edge_id)].members} for edge_id in route.edge_path if str(edge_id) in self.hyperedges):
                    self.routes.pop(route_id, None)
        if drop_cycles:
            for cycle_id, cycle in list(self.cycles.items()):
                if name in {str(vertex) for vertex in cycle.vertex_path}:
                    self.cycles.pop(cycle_id, None)
        return removed

    def add_hyperedge(
        self,
        members: Sequence[str | SectorId],
        *,
        weight: float = 1.0,
        phase_bias: float = 0.0,
        coherence_score: float = 1.0,
        capacity: float = 1.0,
        tags: Sequence[str] = (),
        metadata: Optional[Dict[str, Any]] = None,
        edge_id: Optional[str] = None,
        replace: bool = False,
    ) -> EntanglementHyperedge:
        member_ids = tuple(SectorId(str(member)) for member in members)
        edge_name = edge_id or self.make_edge_id(member_ids)
        edge = EntanglementHyperedge(
            edge_id=HyperedgeId(edge_name),
            members=member_ids,
            weight=float(weight),
            phase_bias=float(phase_bias),
            coherence_score=float(coherence_score),
            capacity=float(capacity),
            tags=tuple(tags),
            metadata={} if metadata is None else dict(metadata),
        )
        edge.validate(vertex_names=self.vertex_names, max_order=self.max_order)
        if not replace and edge_name in self.hyperedges:
            raise ValueError(f"Hyperedge {edge_name!r} already exists")
        self.hyperedges[edge_name] = edge
        return edge

    def add_cycle(
        self,
        vertex_path: Sequence[str | SectorId],
        *,
        holonomy_phase: float = 0.0,
        stability: float = 1.0,
        metadata: Optional[Dict[str, Any]] = None,
        cycle_id: Optional[str] = None,
        replace: bool = False,
    ) -> EntanglementCycle:
        if len(vertex_path) < 3:
            raise ValueError("vertex_path must contain at least 3 vertices before closure")
        names = [str(vertex) for vertex in vertex_path]
        if names[0] != names[-1]:
            names.append(names[0])
        cycle_name = cycle_id or self.make_cycle_id(names[:-1])
        cycle = EntanglementCycle(
            cycle_id=cycle_name,
            vertex_path=tuple(SectorId(name) for name in names),
            holonomy_phase=float(holonomy_phase),
            stability=float(stability),
            metadata={} if metadata is None else dict(metadata),
        )
        cycle.validate(vertex_names=self.vertex_names)
        if not replace and cycle_name in self.cycles:
            raise ValueError(f"Cycle {cycle_name!r} already exists")
        self.cycles[cycle_name] = cycle
        return cycle

    def add_route(
        self,
        source: str | SectorId,
        target: str | SectorId,
        edge_path: Sequence[str | HyperedgeId],
        *,
        score: float = 0.0,
        bandwidth: float = 1.0,
        latency: float = 0.0,
        metadata: Optional[Dict[str, Any]] = None,
        route_id: Optional[str] = None,
        replace: bool = False,
    ) -> EntanglementRoute:
        src = SectorId(str(source))
        dst = SectorId(str(target))
        path_ids = tuple(HyperedgeId(str(edge)) for edge in edge_path)
        route_name = route_id or f"route:{src}->{dst}:{'~'.join(str(edge) for edge in path_ids)}"
        route = EntanglementRoute(
            route_id=route_name,
            source=src,
            target=dst,
            edge_path=path_ids,
            score=float(score),
            bandwidth=float(bandwidth),
            latency=float(latency),
            metadata={} if metadata is None else dict(metadata),
        )
        route.validate(available_edges=set(self.hyperedges.keys()), vertex_names=self.vertex_names)
        self._validate_route_connectivity(route)
        if not replace and route_name in self.routes:
            raise ValueError(f"Route {route_name!r} already exists")
        self.routes[route_name] = route
        return route

    def incident_edge_ids(self, sector_id: str | SectorId) -> List[str]:
        name = str(sector_id)
        return [
            edge_id
            for edge_id, edge in self.hyperedges.items()
            if name in {str(member) for member in edge.members}
        ]

    def neighbors(self, sector_id: str | SectorId) -> Set[str]:
        name = str(sector_id)
        if name not in self.vertices:
            raise KeyError(f"Unknown vertex: {name}")
        out: Set[str] = set()
        for edge in self.hyperedges.values():
            members = {str(member) for member in edge.members}
            if name in members:
                out.update(members)
        out.discard(name)
        return out

    def weighted_degree(self, sector_id: str | SectorId) -> float:
        name = str(sector_id)
        if name not in self.vertices:
            raise KeyError(f"Unknown vertex: {name}")
        total = 0.0
        for edge in self.hyperedges.values():
            if name in {str(member) for member in edge.members}:
                total += edge.weight
        return total

    def pair_projection(self) -> Dict[str, Set[str]]:
        adjacency = {name: set() for name in self.vertices}
        for edge in self.hyperedges.values():
            members = [str(member) for member in edge.members]
            for i, a in enumerate(members):
                for b in members[i + 1 :]:
                    adjacency[a].add(b)
                    adjacency[b].add(a)
        return adjacency

    def detect_pairwise_cycles(self, *, max_cycle_size: int = 6, max_cycles: int = 64) -> List[Tuple[str, ...]]:
        if max_cycle_size < 3:
            raise ValueError("max_cycle_size must be at least 3")
        adjacency = self.pair_projection()
        seen: Set[Tuple[str, ...]] = set()
        cycles: List[Tuple[str, ...]] = []

        def canonicalize(path: Sequence[str]) -> Tuple[str, ...]:
            seq = tuple(path)
            rotations = [seq[i:] + seq[:i] for i in range(len(seq))]
            rev = tuple(reversed(seq))
            rev_rotations = [rev[i:] + rev[:i] for i in range(len(rev))]
            return min(rotations + rev_rotations)

        def dfs(start: str, current: str, path: List[str], used: Set[str]) -> None:
            if len(cycles) >= max_cycles:
                return
            for nxt in adjacency[current]:
                if nxt == start and len(path) >= 3:
                    canon = canonicalize(path)
                    if canon not in seen:
                        seen.add(canon)
                        cycles.append(canon)
                    continue
                if nxt in used or len(path) >= max_cycle_size:
                    continue
                used.add(nxt)
                path.append(nxt)
                dfs(start, nxt, path, used)
                path.pop()
                used.remove(nxt)

        for start in sorted(adjacency):
            dfs(start, start, [start], {start})
            if len(cycles) >= max_cycles:
                break
        return cycles

    def refresh_cycles_from_projection(
        self,
        *,
        max_cycle_size: int = 6,
        max_cycles: int = 64,
        default_stability: float = 1.0,
    ) -> List[EntanglementCycle]:
        discovered = self.detect_pairwise_cycles(max_cycle_size=max_cycle_size, max_cycles=max_cycles)
        rebuilt: Dict[str, EntanglementCycle] = {}
        for nodes in discovered:
            cycle_name = self.make_cycle_id(nodes)
            prior = self.cycles.get(cycle_name)
            rebuilt[cycle_name] = EntanglementCycle(
                cycle_id=cycle_name,
                vertex_path=tuple(SectorId(name) for name in (list(nodes) + [nodes[0]])),
                holonomy_phase=0.0 if prior is None else prior.holonomy_phase,
                stability=default_stability if prior is None else prior.stability,
                metadata={} if prior is None else dict(prior.metadata),
            )
            rebuilt[cycle_name].validate(vertex_names=self.vertex_names)
        self.cycles = rebuilt
        return [self.cycles[key] for key in sorted(self.cycles)]

    def make_edge_id(self, members: Sequence[str | SectorId]) -> str:
        names = sorted(str(member) for member in members)
        return "edge:" + "|".join(names)

    def make_cycle_id(self, vertices: Sequence[str | SectorId]) -> str:
        names = [str(vertex) for vertex in vertices]
        if len(names) != len(set(names)):
            raise ValueError("Cycle identifiers require unique non-closed vertices")
        seq = tuple(names)
        rotations = [seq[i:] + seq[:i] for i in range(len(seq))]
        rev = tuple(reversed(seq))
        rev_rotations = [rev[i:] + rev[:i] for i in range(len(rev))]
        canon = min(rotations + rev_rotations)
        return "cycle:" + "|".join(canon)

    def _validate_route_connectivity(self, route: EntanglementRoute) -> None:
        current_vertices = {str(route.source)}
        for index, edge_id in enumerate(route.edge_path):
            edge = self.hyperedges[str(edge_id)]
            members = {str(member) for member in edge.members}
            if current_vertices.isdisjoint(members):
                raise ValueError(
                    f"Route {route.route_id} is disconnected at step {index}: edge {edge_id} does not touch current frontier"
                )
            current_vertices.update(members)
        if str(route.target) not in current_vertices:
            raise ValueError(f"Route {route.route_id} does not reach target {route.target}")
