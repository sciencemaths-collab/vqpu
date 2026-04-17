"""Circuit Knitting — heterogeneous circuit cutting across mixed backends.

Cuts a circuit whose entangling gates cross partition boundaries into
independent fragments, executes each fragment on the best-fit backend
(GPU, CPU, cloud QPU — simultaneously), and reconstructs the full
probability distribution.

Algorithm
─────────
For controlled gates (CNOT, CZ) cut at the control wire, the
decomposition is EXACT: the gate is diagonal in the control's Z basis,
so measuring the control and classically forwarding the result to
prepare the target reproduces the gate perfectly with zero sampling
overhead.

    P(x_up, x_dn) = P_up(x_up) · P_dn(x_dn | prep = x_up[cut_qubit])

For each cut CNOT, the upstream fragment runs normally and measures
all qubits. The cut qubit's measured value (0 or 1) determines which
preparation the downstream fragment receives. Two downstream executions
(one per prep) are batched, and the results are combined using the
upstream's conditional distribution.

Integration
───────────
- EntanglementScanner → finds where entanglement crosses partition
  boundaries and identifies minimum-weight cuts
- LinkManager → dispatches fragment execution across heterogeneous
  backends with preference routing
- UniversalvQPU → fallback single-backend execution per fragment
- ExecutionResult → output is standard vqpu result format
"""

from __future__ import annotations

import itertools
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from .core import (
    ExecutionResult,
    GateOp,
    QuantumCircuit,
    QuantumRegister,
)
from .universal import (
    EntanglementScanner,
    EntanglementScanResult,
)


# ═══════════════════════════════════════════════════════════
#  CUT SPECIFICATION
# ═══════════════════════════════════════════════════════════

@dataclass
class WireCut:
    """A single gate cut between two partitions.

    The cut gate (CNOT/CZ) is removed from the circuit. The upstream
    fragment measures the control qubit in Z. The downstream fragment
    prepares a virtual qubit based on the measurement result, then
    applies the cut gate locally.
    """
    gate_index: int
    gate_op: GateOp
    upstream_qubit: int
    downstream_qubit: int
    upstream_partition: int
    downstream_partition: int

    def __repr__(self) -> str:
        return (
            f"<WireCut gate={self.gate_op.gate_name}[{self.gate_index}] "
            f"q{self.upstream_qubit}→q{self.downstream_qubit} "
            f"part {self.upstream_partition}↔{self.downstream_partition}>"
        )


@dataclass
class CutPlan:
    """Complete cutting plan for a circuit."""
    circuit: QuantumCircuit
    scan: EntanglementScanResult
    partitions: List[Set[int]]
    cuts: List[WireCut]

    @property
    def n_cuts(self) -> int:
        return len(self.cuts)

    def to_dict(self) -> dict:
        return {
            "n_partitions": len(self.partitions),
            "n_cuts": len(self.cuts),
            "partitions": [sorted(p) for p in self.partitions],
            "cuts": [
                {
                    "gate": c.gate_op.gate_name,
                    "gate_index": c.gate_index,
                    "upstream_qubit": c.upstream_qubit,
                    "downstream_qubit": c.downstream_qubit,
                    "upstream_partition": c.upstream_partition,
                    "downstream_partition": c.downstream_partition,
                }
                for c in self.cuts
            ],
        }


# ═══════════════════════════════════════════════════════════
#  CIRCUIT FRAGMENT
# ═══════════════════════════════════════════════════════════

@dataclass
class _CutEndpoint:
    """One end of a wire cut inside a fragment."""
    cut_index: int
    local_qubit: int
    role: str  # "measure" (upstream) or "prep" (downstream)


@dataclass
class CircuitFragment:
    """A subcircuit representing one partition after cutting."""
    partition_id: int
    original_qubits: List[int]
    qubit_map: Dict[int, int]
    n_qubits: int
    ops: List[GateOp]
    cut_measures: List[_CutEndpoint]
    cut_preps: List[_CutEndpoint]

    def to_circuit(self, name: str = "") -> QuantumCircuit:
        circ = QuantumCircuit(self.n_qubits, name or f"fragment_{self.partition_id}")
        circ.ops = list(self.ops)
        return circ


# ═══════════════════════════════════════════════════════════
#  CUT FINDER
# ═══════════════════════════════════════════════════════════

class CutFinder:
    """Finds optimal cut points using the entanglement topology."""

    @staticmethod
    def auto_partition(
        circuit: QuantumCircuit,
        max_fragment_qubits: int,
        scan: Optional[EntanglementScanResult] = None,
    ) -> CutPlan:
        if scan is None:
            scan = EntanglementScanner().scan(circuit)

        n = circuit.n_qubits
        if n <= max_fragment_qubits:
            return CutPlan(
                circuit=circuit, scan=scan,
                partitions=[set(range(n))], cuts=[],
            )

        partitions = CutFinder._build_partitions(circuit, scan, max_fragment_qubits)
        cuts = CutFinder._find_cuts(circuit, partitions)
        return CutPlan(circuit=circuit, scan=scan, partitions=partitions, cuts=cuts)

    @staticmethod
    def partition_manual(
        circuit: QuantumCircuit,
        partitions: List[Set[int]],
        scan: Optional[EntanglementScanResult] = None,
    ) -> CutPlan:
        if scan is None:
            scan = EntanglementScanner().scan(circuit)

        all_qubits: Set[int] = set()
        for p in partitions:
            if all_qubits & p:
                raise ValueError(f"partitions overlap at qubits {all_qubits & p}")
            all_qubits |= p

        cuts = CutFinder._find_cuts(circuit, partitions)
        return CutPlan(circuit=circuit, scan=scan, partitions=partitions, cuts=cuts)

    @staticmethod
    def _build_partitions(
        circuit: QuantumCircuit,
        scan: EntanglementScanResult,
        max_qubits: int,
    ) -> List[Set[int]]:
        n = circuit.n_qubits

        if scan.components and all(
            len(c.qubits) <= max_qubits for c in scan.components
        ):
            return [set(c.qubits) for c in scan.components]

        adj: Dict[int, Dict[int, int]] = defaultdict(lambda: defaultdict(int))
        for edge in scan.edges:
            a, b = edge.qubits
            adj[a][b] += edge.weight
            adj[b][a] += edge.weight

        partitions: List[Set[int]] = []
        assigned: Set[int] = set()
        remaining = set(range(n))

        while remaining:
            seed = min(remaining, key=lambda q: scan.weighted_degree.get(q, 0))
            partition = {seed}
            assigned.add(seed)
            remaining.discard(seed)

            while len(partition) < max_qubits and remaining:
                best_q = None
                best_w = -1
                for q in remaining:
                    w = sum(adj[q].get(p, 0) for p in partition)
                    if w > best_w:
                        best_w = w
                        best_q = q
                if best_q is None:
                    break
                partition.add(best_q)
                assigned.add(best_q)
                remaining.discard(best_q)

            partitions.append(partition)

        return partitions

    @staticmethod
    def _find_cuts(
        circuit: QuantumCircuit,
        partitions: List[Set[int]],
    ) -> List[WireCut]:
        qubit_to_part: Dict[int, int] = {}
        for pid, part in enumerate(partitions):
            for q in part:
                qubit_to_part[q] = pid

        cuts: List[WireCut] = []
        for gi, op in enumerate(circuit.ops):
            if not op.is_two_qubit or len(op.targets) < 2:
                continue
            q0, q1 = op.targets[0], op.targets[1]
            p0 = qubit_to_part.get(q0)
            p1 = qubit_to_part.get(q1)
            if p0 is not None and p1 is not None and p0 != p1:
                cuts.append(WireCut(
                    gate_index=gi, gate_op=op,
                    upstream_qubit=q0, downstream_qubit=q1,
                    upstream_partition=p0, downstream_partition=p1,
                ))
        return cuts


# ═══════════════════════════════════════════════════════════
#  FRAGMENT BUILDER
# ═══════════════════════════════════════════════════════════

class FragmentBuilder:
    """Builds executable CircuitFragments from a CutPlan.

    Upstream fragments keep all their original qubits — the cut qubit
    is measured normally in Z and its value determines downstream prep.

    Downstream fragments get a virtual qubit that receives the prep
    state, plus the cut gate re-wired to connect virtual → target.
    """

    @staticmethod
    def build(plan: CutPlan) -> List[CircuitFragment]:
        cut_gate_indices = {c.gate_index for c in plan.cuts}
        qubit_to_part: Dict[int, int] = {}
        for pid, part in enumerate(plan.partitions):
            for q in part:
                qubit_to_part[q] = pid

        dn_extra: Dict[int, List[int]] = defaultdict(list)
        measures: Dict[int, List[_CutEndpoint]] = defaultdict(list)
        preps: Dict[int, List[_CutEndpoint]] = defaultdict(list)
        cut_gates: Dict[int, List[Tuple[GateOp, Dict[int, int]]]] = defaultdict(list)

        for ci, cut in enumerate(plan.cuts):
            measures[cut.upstream_partition].append(_CutEndpoint(
                cut_index=ci, local_qubit=cut.upstream_qubit, role="measure",
            ))

            vq = max(plan.partitions[cut.downstream_partition]) + 1 + len(dn_extra[cut.downstream_partition])
            dn_extra[cut.downstream_partition].append(vq)
            preps[cut.downstream_partition].append(_CutEndpoint(
                cut_index=ci, local_qubit=vq, role="prep",
            ))

            cut_gates[cut.downstream_partition].append(
                (cut.gate_op, {cut.upstream_qubit: vq})
            )

        partition_ops: Dict[int, List[GateOp]] = defaultdict(list)
        for gi, op in enumerate(plan.circuit.ops):
            if gi in cut_gate_indices:
                continue
            if not op.targets:
                continue
            pid = qubit_to_part.get(op.targets[0])
            if pid is not None:
                partition_ops[pid].append(op)

        fragments: List[CircuitFragment] = []
        for pid, part in enumerate(plan.partitions):
            all_q = sorted(part) + dn_extra.get(pid, [])
            qmap = {orig: local for local, orig in enumerate(all_q)}
            nq = len(all_q)

            remapped: List[GateOp] = []

            for gate_op, remap in cut_gates.get(pid, []):
                targets = [remap.get(t, t) for t in gate_op.targets]
                new_t = [qmap[t] for t in targets if t in qmap]
                if len(new_t) == len(gate_op.targets):
                    remapped.append(GateOp(
                        gate_name=gate_op.gate_name,
                        gate_matrix=gate_op.gate_matrix,
                        targets=new_t,
                        params=gate_op.params,
                        is_two_qubit=gate_op.is_two_qubit,
                    ))

            for op in partition_ops.get(pid, []):
                new_t = [qmap[t] for t in op.targets if t in qmap]
                if len(new_t) != len(op.targets):
                    continue
                remapped.append(GateOp(
                    gate_name=op.gate_name,
                    gate_matrix=op.gate_matrix,
                    targets=new_t,
                    params=op.params,
                    is_two_qubit=op.is_two_qubit,
                ))

            frag_measures = [
                _CutEndpoint(ep.cut_index, qmap[ep.local_qubit], ep.role)
                for ep in measures.get(pid, [])
            ]
            frag_preps = [
                _CutEndpoint(ep.cut_index, qmap[ep.local_qubit], ep.role)
                for ep in preps.get(pid, [])
            ]

            fragments.append(CircuitFragment(
                partition_id=pid,
                original_qubits=sorted(part),
                qubit_map=qmap,
                n_qubits=nq,
                ops=remapped,
                cut_measures=frag_measures,
                cut_preps=frag_preps,
            ))

        return fragments


# ═══════════════════════════════════════════════════════════
#  KNIT RESULT
# ═══════════════════════════════════════════════════════════

@dataclass
class KnitResult:
    """Reconstructed probability distribution from knitted fragments."""
    counts: Dict[str, int]
    n_qubits: int
    n_cuts: int
    n_fragment_executions: int
    wall_time_s: float
    fragment_details: List[Dict[str, Any]]

    def to_execution_result(self) -> ExecutionResult:
        return ExecutionResult(
            counts=self.counts,
            statevector=None,
            execution_time=self.wall_time_s,
            backend_name="knit",
            circuit_name="knitted",
            n_qubits=self.n_qubits,
            gate_count=0,
            circuit_depth=0,
            entanglement_pairs=[],
            entropy=0.0,
            symmetry_report=None,
            execution_metadata={
                "engine": "circuit_knitting",
                "n_cuts": self.n_cuts,
                "n_fragment_executions": self.n_fragment_executions,
            },
        )


# ═══════════════════════════════════════════════════════════
#  CIRCUIT KNITTER — Z-basis exact decomposition
# ═══════════════════════════════════════════════════════════

class CircuitKnitter:
    """Orchestrates circuit cutting, fragment dispatch, and reconstruction.

    For controlled gates (CNOT, CZ), the Z-basis decomposition is exact:
    measure the upstream control qubit, classically forward the result
    to prepare the downstream virtual qubit, and combine.

    Usage
    ─────
        from vqpu import UniversalvQPU
        from vqpu.knit import CircuitKnitter, CutFinder

        qpu = UniversalvQPU()
        circuit = qpu.circuit(40, "big_one")
        # ... build circuit ...

        plan = CutFinder.auto_partition(circuit, max_fragment_qubits=20)
        knitter = CircuitKnitter(plan)
        result = knitter.run(executor=qpu.run, shots=4096)
        print(result.counts)
    """

    def __init__(self, plan: CutPlan) -> None:
        self.plan = plan
        self.fragments = FragmentBuilder.build(plan) if plan.cuts else []

    def run(
        self,
        executor: Callable[..., ExecutionResult],
        shots: int = 1024,
    ) -> KnitResult:
        """Execute fragments and reconstruct the full distribution.

        ``executor`` signature: ``(circuit, shots) → ExecutionResult``.
        """
        t0 = time.perf_counter()

        if not self.plan.cuts:
            result = executor(self.plan.circuit, shots)
            return KnitResult(
                counts=result.counts,
                n_qubits=self.plan.circuit.n_qubits,
                n_cuts=0,
                n_fragment_executions=1,
                wall_time_s=time.perf_counter() - t0,
                fragment_details=[],
            )

        upstream_frags = [f for f in self.fragments if f.cut_measures]
        downstream_frags = [f for f in self.fragments if f.cut_preps]
        neutral_frags = [
            f for f in self.fragments
            if not f.cut_measures and not f.cut_preps
        ]

        total_executions = 0
        fragment_details: List[Dict[str, Any]] = []

        upstream_results: Dict[int, Dict[str, int]] = {}
        for frag in upstream_frags:
            circ = frag.to_circuit(f"upstream_{frag.partition_id}")
            result = executor(circ, shots)
            upstream_results[frag.partition_id] = result.counts
            total_executions += 1
            fragment_details.append({
                "partition_id": frag.partition_id,
                "role": "upstream",
                "n_qubits": frag.n_qubits,
                "original_qubits": frag.original_qubits,
            })

        neutral_results: Dict[int, Dict[str, int]] = {}
        for frag in neutral_frags:
            circ = frag.to_circuit(f"neutral_{frag.partition_id}")
            result = executor(circ, shots)
            neutral_results[frag.partition_id] = result.counts
            total_executions += 1
            fragment_details.append({
                "partition_id": frag.partition_id,
                "role": "neutral",
                "n_qubits": frag.n_qubits,
                "original_qubits": frag.original_qubits,
            })

        downstream_conditional: Dict[int, Dict[Tuple[int, ...], Dict[str, int]]] = {}
        for frag in downstream_frags:
            prep_cut_indices = [ep.cut_index for ep in frag.cut_preps]
            prep_local_qubits = [ep.local_qubit for ep in frag.cut_preps]
            n_preps = len(prep_cut_indices)
            prep_combos = list(itertools.product([0, 1], repeat=n_preps))

            conditional: Dict[Tuple[int, ...], Dict[str, int]] = {}
            for combo in prep_combos:
                circ = QuantumCircuit(
                    frag.n_qubits,
                    f"dn_{frag.partition_id}_prep{''.join(str(c) for c in combo)}",
                )
                for val, lq in zip(combo, prep_local_qubits):
                    if val == 1:
                        circ.x(lq)
                circ.ops.extend(frag.ops)

                result = executor(circ, shots)
                conditional[combo] = result.counts
                total_executions += 1

            downstream_conditional[frag.partition_id] = conditional
            fragment_details.append({
                "partition_id": frag.partition_id,
                "role": "downstream",
                "n_qubits": frag.n_qubits,
                "original_qubits": frag.original_qubits,
                "prep_combos": len(prep_combos),
            })

        combined = self._reconstruct(
            upstream_frags, upstream_results,
            downstream_frags, downstream_conditional,
            neutral_frags, neutral_results,
            shots,
        )

        return KnitResult(
            counts=combined,
            n_qubits=self.plan.circuit.n_qubits,
            n_cuts=len(self.plan.cuts),
            n_fragment_executions=total_executions,
            wall_time_s=time.perf_counter() - t0,
            fragment_details=fragment_details,
        )

    def _reconstruct(
        self,
        upstream_frags: List[CircuitFragment],
        upstream_results: Dict[int, Dict[str, int]],
        downstream_frags: List[CircuitFragment],
        downstream_conditional: Dict[int, Dict[Tuple[int, ...], Dict[str, int]]],
        neutral_frags: List[CircuitFragment],
        neutral_results: Dict[int, Dict[str, int]],
        shots: int,
    ) -> Dict[str, int]:
        n = self.plan.circuit.n_qubits

        cut_to_dn: Dict[int, Tuple[CircuitFragment, int]] = {}
        for frag in downstream_frags:
            for i, ep in enumerate(frag.cut_preps):
                cut_to_dn[ep.cut_index] = (frag, i)

        dn_dists: Dict[int, Dict[Tuple[int, ...], Dict[str, float]]] = {}
        for frag in downstream_frags:
            orig_idx = [frag.qubit_map[q] for q in frag.original_qubits]
            per_combo: Dict[Tuple[int, ...], Dict[str, float]] = {}
            for combo, counts in downstream_conditional[frag.partition_id].items():
                total = sum(counts.values()) or 1
                d: Dict[str, float] = defaultdict(float)
                for bs, cnt in counts.items():
                    key = ''.join(bs[i] for i in orig_idx)
                    d[key] += cnt / total
                per_combo[combo] = dict(d)
            dn_dists[frag.partition_id] = per_combo

        result_f: Dict[str, float] = defaultdict(float)

        for up_frag in upstream_frags:
            counts = upstream_results[up_frag.partition_id]
            total_up = sum(counts.values()) or 1
            measure_locals = [ep.local_qubit for ep in up_frag.cut_measures]
            cut_indices = [ep.cut_index for ep in up_frag.cut_measures]

            for bs, cnt in counts.items():
                up_p = cnt / total_up
                cut_vals = tuple(int(bs[lq]) for lq in measure_locals)
                up_bits = {
                    q: bs[up_frag.qubit_map[q]]
                    for q in up_frag.original_qubits
                }

                dn_part_dists: List[Tuple[CircuitFragment, Dict[str, float]]] = []
                for ci_idx, ci in enumerate(cut_indices):
                    dn_frag, dn_pos = cut_to_dn[ci]
                    val = cut_vals[ci_idx]
                    matching = {
                        combo: d
                        for combo, d in dn_dists[dn_frag.partition_id].items()
                        if combo[dn_pos] == val
                    }
                    combined_dn: Dict[str, float] = defaultdict(float)
                    for d in matching.values():
                        for k, v in d.items():
                            combined_dn[k] += v
                    n_match = len(matching) or 1
                    combined_dn = {k: v / n_match for k, v in combined_dn.items()}
                    dn_part_dists.append((dn_frag, combined_dn))

                self._assemble(result_f, n, up_bits, up_p, dn_part_dists, 0)

        total = sum(result_f.values())
        if total <= 0:
            return {}
        out: Dict[str, int] = {}
        for bs, p in result_f.items():
            c = max(0, round(p / total * shots))
            if c > 0:
                out[bs] = c
        return out

    def _assemble(
        self,
        result: Dict[str, float],
        n: int,
        bits: Dict[int, str],
        prob: float,
        dn_parts: List[Tuple[CircuitFragment, Dict[str, float]]],
        idx: int,
    ) -> None:
        if idx >= len(dn_parts):
            full = ['0'] * n
            for q, b in bits.items():
                full[q] = b
            result[''.join(full)] += prob
            return

        dn_frag, dn_dist = dn_parts[idx]
        for dn_bs, dn_p in dn_dist.items():
            merged = dict(bits)
            for bi, q in enumerate(dn_frag.original_qubits):
                if bi < len(dn_bs):
                    merged[q] = dn_bs[bi]
            self._assemble(result, n, merged, prob * dn_p, dn_parts, idx + 1)

    def run_heterogeneous(
        self,
        link_manager: Any,
        shots: int = 1024,
        prefer: Optional[Dict[int, Sequence[str]]] = None,
    ) -> KnitResult:
        """Execute fragments across heterogeneous backends via LinkManager."""
        from .link import QuantumTask

        def _exec_via_link(circuit: QuantumCircuit, shots: int) -> ExecutionResult:
            gate_seq = _circuit_to_gate_sequence(circuit)
            task = QuantumTask(
                n_qubits=circuit.n_qubits,
                gate_sequence=gate_seq,
                shots=shots,
            )
            pid = getattr(circuit, '_knit_partition_id', None)
            pref = (prefer or {}).get(pid, ()) if pid is not None else ()
            counts, link = link_manager.submit(task, prefer=pref)

            return ExecutionResult(
                counts=counts, statevector=None,
                execution_time=0, backend_name=link.handle,
                circuit_name=circuit.name, n_qubits=circuit.n_qubits,
                gate_count=len(circuit.ops), circuit_depth=0,
                entanglement_pairs=[], entropy=0,
                symmetry_report=None, execution_metadata=None,
            )

        return self.run(executor=_exec_via_link, shots=shots)


# ═══════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════

def _circuit_to_gate_sequence(circ: QuantumCircuit) -> List[Tuple]:
    seq: List[Tuple] = []
    for op in circ.ops:
        if op.params:
            seq.append((op.gate_name, op.targets, *op.params))
        else:
            seq.append((op.gate_name, op.targets))
    return seq


__all__ = [
    "WireCut",
    "CutPlan",
    "CircuitFragment",
    "CutFinder",
    "FragmentBuilder",
    "CircuitKnitter",
    "KnitResult",
]
