"""AEGIS-Ion-N(12,7,3,1) — nested search optimizer for trapped-ion circuits.

Cascade: 12 global families → 7 local refinements → 3 precision tweaks → 1 winner.

Every candidate is a ``gate_sequence`` in the vqpu format, so this module is a
drop-in post-processor for anything the CHESSO bridge produces. All rewrites
are exact and semantics-preserving; AEGIS picks the Pareto-best variant under
J(x) = w1·F(x) − w2·N2Q(x) − w3·D(x) − w5·(1−F(x)) + w8·H(x).

Two-qubit count and depth are computed on the vqpu gate_sequence directly;
fidelity is computed against the seed's state vector on vqpu's local backend.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

GateTuple = Tuple[Any, ...]
GateSeq = List[GateTuple]


# ══════════════════════════ gate kind + invariants ══════════════════════════

_SELF_INVERSE_1Q = {"H", "X", "Y", "Z"}
_SELF_INVERSE_2Q = {"CNOT", "CZ", "SWAP"}
_ROTATION_1Q = {"Rx", "Ry", "Rz", "Phase"}
_TWO_QUBIT = {"CNOT", "CZ", "SWAP"}


def _targets(g: GateTuple) -> List[int]:
    t = g[1]
    return list(t) if isinstance(t, list) else [t]


def _params(g: GateTuple) -> List[Any]:
    return list(g[2:]) if len(g) > 2 else []


def is_two_qubit(g: GateTuple) -> bool:
    if g[0] in _TWO_QUBIT:
        return True
    if g[0] == "FULL_UNITARY":
        return len(_targets(g)) >= 2
    return False


def count_2q(seq: GateSeq) -> int:
    """Weight FULL_UNITARY(n) as n·(n−1)/2 nearest-neighbour couplings.

    That matches the decomposition cost a linear-chain ion trap actually pays.
    """
    total = 0
    for g in seq:
        if g[0] in _TWO_QUBIT:
            total += 1
        elif g[0] == "FULL_UNITARY":
            n = len(_targets(g))
            if n >= 2:
                total += n * (n - 1) // 2
    return total


def circuit_depth(seq: GateSeq, n_qubits: int) -> int:
    """ASAP depth: each gate scheduled as early as its target wires allow."""
    wire_end = [0] * n_qubits
    for g in seq:
        ts = _targets(g)
        start = max(wire_end[t] for t in ts)
        for t in ts:
            wire_end[t] = start + 1
    return max(wire_end) if wire_end else 0


# ═══════════════════════════ peephole rewrites ════════════════════════════

def _same_targets(a: GateTuple, b: GateTuple) -> bool:
    return _targets(a) == _targets(b)


def _wires_overlap(a: GateTuple, b: GateTuple) -> bool:
    return bool(set(_targets(a)) & set(_targets(b)))


def _commute_on_shared_wire(a: GateTuple, b: GateTuple) -> bool:
    """Lightweight commutation oracle for adjacent ops that share wires.

    Only returns True for cases we are certain about — used to slide ops past
    each other to expose cancellations. Conservative: if unsure, return False.
    """
    if not _wires_overlap(a, b):
        return True
    na, nb = a[0], b[0]
    ta, tb = _targets(a), _targets(b)
    shared = set(ta) & set(tb)

    if na in _ROTATION_1Q and nb in _ROTATION_1Q and na == nb and ta == tb:
        return True  # same-axis rotations always commute
    if na == "Rz" and nb == "CNOT" and ta[0] == tb[0]:
        return True  # Rz on control commutes with CNOT
    if nb == "Rz" and na == "CNOT" and tb[0] == ta[0]:
        return True
    if na in {"X", "Rx"} and nb == "CNOT" and ta[0] == tb[1]:
        return True  # X on target commutes with CNOT
    if nb in {"X", "Rx"} and na == "CNOT" and tb[0] == ta[1]:
        return True
    if na == "Z" and nb == "Rz" and ta == tb:
        return True
    return False


def _merge_rotations(a: GateTuple, b: GateTuple) -> Optional[GateTuple]:
    """Combine two rotations of the same axis on the same target."""
    if a[0] != b[0] or a[0] not in _ROTATION_1Q:
        return None
    if _targets(a) != _targets(b):
        return None
    theta = float(_params(a)[0]) + float(_params(b)[0])
    return (a[0], _targets(a), theta)


def _cancels(a: GateTuple, b: GateTuple) -> bool:
    """Two adjacent gates on identical targets that annihilate."""
    if a[0] != b[0]:
        return False
    if a[0] in _SELF_INVERSE_1Q and _targets(a) == _targets(b):
        return True
    if a[0] in _SELF_INVERSE_2Q and _targets(a) == _targets(b):
        return True
    return False


def _is_zero_rotation(g: GateTuple, tol: float = 1e-12) -> bool:
    if g[0] not in _ROTATION_1Q:
        return False
    theta = float(_params(g)[0]) % (4 * math.pi)
    if theta > 2 * math.pi:
        theta -= 4 * math.pi
    return abs(theta) < tol or abs(theta - 2 * math.pi) < tol or abs(theta + 2 * math.pi) < tol


# ─── passes ─────────────────────────────────────────────────────────────────

def pass_cancel_adjacent(seq: GateSeq) -> GateSeq:
    """Delete strictly-adjacent cancelling pairs."""
    out: GateSeq = []
    for g in seq:
        if out and _cancels(out[-1], g):
            out.pop()
            continue
        out.append(g)
    return out


def pass_merge_rotations(seq: GateSeq) -> GateSeq:
    out: GateSeq = []
    for g in seq:
        if out:
            merged = _merge_rotations(out[-1], g)
            if merged is not None:
                out[-1] = merged
                continue
        out.append(g)
    return out


def pass_drop_zero_rotations(seq: GateSeq) -> GateSeq:
    return [g for g in seq if not _is_zero_rotation(g)]


def pass_commute_and_cancel(seq: GateSeq) -> GateSeq:
    """Slide ops past each other one hop at a time to expose cancellations.

    Only the commutation oracle above is trusted, so this pass cannot introduce
    semantic drift: every swap is between ops we know commute.
    """
    work = list(seq)
    progressed = True
    while progressed:
        progressed = False
        i = 0
        while i < len(work) - 1:
            a, b = work[i], work[i + 1]
            if _cancels(a, b):
                del work[i : i + 2]
                progressed = True
                continue
            merged = _merge_rotations(a, b)
            if merged is not None:
                work[i : i + 2] = [merged]
                progressed = True
                continue
            # Try to pull a matching partner across one commuting neighbour.
            if i + 2 < len(work):
                c = work[i + 2]
                if _commute_on_shared_wire(b, c) and _cancels(a, c):
                    work[i : i + 3] = [b]
                    progressed = True
                    continue
                if _commute_on_shared_wire(b, c):
                    merged_ac = _merge_rotations(a, c)
                    if merged_ac is not None:
                        work[i : i + 3] = [b, merged_ac]
                        progressed = True
                        continue
            i += 1
    return work


def pass_reverse_traversal(seq: GateSeq) -> GateSeq:
    """Run the cancel-pass on the reversed sequence, then un-reverse.

    Inverse-pair cancellation is symmetric, so reversing exposes cancellation
    opportunities the forward pass may miss when the sequence is interleaved.
    """
    return list(reversed(pass_cancel_adjacent(list(reversed(seq)))))


# ════════════════════════ 12 families, 7 refinements, 3 tweaks ══════════════

@dataclass(slots=True, frozen=True)
class Strategy:
    name: str
    passes: Tuple[Callable[[GateSeq], GateSeq], ...]
    repeats: int = 2


def _run_strategy(seed: GateSeq, strat: Strategy) -> GateSeq:
    cur = list(seed)
    for _ in range(strat.repeats):
        for p in strat.passes:
            cur = p(cur)
    return cur


def _global_families() -> List[Strategy]:
    """12 structurally distinct strategies covering the search space."""
    cancel = pass_cancel_adjacent
    merge = pass_merge_rotations
    zerodrop = pass_drop_zero_rotations
    commute = pass_commute_and_cancel
    reverse = pass_reverse_traversal
    return [
        Strategy("F01-identity", (), repeats=1),
        Strategy("F02-cancel", (cancel,), repeats=2),
        Strategy("F03-merge", (merge,), repeats=2),
        Strategy("F04-zerodrop", (zerodrop,), repeats=1),
        Strategy("F05-cancel+merge", (cancel, merge), repeats=2),
        Strategy("F06-merge+cancel", (merge, cancel), repeats=2),
        Strategy("F07-cancel+zerodrop", (cancel, zerodrop), repeats=2),
        Strategy("F08-merge+zerodrop", (merge, zerodrop), repeats=2),
        Strategy("F09-commute", (commute,), repeats=1),
        Strategy("F10-commute+merge", (commute, merge), repeats=2),
        Strategy("F11-reverse+cancel", (reverse, cancel), repeats=2),
        Strategy("F12-full", (cancel, merge, zerodrop, reverse, commute), repeats=3),
    ]


def _refinements(winner_name: str) -> List[Strategy]:
    """7 local structural refinements — medium-radius perturbations."""
    cancel = pass_cancel_adjacent
    merge = pass_merge_rotations
    zerodrop = pass_drop_zero_rotations
    commute = pass_commute_and_cancel
    reverse = pass_reverse_traversal
    base = (cancel, merge, zerodrop)
    return [
        Strategy(f"{winner_name}|R1-base", base, repeats=2),
        Strategy(f"{winner_name}|R2-deep", base, repeats=4),
        Strategy(f"{winner_name}|R3-commute", (commute, merge, zerodrop), repeats=2),
        Strategy(f"{winner_name}|R4-reverse", (reverse, cancel, merge), repeats=2),
        Strategy(f"{winner_name}|R5-merge-first", (merge, cancel, merge, zerodrop), repeats=2),
        Strategy(f"{winner_name}|R6-cancel-first", (cancel, merge, cancel, zerodrop), repeats=2),
        Strategy(f"{winner_name}|R7-cascade", (commute, merge, cancel, zerodrop, reverse), repeats=3),
    ]


def _precision(winner_name: str) -> List[Strategy]:
    cancel = pass_cancel_adjacent
    merge = pass_merge_rotations
    zerodrop = pass_drop_zero_rotations
    commute = pass_commute_and_cancel
    return [
        Strategy(f"{winner_name}|P1-hold", (), repeats=1),
        Strategy(f"{winner_name}|P2-zerodrop+merge", (zerodrop, merge), repeats=1),
        Strategy(f"{winner_name}|P3-commute+cancel", (commute, cancel), repeats=1),
    ]


# ═══════════════════════════════ scoring ════════════════════════════════════

@dataclass(slots=True, frozen=True)
class ScoreWeights:
    w_fidelity: float = 4.0       # w1
    w_2q: float = 1.0             # w2
    w_depth: float = 0.25         # w3
    w_infidelity: float = 50.0    # w5 — huge penalty if rewrite broke semantics
    w_hierarchical: float = 0.5   # w8


@dataclass(slots=True, frozen=True)
class Metrics:
    n_2q: int
    depth: int
    length: int
    fidelity: float

    def j(self, w: ScoreWeights) -> float:
        return (
            w.w_fidelity * self.fidelity
            - w.w_2q * self.n_2q
            - w.w_depth * self.depth
            - w.w_infidelity * (1.0 - self.fidelity)
        )


def _evaluate(
    seq: GateSeq,
    n_qubits: int,
    reference_state: np.ndarray,
    simulator,
) -> Metrics:
    n2q = count_2q(seq)
    depth = circuit_depth(seq, n_qubits)
    length = len(seq)
    try:
        sv = simulator(n_qubits, seq)
    except Exception:
        return Metrics(n2q, depth, length, 0.0)
    if sv.shape != reference_state.shape:
        return Metrics(n2q, depth, length, 0.0)
    fid = float(abs(np.vdot(reference_state, sv)) ** 2)
    return Metrics(n2q, depth, length, fid)


# ═══════════════════════════ public API ═════════════════════════════════════

@dataclass(slots=True)
class CandidateRecord:
    strategy: str
    stage: str  # "12" | "7" | "3" | "1"
    metrics: Metrics
    score: float
    sequence: GateSeq = field(repr=False)


@dataclass(slots=True)
class AegisResult:
    seed_metrics: Metrics
    winner: CandidateRecord
    cascade: List[CandidateRecord]
    weights: ScoreWeights

    def improvement(self) -> Dict[str, Any]:
        d2q = self.seed_metrics.n_2q - self.winner.metrics.n_2q
        dd = self.seed_metrics.depth - self.winner.metrics.depth
        dl = self.seed_metrics.length - self.winner.metrics.length
        return {
            "delta_2q": d2q,
            "delta_depth": dd,
            "delta_length": dl,
            "fidelity": self.winner.metrics.fidelity,
            "seed_2q": self.seed_metrics.n_2q,
            "winner_2q": self.winner.metrics.n_2q,
            "seed_depth": self.seed_metrics.depth,
            "winner_depth": self.winner.metrics.depth,
        }


def _default_simulator() -> Callable[[int, GateSeq], np.ndarray]:
    from vqpu.universal import CPUPlugin
    plugin = CPUPlugin()
    return lambda n, seq: plugin.execute_statevector(n, seq)


def aegis_ion_nested(
    gate_sequence: Sequence[GateTuple],
    n_qubits: int,
    *,
    weights: Optional[ScoreWeights] = None,
    simulator: Optional[Callable[[int, GateSeq], np.ndarray]] = None,
    seed: int = 0,
) -> AegisResult:
    """Run the 12 → 7 → 3 → 1 cascade on an existing vqpu gate sequence."""
    del seed  # reserved for future stochastic variants
    weights = weights or ScoreWeights()
    sim = simulator or _default_simulator()
    seed_seq: GateSeq = list(gate_sequence)
    reference = sim(n_qubits, seed_seq)
    seed_metrics = _evaluate(seed_seq, n_qubits, reference, sim)
    cascade: List[CandidateRecord] = []

    def eval_strategies(strats: Sequence[Strategy], stage: str) -> CandidateRecord:
        best: Optional[CandidateRecord] = None
        for strat in strats:
            candidate = _run_strategy(seed_seq if stage == "12" else best.sequence, strat) \
                if stage != "12" else _run_strategy(seed_seq, strat)
            metrics = _evaluate(candidate, n_qubits, reference, sim)
            score = metrics.j(weights)
            rec = CandidateRecord(strat.name, stage, metrics, score, candidate)
            cascade.append(rec)
            if best is None or score > best.score:
                best = rec
        assert best is not None
        return best

    # Stage 1 — 12-point global family search.
    best_12 = eval_strategies(_global_families(), stage="12")

    # Stage 2 — 7-point refinement on the best family.
    best_7 = _refine_stage(
        best_12, _refinements(best_12.strategy), n_qubits, reference, sim, weights, cascade
    )

    # Stage 3 — 3-point precision tuning on the best refinement.
    best_3 = _refine_stage(
        best_7, _precision(best_7.strategy), n_qubits, reference, sim, weights, cascade, stage="3"
    )

    # Stage 4 — 1-point lock. Apply the hierarchical-consistency bonus before
    # declaring the winner so we reward candidates that stayed strong across
    # every stage, not one-off lucky rewrites.
    history = {
        "12": best_12.score,
        "7": best_7.score,
        "3": best_3.score,
    }
    bonus = weights.w_hierarchical * (history["12"] + history["7"] + history["3"]) / 3.0
    winner = CandidateRecord(
        strategy=best_3.strategy,
        stage="1",
        metrics=best_3.metrics,
        score=best_3.score + bonus,
        sequence=best_3.sequence,
    )
    cascade.append(winner)
    return AegisResult(
        seed_metrics=seed_metrics,
        winner=winner,
        cascade=cascade,
        weights=weights,
    )


def _refine_stage(
    parent: CandidateRecord,
    strats: Sequence[Strategy],
    n_qubits: int,
    reference: np.ndarray,
    sim: Callable[[int, GateSeq], np.ndarray],
    weights: ScoreWeights,
    cascade: List[CandidateRecord],
    stage: str = "7",
) -> CandidateRecord:
    best: Optional[CandidateRecord] = None
    for strat in strats:
        candidate = _run_strategy(parent.sequence, strat)
        metrics = _evaluate(candidate, n_qubits, reference, sim)
        score = metrics.j(weights)
        rec = CandidateRecord(strat.name, stage, metrics, score, candidate)
        cascade.append(rec)
        if best is None or score > best.score:
            best = rec
    # If no refinement improved over the parent, keep the parent.
    if best is None or best.score < parent.score:
        rec = CandidateRecord(
            strategy=f"{parent.strategy}|{stage}-hold",
            stage=stage,
            metrics=parent.metrics,
            score=parent.score,
            sequence=parent.sequence,
        )
        cascade.append(rec)
        return rec
    return best


__all__ = [
    "AegisResult",
    "CandidateRecord",
    "Metrics",
    "ScoreWeights",
    "aegis_ion_nested",
    "circuit_depth",
    "count_2q",
]
