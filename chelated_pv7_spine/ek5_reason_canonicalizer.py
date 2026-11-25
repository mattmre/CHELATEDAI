r"""Lazy audit-export reason & cycle canonicalizer for the EK5 v3 stratified store (Wave 16 / #48, 6A).

Problem
-------
``StratifiedEvidenceStore`` maintains ``val`` / ``cur`` / ``eligibility`` (the *semantic
dispositions*) incrementally, and also maintains two *telemetry* fields along the way:
``pos_cyc`` (positive-cycle membership) and ``reason`` (the justification string).

The incremental engine is deliberately lazy about telemetry.  Stratum 2 uses Pitfall-A/B
cone pruning: a node whose truth value did not change, or whose external shield counter
``sigma`` is still positive, is *not* re-examined when an edge is withdrawn or an
authority ceiling moves.  The WFS value stays exactly right (that is what the shield
guarantees), but the *explanation* attached to it is whatever the last examined mutation
left behind.  A node that was ``F`` because of an unfounded positive cycle can keep
``pos_cyc=True`` / ``UNFOUNDED_CYCLE`` after the cycle is broken but the node stays ``F``
through another refuter; conversely a node can be ``F`` through a cycle that a later
batch created while an older ``REFUTED`` string survives.  A from-scratch rebuild
examines every node once, so it reports the explanation of the *final* graph.  The result
is "same disposition, different ``reason``/``pos_cyc``" between an incrementally built
store and a fresh rebuild of the same log -- harmless for admission, but it makes audit
exports non-reproducible and breaks byte-level diffing of audit trails.

Approach
--------
``export_canonical_ek5_audit_trace(store)`` is evaluated lazily, only at export time, so
the write path pays nothing (no extra bookkeeping inside ``apply_mutation``).  It takes
the incrementally maintained *semantic* state it can trust (``val`` per node, plus the
authority-qualified edge sets ``refs`` / ``sup`` / ``lsup`` / ``deps_v`` / ``deps_c`` and
the mandatory-parent-missing set) and re-derives *everything explanatory* from the final
graph:

* ``pos_cyc``: membership in a non-trivial strongly-connected component of ``deps_v``
  (full structural audit, not the engine's dirty-region approximation).
* quarantine sets for ``SUPERSEDE`` / ``VERSION_LINEAGE_SUPERSEDE`` cycles, recomputed
  from the final edge sets and ``val`` rather than read from the incremental caches.
* ``cur`` and the canonical ``reason`` via one global bottom-up pass.  The derived
  ``cur`` / ``eligibility`` are cross-checked against the store's own; any disagreement
  is counted in ``semantic_mismatches`` (a non-zero value means the *semantic* engine
  state is inconsistent, which is a bug, not a telemetry difference).
* a deterministic minimal justification witness per node (lexicographically smallest
  contributing ids), so the exported explanation is a pure function of the final state.

Distinct epistemic causes never share a canonical reason: ``SUPERSEDED`` (a current
superseder), ``REFUTED`` (a true refuter or a false mandatory parent),
``UNFOUNDED_CYCLE`` (positive dependency cycle) and ``EPISTEMIC_CONTESTED`` (an
undecided peer conflict) keep separate reason strings and separate ``cause`` classes
(``SUPERSEDED`` / ``REFUTED`` / ``UNFOUNDED_POSITIVE_CYCLE`` / ``PEER_CONFLICT_DISPUTED``).

Cost
----
One Tarjan pass over ``deps_v``, one over the lineage graph, one over the supersede graph,
one over ``deps_c`` and a per-node witness scan: O(V + E) with small constants.  On the
small stores exercised by the tests (tens of nodes) this is far below 5 ms; the trace
records its own ``export_ms``.  It is *not* sub-5 ms for arbitrarily large stores -- the
claim is "zero overhead on incremental writes, linear-time export", measured at the sizes
in the tests.

Also provided: ``reference_fresh_rebuild`` (the oracle: replay the visible log into a new
store exactly like ``StratifiedEvidenceStore.as_of``) and ``compare_with_fresh_rebuild``.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from typing import Any, Dict, FrozenSet, List, Mapping, Optional, Set, Tuple

from ek5_v3_stratified import StratifiedEvidenceStore

__all__ = [
    "CAUSE_CLASSES",
    "CanonicalNode",
    "CanonicalAuditTrace",
    "RebuildComparison",
    "canonical_cause",
    "export_canonical_ek5_audit_trace",
    "reference_fresh_rebuild",
    "compare_with_fresh_rebuild",
]

_T, _U, _F = "T", "U", "F"

#: EK5 reason string -> canonical epistemic cause class.  Distinct causes stay distinct.
_REASON_TO_CAUSE: Dict[str, str] = {
    "CURRENT": "CURRENT",
    "REFUTED": "REFUTED",
    "MISSING_MANDATORY_PARENT": "MISSING_MANDATORY_PARENT",
    "UNFOUNDED_CYCLE": "UNFOUNDED_POSITIVE_CYCLE",
    "SUPERSEDED": "SUPERSEDED",
    "SUPERSEDED_ANCESTOR": "SUPERSEDED",
    "EPISTEMIC_CONTESTED": "PEER_CONFLICT_DISPUTED",
    "TEMPORAL_CONTESTED": "TEMPORAL_DISPUTED",
    "TEMPORAL_ANCESTOR_CONTESTED": "TEMPORAL_DISPUTED",
}
CAUSE_CLASSES: FrozenSet[str] = frozenset(_REASON_TO_CAUSE.values())


def canonical_cause(reason: str) -> str:
    """Map an EK5 ``reason`` string to its canonical epistemic cause class."""
    try:
        return _REASON_TO_CAUSE[reason]
    except KeyError as exc:  # pragma: no cover - guarded by the engine's closed vocabulary
        raise ValueError(f"unknown EK5 reason {reason!r}") from exc


# ---------------------------------------------------------------------------
# Minimal local Tarjan (kept private so this module does not depend on the
# engine's private helpers, and so the canonical pass is independently derived).
# ---------------------------------------------------------------------------
def _sccs(nodes: List[str], adj: Mapping[str, Any]) -> List[List[str]]:
    """Strongly-connected components restricted to ``nodes``; sinks are emitted first."""
    node_set = set(nodes)
    index: Dict[str, int] = {}
    low: Dict[str, int] = {}
    on_stack: Set[str] = set()
    stack: List[str] = []
    out: List[List[str]] = []
    counter = 0
    for start in nodes:
        if start in index:
            continue
        index[start] = low[start] = counter
        counter += 1
        stack.append(start)
        on_stack.add(start)
        work: List[Tuple[str, List[str], int]] = [
            (start, sorted(w for w in adj.get(start, ()) if w in node_set), 0)
        ]
        while work:
            v, nbrs, pos = work[-1]
            if pos < len(nbrs):
                work[-1] = (v, nbrs, pos + 1)
                w = nbrs[pos]
                if w not in index:
                    index[w] = low[w] = counter
                    counter += 1
                    stack.append(w)
                    on_stack.add(w)
                    work.append((w, sorted(z for z in adj.get(w, ()) if z in node_set), 0))
                elif w in on_stack and index[w] < low[v]:
                    low[v] = index[w]
            else:
                work.pop()
                if low[v] == index[v]:
                    comp: List[str] = []
                    while True:
                        w = stack.pop()
                        on_stack.discard(w)
                        comp.append(w)
                        if w == v:
                            break
                    out.append(comp)
                if work:
                    parent = work[-1][0]
                    if low[v] < low[parent]:
                        low[parent] = low[v]
    return out


def _is_cyclic(comp: List[str], adj: Mapping[str, Any]) -> bool:
    return len(comp) >= 2 or (len(comp) == 1 and comp[0] in adj.get(comp[0], ()))


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class CanonicalNode:
    """One node of the canonical audit trace."""

    event_id: str
    val: str
    cur: str
    eligibility: str
    reason: str
    cause: str
    pos_cyc: bool
    witness: Tuple[str, ...]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "val": self.val,
            "cur": self.cur,
            "eligibility": self.eligibility,
            "reason": self.reason,
            "cause": self.cause,
            "pos_cyc": self.pos_cyc,
            "witness": list(self.witness),
        }


@dataclass(frozen=True)
class CanonicalAuditTrace:
    """Deterministic audit export; ``export_ms`` is excluded from the digest."""

    nodes: Tuple[CanonicalNode, ...]
    positive_cycles: Tuple[Tuple[str, ...], ...]
    supersede_cycles: Tuple[Tuple[str, ...], ...]
    lineage_cycles: Tuple[Tuple[str, ...], ...]
    semantic_mismatches: Tuple[str, ...]
    export_ms: float

    def by_id(self) -> Dict[str, CanonicalNode]:
        return {n.event_id: n for n in self.nodes}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "nodes": [n.to_dict() for n in self.nodes],
            "positive_cycles": [list(c) for c in self.positive_cycles],
            "supersede_cycles": [list(c) for c in self.supersede_cycles],
            "lineage_cycles": [list(c) for c in self.lineage_cycles],
            "semantic_mismatches": list(self.semantic_mismatches),
        }

    def digest(self) -> str:
        blob = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class RebuildComparison:
    """Differences between a store (and its canonical export) and a fresh rebuild."""

    node_count: int
    #: store vs fresh on val/cur/eligibility (must be 0 for a correct engine)
    semantic_diffs: Tuple[str, ...]
    #: canonical export vs fresh rebuild on reason / pos_cyc (must be 0)
    canonical_reason_diffs: Tuple[str, ...]
    canonical_pos_cyc_diffs: Tuple[str, ...]
    #: raw incremental telemetry vs fresh rebuild (may be non-zero: Pitfall-B pruning)
    raw_reason_diffs: Tuple[str, ...]
    raw_pos_cyc_diffs: Tuple[str, ...]
    #: canonical cur/eligibility vs the store's own (must be 0)
    canonical_semantic_mismatches: Tuple[str, ...]


# ---------------------------------------------------------------------------
# Canonical derivation
# ---------------------------------------------------------------------------
def export_canonical_ek5_audit_trace(store: StratifiedEvidenceStore) -> CanonicalAuditTrace:
    """Lazily derive canonical ``reason`` / ``pos_cyc`` (and witnesses) for ``store``.

    Reads only the store's semantic state and active edge sets; never mutates it and
    never reads the incrementally maintained ``reason`` / ``pos_cyc`` / quarantine caches.
    """
    t0 = time.perf_counter()
    ids: List[str] = sorted(store.events)
    val: Dict[str, str] = {v: store.val[v] for v in ids}
    missing: Set[str] = set(store._missing_dep_nodes) & set(ids)

    # --- canonical positive-cycle membership: non-trivial SCCs of D_V --------------
    dv_cycles: List[Tuple[str, ...]] = []
    pos_cyc: Dict[str, bool] = {v: False for v in ids}
    comp_of_cycle: Dict[str, Tuple[str, ...]] = {}
    for comp in _sccs(ids, store.deps_v):
        if _is_cyclic(comp, store.deps_v):
            key = tuple(sorted(comp))
            dv_cycles.append(key)
            for u in comp:
                pos_cyc[u] = True
                comp_of_cycle[u] = key
    dv_cycles.sort()

    # --- quarantined SUPERSEDE cycles (sources that are F do not forward) ----------
    active_sup_out: Dict[str, Set[str]] = {
        u: {w for w in store.sup_out.get(u, ()) if w in store.events}
        for u in ids
        if val[u] != _F
    }
    q_sup_nodes: Set[str] = set()
    sup_cycles: List[Tuple[str, ...]] = []
    for comp in _sccs(ids, active_sup_out):
        if _is_cyclic(comp, active_sup_out):
            sup_cycles.append(tuple(sorted(comp)))
            q_sup_nodes.update(comp)
    sup_cycles.sort()

    # --- lineage condensation DAG ---------------------------------------------------
    lineage_nodes = sorted(
        u for u in ids if store.lsup.get(u) or store.lsup_out.get(u)
    )
    lsup_out = {u: {w for w in store.lsup_out.get(u, ()) if w in store.events} for u in lineage_nodes}
    lsup_sccs = _sccs(lineage_nodes, lsup_out)
    scc_id: Dict[str, int] = {}
    quarantined_lineage: Set[int] = set()
    lineage_cycles: List[Tuple[str, ...]] = []
    for idx, comp in enumerate(lsup_sccs):
        for u in comp:
            scc_id[u] = idx
        if _is_cyclic(comp, lsup_out) and any(val[u] != _F for u in comp):
            quarantined_lineage.add(idx)
            lineage_cycles.append(tuple(sorted(comp)))
    lineage_cycles.sort()
    has_def = [False] * len(lsup_sccs)
    has_poss = [False] * len(lsup_sccs)
    # Tarjan emits sinks first; superseders are sources, so walk in reverse.
    for idx in reversed(range(len(lsup_sccs))):
        comp = lsup_sccs[idx]
        if idx in quarantined_lineage:
            for u in comp:
                for w in lsup_out.get(u, ()):
                    if scc_id.get(w, idx) != idx:
                        has_poss[scc_id[w]] = True
            continue
        comp_has_t = has_def[idx] or any(val[u] == _T for u in comp)
        comp_has_u = has_poss[idx] or any(val[u] == _U for u in comp)
        for u in comp:
            for w in lsup_out.get(u, ()):
                if scc_id.get(w, idx) != idx:
                    if comp_has_t:
                        has_def[scc_id[w]] = True
                    if comp_has_u:
                        has_poss[scc_id[w]] = True

    # --- local temporal blockers ----------------------------------------------------
    local_cur: Dict[str, str] = {}
    local_reason: Dict[str, str] = {}
    for v in ids:
        vv = val[v]
        if vv == _F:
            local_cur[v] = _F
            if v in missing:
                local_reason[v] = "MISSING_MANDATORY_PARENT"
            elif pos_cyc[v]:
                local_reason[v] = "UNFOUNDED_CYCLE"
            else:
                local_reason[v] = "REFUTED"
            continue
        b_ext = _F
        b_cyc = _F
        for j in store.sup.get(v, ()):
            vj = val.get(j, _F)
            if vj == _F:
                continue
            if j in q_sup_nodes:
                b_cyc = _U
            elif vj == _T:
                b_ext = _T
            elif b_ext != _T:
                b_ext = _U
        if v in scc_id:
            sid = scc_id[v]
            if has_def[sid]:
                b_ext = _T
            elif has_poss[sid] or sid in quarantined_lineage:
                b_cyc = _U
        if b_ext == _T:
            local_cur[v] = _F
            local_reason[v] = "SUPERSEDED"
        elif vv == _U:
            local_cur[v] = _U
            local_reason[v] = "EPISTEMIC_CONTESTED"
        elif b_ext == _U or b_cyc == _U:
            local_cur[v] = _U
            local_reason[v] = "TEMPORAL_CONTESTED"
        else:
            local_cur[v] = _T
            local_reason[v] = "CURRENT"

    # --- D_C propagation, parents first ---------------------------------------------
    deps_c = {u: {d for d in store.deps_c.get(u, ()) if d in store.events} for u in ids}
    for comp in _sccs(ids, deps_c):
        if _is_cyclic(comp, deps_c):
            for u in comp:
                local_cur[u] = _F
                local_reason[u] = "UNFOUNDED_CYCLE"
            continue
        u = comp[0]
        if local_cur[u] == _F:
            continue
        for d in sorted(deps_c[u]):
            cd = local_cur[d]
            if cd == _F:
                local_cur[u] = _F
                local_reason[u] = "SUPERSEDED_ANCESTOR"
                break
            if cd == _U and local_cur[u] == _T:
                local_cur[u] = _U
                local_reason[u] = "TEMPORAL_ANCESTOR_CONTESTED"

    # --- eligibility, cross-check against the store, witnesses ----------------------
    mismatches: List[str] = []
    nodes: List[CanonicalNode] = []
    for v in ids:
        if val[v] == _T and local_cur[v] == _T:
            elig = "ADMIT"
        elif val[v] == _F or local_cur[v] == _F:
            elig = "AUDIT_ONLY"
        else:
            elig = "CONTESTED"
        if store.cur.get(v) != local_cur[v]:
            mismatches.append(f"{v}:cur store={store.cur.get(v)} canonical={local_cur[v]}")
        if store.eligibility.get(v) != elig:
            mismatches.append(f"{v}:eligibility store={store.eligibility.get(v)} canonical={elig}")
        reason = local_reason[v]
        witness = _witness(store, v, reason, val, local_cur, q_sup_nodes, comp_of_cycle)
        nodes.append(
            CanonicalNode(
                event_id=v,
                val=val[v],
                cur=local_cur[v],
                eligibility=elig,
                reason=reason,
                cause=canonical_cause(reason),
                pos_cyc=pos_cyc[v],
                witness=witness,
            )
        )
    export_ms = (time.perf_counter() - t0) * 1000.0
    return CanonicalAuditTrace(
        nodes=tuple(nodes),
        positive_cycles=tuple(dv_cycles),
        supersede_cycles=tuple(sup_cycles),
        lineage_cycles=tuple(lineage_cycles),
        semantic_mismatches=tuple(mismatches),
        export_ms=export_ms,
    )


def _witness(
    store: StratifiedEvidenceStore,
    v: str,
    reason: str,
    val: Mapping[str, str],
    cur: Mapping[str, str],
    q_sup_nodes: Set[str],
    comp_of_cycle: Mapping[str, Tuple[str, ...]],
) -> Tuple[str, ...]:
    """Lexicographically-minimal contributing ids for ``reason`` (pure function of state)."""
    if reason == "CURRENT" or reason == "MISSING_MANDATORY_PARENT":
        return ()
    if reason == "UNFOUNDED_CYCLE":
        if v in comp_of_cycle:
            return comp_of_cycle[v]
        # D_C-cycle nodes are always D_V-cycle nodes (D_C is a subset of D_V).
        return (v,)
    if reason == "REFUTED":
        refuters = sorted(a for a in store.refs.get(v, ()) if val.get(a) == _T)
        if refuters:
            return (refuters[0],)
        false_parents = sorted(d for d in store.deps_v.get(v, ()) if val.get(d) == _F)
        return (false_parents[0],) if false_parents else ()
    if reason == "SUPERSEDED":
        supers = sorted(j for j in store.sup.get(v, ()) if val.get(j) == _T and j not in q_sup_nodes)
        if supers:
            return (supers[0],)
        lineage = sorted(j for j in store.lsup.get(v, ()))
        return (lineage[0],) if lineage else ()
    if reason == "SUPERSEDED_ANCESTOR":
        parents = sorted(d for d in store.deps_c.get(v, ()) if cur.get(d) == _F)
        return (parents[0],) if parents else ()
    if reason == "EPISTEMIC_CONTESTED":
        peers = sorted(a for a in store.refs.get(v, ()) if val.get(a) == _U)
        if peers:
            return (peers[0],)
        undecided = sorted(d for d in store.deps_v.get(v, ()) if val.get(d) == _U)
        return (undecided[0],) if undecided else ()
    if reason == "TEMPORAL_ANCESTOR_CONTESTED":
        parents = sorted(d for d in store.deps_c.get(v, ()) if cur.get(d) == _U)
        return (parents[0],) if parents else ()
    # TEMPORAL_CONTESTED
    contenders = sorted(
        j for j in store.sup.get(v, ()) if val.get(j) == _U or (j in q_sup_nodes and val.get(j) != _F)
    )
    if contenders:
        return (contenders[0],)
    lineage = sorted(j for j in store.lsup.get(v, ()) if val.get(j) != _F)
    return (lineage[0],) if lineage else ()


# ---------------------------------------------------------------------------
# Fresh-rebuild oracle & comparison
# ---------------------------------------------------------------------------
def reference_fresh_rebuild(store: StratifiedEvidenceStore) -> StratifiedEvidenceStore:
    """Replay ``store``'s visible log into a brand-new store (the from-scratch oracle).

    Mirrors ``StratifiedEvidenceStore.as_of`` at the unbounded cutoff: imported cycles are
    tolerated (``strict_dag_supersede=False``) and every active attestation is replayed in
    one atomic batch.
    """
    fresh = StratifiedEvidenceStore(
        strict_dag_supersede=False,
        buffer_pending=False,
        exact_min_rank=store.exact_min_rank,
        max_dirty_edges=store.max_dirty_edges,
    )
    fresh._missing_dep_nodes = set(store._missing_dep_nodes)
    active = [a for a in store.attestations.values() if a.active]
    fresh.apply_mutation(list(store.events.values()), active)
    return fresh


def compare_with_fresh_rebuild(
    store: StratifiedEvidenceStore,
    trace: Optional[CanonicalAuditTrace] = None,
) -> RebuildComparison:
    """Compare ``store`` and its canonical export against a from-scratch rebuild."""
    fresh = reference_fresh_rebuild(store)
    if trace is None:
        trace = export_canonical_ek5_audit_trace(store)
    canon = trace.by_id()
    semantic: List[str] = []
    c_reason: List[str] = []
    c_cyc: List[str] = []
    r_reason: List[str] = []
    r_cyc: List[str] = []
    for v in sorted(store.events):
        for field in ("val", "cur", "eligibility"):
            a, b = getattr(store, field).get(v), getattr(fresh, field).get(v)
            if a != b:
                semantic.append(f"{v}:{field} store={a} fresh={b}")
        node = canon[v]
        if node.reason != fresh.reason.get(v):
            c_reason.append(f"{v}: canonical={node.reason} fresh={fresh.reason.get(v)}")
        if node.pos_cyc != bool(fresh.pos_cyc.get(v)):
            c_cyc.append(f"{v}: canonical={node.pos_cyc} fresh={fresh.pos_cyc.get(v)}")
        if store.reason.get(v) != fresh.reason.get(v):
            r_reason.append(f"{v}: store={store.reason.get(v)} fresh={fresh.reason.get(v)}")
        if bool(store.pos_cyc.get(v)) != bool(fresh.pos_cyc.get(v)):
            r_cyc.append(f"{v}: store={store.pos_cyc.get(v)} fresh={fresh.pos_cyc.get(v)}")
    return RebuildComparison(
        node_count=len(store.events),
        semantic_diffs=tuple(semantic),
        canonical_reason_diffs=tuple(c_reason),
        canonical_pos_cyc_diffs=tuple(c_cyc),
        raw_reason_diffs=tuple(r_reason),
        raw_pos_cyc_diffs=tuple(r_cyc),
        canonical_semantic_mismatches=trace.semantic_mismatches,
    )
