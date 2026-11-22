r"""EK5 v3 — Stratified K=4 Two-Axis Well-Founded Evidence Engine + EM-3 Telemetry + EM-1 M4e.

Implements the exact 3/3 unanimous R3 Tri-Model Council specification:
  - Stratum 0: Bitemporal visibility `(valid_cutoff, tx_cutoff)` + append-only event/attestation log
               with edge multiplicity counts, fail-closed endpoint validation (or `buffer_pending`),
               and `as_of(valid_cutoff, tx_cutoff)` historical snapshot/rewind.
  - Stratum 1: Capability-scoped anti-laundering effective authority ceiling:
               `alpha_tilde(v, c) = min(alpha_raw(v, c), min_{d in D_V(v)} alpha_tilde(d, c))`
               with positive D_V-cycles and missing mandatory parents -> 0, `CITES_AUDIT` outside D_V,
               and symmetric sword-and-shield defeat qualification `alpha_tilde(u, dom(w)) >= alpha_tilde(w, dom(w))`.
  - Stratum 2: Option 1A Epistemic Validity WFS — iterative Tarjan positive-SCC collapse (`pos_cyc=True, rho=0`)
               + Plain-FIFO boundary-aware Lyapunov rank + Pitfall A/B cone pruning (deletion-first net-cancelled
               atomic batches, asymmetric `rho == 0` rule) + checkpoint rank compaction + `exact_min_rank` mode.
  - Stratum 3: Temporal Currency — Direct-B_i for ordinary `SUPERSEDE`, condensation-DAG ancestry for declared
               `(tenant, entity, single_valued_slot, lineage_id)` version lineages, two-mode positive dependencies
               (`COPY_OF`/`DERIVES_CURRENT` in D_C vs `AMENDS`/`DERIVES_SNAPSHOT` in D_V \ D_C), native write
               cycle rejection (`SupersedeCycleError`), and imported cycle quarantine with outgoing quarantine cap
               (`C_Y = U`) & non-forwarding cycle quarantine (`C_B = U`).
  - EM-3: Non-blocking certificate-based diagnostic telemetry (`unfounded_cycle_nodes`, `blocked_authority_edges`,
          `quarantined_supersede_cycles`) + optional `full_structural_scc_audit()`.
  - EM-1 (`M4e`): Minimal-pair contradiction vs. echo split adapter (header stripping, normalized numeric/unit/
                  time/alphanum, negation/antonym polarity, capitalized proper-noun entity diff guard).
  - Independent Textbook Alternating-Reduct WFS Oracle (`compute_alternating_wfs_oracle`) using
    `T* = lfp(Gamma^2)` and `F* = V \ Gamma(T*)` with zero shared propagation code.
"""

from __future__ import annotations

import heapq
import re
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

T, U, F = "T", "U", "F"
INF = 10 ** 18

V3_VALID_EDGE_TYPES: FrozenSet[str] = frozenset({
    "REFUTE",
    "SUPERSEDE",
    "VERSION_LINEAGE_SUPERSEDE",
    "COPY_OF",
    "DERIVES_CURRENT",
    "DERIVES_FROM",      # legacy -> defaults to DERIVES_CURRENT (D_C)
    "AMENDS",            # D_V \ D_C
    "DERIVES_SNAPSHOT",  # D_V \ D_C
    "CITES_AUDIT",       # non-epistemic reference (outside D_V and D_C)
    "SUPPORT",           # post-admission corroboration only
})

DEP_CUR_TYPES: FrozenSet[str] = frozenset({"COPY_OF", "DERIVES_CURRENT", "DERIVES_FROM"})
DEP_EP_ONLY_TYPES: FrozenSet[str] = frozenset({"AMENDS", "DERIVES_SNAPSHOT"})
DEP_ALL_TYPES: FrozenSet[str] = DEP_CUR_TYPES | DEP_EP_ONLY_TYPES


class SupersedeCycleError(ValueError):
    """Raised when an online native write attempts to close a SUPERSEDE cycle."""


@dataclass(frozen=True)
class StratifiedEvent:
    event_id: str
    entity: str = "default"
    slot: str = "default"
    claim_value: str = ""
    text: str = ""
    valid_time: int = 0
    tx_time: int = 0
    valid_end: Optional[int] = None
    tenant: str = "default"
    domain: str = "general"
    authority_tier: int = 0


@dataclass(frozen=True)
class EdgeAttestation:
    attestation_id: str
    src_id: str
    dst_id: str
    edge_type: str
    valid_time: int = 0
    tx_time: int = 0
    lineage_id: Optional[str] = None
    active: bool = True


@dataclass(frozen=True)
class MutationDelta:
    inserted: Dict[str, str]
    transitioned: Dict[str, Tuple[str, str]]
    removed: Dict[str, str]
    rank_changed: Dict[str, Tuple[int, int]]
    pending_buffered: Tuple[str, ...] = ()
    work_stats: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EpistemicTelemetry:
    """EM-3 non-blocking diagnostic telemetry channel (zero feedback into WFS)."""

    certificate_based: bool
    unfounded_cycle_nodes: Tuple[str, ...]
    blocked_authority_edges: Tuple[Tuple[str, str, str], ...]
    quarantined_supersede_cycles: Tuple[Tuple[str, ...], ...]


# ---------------------------------------------------------------------------
# Iterative Tarjan SCC (Zero Recursion Limit Risk on 1M-Node Chains)
# ---------------------------------------------------------------------------
def _iterative_tarjan_sccs(
    nodes: Iterable[str],
    adj: Mapping[str, Iterable[str]],
    subset: Optional[Set[str]] = None,
) -> List[List[str]]:
    """Return all SCCs inside `nodes` (restricted to `subset` if provided) in linear time."""
    node_set = subset if subset is not None else set(nodes)
    index_counter = 0
    index: Dict[str, int] = {}
    lowlink: Dict[str, int] = {}
    on_stack: Set[str] = set()
    tarjan_stack: List[str] = []
    sccs: List[List[str]] = []

    for start in node_set:
        if start in index:
            continue
        call_stack: List[Tuple[str, List[str], int]] = []
        index[start] = lowlink[start] = index_counter
        index_counter += 1
        tarjan_stack.append(start)
        on_stack.add(start)
        nbrs = [w for w in adj.get(start, ()) if w in node_set]
        call_stack.append((start, nbrs, 0))

        while call_stack:
            v, v_nbrs, pos = call_stack[-1]
            if pos < len(v_nbrs):
                w = v_nbrs[pos]
                call_stack[-1] = (v, v_nbrs, pos + 1)
                if w not in index:
                    index[w] = lowlink[w] = index_counter
                    index_counter += 1
                    tarjan_stack.append(w)
                    on_stack.add(w)
                    w_nbrs = [z for z in adj.get(w, ()) if z in node_set]
                    call_stack.append((w, w_nbrs, 0))
                elif w in on_stack:
                    if index[w] < lowlink[v]:
                        lowlink[v] = index[w]
            else:
                call_stack.pop()
                if lowlink[v] == index[v]:
                    comp: List[str] = []
                    while True:
                        w = tarjan_stack.pop()
                        on_stack.remove(w)
                        comp.append(w)
                        if w == v:
                            break
                    sccs.append(comp)
                if call_stack:
                    parent = call_stack[-1][0]
                    if lowlink[v] < lowlink[parent]:
                        lowlink[parent] = lowlink[v]
    return sccs


def _nontrivial_scc_nodes(
    nodes: Iterable[str],
    adj: Mapping[str, Set[str]],
    subset: Optional[Set[str]] = None,
) -> Set[str]:
    node_set = subset if subset is not None else set(nodes)
    cyc: Set[str] = set()
    for comp in _iterative_tarjan_sccs(node_set, adj, subset=node_set):
        if len(comp) >= 2:
            cyc.update(comp)
        elif len(comp) == 1:
            u = comp[0]
            if u in adj.get(u, ()):
                cyc.add(u)
    return cyc


# ---------------------------------------------------------------------------
# Independent Textbook Alternating-Reduct WFS Oracle (Zero Shared Code!)
# ---------------------------------------------------------------------------
def compute_alternating_wfs_oracle(
    nodes: Iterable[str],
    refs_in: Mapping[str, Iterable[str]],
    deps_of: Mapping[str, Iterable[str]],
) -> Dict[str, str]:
    """Independent Van Gelder-Ross-Schlipf alternating-fixpoint WFS oracle.

    For single-conjunctive normal rules `v <- AND_{d in D(v)} d AND_{u in R(v)} not u`:
      - `Gamma(S)` = least model of the positive Horn reduct obtained by removing rules
        whose negative body intersects `S`.
      - `T* = lfp(Gamma^2)` (starting from empty set),
      - `F* = V \\ Gamma(T*)` (exact R2-verified false set; NOT `V \\ Gamma(V \\ T*)`!).
      - Remaining nodes in `Gamma(T*) \\ T*` are `U` (`CONTESTED`).
    """
    v_set = set(nodes)
    r_map = {v: {u for u in refs_in.get(v, ()) if u in v_set} for v in v_set}
    d_map = {v: {d for d in deps_of.get(v, ()) if d in v_set} for v in v_set}

    def gamma(assumed_true: Set[str]) -> Set[str]:
        # Positive Horn reduct: v is derivable iff no r in R(v) is in assumed_true AND all d in D(v) derived
        eligible = {v for v in v_set if not (r_map[v] & assumed_true)}
        derived: Set[str] = set()
        changed = True
        while changed:
            changed = False
            for v in eligible:
                if v not in derived and d_map[v].issubset(derived):
                    derived.add(v)
                    changed = True
        return derived

    t_set: Set[str] = set()
    while True:
        u_poss = gamma(t_set)
        t_next = gamma(u_poss)
        if t_next == t_set:
            break
        t_set = t_next

    poss_true = gamma(t_set)
    out: Dict[str, str] = {}
    for v in v_set:
        if v in t_set:
            out[v] = T
        elif v not in poss_true:
            out[v] = F
        else:
            out[v] = U
    return out


# ---------------------------------------------------------------------------
# EK5 v3: Stratified K=4 Evidence Store (Option 1A + Stratum 2 Currency + EM-3)
# ---------------------------------------------------------------------------
class StratifiedEvidenceStore:
    """EK5 v3 Stratified K=4 Two-Axis Well-Founded Evidence Store.

    Strata:
      0. Bitemporal visibility (`valid_cutoff`, `tx_cutoff`) + append-only multiplicity log
      1. Capability-scoped anti-laundering effective authority ceiling `alpha_tilde`
      2. Epistemic Validity WFS (Option 1A: Tarjan positive-SCC + Pitfall A/B + Plain-FIFO Lyapunov)
      3. Temporal Currency (Direct-B_i + Declared Version-Lineage + Quarantined Cycle Cap)
    """

    def __init__(
        self,
        *,
        strict_dag_supersede: bool = True,
        buffer_pending: bool = False,
        exact_min_rank: bool = False,
        max_dirty_edges: Optional[int] = None,
    ) -> None:
        self.strict_dag_supersede = strict_dag_supersede
        self.buffer_pending = buffer_pending
        self.exact_min_rank = exact_min_rank
        self.max_dirty_edges = max_dirty_edges

        # Append-only canonical log (for bitemporal as-of rewinds & crash replay)
        self.events: Dict[str, StratifiedEvent] = {}
        self.attestations: Dict[str, EdgeAttestation] = {}
        self.pending_attestations: List[EdgeAttestation] = []

        # Active live-view multiplicity counts & graphs (default latest snapshot)
        self._edge_mult: Dict[Tuple[str, str, str], int] = defaultdict(int)
        self._lineage_edges: Dict[Tuple[str, str], Set[str]] = defaultdict(set)  # (src, dst) -> lineage_ids

        # Raw visible topology (before authority filtering)
        self._raw_refs: Dict[str, Set[str]] = defaultdict(set)      # dst -> {src}
        self._raw_ref_out: Dict[str, Set[str]] = defaultdict(set)   # src -> {dst}
        self._raw_sup: Dict[str, Set[str]] = defaultdict(set)       # dst -> {src}
        self._raw_sup_out: Dict[str, Set[str]] = defaultdict(set)   # src -> {dst}
        self._raw_lsup: Dict[str, Set[str]] = defaultdict(set)      # dst -> {src}
        self._raw_lsup_out: Dict[str, Set[str]] = defaultdict(set)  # src -> {dst}
        self.deps_v: Dict[str, Set[str]] = defaultdict(set)         # child -> {parent} (D_V)
        self.dep_v_rev: Dict[str, Set[str]] = defaultdict(set)      # parent -> {child}
        self.deps_c: Dict[str, Set[str]] = defaultdict(set)         # child -> {parent} (D_C <= D_V)
        self.dep_c_rev: Dict[str, Set[str]] = defaultdict(set)      # parent -> {child}
        self.cites_audit: Dict[str, Set[str]] = defaultdict(set)    # auditor -> {cited} (outside D_V)

        # Stratum 1: Effective Authority Ceiling per node (for node's own domain + domain map)
        self.eff_auth: Dict[str, int] = {}

        # Authority-qualified Stratum 2/3 edges
        self.refs: Dict[str, Set[str]] = defaultdict(set)
        self.ref_out: Dict[str, Set[str]] = defaultdict(set)
        self.sup: Dict[str, Set[str]] = defaultdict(set)
        self.sup_out: Dict[str, Set[str]] = defaultdict(set)
        self.lsup: Dict[str, Set[str]] = defaultdict(set)
        self.lsup_out: Dict[str, Set[str]] = defaultdict(set)

        # Stratum 2: Option 1A Epistemic WFS state + Lyapunov rank + Pitfall A/B shield counters
        self.val: Dict[str, str] = {}
        self.rho: Dict[str, int] = {}
        self.pos_cyc: Dict[str, bool] = {}
        self.sigma: Dict[str, int] = {}

        # Stratum 3: Temporal Currency & final eligibility
        self.cur: Dict[str, str] = {}
        self.eligibility: Dict[str, str] = {}
        self.reason: Dict[str, str] = {}
        self._missing_dep_nodes: Set[str] = set()
        self._lsup_has_def_anc: Dict[str, bool] = {}
        self._lsup_has_poss_anc: Dict[str, bool] = {}
        self._q_sup_sccs: Dict[Tuple[str, ...], Set[str]] = {}
        self._q_sup_nodes: Set[str] = set()
        self._q_lsup_sccs: Dict[Tuple[str, ...], Set[str]] = {}
        self._q_lsup_nodes: Set[str] = set()

        # EM-3 Non-blocking telemetry
        self._blocked_auth_edges: Set[Tuple[str, str, str]] = set()
        self._quarantined_sup_cycles: List[Tuple[str, ...]] = []
        self._auto_att_seq = 0

    def _effective_authority_for_domain(self, node_id: str, target_domain: str) -> int:
        ev = self.events[node_id]
        # Scoped capability check: cross-domain actors default to tier 0 unless general/matching
        if ev.domain != "general" and target_domain != "general" and ev.domain != target_domain:
            return 0
        return self.eff_auth.get(node_id, 0)

    def _recompute_authority_cone(self, seeds: Set[str]) -> Set[str]:
        """Recompute monotone foundation ceiling `eff_auth` on forward D_V cone; return changed nodes."""
        if not seeds:
            return set()
        cone: Set[str] = set(seeds)
        q = deque(seeds)
        while q:
            u = q.popleft()
            for w in self.dep_v_rev.get(u, ()):
                if w not in cone:
                    cone.add(w)
                    q.append(w)

        cyc = _nontrivial_scc_nodes(cone, self.deps_v, subset=cone)
        old_eff = {u: self.eff_auth.get(u, 0) for u in cone}
        sccs = _iterative_tarjan_sccs(cone, self.deps_v, subset=cone)
        # Tarjan emits sinks (parents with no outgoing deps in cone) FIRST — exact bottom-up order!
        for comp in sccs:
            if len(comp) >= 2 or (len(comp) == 1 and comp[0] in cyc):
                for u in comp:
                    self.eff_auth[u] = 0
            else:
                u = comp[0]
                if u in self._missing_dep_nodes:
                    self.eff_auth[u] = 0
                    continue
                best = self.events[u].authority_tier
                for d in self.deps_v.get(u, ()):
                    if d not in self.events:
                        best = 0
                        break
                    val_d = self.eff_auth.get(d, 0)
                    if val_d < best:
                        best = val_d
                self.eff_auth[u] = best

        return {u for u in cone if self.eff_auth[u] != old_eff[u]}

    def _solve_stratum2_region(
        self,
        dirty: Set[str],
        *,
        force_exact: Optional[bool] = None,
    ) -> Tuple[Dict[str, Tuple[int, int]], int]:
        """Solve Option 1A positive-SCC + Lyapunov WFS on `dirty` with frozen boundary; update Pitfall B shields."""
        if not dirty:
            return {}, 0
        use_exact = self.exact_min_rank if force_exact is None else force_exact
        old_val = {u: self.val[u] for u in dirty}
        old_rho = {u: self.rho[u] for u in dirty}

        cyc_nodes = _nontrivial_scc_nodes(dirty, self.deps_v, subset=dirty)
        forced_false = cyc_nodes | (self._missing_dep_nodes & dirty)
        new_val: Dict[str, str] = {u: U for u in dirty}
        new_rho: Dict[str, int] = {u: INF for u in dirty}
        rem_sat: Dict[str, List[int]] = {}
        prop_edges = 0

        if use_exact:
            pq: List[Tuple[int, str, str]] = []
            for u in dirty:
                if u in forced_false:
                    heapq.heappush(pq, (0, u, F))
                    continue
                rem = 0
                best_f = INF
                max_r = -1
                for a in self.refs.get(u, ()):
                    prop_edges += 1
                    if a not in dirty:
                        if self.val[a] == T:
                            best_f = min(best_f, self.rho[a] + 1)
                            rem += 1
                        elif self.val[a] == F:
                            max_r = max(max_r, self.rho[a])
                        else:
                            rem += 1
                    else:
                        rem += 1
                for d in self.deps_v.get(u, ()):
                    prop_edges += 1
                    if d not in dirty:
                        if self.val[d] == F:
                            best_f = min(best_f, self.rho[d] + 1)
                            rem += 1
                        elif self.val[d] == T:
                            max_r = max(max_r, self.rho[d])
                        else:
                            rem += 1
                    else:
                        rem += 1
                rem_sat[u] = [rem, max_r]
                if best_f < INF:
                    heapq.heappush(pq, (best_f, u, F))
                elif rem == 0:
                    r0 = 0 if (not self.refs.get(u) and not self.deps_v.get(u)) else (max_r + 1)
                    heapq.heappush(pq, (r0, u, T))

            proc: Set[str] = set()
            while pq:
                r, u, st = heapq.heappop(pq)
                if u in proc:
                    continue
                proc.add(u)
                new_val[u] = st
                new_rho[u] = r
                for w in self.ref_out.get(u, ()):
                    prop_edges += 1
                    if w not in dirty or w in proc or w in forced_false:
                        continue
                    if st == T:
                        heapq.heappush(pq, (r + 1, w, F))
                    else:
                        rem_sat[w][0] -= 1
                        if r > rem_sat[w][1]:
                            rem_sat[w][1] = r
                        if rem_sat[w][0] == 0:
                            heapq.heappush(pq, (rem_sat[w][1] + 1, w, T))
                for w in self.dep_v_rev.get(u, ()):
                    prop_edges += 1
                    if w not in dirty or w in proc or w in forced_false:
                        continue
                    if st == F:
                        heapq.heappush(pq, (r + 1, w, F))
                    else:
                        rem_sat[w][0] -= 1
                        if r > rem_sat[w][1]:
                            rem_sat[w][1] = r
                        if rem_sat[w][0] == 0:
                            heapq.heappush(pq, (rem_sat[w][1] + 1, w, T))
        else:
            fifo: deque[str] = deque()
            for u in dirty:
                if u in forced_false:
                    new_val[u] = F
                    new_rho[u] = 0
                    fifo.append(u)
                    continue
                rem = 0
                best_f = INF
                max_r = -1
                for a in self.refs.get(u, ()):
                    prop_edges += 1
                    if a not in dirty:
                        if self.val[a] == T:
                            best_f = min(best_f, self.rho[a] + 1)
                            rem += 1
                        elif self.val[a] == F:
                            max_r = max(max_r, self.rho[a])
                        else:
                            rem += 1
                    else:
                        rem += 1
                for d in self.deps_v.get(u, ()):
                    prop_edges += 1
                    if d not in dirty:
                        if self.val[d] == F:
                            best_f = min(best_f, self.rho[d] + 1)
                            rem += 1
                        elif self.val[d] == T:
                            max_r = max(max_r, self.rho[d])
                        else:
                            rem += 1
                    else:
                        rem += 1
                rem_sat[u] = [rem, max_r]
                if best_f < INF:
                    new_val[u] = F
                    new_rho[u] = best_f
                    fifo.append(u)
                elif rem == 0:
                    new_val[u] = T
                    new_rho[u] = 0 if (not self.refs.get(u) and not self.deps_v.get(u)) else (max_r + 1)
                    fifo.append(u)

            proc = set()
            while fifo:
                u = fifo.popleft()
                if u in proc:
                    continue
                proc.add(u)
                vu, r = new_val[u], new_rho[u]
                for w in self.ref_out.get(u, ()):
                    prop_edges += 1
                    if w not in dirty or w in proc or w in forced_false:
                        continue
                    if vu == T and new_val[w] != F:
                        new_val[w] = F
                        new_rho[w] = r + 1
                        fifo.append(w)
                    elif vu == F and new_val[w] == U:
                        rem_sat[w][0] -= 1
                        if r > rem_sat[w][1]:
                            rem_sat[w][1] = r
                        if rem_sat[w][0] == 0:
                            new_val[w] = T
                            new_rho[w] = rem_sat[w][1] + 1
                            fifo.append(w)
                for w in self.dep_v_rev.get(u, ()):
                    prop_edges += 1
                    if w not in dirty or w in proc or w in forced_false:
                        continue
                    if vu == F and new_val[w] != F:
                        new_val[w] = F
                        new_rho[w] = r + 1
                        fifo.append(w)
                    elif vu == T and new_val[w] == U:
                        rem_sat[w][0] -= 1
                        if r > rem_sat[w][1]:
                            rem_sat[w][1] = r
                        if rem_sat[w][0] == 0:
                            new_val[w] = T
                            new_rho[w] = rem_sat[w][1] + 1
                            fifo.append(w)

        rank_changed: Dict[str, Tuple[int, int]] = {}
        for u in dirty:
            if old_rho[u] != new_rho[u]:
                rank_changed[u] = (old_rho[u], new_rho[u])
            self.val[u] = new_val[u]
            self.rho[u] = new_rho[u]
            self.pos_cyc[u] = u in cyc_nodes

        # Step 4: Pitfall B O(1) external shield delta + local shield recompute in D
        for u in dirty:
            for w in self.ref_out.get(u, ()):
                if w not in dirty and self.val[w] == F and not self.pos_cyc[w] and w not in self._missing_dep_nodes:
                    now_sh = int(self.val[u] == T and self.rho[u] < self.rho[w])
                    was_sh = int(old_val[u] == T and old_rho[u] < self.rho[w])
                    if now_sh != was_sh:
                        self.sigma[w] += now_sh - was_sh
            for w in self.dep_v_rev.get(u, ()):
                if w not in dirty and self.val[w] == F and not self.pos_cyc[w] and w not in self._missing_dep_nodes:
                    now_sh = int(self.val[u] == F and self.rho[u] < self.rho[w])
                    was_sh = int(old_val[u] == F and old_rho[u] < self.rho[w])
                    if now_sh != was_sh:
                        self.sigma[w] += now_sh - was_sh

        for w in dirty:
            if self.val[w] == F and not self.pos_cyc[w] and w not in self._missing_dep_nodes:
                self.sigma[w] = sum(
                    1 for a in self.refs.get(w, ()) if self.val[a] == T and self.rho[a] < self.rho[w]
                ) + sum(
                    1 for d in self.deps_v.get(w, ()) if self.val[d] == F and self.rho[d] < self.rho[w]
                )
            else:
                self.sigma[w] = 0

        return rank_changed, prop_edges

    def _apply_stratum2_batch(
        self,
        new_nodes: Set[str],
        raw_add_refs: Sequence[Tuple[str, str]],
        raw_del_refs: Sequence[Tuple[str, str]],
        raw_add_deps: Sequence[Tuple[str, str]],
        raw_del_deps: Sequence[Tuple[str, str]],
    ) -> Tuple[Set[str], Dict[str, Tuple[int, int]], Dict[str, Any]]:
        """Option 1A: Exact O(|Delta| + |D| + |E_inc(D)|) Positive-SCC + Pitfall A/B + Lyapunov WFS."""
        # 0. Net-cancel opposite active edges in the same atomic batch
        s_ar, s_dr = set(raw_add_refs), set(raw_del_refs)
        s_ad, s_dd = set(raw_add_deps), set(raw_del_deps)
        add_refs = s_ar - s_dr
        del_refs = s_dr - s_ar
        add_deps = s_ad - s_dd
        del_deps = s_dd - s_ad

        seeds: Set[str] = set()
        actual_del_refs: List[Tuple[str, str]] = []
        actual_del_deps: List[Tuple[str, str]] = []
        actual_add_refs: List[Tuple[str, str]] = []
        actual_add_deps: List[Tuple[str, str]] = []

        for u, w in del_refs:
            if u in self.refs.get(w, ()):
                was = self.val[w] == F and self.val[u] == T and self.rho[u] < self.rho[w]
                self.refs[w].remove(u)
                self.ref_out[u].discard(w)
                if was:
                    self.sigma[w] -= 1
                actual_del_refs.append((u, w))
        for w, d in del_deps:
            if d in self.deps_v.get(w, ()):
                was = self.val[w] == F and self.val[d] == F and self.rho[d] < self.rho[w]
                self.deps_v[w].remove(d)
                self.dep_v_rev[d].discard(w)
                if was:
                    self.sigma[w] -= 1
                actual_del_deps.append((w, d))
        for u, w in add_refs:
            if u not in self.refs.get(w, ()):
                self.refs[w].add(u)
                self.ref_out[u].add(w)
                if self.val[w] == F and self.val[u] == T and self.rho[u] < self.rho[w]:
                    self.sigma[w] += 1
                actual_add_refs.append((u, w))
        for w, d in add_deps:
            if d not in self.deps_v.get(w, ()):
                self.deps_v[w].add(d)
                self.dep_v_rev[d].add(w)
                if self.val[w] == F and self.val[d] == F and self.rho[d] < self.rho[w]:
                    self.sigma[w] += 1
                actual_add_deps.append((w, d))

        any_dep_del = len(actual_del_deps) > 0
        for nid in new_nodes:
            if self.refs.get(nid) or self.deps_v.get(nid) or nid in self._missing_dep_nodes:
                seeds.add(nid)

        for u, w in actual_del_refs:
            if self.val[w] == T or w in self._missing_dep_nodes:
                continue
            if self.val[w] == F and ((self.pos_cyc[w] and not any_dep_del) or self.sigma[w] > 0):
                continue
            if self.val[w] == U and self.val[u] == F:
                continue
            seeds.add(w)
        for w, d in actual_del_deps:
            if self.val[w] == T or w in self._missing_dep_nodes:
                continue
            if self.val[w] == F and (not self.pos_cyc[w]) and self.sigma[w] > 0:
                continue
            if self.val[w] == U and self.val[d] == T:
                continue
            seeds.add(w)
        for u, w in actual_add_refs:
            if w in self._missing_dep_nodes:
                continue
            if self.val[w] == F and ((self.pos_cyc[w] and not any_dep_del) or self.sigma[w] > 0):
                continue
            if self.val[u] == F and self.rho[u] < self.rho[w]:
                continue
            seeds.add(w)
        for w, d in actual_add_deps:
            if w in self._missing_dep_nodes:
                continue
            if self.val[w] == F and ((self.pos_cyc[w] and not any_dep_del) or self.sigma[w] > 0):
                continue
            if self.val[d] == T and self.rho[d] < self.rho[w]:
                continue
            seeds.add(w)

        if not seeds:
            return set(), {}, {"dirty_nodes": 0, "dirty_edges": 0, "used_dense_fallback": False}

        # Step 2: Forward cone D with Pitfall A shield-loss tracking + edge-budget circuit breaker
        dirty: Set[str] = set(seeds)
        q = deque(seeds)
        loss: Dict[str, int] = defaultdict(int)
        bfs_edges = 0
        used_fallback = False

        while q:
            u = q.popleft()
            for w in self.ref_out.get(u, ()):
                bfs_edges += 1
                if w in dirty or w in self._missing_dep_nodes:
                    continue
                if self.val[w] == F:
                    if self.val[u] == T and self.rho[u] < self.rho[w]:
                        loss[w] += 1
                    if (self.pos_cyc[w] and not any_dep_del) or (self.sigma[w] - loss[w] > 0):
                        continue
                dirty.add(w)
                q.append(w)
            for w in self.dep_v_rev.get(u, ()):
                bfs_edges += 1
                if w in dirty or w in self._missing_dep_nodes:
                    continue
                if self.val[w] == F:
                    if self.val[u] == F and self.rho[u] < self.rho[w]:
                        loss[w] += 1
                    if (not self.pos_cyc[w]) and (self.sigma[w] - loss[w] > 0):
                        continue
                dirty.add(w)
                q.append(w)
            if self.max_dirty_edges is not None and bfs_edges > self.max_dirty_edges:
                used_fallback = True
                dirty = set(self.events.keys())
                break

        rank_changed, prop_edges = self._solve_stratum2_region(dirty)
        return dirty, rank_changed, {
            "dirty_nodes": len(dirty),
            "dirty_edges": bfs_edges + prop_edges,
            "used_dense_fallback": used_fallback,
        }

    def _recompute_stratum3_currency(self, seeds: Set[str]) -> Set[str]:
        """Stratum 3: Direct-B_i + Declared Version-Lineage condensation + Quarantined cycle cap + D_C."""
        if not seeds:
            return set()
        # Expand forward over sup_out, lsup_out, dep_c_rev
        cone: Set[str] = set(seeds)
        q = deque(seeds)
        while q:
            u = q.popleft()
            for w in self.sup_out.get(u, ()):
                if w not in cone:
                    cone.add(w)
                    q.append(w)
            for w in self.lsup_out.get(u, ()):
                if w not in cone:
                    cone.add(w)
                    q.append(w)
            for w in self.dep_c_rev.get(u, ()):
                if w not in cone:
                    cone.add(w)
                    q.append(w)

        # In non-DAG (imported cycle) mode, close sup_region across any active sup cycles touching cone
        if not self.strict_dag_supersede or self._q_sup_nodes:
            sup_region: Set[str] = set(cone)
            sq = deque(cone)
            while sq:
                u = sq.popleft()
                for w in self.sup_out.get(u, ()):
                    if w not in sup_region:
                        sup_region.add(w)
                        sq.append(w)
                for a in self.sup.get(u, ()):
                    if a not in sup_region and (a in self._q_sup_nodes or not self.strict_dag_supersede):
                        sup_region.add(a)
                        sq.append(a)
            active_sup_out: Dict[str, Set[str]] = defaultdict(set)
            for u in sup_region:
                if self.val.get(u) != F:
                    for w in self.sup_out.get(u, ()):
                        if w in sup_region:
                            active_sup_out[u].add(w)
            sup_sccs = _iterative_tarjan_sccs(sup_region, active_sup_out, subset=sup_region)
            # Replace only quarantined cycles touching sup_region (preserve unrelated cycles!)
            for key in list(self._q_sup_sccs.keys()):
                if self._q_sup_sccs[key] & sup_region:
                    del self._q_sup_sccs[key]
            for comp in sup_sccs:
                if len(comp) >= 2 or (len(comp) == 1 and comp[0] in active_sup_out.get(comp[0], ())):
                    key = tuple(sorted(comp))
                    self._q_sup_sccs[key] = set(comp)
            self._q_sup_nodes = set().union(*self._q_sup_sccs.values()) if self._q_sup_sccs else set()
            self._quarantined_sup_cycles = sorted(self._q_sup_sccs.keys())
            # Ensure every node whose outgoing sup from a cycle may have changed is in cone
            for u in list(sup_region):
                if u not in cone:
                    cone.add(u)
                    for w in self.dep_c_rev.get(u, ()):
                        if w not in cone:
                            cone.add(w)

        q_sup_nodes = self._q_sup_nodes

        # Declared version-lineage condensation DAG (durable edges: even if V_mid == F, valid head covers!)
        lsup_region: Set[str] = {u for u in cone if self.lsup.get(u) or self.lsup_out.get(u) or u in self._q_lsup_nodes}
        scc_id: Dict[str, int] = {}
        has_def_anc: Dict[int, bool] = defaultdict(bool)
        has_poss_anc: Dict[int, bool] = defaultdict(bool)
        q_lsup_sccs_idx: Set[int] = set()
        if lsup_region:
            if not self.strict_dag_supersede or self._q_lsup_nodes:
                lq = deque(lsup_region)
                while lq:
                    u = lq.popleft()
                    for w in self.lsup_out.get(u, ()):
                        if w not in lsup_region:
                            lsup_region.add(w)
                            cone.add(w)
                            lq.append(w)
                    for a in self.lsup.get(u, ()):
                        if a not in lsup_region and (a in self._q_lsup_nodes or not self.strict_dag_supersede):
                            lsup_region.add(a)
                            cone.add(a)
                            lq.append(a)
            lsup_sccs = _iterative_tarjan_sccs(lsup_region, self.lsup_out, subset=lsup_region)
            for key in list(self._q_lsup_sccs.keys()):
                if self._q_lsup_sccs[key] & lsup_region:
                    del self._q_lsup_sccs[key]
            for idx, comp in enumerate(lsup_sccs):
                for u in comp:
                    scc_id[u] = idx
                if len(comp) >= 2 or (len(comp) == 1 and comp[0] in self.lsup_out.get(comp[0], ())):
                    if any(self.val.get(u) != F for u in comp):
                        q_lsup_sccs_idx.add(idx)
                        key = tuple(sorted(comp))
                        self._q_lsup_sccs[key] = set(comp)
            self._q_lsup_nodes = set().union(*self._q_lsup_sccs.values()) if self._q_lsup_sccs else set()

            # Seed boundary lineage ancestry from frozen predecessors outside lsup_region
            for idx, comp in enumerate(lsup_sccs):
                for v in comp:
                    for j in self.lsup.get(v, ()):
                        if j not in lsup_region:
                            if j in self._q_lsup_nodes:
                                has_poss_anc[idx] = True
                            else:
                                if self.val.get(j) == T or self._lsup_has_def_anc.get(j, False):
                                    has_def_anc[idx] = True
                                if self.val.get(j) == U or self._lsup_has_poss_anc.get(j, False):
                                    has_poss_anc[idx] = True

            # Forward DP on lineage SCC condensation (Tarjan emits sinks first -> reverse for sources first!)
            for idx in reversed(range(len(lsup_sccs))):
                comp = lsup_sccs[idx]
                is_q = idx in q_lsup_sccs_idx
                for u in comp:
                    self._lsup_has_def_anc[u] = has_def_anc[idx] and not is_q
                    self._lsup_has_poss_anc[u] = has_poss_anc[idx] or is_q
                if is_q:
                    for u in comp:
                        for w in self.lsup_out.get(u, ()):
                            if w in scc_id and scc_id[w] != idx:
                                has_poss_anc[scc_id[w]] = True
                else:
                    comp_has_t = has_def_anc[idx] or any(self.val.get(u) == T for u in comp)
                    comp_has_u = has_poss_anc[idx] or any(self.val.get(u) == U for u in comp)
                    for u in comp:
                        for w in self.lsup_out.get(u, ()):
                            if w in scc_id and scc_id[w] != idx:
                                if comp_has_t:
                                    has_def_anc[scc_id[w]] = True
                                if comp_has_u:
                                    has_poss_anc[scc_id[w]] = True

        # Local blocker evaluation for every v in cone
        local_cur: Dict[str, str] = {}
        local_reason: Dict[str, str] = {}
        for v in cone:
            vv = self.val[v]
            if vv == F:
                local_cur[v] = F
                if v in self._missing_dep_nodes:
                    local_reason[v] = "MISSING_MANDATORY_PARENT"
                elif self.pos_cyc.get(v):
                    local_reason[v] = "UNFOUNDED_CYCLE"
                else:
                    local_reason[v] = "REFUTED"
                continue

            b_ext = F
            b_cyc = F
            for j in self.sup.get(v, ()):
                vj = self.val.get(j, F)
                if vj == F:
                    continue
                if j in q_sup_nodes:
                    b_cyc = U
                else:
                    if vj == T:
                        b_ext = T
                    elif vj == U and b_ext != T:
                        b_ext = U
            if v in scc_id:
                sid = scc_id[v]
                if has_def_anc[sid]:
                    b_ext = T
                elif has_poss_anc[sid] or sid in q_lsup_sccs_idx:
                    b_cyc = U

            if b_ext == T:
                local_cur[v] = F
                local_reason[v] = "SUPERSEDED"
            elif vv == U:
                local_cur[v] = U
                local_reason[v] = "EPISTEMIC_CONTESTED"
            elif b_ext == U or b_cyc == U:
                local_cur[v] = U
                local_reason[v] = "TEMPORAL_CONTESTED"
            else:
                local_cur[v] = T
                local_reason[v] = "CURRENT"

        # Propagate currency monotonically along D_C (`COPY_OF`, `DERIVES_CURRENT`) in bottom-up Tarjan order
        dc_sccs = _iterative_tarjan_sccs(cone, self.deps_c, subset=cone)
        for comp in dc_sccs:
            if len(comp) >= 2 or (len(comp) == 1 and comp[0] in self.deps_c.get(comp[0], ())):
                for u in comp:
                    local_cur[u] = F
                    local_reason[u] = "UNFOUNDED_CYCLE"
                continue
            u = comp[0]
            if local_cur[u] == F:
                continue
            for d in self.deps_c.get(u, ()):
                cd = local_cur[d] if d in local_cur else self.cur.get(d, T)
                if cd == F:
                    local_cur[u] = F
                    local_reason[u] = "SUPERSEDED_ANCESTOR"
                    break
                if cd == U and local_cur[u] == T:
                    local_cur[u] = U
                    local_reason[u] = "TEMPORAL_ANCESTOR_CONTESTED"

        for v in cone:
            self.cur[v] = local_cur[v]
            self.reason[v] = local_reason[v]
            if self.val[v] == T and self.cur[v] == T:
                self.eligibility[v] = "ADMIT"
            elif self.val[v] == F or self.cur[v] == F:
                self.eligibility[v] = "AUDIT_ONLY"
            else:
                self.eligibility[v] = "CONTESTED"
        return cone

    def _would_close_supersede_cycle(self, src: str, dst: str, out_map: Mapping[str, Set[str]]) -> bool:
        if src == dst:
            return True
        seen: Set[str] = {dst}
        q = deque([dst])
        while q:
            u = q.popleft()
            for w in out_map.get(u, ()):
                if w == src:
                    return True
                if w not in seen:
                    seen.add(w)
                    q.append(w)
        return False

    def apply_mutation(
        self,
        new_events: Sequence[StratifiedEvent | str] = (),
        add_attestations: Sequence[EdgeAttestation | Tuple[str, str, str]] = (),
        withdraw_attestation_ids: Sequence[str] = (),
        *,
        remove_edges: Sequence[Tuple[str, str, str]] = (),
    ) -> MutationDelta:
        """Apply an atomic batch of events, edge additions, and edge withdrawals/removals."""
        t0 = time.perf_counter()
        old_elig = dict(self.eligibility)
        inserted_ids: Set[str] = set()
        auth_seeds: Set[str] = set()
        pending_ids: List[str] = []

        for item in new_events:
            ev = StratifiedEvent(event_id=item) if isinstance(item, str) else item
            if ev.event_id not in self.events:
                self.events[ev.event_id] = ev
                self.eff_auth[ev.event_id] = 0 if ev.event_id in self._missing_dep_nodes else ev.authority_tier
                self.val[ev.event_id] = T
                self.rho[ev.event_id] = 0
                self.pos_cyc[ev.event_id] = False
                self.sigma[ev.event_id] = 0
                self.cur[ev.event_id] = T
                self.eligibility[ev.event_id] = "ADMIT"
                self.reason[ev.event_id] = "CURRENT"
                inserted_ids.add(ev.event_id)
                auth_seeds.add(ev.event_id)

        # Replay previously pending attestations whose endpoints arrived
        to_process: List[EdgeAttestation] = []
        if self.pending_attestations and inserted_ids:
            still_pending: List[EdgeAttestation] = []
            for p_att in self.pending_attestations:
                if p_att.src_id in self.events and p_att.dst_id in self.events:
                    to_process.append(p_att)
                else:
                    still_pending.append(p_att)
            self.pending_attestations = still_pending

        for raw in add_attestations:
            if isinstance(raw, EdgeAttestation):
                att = raw
            else:
                src_id, dst_id, etype = raw
                self._auto_att_seq += 1
                att = EdgeAttestation(
                    attestation_id=f"auto_{self._auto_att_seq}",
                    src_id=src_id,
                    dst_id=dst_id,
                    edge_type=etype,
                )
            if att.edge_type not in V3_VALID_EDGE_TYPES:
                raise ValueError(f"Unknown edge_type={att.edge_type!r}")
            if att.edge_type in ("SUPERSEDE", "VERSION_LINEAGE_SUPERSEDE") and att.src_id == att.dst_id:
                raise SupersedeCycleError(f"Self-{att.edge_type} rejected on {att.src_id}")
            if att.src_id not in self.events or att.dst_id not in self.events:
                if self.buffer_pending:
                    self.pending_attestations.append(att)
                    pending_ids.append(att.attestation_id)
                    continue
                raise ValueError(f"Dangling edge endpoint in {att.src_id} -[{att.edge_type}]-> {att.dst_id}")
            to_process.append(att)

        cand_ref_pairs: Set[Tuple[str, str]] = set()
        cand_sup_pairs: Set[Tuple[str, str]] = set()
        cand_lsup_pairs: Set[Tuple[str, str]] = set()
        raw_add_deps: List[Tuple[str, str]] = []
        raw_del_deps: List[Tuple[str, str]] = []
        cur_seeds: Set[str] = set(inserted_ids)

        # Phase A: Apply withdrawals to raw topology + multiplicity map
        withdraw_triples: List[Tuple[str, str, str, Optional[str]]] = []
        for aid in withdraw_attestation_ids:
            att = self.attestations.get(aid)
            if att is not None and att.active:
                self.attestations[aid] = EdgeAttestation(
                    attestation_id=att.attestation_id,
                    src_id=att.src_id,
                    dst_id=att.dst_id,
                    edge_type=att.edge_type,
                    valid_time=att.valid_time,
                    tx_time=att.tx_time,
                    lineage_id=att.lineage_id,
                    active=False,
                )
                withdraw_triples.append((att.src_id, att.dst_id, att.edge_type, att.lineage_id))
        for src, dst, etype in remove_edges:
            if etype not in V3_VALID_EDGE_TYPES:
                raise ValueError(f"Unknown edge_type={etype!r}")
            for aid, att in self.attestations.items():
                if att.active and att.src_id == src and att.dst_id == dst and att.edge_type == etype:
                    self.attestations[aid] = EdgeAttestation(
                        attestation_id=att.attestation_id,
                        src_id=att.src_id,
                        dst_id=att.dst_id,
                        edge_type=att.edge_type,
                        valid_time=att.valid_time,
                        tx_time=att.tx_time,
                        lineage_id=att.lineage_id,
                        active=False,
                    )
                    break
            withdraw_triples.append((src, dst, etype, None))

        for src, dst, etype, lin_id in withdraw_triples:
            key = (src, dst, etype)
            if self._edge_mult.get(key, 0) <= 0:
                continue
            self._edge_mult[key] -= 1
            if self._edge_mult[key] > 0:
                continue
            del self._edge_mult[key]
            if etype == "REFUTE":
                self._raw_refs[dst].discard(src)
                self._raw_ref_out[src].discard(dst)
                cand_ref_pairs.add((src, dst))
            elif etype == "SUPERSEDE":
                self._raw_sup[dst].discard(src)
                self._raw_sup_out[src].discard(dst)
                cand_sup_pairs.add((src, dst))
            elif etype == "VERSION_LINEAGE_SUPERSEDE":
                self._raw_lsup[dst].discard(src)
                self._raw_lsup_out[src].discard(dst)
                if lin_id is not None:
                    self._lineage_edges[(src, dst)].discard(lin_id)
                cand_lsup_pairs.add((src, dst))
            elif etype in DEP_ALL_TYPES:
                still_dv = any(self._edge_mult.get((src, dst, t), 0) > 0 for t in DEP_ALL_TYPES)
                still_dc = any(self._edge_mult.get((src, dst, t), 0) > 0 for t in DEP_CUR_TYPES)
                if not still_dv:
                    raw_del_deps.append((src, dst))
                    auth_seeds.add(src)
                if not still_dc and dst in self.deps_c.get(src, ()):
                    self.deps_c[src].discard(dst)
                    self.dep_c_rev[dst].discard(src)
                    cur_seeds.add(src)
            elif etype == "CITES_AUDIT":
                self.cites_audit[src].discard(dst)

        # Phase B: Apply additions to raw topology + multiplicity map
        for att in to_process:
            if att.attestation_id in self.attestations and self.attestations[att.attestation_id].active:
                continue
            if att.edge_type == "VERSION_LINEAGE_SUPERSEDE":
                ev_s, ev_d = self.events[att.src_id], self.events[att.dst_id]
                if not att.lineage_id or (ev_s.tenant, ev_s.entity, ev_s.slot) != (ev_d.tenant, ev_d.entity, ev_d.slot):
                    raise ValueError("VERSION_LINEAGE_SUPERSEDE requires matching (tenant, entity, slot) & lineage_id")
            if self.strict_dag_supersede and att.edge_type == "SUPERSEDE":
                if self._would_close_supersede_cycle(att.src_id, att.dst_id, self._raw_sup_out):
                    raise SupersedeCycleError(f"SUPERSEDE cycle rejected: {att.src_id} -> {att.dst_id}")
            if self.strict_dag_supersede and att.edge_type == "VERSION_LINEAGE_SUPERSEDE":
                if self._would_close_supersede_cycle(att.src_id, att.dst_id, self._raw_lsup_out):
                    raise SupersedeCycleError(f"Lineage SUPERSEDE cycle rejected: {att.src_id} -> {att.dst_id}")

            self.attestations[att.attestation_id] = att
            key = (att.src_id, att.dst_id, att.edge_type)
            self._edge_mult[key] += 1
            if self._edge_mult[key] > 1:
                continue

            src, dst, etype = att.src_id, att.dst_id, att.edge_type
            if etype == "REFUTE":
                self._raw_refs[dst].add(src)
                self._raw_ref_out[src].add(dst)
                cand_ref_pairs.add((src, dst))
            elif etype == "SUPERSEDE":
                self._raw_sup[dst].add(src)
                self._raw_sup_out[src].add(dst)
                cand_sup_pairs.add((src, dst))
            elif etype == "VERSION_LINEAGE_SUPERSEDE":
                self._raw_lsup[dst].add(src)
                self._raw_lsup_out[src].add(dst)
                if att.lineage_id:
                    self._lineage_edges[(src, dst)].add(att.lineage_id)
                cand_lsup_pairs.add((src, dst))
            elif etype in DEP_ALL_TYPES:
                if dst not in self.deps_v.get(src, ()) or (src, dst) in raw_del_deps:
                    raw_add_deps.append((src, dst))
                    auth_seeds.add(src)
                if etype in DEP_CUR_TYPES:
                    self.deps_c[src].add(dst)
                    self.dep_c_rev[dst].add(src)
                    cur_seeds.add(src)
            elif etype == "CITES_AUDIT":
                self.cites_audit[src].add(dst)

        # Phase C: Recompute Stratum 1 authority ceiling over post-batch D_V BEFORE qualifying edges!
        s_ad, s_dd = set(raw_add_deps), set(raw_del_deps)
        net_add_deps = [e for e in (s_ad - s_dd) if e[1] not in self.deps_v.get(e[0], ())]
        net_del_deps = [e for e in (s_dd - s_ad) if e[1] in self.deps_v.get(e[0], ())]
        changed_auth: Set[str] = set()
        if auth_seeds and (net_add_deps or net_del_deps or (self._missing_dep_nodes & inserted_ids)):
            for w, d in net_del_deps:
                self.deps_v[w].discard(d)
                self.dep_v_rev[d].discard(w)
            for w, d in net_add_deps:
                self.deps_v[w].add(d)
                self.dep_v_rev[d].add(w)
            changed_auth = self._recompute_authority_cone(auth_seeds)
            # Revert deps_v so _apply_stratum2_batch tracks sigma cleanly on net transitions
            for w, d in net_add_deps:
                self.deps_v[w].discard(d)
                self.dep_v_rev[d].discard(w)
            for w, d in net_del_deps:
                self.deps_v[w].add(d)
                self.dep_v_rev[d].add(w)

        for u in changed_auth:
            for w in self._raw_ref_out.get(u, ()):
                cand_ref_pairs.add((u, w))
            for a in self._raw_refs.get(u, ()):
                cand_ref_pairs.add((a, u))
            for w in self._raw_sup_out.get(u, ()):
                cand_sup_pairs.add((u, w))
            for a in self._raw_sup.get(u, ()):
                cand_sup_pairs.add((a, u))
            for w in self._raw_lsup_out.get(u, ()):
                cand_lsup_pairs.add((u, w))
            for a in self._raw_lsup.get(u, ()):
                cand_lsup_pairs.add((a, u))

        # Phase D: Reconcile all candidate REFUTE / SUPERSEDE / LSUP edges against final eff_auth
        add_refs: List[Tuple[str, str]] = []
        del_refs: List[Tuple[str, str]] = []
        for u, w in cand_ref_pairs:
            raw_exists = u in self._raw_refs.get(w, ())
            cur_active = u in self.refs.get(w, ())
            if not raw_exists:
                self._blocked_auth_edges.discard((u, w, "REFUTE"))
                if cur_active:
                    del_refs.append((u, w))
            else:
                allowed = self._effective_authority_for_domain(u, self.events[w].domain) >= self.eff_auth[w]
                if allowed:
                    self._blocked_auth_edges.discard((u, w, "REFUTE"))
                    if not cur_active:
                        add_refs.append((u, w))
                else:
                    self._blocked_auth_edges.add((u, w, "REFUTE"))
                    if cur_active:
                        del_refs.append((u, w))

        for u, w in cand_sup_pairs:
            raw_exists = u in self._raw_sup.get(w, ())
            cur_active = u in self.sup.get(w, ())
            if not raw_exists:
                self._blocked_auth_edges.discard((u, w, "SUPERSEDE"))
                if cur_active:
                    self.sup[w].discard(u)
                    self.sup_out[u].discard(w)
                    cur_seeds.add(w)
                    cur_seeds.add(u)
            else:
                allowed = self._effective_authority_for_domain(u, self.events[w].domain) >= self.eff_auth[w]
                if allowed:
                    self._blocked_auth_edges.discard((u, w, "SUPERSEDE"))
                    if not cur_active:
                        self.sup[w].add(u)
                        self.sup_out[u].add(w)
                        cur_seeds.add(w)
                        cur_seeds.add(u)
                else:
                    self._blocked_auth_edges.add((u, w, "SUPERSEDE"))
                    if cur_active:
                        self.sup[w].discard(u)
                        self.sup_out[u].discard(w)
                        cur_seeds.add(w)
                        cur_seeds.add(u)

        for u, w in cand_lsup_pairs:
            raw_exists = u in self._raw_lsup.get(w, ())
            cur_active = u in self.lsup.get(w, ())
            if not raw_exists:
                self._blocked_auth_edges.discard((u, w, "VERSION_LINEAGE_SUPERSEDE"))
                if cur_active:
                    self.lsup[w].discard(u)
                    self.lsup_out[u].discard(w)
                    cur_seeds.add(w)
                    cur_seeds.add(u)
            else:
                allowed = self._effective_authority_for_domain(u, self.events[w].domain) >= self.eff_auth[w]
                if allowed:
                    self._blocked_auth_edges.discard((u, w, "VERSION_LINEAGE_SUPERSEDE"))
                    if not cur_active:
                        self.lsup[w].add(u)
                        self.lsup_out[u].add(w)
                        cur_seeds.add(w)
                        cur_seeds.add(u)
                else:
                    self._blocked_auth_edges.add((u, w, "VERSION_LINEAGE_SUPERSEDE"))
                    if cur_active:
                        self.lsup[w].discard(u)
                        self.lsup_out[u].discard(w)
                        cur_seeds.add(w)
                        cur_seeds.add(u)

        dirty_s2, rank_changed, s2_stats = self._apply_stratum2_batch(
            inserted_ids, add_refs, del_refs, net_add_deps, net_del_deps
        )
        cur_seeds.update(dirty_s2)
        if cur_seeds:
            affected_cur = self._recompute_stratum3_currency(cur_seeds)
            cur_seeds.update(affected_cur)

        inserted_map = {nid: self.eligibility[nid] for nid in sorted(inserted_ids)}
        transitioned: Dict[str, Tuple[str, str]] = {}
        for nid in sorted(cur_seeds - inserted_ids):
            prev = old_elig.get(nid)
            now = self.eligibility.get(nid)
            if prev is not None and now is not None and prev != now:
                transitioned[nid] = (prev, now)

        wall_ms = (time.perf_counter() - t0) * 1000.0
        s2_stats["wall_ms"] = round(wall_ms, 4)
        s2_stats["total_nodes"] = len(self.events)
        return MutationDelta(
            inserted=inserted_map,
            transitioned=transitioned,
            removed={},
            rank_changed=rank_changed,
            pending_buffered=tuple(pending_ids),
            work_stats=s2_stats,
        )

    def compact_ranks(self) -> None:
        """Periodic checkpoint rank compaction recomputing minimal Lyapunov ranks in O(|V| log |V| + |E|)."""
        all_nodes = set(self.events.keys())
        if not all_nodes:
            return
        self._solve_stratum2_region(all_nodes, force_exact=True)

    def as_of(self, valid_cutoff: int, tx_cutoff: int) -> "StratifiedEvidenceStore":
        """Materialize an exact bitemporal snapshot at `(valid_cutoff, tx_cutoff)` (handles rewinds & missing parents!)."""
        snap = StratifiedEvidenceStore(
            strict_dag_supersede=False,  # Historical replay preserves imported cycles fail-closed
            buffer_pending=False,
            exact_min_rank=self.exact_min_rank,
            max_dirty_edges=self.max_dirty_edges,
        )
        vis_events = [
            ev
            for ev in self.events.values()
            if ev.tx_time <= tx_cutoff
            and ev.valid_time <= valid_cutoff
            and (ev.valid_end is None or valid_cutoff < ev.valid_end)
        ]
        vis_ids = {ev.event_id for ev in vis_events}
        vis_atts: List[EdgeAttestation] = []
        missing_dep_nodes: Set[str] = set()
        for att in self.attestations.values():
            if not (att.active and att.tx_time <= tx_cutoff and att.valid_time <= valid_cutoff):
                continue
            if att.src_id in vis_ids and att.dst_id in vis_ids:
                vis_atts.append(att)
            elif att.edge_type in DEP_ALL_TYPES and att.src_id in vis_ids and att.dst_id not in vis_ids:
                # Visible child depends on an invisible/expired mandatory parent -> eff_auth=0, V=F
                missing_dep_nodes.add(att.src_id)
        snap._missing_dep_nodes = missing_dep_nodes
        snap.apply_mutation(vis_events, vis_atts)
        return snap

    def resolve_slot(self, tenant: str, entity: str, slot: str) -> Dict[str, Any]:
        """Single-valued slot reader enforcing slot-scoped `CONTESTED` abstention."""
        slot_eids = [
            eid
            for eid, ev in self.events.items()
            if ev.tenant == tenant and ev.entity == entity and ev.slot == slot
        ]
        admitted = [eid for eid in slot_eids if self.eligibility.get(eid) == "ADMIT"]
        contested = [eid for eid in slot_eids if self.eligibility.get(eid) == "CONTESTED"]
        admit_vals = sorted({self.events[eid].claim_value for eid in admitted if self.events[eid].claim_value})
        if contested:
            return {
                "status": "CONTESTED",
                "reason": "UNRESOLVED_CONFLICT",
                "admitted_ids": admitted,
                "contested_ids": contested,
                "distinct_values": admit_vals,
            }
        if len(admit_vals) > 1:
            return {
                "status": "CONTESTED",
                "reason": "MULTI_CURRENT",
                "admitted_ids": admitted,
                "contested_ids": [],
                "distinct_values": admit_vals,
            }
        if len(admit_vals) == 1:
            return {
                "status": "ADMIT",
                "reason": "UNIQUE_CURRENT",
                "claim_value": admit_vals[0],
                "admitted_ids": admitted,
                "contested_ids": [],
                "distinct_values": admit_vals,
            }
        return {
            "status": "EMPTY",
            "reason": "NO_CURRENT_EVIDENCE",
            "admitted_ids": [],
            "contested_ids": [],
            "distinct_values": [],
        }

    def get_telemetry(self) -> EpistemicTelemetry:
        """EM-3 passive certificate-based diagnostic telemetry (zero feedback into WFS)."""
        return EpistemicTelemetry(
            certificate_based=True,
            unfounded_cycle_nodes=tuple(sorted(u for u, is_c in self.pos_cyc.items() if is_c)),
            blocked_authority_edges=tuple(sorted(self._blocked_auth_edges)),
            quarantined_supersede_cycles=tuple(self._quarantined_sup_cycles),
        )

    def full_structural_scc_audit(self) -> Tuple[str, ...]:
        """Optional full structural positive-cycle audit (addresses Astra R3 EM-3 pruning gap)."""
        return tuple(sorted(_nontrivial_scc_nodes(self.events.keys(), self.deps_v)))


# ---------------------------------------------------------------------------
# EM-1 (M4e): Minimal-Pair Contradiction vs. Echo Split Adapter
# ---------------------------------------------------------------------------
_HEADER_PREFIX_RE = re.compile(
    r"^\s*(?:briefing|session|memo|note|turn|entry|update|memory\s+recap|recap|digest|log|report|reminder|summary)\s*"
    r"(?:#?\d+)?\s*(?:\([^)]*\))?\s*[:\-–—]\s*",
    re.IGNORECASE,
)
_TIME_AMPM_RE = re.compile(r"\b(\d{1,2})(?::(\d{2}))?\s*(am|pm)\b", re.IGNORECASE)
_TIME_24H_RE = re.compile(r"\b([01]?\d|2[0-3]):([0-5]\d)\b")
_NUM_UNIT_RE = re.compile(
    r"(?<![a-z0-9])([a-z]{0,3}\d+(?:,\d{3})*(?:\.\d+)?[a-z%]*)\s*(bar|kpa|psi|mg|g|kg|ml|l|days?|weeks?|hours?|hrs?|minutes?|mins?|usd|dollars?|%)?",
    re.IGNORECASE,
)
_NEGATION_WORDS: FrozenSet[str] = frozenset({
    "not", "no", "never", "none", "neither", "nor", "without", "cannot", "cant", "dont", "doesnt",
    "didnt", "isnt", "arent", "wasnt", "werent", "wont", "wouldnt", "shouldnt", "hasnt", "havent",
})
_ANTONYM_PAIRS: Tuple[Tuple[str, str], ...] = (
    ("enabled", "disabled"),
    ("open", "closed"),
    ("active", "inactive"),
    ("allow", "deny"),
    ("allowed", "blocked"),
    ("approved", "rejected"),
    ("pass", "fail"),
    ("true", "false"),
    ("yes", "no"),
    ("allergic", "nonallergic"),
    ("online", "offline"),
    ("valid", "invalid"),
    ("confirmed", "cancelled"),
    ("married", "divorced"),
    ("employed", "unemployed"),
)
_ANTONYM_MAP: Dict[str, str] = {}
for _a, _b in _ANTONYM_PAIRS:
    _ANTONYM_MAP[_a] = _b
    _ANTONYM_MAP[_b] = _a

_WORD_NUMS: Dict[str, str] = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
    "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10",
    "eleven": "11", "twelve": "12", "fourteen": "14", "twenty": "20", "thirty": "30",
}


_SENTENCE_STARTERS: FrozenSet[str] = frozenset({
    "the", "a", "an", "i", "we", "you", "he", "she", "it", "they", "this", "that", "these", "those",
    "my", "our", "your", "his", "her", "its", "their", "user", "assistant", "system", "note", "update",
    "on", "in", "at", "for", "by", "with", "from", "to", "please", "yes", "no", "always", "never",
    "all", "some", "each", "every", "both", "any", "current", "currently", "previous", "previously",
    "earlier", "prior", "latest", "recent", "updated", "confirmed", "status", "dosage", "meeting",
    "gate", "flight", "room", "price", "cost", "budget", "limit", "threshold", "schedule", "time",
    "date", "location", "address", "city", "country", "favorite", "preferred", "allergic", "medical",
    "briefing", "session", "recap", "reminder", "summary", "memo", "entry", "log", "report",
})
_BOILERPLATE_FRAMING_WORDS: FrozenSet[str] = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being", "to", "of", "in", "on", "at",
    "for", "by", "with", "from", "that", "this", "it", "as", "per", "about", "into",
    "briefing", "session", "recap", "note", "notes", "reminder", "log", "update", "updated", "previous",
    "previously", "earlier", "prior", "recorded", "noted", "mentioned", "stated", "said", "confirmed",
    "confirms", "memory", "entry", "turn", "memo", "report", "summary", "archive", "context", "field",
    "audit", "independent", "latest", "current", "currently", "still", "also", "just",
})
_CANON_UNIT_TOKENS: FrozenSet[str] = frozenset({
    "bar", "kpa", "psi", "mg", "g", "kg", "ml", "l", "day", "days", "week", "weeks",
    "hour", "hours", "hr", "hrs", "minute", "minutes", "min", "mins", "usd", "dollar", "dollars",
    "am", "pm",
})


def strip_boilerplate_headers(text: str) -> str:
    """Strip recurring log/recap headers (`Briefing 0:`, `Session 4:`, `Memory recap 1:`)."""
    cleaned = text.strip()
    for _ in range(3):
        nxt = _HEADER_PREFIX_RE.sub("", cleaned, count=1).strip()
        if nxt == cleaned:
            break
        cleaned = nxt
    return cleaned


def _canonical_value_tokens(text: str) -> Tuple[Set[str], int, Set[str], Set[str], Set[str]]:
    """Extract `(numeric_unit_sigs, negation_parity, status_terms, proper_nouns, residual_content)`."""
    body = strip_boilerplate_headers(text)
    lower = re.sub(r"\$\s*(\d+(?:,\d{3})*(?:\.\d+)?)", r"\1 usd", body.lower())
    for w_num, dig in _WORD_NUMS.items():
        lower = re.sub(rf"\b{w_num}\b", dig, lower)

    sigs: Set[str] = set()

    def _norm_ampm(m: re.Match[str]) -> str:
        hh = int(m.group(1))
        mm = int(m.group(2) or "0")
        ap = m.group(3).lower()
        if ap == "pm" and hh < 12:
            hh += 12
        elif ap == "am" and hh == 12:
            hh = 0
        sigs.add(f"time:{hh:02d}:{mm:02d}")
        return " "

    lower_nt = _TIME_AMPM_RE.sub(_norm_ampm, lower)

    def _norm_24h(m: re.Match[str]) -> str:
        sigs.add(f"time:{int(m.group(1)):02d}:{int(m.group(2)):02d}")
        return " "

    lower_nt = _TIME_24H_RE.sub(_norm_24h, lower_nt)

    def _norm_num(m: re.Match[str]) -> str:
        raw_tok = m.group(1).replace(",", "")
        unit = (m.group(2) or "").lower()
        if re.fullmatch(r"\d+(?:\.\d+)?", raw_tok):
            val = float(raw_tok)
            if unit == "kpa":
                val = val / 100.0
                unit = "bar"
            elif unit in ("week", "weeks"):
                val = val * 7.0
                unit = "day"
            elif unit in ("day", "days"):
                unit = "day"
            elif unit in ("hour", "hours", "hr", "hrs"):
                unit = "hour"
            elif unit in ("minute", "minutes", "min", "mins"):
                unit = "min"
            elif unit in ("dollar", "dollars", "usd"):
                unit = "usd"
            sigs.add(f"num:{val:.6g}:{unit}")
        else:
            sigs.add(f"code:{raw_tok}")
        return " "

    lower_stripped = _NUM_UNIT_RE.sub(_norm_num, lower_nt)

    toks = re.findall(r"[a-z0-9]+", lower.replace("'", ""))
    neg_count = sum(1 for t in toks if t in _NEGATION_WORDS)
    status_terms = {t for t in toks if t in _ANTONYM_MAP}

    residual_content = {
        t
        for t in re.findall(r"[a-z]+", lower_stripped.replace("'", ""))
        if t not in _BOILERPLATE_FRAMING_WORDS
        and t not in _CANON_UNIT_TOKENS
        and t not in _NEGATION_WORDS
    }

    # Proper nouns: non-sentence-initial capitalized tokens OR sentence-initial capitalized non-starters
    proper: Set[str] = set()
    raw_words = re.findall(r"[A-Za-z][A-Za-z0-9_-]*", body)
    for idx, w in enumerate(raw_words):
        wl = w.lower()
        if wl in _WORD_NUMS or wl in _NEGATION_WORDS or wl in _CANON_UNIT_TOKENS or wl in ("user", "assistant", "system"):
            continue
        if w[0].isupper():
            if idx > 0 or w.isupper() or wl not in _SENTENCE_STARTERS:
                proper.add(wl)

    return sigs, neg_count % 2, status_terms, proper, residual_content


def classify_high_similarity_pair(text_a: str, text_b: str) -> Tuple[str, str]:
    """Classify a high-similarity cross-session pair (`EM-1` `M4e`).

    Returns `(action, reason)` where `action` is:
      - `"ECHO_COLLAPSE"`: compatible values, polarity, entities, and content frame -> safe to collapse copy.
      - `"PRESERVE_CONFLICT"`: detected numeric/code/time, negation/antonym, proper-entity, or crisp minimal-pair
                               substitution -> NEVER collapse; retain both witnesses fail-safe
                               (does NOT auto-grant supersession authority).
    """
    sigs_a, neg_a, st_a, prop_a, res_a = _canonical_value_tokens(text_a)
    sigs_b, neg_b, st_b, prop_b, res_b = _canonical_value_tokens(text_b)

    if sigs_a != sigs_b and (sigs_a or sigs_b):
        return "PRESERVE_CONFLICT", "CONFLICT_NUMERIC_OR_CODE"
    if neg_a != neg_b:
        return "PRESERVE_CONFLICT", "CONFLICT_POLARITY"
    for term in st_a:
        if _ANTONYM_MAP.get(term) in st_b:
            return "PRESERVE_CONFLICT", "CONFLICT_ANTONYM"
    if prop_a != prop_b and prop_a and prop_b:
        return "PRESERVE_CONFLICT", "CONFLICT_PROPER_ENTITY"
    diff_a = res_a - res_b
    diff_b = res_b - res_a
    if diff_a and diff_b and len(diff_a) <= 2 and len(diff_b) <= 2:
        return "PRESERVE_CONFLICT", "CONFLICT_MINIMAL_PAIR_SUBSTITUTION"
    return "ECHO_COLLAPSE", "COMPATIBLE_ECHO"


__all__ = [
    "DEP_ALL_TYPES",
    "DEP_CUR_TYPES",
    "DEP_EP_ONLY_TYPES",
    "EdgeAttestation",
    "EpistemicTelemetry",
    "MutationDelta",
    "StratifiedEvent",
    "StratifiedEvidenceStore",
    "SupersedeCycleError",
    "V3_VALID_EDGE_TYPES",
    "classify_high_similarity_pair",
    "compute_alternating_wfs_oracle",
    "strip_boilerplate_headers",
]
