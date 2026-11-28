"""W22-11: large-graph incremental WFS deletion stress test and canonical witness check.

This module extends the Wave-14A differential conformance in ``formal/run_conformance.py``. That run
covered 11 regular nodes and 7,722 checkpoints. Here the graphs are larger, the planted D_V
(positive-dependency) SCCs are larger, and every step carries a deletion schedule. Each checkpoint
compares the live incremental ``ek5_v3_stratified.StratifiedEvidenceStore`` against a store rebuilt from
scratch out of the independent mutation model.

It reuses rather than re-implements the conformance machinery: ``_Model``, ``_gen_batch`` (baseline tiers),
``model_snapshot``, ``fresh_from_model``, ``store_state``, ``compare_states``, ``materialize`` and
``oracle_vals`` are imported from ``formal/run_conformance.py``.

Four comparison classes run at each checkpoint (definitions frozen in
``artifacts/pv7_w23c_w22_11/PREREGISTRATION.md``):

* **semantic**:
  * inc vs fresh on ``eff_auth/val/cur/eligibility`` and the node set;
  * inc ``val`` vs the textbook alternating-fixpoint WFS oracle;
  * the inc Stratum-0 materialization vs the model snapshot;
  * the canonicalizer's own ``cur``/``eligibility`` re-derivation vs each store.
* **raw witness**: inc vs fresh on the engine's own telemetry (``reason``, ``pos_cyc``, ``blocked``,
  ``quarantined``). Known to differ; this class is characterized, not used as the gate.
* **canonical witness**: ``export_canonical_ek5_audit_trace(inc)`` vs the same export of the fresh store.
  The comparison is on the whole trace: per-node reason/cause/pos_cyc/witness tuple plus the positive,
  supersede and lineage cycle lists.
* **order**: fresh store A vs fresh store B. The two are built from different shuffles of the same
  visible log; both the raw witness and the canonical witness are compared.
"""

from __future__ import annotations

import importlib.util
import random
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import ek5_v3_stratified as ek5  # noqa: E402
from chelated_pv7_spine.ek5_reason_canonicalizer import export_canonical_ek5_audit_trace  # noqa: E402


def _load_rc() -> Any:
    path = REPO_ROOT / "formal" / "run_conformance.py"
    name = "w22_11_run_conformance"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load %s" % path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


rc = _load_rc()

SEED = 20261124
SEQ_SEED_MULT = 100003  # same per-sequence seed scheme as W22-05's re-run of the conformance generator
ORDER_RNG_XOR = 0x5EED_0F_F2E5  # second-shuffle RNG is independent of the generation RNG
SEMANTIC_FIELDS: Tuple[str, ...] = ("eff_auth", "val", "cur", "eligibility")
RAW_WITNESS_FIELDS: Tuple[str, ...] = ("reason", "pos_cyc")
MAX_SAMPLES = 40


@dataclass(frozen=True)
class TierConfig:
    """One preregistered tier.  ``generator='baseline'`` uses ``run_conformance._gen_batch`` unchanged."""

    name: str
    generator: str  # "baseline" | "stress"
    max_nodes: int
    steps: int
    sequences: int
    seed_offset: int
    first_nodes: Tuple[int, int] = (3, 5)
    new_nodes_per_step: Tuple[int, int] = (0, 2)
    adds_first: Tuple[int, int] = (2, 6)
    adds_per_step: Tuple[int, int] = (0, 4)
    scc_size: Tuple[int, int] = (3, 8)
    p_plant: float = 0.3
    p_chord: float = 0.3
    p_cycle_refuter: float = 0.5
    p_withdraw: float = 0.6
    max_withdraw: int = 3
    p_remove: float = 0.3
    p_cycle_break: float = 0.5

    def seq_seed(self, i: int) -> int:
        return (SEED + self.seed_offset) * SEQ_SEED_MULT + i


# ---------------------------------------------------------------------------------------------------------------
# Graph helpers
# ---------------------------------------------------------------------------------------------------------------
def nontrivial_scc_membership(nodes: Sequence[str], adj: Mapping[str, Set[str]]) -> Dict[str, int]:
    """node -> size of its non-trivial SCC (self-loop counts as size 1); nodes not on a cycle are absent."""
    node_set = set(nodes)
    out: Dict[str, int] = {}
    for comp in ek5._iterative_tarjan_sccs(node_set, adj, subset=node_set):
        if len(comp) >= 2 or (len(comp) == 1 and comp[0] in adj.get(comp[0], ())):
            for u in comp:
                out[u] = len(comp)
    return out


def snapshot_dv_scc(snapshot: Mapping[str, Any]) -> Dict[str, int]:
    """D_V SCC membership computed only from the raw model snapshot (independent of any store table)."""
    deps = rc.snapshot_deps(snapshot)
    return nontrivial_scc_membership([n["id"] for n in snapshot["nodes"]], deps)


def model_dv_adj(model: Any) -> Dict[str, Set[str]]:
    out: Dict[str, Set[str]] = {}
    for aid in model.active_ids():
        s, d, t, _ = model.att[aid]
        if t in ek5.DEP_ALL_TYPES:
            out.setdefault(s, set()).add(d)
    return out


def scc_bucket(size: int) -> str:
    if size <= 0:
        return "none"
    if size == 1:
        return "self"
    if size == 2:
        return "2"
    if size <= 4:
        return "3-4"
    if size <= 8:
        return "5-8"
    if size <= 16:
        return "9-16"
    if size <= 32:
        return "17-32"
    return "33+"


# ---------------------------------------------------------------------------------------------------------------
# Stress generator (larger graphs, planted D_V SCCs, deletion schedule every step)
# ---------------------------------------------------------------------------------------------------------------
def _new_event(rng: random.Random, model: Any) -> Any:
    nid = "n%03d" % model.node_seq
    model.node_seq += 1
    ent = rng.choice(("e0", "e1", "e2"))
    vt = rng.randint(0, 4)
    ve = None if rng.random() < 0.7 else vt + rng.randint(1, 4)
    event = ek5.StratifiedEvent(
        event_id=nid,
        entity=ent,
        slot="s0",
        claim_value="v%d" % rng.randint(0, 2),
        valid_time=vt,
        tx_time=rng.randint(0, 4),
        valid_end=ve,
        tenant="t0",
        domain=rng.choice(rc.DOMAINS),
        authority_tier=rng.randint(0, 4),
    )
    model.events[nid] = event
    model.nodes.append(nid)
    model.entity[nid] = ent
    return event


def _new_att(rng: random.Random, model: Any, src: str, dst: str, etype: str, lineage: Optional[str] = None) -> Any:
    model.att_seq += 1
    att = ek5.EdgeAttestation(
        attestation_id="a%05d" % model.att_seq,
        src_id=src,
        dst_id=dst,
        edge_type=etype,
        valid_time=rng.randint(0, 4),
        tx_time=rng.randint(0, 4),
        lineage_id=lineage,
    )
    model.add(att)
    return att


def gen_batch_stress(rng: random.Random, model: Any, step: int, cfg: TierConfig) -> Dict[str, Any]:
    """One atomic mutation batch: new events, deletions (withdraw / cycle-break / remove), SCC plant, random adds."""
    new_events: List[Any] = []
    lo, hi = cfg.first_nodes if step == 0 else cfg.new_nodes_per_step
    for _ in range(rng.randint(lo, hi)):
        if len(model.nodes) >= cfg.max_nodes:
            break
        new_events.append(_new_event(rng, model))

    withdraw: List[str] = []
    remove: List[Tuple[str, str, str]] = []
    dep_deleted = 0
    if step > 0:
        # (a) random withdrawals
        active = model.active_ids()
        if active and rng.random() < cfg.p_withdraw:
            for aid in rng.sample(active, k=min(len(active), rng.randint(1, cfg.max_withdraw))):
                withdraw.append(aid)
                model.deactivate(aid)
        # (b) targeted cycle-break: delete one active D_V edge lying inside a non-trivial D_V SCC
        if rng.random() < cfg.p_cycle_break:
            adj = model_dv_adj(model)
            memb = nontrivial_scc_membership(model.nodes, adj)
            comp_of: Dict[str, int] = {}
            if memb:
                # identify which SCC each node sits in, to test "both endpoints in the same SCC"
                for idx, comp in enumerate(ek5._iterative_tarjan_sccs(set(model.nodes), adj, subset=set(model.nodes))):
                    for u in comp:
                        comp_of[u] = idx
            inside = [
                aid
                for aid in model.active_ids()
                if model.att[aid][2] in ek5.DEP_ALL_TYPES
                and model.att[aid][0] in memb
                and comp_of.get(model.att[aid][0]) == comp_of.get(model.att[aid][1])
            ]
            if inside:
                aid = rng.choice(inside)
                if rng.random() < 0.5:
                    withdraw.append(aid)
                    model.deactivate(aid)
                else:
                    triple = model.att[aid][:3]
                    remove.append(triple)
                    model.remove_first(triple)
        # (c) random triple removal
        live = model.active_ids()
        if live and rng.random() < cfg.p_remove:
            triple = model.att[rng.choice(live)][:3]
            remove.append(triple)
            model.remove_first(triple)
    for aid in withdraw:
        if model.att[aid][2] in ek5.DEP_ALL_TYPES:
            dep_deleted += 1
    dep_deleted += sum(1 for t in remove if t[2] in ek5.DEP_ALL_TYPES)

    atts: List[Any] = []
    # (d) planted D_V SCC: a ring of k nodes, optional chords, optional external refuter
    k_lo, k_hi = cfg.scc_size
    if len(model.nodes) >= k_lo and rng.random() < cfg.p_plant:
        k = rng.randint(k_lo, min(k_hi, len(model.nodes)))
        ring = rng.sample(model.nodes, k)
        dep_types = sorted(ek5.DEP_ALL_TYPES)
        for i in range(k):
            atts.append(_new_att(rng, model, ring[i], ring[(i + 1) % k], rng.choice(dep_types)))
        if k >= 4 and rng.random() < cfg.p_chord:
            for _ in range(rng.randint(1, max(1, k // 4))):
                a, b = rng.sample(ring, 2)
                atts.append(_new_att(rng, model, a, b, rng.choice(dep_types)))
        if rng.random() < cfg.p_cycle_refuter:
            outside = [n for n in model.nodes if n not in ring]
            if outside:
                atts.append(_new_att(rng, model, rng.choice(outside), rng.choice(ring), "REFUTE"))

    # (e) random additions, same legality rules as run_conformance._gen_batch
    if not model.strict and step in (1, 2) and len(model.nodes) >= 2 and rng.random() < 0.5:
        a_node, b_node = rng.sample(model.nodes, 2)
        for s_node, d_node in ((a_node, b_node), (b_node, a_node)):
            atts.append(_new_att(rng, model, s_node, d_node, "SUPERSEDE"))
    lo, hi = cfg.adds_first if step == 0 else cfg.adds_per_step
    for _ in range(rng.randint(lo, hi)):
        if len(model.nodes) < 2:
            break
        etype = rc._pick_type(rng)
        src = rng.choice(model.nodes)
        dst = rng.choice(model.nodes)
        lineage: Optional[str] = None
        if etype in ("SUPERSEDE", "VERSION_LINEAGE_SUPERSEDE"):
            if etype == "SUPERSEDE" and not model.strict and rng.random() < 0.4:
                existing = sorted((s, d) for s, ds in model.edges("SUPERSEDE").items() for d in ds)
                if existing:
                    src, dst = rng.choice(existing)
                    src, dst = dst, src
            if etype == "VERSION_LINEAGE_SUPERSEDE":
                peers = [n for n in model.nodes if n != src and model.entity[n] == model.entity[src]]
                if not peers:
                    continue
                dst = rng.choice(peers)
                lineage = "L%d" % rng.randint(0, 2)
            if src == dst:
                continue
            if model.strict and model.would_close(etype, src, dst):
                continue
        elif src == dst and rng.random() < 0.8:
            continue
        reusable = [a for a in model.order if not model.active.get(a)]
        if reusable and rng.random() < 0.1:
            aid = rng.choice(reusable)
            src, dst, etype, lineage = model.att[aid]
            if model.strict and etype in ("SUPERSEDE", "VERSION_LINEAGE_SUPERSEDE") and model.would_close(etype, src, dst):
                continue
            att = ek5.EdgeAttestation(
                attestation_id=aid,
                src_id=src,
                dst_id=dst,
                edge_type=etype,
                valid_time=rng.randint(0, 4),
                tx_time=rng.randint(0, 4),
                lineage_id=lineage,
            )
            model.add(att)
            atts.append(att)
        else:
            atts.append(_new_att(rng, model, src, dst, etype, lineage))
    return {"events": new_events, "atts": atts, "withdraw": withdraw, "remove": remove, "dep_deleted": dep_deleted}


def gen_batch_baseline(rng: random.Random, model: Any, step: int) -> Dict[str, Any]:
    """``run_conformance._gen_batch`` unchanged, plus the D_V-deletion count for characterization."""
    batch = rc._gen_batch(rng, model, step)
    dep_deleted = sum(1 for aid in batch["withdraw"] if model.att[aid][2] in ek5.DEP_ALL_TYPES)
    dep_deleted += sum(1 for t in batch["remove"] if t[2] in ek5.DEP_ALL_TYPES)
    batch["dep_deleted"] = dep_deleted
    return batch


# ---------------------------------------------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------------------------------------------
def canonical_dict(store: Any) -> Dict[str, Any]:
    return export_canonical_ek5_audit_trace(store).to_dict()


def canonical_witness_diffs(a: Mapping[str, Any], b: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """Every difference between two canonical trace dicts (excluding ``semantic_mismatches``, handled separately)."""
    diffs: List[Dict[str, Any]] = []
    na = {n["event_id"]: n for n in a["nodes"]}
    nb = {n["event_id"]: n for n in b["nodes"]}
    if set(na) != set(nb):
        diffs.append({"field": "node_set", "node": None})
    for nid in sorted(set(na) & set(nb)):
        for f in ("val", "cur", "eligibility", "reason", "cause", "pos_cyc", "witness"):
            if na[nid][f] != nb[nid][f]:
                diffs.append({"field": f, "node": nid, "a": na[nid][f], "b": nb[nid][f]})
    for f in ("positive_cycles", "supersede_cycles", "lineage_cycles"):
        if a[f] != b[f]:
            diffs.append({"field": f, "node": None, "a": a[f], "b": b[f]})
    return diffs


def inject_canonical_fault(trace: Mapping[str, Any], kind: int) -> Optional[Dict[str, Any]]:
    """Deterministic corruption of a canonical trace dict (comparator liveness)."""
    if not trace["nodes"]:
        return None
    out = {
        "nodes": [dict(n) for n in trace["nodes"]],
        "positive_cycles": [list(c) for c in trace["positive_cycles"]],
        "supersede_cycles": [list(c) for c in trace["supersede_cycles"]],
        "lineage_cycles": [list(c) for c in trace["lineage_cycles"]],
        "semantic_mismatches": list(trace["semantic_mismatches"]),
    }
    n0 = out["nodes"][0]
    k = kind % 6
    if k == 0:
        n0["reason"] = n0["reason"] + "_X"
    elif k == 1:
        n0["pos_cyc"] = not n0["pos_cyc"]
    elif k == 2:
        n0["witness"] = list(n0["witness"]) + ["__x__"]
    elif k == 3:
        n0["cause"] = n0["cause"] + "_X"
    elif k == 4:
        out["positive_cycles"] = out["positive_cycles"] + [["__x__"]]
    else:
        out["supersede_cycles"] = out["supersede_cycles"] + [["__x__"]]
    return out


def _new_counts() -> Dict[str, Any]:
    return {
        "checkpoints": 0,
        "checkpoints_mutation": 0,
        "checkpoints_as_of": 0,
        "checkpoints_after_dep_deletion": 0,
        "node_comparisons": 0,
        "sequences": 0,
        "max_nodes_seen": 0,
        "max_dv_scc_seen": 0,
        "dv_scc_nodes_total": 0,
        "dv_scc_size_hist": Counter(),
        "deletions": Counter(),
        "semantic": Counter(),
        "raw_witness_inc_vs_fresh": Counter(),
        "raw_witness_inc_vs_fresh_checkpoints": 0,
        "raw_witness_fresh_vs_fresh": Counter(),
        "canonical_inc_vs_fresh": Counter(),
        "canonical_inc_vs_fresh_checkpoints": 0,
        "canonical_fresh_vs_fresh": Counter(),
        "canonical_vs_fresh_raw": Counter(),
        "canonical_pos_cyc_vs_snapshot_scc": 0,
        "raw_diff_characterization": Counter(),
        "liveness": Counter(),
        "samples": {"semantic": [], "raw_witness": [], "canonical": [], "order": []},
    }


def _sample(counts: Dict[str, Any], key: str, item: Dict[str, Any]) -> None:
    if len(counts["samples"][key]) < MAX_SAMPLES:
        counts["samples"][key].append(item)


def check_checkpoint(
    counts: Dict[str, Any],
    *,
    tier: str,
    seq_seed: int,
    step: int,
    kind: str,
    store: Any,
    model: Any,
    rng: random.Random,
    rng_order: random.Random,
    cutoff: Optional[Tuple[int, int]],
    dep_deleted: int,
    liveness: bool,
) -> None:
    """Run every preregistered comparison at one checkpoint and fold the result into ``counts``."""
    cp_id = "%s:%d:%d:%s" % (tier, seq_seed, step, kind)
    snapshot = rc.model_snapshot(model, cp_id, cutoff)
    inc_snap = rc.materialize(store, cp_id)
    fresh = rc.fresh_from_model(model, snapshot, rng, cutoff)  # consumes the generation RNG exactly like rc
    fresh2 = rc.fresh_from_model(model, snapshot, rng_order, cutoff)
    inc_state = rc.store_state(store)
    fresh_state = rc.store_state(fresh)
    fresh2_state = rc.store_state(fresh2)
    oracle = rc.oracle_vals(snapshot)
    scc = snapshot_dv_scc(snapshot)

    counts["checkpoints"] += 1
    counts["checkpoints_" + kind] += 1
    if dep_deleted:
        counts["checkpoints_after_dep_deletion"] += 1
    counts["node_comparisons"] += len(inc_state["nodes"])
    counts["max_nodes_seen"] = max(counts["max_nodes_seen"], len(snapshot["nodes"]))
    counts["dv_scc_nodes_total"] += len(scc)
    if scc:
        counts["max_dv_scc_seen"] = max(counts["max_dv_scc_seen"], max(scc.values()))
    # SCC size histogram (count each SCC once: size-weighted node count / size)
    sizes = Counter(scc.values())
    for size, nodes_in in sizes.items():
        counts["dv_scc_size_hist"][scc_bucket(size)] += nodes_in // size if size > 1 else nodes_in

    # ---- semantic ---------------------------------------------------------------------------------------------
    sem: List[Dict[str, Any]] = []
    if not rc.snapshots_equal(inc_snap, snapshot):
        sem.append({"field": "stratum0_materialization", "node": None})
    for d in rc.compare_states(inc_state, fresh_state, fields=SEMANTIC_FIELDS, telemetry=False):
        sem.append(d)
    for nid, ov in oracle.items():
        if inc_state["nodes"].get(nid, {}).get("val") != ov:
            sem.append({"field": "val_vs_oracle", "node": nid, "a": inc_state["nodes"].get(nid, {}).get("val"), "b": ov})
    c_inc = canonical_dict(store)
    c_fr = canonical_dict(fresh)
    c_fr2 = canonical_dict(fresh2)
    for label, c in (("inc", c_inc), ("fresh", c_fr), ("fresh2", c_fr2)):
        for m in c["semantic_mismatches"]:
            sem.append({"field": "canonical_crosscheck_" + label, "node": m})
    for d in sem:
        counts["semantic"][d["field"]] += 1
        _sample(counts, "semantic", dict(d, cp=cp_id))

    # ---- raw witness: inc vs fresh, fresh vs fresh ------------------------------------------------------------
    raw = rc.compare_states(inc_state, fresh_state, fields=RAW_WITNESS_FIELDS, telemetry=True)
    raw = [d for d in raw if d["field"] != "node_set"]
    if raw:
        counts["raw_witness_inc_vs_fresh_checkpoints"] += 1
    order_raw = rc.compare_states(fresh_state, fresh2_state, fields=RAW_WITNESS_FIELDS, telemetry=True)
    order_raw = [d for d in order_raw if d["field"] != "node_set"]
    order_nodes = {(d["field"], d["node"]) for d in order_raw}
    for d in order_raw:
        counts["raw_witness_fresh_vs_fresh"][d["field"]] += 1
        _sample(counts, "order", dict(d, cp=cp_id, cls="raw"))
    for d in raw:
        counts["raw_witness_inc_vs_fresh"][d["field"]] += 1
        nid = d["node"]
        if nid is not None:
            size = scc.get(nid, 0)
            if d["field"] == "pos_cyc":
                direction = "inc_missed_cycle" if (not d["a"] and d["b"]) else "inc_stale_cycle"
            else:
                direction = "%s->%s" % (d["a"], d["b"])
            char = (
                tier,
                kind,
                d["field"],
                direction,
                "fresh_scc=" + scc_bucket(size),
                "val=" + str(inc_state["nodes"][nid]["val"]),
                "dep_del_in_batch=" + str(bool(dep_deleted)),
                "fresh_order_dependent=" + str((d["field"], nid) in order_nodes),
            )
            counts["raw_diff_characterization"]["|".join(char)] += 1
        _sample(counts, "raw_witness", dict(d, cp=cp_id, fresh_scc=scc.get(nid, 0) if nid else None))

    # ---- canonical witness ------------------------------------------------------------------------------------
    cdiff = canonical_witness_diffs(c_inc, c_fr)
    if cdiff:
        counts["canonical_inc_vs_fresh_checkpoints"] += 1
    for d in cdiff:
        counts["canonical_inc_vs_fresh"][d["field"]] += 1
        _sample(counts, "canonical", dict(d, cp=cp_id))
    for d in canonical_witness_diffs(c_fr, c_fr2):
        counts["canonical_fresh_vs_fresh"][d["field"]] += 1
        _sample(counts, "order", dict(d, cp=cp_id, cls="canonical"))
    # contract: canonical export == the fresh store's raw telemetry; canonical pos_cyc == snapshot SCC membership
    for n in c_inc["nodes"]:
        fn = fresh_state["nodes"].get(n["event_id"])
        if fn is None:
            continue
        if n["reason"] != fn["reason"]:
            counts["canonical_vs_fresh_raw"]["reason"] += 1
        if n["pos_cyc"] != fn["pos_cyc"]:
            counts["canonical_vs_fresh_raw"]["pos_cyc"] += 1
        if n["pos_cyc"] != (n["event_id"] in scc):
            counts["canonical_pos_cyc_vs_snapshot_scc"] += 1

    # ---- comparator liveness ----------------------------------------------------------------------------------
    if liveness:
        for kind_i in range(6):
            bad = inject_canonical_fault(c_inc, kind_i)
            if bad is None:
                continue
            counts["liveness"]["canonical_injected"] += 1
            if canonical_witness_diffs(bad, c_fr):
                counts["liveness"]["canonical_detected"] += 1
        for kind_i in (0, 1, 2, 3, 8):
            bad = rc.inject_fault(inc_state, kind_i)
            if bad is None:
                continue
            counts["liveness"]["semantic_injected"] += 1
            if rc.compare_states(bad, fresh_state, fields=SEMANTIC_FIELDS, telemetry=False):
                counts["liveness"]["semantic_detected"] += 1


def run_sequence(cfg: TierConfig, i: int, counts: Dict[str, Any], liveness_every: int = 50) -> None:
    """One seeded multi-mutation sequence with a checkpoint after every batch plus two ``as_of`` rewinds."""
    seed = cfg.seq_seed(i)
    rng = random.Random(seed)
    rng_order = random.Random(seed ^ ORDER_RNG_XOR)
    strict = rng.random() < 0.6
    store = ek5.StratifiedEvidenceStore(strict_dag_supersede=strict)
    model = rc._Model(strict)
    if cfg.generator == "baseline" and cfg.max_nodes != rc.MAX_NODES:
        raise ValueError("baseline generator is fixed at run_conformance.MAX_NODES=%d" % rc.MAX_NODES)
    liveness = liveness_every > 0 and i % liveness_every == 0
    for step in range(cfg.steps):
        if cfg.generator == "baseline":
            batch = gen_batch_baseline(rng, model, step)
        else:
            batch = gen_batch_stress(rng, model, step, cfg)
        counts["deletions"]["withdraw"] += len(batch["withdraw"])
        counts["deletions"]["remove"] += len(batch["remove"])
        counts["deletions"]["dep_edges"] += batch["dep_deleted"]
        store.apply_mutation(batch["events"], batch["atts"], batch["withdraw"], remove_edges=batch["remove"])
        check_checkpoint(
            counts, tier=cfg.name, seq_seed=seed, step=step, kind="mutation", store=store, model=model,
            rng=rng, rng_order=rng_order, cutoff=None, dep_deleted=batch["dep_deleted"], liveness=liveness,
        )
    grid = [(vc, tc) for vc in range(0, 8) for tc in range(0, 8)]
    missing_grid = [g for g in grid if rc.model_visible(model, g)[2]]
    picks: List[Tuple[int, int]] = []
    if missing_grid:
        picks.append(rng.choice(missing_grid))
    picks.append(rng.choice(grid))
    for j, (vc, tc) in enumerate(picks):
        snap = store.as_of(vc, tc)
        check_checkpoint(
            counts, tier=cfg.name, seq_seed=seed, step=cfg.steps + j, kind="as_of", store=snap, model=model,
            rng=rng, rng_order=rng_order, cutoff=(vc, tc), dep_deleted=0, liveness=False,
        )
    counts["sequences"] += 1


def run_chunk(args: Tuple[Dict[str, Any], int, int]) -> Dict[str, Any]:
    """Worker entry point: run sequences ``[lo, hi)`` of one tier and return JSON-able counts."""
    cfg_d, lo, hi = args
    cfg_d = dict(cfg_d)
    for key in ("first_nodes", "new_nodes_per_step", "adds_first", "adds_per_step", "scc_size"):
        cfg_d[key] = tuple(cfg_d[key])
    cfg = TierConfig(**cfg_d)
    counts = _new_counts()
    for i in range(lo, hi):
        run_sequence(cfg, i, counts)
    return to_jsonable(counts)


def to_jsonable(counts: Mapping[str, Any]) -> Dict[str, Any]:
    return {k: (dict(v) if isinstance(v, Counter) else v) for k, v in counts.items()}


def merge_counts(acc: Dict[str, Any], part: Mapping[str, Any]) -> Dict[str, Any]:
    """Merge two JSON-able count dicts (sums for ints and Counters; max for max_*; capped sample lists)."""
    for k, v in part.items():
        if k == "samples":
            s = acc.setdefault("samples", {})
            for sk, sv in v.items():
                cur = s.setdefault(sk, [])
                cur.extend(sv[: max(0, MAX_SAMPLES - len(cur))])
        elif k.startswith("max_"):
            acc[k] = max(acc.get(k, 0), v)
        elif isinstance(v, dict):
            d = acc.setdefault(k, {})
            for kk, vv in v.items():
                d[kk] = d.get(kk, 0) + vv
        else:
            acc[k] = acc.get(k, 0) + v
    return acc


def tier_dict(cfg: TierConfig) -> Dict[str, Any]:
    return asdict(cfg)


def summarize(counts: Mapping[str, Any]) -> Dict[str, int]:
    return {
        "semantic_total": sum(counts.get("semantic", {}).values()),
        "raw_witness_inc_vs_fresh_total": sum(counts.get("raw_witness_inc_vs_fresh", {}).values()),
        "raw_witness_fresh_vs_fresh_total": sum(counts.get("raw_witness_fresh_vs_fresh", {}).values()),
        "canonical_inc_vs_fresh_total": sum(counts.get("canonical_inc_vs_fresh", {}).values()),
        "canonical_fresh_vs_fresh_total": sum(counts.get("canonical_fresh_vs_fresh", {}).values()),
        "canonical_vs_fresh_raw_total": sum(counts.get("canonical_vs_fresh_raw", {}).values()),
        "canonical_pos_cyc_vs_snapshot_scc": int(counts.get("canonical_pos_cyc_vs_snapshot_scc", 0)),
    }
