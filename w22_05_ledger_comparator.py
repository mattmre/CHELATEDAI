"""W22-05: same-input ordinary bitemporal ledger comparator + real-graph negation-cycle census.

The comparison is between the EK5 v3 stratified evidence kernel (via the arena's
``AuthorityDominatedExocortexV2`` and bundle #55's own projection) and the ordinary
SQL challenger ledger in ``chelated_pv7_spine.challenger_ledger``, which shares no
code with EK5.  Both arms receive the same inputs: matched timestamps, channel
authority, stances, and copy edges.

The kill/advance rule is preregistered in
``artifacts/pv7_w23a_w22_05/PREREGISTRATION.md``.  ``kill_rule_decision`` implements
it verbatim.

All functions are deterministic and CPU-only.  No LLM or embedding model is loaded.
"""

from __future__ import annotations

import random
import re
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np

from chelated_epistemic_proposer_v2 import AuthorityDominatedExocortexV2
from chelated_pv7_spine.challenger_ledger import (
    ChallengerBitemporalLedger,
    LedgerEdgeEnvelope,
    LedgerRecordEnvelope,
    SQLiteChallengerBackend,
)
from chelated_raw_text_exocortex import ExtractedClaimRecord, RawDocumentEnvelope, canonicalize_value

ABSTAIN = "ABSTAIN"
PREVALENCES: Tuple[float, ...] = (0.01, 0.05, 0.25)
PARITY_THRESHOLD = 0.99
HARM_TOLERANCE = 0.005
COVERAGE_TOLERANCE = 0.005
NEGATIVE_EDGE_TYPES = frozenset({"REFUTE", "SUPERSEDE", "VERSION_LINEAGE_SUPERSEDE"})
LEDGER_QUERY_VALID_TIME = 10 ** 6
LEDGER_MODES: Tuple[str, ...] = ("recursive_cte", "closure_table")


def normalize_value(raw: str) -> str:
    """The exact value normalization ``compile_claims_to_ek5_v2`` applies (canonicalize, collapse, lower)."""
    return re.sub(r"\s+", " ", canonicalize_value(raw).strip().lower())


# ---------------------------------------------------------------------------------------------------------------
# Arena 60: deterministic role-derived ("oracle") extraction shared by both arms
# ---------------------------------------------------------------------------------------------------------------
def arena_doc_role(case: Mapping[str, Any], doc_id: str) -> Tuple[str, str]:
    """Return ``(stance, raw_value)`` for one arena document from its generator role.

    Stances follow ``EXTRACTION_SYSTEM_PROMPT_V2``.  Raises ``ValueError`` for an unknown role,
    so a generator change cannot silently produce a mislabelled claim.
    """
    cid = case["case_id"]
    if not doc_id.startswith(cid + "_"):
        raise ValueError(f"{doc_id} does not belong to {cid}")
    role = doc_id[len(cid) + 1:]
    regime = case["regime"]
    gold = case["gold"]
    vals = list(case["stale_or_false_values"])
    if regime == "supersession_under_echo_flood":
        if role == "d12_official_supersede":
            return "SUPERSEDE", gold
        if role == "d00_prelim" or role.startswith("echo_") or role.endswith("_late_archive"):
            return "ASSERT", vals[0]
    elif regime == "rumor_laundering_and_audit_trap":
        if role == "d11_official_debunk":
            return "DEBUNK_RUMOR", gold
        if role == "d00_seed_rumor" or role.startswith("echo_") or role.endswith("_late_launder"):
            return "ASSERT", vals[0]
    elif regime == "multihop_version_lineage_drift":
        if role == "d00_v1" or role.endswith("_v1_echo"):
            return "ASSERT", vals[0]
        if role == "d01_v2":
            return "SUPERSEDE", vals[1]
        if role.startswith("echo_"):
            return "ASSERT", vals[1]
        if role == "d14_v3_final":
            return "SUPERSEDE", gold
    elif regime == "mutual_peer_conflict_abstain":
        if role == "d00_party_a":
            return "REFUTE", vals[0]
        if role == "d01_party_b":
            return "REFUTE", vals[1]
        if role.startswith("echo_b_"):
            return "ASSERT", vals[1]
        if role.startswith("echo_"):
            return "ASSERT", vals[0]
    raise ValueError(f"unknown arena role {role!r} in regime {regime!r}")


def arena_claims(case: Mapping[str, Any], docs: Sequence[RawDocumentEnvelope]) -> List[ExtractedClaimRecord]:
    """One claim per doc.  Entity = case subject, single canonical slot, value and stance from the role."""
    out = []
    for d in docs:
        stance, value = arena_doc_role(case, d.doc_id)
        out.append(
            ExtractedClaimRecord(
                doc_id=d.doc_id,
                entity=case["subject"],
                slot="case_attribute",
                claim_value=value,
                stance=stance,
                raw_json="",
            )
        )
    return out


def build_arena_exocortex(
    docs: Sequence[RawDocumentEnvelope], claims: Sequence[ExtractedClaimRecord]
) -> AuthorityDominatedExocortexV2:
    """The arena's EK5 arm, with deterministic one-hot vectors.

    One-hot doc vectors give zero pairwise cosine.  Slot vectors are identical for identical slot strings.
    """
    n = len(docs)
    doc_emb = np.eye(n, dtype=np.float32)
    slot_names = sorted({c.slot for c in claims})
    slot_idx = {s: i for i, s in enumerate(slot_names)}
    slot_emb = np.zeros((n, len(slot_names)), dtype=np.float32)
    for i, c in enumerate(claims):
        slot_emb[i, slot_idx[c.slot]] = 1.0
    return AuthorityDominatedExocortexV2(docs, claims, doc_emb, slot_emb)


def ek5_arena_verdict(exo: AuthorityDominatedExocortexV2, query_text: str) -> Dict[str, Any]:
    """Set-level EK5 verdict.

    ``ABSTAIN`` if the exocortex's CONTESTED rule fires.  Otherwise the verdict is the sorted
    tuple of distinct admitted values.  ``select_context_for_query`` only decides the
    abstention; the value set is read off ``admit_mask`` before echo de-duplication.
    """
    n = len(exo.doc_ids)
    ctx = exo.select_context_for_query(np.ones(n, dtype=np.float32), query_text, top_k=n)
    slots = sorted({(ev.tenant, ev.entity, ev.slot) for ev in exo.events.values()})
    if ctx["status"] == "CONTESTED_ABSTAIN":
        verdict: Tuple[str, ...] = (ABSTAIN,)
    else:
        verdict = tuple(sorted({exo.events[did].claim_value for i, did in enumerate(exo.doc_ids) if exo.admit_mask[i]}))
    return {
        "verdict": verdict,
        "status": ctx["status"],
        "n_slots": len(slots),
        "admitted_ids": [did for i, did in enumerate(exo.doc_ids) if exo.admit_mask[i]],
        "contested_ids": [did for i, did in enumerate(exo.doc_ids) if exo.contested_mask[i]],
    }


def arena_ledger_ops(
    exo: AuthorityDominatedExocortexV2, *, authority_only: bool = False
) -> Tuple[List[Tuple[str, Any]], Dict[str, str]]:
    """Project the exocortex's own cleaned claims and events (identical input) into ordinary ledger ops.

    Returns ``(ops, entity_key_by_slot)``, where ops is a list of ``("record", envelope)`` or
    ``("edge", envelope)``.  With ``authority_only`` set, every claim is a plain ASSERT and no
    WITHDRAW, CONFLICT or DERIVES ops are emitted (ablation).
    """
    stance_by_id = {c.doc_id: c.stance for c in exo.cleaned_claims}
    ordered = sorted(exo.events.values(), key=lambda e: (e.tx_time, e.valid_time, e.event_id))
    ops: List[Tuple[str, Any]] = []
    seen: List[Any] = []
    first_by_val: Dict[Tuple[str, str], str] = {}
    keys: Dict[str, str] = {}
    for ev in ordered:
        ekey = f"{ev.entity}|{ev.slot}"
        keys[ev.slot] = ekey
        stance = stance_by_id[ev.event_id]
        kind = "CONFLICT" if (stance == "REFUTE" and not authority_only) else "ASSERT"
        ops.append(
            (
                "record",
                LedgerRecordEnvelope(
                    tenant_id=ev.tenant,
                    record_id=ev.event_id,
                    entity_key=ekey,
                    authority_rank=int(ev.authority_tier),
                    valid_from=int(ev.valid_time),
                    valid_to=None,
                    tx_time=int(ev.tx_time),
                    op_kind=kind,
                    claim_value=ev.claim_value,
                ),
            )
        )
        if not authority_only:
            vkey = (ekey, ev.claim_value)
            origin = first_by_val.get(vkey)
            if origin is None:
                first_by_val[vkey] = ev.event_id
            elif stance == "ASSERT":
                ops.append(
                    (
                        "edge",
                        LedgerEdgeEnvelope(
                            tenant_id=ev.tenant,
                            edge_id=f"der:{ev.event_id}:{origin}",
                            src_id=ev.event_id,
                            dst_id=origin,
                            tx_time=int(ev.tx_time),
                        ),
                    )
                )
            if stance in {"SUPERSEDE", "DEBUNK_RUMOR"}:
                for prev in seen:
                    if f"{prev.entity}|{prev.slot}" != ekey or prev.claim_value == ev.claim_value:
                        continue
                    ops.append(
                        (
                            "record",
                            LedgerRecordEnvelope(
                                tenant_id=ev.tenant,
                                record_id=prev.event_id,
                                entity_key=ekey,
                                authority_rank=int(ev.authority_tier),
                                valid_from=int(prev.valid_time),
                                valid_to=None,
                                tx_time=int(ev.tx_time),
                                op_kind="WITHDRAW",
                                claim_value="",
                            ),
                        )
                    )
        seen.append(ev)
    return ops, keys


def apply_ops(ledger: ChallengerBitemporalLedger, ops: Iterable[Tuple[str, Any]]) -> None:
    for kind, env in ops:
        if kind == "record":
            ledger.append(env)
        elif kind == "edge":
            ledger.add_edge(env)
        else:
            raise ValueError(f"unknown op kind {kind!r}")


def ledger_set_verdict(result: Any) -> Tuple[str, ...]:
    """Set-level verdict of an ``AsOfQueryResult``: RESOLVED gives (value,), DISPUTED gives ABSTAIN, else ()."""
    if result.status == "RESOLVED":
        return (result.winner_claim_value,)
    if result.status == "DISPUTED":
        return (ABSTAIN,)
    return ()


def gold_verdict(case: Mapping[str, Any]) -> Tuple[str, ...]:
    return (ABSTAIN,) if case["gold"] == ABSTAIN else (normalize_value(case["gold"]),)


def run_arena_case(case: Mapping[str, Any], docs: Sequence[RawDocumentEnvelope]) -> Dict[str, Any]:
    """Run both arms plus the authority-only ablation on one arena case.  Returns one row."""
    claims = arena_claims(case, docs)
    exo = build_arena_exocortex(docs, claims)
    ek5 = ek5_arena_verdict(exo, case["query_text"])
    tenant = docs[0].tenant
    row: Dict[str, Any] = {
        "case_id": case["case_id"],
        "domain": case["domain"],
        "regime": case["regime"],
        "gold": list(gold_verdict(case)),
        "is_conflict": case["gold"] == ABSTAIN,
        "ek5_verdict": list(ek5["verdict"]),
        "ek5_status": ek5["status"],
        "ek5_n_slots": ek5["n_slots"],
    }
    for label, authority_only in (("ledger", False), ("ledger_authority_only", True)):
        ops, keys = arena_ledger_ops(exo, authority_only=authority_only)
        if len(keys) != 1:
            raise RuntimeError(f"{case['case_id']}: expected one slot, got {sorted(keys)}")
        ekey = next(iter(keys.values()))
        tx = max(env.tx_time for _k, env in ops)
        led = ChallengerBitemporalLedger(SQLiteChallengerBackend())
        apply_ops(led, ops)
        verdicts = {}
        for mode in LEDGER_MODES:
            verdicts[mode] = led.query_as_of(tenant, ekey, LEDGER_QUERY_VALID_TIME, tx, mode=mode)
        led.close()
        if verdicts["recursive_cte"].semantic_key() != verdicts["closure_table"].semantic_key():
            raise RuntimeError(f"{case['case_id']}: ledger modes disagree")
        res = verdicts["closure_table"]
        row[f"{label}_verdict"] = list(ledger_set_verdict(res))
        row[f"{label}_status"] = res.status
        row[f"{label}_ops"] = len(ops)
    gold = tuple(row["gold"])
    for arm in ("ek5", "ledger", "ledger_authority_only"):
        row[f"{arm}_correct"] = tuple(row[f"{arm}_verdict"]) == gold
    row["parity"] = row["ek5_verdict"] == row["ledger_verdict"]
    row["_exo"] = exo
    return row


# ---------------------------------------------------------------------------------------------------------------
# Generic paired metrics + the preregistered kill/advance rule
# ---------------------------------------------------------------------------------------------------------------
def _rate(xs: Sequence[bool]) -> Optional[float]:
    return (sum(1 for x in xs if x) / len(xs)) if xs else None


def harm_at(err_conflict: Optional[float], err_det: Optional[float], p: float) -> Optional[float]:
    """``p*err_conflict + (1-p)*err_determinate``.  Undefined (None) if either subset is empty."""
    if err_conflict is None or err_det is None:
        return None
    return p * err_conflict + (1.0 - p) * err_det


def paired_metrics(items: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Compute paired metrics over a list of items.

    Each item carries ``is_conflict``, ``ek5_verdict``, ``ledger_verdict``, ``ek5_correct``,
    ``ledger_correct``, ``ek5_covered`` and ``ledger_covered``.  The function returns parity,
    win counts, per-subset error, prevalence-weighted harm and coverage.
    """
    n = len(items)
    parity = _rate([tuple(i["ek5_verdict"]) == tuple(i["ledger_verdict"]) for i in items])
    conf = [i for i in items if i["is_conflict"]]
    det = [i for i in items if not i["is_conflict"]]
    out: Dict[str, Any] = {
        "n": n,
        "n_conflict": len(conf),
        "n_determinate": len(det),
        "set_level_parity": parity,
        "parity_mismatches": sum(1 for i in items if tuple(i["ek5_verdict"]) != tuple(i["ledger_verdict"])),
        "ek5_conflict_wins": sum(1 for i in conf if i["ek5_correct"] and not i["ledger_correct"]),
        "ledger_conflict_wins": sum(1 for i in conf if i["ledger_correct"] and not i["ek5_correct"]),
        "ek5_determinate_wins": sum(1 for i in det if i["ek5_correct"] and not i["ledger_correct"]),
        "ledger_determinate_wins": sum(1 for i in det if i["ledger_correct"] and not i["ek5_correct"]),
    }
    for arm in ("ek5", "ledger"):
        acc_c = _rate([i[f"{arm}_correct"] for i in conf])
        acc_d = _rate([i[f"{arm}_correct"] for i in det])
        err_c = None if acc_c is None else 1.0 - acc_c
        err_d = None if acc_d is None else 1.0 - acc_d
        out[arm] = {
            "correct": sum(1 for i in items if i[f"{arm}_correct"]),
            "accuracy": _rate([i[f"{arm}_correct"] for i in items]),
            "err_conflict": err_c,
            "err_determinate": err_d,
            "harm": {str(p): harm_at(err_c, err_d, p) for p in PREVALENCES},
            "coverage": _rate([i[f"{arm}_covered"] for i in items]),
        }
    return out


def kill_rule_checks(m: Mapping[str, Any]) -> Dict[str, Any]:
    """Conditions (a), (b) and (c) of the preregistered rule on one metrics block."""
    parity = m["set_level_parity"]
    a = parity is not None and parity >= PARITY_THRESHOLD
    b = m["ek5_conflict_wins"] == 0
    harm_ok = {}
    for p in PREVALENCES:
        hl, he = m["ledger"]["harm"][str(p)], m["ek5"]["harm"][str(p)]
        harm_ok[str(p)] = None if (hl is None or he is None) else hl <= he + HARM_TOLERANCE
    cov_ok = m["ledger"]["coverage"] >= m["ek5"]["coverage"] - COVERAGE_TOLERANCE
    c = all(v is True for v in harm_ok.values()) and cov_ok
    return {
        "a_parity_ge_0_99": a,
        "b_zero_ek5_conflict_wins": b,
        "c_harm_equivalent_by_p": harm_ok,
        "c_coverage_equivalent": cov_ok,
        "c_harm_and_coverage_equivalent": c,
        "all_hold": bool(a and b and c),
    }


def kill_rule_decision(blocks: Mapping[str, Mapping[str, Any]], gates: Mapping[str, bool]) -> Dict[str, Any]:
    """Apply the preregistered rule.

    ``blocks`` maps corpus name (plus ``"pooled"``) to a ``paired_metrics`` result, and
    ``gates`` maps corpus name to whether it passed its reproduction gate.  Corpora that
    failed their gate are excluded and the verdict gets the suffix ``_PARTIAL_INPUTS``.
    """
    included = [k for k in blocks if k != "pooled" and gates.get(k, False)]
    excluded = [k for k in blocks if k != "pooled" and not gates.get(k, False)]
    checks = {k: kill_rule_checks(blocks[k]) for k in included}
    if "pooled" in blocks:
        checks["pooled"] = kill_rule_checks(blocks["pooled"])
    retire = bool(included) and all(c["all_hold"] for c in checks.values())
    verdict = "RETIRE_EK5_CLEAN_GRAPH_PREMIUM" if retire else "ADVANCE_EK5_RETAINS_MEASURED_PREMIUM"
    if excluded:
        verdict += "_PARTIAL_INPUTS"
    return {"verdict": verdict, "included_corpora": included, "excluded_corpora": excluded, "checks": checks}


# ---------------------------------------------------------------------------------------------------------------
# Real-graph negation-cycle census
# ---------------------------------------------------------------------------------------------------------------
def strongly_connected_components(nodes: Iterable[str], adj: Mapping[str, Sequence[str]]) -> List[List[str]]:
    """Iterative Tarjan.  Deterministic for a fixed node order."""
    index: Dict[str, int] = {}
    low: Dict[str, int] = {}
    on_stack: Set[str] = set()
    stack: List[str] = []
    comps: List[List[str]] = []
    counter = 0
    for root in nodes:
        if root in index:
            continue
        work = [(root, 0)]
        while work:
            v, i = work.pop()
            if i == 0:
                index[v] = low[v] = counter
                counter += 1
                stack.append(v)
                on_stack.add(v)
            succ = adj.get(v, ())
            recurse = False
            while i < len(succ):
                w = succ[i]
                i += 1
                if w not in index:
                    work.append((v, i))
                    work.append((w, 0))
                    recurse = True
                    break
                if w in on_stack:
                    low[v] = min(low[v], index[w])
            if recurse:
                continue
            if low[v] == index[v]:
                comp = []
                while True:
                    w = stack.pop()
                    on_stack.discard(w)
                    comp.append(w)
                    if w == v:
                        break
                comps.append(sorted(comp))
            if work:
                parent = work[-1][0]
                low[parent] = min(low[parent], low[v])
    return comps


def negation_cycle_census(nodes: Sequence[str], edges: Sequence[Tuple[str, str, str]]) -> Dict[str, Any]:
    """Count cycles through at least one negative edge in one attestation graph.

    A negation cycle is a non-trivial SCC (size > 1, or a self-loop) whose internal edges
    include REFUTE, SUPERSEDE or VERSION_LINEAGE_SUPERSEDE.  Edge endpoints that are not in
    ``nodes`` are added as nodes.
    """
    node_list = list(dict.fromkeys(list(nodes) + [s for s, _d, _t in edges] + [d for _s, d, _t in edges]))
    adj: Dict[str, List[str]] = defaultdict(list)
    for s, d, _t in edges:
        adj[s].append(d)
    comp_of: Dict[str, int] = {}
    comps = strongly_connected_components(node_list, adj)
    for ci, comp in enumerate(comps):
        for v in comp:
            comp_of[v] = ci
    neg_in: Dict[int, int] = defaultdict(int)
    pos_in: Dict[int, int] = defaultdict(int)
    self_loop: Set[int] = set()
    for s, d, t in edges:
        if comp_of[s] == comp_of[d]:
            if s == d:
                self_loop.add(comp_of[s])
            if t in NEGATIVE_EDGE_TYPES:
                neg_in[comp_of[s]] += 1
            else:
                pos_in[comp_of[s]] += 1
    nontrivial = [ci for ci, comp in enumerate(comps) if len(comp) > 1 or ci in self_loop]
    neg_cycles = [ci for ci in nontrivial if neg_in[ci] > 0]
    pairs = {(s, d) for s, d, t in edges if t == "REFUTE"}
    mutual = {tuple(sorted((s, d))) for s, d in pairs if (d, s) in pairs and s != d}
    return {
        "nodes": len(node_list),
        "edges": len(edges),
        "negative_edges": sum(1 for _s, _d, t in edges if t in NEGATIVE_EDGE_TYPES),
        "nontrivial_sccs": len(nontrivial),
        "negation_cycle_sccs": len(neg_cycles),
        "nodes_in_negation_cycles": sum(len(comps[ci]) for ci in neg_cycles),
        "purely_positive_cycle_sccs": sum(1 for ci in nontrivial if neg_in[ci] == 0),
        "mutual_refute_2cycles": len(mutual),
        "max_negation_scc_size": max((len(comps[ci]) for ci in neg_cycles), default=0),
    }


def census_summary(per_graph: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Aggregate per-graph censuses.

    Each entry carries ``is_conflict`` and ``census``.  The summary reports raw incidence and
    incidence weighted by conflict prevalence.
    """
    n = len(per_graph)
    has = [g["census"]["negation_cycle_sccs"] > 0 for g in per_graph]
    conf = [h for g, h in zip(per_graph, has) if g["is_conflict"]]
    det = [h for g, h in zip(per_graph, has) if not g["is_conflict"]]
    inc_c, inc_d = _rate(conf), _rate(det)
    return {
        "graphs": n,
        "graphs_with_negation_cycle": sum(has),
        "raw_incidence": _rate(has),
        "incidence_conflict_graphs": inc_c,
        "incidence_determinate_graphs": inc_d,
        "n_conflict_graphs": len(conf),
        "n_determinate_graphs": len(det),
        "prevalence_weighted_incidence": {str(p): harm_at(inc_c, inc_d, p) for p in PREVALENCES},
        "total_negation_cycle_sccs": sum(g["census"]["negation_cycle_sccs"] for g in per_graph),
        "total_mutual_refute_2cycles": sum(g["census"]["mutual_refute_2cycles"] for g in per_graph),
        "total_purely_positive_cycle_sccs": sum(g["census"]["purely_positive_cycle_sccs"] for g in per_graph),
        "total_edges": sum(g["census"]["edges"] for g in per_graph),
        "total_negative_edges": sum(g["census"]["negative_edges"] for g in per_graph),
    }


# ---------------------------------------------------------------------------------------------------------------
# Ledger incremental-vs-fresh explanation divergence (counterpart to /formal's 153-962)
# ---------------------------------------------------------------------------------------------------------------
def ledger_rebuild_divergence(
    ops: Sequence[Tuple[str, Any]],
    queries: Sequence[Tuple[str, str, int, int]],
    *,
    seed: int = 20261126,
) -> Dict[str, Any]:
    """Compare incremental, as-of-time answers against two from-scratch rebuilds of the same op log.

    - Incremental: ops are appended in tx order, and each query runs as soon as every op with
      ``tx <= query tx`` has been appended.
    - Fresh (in order): all ops in tx order, then all queries.
    - Fresh (shuffled): all ops in a seeded shuffled order, then all queries.

    The function counts ``AsOfQueryResult.digest()`` mismatches, broken down by field.  All
    three evaluations use ``closure_table`` mode.
    """
    ordered = sorted(range(len(ops)), key=lambda i: (ops[i][1].tx_time, i))
    qs = sorted(range(len(queries)), key=lambda j: (queries[j][3], j))
    inc = ChallengerBitemporalLedger(SQLiteChallengerBackend())
    inc_res: Dict[int, Any] = {}
    pos = 0
    for j in qs:
        tenant, ekey, vt, tx = queries[j]
        while pos < len(ordered) and ops[ordered[pos]][1].tx_time <= tx:
            apply_ops(inc, [ops[ordered[pos]]])
            pos += 1
        inc_res[j] = inc.query_as_of(tenant, ekey, vt, tx, mode="closure_table")
    inc.close()

    def _fresh(order: Sequence[int]) -> Dict[int, Any]:
        led = ChallengerBitemporalLedger(SQLiteChallengerBackend())
        apply_ops(led, [ops[i] for i in order])
        res = {j: led.query_as_of(*queries[j], mode="closure_table") for j in range(len(queries))}
        led.close()
        return res

    shuffled = list(range(len(ops)))
    random.Random(seed).shuffle(shuffled)
    out: Dict[str, Any] = {"queries": len(queries), "ops": len(ops)}
    fields = ("status", "winner_record_id", "winner_claim_value", "contending_record_ids", "live_record_ids", "retracted_record_ids")
    for label, fresh in (("inc_vs_fresh_ordered", _fresh(ordered)), ("inc_vs_fresh_shuffled", _fresh(shuffled))):
        by_field: Dict[str, int] = defaultdict(int)
        digest_mismatch = 0
        for j in range(len(queries)):
            a, b = inc_res[j], fresh[j]
            if a.digest() != b.digest():
                digest_mismatch += 1
                for f in fields:
                    if getattr(a, f) != getattr(b, f):
                        by_field[f] += 1
        out[label] = {"digest_mismatches": digest_mismatch, "by_field": dict(by_field)}
    return out


def ek5_conformance_explanation_diffs(steps: int, sequences: int = 1000, seed: int = 20261124) -> Dict[str, Any]:
    """Python-only re-run of ``formal/run_conformance.py``'s inc_vs_fresh comparison.

    It uses the conformance generator's own ``run_sequence`` and ``compare_states``, and
    does not call the Lean CLI.  It returns per-field diff counts.
    """
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parent / "formal" / "run_conformance.py"
    spec = importlib.util.spec_from_file_location("w22_05_run_conformance", str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    by_field: Dict[str, int] = defaultdict(int)
    checkpoints = 0
    for i in range(sequences):
        cps, _st = mod.run_sequence(seed * 100003 + i, steps)
        for cp in cps:
            checkpoints += 1
            for d in mod.compare_states(cp["inc"], cp["fresh"]):
                by_field[d["field"]] += 1
    semantic = sum(c for f, c in by_field.items() if f in mod.SEMANTIC_FIELDS or f == "node_set")
    return {
        "steps": steps,
        "sequences": sequences,
        "seed": seed,
        "checkpoints": checkpoints,
        "by_field": dict(sorted(by_field.items())),
        "total": sum(by_field.values()),
        "semantic_total": semantic,
    }


# ---------------------------------------------------------------------------------------------------------------
# Conv 120 knowledge-update slice (descriptive; not a matched-input comparison)
# ---------------------------------------------------------------------------------------------------------------
def conv_ku_ledger_rows(conversations: Sequence[Any], scenarios: Sequence[Any]) -> List[Dict[str, Any]]:
    """Run the ordinary ledger on the 24 knowledge-update questions, using role-derived claims.

    - ``ku_initial`` and the six ``ku_echo`` turns assert the stale value.
    - ``ku_update`` withdraws them and asserts the gold reference.
    - Every record gets authority 1 (same speaker).

    Also runs the authority-only ablation (no WITHDRAW).  The query is as-of the update turn's timestamp for
    both valid and transaction time (turn timestamps are Unix seconds, so the arena's valid=10^6 would precede them).
    """
    rows = []
    for conv, spec in zip(conversations, scenarios):
        if conv.conversation_id != spec.conv_id:
            raise RuntimeError(f"scenario order mismatch {conv.conversation_id} vs {spec.conv_id}")
        q = next(x for x in conv.questions if x.category == "knowledge_update_echo_flood")
        by_text = {t.claim_text: t for t in conv.turns}
        initial = next(t for t in conv.turns if t.turn_id == q.evidence_turn_ids[0])
        update = next(t for t in conv.turns if t.turn_id == q.evidence_turn_ids[1])
        echoes = [by_text[txt] for txt in spec.ku_echoes_6]
        stale = normalize_value(spec.ku_stale_val)
        gold = normalize_value(spec.ku_gold)
        ekey = f"{conv.conversation_id}|ku"
        result: Dict[str, Any] = {"qid": q.qid}
        for label, withdraw in (("ledger", True), ("ledger_authority_only", False)):
            led = ChallengerBitemporalLedger(SQLiteChallengerBackend())
            for t in [initial] + echoes:
                led.append(LedgerRecordEnvelope("conv", t.turn_id, ekey, 1, int(t.timestamp), None, int(t.timestamp), "ASSERT", claim_value=stale))
            if withdraw:
                for t in [initial] + echoes:
                    led.append(LedgerRecordEnvelope("conv", t.turn_id, ekey, 1, int(t.timestamp), None, int(update.timestamp), "WITHDRAW"))
            led.append(LedgerRecordEnvelope("conv", update.turn_id, ekey, 1, int(update.timestamp), None, int(update.timestamp), "ASSERT", claim_value=gold))
            res = led.query_as_of("conv", ekey, int(update.timestamp), int(update.timestamp), mode="closure_table")
            led.close()
            result[f"{label}_status"] = res.status
            result[f"{label}_correct"] = ledger_set_verdict(res) == (gold,)
        rows.append(result)
    return rows


def conv_audit_per_item(audit_log: Sequence[Mapping[str, Any]], arm: str) -> Dict[str, bool]:
    return {r["item_id"]: bool(r["correct"]) for r in audit_log if r["arm"] == arm}
