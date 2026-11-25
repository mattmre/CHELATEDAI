"""Head-to-head drycheck for bundle 55 / item 6C.

Compares EK5 v3, the SQL challenger ledger, a recency baseline, and an
unsigned correlation filter. Synthetic rows stay on the schema-drycheck
cohort until a frozen wild snapshot clears 240 families and 60 per stratum.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

from chelated_pv7_spine.aa_null_calibration import mcnemar_exact
from chelated_pv7_spine.bridge_merge_harness import paired_bootstrap_ci
from chelated_pv7_spine.burn_ledger import (
    BurnedCorpusError,
    assert_corpus_unburned,
    text_fingerprint,
)
from chelated_pv7_spine.challenger_ledger import (
    QUERY_MODES,
    ChallengerBitemporalLedger,
    LedgerEdgeEnvelope,
    LedgerRecordEnvelope,
    Postgres17ChallengerBackend,
    SQLiteChallengerBackend,
)
from chelated_pv7_spine.claim_registry import REGISTRY_PATH, claim_registry_sha256
from chelated_pv7_spine.ek5_reason_canonicalizer import export_canonical_ek5_audit_trace
from chelated_pv7_spine.tra1_weight_sha_guard import reproduce_memt_weight_omission
from ek5_v3_stratified import EdgeAttestation, StratifiedEvent, StratifiedEvidenceStore

from chelated_wild_revision_streams import (
    COHORT_DRYCHECK,
    COHORT_WILD,
    STATUS_DRYCHECK,
    WILD_FAMILY_MINIMUM,
    WILD_PER_STRATUM,
    echo_decision,
    echo_metrics,
    family_oracles,
    infer_echo_roots,
    load_adjudicated_families,
    load_osv_seed_cache,
    oracle_at_cut,
    schema_echo_pairs,
    select_cohort,
    snapshot_directory_sha256,
    unsigned_correlation_collapse,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
BUNDLE_ID = 55
ITEM = "6C"
COST_LIMIT = 1.25
VALID_MODES = ("local-drycheck", "run-wild")
MIRROR_STRATA = frozenset({"unlinked_cross_mirror_echo", "equal_authority_vex_ghsa"})
_ASSERTING = frozenset({"ASSERT", "CONFLICT"})


def _mean(samples):
    if not samples:
        return 0.0
    return sum(samples) / len(samples)


def nearest_rank_p95(samples):
    """Nearest-rank 95th percentile. An empty sample is 0."""
    if not samples:
        return 0.0
    ordered = sorted(samples)
    index = (95 * len(ordered) + 99) // 100 - 1
    index = min(max(index, 0), len(ordered) - 1)
    return ordered[index]


def combined_cost(query_samples, withdrawal_samples):
    """Worse of as-of p95 and mean per-withdrawal maintenance."""
    return max(nearest_rank_p95(query_samples), _mean(withdrawal_samples))


def _slot(family):
    for record in family.records:
        if record.slot:
            return record.slot
    return "status"


def _public_family_view(family):
    """Strip ground-truth echo_of labels so every engine receives 100% identical public inputs."""
    return replace(
        family,
        records=tuple(replace(record, echo_of="") for record in family.records),
    )


def sql_envelopes(family):
    """Project one family into SQL record/edge ops using public fields only (never echo_of)."""
    public = _public_family_view(family)
    by_id = {record.record_id: record for record in public.records}
    envelopes = []
    for record in public.records:
        if record.op_kind in _ASSERTING:
            envelopes.append(
                (
                    "assert",
                    LedgerRecordEnvelope(
                        tenant_id=record.tenant_id,
                        record_id=record.record_id,
                        entity_key=record.entity_key,
                        authority_rank=record.authority_rank,
                        valid_from=record.valid_from,
                        valid_to=record.valid_to,
                        tx_time=record.tx_time,
                        op_kind=record.op_kind,
                        claim_value=record.claim_value,
                    ),
                )
            )
            for parent_id in record.explicit_parent_ids:
                parent = by_id.get(parent_id)
                if parent is not None and parent.tenant_id == record.tenant_id:
                    envelopes.append(
                        (
                            "edge",
                            LedgerEdgeEnvelope(
                                tenant_id=record.tenant_id,
                                edge_id=f"der:{record.record_id}:{parent_id}:{record.tx_time}",
                                src_id=record.record_id,
                                dst_id=parent_id,
                                tx_time=record.tx_time,
                            ),
                        )
                    )
            for target in record.supersedes_ids:
                envelopes.append(
                    (
                        "withdraw",
                        LedgerRecordEnvelope(
                            tenant_id=record.tenant_id,
                            record_id=target,
                            entity_key=record.entity_key,
                            authority_rank=record.authority_rank,
                            valid_from=record.valid_from,
                            valid_to=None,
                            tx_time=record.tx_time,
                            op_kind="WITHDRAW",
                            claim_value="",
                        ),
                    )
                )
        elif record.op_kind in {"WITHDRAW", "REINSTATE"}:
            envelopes.append(
                (
                    "withdraw" if record.op_kind == "WITHDRAW" else "assert",
                    LedgerRecordEnvelope(
                        tenant_id=record.tenant_id,
                        record_id=record.target_id or record.record_id,
                        entity_key=record.entity_key,
                        authority_rank=record.authority_rank,
                        valid_from=record.valid_from,
                        valid_to=record.valid_to,
                        tx_time=record.tx_time,
                        op_kind=record.op_kind,
                        claim_value="",
                    ),
                )
            )
    return envelopes


def sql_prediction(result):
    if result.status == "RESOLVED":
        return result.winner_claim_value or "retracted", tuple(result.live_record_ids)
    if result.status == "DISPUTED":
        return "disputed", tuple(result.contending_record_ids)
    return "retracted", ()


def _lineage_roots(family, admitted_ids):
    """Identify independent roots strictly from public text echo inference and explicit parents."""
    public = _public_family_view(family)
    roots = infer_echo_roots(public.records)
    linked = set()
    for record in public.records:
        if record.explicit_parent_ids:
            linked.add(record.record_id)
    chosen = []
    for event_id in admitted_ids:
        if event_id.startswith(("wd:", "ri:", "conflict-marker:")):
            continue
        if event_id in linked or roots.get(event_id, event_id) != event_id:
            continue
        chosen.append(event_id)
    return tuple(sorted(set(chosen)))


def project_ek5(family):
    """Project assertions, text-inferred same-tenant echo edges, explicit parents, and tx-gated withdrawals."""
    public = _public_family_view(family)
    slot = _slot(public)
    lineage_id = f"{public.tenant_id}:{public.entity_key}:{slot}"
    events = []
    copy_edges = []
    withdraw_edges = []
    by_id = {record.record_id: record for record in public.records}
    for record in public.records:
        if record.op_kind not in _ASSERTING:
            continue
        events.append(
            StratifiedEvent(
                event_id=record.record_id,
                entity=record.entity_key,
                slot=record.slot or slot,
                claim_value=record.claim_value,
                text=record.text,
                valid_time=record.valid_from,
                tx_time=record.tx_time,
                valid_end=record.valid_to,
                tenant=record.tenant_id,
                domain="general",
                authority_tier=record.authority_rank,
            )
        )
        if record.op_kind == "CONFLICT":
            marker = f"conflict-marker:{record.record_id}"
            events.append(
                StratifiedEvent(
                    event_id=marker,
                    entity=record.entity_key,
                    slot=record.slot or slot,
                    claim_value=f"conflict:{record.claim_value}",
                    text="",
                    valid_time=record.valid_from,
                    tx_time=record.tx_time,
                    valid_end=record.valid_to,
                    tenant=record.tenant_id,
                    domain="general",
                    authority_tier=record.authority_rank,
                )
            )
        for target in record.supersedes_ids:
            parent = by_id.get(target)
            if parent is None or (parent.tenant_id, parent.entity_key, parent.slot or slot) != (
                record.tenant_id,
                record.entity_key,
                record.slot or slot,
            ):
                continue
            withdraw_edges.append(
                EdgeAttestation(
                    attestation_id=f"lsup:{record.record_id}:{target}:{record.tx_time}",
                    src_id=record.record_id,
                    dst_id=target,
                    edge_type="VERSION_LINEAGE_SUPERSEDE",
                    valid_time=record.valid_from,
                    tx_time=record.tx_time,
                    lineage_id=lineage_id,
                )
            )
    roots = infer_echo_roots(public.records)
    seen_copies = set()
    for record in public.records:
        if record.op_kind not in _ASSERTING:
            continue
        parents = []
        root = roots.get(record.record_id, record.record_id)
        if root != record.record_id and by_id.get(root) and by_id[root].tenant_id == record.tenant_id:
            parents.append(root)
        parents.extend(parent for parent in record.explicit_parent_ids if parent not in parents)
        for parent_id in parents:
            parent = by_id.get(parent_id)
            if parent is None or parent.tenant_id != record.tenant_id:
                continue
            key = (record.record_id, parent_id)
            if key in seen_copies:
                continue
            seen_copies.add(key)
            copy_edges.append(
                EdgeAttestation(
                    attestation_id=f"copy:{record.record_id}:{parent_id}",
                    src_id=record.record_id,
                    dst_id=parent_id,
                    edge_type="COPY_OF",
                    valid_time=record.valid_from,
                    tx_time=record.tx_time,
                )
            )
    by_target = {}
    for record in public.records:
        if record.op_kind not in {"WITHDRAW", "REINSTATE"}:
            continue
        target = record.target_id or record.record_id
        by_target.setdefault(target, []).append(record)
    for target, ops in by_target.items():
        parent = by_id.get(target)
        if parent is None:
            continue
        open_withdraws = []
        for op in sorted(ops, key=lambda item: (item.tx_time, item.record_id)):
            if op.tenant_id != parent.tenant_id or op.authority_rank < parent.authority_rank:
                continue
            if op.op_kind == "WITHDRAW":
                sentinel = f"wd:{target}:{op.tx_time}:{op.record_id}"
                events.append(
                    StratifiedEvent(
                        event_id=sentinel,
                        entity=parent.entity_key,
                        slot=parent.slot or slot,
                        claim_value="",
                        text="",
                        valid_time=op.valid_from,
                        tx_time=op.tx_time,
                        tenant=parent.tenant_id,
                        domain="general",
                        authority_tier=op.authority_rank,
                    )
                )
                withdraw_edges.append(
                    EdgeAttestation(
                        attestation_id=f"lsup:{sentinel}:{target}",
                        src_id=sentinel,
                        dst_id=target,
                        edge_type="VERSION_LINEAGE_SUPERSEDE",
                        valid_time=op.valid_from,
                        tx_time=op.tx_time,
                        lineage_id=lineage_id,
                    )
                )
                open_withdraws.append((sentinel, op))
            else:
                events.append(
                    StratifiedEvent(
                        event_id=f"ri:{target}:{op.tx_time}:{op.record_id}",
                        entity=parent.entity_key,
                        slot=parent.slot or slot,
                        claim_value="",
                        text="",
                        valid_time=op.valid_from,
                        tx_time=op.tx_time,
                        tenant=parent.tenant_id,
                        domain="general",
                        authority_tier=op.authority_rank,
                    )
                )
                remaining_open = []
                for sentinel, earlier in open_withdraws:
                    if (earlier.tx_time, earlier.record_id) >= (op.tx_time, op.record_id):
                        remaining_open.append((sentinel, earlier))
                        continue
                    if op.authority_rank < earlier.authority_rank:
                        remaining_open.append((sentinel, earlier))
                        continue
                    withdraw_edges.append(
                        EdgeAttestation(
                            attestation_id=f"refute:{op.record_id}:{sentinel}",
                            src_id=f"ri:{target}:{op.tx_time}:{op.record_id}",
                            dst_id=sentinel,
                            edge_type="REFUTE",
                            valid_time=op.valid_from,
                            tx_time=op.tx_time,
                        )
                    )
                open_withdraws = remaining_open
    return events, copy_edges, withdraw_edges


def ek5_prediction(store, family, cut):
    snapshot = store.as_of(cut.valid_time, cut.tx_time)
    resolved = snapshot.resolve_slot(family.tenant_id, family.entity_key, _slot(family))
    status = resolved.get("status")
    if status == "ADMIT":
        disposition = resolved.get("claim_value") or "retracted"
    elif status == "CONTESTED":
        disposition = "disputed"
    else:
        disposition = "retracted"
    admitted = list(resolved.get("admitted_ids") or []) + list(resolved.get("contested_ids") or [])
    return disposition, _lineage_roots(family, admitted)


def unsigned_prediction(family, cut):
    """Collapse near-copies with no tenant boundary, then evaluate disposition."""
    public = _public_family_view(family)
    assertions = [record for record in public.records if record.op_kind in _ASSERTING and record.text]
    parent = {record.record_id: record.record_id for record in assertions}

    def find(record_id):
        while parent[record_id] != record_id:
            parent[record_id] = parent[parent[record_id]]
            record_id = parent[record_id]
        return record_id

    def unite(left, right):
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for index, left in enumerate(assertions):
        for right in assertions[index + 1 :]:
            if unsigned_correlation_collapse(left.text, right.text):
                unite(left.record_id, right.record_id)
    grouped = defaultdict(list)
    for record in assertions:
        grouped[find(record.record_id)].append(record)
    echo_of = {}
    for members in grouped.values():
        root = min(members, key=lambda record: (record.tx_time, record.record_id))
        for record in members:
            if record.record_id != root.record_id:
                echo_of[record.record_id] = root.record_id
    rewritten = []
    for record in public.records:
        if record.record_id in echo_of:
            rewritten.append(
                replace(
                    record,
                    tenant_id=public.tenant_id,
                    entity_key=public.entity_key,
                    echo_of=echo_of[record.record_id],
                )
            )
        else:
            rewritten.append(record)
    clean_cut = replace(cut, expected_disposition=None, expected_admitted_ids=None)
    oracle = oracle_at_cut(replace(public, records=tuple(rewritten)), clean_cut)
    return oracle.disposition, oracle.independent_lineage_ids


def recency_prediction(family, cut):
    """Latest visible assertion. Withdrawals, echoes, and conflicts are ignored."""
    public = _public_family_view(family)
    visible = [
        record
        for record in public.records
        if record.op_kind in _ASSERTING
        and record.tenant_id == public.tenant_id
        and record.entity_key == public.entity_key
        and record.tx_time <= cut.tx_time
        and record.valid_from <= cut.valid_time
        and (record.valid_to is None or cut.valid_time < record.valid_to)
    ]
    if not visible:
        return "retracted", ()
    latest = max(visible, key=lambda record: (record.tx_time, record.record_id))
    return latest.claim_value, (latest.record_id,)


def _hit(disposition, lineage, oracle):
    return int(disposition == oracle.disposition and tuple(lineage) == oracle.independent_lineage_ids)


def _pair_row(family, cut, oracle, disposition, lineage, elapsed):
    hit = _hit(disposition, lineage, oracle)
    return {
        "family_id": family.family_id,
        "stratum": family.stratum,
        "cut": cut.name,
        "hit": hit,
        "predicted_disposition": disposition,
        "predicted_lineage": list(lineage),
        "oracle_disposition": oracle.disposition,
        "oracle_lineage": list(oracle.independent_lineage_ids),
        "adjudication_source": oracle.adjudication_source,
        "elapsed_seconds": elapsed,
    }


def _accuracy(rows):
    if not rows:
        return 0.0
    return sum(row["hit"] for row in rows) / len(rows)


def _stratum_accuracy(rows, strata):
    chosen = [row for row in rows if row["stratum"] in strata]
    return _accuracy(chosen), len(chosen)


def build_sql_ledger(families, backend=None):
    ledger = ChallengerBitemporalLedger(backend or SQLiteChallengerBackend())
    withdrawal_samples = []
    for family in families:
        for kind, envelope in sql_envelopes(family):
            if kind == "withdraw":
                started = time.perf_counter()
                ledger.append(envelope)
                withdrawal_samples.append(time.perf_counter() - started)
            else:
                ledger.append(envelope)
    return ledger, withdrawal_samples


def score_sql(ledger, families, mode):
    rows = []
    for family in families:
        for cut, oracle in zip(family.cuts, family_oracles(family)):
            started = time.perf_counter()
            result = ledger.query_as_of(
                family.tenant_id,
                family.entity_key,
                cut.valid_time,
                cut.tx_time,
                mode=mode,
            )
            elapsed = time.perf_counter() - started
            disposition, lineage = sql_prediction(result)
            rows.append(_pair_row(family, cut, oracle, disposition, lineage, elapsed))
    return rows


def build_ek5_store(families):
    store = StratifiedEvidenceStore()
    events = []
    copy_edges = []
    withdraw_edges = []
    for family in families:
        family_events, family_copies, family_withdraws = project_ek5(family)
        events.extend(family_events)
        copy_edges.extend(family_copies)
        withdraw_edges.extend(family_withdraws)
    store.apply_mutation(events, copy_edges)
    withdrawal_samples = []
    for edge in withdraw_edges:
        started = time.perf_counter()
        store.apply_mutation((), [edge])
        withdrawal_samples.append(time.perf_counter() - started)
    return store, withdrawal_samples


def score_ek5(store, families):
    rows = []
    for family in families:
        for cut, oracle in zip(family.cuts, family_oracles(family)):
            started = time.perf_counter()
            disposition, lineage = ek5_prediction(store, family, cut)
            elapsed = time.perf_counter() - started
            rows.append(_pair_row(family, cut, oracle, disposition, lineage, elapsed))
    return rows


def score_callable(families, predictor):
    rows = []
    for family in families:
        for cut, oracle in zip(family.cuts, family_oracles(family)):
            started = time.perf_counter()
            disposition, lineage = predictor(family, cut)
            elapsed = time.perf_counter() - started
            rows.append(_pair_row(family, cut, oracle, disposition, lineage, elapsed))
    return rows


def _engine_block(name, rows, withdrawal_samples, extra=None):
    block = {
        "name": name,
        "accuracy": _accuracy(rows),
        "hits": sum(row["hit"] for row in rows),
        "trials": len(rows),
        "as_of_p95_seconds": nearest_rank_p95([row["elapsed_seconds"] for row in rows]),
        "per_withdrawal_mean_seconds": _mean(withdrawal_samples),
        "combined_cost_seconds": combined_cost([row["elapsed_seconds"] for row in rows], withdrawal_samples),
        "rows": rows,
    }
    if extra:
        block.update(extra)
    return block


def _dev_best(mode_blocks):
    return max(mode_blocks, key=lambda block: (block["accuracy"], -block["combined_cost_seconds"]))


def retirement_decision(ek5_block, sql_block):
    """1.25x cost gate plus family-cluster paired bootstrap (2,000 resamples) and cluster-level exact sign/McNemar test."""
    ek5_by_family: dict[str, list[int]] = {}
    sql_by_family: dict[str, list[int]] = {}
    for ek5, sql in zip(ek5_block["rows"], sql_block["rows"]):
        fid = ek5["family_id"]
        ek5_by_family.setdefault(fid, []).append(int(ek5["hit"]))
        sql_by_family.setdefault(fid, []).append(int(sql["hit"]))
    family_ids = list(ek5_by_family.keys())
    family_deltas: list[float] = []
    family_pairs: list[tuple[int, int]] = []
    for fid in family_ids:
        ek5_acc = _mean(ek5_by_family[fid])
        sql_acc = _mean(sql_by_family[fid])
        family_deltas.append(float(ek5_acc - sql_acc))
        if ek5_acc > sql_acc:
            family_pairs.append((1, 0))
        elif ek5_acc < sql_acc:
            family_pairs.append((0, 1))
        else:
            family_pairs.append((1, 1))
    mean, lower, upper = paired_bootstrap_ci(family_deltas, n_resamples=2000, alpha=0.05, seed=1601)
    mirror_ek5, mirror_n = _stratum_accuracy(ek5_block["rows"], MIRROR_STRATA)
    mirror_sql, mirror_sql_n = _stratum_accuracy(sql_block["rows"], MIRROR_STRATA)
    ek5_cost = ek5_block["combined_cost_seconds"]
    sql_cost = sql_block["combined_cost_seconds"]
    if ek5_cost <= 0.0 and sql_cost <= 0.0:
        ratio = 1.0
    elif ek5_cost <= 0.0:
        ratio = float("inf")
    else:
        ratio = sql_cost / ek5_cost
    accuracy_superior = lower > 0.0
    sql_within_cost = ratio <= COST_LIMIT
    noninferior = lower >= 0.0
    mirror_superior = mirror_n > 0 and mirror_ek5 > mirror_sql
    ek5_cheaper = ek5_cost > 0.0 and ek5_cost * COST_LIMIT <= sql_cost
    conjunction = noninferior and (mirror_superior or ek5_cheaper)
    retire = (not accuracy_superior and sql_within_cost) or (not conjunction)
    cluster_wins = sum(1 for ek5_w, sql_w in family_pairs if ek5_w > sql_w)
    cluster_losses = sum(1 for ek5_w, sql_w in family_pairs if ek5_w < sql_w)
    cluster_ties = sum(1 for ek5_w, sql_w in family_pairs if ek5_w == sql_w)
    return {
        "cost_limit": COST_LIMIT,
        "sampling_unit": "family_id",
        "family_cluster_count": len(family_ids),
        "bootstrap_resamples": 2000,
        "paired_deltas_ek5_minus_sql": family_deltas,
        "family_deltas_ek5_minus_sql": family_deltas,
        "bootstrap_mean": mean,
        "bootstrap_lcb": lower,
        "bootstrap_ucb": upper,
        "mcnemar_p": mcnemar_exact(family_pairs),
        "wins": cluster_wins,
        "losses": cluster_losses,
        "ties": cluster_ties,
        "cluster_wins": cluster_wins,
        "cluster_losses": cluster_losses,
        "cluster_ties": cluster_ties,
        "ek5_accuracy": ek5_block["accuracy"],
        "sql_accuracy": sql_block["accuracy"],
        "sql_mode": sql_block["name"],
        "sql_over_ek5_cost": ratio,
        "mirror_or_disputed_trials": mirror_n,
        "mirror_or_disputed_sql_trials": mirror_sql_n,
        "mirror_or_disputed_ek5_accuracy": mirror_ek5,
        "mirror_or_disputed_sql_accuracy": mirror_sql,
        "mirror_or_disputed_ek5_superior": mirror_superior,
        "ek5_cost_at_least_1_25x_lower": ek5_cheaper,
        "accuracy_superior_lcb_positive": accuracy_superior,
        "sql_within_1_25x_cost": sql_within_cost,
        "noninferior_accuracy": noninferior,
        "preregistered_conjunction_met": conjunction,
        "retire_ek5_proprietary_superiority": retire,
    }


def _kernel_receipt():
    kernel_root = str(REPO_ROOT / "packages" / "chelated_kernel")
    if kernel_root not in sys.path:
        sys.path.insert(0, kernel_root)
    try:
        from chelated_kernel import EventLedger
    except Exception as exc:
        return {
            "status": "skipped",
            "error": f"{type(exc).__name__}: {exc}",
            "llm_write_tokens": 0,
            "ingested": 0,
        }
    ledger = EventLedger(policy="sec_audit", tenant_id="bundle55", disclosure="non_nda")
    ledger.ingest(
        {
            "id": "OSV-DEMO-55",
            "affected": [{"package": {"name": "widget"}}],
            "status": "affected",
        }
    )
    ledger.ingest(
        {
            "document": {"title": "schema drycheck vex"},
            "vulnerabilities": [
                {"id": "OSV-DEMO-55", "product_status": {"known_not_affected": ["widget"]}}
            ],
        }
    )
    ledger.ingest(
        {
            "specversion": "1.0",
            "type": "com.chelated.sec.8k",
            "id": "sec-402-demo",
            "source": "edgar",
            "data": {"item": "4.02", "supersedes": "sec-10k-demo"},
        }
    )
    return {
        "status": "ingested",
        "error": "",
        "llm_write_tokens": ledger.llm_write_tokens,
        "ingested": len(ledger.receipts),
    }


def _burn_receipt(families, *, cohort_source=COHORT_DRYCHECK, historical_fingerprints=None):
    texts = [record.text for family in families for record in family.records if record.text]
    queries = [f"{family.family_id}:{cut.name}" for family in families for cut in family.cuts] or [
        "schema-drycheck-family query token"
    ]
    rejected = False
    try:
        assert_corpus_unburned("osv_cve_seeds")
    except BurnedCorpusError:
        rejected = True
    if historical_fingerprints is not None:
        assert_corpus_unburned(
            "wild-revision-cohort",
            doc_texts=texts or ["empty-cohort-placeholder"],
            query_texts=queries,
            historical_fingerprints=historical_fingerprints,
        )
        return {
            "osv_cve_seeds_rejected": rejected,
            "scored_cohort_texts_checked": len(texts),
            "drycheck_texts_admitted": True,
            "burn_inventory_scope": "supplied_historical_inventory",
            "wild_inventory_verified": True,
        }
    if cohort_source == COHORT_WILD:
        return {
            "osv_cve_seeds_rejected": rejected,
            "scored_cohort_texts_checked": len(texts),
            "drycheck_texts_admitted": False,
            "burn_inventory_scope": "missing_historical_inventory",
            "wild_inventory_verified": False,
            "error": "historical_fingerprints inventory is required to verify a live wild cohort",
        }
    historical = {
        "documents": [text_fingerprint("historical document marker for schema drycheck burn index")],
        "queries": [text_fingerprint("historical query marker for schema drycheck burn index")],
    }
    assert_corpus_unburned(
        "schema-drycheck-family",
        doc_texts=texts,
        query_texts=queries,
        historical_fingerprints=historical,
    )
    return {
        "osv_cve_seeds_rejected": rejected,
        "scored_cohort_texts_checked": len(texts),
        "drycheck_texts_admitted": True,
        "burn_inventory_scope": "local_drycheck_smoke",
        "wild_inventory_verified": False,
    }


def _echo_block(families):
    pairs = schema_echo_pairs(families)
    aware = echo_metrics(
        pairs,
        predicate=lambda pair: echo_decision(pair.left_text, pair.right_text, pair.left_tenant, pair.right_tenant),
    )
    unsigned = echo_metrics(
        pairs,
        predicate=lambda pair: unsigned_correlation_collapse(pair.left_text, pair.right_text),
    )
    return {"tenant_aware": aware, "unsigned_correlation": unsigned}


def evaluate_sql_family(families, backend):
    """Append once, then score both SQL modes on that ledger."""
    ledger, withdrawal_samples = build_sql_ledger(families, backend)
    try:
        blocks = [_engine_block(mode, score_sql(ledger, families, mode), withdrawal_samples) for mode in QUERY_MODES]
    finally:
        ledger.close()
    return blocks


def _postgres_modes(families, dsn, connect):
    if not dsn:
        return {"status": "not_requested", "error": "", "modes": []}
    try:
        backend = Postgres17ChallengerBackend(dsn, connect=connect)
    except Exception as exc:
        return {"status": "failed", "error": f"{type(exc).__name__}: {exc}", "modes": []}
    try:
        blocks = evaluate_sql_family(families, backend)
    except Exception as exc:
        return {
            "status": "failed",
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
            "modes": [],
        }
    return {"status": "ok", "error": "", "modes": blocks}


def run_wild_revision_comparison(
    *,
    mode="local-drycheck",
    wild_snapshot_dir=None,
    pg17_dsn=None,
    pg_connect=None,
    seed_cache=None,
    historical_fingerprints=None,
):
    """Run the item 6C comparison. Synthetic output is never labeled wild."""
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    if mode not in VALID_MODES:
        raise ValueError(f"mode must be one of {VALID_MODES}, got {mode!r}")
    registry_before = REGISTRY_PATH.read_bytes()
    snapshot_dir = Path(wild_snapshot_dir) if wild_snapshot_dir else None
    snapshot_sha256 = snapshot_directory_sha256(snapshot_dir)
    wild_families = load_adjudicated_families(snapshot_dir) if snapshot_dir else ()
    cohort = select_cohort(wild_families)
    families = tuple(cohort["families"])
    seed_path = Path(seed_cache) if seed_cache else REPO_ROOT / "cache" / "osv_cve_seeds.json"
    seed_advisories = load_osv_seed_cache(seed_path)
    sqlite_modes = evaluate_sql_family(families, SQLiteChallengerBackend())
    sqlite_best = _dev_best(sqlite_modes)
    postgres = _postgres_modes(families, pg17_dsn, pg_connect)
    if postgres["status"] == "ok" and postgres["modes"]:
        comparison_sql = _dev_best(postgres["modes"])
        comparison_engine = "postgres17"
    else:
        comparison_sql = sqlite_best
        comparison_engine = "sqlite"
    store, ek5_withdrawals = build_ek5_store(families)
    ek5_rows = score_ek5(store, families)
    trace = export_canonical_ek5_audit_trace(store)
    ek5_block = _engine_block("ek5_v3", ek5_rows, ek5_withdrawals, {"audit_digest": trace.digest()})
    recency_block = _engine_block("temporal_recency_rag", score_callable(families, recency_prediction), [])
    echo = _echo_block(families)
    unsigned_block = _engine_block(
        "unsigned_correlation_filter",
        score_callable(families, unsigned_prediction),
        [],
        {"cross_tenant_collapse_leaks": echo["unsigned_correlation"]["cross_tenant_collapse_leaks"]},
    )
    retirement = retirement_decision(ek5_block, comparison_sql)
    retirement["comparison_engine"] = comparison_engine
    kernel = _kernel_receipt()
    burn = _burn_receipt(
        families,
        cohort_source=cohort["cohort_source"],
        historical_fingerprints=historical_fingerprints,
    )
    weight_guard = reproduce_memt_weight_omission(
        config_sha256="a" * 64,
        tokenizer_sha256="b" * 64,
        shape=(4, 4),
        base_weights_a=b"base-revision-a",
        base_weights_b=b"base-revision-b",
        adapter_weights_a=b"adapter-revision-a",
        adapter_weights_b=b"adapter-revision-b",
        kv_or_state_dtype="float16",
        rope_freq_base=10000.0,
        rope_scaling_hash="c" * 64,
    )
    registry_after = REGISTRY_PATH.read_bytes()
    per_stratum = {}
    for stratum in ("unlinked_cross_mirror_echo", "equal_authority_vex_ghsa", "sec_item_402_nonreliance", "routine_amendment_control"):
        per_stratum[stratum] = sum(1 for family in families if family.stratum == stratum)
    status = cohort["status"]
    if mode == "local-drycheck" and cohort["cohort_source"] != COHORT_WILD:
        status = STATUS_DRYCHECK
    cohort_source = cohort["cohort_source"] if status != STATUS_DRYCHECK or cohort["cohort_source"] == COHORT_WILD else COHORT_DRYCHECK
    is_wild_evaluated = (
        status == "EVALUATED"
        and cohort_source == COHORT_WILD
        and bool(cohort["census"]["meets_wild_gate"])
        and bool(burn.get("wild_inventory_verified", False))
        and not any(family.synthetic for family in families)
    )
    superiority_is_wild = bool(
        is_wild_evaluated
        and not retirement["retire_ek5_proprietary_superiority"]
        and retirement["preregistered_conjunction_met"]
    )
    retirement["superiority_is_wild_evidence"] = superiority_is_wild
    live_requirements_met = is_wild_evaluated if mode == "run-wild" else True
    report = {
        "bundle_id": BUNDLE_ID,
        "item": ITEM,
        "mode": mode,
        "status": status,
        "passed": bool(live_requirements_met and registry_before == registry_after),
        "cohort_source": cohort_source,
        "snapshot_sha256": snapshot_sha256,
        "family_count": len(families),
        "families_per_stratum": per_stratum,
        "wild_adjudicated_count": cohort["census"]["total"],
        "duplicate_family_ids_dropped": cohort["census"].get("duplicate_family_ids_dropped", 0),
        "wild_gate": {"families": WILD_FAMILY_MINIMUM, "per_stratum": WILD_PER_STRATUM},
        "wild_gate_met": cohort["census"]["meets_wild_gate"],
        "synthetic_labeled_wild": cohort["synthetic_labeled_wild"],
        "descriptive_family_count": len(cohort.get("descriptive_families") or ()),
        "scored_families": [
            {
                "family_id": family.family_id,
                "stratum": family.stratum,
                "synthetic": family.synthetic,
                "cohort_source": COHORT_DRYCHECK if family.synthetic else COHORT_WILD,
            }
            for family in families
        ],
        "parsed_seed_advisories": len(seed_advisories),
        "llm_write_tokens": 0,
        "authored_prompts": 0,
        "echo": echo,
        "engines": {
            "ek5_v3": ek5_block,
            "sqlite": {block["name"]: block for block in sqlite_modes},
            "sqlite_dev_best": sqlite_best["name"],
            "postgres17": {
                "status": postgres["status"],
                "error": postgres["error"],
                "modes": {block["name"]: block for block in postgres["modes"]},
            },
            "temporal_recency_rag": recency_block,
            "unsigned_correlation_filter": unsigned_block,
        },
        "retirement": retirement,
        "ek5_audit_digest": trace.digest(),
        "kernel": kernel,
        "burn": burn,
        "tra1_silent_state_reuses": weight_guard["silent_state_reuses"],
        "registry_mutated": registry_before != registry_after,
        "registry_sha256": claim_registry_sha256(),
        "superiority_is_wild_evidence": superiority_is_wild,
        "confirmatory_n": is_wild_evaluated,
    }
    if report["cohort_source"] == COHORT_WILD and any(family.synthetic for family in families):
        raise RuntimeError("synthetic family entered the frozen wild cohort")
    if report["synthetic_labeled_wild"] != 0:
        raise RuntimeError("synthetic rows were labeled wild")
    return report


def _json_ready(value):
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, float) and value == float("inf"):
        return "inf"
    return value


def main(argv=None):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    parser = argparse.ArgumentParser(description="Bundle 55 item 6C wild-revision comparison")
    parser.add_argument("--mode", default="local-drycheck", choices=VALID_MODES)
    parser.add_argument("--wild-snapshot-dir", default=None)
    parser.add_argument("--pg17-dsn", default=None)
    parser.add_argument("--historical-fingerprints", default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args(argv)
    hist_fp = None
    if args.historical_fingerprints:
        hist_fp = json.loads(Path(args.historical_fingerprints).read_text(encoding="utf-8"))
    report = run_wild_revision_comparison(
        mode=args.mode,
        wild_snapshot_dir=args.wild_snapshot_dir,
        pg17_dsn=args.pg17_dsn,
        historical_fingerprints=hist_fp,
    )
    payload = json.dumps(_json_ready(report), indent=2, sort_keys=True)
    if args.output:
        Path(args.output).write_text(payload + "\n", encoding="utf-8")
    print(payload)
    if args.mode == "run-wild" and not report["passed"]:
        return 2
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
