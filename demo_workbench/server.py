"""Workable Live Solution Demonstration Backend (`demo_workbench/server.py`).

Provides a zero-dependency-bloat stdlib HTTP live execution server backing
`demo_workbench/index.html` and investor architectural evaluations:
  1. `POST /api/v1/epistemic/evaluate` — Live `StratifiedEvidenceStore` (`EK5 v3` 3-valued WFS)
     mutation, bitemporal `as_of` time-travel, `CITES_AUDIT` vs `D_V` provenance trace, and
     `Delta=0s` synchronous retraction comparison against unmasked search.
  2. `POST /api/v1/migration/evaluate` — Live synthetic NumPy `Mode B` (`d_high -> d_low`)
     `ZeroMu` 4-byte norm + 4-byte `CSLS` rescoring & legacy-shard certified shortlist stop check
     (labeled explicitly as a synthetic NumPy simulator alongside physical Qdrant receipts).
  3. `POST /api/v1/tco/model` — Council K6 / Wave-8 M4 parameterized 100M-event COGS identity
     with explicit `MEASURED_RECEIPT` vs `SENSITIVITY_ASSUMPTION` tags, equal-offset base infra,
     separated 500-anchor fit vs 20% backfill, and compact OpenSearch default (`2,178.6` toks).
  4. `GET /api/v1/registry/status` & `GET /api/v1/wave6/summary` — Live on-disk SHA-256
     re-verification of `docs/research/verified_registry.json` and `artifacts/wave6_*` bundles.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
from urllib.parse import parse_qs, urlparse

import numpy as np

WORKBENCH_DIR = Path(__file__).resolve().parent
REPO_ROOT = WORKBENCH_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from demo_workbench.compliance_copilot_engine import (  # noqa: E402
    CopilotService,
    handle_copilot_get,
    handle_copilot_post,
    is_copilot_path,
    set_service,
)
from ek5_v3_stratified import (  # noqa: E402
    EdgeAttestation,
    StratifiedEvent,
    StratifiedEvidenceStore,
    compute_alternating_wfs_oracle,
)
from scripts.night1_common import FastBM25Index  # noqa: E402


def _sha256_bytes(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def evaluate_live_epistemic_scenario(payload: Mapping[str, Any]) -> Dict[str, Any]:
    """Execute live EK5 v3 3-valued WFS + textbook alternating-fixpoint oracle cross-check."""
    t0 = time.perf_counter()
    raw_events = list(payload.get("events") or [])
    raw_edges = list(payload.get("edges") or [])
    query_text = str(payload.get("query") or "").strip()
    as_of_tx: Optional[int] = payload.get("as_of_tx")
    if as_of_tx is not None:
        as_of_tx = int(as_of_tx)

    events: List[StratifiedEvent] = []
    for idx, ev in enumerate(raw_events):
        events.append(
            StratifiedEvent(
                event_id=str(ev.get("event_id") or f"ev_{idx}"),
                text=str(ev.get("text") or ""),
                tenant=str(ev.get("tenant") or "enterprise_t0"),
                entity=str(ev.get("entity") or "default_entity"),
                slot=str(ev.get("slot") or "default_slot"),
                claim_value=str(ev.get("claim_value") or ""),
                authority_tier=int(ev.get("authority_tier", 2)),
                valid_time=int(ev.get("valid_time", idx + 1)),
                tx_time=int(ev.get("tx_time", idx + 1)),
            )
        )

    edges: List[EdgeAttestation] = []
    for idx, ed in enumerate(raw_edges):
        edges.append(
            EdgeAttestation(
                attestation_id=str(ed.get("attestation_id") or f"wb_att_{idx}"),
                src_id=str(ed["src_id"]),
                dst_id=str(ed["dst_id"]),
                edge_type=str(ed["edge_type"]),
                valid_time=int(ed.get("valid_time", idx + 1)),
                tx_time=int(ed.get("tx_time", idx + 1)),
            )
        )

    base_store = StratifiedEvidenceStore(strict_dag_supersede=False)
    base_store.apply_mutation(events, edges)
    if as_of_tx is not None:
        max_vt = max((e.valid_time for e in events), default=10**9)
        store = base_store.as_of(valid_cutoff=max_vt, tx_cutoff=as_of_tx)
        active_events = [e for e in events if e.event_id in store.events]
    else:
        store = base_store
        active_events = events
    kernel_ms = (time.perf_counter() - t0) * 1000.0

    active_nodes = [e.event_id for e in active_events]
    oracle_vals = compute_alternating_wfs_oracle(active_nodes, store.refs, store.deps_v)
    oracle_mismatches = 0
    node_states: List[Dict[str, Any]] = []

    for ev in active_events:
        eid = ev.event_id
        elig = store.eligibility.get(eid, "AUDIT_ONLY")
        wfs_val = store.val.get(eid, "F")
        orac = oracle_vals.get(eid, "F")
        if wfs_val != orac:
            oracle_mismatches += 1
        node_states.append({
            "event_id": eid,
            "entity": ev.entity,
            "slot": ev.slot,
            "claim_value": ev.claim_value,
            "base_authority": ev.authority_tier,
            "effective_authority": int(store.eff_auth.get(eid, ev.authority_tier)),
            "wfs_stratum2_val": wfs_val,
            "oracle_wfs_val": orac,
            "final_eligibility": elig,
            "valid_time": ev.valid_time,
            "tx_time": ev.tx_time,
            "text": ev.text,
        })

    unmasked_hits: List[str] = []
    chelated_hits: List[str] = []
    epistemic_decision = "ADMIT_OK"
    if active_events and query_text:
        bm = FastBM25Index([e.event_id for e in active_events], [e.text for e in active_events])
        scores = bm.score_query(query_text)
        raw_ord = np.argsort(-scores, kind="mergesort")[:5]
        unmasked_hits = [active_events[int(i)].event_id for i in raw_ord]
        admitted_indices = [
            int(i) for i in np.argsort(-scores, kind="mergesort")
            if store.eligibility.get(active_events[int(i)].event_id) == "ADMIT"
        ][:5]
        contested_high = [
            e.event_id for e in active_events if store.eligibility.get(e.event_id) == "CONTESTED"
        ]
        max_cont = max((store.eff_auth.get(cid, 0) for cid in contested_high), default=0)
        max_adm = max((store.eff_auth.get(active_events[i].event_id, 0) for i in admitted_indices), default=0)
        if contested_high and max_cont > max_adm:
            epistemic_decision = "CONTESTED_ABSTAIN"
            chelated_hits = []
        else:
            chelated_hits = [active_events[i].event_id for i in admitted_indices]

    receipt_payload = json.dumps(node_states, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {
        "epistemic_decision": epistemic_decision,
        "kernel_exec_ms": round(kernel_ms, 4),
        "as_of_tx": as_of_tx,
        "wfs_oracle_mismatches": oracle_mismatches,
        "unmasked_delta0s_topk_ids": unmasked_hits,
        "chelated_admitted_topk_ids": chelated_hits,
        "blocked_authority_edges": list(store.get_telemetry().blocked_authority_edges),
        "nodes": node_states,
        "epistemic_receipt_sha256": hashlib.sha256(receipt_payload).hexdigest(),
    }


def active_ids_set(ids: Sequence[str]) -> set[str]:
    return set(ids)


def evaluate_live_migration_stop(payload: Mapping[str, Any]) -> Dict[str, Any]:
    """Run synthetic NumPy Mode B (d_high -> d_low) ZeroMu + CSLS & legacy-shard exact-prefix stop check."""
    t0 = time.perf_counter()
    n_docs = min(int(payload.get("n_docs", 2000)), 20000)
    d_low = min(int(payload.get("d_low", 64)), 512)
    d_high = min(int(payload.get("d_high", 256)), 1024)
    backfill_frac = max(0.0, min(1.0, float(payload.get("backfill_fraction", 0.20))))
    k_os = max(10, min(256, int(payload.get("k_os", 64))))
    seed = int(payload.get("seed", 42))

    rng = np.random.default_rng(seed)
    latent = rng.normal(size=(n_docs + 24, 24))
    w1 = rng.normal(size=(24, d_low))
    w2 = rng.normal(size=(24, d_high))
    cone = np.zeros(d_high, dtype=np.float64)
    cone[0] = 0.80

    low_all = latent @ w1 + 0.04 * rng.normal(size=(n_docs + 24, d_low))
    low_all /= np.maximum(np.linalg.norm(low_all, axis=1, keepdims=True), 1e-12)
    high_all = latent @ w2 + cone + 0.04 * rng.normal(size=(n_docs + 24, d_high))
    high_all /= np.maximum(np.linalg.norm(high_all, axis=1, keepdims=True), 1e-12)

    docs_low = low_all[:n_docs].astype(np.float32)
    docs_high = high_all[:n_docs].astype(np.float32)
    dev_q_high = high_all[n_docs : n_docs + 16].astype(np.float32)
    eval_q_high = high_all[n_docs + 16 :].astype(np.float32)

    n_anchors = min(max(32, int(payload.get("n_anchors", 128))), n_docs)
    fit_idx = np.arange(n_anchors, dtype=np.int64)
    mu_src = docs_high[fit_idx].astype(np.float64).mean(axis=0)
    mu_tgt = docs_low[fit_idx].astype(np.float64).mean(axis=0)
    xc = docs_high[fit_idx].astype(np.float64) - mu_src
    xc /= np.maximum(np.linalg.norm(xc, axis=1, keepdims=True), 1e-12)
    yc = docs_low[fit_idx].astype(np.float64) - mu_tgt
    yc /= np.maximum(np.linalg.norm(yc, axis=1, keepdims=True), 1e-12)
    u, _s, vt = np.linalg.svd(xc.T @ yc, full_matrices=False)
    w_down = u @ vt
    ortho_err = float(np.max(np.abs(w_down.T @ w_down - np.eye(d_low))))

    n_bf = int(round(backfill_frac * n_docs))
    mask_new = np.zeros(n_docs, dtype=bool)
    mask_new[:n_bf] = True
    idx_old = np.flatnonzero(~mask_new)
    idx_new = np.flatnonzero(mask_new)

    q_c = eval_q_high.astype(np.float64) - mu_src
    q_c /= np.maximum(np.linalg.norm(q_c, axis=1, keepdims=True), 1e-12)
    q_hat_low = q_c @ w_down
    q_hat_low /= np.maximum(np.linalg.norm(q_hat_low, axis=1, keepdims=True), 1e-12)

    dev_c = dev_q_high.astype(np.float64) - mu_src
    dev_c /= np.maximum(np.linalg.norm(dev_c, axis=1, keepdims=True), 1e-12)
    dev_hat_low = dev_c @ w_down
    dev_hat_low /= np.maximum(np.linalg.norm(dev_hat_low, axis=1, keepdims=True), 1e-12)

    exact_csls = np.full((eval_q_high.shape[0], n_docs), -1e9, dtype=np.float64)
    shortlist_csls = np.full((eval_q_high.shape[0], n_docs), -1e9, dtype=np.float64)
    legacy_shard_certified_flags: List[bool] = []

    if idx_old.size > 0:
        sub_old = docs_low[idx_old].astype(np.float64)
        mu_old = sub_old.mean(axis=0)
        norms_old = np.maximum(np.linalg.norm(sub_old - mu_old, axis=1).astype(np.float32), np.float32(1e-6))
        n_min_old = float(np.min(norms_old))
        n_max_old = float(np.max(norms_old))
        dev_dots = dev_hat_low @ sub_old.T
        dev_zm = (dev_dots - (dev_hat_low @ mu_old)[:, None]) / norms_old[None, :]
        r_d_old = np.sort(dev_zm, axis=0)[-min(10, dev_zm.shape[0]):, :].mean(axis=0).astype(np.float32)
        r_d_min_old = float(np.min(r_d_old))

        raw_dots = q_hat_low @ sub_old.T
        zm_full = (raw_dots - (q_hat_low @ mu_old)[:, None]) / norms_old[None, :]
        rq_full = np.sort(zm_full, axis=1)[:, -min(10, zm_full.shape[1]):].mean(axis=1, keepdims=True)
        exact_csls[:, idx_old] = 2.0 * zm_full - r_d_old[None, :] - rq_full

        lim = min(k_os, idx_old.size)
        for qi in range(eval_q_high.shape[0]):
            ord_dot = np.argsort(-raw_dots[qi], kind="mergesort")[:lim]
            s_sub = zm_full[qi, ord_dot]
            rq_sub = float(np.mean(np.sort(s_sub)[-min(10, s_sub.size):]))
            csls_sub = 2.0 * s_sub - r_d_old[ord_dot] - rq_sub
            shortlist_csls[qi, idx_old[ord_dot]] = csls_sub
            if lim == idx_old.size:
                legacy_shard_certified_flags.append(True)
            else:
                last_dot = float(raw_dots[qi, ord_dot[-1]])
                shift = float(q_hat_low[qi] @ mu_old)
                num_ub = last_dot - shift
                zm_ub = (num_ub / n_min_old) if num_ub >= 0.0 else (num_ub / n_max_old)
                csls_ub = 2.0 * zm_ub - r_d_min_old - rq_sub
                kth_sc = float(np.sort(csls_sub)[-min(10, csls_sub.size)])
                legacy_shard_certified_flags.append(bool(csls_ub < kth_sc))

    if idx_new.size > 0:
        sub_new = docs_high[idx_new].astype(np.float64)
        dev_new = dev_q_high.astype(np.float64) @ sub_new.T
        r_d_new = np.sort(dev_new, axis=0)[-min(10, dev_new.shape[0]):, :].mean(axis=0).astype(np.float32)
        eval_new = eval_q_high.astype(np.float64) @ sub_new.T
        rq_new = np.sort(eval_new, axis=1)[:, -min(10, eval_new.shape[1]):].mean(axis=1, keepdims=True)
        exact_csls[:, idx_new] = 2.0 * eval_new - r_d_new[None, :] - rq_new
        lim_n = min(k_os, idx_new.size)
        for qi in range(eval_q_high.shape[0]):
            ord_n = np.argsort(-eval_new[qi], kind="mergesort")[:lim_n]
            s_n = eval_new[qi, ord_n]
            rq_sn = float(np.mean(np.sort(s_n)[-min(10, s_n.size):]))
            shortlist_csls[qi, idx_new[ord_n]] = 2.0 * s_n - r_d_new[ord_n] - rq_sn

    recalls = []
    for qi in range(eval_q_high.shape[0]):
        ex10 = set(np.argsort(-exact_csls[qi], kind="mergesort")[:10].tolist())
        sh10 = set(np.argsort(-shortlist_csls[qi], kind="mergesort")[:10].tolist())
        recalls.append(len(ex10 & sh10) / 10.0)

    return {
        "mode": "SYNTHETIC_NUMPY_MODE_B_SIMULATOR",
        "scope_note": (
            "Synthetic Gaussian+cone NumPy simulator verifying Stiefel orthogonality and exact-DOT-prefix "
            "shortlist certification on the unmigrated legacy shard. For physical Qdrant gRPC BEIR receipts, "
            "see dcb84026a15c (Track B) and cba30b931812 (Track E)."
        ),
        "n_docs": n_docs,
        "n_anchors": n_anchors,
        "anchor_fraction": round(n_anchors / float(n_docs), 5),
        "backfill_fraction": backfill_frac,
        "k_os": k_os,
        "egress_sidecar_bytes_per_doc": 8,
        "shared_centroid_bytes": int(d_low * 4),
        "stiefel_orthogonality_error": ortho_err,
        "shortlist_vs_exact_csls_recall_at_10": float(np.mean(recalls)),
        "legacy_shard_exact_prefix_certified_stop_rate": (
            float(np.mean(legacy_shard_certified_flags)) if legacy_shard_certified_flags else 1.0
        ),
        "exec_ms": round((time.perf_counter() - t0) * 1000.0, 3),
    }


def compute_parameterized_100m_tco(params: Mapping[str, Any]) -> Dict[str, Any]:
    """Council K6 & Wave-8 M4 Parameterized TCO Identity: explicit MEASURED vs ASSUMPTION provenance."""
    n_writes = float(params.get("n_writes", 100_000_000))
    n_reads = float(params.get("n_reads", 500_000_000))
    corpus_vectors = float(params.get("corpus_vectors", 10_000_000))
    retraction_events = float(params.get("retraction_events", 50_000))

    llm_in_usd_per_1m = float(params.get("llm_input_usd_per_1m_tokens", 0.60))
    llm_out_usd_per_1m = float(params.get("llm_output_usd_per_1m_tokens", 2.40))
    reader_out_tokens = float(params.get("shared_reader_output_tokens_per_query", 32.0))
    embed_usd_per_1m_docs = float(params.get("gpu_embed_4096_usd_per_1m_docs", 18.00))
    recurring_embed_usd_per_1m = float(params.get("recurring_embed_usd_per_1m_inputs", 0.02))
    base_ann_storage_annual_usd = float(params.get("base_ann_and_storage_annual_usd", 36_000.0))
    ram_usd_per_gib_month = float(params.get("ram_usd_per_gib_month", 6.50))
    ops_audit_usd_per_incident = float(params.get("compliance_incident_audit_usd", 25.0))
    include_equal_offset_infra = bool(params.get("include_equal_offset_infra", True))

    # MEASURED token coefficients from Wave-6 Track C (05442a849c19)
    chelated_conv_write_toks = float(params.get("chelated_conv_write_llm_tokens_per_event", 0.0))
    chelated_conv_read_toks = float(params.get("chelated_conv_read_prompt_tokens_per_query", 1756.4))
    chelated_prose_write_toks = float(params.get("chelated_prose_write_llm_tokens_per_doc", 404.86))
    chelated_prose_read_toks = float(params.get("chelated_prose_read_prompt_tokens_per_query", 170.5))
    mem0_measured_read_toks = float(params.get("mem0_infer_read_prompt_tokens_per_query", 506.65))
    mem0_unbatched_write_toks = float(params.get("mem0_infer_unbatched_write_llm_tokens_per_turn", 4546.4))
    hindsight_measured_read_toks = float(params.get("hindsight_reflect_prompt_tokens_per_query", 2086.7))
    osm_compact_read_toks = float(params.get("opensearch_memory_compact_read_tokens_per_query", 2178.6))

    # SENSITIVITY ASSUMPTION coefficients (explicitly tagged per Council M4; never labeled measured)
    mem0_batched_write_toks = float(params.get("mem0_infer_batched_write_assumption_tokens", 620.0))
    hindsight_batched_write_toks = float(params.get("hindsight_retain_batched_write_assumption_tokens", 740.0))
    osm_batched_write_toks = float(params.get("opensearch_memory_batched_write_assumption_tokens", 580.0))
    osm_long_ctx_read_toks = float(params.get("opensearch_memory_long_context_assumption_tokens", 13500.0))
    mode_b_anchor_count = float(params.get("mode_b_anchor_count", 500.0))
    mode_b_backfill_frac = float(params.get("mode_b_optional_backfill_fraction", 0.20))

    recurring_embed_usd = ((n_writes + n_reads) / 1_000_000.0) * recurring_embed_usd_per_1m
    reader_output_usd = ((n_reads * reader_out_tokens) / 1_000_000.0) * llm_out_usd_per_1m
    equal_offset_base_usd = (
        (recurring_embed_usd + base_ann_storage_annual_usd + reader_output_usd)
        if include_equal_offset_infra
        else 0.0
    )

    def _cost_breakdown(
        *,
        write_toks: float,
        read_toks: float,
        anchor_docs_reencoded: float,
        corpus_backfill_frac: float,
        sidecar_bytes_per_vec: float,
        shared_centroid_bytes: float,
        manual_retraction_review_frac: float,
    ) -> Dict[str, float]:
        write_llm_usd = (n_writes * write_toks / 1_000_000.0) * llm_in_usd_per_1m
        read_llm_usd = (n_reads * read_toks / 1_000_000.0) * llm_in_usd_per_1m
        anchor_fit_usd = (anchor_docs_reencoded / 1_000_000.0) * embed_usd_per_1m_docs
        backfill_gpu_usd = (corpus_vectors * corpus_backfill_frac / 1_000_000.0) * embed_usd_per_1m_docs
        sidecar_ram_gib = (corpus_vectors * sidecar_bytes_per_vec + shared_centroid_bytes) / float(1024 ** 3)
        storage_ram_usd = sidecar_ram_gib * ram_usd_per_gib_month * 12.0
        retraction_ops_usd = retraction_events * manual_retraction_review_frac * ops_audit_usd_per_incident
        variable_diff_usd = (
            write_llm_usd + read_llm_usd + anchor_fit_usd + backfill_gpu_usd + storage_ram_usd + retraction_ops_usd
        )
        total_usd = variable_diff_usd + equal_offset_base_usd
        return {
            "write_llm_usd": round(write_llm_usd, 2),
            "read_llm_usd": round(read_llm_usd, 2),
            "anchor_fit_gpu_usd": round(anchor_fit_usd, 2),
            "encoder_migration_gpu_usd": round(anchor_fit_usd + backfill_gpu_usd, 2),
            "backfill_or_reindex_gpu_usd": round(backfill_gpu_usd, 2),
            "sidecar_or_extra_ram_annual_usd": round(storage_ram_usd, 2),
            "retraction_compliance_ops_usd": round(retraction_ops_usd, 2),
            "equal_offset_base_infra_usd": round(equal_offset_base_usd, 2),
            "variable_differentiation_subtotal_usd": round(variable_diff_usd, 2),
            "total_annualized_tco_usd": round(total_usd, 2),
        }

    systems = {
        "Chelated_RegimeI_Conv_ModeB_20PctBackfill": _cost_breakdown(
            write_toks=chelated_conv_write_toks,
            read_toks=chelated_conv_read_toks,
            anchor_docs_reencoded=mode_b_anchor_count,
            corpus_backfill_frac=mode_b_backfill_frac,
            sidecar_bytes_per_vec=8.0,
            shared_centroid_bytes=16384.0,
            manual_retraction_review_frac=0.02,
        ),
        "Chelated_RegimeII_RawProse_ModeB_20PctBackfill": _cost_breakdown(
            write_toks=chelated_prose_write_toks,
            read_toks=chelated_prose_read_toks,
            anchor_docs_reencoded=mode_b_anchor_count,
            corpus_backfill_frac=mode_b_backfill_frac,
            sidecar_bytes_per_vec=8.0,
            shared_centroid_bytes=16384.0,
            manual_retraction_review_frac=0.02,
        ),
        "Mem0_InferTrue_BatchedWriteAssumption_MeasuredRead": _cost_breakdown(
            write_toks=mem0_batched_write_toks,
            read_toks=mem0_measured_read_toks,
            anchor_docs_reencoded=0.0,
            corpus_backfill_frac=1.0,
            sidecar_bytes_per_vec=0.0,
            shared_centroid_bytes=0.0,
            manual_retraction_review_frac=0.15,
        ),
        "Mem0_InferTrue_UnbatchedSuiteBMeasured": _cost_breakdown(
            write_toks=mem0_unbatched_write_toks,
            read_toks=mem0_measured_read_toks,
            anchor_docs_reencoded=0.0,
            corpus_backfill_frac=1.0,
            sidecar_bytes_per_vec=0.0,
            shared_centroid_bytes=0.0,
            manual_retraction_review_frac=0.15,
        ),
        "Hindsight_4Net_Emulation_BatchedWriteAssumption": _cost_breakdown(
            write_toks=hindsight_batched_write_toks,
            read_toks=hindsight_measured_read_toks,
            anchor_docs_reencoded=0.0,
            corpus_backfill_frac=1.0,
            sidecar_bytes_per_vec=0.0,
            shared_centroid_bytes=0.0,
            manual_retraction_review_frac=0.15,
        ),
        "OpenSearch_Memory_Emulation_CompactReadDefault": _cost_breakdown(
            write_toks=osm_batched_write_toks,
            read_toks=osm_compact_read_toks,
            anchor_docs_reencoded=0.0,
            corpus_backfill_frac=1.0,
            sidecar_bytes_per_vec=0.0,
            shared_centroid_bytes=0.0,
            manual_retraction_review_frac=0.10,
        ),
        "OpenSearch_Memory_LongContext13500_Sensitivity": _cost_breakdown(
            write_toks=osm_batched_write_toks,
            read_toks=osm_long_ctx_read_toks,
            anchor_docs_reencoded=0.0,
            corpus_backfill_frac=1.0,
            sidecar_bytes_per_vec=0.0,
            shared_centroid_bytes=0.0,
            manual_retraction_review_frac=0.10,
        ),
    }
    return {
        "formula": (
            "total_annualized_tco_usd = write_llm_usd + read_llm_usd + anchor_fit_gpu_usd + "
            "backfill_or_reindex_gpu_usd + sidecar_or_extra_ram_annual_usd + "
            "retraction_compliance_ops_usd + equal_offset_base_infra_usd"
        ),
        "parameter_provenance": {
            "chelated_conv_write_llm_tokens_per_event": "MEASURED_RECEIPT (05442a849c19 Suite B: 0.0)",
            "chelated_conv_read_prompt_tokens_per_query": "MEASURED_RECEIPT (05442a849c19 Suite B: 1756.4)",
            "chelated_prose_write_llm_tokens_per_doc": "MEASURED_RECEIPT (05442a849c19 Suite A: 6882.67 / 17 = 404.86)",
            "chelated_prose_read_prompt_tokens_per_query": "MEASURED_RECEIPT (05442a849c19 Suite A: 170.5)",
            "mem0_infer_read_prompt_tokens_per_query": "MEASURED_RECEIPT (05442a849c19 Suite B: 506.65; Suite A: 371.35)",
            "mem0_infer_unbatched_write_llm_tokens_per_turn": "MEASURED_RECEIPT (05442a849c19 Suite B: 363715.6 / 80 = 4546.4)",
            "hindsight_reflect_prompt_tokens_per_query": "MEASURED_RECEIPT (05442a849c19 Suite B emulation: 2086.7)",
            "opensearch_memory_compact_read_tokens_per_query": "MEASURED_RECEIPT (05442a849c19 Suite B emulation: 2178.6)",
            "batched_write_tokens_620_740_580": "SENSITIVITY_ASSUMPTION (turn-batched write projection)",
            "opensearch_13500_read_tokens": "SENSITIVITY_ASSUMPTION (public long-context benchmark configuration)",
            "manual_retraction_review_shares_2pct_10pct_15pct": "SENSITIVITY_ASSUMPTION (compliance escalation review share)",
            "unit_cloud_prices_and_equal_offset_infra": "SENSITIVITY_ASSUMPTION (user-adjustable workload/cloud prices)",
        },
        "inputs_used": {
            "n_writes": n_writes,
            "n_reads": n_reads,
            "corpus_vectors": corpus_vectors,
            "retraction_events": retraction_events,
            "llm_input_usd_per_1m_tokens": llm_in_usd_per_1m,
            "gpu_embed_4096_usd_per_1m_docs": embed_usd_per_1m_docs,
            "equal_offset_base_infra_usd": round(equal_offset_base_usd, 2),
        },
        "systems": systems,
        "disclosure": (
            "Council K6 / Wave-8 M4 Sensitivity Model: under 5:1 read-heavy conversational workloads with turn-batched "
            "writes (ASSUMPTION 620 toks), Mem0 infer=True's compact 506.65-token extracted memories yield lower read COGS "
            "than Chelated Regime I ($463k vs $638k incl. base infra), whereas Chelated wins on unbatched rolling writes "
            "($638k vs $699k), write-heavy telemetry, and Regulated Raw Prose Regime II ($187k + +15.0 pp Suite A accuracy). "
            "No conversational accuracy superiority is claimed on Suite B (LCB97.5 <= 0)."
        ),
    }


def get_registry_status() -> Dict[str, Any]:
    reg_path = REPO_ROOT / "docs" / "research" / "verified_registry.json"
    reg_info: Dict[str, Any] = {"registry_present": reg_path.is_file()}
    if reg_path.is_file():
        data = json.loads(reg_path.read_text(encoding="utf-8"))
        payload_copy = {k: v for k, v in data.items() if k != "registry_content_sha256"}
        recomputed = hashlib.sha256(
            json.dumps(payload_copy, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        reg_info["schema_version"] = data.get("schema_version")
        reg_info["n_bundles"] = len(data.get("bundles") or [])
        reg_info["registry_content_sha256"] = data.get("registry_content_sha256")
        reg_info["registry_sha256_verified_live"] = bool(recomputed == data.get("registry_content_sha256"))
    wave6_bundles = []
    art_dir = REPO_ROOT / "artifacts"
    if art_dir.is_dir():
        for sub in sorted(art_dir.glob("wave6_*")):
            rec = sub / "report_receipt.json"
            if rec.is_file():
                rdata = json.loads(rec.read_text(encoding="utf-8"))
                mname = str(rdata.get("manifest_file") or rdata.get("manifest") or "")
                mpath = sub / mname
                expected_sha = str(rdata.get("manifest_sha256") or "")
                actual_sha = _sha256_bytes(mpath) if mpath.is_file() else ""
                wave6_bundles.append({
                    "bundle": sub.name,
                    "manifest": mname,
                    "manifest_sha256": expected_sha,
                    "sha256_verified_on_disk": bool(actual_sha and actual_sha == expected_sha),
                })
    reg_info["wave6_bundles"] = wave6_bundles
    return reg_info


def get_wave6_summary() -> Dict[str, Any]:
    """Load live Wave-6 Part VI manifest summaries with live on-disk SHA-256 verification."""
    expected = {
        "wave6_pm1b_v4_reranked_sota": "wave6_pm1b_v4_manifest.json",
        "wave6_d2_4096d_3beir_complete": "wave6_d2_3beir_manifest.json",
        "wave6_pm3_v2_competitor_arena": "wave6_pm3_v2_arena_manifest.json",
        "wave6_d4_trained_epistemic_grpo": "wave6_d4_epistemic_grpo_manifest.json",
        "wave6_d5_grpc_interceptor_gateway": "wave6_d5_grpc_gateway_manifest.json",
    }
    art_dir = REPO_ROOT / "artifacts"
    tracks: Dict[str, Any] = {}
    for bname, default_manifest in expected.items():
        bdir = art_dir / bname
        rec = bdir / "report_receipt.json"
        if not rec.is_file():
            tracks[bname] = {"present": False, "status": "PENDING_OR_RUNNING"}
            continue
        rdata = json.loads(rec.read_text(encoding="utf-8"))
        mname = str(rdata.get("manifest_file") or rdata.get("manifest") or default_manifest)
        mpath = bdir / mname
        expected_sha = str(rdata.get("manifest_sha256") or "")
        actual_sha = _sha256_bytes(mpath) if mpath.is_file() else ""
        side_hashes = rdata.get("artifact_hashes") or rdata.get("artifact_sha256") or {}
        side_ok = all(
            (bdir / fname).is_file() and _sha256_bytes(bdir / fname) == str(fsha)
            for fname, fsha in side_hashes.items()
        )
        mdata = json.loads(mpath.read_text(encoding="utf-8")) if mpath.is_file() else {}
        summary_dict: Dict[str, Any] = {
            k: mdata[k]
            for k in (
                "primary_endpoint",
                "primary_confirmatory_virgin_holdout",
                "suite_a_enterprise_n60",
                "suite_b_conversational_n40",
                "system_provenance_labels",
                "heldout_wfs_eval_n120",
                "d1_zero_shot_enterprise_prose_transfer_n60",
                "headline_metrics",
                "flagship_10k_per_arm_open_loop_lan_tail_proof",
                "ek5_v3_retraction_and_epoch_pinned_fence",
            )
            if k in mdata
        }
        if bname == "wave6_pm3_v2_competitor_arena":
            summary_dict["metric_naming_note"] = (
                "In suite_a_enterprise_n60, stale_resurrection_delta0s_immediate (0.7833 on Chelated) is a raw "
                "substring scan over formatted_context including [SUPERSEDED] audit provenance notes, not reader "
                "prediction stale leak (actual Suite A reader accuracy = 96.67%) nor Demo D5 bitset resurrection (0.00%)."
            )
        tracks[bname] = {
            "present": True,
            "status": "SHA256_VERIFIED" if (actual_sha == expected_sha and side_ok) else "HASH_MISMATCH",
            "manifest": mname,
            "manifest_sha256": expected_sha,
            "actual_manifest_sha256": actual_sha,
            "side_artifacts_verified": side_ok,
            "wall_seconds": mdata.get("wall_seconds", mdata.get("pipeline_wall_seconds")),
            "summary": summary_dict,
        }
    return {
        "tracks_completed": sum(1 for v in tracks.values() if v.get("status") == "SHA256_VERIFIED"),
        "tracks_total": len(expected),
        "tracks": tracks,
    }


def dispatch_api_request(
    method: str,
    path: str,
    body_bytes: bytes = b"",
    *,
    headers: Optional[Mapping[str, Any]] = None,
    public_only: bool = False,
) -> Tuple[int, str, bytes]:
    """Pure-function HTTP router testable directly in pytest without opening sockets."""
    parsed = urlparse(path)
    clean_path = parsed.path
    query_params = parse_qs(parsed.query)

    if is_copilot_path(clean_path):
        if method == "GET":
            return handle_copilot_get(clean_path, query_params, headers)
        if method == "POST":
            return handle_copilot_post(clean_path, query_params, headers, body_bytes)
        return 405, "application/json", json.dumps({"error": "method_not_allowed", "message": "Method not allowed."}).encode("utf-8")

    if public_only:
        return 404, "application/json", json.dumps({"error": "not_found", "message": "Not found."}).encode("utf-8")

    if method == "GET" and clean_path in ("/", "/index.html"):
        idx_file = WORKBENCH_DIR / "index.html"
        return 200, "text/html; charset=utf-8", idx_file.read_bytes()
    if method == "GET" and clean_path == "/pm4_human_grader.html":
        gr_file = WORKBENCH_DIR / "pm4_human_grader.html"
        return 200, "text/html; charset=utf-8", gr_file.read_bytes()
    if method == "GET" and clean_path == "/api/v1/health":
        return 200, "application/json", json.dumps({"status": "ok", "service": "chelated-workbench-v2"}).encode("utf-8")
    if method == "GET" and clean_path == "/api/v1/registry/status":
        return 200, "application/json", json.dumps(get_registry_status(), indent=2).encode("utf-8")
    if method == "GET" and clean_path == "/api/v1/wave6/summary":
        return 200, "application/json", json.dumps(get_wave6_summary(), indent=2).encode("utf-8")

    if method == "POST":
        try:
            payload = json.loads(body_bytes.decode("utf-8")) if body_bytes else {}
        except Exception as exc:
            return 400, "application/json", json.dumps({"error": f"Invalid JSON: {exc}"}).encode("utf-8")

        if clean_path == "/api/v1/epistemic/evaluate":
            return 200, "application/json", json.dumps(evaluate_live_epistemic_scenario(payload), indent=2).encode("utf-8")
        if clean_path == "/api/v1/migration/evaluate":
            return 200, "application/json", json.dumps(evaluate_live_migration_stop(payload), indent=2).encode("utf-8")
        if clean_path == "/api/v1/tco/model":
            return 200, "application/json", json.dumps(compute_parameterized_100m_tco(payload), indent=2).encode("utf-8")

    return 404, "application/json", json.dumps({"error": f"Not found: {method} {clean_path}"}).encode("utf-8")


class WorkbenchRequestHandler(BaseHTTPRequestHandler):
    public_only: bool = False

    def _handle(self, method: str) -> None:
        clean_path = urlparse(self.path).path
        copilot_req = is_copilot_path(clean_path)
        length = int(self.headers.get("Content-Length", "0") or "0")
        body = self.rfile.read(length) if length > 0 else b""
        status, ctype, data = dispatch_api_request(
            method,
            self.path,
            body,
            headers=dict(self.headers.items()),
            public_only=bool(self.public_only),
        )
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        if copilot_req:
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
        else:
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:  # noqa: N802
        self._handle("GET")

    def do_POST(self) -> None:  # noqa: N802
        self._handle("POST")

    def do_OPTIONS(self) -> None:  # noqa: N802
        clean_path = urlparse(self.path).path
        if is_copilot_path(clean_path):
            err_bytes = json.dumps({"error": "same_origin_only", "message": "Cross-origin requests are not supported."}).encode("utf-8")
            self.send_response(405)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(err_bytes)))
            self.end_headers()
            self.wfile.write(err_bytes)
            return
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Chelated AI Live Solution Workbench Server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8787)
    parser.add_argument("--public", action="store_true", help="Serve ONLY Non-NDA /copilot routes (404s internal /api/v1/*)")
    parser.add_argument("--nda-token", default="", help="Optional NDA unlock token for /copilot NDA mode")
    args = parser.parse_args()
    if args.nda_token:
        set_service(CopilotService(nda_token=args.nda_token))
    handler_cls = type("ConfiguredWorkbenchHandler", (WorkbenchRequestHandler,), {"public_only": bool(args.public)})
    srv = ThreadingHTTPServer((args.host, args.port), handler_cls)
    print(f"Chelated Live Solution Workbench listening on http://{args.host}:{args.port} (public_only={args.public})", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
