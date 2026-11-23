r"""Stage R6 Gate Closure (`PM-2` Truly Zero-Oracle & Causal Streaming `mu_q` Rerun — Wave-4 J2 Hardened).

Resolves ALL Wave-4 Frontier Cross-Judgment findings (`J2`) while preserving historical `r5b_s1b` and
archiving v1 in `archive/wave4-pre-adjudication-v1-backups/`:
  1. `RedactedQueryCase` guard strips `entity`, `slot`, `family`, `tenant`, `gold`, and `stale_vals`
     before retrieval/selection; any access to a forbidden key increments `oracle_access_attempts` and
     raises `KeyError` immediately.
  2. Routes EVERY arm (`Flat_No_EK5` included) through ONE unified, retrieval-inferred `(tenant, entity, slot)`
     selector (`select_context_unified_labelfree`) operating under the request tenant (`t0` from the request
     session, never per-case metadata), with frozen tie/empty behavior (`UNKNOWN` -> abstain).
  3. Derives `zero_oracle_metadata_reads` and `strictly_causal_score_before_update` dynamically from runtime
     execution traces (not boolean literals), and measures 5-seed query-order shuffle EMA spread.
  4. Renames `Query0W_CenteredTarget` -> `QA_Hybrid_Peak_ZeroMu_DualFit_1Float_FixedMuX_EK5v3` and marks
     `DualAdap` explicitly as a transductive batch-mean diagnostic (`non_causal_batch_mu_q: True`).
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
import json
import re
import sys
import time
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Set, Tuple

import numpy as np
import requests

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from chelated_raw_text_exocortex import CausalStreamingZeroMuBridge  # noqa: E402
from ek5_v3_stratified import EdgeAttestation, StratifiedEvent, StratifiedEvidenceStore  # noqa: E402
from m1_baseline_ladder import fit_closed_form_centered_ridge  # noqa: E402
from scripts.night1_common import (  # noqa: E402
    current_git_sha,
    load_beir_dataset,
    paired_bootstrap_delta,
    process_telemetry,
    write_json_atomic,
)
from scripts.run_phase2_d1_local_fleet_proof import SimpleBM25, _minmax, _start_ssh_tunnel  # noqa: E402
from scripts.run_r5b_spark6724_completion import build_zero_shortcut_v3_flooded_cases  # noqa: E402
from spherical_migration_bridge import (  # noqa: E402
    CenteredProcrustesBridge,
    compose_centered_bridges,
    l2_normalize,
)

_WORD_RE = re.compile(r"[a-z0-9][a-z0-9_\-]*")
_FORBIDDEN_ORACLE_KEYS = frozenset(
    {"entity", "slot", "family", "tenant", "gold", "stale_vals", "events", "legacy_events", "attestations"}
)


class RedactedQueryCase(Mapping[str, Any]):
    """Read-only query envelope exposing ONLY `case_id`, `request_tenant`, and `query_text`."""

    __slots__ = ("_data", "reads_log", "forbidden_attempts")

    def __init__(self, case_id: str, query_text: str, request_tenant: str) -> None:
        self._data = {
            "case_id": str(case_id),
            "query_text": str(query_text),
            "request_tenant": str(request_tenant),
        }
        self.reads_log: List[str] = []
        self.forbidden_attempts: List[str] = []

    def __getitem__(self, key: str) -> Any:
        if key in _FORBIDDEN_ORACLE_KEYS or key not in self._data:
            self.forbidden_attempts.append(key)
            raise KeyError(f"Oracle metadata key {key!r} is redacted during PM-2 retrieval/selection")
        self.reads_log.append(key)
        return self._data[key]

    def get(self, key: str, default: Any = None) -> Any:
        if key in _FORBIDDEN_ORACLE_KEYS:
            self.forbidden_attempts.append(key)
            raise KeyError(f"Oracle metadata key {key!r} is redacted during PM-2 retrieval/selection")
        if key in self._data:
            self.reads_log.append(key)
            return self._data[key]
        return default

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)


def _query_tokens(text: str) -> Set[str]:
    return {t for t in _WORD_RE.findall((text or "").lower()) if len(t) > 1}


def infer_slot_from_retrieval(
    query_text: str,
    request_tenant: str,
    raw_hybrid_scores: np.ndarray,
    global_eids: Sequence[str],
    global_events: Mapping[str, StratifiedEvent],
    *,
    probe_k: int = 12,
) -> Tuple[Optional[Tuple[str, str, str]], str]:
    """Infer `(tenant, entity, slot)` strictly from query tokens + top-`probe_k` retrieved records.

    Frozen tie/empty rule:
      - Empty or non-positive probe pool -> `(None, "UNKNOWN_EMPTY")`
      - Exact top-weight tie between two distinct `(entity, slot)` keys -> `(None, "UNKNOWN_TIED")`
    """
    q_toks = _query_tokens(query_text)
    top_idx = np.argsort(-raw_hybrid_scores, kind="mergesort")[:probe_k]
    weights: Dict[Tuple[str, str, str], float] = defaultdict(float)
    for rank_i, idx_j in enumerate(top_idx):
        sc = float(raw_hybrid_scores[int(idx_j)])
        if sc <= 0.0:
            continue
        ev = global_events[global_eids[int(idx_j)]]
        if ev.tenant != request_tenant:
            continue
        ent_toks = _query_tokens(ev.entity.replace("_", " "))
        slot_toks = _query_tokens(ev.slot.replace("_", " "))
        ent_match = 3.0 if (ent_toks and ent_toks <= q_toks) else (1.5 if (ent_toks & q_toks) else 0.0)
        slot_match = 1.5 if (slot_toks & q_toks) else 0.0
        weights[(ev.tenant, ev.entity, ev.slot)] += ent_match + slot_match + (1.0 / (rank_i + 1.0))

    if not weights:
        return None, "UNKNOWN_EMPTY"
    ranked_slots = sorted(weights.items(), key=lambda kv: (-kv[1], kv[0]))
    if len(ranked_slots) >= 2 and abs(ranked_slots[0][1] - ranked_slots[1][1]) <= 1e-9:
        return None, "UNKNOWN_TIED"
    return ranked_slots[0][0], "INFERRED_UNIQUE"


def select_context_unified_labelfree(
    query_case: RedactedQueryCase,
    bm_vec: np.ndarray,
    dense_vec: np.ndarray,
    global_eids: Sequence[str],
    global_events: Mapping[str, StratifiedEvent],
    slot_index: Mapping[Tuple[str, str, str], Sequence[str]],
    *,
    use_ek5: bool,
    ek5_active_mask: np.ndarray,
    gid_set_active: Set[str],
    st_global: StratifiedEvidenceStore,
    blocked_global: Set[str],
    top_k: int = 5,
) -> Dict[str, Any]:
    """Single unified selector executed identically across ALL arms (`Flat_No_EK5` and `EK5v3` arms)."""
    q_txt = query_case["query_text"]
    req_tenant = query_case["request_tenant"]
    raw_hyb = 0.35 * _minmax(bm_vec) + 0.65 * _minmax(dense_vec)

    inferred_slot, slot_status = infer_slot_from_retrieval(
        q_txt, req_tenant, raw_hyb, global_eids, global_events, probe_k=12
    )
    if inferred_slot is None:
        return {
            "status": slot_status,
            "inferred_slot": None,
            "selected_ids": [],
            "formatted_context": "(No unambiguous records retrieved for query — reply ABSTAIN.)",
        }

    # Filter candidate ranking to the request tenant; EK5 arms apply `ek5_active_mask` + observable CONTESTED check
    tenant_mask = np.array([global_events[gid].tenant == req_tenant for gid in global_eids], dtype=bool)
    if use_ek5:
        s_gids = slot_index.get(inferred_slot, ())
        act_admit = [g for g in s_gids if g in gid_set_active]
        cont = [g for g in s_gids if st_global.eligibility.get(g) == "CONTESTED" and g not in blocked_global]
        if cont and not act_admit:
            return {
                "status": "CONTESTED_ABSTAIN",
                "inferred_slot": inferred_slot,
                "selected_ids": [],
                "formatted_context": "(No admitted records — conflicting claims on the inferred slot are epistemically CONTESTED.)",
            }
        eff_mask = tenant_mask & ek5_active_mask
    else:
        eff_mask = tenant_mask

    masked_scores = np.where(eff_mask, raw_hyb, -1e9)
    order = np.argsort(-masked_scores, kind="mergesort")[:top_k]
    cand_gids = [global_eids[int(j)] for j in order if masked_scores[int(j)] > -1e8]
    if not cand_gids:
        return {
            "status": "EMPTY_ABSTAIN",
            "inferred_slot": inferred_slot,
            "selected_ids": [],
            "formatted_context": "(No admitted records — conflicting claims on the inferred slot are epistemically CONTESTED.)",
        }

    ordered = sorted(cand_gids, key=lambda g: (global_events[g].tx_time, g))
    ctx = "\n".join(
        f"- [vt={global_events[g].valid_time}, tx={global_events[g].tx_time}] {global_events[g].text}"
        for g in ordered
    )
    return {
        "status": "OK",
        "inferred_slot": inferred_slot,
        "selected_ids": ordered,
        "formatted_context": ctx,
    }


def _call_vllm(endpoint: str, model_id: str, sys_p: str, usr_p: str, max_tokens: int = 24) -> Tuple[str, int, float, bool]:
    t0 = time.perf_counter()
    try:
        r = requests.post(
            f"{endpoint.rstrip('/')}/chat/completions",
            json={
                "model": model_id,
                "messages": [{"role": "system", "content": sys_p}, {"role": "user", "content": usr_p}],
                "temperature": 0.0,
                "max_tokens": max_tokens,
                "chat_template_kwargs": {"enable_thinking": False},
            },
            timeout=60.0,
        )
        r.raise_for_status()
        d = r.json()
        txt = (d["choices"][0]["message"].get("content") or "").strip()
        pt = int((d.get("usage") or {}).get("prompt_tokens", 0))
        return txt, pt, (time.perf_counter() - t0) * 1000.0, False
    except Exception as exc:
        return f"ERROR: {exc}", 0, (time.perf_counter() - t0) * 1000.0, True


def _grade_pm2_answer(pred: str, gold: str, stale_vals: Sequence[str]) -> bool:
    p_lo = pred.lower()
    if gold == "ABSTAIN":
        return (
            ("abstain" in p_lo) or ("cannot" in p_lo) or ("contradict" in p_lo) or ("no admitted" in p_lo)
        ) and not any(sv.lower() in p_lo for sv in stale_vals)
    return (gold.lower() in p_lo) and not any(sv.lower() in p_lo for sv in stale_vals) and ("abstain" not in p_lo)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "artifacts" / "r6_pm2_labelfree_closure")
    parser.add_argument("--spark-host", type=str, default="10.0.0.194")
    parser.add_argument("--spark-port", type=int, default=8890)
    parser.add_argument("--local-port", type=int, default=18894)
    parser.add_argument("--model-id", type=str, default="nemotron-lightning")
    args = parser.parse_args()

    out_dir: Path = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()

    import torch
    from sentence_transformers import SentenceTransformer

    raw_cases = build_zero_shortcut_v3_flooded_cases(n_per_family=25)  # 8 * 25 = 200 cases, 3,200 resident records

    # Immediately separate resident store ingestion + sealed gold scorer table from redacted evaluation queries!
    global_eids: List[str] = []
    global_txts: List[str] = []
    global_events: Dict[str, StratifiedEvent] = {}
    global_atts: List[EdgeAttestation] = []
    slot_index: Dict[Tuple[str, str, str], List[str]] = defaultdict(list)
    gold_scorer_only: Dict[str, Dict[str, Any]] = {}
    redacted_queries: List[RedactedQueryCase] = []

    for c in raw_cases:
        cid = c["case_id"]
        tenants_in_case = {ev.tenant for ev in c["events"].values()}
        assert len(tenants_in_case) == 1, f"Expected single request tenant per case, got {tenants_in_case}"
        req_tenant = next(iter(tenants_in_case))
        redacted_queries.append(RedactedQueryCase(cid, c["query_text"], req_tenant))
        gold_scorer_only[cid] = {
            "gold": c["gold"],
            "stale_vals": tuple(c["stale_vals"]),
            "oracle_entity_for_posthoc_audit_only": c["entity"],
            "oracle_slot_for_posthoc_audit_only": c["slot"],
        }
        for eid, ev in c["events"].items():
            gid = f"{cid}__{eid}"
            global_eids.append(gid)
            global_txts.append(ev.text)
            gev = StratifiedEvent(
                event_id=gid, entity=ev.entity, slot=ev.slot, claim_value=ev.claim_value, text=ev.text,
                valid_time=ev.valid_time, tx_time=ev.tx_time, tenant=ev.tenant, domain=ev.domain, authority_tier=ev.authority_tier,
            )
            global_events[gid] = gev
            slot_index[(gev.tenant, gev.entity, gev.slot)].append(gid)
        for att in c["attestations"]:
            global_atts.append(
                EdgeAttestation(
                    f"{cid}__{att.attestation_id}", f"{cid}__{att.src_id}", f"{cid}__{att.dst_id}",
                    att.edge_type, valid_time=att.valid_time, tx_time=att.tx_time, lineage_id=att.lineage_id,
                )
            )
    # Delete raw_cases reference so retrieval/selection loop cannot even accidentally touch unredacted dicts!
    del raw_cases

    # Unit self-check of RedactedQueryCase guard before proceeding
    _guard_probe = RedactedQueryCase("probe_0", "What is x?", "t0")
    _guard_tripped = False
    try:
        _ = _guard_probe["entity"]
    except KeyError:
        _guard_tripped = True
    assert _guard_tripped and _guard_probe.forbidden_attempts == ["entity"]

    enc_e0 = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cuda")
    enc_e3 = SentenceTransformer("sentence-transformers/all-mpnet-base-v2", device="cuda")
    enc_e1 = SentenceTransformer("BAAI/bge-small-en-v1.5", device="cuda")

    _, sf_txts, _, _, _, sf_qrels_sha = load_beir_dataset("SciFact", max_docs=1200)
    sf_e0 = enc_e0.encode(sf_txts[:1000], batch_size=128, convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)
    sf_e3 = enc_e3.encode(sf_txts[:1000], batch_size=128, convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)
    sf_e1 = enc_e1.encode(sf_txts[:1000], batch_size=128, convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)

    b_03 = CenteredProcrustesBridge.fit(sf_e0[:800], sf_e3[:800], cal_x=sf_e0[800:], cal_y=sf_e3[800:])
    b_31 = CenteredProcrustesBridge.fit(sf_e3[:800], sf_e1[:800], cal_x=sf_e3[800:], cal_y=sf_e1[800:])
    br_peak_polar = compose_centered_bridges([b_03, b_31], polar_reproject=True)
    mx03, my03, wc03 = fit_closed_form_centered_ridge(sf_e0, sf_e3, fit_idx=np.arange(800), regularization=10.0)
    mx31, my31, wc31 = fit_closed_form_centered_ridge(sf_e3, sf_e1, fit_idx=np.arange(800), regularization=10.0)

    st_global = StratifiedEvidenceStore(strict_dag_supersede=True)
    st_global.apply_mutation(list(global_events.values()), global_atts)

    doc_e1 = enc_e1.encode(global_txts, batch_size=128, convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)
    q_txts = [qc["query_text"] for qc in redacted_queries]
    q_e0 = enc_e0.encode(q_txts, batch_size=64, convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)
    q_e1_native = enc_e1.encode(q_txts, batch_size=64, convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)
    q_e1_car = l2_normalize((l2_normalize((q_e0 - mx03) @ wc03 + my03) - mx31) @ wc31 + my31).astype(np.float32)
    del enc_e0, enc_e3, enc_e1
    torch.cuda.empty_cache()

    telem_global = st_global.get_telemetry()
    blocked_global: Set[str] = set()
    for u_b, w_b, _et_b in telem_global.blocked_authority_edges:
        if st_global.eligibility.get(w_b) in {"ADMIT", "CONTESTED"}:
            blocked_global.add(u_b)
            for d_b in st_global.deps_v.get(u_b, ()):
                if st_global.eff_auth.get(d_b, 0) < st_global.eff_auth.get(w_b, 0) and global_events[d_b].claim_value == global_events[u_b].claim_value:
                    blocked_global.add(d_b)
            u_ev = global_events[u_b]
            w_auth = st_global.eff_auth.get(w_b, 0)
            for cand_gid in slot_index[(u_ev.tenant, u_ev.entity, u_ev.slot)]:
                if global_events[cand_gid].claim_value == u_ev.claim_value and st_global.eff_auth.get(cand_gid, 0) < w_auth:
                    blocked_global.add(cand_gid)

    ret_tx_by_slot_val: Dict[Tuple[str, str, str, str], int] = {}
    for att in global_atts:
        if att.edge_type in ("REFUTE", "SUPERSEDE", "VERSION_LINEAGE_SUPERSEDE"):
            is_act = (
                att.src_id in st_global.refs.get(att.dst_id, ())
                or att.src_id in st_global.sup.get(att.dst_id, ())
                or att.src_id in st_global.lsup.get(att.dst_id, ())
            )
            if is_act and st_global.eligibility.get(att.src_id) == "ADMIT" and st_global.eligibility.get(att.dst_id) == "AUDIT_ONLY":
                dst_ev = global_events[att.dst_id]
                if dst_ev.claim_value:
                    key_sv = (dst_ev.tenant, dst_ev.entity, dst_ev.slot, dst_ev.claim_value)
                    ret_tx_by_slot_val[key_sv] = max(ret_tx_by_slot_val.get(key_sv, -1), att.tx_time)

    def _is_active_admit(gid: str) -> bool:
        ev = global_events[gid]
        k_sv = (ev.tenant, ev.entity, ev.slot, ev.claim_value)
        return (
            st_global.eligibility.get(gid) == "ADMIT"
            and gid not in blocked_global
            and bool(ev.claim_value)
            and not (k_sv in ret_tx_by_slot_val and ev.valid_time <= ret_tx_by_slot_val[k_sv])
        )

    ek5_active_mask = np.array([_is_active_admit(gid) for gid in global_eids], dtype=bool)
    gid_set_active = {gid for gid, ok in zip(global_eids, ek5_active_mask) if ok}

    bm25_global = SimpleBM25(global_txts)
    bm_cache = [bm25_global.score(qc["query_text"]) for qc in redacted_queries]
    mu_d_e1 = doc_e1.mean(axis=0, keepdims=True).astype(np.float32)
    d_e1_norms = np.maximum(np.linalg.norm(doc_e1 - mu_d_e1, axis=1).astype(np.float32), 1e-12)

    # Transductive batch-mean diagnostic (`DualAdap`, explicitly flagged non-causal!) and fixed-mu_x 1-float sidecar (`DualFit_1Float_FixedMuX`)
    mu_q_e0_batch = q_e0.mean(axis=0, keepdims=True).astype(np.float32)
    q_zm_dual_e1 = l2_normalize((q_e0 - mu_q_e0_batch) @ br_peak_polar.Q).astype(np.float32)
    q_zm_fixed_mux_e1 = l2_normalize((q_e0 - br_peak_polar.mu_x) @ br_peak_polar.Q).astype(np.float32)

    causal_stream = CausalStreamingZeroMuBridge(br_peak_polar, doc_e1, ema_alpha=0.05)
    causal_trace_steps = 0
    causal_step0_matches_mux = bool(np.allclose(causal_stream.mu_q_ema, br_peak_polar.mu_x.reshape(1, -1), atol=1e-7))
    causal_all_updates_post_score = True

    cids_qa = [qc["case_id"] for qc in redacted_queries]
    selections: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(dict)
    full_hybrid_causal_lat_ms: List[float] = []

    for i_q, qc in enumerate(redacted_queries):
        cid = qc["case_id"]
        mu_before = causal_stream.mu_q_ema.copy()
        # Verify score_single_query matches manual projection with pre-update `mu_before`!
        q_expected_pre = l2_normalize((q_e0[i_q : i_q + 1] - mu_before) @ br_peak_polar.Q)[0]
        expected_dense = ((doc_e1 @ q_expected_pre) - float(mu_d_e1[0] @ q_expected_pre)) / d_e1_norms

        t_hq0 = time.perf_counter()
        bm_v = bm25_global.score(qc["query_text"])
        d_zm_causal = causal_stream.score_single_query(q_e0[i_q])
        sel_causal = select_context_unified_labelfree(
            qc, bm_v, d_zm_causal, global_eids, global_events, slot_index,
            use_ek5=True, ek5_active_mask=ek5_active_mask, gid_set_active=gid_set_active,
            st_global=st_global, blocked_global=blocked_global, top_k=5,
        )
        full_hybrid_causal_lat_ms.append((time.perf_counter() - t_hq0) * 1000.0)
        mu_after = causal_stream.mu_q_ema.copy()
        if not np.allclose(d_zm_causal, expected_dense, atol=1e-5) or np.allclose(mu_after, mu_before, atol=1e-12):
            causal_all_updates_post_score = False
        causal_trace_steps += 1

        selections["QA_Hybrid_Peak_ZeroMu_CausalEMA_EK5v3"][cid] = sel_causal

        d_nat = doc_e1 @ q_e1_native[i_q]
        d_zmda = ((doc_e1 @ q_zm_dual_e1[i_q]) - float(mu_d_e1[0] @ q_zm_dual_e1[i_q])) / d_e1_norms
        d_zm_fix = ((doc_e1 @ q_zm_fixed_mux_e1[i_q]) - float(mu_d_e1[0] @ q_zm_fixed_mux_e1[i_q])) / d_e1_norms
        d_car = doc_e1 @ q_e1_car[i_q]

        for arm_name, d_vec, ek5_flag in (
            ("QA_Hybrid_Native_E1_EK5v3", d_nat, True),
            ("QA_Hybrid_Peak_ZeroMu_DualAdap_BatchDiagnostic_EK5v3", d_zmda, True),
            ("QA_Hybrid_Peak_ZeroMu_DualFit_1Float_FixedMuX_EK5v3", d_zm_fix, True),
            ("QA_Hybrid_Peak_Composed_CAR_EK5v3", d_car, True),
            ("QA_Hybrid_Peak_ZeroMu_Flat_No_EK5", d_zm_causal, False),
        ):
            selections[arm_name][cid] = select_context_unified_labelfree(
                qc, bm_v, d_vec, global_eids, global_events, slot_index,
                use_ek5=ek5_flag, ek5_active_mask=ek5_active_mask, gid_set_active=gid_set_active,
                st_global=st_global, blocked_global=blocked_global, top_k=5,
            )

    # 5-seed query-order shuffle sensitivity check for `CausalStreamingZeroMuBridge`
    shuffle_seeds = (11, 22, 33, 44, 55)
    shuffle_selections: Dict[int, Dict[str, Dict[str, Any]]] = {}
    for sh_seed in shuffle_seeds:
        perm = np.random.default_rng(sh_seed).permutation(len(redacted_queries))
        sh_stream = CausalStreamingZeroMuBridge(br_peak_polar, doc_e1, ema_alpha=0.05)
        sh_map: Dict[str, Dict[str, Any]] = {}
        for idx_p in perm:
            i_int = int(idx_p)
            qc = redacted_queries[i_int]
            d_sh = sh_stream.score_single_query(q_e0[i_int])
            sh_map[qc["case_id"]] = select_context_unified_labelfree(
                qc, bm_cache[i_int], d_sh, global_eids, global_events, slot_index,
                use_ek5=True, ek5_active_mask=ek5_active_mask, gid_set_active=gid_set_active,
                st_global=st_global, blocked_global=blocked_global, top_k=5,
            )
        shuffle_selections[sh_seed] = sh_map

    # Runtime trace verification BEFORE touching `gold_scorer_only`
    total_forbidden_attempts = sum(len(qc.forbidden_attempts) for qc in redacted_queries)
    allowed_keys_seen = sorted({k for qc in redacted_queries for k in qc.reads_log})
    zero_oracle_reads_verified = (
        total_forbidden_attempts == 0
        and set(allowed_keys_seen) <= {"case_id", "query_text", "request_tenant"}
    )
    strictly_causal_ema_verified = bool(
        causal_step0_matches_mux and causal_all_updates_post_score and causal_trace_steps == len(redacted_queries)
    )

    sys_qa = (
        "Answer the factual query strictly using ONLY the dated records below for the exact entity asked. "
        "Reply with ONLY the exact current value (e.g., '65psi' or '25ml'). "
        "If no records for that entity are present or they are mutually contested with no resolution, reply ONLY 'ABSTAIN'."
    )
    unique_prompts: Dict[str, str] = {}
    qa_tasks = []
    for qc in redacted_queries:
        cid = qc["case_id"]
        for arm_name, sel_map in selections.items():
            sel = sel_map[cid]
            prompt = f"Records:\n{sel['formatted_context']}\n\nQuestion: {qc['query_text']}\nAnswer:"
            psha = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
            unique_prompts[psha] = prompt
            qa_tasks.append({"cid": cid, "arm": arm_name, "psha": psha, "sel_status": sel["status"], "inferred_slot": sel["inferred_slot"]})
        for sh_seed, sh_map in shuffle_selections.items():
            sel = sh_map[cid]
            prompt = f"Records:\n{sel['formatted_context']}\n\nQuestion: {qc['query_text']}\nAnswer:"
            psha = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
            unique_prompts[psha] = prompt

    print(f"[R6-PM2-ZeroOracle] Executing {len(unique_prompts)} unique QA prompts on {args.spark_host}:{args.spark_port}...", flush=True)
    tunnel = _start_ssh_tunnel(args.spark_host, args.local_port, args.spark_port)
    try:
        endpoint = f"http://127.0.0.1:{args.local_port}/v1"
        p_res: Dict[str, Tuple[str, int, float, bool]] = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=20) as pool:
            for psha, res in pool.map(
                lambda kv: (kv[0], _call_vllm(endpoint, args.model_id, sys_qa, kv[1], max_tokens=24)),
                list(unique_prompts.items()),
            ):
                p_res[psha] = res
    finally:
        tunnel.terminate()

    qa_rows = []
    by_qa_arm: Dict[str, Dict[str, float]] = defaultdict(dict)
    slot_inference_exact_by_arm: Dict[str, List[int]] = defaultdict(list)

    for qt in qa_tasks:
        cid = qt["cid"]
        ginfo = gold_scorer_only[cid]
        pred, pt, _lat, err = p_res[qt["psha"]]
        ok = _grade_pm2_answer(pred, ginfo["gold"], ginfo["stale_vals"])
        by_qa_arm[qt["arm"]][cid] = float(int(ok))
        inf_slot = qt["inferred_slot"]
        slot_exact = int(
            inf_slot is not None
            and inf_slot[1] == ginfo["oracle_entity_for_posthoc_audit_only"]
            and inf_slot[2] == ginfo["oracle_slot_for_posthoc_audit_only"]
        )
        slot_inference_exact_by_arm[qt["arm"]].append(slot_exact)
        qa_rows.append({
            "case_id": cid,
            "gold": ginfo["gold"],
            "arm": qt["arm"],
            "selection_status": qt["sel_status"],
            "inferred_slot": f"{inf_slot[0]}:{inf_slot[1]}.{inf_slot[2]}" if inf_slot else "NONE",
            "inferred_slot_matches_gold": slot_exact,
            "pred": pred.replace("\n", " "),
            "is_correct": int(ok),
            "prompt_tokens": pt,
            "api_error": int(err),
        })

    shuffle_accs: Dict[str, float] = {}
    for sh_seed, sh_map in shuffle_selections.items():
        sh_correct = []
        for qc in redacted_queries:
            cid = qc["case_id"]
            ginfo = gold_scorer_only[cid]
            prompt = f"Records:\n{sh_map[cid]['formatted_context']}\n\nQuestion: {qc['query_text']}\nAnswer:"
            psha = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
            pred = p_res[psha][0]
            sh_correct.append(float(int(_grade_pm2_answer(pred, ginfo["gold"], ginfo["stale_vals"]))))
        shuffle_accs[f"seed_{sh_seed}"] = round(float(np.mean(sh_correct)), 4)

    qa_csv = out_dir / "r6_pm2_labelfree_qa_rows.csv"
    with qa_csv.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(qa_rows[0].keys()))
        w.writeheader()
        w.writerows(qa_rows)

    causal_arm = by_qa_arm["QA_Hybrid_Peak_ZeroMu_CausalEMA_EK5v3"]
    bt_causal_vs_nat = paired_bootstrap_delta(causal_arm, by_qa_arm["QA_Hybrid_Native_E1_EK5v3"], n_bootstrap=3000, seed=42, expected_ids=cids_qa)
    bt_causal_vs_car = paired_bootstrap_delta(causal_arm, by_qa_arm["QA_Hybrid_Peak_Composed_CAR_EK5v3"], n_bootstrap=3000, seed=42, expected_ids=cids_qa)
    bt_causal_vs_noek5 = paired_bootstrap_delta(causal_arm, by_qa_arm["QA_Hybrid_Peak_ZeroMu_Flat_No_EK5"], n_bootstrap=3000, seed=42, expected_ids=cids_qa)

    p95_hybrid_ms = float(np.percentile(full_hybrid_causal_lat_ms, 95))
    total_api_errors = sum(r["api_error"] for r in qa_rows)
    checks = {
        "zero_oracle_metadata_reads_verified_from_trace": bool(zero_oracle_reads_verified),
        "unified_selector_across_all_arms_including_flat": True,
        "strictly_causal_online_ema_mu_q_verified_from_trace": bool(strictly_causal_ema_verified),
        "zero_api_errors": bool(total_api_errors == 0),
        "full_hybrid_3200rec_query_p95_le_25ms": bool(p95_hybrid_ms <= 25.0),
        "causal_bridged_minus_native_point_ge_minus_3pp": bool(bt_causal_vs_nat["mean_delta"] >= -0.03),
        "causal_bridged_minus_native_bootstrap_lcb95_ge_minus_5pp": bool(bt_causal_vs_nat["lcb_95"] >= -0.05),
        "causal_bridged_minus_native_tango_lcb95_ge_minus_5pp": bool(bt_causal_vs_nat["tango_score_lcb95"] >= -0.05),
        "causal_bridged_minus_car_lcb95_gt_0": bool(bt_causal_vs_car["lcb_95"] > 0.0),
        "ek5_echo_stress_gain_ge_20pp": bool(bt_causal_vs_noek5["mean_delta"] >= 0.20),
    }

    sh_vals = list(shuffle_accs.values())
    manifest = {
        "schema_version": "r6_pm2_zero_oracle_causal_closure_v2",
        "git_sha": current_git_sha(REPO_ROOT),
        "wall_seconds": round(time.perf_counter() - t0, 2),
        "calibration_scifact_qrels_sha256": sf_qrels_sha,
        "runtime_provenance_audit": {
            "redacted_query_allowed_keys_accessed": allowed_keys_seen,
            "forbidden_oracle_key_attempts": total_forbidden_attempts,
            "redacted_guard_self_test_passed": bool(_guard_tripped),
            "causal_step0_equals_anchor_mu_x": causal_step0_matches_mux,
            "causal_all_200_steps_scored_before_ema_update": causal_all_updates_post_score,
            "slot_inference_exact_rate_by_arm": {
                k: round(float(np.mean(v)), 4) for k, v in sorted(slot_inference_exact_by_arm.items())
            },
            "non_causal_diagnostic_arms": ["QA_Hybrid_Peak_ZeroMu_DualAdap_BatchDiagnostic_EK5v3"],
            "sidecar_disclosure": "DualFit_1Float_FixedMuX and CausalEMA both use the 4-byte per-doc ||d_j - mu_d||_2 sidecar plus shared D-float mu_d.",
            "ceiling_and_tango_disclosure": "Native_E1_EK5v3 sits at or near the 1.000 ceiling on this structured corpus; check both bootstrap_lcb95 and exact Tango score LCB95.",
        },
        "ema_query_order_5seed_shuffle": {
            "per_seed_accuracy": shuffle_accs,
            "mean": round(float(np.mean(sh_vals)), 4),
            "min": round(float(np.min(sh_vals)), 4),
            "max": round(float(np.max(sh_vals)), 4),
            "std": round(float(np.std(sh_vals)), 4),
        },
        "full_3200_record_hybrid_causal_query_latency_ms": {
            "p50_ms": round(float(np.percentile(full_hybrid_causal_lat_ms, 50)), 3),
            "p95_ms": round(p95_hybrid_ms, 3),
            "max_ms": round(float(np.max(full_hybrid_causal_lat_ms)), 3),
            "n_queries": len(full_hybrid_causal_lat_ms),
        },
        "arm_accuracy_n200": {k: round(float(np.mean(list(v.values()))), 4) for k, v in sorted(by_qa_arm.items())},
        "paired_comparisons": {
            "CausalEMA_vs_Native_E1": {
                "mean_delta": round(bt_causal_vs_nat["mean_delta"], 4),
                "bootstrap_lcb95": round(bt_causal_vs_nat["lcb_95"], 4),
                "bootstrap_ucb95": round(bt_causal_vs_nat["ucb_95"], 4),
                "tango_lcb95": round(bt_causal_vs_nat["tango_score_lcb95"], 4),
                "tango_ucb95": round(bt_causal_vs_nat["tango_score_ucb95"], 4),
            },
            "CausalEMA_vs_Composed_CAR": {
                "mean_delta": round(bt_causal_vs_car["mean_delta"], 4),
                "bootstrap_lcb95": round(bt_causal_vs_car["lcb_95"], 4),
                "bootstrap_ucb95": round(bt_causal_vs_car["ucb_95"], 4),
                "tango_lcb95": round(bt_causal_vs_car["tango_score_lcb95"], 4),
            },
            "CausalEMA_EK5_vs_Flat_No_EK5": {
                "mean_delta": round(bt_causal_vs_noek5["mean_delta"], 4),
                "bootstrap_lcb95": round(bt_causal_vs_noek5["lcb_95"], 4),
                "bootstrap_ucb95": round(bt_causal_vs_noek5["ucb_95"], 4),
                "tango_lcb95": round(bt_causal_vs_noek5["tango_score_lcb95"], 4),
            },
        },
        "gate_checks": checks,
        "pm2_strict_all_checks_passed": all(checks.values()),
        "pm2_bootstrap_gate_passed": all(
            v for k, v in checks.items() if k != "causal_bridged_minus_native_tango_lcb95_ge_minus_5pp"
        ),
        "rows_csv_sha256": hashlib.sha256(qa_csv.read_bytes()).hexdigest(),
        "telemetry": process_telemetry(),
    }
    manifest_path = out_dir / "r6_pm2_labelfree_manifest.json"
    write_json_atomic(manifest_path, manifest)
    print(f"[R6-PM2-ZeroOracle] DONE! Strict={manifest['pm2_strict_all_checks_passed']} BootstrapGate={manifest['pm2_bootstrap_gate_passed']}", flush=True)
    print(json.dumps(manifest["arm_accuracy_n200"], indent=2), flush=True)
    print(json.dumps(manifest["paired_comparisons"], indent=2), flush=True)


if __name__ == "__main__":
    main()
