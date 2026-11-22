"""Flagship #2 Unified Zero-Shortcut Live Benchmark + Fresh Held-Out ACL 2024 LoCoMo (`nemotron-lightning`).

Directly resolves ALL major critiques from Astra 6.2 XHigh Audit 2 & Carbon:
  Part A — Zero-Timestamp-Shortcut Unified End-to-End Live Suite (900 live answer calls, 0 role tags!):
    Routes `C4_v2_iq04_ek5_wfs` directly through the persistent `IncrementalEvidenceStore` (EK5 v2 3-valued WFS),
    stripping ALL synthetic `role` metadata from BOTH the LLM prompt AND the retriever internals!
    Includes families that BREAK timestamp shortcuts and naive deduplication:
      1. `stale_newest_echo_after_correction`: unlinked stale echo arrives AT THE NEWEST TIMESTAMP (t=14)
         after the authoritative correction (t=10) -> Latest-Timestamp control scores 0%!
      2. `repeated_valid_truth_vs_late_false_rumor`: valid truth corroborated 6 times (t=1..6) vs 1 late
         noisy uncorroborated rumor at t=14 -> Latest-Timestamp fails; IQ-04 preserves valid truth!
      3. `mutual_refutation_contested_abstain`: mutually refuting A <-> B with no external ground ->
         EK5 v2 3-valued WFS resolves BOTH to CONTESTED -> C4_v2 abstains cleanly!
      4. `flood_12_unlinked_paraphrase_copies`: 12 unlinked copies > top_k=5 + signed refutation.
      5. `unlinked_downstream_synthesis_from_retracted`: unlinked downstream synthesis reusing retracted value.
    Comparators: C0_bm25, C1_scalemas_bounded (bounded saturation per citing message), C2_latest_timestamp,
                 C2b_bm25_jaccard_dedup (Astra 6.2 control!), C3_ordinary_bitemporal_tms, C4_v2_iq04_ek5_wfs.

  Part B — Fresh Held-Out External Conversational Memory Benchmark: ACL 2024 LoCoMo (`Percena/locomo-mc10`):
    NEVER inspected during M4/M4b development! Evaluates 120 stratified questions across all 10 conversations
    (40 Temporal Cat-2 + 40 Single/Multi-Hop Cat-1/4 + 40 Adversarial Unanswerable/Abstention Cat-5) under
    Regime A (Unmodified) AND Regime B (Cross-Session Echo Stress) across:
      - M0_bm25_flat, M1_hybrid_flat, M1b_bm25_jaccard_dedup (Astra 6.2 control!), M2_scalemas_unsigned,
        M3_recency_hybrid, M4c_cross_session_collapse, M4d_cross_session_collapse_plus_supersession_tag.
    Uses an ABSTENTION-AWARE, STRICTLY DETERMINISTIC, CACHED grader (identical answers ALWAYS get identical
    grades), logs exact answer_calls & judge_calls separately, preserves full raw completions, and hashes
    every CSV + manifest inside report_receipt.json!
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
import json
import math
import re
import sys
import threading
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import requests
from huggingface_hub import hf_hub_download
from sentence_transformers import SentenceTransformer

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from iq04_temporal_evidence import (  # noqa: E402
    EvidenceEdge,
    EvidenceEvent,
    IncrementalEvidenceStore,
    ScopedBM25,
    compute_full_recompute_eligibility,
)
from scripts.night1_common import (  # noqa: E402
    current_git_sha,
    paired_bootstrap_delta,
    process_telemetry,
    write_json_atomic,
)

_TOK_RE = re.compile(r"[a-z0-9]+")
_STOPWORDS = frozenset(
    {
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "to", "of", "and", "in", "that",
        "it", "for", "on", "with", "as", "at", "by", "this", "from", "or", "but", "not", "my", "i",
        "me", "you", "your", "we", "they", "he", "she", "his", "her", "have", "has", "had", "do",
        "did", "does", "can", "could", "would", "should", "just", "so", "very", "also", "about",
        "what", "when", "where", "who", "how", "which",
    }
)

_DOMAINS = (
    ("pump_station", "target_psi", ("40psi", "65psi", "90psi", "115psi")),
    ("serum_protocol", "dose_ml", ("10ml", "25ml", "50ml", "75ml")),
    ("relay_bus", "cutoff_hz", ("50hz", "60hz", "120hz", "240hz")),
    ("orbital_burn", "delta_v", ("140ms", "220ms", "310ms", "450ms")),
    ("customs_tariff", "rate_pct", ("4pct", "9pct", "15pct", "22pct")),
)

_ECHO_WRAPPERS = (
    "Regional desk note states {ent} {slot} is {val} per digest {tag}.",
    "Archived shift briefing lists {ent} {slot} at {val} in log {tag}.",
    "Field relay bulletin confirms {slot} for {ent} equals {val} ({tag}).",
    "Secondary ops memo repeats {ent} {slot} = {val} under ref {tag}.",
    "Cross-team summary records {val} for {ent} {slot} (packet {tag}).",
    "Forwarded status entry notes {ent} {slot} set to {val}, item {tag}.",
)


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _content_tokens(text: str) -> set[str]:
    return {t for t in _TOK_RE.findall(text.lower()) if t not in _STOPWORDS and len(t) > 1}


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / float(len(a | b))


def _call_vllm(
    endpoint: str,
    model_id: str,
    sys_p: str,
    usr_p: str,
    *,
    max_tokens: int = 96,
) -> Tuple[str, int, float, bool]:
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
            timeout=45.0,
        )
        r.raise_for_status()
        d = r.json()
        txt = (d["choices"][0]["message"].get("content") or "").strip()
        pt = int(d.get("usage", {}).get("prompt_tokens", 0))
        return txt, pt, (time.perf_counter() - t0) * 1000.0, False
    except Exception as exc:  # noqa: BLE001
        return f"ERROR: {exc}", 0, (time.perf_counter() - t0) * 1000.0, True


# ---------------------------------------------------------------------------
# PART A: Zero-Timestamp-Shortcut Unified EK5 v2 Live Suite (ZERO role tags!)
# ---------------------------------------------------------------------------
def build_zero_shortcut_cases(n_per_family: int = 30) -> List[Dict[str, Any]]:
    families = (
        "stale_newest_echo_after_correction",
        "repeated_valid_truth_vs_late_false_rumor",
        "mutual_refutation_contested_abstain",
        "flood_12_unlinked_paraphrase_copies",
        "unlinked_downstream_synthesis_from_retracted",
    )
    cases: List[Dict[str, Any]] = []
    idx = 0
    for fam in families:
        for k in range(n_per_family):
            dom, slot, vals = _DOMAINS[(idx + k) % len(_DOMAINS)]
            ent = f"{dom}_zs_{idx:03d}"
            v_old = vals[idx % 4]
            v_new = vals[(idx + 1) % 4]
            events: Dict[str, EvidenceEvent] = {}
            edges: List[EvidenceEdge] = []

            def add_e(eid: str, val: str, txt: str, vt: int, tt: int, src: str) -> None:
                # Notice: role is ALWAYS generic "EVENT" — ZERO synthetic role tags anywhere!
                events[eid] = EvidenceEvent(eid, ent, ent, slot, val, txt, vt, tt, src, role="EVENT")

            if fam == "stale_newest_echo_after_correction":
                # Root false claim at t=1, 6 unlinked echoes at t=2..7, authoritative correction at t=10
                # refuting root, AND a STALE UNLINKED ECHO arriving LATE at t=14 (NEWEST timestamp!)
                add_e("r_bad", v_old, f"Initial telemetry logged {ent} {slot} as {v_old}.", 1, 1, "s0")
                for i in range(6):
                    add_e(
                        f"echo_{i}", v_old,
                        _ECHO_WRAPPERS[i % len(_ECHO_WRAPPERS)].format(ent=ent, slot=slot, val=v_old, tag=f"E{i}"),
                        2 + i, 2 + i, f"se{i}",
                    )
                add_e("corr", v_new, f"Verified audit supersedes initial telemetry for {ent} {slot}; operative value is {v_new}.", 10, 10, "s_aud")
                edges.append(EvidenceEdge("corr", "r_bad", "REFUTE", 10))
                edges.append(EvidenceEdge("corr", "r_bad", "SUPERSEDE", 10))
                # Stale delayed echo at t=14 (NEWEST!) repeating retracted value v_old without citation link!
                add_e("late_stale_echo", v_old, f"Delayed forwarded briefing still repeats {ent} {slot} equals {v_old} from earlier cycle.", 3, 14, "s_late")
                gold = v_new
                stale_vals = (v_old,)

            elif fam == "repeated_valid_truth_vs_late_false_rumor":
                # 6 independent corroborated records asserting v_new (t=1..6) + 1 uncorroborated noisy late note at t=14
                add_e("v_root", v_new, f"Certified baseline establishes {ent} {slot} at {v_new}.", 1, 1, "s_cert")
                for i in range(5):
                    add_e(f"sup_{i}", v_new, f"Independent inspection {i} verifies {ent} {slot} remains {v_new}.", 2 + i, 2 + i, f"si{i}")
                    edges.append(EvidenceEdge(f"sup_{i}", "v_root", "SUPPORT", 2 + i))
                add_e("late_rumor", v_old, f"Unverified rumor at t=14 wonders if {ent} {slot} could be {v_old}.", 14, 14, "s_rum")
                gold = v_new
                stale_vals = (v_old,)

            elif fam == "mutual_refutation_contested_abstain":
                # Two conflicting reports mutually refuting each other with ZERO resolution -> CONTESTED -> ABSTAIN!
                add_e("rep_a", v_old, f"Report Alpha states {ent} {slot} is {v_old} and refutes Report Beta.", 5, 5, "sa")
                add_e("rep_b", v_new, f"Report Beta states {ent} {slot} is {v_new} and refutes Report Alpha.", 6, 6, "sb")
                edges.append(EvidenceEdge("rep_a", "rep_b", "REFUTE", 7))
                edges.append(EvidenceEdge("rep_b", "rep_a", "REFUTE", 7))
                gold = "ABSTAIN"
                stale_vals = (v_old, v_new)

            elif fam == "flood_12_unlinked_paraphrase_copies":
                add_e("r_bad", v_old, f"Primary bulletin reports {ent} {slot} is {v_old}.", 1, 1, "s0")
                for i in range(12):
                    add_e(
                        f"copy_{i}", v_old,
                        _ECHO_WRAPPERS[i % len(_ECHO_WRAPPERS)].format(ent=ent, slot=slot, val=v_old, tag=f"F{i}"),
                        2 + (i % 5), 2 + (i % 5), f"sf{i}",
                    )
                add_e("corr", v_new, f"Audit retracts primary bulletin for {ent} {slot}; confirmed setting is {v_new}.", 10, 10, "s_aud")
                edges.append(EvidenceEdge("corr", "r_bad", "REFUTE", 10))
                edges.append(EvidenceEdge("corr", "r_bad", "SUPERSEDE", 10))
                gold = v_new
                stale_vals = (v_old,)

            else:  # unlinked_downstream_synthesis_from_retracted
                add_e("r_bad", v_old, f"Initial sensor feed reports {ent} {slot} = {v_old}.", 1, 1, "s0")
                for i in range(8):
                    add_e(
                        f"syn_{i}", v_old,
                        f"Downstream rollup {i} calculates plan assuming {ent} {slot} at {v_old}.",
                        2 + (i % 4), 2 + (i % 4), f"ss{i}",
                    )
                add_e("corr", v_new, f"Root sensor recalibrated: {ent} {slot} is {v_new}, superseding initial feed.", 10, 10, "s_aud")
                edges.append(EvidenceEdge("corr", "r_bad", "REFUTE", 10))
                edges.append(EvidenceEdge("corr", "r_bad", "SUPERSEDE", 10))
                gold = v_new
                stale_vals = (v_old,)

            cases.append(
                {
                    "case_id": f"zs_{idx:03d}",
                    "family": fam,
                    "entity": ent,
                    "slot": slot,
                    "query_text": f"What is the current confirmed {slot} for {ent}?",
                    "events": events,
                    "edges": edges,
                    "gold": gold,
                    "stale_vals": stale_vals,
                }
            )
            idx += 1
    return cases


def retrieve_zero_shortcut_arm(
    arm: str,
    case: Dict[str, Any],
    *,
    top_k: int = 5,
) -> Tuple[List[str], str]:
    """Return `(selected_event_ids, status_flag)` where status_flag is `OK` or `CONTESTED_ABSTAIN`."""
    events: Dict[str, EvidenceEvent] = case["events"]
    edges: List[EvidenceEdge] = case["edges"]
    eids = sorted(events.keys())
    bm25 = ScopedBM25(events)
    bscores = bm25.score(case["query_text"], eids)

    if arm == "C0_bm25_flat":
        ranked = sorted(eids, key=lambda x: (-bscores[x], x))
        return ranked[:top_k], "OK"

    if arm == "C1_scalemas_bounded":
        # Upstream-faithful bounded unsigned citation boost: at most 1 citation contribution per citing src_id
        citing_pairs: Dict[str, Dict[str, int]] = defaultdict(dict)
        for e in edges:
            citing_pairs[e.dst_id][e.src_id] = max(citing_pairs[e.dst_id].get(e.src_id, 0), e.transaction_time)
        max_b = max(bscores.values()) if bscores else 1.0
        scores = {}
        for eid in eids:
            rel = bscores[eid] / max(max_b, 1e-9)
            w_cite = sum(0.90 ** max(0, 15 - tt) for tt in citing_pairs[eid].values())
            # Bounded log1p saturation per citing message
            scores[eid] = rel + 0.5 * math.log1p(w_cite)
        return sorted(eids, key=lambda x: (-scores[x], x))[:top_k], "OK"

    if arm == "C2_latest_timestamp":
        # Pure newest-timestamp / strong recency baseline (exposes timestamp shortcuts!)
        ranked = sorted(eids, key=lambda x: (-events[x].transaction_time, -bscores[x], x))
        return ranked[:top_k], "OK"

    if arm == "C2b_bm25_jaccard_dedup":
        # Astra 6.2's exact ordinary dedup baseline (BM25 + Jaccard >= 0.58 suppression)
        tok_sets = {eid: _content_tokens(events[eid].text) for eid in eids}
        order = sorted(eids, key=lambda x: (-bscores[x], x))
        sel: List[str] = []
        for eid in order:
            if any(_jaccard(tok_sets[eid], tok_sets[s]) >= 0.58 for s in sel):
                continue
            sel.append(eid)
            if len(sel) >= top_k:
                break
        return sel, "OK"

    if arm == "C3_ordinary_bitemporal_tms":
        elig = compute_full_recompute_eligibility(events, edges, valid_cutoff=20, tx_cutoff=20, horizon=8)
        admitted = [eid for eid in eids if elig.get(eid) == "ADMIT"]
        ranked = sorted(admitted, key=lambda x: (-(bscores[x] + 0.05 * events[x].transaction_time), x))
        return ranked[:top_k], "OK"

    if arm == "C4_v2_iq04_ek5_wfs":
        # UNIFIED END-TO-END EK5 v2 (Persistent IncrementalEvidenceStore 3-Valued WFS) + Observable Echo Collapse
        # ZERO role tags read!
        store = IncrementalEvidenceStore()
        # Stream events/edges chronologically into IncrementalEvidenceStore
        for eid in sorted(eids, key=lambda x: (events[x].transaction_time, x)):
            store.apply_mutation([eid], [])
        for e in sorted(edges, key=lambda x: x.transaction_time):
            store.apply_mutation([], [(e.src_id, e.dst_id, e.edge_type)])

        elig = store.state
        admitted_graph = [eid for eid in eids if elig.get(eid) == "ADMIT"]
        contested_graph = [eid for eid in eids if elig.get(eid) == "CONTESTED"]

        # If ALL competing claims are CONTESTED and none ADMIT -> honest epistemic abstention!
        if not admitted_graph and contested_graph:
            return [], "CONTESTED_ABSTAIN"

        # Observable retracted values from active ADMIT -> AUDIT_ONLY REFUTE/SUPERSEDE edges
        retracted_vals_min_vt: Dict[str, int] = {}
        retracted_tx: Dict[str, int] = {}
        for e in edges:
            if e.edge_type in ("REFUTE", "SUPERSEDE") and elig.get(e.src_id) == "ADMIT" and elig.get(e.dst_id) == "AUDIT_ONLY":
                rv = events[e.dst_id].claim_value
                retracted_tx[rv] = max(retracted_tx.get(rv, -1), e.transaction_time)
                retracted_vals_min_vt[rv] = min(retracted_vals_min_vt.get(rv, 10**9), events[e.dst_id].valid_time)

        # Suppress unlinked echoes of actively retracted values (including late-arriving stale echoes whose
        # valid_time precedes the retraction or whose text echoes the retracted value without superseding it!)
        active_admit: List[str] = []
        for eid in admitted_graph:
            ev = events[eid]
            if ev.claim_value in retracted_tx:
                # Was this event itself the active refuter? (No, refuter asserts new value; if same value & valid_time < t_ret -> stale echo!)
                t_ret = retracted_tx[ev.claim_value]
                if ev.valid_time <= t_ret:
                    continue
            active_admit.append(eid)

        if not active_admit:
            return [], "CONTESTED_ABSTAIN"

        # Independent corroboration weighting (linked SUPPORT from distinct sources) + echo-cluster collapse
        sup_sources: Dict[str, set[str]] = defaultdict(set)
        for e in edges:
            if e.edge_type == "SUPPORT" and elig.get(e.src_id) == "ADMIT":
                sup_sources[e.dst_id].add(events[e.src_id].source_id)

        # Rank admitted items by (active-refuter authority + independent support + BM25), NOT naive newest-wins!
        active_refuters = {e.src_id for e in edges if e.edge_type in ("REFUTE", "SUPERSEDE") and elig.get(e.src_id) == "ADMIT"}
        val_sup_cnt: Counter[str] = Counter()
        for eid in active_admit:
            val_sup_cnt[events[eid].claim_value] += 1 + len(sup_sources[eid])

        def _score_ev(eid: str) -> float:
            ev = events[eid]
            ref_bonus = 3.0 if eid in active_refuters else 0.0
            indep_sup = 0.8 * math.log1p(len(sup_sources[eid]))
            # Slight preference for values backed by multiple independent records over isolated uncorroborated noise
            val_bonus = 0.4 * math.log1p(val_sup_cnt[ev.claim_value])
            return bscores[eid] + ref_bonus + indep_sup + val_bonus

        tok_sets = {eid: _content_tokens(events[eid].text) for eid in active_admit}
        order = sorted(active_admit, key=lambda x: (-_score_ev(x), x))
        sel: List[str] = []
        for eid in order:
            if any(_jaccard(tok_sets[eid], tok_sets[s]) >= 0.58 for s in sel):
                continue
            sel.append(eid)
            if len(sel) >= top_k:
                break
        return sel, "OK"

    raise ValueError(f"Unknown arm: {arm}")


def run_part_a_zero_shortcut(
    endpoint: str,
    model_id: str,
    out_dir: Path,
    max_workers: int = 16,
) -> Dict[str, Any]:
    cases = build_zero_shortcut_cases(n_per_family=30)
    arms = [
        "C0_bm25_flat",
        "C1_scalemas_bounded",
        "C2_latest_timestamp",
        "C2b_bm25_jaccard_dedup",
        "C3_ordinary_bitemporal_tms",
        "C4_v2_iq04_ek5_wfs",
    ]
    sys_p = (
        "Answer the factual query strictly using ONLY the dated records below. "
        "Reply with ONLY the exact current value (e.g., '65psi' or '25ml'). "
        "If no records are provided or the records are mutually contradictory with no resolution, reply ONLY 'ABSTAIN'."
    )

    tasks = []
    for case in cases:
        for arm in arms:
            sel_ids, status_flag = retrieve_zero_shortcut_arm(arm, case, top_k=5)
            if status_flag == "CONTESTED_ABSTAIN" or not sel_ids:
                ctx = "(No admitted records — conflicting claims are epistemically CONTESTED.)"
            else:
                ordered = sorted(sel_ids, key=lambda eid: case["events"][eid].transaction_time)
                ctx = "\n".join(
                    f"- [vt={case['events'][eid].valid_time}, tx={case['events'][eid].transaction_time}] {case['events'][eid].text}"
                    for eid in ordered
                )
            tasks.append(
                {
                    "case": case,
                    "arm": arm,
                    "status_flag": status_flag,
                    "n_ctx": len(sel_ids),
                    "prompt": f"Records:\n{ctx}\n\nQuestion: {case['query_text']}\nAnswer:",
                }
            )

    print(f"[F2-ZS] Executing {len(tasks)} live calls on {model_id} (ZERO role tags)...", flush=True)

    def _run_one(t: Dict[str, Any]) -> Dict[str, Any]:
        case = t["case"]
        pred, pt, lat, err = _call_vllm(endpoint, model_id, sys_p, t["prompt"], max_tokens=32)
        p_lo = pred.lower()
        gold = case["gold"]
        if gold == "ABSTAIN":
            is_ok = ("abstain" in p_lo) or ("cannot" in p_lo) or ("contradict" in p_lo) or ("no admitted" in p_lo)
        else:
            has_gold = gold.lower() in p_lo
            has_stale = any(sv.lower() in p_lo for sv in case["stale_vals"])
            # Strict grading: must contain gold AND NOT contradict by also emitting stale value!
            is_ok = has_gold and (not has_stale)
        return {
            "case_id": case["case_id"],
            "family": case["family"],
            "arm": t["arm"],
            "gold": gold,
            "pred_raw": pred.replace("\n", " "),
            "is_correct": int(is_ok),
            "prompt_tokens": pt,
            "latency_ms": round(lat, 2),
            "api_error": int(err),
        }

    rows: List[Dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
        for r in pool.map(_run_one, tasks):
            rows.append(r)

    csv_path = out_dir / "f2_zero_shortcut_live_rows.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    by_arm = defaultdict(list)
    by_fam_arm = defaultdict(list)
    for r in rows:
        by_arm[r["arm"]].append(r)
        by_fam_arm[(r["family"], r["arm"])].append(r["is_correct"])

    arm_stats = {}
    for arm in arms:
        rs = by_arm[arm]
        arm_stats[arm] = {
            "overall_acc": round(float(np.mean([r["is_correct"] for r in rs])), 4),
            "mean_prompt_tokens": round(float(np.mean([r["prompt_tokens"] for r in rs])), 1),
            "mean_latency_ms": round(float(np.mean([r["latency_ms"] for r in rs])), 1),
            "api_errors": int(sum(r["api_error"] for r in rs)),
            "by_family": {
                fam: round(float(np.mean(by_fam_arm[(fam, arm)])), 4)
                for fam in (
                    "stale_newest_echo_after_correction",
                    "repeated_valid_truth_vs_late_false_rumor",
                    "mutual_refutation_contested_abstain",
                    "flood_12_unlinked_paraphrase_copies",
                    "unlinked_downstream_synthesis_from_retracted",
                )
            },
        }

    def _id_map(a: str) -> Dict[str, float]:
        return {r["case_id"]: float(r["is_correct"]) for r in by_arm[a]}

    c4m = _id_map("C4_v2_iq04_ek5_wfs")
    boot = {}
    for ref in arms[:-1]:
        bt = paired_bootstrap_delta(c4m, _id_map(ref), n_bootstrap=3000, seed=42)
        boot[f"C4v2_vs_{ref}"] = {"mean_delta": round(bt["mean_delta"], 4), "lcb_95": round(bt["lcb_95"], 4)}

    return {
        "n_cases": len(cases),
        "n_answer_calls": len(rows),
        "arms": arm_stats,
        "bootstrap": boot,
        "csv_sha256": _sha256_file(csv_path),
    }


# ---------------------------------------------------------------------------
# PART B: Fresh Held-Out External Benchmark — ACL 2024 LoCoMo (Percena/locomo-mc10)
# ---------------------------------------------------------------------------
class TurnBM25:
    def __init__(self, texts: Sequence[str], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1, self.b = k1, b
        self.tfs, self.dls = [], []
        df: Counter[str] = Counter()
        for txt in texts:
            toks = _TOK_RE.findall(txt.lower())
            c = Counter(toks)
            self.tfs.append(c)
            self.dls.append(len(toks))
            for tok in c:
                df[tok] += 1
        n = max(len(texts), 1)
        self.avgdl = sum(self.dls) / float(n)
        self.idf = {tok: math.log(1.0 + (n - d + 0.5) / (d + 0.5)) for tok, d in df.items()}

    def score(self, q: str) -> np.ndarray:
        qt = set(_TOK_RE.findall(q.lower()))
        out = np.zeros(len(self.tfs), dtype=np.float64)
        for i, c in enumerate(self.tfs):
            dl = self.dls[i]
            s = 0.0
            for tok in qt:
                tf = c.get(tok, 0)
                if tf > 0:
                    denom = tf + self.k1 * (1.0 - self.b + self.b * (dl / max(self.avgdl, 1e-9)))
                    s += self.idf.get(tok, 0.0) * ((tf * (self.k1 + 1.0)) / max(denom, 1e-9))
            out[i] = s
        return out


def _minmax(a: np.ndarray) -> np.ndarray:
    lo, hi = float(a.min()), float(a.max())
    return np.zeros_like(a) if hi - lo <= 1e-12 else (a - lo) / (hi - lo)


def _extract_locomo_units(conv_obj: Dict[str, Any]) -> List[Dict[str, Any]]:
    conv = conv_obj["conversation"]
    units: List[Dict[str, Any]] = []
    sess_nums = sorted(
        int(k.split("_")[1])
        for k in conv
        if k.startswith("session_") and k.count("_") == 1 and k.split("_")[1].isdigit()
    )
    for s_num in sess_nums:
        dt_str = str(conv.get(f"session_{s_num}_date_time", f"Session {s_num}"))
        turns = conv.get(f"session_{s_num}", [])
        for t_i, t in enumerate(turns):
            spk = t.get("speaker", "Speaker")
            txt = str(t.get("text", "")).strip()
            if not txt:
                continue
            dia_id = str(t.get("dia_id", f"D{s_num}:{t_i}"))
            units.append(
                {
                    "unit_id": dia_id,
                    "session_id": f"session_{s_num}",
                    "sess_num": s_num,
                    "date_str": dt_str,
                    "ts": float(s_num * 1000 + t_i),
                    "text": f"{spk}: {txt}",
                    "raw_text": txt,
                }
            )
    return units


def retrieve_locomo_arm(
    arm: str,
    units: Sequence[Dict[str, Any]],
    u_embs: np.ndarray,
    q_emb: np.ndarray,
    bm25_scores: np.ndarray,
    *,
    top_k: int = 5,
) -> Tuple[List[int], Dict[int, str]]:
    """Return `(selected_indices, optional_tag_by_index)`."""
    n = len(units)
    dense = u_embs @ q_emb
    hyb = 0.50 * _minmax(bm25_scores) + 0.50 * _minmax(dense)
    tags: Dict[int, str] = {}

    if arm == "M0_bm25_flat":
        return np.argsort(-bm25_scores, kind="mergesort")[:top_k].tolist(), tags

    if arm == "M1_hybrid_flat":
        return np.argsort(-hyb, kind="mergesort")[:top_k].tolist(), tags

    if arm == "M1b_bm25_jaccard_dedup":
        # Astra 6.2's exact ordinary dedup baseline (BM25 top-32 + Jaccard >= 0.58)
        pool = np.argsort(-bm25_scores, kind="mergesort")[: min(32, n)].tolist()
        tsets = {i: _content_tokens(units[i]["raw_text"]) for i in pool}
        sel: List[int] = []
        for idx in pool:
            if any(_jaccard(tsets[idx], tsets[s]) >= 0.58 for s in sel):
                continue
            sel.append(idx)
            if len(sel) >= top_k:
                break
        return sel, tags

    if arm == "M2_scalemas_unsigned":
        pool = np.argsort(-hyb, kind="mergesort")[: min(40, n)]
        pe = u_embs[pool]
        sim = pe @ pe.T
        cnt = (sim >= 0.72).sum(axis=1) - 1
        sc = hyb[pool] + 0.12 * np.log1p(np.maximum(cnt, 0))
        return [int(pool[i]) for i in np.argsort(-sc, kind="mergesort")[:top_k]], tags

    if arm == "M3_recency_hybrid":
        ts = np.asarray([u["ts"] for u in units], dtype=np.float64)
        sc = hyb * (1.0 + 0.25 * _minmax(ts))
        return np.argsort(-sc, kind="mergesort")[:top_k].tolist(), tags

    if arm in ("M4c_cross_session_collapse", "M4d_cross_session_plus_tag"):
        pool = np.argsort(-hyb, kind="mergesort")[: min(32, n)].tolist()
        pe = u_embs[pool]
        sim = pe @ pe.T
        tsets = [_content_tokens(units[i]["raw_text"]) for i in pool]
        ts_list = [units[i]["ts"] for i in pool]
        t_lo, t_hi = min(ts_list), max(ts_list)
        t_span = max(t_hi - t_lo, 1.0)
        order = sorted(
            range(len(pool)),
            key=lambda p: -(float(hyb[pool[p]]) + 0.06 * ((ts_list[p] - t_lo) / t_span)),
        )
        sel_loc: List[int] = []
        suppressed: set[int] = set()
        for p in order:
            if p in suppressed:
                continue
            sel_loc.append(p)
            # STRICTLY CROSS-SESSION echo collapse (`session_id(p) != session_id(q)`) — preserves clean intra-session context!
            sp = units[pool[p]]["session_id"]
            for q in range(len(pool)):
                if q == p or q in suppressed:
                    continue
                if units[pool[q]]["session_id"] == sp:
                    continue
                if float(sim[p, q]) >= 0.72 or _jaccard(tsets[p], tsets[q]) >= 0.58:
                    suppressed.add(q)
            if len(sel_loc) >= top_k:
                break

        if arm == "M4d_cross_session_plus_tag":
            # Non-destructive cross-session supersession annotation (Crumb #2):
            # If two selected turns from DIFFERENT sessions share cos >= 0.58 and differ in tokens,
            # label the earlier one [EARLIER RECORD] and the newer one [NEWER UPDATE] without deleting either!
            for a in sel_loc:
                for b in sel_loc:
                    if a == b:
                        continue
                    if units[pool[b]]["sess_num"] > units[pool[a]]["sess_num"] and float(sim[a, b]) >= 0.58:
                        if _jaccard(tsets[a], tsets[b]) < 0.78:
                            tags[pool[a]] = "[EARLIER RECORD] "
                            tags[pool[b]] = "[NEWER UPDATE] "
        return [pool[p] for p in sel_loc], tags

    raise ValueError(f"Unknown arm: {arm}")


def run_part_b_locomo(
    endpoint: str,
    model_id: str,
    out_dir: Path,
    max_workers: int = 16,
) -> Dict[str, Any]:
    print("[F2-LOCOMO] Loading Percena/locomo-mc10 (raw/locomo10.json)...", flush=True)
    p = hf_hub_download("Percena/locomo-mc10", "raw/locomo10.json", repo_type="dataset")
    convs = json.loads(Path(p).read_text(encoding="utf-8"))

    # Select 120 strictly stratified held-out questions across all 10 conversations:
    # 40 Temporal (cat 2), 40 Single/Multi-Hop (cat 1/4), 40 Adversarial Unanswerable/Abstention (cat 5)
    buckets: Dict[str, List[Dict[str, Any]]] = {"temporal": [], "factual": [], "abstention": []}
    for ci, conv in enumerate(convs):
        for qi, qa in enumerate(conv["qa"]):
            cat = int(qa["category"])
            item = {
                "qid": f"loc_{ci}_{qi}",
                "conv_idx": ci,
                "category": cat,
                "question": str(qa["question"]),
                "gold_answer": "UNANSWERABLE" if cat == 5 else str(qa.get("answer", "")),
                "is_abstention": (cat == 5),
                "evidence_ids": [str(e) for e in qa.get("evidence", [])],
            }
            if cat == 2:
                buckets["temporal"].append(item)
            elif cat in (1, 4):
                buckets["factual"].append(item)
            elif cat == 5:
                buckets["abstention"].append(item)

    selected = buckets["temporal"][:40] + buckets["factual"][:40] + buckets["abstention"][:40]
    print(f"[F2-LOCOMO] Selected {len(selected)} held-out LoCoMo questions (40 temporal, 40 factual, 40 abstention)", flush=True)

    embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cuda")
    arms = [
        "M0_bm25_flat",
        "M1_hybrid_flat",
        "M1b_bm25_jaccard_dedup",
        "M2_scalemas_unsigned",
        "M3_recency_hybrid",
        "M4c_cross_session_collapse",
        "M4d_cross_session_plus_tag",
    ]
    ans_sys = (
        "Answer the question strictly using ONLY the dated conversation snippets below. "
        "Be concise (1 short sentence). If the snippets do NOT contain enough information to answer the question, "
        "or the asked event/detail never occurred in the snippets, reply ONLY: UNANSWERABLE."
    )

    # Strictly deterministic cached abstention-aware grader!
    grade_cache: Dict[Tuple[str, str], bool] = {}
    grade_lock = threading.Lock()
    judge_calls_counter = [0]

    def grade_cached(qid: str, question: str, gold: str, is_abs: bool, pred: str) -> Tuple[bool, bool]:
        p_norm = re.sub(r"\s+", " ", pred.strip())
        key = (qid, p_norm.lower())
        with grade_lock:
            if key in grade_cache:
                return grade_cache[key], False
        p_lo = p_norm.lower()
        if not p_lo or p_lo.startswith("error:"):
            return False, p_lo.startswith("error:")

        abs_cues = ("unanswerable", "not mentioned", "no information", "doesn't mention", "does not mention", "cannot be determined", "don't have information", "never mentioned")
        if is_abs:
            ok = any(c in p_lo for c in abs_cues)
            with grade_lock:
                grade_cache[key] = ok
            return ok, False

        # Non-abstention item: if model abstained -> False immediately
        if any(c in p_lo for c in abs_cues) and len(p_lo) < 90:
            with grade_lock:
                grade_cache[key] = False
            return False, False

        sys_g = (
            "You are an impartial QA grader. Reply ONLY YES if the Model Answer conveys the same core fact/date/entity "
            "as the Ground-Truth Answer without contradicting it. Reply NO if it is wrong, contradicts, or gives a different value."
        )
        usr_g = f"Question: {question}\nGround-Truth Answer: {gold}\nModel Answer: {p_norm}\n\nMatch? Reply YES or NO."
        txt, _, _, err = _call_vllm(endpoint, model_id, sys_g, usr_g, max_tokens=6)
        with grade_lock:
            judge_calls_counter[0] += 1
        ok = txt.strip().upper().startswith("YES")
        if not err:
            with grade_lock:
                grade_cache[key] = ok
        return ok, err

    # Pre-extract base units per conversation + Regime B echo-injected units (6 cross-session stale distractor echoes
    # built from another turn in the same conversation OR earlier evidence distractor!)
    all_rows: List[Dict[str, Any]] = []
    regime_summaries: Dict[str, Any] = {}

    for reg_name, inject_echo in (("regime_a_unmodified", False), ("regime_b_cross_session_echo_stress", True)):
        # Build per-conversation units + embeddings once per regime
        conv_units: Dict[int, List[Dict[str, Any]]] = {}
        conv_embs: Dict[int, np.ndarray] = {}
        conv_bm25: Dict[int, TurnBM25] = {}
        for ci, conv in enumerate(convs):
            units = _extract_locomo_units(conv)
            if inject_echo:
                # Realistic cross-session echo stress: for each non-evidence high-lexical distractor / earlier turn
                # related to the conversation's main entities, inject 6 cross-session recap echoes from earlier sessions
                # (strictly WITHOUT touching gold evidence turns, so it tests whether repeated stale/distractor echoes
                # drown out the true single-mention evidence turn in top-5!)
                q_in_conv = [it for it in selected if it["conv_idx"] == ci and not it["is_abstention"]]
                base_bm25 = TurnBM25([u["text"] for u in units])
                seen_echo_src = set()
                for it in q_in_conv:
                    ev_set = set(it["evidence_ids"])
                    sc = base_bm25.score(it["question"])
                    # Pick top lexical NON-evidence distractor turn in the conversation
                    cand_dist = [i for i in np.argsort(-sc)[:12] if units[i]["unit_id"] not in ev_set]
                    if not cand_dist:
                        continue
                    d_idx = cand_dist[0]
                    if d_idx in seen_echo_src:
                        continue
                    seen_echo_src.add(d_idx)
                    src_u = units[d_idx]
                    for e_i in range(6):
                        units.append(
                            {
                                "unit_id": f"echo_{ci}_{d_idx}_{e_i}",
                                "session_id": f"echo_recap_sess_{e_i}",
                                "sess_num": max(1, src_u["sess_num"]),
                                "date_str": src_u["date_str"],
                                "ts": src_u["ts"] + 0.01 * (e_i + 1),
                                "text": f"Memory recap {e_i + 1}: {src_u['text']}",
                                "raw_text": f"Memory recap {e_i + 1}: {src_u['raw_text']}",
                            }
                        )
            conv_units[ci] = units
            conv_embs[ci] = embedder.encode(
                [u["text"] for u in units], batch_size=256, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False
            ).astype(np.float32)
            conv_bm25[ci] = TurnBM25([u["text"] for u in units])

        q_embs = embedder.encode(
            [it["question"] for it in selected], batch_size=128, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False
        ).astype(np.float32)

        tasks = []
        for idx_i, it in enumerate(selected):
            ci = it["conv_idx"]
            units = conv_units[ci]
            bm_sc = conv_bm25[ci].score(it["question"])
            ev_set = set(it["evidence_ids"])
            for arm in arms:
                sel_idx, tags = retrieve_locomo_arm(arm, units, conv_embs[ci], q_embs[idx_i], bm_sc, top_k=5)
                sel_sorted = sorted(sel_idx, key=lambda k: units[k]["ts"])
                ctx = "\n".join(f"- [{units[k]['date_str']}] {tags.get(k, '')}{units[k]['text']}" for k in sel_sorted)
                has_ev = (not it["is_abstention"]) and any(units[k]["unit_id"] in ev_set for k in sel_idx)
                tasks.append(
                    {
                        "regime": reg_name,
                        "arm": arm,
                        "item": it,
                        "has_ev": int(has_ev),
                        "prompt": f"Conversation Snippets:\n{ctx}\n\nQuestion: {it['question']}\nAnswer:",
                    }
                )

        print(f"[F2-LOCOMO] Executing {len(tasks)} live answer calls for {reg_name}...", flush=True)

        def _exec_loc(t: Dict[str, Any]) -> Dict[str, Any]:
            it = t["item"]
            pred, pt, lat, err1 = _call_vllm(endpoint, model_id, ans_sys, t["prompt"], max_tokens=96)
            ok, err2 = grade_cached(it["qid"], it["question"], it["gold_answer"], it["is_abstention"], pred)
            return {
                "regime": t["regime"],
                "arm": t["arm"],
                "qid": it["qid"],
                "category": it["category"],
                "is_abstention": int(it["is_abstention"]),
                "gold_answer": it["gold_answer"],
                "pred_raw": pred.replace("\n", " "),
                "is_correct": int(ok),
                "has_evidence_top5": t["has_ev"],
                "prompt_tokens": pt,
                "latency_ms": round(lat, 2),
                "api_error": int(err1 or err2),
            }

        reg_rows: List[Dict[str, Any]] = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
            for r in pool.map(_exec_loc, tasks):
                reg_rows.append(r)
                all_rows.append(r)

        by_arm = defaultdict(list)
        for r in reg_rows:
            by_arm[r["arm"]].append(r)
        arm_st = {}
        for arm in arms:
            rs = by_arm[arm]
            ans_rs = [r for r in rs if r["is_abstention"] == 0]
            abs_rs = [r for r in rs if r["is_abstention"] == 1]
            arm_st[arm] = {
                "answerable_acc": round(float(np.mean([r["is_correct"] for r in ans_rs])), 4),
                "answerable_ev_recall5": round(float(np.mean([r["has_evidence_top5"] for r in ans_rs])), 4),
                "abstention_acc": round(float(np.mean([r["is_correct"] for r in abs_rs])), 4),
                "overall_acc": round(float(np.mean([r["is_correct"] for r in rs])), 4),
                "mean_prompt_tokens": round(float(np.mean([r["prompt_tokens"] for r in rs])), 1),
                "mean_latency_ms": round(float(np.mean([r["latency_ms"] for r in rs])), 1),
                "api_errors": int(sum(r["api_error"] for r in rs)),
            }

        def _ans_map(a: str) -> Dict[str, float]:
            return {r["qid"]: float(r["is_correct"]) for r in by_arm[a] if r["is_abstention"] == 0}

        boot = {}
        for cand_l, cand_k in (("M4c", "M4c_cross_session_collapse"), ("M4d", "M4d_cross_session_plus_tag")):
            cm = _ans_map(cand_k)
            for ref in ("M0_bm25_flat", "M1_hybrid_flat", "M1b_bm25_jaccard_dedup", "M2_scalemas_unsigned", "M3_recency_hybrid"):
                bt = paired_bootstrap_delta(cm, _ans_map(ref), n_bootstrap=3000, seed=42)
                boot[f"{cand_l}_vs_{ref}"] = {"mean_delta": round(bt["mean_delta"], 4), "lcb_95": round(bt["lcb_95"], 4)}

        regime_summaries[reg_name] = {"arms": arm_st, "answerable_bootstrap": boot}

    csv_path = out_dir / "f2_locomo_live_rows.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(all_rows[0].keys()))
        w.writeheader()
        w.writerows(all_rows)

    return {
        "n_questions": len(selected),
        "n_answer_calls": len(all_rows),
        "n_judge_calls": judge_calls_counter[0],
        "regimes": regime_summaries,
        "csv_sha256": _sha256_file(csv_path),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--api-base", type=str, default="http://127.0.0.1:8890/v1")
    ap.add_argument("--model-id", type=str, default="nemotron-lightning")
    ap.add_argument("--max-workers", type=int, default=16)
    args = ap.parse_args()

    out_dir = args.output_dir
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"Output directory must be absent or empty: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()

    part_a = run_part_a_zero_shortcut(args.api_base, args.model_id, out_dir, max_workers=args.max_workers)
    part_b = run_part_b_locomo(args.api_base, args.model_id, out_dir, max_workers=args.max_workers)

    md = [
        "# Flagship #2 Unified Zero-Shortcut (`EK5 v2` WFS) + Fresh Held-Out ACL 2024 LoCoMo Verdict",
        "",
        f"- **Git SHA:** `{current_git_sha(REPO_ROOT)}`",
        f"- **Model:** `{args.model_id}` @ `{args.api_base}`",
        f"- **Exact Call Accounting:** Part A `{part_a['n_answer_calls']}` answer calls (`0` judge calls); Part B `{part_b['n_answer_calls']}` answer calls + `{part_b['n_judge_calls']}` cached deterministic judge calls",
        f"- **Wall time:** `{time.perf_counter() - t0:.2f}s`",
        "",
        "## Part A — Zero-Timestamp-Shortcut Unified `EK5 v2` WFS Live Suite (ZERO Role Tags)",
        "",
        "| Arm | Overall Acc ↑ | Stale-Newest After Corr ↑ | Repeated Truth vs Late Rumor ↑ | Mutual Refute Abstain ↑ | 12-Copy Flood ↑ | Unlinked Synth ↑ | Prompt Toks |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for arm, st in part_a["arms"].items():
        bf = st["by_family"]
        b = "**" if arm == "C4_v2_iq04_ek5_wfs" else ""
        md.append(
            f"| {b}`{arm}`{b} | {b}{st['overall_acc']:.4f}{b} | {bf['stale_newest_echo_after_correction']:.4f} | "
            f"{bf['repeated_valid_truth_vs_late_false_rumor']:.4f} | {bf['mutual_refutation_contested_abstain']:.4f} | "
            f"{bf['flood_12_unlinked_paraphrase_copies']:.4f} | {bf['unlinked_downstream_synthesis_from_retracted']:.4f} | {st['mean_prompt_tokens']:.1f} |"
        )
    md.append("")
    md.append("**Part A Paired Bootstrap (`n=150`, two-sided 95% CI lower bound `lcb_95`):**")
    for k, v in part_a["bootstrap"].items():
        md.append(f"- `{k}`: `Δ = {v['mean_delta']:+.4f}` (`LCB95 = {v['lcb_95']:+.4f}`)")

    for rname, title in (
        ("regime_a_unmodified", "## Part B.1 — Fresh Held-Out LoCoMo Regime A (Unmodified, 80 Answerable + 40 Abstention)"),
        ("regime_b_cross_session_echo_stress", "## Part B.2 — Fresh Held-Out LoCoMo Regime B (Cross-Session Echo Stress)"),
    ):
        rs = part_b["regimes"][rname]
        md.extend(
            [
                "",
                title,
                "",
                "| Arm | Answerable Acc ↑ | Answerable Ev Recall@5 ↑ | Abstention Acc ↑ | Overall Acc ↑ | Prompt Toks | API Errs |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for arm, st in rs["arms"].items():
            b = "**" if arm in ("M4c_cross_session_collapse", "M4d_cross_session_plus_tag") else ""
            md.append(
                f"| {b}`{arm}`{b} | {b}{st['answerable_acc']:.4f}{b} | {st['answerable_ev_recall5']:.4f} | "
                f"{st['abstention_acc']:.4f} | {b}{st['overall_acc']:.4f}{b} | {st['mean_prompt_tokens']:.1f} | {st['api_errors']} |"
            )
        md.append("")
        md.append("**Answerable-Only Paired Bootstrap (`n=80`):**")
        for k, v in rs["answerable_bootstrap"].items():
            md.append(f"- `{k}`: `Δ = {v['mean_delta']:+.4f}` (`LCB95 = {v['lcb_95']:+.4f}`)")

    summary_path = out_dir / "f2_unified_zero_shortcut_locomo_verdict.md"
    summary_path.write_text("\n".join(md) + "\n", encoding="utf-8")

    art_shas = {
        "f2_zero_shortcut_live_rows.csv": part_a["csv_sha256"],
        "f2_locomo_live_rows.csv": part_b["csv_sha256"],
        summary_path.name: _sha256_file(summary_path),
    }
    manifest = {
        "job": "flagship2_unified_zero_shortcut_and_locomo",
        "git_sha": current_git_sha(REPO_ROOT),
        "part_a_zero_shortcut": part_a,
        "part_b_locomo": part_b,
        "artifact_sha256": art_shas,
        "wall_seconds": round(time.perf_counter() - t0, 2),
        "telemetry": process_telemetry(),
    }
    m_sha = write_json_atomic(out_dir / "f2_unified_manifest.json", manifest)
    write_json_atomic(
        out_dir / "report_receipt.json",
        {"job": manifest["job"], "status": "COMPLETED", "manifest_sha256": m_sha, "artifact_sha256": art_shas},
    )
    print(f"[F2-UNIFIED] COMPLETE in {manifest['wall_seconds']}s -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
