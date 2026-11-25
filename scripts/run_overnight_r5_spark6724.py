r"""Overnight R5 Master Validation Worker 1 (`gx10-6724`): `PM-1a`, `PM-1b` (`EM-4` + Full `LongMemEval-S N=500`), & `PM-4` (`2x2` Cross-Family + Blind Human Audit Packet).

Strictly ADDITIVE & NON-DESTRUCTIVE:
  - Writes ONLY to `--output-root` (default `/home/mattmre/chelated-night1/out/20261122-overnight-r5/spark6724`).
  - Never modifies R3/R4 artifacts or code.
  - Emits independent atomic manifests + 100%-SHA-256 `report_receipt.json` after EVERY stage:
      * Stage 1 (`r5_s1_zero_shortcut_v3_flooded`): `N=240 = 8x30` realistic `> top_k=5` density (`16` records/entity) closing `PM-1a`.
      * Stage 2 (`r5_s2_locomo_300_em4`): Balanced 10-conv `LoCoMo-300` across 3 regimes testing Carbon's `M4f` & `M4g` (`EM-4`) alongside `M0..M4e`.
      * Stage 3 (`r5_s3_longmemeval_s_full_n500`): ALL `500` unmodified ICLR 2025 `LongMemEval-S` questions (`70` `single-session-user` separately) + `78` `knowledge-update` 6x verbatim+paraphrased echo stress (`PM-1b`).
      * Stage 4 (`r5_s4_pm4_cross_family_2x2_and_audit`): Sealed blind `100`-item human audit packet + live `2x2` `{Nemotron-3.5-30B, Qwen2.5-7B-Instruct} x {Nemotron Judge, Qwen2.5 Judge}` Cohen's kappa (`PM-4`).
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
import json
import math
import random
import re
import sys
import time
import traceback
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Sequence, Set, Tuple

import numpy as np
import requests
from huggingface_hub import hf_hub_download
from sentence_transformers import SentenceTransformer

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ek5_v3_stratified import (  # noqa: E402
    EdgeAttestation,
    StratifiedEvent,
    classify_high_similarity_pair,
)
from em4_soft_telemetry_routing import route_soft_telemetry_top_k  # noqa: E402
from iq04_temporal_evidence import (  # noqa: E402
    EvidenceEdge,
    EvidenceEvent,
)
from scripts.night1_common import (  # noqa: E402
    current_git_sha,
    paired_bootstrap_delta,
    process_telemetry,
    write_json_atomic,
)

_TOK_RE = re.compile(r"[a-z0-9]+")
_EVID_SPLIT_RE = re.compile(r"[;,\s]+")
_STOPWORDS = frozenset({
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "to", "of", "and", "in", "that",
    "it", "for", "on", "with", "as", "at", "by", "this", "from", "or", "but", "not", "my", "i",
    "me", "you", "your", "we", "they", "he", "she", "his", "her", "have", "has", "had", "do",
    "did", "does", "can", "could", "would", "should", "just", "so", "very", "also", "about",
    "what", "when", "where", "who", "how", "which", "been", "being", "into", "out", "up",
})

_DOMAINS = (
    ("pump_station", "target_psi", ("40psi", "65psi", "90psi", "115psi")),
    ("serum_protocol", "dose_ml", ("10ml", "25ml", "50ml", "75ml")),
    ("relay_bus", "cutoff_hz", ("50hz", "60hz", "120hz", "240hz")),
    ("orbital_burn", "delta_v", ("140ms", "220ms", "310ms", "450ms")),
    ("customs_tariff", "rate_pct", ("4pct", "9pct", "15pct", "22pct")),
)
_ECHO_WRAPPERS = (
    "Current confirmed {slot} for {ent} is logged as {val} in regional digest {tag}.",
    "Archived shift briefing confirms {ent} {slot} equals {val} ({tag}).",
    "Field status bulletin lists current {slot} for {ent} at {val} ({tag}).",
    "Secondary ops memo repeats {ent} {slot} = {val} under ref {tag}.",
    "Cross-team summary records {val} as {ent} {slot} (packet {tag}).",
    "Forwarded status entry notes {ent} {slot} set to {val}, item {tag}.",
)
_LME_ECHO_PREFIXES = (
    "Session recap note from earlier chat: user mentioned — ",
    "Memory summary log (auto-archived): ",
    "Background context carried forward: ",
    "Follow-up note recalling earlier detail: ",
    "Periodic assistant digest citing prior session: ",
    "Stored profile reminder from previous conversation: ",
)


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _content_tokens(text: str) -> Set[str]:
    return {t for t in _TOK_RE.findall(text.lower()) if t not in _STOPWORDS and len(t) > 1}


def _jaccard(a: Set[str], b: Set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / float(len(a | b))


def _minmax(a: np.ndarray) -> np.ndarray:
    lo, hi = float(a.min()), float(a.max())
    return np.zeros_like(a) if hi - lo <= 1e-12 else (a - lo) / (hi - lo)


def _call_vllm(
    endpoint: str,
    model_id: str,
    sys_p: str,
    usr_p: str,
    *,
    max_tokens: int = 80,
    timeout: float = 60.0,
) -> Tuple[str, int, float, bool]:
    t0 = time.perf_counter()
    for attempt in range(3):
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
                timeout=timeout,
            )
            r.raise_for_status()
            d = r.json()
            txt = (d["choices"][0]["message"].get("content") or "").strip()
            pt = int(d.get("usage", {}).get("prompt_tokens", 0))
            return txt, pt, (time.perf_counter() - t0) * 1000.0, False
        except Exception as exc:  # noqa: BLE001
            if attempt == 2:
                return f"ERROR: {exc}", 0, (time.perf_counter() - t0) * 1000.0, True
            time.sleep(0.5 * (attempt + 1))
    return "ERROR: unreachable", 0, 0.0, True


class TurnBM25:
    def __init__(self, texts: Sequence[str], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1, self.b = k1, b
        self.tfs: List[Counter[str]] = []
        self.dls: List[int] = []
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


# ===========================================================================
# STAGE 1: ZERO-SHORTCUT v3 (N=240 = 8x30, REALISTIC > top_k=5 DENSITY = 15-17 RECORDS/CASE)
# ===========================================================================
def build_zero_shortcut_v3_flooded_cases(n_per_family: int = 30) -> List[Dict[str, Any]]:
    families = (
        "stale_newest_echo_after_correction",
        "repeated_valid_truth_vs_late_false_rumor",
        "mutual_refutation_contested_abstain",
        "flood_12_unlinked_paraphrase_copies",
        "unlinked_downstream_synthesis_from_retracted",
        "authority_laundering_rumor_attack",
        "version_lineage_refuted_middle_v3_v2_v1",
        "amends_vs_copy_of_currency_split",
    )
    cases: List[Dict[str, Any]] = []
    idx = 0
    for fam in families:
        for k in range(n_per_family):
            dom, slot, vals = _DOMAINS[(idx + k) % len(_DOMAINS)]
            ent = f"{dom}_zs3_{idx:03d}"
            v_old, v_mid, v_new, v_anc = vals[idx % 4], vals[(idx + 1) % 4], vals[(idx + 2) % 4], vals[(idx + 3) % 4]
            events: Dict[str, StratifiedEvent] = {}
            legacy_events: Dict[str, EvidenceEvent] = {}
            attestations: List[EdgeAttestation] = []
            legacy_edges: List[EvidenceEdge] = []

            def add_e(eid: str, val: str, txt: str, vt: int, tt: int, auth: int = 2, src: str = "s") -> None:
                events[eid] = StratifiedEvent(
                    event_id=eid, entity=ent, slot=slot, claim_value=val, text=txt,
                    valid_time=vt, tx_time=tt, tenant="t0", domain=dom, authority_tier=auth,
                )
                legacy_events[eid] = EvidenceEvent(eid, ent, ent, slot, val, txt, vt, tt, src, role="EVENT")

            def add_a(src: str, dst: str, et: str, tt: int, lin: str | None = None) -> None:
                aid = f"att_{idx}_{len(attestations)}"
                attestations.append(EdgeAttestation(aid, src, dst, et, valid_time=tt, tx_time=tt, lineage_id=lin))
                leg_et = "SUPERSEDE" if et == "VERSION_LINEAGE_SUPERSEDE" else ("DERIVES_FROM" if et in ("DERIVES_CURRENT", "AMENDS") else et)
                if leg_et in ("SUPPORT", "REFUTE", "SUPERSEDE", "DERIVES_FROM", "COPY_OF"):
                    legacy_edges.append(EvidenceEdge(src, dst, leg_et, tt))

            # Every case starts with a superseded legacy epoch (v_anc) + 3 historical mirrors so store size >= 15 (> top_k=5)
            add_e("hist_0", v_anc, f"Bootstrap commissioning record for {ent} {slot} logged {v_anc}.", 0, 0, 2, "sh0")
            for h_i in range(3):
                add_e(f"hist_m{h_i}", v_anc, f"Commissioning archive mirror {h_i} notes {ent} {slot} was {v_anc}.", 0, 1, 2, f"shm{h_i}")
                add_a(f"hist_m{h_i}", "hist_0", "COPY_OF", 1)

            if fam == "stale_newest_echo_after_correction":
                add_e("r_bad", v_old, f"Shift 1 telemetry logged {ent} {slot} as {v_old}.", 2, 2, 2, "s0")
                add_a("r_bad", "hist_0", "SUPERSEDE", 2)
                for i in range(5):
                    add_e(f"echo_{i}", v_old, _ECHO_WRAPPERS[i % len(_ECHO_WRAPPERS)].format(ent=ent, slot=slot, val=v_old, tag=f"E{i}"), 3 + i, 3 + i, 2, f"se{i}")
                add_e("corr", v_new, f"Verified engineering audit: {ent} {slot} operative setting is {v_new}.", 10, 10, 3, "sa")
                add_a("corr", "r_bad", "REFUTE", 10)
                add_a("corr", "r_bad", "SUPERSEDE", 10)
                # 6 delayed low-tier forwarded echoes arriving at tx=12..17 (valid_time=3..5 <= t_ret=10!)
                for j in range(6):
                    add_e(
                        f"late_echo_{j}", v_old,
                        f"Delayed packet {j}: current confirmed {slot} for {ent} repeated as {v_old} from earlier shift.",
                        3 + (j % 3), 12 + j, 1, f"sl{j}",
                    )
                gold, stale = v_new, (v_old, v_anc)

            elif fam == "repeated_valid_truth_vs_late_false_rumor":
                add_e("v_root", v_new, f"Certified baseline establishes {ent} {slot} at {v_new}.", 2, 2, 3, "sc")
                add_a("v_root", "hist_0", "SUPERSEDE", 2)
                for i in range(5):
                    add_e(f"sup_{i}", v_new, f"Independent inspection {i} verifies {ent} {slot} remains {v_new}.", 3 + i, 3 + i, 3, f"si{i}")
                    add_a(f"sup_{i}", "v_root", "SUPPORT", 3 + i)
                # 6 late unverified tier-1 rumor relays at tx=12..17 attempting to refute tier-3 baseline
                for j in range(6):
                    add_e(
                        f"late_rumor_{j}", v_old,
                        f"Unverified external chatter {j} claims current confirmed {slot} for {ent} is {v_old}.",
                        12 + j, 12 + j, 1, f"sr{j}",
                    )
                    add_a(f"late_rumor_{j}", "v_root", "REFUTE", 12 + j)
                gold, stale = v_new, (v_old, v_anc)

            elif fam == "mutual_refutation_contested_abstain":
                add_e("rep_a", v_old, f"Sensor Channel Alpha states current confirmed {slot} for {ent} is {v_old}.", 5, 5, 3, "sa")
                add_e("rep_b", v_new, f"Sensor Channel Beta states current confirmed {slot} for {ent} is {v_new}.", 6, 6, 3, "sb")
                add_a("rep_a", "hist_0", "SUPERSEDE", 5)
                add_a("rep_b", "hist_0", "SUPERSEDE", 6)
                add_a("rep_a", "rep_b", "REFUTE", 7)
                add_a("rep_b", "rep_a", "REFUTE", 7)
                for j in range(5):
                    add_e(f"ca_{j}", v_old, f"Relay mirror A{j} repeats current confirmed {slot} for {ent} is {v_old}.", 8 + j, 8 + j, 3, f"mca{j}")
                    add_a(f"ca_{j}", "rep_a", "COPY_OF", 8 + j)
                    add_e(f"cb_{j}", v_new, f"Relay mirror B{j} repeats current confirmed {slot} for {ent} is {v_new}.", 8 + j, 13 + j, 3, f"mcb{j}")
                    add_a(f"cb_{j}", "rep_b", "COPY_OF", 13 + j)
                gold, stale = "ABSTAIN", (v_old, v_new, v_anc)

            elif fam == "flood_12_unlinked_paraphrase_copies":
                add_e("r_bad", v_old, f"Primary bulletin reports {ent} {slot} is {v_old}.", 2, 2, 2, "s0")
                add_a("r_bad", "hist_0", "SUPERSEDE", 2)
                for i in range(12):
                    add_e(f"copy_{i}", v_old, _ECHO_WRAPPERS[i % len(_ECHO_WRAPPERS)].format(ent=ent, slot=slot, val=v_old, tag=f"F{i}"), 3 + (i % 5), 3 + (i % 5), 2, f"sf{i}")
                add_e("corr", v_new, f"Audit override for {ent}: {slot} recalibrated to {v_new}.", 10, 10, 3, "sa")
                add_a("corr", "r_bad", "REFUTE", 10)
                add_a("corr", "r_bad", "SUPERSEDE", 10)
                gold, stale = v_new, (v_old, v_anc)

            elif fam == "unlinked_downstream_synthesis_from_retracted":
                add_e("r_bad", v_old, f"Initial sensor feed reports {ent} {slot} = {v_old}.", 2, 2, 2, "s0")
                add_a("r_bad", "hist_0", "SUPERSEDE", 2)
                for i in range(6):
                    add_e(f"syn_d_{i}", v_old, f"Derived rollup D{i}: current confirmed {slot} for {ent} computed as {v_old}.", 4 + i, 4 + i, 2, f"ssd{i}")
                    add_a(f"syn_d_{i}", "r_bad", "DERIVES_CURRENT", 4 + i)
                for i in range(5):
                    add_e(f"syn_u_{i}", v_old, f"Summary cache U{i}: current confirmed {slot} for {ent} listed at {v_old}.", 3 + i, 3 + i, 2, f"ssu{i}")
                add_e("corr", v_new, f"Root sensor recalibrated: {ent} {slot} = {v_new}.", 12, 12, 3, "sa")
                add_a("corr", "r_bad", "REFUTE", 12)
                add_a("corr", "r_bad", "SUPERSEDE", 12)
                gold, stale = v_new, (v_old, v_anc)

            elif fam == "authority_laundering_rumor_attack":
                add_e("verified", v_new, f"Certified spec sets {ent} {slot} to {v_new}.", 2, 2, 3, "sc")
                add_a("verified", "hist_0", "SUPERSEDE", 2)
                add_e("rumor", v_old, f"Anonymous tip claims {ent} {slot} changed to {v_old}.", 8, 8, 1, "sr")
                for j in range(8):
                    add_e(
                        f"laundered_{j}", v_old,
                        f"Executive digest L{j}: current confirmed {slot} for {ent} reported as {v_old} per tip.",
                        11 + j, 11 + j, 5, f"sl{j}",
                    )
                    add_a(f"laundered_{j}", "rumor", "DERIVES_CURRENT", 11 + j)
                    add_a(f"laundered_{j}", "verified", "REFUTE", 11 + j)
                    add_a(f"laundered_{j}", "verified", "SUPERSEDE", 11 + j)
                gold, stale = v_new, (v_old, v_anc)

            elif fam == "version_lineage_refuted_middle_v3_v2_v1":
                add_e("v1", v_old, f"Revision 1 sets current confirmed {slot} for {ent} to {v_old}.", 2, 2, 3, "s1")
                add_a("v1", "hist_0", "SUPERSEDE", 2)
                add_e("v2", v_mid, f"Revision 2 sets {ent} {slot} to {v_mid}.", 5, 5, 3, "s2")
                add_e("v3", v_new, f"Revision 3 sets {ent} {slot} to {v_new}.", 10, 10, 3, "s3")
                add_e("aud", "", f"Audit invalidates defective Revision 2 for {ent} {slot}.", 11, 11, 4, "sa")
                add_a("v2", "v1", "VERSION_LINEAGE_SUPERSEDE", 5, lin="L1")
                add_a("v3", "v2", "VERSION_LINEAGE_SUPERSEDE", 10, lin="L1")
                add_a("aud", "v2", "REFUTE", 11)
                for j in range(7):
                    add_e(
                        f"v1_mir_{j}", v_old,
                        f"Replica node M{j} mirrors Revision 1: current confirmed {slot} for {ent} is {v_old}.",
                        2, 12 + j, 3, f"sm{j}",
                    )
                    add_a(f"v1_mir_{j}", "v1", "COPY_OF", 12 + j)
                gold, stale = v_new, (v_old, v_mid, v_anc)

            else:  # amends_vs_copy_of_currency_split
                add_e("orig", v_old, f"Original charter sets current confirmed {slot} for {ent} at {v_old}.", 2, 2, 3, "s0")
                add_a("orig", "hist_0", "SUPERSEDE", 2)
                add_e("mid_upd", v_mid, f"Interim directive shifts {ent} {slot} to {v_mid}.", 5, 5, 3, "s2")
                add_a("mid_upd", "orig", "SUPERSEDE", 5)
                add_e("amend_node", v_new, f"Signed addendum amending charter sets {ent} {slot} to {v_new}.", 9, 9, 3, "s3")
                add_a("amend_node", "orig", "AMENDS", 9)
                add_a("amend_node", "mid_upd", "SUPERSEDE", 9)
                for j in range(8):
                    add_e(
                        f"copy_orig_{j}", v_old,
                        f"Synced cache C{j} copies original charter: current confirmed {slot} for {ent} is {v_old}.",
                        2, 11 + j, 3, f"sc{j}",
                    )
                    add_a(f"copy_orig_{j}", "orig", "COPY_OF", 11 + j)
                gold, stale = v_new, (v_old, v_mid, v_anc)

            cases.append({
                "case_id": f"zs3_{idx:03d}",
                "family": fam,
                "entity": ent,
                "slot": slot,
                "query_text": f"What is the current confirmed {slot} for {ent}?",
                "events": events,
                "legacy_events": legacy_events,
                "attestations": attestations,
                "legacy_edges": legacy_edges,
                "gold": gold,
                "stale_vals": stale,
            })
            idx += 1
    return cases


def retrieve_zs_v3_arm(arm: str, case: Dict[str, Any], *, top_k: int = 5) -> Tuple[List[str], str]:
    # Identical zero-oracle retrieval implementations from R4 across all 8 arms!
    from scripts.run_flagship2_r4_ek5_v3_and_suites import retrieve_zs_v2_arm

    return retrieve_zs_v2_arm(arm, case, top_k=top_k)


def run_stage1_zs_v3(endpoint: str, model_id: str, root_dir: Path, max_workers: int = 16) -> Dict[str, Any]:
    out_dir = root_dir / "r5_s1_zero_shortcut_v3_flooded"
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    cases = build_zero_shortcut_v3_flooded_cases(n_per_family=30)
    arms = [
        "C0_bm25_flat",
        "C1_scalemas_bounded",
        "C2_latest_timestamp",
        "C2b_bm25_jaccard_dedup",
        "C2c_astra_4line_rule",
        "C3_ordinary_bitemporal_tms",
        "C4_v2_ek5_v2",
        "C4_v3_ek5_v3_k4",
    ]
    sys_p = (
        "Answer the factual query strictly using ONLY the dated records below. "
        "Reply with ONLY the exact current value (e.g., '65psi' or '25ml'). "
        "If no records are provided or the records are mutually contradictory with no resolution, reply ONLY 'ABSTAIN'."
    )
    unique_prompts: Dict[str, str] = {}
    task_meta = []
    mean_store_size = float(np.mean([len(c["events"]) for c in cases]))
    for case in cases:
        for arm in arms:
            sel_ids, status_flag = retrieve_zs_v3_arm(arm, case, top_k=5)
            if status_flag == "CONTESTED_ABSTAIN" or not sel_ids:
                ctx = "(No admitted records — conflicting claims are epistemically CONTESTED.)"
            else:
                ordered = sorted(sel_ids, key=lambda eid: case["events"][eid].tx_time)
                ctx = "\n".join(
                    f"- [vt={case['events'][eid].valid_time}, tx={case['events'][eid].tx_time}] {case['events'][eid].text}"
                    for eid in ordered
                )
            prompt = f"Records:\n{ctx}\n\nQuestion: {case['query_text']}\nAnswer:"
            psha = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
            unique_prompts[psha] = prompt
            task_meta.append({"case": case, "arm": arm, "psha": psha})

    print(f"[R5-S1-ZSv3] {len(task_meta)} evals ({len(unique_prompts)} unique prompts, mean_events={mean_store_size:.1f} > top_k=5)...", flush=True)
    prompt_results: Dict[str, Tuple[str, int, float, bool]] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
        for psha, res in pool.map(lambda kv: (kv[0], _call_vllm(endpoint, model_id, sys_p, kv[1], max_tokens=32)), list(unique_prompts.items())):
            prompt_results[psha] = res

    rows = []
    for tm in task_meta:
        case = tm["case"]
        pred, pt, lat, err = prompt_results[tm["psha"]]
        p_lo = pred.lower()
        gold = case["gold"]
        if gold == "ABSTAIN":
            is_ok = ("abstain" in p_lo) or ("cannot" in p_lo) or ("contradict" in p_lo) or ("no admitted" in p_lo)
        else:
            has_gold = gold.lower() in p_lo
            has_stale = any(sv.lower() in p_lo for sv in case["stale_vals"])
            is_ok = has_gold and (not has_stale)
        rows.append({
            "case_id": case["case_id"],
            "family": case["family"],
            "arm": tm["arm"],
            "n_case_events": len(case["events"]),
            "gold": gold,
            "pred_raw": pred.replace("\n", " "),
            "is_correct": int(is_ok),
            "prompt_tokens": pt,
            "latency_ms": round(lat, 2),
            "api_error": int(err),
            "prompt_sha256": tm["psha"],
        })

    csv_path = out_dir / "r5_s1_zero_shortcut_v3_rows.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    by_arm = defaultdict(list)
    by_fam_arm = defaultdict(list)
    for r in rows:
        by_arm[r["arm"]].append(r)
        by_fam_arm[(r["family"], r["arm"])].append(r["is_correct"])
    fams = sorted({c["family"] for c in cases})
    arm_stats = {
        arm: {
            "overall_acc": round(float(np.mean([r["is_correct"] for r in by_arm[arm]])), 4),
            "mean_prompt_tokens": round(float(np.mean([r["prompt_tokens"] for r in by_arm[arm]])), 1),
            "api_errors": int(sum(r["api_error"] for r in by_arm[arm])),
            "by_family": {f: round(float(np.mean(by_fam_arm[(f, arm)])), 4) for f in fams},
        }
        for arm in arms
    }

    def _imap(a: str) -> Dict[str, float]:
        return {r["case_id"]: float(r["is_correct"]) for r in by_arm[a]}

    c4v3 = _imap("C4_v3_ek5_v3_k4")
    all_cids = [c["case_id"] for c in cases]
    boot = {}
    all_lcb_gt_15pp = True
    for ref in arms[:-1]:
        bt = paired_bootstrap_delta(c4v3, _imap(ref), n_bootstrap=3000, seed=42, expected_ids=all_cids)
        boot[f"C4v3_vs_{ref}"] = {
            "mean_delta": round(bt["mean_delta"], 4),
            "lcb_95": round(bt["lcb_95"], 4),
            "tango_ci95_low": round(bt["tango_score_lcb95"], 4),
            "clopper_pearson_ci95_low": round(bt["clopper_pearson_lcb95"], 4),
        }
        if bt["lcb_95"] <= 0.15:
            all_lcb_gt_15pp = False

    manifest = {
        "schema_version": "r5_s1_zero_shortcut_v3_flooded_v1",
        "git_sha": current_git_sha(REPO_ROOT),
        "model_id": model_id,
        "wall_seconds": round(time.perf_counter() - t0, 2),
        "n_cases": len(cases),
        "mean_events_per_case": round(mean_store_size, 2),
        "top_k": 5,
        "arms": arm_stats,
        "bootstrap": boot,
        "pm1a_all_comparators_lcb95_gt_15pp_pass": bool(
            arm_stats["C4_v3_ek5_v3_k4"]["overall_acc"] >= 0.98 and all_lcb_gt_15pp
        ),
        "csv_path": csv_path.name,
        "csv_sha256": _sha256_file(csv_path),
    }
    man_path = out_dir / "r5_s1_manifest.json"
    write_json_atomic(man_path, manifest)
    write_json_atomic(
        out_dir / "report_receipt.json",
        {
            "schema_version": "r5_report_receipt_v1",
            "git_sha": manifest["git_sha"],
            "manifest_path": man_path.name,
            "manifest_sha256": _sha256_file(man_path),
            "artifact_hashes": {man_path.name: _sha256_file(man_path), csv_path.name: manifest["csv_sha256"]},
            "wall_seconds": manifest["wall_seconds"],
        },
    )
    print(f"[R5-S1-ZSv3] COMPLETE in {manifest['wall_seconds']}s | C4_v3={arm_stats['C4_v3_ek5_v3_k4']['overall_acc']} | PM-1a PASS={manifest['pm1a_all_comparators_lcb95_gt_15pp_pass']}", flush=True)
    return manifest


# ===========================================================================
# STAGE 2: LOCOMO-300 BALANCED 10-CONV WITH EM-4 (`M4f` & `M4g`)
# ===========================================================================
def run_stage2_locomo_300_em4(endpoint: str, model_id: str, root_dir: Path, embedder: SentenceTransformer, max_workers: int = 16) -> Dict[str, Any]:
    from scripts.run_flagship2_r4_ek5_v3_and_suites import (
        _cluster_bootstrap_delta,
        _extract_locomo_units,
        retrieve_locomo_r4_arm,
    )

    out_dir = root_dir / "r5_s2_locomo_300_em4"
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()

    p = hf_hub_download("Percena/locomo-mc10", "raw/locomo10.json", repo_type="dataset")
    convs = json.loads(Path(p).read_text(encoding="utf-8"))

    selected: List[Dict[str, Any]] = []
    for ci, conv in enumerate(convs):
        b_temp, b_fact, b_abs = [], [], []
        for qi, qa in enumerate(conv["qa"]):
            cat = int(qa["category"])
            ev_ids = [tok for item in qa.get("evidence", []) for tok in _EVID_SPLIT_RE.split(str(item).strip()) if tok]
            rec = {
                "qid": f"loc_{ci}_{qi}",
                "conv_idx": ci,
                "category": cat,
                "question": str(qa["question"]),
                "gold_answer": "UNANSWERABLE" if cat == 5 else str(qa.get("answer", "")),
                "is_abstention": (cat == 5),
                "evidence_ids": ev_ids,
            }
            if cat == 2:
                b_temp.append(rec)
            elif cat in (1, 4):
                b_fact.append(rec)
            elif cat == 5:
                b_abs.append(rec)
        b_temp.sort(key=lambda r: hashlib.sha256(r["qid"].encode()).hexdigest())
        b_fact.sort(key=lambda r: hashlib.sha256(r["qid"].encode()).hexdigest())
        b_abs.sort(key=lambda r: hashlib.sha256(r["qid"].encode()).hexdigest())
        selected.extend(b_temp[:10] + b_fact[:10] + b_abs[:10])

    conv_by_qid = {it["qid"]: it["conv_idx"] for it in selected}
    all_qids = [it["qid"] for it in selected]
    arms = [
        "M0_bm25_flat",
        "M1_hybrid_flat",
        "M1c_hybrid_jaccard",
        "M2_scalemas_unsigned",
        "M3_recency_hybrid",
        "M4b_within_and_cross_collapse",
        "M4e_minimal_pair_split",
        "M4f_soft_telemetry_fallback",
        "M4g_ek5_soft_telemetry_recency",
    ]
    ans_sys = (
        "Answer the question strictly using ONLY the dated conversation snippets below. "
        "Be concise (1 short sentence). If the snippets do NOT contain enough information to answer the question, "
        "or the asked event/detail never occurred in the snippets, reply ONLY: UNANSWERABLE."
    )

    unique_ans_prompts: Dict[str, str] = {}
    pending_records: List[Dict[str, Any]] = []
    q_embs = embedder.encode([it["question"] for it in selected], batch_size=64, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False).astype(np.float32)

    for reg_name, inject_echo in (("regime_a_unmodified", False), ("regime_b_blind_neutral_echo_stress", True)):
        conv_units: Dict[int, List[Dict[str, Any]]] = {}
        conv_embs: Dict[int, np.ndarray] = {}
        conv_bm25: Dict[int, TurnBM25] = {}
        for ci, conv in enumerate(convs):
            units = _extract_locomo_units(conv)
            if inject_echo:
                base_embs = embedder.encode([u["text"] for u in units], batch_size=128, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False).astype(np.float32)
                centroid = base_embs.mean(axis=0)
                centroid = centroid / max(float(np.linalg.norm(centroid)), 1e-12)
                cent_sims = base_embs @ centroid
                cand_idx = [i for i in np.argsort(-cent_sims) if len(units[i]["raw_text"].split()) >= 12][:15]
                for d_idx in cand_idx:
                    src_u = units[d_idx]
                    for e_i in range(6):
                        units.append({
                            "unit_id": f"neutral_echo_{ci}_{d_idx}_{e_i}",
                            "session_id": f"recap_session_{e_i}",
                            "sess_num": max(1, src_u["sess_num"]),
                            "date_str": src_u["date_str"],
                            "ts": src_u["ts"] + 0.01 * (e_i + 1),
                            "text": f"Briefing {e_i}: {src_u['text']}",
                            "raw_text": f"Briefing {e_i}: {src_u['raw_text']}",
                        })
            conv_units[ci] = units
            conv_embs[ci] = embedder.encode([u["text"] for u in units], batch_size=128, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False).astype(np.float32)
            conv_bm25[ci] = TurnBM25([u["text"] for u in units])

        for qi, it in enumerate(selected):
            ci = it["conv_idx"]
            units = conv_units[ci]
            u_embs = conv_embs[ci]
            bm_sc = conv_bm25[ci].score(it["question"])
            ev_set = set(it["evidence_ids"])
            for arm in arms:
                if arm in ("M4f_soft_telemetry_fallback", "M4g_ek5_soft_telemetry_recency"):
                    rt = route_soft_telemetry_top_k(units, u_embs, q_embs[qi], bm_sc, top_k=5, mode=arm)
                    sel_idx = list(rt.selected_indices)
                    tag_map = dict(rt.status_tags)
                else:
                    sel_idx, tag_map = retrieve_locomo_r4_arm(arm, units, u_embs, q_embs[qi], bm_sc, top_k=5)
                ev_hit = int(any(units[j]["unit_id"] in ev_set for j in sel_idx)) if ev_set else 0
                ordered = sorted(sel_idx, key=lambda j: units[j]["ts"])
                ctx = "\n".join(f"- [{units[j]['date_str']}] {tag_map.get(j, '')}{units[j]['text']}" for j in ordered)
                prompt = f"Conversation Snippets:\n{ctx}\n\nQuestion: {it['question']}\nAnswer:"
                psha = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
                unique_ans_prompts[psha] = prompt
                pending_records.append({"regime": reg_name, "arm": arm, "item": it, "ev_hit": ev_hit, "psha": psha})

    print(f"[R5-S2-LoCoMo] Executing {len(unique_ans_prompts)} unique answer prompts...", flush=True)
    ans_cache: Dict[str, Tuple[str, int, float, bool]] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
        for psha, res in pool.map(lambda kv: (kv[0], _call_vllm(endpoint, model_id, ans_sys, kv[1], max_tokens=64)), list(unique_ans_prompts.items())):
            ans_cache[psha] = res

    abs_cues = (
        "unanswerable", "not mentioned", "no information", "doesn't mention",
        "does not mention", "cannot be determined", "don't have information", "never mentioned",
    )
    unique_grade_keys: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for rec in pending_records:
        pred, _, _, _ = ans_cache[rec["psha"]]
        p_norm = re.sub(r"\s+", " ", pred.strip())
        key = (rec["item"]["qid"], p_norm.lower())
        if key not in unique_grade_keys:
            unique_grade_keys[key] = {"item": rec["item"], "p_norm": p_norm}

    grade_map: Dict[Tuple[str, str], Tuple[bool, bool]] = {}
    llm_grade_jobs = []
    for key, payload in unique_grade_keys.items():
        p_lo = key[1]
        it = payload["item"]
        if not p_lo or p_lo.startswith("error:"):
            grade_map[key] = (False, p_lo.startswith("error:"))
        elif it["is_abstention"]:
            grade_map[key] = (any(c in p_lo for c in abs_cues), False)
        elif any(c in p_lo for c in abs_cues) and len(p_lo) < 90:
            grade_map[key] = (False, False)
        else:
            llm_grade_jobs.append((key, payload))

    sys_g = (
        "You are an impartial QA grader. Reply ONLY YES if the Model Answer conveys the same core fact/date/entity "
        "as the Ground-Truth Answer without contradicting it. Reply NO if it is wrong, contradicts, or gives a different value."
    )

    def _grade_one(job: Tuple[Tuple[str, str], Dict[str, Any]]) -> Tuple[Tuple[str, str], Tuple[bool, bool]]:
        key, payload = job
        it = payload["item"]
        usr_g = f"Question: {it['question']}\nGround-Truth Answer: {it['gold_answer']}\nModel Answer: {payload['p_norm']}\n\nMatch? Reply YES or NO."
        txt, _, _, err = _call_vllm(endpoint, model_id, sys_g, usr_g, max_tokens=6)
        return key, (txt.strip().upper().startswith("YES"), err)

    print(f"[R5-S2-LoCoMo] Offline lock-free grading {len(llm_grade_jobs)} unique keys...", flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
        for key, res in pool.map(_grade_one, llm_grade_jobs):
            grade_map[key] = res

    all_rows = []
    verify_keys: Dict[Tuple[str, str], Set[int]] = defaultdict(set)
    for rec in pending_records:
        it = rec["item"]
        pred, pt, lat, err = ans_cache[rec["psha"]]
        p_norm = re.sub(r"\s+", " ", pred.strip())
        key = (it["qid"], p_norm.lower())
        ok, g_err = grade_map[key]
        verify_keys[key].add(int(ok))
        all_rows.append({
            "regime": rec["regime"], "arm": rec["arm"], "qid": it["qid"], "conv_idx": it["conv_idx"],
            "category": it["category"], "is_abstention": int(it["is_abstention"]), "evidence_hit": rec["ev_hit"],
            "gold": it["gold_answer"], "pred_raw": p_norm, "is_correct": int(ok),
            "prompt_tokens": pt, "latency_ms": round(lat, 2), "api_error": int(err or g_err), "prompt_sha256": rec["psha"],
        })
    assert sum(1 for v in verify_keys.values() if len(v) > 1) == 0

    csv_path = out_dir / "r5_s2_locomo_300_em4_rows.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(all_rows[0].keys()))
        w.writeheader()
        w.writerows(all_rows)

    regimes_out = {}
    for reg_name in ("regime_a_unmodified", "regime_b_blind_neutral_echo_stress"):
        reg_rows = [r for r in all_rows if r["regime"] == reg_name]
        by_arm = defaultdict(list)
        for r in reg_rows:
            by_arm[r["arm"]].append(r)
        astats = {}
        for arm in arms:
            rs = by_arm[arm]
            non_abs = [r for r in rs if not r["is_abstention"]]
            astats[arm] = {
                "overall_acc": round(float(np.mean([r["is_correct"] for r in rs])), 4),
                "temporal_acc": round(float(np.mean([r["is_correct"] for r in rs if r["category"] == 2])), 4),
                "factual_acc": round(float(np.mean([r["is_correct"] for r in rs if r["category"] in (1, 4)])), 4),
                "abstention_acc": round(float(np.mean([r["is_correct"] for r in rs if r["is_abstention"]])), 4),
                "evidence_recall_at_5": round(float(np.mean([r["evidence_hit"] for r in non_abs])), 4) if non_abs else 0.0,
                "mean_prompt_tokens": round(float(np.mean([r["prompt_tokens"] for r in rs])), 1),
                "api_errors": int(sum(r["api_error"] for r in rs)),
            }

        def _qmap(a: str) -> Dict[str, float]:
            return {r["qid"]: float(r["is_correct"]) for r in by_arm[a]}

        comps = {}
        for cand in ("M4e_minimal_pair_split", "M4f_soft_telemetry_fallback", "M4g_ek5_soft_telemetry_recency"):
            for ref in ("M1_hybrid_flat", "M1c_hybrid_jaccard", "M4b_within_and_cross_collapse"):
                bt = paired_bootstrap_delta(_qmap(cand), _qmap(ref), n_bootstrap=3000, seed=42, expected_ids=all_qids)
                cl = _cluster_bootstrap_delta(_qmap(cand), _qmap(ref), conv_by_qid, n_bootstrap=3000, seed=42)
                comps[f"{cand}_vs_{ref}"] = {
                    "mean_delta": round(bt["mean_delta"], 4),
                    "lcb_95": round(bt["lcb_95"], 4),
                    "tango_ci95_low": round(bt["tango_score_lcb95"], 4),
                    "cluster_ci95_low": cl["cluster_ci95_low"],
                    "cluster_ci95_high": cl["cluster_ci95_high"],
                }
        regimes_out[reg_name] = {"arms": astats, "comparisons": comps}

    m4f_clean_delta = regimes_out["regime_a_unmodified"]["comparisons"]["M4f_soft_telemetry_fallback_vs_M1_hybrid_flat"]["mean_delta"]
    manifest = {
        "schema_version": "r5_s2_locomo_300_em4_v1",
        "git_sha": current_git_sha(REPO_ROOT),
        "model_id": model_id,
        "wall_seconds": round(time.perf_counter() - t0, 2),
        "n_questions": len(selected),
        "regimes": regimes_out,
        "m4f_locomo_clean_ge_minus_1pp_pass": bool(m4f_clean_delta >= -0.01),
        "csv_path": csv_path.name,
        "csv_sha256": _sha256_file(csv_path),
    }
    man_path = out_dir / "r5_s2_manifest.json"
    write_json_atomic(man_path, manifest)
    write_json_atomic(
        out_dir / "report_receipt.json",
        {
            "schema_version": "r5_report_receipt_v1",
            "git_sha": manifest["git_sha"],
            "manifest_path": man_path.name,
            "manifest_sha256": _sha256_file(man_path),
            "artifact_hashes": {man_path.name: _sha256_file(man_path), csv_path.name: manifest["csv_sha256"]},
            "wall_seconds": manifest["wall_seconds"],
        },
    )
    print(f"[R5-S2-LoCoMo] COMPLETE in {manifest['wall_seconds']}s | M4f_clean_delta={m4f_clean_delta:+.4f} (pass={manifest['m4f_locomo_clean_ge_minus_1pp_pass']})", flush=True)
    return manifest


# ===========================================================================
# STAGE 3: FULL ICLR 2025 LONGMEMEVAL-S (ALL N=500 CLEAN + 78 KU ECHO STRESS!)
# ===========================================================================
def _parse_lme_date(date_str: str) -> float:
    cleaned = re.sub(r"\([^)]*\)", "", date_str).strip()
    cleaned = re.sub(r"\s+", " ", cleaned)
    for fmt in ("%Y/%m/%d %H:%M", "%Y/%m/%d", "%Y-%m-%d %H:%M"):
        try:
            return datetime.strptime(cleaned, fmt).timestamp()
        except ValueError:
            continue
    return 0.0


def _extract_lme_units(item: Dict[str, Any], *, inject_echo_stress: bool = False, n_echoes: int = 6) -> List[Dict[str, Any]]:
    units: List[Dict[str, Any]] = []
    sessions = item["haystack_sessions"]
    dates = item["haystack_dates"]
    sids = item["haystack_session_ids"]
    answer_sids = set(item.get("answer_session_ids", []))
    ans_turn_records: List[Tuple[float, str, str]] = []

    for sid, dt_str, turns in zip(sids, dates, sessions):
        ts = _parse_lme_date(dt_str)
        i = 0
        while i < len(turns):
            t = turns[i]
            if t["role"] == "user":
                user_txt = t["content"].strip()
                asst_txt = ""
                has_ans = bool(t.get("has_answer", False))
                if i + 1 < len(turns) and turns[i + 1]["role"] == "assistant":
                    asst_txt = turns[i + 1]["content"].strip()[:320]
                    has_ans = has_ans or bool(turns[i + 1].get("has_answer", False))
                    i += 2
                else:
                    i += 1
                combined = f"User: {user_txt[:900]}" + (f" | Assistant: {asst_txt}" if asst_txt else "")
                units.append({
                    "unit_id": f"{sid}_u{len(units)}",
                    "session_id": sid,
                    "date_str": dt_str,
                    "ts": ts + 0.001 * len(units),
                    "text": combined,
                    "user_text": user_txt[:900],
                    "raw_text": user_txt[:900],
                })
                if sid in answer_sids and has_ans and user_txt:
                    ans_turn_records.append((ts, dt_str, user_txt[:600]))
            else:
                i += 1

    if inject_echo_stress and item["question_type"] == "knowledge-update" and len(ans_turn_records) >= 2:
        ans_turn_records.sort(key=lambda r: r[0])
        old_ts, old_dt, old_txt = ans_turn_records[0]
        new_ts = ans_turn_records[-1][0]
        span = max(new_ts - old_ts, 3600.0 * 6)
        for e_i in range(n_echoes):
            frac = (e_i + 1) / float(n_echoes + 1)
            echo_ts = old_ts + frac * min(span * 0.85, max(new_ts - old_ts - 600.0, 60.0))
            echo_dt = datetime.fromtimestamp(echo_ts).strftime("%Y/%m/%d (%a) %H:%M") if echo_ts > 0 else old_dt
            prefix = _LME_ECHO_PREFIXES[e_i % len(_LME_ECHO_PREFIXES)]
            units.append({
                "unit_id": f"echo_{e_i}",
                "session_id": f"echo_sess_{e_i}",
                "date_str": echo_dt,
                "ts": echo_ts,
                "text": f"User: {prefix}{old_txt}",
                "user_text": f"{prefix}{old_txt}",
                "raw_text": f"{prefix}{old_txt}",
            })

    units.sort(key=lambda u: (u["ts"], u["unit_id"]))
    return units


def _retrieve_lme_r5_arm(
    arm: str,
    units: Sequence[Dict[str, Any]],
    u_embs: np.ndarray,
    q_emb: np.ndarray,
    bm25_scores: np.ndarray,
    *,
    top_k: int = 5,
) -> Tuple[List[int], Dict[int, str]]:
    n = len(units)
    if n == 0:
        return [], {}
    dense = u_embs @ q_emb
    hyb = 0.50 * _minmax(bm25_scores) + 0.50 * _minmax(dense)
    tags: Dict[int, str] = {}

    if arm == "M0_bm25_flat":
        return np.argsort(-bm25_scores, kind="mergesort")[:top_k].tolist(), tags
    if arm == "M1_hybrid_flat":
        return np.argsort(-hyb, kind="mergesort")[:top_k].tolist(), tags
    if arm == "M1c_hybrid_jaccard":
        pool = np.argsort(-hyb, kind="mergesort")[: min(32, n)].tolist()
        tsets = {i: _content_tokens(units[i]["user_text"]) for i in pool}
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
        cnt = ((pe @ pe.T) >= 0.72).sum(axis=1) - 1
        sc = hyb[pool] + 0.12 * np.log1p(np.maximum(cnt, 0))
        return [int(pool[i]) for i in np.argsort(-sc, kind="mergesort")[:top_k]], tags
    if arm == "M3_recency_hybrid":
        ts = np.asarray([u["ts"] for u in units], dtype=np.float64)
        sc = hyb * (1.0 + 0.25 * _minmax(ts))
        return np.argsort(-sc, kind="mergesort")[:top_k].tolist(), tags
    if arm == "M4b_within_and_cross_collapse":
        pool = np.argsort(-hyb, kind="mergesort")[: min(32, n)].tolist()
        pe = u_embs[pool]
        sim = pe @ pe.T
        tsets = [_content_tokens(units[i]["user_text"]) for i in pool]
        ts_list = [units[i]["ts"] for i in pool]
        t_lo, t_hi = min(ts_list), max(ts_list)
        t_span = max(t_hi - t_lo, 1.0)
        order = sorted(range(len(pool)), key=lambda p: -(float(hyb[pool[p]]) + 0.06 * ((ts_list[p] - t_lo) / t_span)))
        sel_loc, suppressed = [], set()
        for p in order:
            if p in suppressed:
                continue
            sel_loc.append(p)
            for q in range(len(pool)):
                if q != p and q not in suppressed and (float(sim[p, q]) >= 0.72 or _jaccard(tsets[p], tsets[q]) >= 0.58):
                    suppressed.add(q)
            if len(sel_loc) >= top_k:
                break
        return [pool[p] for p in sel_loc], tags
    if arm == "M4e_minimal_pair_split":
        pool = np.argsort(-hyb, kind="mergesort")[: min(32, n)].tolist()
        pe = u_embs[pool]
        sim = pe @ pe.T
        tsets = [_content_tokens(units[i]["user_text"]) for i in pool]
        ts_list = [units[i]["ts"] for i in pool]
        t_lo, t_hi = min(ts_list), max(ts_list)
        t_span = max(t_hi - t_lo, 1.0)
        order = sorted(range(len(pool)), key=lambda p: -(float(hyb[pool[p]]) + 0.06 * ((ts_list[p] - t_lo) / t_span)))
        sel_loc, suppressed = [], set()
        for p in order:
            if p in suppressed:
                continue
            sel_loc.append(p)
            sp = units[pool[p]]["session_id"]
            for q in range(len(pool)):
                if q == p or q in suppressed or units[pool[q]]["session_id"] == sp:
                    continue
                if float(sim[p, q]) >= 0.72 or _jaccard(tsets[p], tsets[q]) >= 0.58:
                    act, _ = classify_high_similarity_pair(units[pool[p]]["user_text"], units[pool[q]]["user_text"])
                    if act == "PRESERVE_CONFLICT":
                        continue
                    suppressed.add(q)
            if len(sel_loc) >= top_k:
                break
        return [pool[p] for p in sel_loc], tags
    if arm in ("M4f_soft_telemetry_fallback", "M4g_ek5_soft_telemetry_recency"):
        rt = route_soft_telemetry_top_k(units, u_embs, q_emb, bm25_scores, top_k=top_k, mode=arm)
        return list(rt.selected_indices), dict(rt.status_tags)
    raise ValueError(f"Unknown arm: {arm}")


def run_stage3_longmemeval_full_n500(endpoint: str, model_id: str, root_dir: Path, embedder: SentenceTransformer, max_workers: int = 16) -> Dict[str, Any]:
    out_dir = root_dir / "r5_s3_longmemeval_s_full_n500"
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()

    lme_path = hf_hub_download("xiaowu0162/longmemeval-cleaned", "longmemeval_s_cleaned.json", repo_type="dataset")
    items = json.loads(Path(lme_path).read_text(encoding="utf-8"))
    print(f"[R5-S3-LME500] Loaded ALL {len(items)} LongMemEval-S items: {dict(Counter(x['question_type'] for x in items))}", flush=True)

    arms = [
        "M0_bm25_flat",
        "M1_hybrid_flat",
        "M1c_hybrid_jaccard",
        "M2_scalemas_unsigned",
        "M3_recency_hybrid",
        "M4b_within_and_cross_collapse",
        "M4e_minimal_pair_split",
        "M4f_soft_telemetry_fallback",
        "M4g_ek5_soft_telemetry_recency",
    ]
    ans_sys = (
        "You are a precise conversational memory assistant. Answer the user's question strictly using the "
        "dated conversation excerpts below. When an earlier detail was later updated in a newer dated session "
        "(or marked as an earlier echoed record superseded by a newer update), report the latest current fact "
        "unless the question explicitly asks about the past. Be concise (1-2 sentences)."
    )

    unique_ans_prompts: Dict[str, str] = {}
    pending_records: List[Dict[str, Any]] = []

    q_texts = [str(it["question"]) for it in items]
    q_embs = embedder.encode(q_texts, batch_size=64, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False).astype(np.float32)

    # Regime A: ALL 500 unmodified LongMemEval-S items!
    # Regime B: ALL 78 `knowledge-update` items under 6x echo-persistence stress!
    for idx_i, it in enumerate(items):
        qid = str(it["question_id"])
        qtype = str(it["question_type"])
        q_date = str(it.get("question_date", ""))
        regimes_for_item = [("regime_a_clean_n500", False)]
        if qtype == "knowledge-update":
            regimes_for_item.append(("regime_b_ku_echo_stress_78", True))

        for reg_name, inj in regimes_for_item:
            units = _extract_lme_units(it, inject_echo_stress=inj, n_echoes=6)
            u_embs = embedder.encode([u["text"] for u in units], batch_size=128, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False).astype(np.float32)
            bm_sc = TurnBM25([u["text"] for u in units]).score(it["question"])
            for arm in arms:
                sel_idx, tag_map = _retrieve_lme_r5_arm(arm, units, u_embs, q_embs[idx_i], bm_sc, top_k=5)
                ordered = sorted(sel_idx, key=lambda j: units[j]["ts"])
                ctx = "\n".join(f"- [{units[j]['date_str']}] {tag_map.get(j, '')}{units[j]['text']}" for j in ordered)
                prompt = f"Current Question Date: {q_date}\nConversation Excerpts:\n{ctx}\n\nQuestion: {it['question']}\nAnswer:"
                psha = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
                unique_ans_prompts[psha] = prompt
                pending_records.append({
                    "regime": reg_name,
                    "arm": arm,
                    "qid": qid,
                    "question_type": qtype,
                    "question": str(it["question"]),
                    "gold": str(it["answer"]),
                    "psha": psha,
                })
        if (idx_i + 1) % 100 == 0:
            print(f"[R5-S3-LME500] Encoded & retrieved {idx_i + 1}/{len(items)} items (unique_prompts={len(unique_ans_prompts)})...", flush=True)

    # Incremental checkpointed LLM execution!
    ckpt_ans_path = out_dir / "checkpoint_ans_cache.json"
    ans_cache: Dict[str, Tuple[str, int, float, bool]] = {}
    if ckpt_ans_path.exists():
        raw_ck = json.loads(ckpt_ans_path.read_text(encoding="utf-8"))
        ans_cache = {k: (v[0], int(v[1]), float(v[2]), bool(v[3])) for k, v in raw_ck.items()}

    todo_prompts = [(k, v) for k, v in unique_ans_prompts.items() if k not in ans_cache]
    print(f"[R5-S3-LME500] Phase 1: Executing {len(todo_prompts)} unique answer prompts (cached={len(ans_cache)})...", flush=True)
    chunk_sz = 400
    for start in range(0, len(todo_prompts), chunk_sz):
        batch = todo_prompts[start : start + chunk_sz]
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
            for psha, res in pool.map(lambda kv: (kv[0], _call_vllm(endpoint, model_id, ans_sys, kv[1], max_tokens=96)), batch):
                ans_cache[psha] = res
        write_json_atomic(ckpt_ans_path, ans_cache)
        print(f"[R5-S3-LME500] Answer cache checkpointed: {len(ans_cache)}/{len(unique_ans_prompts)}", flush=True)

    # Phase 2: Lock-free deduplicated grading!
    unique_grade_keys: Dict[Tuple[str, str], Dict[str, str]] = {}
    for rec in pending_records:
        pred, _, _, _ = ans_cache[rec["psha"]]
        p_norm = re.sub(r"\s+", " ", pred.strip())
        key = (rec["qid"], p_norm.lower())
        if key not in unique_grade_keys:
            unique_grade_keys[key] = {"question": rec["question"], "gold": rec["gold"], "p_norm": p_norm}

    grade_map: Dict[Tuple[str, str], Tuple[bool, bool]] = {}
    llm_jobs = []
    for key, payload in unique_grade_keys.items():
        p_lo = key[1]
        g_lo = re.sub(r"\s+", " ", payload["gold"].strip().lower())
        if not p_lo or p_lo.startswith("error:"):
            grade_map[key] = (False, p_lo.startswith("error:"))
        elif g_lo and len(g_lo) >= 4 and g_lo in p_lo and len(p_lo) <= len(g_lo) + 70:
            grade_map[key] = (True, False)
        else:
            llm_jobs.append((key, payload))

    sys_g = (
        "You are an impartial QA evaluator for LongMemEval. Reply ONLY YES if the Model Answer accurately "
        "provides the core information in the Reference Answer without contradicting it on updated facts. "
        "Reply NO if it gives an outdated/superseded value, contradicts the reference, or fails to answer."
    )

    def _grade_lme(job: Tuple[Tuple[str, str], Dict[str, str]]) -> Tuple[Tuple[str, str], Tuple[bool, bool]]:
        key, payload = job
        usr_g = f"Question: {payload['question']}\nReference Answer: {payload['gold']}\nModel Answer: {payload['p_norm']}\n\nCorrect? Reply YES or NO."
        txt, _, _, err = _call_vllm(endpoint, model_id, sys_g, usr_g, max_tokens=6)
        return key, (txt.strip().upper().startswith("YES"), err)

    print(f"[R5-S3-LME500] Phase 2: Lock-free blind grading {len(llm_jobs)} unique (qid, pred) keys...", flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
        for key, res in pool.map(_grade_lme, llm_jobs):
            grade_map[key] = res

    if ckpt_ans_path.exists():
        ckpt_ans_path.unlink()

    all_rows = []
    verify_keys: Dict[Tuple[str, str], Set[int]] = defaultdict(set)
    for rec in pending_records:
        pred, pt, lat, err = ans_cache[rec["psha"]]
        p_norm = re.sub(r"\s+", " ", pred.strip())
        key = (rec["qid"], p_norm.lower())
        ok, g_err = grade_map[key]
        verify_keys[key].add(int(ok))
        all_rows.append({
            "regime": rec["regime"], "arm": rec["arm"], "qid": rec["qid"], "question_type": rec["question_type"],
            "gold": rec["gold"].replace("\n", " "), "pred_raw": p_norm, "is_correct": int(ok),
            "prompt_tokens": pt, "latency_ms": round(lat, 2), "api_error": int(err or g_err), "prompt_sha256": rec["psha"],
        })
    assert sum(1 for v in verify_keys.values() if len(v) > 1) == 0

    csv_path = out_dir / "r5_s3_longmemeval_n500_rows.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(all_rows[0].keys()))
        w.writeheader()
        w.writerows(all_rows)

    reg_summaries = {}
    for reg_name in ("regime_a_clean_n500", "regime_b_ku_echo_stress_78"):
        r_rows = [r for r in all_rows if r["regime"] == reg_name]
        by_arm = defaultdict(list)
        for r in r_rows:
            by_arm[r["arm"]].append(r)
        exp_qids = sorted({r["qid"] for r in r_rows})
        qtypes = sorted({r["question_type"] for r in r_rows})
        astats = {}
        for arm in arms:
            rs = by_arm[arm]
            astats[arm] = {
                "n_items": len(rs),
                "overall_acc": round(float(np.mean([r["is_correct"] for r in rs])), 4),
                "by_question_type": {
                    qt: round(float(np.mean([r["is_correct"] for r in rs if r["question_type"] == qt])), 4)
                    for qt in qtypes
                },
                "mean_prompt_tokens": round(float(np.mean([r["prompt_tokens"] for r in rs])), 1),
                "api_errors": int(sum(r["api_error"] for r in rs)),
            }

        def _qmap(a: str) -> Dict[str, float]:
            return {r["qid"]: float(r["is_correct"]) for r in by_arm[a]}

        comps = {}
        for cand in ("M4b_within_and_cross_collapse", "M4e_minimal_pair_split", "M4f_soft_telemetry_fallback", "M4g_ek5_soft_telemetry_recency"):
            for ref in ("M1_hybrid_flat", "M1c_hybrid_jaccard"):
                bt = paired_bootstrap_delta(_qmap(cand), _qmap(ref), n_bootstrap=3000, seed=42, expected_ids=exp_qids)
                comps[f"{cand}_vs_{ref}"] = {
                    "mean_delta": round(bt["mean_delta"], 4),
                    "lcb_95": round(bt["lcb_95"], 4),
                    "tango_ci95_low": round(bt["tango_score_lcb95"], 4),
                    "clopper_pearson_ci95_low": round(bt["clopper_pearson_lcb95"], 4),
                }
        reg_summaries[reg_name] = {"arms": astats, "comparisons": comps}

    # Separate 70-item `single-session-user` clean check!
    ssu_rows = [r for r in all_rows if r["regime"] == "regime_a_clean_n500" and r["question_type"] == "single-session-user"]
    ssu_ids = sorted({r["qid"] for r in ssu_rows})
    ssu_m1 = {r["qid"]: float(r["is_correct"]) for r in ssu_rows if r["arm"] == "M1_hybrid_flat"}
    ssu_m4f = {r["qid"]: float(r["is_correct"]) for r in ssu_rows if r["arm"] == "M4f_soft_telemetry_fallback"}
    ssu_bt = paired_bootstrap_delta(ssu_m4f, ssu_m1, n_bootstrap=3000, seed=42, expected_ids=ssu_ids)

    c_m4f_clean = reg_summaries["regime_a_clean_n500"]["comparisons"]["M4f_soft_telemetry_fallback_vs_M1_hybrid_flat"]
    c_m4g_stress = reg_summaries["regime_b_ku_echo_stress_78"]["comparisons"]["M4g_ek5_soft_telemetry_recency_vs_M1_hybrid_flat"]
    c_m4f_stress = reg_summaries["regime_b_ku_echo_stress_78"]["comparisons"]["M4f_soft_telemetry_fallback_vs_M1_hybrid_flat"]

    manifest = {
        "schema_version": "r5_s3_longmemeval_s_full_n500_v1",
        "git_sha": current_git_sha(REPO_ROOT),
        "model_id": model_id,
        "wall_seconds": round(time.perf_counter() - t0, 2),
        "n_clean_total": len(items),
        "n_single_session_user": len(ssu_ids),
        "single_session_user_m4f_vs_m1": {
            "mean_delta": round(ssu_bt["mean_delta"], 4),
            "lcb_95": round(ssu_bt["lcb_95"], 4),
            "tango_ci95_low": round(ssu_bt["tango_score_lcb95"], 4),
        },
        "regimes": reg_summaries,
        "pm1b_lme500_checks": {
            "m4f_clean_point_ge_minus_1pp": bool(c_m4f_clean["mean_delta"] >= -0.01),
            "m4f_clean_lcb95_ge_minus_3pp": bool(c_m4f_clean["lcb_95"] >= -0.03),
            "m4f_ssu70_point_ge_minus_1pp": bool(ssu_bt["mean_delta"] >= -0.01),
            "m4f_stress_delta": c_m4f_stress["mean_delta"],
            "m4g_stress_delta": c_m4g_stress["mean_delta"],
        },
        "csv_path": csv_path.name,
        "csv_sha256": _sha256_file(csv_path),
    }
    man_path = out_dir / "r5_s3_manifest.json"
    write_json_atomic(man_path, manifest)
    write_json_atomic(
        out_dir / "report_receipt.json",
        {
            "schema_version": "r5_report_receipt_v1",
            "git_sha": manifest["git_sha"],
            "manifest_path": man_path.name,
            "manifest_sha256": _sha256_file(man_path),
            "artifact_hashes": {man_path.name: _sha256_file(man_path), csv_path.name: manifest["csv_sha256"]},
            "wall_seconds": manifest["wall_seconds"],
        },
    )
    print(f"[R5-S3-LME500] COMPLETE in {manifest['wall_seconds']}s | M4f_N500_delta={c_m4f_clean['mean_delta']:+.4f} | SSU70_delta={ssu_bt['mean_delta']:+.4f} | Stress_M4f={c_m4f_stress['mean_delta']:+.4f}", flush=True)
    return manifest


# ===========================================================================
# STAGE 4: PM-4 2x2 CROSS-FAMILY (NEMOTRON x QWEN2.5) + BLIND 100-ITEM AUDIT PACKET
# ===========================================================================
def _cohens_kappa(y1: Sequence[int], y2: Sequence[int]) -> float:
    assert len(y1) == len(y2) and len(y1) > 0
    n = float(len(y1))
    po = sum(1 for a, b in zip(y1, y2) if a == b) / n
    p1 = sum(y1) / n
    p2 = sum(y2) / n
    pe = p1 * p2 + (1.0 - p1) * (1.0 - p2)
    if abs(1.0 - pe) <= 1e-12:
        return 1.0
    return (po - pe) / (1.0 - pe)


def run_stage4_pm4_cross_family_and_audit_packet(
    endpoint: str,
    nemotron_id: str,
    root_dir: Path,
    second_hf_model: str = "Qwen/Qwen2.5-7B-Instruct",
) -> Dict[str, Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    out_dir = root_dir / "r5_s4_pm4_cross_family_2x2_and_audit"
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()

    # 1. Build stratified 120-item cross-family evaluation set + 100-item sealed blind human audit packet
    #    drawing from Stage 1 (ZS v3), Stage 2 (LoCoMo-300), and Stage 3 (LongMemEval-S N=500)!
    s2_csv = root_dir / "r5_s2_locomo_300_em4" / "r5_s2_locomo_300_em4_rows.csv"
    s3_csv = root_dir / "r5_s3_longmemeval_s_full_n500" / "r5_s3_longmemeval_n500_rows.csv"

    rng = random.Random(20261122)
    loc_rows = [r for r in csv.DictReader(s2_csv.open("r", encoding="utf-8")) if r["arm"] in ("M1_hybrid_flat", "M4f_soft_telemetry_fallback") and not int(r["is_abstention"])]
    lme_rows = [r for r in csv.DictReader(s3_csv.open("r", encoding="utf-8")) if r["arm"] in ("M1_hybrid_flat", "M4f_soft_telemetry_fallback")]
    rng.shuffle(loc_rows)
    rng.shuffle(lme_rows)

    # Construct 100-item blind human audit sheet (50 LoCoMo + 50 LongMemEval non-trivial QA pairs, arm labels stripped!)
    audit_pool = []
    seen_pairs = set()
    for src_name, pool_r, quota in (("LoCoMo-300", loc_rows, 50), ("LongMemEval-S", lme_rows, 50)):
        cnt = 0
        for r in pool_r:
            pair_k = (src_name, r["qid"], r["pred_raw"])
            if pair_k in seen_pairs or not r["pred_raw"]:
                continue
            seen_pairs.add(pair_k)
            audit_pool.append({
                "source": src_name,
                "qid": r["qid"],
                "regime": r["regime"],
                "arm_sealed": r["arm"],
                "gold_answer": r["gold"],
                "model_prediction": r["pred_raw"],
                "nemotron_judge_grade": int(r["is_correct"]),
            })
            cnt += 1
            if cnt >= quota:
                break
    rng.shuffle(audit_pool)

    blind_sheet_path = out_dir / "pm4_human_audit_100_blind_sheet.csv"
    sealed_key_path = out_dir / "pm4_human_audit_100_sealed_key.json"
    with blind_sheet_path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["audit_id", "dataset", "qid", "reference_answer", "candidate_answer", "human_yes_or_no", "notes"])
        w.writeheader()
        for idx, it in enumerate(audit_pool):
            it["audit_id"] = f"AUD_{idx:03d}"
            w.writerow({
                "audit_id": it["audit_id"],
                "dataset": it["source"],
                "qid": it["qid"],
                "reference_answer": it["gold_answer"],
                "candidate_answer": it["model_prediction"],
                "human_yes_or_no": "",
                "notes": "",
            })
    write_json_atomic(sealed_key_path, {"n_items": len(audit_pool), "items": audit_pool})

    # 2. Load independent second-family model (`Qwen/Qwen2.5-7B-Instruct`) in `.venv` on free GB10 memory!
    print(f"[R5-S4-PM4] Loading second model family {second_hf_model} in bfloat16 for 2x2 Gen x Judge...", flush=True)
    tok = AutoTokenizer.from_pretrained(second_hf_model)
    mdl = AutoModelForCausalLM.from_pretrained(
        second_hf_model,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
    )
    mdl.eval()

    def _qwen_chat(sys_p: str, usr_p: str, max_new: int = 48) -> str:
        msgs = [{"role": "system", "content": sys_p}, {"role": "user", "content": usr_p}]
        txt_in = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inp = tok([txt_in], return_tensors="pt").to(mdl.device)
        with torch.no_grad():
            out_ids = mdl.generate(**inp, max_new_tokens=max_new, do_sample=False, pad_token_id=tok.eos_token_id)
        gen_ids = out_ids[0][inp.input_ids.shape[1] :]
        return tok.decode(gen_ids, skip_special_tokens=True).strip()

    # Part 4A: Inter-family Judge Agreement (Cohen's kappa) across all 100 stratified real-conversation items
    sys_g = (
        "You are an impartial QA evaluator. Reply ONLY YES if the Candidate Answer conveys the same core fact, "
        "date, or entity as the Reference Answer without contradicting it. Reply NO otherwise."
    )
    nem_grades, qwen_grades = [], []
    for it in audit_pool:
        usr_g = f"Question ID: {it['qid']}\nReference Answer: {it['gold_answer']}\nCandidate Answer: {it['model_prediction']}\n\nCorrect? Reply YES or NO."
        t_nem, _, _, _ = _call_vllm(endpoint, nemotron_id, sys_g, usr_g, max_tokens=6)
        t_qw = _qwen_chat(sys_g, usr_g, max_new=6)
        g_n = int(t_nem.strip().upper().startswith("YES"))
        g_q = int(t_qw.strip().upper().startswith("YES"))
        nem_grades.append(g_n)
        qwen_grades.append(g_q)
        it["nemotron_fresh_judge"] = g_n
        it["qwen_judge"] = g_q

    kappa_real_conv = _cohens_kappa(nem_grades, qwen_grades)
    raw_agree_real = float(np.mean([int(a == b) for a, b in zip(nem_grades, qwen_grades)]))

    # Part 4B: Full 2x2 Generator x Judge Matrix on Stratified Zero-Shortcut v3 (`N=80 = 8x10` x 4 key arms = 320 evals!)
    zs_cases = build_zero_shortcut_v3_flooded_cases(n_per_family=10)
    zs_arms = ["C0_bm25_flat", "C2c_astra_4line_rule", "C3_ordinary_bitemporal_tms", "C4_v3_ek5_v3_k4"]
    zs_sys = (
        "Answer the factual query strictly using ONLY the dated records below. "
        "Reply with ONLY the exact current value (e.g., '65psi' or '25ml'). "
        "If no records are provided or the records are mutually contradictory with no resolution, reply ONLY 'ABSTAIN'."
    )
    zs_judge_sys = (
        "You are an impartial factual QA judge. Reply ONLY YES if the Candidate Answer matches the Gold Value "
        "and does NOT assert any Stale Retracted Value. For Gold='ABSTAIN', reply YES iff the Candidate abstains or notes unresolved contradiction. Otherwise reply NO."
    )

    matrix_rows = []
    j_nem_all, j_qwen_all = [], []
    for case in zs_cases:
        for arm in zs_arms:
            sel_ids, st_flag = retrieve_zs_v3_arm(arm, case, top_k=5)
            if st_flag == "CONTESTED_ABSTAIN" or not sel_ids:
                ctx = "(No admitted records — conflicting claims are epistemically CONTESTED.)"
            else:
                ordered = sorted(sel_ids, key=lambda eid: case["events"][eid].tx_time)
                ctx = "\n".join(f"- [vt={case['events'][eid].valid_time}, tx={case['events'][eid].tx_time}] {case['events'][eid].text}" for eid in ordered)
            prompt = f"Records:\n{ctx}\n\nQuestion: {case['query_text']}\nAnswer:"
            pred_nem, _, _, _ = _call_vllm(endpoint, nemotron_id, zs_sys, prompt, max_tokens=24)
            pred_qw = _qwen_chat(zs_sys, prompt, max_new=24)

            for gen_name, pred_txt in (("Nemotron_30B", pred_nem), ("Qwen2.5_7B", pred_qw)):
                jp = (
                    f"Question: {case['query_text']}\nGold Value: {case['gold']}\n"
                    f"Stale Retracted Values: {', '.join(case['stale_vals'])}\n"
                    f"Candidate Answer: {pred_txt}\n\nCorrect? Reply YES or NO."
                )
                jn_txt, _, _, _ = _call_vllm(endpoint, nemotron_id, zs_judge_sys, jp, max_tokens=6)
                jq_txt = _qwen_chat(zs_judge_sys, jp, max_new=6)
                jn = int(jn_txt.strip().upper().startswith("YES"))
                jq = int(jq_txt.strip().upper().startswith("YES"))
                j_nem_all.append(jn)
                j_qwen_all.append(jq)
                matrix_rows.append({
                    "case_id": case["case_id"], "family": case["family"], "arm": arm,
                    "generator": gen_name, "gold": case["gold"], "pred": pred_txt.replace("\n", " "),
                    "judge_nemotron": jn, "judge_qwen": jq,
                })

    kappa_2x2 = _cohens_kappa(j_nem_all, j_qwen_all)
    combined_nem = nem_grades + j_nem_all
    combined_qwen = qwen_grades + j_qwen_all
    kappa_combined = _cohens_kappa(combined_nem, combined_qwen)

    # Bootstrap 95% CI on combined Cohen's kappa
    rng_np = np.random.default_rng(42)
    k_boots = []
    n_tot = len(combined_nem)
    arr_n = np.asarray(combined_nem)
    arr_q = np.asarray(combined_qwen)
    for _ in range(2000):
        b_idx = rng_np.integers(0, n_tot, size=n_tot)
        k_boots.append(_cohens_kappa(arr_n[b_idx].tolist(), arr_q[b_idx].tolist()))

    mat_csv = out_dir / "r5_s4_pm4_2x2_matrix_rows.csv"
    with mat_csv.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(matrix_rows[0].keys()))
        w.writeheader()
        w.writerows(matrix_rows)

    cell_2x2 = {}
    for gen_name in ("Nemotron_30B", "Qwen2.5_7B"):
        for judge_col, j_label in (("judge_nemotron", "Nemotron_Judge"), ("judge_qwen", "Qwen2.5_Judge")):
            cell_2x2[f"{gen_name}_x_{j_label}"] = {
                arm: round(float(np.mean([r[judge_col] for r in matrix_rows if r["generator"] == gen_name and r["arm"] == arm])), 4)
                for arm in zs_arms
            }

    mdl = None
    torch.cuda.empty_cache()

    manifest = {
        "schema_version": "r5_s4_pm4_cross_family_2x2_and_audit_v1",
        "git_sha": current_git_sha(REPO_ROOT),
        "wall_seconds": round(time.perf_counter() - t0, 2),
        "models_crossed": {"family_1": nemotron_id, "family_2": second_hf_model},
        "inter_judge_agreement": {
            "real_conv_100_cohens_kappa": round(kappa_real_conv, 4),
            "real_conv_100_raw_agreement": round(raw_agree_real, 4),
            "zs_2x2_640_cohens_kappa": round(kappa_2x2, 4),
            "combined_740_cohens_kappa": round(kappa_combined, 4),
            "combined_kappa_ci95": [round(float(np.percentile(k_boots, 2.5)), 4), round(float(np.percentile(k_boots, 97.5)), 4)],
            "kappa_ge_085_pass": bool(kappa_combined >= 0.85),
        },
        "matrix_2x2_accuracy_by_arm": cell_2x2,
        "human_audit_packet": {
            "status": "READY_FOR_HUMAN_SIGN_OFF",
            "n_blind_items": len(audit_pool),
            "blind_sheet_csv": blind_sheet_path.name,
            "blind_sheet_sha256": _sha256_file(blind_sheet_path),
            "sealed_key_json": sealed_key_path.name,
            "sealed_key_sha256": _sha256_file(sealed_key_path),
        },
        "matrix_csv": mat_csv.name,
        "matrix_csv_sha256": _sha256_file(mat_csv),
    }
    man_path = out_dir / "r5_s4_manifest.json"
    write_json_atomic(man_path, manifest)
    write_json_atomic(
        out_dir / "report_receipt.json",
        {
            "schema_version": "r5_report_receipt_v1",
            "git_sha": manifest["git_sha"],
            "manifest_path": man_path.name,
            "manifest_sha256": _sha256_file(man_path),
            "artifact_hashes": {
                man_path.name: _sha256_file(man_path),
                mat_csv.name: manifest["matrix_csv_sha256"],
                blind_sheet_path.name: manifest["human_audit_packet"]["blind_sheet_sha256"],
                sealed_key_path.name: manifest["human_audit_packet"]["sealed_key_sha256"],
            },
            "wall_seconds": manifest["wall_seconds"],
        },
    )
    print(f"[R5-S4-PM4] COMPLETE in {manifest['wall_seconds']}s | Combined Cohen's kappa={kappa_combined:.4f} (pass>=0.85: {manifest['inter_judge_agreement']['kappa_ge_085_pass']})", flush=True)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Overnight R5 Spark 6724 Master Pipeline")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--endpoint", type=str, default="http://127.0.0.1:8890/v1")
    parser.add_argument("--model-id", type=str, default="nemotron-lightning")
    parser.add_argument("--second-hf-model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--max-workers", type=int, default=16)
    args = parser.parse_args()

    args.output_root.mkdir(parents=True, exist_ok=True)
    t0_all = time.perf_counter()
    embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cuda")

    summary: Dict[str, Any] = {"started_at": datetime.now().isoformat(), "stages": {}}
    stage_fns = [
        ("stage1_zs_v3_flooded", lambda: run_stage1_zs_v3(args.endpoint, args.model_id, args.output_root, args.max_workers)),
        ("stage2_locomo_300_em4", lambda: run_stage2_locomo_300_em4(args.endpoint, args.model_id, args.output_root, embedder, args.max_workers)),
        ("stage3_longmemeval_n500", lambda: run_stage3_longmemeval_full_n500(args.endpoint, args.model_id, args.output_root, embedder, args.max_workers)),
        ("stage4_pm4_2x2_and_audit", lambda: run_stage4_pm4_cross_family_and_audit_packet(args.endpoint, args.model_id, args.output_root, args.second_hf_model)),
    ]
    for s_name, fn in stage_fns:
        try:
            res = fn()
            summary["stages"][s_name] = {"status": "COMPLETED", "wall_seconds": res["wall_seconds"]}
        except Exception as exc:  # noqa: BLE001
            tb = traceback.format_exc()
            print(f"[R5-6724] ERROR in {s_name}: {exc}\n{tb}", flush=True)
            summary["stages"][s_name] = {"status": "ERROR", "error": str(exc), "traceback": tb}
        write_json_atomic(args.output_root / "spark6724_master_status.json", summary)

    summary["total_wall_seconds"] = round(time.perf_counter() - t0_all, 2)
    summary["telemetry"] = process_telemetry()
    write_json_atomic(args.output_root / "spark6724_master_status.json", summary)
    print(f"[R5-6724] ALL STAGES FINISHED in {summary['total_wall_seconds']}s", flush=True)


if __name__ == "__main__":
    main()
