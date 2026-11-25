r"""Stage R6 PM-4 — Open-Conversational Cross-Family Judge Calibration (`N=200` + Pooled `N=520`).

Addresses the R5b conversational inter-judge gap (`κ = 0.7414`, `87/100` raw agreement on open-ended
LoCoMo + LongMemEval-S vs `0.9847` on structured Zero-Shortcut QA) by replacing the underspecified
two-line prompt with a crisp, symmetric, rubric-grounded binary evaluation rubric applied
IDENTICALLY to two completely independent model families:
  - Judge A: NVIDIA `Nemotron-3.5-Lightning-30B-A3B` (served on DGX Spark #3 `10.0.0.24:8801` via
    SSH tunnel port `18851`, `enable_thinking=False`, `temperature=0.0`).
  - Judge B: Alibaba `Qwen/Qwen2.5-7B-Instruct` (loaded locally on Desktop RTX 3090 in `bfloat16`,
    greedy decoding `do_sample=False`).

Sampling Protocol (Deterministic, Zero-Leak):
  - `N=100` LoCoMo items sampled deterministically via SHA-256 seed from R5 `r5_s2_locomo_300_em4_rows.csv`
    (`regime_a_unmodified`), stratified across all 4 LoCoMo categories (`1, 2, 4, 5` including `UNANSWERABLE`).
  - `N=100` LongMemEval-S items sampled deterministically via SHA-256 seed from R5 `r5_s3_longmemeval_n500_rows.csv`
    (`regime_a_clean_n500`), stratified across all available question types (`knowledge-update`,
    `multi-session`, `single-session-assistant`, `single-session-preference`, `single-session-user`,
    `temporal-reasoning`).
  - Pooled `N=520` (`320` stratified structured items from R5b `r5b_s4_pm4_2x2_matrix_rows.csv` +
    `200` open-conversational items) and full pooled `N=1000` (`800` structured + `200` conversational),
    with 3,000-resample bootstrap 95% confidence intervals.

Strictly ADDITIVE: reads R5/R5b CSVs read-only and NEVER touches any sealed key or existing artifact.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
import json
import re
import socket
import subprocess
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import requests

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.night1_common import (  # noqa: E402
    current_git_sha,
    process_telemetry,
    write_json_atomic,
)

RUBRIC_VERSION = "pm4_symmetric_question_grounded_rubric_v8"

# Crisp, symmetric, question-grounded, dataset-blind binary judge prompt applied identically to Judge A and Judge B.
SYMMETRIC_CONV_JUDGE_SYSTEM_PROMPT = (
    "You are an impartial QA grader. Given a Question, Reference Answer, and Candidate Answer, "
    "determine if the Candidate Answer is CORRECT using these 4 symmetric rules. Reply with ONLY YES or NO.\n"
    "1. CORE FACT MATCH: Reply YES if Candidate correctly answers the Question with the core entity, fact, "
    "number, or concept in Reference without contradicting it (full sentences and extra accurate context are YES). "
    "Reply NO if Candidate answers the wrong entity, contradicts Reference, or omits a required item from a multi-item Reference.\n"
    "2. NUMERIC, UNIT, & DATE EQUIVALENCE: Equivalent formats ('4' == 'four', '25:50' == '25 minutes 50 seconds', "
    "'Oct 2023' == 'October, 2023', minor spelling/quote differences, or matching an explicit acceptable alternative "
    "in Reference) are YES. Different calendar dates/numbers or vague relative words ('next month', 'this week') "
    "without the matching calendar date are NO.\n"
    "3. ABSTENTION EQUIVALENCE: If Reference is 'UNANSWERABLE', reply YES iff Candidate explicitly abstains "
    "('UNANSWERABLE', 'not mentioned', 'unknown', 'cannot determine'). If Reference is a concrete fact "
    "(NOT 'UNANSWERABLE') and Candidate says 'UNANSWERABLE' or abstains, reply NO.\n"
    "4. PREFERENCE RUBRIC: If Reference starts with 'The user would prefer...', reply YES iff Candidate "
    "clearly follows that stated user preference with relevant specifics.\n\n"
    "Examples:\n"
    "- Q: What is the capital of Japan? | Ref: Tokyo | Cand: The capital of Japan is Tokyo. -> YES\n"
    "- Q: How many miles did I run? | Ref: 12 | Cand: You ran 12 miles last week. -> YES\n"
    "- Q: What car do I drive? | Ref: Blue sedan | Cand: UNANSWERABLE -> NO\n"
    "- Q: What is Bob's dog's name? | Ref: UNANSWERABLE | Cand: That detail was not mentioned. -> YES\n"
    "- Q: Which fruits did I buy? | Ref: Apple, banana | Cand: Apple -> NO\n\n"
    "Output ONLY one word: YES or NO."
)

_VERDICT_LINE_RE = re.compile(r"Verdict\s*:\s*(YES|NO)\b", re.IGNORECASE)
_VERDICT_YES_RE = re.compile(r"\bYES\b", re.IGNORECASE)
_VERDICT_NO_RE = re.compile(r"\bNO\b", re.IGNORECASE)


def load_question_lookup() -> Dict[str, str]:
    """Load exact `(dataset, qid) -> question` lookup from cached HF LoCoMo & LongMemEval-S files."""
    from huggingface_hub import hf_hub_download

    qmap: Dict[str, str] = {}
    loc_path = hf_hub_download("Percena/locomo-mc10", "raw/locomo10.json", repo_type="dataset")
    convs = json.loads(Path(loc_path).read_text(encoding="utf-8"))
    for ci, conv in enumerate(convs):
        for qi, qa in enumerate(conv.get("qa", [])):
            qmap[f"LoCoMo-300::loc_{ci}_{qi}"] = str(qa.get("question", "")).strip()

    lme_path = hf_hub_download("xiaowu0162/longmemeval-cleaned", "longmemeval_s_cleaned.json", repo_type="dataset")
    lme_items = json.loads(Path(lme_path).read_text(encoding="utf-8"))
    for it in lme_items:
        qmap[f"LongMemEval-S::{it['question_id']}"] = str(it.get("question", "")).strip()
    return qmap


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def build_symmetric_judge_user_prompt(
    reference_answer: str,
    candidate_answer: str,
    *,
    question: str = "",
    dataset: str = "",
    question_type: str = "",
) -> str:
    """Build deterministic, dataset-blind, question-grounded symmetric user prompt."""
    _ = (dataset, question_type)  # Deliberately omitted so judging is 100% dataset-blind.
    q_txt = (question or "").strip()
    ref = (reference_answer or "").strip()
    cand = (candidate_answer or "").strip()
    q_line = f"Question: {q_txt}\n" if q_txt else ""
    return (
        f"{q_line}"
        f"Reference Answer: {ref}\n"
        f"Candidate Answer: {cand}\n\n"
        "Correct? Reply ONLY YES or NO."
    )


def parse_binary_verdict(raw_output: str) -> Tuple[int, bool]:
    """Deterministically parse a YES/NO judge output into `(grade_0_or_1, is_clean_parse)`.

    Prioritizes explicit `Verdict: YES|NO` lines, then falls back to leading/standalone YES/NO tokens.
    Returns `(1, True)` for YES, `(0, True)` for NO, and `(0, False)` for malformed/empty text.
    """
    cleaned = (raw_output or "").strip()
    if "</think>" in cleaned:
        cleaned = cleaned.split("</think>", 1)[1].strip()
    if not cleaned:
        return 0, False
    matches = _VERDICT_LINE_RE.findall(cleaned)
    if matches:
        uniq = {m.upper() for m in matches}
        if len(uniq) > 1:
            return 0, False
        return (1 if matches[-1].upper() == "YES" else 0), True
    m_yes = _VERDICT_YES_RE.search(cleaned)
    m_no = _VERDICT_NO_RE.search(cleaned)
    if m_yes and m_no:
        return 0, False
    first_line = cleaned.splitlines()[0].strip().strip(".,:;!*`\"'()[]{}")
    up = first_line.upper()
    if up == "YES" or up.startswith("YES ") or up.startswith("YES:"):
        return 1, True
    if up == "NO" or up.startswith("NO ") or up.startswith("NO:"):
        return 0, True
    last_line = cleaned.splitlines()[-1].strip().strip(".,:;!*`\"'()[]{}").upper()
    if last_line == "YES":
        return 1, True
    if last_line == "NO":
        return 0, True
    if m_yes and not m_no:
        return 1, True
    if m_no and not m_yes:
        return 0, True
    return 0, False


def cohens_kappa_safe(y1: Sequence[int], y2: Sequence[int]) -> Tuple[float, float, bool]:
    """Return `(kappa, raw_agreement, degenerate_marginals)` with zero-denominator guard."""
    if len(y1) != len(y2) or len(y1) == 0:
        raise ValueError("y1 and y2 must be non-empty sequences of equal length")
    n = float(len(y1))
    po = sum(1 for a, b in zip(y1, y2) if int(a) == int(b)) / n
    p1 = sum(int(a) for a in y1) / n
    p2 = sum(int(b) for b in y2) / n
    pe = p1 * p2 + (1.0 - p1) * (1.0 - p2)
    if abs(1.0 - pe) <= 1e-12:
        return (1.0 if abs(po - 1.0) <= 1e-12 else 0.0), po, True
    return (po - pe) / (1.0 - pe), po, False


def bootstrap_cohens_kappa(
    y1: Sequence[int],
    y2: Sequence[int],
    *,
    n_bootstrap: int = 3000,
    seed: int = 42,
) -> Dict[str, Any]:
    """Compute point Cohen's kappa, raw agreement, confusion counts, and bootstrap 95% CI."""
    if len(y1) != len(y2) or len(y1) == 0:
        raise ValueError("y1 and y2 must be non-empty sequences of equal length")
    a = np.asarray([int(x) for x in y1], dtype=np.int64)
    b = np.asarray([int(x) for x in y2], dtype=np.int64)
    n = len(a)
    kappa, raw_agree, deg = cohens_kappa_safe(a.tolist(), b.tolist())

    tp = int(np.sum((a == 1) & (b == 1)))
    tn = int(np.sum((a == 0) & (b == 0)))
    fp = int(np.sum((a == 1) & (b == 0)))  # Judge A = YES, Judge B = NO
    fn = int(np.sum((a == 0) & (b == 1)))  # Judge A = NO, Judge B = YES

    rng = np.random.default_rng(seed)
    idx_mat = rng.integers(0, n, size=(n_bootstrap, n))
    a_b = a[idx_mat]
    b_b = b[idx_mat]
    po_b = np.mean(a_b == b_b, axis=1)
    p1_b = np.mean(a_b, axis=1)
    p2_b = np.mean(b_b, axis=1)
    pe_b = p1_b * p2_b + (1.0 - p1_b) * (1.0 - p2_b)
    denom = 1.0 - pe_b
    k_b = np.where(
        np.abs(denom) <= 1e-12,
        np.where(np.abs(po_b - 1.0) <= 1e-12, 1.0, 0.0),
        (po_b - pe_b) / np.where(np.abs(denom) <= 1e-12, 1.0, denom),
    )

    ci_lo = float(np.percentile(k_b, 2.5))
    ci_hi = float(np.percentile(k_b, 97.5))
    return {
        "n_items": n,
        "cohens_kappa": round(float(kappa), 4),
        "raw_agreement": round(float(raw_agree), 4),
        "n_agree": int(tp + tn),
        "n_disagree": int(fp + fn),
        "ci95": [round(ci_lo, 4), round(ci_hi, 4)],
        "n_bootstrap": int(n_bootstrap),
        "degenerate_marginals": bool(deg),
        "judge_a_yes_rate": round(float(np.mean(a)), 4),
        "judge_b_yes_rate": round(float(np.mean(b)), 4),
        "confusion": {
            "both_yes": tp,
            "both_no": tn,
            "judge_a_yes_judge_b_no": fp,
            "judge_a_no_judge_b_yes": fn,
        },
    }


def _locomo_conv_cluster_id(qid: str) -> str:
    parts = str(qid).split("_")
    return f"loc_conv_{parts[1]}" if len(parts) >= 3 and parts[0] == "loc" else str(qid)


def cluster_and_stratified_bootstrap_kappa(
    rows: Sequence[Mapping[str, Any]],
    *,
    n_bootstrap: int = 3000,
    seed: int = 42,
) -> Dict[str, Any]:
    """Bootstrap intact `(Judge A, Judge B)` pairs using LoCoMo conversation clusters + stratified LME items."""
    loc_by_conv: Dict[str, List[Tuple[int, int]]] = defaultdict(list)
    lme_by_stratum: Dict[str, List[Tuple[int, int]]] = defaultdict(list)
    strata_pairs: Dict[str, List[Tuple[int, int]]] = defaultdict(list)

    for r in rows:
        pair = (int(r["nemotron_grade"]), int(r["qwen_grade"]))
        sk = f"{r['dataset']}::{r['stratum']}"
        strata_pairs[sk].append(pair)
        if r["dataset"] == "LoCoMo-300":
            loc_by_conv[_locomo_conv_cluster_id(str(r["qid"]))].append(pair)
        else:
            lme_by_stratum[str(r["stratum"])].append(pair)

    conv_keys = sorted(loc_by_conv.keys())
    lme_keys = sorted(lme_by_stratum.keys())
    rng = np.random.default_rng(seed)

    conv_kappas: List[float] = []
    loc_kappas: List[float] = []
    lme_kappas: List[float] = []

    for _ in range(n_bootstrap):
        sampled_loc: List[Tuple[int, int]] = []
        if conv_keys:
            picked_convs = rng.integers(0, len(conv_keys), size=len(conv_keys))
            for ci in picked_convs:
                sampled_loc.extend(loc_by_conv[conv_keys[int(ci)]])
            k_loc, _, _ = cohens_kappa_safe([p[0] for p in sampled_loc], [p[1] for p in sampled_loc])
            loc_kappas.append(k_loc)

        sampled_lme: List[Tuple[int, int]] = []
        for lk in lme_keys:
            bucket = lme_by_stratum[lk]
            picks = rng.integers(0, len(bucket), size=len(bucket))
            for pi in picks:
                sampled_lme.append(bucket[int(pi)])
        if sampled_lme:
            k_lme, _, _ = cohens_kappa_safe([p[0] for p in sampled_lme], [p[1] for p in sampled_lme])
            lme_kappas.append(k_lme)

        combined = sampled_loc + sampled_lme
        k_all, _, _ = cohens_kappa_safe([p[0] for p in combined], [p[1] for p in combined])
        conv_kappas.append(k_all)

    by_stratum_ci: Dict[str, Any] = {}
    for sk in sorted(strata_pairs.keys()):
        pairs = strata_pairs[sk]
        ys_a = [p[0] for p in pairs]
        ys_b = [p[1] for p in pairs]
        st = bootstrap_cohens_kappa(ys_a, ys_b, n_bootstrap=n_bootstrap, seed=seed + 101)
        by_stratum_ci[sk] = {
            "n_items": st["n_items"],
            "cohens_kappa": st["cohens_kappa"],
            "raw_agreement": st["raw_agreement"],
            "ci95": st["ci95"],
            "degenerate_marginals": st["degenerate_marginals"],
        }

    return {
        "conversational_n200_cluster_ci95": [
            round(float(np.percentile(conv_kappas, 2.5)), 4),
            round(float(np.percentile(conv_kappas, 97.5)), 4),
        ] if conv_kappas else None,
        "locomo_n100_conv_cluster_ci95": [
            round(float(np.percentile(loc_kappas, 2.5)), 4),
            round(float(np.percentile(loc_kappas, 97.5)), 4),
        ] if loc_kappas else None,
        "locomo_n_conversation_clusters": len(conv_keys),
        "longmemeval_n100_stratified_ci95": [
            round(float(np.percentile(lme_kappas, 2.5)), 4),
            round(float(np.percentile(lme_kappas, 97.5)), 4),
        ] if lme_kappas else None,
        "by_stratum_with_ci95": by_stratum_ci,
    }


def _allocate_stratified_quotas(
    strata_counts: Mapping[str, int],
    target_n: int,
) -> Dict[str, int]:
    """Allocate `target_n` across strata evenly (capped by available items per stratum)."""
    keys = sorted(strata_counts.keys())
    if not keys:
        raise ValueError("Cannot allocate quotas over empty strata")
    total_avail = sum(strata_counts.values())
    if total_avail < target_n:
        raise ValueError(f"Requested {target_n} items, but only {total_avail} available")

    quotas = {k: 0 for k in keys}
    remaining = target_n
    active = [k for k in keys if strata_counts[k] > 0]
    while remaining > 0 and active:
        base = max(1, remaining // len(active))
        progressed = False
        for k in list(active):
            if remaining <= 0:
                break
            room = strata_counts[k] - quotas[k]
            take = min(base, room, remaining)
            if take > 0:
                quotas[k] += take
                remaining -= take
                progressed = True
            if quotas[k] >= strata_counts[k]:
                active.remove(k)
        if not progressed:
            break
    return quotas


def _deterministic_hash_key(seed: str, *parts: str) -> str:
    joined = "|".join([seed, *parts])
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


def sample_locomo_items(
    locomo_csv: Path,
    *,
    n_target: int = 100,
    seed: str = "r6_pm4_conversational_kappa_n200_v1",
    preferred_arms: Sequence[str] = (
        "M4f_soft_telemetry_fallback",
        "M4g_ek5_soft_telemetry_recency",
        "M4e_minimal_pair_split",
        "M1_hybrid_flat",
    ),
) -> List[Dict[str, str]]:
    """Deterministically sample `n_target` unique LoCoMo questions stratified across all categories."""
    arm_set = set(preferred_arms)
    by_qid: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    with locomo_csv.open("r", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row.get("regime") != "regime_a_unmodified":
                continue
            if int(row.get("api_error", "0")) != 0:
                continue
            if arm_set and row.get("arm") not in arm_set:
                continue
            by_qid[row["qid"]].append(row)

    # Pick one deterministic representative arm per unique qid so all 100 questions are distinct.
    per_cat: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for qid in sorted(by_qid.keys()):
        candidates = sorted(
            by_qid[qid],
            key=lambda r: _deterministic_hash_key(seed, "locomo_arm", qid, r["arm"]),
        )
        chosen = candidates[0]
        per_cat[str(chosen["category"])].append(chosen)

    quotas = _allocate_stratified_quotas({k: len(v) for k, v in per_cat.items()}, n_target)
    selected: List[Dict[str, str]] = []
    for cat in sorted(per_cat.keys()):
        pool = sorted(
            per_cat[cat],
            key=lambda r: _deterministic_hash_key(seed, "locomo_pick", cat, r["qid"]),
        )
        for r in pool[: quotas[cat]]:
            selected.append(
                {
                    "item_id": f"LOC_{r['qid']}_{r['arm']}",
                    "dataset": "LoCoMo-300",
                    "stratum": f"cat_{cat}" + ("_abstention" if str(r.get("is_abstention")) == "1" else ""),
                    "qid": r["qid"],
                    "arm": r["arm"],
                    "gold_answer": r["gold"],
                    "candidate_answer": r["pred_raw"],
                    "r5_orig_is_correct": str(r.get("is_correct", "")),
                }
            )
    selected.sort(key=lambda x: _deterministic_hash_key(seed, "order", x["item_id"]))
    return selected


def sample_longmemeval_items(
    lme_csv: Path,
    *,
    n_target: int = 100,
    seed: str = "r6_pm4_conversational_kappa_n200_v1",
    preferred_arms: Sequence[str] = (
        "M4f_soft_telemetry_fallback",
        "M4g_ek5_soft_telemetry_recency",
        "M4e_minimal_pair_split",
        "M1_hybrid_flat",
    ),
) -> List[Dict[str, str]]:
    """Deterministically sample `n_target` unique LongMemEval-S items across all question types."""
    arm_set = set(preferred_arms)
    by_qid: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    with lme_csv.open("r", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row.get("regime") != "regime_a_clean_n500":
                continue
            if int(row.get("api_error", "0")) != 0:
                continue
            if arm_set and row.get("arm") not in arm_set:
                continue
            by_qid[row["qid"]].append(row)

    per_qtype: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for qid in sorted(by_qid.keys()):
        candidates = sorted(
            by_qid[qid],
            key=lambda r: _deterministic_hash_key(seed, "lme_arm", qid, r["arm"]),
        )
        chosen = candidates[0]
        per_qtype[str(chosen["question_type"])].append(chosen)

    quotas = _allocate_stratified_quotas({k: len(v) for k, v in per_qtype.items()}, n_target)
    selected: List[Dict[str, str]] = []
    for qtype in sorted(per_qtype.keys()):
        pool = sorted(
            per_qtype[qtype],
            key=lambda r: _deterministic_hash_key(seed, "lme_pick", qtype, r["qid"]),
        )
        for r in pool[: quotas[qtype]]:
            selected.append(
                {
                    "item_id": f"LME_{r['qid']}_{r['arm']}",
                    "dataset": "LongMemEval-S",
                    "stratum": qtype,
                    "qid": r["qid"],
                    "arm": r["arm"],
                    "gold_answer": r["gold"],
                    "candidate_answer": r["pred_raw"],
                    "r5_orig_is_correct": str(r.get("is_correct", "")),
                }
            )
    selected.sort(key=lambda x: _deterministic_hash_key(seed, "order", x["item_id"]))
    return selected


def sample_structured_r5b_items(
    matrix_csv: Path,
    *,
    n_target: int = 320,
    seed: str = "r6_pm4_conversational_kappa_n200_v1",
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Load all R5b structured rows and return `(stratified_n320, all_rows)`."""
    all_rows: List[Dict[str, Any]] = []
    per_cell: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    with matrix_csv.open("r", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            rec = {
                "case_id": row["case_id"],
                "family": row["family"],
                "arm": row["arm"],
                "generator": row["generator"],
                "judge_nemotron": int(row["judge_nemotron"]),
                "judge_qwen": int(row["judge_qwen"]),
            }
            all_rows.append(rec)
            cell_key = f"{rec['family']}|{rec['arm']}|{rec['generator']}"
            per_cell[cell_key].append(rec)

    quotas = _allocate_stratified_quotas({k: len(v) for k, v in per_cell.items()}, n_target)
    sampled: List[Dict[str, Any]] = []
    for cell_key in sorted(per_cell.keys()):
        pool = sorted(
            per_cell[cell_key],
            key=lambda r: _deterministic_hash_key(seed, "struct_pick", cell_key, r["case_id"]),
        )
        sampled.extend(pool[: quotas[cell_key]])
    return sampled, all_rows


def _wait_port(port: int, timeout_s: float = 12.0) -> bool:
    deadline = time.perf_counter() + timeout_s
    while time.perf_counter() < deadline:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(0.4)
            if s.connect_ex(("127.0.0.1", port)) == 0:
                return True
        time.sleep(0.2)
    return False


def start_ssh_tunnel(host: str, local_port: int, remote_port: int, user: str = "mattmre") -> subprocess.Popen:
    """Open a local-to-remote OpenSSH port forward with `stdin=subprocess.DEVNULL` (Windows-safe)."""
    cmd = [
        "ssh",
        "-N",
        "-o",
        "BatchMode=yes",
        "-o",
        "StrictHostKeyChecking=accept-new",
        "-o",
        "ExitOnForwardFailure=yes",
        "-L",
        f"127.0.0.1:{local_port}:127.0.0.1:{remote_port}",
        f"{user}@{host}",
    ]
    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if not _wait_port(local_port, timeout_s=12.0):
        proc.terminate()
        raise RuntimeError(f"Failed to establish SSH tunnel {user}@{host}:{remote_port} -> 127.0.0.1:{local_port}")
    return proc


def _call_nemotron_judge(
    endpoint: str,
    model_id: str,
    sys_prompt: str,
    user_prompt: str,
    *,
    max_tokens: int = 8,
    timeout: float = 60.0,
) -> Tuple[str, float, bool]:
    t0 = time.perf_counter()
    for attempt in range(3):
        try:
            r = requests.post(
                f"{endpoint.rstrip('/')}/chat/completions",
                json={
                    "model": model_id,
                    "messages": [
                        {"role": "system", "content": sys_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    "temperature": 0.0,
                    "max_tokens": max_tokens,
                    "chat_template_kwargs": {"enable_thinking": False},
                },
                timeout=timeout,
            )
            r.raise_for_status()
            payload = r.json()
            txt = (payload["choices"][0]["message"].get("content") or "").strip()
            return txt, (time.perf_counter() - t0) * 1000.0, False
        except Exception as exc:  # noqa: BLE001
            if attempt == 2:
                return f"ERROR: {exc}", (time.perf_counter() - t0) * 1000.0, True
            time.sleep(0.5 * (attempt + 1))
    return "ERROR: unreachable", 0.0, True


def _grade_with_local_qwen(
    prompts: Sequence[str],
    sys_prompt: str,
    hf_model: str,
    *,
    batch_size: int = 8,
    max_new_tokens: int = 8,
) -> Tuple[List[str], float, float]:
    """Run greedy `bfloat16` local RTX 3090 inference with `Qwen/Qwen2.5-7B-Instruct`."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    t_load0 = time.perf_counter()
    tok = AutoTokenizer.from_pretrained(hf_model)
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(hf_model, dtype=torch.bfloat16).to("cuda").eval()
    load_sec = time.perf_counter() - t_load0

    t_gen0 = time.perf_counter()
    formatted = [
        tok.apply_chat_template(
            [{"role": "system", "content": sys_prompt}, {"role": "user", "content": up}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for up in prompts
    ]

    outputs: List[str] = []
    for start in range(0, len(formatted), batch_size):
        chunk = formatted[start : start + batch_size]
        enc = tok(chunk, return_tensors="pt", padding=True, truncation=True, max_length=2048).to("cuda")
        input_len = enc.input_ids.shape[1]
        with torch.no_grad():
            gen = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tok.pad_token_id,
            )
        for row_idx in range(gen.shape[0]):
            gen_ids = gen[row_idx, input_len:]
            outputs.append(tok.decode(gen_ids, skip_special_tokens=True).strip())
        if (start + len(chunk)) % 40 == 0 or (start + len(chunk)) == len(formatted):
            print(f"[R6-PM4-Qwen] Graded {start + len(chunk)}/{len(formatted)} items on RTX 3090...", flush=True)

    gen_sec = time.perf_counter() - t_gen0
    del model
    torch.cuda.empty_cache()
    return outputs, load_sec, gen_sec


def run_r6_pm4_conversational_kappa(
    *,
    locomo_csv: Path,
    lme_csv: Path,
    structured_matrix_csv: Path,
    output_dir: Path,
    spark_host: str = "10.0.0.24",
    spark_port: int = 8801,
    local_tunnel_port: int = 18851,
    nemotron_model_id: str = "nemotron-lightning",
    qwen_hf_model: str = "Qwen/Qwen2.5-7B-Instruct",
    seed: str = "r6_pm4_conversational_kappa_n200_v1",
    n_bootstrap: int = 3000,
    max_workers: int = 8,
    qwen_batch_size: int = 8,
    direct_endpoint: Optional[str] = None,
) -> Dict[str, Any]:
    """Execute full Stage R6 PM-4 N=200 conversational + pooled N=520 calibration."""
    t0 = time.perf_counter()
    output_dir.mkdir(parents=True, exist_ok=True)

    loc_items = sample_locomo_items(locomo_csv, n_target=100, seed=seed)
    lme_items = sample_longmemeval_items(lme_csv, n_target=100, seed=seed)
    conv_items = loc_items + lme_items
    assert len(conv_items) == 200, f"Expected 200 items, got {len(conv_items)}"

    question_map = load_question_lookup()
    matched_questions = 0
    for it in conv_items:
        q_txt = question_map.get(f"{it['dataset']}::{it['qid']}", "")
        it["question"] = q_txt
        if q_txt:
            matched_questions += 1
    print(f"[R6-PM4] Loaded question texts for {matched_questions}/{len(conv_items)} conversational items.", flush=True)

    user_prompts = [
        build_symmetric_judge_user_prompt(
            it["gold_answer"],
            it["candidate_answer"],
            question=it["question"],
            dataset=it["dataset"],
            question_type=it["stratum"],
        )
        for it in conv_items
    ]

    # 1) Judge A: Nemotron-3.5-30B via unique SSH tunnel port 18851 -> 10.0.0.24:8801
    tunnel_proc: Optional[subprocess.Popen] = None
    endpoint = direct_endpoint
    if not endpoint:
        print(f"[R6-PM4] Opening SSH tunnel 127.0.0.1:{local_tunnel_port} -> {spark_host}:{spark_port}...", flush=True)
        tunnel_proc = start_ssh_tunnel(spark_host, local_tunnel_port, spark_port)
        endpoint = f"http://127.0.0.1:{local_tunnel_port}/v1"

    nem_raw: List[str] = [""] * len(conv_items)
    nem_lat: List[float] = [0.0] * len(conv_items)
    nem_err: List[bool] = [False] * len(conv_items)
    t_nem0 = time.perf_counter()
    try:
        print(f"[R6-PM4-Nemotron] Grading {len(conv_items)} open-conversational items via {endpoint}...", flush=True)

        def _worker(idx_up: Tuple[int, str]) -> Tuple[int, str, float, bool]:
            idx, up = idx_up
            txt, lat, err = _call_nemotron_judge(
                endpoint, nemotron_model_id, SYMMETRIC_CONV_JUDGE_SYSTEM_PROMPT, up, max_tokens=6
            )
            return idx, txt, lat, err

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
            done = 0
            for idx, txt, lat, err in pool.map(_worker, list(enumerate(user_prompts))):
                nem_raw[idx] = txt
                nem_lat[idx] = lat
                nem_err[idx] = err
                done += 1
                if done % 50 == 0 or done == len(conv_items):
                    print(f"[R6-PM4-Nemotron] Graded {done}/{len(conv_items)} items...", flush=True)
    finally:
        if tunnel_proc is not None:
            tunnel_proc.terminate()
            try:
                tunnel_proc.wait(timeout=5)
            except Exception:
                tunnel_proc.kill()
    nem_sec = time.perf_counter() - t_nem0

    # 2) Judge B: Local RTX 3090 Qwen/Qwen2.5-7B-Instruct (bfloat16)
    print(f"[R6-PM4-Qwen] Loading {qwen_hf_model} in bfloat16 on local RTX 3090...", flush=True)
    qwen_raw, qwen_load_sec, qwen_gen_sec = _grade_with_local_qwen(
        user_prompts,
        SYMMETRIC_CONV_JUDGE_SYSTEM_PROMPT,
        qwen_hf_model,
        batch_size=qwen_batch_size,
        max_new_tokens=6,
    )

    # 3) Parse verdicts & build per-item CSV rows
    rows_out: List[Dict[str, Any]] = []
    nem_grades: List[int] = []
    qwen_grades: List[int] = []
    nem_clean_parses = 0
    qwen_clean_parses = 0

    for idx, it in enumerate(conv_items):
        gn, clean_n = parse_binary_verdict(nem_raw[idx])
        gq, clean_q = parse_binary_verdict(qwen_raw[idx])
        nem_grades.append(gn)
        qwen_grades.append(gq)
        nem_clean_parses += int(clean_n)
        qwen_clean_parses += int(clean_q)
        rows_out.append(
            {
                "item_idx": idx + 1,
                "item_id": it["item_id"],
                "dataset": it["dataset"],
                "stratum": it["stratum"],
                "qid": it["qid"],
                "arm": it["arm"],
                "question": it.get("question", ""),
                "gold_answer": it["gold_answer"],
                "candidate_answer": it["candidate_answer"],
                "r5_orig_is_correct": it["r5_orig_is_correct"],
                "nemotron_raw": nem_raw[idx].replace("\n", " "),
                "nemotron_grade": gn,
                "nemotron_clean_parse": int(clean_n),
                "nemotron_latency_ms": round(nem_lat[idx], 2),
                "nemotron_api_error": int(nem_err[idx]),
                "qwen_raw": qwen_raw[idx].replace("\n", " "),
                "qwen_grade": gq,
                "qwen_clean_parse": int(clean_q),
                "agree": int(gn == gq),
            }
        )

    csv_path = output_dir / "r6_pm4_conversational_kappa_n200_rows.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows_out[0].keys()))
        writer.writeheader()
        writer.writerows(rows_out)

    # 4) Compute Cohen's kappa + 3,000-resample bootstrap 95% CI across splits and pooled N=520
    conv_stats = bootstrap_cohens_kappa(nem_grades, qwen_grades, n_bootstrap=n_bootstrap, seed=42)

    loc_idx = [i for i, r in enumerate(rows_out) if r["dataset"] == "LoCoMo-300"]
    lme_idx = [i for i, r in enumerate(rows_out) if r["dataset"] == "LongMemEval-S"]
    loc_stats = bootstrap_cohens_kappa(
        [nem_grades[i] for i in loc_idx],
        [qwen_grades[i] for i in loc_idx],
        n_bootstrap=n_bootstrap,
        seed=43,
    )
    lme_stats = bootstrap_cohens_kappa(
        [nem_grades[i] for i in lme_idx],
        [qwen_grades[i] for i in lme_idx],
        n_bootstrap=n_bootstrap,
        seed=44,
    )

    cluster_block = cluster_and_stratified_bootstrap_kappa(rows_out, n_bootstrap=n_bootstrap, seed=48)
    conv_stats["cluster_stratified_ci95"] = cluster_block["conversational_n200_cluster_ci95"]
    loc_stats["conv_cluster_ci95"] = cluster_block["locomo_n100_conv_cluster_ci95"]
    loc_stats["n_conversation_clusters"] = cluster_block["locomo_n_conversation_clusters"]
    lme_stats["stratified_ci95"] = cluster_block["longmemeval_n100_stratified_ci95"]
    by_stratum = cluster_block["by_stratum_with_ci95"]

    # Historical structured R5b matrix is kept STRICTLY SEPARATE (different rubric/domain; never pooled into v8 gate!)
    struct_320, struct_all = sample_structured_r5b_items(structured_matrix_csv, n_target=320, seed=seed)
    s320_nem = [r["judge_nemotron"] for r in struct_320]
    s320_qw = [r["judge_qwen"] for r in struct_320]
    struct_320_stats = bootstrap_cohens_kappa(s320_nem, s320_qw, n_bootstrap=n_bootstrap, seed=45)
    struct_800_stats = bootstrap_cohens_kappa(
        [r["judge_nemotron"] for r in struct_all],
        [r["judge_qwen"] for r in struct_all],
        n_bootstrap=n_bootstrap,
        seed=46,
    )

    disagreements_sample = [
        {
            "item_id": r["item_id"],
            "dataset": r["dataset"],
            "stratum": r["stratum"],
            "gold": r["gold_answer"][:120],
            "candidate": r["candidate_answer"][:140],
            "nemotron_grade": r["nemotron_grade"],
            "qwen_grade": r["qwen_grade"],
        }
        for r in rows_out
        if int(r["agree"]) == 0
    ]

    cluster_ci = conv_stats["cluster_stratified_ci95"] or conv_stats["ci95"]
    manifest = {
        "schema_version": "r6_pm4_conversational_kappa_n200_v2_isolated",
        "rubric_version": RUBRIC_VERSION,
        "rubric_system_prompt_sha256": sha256_text(SYMMETRIC_CONV_JUDGE_SYSTEM_PROMPT),
        "sampling_seed": seed,
        "git_sha": current_git_sha(REPO_ROOT),
        "timestamp": datetime.now().isoformat(),
        "wall_seconds": round(time.perf_counter() - t0, 2),
        "judges": {
            "judge_a_family": "NVIDIA Nemotron-3.5-Lightning-30B-A3B-NVFP4",
            "judge_a_model_id": nemotron_model_id,
            "judge_a_host": f"{spark_host}:{spark_port}",
            "judge_a_wall_seconds": round(nem_sec, 2),
            "judge_a_api_errors": int(sum(nem_err)),
            "judge_a_clean_parses": f"{nem_clean_parses}/{len(conv_items)}",
            "judge_b_family": "Alibaba Qwen2.5-7B-Instruct (local RTX 3090 bfloat16)",
            "judge_b_model_id": qwen_hf_model,
            "judge_b_load_seconds": round(qwen_load_sec, 2),
            "judge_b_gen_seconds": round(qwen_gen_sec, 2),
            "judge_b_clean_parses": f"{qwen_clean_parses}/{len(conv_items)}",
            "self_judge_caveat": "Nemotron-30B also generated the R5 conversational candidate answers; Qwen2.5-7B provides the independent cross-family check, and human PM-4b remains required.",
        },
        "source_inputs": {
            "locomo_csv": str(locomo_csv.relative_to(REPO_ROOT)).replace("\\", "/"),
            "locomo_csv_sha256": sha256_file(locomo_csv),
            "longmemeval_csv": str(lme_csv.relative_to(REPO_ROOT)).replace("\\", "/"),
            "longmemeval_csv_sha256": sha256_file(lme_csv),
            "structured_r5b_csv": str(structured_matrix_csv.relative_to(REPO_ROOT)).replace("\\", "/"),
            "structured_r5b_csv_sha256": sha256_file(structured_matrix_csv),
        },
        "sampling_distribution": {
            "locomo_strata": dict(sorted(Counter(it["stratum"] for it in loc_items).items())),
            "longmemeval_strata": dict(sorted(Counter(it["stratum"] for it in lme_items).items())),
        },
        "conversational_n200": conv_stats,
        "locomo_n100": loc_stats,
        "longmemeval_n100": lme_stats,
        "by_stratum": by_stratum,
        "historical_r5b_structured_separate_rubric_do_not_pool": {
            "note": "R5b structured Zero-Shortcut items were graded under the R5b structured rubric, NOT rubric v8; reported separately and excluded from conversational PM-4 gates.",
            "structured_n320": struct_320_stats,
            "structured_n800": struct_800_stats,
        },
        "baseline_comparison_vs_r5b": {
            "r5b_real_conv_n100_kappa": 0.7414,
            "r5b_real_conv_n100_raw_agreement": 0.87,
            "r6_conv_n200_kappa_delta_vs_r5b": round(conv_stats["cohens_kappa"] - 0.7414, 4),
            "cohort_disclosure": "Cross-cohort comparison (R5b N=100 vs R6 N=200 stratified sample); with balanced ~0.50 YES marginals, kappa closely tracks 2*p_o - 1.",
        },
        "gate_checks": {
            "conv_n200_point_kappa_ge_080": bool(conv_stats["cohens_kappa"] >= 0.80),
            "conv_n200_iid_ci95_lcb_ge_080": bool(conv_stats["ci95"][0] >= 0.80),
            "conv_n200_cluster_ci95_lcb_ge_080": bool(cluster_ci[0] >= 0.80),
            "zero_api_errors": bool(sum(nem_err) == 0),
            "all_200_strict_unambiguous_parses": bool(nem_clean_parses == 200 and qwen_clean_parses == 200),
            "symmetric_identical_prompt": True,
        },
        "disagreements_count": len(disagreements_sample),
        "disagreements_preview": disagreements_sample[:25],
        "rows_csv": csv_path.name,
        "rows_csv_sha256": sha256_file(csv_path),
        "telemetry": process_telemetry(),
    }

    man_path = output_dir / "r6_pm4_conversational_kappa_manifest.json"
    write_json_atomic(man_path, manifest)

    receipt_path = output_dir / "report_receipt.json"
    write_json_atomic(
        receipt_path,
        {
            "schema_version": "r6_pm4_report_receipt_v2",
            "git_sha": manifest["git_sha"],
            "manifest_path": man_path.name,
            "manifest_sha256": sha256_file(man_path),
            "artifact_hashes": {
                man_path.name: sha256_file(man_path),
                csv_path.name: manifest["rows_csv_sha256"],
            },
            "wall_seconds": manifest["wall_seconds"],
        },
    )

    print(
        f"[R6-PM4] COMPLETE in {manifest['wall_seconds']}s | "
        f"Conv N=200 kappa={conv_stats['cohens_kappa']:.4f} (iid CI {conv_stats['ci95']}, cluster CI {cluster_ci}, raw={conv_stats['raw_agreement']:.4f}) | "
        f"Point >=0.80: {manifest['gate_checks']['conv_n200_point_kappa_ge_080']} | CI95 LCB >=0.80: {manifest['gate_checks']['conv_n200_cluster_ci95_lcb_ge_080']}",
        flush=True,
    )
    return manifest


def recompute_pm4_manifest_from_existing_csv(
    csv_path: Path,
    *,
    locomo_csv: Path,
    lme_csv: Path,
    structured_matrix_csv: Path,
    output_dir: Path,
    seed: str = "r6_pm4_conversational_kappa_n200_v1",
    n_bootstrap: int = 3000,
) -> Dict[str, Any]:
    """Re-parse raw judge outputs under hardened `parse_binary_verdict` and compute cluster + stratum CIs offline."""
    t0 = time.perf_counter()
    prev_man_path = output_dir / "r6_pm4_conversational_kappa_manifest.json"
    prev_man = json.loads(prev_man_path.read_text(encoding="utf-8")) if prev_man_path.is_file() else {}
    prev_judges = dict(prev_man.get("judges") or {})

    rows_in = list(csv.DictReader(csv_path.open("r", encoding="utf-8")))
    assert len(rows_in) == 200, f"Expected 200 rows in {csv_path}, got {len(rows_in)}"

    nem_grades: List[int] = []
    qwen_grades: List[int] = []
    nem_clean = 0
    qwen_clean = 0
    nem_err = 0
    rows_rechecked: List[Dict[str, Any]] = []
    for r in rows_in:
        gn, cn = parse_binary_verdict(r["nemotron_raw"])
        gq, cq = parse_binary_verdict(r["qwen_raw"])
        nem_grades.append(gn)
        qwen_grades.append(gq)
        nem_clean += int(cn)
        qwen_clean += int(cq)
        nem_err += int(r.get("nemotron_api_error", 0))
        rec = dict(r)
        rec["nemotron_grade"] = gn
        rec["nemotron_clean_parse"] = int(cn)
        rec["qwen_grade"] = gq
        rec["qwen_clean_parse"] = int(cq)
        rec["agree"] = int(gn == gq)
        rows_rechecked.append(rec)

    conv_stats = bootstrap_cohens_kappa(nem_grades, qwen_grades, n_bootstrap=n_bootstrap, seed=42)
    loc_idx = [i for i, r in enumerate(rows_rechecked) if r["dataset"] == "LoCoMo-300"]
    lme_idx = [i for i, r in enumerate(rows_rechecked) if r["dataset"] == "LongMemEval-S"]
    loc_stats = bootstrap_cohens_kappa([nem_grades[i] for i in loc_idx], [qwen_grades[i] for i in loc_idx], n_bootstrap=n_bootstrap, seed=43)
    lme_stats = bootstrap_cohens_kappa([nem_grades[i] for i in lme_idx], [qwen_grades[i] for i in lme_idx], n_bootstrap=n_bootstrap, seed=44)

    cluster_block = cluster_and_stratified_bootstrap_kappa(rows_rechecked, n_bootstrap=n_bootstrap, seed=48)
    conv_stats["cluster_stratified_ci95"] = cluster_block["conversational_n200_cluster_ci95"]
    loc_stats["conv_cluster_ci95"] = cluster_block["locomo_n100_conv_cluster_ci95"]
    loc_stats["n_conversation_clusters"] = cluster_block["locomo_n_conversation_clusters"]
    lme_stats["stratified_ci95"] = cluster_block["longmemeval_n100_stratified_ci95"]

    struct_320, struct_all = sample_structured_r5b_items(structured_matrix_csv, n_target=320, seed=seed)
    struct_320_stats = bootstrap_cohens_kappa(
        [r["judge_nemotron"] for r in struct_320], [r["judge_qwen"] for r in struct_320], n_bootstrap=n_bootstrap, seed=45
    )
    struct_800_stats = bootstrap_cohens_kappa(
        [r["judge_nemotron"] for r in struct_all], [r["judge_qwen"] for r in struct_all], n_bootstrap=n_bootstrap, seed=46
    )

    disagreements_sample = [
        {
            "item_id": r["item_id"],
            "dataset": r["dataset"],
            "stratum": r["stratum"],
            "gold": r["gold_answer"][:120],
            "candidate": r["candidate_answer"][:140],
            "nemotron_grade": r["nemotron_grade"],
            "qwen_grade": r["qwen_grade"],
        }
        for r in rows_rechecked
        if int(r["agree"]) == 0
    ]
    cluster_ci = conv_stats["cluster_stratified_ci95"] or conv_stats["ci95"]
    manifest = {
        "schema_version": "r6_pm4_conversational_kappa_n200_v2_isolated",
        "rubric_version": RUBRIC_VERSION,
        "rubric_system_prompt_sha256": sha256_text(SYMMETRIC_CONV_JUDGE_SYSTEM_PROMPT),
        "sampling_seed": seed,
        "git_sha": current_git_sha(REPO_ROOT),
        "timestamp": datetime.now().isoformat(),
        "wall_seconds": prev_man.get("wall_seconds", round(time.perf_counter() - t0, 2)),
        "recompute_wall_seconds": round(time.perf_counter() - t0, 3),
        "judges": {
            **prev_judges,
            "judge_a_api_errors": nem_err,
            "judge_a_clean_parses": f"{nem_clean}/200",
            "judge_b_clean_parses": f"{qwen_clean}/200",
            "self_judge_caveat": "Nemotron-30B also generated the R5 conversational candidate answers; Qwen2.5-7B provides the independent cross-family check, and human PM-4b remains required.",
        },
        "source_inputs": {
            "locomo_csv": str(locomo_csv.relative_to(REPO_ROOT)).replace("\\", "/"),
            "locomo_csv_sha256": sha256_file(locomo_csv),
            "longmemeval_csv": str(lme_csv.relative_to(REPO_ROOT)).replace("\\", "/"),
            "longmemeval_csv_sha256": sha256_file(lme_csv),
            "structured_r5b_csv": str(structured_matrix_csv.relative_to(REPO_ROOT)).replace("\\", "/"),
            "structured_r5b_csv_sha256": sha256_file(structured_matrix_csv),
        },
        "sampling_distribution": {
            "locomo_strata": dict(sorted(Counter(r["stratum"] for r in rows_rechecked if r["dataset"] == "LoCoMo-300").items())),
            "longmemeval_strata": dict(sorted(Counter(r["stratum"] for r in rows_rechecked if r["dataset"] == "LongMemEval-S").items())),
        },
        "conversational_n200": conv_stats,
        "locomo_n100": loc_stats,
        "longmemeval_n100": lme_stats,
        "by_stratum": cluster_block["by_stratum_with_ci95"],
        "historical_r5b_structured_separate_rubric_do_not_pool": {
            "note": "R5b structured Zero-Shortcut items were graded under the R5b structured rubric, NOT rubric v8; reported separately and excluded from conversational PM-4 gates.",
            "structured_n320": struct_320_stats,
            "structured_n800": struct_800_stats,
        },
        "baseline_comparison_vs_r5b": {
            "r5b_real_conv_n100_kappa": 0.7414,
            "r5b_real_conv_n100_raw_agreement": 0.87,
            "r6_conv_n200_kappa_delta_vs_r5b": round(conv_stats["cohens_kappa"] - 0.7414, 4),
            "cohort_disclosure": "Cross-cohort comparison (R5b N=100 vs R6 N=200 stratified sample); with balanced ~0.50 YES marginals, kappa closely tracks 2*p_o - 1.",
        },
        "gate_checks": {
            "conv_n200_point_kappa_ge_080": bool(conv_stats["cohens_kappa"] >= 0.80),
            "conv_n200_iid_ci95_lcb_ge_080": bool(conv_stats["ci95"][0] >= 0.80),
            "conv_n200_cluster_ci95_lcb_ge_080": bool(cluster_ci[0] >= 0.80),
            "zero_api_errors": bool(nem_err == 0),
            "all_200_strict_unambiguous_parses": bool(nem_clean == 200 and qwen_clean == 200),
            "symmetric_identical_prompt": True,
        },
        "disagreements_count": len(disagreements_sample),
        "disagreements_preview": disagreements_sample[:25],
        "rows_csv": csv_path.name,
        "rows_csv_sha256": sha256_file(csv_path),
        "telemetry": process_telemetry(),
    }
    man_path = output_dir / "r6_pm4_conversational_kappa_manifest.json"
    write_json_atomic(man_path, manifest)
    write_json_atomic(
        output_dir / "report_receipt.json",
        {
            "schema_version": "r6_pm4_report_receipt_v2",
            "git_sha": manifest["git_sha"],
            "manifest_path": man_path.name,
            "manifest_sha256": sha256_file(man_path),
            "artifact_hashes": {
                man_path.name: sha256_file(man_path),
                csv_path.name: manifest["rows_csv_sha256"],
            },
            "wall_seconds": manifest["wall_seconds"],
        },
    )
    print(
        f"[R6-PM4-Recompute] kappa={conv_stats['cohens_kappa']:.4f} | iid_CI={conv_stats['ci95']} | cluster_CI={cluster_ci} | clean={nem_clean}/200,{qwen_clean}/200",
        flush=True,
    )
    return manifest


def _resolve_default_paths() -> Tuple[Path, Path, Path]:
    r5_base = REPO_ROOT / "artifacts" / "flagship1-night1" / "20261122-overnight-r5" / "spark6724"
    r5b_base = REPO_ROOT / "artifacts" / "flagship1-night1" / "20261122-r5b-completion" / "spark6724"
    loc = r5_base / "r5_s2_locomo_300_em4" / "r5_s2_locomo_300_em4_rows.csv"
    lme = r5_base / "r5_s3_longmemeval_s_full_n500" / "r5_s3_longmemeval_n500_rows.csv"
    mat = r5b_base / "r5b_s4_pm4_cross_family_2x2" / "r5b_s4_pm4_2x2_matrix_rows.csv"
    return loc, lme, mat


def main() -> None:
    loc_def, lme_def, mat_def = _resolve_default_paths()
    parser = argparse.ArgumentParser(description="Stage R6 PM-4 Open-Conversational Cross-Family Judge Calibration (N=200)")
    parser.add_argument("--locomo-csv", type=Path, default=loc_def)
    parser.add_argument("--lme-csv", type=Path, default=lme_def)
    parser.add_argument("--structured-csv", type=Path, default=mat_def)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "artifacts" / "r6_pm4_conversational_kappa_n200",
    )
    parser.add_argument("--spark-host", type=str, default="10.0.0.24")
    parser.add_argument("--spark-port", type=int, default=8801)
    parser.add_argument("--local-tunnel-port", type=int, default=18851)
    parser.add_argument("--nemotron-model-id", type=str, default="nemotron-lightning")
    parser.add_argument("--qwen-hf-model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--seed", type=str, default="r6_pm4_conversational_kappa_n200_v1")
    parser.add_argument("--n-bootstrap", type=int, default=3000)
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--qwen-batch-size", type=int, default=8)
    parser.add_argument("--direct-endpoint", type=str, default=None)
    parser.add_argument("--recompute-from-csv", type=Path, default=None)
    args = parser.parse_args()

    if args.recompute_from_csv is not None:
        recompute_pm4_manifest_from_existing_csv(
            args.recompute_from_csv,
            locomo_csv=args.locomo_csv,
            lme_csv=args.lme_csv,
            structured_matrix_csv=args.structured_csv,
            output_dir=args.output_dir,
            seed=args.seed,
            n_bootstrap=args.n_bootstrap,
        )
        return

    run_r6_pm4_conversational_kappa(
        locomo_csv=args.locomo_csv,
        lme_csv=args.lme_csv,
        structured_matrix_csv=args.structured_csv,
        output_dir=args.output_dir,
        spark_host=args.spark_host,
        spark_port=args.spark_port,
        local_tunnel_port=args.local_tunnel_port,
        nemotron_model_id=args.nemotron_model_id,
        qwen_hf_model=args.qwen_hf_model,
        seed=args.seed,
        n_bootstrap=args.n_bootstrap,
        max_workers=args.max_workers,
        qwen_batch_size=args.qwen_batch_size,
        direct_endpoint=args.direct_endpoint,
    )


if __name__ == "__main__":
    main()
