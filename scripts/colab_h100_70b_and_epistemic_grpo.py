r"""Colab H100 80GB (Google AI Ultra) 24h Package (Wave-4 J1 Hardened):
  Part 1 — Frontier `70B/72B` Evaluation (`Qwen2.5-72B-Instruct-AWQ` / `Llama-3.3-70B-Instruct`)
           on `prospective_realworld_drift_corpus.py` (`Bare HybridRAG` vs `Chelated v2 EK5 v3`).
  Part 2 — Epistemic RLVR (`TRL GRPOTrainer + LoRA`) on `Qwen/Qwen2.5-7B-Instruct` wired directly
           to `ek5_wfs_rlvr_env.py` with entity-, value-, scenario/template-, and seed-disjoint
           `TRAIN / DEV / EVAL` splits, canonical exact-match / closed-lexicon `ABSTAIN` scoring,
           negation & mixed-commitment rejection, and gated format bonus `<= 0.05` paid ONLY when
           valid parse AND core reward `== 1.0`.

Usage on Google Colab H100 80GB:
  !pip install -q vllm trl peft accelerate datasets bitsandbytes sentence-transformers
  !python scripts/colab_h100_70b_and_epistemic_grpo.py --mode all --output-dir artifacts/colab_h100_70b_grpo

Dry-run split-disjointness & anti-hacking verifier check (CPU / local):
  python scripts/colab_h100_70b_and_epistemic_grpo.py --mode verify-only --output-dir /tmp/colab_verify
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import json
import math
import os
import re
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Set, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import ek5_wfs_rlvr_env as rlvr_env  # noqa: E402
from ek5_wfs_rlvr_env import (  # noqa: E402
    ABSTAIN,
    LABEL_F,
    LABEL_T,
    LABEL_U,
    EpistemicEpisode,
    audit_episode,
    oracle_completion,
    score_epistemic_completion_detailed,
)
from prospective_realworld_drift_corpus import (  # noqa: E402
    build_prospective_realworld_drift_suite,
)
from scripts.night1_common import (  # noqa: E402
    current_git_sha,
    paired_bootstrap_delta,
    write_json_atomic,
)

# ---------------------------------------------------------------------------
# Shared System Prompts & Strict Canonical Epistemic Scoring
# ---------------------------------------------------------------------------
ENTERPRISE_QA_SYSTEM_PROMPT = (
    "You are an enterprise epistemic verification assistant. Answer the question strictly using ONLY "
    "the dated records below.\n"
    "- Output ONLY the exact current verified value/identifier (no extra explanation).\n"
    "- Higher-authority official filings, signed advisories, and postmortems supersede earlier "
    "preliminary notes and unverified forum/social rumors.\n"
    "- If equal-authority official sources are in unresolved mutual conflict (epistemic state U / CONTESTED) "
    "or no verified record is admitted, reply ONLY with 'ABSTAIN'."
)

GRPO_SYSTEM_PROMPT = (
    "You are an epistemic reasoning model grounded in 3-valued Well-Founded Semantics (T=Admitted, "
    "F=Refuted/Superseded, U=Contested). Given the multi-agent evidence log, reason briefly inside "
    "<think>...</think>, then output EXACTLY two lines after </think>:\n"
    "VERDICT: T | F | U\n"
    "ANSWER: <exact unique current value> | ABSTAIN"
)

_ANSWER_TAG_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.IGNORECASE | re.DOTALL)
_THINK_CLOSED_RE = re.compile(r"^\s*<think>(.+?)</think>\s*(.+)$", re.IGNORECASE | re.DOTALL)
_VERDICT_LINE_RE = re.compile(r"^\s*VERDICT\s*:\s*(T|F|U)\s*$", re.IGNORECASE | re.MULTILINE)
_ANSWER_LINE_RE = re.compile(r"^\s*ANSWER\s*:\s*([^\r\n]+?)\s*$", re.IGNORECASE | re.MULTILINE)
_NEGATION_RE = re.compile(
    r"\b(?:not|never|no\s+longer|other\s+than|except|cannot\s+be|isn't|is\s+not|neither|nor)\b",
    re.IGNORECASE,
)
_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9_.\-/]*")
_CLOSED_ABSTAIN_TOKENS = {"abstain"}

# ---------------------------------------------------------------------------
# Disjoint Split Specifications (Entity x Slot x Value x Scenario x Seed)
# ---------------------------------------------------------------------------
SPLIT_SLOT_POOLS: Dict[str, Tuple[Tuple[str, str, Tuple[str, ...]], ...]] = {
    "TRAIN": rlvr_env.SLOT_POOL[:4],
    "DEV": rlvr_env.SLOT_POOL[4:6],
    "EVAL": rlvr_env.SLOT_POOL[6:8],
}

SPLIT_SCENARIOS: Dict[str, Tuple[Tuple[str, Tuple[str, ...]], ...]] = {
    "TRAIN": (
        ("conflict_refute", (LABEL_T, LABEL_F)),
        ("supersede_chain", (LABEL_T, LABEL_F)),
        ("sybil_corroboration", (LABEL_T, LABEL_F)),
        ("peer_deadlock", (LABEL_U,)),
        ("silent_multi_current", (LABEL_U,)),
    ),
    "DEV": (
        ("bitemporal_as_of", (LABEL_T, LABEL_F)),
        ("cross_domain_standoff", (LABEL_U,)),
    ),
    "EVAL": (
        ("auditor_cites_poison", (LABEL_T, LABEL_F)),
        ("summarizer_laundering", (LABEL_T, LABEL_F)),
        ("odd_refutation_cycle", (LABEL_U,)),
        ("contested_successor", (LABEL_U,)),
    ),
}

SPLIT_SEEDS: Dict[str, int] = {
    "TRAIN": 101001,
    "DEV": 202002,
    "EVAL": 303003,
}


@contextlib.contextmanager
def _scoped_rlvr_split(split_name: str) -> Iterator[Tuple[Tuple[str, str, str], ...]]:
    """Scope `ek5_wfs_rlvr_env` globals so target AND distractor slots/scenarios stay strictly inside `split_name`."""
    orig_pool = rlvr_env.SLOT_POOL
    orig_fams = rlvr_env.FAMILIES
    orig_dist = rlvr_env._DISTRACTOR_SCENARIOS
    split_fams = tuple(
        (f"{sc}:{lab}", sc, lab)
        for sc, labs in SPLIT_SCENARIOS[split_name]
        for lab in labs
    )
    # Exclude bitemporal_as_of from distractor helper since distractors do not set global episode as_of
    split_dist = tuple(
        sc for sc, _labs in SPLIT_SCENARIOS[split_name] if sc != "bitemporal_as_of"
    )
    try:
        rlvr_env.SLOT_POOL = SPLIT_SLOT_POOLS[split_name]
        rlvr_env.FAMILIES = split_fams
        rlvr_env._DISTRACTOR_SCENARIOS = split_dist
        yield split_fams
    finally:
        rlvr_env.SLOT_POOL = orig_pool
        rlvr_env.FAMILIES = orig_fams
        rlvr_env._DISTRACTOR_SCENARIOS = orig_dist


def build_disjoint_rlvr_split(split_name: str, n_episodes: int, seed: Optional[int] = None) -> List[EpistemicEpisode]:
    """Generate `n_episodes` oracle-audited WFS episodes strictly confined to `split_name`'s disjoint subspace."""
    if split_name not in SPLIT_SLOT_POOLS:
        raise KeyError(f"Unknown split {split_name!r}; expected one of {sorted(SPLIT_SLOT_POOLS)}")
    use_seed = SPLIT_SEEDS[split_name] if seed is None else int(seed)
    with _scoped_rlvr_split(split_name) as split_fams:
        episodes: List[EpistemicEpisode] = []
        for idx in range(n_episodes):
            fam_name = split_fams[idx % len(split_fams)][0]
            ep = rlvr_env.generate_episode(idx, seed=use_seed, family=fam_name)
            aud = audit_episode(ep)
            mismatches = sum(v for k, v in aud.items() if k != "nodes_checked")
            if mismatches != 0:
                raise RuntimeError(f"Oracle mismatch in {ep.episode_id}: {aud}")
            episodes.append(ep)
    return episodes


def _episode_footprint(episodes: Sequence[EpistemicEpisode]) -> Dict[str, Set[str]]:
    entities: Set[str] = set()
    slots: Set[str] = set()
    values: Set[str] = set()
    scenarios: Set[str] = set()
    families: Set[str] = set()
    prompt_shas: Set[str] = set()
    for ep in episodes:
        scenarios.add(ep.scenario)
        families.add(ep.family)
        prompt_shas.add(hashlib.sha256(ep.prompt.encode("utf-8")).hexdigest())
        for ev in ep.events:
            if ev.entity == "noise-retractor":
                continue
            entities.add(ev.entity.casefold())
            slots.add(ev.slot.casefold())
            values.add(ev.claim_value.strip().casefold())
    return {
        "entities": entities,
        "slots": slots,
        "values": values,
        "scenarios": scenarios,
        "families": families,
        "prompt_shas": prompt_shas,
    }


def assert_pairwise_disjoint_splits(splits: Dict[str, Sequence[EpistemicEpisode]]) -> Dict[str, Any]:
    """Hash-assert zero overlap on entities, slots, values, scenarios, families, and prompts across splits."""
    footprints = {k: _episode_footprint(v) for k, v in splits.items()}
    names = sorted(footprints.keys())
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            for dim in ("entities", "slots", "values", "scenarios", "families", "prompt_shas"):
                inter = footprints[a][dim] & footprints[b][dim]
                if inter:
                    raise AssertionError(f"Split leakage between {a} and {b} on {dim}: {sorted(inter)[:5]}")
    return {
        k: {
            "n_episodes": len(splits[k]),
            "n_entities": len(fp["entities"]),
            "n_values": len(fp["values"]),
            "scenarios": sorted(fp["scenarios"]),
            "split_prompt_bundle_sha256": hashlib.sha256(
                "\n".join(sorted(fp["prompt_shas"])).encode("utf-8")
            ).hexdigest(),
        }
        for k, fp in footprints.items()
    }


def reference_cot_for_episode(ep: EpistemicEpisode) -> str:
    """Build deterministic SFT warm-start trajectory using ONLY observable store slot resolution."""
    ans = ep.gold_value if ep.gold_value is not None else ABSTAIN
    return (
        f"<think>Target {ep.entity}.{ep.slot} resolves to status={ep.store_slot_status} "
        f"({ep.store_slot_reason}); query [{ep.query_event_id}] proposed={ep.proposed_value} "
        f"-> verdict={ep.gold_label}.</think>\n"
        f"VERDICT: {ep.gold_label}\n"
        f"ANSWER: {ans}"
    )


def extract_answer_span(completion: str) -> str:
    """Extract `<answer>...</answer>` payload if present, else stripped final line."""
    txt = (completion or "").strip()
    m = _ANSWER_TAG_RE.search(txt)
    if m:
        return m.group(1).strip()
    if "</think>" in txt.lower():
        txt = txt[txt.lower().rfind("</think>") + len("</think>") :].strip()
    lines = [ln.strip() for ln in txt.splitlines() if ln.strip()]
    return lines[-1] if lines else ""


def score_enterprise_prediction(pred: str, gold: str, stale_values: Sequence[str]) -> bool:
    """Canonical enterprise evaluation scorer (rejects negations, stale leakage, and mixed abstain+value)."""
    p = extract_answer_span(pred).strip().strip("`\"'.,; ")
    p_lo = p.casefold()
    g_lo = gold.strip().casefold()
    if not p_lo or _NEGATION_RE.search(p_lo):
        return False
    has_stale = any(sv.strip().casefold() in p_lo for sv in stale_values if sv.strip())
    is_abstain = p_lo in _CLOSED_ABSTAIN_TOKENS
    if g_lo == "abstain":
        return bool(is_abstain and not has_stale)
    if is_abstain or "abstain" in p_lo or has_stale:
        return False
    return bool(p_lo == g_lo)


def _strict_score_single_rlvr(completion_text: Any, episode_payload: Any) -> Dict[str, Any]:
    """Strict canonical WFS verifier rejecting unclosed `<think>`, negations, non-canonical abstain, and mixed commitments."""
    ep = rlvr_env._coerce_episode(episode_payload)
    raw = (
        "".join(str(m.get("content", "")) for m in completion_text if isinstance(m, dict))
        if isinstance(completion_text, list)
        else str(completion_text or "")
    )
    low = raw.lower()
    if "<think>" in low and "</think>" not in low:
        return {"reward": -1.0, "exact_match": 0.0, "valid_format": False, "outcome": "unclosed_think"}

    post = raw[low.rfind("</think>") + len("</think>") :].strip() if "</think>" in low else raw.strip()
    v_lines = _VERDICT_LINE_RE.findall(post)
    a_lines = _ANSWER_LINE_RE.findall(post)
    if len(v_lines) != 1 or len(a_lines) != 1:
        return {"reward": -1.0, "exact_match": 0.0, "valid_format": False, "outcome": "malformed_lines"}

    ans_raw = a_lines[0].strip().strip("*`\"' ").rstrip(".;,").strip("*`\"' ")
    ans_lo = ans_raw.casefold()
    if not ans_lo or _NEGATION_RE.search(ans_raw):
        return {"reward": -1.0, "exact_match": 0.0, "valid_format": False, "outcome": "negated_or_empty"}

    cand_hits = [
        cv for cv in ep.candidate_values if cv.casefold() in ans_lo
    ]
    cand_hits = [
        v for v in cand_hits if not any(v != w and v.casefold() in w.casefold() for w in cand_hits)
    ]
    has_abstain_word = "abstain" in ans_lo
    # Reject mixed commitment (`ABSTAIN or val`, or multiple distinct candidate values)
    if (has_abstain_word and ans_lo not in _CLOSED_ABSTAIN_TOKENS) or (has_abstain_word and cand_hits) or len(cand_hits) > 1:
        return {"reward": -1.0, "exact_match": 0.0, "valid_format": False, "outcome": "mixed_commitment"}
    # Closed-lexicon ABSTAIN: reject informal refusal synonyms (`unknown`, `unresolved`, `none`, `n/a`)
    if ans_lo in {"unknown", "unresolved", "none", "n/a", "contested", "undetermined"}:
        return {"reward": -1.0, "exact_match": 0.0, "valid_format": False, "outcome": "non_canonical_abstain"}

    canonical_post = f"VERDICT: {v_lines[0].upper()}\nANSWER: {'ABSTAIN' if ans_lo == 'abstain' else ans_raw}"
    det = score_epistemic_completion_detailed(canonical_post, ep)
    core_reward = float(det["reward"])
    has_clean_think = bool(_THINK_CLOSED_RE.match(raw)) and post == "\n".join(
        ln.strip() for ln in post.splitlines() if ln.strip()
    ) and len([ln for ln in post.splitlines() if ln.strip()]) == 2
    exact_match = 1.0 if abs(core_reward - 1.0) < 1e-9 else 0.0
    return {
        "reward": core_reward,
        "exact_match": exact_match,
        "valid_format": bool(has_clean_think),
        "outcome": det["outcome"],
    }


def ek5_epistemic_verifiability_reward(
    completions: Sequence[Any],
    episode: Sequence[Any],
    **_: Any,
) -> List[float]:
    """Deterministic EK5 v3 3-valued (`T / F / U`) verifiable reward wired to `ek5_wfs_rlvr_env.py`."""
    return [_strict_score_single_rlvr(c, ep)["reward"] for c, ep in zip(completions, episode)]


def format_epistemic_tags_reward(
    completions: Sequence[Any],
    episode: Sequence[Any],
    **_: Any,
) -> List[float]:
    """Gated `<= 0.05` structural bonus paid ONLY when valid `<think>+VERDICT+ANSWER` parse AND `core_reward == 1.0`."""
    bonuses: List[float] = []
    for c, ep in zip(completions, episode):
        sc = _strict_score_single_rlvr(c, ep)
        bonuses.append(0.05 if (sc["valid_format"] and sc["exact_match"] == 1.0) else 0.0)
    return bonuses


# ---------------------------------------------------------------------------
# Lightweight Self-Contained BM25 Retriever (for 70B Bare HybridRAG arm)
# ---------------------------------------------------------------------------
class CompactBM25:
    def __init__(self, texts: Sequence[str], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1, self.b = k1, b
        self.n = len(texts)
        self.tfs: List[Dict[str, int]] = []
        self.dls = np.zeros(self.n, dtype=np.float32)
        df: Dict[str, int] = defaultdict(int)
        for i, t in enumerate(texts):
            toks = _TOKEN_RE.findall(t.lower())
            self.dls[i] = float(len(toks))
            tf: Dict[str, int] = defaultdict(int)
            for tok in toks:
                tf[tok] += 1
            self.tfs.append(dict(tf))
            for tok in tf:
                df[tok] += 1
        self.avgdl = float(np.mean(self.dls)) if self.n else 1.0
        self.idf = {tok: float(math.log(1.0 + (self.n - c + 0.5) / (c + 0.5))) for tok, c in df.items()}

    def top_k(self, query: str, k: int = 5) -> List[int]:
        q_toks = _TOKEN_RE.findall(query.lower())
        scores = np.zeros(self.n, dtype=np.float32)
        for qt in q_toks:
            idf = self.idf.get(qt)
            if idf is None:
                continue
            for i in range(self.n):
                tf = self.tfs[i].get(qt, 0)
                if tf:
                    denom = tf + self.k1 * (1.0 - self.b + self.b * (self.dls[i] / self.avgdl))
                    scores[i] += idf * ((tf * (self.k1 + 1.0)) / denom)
        return np.argsort(-scores)[:k].tolist()


# ---------------------------------------------------------------------------
# Part 1: 70B / 72B Evaluation on Prospective Real-World Drift Corpus
# ---------------------------------------------------------------------------
def _build_70b_generator(model_id: str, quantization: Optional[str] = "awq") -> Callable[[List[Tuple[str, str]]], List[str]]:
    from vllm import LLM, SamplingParams

    llm_kwargs: Dict[str, Any] = {
        "model": model_id,
        "tensor_parallel_size": 1,
        "dtype": "bfloat16",
        "gpu_memory_utilization": 0.90,
        "max_model_len": 8192,
        "trust_remote_code": True,
    }
    if quantization and quantization.lower() != "none":
        llm_kwargs["quantization"] = quantization

    llm = LLM(**llm_kwargs)
    tokenizer = llm.get_tokenizer()
    sampling = SamplingParams(temperature=0.0, max_tokens=48)

    def _run_batch(pairs: List[Tuple[str, str]]) -> List[str]:
        prompts = [
            tokenizer.apply_chat_template(
                [{"role": "system", "content": sp}, {"role": "user", "content": up}],
                tokenize=False,
                add_generation_prompt=True,
            )
            for sp, up in pairs
        ]
        outs = llm.generate(prompts, sampling)
        return [o.outputs[0].text.strip() for o in outs]

    return _run_batch


def run_part1_70b_drift_eval(
    *,
    model_id: str,
    quantization: Optional[str],
    output_dir: Path,
    osv_cache_path: Path,
    top_k: int = 5,
) -> Dict[str, Any]:
    from sentence_transformers import SentenceTransformer

    from chelated_epistemic_proposer_v2 import (
        AuthorityDominatedExocortexV2,
        EXTRACTION_SYSTEM_PROMPT_V2,
        ground_and_clean_claims,
    )
    from chelated_raw_text_exocortex import parse_extracted_claim_json

    t0 = time.perf_counter()
    output_dir.mkdir(parents=True, exist_ok=True)
    cases, all_docs, corpus_sha = build_prospective_realworld_drift_suite(osv_cache_path)

    doc_texts = [d.text for d in all_docs]
    bm25 = CompactBM25(doc_texts)
    emb_model = SentenceTransformer("BAAI/bge-m3", device="cuda")
    doc_embs = emb_model.encode(doc_texts, normalize_embeddings=True, batch_size=64, show_progress_bar=False)
    query_embs = emb_model.encode(
        [c["query_text"] for c in cases], normalize_embeddings=True, batch_size=64, show_progress_bar=False
    )

    run_70b = _build_70b_generator(model_id, quantization=quantization)

    bare_prompts: List[Tuple[str, str]] = []
    for idx, case in enumerate(cases):
        bm_scores = np.zeros(len(all_docs), dtype=np.float32)
        for rk, di in enumerate(bm25.top_k(case["query_text"], k=len(all_docs))):
            bm_scores[di] = 1.0 - (rk / float(len(all_docs)))
        cos_scores = doc_embs @ query_embs[idx]
        hybrid = 0.5 * bm_scores + 0.5 * cos_scores
        top_ids = np.argsort(-hybrid)[:top_k].tolist()
        ctx = "\n".join(
            f"- [channel={all_docs[i].source_channel}, pub_day={all_docs[i].published_ts}] {all_docs[i].text}"
            for i in top_ids
        )
        bare_prompts.append((ENTERPRISE_QA_SYSTEM_PROMPT, f"Records:\n{ctx}\n\nQuestion: {case['query_text']}\nAnswer:"))

    bare_preds = run_70b(bare_prompts)

    extract_pairs = [
        (EXTRACTION_SYSTEM_PROMPT_V2, f"Channel: {d.source_channel}\nPublished Day: {d.published_ts}\nDocument: {d.text}")
        for d in all_docs
    ]
    raw_jsons = run_70b(extract_pairs)
    raw_claims = [parse_extracted_claim_json(d.doc_id, rj) for d, rj in zip(all_docs, raw_jsons)]
    slot_embs = emb_model.encode([c.slot for c in raw_claims], normalize_embeddings=True, batch_size=128, show_progress_bar=False)
    cleaned_claims = ground_and_clean_claims(all_docs, raw_claims, slot_embs)
    exo = AuthorityDominatedExocortexV2(all_docs, cleaned_claims, doc_embs, slot_embs)

    chelated_prompts: List[Tuple[str, str]] = []
    for idx, case in enumerate(cases):
        bm_scores = np.zeros(len(all_docs), dtype=np.float32)
        for rk, di in enumerate(bm25.top_k(case["query_text"], k=len(all_docs))):
            bm_scores[di] = 1.0 - (rk / float(len(all_docs)))
        cos_scores = doc_embs @ query_embs[idx]
        sel = exo.select_context_for_query(0.5 * bm_scores + 0.5 * cos_scores, case["query_text"], top_k=top_k)
        chelated_prompts.append(
            (ENTERPRISE_QA_SYSTEM_PROMPT, f"Records:\n{sel['formatted_context']}\n\nQuestion: {case['query_text']}\nAnswer:")
        )

    chelated_preds = run_70b(chelated_prompts)

    rows: List[Dict[str, Any]] = []
    scores_by_arm: Dict[str, Dict[str, float]] = {"Bare_70B_HybridRAG": {}, "Chelated_V2_EK5v3_70B": {}}
    for arm_name, preds in (("Bare_70B_HybridRAG", bare_preds), ("Chelated_V2_EK5v3_70B", chelated_preds)):
        for case, pred in zip(cases, preds):
            ok = int(score_enterprise_prediction(pred, case["gold"], case["stale_or_false_values"]))
            scores_by_arm[arm_name][case["case_id"]] = float(ok)
            rows.append(
                {
                    "case_id": case["case_id"],
                    "domain": case["domain"],
                    "regime": case["regime"],
                    "arm": arm_name,
                    "gold": case["gold"],
                    "pred": pred.replace("\n", " "),
                    "is_correct": ok,
                }
            )

    csv_path = output_dir / "colab_h100_70b_drift_rows.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    cids = [c["case_id"] for c in cases]
    boot = paired_bootstrap_delta(
        scores_by_arm["Chelated_V2_EK5v3_70B"],
        scores_by_arm["Bare_70B_HybridRAG"],
        n_bootstrap=3000,
        seed=42,
        expected_ids=cids,
    )
    summary = {
        "schema_version": "colab_h100_70b_drift_eval_v2",
        "model_id": model_id,
        "quantization": quantization,
        "corpus_sha256": corpus_sha,
        "n_cases": len(cases),
        "wall_seconds": round(time.perf_counter() - t0, 2),
        "arm_accuracy": {
            arm: round(float(np.mean(list(m.values()))), 4) for arm, m in scores_by_arm.items()
        },
        "paired_bootstrap_chelated_minus_bare": {
            "mean_delta": round(boot["mean_delta"], 4),
            "lcb_95": round(boot["lcb_95"], 4),
            "ucb_95": round(boot["ucb_95"], 4),
        },
        "rows_csv": csv_path.name,
        "rows_csv_sha256": hashlib.sha256(csv_path.read_bytes()).hexdigest(),
    }
    write_json_atomic(output_dir / "colab_h100_70b_manifest.json", summary)
    return summary


# ---------------------------------------------------------------------------
# Part 2: TRL GRPOTrainer Wired to Disjoint EK5-WFS-RLVR Splits
# ---------------------------------------------------------------------------
def run_part2_epistemic_grpo(
    *,
    base_model: str = "Qwen/Qwen2.5-7B-Instruct",
    output_dir: Path,
    n_train_episodes: int = 320,
    n_dev_episodes: int = 90,
    n_eval_episodes: int = 120,
    grpo_num_generations: int = 8,
    max_grpo_steps: int = 300,
) -> Dict[str, Any]:
    """Train SFT warm-start vs SFT + TRL GRPOTrainer (LoRA) on disjoint `TRAIN / DEV / EVAL` WFS splits."""
    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import GRPOConfig, GRPOTrainer, SFTConfig, SFTTrainer

    t0 = time.perf_counter()
    output_dir.mkdir(parents=True, exist_ok=True)
    train_eps = build_disjoint_rlvr_split("TRAIN", n_train_episodes)
    dev_eps = build_disjoint_rlvr_split("DEV", n_dev_episodes)
    eval_eps = build_disjoint_rlvr_split("EVAL", n_eval_episodes)
    disjoint_proof = assert_pairwise_disjoint_splits({"TRAIN": train_eps, "DEV": dev_eps, "EVAL": eval_eps})

    tok = AutoTokenizer.from_pretrained(base_model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    lora_cfg = LoraConfig(
        r=32,
        lora_alpha=64,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )

    sft_rows = [
        {
            "messages": [
                {"role": "system", "content": GRPO_SYSTEM_PROMPT},
                {"role": "user", "content": ep.prompt},
                {"role": "assistant", "content": reference_cot_for_episode(ep)},
            ]
        }
        for ep in train_eps
    ]
    sft_ds = Dataset.from_list(sft_rows)
    sft_dir = output_dir / "sft_checkpoint"
    model_sft = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=torch.bfloat16, device_map="auto")
    sft_trainer = SFTTrainer(
        model=model_sft,
        processing_class=tok,
        train_dataset=sft_ds,
        peft_config=lora_cfg,
        args=SFTConfig(
            output_dir=str(sft_dir),
            num_train_epochs=1,
            per_device_train_batch_size=8,
            gradient_accumulation_steps=2,
            learning_rate=1.5e-4,
            bf16=True,
            logging_steps=10,
            save_strategy="no",
            report_to="none",
        ),
    )
    sft_trainer.train()
    sft_trainer.save_model(str(sft_dir))

    def _eval_model(mdl: Any, split_eps: Sequence[EpistemicEpisode]) -> Tuple[Dict[str, float], Dict[str, float]]:
        mdl.eval()
        exact_scores: Dict[str, float] = {}
        verifier_rewards: Dict[str, float] = {}
        for ep in split_eps:
            txt_in = tok.apply_chat_template(
                [{"role": "system", "content": GRPO_SYSTEM_PROMPT}, {"role": "user", "content": ep.prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
            enc = tok([txt_in], return_tensors="pt").to(mdl.device)
            with torch.no_grad():
                out = mdl.generate(**enc, max_new_tokens=192, do_sample=False, pad_token_id=tok.pad_token_id)
            comp = tok.decode(out[0, enc.input_ids.shape[1] :], skip_special_tokens=True)
            sc = _strict_score_single_rlvr(comp, ep)
            exact_scores[ep.episode_id] = float(sc["exact_match"])
            verifier_rewards[ep.episode_id] = float(sc["reward"])
        return exact_scores, verifier_rewards

    sft_dev_exact, sft_dev_rew = _eval_model(sft_trainer.model, dev_eps)
    sft_eval_exact, sft_eval_rew = _eval_model(sft_trainer.model, eval_eps)

    grpo_rows = [
        {
            "prompt": tok.apply_chat_template(
                [{"role": "system", "content": GRPO_SYSTEM_PROMPT}, {"role": "user", "content": ep.prompt}],
                tokenize=False,
                add_generation_prompt=True,
            ),
            "episode": ep.to_json(),
        }
        for ep in train_eps
    ]
    grpo_ds = Dataset.from_list(grpo_rows)
    grpo_dir = output_dir / "sft_plus_grpo_checkpoint"

    grpo_cfg = GRPOConfig(
        output_dir=str(grpo_dir),
        max_steps=max_grpo_steps,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        num_generations=grpo_num_generations,
        max_completion_length=224,
        learning_rate=5e-6,
        beta=0.04,
        bf16=True,
        logging_steps=10,
        save_strategy="no",
        report_to="none",
    )
    grpo_model = sft_trainer.model if hasattr(sft_trainer.model, "peft_config") else get_peft_model(model_sft, lora_cfg)
    grpo_trainer = GRPOTrainer(
        model=grpo_model,
        processing_class=tok,
        reward_funcs=[ek5_epistemic_verifiability_reward, format_epistemic_tags_reward],
        args=grpo_cfg,
        train_dataset=grpo_ds,
    )
    grpo_trainer.train()
    grpo_trainer.save_model(str(grpo_dir))

    grpo_dev_exact, grpo_dev_rew = _eval_model(grpo_trainer.model, dev_eps)
    grpo_eval_exact, grpo_eval_rew = _eval_model(grpo_trainer.model, eval_eps)
    eids = [ep.episode_id for ep in eval_eps]
    boot_exact = paired_bootstrap_delta(grpo_eval_exact, sft_eval_exact, n_bootstrap=3000, seed=42, expected_ids=eids)
    boot_rew = paired_bootstrap_delta(grpo_eval_rew, sft_eval_rew, n_bootstrap=3000, seed=43, expected_ids=eids)

    manifest = {
        "schema_version": "colab_h100_epistemic_grpo_v2_disjoint_wfs",
        "git_sha": current_git_sha(REPO_ROOT),
        "timestamp": datetime.now().isoformat(),
        "base_model": base_model,
        "disjoint_split_proof": disjoint_proof,
        "reward_design": {
            "core_correct_answer": 0.75,
            "core_correct_verdict": 0.25,
            "abstain_on_determinate_verdict_plus_answer_note": "VERDICT: U + ANSWER: ABSTAIN on determinate item scores 0.0; correct verdict + ABSTAIN scores +0.25",
            "format_bonus_max": 0.05,
            "format_bonus_gated_on_exact_match": True,
        },
        "dev_metrics": {
            "sft_exact_match": round(float(np.mean(list(sft_dev_exact.values()))), 4),
            "sft_mean_verifier_reward": round(float(np.mean(list(sft_dev_rew.values()))), 4),
            "grpo_exact_match": round(float(np.mean(list(grpo_dev_exact.values()))), 4),
            "grpo_mean_verifier_reward": round(float(np.mean(list(grpo_dev_rew.values()))), 4),
        },
        "heldout_eval_metrics": {
            "sft_exact_match": round(float(np.mean(list(sft_eval_exact.values()))), 4),
            "sft_mean_verifier_reward": round(float(np.mean(list(sft_eval_rew.values()))), 4),
            "sft_plus_grpo_exact_match": round(float(np.mean(list(grpo_eval_exact.values()))), 4),
            "sft_plus_grpo_mean_verifier_reward": round(float(np.mean(list(grpo_eval_rew.values()))), 4),
            "paired_bootstrap_exact_match_grpo_minus_sft": boot_exact,
            "paired_bootstrap_verifier_reward_grpo_minus_sft": boot_rew,
        },
        "wall_seconds": round(time.perf_counter() - t0, 2),
    }
    write_json_atomic(output_dir / "colab_h100_epistemic_grpo_manifest.json", manifest)
    return manifest


def run_offline_verify(output_dir: Path) -> Dict[str, Any]:
    """CPU-only verification of disjoint `TRAIN/DEV/EVAL` splits, 3-way WFS oracle audit, and anti-hacking reward gates."""
    train_eps = build_disjoint_rlvr_split("TRAIN", 24)
    dev_eps = build_disjoint_rlvr_split("DEV", 12)
    eval_eps = build_disjoint_rlvr_split("EVAL", 18)
    disjoint_proof = assert_pairwise_disjoint_splits({"TRAIN": train_eps, "DEV": dev_eps, "EVAL": eval_eps})

    all_eps = train_eps + dev_eps + eval_eps
    ep_jsons = [ep.to_json() for ep in all_eps]

    # 1) Reference `<think>+VERDICT+ANSWER` trajectories achieve 1.0 core + 0.05 format bonus
    ref_cots = [reference_cot_for_episode(ep) for ep in all_eps]
    core_ref = ek5_epistemic_verifiability_reward(ref_cots, ep_jsons)
    fmt_ref = format_epistemic_tags_reward(ref_cots, ep_jsons)
    assert all(r == 1.0 for r in core_ref), f"Expected all +1.0 on reference CoTs, got {core_ref}"
    assert all(f == 0.05 for f in fmt_ref), f"Expected all +0.05 format bonus on reference CoTs, got {fmt_ref}"

    # 2) Bare oracle (`VERDICT + ANSWER` without `<think>`) scores 1.0 core and 0.0 format (beats any wrong answer!)
    bare_oracles = [oracle_completion(ep) for ep in all_eps]
    core_bare = ek5_epistemic_verifiability_reward(bare_oracles, ep_jsons)
    fmt_bare = format_epistemic_tags_reward(bare_oracles, ep_jsons)
    assert all(r == 1.0 for r in core_bare)
    assert all(f == 0.0 for f in fmt_bare)

    # 3) Well-formatted wrong/negated/mixed-commitment completions receive <= 0.0 format bonus and negative core reward
    negated = [
        f"<think>careful check</think>\nVERDICT: {ep.gold_label}\nANSWER: not {ep.gold_value or 'ABSTAIN'}"
        for ep in all_eps
    ]
    core_neg = ek5_epistemic_verifiability_reward(negated, ep_jsons)
    fmt_neg = format_epistemic_tags_reward(negated, ep_jsons)
    assert all(r == -1.0 for r in core_neg), f"Negation must be rejected (-1.0), got {core_neg}"
    assert all(f == 0.0 for f in fmt_neg), f"Wrong/negated completion must get 0.0 format bonus, got {fmt_neg}"

    mixed_u = [
        f"<think>hedging</think>\nVERDICT: U\nANSWER: ABSTAIN ({ep.candidate_values[0]})"
        for ep in all_eps
    ]
    core_mix = ek5_epistemic_verifiability_reward(mixed_u, ep_jsons)
    fmt_mix = format_epistemic_tags_reward(mixed_u, ep_jsons)
    assert all(r == -1.0 for r in core_mix), f"Mixed commitment must be rejected (-1.0), got {core_mix}"
    assert all(f == 0.0 for f in fmt_mix)

    res = {
        "schema_version": "colab_h100_package_offline_verify_v2_disjoint_wfs",
        "git_sha": current_git_sha(REPO_ROOT),
        "n_verified_rlvr_episodes": len(all_eps),
        "disjoint_split_proof": disjoint_proof,
        "perfect_core_reward_mean": float(np.mean(core_ref)),
        "perfect_format_bonus_mean": float(np.mean(fmt_ref)),
        "negation_attack_reward_mean": float(np.mean(core_neg)),
        "mixed_commitment_reward_mean": float(np.mean(core_mix)),
        "wrong_answer_max_format_bonus": float(max(fmt_neg + fmt_mix)),
        "trained_grpo_checkpoint_present": False,
        "scope_note": "Offline WFS verifier & disjoint-split check ONLY; no H100 LoRA GRPO checkpoint trained yet.",
        "status": "OK",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json_atomic(output_dir / "verify_only_check.json", res)
    print(f"[Colab-Verify-v2] PASSED: {json.dumps(res, indent=2)}", flush=True)
    return res


def main() -> None:
    parser = argparse.ArgumentParser(description="Colab H100 80GB 70B Drift Eval + Disjoint EK5-WFS Epistemic GRPO Package")
    parser.add_argument("--mode", choices=("all", "eval70b", "grpo", "verify-only"), default="all")
    parser.add_argument("--model-70b", type=str, default=os.environ.get("CHELATED_70B_MODEL", "Qwen/Qwen2.5-72B-Instruct-AWQ"))
    parser.add_argument("--quantization", type=str, default="awq")
    parser.add_argument("--grpo-base-model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--grpo-steps", type=int, default=300)
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "artifacts" / "colab_h100_70b_grpo")
    parser.add_argument("--osv-cache", type=Path, default=REPO_ROOT / "artifacts" / "phase2_d1_4node_local_fleet_proof" / "_work" / "osv_cve_seeds.json")
    args = parser.parse_args()

    if args.mode == "verify-only":
        run_offline_verify(args.output_dir)
        return

    if args.mode in ("all", "eval70b"):
        run_part1_70b_drift_eval(
            model_id=args.model_70b,
            quantization=args.quantization,
            output_dir=args.output_dir,
            osv_cache_path=args.osv_cache,
        )
    if args.mode in ("all", "grpo"):
        run_part2_epistemic_grpo(
            base_model=args.grpo_base_model,
            output_dir=args.output_dir,
            max_grpo_steps=args.grpo_steps,
        )


if __name__ == "__main__":
    main()
