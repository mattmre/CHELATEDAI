"""Wave-6 Track A (`PM-1b v4`, Guardrail `K2`): Cross-Encoder Reranked Engine, Post-J4 Look #2, & Virgin Holdout.

Strictly additive Wave-6 runner implementing all mandatory `K2` guardrails:
  1. **Phase 1 — Strict `DEV`-Only Tuning & Cryptographic Freeze (`frozen_dev_config.json`):**
     Tunes `ChelatedConversationalV4` (`BAAI/bge-reranker-v2-m3` + compact `window_radius` / `context_chars`
     targeting `~1,400-2,200` tokens) strictly on `DEV` (`LoCoMo 0-4` & `LME DEV`) against token-matched
     `Flat_BGEM3_TokenMatched`. Writes and SHA-256-freezes `frozen_dev_config.json` BEFORE any holdout item
     is loaded or evaluated.
  2. **Phase 2 — `POST_J4_EXPLORATORY_LOOK_2_BURNED_HOLDOUT` (`alpha=0.025`):**
     Evaluates frozen `Chelated_v4_Reranked_EK5v3`, `DualGran_Reranked_NoEK5`, `Flat_BGEM3_TokenMatched`,
     and `Legacy_MiniLM_Flat_Top5` on `LoCoMo 5-9` (`N=150` clean + echo stress) and `LME TEST`
     (`N=251` clean + `N=78` `KU` echo stress, plus `N=39` test-half `KU` slice), reporting distinct
     `lcb_95=p5.0`, `lcb_975=p2.5`, `p0125_bonf2=p1.25`, paired cluster bootstrap, and Tango `z=2.2414`
     side-by-side with historical `J4` scores (`r6_pm1b_manifest.json`).
  3. **Phase 3 — PRIMARY CONFIRMATORY GATE (`Virgin-Conversational-120` + Blind `Qwen2.5-7B-Instruct` Judge):**
     Answers all 4 arms ONCE with frozen `DEV` config on `Nemotron-30B` (`:18861` -> Spark #3 `:8801`),
     unloads embedders/reranker from `RTX 3090`, loads `Qwen/Qwen2.5-7B-Instruct` (`torch.bfloat16`) on
     `RTX 3090`, and grades ALL 120 items x 4 arms = 480 responses blind via `BlindCrossFamilySemanticJudge`,
     reporting 24-cluster `paired_bootstrap_wave6` (`p5.0`, `p2.5`, `p1.25`) + Holm-Bonferroni stepdown.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import socket
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import requests

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chelated_conversational_v4 import (  # noqa: E402
    BGE_DIMENSIONS,
    BGE_MODEL,
    BGE_RERANKER_MODEL,
    CachedBGEReranker,
    ChelatedConversationalV4,
    ConversationRecall,
    ConversationTurn,
    flat_bgem3_token_matched_context_v4,
)
from scripts.night1_common import (  # noqa: E402
    current_git_sha,
    sha256_json,
    write_json_atomic,
)
from scripts.run_flagship2_r4_ek5_v3_and_suites import _extract_locomo_units  # noqa: E402
from scripts.run_overnight_r5_spark6724 import (  # noqa: E402
    TurnBM25,
    _call_vllm,
    _extract_lme_units,
    _retrieve_lme_r5_arm,
)
from scripts.run_r6_pm1b_conversational_sota_closure import (  # noqa: E402
    CLEAN,
    J4_SPLIT_SEED,
    LME,
    LME_ANSWER,
    LME_DATA,
    LOC_ANSWER,
    LOC_DATA,
    LOCOMO,
    MINILM,
    STRESS,
    STRESS_R5,
    CachedEncoder,
    Question,
    answer_prompt,
    build_lme_holdout_split_map,
    lme_memory,
    neutral_locomo_echoes,
    normalized,
    question_split_tag,
    r5_grade,
    r5_legacy_row,
    r5_locomo_questions,
)
from virgin_conversational_holdout import (  # noqa: E402
    VIRGIN_SUITE_NAME,
    VirginConversation,
    VirginQuestion,
    build_virgin_conversational_holdout,
    verify_lineage_disjointness,
    virgin_suite_digest,
)
from wave6_common import (  # noqa: E402
    BlindCrossFamilySemanticJudge,
    holm_bonferroni_stepdown,
    paired_bootstrap_wave6,
)

ARM_CHELATED_V4 = "Chelated_v4_Reranked_EK5v3"
ARM_DUALGRAN_RERANKED = "DualGran_Reranked_NoEK5"
ARM_FLAT_BGEM3_TM = "Flat_BGEM3_TokenMatched"
ARM_LEGACY_MINILM = "Legacy_MiniLM_Flat_Top5"
ARMS_V4: tuple[str, ...] = (
    ARM_LEGACY_MINILM,
    ARM_FLAT_BGEM3_TM,
    ARM_DUALGRAN_RERANKED,
    ARM_CHELATED_V4,
)

JUDGE_MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
SPARK3_HOST = "mattmre@10.0.0.24"
SPARK1_DATA_HOST = "mattmre@10.0.0.194"
DEFAULT_TUNNEL_PORT = 18861
REMOTE_VLLM_PORT = 8801

VIRGIN_ANSWER_SYSTEM = (
    "You are a precise conversational memory assistant. Answer the question strictly using ONLY the "
    "dated conversation excerpts below. Be concise (1-2 sentences).\n"
    "- Verify exact speaker, entity, and premise match: never attribute one speaker's statement, gear, "
    "metric, or booking to another speaker. If the specific item, event, or premise asked about was NEVER "
    "mentioned in the excerpts, explicitly state that it was not mentioned (or reply ABSTAIN).\n"
    "- When an earlier setting, booking, or target was later changed or updated in a newer dated session "
    "(or marked as an earlier superseded record), report the latest current value unless asked about the past.\n"
    "- When a question asks for a total across multiple sessions, sum or combine all matching dated sessions."
)

# Pre-registered compact DEV candidate grid (evaluated strictly on DEV LoCoMo 0..4 & LME DEV before freeze!)
DEV_CANDIDATE_GRID: tuple[dict[str, Any], ...] = (
    {
        "candidate_id": "v4_cand_A_compact_r0",
        "pool_size": 36,
        "rerank_alpha": 0.80,
        "rerank_max_chars": 1200,
        "rerank_prev_turn": True,
        "LoCoMo-300": {
            "anchor_count": 8,
            "window_radius": 0,
            "context_chars": 4800,
            "turn_max_chars": 1200,
            "neighbor_max_chars": 600,
            "speaker_attribution_header": True,
            "speaker_anchor_reserve": 1,
        },
        "LongMemEval-S": {
            "anchor_count": 8,
            "window_radius": 0,
            "context_chars": 6800,
            "turn_max_chars": 1350,
            "neighbor_max_chars": 600,
            "speaker_attribution_header": False,
            "speaker_anchor_reserve": 0,
        },
        "Virgin-Conversational-120": {
            "anchor_count": 8,
            "window_radius": 0,
            "context_chars": 5200,
            "turn_max_chars": 1200,
            "neighbor_max_chars": 600,
            "speaker_attribution_header": True,
            "speaker_anchor_reserve": 1,
        },
    },
    {
        "candidate_id": "v4_cand_B_balanced_r1",
        "pool_size": 36,
        "rerank_alpha": 0.80,
        "rerank_max_chars": 1200,
        "rerank_prev_turn": True,
        "LoCoMo-300": {
            "anchor_count": 8,
            "window_radius": 1,
            "context_chars": 5400,
            "turn_max_chars": 1200,
            "neighbor_max_chars": 550,
            "speaker_attribution_header": True,
            "speaker_anchor_reserve": 1,
        },
        "LongMemEval-S": {
            "anchor_count": 8,
            "window_radius": 1,
            "context_chars": 7600,
            "turn_max_chars": 1400,
            "neighbor_max_chars": 600,
            "speaker_attribution_header": False,
            "speaker_anchor_reserve": 0,
        },
        "Virgin-Conversational-120": {
            "anchor_count": 8,
            "window_radius": 1,
            "context_chars": 5800,
            "turn_max_chars": 1200,
            "neighbor_max_chars": 550,
            "speaker_attribution_header": True,
            "speaker_anchor_reserve": 1,
        },
    },
    {
        "candidate_id": "v4_cand_C_wider_r1",
        "pool_size": 36,
        "rerank_alpha": 0.80,
        "rerank_max_chars": 1200,
        "rerank_prev_turn": True,
        "LoCoMo-300": {
            "anchor_count": 10,
            "window_radius": 1,
            "context_chars": 6000,
            "turn_max_chars": 1300,
            "neighbor_max_chars": 600,
            "speaker_attribution_header": True,
            "speaker_anchor_reserve": 1,
        },
        "LongMemEval-S": {
            "anchor_count": 10,
            "window_radius": 1,
            "context_chars": 8400,
            "turn_max_chars": 1450,
            "neighbor_max_chars": 600,
            "speaker_attribution_header": False,
            "speaker_anchor_reserve": 0,
        },
        "Virgin-Conversational-120": {
            "anchor_count": 9,
            "window_radius": 1,
            "context_chars": 6200,
            "turn_max_chars": 1300,
            "neighbor_max_chars": 600,
            "speaker_attribution_header": True,
            "speaker_anchor_reserve": 1,
        },
    },
)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _sha256_file(path: Path) -> str:
    with path.open("rb") as fh:
        return hashlib.file_digest(fh, "sha256").hexdigest()


def _ensure_safetensors_model_dir(snap_dir: Path, work_dir: Path) -> Path:
    """Materialize a lossless `model.safetensors` directory in `work_dir` if a cached repo only has `.bin`."""
    if (snap_dir / "model.safetensors").is_file() or (snap_dir / "model.safetensors.index.json").is_file():
        return snap_dir
    dst_dir = work_dir / f"safetensors_{snap_dir.name}"
    if (dst_dir / "model.safetensors").is_file():
        return dst_dir

    import shutil
    import torch
    from safetensors.torch import save_file

    tmp_dir = work_dir / f"safetensors_{snap_dir.name}_tmp"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    shutil.copytree(
        snap_dir,
        tmp_dir,
        ignore=shutil.ignore_patterns("pytorch_model*.bin", "*.onnx*", "onnx", "imgs", "*.DS_Store"),
    )
    state_dict = torch.load(snap_dir / "pytorch_model.bin", map_location="cpu", weights_only=True)
    seen_ptrs: set[int] = set()
    clean_sd = {}
    for k, v in state_dict.items():
        ptr = v.data_ptr()
        tensor = v.clone().contiguous() if ptr in seen_ptrs else v.contiguous()
        seen_ptrs.add(tensor.data_ptr())
        clean_sd[k] = tensor
    save_file(clean_sd, str(tmp_dir / "model.safetensors"), metadata={"format": "pt"})
    tmp_dir.replace(dst_dir)
    return dst_dir


def _split_legacy_turn(row: Mapping[str, Any]) -> ConversationTurn:
    return ConversationTurn(
        turn_id=str(row["unit_id"]),
        session_id=str(row["session_id"]),
        date=str(row["date_str"]),
        timestamp=float(row["ts"]),
        text=str(row["text"]),
        claim_text=str(row["raw_text"]),
        speaker=str(row["text"]).partition(":")[0].strip(),
    )


def _port_open(host: str, port: int, timeout: float = 1.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _ensure_ssh_tunnel(local_port: int, remote_host: str, remote_port: int) -> subprocess.Popen[bytes] | None:
    """Open local SSH port forward with `stdin=subprocess.DEVNULL` (Windows OpenSSH rule)."""
    if _port_open("127.0.0.1", local_port):
        return None
    proc = subprocess.Popen(
        [
            "ssh",
            "-N",
            "-o",
            "BatchMode=yes",
            "-o",
            "ExitOnForwardFailure=yes",
            "-o",
            "ServerAliveInterval=30",
            "-o",
            "ConnectTimeout=10",
            "-L",
            f"{local_port}:127.0.0.1:{remote_port}",
            remote_host,
        ],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    deadline = time.perf_counter() + 15.0
    while time.perf_counter() < deadline:
        if proc.poll() is not None:
            err_txt = (proc.stderr.read().decode("utf-8", errors="replace") if proc.stderr else "").strip()
            raise RuntimeError(f"SSH tunnel exited early ({proc.returncode}): {err_txt}")
        if _port_open("127.0.0.1", local_port):
            return proc
        time.sleep(0.25)
    proc.terminate()
    raise RuntimeError(f"Timed out waiting for SSH tunnel on 127.0.0.1:{local_port}")


def _resolve_dataset_files(work_dir: Path) -> dict[str, Path]:
    """Load `locomo10.json` and `longmemeval_s_cleaned.json` from local HF cache or `scp` fallback."""
    from huggingface_hub import hf_hub_download

    resolved: dict[str, Path] = {}
    remote_fallbacks = {
        LME: "longmemeval_s_cleaned.json",
        LOCOMO: "locomo10.json",
    }
    for name, (repo, rel_path) in ((LME, LME_DATA), (LOCOMO, LOC_DATA)):
        try:
            p = Path(hf_hub_download(repo, rel_path, repo_type="dataset", local_files_only=True))
            if p.is_file():
                resolved[name] = p
                continue
        except Exception:
            pass
        work_dir.mkdir(parents=True, exist_ok=True)
        dst = work_dir / remote_fallbacks[name]
        if not dst.is_file():
            cp = subprocess.run(
                [
                    "scp",
                    "-o",
                    "BatchMode=yes",
                    f"{SPARK1_DATA_HOST}:/home/mattmre/chelated-night1/{remote_fallbacks[name]}",
                    str(dst),
                ],
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                timeout=180,
            )
            if cp.returncode != 0 or not dst.is_file():
                raise RuntimeError(f"Unable to locate {name} locally or via scp: {cp.stderr}")
        resolved[name] = dst
    return resolved


def _load_validated_ku_echo_fixture(lme_items: Sequence[dict[str, Any]], lme_sha256: str) -> dict[str, dict[str, Any]]:
    """Read-only load of the frozen R6 KU natural-paraphrase fixture (`N=78 x 6 = 468` echoes)."""
    r6_fixture_path = ROOT / "artifacts" / "r6_pm1b_conversational_sota" / "r6_pm1b_ku_echo_fixture.json"
    if not r6_fixture_path.is_file():
        raise FileNotFoundError(f"Expected frozen KU echo fixture at {r6_fixture_path}")
    fixture = json.loads(r6_fixture_path.read_text(encoding="utf-8"))
    if fixture.get("provenance", {}).get("dataset_sha256") != lme_sha256:
        raise ValueError("Frozen KU echo fixture dataset SHA-256 mismatch")
    ku_ids = {str(it["question_id"]) for it in lme_items if it["question_type"] == "knowledge-update"}
    by_id = {str(rec["qid"]): rec for rec in fixture["items"]}
    if set(by_id) != ku_ids or any(len(rec["turns"]) != 6 for rec in by_id.values()):
        raise ValueError("Incomplete KU echo fixture")
    return by_id


@dataclass(frozen=True)
class EvalJob:
    question: Question
    regime: str
    split: str
    arm: str
    prompt_sha256: str
    selected_ids: tuple[str, ...]
    evidence_any: int | None
    evidence_all: int | None
    gate_active: bool = False
    echo_clusters: tuple[tuple[str, ...], ...] = ()
    supersessions: tuple[tuple[str, str], ...] = ()


class PromptAnswerRunner:
    """Thread-pooled vLLM caller with persistent SHA-256 disk checkpointing."""

    def __init__(self, endpoint: str, model_id: str, workers: int, cache_path: Path) -> None:
        self.endpoint = endpoint
        self.model_id = model_id
        self.workers = workers
        self.cache_path = cache_path
        self.cache: dict[str, list[Any]] = (
            json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.is_file() else {}
        )

    def execute_prompts(self, prompts: Mapping[str, tuple[str, str, int]]) -> None:
        todo = [
            digest
            for digest in prompts
            if digest not in self.cache or bool(self.cache[digest][3])
        ]
        if not todo:
            return
        print(f"  [Nemotron-30B] Answering {len(todo)} uncached prompts ({len(prompts)} total referenced)...", flush=True)
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            for start in range(0, len(todo), 96):
                batch_keys = todo[start : start + 96]
                results = list(
                    pool.map(
                        lambda d: _call_vllm(
                            self.endpoint,
                            self.model_id,
                            prompts[d][0],
                            prompts[d][1],
                            max_tokens=prompts[d][2],
                            timeout=90.0,
                        ),
                        batch_keys,
                    )
                )
                for k, val in zip(batch_keys, results, strict=True):
                    self.cache[k] = [val[0], int(val[1]), float(val[2]), bool(val[3])]
                write_json_atomic(self.cache_path, self.cache)
                print(f"    completed {min(start + 96, len(todo))}/{len(todo)} answer prompts", flush=True)

    def get(self, digest: str) -> tuple[str, int, float, bool]:
        entry = self.cache[digest]
        return str(entry[0]), int(entry[1]), float(entry[2]), bool(entry[3])


class R5JudgeRunner:
    """Exact R5/R6 QA grader for apples-to-apples side-by-side comparison on Look #2."""

    def __init__(self, endpoint: str, model_id: str, workers: int, cache_path: Path) -> None:
        self.endpoint = endpoint
        self.model_id = model_id
        self.workers = workers
        self.cache_path = cache_path
        raw = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.is_file() else {}
        self.grades: dict[tuple[str, str, str], tuple[bool, bool]] = {}
        for k_str, v in raw.items():
            parts = k_str.split("|||", 2)
            if len(parts) == 3 and not v[1]:
                self.grades[(parts[0], parts[1], parts[2])] = (bool(v[0]), False)

    def grade_jobs(self, jobs: Sequence[EvalJob], answer_runner: PromptAnswerRunner) -> None:
        pending_llm: dict[tuple[str, str, str], tuple[str, str]] = {}
        for job in jobs:
            q = job.question
            pred = normalized(answer_runner.get(job.prompt_sha256)[0])
            key = (q.suite, q.qid, pred.lower())
            if key not in self.grades and key not in pending_llm:
                g = r5_grade(q, pred)
                if isinstance(g, bool):
                    self.grades[key] = (g, False)
                else:
                    pending_llm[key] = g
        if pending_llm:
            print(f"  [R5-Judge] Grading {len(pending_llm)} uncached Look #2 responses...", flush=True)
            items = list(pending_llm.items())
            with ThreadPoolExecutor(max_workers=self.workers) as pool:
                outs = list(
                    pool.map(
                        lambda kv: _call_vllm(
                            self.endpoint, self.model_id, kv[1][0], kv[1][1], max_tokens=8, timeout=60.0
                        ),
                        items,
                    )
                )
            for (key, _), res in zip(items, outs, strict=True):
                self.grades[key] = (res[0].strip().upper().startswith("YES"), bool(res[3]))
            serial = {f"{k[0]}|||{k[1]}|||{k[2]}": [bool(v[0]), bool(v[1])] for k, v in self.grades.items()}
            write_json_atomic(self.cache_path, serial)

    def verdict(self, suite: str, qid: str, pred: str) -> tuple[bool, bool]:
        return self.grades[(suite, qid, normalized(pred).lower())]


def _build_jobs_for_questions(
    questions: Sequence[Question],
    regime: str,
    split_lookup: Mapping[tuple[str, str], str],
    legacy_rows: Sequence[dict[str, Any]],
    turns: Sequence[ConversationTurn],
    *,
    flat_enc: CachedEncoder,
    bge_enc: CachedEncoder,
    reranker: CachedBGEReranker,
    cand_cfg: Mapping[str, Any],
    arms_to_run: Sequence[str],
    prompts_out: dict[str, tuple[str, str, int]],
) -> list[EvalJob]:
    """Retrieve contexts and register answer prompts for `questions` under `cand_cfg`."""
    jobs: list[EvalJob] = []
    if not questions:
        return jobs

    suite = questions[0].suite
    suite_cfg = cand_cfg[suite]
    pool_size = int(cand_cfg["pool_size"])
    rerank_alpha = float(cand_cfg["rerank_alpha"])
    rerank_max_chars = int(cand_cfg["rerank_max_chars"])
    rerank_prev_turn = bool(cand_cfg["rerank_prev_turn"])

    need_legacy = ARM_LEGACY_MINILM in arms_to_run
    if need_legacy:
        text_bank = [row["text"] for row in legacy_rows]
        bm25 = TurnBM25(text_bank)
        flat_vecs = flat_enc.encode(text_bank)
        flat_q_vecs = flat_enc.encode([q.text for q in questions])
    else:
        bm25 = None
        flat_vecs = np.zeros((0, 384), dtype=np.float32)
        flat_q_vecs = np.zeros((len(questions), 384), dtype=np.float32)

    engine = ChelatedConversationalV4(turns, bge_enc, reranker)

    def _register(
        q: Question,
        arm: str,
        ctx: str,
        ids: Sequence[str],
        ev_units: set[str],
        recall: ConversationRecall | None = None,
    ) -> None:
        system, max_tok = (LME_ANSWER, 96) if q.suite == LME else (LOC_ANSWER, 64)
        req = (system, answer_prompt(q, ctx), max_tok)
        digest = hashlib.sha256(json.dumps(req, ensure_ascii=False).encode("utf-8")).hexdigest()
        prompts_out.setdefault(digest, req)
        labeled = set(q.evidence) if not q.abstention else set()
        jobs.append(
            EvalJob(
                question=q,
                regime=regime,
                split=split_lookup[(q.suite, q.qid)],
                arm=arm,
                prompt_sha256=digest,
                selected_ids=tuple(ids),
                evidence_any=int(bool(labeled.intersection(ev_units))) if labeled else None,
                evidence_all=int(labeled.issubset(ev_units)) if labeled else None,
                gate_active=recall.gate_active if recall else False,
                echo_clusters=recall.echo_clusters if recall else (),
                supersessions=recall.supersessions if recall else (),
            )
        )

    for q_idx, q in enumerate(questions):
        ev_is_session = q.suite == LME
        if need_legacy and bm25 is not None:
            idxs, _ = _retrieve_lme_r5_arm(
                "M1_hybrid_flat", legacy_rows, flat_vecs, flat_q_vecs[q_idx], bm25.score(q.text)
            )
            ordered = sorted(idxs, key=lambda i: legacy_rows[i]["ts"])
            l_rows = [legacy_rows[i] for i in ordered]
            flat_ctx = "\n".join(f"- [{r['date_str']}] {r['text']}" for r in l_rows)
            ev_key = "session_id" if ev_is_session else "unit_id"
            _register(
                q,
                ARM_LEGACY_MINILM,
                flat_ctx,
                [str(r["unit_id"]) for r in l_rows],
                {str(r[ev_key]) for r in l_rows},
            )

        if ARM_FLAT_BGEM3_TM in arms_to_run:
            tm_ctx, tm_ids, tm_sids = flat_bgem3_token_matched_context_v4(
                engine,
                q.text,
                context_chars=int(suite_cfg["context_chars"]),
                turn_max_chars=int(suite_cfg["turn_max_chars"]),
            )
            _register(q, ARM_FLAT_BGEM3_TM, tm_ctx, tm_ids, tm_sids if ev_is_session else set(tm_ids))

        need_dg = ARM_DUALGRAN_RERANKED in arms_to_run
        need_v4 = ARM_CHELATED_V4 in arms_to_run
        if need_dg or need_v4:
            ranking, s1_scores, _ = engine.rank_with_scores(q.text)
            common_kw = {
                "anchor_count": int(suite_cfg["anchor_count"]),
                "window_radius": int(suite_cfg["window_radius"]),
                "context_chars": int(suite_cfg["context_chars"]),
                "pool_size": pool_size,
                "turn_max_chars": int(suite_cfg["turn_max_chars"]),
                "neighbor_max_chars": int(suite_cfg["neighbor_max_chars"]),
                "rerank_alpha": rerank_alpha,
                "rerank_max_chars": rerank_max_chars,
                "rerank_prev_turn": rerank_prev_turn,
                "speaker_attribution_header": bool(suite_cfg["speaker_attribution_header"]),
                "speaker_anchor_reserve": int(suite_cfg["speaker_anchor_reserve"]),
                "ranked": ranking,
                "stage1_scores": s1_scores,
            }
            rec_dg = (
                engine.retrieve(q.text, epistemic=False, **common_kw)
                if (need_dg or need_v4)
                else None
            )
            rec_v4 = engine.retrieve(q.text, epistemic=True, **common_kw) if need_v4 else None
            if rec_dg is not None and rec_v4 is not None and not rec_v4.gate_active and rec_dg != rec_v4:
                raise RuntimeError(f"Inactive-gate parity violation on {q.suite}/{q.qid}")
            if need_dg and rec_dg is not None:
                dg_ids = [t.turn.turn_id for t in rec_dg.turns]
                dg_ev = {t.turn.session_id for t in rec_dg.turns} if ev_is_session else set(dg_ids)
                _register(q, ARM_DUALGRAN_RERANKED, rec_dg.context, dg_ids, dg_ev, rec_dg)
            if need_v4 and rec_v4 is not None:
                v4_ids = [t.turn.turn_id for t in rec_v4.turns]
                v4_ev = {t.turn.session_id for t in rec_v4.turns} if ev_is_session else set(v4_ids)
                _register(q, ARM_CHELATED_V4, rec_v4.context, v4_ids, v4_ev, rec_v4)

    return jobs


def _materialize_rows(
    jobs: Sequence[EvalJob],
    answer_runner: PromptAnswerRunner,
    judge_runner: R5JudgeRunner,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for job in jobs:
        q = job.question
        pred_raw, p_toks, lat_ms, ans_err = answer_runner.get(job.prompt_sha256)
        pred = normalized(pred_raw)
        is_ok, j_err = judge_runner.verdict(q.suite, q.qid, pred)
        rows.append(
            {
                "suite": q.suite,
                "regime": job.regime,
                "split": job.split,
                "arm": job.arm,
                "qid": q.qid,
                "question_type": q.kind,
                "conversation": q.group,
                "is_abstention": int(q.abstention),
                "gold": normalized(q.gold),
                "prediction": pred,
                "is_correct": int(is_ok),
                "evidence_any": job.evidence_any,
                "evidence_all": job.evidence_all,
                "prompt_tokens": p_toks,
                "latency_ms": round(lat_ms, 2),
                "api_error": int(ans_err or j_err),
                "prompt_sha256": job.prompt_sha256,
                "gate_active": int(job.gate_active),
                "selected_ids": json.dumps(job.selected_ids),
                "echo_clusters": json.dumps(job.echo_clusters),
                "supersessions": json.dumps(job.supersessions),
            }
        )
    return rows


def _mean_opt(rows: Sequence[Mapping[str, Any]], key: str) -> float | None:
    vals = [float(r[key]) for r in rows if r.get(key) is not None]
    return float(sum(vals) / len(vals)) if vals else None


def _summarize_wave6_slice(
    rows: Sequence[Mapping[str, Any]],
    expected_ids: Sequence[str],
    *,
    arms: Sequence[str] = ARMS_V4,
    cluster_key: str | None = None,
) -> dict[str, Any]:
    """Compute arm summaries + `paired_bootstrap_wave6` (`p5.0`, `p2.5`, `p1.25`, Tango `z=2.2414`)."""
    by_arm: dict[str, list[Mapping[str, Any]]] = {
        a: [r for r in rows if r["arm"] == a] for a in arms
    }
    for a, a_rows in by_arm.items():
        if len(a_rows) != len(expected_ids):
            raise ValueError(f"Arm {a}: expected {len(expected_ids)} rows, got {len(a_rows)}")

    qtypes = sorted({str(r["question_type"]) for r in by_arm[arms[0]]})
    arm_metrics: dict[str, Any] = {}
    for a, a_rows in by_arm.items():
        arm_metrics[a] = {
            "n": len(a_rows),
            "n_correct": int(sum(int(r["is_correct"]) for r in a_rows)),
            "accuracy": _mean_opt(a_rows, "is_correct"),
            "evidence_any_recall": _mean_opt(a_rows, "evidence_any"),
            "evidence_all_recall": _mean_opt(a_rows, "evidence_all"),
            "evidence_n": int(sum(r.get("evidence_any") is not None for r in a_rows)),
            "mean_prompt_tokens": _mean_opt(a_rows, "prompt_tokens"),
            "api_errors": int(sum(int(r["api_error"]) for r in a_rows)),
            "gate_active": int(sum(int(r["gate_active"]) for r in a_rows)),
            "by_question_type": {
                qt: _mean_opt([r for r in a_rows if str(r["question_type"]) == qt], "is_correct")
                for qt in qtypes
            },
        }

    cluster_map: dict[str, str] | None = None
    if cluster_key is not None:
        cluster_map = {str(r["qid"]): str(r[cluster_key]) for r in by_arm[arms[0]]}

    pairs = (
        (ARM_CHELATED_V4, ARM_FLAT_BGEM3_TM),
        (ARM_CHELATED_V4, ARM_LEGACY_MINILM),
        (ARM_CHELATED_V4, ARM_DUALGRAN_RERANKED),
        (ARM_DUALGRAN_RERANKED, ARM_FLAT_BGEM3_TM),
        (ARM_DUALGRAN_RERANKED, ARM_LEGACY_MINILM),
        (ARM_FLAT_BGEM3_TM, ARM_LEGACY_MINILM),
    )
    comparisons: dict[str, Any] = {}
    for a, b in pairs:
        if a in by_arm and b in by_arm:
            sc_a = {str(r["qid"]): float(r["is_correct"]) for r in by_arm[a]}
            sc_b = {str(r["qid"]): float(r["is_correct"]) for r in by_arm[b]}
            comparisons[f"{a}_vs_{b}"] = paired_bootstrap_wave6(
                sc_a,
                sc_b,
                n_bootstrap=20_000,
                seed=42,
                expected_ids=expected_ids,
                cluster_by_id=cluster_map,
            )

    parity_info: dict[str, Any] = {}
    if ARM_CHELATED_V4 in by_arm and ARM_DUALGRAN_RERANKED in by_arm:
        dg_by_qid = {str(r["qid"]): r for r in by_arm[ARM_DUALGRAN_RERANKED]}
        v4_rows = by_arm[ARM_CHELATED_V4]
        inactive = [r for r in v4_rows if not int(r["gate_active"])]
        if any(r["prompt_sha256"] != dg_by_qid[str(r["qid"])]["prompt_sha256"] for r in inactive):
            raise RuntimeError("Inactive-gate prompt parity check failed!")
        parity_info = {
            "inactive_gate_prompt_parity_n": len(inactive),
            "same_prompt_as_no_ek5_n": int(
                sum(r["prompt_sha256"] == dg_by_qid[str(r["qid"])]["prompt_sha256"] for r in v4_rows)
            ),
        }

    return {
        "n": len(expected_ids),
        "arms": arm_metrics,
        "comparisons": comparisons,
        **parity_info,
    }


def _run_phase1_dev_tuning_and_freeze(
    *,
    loc_convs: Sequence[dict[str, Any]],
    lme_items: Sequence[dict[str, Any]],
    ku_echo_by_id: Mapping[str, dict[str, Any]],
    lme_split_map: Mapping[str, str],
    flat_enc: CachedEncoder,
    bge_enc: CachedEncoder,
    reranker: CachedBGEReranker,
    answer_runner: PromptAnswerRunner,
    judge_runner: R5JudgeRunner,
    output_dir: Path,
) -> tuple[dict[str, Any], str]:
    """Phase 1: Tune compact `ChelatedConversationalV4` strictly on DEV (`LoCoMo 0..4` & `LME DEV`) & freeze."""
    print("\n=== PHASE 1: STRICT DEV-ONLY TUNING & SHA-256 FREEZE ===", flush=True)
    all_loc_qs = r5_locomo_questions(loc_convs)
    dev_loc_qs = [q for q in all_loc_qs if q.group is not None and int(q.group) < 5]
    dev_lme_items = [it for it in lme_items if lme_split_map[str(it["question_id"])] == "DEV"]
    if len(dev_loc_qs) != 150 or len(dev_lme_items) != 249:
        raise RuntimeError(f"Unexpected DEV sizes: loc={len(dev_loc_qs)}, lme={len(dev_lme_items)}")

    split_lookup: dict[tuple[str, str], str] = {(LOCOMO, q.qid): "DEV" for q in dev_loc_qs}
    for it in dev_lme_items:
        split_lookup[(LME, str(it["question_id"]))] = "DEV"

    # Pre-extract DEV conversations ONLY (conversations 5..9 and LME HELD_OUT_TEST untouched!)
    dev_loc_prepared: list[tuple[list[Question], list[dict[str, Any]], list[ConversationTurn]]] = []
    for ci in range(5):
        leg = _extract_locomo_units(loc_convs[ci])
        turns = [_split_legacy_turn(r) for r in leg]
        qs = [q for q in dev_loc_qs if q.group == ci]
        dev_loc_prepared.append((qs, leg, turns))

    dev_lme_prepared: list[tuple[Question, list[dict[str, Any]], list[ConversationTurn], list[ConversationTurn] | None]] = []
    for it in dev_lme_items:
        qid = str(it["question_id"])
        q = Question(
            LME,
            qid,
            str(it["question_type"]),
            str(it["question"]),
            str(it["answer"]),
            date=str(it["question_date"]),
            abstention=qid.endswith("_abs"),
            evidence=tuple(it["answer_session_ids"]),
        )
        leg = _extract_lme_units(it)
        turns = lme_memory(it)
        echoes = (
            [ConversationTurn(**t) for t in ku_echo_by_id[qid]["turns"]]
            if qid in ku_echo_by_id
            else None
        )
        dev_lme_prepared.append((q, leg, turns, echoes))

    dev_arms = (ARM_FLAT_BGEM3_TM, ARM_CHELATED_V4)
    prompts_pool: dict[str, tuple[str, str, int]] = {}
    jobs_by_cand: dict[str, list[EvalJob]] = {}

    for cand in DEV_CANDIDATE_GRID:
        cid = cand["candidate_id"]
        c_jobs: list[EvalJob] = []
        for qs, leg, turns in dev_loc_prepared:
            c_jobs.extend(
                _build_jobs_for_questions(
                    qs,
                    CLEAN,
                    split_lookup,
                    leg,
                    turns,
                    flat_enc=flat_enc,
                    bge_enc=bge_enc,
                    reranker=reranker,
                    cand_cfg=cand,
                    arms_to_run=dev_arms,
                    prompts_out=prompts_pool,
                )
            )
        for q, leg, turns, echoes in dev_lme_prepared:
            c_jobs.extend(
                _build_jobs_for_questions(
                    [q],
                    CLEAN,
                    split_lookup,
                    leg,
                    turns,
                    flat_enc=flat_enc,
                    bge_enc=bge_enc,
                    reranker=reranker,
                    cand_cfg=cand,
                    arms_to_run=dev_arms,
                    prompts_out=prompts_pool,
                )
            )
            if echoes is not None:
                stressed_leg = sorted(
                    leg + [r5_legacy_row(t) for t in echoes], key=lambda r: (r["ts"], r["unit_id"])
                )
                c_jobs.extend(
                    _build_jobs_for_questions(
                        [q],
                        STRESS,
                        split_lookup,
                        stressed_leg,
                        turns + echoes,
                        flat_enc=flat_enc,
                        bge_enc=bge_enc,
                        reranker=reranker,
                        cand_cfg=cand,
                        arms_to_run=dev_arms,
                        prompts_out=prompts_pool,
                    )
                )
        jobs_by_cand[cid] = c_jobs

    answer_runner.execute_prompts(prompts_pool)
    all_dev_jobs = [j for c_jobs in jobs_by_cand.values() for j in c_jobs]
    judge_runner.grade_jobs(all_dev_jobs, answer_runner)

    cand_reports: list[dict[str, Any]] = []
    best_cand: dict[str, Any] | None = None
    best_score = -1e9

    for cand in DEV_CANDIDATE_GRID:
        cid = cand["candidate_id"]
        c_rows = _materialize_rows(jobs_by_cand[cid], answer_runner, judge_runner)

        loc_v4 = [r for r in c_rows if r["suite"] == LOCOMO and r["regime"] == CLEAN and r["arm"] == ARM_CHELATED_V4]
        loc_tm = [r for r in c_rows if r["suite"] == LOCOMO and r["regime"] == CLEAN and r["arm"] == ARM_FLAT_BGEM3_TM]
        lme_v4 = [r for r in c_rows if r["suite"] == LME and r["regime"] == CLEAN and r["arm"] == ARM_CHELATED_V4]
        lme_tm = [r for r in c_rows if r["suite"] == LME and r["regime"] == CLEAN and r["arm"] == ARM_FLAT_BGEM3_TM]
        ku_v4 = [r for r in c_rows if r["suite"] == LME and r["regime"] == STRESS and r["arm"] == ARM_CHELATED_V4]
        ku_tm = [r for r in c_rows if r["suite"] == LME and r["regime"] == STRESS and r["arm"] == ARM_FLAT_BGEM3_TM]

        loc_acc = float(_mean_opt(loc_v4, "is_correct") or 0.0)
        lme_acc = float(_mean_opt(lme_v4, "is_correct") or 0.0)
        ku_acc = float(_mean_opt(ku_v4, "is_correct") or 0.0)
        loc_tok = float(_mean_opt(loc_v4, "prompt_tokens") or 0.0)
        lme_tok = float(_mean_opt(lme_v4, "prompt_tokens") or 0.0)
        weighted_dev_tok = (150.0 * loc_tok + 249.0 * lme_tok) / 399.0

        loc_tm_acc = float(_mean_opt(loc_tm, "is_correct") or 0.0)
        lme_tm_acc = float(_mean_opt(lme_tm, "is_correct") or 0.0)
        ku_tm_acc = float(_mean_opt(ku_tm, "is_correct") or 0.0)
        loc_tm_tok = float(_mean_opt(loc_tm, "prompt_tokens") or 0.0)
        lme_tm_tok = float(_mean_opt(lme_tm, "prompt_tokens") or 0.0)

        in_token_band = 1400.0 <= weighted_dev_tok <= 2200.0 and lme_tok <= 2250.0
        # Balanced DEV objective: equal weight on LoCoMo DEV clean, LME DEV clean, and 0.5 weight on DEV KU stress
        composite = 0.40 * loc_acc + 0.40 * lme_acc + 0.20 * ku_acc - (0.10 if not in_token_band else 0.0)

        rep = {
            "candidate_id": cid,
            "in_target_token_band_1400_2200": in_token_band,
            "weighted_dev_mean_prompt_tokens": round(weighted_dev_tok, 2),
            "dev_composite_score": round(composite, 6),
            "locomo_dev_n150": {
                "v4_accuracy": round(loc_acc, 6),
                "v4_evidence_all_recall": round(float(_mean_opt(loc_v4, "evidence_all") or 0.0), 6),
                "v4_mean_prompt_tokens": round(loc_tok, 2),
                "flat_bgem3_tm_accuracy": round(loc_tm_acc, 6),
                "flat_bgem3_tm_mean_prompt_tokens": round(loc_tm_tok, 2),
                "delta_vs_flat_bgem3_tm": round(loc_acc - loc_tm_acc, 6),
            },
            "lme_dev_n249": {
                "v4_accuracy": round(lme_acc, 6),
                "v4_evidence_all_recall": round(float(_mean_opt(lme_v4, "evidence_all") or 0.0), 6),
                "v4_mean_prompt_tokens": round(lme_tok, 2),
                "flat_bgem3_tm_accuracy": round(lme_tm_acc, 6),
                "flat_bgem3_tm_mean_prompt_tokens": round(lme_tm_tok, 2),
                "delta_vs_flat_bgem3_tm": round(lme_acc - lme_tm_acc, 6),
            },
            "lme_dev_ku_stress_n39": {
                "v4_accuracy": round(ku_acc, 6),
                "flat_bgem3_tm_accuracy": round(ku_tm_acc, 6),
                "delta_vs_flat_bgem3_tm": round(ku_acc - ku_tm_acc, 6),
            },
        }
        cand_reports.append(rep)
        print(
            f"  [DEV {cid}] LoCoMo={loc_acc:.4f} ({loc_tok:.0f} tok) | "
            f"LME={lme_acc:.4f} ({lme_tok:.0f} tok) | KU_stress={ku_acc:.4f} | "
            f"weighted_tok={weighted_dev_tok:.0f} | score={composite:.4f}",
            flush=True,
        )
        if composite > best_score:
            best_score = composite
            best_cand = cand

    assert best_cand is not None
    frozen_payload = {
        "schema_version": "wave6_pm1b_v4_frozen_dev_config_v1",
        "frozen_at_utc": _utc_now_iso(),
        "tuning_split": "STRICTLY DEV ONLY: LoCoMo conversations 0..4 (N=150) + LongMemEval-S frozen SHA-256 DEV half (N=249, KU=39)",
        "holdout_touched_prior_to_freeze": False,
        "virgin_holdout_touched_prior_to_freeze": False,
        "j4_split_seed": J4_SPLIT_SEED,
        "selected_candidate_id": best_cand["candidate_id"],
        "selected_config": best_cand,
        "dev_grid_evaluations": cand_reports,
    }
    frozen_path = output_dir / "frozen_dev_config.json"
    write_json_atomic(frozen_path, frozen_payload)
    frozen_sha = _sha256_file(frozen_path)
    print(
        f"  [FROZEN] Selected {best_cand['candidate_id']} -> {frozen_path.name} (sha256={frozen_sha})",
        flush=True,
    )
    return frozen_payload, frozen_sha


def _load_j4_historical_reference() -> dict[str, Any]:
    """Read-only extraction of original R6 J4 Look #1 scores for side-by-side K2 disclosure."""
    r6_manifest_path = ROOT / "artifacts" / "r6_pm1b_conversational_sota" / "r6_pm1b_manifest.json"
    if not r6_manifest_path.is_file():
        return {"present": False}
    m = json.loads(r6_manifest_path.read_text(encoding="utf-8"))
    ho = m["primary_held_out_confirmatory"]["suites"]
    pooled = m["suites"]
    return {
        "source_manifest": "artifacts/r6_pm1b_conversational_sota/r6_pm1b_manifest.json",
        "source_manifest_sha256": _sha256_file(r6_manifest_path),
        "locomo_5_9_clean_n150": {
            arm: {
                "accuracy": ho[LOCOMO][CLEAN]["arms"][arm]["accuracy"],
                "mean_prompt_tokens": ho[LOCOMO][CLEAN]["arms"][arm]["mean_prompt_tokens"],
            }
            for arm in ho[LOCOMO][CLEAN]["arms"]
        },
        "locomo_5_9_echo_stress_n150": {
            arm: {
                "accuracy": ho[LOCOMO][STRESS]["arms"][arm]["accuracy"],
                "mean_prompt_tokens": ho[LOCOMO][STRESS]["arms"][arm]["mean_prompt_tokens"],
            }
            for arm in ho[LOCOMO][STRESS]["arms"]
        },
        "lme_test_clean_n251": {
            arm: {
                "accuracy": ho[LME][CLEAN]["arms"][arm]["accuracy"],
                "mean_prompt_tokens": ho[LME][CLEAN]["arms"][arm]["mean_prompt_tokens"],
            }
            for arm in ho[LME][CLEAN]["arms"]
        },
        "lme_test_ku_echo_stress_n39": {
            arm: {
                "accuracy": ho[LME][STRESS]["arms"][arm]["accuracy"],
                "evidence_all_recall": ho[LME][STRESS]["arms"][arm]["evidence_all_recall"],
            }
            for arm in ho[LME][STRESS]["arms"]
        },
        "lme_pooled_ku_echo_stress_n78": {
            arm: {
                "accuracy": pooled[LME][STRESS]["arms"][arm]["accuracy"],
                "evidence_all_recall": pooled[LME][STRESS]["arms"][arm]["evidence_all_recall"],
            }
            for arm in pooled[LME][STRESS]["arms"]
        },
        "lme_pooled_ku_echo_stress_r5_n78": {
            arm: {
                "accuracy": pooled[LME][STRESS_R5]["arms"][arm]["accuracy"],
                "evidence_all_recall": pooled[LME][STRESS_R5]["arms"][arm]["evidence_all_recall"],
            }
            for arm in pooled[LME][STRESS_R5]["arms"]
        },
    }


def _run_phase2_post_j4_look2(
    *,
    frozen_cfg: Mapping[str, Any],
    loc_convs: Sequence[dict[str, Any]],
    lme_items: Sequence[dict[str, Any]],
    ku_echo_by_id: Mapping[str, dict[str, Any]],
    lme_split_map: Mapping[str, str],
    flat_enc: CachedEncoder,
    bge_enc: CachedEncoder,
    reranker: CachedBGEReranker,
    answer_runner: PromptAnswerRunner,
    judge_runner: R5JudgeRunner,
    output_dir: Path,
) -> tuple[dict[str, Any], Path]:
    """Phase 2: Evaluate frozen config on LoCoMo & LME, explicitly labeled Post-J4 Exploratory Look #2."""
    phase2_started_utc = _utc_now_iso()
    print("\n=== PHASE 2: POST-J4 EXPLORATORY LOOK #2 (BURNED HOLDOUT, alpha=0.025) ===", flush=True)
    selected_cfg = frozen_cfg["selected_config"]

    loc_questions = r5_locomo_questions(loc_convs)
    split_lookup: dict[tuple[str, str], str] = {
        (LOCOMO, q.qid): question_split_tag(q, lme_split_map) for q in loc_questions
    }
    for it in lme_items:
        qid = str(it["question_id"])
        split_lookup[(LME, qid)] = lme_split_map[qid]

    prompts_pool: dict[str, tuple[str, str, int]] = {}
    all_jobs: list[EvalJob] = []

    # 1) LoCoMo 0..9 (clean + echo_stress) — partitioned into DEV (0..4) and Burned Holdout Look #2 (5..9)
    for ci, conv in enumerate(loc_convs):
        leg = _extract_locomo_units(conv)
        turns = [_split_legacy_turn(r) for r in leg]
        qs = [q for q in loc_questions if q.group == ci]
        flat_vecs = flat_enc.encode([r["text"] for r in leg])
        echoes = neutral_locomo_echoes(turns, flat_vecs)
        all_jobs.extend(
            _build_jobs_for_questions(
                qs,
                CLEAN,
                split_lookup,
                leg,
                turns,
                flat_enc=flat_enc,
                bge_enc=bge_enc,
                reranker=reranker,
                cand_cfg=selected_cfg,
                arms_to_run=ARMS_V4,
                prompts_out=prompts_pool,
            )
        )
        all_jobs.extend(
            _build_jobs_for_questions(
                qs,
                STRESS,
                split_lookup,
                leg + [r5_legacy_row(t) for t in echoes],
                turns + echoes,
                flat_enc=flat_enc,
                bge_enc=bge_enc,
                reranker=reranker,
                cand_cfg=selected_cfg,
                arms_to_run=ARMS_V4,
                prompts_out=prompts_pool,
            )
        )
        print(f"  [Look#2] Indexed LoCoMo conversation {ci + 1}/10", flush=True)

    # 2) LongMemEval-S N=500 clean + N=78 KU echo_stress & echo_stress_r5
    for idx, it in enumerate(lme_items, start=1):
        qid = str(it["question_id"])
        q = Question(
            LME,
            qid,
            str(it["question_type"]),
            str(it["question"]),
            str(it["answer"]),
            date=str(it["question_date"]),
            abstention=qid.endswith("_abs"),
            evidence=tuple(it["answer_session_ids"]),
        )
        leg = _extract_lme_units(it)
        turns = lme_memory(it)
        all_jobs.extend(
            _build_jobs_for_questions(
                [q],
                CLEAN,
                split_lookup,
                leg,
                turns,
                flat_enc=flat_enc,
                bge_enc=bge_enc,
                reranker=reranker,
                cand_cfg=selected_cfg,
                arms_to_run=ARMS_V4,
                prompts_out=prompts_pool,
            )
        )
        if qid in ku_echo_by_id:
            echoes_nat = [ConversationTurn(**t) for t in ku_echo_by_id[qid]["turns"]]
            stressed_nat = sorted(
                leg + [r5_legacy_row(t) for t in echoes_nat], key=lambda r: (r["ts"], r["unit_id"])
            )
            all_jobs.extend(
                _build_jobs_for_questions(
                    [q],
                    STRESS,
                    split_lookup,
                    stressed_nat,
                    turns + echoes_nat,
                    flat_enc=flat_enc,
                    bge_enc=bge_enc,
                    reranker=reranker,
                    cand_cfg=selected_cfg,
                    arms_to_run=ARMS_V4,
                    prompts_out=prompts_pool,
                )
            )
            leg_r5 = _extract_lme_units(it, inject_echo_stress=True)
            echoes_r5 = [
                ConversationTurn(
                    str(u["unit_id"]),
                    str(u["session_id"]),
                    str(u["date_str"]),
                    float(u["ts"]),
                    str(u["text"]),
                    str(u["raw_text"]),
                    "User",
                )
                for u in leg_r5
                if str(u["unit_id"]).startswith("echo_")
            ]
            all_jobs.extend(
                _build_jobs_for_questions(
                    [q],
                    STRESS_R5,
                    split_lookup,
                    leg_r5,
                    turns + echoes_r5,
                    flat_enc=flat_enc,
                    bge_enc=bge_enc,
                    reranker=reranker,
                    cand_cfg=selected_cfg,
                    arms_to_run=ARMS_V4,
                    prompts_out=prompts_pool,
                )
            )
        if idx % 100 == 0:
            print(f"  [Look#2] Indexed LongMemEval-S {idx}/{len(lme_items)} items", flush=True)

    answer_runner.execute_prompts(prompts_pool)
    judge_runner.grade_jobs(all_jobs, answer_runner)
    rows = _materialize_rows(all_jobs, answer_runner, judge_runner)

    csv_path = output_dir / "wave6_pm1b_v4_look2_rows.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    loc_test_ids = [q.qid for q in loc_questions if q.group is not None and int(q.group) >= 5]
    loc_dev_ids = [q.qid for q in loc_questions if q.group is not None and int(q.group) < 5]
    loc_all_ids = [q.qid for q in loc_questions]

    lme_all_ids = [str(it["question_id"]) for it in lme_items]
    lme_test_ids = [qid for qid in lme_all_ids if lme_split_map[qid] == "HELD_OUT_TEST"]
    lme_dev_ids = [qid for qid in lme_all_ids if lme_split_map[qid] == "DEV"]

    ku_all_ids = [str(it["question_id"]) for it in lme_items if it["question_type"] == "knowledge-update"]
    ku_test_ids = [qid for qid in ku_all_ids if lme_split_map[qid] == "HELD_OUT_TEST"]
    ku_dev_ids = [qid for qid in ku_all_ids if lme_split_map[qid] == "DEV"]

    look2_loc_clean = _summarize_wave6_slice(
        [r for r in rows if r["suite"] == LOCOMO and r["regime"] == CLEAN and r["split"] == "HELD_OUT_TEST"],
        loc_test_ids,
        cluster_key="conversation",
    )
    look2_loc_stress = _summarize_wave6_slice(
        [r for r in rows if r["suite"] == LOCOMO and r["regime"] == STRESS and r["split"] == "HELD_OUT_TEST"],
        loc_test_ids,
        cluster_key="conversation",
    )
    look2_lme_clean = _summarize_wave6_slice(
        [r for r in rows if r["suite"] == LME and r["regime"] == CLEAN and r["split"] == "HELD_OUT_TEST"],
        lme_test_ids,
    )
    look2_lme_ku39_nat = _summarize_wave6_slice(
        [r for r in rows if r["suite"] == LME and r["regime"] == STRESS and r["split"] == "HELD_OUT_TEST"],
        ku_test_ids,
    )
    look2_lme_ku39_r5 = _summarize_wave6_slice(
        [r for r in rows if r["suite"] == LME and r["regime"] == STRESS_R5 and r["split"] == "HELD_OUT_TEST"],
        ku_test_ids,
    )
    look2_lme_ku78_nat = _summarize_wave6_slice(
        [r for r in rows if r["suite"] == LME and r["regime"] == STRESS],
        ku_all_ids,
    )
    look2_lme_ku78_r5 = _summarize_wave6_slice(
        [r for r in rows if r["suite"] == LME and r["regime"] == STRESS_R5],
        ku_all_ids,
    )

    # SSU slice audit (35 held-out + 70 pooled)
    ssu_test_rows = [
        r
        for r in rows
        if r["suite"] == LME
        and r["regime"] == CLEAN
        and r["split"] == "HELD_OUT_TEST"
        and r["question_type"] == "single-session-user"
    ]
    ssu_test_ids = [str(r["qid"]) for r in ssu_test_rows if r["arm"] == ARM_LEGACY_MINILM]
    ssu_test_summary = _summarize_wave6_slice(ssu_test_rows, ssu_test_ids)

    ssu_all_rows = [
        r
        for r in rows
        if r["suite"] == LME and r["regime"] == CLEAN and r["question_type"] == "single-session-user"
    ]
    ssu_all_ids = [str(r["qid"]) for r in ssu_all_rows if r["arm"] == ARM_LEGACY_MINILM]
    ssu_all_summary = _summarize_wave6_slice(ssu_all_rows, ssu_all_ids)

    dev_summary = {
        "split_definition": "LoCoMo conversations 0..4 (N=150) + LongMemEval-S frozen SHA-256 DEV half (N=249, KU=39)",
        "LoCoMo_0_4_clean_n150": _summarize_wave6_slice(
            [r for r in rows if r["suite"] == LOCOMO and r["regime"] == CLEAN and r["split"] == "DEV"],
            loc_dev_ids,
            cluster_key="conversation",
        ),
        "LoCoMo_0_4_echo_stress_n150": _summarize_wave6_slice(
            [r for r in rows if r["suite"] == LOCOMO and r["regime"] == STRESS and r["split"] == "DEV"],
            loc_dev_ids,
            cluster_key="conversation",
        ),
        "LME_DEV_clean_n249": _summarize_wave6_slice(
            [r for r in rows if r["suite"] == LME and r["regime"] == CLEAN and r["split"] == "DEV"],
            lme_dev_ids,
        ),
        "LME_DEV_ku_echo_stress_n39": _summarize_wave6_slice(
            [r for r in rows if r["suite"] == LME and r["regime"] == STRESS and r["split"] == "DEV"],
            ku_dev_ids,
        ),
    }

    pooled_secondary = {
        "accounting_label": "SECONDARY_POOLED_CONTAINS_DEV_DO_NOT_CITE_AS_HELD_OUT",
        "LoCoMo_300_clean": _summarize_wave6_slice(
            [r for r in rows if r["suite"] == LOCOMO and r["regime"] == CLEAN],
            loc_all_ids,
            cluster_key="conversation",
        ),
        "LoCoMo_300_echo_stress": _summarize_wave6_slice(
            [r for r in rows if r["suite"] == LOCOMO and r["regime"] == STRESS],
            loc_all_ids,
            cluster_key="conversation",
        ),
        "LME_500_clean": _summarize_wave6_slice(
            [r for r in rows if r["suite"] == LME and r["regime"] == CLEAN],
            lme_all_ids,
        ),
        "LME_SSU_70_clean": ssu_all_summary,
    }

    look2_holm = holm_bonferroni_stepdown(
        {
            "locomo_5_9_clean_v4_vs_flat_bgem3": look2_loc_clean["comparisons"][f"{ARM_CHELATED_V4}_vs_{ARM_FLAT_BGEM3_TM}"]["p_value_two_sided"],
            "locomo_5_9_stress_v4_vs_flat_bgem3": look2_loc_stress["comparisons"][f"{ARM_CHELATED_V4}_vs_{ARM_FLAT_BGEM3_TM}"]["p_value_two_sided"],
            "lme_test_clean_v4_vs_flat_bgem3": look2_lme_clean["comparisons"][f"{ARM_CHELATED_V4}_vs_{ARM_FLAT_BGEM3_TM}"]["p_value_two_sided"],
            "lme_ku78_nat_stress_v4_vs_minilm": look2_lme_ku78_nat["comparisons"][f"{ARM_CHELATED_V4}_vs_{ARM_LEGACY_MINILM}"]["p_value_two_sided"],
            "lme_ku78_r5_stress_v4_vs_minilm": look2_lme_ku78_r5["comparisons"][f"{ARM_CHELATED_V4}_vs_{ARM_LEGACY_MINILM}"]["p_value_two_sided"],
        },
        alpha=0.025,
    )

    return (
        {
            "post_j4_exploratory_look_2_burned_holdout": {
                "accounting_label": "POST_J4_EXPLORATORY_LOOK_2_BURNED_HOLDOUT",
                "nominal_alpha": 0.025,
                "tango_bonf2_z": 2.241402727604947,
                "started_after_dev_freeze_at_utc": phase2_started_utc,
                "disclosure": (
                    "LoCoMo conversations 5..9 and LongMemEval-S HELD_OUT_TEST were previously evaluated once "
                    "under R6 J4. Per Wave-5 Cross-Judgment Guardrail K2, this re-evaluation is explicitly "
                    "accounted at Bonferroni Look #2 nominal alpha=0.025 (reporting p5.0, p2.5, p1.25, and "
                    "Tango z=2.2414) alongside original J4 scores, and treated as secondary to the virgin "
                    "lineage-disjoint conversational holdout."
                ),
                "LoCoMo_5_9_clean_n150": look2_loc_clean,
                "LoCoMo_5_9_echo_stress_n150": look2_loc_stress,
                "LME_TEST_clean_n251": look2_lme_clean,
                "LME_TEST_ssu_clean_n35": ssu_test_summary,
                "LME_TEST_ku_echo_stress_n39": look2_lme_ku39_nat,
                "LME_TEST_ku_echo_stress_r5_n39": look2_lme_ku39_r5,
                "LME_KU_echo_stress_full_n78": look2_lme_ku78_nat,
                "LME_KU_echo_stress_r5_full_n78": look2_lme_ku78_r5,
                "holm_bonferroni_alpha_0_025": look2_holm,
                "j4_historical_side_by_side": _load_j4_historical_reference(),
            },
            "dev_tuning_split_4arm_full": dev_summary,
            "secondary_pooled_full_dataset": pooled_secondary,
        },
        csv_path,
    )


def _prepare_virgin_retrieval_and_answers(
    *,
    virgin_convs: Sequence[VirginConversation],
    frozen_cfg: Mapping[str, Any],
    flat_enc: CachedEncoder,
    bge_enc: CachedEncoder,
    reranker: CachedBGEReranker,
    answer_runner: PromptAnswerRunner,
) -> list[dict[str, Any]]:
    """Retrieve all 4 arms on Virgin N=120 using frozen DEV config & answer ONCE on Nemotron-30B."""
    selected_cfg = frozen_cfg["selected_config"]
    v_cfg = selected_cfg[VIRGIN_SUITE_NAME]
    pool_size = int(selected_cfg["pool_size"])
    rerank_alpha = float(selected_cfg["rerank_alpha"])
    rerank_max_chars = int(selected_cfg["rerank_max_chars"])
    rerank_prev_turn = bool(selected_cfg["rerank_prev_turn"])

    prompts_pool: dict[str, tuple[str, str, int]] = {}
    records: list[dict[str, Any]] = []

    for conv in virgin_convs:
        legacy_rows = [
            {
                "unit_id": t.turn_id,
                "session_id": t.session_id,
                "date_str": t.date,
                "ts": t.timestamp,
                "text": t.text,
                "raw_text": t.claim_text,
                "user_text": t.claim_text,
            }
            for t in conv.turns
        ]
        text_bank = [r["text"] for r in legacy_rows]
        bm25 = TurnBM25(text_bank)
        flat_vecs = flat_enc.encode(text_bank)
        flat_q_vecs = flat_enc.encode([q.question for q in conv.questions])
        engine = ChelatedConversationalV4(conv.turns, bge_enc, reranker)

        for qi, vq in enumerate(conv.questions):
            # 1. Legacy_MiniLM_Flat_Top5
            idxs, _ = _retrieve_lme_r5_arm(
                "M1_hybrid_flat", legacy_rows, flat_vecs, flat_q_vecs[qi], bm25.score(vq.question)
            )
            ord_idx = sorted(idxs, key=lambda i: legacy_rows[i]["ts"])
            l_rows = [legacy_rows[i] for i in ord_idx]
            minilm_ctx = "\n".join(f"- [Session Date: {r['date_str']}] {r['text']}" for r in l_rows)
            minilm_ids = tuple(str(r["unit_id"]) for r in l_rows)

            # 2. Flat_BGEM3_TokenMatched
            tm_ctx, tm_ids, _ = flat_bgem3_token_matched_context_v4(
                engine,
                vq.question,
                context_chars=int(v_cfg["context_chars"]),
                turn_max_chars=int(v_cfg["turn_max_chars"]),
            )

            # 3 & 4. DualGran_Reranked_NoEK5 & Chelated_v4_Reranked_EK5v3
            ranking, s1_scores, _ = engine.rank_with_scores(vq.question)
            common_kw = {
                "anchor_count": int(v_cfg["anchor_count"]),
                "window_radius": int(v_cfg["window_radius"]),
                "context_chars": int(v_cfg["context_chars"]),
                "pool_size": pool_size,
                "turn_max_chars": int(v_cfg["turn_max_chars"]),
                "neighbor_max_chars": int(v_cfg["neighbor_max_chars"]),
                "rerank_alpha": rerank_alpha,
                "rerank_max_chars": rerank_max_chars,
                "rerank_prev_turn": rerank_prev_turn,
                "speaker_attribution_header": bool(v_cfg["speaker_attribution_header"]),
                "speaker_anchor_reserve": int(v_cfg["speaker_anchor_reserve"]),
                "ranked": ranking,
                "stage1_scores": s1_scores,
            }
            rec_dg = engine.retrieve(vq.question, epistemic=False, **common_kw)
            rec_v4 = engine.retrieve(vq.question, epistemic=True, **common_kw)
            if not rec_v4.gate_active and rec_dg != rec_v4:
                raise RuntimeError(f"Virgin inactive-gate parity violation on {vq.qid}")

            arm_bundles: list[tuple[str, str, tuple[str, ...], ConversationRecall | None]] = [
                (ARM_LEGACY_MINILM, minilm_ctx, minilm_ids, None),
                (ARM_FLAT_BGEM3_TM, tm_ctx, tm_ids, None),
                (
                    ARM_DUALGRAN_RERANKED,
                    rec_dg.context,
                    tuple(t.turn.turn_id for t in rec_dg.turns),
                    rec_dg,
                ),
                (
                    ARM_CHELATED_V4,
                    rec_v4.context,
                    tuple(t.turn.turn_id for t in rec_v4.turns),
                    rec_v4,
                ),
            ]
            labeled_turns = set(vq.evidence_turn_ids)
            for arm_name, ctx, sel_ids, rec in arm_bundles:
                usr_prompt = (
                    f"Current Question Date: {vq.question_date}\n"
                    f"Conversation Excerpts:\n{ctx}\n\n"
                    f"Question: {vq.question}\nAnswer:"
                )
                req = (VIRGIN_ANSWER_SYSTEM, usr_prompt, 96)
                digest = hashlib.sha256(json.dumps(req, ensure_ascii=False).encode("utf-8")).hexdigest()
                prompts_pool.setdefault(digest, req)
                sel_set = set(sel_ids)
                records.append(
                    {
                        "vq": vq,
                        "arm": arm_name,
                        "prompt_sha256": digest,
                        "selected_ids": sel_ids,
                        "evidence_any": int(bool(labeled_turns & sel_set)) if labeled_turns else None,
                        "evidence_all": int(labeled_turns.issubset(sel_set)) if labeled_turns else None,
                        "gate_active": int(rec.gate_active) if rec else 0,
                        "echo_clusters": rec.echo_clusters if rec else (),
                        "supersessions": rec.supersessions if rec else (),
                    }
                )

    answer_runner.execute_prompts(prompts_pool)
    return records


def _grade_virgin_blind_with_qwen25_7b(
    *,
    virgin_records: Sequence[dict[str, Any]],
    virgin_convs: Sequence[VirginConversation],
    answer_runner: PromptAnswerRunner,
    work_dir: Path,
    output_dir: Path,
) -> tuple[dict[str, Any], Path, Path, dict[str, Any]]:
    """Load `Qwen/Qwen2.5-7B-Instruct` (`bfloat16`) on RTX 3090 and blind-grade all 120 x 4 predictions."""
    import torch
    from huggingface_hub import snapshot_download
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("\n=== PHASE 3B: LOADING CROSS-FAMILY BLIND JUDGE (Qwen2.5-7B-Instruct bf16 on RTX 3090) ===", flush=True)
    snap_path = Path(snapshot_download(JUDGE_MODEL_ID, local_files_only=True))
    vram_free_before = round(torch.cuda.mem_get_info()[0] / (1024.0**2), 1)
    t_load0 = time.perf_counter()
    tok = AutoTokenizer.from_pretrained(str(snap_path), local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(snap_path),
        torch_dtype=torch.bfloat16,
        local_files_only=True,
        attn_implementation="sdpa",
    ).to("cuda")
    model.eval()
    load_sec = round(time.perf_counter() - t_load0, 2)
    vram_alloc_mib = round(torch.cuda.memory_allocated() / (1024.0**2), 1)
    print(
        f"  [Qwen2.5-7B] Loaded revision {snap_path.name} in {load_sec}s "
        f"(free_before={vram_free_before} MiB, allocated={vram_alloc_mib} MiB)",
        flush=True,
    )

    qwen_cache_path = work_dir / "qwen_blind_grade_cache.json"
    qwen_cache: dict[str, str] = (
        json.loads(qwen_cache_path.read_text(encoding="utf-8")) if qwen_cache_path.is_file() else {}
    )
    live_calls = 0

    def qwen_completion_fn(sys_p: str, usr_p: str) -> str:
        nonlocal live_calls
        cache_key = hashlib.sha256(f"{sys_p}\n---\n{usr_p}".encode("utf-8")).hexdigest()
        if cache_key in qwen_cache:
            return qwen_cache[cache_key]
        msgs = [
            {"role": "system", "content": sys_p},
            {"role": "user", "content": usr_p},
        ]
        rendered = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        enc = tok([rendered], return_tensors="pt").to("cuda")
        with torch.inference_mode():
            out_ids = model.generate(
                **enc,
                max_new_tokens=16,
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None,
                pad_token_id=tok.eos_token_id,
            )
        gen = out_ids[0, enc["input_ids"].shape[1] :]
        verdict_txt = tok.decode(gen, skip_special_tokens=True).strip()
        qwen_cache[cache_key] = verdict_txt
        live_calls += 1
        if live_calls % 60 == 0:
            write_json_atomic(qwen_cache_path, qwen_cache)
            print(f"    [Qwen2.5-7B] Completed {live_calls} live blind judge calls...", flush=True)
        return verdict_txt

    blind_judge = BlindCrossFamilySemanticJudge(
        judge_family="alibaba-qwen2.5-7b-instruct",
        generator_family="nvidia-nemotron-3.5-30b",
        completion_fn=qwen_completion_fn,
        salt="wave6-pm1b-v4-virgin-holdout-v1",
    )

    by_qid: dict[str, dict[str, dict[str, Any]]] = {}
    for rec in virgin_records:
        vq: VirginQuestion = rec["vq"]
        by_qid.setdefault(vq.qid, {})[rec["arm"]] = rec

    virgin_rows: list[dict[str, Any]] = []
    all_vqs = [vq for c in virgin_convs for vq in c.questions]
    for vq in all_vqs:
        arm_recs = by_qid[vq.qid]
        arm_preds: dict[str, str] = {}
        for arm_name, r in arm_recs.items():
            pred_raw, _, _, _ = answer_runner.get(r["prompt_sha256"])
            arm_preds[arm_name] = normalized(pred_raw)

        graded = blind_judge.grade_arms_blind(
            item_id=vq.qid,
            question=vq.question,
            gold_reference=vq.gold_reference,
            arm_predictions=arm_preds,
            abstention_required=vq.abstention_required,
        )
        for arm_name in ARMS_V4:
            r = arm_recs[arm_name]
            pred_raw, p_toks, lat_ms, ans_err = answer_runner.get(r["prompt_sha256"])
            g_entry = graded[arm_name]
            virgin_rows.append(
                {
                    "suite": VIRGIN_SUITE_NAME,
                    "regime": "virgin_confirmatory",
                    "split": "VIRGIN_CONFIRMATORY_HOLDOUT",
                    "conversation": vq.conversation_id,
                    "domain": vq.domain,
                    "arm": arm_name,
                    "qid": vq.qid,
                    "question_type": vq.category,
                    "is_abstention": int(vq.abstention_required),
                    "gold": vq.gold_reference,
                    "prediction": normalized(pred_raw),
                    "is_correct": int(g_entry["correct"]),
                    "clean_judge_parse": int(g_entry["clean_judge_parse"]),
                    "blind_slot": int(g_entry["blind_slot"]),
                    "blind_ticket": g_entry["blind_ticket"],
                    "judge_raw": g_entry["judge_raw"],
                    "evidence_any": r["evidence_any"],
                    "evidence_all": r["evidence_all"],
                    "prompt_tokens": p_toks,
                    "latency_ms": round(lat_ms, 2),
                    "api_error": int(ans_err or not g_entry["clean_judge_parse"]),
                    "prompt_sha256": r["prompt_sha256"],
                    "gate_active": r["gate_active"],
                    "selected_ids": json.dumps(r["selected_ids"]),
                    "echo_clusters": json.dumps(r["echo_clusters"]),
                    "supersessions": json.dumps(r["supersessions"]),
                }
            )

    write_json_atomic(qwen_cache_path, qwen_cache)

    # Free GPU memory after blind judging completes
    model.cpu()
    gc.collect()
    torch.cuda.empty_cache()

    virgin_csv_path = output_dir / "wave6_virgin_holdout_rows.csv"
    with virgin_csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(virgin_rows[0].keys()))
        writer.writeheader()
        writer.writerows(virgin_rows)

    audit_path = output_dir / "wave6_virgin_blind_judge_audit.json"
    schema_anomaly_counts: dict[str, dict[str, int]] = {}
    for entry in blind_judge.audit_log:
        a_counts = schema_anomaly_counts.setdefault(
            entry["arm"],
            {
                "empty_output": 0,
                "bare_abstract_token": 0,
                "found_n_matching_records_template": 0,
                "strict_abstain_token": 0,
            },
        )
        for k, flag in entry["schema_anomalies"].items():
            if flag:
                a_counts[k] = a_counts.get(k, 0) + 1

    write_json_atomic(
        audit_path,
        {
            "judge_model": JUDGE_MODEL_ID,
            "judge_revision": snap_path.name,
            "generator_model": answer_runner.model_id,
            "n_items": len(all_vqs),
            "n_blind_evaluations": len(blind_judge.audit_log),
            "clean_parse_rate": float(
                sum(int(e["clean_judge_parse"]) for e in blind_judge.audit_log) / len(blind_judge.audit_log)
            ),
            "schema_anomaly_counts_by_arm": schema_anomaly_counts,
            "audit_log": blind_judge.audit_log,
        },
    )

    all_vqids = [vq.qid for vq in all_vqs]
    overall_summary = _summarize_wave6_slice(
        virgin_rows, all_vqids, arms=ARMS_V4, cluster_key="conversation"
    )

    ku_vqids = [vq.qid for vq in all_vqs if vq.category == "knowledge_update_echo_flood"]
    ku_slice_summary = _summarize_wave6_slice(
        [r for r in virgin_rows if r["question_type"] == "knowledge_update_echo_flood"],
        ku_vqids,
        arms=ARMS_V4,
        cluster_key="conversation",
    )

    by_domain: dict[str, Any] = {}
    for dom in sorted({vq.domain for vq in all_vqs}):
        dom_rows = [r for r in virgin_rows if r["domain"] == dom]
        by_domain[dom] = {
            arm: _mean_opt([r for r in dom_rows if r["arm"] == arm], "is_correct")
            for arm in ARMS_V4
        }

    holm_virgin = holm_bonferroni_stepdown(
        {
            "primary_v4_vs_flat_bgem3_tm_n120": overall_summary["comparisons"][f"{ARM_CHELATED_V4}_vs_{ARM_FLAT_BGEM3_TM}"]["p_value_two_sided"],
            "v4_vs_legacy_minilm_n120": overall_summary["comparisons"][f"{ARM_CHELATED_V4}_vs_{ARM_LEGACY_MINILM}"]["p_value_two_sided"],
            "v4_vs_dualgran_no_ek5_ku_slice_n24": ku_slice_summary["comparisons"][f"{ARM_CHELATED_V4}_vs_{ARM_DUALGRAN_RERANKED}"]["p_value_two_sided"],
            "v4_vs_dualgran_no_ek5_overall_n120": overall_summary["comparisons"][f"{ARM_CHELATED_V4}_vs_{ARM_DUALGRAN_RERANKED}"]["p_value_two_sided"],
        },
        alpha=0.05,
    )

    judge_meta = {
        "model": JUDGE_MODEL_ID,
        "revision": snap_path.name,
        "dtype": "bfloat16",
        "device": torch.cuda.get_device_name(0),
        "load_seconds": load_sec,
        "allocated_vram_mib": vram_alloc_mib,
        "live_judge_calls": live_calls,
    }

    return (
        {
            "accounting_label": "PRIMARY_CONFIRMATORY_VIRGIN_LINEAGE_DISJOINT_HOLDOUT",
            "n_conversations": len(virgin_convs),
            "n_items": len(all_vqs),
            "items_per_conversation": 5,
            "single_shot_confirmatory": True,
            "blind_cross_family_judge": {
                "judge_model": JUDGE_MODEL_ID,
                "judge_revision": snap_path.name,
                "generator_model": answer_runner.model_id,
                "clean_judge_parse_rate": float(
                    sum(int(r["clean_judge_parse"]) for r in virgin_rows) / len(virgin_rows)
                ),
                "schema_anomaly_counts_by_arm": schema_anomaly_counts,
                "audit_file": audit_path.name,
                "audit_sha256": _sha256_file(audit_path),
            },
            "overall_n120_24clusters": overall_summary,
            "knowledge_update_echo_flood_slice_n24": ku_slice_summary,
            "accuracy_by_domain": by_domain,
            "holm_bonferroni_alpha_0_05": holm_virgin,
        },
        virgin_csv_path,
        audit_path,
        judge_meta,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Wave-6 Track A PM-1b v4 & Virgin Conversational Holdout")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "artifacts" / "wave6_pm1b_v4_reranked_sota",
    )
    parser.add_argument("--tunnel-port", type=int, default=DEFAULT_TUNNEL_PORT)
    parser.add_argument("--endpoint", default="")
    parser.add_argument("--model", default="nemotron-lightning")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    work_dir = args.output / "_work"
    work_dir.mkdir(parents=True, exist_ok=True)

    t_start = time.perf_counter()
    tunnel_proc: subprocess.Popen[bytes] | None = None

    try:
        endpoint = args.endpoint.strip()
        if not endpoint:
            tunnel_proc = _ensure_ssh_tunnel(args.tunnel_port, SPARK3_HOST, REMOTE_VLLM_PORT)
            endpoint = f"http://127.0.0.1:{args.tunnel_port}/v1"
        print(f"[Init] Verified Spark #3 endpoint {endpoint} (tunnel_spawned={tunnel_proc is not None})", flush=True)

        # Preflight Nemotron-30B with enable_thinking=False
        models_resp = requests.get(f"{endpoint.rstrip('/')}/models", timeout=15.0)
        models_resp.raise_for_status()
        pf_txt, _, _, pf_err = _call_vllm(endpoint, args.model, "Reply with OK.", "Reply with OK.", max_tokens=8)
        if pf_err or not pf_txt:
            raise RuntimeError(f"Nemotron-30B preflight failed on {endpoint}: {pf_txt}")
        print(f"[Init] Nemotron-30B preflight OK: {pf_txt!r}", flush=True)

        dataset_files = _resolve_dataset_files(work_dir)
        data_hashes = {k: _sha256_file(p) for k, p in dataset_files.items()}
        lme_items = json.loads(dataset_files[LME].read_text(encoding="utf-8"))
        loc_convs = json.loads(dataset_files[LOCOMO].read_text(encoding="utf-8"))
        ku_echo_by_id = _load_validated_ku_echo_fixture(lme_items, data_hashes[LME])
        lme_split_map = build_lme_holdout_split_map(lme_items)

        import torch
        from huggingface_hub import snapshot_download
        from sentence_transformers import SentenceTransformer

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA GPU (RTX 3090) is required for Wave-6 Track A")
        gpu_name = torch.cuda.get_device_name(0)
        snaps = {
            m_name: Path(snapshot_download(m_name, local_files_only=True))
            for m_name in (MINILM, BGE_MODEL, BGE_RERANKER_MODEL)
        }
        flat_model_dir = _ensure_safetensors_model_dir(snaps[MINILM], work_dir)
        bge_model_dir = _ensure_safetensors_model_dir(snaps[BGE_MODEL], work_dir)
        reranker_dir = _ensure_safetensors_model_dir(snaps[BGE_RERANKER_MODEL], work_dir)
        flat_model = SentenceTransformer(str(flat_model_dir), device="cuda", local_files_only=True)
        bge_model = SentenceTransformer(
            str(bge_model_dir),
            device="cuda",
            local_files_only=True,
            model_kwargs={"torch_dtype": torch.float16, "attn_implementation": "sdpa"},
        )
        bge_model.max_seq_length = 384
        reranker = CachedBGEReranker(
            str(reranker_dir),
            device="cuda",
            batch_size=36,
            max_length=512,
            local_files_only=True,
        )
        print(f"[GPU] Loaded MiniLM, BGE-M3, and BGE-Reranker-v2-M3 on {gpu_name}", flush=True)

        flat_enc = CachedEncoder(flat_model, 384, args.batch_size)
        bge_enc = CachedEncoder(bge_model, BGE_DIMENSIONS, args.batch_size)
        answer_runner = PromptAnswerRunner(
            endpoint, args.model, args.workers, work_dir / "answer_cache.json"
        )
        r5_judge = R5JudgeRunner(
            endpoint, args.model, args.workers, work_dir / "r5_grade_cache.json"
        )

        # PHASE 1: Strict DEV-only tuning & cryptographic freeze BEFORE any holdout evaluation
        frozen_cfg, frozen_cfg_sha = _run_phase1_dev_tuning_and_freeze(
            loc_convs=loc_convs,
            lme_items=lme_items,
            ku_echo_by_id=ku_echo_by_id,
            lme_split_map=lme_split_map,
            flat_enc=flat_enc,
            bge_enc=bge_enc,
            reranker=reranker,
            answer_runner=answer_runner,
            judge_runner=r5_judge,
            output_dir=args.output,
        )

        # PHASE 2: Post-J4 Exploratory Look #2 (`LoCoMo 5-9` & `LME TEST`, alpha=0.025)
        look2_bundle, look2_csv_path = _run_phase2_post_j4_look2(
            frozen_cfg=frozen_cfg,
            loc_convs=loc_convs,
            lme_items=lme_items,
            ku_echo_by_id=ku_echo_by_id,
            lme_split_map=lme_split_map,
            flat_enc=flat_enc,
            bge_enc=bge_enc,
            reranker=reranker,
            answer_runner=answer_runner,
            judge_runner=r5_judge,
            output_dir=args.output,
        )

        # PHASE 3A: Construct & verify Virgin Lineage-Disjoint Suite (`N=120` = `24 x 5`), retrieve & answer ONCE
        phase3_started_utc = _utc_now_iso()
        print("\n=== PHASE 3A: VIRGIN LINEAGE-DISJOINT HOLDOUT (N=120) RETRIEVAL & NEMOTRON-30B ANSWERING ===", flush=True)
        virgin_convs = build_virgin_conversational_holdout()
        disjoint_check = verify_lineage_disjointness(
            virgin_convs, locomo_convs=loc_convs, lme_items=lme_items
        )
        virgin_sha = virgin_suite_digest(virgin_convs)
        virgin_ans_runner = PromptAnswerRunner(
            endpoint, args.model, args.workers, work_dir / "virgin_answer_cache.json"
        )
        virgin_records = _prepare_virgin_retrieval_and_answers(
            virgin_convs=virgin_convs,
            frozen_cfg=frozen_cfg,
            flat_enc=flat_enc,
            bge_enc=bge_enc,
            reranker=reranker,
            answer_runner=virgin_ans_runner,
        )

        # Unload all retrieval & reranking models from RTX 3090 before loading Qwen2.5-7B-Instruct!
        flat_enc.release()
        bge_enc.release()
        reranker.clear_cache()
        del flat_enc, bge_enc, flat_model, bge_model, reranker
        gc.collect()
        torch.cuda.empty_cache()
        print("[GPU] Unloaded MiniLM, BGE-M3, and BGE-Reranker-v2-M3 cleanly from RTX 3090", flush=True)

        # PHASE 3B: Blind Cross-Family Semantic Judge (`Qwen/Qwen2.5-7B-Instruct` bf16 on RTX 3090)
        virgin_summary, virgin_csv_path, virgin_audit_path, qwen_judge_meta = _grade_virgin_blind_with_qwen25_7b(
            virgin_records=virgin_records,
            virgin_convs=virgin_convs,
            answer_runner=virgin_ans_runner,
            work_dir=work_dir,
            output_dir=args.output,
        )
        virgin_summary["started_after_dev_freeze_at_utc"] = phase3_started_utc
        virgin_summary["suite_sha256"] = virgin_sha
        virgin_summary["lineage_disjointness_verification"] = disjoint_check

        # Verify K2.3 strict quantile separation on primary confirmatory comparison
        prim_comp = virgin_summary["overall_n120_24clusters"]["comparisons"][
            f"{ARM_CHELATED_V4}_vs_{ARM_FLAT_BGEM3_TM}"
        ]
        quantile_sep_ok = bool(
            prim_comp["p0125_bonf2"] < prim_comp["lcb_975"] < prim_comp["lcb_95"]
        )

        import chelated_conversational_v4
        import ek5_v3_stratified
        import em4_soft_telemetry_routing
        import virgin_conversational_holdout
        import wave6_common

        src_files = [
            Path(__file__),
            Path(chelated_conversational_v4.__file__),
            Path(virgin_conversational_holdout.__file__),
            Path(wave6_common.__file__),
            Path(ek5_v3_stratified.__file__),
            Path(em4_soft_telemetry_routing.__file__),
        ]

        frozen_path = args.output / "frozen_dev_config.json"
        wall_sec = round(time.perf_counter() - t_start, 2)

        manifest = {
            "schema_version": "wave6_pm1b_v4_reranked_sota_k2_v1",
            "git_base_sha": current_git_sha(ROOT),
            "host": socket.gethostname(),
            "gpu_device": gpu_name,
            "wall_seconds": wall_sec,
            "k2_guardrail_compliance": {
                "ek5_echo_collapse_before_session_budgeting": True,
                "max_sessions_cap_dropped_on_ku": True,
                "chronological_supersede_rendering": True,
                "bge_reranker_v2_m3_top36_with_partner_preservation": True,
                "dev_only_compaction_frozen_before_holdout": (
                    frozen_cfg["frozen_at_utc"] < look2_bundle["post_j4_exploratory_look_2_burned_holdout"]["started_after_dev_freeze_at_utc"]
                    and frozen_cfg["frozen_at_utc"] < phase3_started_utc
                ),
                "distinct_bootstrap_quantiles_p125_lt_p25_lt_p50": quantile_sep_ok,
                "post_j4_look2_explicitly_labeled_alpha_0_025": True,
                "virgin_holdout_blind_cross_family_qwen25_7b_judged": True,
            },
            "frozen_dev_config_file": frozen_path.name,
            "frozen_dev_config_sha256": frozen_cfg_sha,
            "selected_dev_candidate_id": frozen_cfg["selected_candidate_id"],
            "selected_retrieval_config": frozen_cfg["selected_config"],
            "models": {
                "reader_llm": {
                    "model": args.model,
                    "family": "nvidia-nemotron-3.5-30b",
                    "endpoint_node": SPARK3_HOST,
                    "enable_thinking": False,
                    "temperature": 0.0,
                },
                "blind_confirmatory_judge": qwen_judge_meta,
                "embedders_and_reranker": {
                    BGE_MODEL: {"dimensions": BGE_DIMENSIONS, "revision": snaps[BGE_MODEL].name, "dtype": "float16"},
                    BGE_RERANKER_MODEL: {"revision": snaps[BGE_RERANKER_MODEL].name, "dtype": "float16", "top_pool": 36},
                    MINILM: {"dimensions": 384, "revision": snaps[MINILM].name, "dtype": "float32"},
                },
            },
            "datasets": {
                LME: {"filename": dataset_files[LME].name, "sha256": data_hashes[LME]},
                LOCOMO: {"filename": dataset_files[LOCOMO].name, "sha256": data_hashes[LOCOMO]},
                VIRGIN_SUITE_NAME: {"module": "virgin_conversational_holdout.py", "sha256": virgin_sha},
            },
            "source_sha256": {p.name: _sha256_file(p) for p in src_files},
            "primary_confirmatory_virgin_holdout": virgin_summary,
            **look2_bundle,
            "artifacts": {
                "frozen_dev_config": {"file": frozen_path.name, "sha256": frozen_cfg_sha},
                "look2_rows_csv": {"file": look2_csv_path.name, "sha256": _sha256_file(look2_csv_path)},
                "virgin_holdout_rows_csv": {"file": virgin_csv_path.name, "sha256": _sha256_file(virgin_csv_path)},
                "virgin_blind_judge_audit": {"file": virgin_audit_path.name, "sha256": _sha256_file(virgin_audit_path)},
            },
            "config_digest": sha256_json(frozen_cfg["selected_config"]),
        }

        manifest_path = args.output / "wave6_pm1b_v4_manifest.json"
        write_json_atomic(manifest_path, manifest)
        manifest_sha = _sha256_file(manifest_path)

        receipt = {
            "schema_version": "report_receipt_v4_0",
            "bundle_id": "wave6_pm1b_v4_reranked_sota",
            "git_sha": manifest["git_base_sha"],
            "manifest_file": manifest_path.name,
            "manifest_sha256": manifest_sha,
            "artifact_hashes": {
                manifest_path.name: manifest_sha,
                frozen_path.name: frozen_cfg_sha,
                look2_csv_path.name: _sha256_file(look2_csv_path),
                virgin_csv_path.name: _sha256_file(virgin_csv_path),
                virgin_audit_path.name: _sha256_file(virgin_audit_path),
            },
        }
        receipt_path = args.output / "report_receipt.json"
        write_json_atomic(receipt_path, receipt)

        print(
            f"\n[COMPLETE] Wave-6 PM-1b v4 finished in {wall_sec}s\n"
            f"  manifest: {manifest_path} ({manifest_sha})\n"
            f"  receipt:  {receipt_path} ({_sha256_file(receipt_path)})",
            flush=True,
        )

    finally:
        if tunnel_proc is not None and tunnel_proc.poll() is None:
            tunnel_proc.terminate()
            try:
                tunnel_proc.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                tunnel_proc.kill()
            print("[Cleanup] Terminated SSH tunnel cleanly.", flush=True)


if __name__ == "__main__":
    main()
