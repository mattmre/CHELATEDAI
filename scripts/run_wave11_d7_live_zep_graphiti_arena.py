"""Wave-11 Demo D7 live arena: Chelated v4 vs Graphiti/Zep 0.31.2 vs Mem0 2.2.1.

``--mode local-check`` validates the docker contract and imports. It does not
start Neo4j, vLLM, or a GPU job, and it does not write a live receipt.

``--mode spark1-live`` is the runtime the orchestrator runs on Spark #1
(``mattmre@10.0.0.49``). This file does not open an SSH session. Docker is
bound to ``127.0.0.1`` and the container name is the literal ``wave11-d7-neo4j``.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from chelated_graphiti_arena_v1 import (  # noqa: E402
    ARENA_ARMS,
    ARM_CHELATED,
    ARM_GRAPHITI,
    ARM_MEM0,
    BGE_M3_MODEL,
    BGE_RERANKER_MODEL,
    BOOTSTRAP_SEED,
    COMMUNITY_DATABASE,
    MAX_COROUTINES,
    N_BOOTSTRAP,
    VLLM_BASE_URL,
    VLLM_MODEL,
    MeteredVllmOpenAIGenericClient,
    apply_horizon_grades,
    call_nemotron_reader,
    case_group_id,
    format_active_facts,
    format_mem0_results,
    knowledge_update_retraction_text,
    partition_retraction_documents,
    reference_time_from_ts,
    run_delta0_then_settled,
    select_retraction_message,
    select_suite_b_items,
    split_stale_leakage,
    suite_a_horizon_inference,
    summarize_suite_b,
    synchronous_fence_horizons,
    verify_case_graph_populated,
)

CONTAINER_NAME = "wave11-d7-neo4j"
NEO4J_IMAGE = "neo4j:5.26-community"
NEO4J_HTTP_PORT = 17474
NEO4J_BOLT_PORT = 17687
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "chelated_bench_2026"
NEO4J_AUTH = f"{NEO4J_USER}/{NEO4J_PASSWORD}"
BOLT_URI = f"bolt://127.0.0.1:{NEO4J_BOLT_PORT}"
SPARK1_HOST = "mattmre@10.0.0.49"
UNTOUCHED_CONTAINER = "nemotron-37c3"
GRAPHITI_VERSION = "0.31.2"
MEM0_VERSION = "2.2.1"
JUDGE_HF_MODEL = "Qwen/Qwen2.5-7B-Instruct"
JUDGE_FAMILY = "alibaba-qwen2.5"
GENERATOR_FAMILY = "nvidia-nemotron"
SUITE_A_N = 60
SUITE_B_N = 20
SCHEMA_VERSION = "wave11_d7_live_zep_graphiti_v1"
DEFAULT_OUTPUT = REPO_ROOT / "artifacts" / "wave11_d7_live_zep_graphiti_arena"
DOCKER_RM_ARGV = ("docker", "rm", "-f", CONTAINER_NAME)

ENTERPRISE_READER_SYSTEM = (
    "You are an enterprise verification assistant. Answer the question concisely using ONLY the provided records.\n"
    "- Higher-authority official filings, signed advisories, and postmortems supersede preliminary notes and unverified rumors.\n"
    "- If equal-authority official sources mutually conflict with no authoritative resolution (CONTESTED), or if the "
    "verified answer is absent from the records, reply ONLY: ABSTAIN."
)

CONVERSATIONAL_READER_SYSTEM = (
    "Answer the user's question concisely (1-2 sentences) using ONLY the dated records below.\n"
    "- When records contain an earlier value and a newer dated update (or [LATEST CURRENT UPDATE]), report the latest current fact "
    "unless the question explicitly asks about the earlier state.\n"
    "- Verify exact entity and premise match: if the specific person, pet, item, place, or metric asked about was NEVER "
    "mentioned in the records, clearly refuse / reply ABSTAIN instead of guessing from a different item."
)


def docker_run_argv() -> List[str]:
    """Ephemeral Community Neo4j. Ports are bound to Spark's loopback only."""
    return [
        "docker",
        "run",
        "--detach",
        "--name",
        CONTAINER_NAME,
        "--memory=8g",
        "--publish",
        f"127.0.0.1:{NEO4J_HTTP_PORT}:7474",
        "--publish",
        f"127.0.0.1:{NEO4J_BOLT_PORT}:7687",
        "--env",
        f"NEO4J_AUTH={NEO4J_AUTH}",
        NEO4J_IMAGE,
    ]


def assert_docker_argv_safe(argv: Sequence[str]) -> None:
    if not argv or argv[0] != "docker":
        raise RuntimeError("refusing a non-docker argv")
    blob = " ".join(argv)
    lowered = blob.lower()
    if "nemotron" in lowered:
        raise RuntimeError(f"refusing an argv that names {UNTOUCHED_CONTAINER}")
    if "*" in blob or "?" in blob:
        raise RuntimeError("refusing a glob in a docker argv")
    if "--name" in argv:
        name = argv[argv.index("--name") + 1]
    elif len(argv) >= 2 and argv[1] == "rm":
        name = argv[-1]
    else:
        raise RuntimeError("docker argv has no container name")
    if name != CONTAINER_NAME:
        raise RuntimeError(f"refusing container {name!r}; the only allowed name is {CONTAINER_NAME}")


def _invoke_docker(argv: Sequence[str], run_cmd: Optional[Callable[..., Any]]) -> Any:
    assert_docker_argv_safe(argv)
    runner = run_cmd or subprocess.run
    return runner(list(argv), check=False)


def stop_wave11_neo4j(run_cmd: Optional[Callable[..., Any]] = None) -> Any:
    argv = list(DOCKER_RM_ARGV)
    if argv != ["docker", "rm", "-f", CONTAINER_NAME]:
        raise RuntimeError("refusing a docker rm that is not the literal wave11-d7-neo4j name")
    return _invoke_docker(argv, run_cmd)


def start_wave11_neo4j(run_cmd: Optional[Callable[..., Any]] = None) -> List[str]:
    """Remove a leftover of this exact container name, then start a new one."""
    stop_wave11_neo4j(run_cmd=run_cmd)
    argv = docker_run_argv()
    result = _invoke_docker(argv, run_cmd)
    code = getattr(result, "returncode", None)
    if code not in (None, 0):
        raise RuntimeError(f"docker run {CONTAINER_NAME} failed with status {code}")
    return argv


def wait_for_tcp(host: str, port: int, *, attempts: int = 60, pause_s: float = 2.0) -> None:
    import socket

    last: Optional[BaseException] = None
    for _ in range(attempts):
        try:
            with socket.create_connection((host, port), timeout=2.0):
                return
        except OSError as exc:
            last = exc
            time.sleep(pause_s)
    raise RuntimeError(f"TCP {host}:{port} did not accept a connection: {last}")


def _public_error(exc: BaseException) -> str:
    text = f"{type(exc).__name__}: {exc}".replace(NEO4J_PASSWORD, "[redacted]")
    return text[:500]


def rrf_combine(bm25_scores: np.ndarray, dense_scores: np.ndarray, k: int = 60) -> np.ndarray:
    n = int(bm25_scores.shape[0])
    if n == 0:
        return np.zeros(0, dtype=np.float64)
    out = np.zeros(n, dtype=np.float64)
    for arr in (bm25_scores, dense_scores):
        order = np.argsort(-arr, kind="mergesort")
        ranks = np.empty(n, dtype=np.float64)
        ranks[order] = np.arange(1, n + 1, dtype=np.float64)
        out += 1.0 / (float(k) + ranks)
    return out


def new_arm_slot() -> Dict[str, Any]:
    return {
        "pred": "",
        "pred_delta0s": "",
        "correct_settled": False,
        "correct_delta0s": False,
        "arm_failure": False,
        "error": None,
        "write_ack_ms": 0.0,
        "consolidation_bg_ms": 0.0,
        "write_prompt_tokens": 0,
        "write_completion_tokens": 0,
        "read_prompt_tokens": 0,
        "read_completion_tokens": 0,
        "stale_context_delta0s": False,
        "stale_answer_delta0s": False,
        "stale_context_settled": False,
        "stale_answer_settled": False,
        "graph_audit_delta0": None,
        "graph_audit_settled": None,
        "task_done_at_delta0": None,
        "task_done_after_settle": None,
        "synchronous_ek5_fence": False,
    }


def _mark_arm_failure(slot: Dict[str, Any], exc: BaseException) -> None:
    slot["arm_failure"] = True
    slot["error"] = _public_error(exc)
    slot["pred"] = ""
    slot["pred_delta0s"] = ""
    audit = {"status": "ARM_FAILURE", "reason": "exception", "n_cnt": None, "r_cnt": None}
    slot["graph_audit_delta0"] = dict(audit)
    slot["graph_audit_settled"] = dict(audit)


def _apply_stale(slot: Dict[str, Any], stale_values: Sequence[str], horizon: str, context: str, answer: str) -> None:
    flags = split_stale_leakage(context, answer, stale_values)
    if horizon == "delta0":
        slot["stale_context_delta0s"] = flags["stale_context"]
        slot["stale_answer_delta0s"] = flags["stale_answer"]
    else:
        slot["stale_context_settled"] = flags["stale_context"]
        slot["stale_answer_settled"] = flags["stale_answer"]


def _read_pair_per_horizon(
    endpoint: str,
    model: str,
    system: str,
    question: str,
    delta0_context: str,
    settled_context: str,
) -> Tuple[str, str, int, int, int, int]:
    user_delta0 = f"Records:\n{delta0_context}\n\nQuestion: {question}\nAnswer:"
    user_settled = f"Records:\n{settled_context}\n\nQuestion: {question}\nAnswer:"
    if delta0_context == settled_context:
        answer, prompt, completion, _wall = call_nemotron_reader(endpoint, model, system, user_delta0, max_tokens=128)
        return answer, answer, int(prompt), int(completion), int(prompt), int(completion)
    answer_d, prompt_d, completion_d, _wall_d = call_nemotron_reader(
        endpoint, model, system, user_delta0, max_tokens=128
    )
    answer_s, prompt_s, completion_s, _wall_s = call_nemotron_reader(
        endpoint, model, system, user_settled, max_tokens=128
    )
    return answer_d, answer_s, int(prompt_d), int(completion_d), int(prompt_s), int(completion_s)


def _read_pair(
    endpoint: str,
    model: str,
    system: str,
    question: str,
    delta0_context: str,
    settled_context: str,
) -> Tuple[str, str, int, int]:
    ans_d, ans_s, _pd, _cd, ps, cs = _read_pair_per_horizon(
        endpoint, model, system, question, delta0_context, settled_context
    )
    return ans_d, ans_s, ps, cs


def load_jsonl_rows(path: Path) -> List[Dict[str, Any]]:
    if not path.is_file():
        return []
    raw = path.read_bytes()
    if b"\r" in raw:
        raise RuntimeError(f"JSONL {path} contains CR; refusing to skip a corrupt checkpoint")
    rows: List[Dict[str, Any]] = []
    seen = set()
    for lineno, line in enumerate(raw.decode("utf-8").split("\n"), start=1):
        if line == "":
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"corrupt JSONL {path}:{lineno}") from exc
        if not isinstance(obj, dict) or "item_id" not in obj:
            raise RuntimeError(f"JSONL {path}:{lineno} is missing item_id")
        item_id = str(obj["item_id"])
        if item_id in seen:
            raise RuntimeError(f"duplicate item_id {item_id} in {path}")
        seen.add(item_id)
        rows.append(obj)
    return rows


def append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = {str(item["item_id"]) for item in load_jsonl_rows(path)} if path.is_file() else set()
    item_id = str(row["item_id"])
    if item_id in existing:
        raise RuntimeError(f"refusing to append duplicate item_id {item_id}")
    line = json.dumps(dict(row), ensure_ascii=True, sort_keys=True, separators=(",", ":")) + "\n"
    if "\r" in line:
        raise RuntimeError("refusing to write a CR into JSONL")
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(line)
        handle.flush()
        os.fsync(handle.fileno())


def rewrite_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    payload = "".join(
        json.dumps(dict(row), ensure_ascii=True, sort_keys=True, separators=(",", ":")) + "\n" for row in rows
    )
    if "\r" in payload:
        raise RuntimeError("refusing to write a CR into JSONL")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(payload, encoding="utf-8", newline="\n")
    os.replace(tmp, path)


def checkpoints_complete(rows_a: Sequence[Mapping[str, Any]], rows_b: Sequence[Mapping[str, Any]]) -> bool:
    if len(rows_a) != SUITE_A_N or len(rows_b) != SUITE_B_N:
        return False
    if len({str(row["item_id"]) for row in rows_a}) != SUITE_A_N:
        return False
    if len({str(row["item_id"]) for row in rows_b}) != SUITE_B_N:
        return False
    return all(bool(row.get("judge_complete")) for row in list(rows_a) + list(rows_b))


def build_manifest(
    rows_a: Sequence[Mapping[str, Any]],
    rows_b: Sequence[Mapping[str, Any]],
    *,
    output_dir: Path,
    probe: Optional[Mapping[str, Any]] = None,
    corpus_sha256: str = "",
) -> Dict[str, Any]:
    if len(rows_a) != SUITE_A_N:
        raise RuntimeError(f"Suite A primary inference requires N={SUITE_A_N}, got {len(rows_a)}")
    if len(rows_b) != SUITE_B_N:
        raise RuntimeError(f"Suite B requires N={SUITE_B_N}, got {len(rows_b)}")
    settled = suite_a_horizon_inference(rows_a, "settled", n_bootstrap=N_BOOTSTRAP, seed=BOOTSTRAP_SEED)
    delta0 = suite_a_horizon_inference(rows_a, "delta0", n_bootstrap=N_BOOTSTRAP, seed=BOOTSTRAP_SEED)
    suite_b = summarize_suite_b(rows_b)
    if suite_b.get("holm_bonferroni") is not None or suite_b.get("unpooled") is not True:
        raise RuntimeError("Suite B must stay unpooled and outside the Holm family")
    if suite_b.get("inferential_claim") is not False:
        raise RuntimeError("Suite B is descriptive-only")
    from scripts.night1_common import current_git_sha

    scratch = output_dir / "scratch"
    return {
        "schema_version": SCHEMA_VERSION,
        "git_sha": current_git_sha(REPO_ROOT),
        "mode": "spark1-live",
        "host": SPARK1_HOST,
        "container": CONTAINER_NAME,
        "untouched_container": UNTOUCHED_CONTAINER,
        "bolt_uri": BOLT_URI,
        "http_publish": f"127.0.0.1:{NEO4J_HTTP_PORT}",
        "neo4j_image": NEO4J_IMAGE,
        "neo4j_database": COMMUNITY_DATABASE,
        "docker_run_argv": docker_run_argv(),
        "docker_rm_argv": list(DOCKER_RM_ARGV),
        "graphiti_core": GRAPHITI_VERSION,
        "mem0ai": MEM0_VERSION,
        "reader_model": VLLM_MODEL,
        "reader_endpoint": VLLM_BASE_URL,
        "judge_hf_model": JUDGE_HF_MODEL,
        "judge_family": JUDGE_FAMILY,
        "generator_family": GENERATOR_FAMILY,
        "embedder": BGE_M3_MODEL,
        "reranker": BGE_RERANKER_MODEL,
        "max_coroutines": MAX_COROUTINES,
        "search_recipe": "EDGE_HYBRID_SEARCH_CROSS_ENCODER",
        "active_edge_filter": "invalid_at IS NULL",
        "suite_a_n": SUITE_A_N,
        "suite_b_n": SUITE_B_N,
        "suite_b_pooled": False,
        "suite_b_episode_granularity": "per_session_graphiti_8k_mem0_16k_standalone_retraction",
        "horizons": ["delta0", "settled"],
        "primary_score": "settled_accuracy",
        "n_bootstrap": N_BOOTSTRAP,
        "seed": BOOTSTRAP_SEED,
        "corpus_sha256": corpus_sha256,
        "probe": dict(probe or {}),
        "suite_a_jsonl": str(output_dir / "suite_a.jsonl"),
        "suite_b_jsonl": str(output_dir / "suite_b.jsonl"),
        "suite_a_inference": {"delta0": delta0, "settled": settled},
        "suite_b_descriptive": suite_b,
        "scratch_dir": str(scratch),
        "scratch_is_removable": True,
        "local_check_is_not_evidence": True,
        "evidence_tier": "spark1-live",
    }


def emit_wave11_artifacts(output_dir: Path, manifest: Mapping[str, Any]) -> Dict[str, str]:
    from scripts.night1_common import write_json_atomic

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "wave11_d7_zep_graphiti_manifest.json"
    digest = write_json_atomic(manifest_path, dict(manifest))
    receipt = {
        "schema_version": "wave11_d7_live_zep_graphiti_receipt_v1",
        "manifest_path": str(manifest_path),
        "manifest_sha256": digest,
        "suite_a_n": SUITE_A_N,
        "suite_b_n": SUITE_B_N,
        "horizons": ["delta0", "settled"],
        "primary_score": "settled_accuracy",
        "suite_b_pooled": False,
        "local_check_is_not_evidence": True,
        "evidence_tier": "spark1-live",
    }
    receipt_path = output_dir / "report_receipt.json"
    write_json_atomic(receipt_path, receipt)
    return {
        "manifest_path": str(manifest_path),
        "receipt_path": str(receipt_path),
        "manifest_sha256": digest,
    }


def installed_graphiti_version() -> str:
    from importlib.metadata import version

    return version("graphiti-core")


def run_local_check() -> Dict[str, Any]:
    """Import and contract check. Does not call Docker, SSH, Neo4j, or vLLM."""
    argv = docker_run_argv()
    assert_docker_argv_safe(argv)
    assert_docker_argv_safe(list(DOCKER_RM_ARGV))
    found = installed_graphiti_version()
    if found != GRAPHITI_VERSION:
        raise RuntimeError(f"graphiti-core {found} != required {GRAPHITI_VERSION}")
    required = (
        MeteredVllmOpenAIGenericClient,
        verify_case_graph_populated,
        format_active_facts,
        run_delta0_then_settled,
        suite_a_horizon_inference,
    )
    if any(item is None for item in required):
        raise RuntimeError("arena symbols failed to import")
    if MAX_COROUTINES != 6 or N_BOOTSTRAP != 20000 or BOOTSTRAP_SEED != 42:
        raise RuntimeError("live statistical or concurrency constants drifted")
    if ARENA_ARMS != (ARM_CHELATED, ARM_GRAPHITI, ARM_MEM0):
        raise RuntimeError(f"arena arms drifted: {ARENA_ARMS!r}")
    return {
        "mode": "local-check",
        "live_executed": False,
        "docker_executed": False,
        "live_receipt_written": False,
        "graphiti_core": found,
        "max_coroutines": MAX_COROUTINES,
        "n_bootstrap": N_BOOTSTRAP,
        "docker_run_argv": argv,
        "docker_rm_argv": list(DOCKER_RM_ARGV),
        "host_recorded_for_live_mode": SPARK1_HOST,
        "note": "local-check did not run Neo4j, vLLM, Mem0, or the judge. It is not a live receipt.",
    }


def _require_live_identity(args: Any) -> None:
    endpoint = str(getattr(args, "vllm_endpoint", VLLM_BASE_URL)).rstrip("/")
    model = str(getattr(args, "vllm_model", VLLM_MODEL))
    judge = str(getattr(args, "judge_hf_model", JUDGE_HF_MODEL))
    if endpoint != VLLM_BASE_URL:
        raise ValueError(f"reader endpoint must be {VLLM_BASE_URL}, got {endpoint!r}")
    if model != VLLM_MODEL:
        raise ValueError(f"reader model must be {VLLM_MODEL}, got {model!r}")
    if judge != JUDGE_HF_MODEL:
        raise ValueError(f"judge must be {JUDGE_HF_MODEL}, got {judge!r}")


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def meter_mem0_writes(memory: Any) -> Dict[str, int]:
    """Count Mem0 chat-completion usage. Missing usage fails the add."""
    box = {"write_prompt_tokens": 0, "write_completion_tokens": 0, "llm_call_count": 0}
    completions = memory.llm.client.chat.completions
    original = completions.create

    def wrapped(*args: Any, **kwargs: Any) -> Any:
        response = original(*args, **kwargs)
        usage = getattr(response, "usage", None)
        prompt = getattr(usage, "prompt_tokens", None) if usage is not None else None
        completion = getattr(usage, "completion_tokens", None) if usage is not None else None
        if prompt is None or completion is None:
            raise RuntimeError("mem0 vLLM response.usage is missing prompt_tokens or completion_tokens")
        box["write_prompt_tokens"] += int(prompt)
        box["write_completion_tokens"] += int(completion)
        box["llm_call_count"] += 1
        return response

    completions.create = wrapped
    return box


async def _add_graphiti_text(graphiti: Any, *, name: str, body: str, source_description: str, ts: float, gid: str) -> Any:
    from graphiti_core.nodes import EpisodeType

    return await graphiti.add_episode(
        name=name[:180],
        episode_body=body,
        source_description=source_description[:180],
        reference_time=reference_time_from_ts(ts),
        source=EpisodeType.text,
        group_id=gid,
    )


async def _chelated_suite_a_slot(
    embedder: Any,
    case: Mapping[str, Any],
    docs: Sequence[Any],
    endpoint: str,
    model: str,
) -> Dict[str, Any]:
    from chelated_epistemic_proposer_v2 import EXTRACTION_SYSTEM_PROMPT_V2, AuthorityDominatedExocortexV2
    from chelated_raw_text_exocortex import parse_extracted_claim_json
    from scripts.night1_common import FastBM25Index
    from spherical_migration_bridge import l2_normalize

    slot = new_arm_slot()
    slot["synchronous_ek5_fence"] = True
    ordered = sorted(docs, key=lambda doc: (int(doc.ingested_ts), int(doc.published_ts), str(doc.doc_id)))
    write_prompt = 0
    write_completion = 0
    started = time.perf_counter()
    claims = []
    for doc in ordered:
        user = f"Source channel: {doc.source_channel}\nPublished day: {doc.published_ts}\nDocument:\n{doc.text}"
        _text, prompt, completion, _wall = call_nemotron_reader(
            endpoint,
            model,
            EXTRACTION_SYSTEM_PROMPT_V2,
            user,
            max_tokens=256,
        )
        write_prompt += int(prompt)
        write_completion += int(completion)
        claims.append(parse_extracted_claim_json(doc.doc_id, _text))
    extract_ms = (time.perf_counter() - started) * 1000.0
    ack_started = time.perf_counter()
    doc_vecs = np.asarray(embedder.encode_texts([doc.text for doc in ordered]), dtype=np.float32)
    slot_vecs = np.asarray(embedder.encode_texts([f"{claim.entity} {claim.slot}" for claim in claims]), dtype=np.float32)
    exo = AuthorityDominatedExocortexV2(ordered, claims, doc_vecs, slot_vecs)
    bm25 = FastBM25Index([doc.doc_id for doc in ordered], [doc.text for doc in ordered])
    ack_ms = (time.perf_counter() - ack_started) * 1000.0
    read_started = time.perf_counter()
    query = str(case["query_text"])
    query_vec = np.asarray(embedder.encode_texts([query])[0], dtype=np.float32)
    hybrid = rrf_combine(bm25.score_query(query), l2_normalize(doc_vecs) @ query_vec)
    selected = exo.select_context_for_query(hybrid, query, top_k=5)
    read_ms = (time.perf_counter() - read_started) * 1000.0
    context = str(selected["formatted_context"])
    read_prompt = 0
    read_completion = 0
    if selected["status"] == "CONTESTED_ABSTAIN":
        answer = "ABSTAIN"
    else:
        answer, read_prompt, read_completion, _wall = call_nemotron_reader(
            endpoint,
            model,
            ENTERPRISE_READER_SYSTEM,
            f"Records:\n{context}\n\nQuestion: {query}\nAnswer:",
            max_tokens=128,
        )
    clocks = synchronous_fence_horizons(context, write_ack_ms=ack_ms, consolidation_bg_ms=extract_ms, read_ms=read_ms)
    slot["pred"] = answer
    slot["pred_delta0s"] = answer
    slot["write_ack_ms"] = clocks["write_ack_ms"]
    slot["consolidation_bg_ms"] = clocks["consolidation_bg_ms"]
    slot["task_done_at_delta0"] = True
    slot["task_done_after_settle"] = True
    slot["write_prompt_tokens"] = write_prompt
    slot["write_completion_tokens"] = write_completion
    slot["read_prompt_tokens"] = int(read_prompt)
    slot["read_completion_tokens"] = int(read_completion)
    stale = list(case.get("stale_or_false_values") or [])
    _apply_stale(slot, stale, "delta0", context, answer)
    _apply_stale(slot, stale, "settled", context, answer)
    return slot


async def _graphiti_documents_slot(
    graphiti: Any,
    driver: Any,
    llm: MeteredVllmOpenAIGenericClient,
    *,
    gid: str,
    query: str,
    stale_values: Sequence[str],
    prior: Sequence[Any],
    retraction: Any,
    body_of,
    endpoint: str,
    model: str,
    reader_system: str,
) -> Dict[str, Any]:
    slot = new_arm_slot()
    llm.reset_counters(phase="write")
    try:
        for doc in prior:
            name, body, source, ts = body_of(doc)
            await _add_graphiti_text(graphiti, name=name, body=body, source_description=source, ts=ts, gid=gid)
        name, body, source, ts = body_of(retraction)

        async def start_retraction() -> Any:
            return await _add_graphiti_text(graphiti, name=name, body=body, source_description=source, ts=ts, gid=gid)

        calls = {"n": 0}

        async def search_once() -> Dict[str, Any]:
            calls["n"] += 1
            context = await format_active_facts(graphiti, query, gid, num_results=8)
            audit = await verify_case_graph_populated(driver, gid, settled_gate=calls["n"] >= 2)
            return {"context": context, "audit": audit}

        raced = await run_delta0_then_settled(start_retraction, search_once)
    except Exception as exc:
        _mark_arm_failure(slot, exc)
        snap = llm.token_snapshot()
        slot["write_prompt_tokens"] = int(snap["write_prompt_tokens"])
        slot["write_completion_tokens"] = int(snap["write_completion_tokens"])
        return slot
    snap = llm.token_snapshot()
    delta0_ctx = str(raced["delta0"]["context"])
    settled_ctx = str(raced["settled"]["context"])
    answer_d, answer_s, pd0, cd0, ps, cs = _read_pair_per_horizon(
        endpoint, model, reader_system, query, delta0_ctx, settled_ctx
    )
    slot["pred_delta0s"] = answer_d
    slot["pred"] = answer_s
    slot["write_prompt_tokens"] = int(snap["write_prompt_tokens"])
    slot["write_completion_tokens"] = int(snap["write_completion_tokens"])
    slot["read_prompt_tokens"] = ps
    slot["read_completion_tokens"] = cs
    slot["read_prompt_tokens_delta0s"] = pd0
    slot["read_completion_tokens_delta0s"] = cd0
    slot["read_prompt_tokens_settled"] = ps
    slot["read_completion_tokens_settled"] = cs
    slot["write_ack_ms"] = float(raced["write_ack_ms"])
    slot["consolidation_bg_ms"] = float(raced["consolidation_bg_ms"])
    slot["task_done_at_delta0"] = bool(raced["task_done_at_delta0"])
    slot["task_done_after_settle"] = bool(raced["task_done_after_settle"])
    slot["graph_audit_delta0"] = raced["delta0"]["audit"]
    slot["graph_audit_settled"] = raced["settled"]["audit"]
    if raced["settled"]["audit"].get("status") == "ARM_FAILURE":
        slot["arm_failure"] = True
    _apply_stale(slot, stale_values, "delta0", delta0_ctx, answer_d)
    _apply_stale(slot, stale_values, "settled", settled_ctx, answer_s)
    return slot


async def _mem0_texts_slot(
    *,
    gid: str,
    query: str,
    stale_values: Sequence[str],
    prior_texts: Sequence[str],
    retraction_text: str,
    scratch_dir: Path,
    endpoint: str,
    model: str,
    reader_system: str,
) -> Dict[str, Any]:
    from scripts.run_wave6_pm3_v2_4way_arena import close_live_mem0, make_live_mem0_gpu

    slot = new_arm_slot()
    scratch_dir.mkdir(parents=True, exist_ok=True)
    qpath = Path(tempfile.mkdtemp(prefix=f"mem0_{gid}_", dir=str(scratch_dir)))
    memory = None
    usage = {"write_prompt_tokens": 0, "write_completion_tokens": 0, "llm_call_count": 0}
    try:
        memory, _tbox = make_live_mem0_gpu(endpoint, model, qpath, infer_prompt=True)
        usage = meter_mem0_writes(memory)
        for text in prior_texts:
            await asyncio.to_thread(memory.add, [{"role": "user", "content": text}], user_id=gid, infer=True)

        async def start_retraction() -> Any:
            return await asyncio.to_thread(
                memory.add,
                [{"role": "user", "content": retraction_text}],
                user_id=gid,
                infer=True,
            )

        async def search_once() -> str:
            found = await asyncio.to_thread(memory.search, query, top_k=8, filters={"user_id": gid})
            return format_mem0_results(found)

        raced = await run_delta0_then_settled(start_retraction, search_once)
    except Exception as exc:
        _mark_arm_failure(slot, exc)
        slot["write_prompt_tokens"] = int(usage["write_prompt_tokens"])
        slot["write_completion_tokens"] = int(usage["write_completion_tokens"])
        return slot
    finally:
        if memory is not None:
            close_live_mem0(memory, qpath)
    delta0_ctx = str(raced["delta0"])
    settled_ctx = str(raced["settled"])
    answer_d, answer_s, pd0, cd0, ps, cs = _read_pair_per_horizon(
        endpoint, model, reader_system, query, delta0_ctx, settled_ctx
    )
    slot["pred_delta0s"] = answer_d
    slot["pred"] = answer_s
    slot["write_prompt_tokens"] = int(usage["write_prompt_tokens"])
    slot["write_completion_tokens"] = int(usage["write_completion_tokens"])
    slot["read_prompt_tokens"] = ps
    slot["read_completion_tokens"] = cs
    slot["read_prompt_tokens_delta0s"] = pd0
    slot["read_completion_tokens_delta0s"] = cd0
    slot["read_prompt_tokens_settled"] = ps
    slot["read_completion_tokens_settled"] = cs
    slot["write_ack_ms"] = float(raced["write_ack_ms"])
    slot["consolidation_bg_ms"] = float(raced["consolidation_bg_ms"])
    slot["task_done_at_delta0"] = bool(raced["task_done_at_delta0"])
    slot["task_done_after_settle"] = bool(raced["task_done_after_settle"])
    _apply_stale(slot, stale_values, "delta0", delta0_ctx, answer_d)
    _apply_stale(slot, stale_values, "settled", settled_ctx, answer_s)
    return slot


def _doc_episode_parts(doc: Any) -> Tuple[str, str, str, float]:
    return (str(doc.doc_id), str(doc.text), str(doc.source_channel), float(doc.published_ts))


def _message_episode_parts(message: Mapping[str, Any]) -> Tuple[str, str, str, float]:
    from datetime import datetime, timezone

    ts = float(message["ts"])
    stamp = datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y/%m/%d %H:%M")
    body = f"[{stamp}] {message['role']}: {message['content']}"
    return (f"{message['session_id']}:{stamp}", body, "longmemeval", ts)


def group_prior_messages_into_session_episodes(
    prior: Sequence[Mapping[str, Any]],
    *,
    max_chars: int = 8000,
) -> List[Dict[str, Any]]:
    """Group ordered conversational turns by contiguous ``session_id`` (capped at ``max_chars``).

    Preserves 100% of turns in chronological order and exact per-session ``reference_time``
    without exploding a 50-session haystack into ~500 isolated single-turn episodes.
    """
    if max_chars < 256:
        raise ValueError("max_chars must be >= 256")
    episodes: List[Dict[str, Any]] = []
    cur_sid: Optional[str] = None
    cur_ts: float = 0.0
    cur_lines: List[str] = []
    cur_chars = 0
    part_idx = 0

    def _flush() -> None:
        nonlocal cur_lines, cur_chars, part_idx
        if not cur_lines or cur_sid is None:
            return
        from datetime import datetime, timezone

        stamp = datetime.fromtimestamp(cur_ts, tz=timezone.utc).strftime("%Y/%m/%d %H:%M")
        episodes.append(
            {
                "name": f"{cur_sid}:p{part_idx}:{stamp}",
                "body": "\n".join(cur_lines),
                "source": "longmemeval",
                "ts": cur_ts,
                "session_id": cur_sid,
            }
        )
        cur_lines = []
        cur_chars = 0
        part_idx += 1

    for message in prior:
        sid = str(message["session_id"])
        _name, line, _src, ts = _message_episode_parts(message)
        if cur_sid is not None and sid != cur_sid:
            _flush()
            cur_sid = None
            part_idx = 0
        if cur_sid is None:
            cur_sid = sid
            cur_ts = ts
        elif cur_lines and cur_chars + len(line) + 1 > max_chars:
            _flush()
            cur_ts = ts
        cur_lines.append(line)
        cur_chars += len(line) + (1 if len(cur_lines) > 1 else 0)
    _flush()
    return episodes


def chunk_prior_message_lines(lines: Sequence[str], *, max_chars: int = 16000) -> List[str]:
    """Pack dated conversational lines into chronological chunks without dropping or splitting turns."""
    if max_chars < 256:
        raise ValueError("max_chars must be >= 256")
    chunks: List[str] = []
    cur: List[str] = []
    cur_len = 0
    for line in lines:
        if cur and cur_len + len(line) + 1 > max_chars:
            chunks.append("\n".join(cur))
            cur = []
            cur_len = 0
        cur.append(line)
        cur_len += len(line) + (1 if len(cur) > 1 else 0)
    if cur:
        chunks.append("\n".join(cur))
    return chunks


def _session_episode_parts(ep: Mapping[str, Any]) -> Tuple[str, str, str, float]:
    return (str(ep["name"]), str(ep["body"]), str(ep["source"]), float(ep["ts"]))


async def run_suite_a_case(
    stack: Mapping[str, Any],
    case: Mapping[str, Any],
    documents: Mapping[str, Any],
    *,
    index: int,
    endpoint: str,
    model: str,
    scratch_dir: Path,
) -> Dict[str, Any]:
    gid = case_group_id(index)
    docs = [documents[str(doc_id)] for doc_id in case["doc_ids"]]
    ordered = sorted(docs, key=lambda doc: (int(doc.ingested_ts), int(doc.published_ts), str(doc.doc_id)))
    lines = [f"[source={doc.source_channel}, published_day={doc.published_ts}] {doc.text}" for doc in ordered]
    prior, retraction = partition_retraction_documents(docs)
    stale = list(case.get("stale_or_false_values") or [])
    query = str(case["query_text"])
    arms: Dict[str, Any] = {}
    try:
        arms[ARM_CHELATED] = await _chelated_suite_a_slot(stack["embedder"], case, docs, endpoint, model)
    except Exception as exc:
        arms[ARM_CHELATED] = new_arm_slot()
        _mark_arm_failure(arms[ARM_CHELATED], exc)
    arms[ARM_GRAPHITI] = await _graphiti_documents_slot(
        stack["graphiti"],
        stack["driver"],
        stack["llm"],
        gid=gid,
        query=query,
        stale_values=stale,
        prior=prior,
        retraction=retraction,
        body_of=_doc_episode_parts,
        endpoint=endpoint,
        model=model,
        reader_system=ENTERPRISE_READER_SYSTEM,
    )
    arms[ARM_MEM0] = await _mem0_texts_slot(
        gid=gid,
        query=query,
        stale_values=stale,
        prior_texts=[str(doc.text) for doc in prior],
        retraction_text=str(retraction.text),
        scratch_dir=scratch_dir,
        endpoint=endpoint,
        model=model,
        reader_system=ENTERPRISE_READER_SYSTEM,
    )
    return {
        "item_id": str(case["case_id"]),
        "group_id": gid,
        "regime": str(case["regime"]),
        "question": query,
        "gold": str(case["gold"]),
        "abstention_required": str(case["gold"]).strip().upper() == "ABSTAIN",
        "stale_values": stale,
        "input_sha256": _sha256_text("\n".join(lines)),
        "judge_complete": False,
        "arms": arms,
    }


async def run_suite_b_case(
    stack: Mapping[str, Any],
    item: Mapping[str, Any],
    *,
    index: int,
    endpoint: str,
    model: str,
    scratch_dir: Path,
) -> Dict[str, Any]:
    from scripts.run_r6_pm3_and_d2_qdrant_csls_backfill import raw_conversational_messages
    from scripts.run_wave6_pm3_v2_4way_arena import chelated_conversational_gpu_retrieve

    gid = case_group_id(index, prefix="d7b_ku")
    messages = raw_conversational_messages(item, inject_echoes=True)
    retraction_text = knowledge_update_retraction_text(item)
    retraction = select_retraction_message(messages, retraction_text)
    prior = [message for message in messages if message is not retraction]
    prior.sort(key=lambda message: (float(message["ts"]), str(message["session_id"])))
    query = str(item["question"])
    gold = str(item.get("answer") or "")
    arms: Dict[str, Any] = {}
    slot = new_arm_slot()
    slot["synchronous_ek5_fence"] = True
    try:
        def _encode(texts: Sequence[str]) -> np.ndarray:
            return np.asarray(stack["embedder"].encode_texts(list(texts)), dtype=np.float32)

        def _rerank(pairs: Sequence[Sequence[str]]) -> np.ndarray:
            return np.asarray(stack["reranker"].predict_pairs(pairs), dtype=np.float64)

        context, write_ack_ms, read_ms, status = chelated_conversational_gpu_retrieve(messages, query, _encode, _rerank)
        answer, prompt, completion, _wall = call_nemotron_reader(
            endpoint,
            model,
            CONVERSATIONAL_READER_SYSTEM,
            f"Records:\n{context}\n\nQuestion: {query}\nAnswer:",
            max_tokens=128,
        )
        clocks = synchronous_fence_horizons(
            str(context),
            write_ack_ms=float(write_ack_ms),
            consolidation_bg_ms=0.0,
            read_ms=float(read_ms),
        )
        slot["pred"] = answer
        slot["pred_delta0s"] = answer
        slot["status"] = status
        slot["write_ack_ms"] = clocks["write_ack_ms"]
        slot["consolidation_bg_ms"] = clocks["consolidation_bg_ms"]
        slot["task_done_at_delta0"] = True
        slot["task_done_after_settle"] = True
        slot["read_prompt_tokens"] = int(prompt)
        slot["read_completion_tokens"] = int(completion)
    except Exception as exc:
        _mark_arm_failure(slot, exc)
    arms[ARM_CHELATED] = slot
    prior_session_episodes = group_prior_messages_into_session_episodes(prior, max_chars=8000)
    ret_name, ret_body, ret_src, ret_ts = _message_episode_parts(retraction)
    retraction_episode = {"name": ret_name, "body": ret_body, "source": ret_src, "ts": ret_ts, "session_id": str(retraction["session_id"])}
    arms[ARM_GRAPHITI] = await _graphiti_documents_slot(
        stack["graphiti"],
        stack["driver"],
        stack["llm"],
        gid=gid,
        query=query,
        stale_values=[],
        prior=prior_session_episodes,
        retraction=retraction_episode,
        body_of=_session_episode_parts,
        endpoint=endpoint,
        model=model,
        reader_system=CONVERSATIONAL_READER_SYSTEM,
    )
    retraction_body = ret_body
    prior_mem0_chunks = chunk_prior_message_lines([_message_episode_parts(message)[1] for message in prior], max_chars=16000)
    arms[ARM_MEM0] = await _mem0_texts_slot(
        gid=gid,
        query=query,
        stale_values=[],
        prior_texts=prior_mem0_chunks,
        retraction_text=retraction_body,
        scratch_dir=scratch_dir,
        endpoint=endpoint,
        model=model,
        reader_system=CONVERSATIONAL_READER_SYSTEM,
    )
    joined = "\n".join(_message_episode_parts(message)[1] for message in messages)
    return {
        "item_id": str(item["question_id"]),
        "group_id": gid,
        "regime": "knowledge-update",
        "question": query,
        "gold": gold,
        "abstention_required": False,
        "stale_values": [],
        "input_sha256": _sha256_text(joined),
        "judge_complete": False,
        "arms": arms,
    }


def _load_judge(judge_model: str) -> Any:
    import torch
    import transformers as _tf
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from wave6_common import BlindCrossFamilySemanticJudge

    if not torch.cuda.is_available():
        raise RuntimeError("spark1-live judge requires CUDA")
    device = "cuda"
    tokenizer = AutoTokenizer.from_pretrained(judge_model, padding_side="left")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    _dtype_kw = {"dtype": dtype} if int(str(_tf.__version__).split(".", 1)[0]) >= 5 else {"torch_dtype": dtype}
    model = AutoModelForCausalLM.from_pretrained(judge_model, **_dtype_kw).to(device)
    model.eval()

    def complete(system: str, user: str) -> str:
        prompt = tokenizer.apply_chat_template(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            tokenize=False,
            add_generation_prompt=True,
        )
        encoded = tokenizer([prompt], return_tensors="pt").to(device)
        with torch.inference_mode():
            output = model.generate(
                **encoded,
                max_new_tokens=20,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        return tokenizer.decode(output[0, encoded.input_ids.shape[1] :], skip_special_tokens=True).strip()

    return BlindCrossFamilySemanticJudge(
        judge_family=JUDGE_FAMILY,
        generator_family=GENERATOR_FAMILY,
        completion_fn=complete,
    )


def grade_rows(rows: Sequence[Dict[str, Any]], *, suite_tag: str, judge: Any) -> List[Dict[str, Any]]:
    graded: List[Dict[str, Any]] = []
    for row in rows:
        if row.get("judge_complete"):
            graded.append(row)
            continue
        settled = judge.grade_arms_blind(
            item_id=f"{suite_tag}_settled:{row['item_id']}",
            question=row["question"],
            gold_reference=row["gold"],
            arm_predictions={arm: row["arms"][arm]["pred"] for arm in ARENA_ARMS},
            abstention_required=bool(row["abstention_required"]),
        )
        delta0 = judge.grade_arms_blind(
            item_id=f"{suite_tag}_delta0:{row['item_id']}",
            question=row["question"],
            gold_reference=row["gold"],
            arm_predictions={arm: row["arms"][arm]["pred_delta0s"] for arm in ARENA_ARMS},
            abstention_required=bool(row["abstention_required"]),
        )
        updated = apply_horizon_grades(dict(row), settled, delta0)
        updated["arms"] = row["arms"]
        updated["judge_complete"] = True
        graded.append(updated)
    return graded


async def _boot_live_stack(args: Any) -> Dict[str, Any]:
    import torch
    from openai import AsyncOpenAI

    from graphiti_core import Graphiti
    from graphiti_core.llm_client.config import LLMConfig
    from importlib.metadata import version

    from chelated_graphiti_arena_v1 import (
        LocalGpuBgeM3Embedder,
        LocalGpuBgeReranker,
        SingleDbCommunityNeo4jDriver,
        pin_graphiti_request_scope,
    )

    if version("graphiti-core") != GRAPHITI_VERSION:
        raise RuntimeError(f"graphiti-core {version('graphiti-core')} != {GRAPHITI_VERSION}")
    if version("mem0ai") != MEM0_VERSION:
        raise RuntimeError(f"mem0ai {version('mem0ai')} != {MEM0_VERSION}")
    if not torch.cuda.is_available():
        raise RuntimeError("spark1-live requires CUDA for BGE-M3, the reranker, and Mem0")
    _require_live_identity(args)
    wait_for_tcp("127.0.0.1", NEO4J_BOLT_PORT)
    driver = SingleDbCommunityNeo4jDriver(BOLT_URI, NEO4J_USER, NEO4J_PASSWORD, database=COMMUNITY_DATABASE)
    init_task = getattr(driver, "_init_task", None)
    if init_task is not None and not init_task.done():
        init_task.cancel()
        try:
            await init_task
        except asyncio.CancelledError:
            pass
    await driver.build_indices_and_constraints()
    ping = await driver.execute_query("RETURN 1 AS ok", params={})
    if ping is None:
        raise RuntimeError("Neo4j ping returned None")
    llm = MeteredVllmOpenAIGenericClient(
        config=LLMConfig(api_key="local-vllm", model=VLLM_MODEL, base_url=VLLM_BASE_URL, temperature=0.0),
        client=AsyncOpenAI(api_key="local-vllm", base_url=VLLM_BASE_URL),
    )
    probe_mode = await llm.probe_json_schema()
    probe = llm.token_snapshot()
    probe["structured_output_mode"] = probe_mode
    llm.reset_counters(phase="write")
    embedder = LocalGpuBgeM3Embedder()
    reranker = LocalGpuBgeReranker()
    graphiti = Graphiti(
        uri=BOLT_URI,
        user=NEO4J_USER,
        password=NEO4J_PASSWORD,
        llm_client=llm,
        embedder=embedder,
        cross_encoder=reranker,
        graph_driver=driver,
        max_coroutines=MAX_COROUTINES,
        store_raw_episode_content=True,
    )
    pin_graphiti_request_scope(graphiti)
    return {
        "driver": driver,
        "llm": llm,
        "embedder": embedder,
        "reranker": reranker,
        "graphiti": graphiti,
        "probe": probe,
    }


async def _close_stack(stack: Mapping[str, Any]) -> None:
    graphiti = stack.get("graphiti")
    if graphiti is not None and hasattr(graphiti, "close"):
        await graphiti.close()
        return
    driver = stack.get("driver")
    if driver is not None and hasattr(driver, "close"):
        await driver.close()


async def execute_spark1_live(
    args: Any,
    *,
    start_container: Optional[Callable[..., Any]] = None,
    stop_container: Optional[Callable[..., Any]] = None,
) -> Dict[str, Any]:
    """Run or resume the live arena. Docker starts only when a case is still missing."""
    _require_live_identity(args)
    osv = Path(str(args.osv_cache))
    lme = Path(str(args.lme_json))
    output_dir = Path(str(getattr(args, "output_dir", DEFAULT_OUTPUT)))
    if not osv.is_file():
        raise FileNotFoundError(osv)
    if not lme.is_file():
        raise FileNotFoundError(lme)
    suite_a_path = output_dir / "suite_a.jsonl"
    suite_b_path = output_dir / "suite_b.jsonl"
    rows_a = load_jsonl_rows(suite_a_path)
    rows_b = load_jsonl_rows(suite_b_path)
    if checkpoints_complete(rows_a, rows_b):
        manifest = build_manifest(rows_a, rows_b, output_dir=output_dir, probe={"resumed_complete_checkpoint": True})
        emit_wave11_artifacts(output_dir, manifest)
        return manifest

    from prospective_realworld_drift_corpus import build_prospective_realworld_drift_suite

    cases, documents, corpus_sha = build_prospective_realworld_drift_suite(osv)
    if len(cases) != SUITE_A_N:
        raise RuntimeError(f"Suite A corpus produced {len(cases)} cases, expected {SUITE_A_N}")
    docs_by_id = {str(doc.doc_id): doc for doc in documents}
    suite_b_items = select_suite_b_items(json.loads(lme.read_text(encoding="utf-8")), n=SUITE_B_N)
    done_a = {str(row["item_id"]) for row in rows_a}
    done_b = {str(row["item_id"]) for row in rows_b}
    missing_a = [idx for idx, case in enumerate(cases) if str(case["case_id"]) not in done_a]
    missing_b = [idx for idx, item in enumerate(suite_b_items) if str(item["question_id"]) not in done_b]
    need_judge = any(not row.get("judge_complete") for row in list(rows_a) + list(rows_b)) or bool(missing_a or missing_b)
    starter = start_container or start_wave11_neo4j
    stopper = stop_container or stop_wave11_neo4j
    container_started = False
    stack: Optional[Dict[str, Any]] = None
    probe: Dict[str, Any] = {}
    try:
        if missing_a or missing_b:
            container_started = True
            starter()
            stack = await _boot_live_stack(args)
            probe = dict(stack["probe"])
            scratch = output_dir / "scratch"
            for idx in missing_a:
                case = cases[idx]
                row = await run_suite_a_case(
                    stack,
                    case,
                    docs_by_id,
                    index=idx,
                    endpoint=VLLM_BASE_URL,
                    model=VLLM_MODEL,
                    scratch_dir=scratch,
                )
                append_jsonl(suite_a_path, row)
                rows_a.append(row)
                print(f"[wave11-d7] suite A {len(rows_a)}/{SUITE_A_N} {row['item_id']}", flush=True)
            for idx in missing_b:
                row = await run_suite_b_case(
                    stack,
                    suite_b_items[idx],
                    index=idx,
                    endpoint=VLLM_BASE_URL,
                    model=VLLM_MODEL,
                    scratch_dir=scratch,
                )
                append_jsonl(suite_b_path, row)
                rows_b.append(row)
                print(f"[wave11-d7] suite B {len(rows_b)}/{SUITE_B_N} {row['item_id']}", flush=True)
            await _close_stack(stack)
            stack = None
            stopper()
            container_started = False
        if need_judge:
            judge = _load_judge(JUDGE_HF_MODEL)
            rows_a = grade_rows(rows_a, suite_tag="suiteA", judge=judge)
            rows_b = grade_rows(rows_b, suite_tag="suiteB", judge=judge)
            rewrite_jsonl(suite_a_path, rows_a)
            rewrite_jsonl(suite_b_path, rows_b)
        if not checkpoints_complete(rows_a, rows_b):
            raise RuntimeError(
                f"arena finished incomplete: suite A {len(rows_a)} suite B {len(rows_b)}; refusing a live receipt"
            )
        manifest = build_manifest(
            rows_a,
            rows_b,
            output_dir=output_dir,
            probe=probe,
            corpus_sha256=str(corpus_sha),
        )
        emit_wave11_artifacts(output_dir, manifest)
        return manifest
    finally:
        close_error: Optional[BaseException] = None
        if stack is not None:
            try:
                await _close_stack(stack)
            except Exception as exc:
                close_error = exc
                print(f"[wave11-d7] stack close failed: {_public_error(exc)}", flush=True)
        if container_started:
            try:
                stopper()
            except Exception as exc:
                print(f"[wave11-d7] docker rm failed: {_public_error(exc)}", flush=True)
                if close_error is None:
                    close_error = exc
        if close_error is not None and sys.exc_info()[0] is None:
            raise close_error


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Wave-11 D7 live Zep/Graphiti arena")
    parser.add_argument("--mode", required=True, choices=("local-check", "spark1-live"))
    parser.add_argument("--osv-cache", default="")
    parser.add_argument("--lme-json", default="")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--vllm-endpoint", default=VLLM_BASE_URL)
    parser.add_argument("--vllm-model", default=VLLM_MODEL)
    parser.add_argument("--judge-hf-model", default=JUDGE_HF_MODEL)
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if args.mode == "local-check":
        print(json.dumps(run_local_check(), sort_keys=True))
        return 0
    if not args.osv_cache or not args.lme_json:
        raise SystemExit("spark1-live requires --osv-cache and --lme-json")
    asyncio.run(execute_spark1_live(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
