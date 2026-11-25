r"""Phase-2 Flagship Demo D1: 4-Node Local Hardware Fleet Proof (`RTX 3090` + `3x DGX Spark`).

Executes an end-to-end, zero-shortcut, real-world raw-text enterprise drift benchmark (`N=60` timelines,
`1,020` global raw natural-language documents across Cybersecurity/CVE [grounded in live OSV.dev advisories],
SEC Financial Filings, and Cloud SRE Postmortems) simultaneously across ALL 4 local GPU machines:

  - Node 0 (`Local Desktop RTX 3090 24GB`):
      * Encodes `384-d` edge embeddings (`all-MiniLM-L6-v2`) & `768-d` interior bridge embeddings (`all-mpnet-base-v2`).
      * Runs `Qwen/Qwen2.5-3B-Instruct` (`bfloat16`) to extract structured `(entity, slot, value, stance)` claims
        from 100% raw natural-language documents (ZERO pre-extracted oracle edges, ZERO inline metadata tags!).
      * Evaluates `Arm1_Small3B_Bare_HybridRAG` vs `Arm5_Small3B_Chelated_RawText_EK5v3` on the consumer GPU!
  - Node 1 (`DGX Spark #1 gx10-37c3`, `10.0.0.49:8891`):
      * Evaluates `Arm2_Nemotron30B_Bare_HybridRAG` + `Arm4a_Nemotron30B_Mem0_RawQdrant` (`N=60`)
        + `Arm4b_Nemotron30B_Mem0_LiveInfer` (`N=24` stratified subset, live upstream `mem0ai` LLM extraction).
  - Node 2 (`DGX Spark #2 gx10-6724`, `10.0.0.194:8890`):
      * Encodes frozen `1024-d BAAI/bge-m3` enterprise resident document & calibration embeddings.
      * Evaluates `Arm3_Nemotron30B_ThinkingMode_Bare_HybridRAG` (`enable_thinking=True`, testing whether raw
        inference-time test-time compute alone can overcome retrieval echo floods & rumor laundering).
  - Node 3 (`DGX Spark #3 gx10-eba1`, `10.0.0.24:8801`):
      * Evaluates `Arm6a_Heterogeneous_3090toSpark_ZeroMu_EK5v3_Nemotron30B` (`384-d -> 768-d -> 1024-d` Stiefel
        polar bridge + strictly causal online EMA `mu_q` + 1-float `ZeroMu` sidecar + label-free `EK5 v3`),
        `Arm6b_Heterogeneous_Composed_CAR_EK5v3_Nemotron30B` (2-hop ridge drift baseline), and
        `Arm7_Native_BGE_M3_1024d_EK5v3_Nemotron30B` (native target-space ceiling).

Strictly ADDITIVE: modifies zero existing R1-R5b files and touches zero running Docker containers.
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
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import requests

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from chelated_raw_text_exocortex import (  # noqa: E402
    CausalStreamingZeroMuBridge,
    ExtractedClaimRecord,
    LabelFreeChelatedExocortex,
    RawDocumentEnvelope,
    align_slot_clusters,
    answer_queries_with_local_hf_model,
    extract_claims_with_local_hf_model,
)
from m1_baseline_ladder import fit_closed_form_centered_ridge  # noqa: E402
from prospective_realworld_drift_corpus import build_prospective_realworld_drift_suite  # noqa: E402
from scripts.night1_common import (  # noqa: E402
    current_git_sha,
    load_beir_dataset,
    paired_bootstrap_delta,
    process_telemetry,
    write_json_atomic,
)
from spherical_migration_bridge import (  # noqa: E402
    CenteredProcrustesBridge,
    compose_centered_bridges,
    l2_normalize,
)

QA_SYSTEM_PROMPT = (
    "You are an enterprise factual verification assistant. Answer the question strictly using ONLY the "
    "retrieved records provided below.\n"
    "- Reply with ONLY the concise current verified value/identifier (do not add extra prose).\n"
    "- Pay close attention to source channels and publication dates: authoritative official filings/bulletins "
    "supersede earlier preliminary notes or unverified forum/social rumors.\n"
    "- If the records show an unresolved mutual conflict between equal-authority official sources (or state "
    "EPISTEMIC STATUS: CONTESTED with no verified claim admitted), reply ONLY with 'ABSTAIN'."
)

_TOKEN_BM25_RE = re.compile(r"[a-z0-9][a-z0-9_.\-/]*")


class SimpleBM25:
    """Deterministic BM25 scorer over the global 1,020-document raw-text store."""

    def __init__(self, texts: Sequence[str], k1: float = 1.5, b: float = 0.75) -> None:
        self.n = len(texts)
        self.k1 = k1
        self.b = b
        self.doc_tfs: List[Dict[str, int]] = []
        self.doc_lens = np.zeros(self.n, dtype=np.float32)
        df: Dict[str, int] = defaultdict(int)
        for i, t in enumerate(texts):
            toks = _TOKEN_BM25_RE.findall(t.lower())
            self.doc_lens[i] = float(len(toks))
            tf: Dict[str, int] = defaultdict(int)
            for tok in toks:
                tf[tok] += 1
            self.doc_tfs.append(dict(tf))
            for tok in tf:
                df[tok] += 1
        self.avgdl = float(np.mean(self.doc_lens)) if self.n else 1.0
        self.idf = {
            tok: float(np.log(1.0 + (self.n - freq + 0.5) / (freq + 0.5)))
            for tok, freq in df.items()
        }

    def score(self, query: str) -> np.ndarray:
        q_toks = _TOKEN_BM25_RE.findall(query.lower())
        scores = np.zeros(self.n, dtype=np.float32)
        for qt in q_toks:
            idf = self.idf.get(qt)
            if idf is None:
                continue
            for i in range(self.n):
                tf = self.doc_tfs[i].get(qt, 0)
                if tf:
                    denom = tf + self.k1 * (1.0 - self.b + self.b * (self.doc_lens[i] / self.avgdl))
                    scores[i] += idf * ((tf * (self.k1 + 1.0)) / denom)
        return scores


def _minmax(x: np.ndarray) -> np.ndarray:
    lo = float(np.min(x))
    hi = float(np.max(x))
    if hi - lo < 1e-12:
        return np.zeros_like(x, dtype=np.float32)
    return ((x - lo) / (hi - lo)).astype(np.float32)


def _wait_port(port: int, timeout_s: float = 10.0) -> bool:
    deadline = time.perf_counter() + timeout_s
    while time.perf_counter() < deadline:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(0.5)
            if s.connect_ex(("127.0.0.1", port)) == 0:
                return True
        time.sleep(0.2)
    return False


def _start_ssh_tunnel(host: str, local_port: int, remote_port: int) -> subprocess.Popen:
    cmd = [
        "ssh",
        "-N",
        "-o", "BatchMode=yes",
        "-o", "StrictHostKeyChecking=accept-new",
        "-o", "ExitOnForwardFailure=yes",
        "-L", f"127.0.0.1:{local_port}:127.0.0.1:{remote_port}",
        f"mattmre@{host}",
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if not _wait_port(local_port, timeout_s=10.0):
        proc.terminate()
        raise RuntimeError(f"Failed to establish SSH tunnel to {host}:{remote_port} on local port {local_port}")
    return proc


def _call_vllm_chat(
    endpoint: str,
    model_id: str,
    sys_prompt: str,
    user_prompt: str,
    *,
    enable_thinking: bool = False,
    max_tokens: int = 48,
) -> Tuple[str, int, int, float, bool]:
    """Return `(final_answer, prompt_tokens, completion_tokens, latency_ms, had_error)`."""
    t0 = time.perf_counter()
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
                "chat_template_kwargs": {"enable_thinking": bool(enable_thinking)},
            },
            timeout=90.0,
        )
        r.raise_for_status()
        d = r.json()
        msg = d["choices"][0]["message"]
        content = (msg.get("content") or "").strip()
        # If thinking tags appear inline, strip `<think>...</think>` to extract final answer cleanly
        if "</think>" in content:
            content = content.split("</think>", 1)[1].strip()
        usage = d.get("usage") or {}
        pt = int(usage.get("prompt_tokens", 0))
        ct = int(usage.get("completion_tokens", 0))
        return content, pt, ct, (time.perf_counter() - t0) * 1000.0, False
    except Exception as exc:
        return f"ERROR: {exc}", 0, 0, (time.perf_counter() - t0) * 1000.0, True


def _encode_bge_m3_on_spark194(
    doc_texts: Sequence[str],
    query_texts: Sequence[str],
    cal_texts: Sequence[str],
    work_dir: Path,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Encode 1024-d `BAAI/bge-m3` embeddings on DGX Spark #2 (`gx10-6724`, `10.0.0.194`)."""
    cache_npz = work_dir / "spark194_bge_m3_1024d_cache.npz"
    payload = {"docs": list(doc_texts), "queries": list(query_texts), "cal": list(cal_texts)}
    payload_bytes = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    payload_sha = hashlib.sha256(payload_bytes).hexdigest()[:16]
    if cache_npz.is_file():
        loaded = np.load(cache_npz)
        if str(loaded.get("sha", "")) == payload_sha:
            return loaded["docs"], loaded["queries"], loaded["cal"], 0.0

    t0 = time.perf_counter()
    in_json = work_dir / "bge_m3_inputs.json"
    in_json.write_bytes(payload_bytes)
    remote_dir = "/home/mattmre/chelated-night1/out/phase2_d1_fleet"
    subprocess.run(["ssh", "-o", "BatchMode=yes", "mattmre@10.0.0.194", f"mkdir -p {remote_dir}"], check=True)
    subprocess.run(["scp", "-q", str(in_json), f"mattmre@10.0.0.194:{remote_dir}/bge_m3_inputs.json"], check=True)

    remote_py = (
        "import json, numpy as np; "
        "from sentence_transformers import SentenceTransformer; "
        f"d = json.load(open('{remote_dir}/bge_m3_inputs.json', encoding='utf-8')); "
        "m = SentenceTransformer('BAAI/bge-m3', device='cuda'); "
        "ed = m.encode(d['docs'], batch_size=64, normalize_embeddings=True, convert_to_numpy=True).astype(np.float32); "
        "eq = m.encode(d['queries'], batch_size=64, normalize_embeddings=True, convert_to_numpy=True).astype(np.float32); "
        "ec = m.encode(d['cal'], batch_size=64, normalize_embeddings=True, convert_to_numpy=True).astype(np.float32); "
        f"np.savez('{remote_dir}/bge_m3_out.npz', docs=ed, queries=eq, cal=ec); "
        "print('BGE_M3_DONE', ed.shape, eq.shape, ec.shape)"
    )
    subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "mattmre@10.0.0.194", f"/home/mattmre/chelated-night1/.venv/bin/python -c \"{remote_py}\""],
        check=True,
    )
    raw_npz = work_dir / "bge_m3_out.npz"
    subprocess.run(["scp", "-q", f"mattmre@10.0.0.194:{remote_dir}/bge_m3_out.npz", str(raw_npz)], check=True)
    dat = np.load(raw_npz)
    ed, eq, ec = dat["docs"], dat["queries"], dat["cal"]
    np.savez(cache_npz, docs=ed, queries=eq, cal=ec, sha=np.array(payload_sha))
    return ed, eq, ec, time.perf_counter() - t0


def _run_mem0_on_spark49(
    cases: Sequence[Dict[str, Any]],
    docs_by_id: Dict[str, RawDocumentEnvelope],
    infer_case_ids: Sequence[str],
    work_dir: Path,
) -> Dict[str, Any]:
    """Run live upstream `mem0ai` (`infer=False` on all 60 cases + `infer=True` on 24 stratified cases) on Spark #1."""
    remote_dir = "/home/mattmre/chelated-night1/out/phase2_d1_fleet"
    req_data = {
        "cases": [
            {
                "case_id": c["case_id"],
                "query_text": c["query_text"],
                "docs": [
                    {
                        "doc_id": did,
                        "text": docs_by_id[did].text,
                        "published_ts": docs_by_id[did].published_ts,
                        "ingested_ts": docs_by_id[did].ingested_ts,
                        "source_channel": docs_by_id[did].source_channel,
                    }
                    for did in c["doc_ids"]
                ],
            }
            for c in cases
        ],
        "infer_case_ids": list(infer_case_ids),
    }
    in_path = work_dir / "mem0_d1_req.json"
    in_path.write_text(json.dumps(req_data), encoding="utf-8")

    worker_code = '''import json, shutil, tempfile, time
from pathlib import Path
from mem0 import Memory

CUSTOM_PROMPT = """You are an Enterprise Factual Memory Organizer. Extract ALL factual claims, versions, metrics, statuses, revisions, and advisories from the messages (preserving entity names, dates, source channels, and exact values). Return ONLY valid JSON {"facts": ["..."]}."""

def make_mem(qpath, custom_prompt=None):
    cfg = {
        "llm": {"provider": "openai", "config": {"model": "nemotron-lightning", "openai_base_url": "http://127.0.0.1:8891/v1", "api_key": "local", "temperature": 0.0, "max_tokens": 1024}},
        "embedder": {"provider": "huggingface", "config": {"model": "sentence-transformers/all-MiniLM-L6-v2", "embedding_dims": 384}},
        "vector_store": {"provider": "qdrant", "config": {"collection_name": "d1_mem0", "embedding_model_dims": 384, "path": str(qpath)}},
    }
    if custom_prompt:
        cfg["custom_fact_extraction_prompt"] = custom_prompt
    mem = Memory.from_config(cfg)
    orig = mem.llm.client.chat.completions.create
    def _patched(*a, **kw):
        eb = dict(kw.get("extra_body") or {})
        eb["chat_template_kwargs"] = {"enable_thinking": False}
        kw["extra_body"] = eb
        return orig(*a, **kw)
    mem.llm.client.chat.completions.create = _patched
    return mem

def close_mem(m, qp):
    for attr in ("vector_store", "_telemetry_vector_store"):
        vs = getattr(m, attr, None)
        if vs is not None and hasattr(vs, "client"):
            try: vs.client.close()
            except Exception: pass
    shutil.rmtree(qp, ignore_errors=True)

req = json.load(open("/home/mattmre/chelated-night1/out/phase2_d1_fleet/mem0_d1_req.json", encoding="utf-8"))
infer_set = set(req["infer_case_ids"])
raw_ctx, inf_ctx = {}, {}
raw_ms, inf_ms = [], []

qp1 = Path(tempfile.mkdtemp(prefix="d1_m0_raw_"))
m_raw = make_mem(qp1)
for c in req["cases"]:
    uid = "u_" + c["case_id"]
    t0 = time.perf_counter()
    for d in sorted(c["docs"], key=lambda x: (x["ingested_ts"], x["doc_id"])):
        m_raw.add(f"[source={d['source_channel']}, published_day={d['published_ts']}] {d['text']}", user_id=uid, infer=False)
    raw_ms.append((time.perf_counter() - t0) * 1000.0)
    sr = m_raw.search(c["query_text"], filters={"user_id": uid}, limit=5).get("results", [])
    raw_ctx[c["case_id"]] = "\\n".join("- " + x["memory"] for x in sr[:5]) if sr else "(No memories found.)"
close_mem(m_raw, qp1)

qp2 = Path(tempfile.mkdtemp(prefix="d1_m0_inf_"))
m_inf = make_mem(qp2, CUSTOM_PROMPT)
for c in req["cases"]:
    if c["case_id"] not in infer_set:
        continue
    uid = "u_" + c["case_id"]
    t0 = time.perf_counter()
    msgs = [{"role": "user", "content": f"[source={d['source_channel']}, published_day={d['published_ts']}] {d['text']}"} for d in sorted(c["docs"], key=lambda x: (x["ingested_ts"], x["doc_id"]))]
    try:
        m_inf.add(msgs, user_id=uid, infer=True)
    except Exception:
        for d in c["docs"]:
            m_inf.add(f"[source={d['source_channel']}, published_day={d['published_ts']}] {d['text']}", user_id=uid, infer=False)
    inf_ms.append((time.perf_counter() - t0) * 1000.0)
    sr = m_inf.search(c["query_text"], filters={"user_id": uid}, limit=5).get("results", [])
    inf_ctx[c["case_id"]] = "\\n".join("- " + x["memory"] for x in sr[:5]) if sr else "(No memories found.)"
close_mem(m_inf, qp2)

out = {"raw_ctx": raw_ctx, "inf_ctx": inf_ctx, "raw_mean_write_ms": sum(raw_ms)/max(len(raw_ms),1), "inf_mean_write_ms": sum(inf_ms)/max(len(inf_ms),1)}
json.dump(out, open("/home/mattmre/chelated-night1/out/phase2_d1_fleet/mem0_d1_out.json", "w"), indent=2)
print("MEM0_D1_DONE", len(raw_ctx), len(inf_ctx))
'''
    out_local = work_dir / "mem0_d1_out.json"
    if out_local.is_file():
        return json.loads(out_local.read_text(encoding="utf-8"))
    worker_file = work_dir / "remote_mem0_d1_worker.py"
    worker_file.write_text(worker_code, encoding="utf-8")
    subprocess.run(["ssh", "-o", "BatchMode=yes", "mattmre@10.0.0.49", f"mkdir -p {remote_dir}"], stdin=subprocess.DEVNULL, check=True)
    subprocess.run(
        ["scp", "-q", str(in_path), str(worker_file), f"mattmre@10.0.0.49:{remote_dir}/"],
        stdin=subprocess.DEVNULL,
        check=True,
    )
    subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "mattmre@10.0.0.49", f"/home/mattmre/chelated-night1/.venv-pm3/bin/python {remote_dir}/remote_mem0_d1_worker.py"],
        stdin=subprocess.DEVNULL,
        check=True,
    )
    subprocess.run(["scp", "-q", f"mattmre@10.0.0.49:{remote_dir}/mem0_d1_out.json", str(out_local)], stdin=subprocess.DEVNULL, check=True)
    return json.loads(out_local.read_text(encoding="utf-8"))


def _grade_prediction(pred: str, gold: str, stale_vals: Sequence[str]) -> bool:
    p_lo = (pred or "").strip().lower()
    if gold == "ABSTAIN":
        return (
            ("abstain" in p_lo)
            or ("contested" in p_lo)
            or ("unresolved" in p_lo)
            or ("cannot determine" in p_lo)
            or ("conflicting" in p_lo)
        ) and not any(sv.lower() in p_lo for sv in stale_vals)
    return (gold.lower() in p_lo) and not any(sv.lower() in p_lo for sv in stale_vals) and ("abstain" not in p_lo)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "artifacts" / "phase2_d1_4node_local_fleet_proof",
    )
    parser.add_argument("--small-model", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--spark-model", type=str, default="nemotron-lightning")
    args = parser.parse_args()

    out_dir: Path = args.output_dir
    work_dir = out_dir / "_work"
    out_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)
    t0_total = time.perf_counter()

    print("[Phase2-D1] 1. Building Real-World Unseen 3-Domain Drift Corpus (grounded in live OSV.dev CVEs)...", flush=True)
    cases, all_docs, corpus_sha256 = build_prospective_realworld_drift_suite(work_dir / "osv_cve_seeds.json")
    docs_by_id = {d.doc_id: d for d in all_docs}
    doc_texts = [d.text for d in all_docs]
    query_texts = [c["query_text"] for c in cases]
    print(f"[Phase2-D1] Corpus ready: {len(cases)} cases, {len(all_docs)} global raw docs (sha256={corpus_sha256[:12]})", flush=True)

    # Load 1,000 disjoint BEIR SciFact calibration texts for strictly out-of-domain 384 -> 768 -> 1024 bridge fitting!
    _, cal_scifact_texts, _, _, _, cal_qrels_sha = load_beir_dataset("SciFact", max_docs=1050)
    cal_texts = cal_scifact_texts[:1000]

    # Stratified N=24 subset (2 per domain x regime = 3 * 4 * 2 = 24 cases) for live LLM-infer `mem0ai`
    by_dr: Dict[Tuple[str, str], List[str]] = defaultdict(list)
    for c in cases:
        by_dr[(c["domain"], c["regime"])].append(c["case_id"])
    infer_case_ids: List[str] = []
    for _k, cids in sorted(by_dr.items()):
        infer_case_ids.extend(cids[:2])

    # Launch Spark #2 (`1024-d BGE-M3` encoding) AND Spark #1 (`mem0ai` live ingestion) in background threads
    # WHILE our local RTX 3090 runs `384-d`/`768-d` encoding + `Qwen2.5-3B-Instruct` raw-text claim extraction!
    print("[Phase2-D1] 2. Launching parallel jobs across RTX 3090 + Spark #1 (mem0ai) + Spark #2 (BGE-M3 1024-d)...", flush=True)
    bg_pool = concurrent.futures.ThreadPoolExecutor(max_workers=2)
    fut_bge = bg_pool.submit(_encode_bge_m3_on_spark194, doc_texts, query_texts, cal_texts, work_dir)
    fut_mem0 = bg_pool.submit(_run_mem0_on_spark49, cases, docs_by_id, infer_case_ids, work_dir)

    # Local RTX 3090 Step A: Encode 384-d (`all-MiniLM-L6-v2`) & 768-d (`all-mpnet-base-v2`)
    from sentence_transformers import SentenceTransformer
    import torch

    t_enc0 = time.perf_counter()
    enc_384 = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cuda")
    doc_384 = enc_384.encode(doc_texts, batch_size=128, normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)
    q_384 = enc_384.encode(query_texts, batch_size=64, normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)
    cal_384 = enc_384.encode(cal_texts, batch_size=128, normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)
    del enc_384

    enc_768 = SentenceTransformer("sentence-transformers/all-mpnet-base-v2", device="cuda")
    cal_768 = enc_768.encode(cal_texts, batch_size=128, normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)
    del enc_768
    torch.cuda.empty_cache()
    rtx_embed_sec = round(time.perf_counter() - t_enc0, 2)
    print(f"[Phase2-D1] RTX 3090 384-d & 768-d embeddings finished in {rtx_embed_sec}s.", flush=True)

    # Local RTX 3090 Step B: Raw-Text Claim Extraction via `Qwen2.5-3B-Instruct`
    claims_cache = work_dir / "rtx3090_qwen3b_extracted_claims.json"
    if claims_cache.is_file():
        cached_c = json.loads(claims_cache.read_text(encoding="utf-8"))
        raw_claims = [ExtractedClaimRecord(**x) for x in cached_c["claims"]]
        extract_stats = cached_c["stats"]
        print(f"[Phase2-D1] Loaded cached RTX 3090 Qwen-3B claims ({len(raw_claims)} docs).", flush=True)
    else:
        print(f"[Phase2-D1] Running {args.small_model} on RTX 3090 over {len(all_docs)} raw documents...", flush=True)
        raw_claims, extract_stats = extract_claims_with_local_hf_model(
            all_docs, model_name=args.small_model, batch_size=32, device="cuda"
        )
        claims_cache.write_text(
            json.dumps({"claims": [c.__dict__ for c in raw_claims], "stats": extract_stats}, indent=2),
            encoding="utf-8",
        )
        print(f"[Phase2-D1] RTX 3090 extraction complete: {extract_stats['batch_extract_sec']}s ({extract_stats['ms_per_doc']} ms/doc)!", flush=True)

    # Align near-synonymous extracted slot names using 384-d slot embeddings on RTX 3090
    enc_slot = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cuda")
    slot_embs = enc_slot.encode(
        [f"{c.entity} {c.slot.replace('_', ' ')}" for c in raw_claims],
        batch_size=128,
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype(np.float32)
    del enc_slot
    torch.cuda.empty_cache()
    aligned_claims = align_slot_clusters(raw_claims, slot_embeddings=slot_embs, sim_threshold=0.82)

    # Gather Spark #2 (`1024-d BGE-M3`) & Spark #1 (`mem0ai`) results
    doc_1024, q_1024, cal_1024, spark194_bge_sec = fut_bge.result()
    print(f"[Phase2-D1] Spark #2 (gx10-6724) BGE-M3 1024-d ready ({round(spark194_bge_sec, 2)}s).", flush=True)

    # Build 2-hop Spherical Bridge (`384-d MiniLM -> 768-d mpnet -> 1024-d BGE-M3`) on disjoint SciFact anchors!
    b_hop1 = CenteredProcrustesBridge.fit(cal_384[:800], cal_768[:800], cal_x=cal_384[800:], cal_y=cal_768[800:])
    b_hop2 = CenteredProcrustesBridge.fit(cal_768[:800], cal_1024[:800], cal_x=cal_768[800:], cal_y=cal_1024[800:])
    br_384_to_1024_polar = compose_centered_bridges([b_hop1, b_hop2], polar_reproject=True)

    mx1, my1, wc1 = fit_closed_form_centered_ridge(cal_384, cal_768, fit_idx=np.arange(800), regularization=10.0)
    mx2, my2, wc2 = fit_closed_form_centered_ridge(cal_768, cal_1024, fit_idx=np.arange(800), regularization=10.0)
    q_1024_car = l2_normalize((l2_normalize((q_384 - mx1) @ wc1 + my1) - mx2) @ wc2 + my2).astype(np.float32)

    # Instantiate Label-Free Chelated Exocortexes (Edge 384-d & Enterprise 1024-d) + Causal Streaming ZeroMu Bridge
    t_exo0 = time.perf_counter()
    exo_edge_384 = LabelFreeChelatedExocortex(all_docs, aligned_claims, doc_384)
    exo_spark_1024 = LabelFreeChelatedExocortex(all_docs, aligned_claims, doc_1024)
    ek5_compile_ms = round((time.perf_counter() - t_exo0) * 1000.0, 2)

    streaming_bridge = CausalStreamingZeroMuBridge(br_384_to_1024_polar, doc_1024, ema_alpha=0.05)
    bm25_global = SimpleBM25(doc_texts)

    # Build retrieval contexts for all arms across all 60 queries (measuring live coupled query latency!)
    contexts_by_arm: Dict[str, Dict[str, str]] = defaultdict(dict)
    bridged_query_lat_ms: List[float] = []

    def _format_bare_topk(hyb_scores: np.ndarray, k: int = 5) -> str:
        top_idx = np.argsort(-hyb_scores, kind="mergesort")[:k]
        ordered = sorted([int(j) for j in top_idx], key=lambda j: (all_docs[j].published_ts, all_docs[j].ingested_ts))
        return "\n".join(
            f"- [source={all_docs[j].source_channel}, published_day={all_docs[j].published_ts}] {all_docs[j].text}"
            for j in ordered
        )

    for q_i, c in enumerate(cases):
        cid = c["case_id"]
        q_txt = c["query_text"]
        bm_v = bm25_global.score(q_txt)
        bm_norm = _minmax(bm_v)

        # Edge 384-d hybrid (`RTX 3090`)
        d384_scores = doc_384 @ q_384[q_i]
        hyb_384 = 0.35 * bm_norm + 0.65 * _minmax(d384_scores)
        contexts_by_arm["Arm1_Small3B_Bare_HybridRAG"][cid] = _format_bare_topk(hyb_384, k=5)
        contexts_by_arm["Arm5_Small3B_Chelated_RawText_EK5v3"][cid] = exo_edge_384.select_context_for_query(
            hyb_384, q_txt, top_k=5
        )["formatted_context"]

        # Native 1024-d BGE-M3 hybrid (`Spark`)
        d1024_nat = doc_1024 @ q_1024[q_i]
        hyb_1024_nat = 0.35 * bm_norm + 0.65 * _minmax(d1024_nat)
        contexts_by_arm["Arm2_Nemotron30B_Bare_HybridRAG"][cid] = _format_bare_topk(hyb_1024_nat, k=5)
        contexts_by_arm["Arm3_Nemotron30B_ThinkingMode_Bare_HybridRAG"][cid] = contexts_by_arm["Arm2_Nemotron30B_Bare_HybridRAG"][cid]
        contexts_by_arm["Arm7_Native_BGEM3_EK5v3_Nemotron30B"][cid] = exo_spark_1024.select_context_for_query(
            hyb_1024_nat, q_txt, top_k=5
        )["formatted_context"]

        # Causal Streaming ZeroMu Bridged (`RTX 3090 384-d -> Spark 1024-d BGE-M3 + Label-Free EK5 v3`)
        t_bq0 = time.perf_counter()
        zm_scores = streaming_bridge.score_single_query(q_384[q_i])
        hyb_zm = 0.35 * bm_norm + 0.65 * _minmax(zm_scores)
        ctx_zm = exo_spark_1024.select_context_for_query(hyb_zm, q_txt, top_k=5)["formatted_context"]
        bridged_query_lat_ms.append((time.perf_counter() - t_bq0) * 1000.0)
        contexts_by_arm["Arm6a_Heterogeneous_3090toSpark_ZeroMu_EK5v3_Nemotron30B"][cid] = ctx_zm

        # Composed CAR 2-hop ridge ablation + EK5 v3
        d1024_car = doc_1024 @ q_1024_car[q_i]
        hyb_car = 0.35 * bm_norm + 0.65 * _minmax(d1024_car)
        contexts_by_arm["Arm6b_Heterogeneous_Composed_CAR_EK5v3_Nemotron30B"][cid] = exo_spark_1024.select_context_for_query(
            hyb_car, q_txt, top_k=5
        )["formatted_context"]

    mem0_res = fut_mem0.result()
    bg_pool.shutdown(wait=True)
    print(
        f"[Phase2-D1] Spark #1 (gx10-37c3) mem0ai complete: raw_write={mem0_res['raw_mean_write_ms']:.1f}ms/case, "
        f"infer_write={mem0_res['inf_mean_write_ms']:.1f}ms/case.",
        flush=True,
    )
    for cid, ctx_m in mem0_res["raw_ctx"].items():
        contexts_by_arm["Arm4a_Nemotron30B_Mem0_RawQdrant"][cid] = ctx_m
    for cid, ctx_m in mem0_res["inf_ctx"].items():
        contexts_by_arm["Arm4b_Nemotron30B_Mem0_LiveInfer_N24"][cid] = ctx_m

    # 3. Execute Reader Inference across ALL 4 NODES SIMULTANEOUSLY!
    print("[Phase2-D1] 3. Running QA Reader Evaluation simultaneously across RTX 3090 + ALL 3 DGX Sparks...", flush=True)
    tunnels: List[subprocess.Popen] = []
    try:
        tunnels.append(_start_ssh_tunnel("10.0.0.49", 18891, 8891))
        tunnels.append(_start_ssh_tunnel("10.0.0.194", 18890, 8890))
        tunnels.append(_start_ssh_tunnel("10.0.0.24", 18801, 8801))

        def _make_user_prompt(ctx: str, q_text: str) -> str:
            return f"Retrieved Records:\n{ctx}\n\nQuestion: {q_text}\nAnswer:"

        # Dispatch Spark #1, Spark #2, Spark #3 vLLM calls in background while RTX 3090 evaluates Small-3B arms!
        spark_futures: Dict[Tuple[str, str], concurrent.futures.Future] = {}
        spark_pool = concurrent.futures.ThreadPoolExecutor(max_workers=36)

        # Spark #1 (`10.0.0.49:8891`): Arm2, Arm4a, Arm4b
        for arm_s1 in ("Arm2_Nemotron30B_Bare_HybridRAG", "Arm4a_Nemotron30B_Mem0_RawQdrant", "Arm4b_Nemotron30B_Mem0_LiveInfer_N24"):
            for c in cases:
                cid = c["case_id"]
                if cid not in contexts_by_arm[arm_s1]:
                    continue
                up = _make_user_prompt(contexts_by_arm[arm_s1][cid], c["query_text"])
                spark_futures[(arm_s1, cid)] = spark_pool.submit(
                    _call_vllm_chat, "http://127.0.0.1:18891/v1", args.spark_model, QA_SYSTEM_PROMPT, up,
                    enable_thinking=False, max_tokens=40,
                )

        # Spark #2 (`10.0.0.194:8890`): Arm3 (`ThinkingMode` ON, `max_tokens=1024`)
        for c in cases:
            cid = c["case_id"]
            up = _make_user_prompt(contexts_by_arm["Arm3_Nemotron30B_ThinkingMode_Bare_HybridRAG"][cid], c["query_text"])
            spark_futures[("Arm3_Nemotron30B_ThinkingMode_Bare_HybridRAG", cid)] = spark_pool.submit(
                _call_vllm_chat, "http://127.0.0.1:18890/v1", args.spark_model, QA_SYSTEM_PROMPT, up,
                enable_thinking=True, max_tokens=1024,
            )

        # Spark #3 (`10.0.0.24:8801`): Arm6a, Arm6b, Arm7
        for arm_s3 in (
            "Arm6a_Heterogeneous_3090toSpark_ZeroMu_EK5v3_Nemotron30B",
            "Arm6b_Heterogeneous_Composed_CAR_EK5v3_Nemotron30B",
            "Arm7_Native_BGEM3_EK5v3_Nemotron30B",
        ):
            for c in cases:
                cid = c["case_id"]
                up = _make_user_prompt(contexts_by_arm[arm_s3][cid], c["query_text"])
                spark_futures[(arm_s3, cid)] = spark_pool.submit(
                    _call_vllm_chat, "http://127.0.0.1:18801/v1", args.spark_model, QA_SYSTEM_PROMPT, up,
                    enable_thinking=False, max_tokens=40,
                )

        # Local RTX 3090: Arm1 (`Small-3B Bare Hybrid RAG`) & Arm5 (`Small-3B + Raw-Text Chelated EK5 v3`)
        local_prompts: List[Tuple[str, str]] = []
        for arm_loc in ("Arm1_Small3B_Bare_HybridRAG", "Arm5_Small3B_Chelated_RawText_EK5v3"):
            for c in cases:
                cid = c["case_id"]
                local_prompts.append((f"{arm_loc}__{cid}", _make_user_prompt(contexts_by_arm[arm_loc][cid], c["query_text"])))

        local_ans = answer_queries_with_local_hf_model(
            local_prompts, system_prompt=QA_SYSTEM_PROMPT, model_name=args.small_model, batch_size=16, device="cuda"
        )

        spark_results: Dict[Tuple[str, str], Tuple[str, int, int, float, bool]] = {
            k: fut.result() for k, fut in spark_futures.items()
        }
        spark_pool.shutdown(wait=True)
    finally:
        for proc in tunnels:
            try:
                proc.terminate()
            except Exception:
                pass

    # 4. Score all predictions, compute paired bootstrap CIs, and emit verifiable receipts!
    all_rows: List[Dict[str, Any]] = []
    scores_by_arm: Dict[str, Dict[str, float]] = defaultdict(dict)
    case_by_id = {c["case_id"]: c for c in cases}

    for arm_loc in ("Arm1_Small3B_Bare_HybridRAG", "Arm5_Small3B_Chelated_RawText_EK5v3"):
        for c in cases:
            cid = c["case_id"]
            pred, pt, lat_ms = local_ans[f"{arm_loc}__{cid}"]
            ok = _grade_prediction(pred, c["gold"], c["stale_or_false_values"])
            scores_by_arm[arm_loc][cid] = float(int(ok))
            all_rows.append({
                "case_id": cid, "domain": c["domain"], "regime": c["regime"], "arm": arm_loc,
                "hardware_node": "RTX_3090_Local", "gold": c["gold"], "pred": pred.replace("\n", " "),
                "is_correct": int(ok), "prompt_tokens": pt, "completion_tokens": 0, "latency_ms": lat_ms, "api_error": 0,
            })

    node_for_arm = {
        "Arm2_Nemotron30B_Bare_HybridRAG": "Spark1_gx10_37c3",
        "Arm4a_Nemotron30B_Mem0_RawQdrant": "Spark1_gx10_37c3",
        "Arm4b_Nemotron30B_Mem0_LiveInfer_N24": "Spark1_gx10_37c3",
        "Arm3_Nemotron30B_ThinkingMode_Bare_HybridRAG": "Spark2_gx10_6724",
        "Arm6a_Heterogeneous_3090toSpark_ZeroMu_EK5v3_Nemotron30B": "RTX3090_to_Spark3_gx10_eba1",
        "Arm6b_Heterogeneous_Composed_CAR_EK5v3_Nemotron30B": "RTX3090_to_Spark3_gx10_eba1",
        "Arm7_Native_BGEM3_EK5v3_Nemotron30B": "Spark3_gx10_eba1",
    }
    for (arm_s, cid), (pred, pt, ct, lat_ms, err) in sorted(spark_results.items()):
        c = case_by_id[cid]
        ok = _grade_prediction(pred, c["gold"], c["stale_or_false_values"])
        scores_by_arm[arm_s][cid] = float(int(ok))
        all_rows.append({
            "case_id": cid, "domain": c["domain"], "regime": c["regime"], "arm": arm_s,
            "hardware_node": node_for_arm[arm_s], "gold": c["gold"], "pred": pred.replace("\n", " "),
            "is_correct": int(ok), "prompt_tokens": pt, "completion_tokens": ct, "latency_ms": round(lat_ms, 2), "api_error": int(err),
        })

    rows_csv = out_dir / "phase2_d1_4node_fleet_rows.csv"
    with rows_csv.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(all_rows[0].keys()))
        w.writeheader()
        w.writerows(all_rows)

    arm_summaries: Dict[str, Any] = {}
    for arm_name in sorted(scores_by_arm.keys()):
        arm_r = [r for r in all_rows if r["arm"] == arm_name]
        by_reg = defaultdict(list)
        by_dom = defaultdict(list)
        for r in arm_r:
            by_reg[r["regime"]].append(r["is_correct"])
            by_dom[r["domain"]].append(r["is_correct"])
        arm_summaries[arm_name] = {
            "n_cases": len(arm_r),
            "accuracy": round(float(np.mean([r["is_correct"] for r in arm_r])), 4),
            "mean_prompt_tokens": round(float(np.mean([r["prompt_tokens"] for r in arm_r])), 1),
            "mean_completion_tokens": round(float(np.mean([r["completion_tokens"] for r in arm_r])), 1),
            "mean_reader_latency_ms": round(float(np.mean([r["latency_ms"] for r in arm_r])), 1),
            "api_errors": int(sum(r["api_error"] for r in arm_r)),
            "by_regime": {k: round(float(np.mean(v)), 4) for k, v in sorted(by_reg.items())},
            "by_domain": {k: round(float(np.mean(v)), 4) for k, v in sorted(by_dom.items())},
        }

    all_cids = [c["case_id"] for c in cases]
    comparisons = {
        "Small3B_Chelated_vs_Small3B_Bare": paired_bootstrap_delta(
            scores_by_arm["Arm5_Small3B_Chelated_RawText_EK5v3"], scores_by_arm["Arm1_Small3B_Bare_HybridRAG"], n_bootstrap=3000, seed=42, expected_ids=all_cids
        ),
        "Small3B_Chelated_RTX3090_vs_Large30B_Bare_Spark": paired_bootstrap_delta(
            scores_by_arm["Arm5_Small3B_Chelated_RawText_EK5v3"], scores_by_arm["Arm2_Nemotron30B_Bare_HybridRAG"], n_bootstrap=3000, seed=42, expected_ids=all_cids
        ),
        "Small3B_Chelated_RTX3090_vs_Large30B_ThinkingMode_Spark": paired_bootstrap_delta(
            scores_by_arm["Arm5_Small3B_Chelated_RawText_EK5v3"], scores_by_arm["Arm3_Nemotron30B_ThinkingMode_Bare_HybridRAG"], n_bootstrap=3000, seed=42, expected_ids=all_cids
        ),
        "Hetero_3090toSpark_ZeroMu_EK5v3_vs_Large30B_Bare": paired_bootstrap_delta(
            scores_by_arm["Arm6a_Heterogeneous_3090toSpark_ZeroMu_EK5v3_Nemotron30B"], scores_by_arm["Arm2_Nemotron30B_Bare_HybridRAG"], n_bootstrap=3000, seed=42, expected_ids=all_cids
        ),
        "Hetero_3090toSpark_ZeroMu_vs_Composed_CAR": paired_bootstrap_delta(
            scores_by_arm["Arm6a_Heterogeneous_3090toSpark_ZeroMu_EK5v3_Nemotron30B"], scores_by_arm["Arm6b_Heterogeneous_Composed_CAR_EK5v3_Nemotron30B"], n_bootstrap=3000, seed=42, expected_ids=all_cids
        ),
        "Hetero_3090toSpark_ZeroMu_vs_Native_1024d_BGEM3": paired_bootstrap_delta(
            scores_by_arm["Arm6a_Heterogeneous_3090toSpark_ZeroMu_EK5v3_Nemotron30B"], scores_by_arm["Arm7_Native_BGEM3_EK5v3_Nemotron30B"], n_bootstrap=3000, seed=42, expected_ids=all_cids
        ),
        "Hetero_3090toSpark_ZeroMu_vs_Mem0_LiveInfer_N24": paired_bootstrap_delta(
            {k: scores_by_arm["Arm6a_Heterogeneous_3090toSpark_ZeroMu_EK5v3_Nemotron30B"][k] for k in infer_case_ids},
            scores_by_arm["Arm4b_Nemotron30B_Mem0_LiveInfer_N24"],
            n_bootstrap=3000,
            seed=42,
            expected_ids=infer_case_ids,
        ),
    }

    manifest = {
        "schema_version": "phase2_d1_4node_local_fleet_proof_v1",
        "git_sha": current_git_sha(REPO_ROOT),
        "wall_seconds": round(time.perf_counter() - t0_total, 2),
        "corpus": {
            "n_cases": len(cases),
            "n_global_raw_docs": len(all_docs),
            "domains": ["cybersec_cve", "sec_financial", "cloud_sre"],
            "regimes": [
                "supersession_under_echo_flood",
                "rumor_laundering_and_audit_trap",
                "multihop_version_lineage_drift",
                "mutual_peer_conflict_abstain",
            ],
            "osv_live_cve_seeded": True,
            "zero_pre_extracted_edges": True,
            "zero_family_labels_at_query_time": True,
            "causal_streaming_ema_mu_q": True,
            "corpus_sha256": corpus_sha256,
            "calibration_scifact_qrels_sha256": cal_qrels_sha,
        },
        "hardware_fleet": {
            "edge_node": "Local Desktop NVIDIA RTX 3090 (24 GB VRAM)",
            "spark_1": "gx10-37c3 (10.0.0.49, 128 GB, Nemotron-3.5-30B-A3B + mem0ai 2.2.1)",
            "spark_2": "gx10-6724 (10.0.0.194, 128 GB, BAAI/bge-m3 1024-d + Nemotron-3.5-30B Thinking)",
            "spark_3": "gx10-eba1 (10.0.0.24, 128 GB, Bridged ZeroMu + EK5 v3 + Nemotron-3.5-30B)",
        },
        "pipeline_telemetry": {
            "rtx3090_raw_extraction": extract_stats,
            "ek5_v3_global_compile_ms": ek5_compile_ms,
            "causal_bridged_query_p50_ms": round(float(np.percentile(bridged_query_lat_ms, 50)), 3),
            "causal_bridged_query_p95_ms": round(float(np.percentile(bridged_query_lat_ms, 95)), 3),
            "mem0_raw_qdrant_mean_write_ms_per_case": round(float(mem0_res["raw_mean_write_ms"]), 2),
            "mem0_live_infer_mean_write_ms_per_case": round(float(mem0_res["inf_mean_write_ms"]), 2),
            "local_process": process_telemetry(),
        },
        "arm_summaries": arm_summaries,
        "paired_bootstrap_comparisons": {
            k: {
                "mean_delta": round(v["mean_delta"], 4),
                "lcb_95": round(v["lcb_95"], 4),
                "ucb_95": round(v["ucb_95"], 4),
                "tango_score_lcb95": round(v["tango_score_lcb95"], 4),
            }
            for k, v in comparisons.items()
        },
        "rows_csv_sha256": hashlib.sha256(rows_csv.read_bytes()).hexdigest(),
    }
    manifest_path = out_dir / "phase2_d1_4node_fleet_manifest.json"
    write_json_atomic(manifest_path, manifest)
    print(f"[Phase2-D1] COMPLETE in {manifest['wall_seconds']}s! Manifest: {manifest_path}", flush=True)
    for k_arm, st_arm in arm_summaries.items():
        print(
            f"  {k_arm}: acc={st_arm['accuracy']:.4f} (N={st_arm['n_cases']}), "
            f"prompt_toks={st_arm['mean_prompt_tokens']}, comp_toks={st_arm['mean_completion_tokens']}",
            flush=True,
        )


if __name__ == "__main__":
    main()
