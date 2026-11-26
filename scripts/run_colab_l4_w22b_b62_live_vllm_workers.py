#!/usr/bin/env python3
"""Colab L4 Wave 22B: Bundle #62 / Demo D13 Live Fast-Worker SLA Remediation (`Qwen/Qwen2.5-1.5B-Instruct`).

Starts a local OpenAI-compatible `/v1/chat/completions` dynamic micro-batching server backed by
live `Qwen/Qwen2.5-1.5B-Instruct` BF16 weights on the Colab NVIDIA L4 GPU (preserving Colab's native
CUDA PyTorch without wheel downgrades), warms CUDA kernels, verifies single & 4-concurrent request
latency, and then invokes the UNMODIFIED production runner:
    `scripts/run_pv7_w20_d13_pooled_agent_loops.py --mode run-live-workers`
    `--vllm-url http://127.0.0.1:8000/v1 --model Qwen/Qwen2.5-1.5B-Instruct`
    `--pool-sizes 4 --schedules S1_steady`

Matches Spark #3's exact 70-cell / 1,680-job live sweep (`JOB_SLA_DEADLINE_MS = 560.0` untouched!)
while stamping explicit `hardware_cohort: "colab:NVIDIA L4"` provenance so Colab results are never
conflated with local DGX Spark GB10 receipts.
"""

from __future__ import annotations

import argparse
import hashlib
import http.server
import json
import os
import platform
import queue
import socketserver
import subprocess
import sys
import threading
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from chelated_pv7_spine.claim_registry import claim_registry_sha256  # noqa: E402

DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


class MicroBatchedCausalServer:
    """Thread-safe dynamic micro-batching greedy decoder over a real HuggingFace CausalLM on CUDA."""

    def __init__(
        self,
        model_id: str,
        *,
        device: str = "cuda",
        max_batch_size: int = 8,
        batch_wait_ms: float = 3.0,
    ) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_id = model_id
        self.device = device
        self.max_batch_size = max_batch_size
        self.batch_wait_s = batch_wait_ms / 1000.0
        self._queue: "queue.Queue[Tuple[str, int, threading.Event, Dict[str, Any]]]" = queue.Queue()
        self._stop = threading.Event()

        print(f"[w22b:server] Loading {model_id} on {device} (bfloat16) ...", flush=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map=device,
        )
        self.model.eval()
        self._eos_id = int(self.tokenizer.eos_token_id)
        # Match ANY BPE token containing '}' (e.g. '}', '"}', '}\n') so JSON early-stop triggers reliably
        self._close_brace_ids = {
            int(tid) for tok_str, tid in self.tokenizer.get_vocab().items() if "}" in str(tok_str)
        }
        self._compiled_decode = None
        if device.startswith("cuda"):
            try:
                # Compile single-token decode step with dynamic shapes so KV-cache decode eliminates Python overhead
                def _decode_one(next_tok: torch.Tensor, attn_mask: torch.Tensor, pkv: Any) -> Any:
                    return self.model(
                        input_ids=next_tok,
                        attention_mask=attn_mask,
                        past_key_values=pkv,
                        use_cache=True,
                    )

                self._compiled_decode = torch.compile(_decode_one, mode="default", dynamic=True)
                print("[w22b:server] Enabled torch.compile(dynamic=True) for decode step", flush=True)
            except Exception as exc:
                print(f"[w22b:server] torch.compile skipped ({type(exc).__name__}: {exc})", flush=True)
                self._compiled_decode = None
        self._reqs_served = 0
        self._recent_ms: List[float] = []
        self._worker = threading.Thread(target=self._batch_loop, daemon=True)
        self._worker.start()

    def _run_batch(self, prompts: List[str], max_new_tokens: int) -> List[str]:
        import torch

        t_b = time.perf_counter()
        enc = self.tokenizer(prompts, return_tensors="pt", padding=True)
        input_ids = enc["input_ids"].to(self.device)
        attention_mask = enc["attention_mask"].to(self.device)
        batch_size = input_ids.shape[0]
        finished = [False] * batch_size
        generated: List[List[int]] = [[] for _ in range(batch_size)]
        cap_tokens = min(int(max_new_tokens), 52)

        with torch.inference_mode():
            out = self.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True)
            past_key_values = out.past_key_values
            next_tokens = torch.argmax(out.logits[:, -1, :], dim=-1)

            for step in range(cap_tokens):
                tok_list = next_tokens.tolist()
                all_done = True
                for i, tid in enumerate(tok_list):
                    if not finished[i]:
                        if tid == self._eos_id:
                            finished[i] = True
                        else:
                            generated[i].append(int(tid))
                            if tid in self._close_brace_ids and len(generated[i]) >= 6:
                                dec = self.tokenizer.decode(generated[i], skip_special_tokens=True)
                                if dec.count("{") > 0 and dec.count("}") >= dec.count("{"):
                                    finished[i] = True
                    if not finished[i]:
                        all_done = False
                if all_done or step + 1 >= cap_tokens:
                    break
                attention_mask = torch.cat(
                    [attention_mask, torch.ones((batch_size, 1), dtype=attention_mask.dtype, device=self.device)],
                    dim=1,
                )
                if self._compiled_decode is not None:
                    try:
                        out = self._compiled_decode(next_tokens.unsqueeze(1), attention_mask, past_key_values)
                    except Exception as exc:
                        print(f"[w22b:server] compiled decode fallback to eager ({type(exc).__name__}: {exc})", flush=True)
                        self._compiled_decode = None
                        out = self.model(
                            input_ids=next_tokens.unsqueeze(1),
                            attention_mask=attention_mask,
                            past_key_values=past_key_values,
                            use_cache=True,
                        )
                else:
                    out = self.model(
                        input_ids=next_tokens.unsqueeze(1),
                        attention_mask=attention_mask,
                        past_key_values=past_key_values,
                        use_cache=True,
                    )
                past_key_values = out.past_key_values
                next_tokens = torch.argmax(out.logits[:, -1, :], dim=-1)

        elapsed_ms = (time.perf_counter() - t_b) * 1000.0
        self._reqs_served += batch_size
        self._recent_ms.append(elapsed_ms)
        if self._reqs_served % 400 < batch_size:
            tail = sorted(self._recent_ms[-100:])
            p50 = tail[len(tail) // 2]
            p95 = tail[min(len(tail) - 1, int(0.95 * len(tail)))]
            print(
                f"[w22b:progress] reqs_served={self._reqs_served} last_bs={batch_size} "
                f"batch_p50_ms={p50:.1f} batch_p95_ms={p95:.1f}",
                flush=True,
            )

        return [self.tokenizer.decode(ids, skip_special_tokens=True).strip() for ids in generated]

    def _batch_loop(self) -> None:
        while not self._stop.is_set():
            try:
                first = self._queue.get(timeout=0.05)
            except queue.Empty:
                continue
            batch = [first]
            deadline = time.perf_counter() + self.batch_wait_s
            while len(batch) < self.max_batch_size:
                rem = deadline - time.perf_counter()
                if rem <= 0.0:
                    break
                try:
                    batch.append(self._queue.get(timeout=rem))
                except queue.Empty:
                    break
            prompts = [item[0] for item in batch]
            max_tok = max(item[1] for item in batch)
            try:
                replies = self._run_batch(prompts, max_tok)
                for (_, _, ev, box), text in zip(batch, replies):
                    box["text"] = text
                    ev.set()
            except Exception as exc:  # pragma: no cover
                for _, _, ev, box in batch:
                    box["error"] = f"{type(exc).__name__}: {exc}"
                    ev.set()

    def generate_chat(self, messages: List[Dict[str, str]], max_tokens: int = 96) -> str:
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        ev = threading.Event()
        box: Dict[str, Any] = {}
        self._queue.put((prompt, min(int(max_tokens), 96), ev, box))
        ev.wait()
        if "error" in box:
            raise RuntimeError(box["error"])
        return str(box.get("text", ""))

    def shutdown(self) -> None:
        self._stop.set()


class _ThreadedHTTPServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def start_openai_server(engine: Any, model_id: str, port: int) -> Tuple[_ThreadedHTTPServer, threading.Thread]:
    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:
            return None

        def do_GET(self) -> None:
            if self.path.rstrip("/") == "/v1/models":
                body = json.dumps({"object": "list", "data": [{"id": model_id, "object": "model"}]}).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            self.send_response(404)
            self.end_headers()

        def do_POST(self) -> None:
            if self.path.rstrip("/") != "/v1/chat/completions":
                self.send_response(404)
                self.end_headers()
                return
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            messages = payload.get("messages") or []
            max_tokens = int(payload.get("max_tokens") or 96)
            try:
                text = engine.generate_chat(messages, max_tokens=max_tokens)
                resp = {
                    "id": f"chatcmpl-{int(time.time() * 1000)}",
                    "object": "chat.completion",
                    "model": model_id,
                    "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
                }
                body = json.dumps(resp).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            except Exception as exc:
                err = json.dumps({"error": f"{type(exc).__name__}: {exc}"}).encode("utf-8")
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(err)))
                self.end_headers()
                self.wfile.write(err)

    srv = _ThreadedHTTPServer(("127.0.0.1", port), Handler)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    return srv, t


def run_self_test() -> Dict[str, Any]:
    """Verify OpenAI server compatibility with unmodified `LiveVLLMWorkerBackend` on CPU."""
    from chelated_pooled_agent_runtime import LiveVLLMWorkerBackend, build_24_matter_templates

    class _MockEngine:
        def generate_chat(self, messages: List[Dict[str, str]], max_tokens: int = 96) -> str:
            del messages, max_tokens
            return '{"field_value": "May 30, 2026", "cited_source_id": "M01_src_memo"}'

    srv, _ = start_openai_server(_MockEngine(), DEFAULT_MODEL, 0)
    port = srv.server_address[1]
    try:
        tmpl = build_24_matter_templates()[0]
        docs = tmpl.build_branch_events("A_derivative_only", include_withdrawal=False)
        backend = LiveVLLMWorkerBackend(vllm_url=f"http://127.0.0.1:{port}/v1", model=DEFAULT_MODEL, timeout_s=5.0)
        draft = backend.extract_field(template=tmpl, field_name="delivery_date", candidate_docs=[docs[3]])
        assert not draft.timed_out and draft.field_value == "May 30, 2026" and draft.cited_source_id == "M01_src_memo"
        return {"self_test": "PASS", "served_model": backend.served_model, "exec_ms": round(draft.exec_ms, 2)}
    finally:
        srv.shutdown()
        srv.server_close()


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--pool-sizes", default="4")
    ap.add_argument("--schedules", default="S1_steady")
    ap.add_argument("--output-dir", type=Path, default=Path("/content/out/pv7_w22b_b62_d13_colab_l4"))
    ap.add_argument("--git-commit-sha", type=str, default=os.environ.get("CHELATED_GIT_SHA", "unknown"))
    ap.add_argument("--source-archive-sha256", type=str, default=os.environ.get("CHELATED_SRC_TAR_SHA256", "unknown"))
    args = ap.parse_args(argv)

    if args.self_test:
        print(json.dumps(run_self_test(), indent=2, sort_keys=True))
        return 0

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required for Wave 22B live worker remediation")

    t0 = time.perf_counter()
    gpu_name = torch.cuda.get_device_name(0)
    vram_mib = round(float(torch.cuda.get_device_properties(0).total_memory / (1024.0 ** 2)), 2)
    hardware_cohort = f"colab:{gpu_name}"
    print(f"[w22b] hardware_cohort={hardware_cohort} vram_mib={vram_mib} torch={torch.__version__}", flush=True)

    engine = MicroBatchedCausalServer(args.model, device="cuda", max_batch_size=8, batch_wait_ms=3.0)
    srv, _ = start_openai_server(engine, args.model, args.port)
    base_url = f"http://127.0.0.1:{args.port}/v1"

    try:
        # Warmup: 2 rounds of 4 concurrent requests to warm CUDA allocator & kernels
        from concurrent.futures import ThreadPoolExecutor

        def _warmup_call(i: int) -> float:
            t_c = time.perf_counter()
            req_body = json.dumps(
                {
                    "model": args.model,
                    "temperature": 0.0,
                    "max_tokens": 64,
                    "messages": [
                        {
                            "role": "system",
                            "content": 'Return ONLY JSON with keys "field_value" and "cited_source_id".',
                        },
                        {
                            "role": "user",
                            "content": f"Visible: - [M01_src_memo]: delivery_date is May 30, 2026 (#{i}). Extract JSON.",
                        },
                    ],
                }
            ).encode("utf-8")
            req = urllib.request.Request(
                f"{base_url}/chat/completions",
                data=req_body,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=180.0) as resp:
                parsed_resp = json.loads(resp.read().decode("utf-8"))
            if i == 0:
                sample_txt = parsed_resp.get("choices", [{}])[0].get("message", {}).get("content", "")
                print(f"[w22b:warmup-sample] {sample_txt[:120]!r}", flush=True)
            return (time.perf_counter() - t_c) * 1000.0

        warmup_latencies: List[float] = []
        for r_idx in range(3):
            with ThreadPoolExecutor(max_workers=4) as pool:
                lats = list(pool.map(_warmup_call, range(4)))
            warmup_latencies.extend(lats)
            print(f"[w22b:warmup] round {r_idx + 1} 4-concurrent latencies_ms={[round(x, 1) for x in lats]}", flush=True)

        # Run the UNMODIFIED production runner `scripts/run_pv7_w20_d13_pooled_agent_loops.py`!
        args.output_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable,
            "-u",
            str(REPO_ROOT / "scripts" / "run_pv7_w20_d13_pooled_agent_loops.py"),
            "--mode",
            "run-live-workers",
            "--vllm-url",
            base_url,
            "--model",
            args.model,
            "--timeout-s",
            "15.0",
            "--pool-sizes",
            args.pool_sizes,
            "--schedules",
            args.schedules,
            "--output-dir",
            str(args.output_dir),
            "--json-summary",
        ]
        print(f"[w22b:exec] Running unmodified production runner: {' '.join(cmd)}", flush=True)
        proc = subprocess.Popen(
            cmd,
            cwd=str(REPO_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="", flush=True)
        runner_rc = proc.wait()
        wall_total = round(time.perf_counter() - t0, 2)

        manifest_path = args.output_dir / "pv7_w20_d13_manifest.json"
        receipt_path = args.output_dir / "report_receipt.json"
        manifest_obj = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}
        colab_provenance = {
            "schema_version": "colab_l4_w22b_b62_d13_provenance_v1",
            "shard_id": "w22b_b62_d13_live",
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "hardware_cohort": hardware_cohort,
            "is_local_dgx_spark_evidence": False,
            "blind_fixtures_touched": False,
            "git_commit_sha": args.git_commit_sha,
            "source_archive_sha256": args.source_archive_sha256,
            "spine_claim_registry_sha256": claim_registry_sha256(),
            "gpu_name": gpu_name,
            "gpu_total_vram_mib": vram_mib,
            "python_version": platform.python_version(),
            "torch_version": torch.__version__,
            "model_id": args.model,
            "warmup_4concurrent_latencies_ms": [round(x, 2) for x in warmup_latencies],
            "runner_exit_code": int(runner_rc),
            "overall_pass": bool(manifest_obj.get("overall_pass", False)),
            "status": manifest_obj.get("status"),
            "live_model_calls": int(manifest_obj.get("live_model_calls", 0)),
            "gates": manifest_obj.get("gates"),
            "manifest_sha256": sha256_file(manifest_path) if manifest_path.is_file() else "",
            "report_receipt_sha256": sha256_file(receipt_path) if receipt_path.is_file() else "",
            "wall_seconds_total": wall_total,
        }
        (args.output_dir / "colab_l4_provenance.json").write_bytes(
            (json.dumps(colab_provenance, indent=2, sort_keys=True) + "\n").encode("utf-8")
        )
        print(
            f"[w22b] DONE runner_rc={runner_rc} overall_pass={colab_provenance['overall_pass']} "
            f"live_calls={colab_provenance['live_model_calls']} wall={wall_total}s",
            flush=True,
        )
        # Always exit 0 after capturing authentic manifest + receipt so Colab downloads honest pass OR fail results!
        return 0
    finally:
        srv.shutdown()
        srv.server_close()
        engine.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
