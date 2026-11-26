#!/usr/bin/env python3
"""Colab L4 Wave 24-05 (Lane 8): Bundle #49 Live 3-Family 7B-Class Open-Weight Coder Cold-Start Benchmark (36 Live Sessions).

Tests the EXACT unmodified `chelated-kernel` wheel and sanitized public `packages/chelated_kernel/README.md`
against 3 genuinely unrelated, ungated 4B-8B coding/instruct model families:
  1. `Qwen/Qwen2.5-Coder-7B-Instruct` (Alibaba Qwen2.5-Coder 7B)
  2. `microsoft/Phi-4-mini-instruct` (Microsoft Phi-4-mini 3.8B)
  3. `deepseek-ai/deepseek-coder-6.7b-instruct` (DeepSeek Coder 6.7B)

All 36 sessions (3 families x 4 verticals x 3 seeds) execute inside an isolated subprocess with ONLY
the freshly built wheel on PYTHONPATH, enforcing Bundle #49 pass criteria:
  >= 30/36 overall, >= 8/12 per family, 0 private imports, 100% unsupported-capability canary blocking,
  and 0 cross-tenant leaks.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from chelated_pv7_spine.claim_registry import claim_registry_sha256  # noqa: E402
from scripts.run_colab_l4_w22d_b49_sdk_3family_live import (  # noqa: E402
    run_benchmark_with_generator,
    run_self_test,
    sha256_bytes,
)

LIVE_FAMILIES_7B: Tuple[Tuple[str, str], ...] = (
    ("qwen2.5-coder-7b", "Qwen/Qwen2.5-Coder-7B-Instruct"),
    ("phi-4-mini", "microsoft/Phi-4-mini-instruct"),
    ("deepseek-coder-6.7b", "deepseek-ai/deepseek-coder-6.7b-instruct"),
)

SYSTEM_PROMPT = (
    "You are a precise Python SDK engineer. Output ONLY one ```python ... ``` block "
    "implementing run_adapter(ctx) strictly from the provided documentation, returning "
    "every requested dictionary key including 'canary_blocked'."
)


def _format_chat(tok: Any, full_prompt: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": full_prompt},
    ]
    try:
        return str(tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
    except Exception:
        # Some coder templates (e.g., older DeepSeek) only accept user/assistant roles
        merged = [{"role": "user", "content": f"{SYSTEM_PROMPT}\n\n{full_prompt}"}]
        return str(tok.apply_chat_template(merged, tokenize=False, add_generation_prompt=True))


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--output-dir", type=Path, default=Path("/content/out/pv7_w24_05_b49_sdk_7b_coders"))
    ap.add_argument("--git-commit-sha", type=str, default=os.environ.get("CHELATED_GIT_SHA", "unknown"))
    ap.add_argument("--source-archive-sha256", type=str, default=os.environ.get("CHELATED_SRC_TAR_SHA256", "unknown"))
    args = ap.parse_args(argv)

    if args.self_test:
        print(json.dumps(run_self_test(), indent=2, sort_keys=True))
        return 0

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required for Wave 24-05 live 3-family 7B coder cold-start benchmark")

    t0 = time.perf_counter()
    gpu_name = torch.cuda.get_device_name(0)
    vram_mib = round(float(torch.cuda.get_device_properties(0).total_memory / (1024.0 ** 2)), 2)
    hardware_cohort = f"colab:{gpu_name}"
    print(f"[w24-05] hardware_cohort={hardware_cohort} vram_mib={vram_mib} torch={torch.__version__}", flush=True)

    loaded_cache: Dict[str, Any] = {"model_id": None, "model": None, "tok": None}

    def _live_generate(fam: str, model_id: str, full_prompt: str, seed: int) -> str:
        if loaded_cache["model_id"] != model_id:
            if loaded_cache["model"] is not None:
                del loaded_cache["model"], loaded_cache["tok"]
                loaded_cache["model"] = None
                loaded_cache["tok"] = None
                gc.collect()
                torch.cuda.empty_cache()
            print(f"[w24-05] Loading live family {fam} ({model_id}) in bfloat16 ...", flush=True)
            tok = AutoTokenizer.from_pretrained(model_id)
            mdl = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="cuda")
            mdl.eval()
            loaded_cache.update({"model_id": model_id, "model": mdl, "tok": tok})

        tok = loaded_cache["tok"]
        mdl = loaded_cache["model"]
        chat_text = _format_chat(tok, full_prompt)
        inputs = tok(chat_text, return_tensors="pt").to("cuda")
        torch.manual_seed(24050 + seed)
        eos_id = tok.eos_token_id
        if isinstance(eos_id, list):
            eos_id = eos_id[0]
        gen_kwargs: Dict[str, Any] = {
            "max_new_tokens": 512,
            "pad_token_id": eos_id,
        }
        if seed == 0:
            gen_kwargs["do_sample"] = False
        else:
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = 0.15
            gen_kwargs["top_p"] = 0.9
        with torch.no_grad():
            out_ids = mdl.generate(**inputs, **gen_kwargs)
        new_ids = out_ids[0][inputs["input_ids"].shape[1] :]
        return str(tok.decode(new_ids, skip_special_tokens=True))

    with tempfile.TemporaryDirectory(prefix="w24-05-wheel-") as td:
        bench = run_benchmark_with_generator(
            repo_root=REPO_ROOT,
            work_dir=Path(td),
            families=LIVE_FAMILIES_7B,
            generate_fn=_live_generate,
        )

    if loaded_cache["model"] is not None:
        del loaded_cache["model"], loaded_cache["tok"]
        gc.collect()
        torch.cuda.empty_cache()

    wall_total = round(time.perf_counter() - t0, 2)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": "colab_l4_w24_05_b49_sdk_7b_coders_v1",
        "wave_item_id": "W24-05",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "provenance": {
            "git_commit_sha": args.git_commit_sha,
            "source_archive_sha256": args.source_archive_sha256,
            "spine_claim_registry_sha256": claim_registry_sha256(),
            "hardware_cohort": hardware_cohort,
            "is_local_dgx_spark_evidence": False,
            "blind_fixtures_touched": False,
            "gpu_name": gpu_name,
            "gpu_total_vram_mib": vram_mib,
            "python_version": platform.python_version(),
            "torch_version": torch.__version__,
            "wall_seconds_total": wall_total,
            "families": [{"name": k, "model_id": v} for k, v in LIVE_FAMILIES_7B],
        },
        "benchmark": bench,
    }
    m_path = args.output_dir / "w24_05_b49_sdk_7b_coders_manifest.json"
    m_bytes = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")
    m_path.write_bytes(m_bytes)
    m_sha = sha256_bytes(m_bytes)

    receipt = {
        "schema_version": "colab_l4_burst_receipt_v1",
        "shard_id": "w24_05_b49_sdk_7b_coders",
        "wave_item_id": "W24-05",
        "hardware_cohort": hardware_cohort,
        "git_commit_sha": args.git_commit_sha,
        "source_archive_sha256": args.source_archive_sha256,
        "manifest_file": m_path.name,
        "manifest_sha256": m_sha,
        "exit_code": 0,
        "wall_seconds_total": wall_total,
    }
    (args.output_dir / "w24_05_b49_sdk_7b_coders_receipt.json").write_bytes(
        (json.dumps(receipt, indent=2, sort_keys=True) + "\n").encode("utf-8")
    )
    print(
        f"[w24-05] DONE sha256={m_sha[:12]} passed={bench['sessions_passed']}/36 "
        f"per_family={bench['per_family_passed']} gate_pass={bench['gate_pass']} wall={wall_total}s",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
