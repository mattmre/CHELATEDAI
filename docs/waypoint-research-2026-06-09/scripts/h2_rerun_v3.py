"""H2 re-run v3 (H1-correct, memory-safe, resilient).

Each cell is its own subprocess so RAM is fully released between the 44 runs
(the in-process driver accumulates MiniLM+mpnet and OOMs ~10 cells in). Adds:
  - retry-on-failure (3 attempts, 30s backoff) to survive transient external
    memory spikes from the machine's other overnight work;
  - a per-cell timeout (15 min) so a hung HF fetch can't wedge the campaign;
  - skip-if-exists so a crash/relaunch resumes where it stopped.

Artifacts were cleared manually before launch, so this is a fully fresh run.
Regenerates both swap manifests + both result docs with the H1 baseline fix.
"""
import os
import sys
import json
import time
import subprocess
from pathlib import Path

sys.path.insert(0, r"D:\GITHUB\CHELATEDAI")
os.chdir(r"D:\GITHUB\CHELATEDAI")
from run_drift_recovery_swap_campaign import SwapCampaignConfig, run_swap_campaign  # noqa: E402

PYTHON = sys.executable
HARNESS = "run_drift_recovery_experiment.py"
# Machine freed up (RAM ~23GB, GPU ~23GB free), so NFCorpus runs on GPU like SciFact
# — both datasets uniformly GPU. One cell at a time, modest footprint; the retry
# wrapper absorbs any transient crash if the operator's work resumes.
SUB_ENV = dict(os.environ, HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1",
               PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
ATTEMPTS = 3
BACKOFF_S = 30
CELL_TIMEOUT_S = 1800


def subprocess_runner(run_config):
    out = Path(run_config.output)
    label = f"{run_config.condition} seed{run_config.seed} steps{run_config.correction_steps}"
    if out.exists():
        print(f"  [skip] {label} (exists)", flush=True)
        return json.loads(out.read_text(encoding="utf-8"))
    args = [
        PYTHON, HARNESS,
        "--task", run_config.task,
        "--condition", run_config.condition,
        "--drift", run_config.drift,
        "--fraction", str(run_config.fraction),
        "--angle", str(run_config.angle),
        "--sigma", str(run_config.sigma),
        "--cycles", str(run_config.cycles),
        "--seed", str(run_config.seed),
        "--max-queries", str(run_config.max_queries),
        "--sample-docs", str(run_config.sample_docs),
        "--model", run_config.model,
        "--swap-model", run_config.swap_model,
        "--anchor-fraction", str(run_config.anchor_fraction),
        "--correction-steps", str(run_config.correction_steps),
        "--correction-lr", str(run_config.correction_lr),
        "--bound-epsilon", str(run_config.bound_epsilon),
        "--trigger-threshold", str(run_config.trigger_threshold),
        "--max-temperature", str(run_config.max_temperature),
        "--epochs-scale", str(run_config.epochs_scale),
        "--output", str(run_config.output),
    ]
    last_err = ""
    for attempt in range(1, ATTEMPTS + 1):
        tag = f"{label} (attempt {attempt}/{ATTEMPTS})"
        print(f"  [run ] {tag} ...", flush=True)
        try:
            r = subprocess.run(args, env=SUB_ENV, capture_output=True,
                               text=True, timeout=CELL_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            last_err = f"timeout after {CELL_TIMEOUT_S}s"
            print(f"  [warn] {tag} TIMEOUT", flush=True)
            time.sleep(BACKOFF_S)
            continue
        if r.returncode == 0 and out.exists():
            print(f"  [done] {label}", flush=True)
            return json.loads(out.read_text(encoding="utf-8"))
        last_err = f"rc={r.returncode}\nSTDERR tail:\n{(r.stderr or '')[-2000:]}"
        print(f"  [warn] {tag} FAILED ({r.returncode}); backing off {BACKOFF_S}s", flush=True)
        time.sleep(BACKOFF_S)
    raise RuntimeError(f"cell FAILED after {ATTEMPTS} attempts: {label}\n{last_err}")


print("=== H2 RE-RUN v3 START ===", flush=True)
print("=== [1/2] SciFact ===", flush=True)
run_swap_campaign(SwapCampaignConfig(device="cpu"), runner=subprocess_runner)
print("=== [1/2] SciFact DONE ===", flush=True)

print("=== [2/2] NFCorpus ===", flush=True)
run_swap_campaign(
    SwapCampaignConfig(
        task="NFCorpus",
        output_dir="experiment_runs/drift-recovery/swap-nfcorpus",
        report_md="docs/drift-recovery-swap-nfcorpus-results-2026-06.md",
        device="cpu",
    ),
    runner=subprocess_runner,
)
print("=== [2/2] NFCorpus DONE ===", flush=True)
print("=== H2 ALL DONE ===", flush=True)
