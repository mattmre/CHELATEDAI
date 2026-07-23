"""H2 re-run v4 — runs ENTIRELY inside an isolated git worktree so a concurrent
session switching the main working tree's branch cannot corrupt the harness or the
output. Both datasets are run fresh on the worktree's H1-fixed + telemetry-fixed
harness; resilient (retry 3x + timeout + skip-if-exists)."""
import os
import sys
import json
import time
import subprocess
from pathlib import Path

WT = r"D:\GITHUB\CHELATEDAI\.claude\worktrees\h2-rerun"
sys.path.insert(0, WT)
os.chdir(WT)
from run_drift_recovery_swap_campaign import SwapCampaignConfig, run_swap_campaign  # noqa: E402

PYTHON = sys.executable
HARNESS = "run_drift_recovery_experiment.py"
SUB_ENV = dict(os.environ, HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1",
               PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
ATTEMPTS = 3
BACKOFF_S = 30
CELL_TIMEOUT_S = 1200


def subprocess_runner(run_config):
    out = Path(run_config.output)
    label = f"{run_config.condition} seed{run_config.seed} steps{run_config.correction_steps}"
    if out.exists():
        print(f"  [skip] {label} (exists)", flush=True)
        return json.loads(out.read_text(encoding="utf-8"))
    args = [
        PYTHON, HARNESS,
        "--task", run_config.task, "--condition", run_config.condition,
        "--drift", run_config.drift, "--fraction", str(run_config.fraction),
        "--angle", str(run_config.angle), "--sigma", str(run_config.sigma),
        "--cycles", str(run_config.cycles), "--seed", str(run_config.seed),
        "--max-queries", str(run_config.max_queries), "--sample-docs", str(run_config.sample_docs),
        "--model", run_config.model, "--swap-model", run_config.swap_model,
        "--anchor-fraction", str(run_config.anchor_fraction),
        "--correction-steps", str(run_config.correction_steps),
        "--correction-lr", str(run_config.correction_lr),
        "--bound-epsilon", str(run_config.bound_epsilon),
        "--trigger-threshold", str(run_config.trigger_threshold),
        "--max-temperature", str(run_config.max_temperature),
        "--epochs-scale", str(run_config.epochs_scale),
        "--output", str(run_config.output),
    ]
    last = ""
    for attempt in range(1, ATTEMPTS + 1):
        print(f"  [run ] {label} (attempt {attempt}/{ATTEMPTS}) ...", flush=True)
        try:
            r = subprocess.run(args, env=SUB_ENV, capture_output=True, text=True, timeout=CELL_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            last = "timeout"; print(f"  [warn] {label} TIMEOUT", flush=True); time.sleep(BACKOFF_S); continue
        if r.returncode == 0 and out.exists():
            print(f"  [done] {label}", flush=True)
            return json.loads(out.read_text(encoding="utf-8"))
        last = f"rc={r.returncode}\n{(r.stderr or '')[-1500:]}"
        print(f"  [warn] {label} FAILED ({r.returncode}); backoff {BACKOFF_S}s", flush=True)
        time.sleep(BACKOFF_S)
    raise RuntimeError(f"cell FAILED after {ATTEMPTS}: {label}\n{last}")


print(f"=== H2 RE-RUN v4 (worktree-isolated: {WT}) ===", flush=True)
print("=== [1/2] SciFact ===", flush=True)
run_swap_campaign(SwapCampaignConfig(), runner=subprocess_runner)
print("=== [1/2] SciFact DONE ===", flush=True)
print("=== [2/2] NFCorpus ===", flush=True)
run_swap_campaign(
    SwapCampaignConfig(task="NFCorpus",
                       output_dir="experiment_runs/drift-recovery/swap-nfcorpus",
                       report_md="docs/drift-recovery-swap-nfcorpus-results-2026-06.md"),
    runner=subprocess_runner,
)
print("=== [2/2] NFCorpus DONE ===", flush=True)
print("=== H2 ALL DONE ===", flush=True)
