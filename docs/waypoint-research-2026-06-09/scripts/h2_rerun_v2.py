"""H2 re-run, memory-safe: each cell runs as its own subprocess so RAM is
released between the 44 runs (the in-process driver accumulates MiniLM+mpnet
copies and OOMs ~10 cells in). Restartable: completed cells are skipped.

One-time clean (guarded by a marker) removes stale pre-H1 (#274/#275) per-run
files so skip-if-exists never reuses contaminated results; restarts resume."""
import os
import sys
import json
import subprocess
from pathlib import Path

sys.path.insert(0, r"D:\GITHUB\CHELATEDAI")
os.chdir(r"D:\GITHUB\CHELATEDAI")
from run_drift_recovery_swap_campaign import SwapCampaignConfig, run_swap_campaign  # noqa: E402

PYTHON = sys.executable
HARNESS = "run_drift_recovery_experiment.py"
SUB_ENV = dict(os.environ, HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1",
               PYTHONUTF8="1", PYTHONIOENCODING="utf-8")

BASES = ["experiment_runs/drift-recovery/swap",
         "experiment_runs/drift-recovery/swap-nfcorpus"]
MARKER = Path("experiment_runs/drift-recovery/.h2_rerun_v1_started")


def one_time_clean():
    if MARKER.exists():
        print("=== resume mode (marker present) — keeping completed cells ===", flush=True)
        return
    removed = 0
    for base in BASES:
        for sub in ("main", "budget", "budget-confirm"):
            d = Path(base) / sub
            if d.exists():
                for f in d.glob("*.json"):
                    f.unlink()
                    removed += 1
        m = Path(base) / "swap-campaign-manifest-2026-06.json"
        if m.exists():
            m.unlink()
            removed += 1
    MARKER.parent.mkdir(parents=True, exist_ok=True)
    MARKER.write_text("started", encoding="utf-8")
    print(f"=== one-time clean removed {removed} stale files ===", flush=True)


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
    print(f"  [run ] {label} ...", flush=True)
    r = subprocess.run(args, env=SUB_ENV, capture_output=True, text=True)
    if r.returncode != 0 or not out.exists():
        raise RuntimeError(
            f"cell FAILED: {label} rc={r.returncode}\nSTDERR tail:\n{(r.stderr or '')[-3000:]}"
        )
    print(f"  [done] {label}", flush=True)
    return json.loads(out.read_text(encoding="utf-8"))


print("=== H2 SUBPROCESS RE-RUN START ===", flush=True)
one_time_clean()

print("=== [1/2] SciFact ===", flush=True)
run_swap_campaign(SwapCampaignConfig(device="cuda"), runner=subprocess_runner)
print("=== [1/2] SciFact DONE ===", flush=True)

print("=== [2/2] NFCorpus ===", flush=True)
run_swap_campaign(
    SwapCampaignConfig(
        task="NFCorpus",
        output_dir="experiment_runs/drift-recovery/swap-nfcorpus",
        report_md="docs/drift-recovery-swap-nfcorpus-results-2026-06.md",
        device="cuda",
    ),
    runner=subprocess_runner,
)
print("=== [2/2] NFCorpus DONE ===", flush=True)
print("=== H2 ALL DONE ===", flush=True)
