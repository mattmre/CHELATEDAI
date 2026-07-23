"""Full-scale GPU campaigns (operator: GPU free). Sequential, resilient, real numbers.
Order: H5 head-to-head (load-bearing) first, then H3 C3b, then H2 NFCorpus re-run.
Each campaign is independent (continue-on-error) and skips if its manifest already exists."""
import json
import os
import sys
import time
import traceback

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

WT = r"D:\GITHUB\CHELATEDAI\.claude\worktrees\gpu-campaigns"
sys.path.insert(0, WT)
os.chdir(WT)

from run_drift_recovery_swap_campaign import (  # noqa: E402
    SwapCampaignConfig,
    run_condition_head_to_head,
    run_swap_campaign,
)

OUT = os.path.join(WT, "experiment_runs", "fullscale-2026-06-29")
os.makedirs(OUT, exist_ok=True)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _h2h(task, sub, conditions=None):
    cfg = SwapCampaignConfig(
        task=task,
        output_dir=os.path.join(OUT, sub),
        report_md=os.path.join(OUT, sub + "-report.md"),
    )
    manifest = os.path.join(OUT, sub, "post-bank-headtohead-manifest-2026-06.json")
    if os.path.exists(manifest):
        log(f"    (skip {sub}: manifest exists)")
        return json.load(open(manifest, encoding="utf-8"))
    if conditions is None:
        return run_condition_head_to_head(cfg)
    return run_condition_head_to_head(cfg, conditions=conditions)


def _h2(task, sub):
    cfg = SwapCampaignConfig(
        task=task,
        output_dir=os.path.join(OUT, sub),
        report_md=os.path.join(OUT, sub + "-report.md"),
    )
    manifest = os.path.join(OUT, sub, "swap-campaign-manifest-2026-06.json")
    if os.path.exists(manifest):
        log(f"    (skip {sub}: manifest exists)")
        return json.load(open(manifest, encoding="utf-8"))
    return run_swap_campaign(cfg)


CAMPAIGNS = [
    ("H5-headtohead-SciFact", lambda: _h2h("SciFact", "h5-scifact")),
    ("H5-headtohead-NFCorpus", lambda: _h2h("NFCorpus", "h5-nfcorpus")),
    ("H3-C3b-SciFact", lambda: _h2h("SciFact", "c3b-scifact", ("C0", "C2O", "C3a", "C3b"))),
    ("H3-C3b-NFCorpus", lambda: _h2h("NFCorpus", "c3b-nfcorpus", ("C0", "C2O", "C3a", "C3b"))),
    ("H2-NFCorpus-rerun", lambda: _h2("NFCorpus", "h2-nfcorpus")),
]

log("=== FULL-SCALE GPU CAMPAIGNS START ===")
log(f"out dir: {OUT}")
for name, fn in CAMPAIGNS:
    try:
        log(f">>> {name} starting")
        t0 = time.time()
        m = fn()
        dt = time.time() - t0
        log(f"<<< {name} DONE in {dt:.0f}s")
        if isinstance(m, dict):
            if "verdict" in m:
                log(f"    VERDICT: {json.dumps(m['verdict'])}")
            for key in ("summary", "main_summary"):
                if key in m:
                    for row in m[key]:
                        log(f"    {row['condition']}: final_mean={row.get('final_mean', float('nan')):.4f} "
                            f"baseline={row.get('baseline_mean', float('nan')):.4f} "
                            f"applied={row.get('applied_runs', '?')}/{row.get('run_count', '?')}")
    except Exception as e:
        log(f"!!! {name} FAILED: {e}")
        traceback.print_exc()
log("=== FULL-SCALE GPU CAMPAIGNS COMPLETE ===")
