"""Remaining full-scale campaigns (H5 already done) with SUBPROCESS-PER-CAMPAIGN isolation
so GPU memory is released between campaigns (the v1 run crashed ~30 cells in from accumulation).
Orchestrator spawns a fresh `python fullscale_v2.py <name>` per campaign."""
import os
import subprocess
import sys
import time

SELF = os.path.abspath(__file__)
WT = r"D:\GITHUB\CHELATEDAI\.claude\worktrees\gpu-campaigns"
OUT = os.path.join(WT, "experiment_runs", "fullscale-2026-06-29")

# (name, kind, task, sub, conditions)
SPECS = [
    ("H3-C3b-SciFact", "h2h", "SciFact", "c3b-scifact", ["C0", "C2O", "C3a", "C3b"]),
    ("H3-C3b-NFCorpus", "h2h", "NFCorpus", "c3b-nfcorpus", ["C0", "C2O", "C3a", "C3b"]),
    ("H2-NFCorpus-rerun", "h2", "NFCorpus", "h2-nfcorpus", None),
]


def _manifest(kind, sub):
    fn = "post-bank-headtohead-manifest-2026-06.json" if kind == "h2h" else "swap-campaign-manifest-2026-06.json"
    return os.path.join(OUT, sub, fn)


def run_one(name, kind, task, sub, conditions):
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    sys.path.insert(0, WT)
    os.chdir(WT)
    from run_drift_recovery_swap_campaign import (
        SwapCampaignConfig, run_condition_head_to_head, run_swap_campaign,
    )
    cfg = SwapCampaignConfig(
        task=task, output_dir=os.path.join(OUT, sub),
        report_md=os.path.join(OUT, sub + "-report.md"),
    )
    if kind == "h2h":
        run_condition_head_to_head(cfg, conditions=tuple(conditions))
    else:
        run_swap_campaign(cfg)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        spec = next(s for s in SPECS if s[0] == sys.argv[1])
        run_one(*spec)
    else:
        print(f"[{time.strftime('%H:%M:%S')}] === REMAINING CAMPAIGNS (subprocess-isolated) ===", flush=True)
        for spec in SPECS:
            name, kind, _, sub, _ = spec
            if os.path.exists(_manifest(kind, sub)):
                print(f"[{time.strftime('%H:%M:%S')}] [skip] {name} (manifest exists)", flush=True)
                continue
            print(f"[{time.strftime('%H:%M:%S')}] >>> {name} (fresh subprocess)", flush=True)
            t0 = time.time()
            r = subprocess.run([sys.executable, SELF, name], cwd=WT)
            print(f"[{time.strftime('%H:%M:%S')}] <<< {name} exit={r.returncode} in {time.time()-t0:.0f}s", flush=True)
        print(f"[{time.strftime('%H:%M:%S')}] === REMAINING CAMPAIGNS COMPLETE ===", flush=True)
