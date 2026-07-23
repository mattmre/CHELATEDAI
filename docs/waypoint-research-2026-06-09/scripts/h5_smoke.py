"""VERY SMALL validation smoke (operator-allowed) — verify the H5 head-to-head driver
runs END-TO-END on the real swap models. NOT the campaign: 1 seed, 15 docs, 2 cycles,
4 conditions. The headline numbers come from the full GPU campaign, not this."""
import json
import os
import sys

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

WT = r"D:\GITHUB\CHELATEDAI\.claude\worktrees\h5-fitness"
sys.path.insert(0, WT)
os.chdir(WT)

import run_drift_recovery_swap_campaign as camp  # noqa: E402

camp.SEEDS = (42,)  # one seed for the smoke (validate the pipeline, not the campaign)
from run_drift_recovery_swap_campaign import SwapCampaignConfig, run_condition_head_to_head  # noqa: E402

OUT = r"C:\Users\mattm\AppData\Local\Temp\claude\D--GITHUB-CHELATEDAI--claude-worktrees-relaxed-wozniak-271e04\00b9ffaa-9115-4aad-9219-bcc10f287c90\scratchpad\h5_smoke_out"
cfg = SwapCampaignConfig(
    task="SciFact", max_queries=6, sample_docs=15, cycles=2,
    output_dir=OUT, report_md=OUT + r"\report.md",
)
print("=== running tiny H5 head-to-head smoke (C0/C5/C5s/C5r, 1 seed, 15 docs) ===", flush=True)
manifest = run_condition_head_to_head(cfg, conditions=("C0", "C5", "C5s", "C5r"))
print("=== SMOKE RESULT ===", flush=True)
print("verdict:", json.dumps(manifest["verdict"]), flush=True)
for row in manifest["summary"]:
    print(f"  {row['condition']}: final_mean={row['final_mean']:.4f} "
          f"baseline={row['baseline_mean']:.4f} applied={row['applied_runs']}/{row['run_count']}", flush=True)
print("DRIVER_END_TO_END: OK", flush=True)
