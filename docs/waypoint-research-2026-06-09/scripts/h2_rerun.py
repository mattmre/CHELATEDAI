"""H2: re-run both swap campaigns on the H1-fixed harness, regenerating the
auto-generated result docs + manifests with correct (uncontaminated) C3a numbers."""
import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
import sys
sys.path.insert(0, r"D:\GITHUB\CHELATEDAI")
os.chdir(r"D:\GITHUB\CHELATEDAI")  # relative output paths (experiment_runs/, docs/) resolve here
from run_drift_recovery_swap_campaign import SwapCampaignConfig, run_swap_campaign
from run_drift_recovery_experiment import _detect_device

dev = _detect_device()
print(f"=== H2 RE-RUN START — device={dev} ===", flush=True)

print("=== [1/2] SciFact campaign ===", flush=True)
run_swap_campaign(SwapCampaignConfig(device=dev))  # defaults: SciFact, swap/, swap-results doc
print("=== [1/2] SciFact DONE ===", flush=True)

print("=== [2/2] NFCorpus campaign ===", flush=True)
run_swap_campaign(SwapCampaignConfig(
    task="NFCorpus",
    output_dir="experiment_runs/drift-recovery/swap-nfcorpus",
    report_md="docs/drift-recovery-swap-nfcorpus-results-2026-06.md",
    device=dev,
))
print("=== [2/2] NFCorpus DONE ===", flush=True)
print("=== H2 ALL DONE ===", flush=True)
