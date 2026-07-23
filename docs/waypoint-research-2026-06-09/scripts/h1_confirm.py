"""H1 runtime confirm: after the build-time-adapter fix, all swap conditions
share one per-seed baseline on real NFCorpus (the C3a divergence is gone)."""
import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
import sys
sys.path.insert(0, r"D:\GITHUB\CHELATEDAI")
import tempfile
import pathlib
from run_drift_recovery_experiment import DriftRecoveryConfig, run_experiment, _detect_device

dev = _detect_device()
out = pathlib.Path(tempfile.mkdtemp())
print("device:", dev)


def cfg(cond):
    return DriftRecoveryConfig(
        task="NFCorpus", condition=cond, drift="query_encoder_swap",
        fraction=0.5, angle=25.0, sigma=0.05, cycles=1, seed=42,
        max_queries=100, sample_docs=1200,
        model="sentence-transformers/all-MiniLM-L6-v2",
        swap_model="all-mpnet-base-v2",
        output=str(out / f"{cond}.json"),
        device=dev, anchor_fraction=0.4,
    )


res = {}
for cond in ["C0", "C2", "C2O", "C3a", "C4a"]:
    r = run_experiment(cfg(cond))
    res[cond] = r["baseline"]["ndcg_at_10"]
    print(f"{cond:4s} baseline_ndcg = {res[cond]!r}")

base = res["C0"]
all_equal = all(res[c] == base for c in res)
print("ALL CONDITIONS SHARE BASELINE:", all_equal)
print("C3a - C0 =", res["C3a"] - base)
