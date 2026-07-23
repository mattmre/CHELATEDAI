"""C2 follow-up: trivial-baseline and continuous-feature leave-one-regime-out analysis.

Grok's C2 adversarial review showed the shipped multi-feature alpha=10 calibrator
(a) loses to trivial baselines and (b) buries a live continuous signal
(``oracle_margin_mean``) that it never calibrates. This module recomputes those
comparisons directly from the frozen per-regime feature JSONs so the D3 report can
state the honest, machine-verified scope of the negative. No numbers are hand-entered.

Run: ``python -m research.drift_recovery.estimator.baseline_analysis``
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Mapping, Sequence, Tuple

import numpy as np

from research.drift_recovery.estimator.calibration import FEATURE_NAMES, fit_calibration

OUT_DIR = Path("research/drift_recovery/out/estimator")
FEATURE_FILES = (
    "scifact_minilm_to_mpnet_features.json",
    "nfcorpus_minilm_to_mpnet_features.json",
    "scifact_minilm_to_bge_large_features.json",
    "nfcorpus_minilm_to_bge_large_features.json",
)
BIN_LOW, BIN_HIGH = 1.0 / 3.0, 2.0 / 3.0


def _rankdata(values: np.ndarray) -> np.ndarray:
    """Average-rank of values (ties averaged), dependency-free."""
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = np.arange(1, len(values) + 1, dtype=np.float64)
    # average ties
    _, inv, counts = np.unique(values, return_inverse=True, return_counts=True)
    sums = np.zeros(len(counts))
    np.add.at(sums, inv, ranks)
    return (sums / counts)[inv]


def spearman(a: Sequence[float], b: Sequence[float]) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    ra, rb = _rankdata(a), _rankdata(b)
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    denom = np.sqrt((ra @ ra) * (rb @ rb))
    return float(ra @ rb / denom) if denom > 0 else float("nan")


def _bin(r: float) -> str:
    return "low" if r < BIN_LOW else ("medium" if r < BIN_HIGH else "high")


def _bin_acc(pred: Sequence[float], actual: Sequence[float]) -> float:
    return float(np.mean([_bin(p) == _bin(a) for p, a in zip(pred, actual)]))


def _mae(pred: Sequence[float], actual: Sequence[float]) -> float:
    return float(np.mean(np.abs(np.asarray(pred) - np.asarray(actual))))


def _load() -> Tuple[List[Mapping], np.ndarray, List[str]]:
    records, recoveries, names = [], [], []
    for fname in FEATURE_FILES:
        rec = json.loads((OUT_DIR / fname).read_text())
        records.append(rec)
        recoveries.append(float(rec["ground_truth"]["ridge_recovery"]))
        names.append(rec["regime_name"])
    return records, np.asarray(recoveries, dtype=np.float64), names


def _loo_scalar(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Leave-one-out univariate OLS predictions of y from single feature x."""
    preds = np.empty(len(y))
    for i in range(len(y)):
        tr = np.arange(len(y)) != i
        xt, yt = x[tr], y[tr]
        # closed-form OLS with intercept
        xm, ym = xt.mean(), yt.mean()
        var = float(((xt - xm) ** 2).sum())
        slope = float(((xt - xm) * (yt - ym)).sum() / var) if var > 1e-15 else 0.0
        preds[i] = ym + slope * (x[i] - xm)
    return preds


def _loo_constant(y: np.ndarray, const: float) -> np.ndarray:
    return np.full(len(y), const)


def _loo_mean(y: np.ndarray) -> np.ndarray:
    preds = np.empty(len(y))
    for i in range(len(y)):
        tr = np.arange(len(y)) != i
        preds[i] = y[tr].mean()
    return preds


def _loo_multifeature(records: List[Mapping], y: np.ndarray, names: List[str], reg: float) -> np.ndarray:
    preds = np.empty(len(y))
    for i in range(len(y)):
        tr = [j for j in range(len(y)) if j != i]
        cal = fit_calibration(
            [records[j] for j in tr], [y[j] for j in tr],
            regime_names=[names[j] for j in tr], regularization=reg,
        )
        preds[i] = cal.predict(records[i])
    return preds


def run() -> dict:
    records, y, names = _load()
    summ = [r["summary"] for r in records]
    floor = np.asarray([r["ground_truth"]["floor_ndcg"] for r in records])
    gap = np.asarray([r["ground_truth"]["oracle_gap"] for r in records])

    def row(label: str, preds: np.ndarray) -> dict:
        return {
            "predictor": label,
            "spearman": round(spearman(preds, y), 4),
            "mae": round(_mae(preds, y), 4),
            "bin_accuracy": round(_bin_acc(preds, y), 4),
            "predictions": [round(float(p), 4) for p in preds],
        }

    rows = [
        row("shipped multi-feature R_hat (alpha=10)", _loo_multifeature(records, y, names, 10.0)),
        row("multi-feature ablation (alpha=0.01)", _loo_multifeature(records, y, names, 0.01)),
        row("trivial: mean train R", _loo_mean(y)),
        row("trivial: constant 0.7", _loo_constant(y, 0.7)),
        row("trivial: floor NDCG -> R (OLS)", _loo_scalar(floor, y)),
        row("trivial: oracle gap -> R (OLS)", _loo_scalar(gap, y)),
        row("continuous: oracle_margin_mean (OLS)", _loo_scalar(np.asarray([s["oracle_margin_mean"] for s in summ]), y)),
        row("continuous: oracle_margin_q50 (OLS)", _loo_scalar(np.asarray([s["oracle_margin_q50"] for s in summ]), y)),
        row("continuous: oracle_margin_sign_rate (OLS)", _loo_scalar(np.asarray([s["oracle_margin_sign_rate"] for s in summ]), y)),
    ]

    # univariate Spearman of each continuous feature vs R (in-sample descriptive)
    univ = {
        name: round(spearman([s[name] for s in summ], y), 4)
        for name in ("oracle_margin_mean", "oracle_margin_q50", "oracle_margin_sign_rate",
                     "predicted_inversion_rate")
    }

    result = {
        "record_type": "d3_c2_baseline_analysis",
        "regimes": names,
        "actual_recovery": [round(float(v), 4) for v in y],
        "oracle_margin_mean_in_feature_names": "oracle_margin_mean" in FEATURE_NAMES,
        "univariate_spearman_vs_R": univ,
        "loo_rows": rows,
        "note": (
            "n=4 held-out regimes: Spearman flips on a single swap and cannot support a "
            "high-power claim in EITHER direction. The binary predicted_inversion_rate is "
            "constant (dead); the continuous oracle-margin structure ranks R on this n=4 set. "
            "The shipped alpha=10 map over-shrinks (8 features, 3 train points) toward the "
            "train mean and does not beat trivial baselines."
        ),
    }
    return result


if __name__ == "__main__":
    out = run()
    (OUT_DIR / "baseline_analysis.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
