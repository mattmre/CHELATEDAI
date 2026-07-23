"""Pre-registered Obj A scale validation for ``oracle_margin_mean``.

This module deliberately has one primary feature and two frozen nulls. It
verifies the pre-registration hashes before loading or building packs, builds
new regimes sequentially, and evaluates whole-dataset and whole-encoder-family
leave-one-block-out predictions.

Run::

    python -m research.drift_recovery.estimator.objA --device cuda
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from research.drift_recovery.artifacts import EmbeddingPack  # noqa: E402
from research.drift_recovery.estimator.baseline_analysis import spearman  # noqa: E402
from research.drift_recovery.estimator.features import (  # noqa: E402
    extract_regime_features,
    leakage_safe_fit_indices,
)
from research.drift_recovery.harness_bridge import assert_harness_parity  # noqa: E402
from research.drift_recovery.run_estimator import (  # noqa: E402
    RegimeSpec,
    RegimeUnavailableError,
    build_regime_pack,
    frozen_ridge_ground_truth,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
ESTIMATOR_DIR = Path(__file__).resolve().parent
OUT_DIR = REPO_ROOT / "research" / "drift_recovery" / "out" / "estimator"
PREREG_PATH = ESTIMATOR_DIR / "prereg_objA.json"
PREREG_HASH_PATH = ESTIMATOR_DIR / "prereg_objA.sha256"
PRIMARY_FEATURE_NAMES = ("oracle_margin_mean",)
NULL_NAMES = ("oracle_gap_ols", "mean_R")


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def verify_frozen_preregistration() -> Dict[str, str]:
    """Fail closed if either frozen preregistration file has changed."""

    if not PREREG_HASH_PATH.exists():
        raise RuntimeError(f"missing preregistration hash lock: {PREREG_HASH_PATH}")
    expected: Dict[str, str] = {}
    for raw_line in PREREG_HASH_PATH.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        digest, filename = line.split(maxsplit=1)
        expected[filename.strip()] = digest.lower()
    required = ("prereg_objA.md", "prereg_objA.json")
    if set(expected) != set(required):
        raise RuntimeError(f"preregistration hash lock must contain exactly {required}")
    actual = {name: _sha256(ESTIMATOR_DIR / name) for name in required}
    mismatches = {
        name: {"expected": expected[name], "actual": actual[name]}
        for name in required
        if actual[name] != expected[name]
    }
    if mismatches:
        raise RuntimeError(f"frozen preregistration hash mismatch: {mismatches}")
    prereg = json.loads(PREREG_PATH.read_text(encoding="utf-8"))
    if prereg.get("primary_predictor", {}).get("allowed_features") != list(
        PRIMARY_FEATURE_NAMES
    ):
        raise RuntimeError("machine preregistration no longer locks the one primary feature")
    return actual


def load_preregistration() -> Dict[str, Any]:
    verify_frozen_preregistration()
    return json.loads(PREREG_PATH.read_text(encoding="utf-8"))


def _resolve_repo_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def _encoder_family(model: str) -> str:
    lowered = str(model).lower()
    if "mpnet" in lowered:
        return "mpnet"
    if "bge-large" in lowered:
        return "bge-large"
    raise ValueError(f"unregistered encoder family for model {model}")


def _release_gpu_memory() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        return


def _ols_predict(
    train_x: Sequence[float],
    train_y: Sequence[float],
    test_x: Sequence[float],
) -> np.ndarray:
    """Univariate OLS with intercept; no clipping and no feature fallback."""

    x = np.asarray(train_x, dtype=np.float64)
    y = np.asarray(train_y, dtype=np.float64)
    test = np.asarray(test_x, dtype=np.float64)
    if x.ndim != 1 or y.ndim != 1 or test.ndim != 1 or len(x) != len(y):
        raise ValueError("univariate OLS expects one-dimensional aligned arrays")
    if len(x) < 2 or not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("univariate OLS needs at least two finite training rows")
    x_mean = float(np.mean(x))
    y_mean = float(np.mean(y))
    centered = x - x_mean
    variance = float(centered @ centered)
    slope = float(centered @ (y - y_mean) / variance) if variance > 1e-15 else 0.0
    return y_mean + slope * (test - x_mean)


def _spearman_or_none(a: Sequence[float], b: Sequence[float]) -> Any:
    value = float(spearman(a, b))
    return value if math.isfinite(value) else None


def _pearson_or_none(a: Sequence[float], b: Sequence[float]) -> Any:
    left = np.asarray(a, dtype=np.float64)
    right = np.asarray(b, dtype=np.float64)
    if left.shape != right.shape or left.ndim != 1 or len(left) == 0:
        raise ValueError("Pearson correlation expects non-empty aligned vectors")
    left = left - float(np.mean(left))
    right = right - float(np.mean(right))
    denominator = float(np.sqrt((left @ left) * (right @ right)))
    return float(left @ right / denominator) if denominator > 1e-15 else None


def _metrics(predictions: Sequence[float], actual: Sequence[float]) -> Dict[str, Any]:
    pred = np.asarray(predictions, dtype=np.float64)
    truth = np.asarray(actual, dtype=np.float64)
    if pred.shape != truth.shape or pred.ndim != 1 or len(pred) == 0:
        raise ValueError("metric arrays must be non-empty aligned vectors")
    return {
        "spearman": _spearman_or_none(pred, truth),
        "mae": float(np.mean(np.abs(pred - truth))),
        "cell_n": int(len(truth)),
    }


def _strictly_beats(candidate: Mapping[str, Any], null: Mapping[str, Any]) -> bool:
    candidate_rho = candidate.get("spearman")
    null_rho = null.get("spearman")
    return bool(
        candidate_rho is not None
        and null_rho is not None
        and float(candidate_rho) > float(null_rho)
        and float(candidate["mae"]) < float(null["mae"])
    )


def block_loo_analysis(
    records: Sequence[Mapping[str, Any]],
    block_field: str,
) -> Dict[str, Any]:
    """Run the frozen same-fold primary and null predictions by whole block."""

    if block_field not in {"dataset", "encoder_family"}:
        raise ValueError("block_field must be dataset or encoder_family")
    if not records:
        raise ValueError("block-LOO requires records")
    names = np.asarray([str(record["name"]) for record in records], dtype=str)
    blocks = np.asarray([str(record[block_field]) for record in records], dtype=str)
    margin = np.asarray(
        [float(record[PRIMARY_FEATURE_NAMES[0]]) for record in records], dtype=np.float64
    )
    gap = np.asarray([float(record["oracle_gap"]) for record in records], dtype=np.float64)
    recovery = np.asarray([float(record["R"]) for record in records], dtype=np.float64)
    unique_blocks = sorted(set(blocks.tolist()))
    if len(unique_blocks) < 2:
        raise ValueError(f"{block_field} block-LOO requires at least two blocks")

    model_predictions = {
        "oracle_margin_mean": np.empty(len(records), dtype=np.float64),
        "oracle_gap_ols": np.empty(len(records), dtype=np.float64),
        "mean_R": np.empty(len(records), dtype=np.float64),
    }
    rows: List[Dict[str, Any]] = []
    for held_out_block in unique_blocks:
        held_mask = blocks == held_out_block
        train_mask = ~held_mask
        if not np.any(held_mask) or np.count_nonzero(train_mask) < 2:
            raise ValueError(f"insufficient rows for held-out {block_field}={held_out_block}")
        primary_fold = _ols_predict(margin[train_mask], recovery[train_mask], margin[held_mask])
        gap_fold = _ols_predict(gap[train_mask], recovery[train_mask], gap[held_mask])
        mean_fold = np.full(np.count_nonzero(held_mask), float(np.mean(recovery[train_mask])))
        model_predictions["oracle_margin_mean"][held_mask] = primary_fold
        model_predictions["oracle_gap_ols"][held_mask] = gap_fold
        model_predictions["mean_R"][held_mask] = mean_fold

        held_indices = np.flatnonzero(held_mask)
        training_regimes = names[train_mask].tolist()
        training_blocks = sorted(set(blocks[train_mask].tolist()))
        for fold_row, index in enumerate(held_indices):
            rows.append(
                {
                    "regime": str(names[index]),
                    "held_out_block": held_out_block,
                    "actual_R": float(recovery[index]),
                    "predictions": {
                        "oracle_margin_mean": float(primary_fold[fold_row]),
                        "oracle_gap_ols": float(gap_fold[fold_row]),
                        "mean_R": float(mean_fold[fold_row]),
                    },
                    "training_regimes": training_regimes,
                    "training_blocks": training_blocks,
                }
            )

    metrics = {
        name: _metrics(predictions, recovery)
        for name, predictions in model_predictions.items()
    }
    comparisons = {
        null_name: {
            "strictly_better_spearman": bool(
                metrics["oracle_margin_mean"]["spearman"] is not None
                and metrics[null_name]["spearman"] is not None
                and metrics["oracle_margin_mean"]["spearman"]
                > metrics[null_name]["spearman"]
            ),
            "strictly_lower_mae": bool(
                metrics["oracle_margin_mean"]["mae"] < metrics[null_name]["mae"]
            ),
            "beats_on_both": _strictly_beats(metrics["oracle_margin_mean"], metrics[null_name]),
        }
        for null_name in NULL_NAMES
    }
    return {
        "block_field": block_field,
        "held_out_unit_count": len(unique_blocks),
        "held_out_units": unique_blocks,
        "cell_n": len(records),
        "metrics": metrics,
        "comparisons": comparisons,
        "primary_beats_both_nulls": all(
            item["beats_on_both"] for item in comparisons.values()
        ),
        "rows": sorted(rows, key=lambda row: row["regime"]),
    }


def _average_ranks(values: Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    order = np.argsort(array, kind="mergesort")
    ordinal = np.empty(len(array), dtype=np.float64)
    ordinal[order] = np.arange(1, len(array) + 1, dtype=np.float64)
    unique, inverse, counts = np.unique(array, return_inverse=True, return_counts=True)
    del unique
    sums = np.zeros(len(counts), dtype=np.float64)
    np.add.at(sums, inverse, ordinal)
    return (sums / counts)[inverse]


def partial_spearman_controlling_gap(records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Locked rank-residual partial correlation and raw rank descriptives."""

    margin = np.asarray([float(row["oracle_margin_mean"]) for row in records])
    recovery = np.asarray([float(row["R"]) for row in records])
    gap = np.asarray([float(row["oracle_gap"]) for row in records])
    ranked_margin = _average_ranks(margin)
    ranked_recovery = _average_ranks(recovery)
    ranked_gap = _average_ranks(gap)
    design = np.column_stack([np.ones(len(records), dtype=np.float64), ranked_gap])
    margin_residual = ranked_margin - design @ np.linalg.lstsq(
        design, ranked_margin, rcond=None
    )[0]
    recovery_residual = ranked_recovery - design @ np.linalg.lstsq(
        design, ranked_recovery, rcond=None
    )[0]
    partial = _pearson_or_none(margin_residual, recovery_residual)
    raw_margin = _spearman_or_none(margin, recovery)
    raw_gap = _spearman_or_none(gap, recovery)
    raw_delta = None if raw_margin is None or raw_gap is None else raw_margin - raw_gap
    return {
        "definition": (
            "Pearson correlation of ranked-margin and ranked-R OLS residuals after "
            "separate adjustment for intercept plus ranked oracle_gap."
        ),
        "partial_spearman_margin_R_controlling_oracle_gap": partial,
        "raw_spearman_margin_R": raw_margin,
        "raw_spearman_gap_R": raw_gap,
        "raw_spearman_margin_minus_gap": raw_delta,
        "independent_pair_block_n": len(records),
    }


def _validate_registered_regimes(prereg: Mapping[str, Any]) -> List[Mapping[str, Any]]:
    regimes = list(prereg["frozen_regimes"])
    new_names = [row["name"] for row in regimes if row["source"] == "build_new"]
    required_new = {
        "fiqa2018_minilm_to_mpnet_af040",
        "fiqa2018_minilm_to_bge_large_af040",
    }
    if len(regimes) != 6 or set(new_names) != required_new:
        raise RuntimeError("frozen Obj A scope must contain six pairs and exactly two FiQA builds")
    pairs = {(row["dataset"], row["encoder_family"]) for row in regimes}
    if len(pairs) != len(regimes):
        raise RuntimeError("frozen Obj A scope contains a dataset/encoder pseudo-replicate")
    if any(float(row["anchor_fraction"]) != 0.4 for row in regimes):
        raise RuntimeError("frozen Obj A scope contains a non-0.40 anchor variant")
    return regimes


def _existing_feature_audit(name: str, record: Mapping[str, Any]) -> Dict[str, Any]:
    feature_path = OUT_DIR / f"{name}_features.json"
    if not feature_path.exists():
        return {"status": "not_present"}
    prior = json.loads(feature_path.read_text(encoding="utf-8"))
    comparisons = {
        "oracle_margin_mean": abs(
            float(prior["summary"]["oracle_margin_mean"])
            - float(record["oracle_margin_mean"])
        ),
        "oracle_gap": abs(
            float(prior["ground_truth"]["oracle_gap"]) - float(record["oracle_gap"])
        ),
        "R": abs(float(prior["ground_truth"]["ridge_recovery"]) - float(record["R"])),
    }
    if max(comparisons.values()) > 1e-12:
        raise AssertionError(f"existing feature artifact drift for {name}: {comparisons}")
    return {"status": "match", "max_abs_difference": max(comparisons.values())}


def collect_regimes(
    prereg: Mapping[str, Any],
    device: str,
    rebuild_new: bool = False,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Load four frozen packs and build at most the two registered FiQA packs."""

    records: List[Dict[str, Any]] = []
    inventory: List[Dict[str, Any]] = []
    for registered in _validate_registered_regimes(prereg):
        prefix = _resolve_repo_path(str(registered["pack_prefix"]))
        source = str(registered["source"])
        name = str(registered["name"])
        spec = RegimeSpec(
            name=name,
            dataset=str(registered["dataset"]),
            swap_model=str(registered["swap_model"]),
            anchor_fraction=float(registered["anchor_fraction"]),
            seed=42,
        )
        try:
            if source == "reuse_existing":
                if not prefix.with_suffix(".npz").exists():
                    raise FileNotFoundError(f"registered existing pack is missing: {prefix}")
                pack = EmbeddingPack.load(prefix)
                build_action = "reused_existing"
            elif source == "build_new":
                if rebuild_new or not prefix.with_suffix(".npz").exists():
                    print(f"[Obj A] building {name} sequentially", flush=True)
                    pack = build_regime_pack(spec, prefix, device=device)
                    build_action = "built_new"
                else:
                    print(f"[Obj A] reusing already-frozen {name}", flush=True)
                    pack = EmbeddingPack.load(prefix)
                    build_action = "reused_new"
            else:
                raise RuntimeError(f"unknown registered source {source}")

            actual_dataset = str(pack.metadata.get("dataset"))
            actual_swap = str(pack.metadata.get("models", {}).get("swap"))
            if actual_dataset != registered["dataset"]:
                raise AssertionError(
                    f"{name} dataset mismatch: {actual_dataset} != {registered['dataset']}"
                )
            if _encoder_family(actual_swap) != registered["encoder_family"]:
                raise AssertionError(f"{name} encoder-family mismatch for {actual_swap}")
            parity = assert_harness_parity(pack, atol=1e-12)
            fit_idx = leakage_safe_fit_indices(pack)
            features = extract_regime_features(pack)
            ground_truth = frozen_ridge_ground_truth(pack)
            record = {
                "name": name,
                "dataset": str(registered["dataset"]),
                "encoder_family": str(registered["encoder_family"]),
                "swap_model": str(registered["swap_model"]),
                "anchor_fraction": float(registered["anchor_fraction"]),
                "oracle_margin_mean": float(features["summary"]["oracle_margin_mean"]),
                "oracle_gap": float(ground_truth["oracle_gap"]),
                "floor_ndcg": float(ground_truth["floor_ndcg"]),
                "oracle_ndcg": float(ground_truth["oracle_ndcg"]),
                "ridge_ndcg": float(ground_truth["ridge_ndcg"]),
                "R": float(ground_truth["ridge_recovery"]),
            }
            feature_audit = (
                _existing_feature_audit(name, record)
                if source == "reuse_existing"
                else {"status": "new"}
            )
            if source == "build_new":
                feature_payload = {
                    **features,
                    "regime_name": name,
                    "ground_truth": ground_truth,
                    "pack_prefix": str(prefix.relative_to(REPO_ROOT)).replace("\\", "/"),
                    "harness_parity": parity,
                }
                _write_json(OUT_DIR / f"{name}_features.json", feature_payload)
            records.append(record)
            inventory.append(
                {
                    **record,
                    "status": "available",
                    "source": source,
                    "build_action": build_action,
                    "pack_prefix": str(prefix.relative_to(REPO_ROOT)).replace("\\", "/"),
                    "pack_sha256": _sha256(prefix.with_suffix(".npz")),
                    "harness_parity": parity,
                    "harness_parity_max_abs": max(parity.values()),
                    "leakage_safe_fit_count": int(len(fit_idx)),
                    "eval_positive_docs_in_fit": 0,
                    "existing_feature_audit": feature_audit,
                }
            )
        except RegimeUnavailableError as exc:
            inventory.append(
                {
                    "name": name,
                    "dataset": str(registered["dataset"]),
                    "encoder_family": str(registered["encoder_family"]),
                    "status": "skipped_unavailable",
                    "reason": f"{type(exc).__name__}: {exc}",
                }
            )
        except RuntimeError as exc:
            if source == "build_new" and "oracle gap is non-positive" in str(exc):
                inventory.append(
                    {
                        "name": name,
                        "dataset": str(registered["dataset"]),
                        "encoder_family": str(registered["encoder_family"]),
                        "status": "skipped_degenerate_gap",
                        "reason": f"{type(exc).__name__}: {exc}",
                    }
                )
            else:
                raise
        finally:
            _release_gpu_memory()
    return records, inventory


def _fiqa_holdout(dataset_analysis: Mapping[str, Any]) -> Dict[str, Any]:
    rows = [
        row
        for row in dataset_analysis["rows"]
        if str(row["held_out_block"]) == "FiQA2018"
    ]
    if not rows:
        return {"status": "unavailable", "rows": []}
    actual = [row["actual_R"] for row in rows]
    metrics = {
        model: _metrics([row["predictions"][model] for row in rows], actual)
        for model in ("oracle_margin_mean", "oracle_gap_ols", "mean_R")
    }
    return {
        "status": "available",
        "held_out_training_regimes": rows[0]["training_regimes"],
        "rows": rows,
        "metrics_within_two_fiqa_cells": metrics,
        "margin_beats_gap_on_both_within_fiqa": _strictly_beats(
            metrics["oracle_margin_mean"], metrics["oracle_gap_ols"]
        ),
    }


def analyze_objA(
    records: Sequence[Mapping[str, Any]],
    inventory: Sequence[Mapping[str, Any]],
    prereg_hashes: Mapping[str, str],
) -> Dict[str, Any]:
    pair_block_n = len({(row["dataset"], row["encoder_family"]) for row in records})
    cell_n = len(records)
    counts = {
        "registered_cell_n": 6,
        "available_cell_n": cell_n,
        "cell_n": cell_n,
        "independent_pair_block_n": pair_block_n,
        "dataset_holdout_unit_count": len({row["dataset"] for row in records}),
        "encoder_family_holdout_unit_count": len(
            {row["encoder_family"] for row in records}
        ),
        "pseudo_replicate_cell_n": cell_n - pair_block_n,
    }
    if cell_n < 2:
        return {
            "record_type": "drift_recovery_objA_validation",
            "schema_version": 1,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "preregistration": {
                "id": "objA-oracle-margin-block-loo-v1",
                "hashes": dict(prereg_hashes),
            },
            "counts": counts,
            "regimes": list(records),
            "inventory": list(inventory),
            "success_bar_met": False,
            "verdict": "UNDERPOWERED-NEGATIVE",
            "reason": "Fewer than two registered independent pairs were available.",
        }

    dataset_analysis = block_loo_analysis(records, "dataset")
    encoder_analysis = block_loo_analysis(records, "encoder_family")
    success_bar_met = bool(
        dataset_analysis["primary_beats_both_nulls"]
        and encoder_analysis["primary_beats_both_nulls"]
    )
    gap_failure = bool(
        not dataset_analysis["comparisons"]["oracle_gap_ols"]["beats_on_both"]
        or not encoder_analysis["comparisons"]["oracle_gap_ols"]["beats_on_both"]
    )
    if pair_block_n < 6:
        verdict = "UNDERPOWERED-NEGATIVE"
    elif success_bar_met:
        verdict = "PROMISING-BUT-UNDERPOWERED"
    else:
        verdict = "NEGATIVE"
    return {
        "record_type": "drift_recovery_objA_validation",
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "preregistration": {
            "id": "objA-oracle-margin-block-loo-v1",
            "hashes": dict(prereg_hashes),
            "primary_feature_names": list(PRIMARY_FEATURE_NAMES),
            "null_names": list(NULL_NAMES),
        },
        "offline_environment": {
            "HF_HUB_OFFLINE": os.environ.get("HF_HUB_OFFLINE"),
            "HF_DATASETS_OFFLINE": os.environ.get("HF_DATASETS_OFFLINE"),
            "TRANSFORMERS_OFFLINE": os.environ.get("TRANSFORMERS_OFFLINE"),
            "execution": "sequential",
        },
        "counts": counts,
        "regimes": list(records),
        "inventory": list(inventory),
        "partial_spearman": partial_spearman_controlling_gap(records),
        "block_loo": {
            "dataset": dataset_analysis,
            "encoder_family": encoder_analysis,
        },
        "fiqa_held_out": _fiqa_holdout(dataset_analysis),
        "success_bar_met": success_bar_met,
        "gap_only_failure_under_either_scheme": gap_failure,
        "validated_positive": False,
        "verdict": verdict,
        "verdict_policy_note": (
            "Even a met AND-bar is only PROMISING-BUT-UNDERPOWERED because the offline "
            "scope has three dataset and two encoder-family holdout units."
        ),
    }


def _format_metric(value: Any) -> str:
    return "undefined" if value is None else f"{float(value):.4f}"


def render_validation(result: Mapping[str, Any]) -> str:
    lines = [
        "# Obj A validation — pre-registered block-LOO",
        "",
        f"**Verdict: {result['verdict']}**. Success bar met: "
        f"**{str(result['success_bar_met']).upper()}**. This result is not labeled validated.",
        "",
        "## Achieved scale",
        "",
        f"- Independent dataset×encoder-family block-n: **{result['counts']['independent_pair_block_n']}**",
        f"- Cell-n: **{result['counts']['cell_n']}**",
        f"- Whole-dataset holdout units: **{result['counts']['dataset_holdout_unit_count']}**",
        f"- Whole-encoder-family holdout units: **{result['counts']['encoder_family_holdout_unit_count']}**",
        f"- Pseudo-replicate cells: **{result['counts']['pseudo_replicate_cell_n']}**",
        "",
        "## Frozen block-LOO results",
        "",
        "| Scheme | Predictor | Spearman | MAE | Cell-n |",
        "|---|---|---:|---:|---:|",
    ]
    for scheme in ("dataset", "encoder_family"):
        analysis = result["block_loo"][scheme]
        for predictor in ("oracle_margin_mean", "oracle_gap_ols", "mean_R"):
            metric = analysis["metrics"][predictor]
            lines.append(
                f"| {scheme} | `{predictor}` | {_format_metric(metric['spearman'])} | "
                f"{metric['mae']:.4f} | {metric['cell_n']} |"
            )
    lines.extend(["", "Strict comparisons (both greater Spearman and lower MAE required):", ""])
    for scheme in ("dataset", "encoder_family"):
        comparisons = result["block_loo"][scheme]["comparisons"]
        for null_name in NULL_NAMES:
            outcome = comparisons[null_name]
            lines.append(
                f"- {scheme}: margin vs `{null_name}` — **"
                f"{'PASS' if outcome['beats_on_both'] else 'FAIL'}** "
                f"(Spearman: {outcome['strictly_better_spearman']}; "
                f"MAE: {outcome['strictly_lower_mae']})."
            )
    partial = result["partial_spearman"]
    lines.extend(
        [
            "",
            "## Increment over oracle gap",
            "",
            f"- Partial Spearman(margin, R | gap): **{_format_metric(partial['partial_spearman_margin_R_controlling_oracle_gap'])}**",
            f"- Raw Spearman(margin, R): **{_format_metric(partial['raw_spearman_margin_R'])}**",
            f"- Raw Spearman(gap, R): **{_format_metric(partial['raw_spearman_gap_R'])}**",
            f"- Raw margin-minus-gap Spearman: **{_format_metric(partial['raw_spearman_margin_minus_gap'])}**",
            "",
            "## Per-regime values",
            "",
            "| Regime | Dataset | Encoder | Margin mean | Floor | Oracle gap | R |",
            "|---|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in result["regimes"]:
        lines.append(
            f"| `{row['name']}` | {row['dataset']} | {row['encoder_family']} | "
            f"{row['oracle_margin_mean']:.6f} | {row['floor_ndcg']:.4f} | "
            f"{row['oracle_gap']:.4f} | {row['R']:.4f} |"
        )
    lines.extend(["", "## FiQA held out", ""])
    fiqa = result["fiqa_held_out"]
    if fiqa["status"] != "available":
        lines.append("FiQA was unavailable or had a degenerate non-positive oracle gap.")
    else:
        for row in fiqa["rows"]:
            predictions = row["predictions"]
            lines.append(
                f"- `{row['regime']}` actual R={row['actual_R']:.4f}; "
                f"margin={predictions['oracle_margin_mean']:.4f}, "
                f"gap-only={predictions['oracle_gap_ols']:.4f}, "
                f"mean-R={predictions['mean_R']:.4f}."
            )
        fiqa_metrics = fiqa["metrics_within_two_fiqa_cells"]
        lines.append(
            "- FiQA-only MAE: margin "
            f"{fiqa_metrics['oracle_margin_mean']['mae']:.4f}, gap-only "
            f"{fiqa_metrics['oracle_gap_ols']['mae']:.4f}, mean-R "
            f"{fiqa_metrics['mean_R']['mae']:.4f}."
        )
    lines.extend(["", "## Availability and audit", ""])
    for item in result["inventory"]:
        if item["status"] == "available":
            lines.append(
                f"- `{item['name']}`: available; action={item['build_action']}; "
                f"parity max={item['harness_parity_max_abs']:.3g}; safe fit n="
                f"{item['leakage_safe_fit_count']}; eval-positive fit docs=0."
            )
        else:
            lines.append(f"- `{item['name']}`: {item['status']} — {item['reason']}")
    return "\n".join(lines) + "\n"


def render_report(result: Mapping[str, Any]) -> str:
    dataset = result["block_loo"]["dataset"]["metrics"]
    encoder = result["block_loo"]["encoder_family"]["metrics"]
    partial = result["partial_spearman"]
    fiqa = result["fiqa_held_out"]
    if fiqa["status"] == "available":
        fiqa_metrics = fiqa["metrics_within_two_fiqa_cells"]
        fiqa_sentence = (
            "With the whole FiQA dataset held out, margin MAE was "
            f"{fiqa_metrics['oracle_margin_mean']['mae']:.3f} versus "
            f"{fiqa_metrics['oracle_gap_ols']['mae']:.3f} for gap-only."
        )
    else:
        fiqa_sentence = "FiQA could not enter the primary analysis; its skip reason is in the validation artifact."
    beats_gap_dataset = result["block_loo"]["dataset"]["comparisons"]["oracle_gap_ols"]
    beats_gap_encoder = result["block_loo"]["encoder_family"]["comparisons"]["oracle_gap_ols"]
    honesty = (
        "It beat oracle-gap-alone under both schemes on both locked metrics."
        if beats_gap_dataset["beats_on_both"] and beats_gap_encoder["beats_on_both"]
        else "It did not beat oracle-gap-alone under both schemes on both locked metrics; the lead does not survive the registered robustness bar."
    )
    return "\n".join(
        [
            "# Obj A report — honest scale-up of `oracle_margin_mean`",
            "",
            "## Built",
            "",
            "Reused the four frozen SciFact/NFCorpus packs and built only the two registered "
            "FiQA2018 packs (mpnet and bge-large, anchor 0.40), sequentially and offline. "
            "Every available pack passed merged-harness NDCG parity at 1e-12 and the feature "
            "fit used a leakage-safe document index.",
            "",
            "## Scale and result",
            "",
            f"Achieved independent pair block-n **{result['counts']['independent_pair_block_n']}** "
            f"and cell-n **{result['counts']['cell_n']}**, spanning "
            f"{result['counts']['dataset_holdout_unit_count']} whole-dataset holdouts and "
            f"{result['counts']['encoder_family_holdout_unit_count']} whole-encoder-family holdouts. "
            f"Verdict: **{result['verdict']}**; the frozen AND-bar was "
            f"{'met' if result['success_bar_met'] else 'not met'}.",
            "",
            f"Dataset-block LOO: margin Spearman {_format_metric(dataset['oracle_margin_mean']['spearman'])}, "
            f"MAE {dataset['oracle_margin_mean']['mae']:.3f}; gap-only Spearman "
            f"{_format_metric(dataset['oracle_gap_ols']['spearman'])}, MAE "
            f"{dataset['oracle_gap_ols']['mae']:.3f}. Encoder-block LOO: margin Spearman "
            f"{_format_metric(encoder['oracle_margin_mean']['spearman'])}, MAE "
            f"{encoder['oracle_margin_mean']['mae']:.3f}; gap-only Spearman "
            f"{_format_metric(encoder['oracle_gap_ols']['spearman'])}, MAE "
            f"{encoder['oracle_gap_ols']['mae']:.3f}.",
            "",
            f"Partial Spearman(margin, R | oracle gap) was "
            f"**{_format_metric(partial['partial_spearman_margin_R_controlling_oracle_gap'])}**. "
            f"{fiqa_sentence}",
            "",
            "## Brutal conclusion",
            "",
            honesty
            + (
                " However, inside the two-cell FiQA holdout alone, margin tied gap-only on "
                "Spearman rather than strictly beating it, although its MAE was lower."
                if fiqa["status"] == "available"
                and not fiqa["margin_beats_gap_on_both_within_fiqa"]
                else ""
            )
            + " The n=4 +1.0 ordering is descriptive history, not validation. Even a passing "
            "six-pair point estimate would remain underpowered because the offline cache supplies "
            "only three dataset and two encoder-family holdout units.",
            "",
        ]
    )


def run_objA(device: str = "cuda", rebuild_new: bool = False) -> Dict[str, Any]:
    prereg_hashes = verify_frozen_preregistration()
    prereg = json.loads(PREREG_PATH.read_text(encoding="utf-8"))
    records, inventory = collect_regimes(prereg, device=device, rebuild_new=rebuild_new)
    result = analyze_objA(records, inventory, prereg_hashes)
    _write_json(OUT_DIR / "objA_validation.json", result)
    (OUT_DIR / "objA_validation.md").write_text(render_validation(result), encoding="utf-8")
    (OUT_DIR / "objA_REPORT.md").write_text(render_report(result), encoding="utf-8")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run pre-registered Obj A block-LOO validation")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--rebuild-new", action="store_true")
    return parser


def main(argv: Any = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_objA(device=args.device, rebuild_new=args.rebuild_new)
    print(
        json.dumps(
            {
                "verdict": result["verdict"],
                "success_bar_met": result["success_bar_met"],
                "counts": result["counts"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
