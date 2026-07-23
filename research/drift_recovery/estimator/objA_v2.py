"""Powered 4-dataset x 3-encoder Obj A v2 validation.

This additive runner leaves Obj A v1 byte-for-byte untouched. It verifies the
v2 preregistration lock before it can build a pack, reuses the six v1 packs,
builds only the six registered additions, and applies the frozen v1 AND-bar.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from research.drift_recovery.artifacts import EmbeddingPack  # noqa: E402
from research.drift_recovery.estimator.features import (  # noqa: E402
    extract_regime_features,
    leakage_safe_fit_indices,
)
from research.drift_recovery.estimator.objA import (  # noqa: E402
    NULL_NAMES,
    PRIMARY_FEATURE_NAMES,
    _metrics,
    _strictly_beats,
    block_loo_analysis,
    partial_spearman_controlling_gap,
)
from research.drift_recovery.harness_bridge import assert_harness_parity  # noqa: E402
from research.drift_recovery.run_estimator import (  # noqa: E402
    E5_MODEL,
    RegimeSpec,
    RegimeUnavailableError,
    build_regime_pack,
    frozen_ridge_ground_truth,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
ESTIMATOR_DIR = Path(__file__).resolve().parent
OUT_DIR = REPO_ROOT / "research" / "drift_recovery" / "out" / "estimator"
PREREG_PATH = ESTIMATOR_DIR / "prereg_objA_v2.json"
PREREG_HASH_PATH = ESTIMATOR_DIR / "prereg_objA_v2.sha256"
V2_OUTPUT_NAMES = (
    "objA_v2_validation.json",
    "objA_v2_validation.md",
    "objA_v2_REPORT.md",
)
EXPECTED_DATASETS = ("SciFact", "NFCorpus", "FiQA2018", "ArguAna")
EXPECTED_ENCODERS = ("mpnet", "bge-large", "e5-base-v2")
EXPECTED_MODELS = {
    "mpnet": "all-mpnet-base-v2",
    "bge-large": "BAAI/bge-large-en-v1.5",
    "e5-base-v2": E5_MODEL,
}
EXPECTED_NEW_NAMES = {
    "scifact_minilm_to_e5_base_v2_af040",
    "nfcorpus_minilm_to_e5_base_v2_af040",
    "fiqa2018_minilm_to_e5_base_v2_af040",
    "arguana_minilm_to_mpnet_af040",
    "arguana_minilm_to_bge_large_af040",
    "arguana_minilm_to_e5_base_v2_af040",
}
ALLOWED_VERDICTS = ("POSITIVE", "PROMISING-BUT-UNDERPOWERED", "NEGATIVE")


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def verify_frozen_preregistration_v2() -> Dict[str, str]:
    """Fail closed unless the two v2 preregistration files match their lock."""

    if not PREREG_HASH_PATH.exists():
        raise RuntimeError(f"missing v2 preregistration hash lock: {PREREG_HASH_PATH}")
    expected: Dict[str, str] = {}
    for raw_line in PREREG_HASH_PATH.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        digest, filename = line.split(maxsplit=1)
        expected[filename.strip()] = digest.lower()
    required = ("prereg_objA_v2.md", "prereg_objA_v2.json")
    if set(expected) != set(required):
        raise RuntimeError(f"v2 preregistration hash lock must contain exactly {required}")
    actual = {name: _sha256(ESTIMATOR_DIR / name) for name in required}
    mismatches = {
        name: {"expected": expected[name], "actual": actual[name]}
        for name in required
        if actual[name] != expected[name]
    }
    if mismatches:
        raise RuntimeError(f"frozen v2 preregistration hash mismatch: {mismatches}")
    prereg = json.loads(PREREG_PATH.read_text(encoding="utf-8"))
    if prereg.get("primary_predictor", {}).get("allowed_features") != list(
        PRIMARY_FEATURE_NAMES
    ):
        raise RuntimeError("v2 preregistration no longer locks the v1 primary feature")
    if prereg.get("verdict_policy", {}).get("allowed_labels") != list(ALLOWED_VERDICTS):
        raise RuntimeError("v2 preregistration verdict labels drifted")
    return actual


def load_preregistration_v2() -> Dict[str, Any]:
    verify_frozen_preregistration_v2()
    return json.loads(PREREG_PATH.read_text(encoding="utf-8"))


def _resolve_repo_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def _release_gpu_memory() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def validate_registered_regimes_v2(
    prereg: Mapping[str, Any],
) -> List[Mapping[str, Any]]:
    regimes = list(prereg["frozen_regimes"])
    if len(regimes) != 12:
        raise RuntimeError("frozen Obj A v2 scope must contain exactly 12 cells")
    pairs = {(row["dataset"], row["encoder_family"]) for row in regimes}
    expected_pairs = {
        (dataset, encoder)
        for dataset in EXPECTED_DATASETS
        for encoder in EXPECTED_ENCODERS
    }
    if pairs != expected_pairs:
        raise RuntimeError("frozen Obj A v2 scope is not the registered 4x3 matrix")
    new_names = {row["name"] for row in regimes if row["source"] == "build_new"}
    if new_names != EXPECTED_NEW_NAMES or sum(
        row["source"] == "reuse_existing" for row in regimes
    ) != 6:
        raise RuntimeError("v2 must reuse six packs and build exactly the registered six")
    for row in regimes:
        if float(row["anchor_fraction"]) != 0.4 or int(row["seed"]) != 42:
            raise RuntimeError("every v2 regime must use anchor fraction 0.40 and seed 42")
        if row["swap_model"] != EXPECTED_MODELS[row["encoder_family"]]:
            raise RuntimeError(f"unexpected model/family pairing in {row['name']}")
    return regimes


def _audit_pack_contract(
    pack: EmbeddingPack,
    registered: Mapping[str, Any],
    source: str,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Assert the common recipe and strict new-pack evidence contract."""

    metadata = pack.metadata
    actual_dataset = str(metadata.get("dataset"))
    actual_swap = str(metadata.get("models", {}).get("swap"))
    if actual_dataset != registered["dataset"]:
        raise AssertionError(
            f"{registered['name']} dataset mismatch: {actual_dataset} != {registered['dataset']}"
        )
    if actual_swap != registered["swap_model"]:
        raise AssertionError(
            f"{registered['name']} model mismatch: {actual_swap} != {registered['swap_model']}"
        )
    if int(metadata.get("seed", -1)) != 42:
        raise AssertionError(f"{registered['name']} seed is not 42")
    if float(metadata.get("anchor_fraction", -1.0)) != 0.4:
        raise AssertionError(f"{registered['name']} anchor fraction is not 0.40")
    if source == "build_new":
        if int(metadata.get("max_queries", -1)) != 100 or int(
            metadata.get("sample_docs", -1)
        ) != 1200:
            raise AssertionError(f"{registered['name']} did not use the frozen slice recipe")
        required_scores = {"floor", "oracle", "ridge"}
        if not required_scores.issubset(pack.per_query_scores):
            raise AssertionError(f"{registered['name']} lacks frozen per-query scores")
        for score_name in required_scores:
            scores = np.asarray(pack.per_query_scores[score_name], dtype=np.float64)
            if scores.shape != (len(pack.query_ids),) or not np.all(np.isfinite(scores)):
                raise AssertionError(
                    f"{registered['name']} has invalid frozen {score_name} scores"
                )
        safe_extra = np.asarray(
            pack.extra_arrays.get("leakage_safe_fit_idx", []), dtype=np.int64
        )
        if not np.array_equal(safe_extra, np.asarray(pack.fit_idx, dtype=np.int64)):
            raise AssertionError(
                f"{registered['name']} fit_idx differs from leakage_safe_fit_idx"
            )
        fit_contract = metadata.get("fit_contract", {})
        if fit_contract.get("eval_qrels_used_by_optimizer") is not False or int(
            fit_contract.get("eval_positive_docs_in_fit", -1)
        ) != 0:
            raise AssertionError(f"{registered['name']} fit contract is not leakage-safe")
    fit_idx = leakage_safe_fit_indices(pack)
    parity = assert_harness_parity(pack, atol=1e-12)
    if max(parity.values()) > 1e-12:
        raise AssertionError(f"{registered['name']} harness parity exceeded 1e-12")
    return fit_idx, parity


def _ordering_audit(regimes: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    prereg_files = [
        ESTIMATOR_DIR / "prereg_objA_v2.md",
        ESTIMATOR_DIR / "prereg_objA_v2.json",
        PREREG_HASH_PATH,
    ]
    prereg_mtimes = {
        path.name: datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat()
        for path in prereg_files
    }
    pack_files: List[Path] = []
    for row in regimes:
        if row["source"] != "build_new":
            continue
        prefix = _resolve_repo_path(str(row["pack_prefix"]))
        for suffix in (".npz", ".json"):
            candidate = prefix.with_suffix(suffix)
            if candidate.exists():
                pack_files.append(candidate)
    pack_mtimes = {
        str(path.relative_to(REPO_ROOT)).replace("\\", "/"): datetime.fromtimestamp(
            path.stat().st_mtime, timezone.utc
        ).isoformat()
        for path in pack_files
    }
    latest_prereg = max(path.stat().st_mtime for path in prereg_files)
    earliest_pack = min((path.stat().st_mtime for path in pack_files), default=None)
    return {
        "preregistration_file_mtimes_utc": prereg_mtimes,
        "new_pack_file_mtimes_utc": pack_mtimes,
        "checked_new_pack_file_count": len(pack_files),
        "preregistration_precedes_every_new_pack": bool(
            earliest_pack is None or latest_prereg <= earliest_pack
        ),
        "note": "Filesystem ordering is execution evidence for this worktree; hashes are the content freeze.",
    }


def collect_regimes_v2(
    prereg: Mapping[str, Any],
    device: str,
    rebuild_new: bool = False,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Reuse six frozen packs and build at most the six registered v2 packs."""

    records: List[Dict[str, Any]] = []
    inventory: List[Dict[str, Any]] = []
    for registered in validate_registered_regimes_v2(prereg):
        prefix = _resolve_repo_path(str(registered["pack_prefix"]))
        source = str(registered["source"])
        name = str(registered["name"])
        spec = RegimeSpec(
            name=name,
            dataset=str(registered["dataset"]),
            swap_model=str(registered["swap_model"]),
            anchor_fraction=float(registered["anchor_fraction"]),
            seed=int(registered["seed"]),
        )
        try:
            if source == "reuse_existing":
                if not prefix.with_suffix(".npz").exists():
                    raise FileNotFoundError(f"registered existing pack is missing: {prefix}")
                pack = EmbeddingPack.load(prefix)
                build_action = "reused_existing"
            elif source == "build_new":
                if rebuild_new or not prefix.with_suffix(".npz").exists():
                    print(f"[Obj A v2] building {name} sequentially", flush=True)
                    pack = build_regime_pack(spec, prefix, device=device)
                    build_action = "built_new"
                else:
                    print(f"[Obj A v2] reusing already-frozen {name}", flush=True)
                    pack = EmbeddingPack.load(prefix)
                    build_action = "reused_new"
            else:
                raise RuntimeError(f"unknown registered source {source}")

            fit_idx, parity = _audit_pack_contract(pack, registered, source)
            features = extract_regime_features(pack)
            ground_truth = frozen_ridge_ground_truth(pack)
            record = {
                "name": name,
                "dataset": str(registered["dataset"]),
                "encoder_family": str(registered["encoder_family"]),
                "swap_model": str(registered["swap_model"]),
                "anchor_fraction": float(registered["anchor_fraction"]),
                "seed": int(registered["seed"]),
                "oracle_margin_mean": float(features["summary"]["oracle_margin_mean"]),
                "oracle_gap": float(ground_truth["oracle_gap"]),
                "floor_ndcg": float(ground_truth["floor_ndcg"]),
                "oracle_ndcg": float(ground_truth["oracle_ndcg"]),
                "ridge_ndcg": float(ground_truth["ridge_ndcg"]),
                "R": float(ground_truth["ridge_recovery"]),
            }
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
                    "document_count": int(len(pack.doc_ids)),
                    "query_count": int(len(pack.query_ids)),
                    "frozen_per_query_score_names": sorted(pack.per_query_scores),
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


def _try_block_analysis(
    records: Sequence[Mapping[str, Any]], block_field: str
) -> Optional[Dict[str, Any]]:
    if len({str(row[block_field]) for row in records}) < 2:
        return None
    return block_loo_analysis(records, block_field)


def held_out_block_check(
    analysis: Optional[Mapping[str, Any]], held_out_block: str
) -> Dict[str, Any]:
    """Evaluate an already-out-of-fold novel block without a second refit."""

    if analysis is None:
        return {"status": "unavailable", "held_out_block": held_out_block, "rows": []}
    rows = [
        row
        for row in analysis["rows"]
        if str(row["held_out_block"]) == str(held_out_block)
    ]
    if not rows:
        return {"status": "unavailable", "held_out_block": held_out_block, "rows": []}
    actual = [row["actual_R"] for row in rows]
    metrics = {
        model: _metrics([row["predictions"][model] for row in rows], actual)
        for model in ("oracle_margin_mean", "oracle_gap_ols", "mean_R")
    }
    margin_rho = metrics["oracle_margin_mean"]["spearman"]
    gap_rho = metrics["oracle_gap_ols"]["spearman"]
    if margin_rho is None or gap_rho is None:
        relation = "undefined"
    elif margin_rho > gap_rho:
        relation = "beats"
    elif margin_rho == gap_rho:
        relation = "ties"
    else:
        relation = "loses"
    comparison = {
        "strictly_better_spearman": bool(
            margin_rho is not None and gap_rho is not None and margin_rho > gap_rho
        ),
        "strictly_lower_mae": bool(
            metrics["oracle_margin_mean"]["mae"] < metrics["oracle_gap_ols"]["mae"]
        ),
        "beats_on_both": _strictly_beats(
            metrics["oracle_margin_mean"], metrics["oracle_gap_ols"]
        ),
        "spearman_relation": relation,
    }
    return {
        "status": "available",
        "held_out_block": held_out_block,
        "cell_n": len(rows),
        "rows": rows,
        "metrics": metrics,
        "margin_vs_gap": comparison,
    }


def verdict_for_v2(
    success_bar_met: bool, dataset_block_count: int, encoder_block_count: int
) -> str:
    if not success_bar_met:
        return "NEGATIVE"
    if dataset_block_count >= 4 and encoder_block_count >= 3:
        return "POSITIVE"
    return "PROMISING-BUT-UNDERPOWERED"


def analyze_objA_v2(
    records: Sequence[Mapping[str, Any]],
    inventory: Sequence[Mapping[str, Any]],
    prereg_hashes: Mapping[str, str],
    ordering_audit: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    pair_block_n = len({(row["dataset"], row["encoder_family"]) for row in records})
    dataset_count = len({row["dataset"] for row in records})
    encoder_count = len({row["encoder_family"] for row in records})
    counts = {
        "registered_cell_n": 12,
        "available_cell_n": len(records),
        "cell_n": len(records),
        "independent_pair_block_n": pair_block_n,
        "dataset_holdout_unit_count": dataset_count,
        "encoder_family_holdout_unit_count": encoder_count,
        "pseudo_replicate_cell_n": len(records) - pair_block_n,
        "excluded_cell_n": 12 - len(records),
    }
    dataset_analysis = _try_block_analysis(records, "dataset")
    encoder_analysis = _try_block_analysis(records, "encoder_family")
    success_bar_met = bool(
        dataset_analysis is not None
        and encoder_analysis is not None
        and dataset_analysis["primary_beats_both_nulls"]
        and encoder_analysis["primary_beats_both_nulls"]
    )
    verdict = verdict_for_v2(success_bar_met, dataset_count, encoder_count)
    arguana = held_out_block_check(dataset_analysis, "ArguAna")
    e5 = held_out_block_check(encoder_analysis, "e5-base-v2")
    aggregate_gap_pass = bool(
        dataset_analysis is not None
        and dataset_analysis["comparisons"]["oracle_gap_ols"]["beats_on_both"]
    )
    if arguana["status"] == "available":
        arguana["agrees_with_aggregate_gap_result"] = bool(
            arguana["margin_vs_gap"]["beats_on_both"] == aggregate_gap_pass
        )
    if e5["status"] == "available":
        encoder_gap_pass = bool(
            encoder_analysis is not None
            and encoder_analysis["comparisons"]["oracle_gap_ols"]["beats_on_both"]
        )
        e5["agrees_with_aggregate_gap_result"] = bool(
            e5["margin_vs_gap"]["beats_on_both"] == encoder_gap_pass
        )
    result = {
        "record_type": "drift_recovery_objA_v2_validation",
        "schema_version": 2,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "preregistration": {
            "id": "objA-oracle-margin-block-loo-v2",
            "hashes": dict(prereg_hashes),
            "hash_lock_sha256": _sha256(PREREG_HASH_PATH),
            "ordering_audit": dict(ordering_audit or {}),
            "primary_feature_names": list(PRIMARY_FEATURE_NAMES),
            "null_names": list(NULL_NAMES),
        },
        "offline_environment": {
            "HF_HUB_OFFLINE": os.environ.get("HF_HUB_OFFLINE"),
            "HF_DATASETS_OFFLINE": os.environ.get("HF_DATASETS_OFFLINE"),
            "TRANSFORMERS_OFFLINE": os.environ.get("TRANSFORMERS_OFFLINE"),
            "execution": "sequential",
        },
        "e5_protocol": {
            "model": E5_MODEL,
            "plain_harness_encode_path": True,
            "query_or_passage_prefixes": False,
            "caveat": "This may understate e5 absolute quality but keeps the encoder-swap protocol identical across families.",
        },
        "counts": counts,
        "regimes": list(records),
        "inventory": list(inventory),
        "excluded_cells": [item for item in inventory if item["status"] != "available"],
        "partial_spearman": (
            partial_spearman_controlling_gap(records) if len(records) >= 2 else None
        ),
        "block_loo": {"dataset": dataset_analysis, "encoder_family": encoder_analysis},
        "novel_holdouts": {"ArguAna": arguana, "e5-base-v2": e5},
        "dataset_margin_beats_gap_on_both": aggregate_gap_pass,
        "dataset_margin_beats_gap_at_four_blocks": bool(
            dataset_count >= 4 and aggregate_gap_pass
        ),
        "success_bar_met": success_bar_met,
        "gap_only_failure_under_either_scheme": bool(
            not aggregate_gap_pass
            or encoder_analysis is None
            or not encoder_analysis["comparisons"]["oracle_gap_ols"]["beats_on_both"]
        ),
        "verdict": verdict,
        "allowed_verdict_labels": list(ALLOWED_VERDICTS),
    }
    if verdict not in ALLOWED_VERDICTS:
        raise AssertionError(f"unregistered v2 verdict: {verdict}")
    return result


def _fmt(value: Any) -> str:
    return "undefined" if value is None else f"{float(value):.4f}"


def render_validation_v2(result: Mapping[str, Any]) -> str:
    lines = [
        "# Obj A v2 validation — powered block-LOO",
        "",
        f"**Frozen-label verdict: {result['verdict']}**. AND-bar met: "
        f"**{str(result['success_bar_met']).upper()}**.",
        "",
        "## Achieved scale",
        "",
        f"- Independent dataset×encoder-family block-n: **{result['counts']['independent_pair_block_n']}**",
        f"- Cell-n: **{result['counts']['cell_n']}** of 12 registered",
        f"- Dataset blocks: **{result['counts']['dataset_holdout_unit_count']}** of 4",
        f"- Encoder-family blocks: **{result['counts']['encoder_family_holdout_unit_count']}** of 3",
        f"- Excluded cells: **{result['counts']['excluded_cell_n']}**",
        "",
        "## Frozen block-LOO tables",
    ]
    for scheme in ("dataset", "encoder_family"):
        analysis = result["block_loo"][scheme]
        lines.extend(["", f"### {scheme}", ""])
        if analysis is None:
            lines.append("Unavailable: fewer than two blocks survived.")
            continue
        lines.extend(
            [
                "| Predictor | Spearman | MAE | Cell-n |",
                "|---|---:|---:|---:|",
            ]
        )
        for predictor in ("oracle_margin_mean", "oracle_gap_ols", "mean_R"):
            metric = analysis["metrics"][predictor]
            lines.append(
                f"| `{predictor}` | {_fmt(metric['spearman'])} | "
                f"{metric['mae']:.4f} | {metric['cell_n']} |"
            )
        for null_name in NULL_NAMES:
            comparison = analysis["comparisons"][null_name]
            lines.append(
                f"- Margin vs `{null_name}`: **{'PASS' if comparison['beats_on_both'] else 'FAIL'}** "
                f"(Spearman strict={comparison['strictly_better_spearman']}; "
                f"MAE strict={comparison['strictly_lower_mae']})."
            )
    partial = result["partial_spearman"]
    lines.extend(["", "## Partial and raw rank associations", ""])
    if partial is None:
        lines.append("Unavailable: fewer than two cells survived.")
    else:
        lines.extend(
            [
                f"- Partial Spearman(margin, R | gap): **{_fmt(partial['partial_spearman_margin_R_controlling_oracle_gap'])}**",
                f"- Raw Spearman(margin, R): **{_fmt(partial['raw_spearman_margin_R'])}**",
                f"- Raw Spearman(gap, R): **{_fmt(partial['raw_spearman_gap_R'])}**",
                f"- Margin-minus-gap raw Spearman: **{_fmt(partial['raw_spearman_margin_minus_gap'])}**",
            ]
        )
    lines.extend(
        [
            "",
            "## Per-cell R",
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
    lines.extend(["", "## Novel held-out block checks", ""])
    for label in ("ArguAna", "e5-base-v2"):
        check = result["novel_holdouts"][label]
        if check["status"] != "available":
            lines.append(f"- **{label}: unavailable**; no within-block comparison was imputed.")
            continue
        comparison = check["margin_vs_gap"]
        margin = check["metrics"]["oracle_margin_mean"]
        gap = check["metrics"]["oracle_gap_ols"]
        lines.append(
            f"- **{label}** ({check['cell_n']} cells): margin rho={_fmt(margin['spearman'])}, "
            f"MAE={margin['mae']:.4f}; gap rho={_fmt(gap['spearman'])}, "
            f"MAE={gap['mae']:.4f}; rank relation={comparison['spearman_relation']}; "
            f"strict both={'PASS' if comparison['beats_on_both'] else 'FAIL'}; "
            f"aggregate agreement={check['agrees_with_aggregate_gap_result']}."
        )
        for row in check["rows"]:
            predictions = row["predictions"]
            lines.append(
                f"  - `{row['regime']}` actual={row['actual_R']:.4f}; "
                f"margin={predictions['oracle_margin_mean']:.4f}; "
                f"gap={predictions['oracle_gap_ols']:.4f}; mean-R={predictions['mean_R']:.4f}."
            )
    lines.extend(["", "## Availability, parity, and leakage audit", ""])
    for item in result["inventory"]:
        if item["status"] == "available":
            lines.append(
                f"- `{item['name']}`: {item['build_action']}; parity max="
                f"{item['harness_parity_max_abs']:.3g}; safe fit n="
                f"{item['leakage_safe_fit_count']}; eval-positive fit docs=0; "
                f"frozen scores={','.join(item['frozen_per_query_score_names'])}."
            )
        else:
            lines.append(f"- `{item['name']}`: **{item['status']}** — {item['reason']}")
    ordering = result["preregistration"]["ordering_audit"]
    lines.extend(
        [
            "",
            "## Freeze ordering",
            "",
            f"Hash lock verified. Preregistration precedes every new pack file: "
            f"**{str(ordering.get('preregistration_precedes_every_new_pack')).upper()}**.",
            "",
            "E5 used the same plain harness text path as mpnet and bge-large, without "
            "`query:`/`passage:` prefixes. This may understate e5 absolute quality but "
            "keeps the encoder-swap protocol identical.",
        ]
    )
    return "\n".join(lines) + "\n"


def _holdout_report_sentence(label: str, check: Mapping[str, Any]) -> str:
    if check["status"] != "available":
        return f"{label} was unavailable, so it provides no holdout evidence."
    comparison = check["margin_vs_gap"]
    margin = check["metrics"]["oracle_margin_mean"]
    gap = check["metrics"]["oracle_gap_ols"]
    return (
        f"{label} **{'agrees' if check['agrees_with_aggregate_gap_result'] else 'disagrees'}** "
        f"with its aggregate gap-only result: margin rho {_fmt(margin['spearman'])} / "
        f"MAE {margin['mae']:.3f} versus gap rho {_fmt(gap['spearman'])} / "
        f"MAE {gap['mae']:.3f}; the within-block rank relation is "
        f"**{comparison['spearman_relation']}** and the strict two-metric check "
        f"**{'passes' if comparison['beats_on_both'] else 'fails'}**."
    )


def render_report_v2(result: Mapping[str, Any]) -> str:
    excluded = result["excluded_cells"]
    excluded_text = (
        "No registered cell failed or had a non-positive oracle gap."
        if not excluded
        else "Excluded: "
        + "; ".join(f"{item['name']} ({item['reason']})" for item in excluded)
        + "."
    )
    dataset = result["block_loo"]["dataset"]
    encoder = result["block_loo"]["encoder_family"]
    if dataset is None or encoder is None:
        metric_text = "At least one block scheme was unavailable after exclusions."
    else:
        dm = dataset["metrics"]["oracle_margin_mean"]
        dg = dataset["metrics"]["oracle_gap_ols"]
        em = encoder["metrics"]["oracle_margin_mean"]
        eg = encoder["metrics"]["oracle_gap_ols"]
        metric_text = (
            f"Dataset-block: margin rho {_fmt(dm['spearman'])}, MAE {dm['mae']:.3f}; "
            f"gap rho {_fmt(dg['spearman'])}, MAE {dg['mae']:.3f}. Encoder-block: "
            f"margin rho {_fmt(em['spearman'])}, MAE {em['mae']:.3f}; gap rho "
            f"{_fmt(eg['spearman'])}, MAE {eg['mae']:.3f}."
        )
    four_block_sentence = (
        "Margin **still beats gap-only on both locked metrics at four dataset blocks**."
        if result["dataset_margin_beats_gap_at_four_blocks"]
        else "Margin **does not beat gap-only on both locked metrics at four dataset blocks**."
    )
    return "\n".join(
        [
            "# Obj A v2 report — powered margin-predictor validation",
            "",
            "## Build and verdict",
            "",
            "Reused all six frozen v1 packs and built only the six registered additions: "
            "ArguAna×{mpnet, bge-large, e5} plus SciFact/NFCorpus/FiQA2018×e5, "
            "sequentially and fully offline. Every available new pack used the frozen "
            "1,200-document slice recipe, safe fit index, frozen per-query scores, and "
            "passed harness parity at 1e-12. "
            + excluded_text,
            "",
            f"**Frozen-label verdict: {result['verdict']}.** The AND-bar "
            f"{'passed' if result['success_bar_met'] else 'failed'} at "
            f"{result['counts']['dataset_holdout_unit_count']} dataset blocks, "
            f"{result['counts']['encoder_family_holdout_unit_count']} encoder blocks, and "
            f"{result['counts']['cell_n']} cells. {metric_text}",
            "",
            "## Brutal holdout read",
            "",
            four_block_sentence,
            "",
            _holdout_report_sentence("ArguAna", result["novel_holdouts"]["ArguAna"]),
            "",
            _holdout_report_sentence("E5", result["novel_holdouts"]["e5-base-v2"]),
            "",
            "E5 was encoded through the same plain harness path as mpnet/bge, with no "
            "`query:` or `passage:` prefixes. That may understate its absolute quality, but "
            "it preserves the encoder-swap protocol. The estimator therefore predicts the "
            "recovery of the swap actually run, not an E5-optimized retrieval deployment.",
            "",
        ]
    )


def run_objA_v2(device: str = "cuda", rebuild_new: bool = False) -> Dict[str, Any]:
    prereg_hashes = verify_frozen_preregistration_v2()
    prereg = json.loads(PREREG_PATH.read_text(encoding="utf-8"))
    regimes = validate_registered_regimes_v2(prereg)
    records, inventory = collect_regimes_v2(
        prereg, device=device, rebuild_new=rebuild_new
    )
    result = analyze_objA_v2(
        records,
        inventory,
        prereg_hashes,
        ordering_audit=_ordering_audit(regimes),
    )
    _write_json(OUT_DIR / V2_OUTPUT_NAMES[0], result)
    (OUT_DIR / V2_OUTPUT_NAMES[1]).write_text(
        render_validation_v2(result), encoding="utf-8"
    )
    (OUT_DIR / V2_OUTPUT_NAMES[2]).write_text(
        render_report_v2(result), encoding="utf-8"
    )
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run pre-registered Obj A v2 powered block-LOO validation"
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--rebuild-new", action="store_true")
    return parser


def main(argv: Any = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_objA_v2(device=args.device, rebuild_new=args.rebuild_new)
    print(
        json.dumps(
            {
                "verdict": result["verdict"],
                "success_bar_met": result["success_bar_met"],
                "counts": result["counts"],
            },
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
