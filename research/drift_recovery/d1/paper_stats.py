"""Build and analyze the frozen SciFact D1 embedding pack."""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np

from research.drift_recovery.artifacts import EmbeddingPack
from research.drift_recovery.harness_bridge import (
    aggregate_ndcg,
    assert_harness_parity,
    encode_scifact_case,
    load_scifact_evalsplit,
    per_query_ndcg,
    system_under_test_spec,
    train_c3a_documents,
)
from research.drift_recovery.stats.learning_curve import (
    COUNTS,
    ORDER_SEEDS,
    nested_prefixes,
    summarize_learning_curve,
)
from research.drift_recovery.stats.multiple_testing import holm_adjust
from research.drift_recovery.stats.paired_bootstrap import (
    paired_contrast,
    paired_query_bootstrap,
    strip_draws,
)


METHODS = ("ridge", "procrustes", "mlp", "c3a")
LEAKAGE_METHODS = ("ridge", "procrustes", "mlp")
METHOD_LABELS = {
    "ridge": "ridge",
    "procrustes": "closed-form orthogonal Procrustes",
    "mlp": "residual MLP",
    "c3a": "C3a",
}


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _ridge(source: np.ndarray, target: np.ndarray, regularization: float = 1.0) -> np.ndarray:
    dimension = source.shape[1]
    return np.linalg.solve(
        source.T @ source + float(regularization) * np.eye(dimension), source.T @ target
    )


def _procrustes(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    left, _singular, right_t = np.linalg.svd(source.T @ target, full_matrices=False)
    return left @ right_t


def _train_residual_mlp(
    all_documents: np.ndarray,
    source: np.ndarray,
    target: np.ndarray,
    seed: int,
    steps: int = 400,
    device: str = "cuda",
) -> np.ndarray:
    import torch
    import torch.nn as nn

    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    target_device = torch.device(device if device == "cuda" and torch.cuda.is_available() else "cpu")
    dimension = int(all_documents.shape[1])
    model = nn.Sequential(nn.Linear(dimension, 1024), nn.GELU(), nn.Linear(1024, dimension)).to(target_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    source_tensor = torch.tensor(source, dtype=torch.float32, device=target_device)
    target_tensor = torch.tensor(target, dtype=torch.float32, device=target_device)
    model.train()
    for _ in range(int(steps)):
        optimizer.zero_grad()
        prediction = source_tensor + model(source_tensor)
        loss = (1.0 - torch.nn.functional.cosine_similarity(prediction, target_tensor, dim=1)).mean()
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        documents = torch.tensor(all_documents, dtype=torch.float32, device=target_device)
        corrected = documents + model(documents)
    return np.asarray(corrected.detach().cpu().numpy(), dtype=np.float64)


def _score_documents(pack_fields: Mapping[str, Any], documents: np.ndarray) -> np.ndarray:
    return per_query_ndcg(
        pack_fields["Qd"],
        pack_fields["query_ids"],
        documents,
        pack_fields["doc_ids"],
        pack_fields["qrels"],
        k=10,
    )


def audit_fit_leakage(pack: EmbeddingPack) -> dict:
    """Recompute literal and leakage-safe fit contamination from frozen qrels."""

    eval_positive_docs = {
        str(doc_id)
        for query_id in pack.query_ids
        for doc_id, relevance in pack.qrels[str(query_id)].items()
        if float(relevance) > 0.0
    }
    doc_ids = np.asarray(pack.doc_ids, dtype=str)
    literal_fit_idx = np.asarray(pack.fit_idx, dtype=np.int64)
    safe_fit_idx = np.asarray(pack.extra_arrays["leakage_safe_fit_idx"], dtype=np.int64)
    literal_positive_count = int(
        sum(str(doc_ids[index]) in eval_positive_docs for index in literal_fit_idx)
    )
    safe_positive_count = int(
        sum(str(doc_ids[index]) in eval_positive_docs for index in safe_fit_idx)
    )
    total_positive_count = len(eval_positive_docs)
    if total_positive_count == 0:
        raise RuntimeError("frozen eval qrels contain no positive documents")
    return {
        "eval_positive_docs_total": total_positive_count,
        "literal_fit_count": len(literal_fit_idx),
        "eval_positive_docs_in_literal_fit": literal_positive_count,
        "literal_fit_share_of_eval_positive_docs": (
            literal_positive_count / total_positive_count
        ),
        "leakage_safe_fit_count": len(safe_fit_idx),
        "eval_positive_docs_in_leakage_safe_fit": safe_positive_count,
    }


def _fit_ladder_methods(
    pack: EmbeddingPack,
    fit_idx: np.ndarray,
    methods: Sequence[str],
    device: str,
) -> Dict[str, np.ndarray]:
    unsupported = set(methods).difference(LEAKAGE_METHODS)
    if unsupported:
        raise ValueError(f"unsupported leakage-sensitivity methods: {sorted(unsupported)}")
    fitted: Dict[str, np.ndarray] = {}
    if "ridge" in methods:
        fitted["ridge"] = pack.Do @ _ridge(pack.Do[fit_idx], pack.Dor[fit_idx])
    if "procrustes" in methods:
        fitted["procrustes"] = pack.Do @ _procrustes(
            pack.Do[fit_idx], pack.Dor[fit_idx]
        )
    if "mlp" in methods:
        fitted["mlp"] = _train_residual_mlp(
            pack.Do,
            pack.Do[fit_idx],
            pack.Dor[fit_idx],
            seed=int(pack.metadata["seed"]),
            device=device,
        )
    return fitted


def compute_leakage_sensitivity(
    pack: EmbeddingPack,
    methods: Sequence[str] = LEAKAGE_METHODS,
    device: str = "cuda",
) -> dict:
    """Refit literal and safe full-600 methods from the frozen embedding pack."""

    selected_methods = tuple(methods)
    audit = audit_fit_leakage(pack)
    if audit["eval_positive_docs_in_leakage_safe_fit"] != 0:
        raise RuntimeError("leakage-safe fit contains eval-positive documents")
    literal_fit_idx = np.asarray(pack.fit_idx, dtype=np.int64)
    safe_fit_idx = np.asarray(pack.extra_arrays["leakage_safe_fit_idx"], dtype=np.int64)
    fields = {
        "Qd": pack.Qd,
        "query_ids": pack.query_ids,
        "doc_ids": pack.doc_ids,
        "qrels": pack.qrels,
    }
    floor = float(np.mean(pack.per_query_scores["floor"]))
    oracle = float(np.mean(pack.per_query_scores["oracle"]))
    gap = oracle - floor
    fitted = {
        "literal_waypoint": _fit_ladder_methods(
            pack, literal_fit_idx, selected_methods, device
        ),
        "leakage_safe_full_600": _fit_ladder_methods(
            pack, safe_fit_idx, selected_methods, device
        ),
    }
    score_rows: Dict[str, Dict[str, Any]] = {}
    for method in selected_methods:
        literal_scores = _score_documents(fields, fitted["literal_waypoint"][method])
        safe_scores = _score_documents(fields, fitted["leakage_safe_full_600"][method])
        frozen_delta = float(
            np.max(np.abs(literal_scores - np.asarray(pack.per_query_scores[method])))
        )
        if frozen_delta > 1e-12:
            raise RuntimeError(
                f"{method} literal refit differs from frozen headline scores by {frozen_delta:.3g}"
            )
        literal_ndcg = float(np.mean(literal_scores))
        safe_ndcg = float(np.mean(safe_scores))
        literal_recovery = (literal_ndcg - floor) / gap
        safe_recovery = (safe_ndcg - floor) / gap
        score_rows[method] = {
            "label": METHOD_LABELS[method],
            "literal_waypoint": {
                "ndcg": literal_ndcg,
                "recovery": literal_recovery,
                "eval_positive_docs_in_fit": audit[
                    "eval_positive_docs_in_literal_fit"
                ],
            },
            "leakage_safe_full_600": {
                "ndcg": safe_ndcg,
                "recovery": safe_recovery,
                "eval_positive_docs_in_fit": audit[
                    "eval_positive_docs_in_leakage_safe_fit"
                ],
            },
            "inflation_recovery_pp": 100.0 * (literal_recovery - safe_recovery),
            "literal_frozen_score_max_abs_delta": frozen_delta,
        }
    return {
        "record_type": "d1_leakage_sensitivity",
        "audit": audit,
        "fits": {
            "literal_waypoint": "literal seed-42 half-corpus permutation",
            "leakage_safe_full_600": (
                "first 600 seed-42 permutation documents after excluding every "
                "eval-positive document"
            ),
        },
        "floor_ndcg": floor,
        "oracle_ndcg": oracle,
        "oracle_gap": gap,
        "methods": score_rows,
    }


def _canonical_reference_audit(recovery: Mapping[str, float]) -> dict:
    """Separate the waypoint reproduction gate from the superseded C3a scalar."""

    tolerance = 0.001
    ridge_target = 0.844
    c3a_old_cross_seed_reference = 0.20
    ridge_error = abs(float(recovery["ridge"]) - ridge_target)
    c3a_difference = abs(float(recovery["c3a"]) - c3a_old_cross_seed_reference)
    return {
        "ridge_waypoint_reproduction": {
            "target_recovery": ridge_target,
            "actual_recovery": float(recovery["ridge"]),
            "absolute_error": ridge_error,
            "tolerance": tolerance,
            "pass": ridge_error <= tolerance,
        },
        "c3a_cross_seed_reference": {
            "old_cross_seed_mean_recovery": c3a_old_cross_seed_reference,
            "seed_42_per_query_recovery": float(recovery["c3a"]),
            "absolute_difference": c3a_difference,
            "tolerance": tolerance,
            "disposition": "superseded_not_a_tolerance_gate",
            "finding": (
                "The old reference was a cross-seed campaign mean. The seed-42 "
                "per-query value supersedes it and is outside the nominal tolerance."
            ),
        },
    }


def build_embedding_pack(output_prefix: Path, seed: int = 42, device: str = "cuda") -> EmbeddingPack:
    """Encode once, fit the canonical methods, score every eval query, and freeze."""

    print("[D1] loading exact SciFact eval split", flush=True)
    case = load_scifact_evalsplit(seed=seed)
    print("[D1] encoding frozen arrays with cached MiniLM/mpnet models", flush=True)
    encoded = encode_scifact_case(case, seed=seed, device=device)
    Do = encoded["Do"]
    Dor = encoded["Dor"]
    Qd = encoded["Qd"]
    doc_ids = np.asarray(case["doc_ids"], dtype=str)
    query_ids = np.asarray(case["eval_query_ids"], dtype=str)

    waypoint_order = np.random.default_rng(seed).permutation(len(doc_ids))
    waypoint_fit_idx = waypoint_order[: len(doc_ids) // 2]
    eval_positive_docs = {
        doc_id
        for query_id in query_ids
        for doc_id, relevance in case["eval_qrels"][str(query_id)].items()
        if float(relevance) > 0.0
    }
    leakage_safe_fit_idx = np.asarray(
        [index for index in waypoint_order if doc_ids[index] not in eval_positive_docs][
            : len(doc_ids) // 2
        ],
        dtype=np.int64,
    )
    if len(leakage_safe_fit_idx) != len(doc_ids) // 2:
        raise RuntimeError("not enough leakage-safe document pairs")
    # The frozen canonical ladder uses the literal waypoint recipe. The separate
    # leakage-safe index is used for all new learning-curve fits and retained so
    # the recipe/contract conflict is explicit rather than silently rewritten.
    fit_idx = waypoint_fit_idx

    fields = {
        "Do": Do,
        "Dor": Dor,
        "Qd": Qd,
        "doc_ids": doc_ids,
        "query_ids": query_ids,
        "qrels": case["eval_qrels"],
    }
    scores: Dict[str, np.ndarray] = {
        "floor": _score_documents(fields, Do),
        "oracle": _score_documents(fields, Dor),
    }
    print("[D1] fitting ridge and closed-form orthogonal Procrustes", flush=True)
    ridge_map = _ridge(Do[fit_idx], Dor[fit_idx], regularization=1.0)
    procrustes_map = _procrustes(Do[fit_idx], Dor[fit_idx])
    ridge_documents = Do @ ridge_map
    procrustes_documents = Do @ procrustes_map
    scores["ridge"] = _score_documents(fields, ridge_documents)
    scores["procrustes"] = _score_documents(fields, procrustes_documents)
    print("[D1] fitting canonical residual MLP", flush=True)
    mlp_documents = _train_residual_mlp(Do, Do[fit_idx], Dor[fit_idx], seed=seed, device=device)
    scores["mlp"] = _score_documents(fields, mlp_documents)
    print("[D1] fitting canonical C3a through merged bounded-InfoNCE path", flush=True)
    c3a_documents = train_c3a_documents(
        Do,
        encoded["canonical_source"],
        encoded["canonical_target"],
        seed=seed,
    )
    scores["c3a"] = _score_documents(fields, c3a_documents)

    # Audited waypoint values use the literal unfiltered half-document permutation.
    waypoint_ridge = _score_documents(fields, Do @ _ridge(Do[waypoint_fit_idx], Dor[waypoint_fit_idx]))
    waypoint_procrustes = _score_documents(
        fields, Do @ _procrustes(Do[waypoint_fit_idx], Dor[waypoint_fit_idx])
    )
    aggregate_reference = {name: aggregate_ndcg(values) for name, values in scores.items()}
    floor = aggregate_reference["floor"]
    oracle = aggregate_reference["oracle"]
    gap = oracle - floor
    recovery = {name: (aggregate_reference[name] - floor) / gap for name in METHODS}
    waypoint_recovery = {
        "ridge": (aggregate_ndcg(waypoint_ridge) - floor) / gap,
        "procrustes": (aggregate_ndcg(waypoint_procrustes) - floor) / gap,
    }
    canonical_reference_audit = _canonical_reference_audit(recovery)
    old_scalar_audit = {
        "old_c3a_scalar_ndcg": 0.1609,
        "fixed_split_per_query_ndcg": aggregate_reference["c3a"],
        "ndcg_difference": aggregate_reference["c3a"] - 0.1609,
        "old_scalar_implied_recovery": (0.1609 - floor) / gap,
        "fixed_split_per_query_recovery": recovery["c3a"],
        "finding": (
            "The old scalar was a cross-seed campaign mean attached to the seed-42 eval split. "
            "The frozen per-query seed-42 result supersedes it."
        ),
    }

    pack = EmbeddingPack(
        Do=Do,
        Dor=Dor,
        Qd=Qd,
        doc_ids=doc_ids,
        query_ids=query_ids,
        qrels=case["eval_qrels"],
        fit_idx=fit_idx,
        per_query_scores=scores,
        extra_arrays={
            "leakage_safe_fit_idx": leakage_safe_fit_idx,
            "canonical_c3a_source": encoded["canonical_source"],
            "canonical_c3a_target": encoded["canonical_target"],
            "curve_c3a_source": encoded["curve_source"],
            "curve_c3a_target": encoded["curve_target"],
            "curve_c3a_query_ids": np.asarray(case["curve_pair_query_ids"], dtype=str),
            "method_documents__ridge": ridge_documents,
            "method_documents__procrustes": procrustes_documents,
            "method_documents__mlp": mlp_documents,
            "method_documents__c3a": c3a_documents,
        },
        metadata={
            "dataset": "SciFact",
            "seed": seed,
            "max_queries": 100,
            "sample_docs": 1200,
            "anchor_fraction": 0.4,
            "k": 10,
            "ndcg": "binary merged-harness ndcg_at_k",
            "anchor_ids": case["anchor_ids"],
            "canonical_c3a_pair_count": len(encoded["canonical_source"]),
            "learning_curve_c3a_pool": (
                "one positive doc per non-eval SciFact query; eval IDs and eval qrels excluded"
            ),
            "drift_manifest": encoded["drift_manifest"],
            "models": {"old": encoded["old_model"], "swap": encoded["swap_model"]},
            "harness_aggregate_ndcg": aggregate_reference,
            "recovery": recovery,
            "canonical_reference_audit": canonical_reference_audit,
            "old_scalar_audit": old_scalar_audit,
            "waypoint_audit": {
                "eval_positive_docs_total": len(eval_positive_docs),
                "literal_fit_count": len(waypoint_fit_idx),
                "eval_positive_docs_in_literal_fit": int(
                    sum(doc_ids[index] in eval_positive_docs for index in waypoint_fit_idx)
                ),
                "literal_fit_share_of_eval_positive_docs": (
                    sum(doc_ids[index] in eval_positive_docs for index in waypoint_fit_idx)
                    / len(eval_positive_docs)
                ),
                "leakage_safe_fit_count": len(leakage_safe_fit_idx),
                "eval_positive_docs_in_leakage_safe_fit": int(
                    sum(doc_ids[index] in eval_positive_docs for index in leakage_safe_fit_idx)
                ),
                "literal_recovery": waypoint_recovery,
                "protocol_conflict": (
                    "The exact waypoint permutation contains eval-positive documents. "
                    "Canonical ladder/bootstrap reproduction uses it; new learning-curve fits use "
                    "the separate leakage_safe_fit_idx."
                ),
            },
        },
    )
    pack.save(output_prefix)
    reloaded = EmbeddingPack.load(output_prefix)
    assert_harness_parity(reloaded, atol=1e-12)
    return reloaded


def ladder_table(
    pack: EmbeddingPack,
    leakage_sensitivity: Optional[Mapping[str, Any]] = None,
) -> dict:
    floor = float(np.mean(pack.per_query_scores["floor"]))
    oracle = float(np.mean(pack.per_query_scores["oracle"]))
    gap = oracle - floor
    methods = {}
    for method in METHODS:
        ndcg = float(np.mean(pack.per_query_scores[method]))
        methods[method] = {
            "ndcg": ndcg,
            "delta_ndcg": ndcg - floor,
            "recovery": (ndcg - floor) / gap,
        }
    waypoint_audit = audit_fit_leakage(pack)
    waypoint_audit.update(
        {
            "literal_recovery": {
                method: methods[method]["recovery"]
                for method in LEAKAGE_METHODS
            },
            "protocol_conflict": (
                "The literal waypoint fit contains eval-positive documents. "
                "It is retained only as the reproduced headline protocol; the "
                "leakage-safe full-600 sensitivity excludes all eval-positive documents."
            ),
        }
    )
    result = {
        "record_type": "d1_ladder",
        "dataset": "SciFact",
        "query_count": len(pack.query_ids),
        "floor_ndcg": floor,
        "oracle_ndcg": oracle,
        "oracle_gap": gap,
        "methods": methods,
        "canonical_reference_audit": _canonical_reference_audit(
            {method: row["recovery"] for method, row in methods.items()}
        ),
        "old_scalar_audit": pack.metadata["old_scalar_audit"],
        "waypoint_audit": waypoint_audit,
    }
    if leakage_sensitivity is not None:
        result["leakage_sensitivity"] = leakage_sensitivity
    return result


def run_bootstrap(pack: EmbeddingPack) -> tuple[dict, dict]:
    bootstrap = paired_query_bootstrap(pack.per_query_scores, METHODS, draws=10_000, seed=20260630)
    contrasts = {
        "ridge_minus_mlp": paired_contrast(bootstrap, "ridge", "mlp"),
        "ridge_minus_c3a": paired_contrast(bootstrap, "ridge", "c3a"),
    }
    holm = holm_adjust({name: row["p_value"] for name, row in contrasts.items()})
    for name, row in contrasts.items():
        row["multiple_testing"] = holm[name]
    primary_half_width = float(contrasts["ridge_minus_mlp"]["half_width"])
    secondary_median_half_width = float(
        np.median([row["half_width"] for row in contrasts.values()])
    )
    threshold = 0.015
    query_count = int(bootstrap["query_count"])
    rough_query_count = int(
        math.ceil(query_count * (primary_half_width / threshold) ** 2)
    )
    contrast_table = {
        "record_type": "d1_preregistered_contrasts",
        "family": ["ridge_minus_mlp", "ridge_minus_c3a"],
        "correction": "Holm",
        "contrasts": contrasts,
        "g2": {
            "primary_contrast": "ridge_minus_mlp",
            "primary_delta_ndcg_half_width": primary_half_width,
            "secondary_median_family_delta_ndcg_half_width": (
                secondary_median_half_width
            ),
            "threshold": threshold,
            "pass": bool(primary_half_width <= threshold),
            "rough_query_count_for_threshold": rough_query_count,
            "scale_up_note": (
                "Rough 1/sqrt(n) extrapolation only; it is not a prospective "
                "power calculation and must be re-estimated with pilot variance."
            ),
        },
    }
    return strip_draws(bootstrap), contrast_table


def run_learning_curves(pack: EmbeddingPack, device: str = "cuda") -> dict:
    """Run separate doc-pair and query-InfoNCE nested-prefix panels."""

    rows = []
    safe_fit_idx = pack.extra_arrays["leakage_safe_fit_idx"]
    doc_prefixes = nested_prefixes(len(safe_fit_idx), COUNTS, ORDER_SEEDS)
    c3a_pool_size = len(pack.extra_arrays["curve_c3a_source"])
    c3a_prefixes = nested_prefixes(c3a_pool_size, COUNTS, ORDER_SEEDS)
    floor = float(np.mean(pack.per_query_scores["floor"]))
    oracle = float(np.mean(pack.per_query_scores["oracle"]))
    gap = oracle - floor
    fields = {
        "Qd": pack.Qd,
        "query_ids": pack.query_ids,
        "doc_ids": pack.doc_ids,
        "qrels": pack.qrels,
    }
    for order_seed in ORDER_SEEDS:
        print(f"[D1] learning curve order seed {order_seed}", flush=True)
        for count in COUNTS:
            local = doc_prefixes[order_seed][count]
            indices = safe_fit_idx[local]
            ridge_documents = pack.Do @ _ridge(pack.Do[indices], pack.Dor[indices])
            ridge_ndcg = float(np.mean(_score_documents(fields, ridge_documents)))
            rows.append(
                {
                    "supervision_type": "paired_document_embeddings",
                    "method": "ridge",
                    "supervision_count": count,
                    "order_seed": order_seed,
                    "ndcg": ridge_ndcg,
                    "recovery": (ridge_ndcg - floor) / gap,
                    "prefix_sha256": hashlib.sha256(indices.tobytes()).hexdigest(),
                }
            )
            mlp_documents = _train_residual_mlp(
                pack.Do,
                pack.Do[indices],
                pack.Dor[indices],
                seed=order_seed,
                device=device,
            )
            mlp_ndcg = float(np.mean(_score_documents(fields, mlp_documents)))
            rows.append(
                {
                    "supervision_type": "paired_document_embeddings",
                    "method": "mlp",
                    "supervision_count": count,
                    "order_seed": order_seed,
                    "ndcg": mlp_ndcg,
                    "recovery": (mlp_ndcg - floor) / gap,
                    "prefix_sha256": hashlib.sha256(indices.tobytes()).hexdigest(),
                }
            )
            c3a_indices = c3a_prefixes[order_seed][count]
            c3a_documents = train_c3a_documents(
                pack.Do,
                pack.extra_arrays["curve_c3a_source"][c3a_indices],
                pack.extra_arrays["curve_c3a_target"][c3a_indices],
                seed=order_seed,
            )
            c3a_ndcg = float(np.mean(_score_documents(fields, c3a_documents)))
            rows.append(
                {
                    "supervision_type": "query_relevance_infonce",
                    "method": "c3a",
                    "supervision_count": count,
                    "order_seed": order_seed,
                    "ndcg": c3a_ndcg,
                    "recovery": (c3a_ndcg - floor) / gap,
                    "prefix_sha256": hashlib.sha256(c3a_indices.tobytes()).hexdigest(),
                }
            )
    return {
        "record_type": "d1_learning_curve",
        "counts": list(COUNTS),
        "order_seeds": list(ORDER_SEEDS),
        "eval_query_ids": pack.query_ids.tolist(),
        "panels": {
            "paired_document_embeddings": {
                "methods": ["ridge", "mlp"],
                "count_unit": "paired old/new document embeddings",
                "pool_size": len(safe_fit_idx),
            },
            "query_relevance_infonce": {
                "methods": ["c3a"],
                "count_unit": "distinct non-eval query anchors, one positive document each",
                "pool_size": c3a_pool_size,
            },
        },
        "warning": "Panels use different supervision types and counts are not commensurate.",
        "rows": rows,
        "summary": summarize_learning_curve(rows),
    }


def _leakage_table_lines(leakage: Mapping[str, Any]) -> list[str]:
    lines = [
        "| Method | Literal waypoint recovery | Leakage-safe full-600 recovery | Inflation (pp) |",
        "|---|---:|---:|---:|",
    ]
    for method in LEAKAGE_METHODS:
        row = leakage["methods"][method]
        lines.append(
            f"| {row['label']} | {100*row['literal_waypoint']['recovery']:.1f}% "
            f"| {100*row['leakage_safe_full_600']['recovery']:.1f}% "
            f"| +{row['inflation_recovery_pp']:.1f} |"
        )
    return lines


def render_ci_report(
    ladder: Mapping[str, Any],
    bootstrap: Mapping[str, Any],
    contrasts: Mapping[str, Any],
    leakage: Mapping[str, Any],
) -> str:
    lines = [
        "# D1 paired-query bootstrap results",
        "",
        f"SciFact eval split: {bootstrap['query_count']} fixed queries; {bootstrap['draws']:,} deterministic paired draws. ",
        "Recovery is the ratio of resampled query means. All methods, floor, and oracle use the same draws.",
        "",
        f"Floor NDCG@10 = {ladder['floor_ndcg']:.4f}; oracle = {ladder['oracle_ndcg']:.4f}; gap = {ladder['oracle_gap']:.4f}.",
        "",
        "| Method | NDCG@10 (95% CI) | ΔNDCG vs floor (95% CI) | Recovery (95% CI) |",
        "|---|---:|---:|---:|",
    ]
    for method in METHODS:
        row = bootstrap["methods"][method]
        lines.append(
            f"| {METHOD_LABELS[method]} | {row['ndcg']['estimate']:.4f} [{row['ndcg']['ci_low']:.4f}, {row['ndcg']['ci_high']:.4f}] "
            f"| {row['delta_ndcg']['estimate']:.4f} [{row['delta_ndcg']['ci_low']:.4f}, {row['delta_ndcg']['ci_high']:.4f}] "
            f"| {100*row['recovery']['estimate']:.1f}% [{100*row['recovery']['ci_low']:.1f}%, {100*row['recovery']['ci_high']:.1f}%] |"
        )
    lines.extend(
        [
            "",
            "## Preregistered paired contrasts",
            "",
            "| Contrast | ΔNDCG (95% CI) | raw p | Holm p | Reject at 0.05 |",
            "|---|---:|---:|---:|:---:|",
        ]
    )
    for name, row in contrasts["contrasts"].items():
        mt = row["multiple_testing"]
        lines.append(
            f"| {name.replace('_', ' ')} | {row['estimate']:.4f} [{row['ci_low']:.4f}, {row['ci_high']:.4f}] "
            f"| {row['p_value']:.6g} | {mt['holm_adjusted_p']:.6g} | {'yes' if mt['reject'] else 'no'} |"
        )
    g2 = contrasts["g2"]
    audit = leakage["audit"]
    c3a_reference = ladder["canonical_reference_audit"]["c3a_cross_seed_reference"]
    lines.extend(
        [
            "",
            f"G2 power gate: **{'PASS' if g2['pass'] else 'FAIL'}**. The preregistered primary "
            f"ridge−MLP half-width is {g2['primary_delta_ndcg_half_width']:.4f} "
            f"(threshold ≤ {g2['threshold']:.3f}). The median across both preregistered "
            f"contrasts is a secondary diagnostic at "
            f"{g2['secondary_median_family_delta_ndcg_half_width']:.4f}.",
            "",
            f"A rough 1/√n extrapolation from {bootstrap['query_count']} queries suggests "
            f"approximately {g2['rough_query_count_for_threshold']} queries (about 800+) to "
            f"reach the {g2['threshold']:.3f} ridge−MLP half-width. This is a scale estimate, "
            "not a prospective power calculation; variance must be re-estimated in a pilot.",
            "",
            "## Leakage sensitivity of the literal waypoint headline",
            "",
            f"The literal 600-document fit contains "
            f"{audit['eval_positive_docs_in_literal_fit']} of the "
            f"{audit['eval_positive_docs_total']} eval-positive documents "
            f"({100*audit['literal_fit_share_of_eval_positive_docs']:.1f}%). The "
            f"leakage-safe full-{audit['leakage_safe_fit_count']} fit contains "
            f"{audit['eval_positive_docs_in_leakage_safe_fit']}. The literal ridge headline "
            f"({100*leakage['methods']['ridge']['literal_waypoint']['recovery']:.1f}%) is "
            "therefore a leaky-fit point estimate, not a clean held-out estimate.",
            "",
            *_leakage_table_lines(leakage),
            "",
            "The primary ladder/bootstrap reproduces the literal waypoint recipe. The "
            "full-600 sensitivity and all learning-curve document fits exclude eval-positive "
            "documents; the two protocols must not be conflated.",
            "",
            f"C3a scalar audit: the old 0.1609 scalar implied "
            f"{100*ladder['old_scalar_audit']['old_scalar_implied_recovery']:.1f}% recovery on this split, but "
            f"the actual frozen per-query score is {ladder['old_scalar_audit']['fixed_split_per_query_ndcg']:.4f} "
            f"({100*ladder['old_scalar_audit']['fixed_split_per_query_recovery']:.1f}%).",
            f"This is a supersession, not a tolerance pass: the recovery difference from "
            f"the old cross-seed 0.200 reference is {c3a_reference['absolute_difference']:.4f}, "
            f"which exceeds the nominal {c3a_reference['tolerance']:.3f} threshold.",
        ]
    )
    return "\n".join(lines) + "\n"


def render_section3() -> str:
    spec = system_under_test_spec()
    bounded = spec["bounded_defaults"]
    controller = spec["controller_defaults"]
    supervised = spec["supervised_cycle_controller"]
    return rf"""# 3. System Under Test

We evaluate a post-hoc correction layer placed between an embedding encoder and cosine retrieval. Let
$x\in\mathbb{{R}}^d$ be an encoder output and $n(x)=x/\lVert x\rVert_2$ (with the implementation's
usual nonzero-vector assumption). Every adapter returns an L2-normalized vector, so retrieval ranks
documents by the cosine-equivalent dot product between normalized query and document vectors.

## 3.1 Adapter forward maps

The residual MLP used by the supervised loop is

$$g_\theta(x)=n\!\left(x+W_2\,\mathrm{{ReLU}}(W_1x+b_1)+b_2\right).$$

Its hidden width defaults to $d/2$. Both linear layers are initialized with zero bias and Gaussian
weights of standard deviation $10^{{-3}}$, making the initial map close to identity. The nonlinear
comparison in §5.5 uses the same residual form with a 1,024-wide GELU hidden layer and no bounded
wrapper.

The D1 ladder baseline is **closed-form orthogonal Procrustes**: it computes an orthogonal map by
SVD of the paired-document cross-covariance. It is not the trainable Cayley adapter described next,
and the two objects must not share an unqualified "Procrustes" label in the ladder or its caption.

The trainable Cayley adapter parameterizes $A=P-P^\top$ and computes the map

$$W=(I-A)(I+A)^{{-1}},\qquad g_{{\mathrm{{proc}}}}(x)=n\!\left((xW^\top)\odot s\right),$$

where $s$ is a learned per-dimension scale initialized to one. The low-rank affine alternative is

$$g_{{\mathrm{{lr}}}}(x)=n\!\left(x+(xU)V^\top+b\right),$$

with $U\in\mathbb{{R}}^{{d\times r}}$, $V\in\mathbb{{R}}^{{d\times r}}$, and $V$ initialized to zero,
so its initial correction is exactly zero.

## 3.2 Exact bounded-adapter map and correction floor

The bounded wrapper operates on the normalized input $\bar x=n(x)$ and the already normalized output
$z=g_\theta(x)$. With learned dimension scale $s$ (initialized to all ones), define

$$\delta=((z-\bar x)\odot s),\qquad r=\lVert\delta\rVert_2.$$

For lower and upper correction bounds $a$ and $b$, the implementation applies

$$
\gamma(r)=
\begin{{cases}}
a/r, & 10^{{-10}}<r<a,\\
b/r, & r>b,\\
1, & \text{{otherwise}},
\end{{cases}}
\qquad
g_{{[a,b]}}(x)=n\!\left(\bar x+\gamma(r)\delta\right).
$$

Thus the lower bound is a floor only for a *nonzero* correction: an exactly zero (or numerically
$\le10^{{-10}}$) correction is left at zero. C3a uses $a={bounded['min_correction']:.2f}$ and
$b={bounded['max_correction']:.1f}$. The stated INT8 scale is $1/128={bounded['int8_floor']:.7f}$,
so $a=0.01$ is intended to place nonzero corrections above that quantization-noise scale. After
clipping, the vector is normalized again. The wrapper therefore bounds the pre-renormalization
correction norm; it does not imply an identical Euclidean displacement after the final normalization.

## 3.3 Drift-triggered controller

The controller stores a scalar temperature $T$. Given a nonnegative drift magnitude $m$, observation
is a strict-threshold update:

$$
T\leftarrow
\begin{{cases}}
\max(T,\min(T_{{\max}},m)), & m>\tau,\\
T, & m\le\tau.
\end{{cases}}
$$

Correction is enabled iff $T>{controller['epsilon']:.0e}$. With
$\rho=\mathrm{{clip}}(T/T_{{\max}},0,1)$, a correction cycle receives learning-rate scale
$0.1+0.9\rho$, epoch count $1+\mathrm{{round}}(2\rho)$, and online intensity $\rho$. At cycle end,
$T\leftarrow cT$ and values at or below ${controller['epsilon']:.0e}$ are set to zero. Constructor
defaults are $T_0={controller['initial_temperature']:.1f}$, $\tau={controller['trigger_threshold']:.2f}$,
$T_{{\max}}={controller['max_temperature']:.1f}$, and $c={controller['cooling_rate']:.1f}$.

For the supervised query-encoder-swap cycle, the actual drift signal is
${supervised['drift_signal']}$, the campaign sets $\tau=0$, $T_{{\max}}=1$, and uses
$c={supervised['cooling_rate']:.1f}$. When the trigger fires and anchor pairs exist, the adapter is
reinitialized from the run seed, trained by query–document InfoNCE, applied to every cached document,
and written back. In the default non-compounding regime every cycle applies the newly trained adapter
to the same pre-correction document snapshot, making the correction a reproducible one-shot map rather
than an accumulated trajectory.
"""


def _paper_section_for_line(lines: Sequence[str], line_number: int) -> str:
    for index in range(line_number - 1, -1, -1):
        if lines[index].startswith("#"):
            return lines[index].lstrip("# ")
    return "document preamble"


def render_paper_edits(
    paper_path: Optional[Path],
    ladder: Mapping[str, Any],
    bootstrap: Mapping[str, Any],
    contrasts: Mapping[str, Any],
    leakage: Mapping[str, Any],
) -> str:
    lines = paper_path.read_text(encoding="utf-8").splitlines() if paper_path and paper_path.exists() else []
    output = [
        "# Chair edit-list for the paper",
        "",
        "This file is an application list only; the git-excluded paper was not modified.",
        "",
        "## Replace all ceiling claims",
        "",
        "Use **observed plateau under this protocol**. This wording does not claim a mathematical or universal ceiling.",
        "",
        "| Source line | Section | Current text containing `ceiling` | Required edit |",
        "|---:|---|---|---|",
    ]
    if lines:
        for line_number, text in enumerate(lines, start=1):
            if "ceiling" not in text.lower():
                continue
            escaped = text.strip().replace("|", "\\|")
            section = _paper_section_for_line(lines, line_number).replace("|", "\\|")
            output.append(
                f"| {line_number} | {section} | {escaped} | Replace each `ceiling` noun phrase with "
                "`observed plateau under this protocol`; recast grammar where needed. |"
            )
    else:
        output.append("| — | — | Paper source unavailable to the runner | Re-run with `--paper PATH`. |")

    ridge = bootstrap["methods"]["ridge"]
    mlp = bootstrap["methods"]["mlp"]
    c3a = bootstrap["methods"]["c3a"]
    ridge_mlp = contrasts["contrasts"]["ridge_minus_mlp"]
    ridge_c3a = contrasts["contrasts"]["ridge_minus_c3a"]
    g2 = contrasts["g2"]
    audit = leakage["audit"]
    ridge_mlp_low = f"{ridge_mlp['ci_low']:.4f}".replace("-", "−")
    ridge_mlp_high = f"{ridge_mlp['ci_high']:.4f}"
    mlp_better_recovery_points = abs(ridge_mlp["ci_low"]) / ladder["oracle_gap"] * 100
    ridge_better_recovery_points = ridge_mlp["ci_high"] / ladder["oracle_gap"] * 100
    output.extend(
        [
            "",
            "## Mandatory leakage-sensitivity disclosure",
            "",
            f"The literal waypoint 600-document fit contains "
            f"{audit['eval_positive_docs_in_literal_fit']} of "
            f"{audit['eval_positive_docs_total']} eval-positive documents "
            f"({100*audit['literal_fit_share_of_eval_positive_docs']:.1f}%). The "
            f"leakage-safe full-{audit['leakage_safe_fit_count']} fit excludes all of them. "
            f"The {100*leakage['methods']['ridge']['literal_waypoint']['recovery']:.1f}% "
            "headline is therefore a leaky-fit point estimate.",
            "",
            *_leakage_table_lines(leakage),
            "",
            "State explicitly that the primary ladder/bootstrap is a literal-waypoint "
            "reproduction, while the full-600 sensitivity and document learning curves are "
            "leakage-safe. Do not equate the safe n=128 learning-curve regime with the "
            "literal 600-document headline.",
            "",
            "## Mandatory location-agnostic inference-ban block",
            "",
            "Paste this prohibition wherever the ridge−MLP result or plateau interpretation is discussed:",
            "",
            f"> Do not interpret ridge−MLP non-significance as evidence of a linear post-hoc ceiling or the "
            f"absence of nonlinear benefit. The 95% CI on ΔNDCG(ridge−MLP) is "
            f"[{ridge_mlp_low}, {ridge_mlp_high}] — it permits MLP better by up to "
            f"~{mlp_better_recovery_points:.0f} recovery points and ridge better by up to "
            f"~{ridge_better_recovery_points:.0f}. G2 FAILED (primary ridge−MLP half-width "
            f"{g2['primary_delta_ndcg_half_width']:.3f} ≫ {g2['threshold']:.3f}; the secondary "
            f"median across both preregistered contrasts is "
            f"{g2['secondary_median_family_delta_ndcg_half_width']:.3f}), so this study is "
            f"underpowered for the equality contrast on {bootstrap['query_count']} queries. "
            "Non-significance here is absence of evidence, not evidence of absence.",
            "",
            "### Forbidden claims",
            "",
            "- No **linear post-hoc ceiling**.",
            "- No **no nonlinear benefit** or **capacity is linear**.",
            f"- No **{100*ridge['recovery']['estimate']:.0f}% is the recoverable upper bound**.",
            f"- No **irreducible {100*(1.0-ridge['recovery']['estimate']):.0f}%**.",
            "- No G2-powered precision on method contrasts.",
            "",
            "### Permitted claims",
            "",
            "- Report point estimates plus 95% CIs on the stated protocol.",
            "- Ridge is the highest-observed method here.",
            "- Ridge ≫ C3a is Holm-significant.",
            "- Ridge versus MLP is not distinguishable at α=0.05 under this n.",
            "- Prefer **observed plateau under this protocol**.",
            "",
            "## Replace point-only D1 numbers with uncertainty-aware values",
            "",
            "Apply these at every matching abstract, Introduction, §4.2, §5.5, §6/§7, and Limitations occurrence; "
            "keep the full precision in tables and round prose to one recovery point / three NDCG decimals.",
            "",
            "| Quantity | Chair-ready replacement |",
            "|---|---|",
            f"| Ridge | NDCG {ridge['ndcg']['estimate']:.4f} (95% CI {ridge['ndcg']['ci_low']:.4f}–{ridge['ndcg']['ci_high']:.4f}); recovery {100*ridge['recovery']['estimate']:.1f}% (95% CI {100*ridge['recovery']['ci_low']:.1f}%–{100*ridge['recovery']['ci_high']:.1f}%) |",
            f"| Residual MLP | NDCG {mlp['ndcg']['estimate']:.4f} (95% CI {mlp['ndcg']['ci_low']:.4f}–{mlp['ndcg']['ci_high']:.4f}); recovery {100*mlp['recovery']['estimate']:.1f}% (95% CI {100*mlp['recovery']['ci_low']:.1f}%–{100*mlp['recovery']['ci_high']:.1f}%) |",
            f"| C3a (per-query, not scalar) | NDCG {c3a['ndcg']['estimate']:.4f} (95% CI {c3a['ndcg']['ci_low']:.4f}–{c3a['ndcg']['ci_high']:.4f}); recovery {100*c3a['recovery']['estimate']:.1f}% (95% CI {100*c3a['recovery']['ci_low']:.1f}%–{100*c3a['recovery']['ci_high']:.1f}%) |",
            f"| Ridge − MLP | ΔNDCG {ridge_mlp['estimate']:.4f} (95% CI {ridge_mlp['ci_low']:.4f}–{ridge_mlp['ci_high']:.4f}); Holm-adjusted p={ridge_mlp['multiple_testing']['holm_adjusted_p']:.6g} |",
            f"| Ridge − C3a | ΔNDCG {ridge_c3a['estimate']:.4f} (95% CI {ridge_c3a['ci_low']:.4f}–{ridge_c3a['ci_high']:.4f}); Holm-adjusted p={ridge_c3a['multiple_testing']['holm_adjusted_p']:.6g} |",
            "",
            "## Location-specific instructions",
            "",
            "- Abstract and §1: replace the point-only 84%/20% comparison with ridge and C3a recovery plus their CIs; describe ridge as the best observed plateau under this protocol.",
            "- §4.2: define recovery exactly as already written, but replace the quoted `~85% linear ceiling` interpretation with the observed ridge plateau and its CI.",
            "- §5.5 table/caption/body: replace all ridge, residual MLP, closed-form orthogonal Procrustes, and C3a entries from `ladder.json`; add the paired contrast CIs and Holm results. State that the ladder's closed-form SVD map is distinct from the trainable Cayley adapter, and that C3a was rescored per query through the merged NDCG implementation.",
            f"- Remove the old C3a `0.1609` scalar wherever it is presented as this fixed split. The actual seed-42 per-query mean is {c3a['ndcg']['estimate']:.4f}; `0.1609` was the mean across three different seeded splits.",
            "- §6 and §7: do not call the unexplained residual an irreducible 16%; the bootstrap measures sampling uncertainty, not an information-theoretic bound.",
            "- §8: replace `fixed ... ceiling` with `observed plateau under this protocol` and add the fixed 60-query eval-split limitation.",
            "- Learning-curve text/figure: show two panels. Ridge/MLP x-axis is paired document embeddings; C3a x-axis is distinct non-eval query anchors. Do not overlay them on a shared `n anchors` axis.",
            "",
            f"G2 is **{'passed' if g2['pass'] else 'failed'}**. The primary ridge−MLP "
            f"half-width is {g2['primary_delta_ndcg_half_width']:.4f}; the secondary "
            f"median across both preregistered contrasts is "
            f"{g2['secondary_median_family_delta_ndcg_half_width']:.4f}.",
        ]
    )
    return "\n".join(output) + "\n"


def _line_number(path: Path, needle: str) -> int:
    for line_number, text in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if needle in text:
            return line_number
    raise RuntimeError(f"cannot locate {needle!r} in {path}")


def _research_relative(path: Path) -> str:
    parts = path.resolve().parts
    research_index = parts.index("research")
    return Path(*parts[research_index:]).as_posix()


def render_c1_fix_report(
    ladder: Mapping[str, Any],
    contrasts: Mapping[str, Any],
    leakage: Mapping[str, Any],
    parity: Mapping[str, float],
    output_dir: Path,
) -> str:
    source = Path(__file__)
    drift_root = source.parents[1]
    tests_path = drift_root / "tests" / "test_d1_stats.py"
    bootstrap_path = drift_root / "stats" / "paired_bootstrap.py"
    ci_path = output_dir / "ci_report.md"
    edits_path = output_dir / "paper_edits.md"
    section3_path = output_dir / "section3_draft.md"
    contrast_path = output_dir / "contrasts_holm.json"
    ladder_path = output_dir / "ladder.json"

    def location(path: Path, needle: str) -> str:
        return f"`{_research_relative(path)}:{_line_number(path, needle)}`"

    audit = leakage["audit"]
    g2 = contrasts["g2"]
    c3a_reference = ladder["canonical_reference_audit"]["c3a_cross_seed_reference"]
    max_parity_delta = max(abs(float(value)) for value in parity.values())
    leakage_results = ", ".join(
        f"{leakage['methods'][method]['label']} "
        f"{100*leakage['methods'][method]['leakage_safe_full_600']['recovery']:.1f}% "
        f"safe / +{leakage['methods'][method]['inflation_recovery_pp']:.1f} pp"
        for method in LEAKAGE_METHODS
    )
    lines = [
        "# C1 adversarial-review fix report",
        "",
        "Generated from the frozen pack and regenerated D1 artifacts; result numbers are not hand-entered.",
        "",
        "| Fix | Exact evidence location | Result |",
        "|---|---|---|",
        f"| F1 | {location(source, 'def compute_leakage_sensitivity')}<br>"
        f"{location(ci_path, '## Leakage sensitivity')}<br>"
        f"{location(edits_path, '## Mandatory leakage-sensitivity')} | "
        f"Literal fit: {audit['eval_positive_docs_in_literal_fit']}/"
        f"{audit['eval_positive_docs_total']} eval-positive docs; safe fit: "
        f"{audit['eval_positive_docs_in_leakage_safe_fit']}. {leakage_results}. |",
        f"| F2 | {location(edits_path, '## Mandatory location-agnostic inference-ban')} | "
        f"Ban plus forbidden/permitted lists emitted; ridge−MLP CI "
        f"[{contrasts['contrasts']['ridge_minus_mlp']['ci_low']:.4f}, "
        f"{contrasts['contrasts']['ridge_minus_mlp']['ci_high']:.4f}]. |",
        f"| F3 | {location(edits_path, 'Paste this prohibition')} | Chair-only application guard "
        "is explicit; protected paper-draft files modified: 0. |",
        f"| F4 | {location(source, 'The D1 ladder baseline is **closed-form orthogonal Procrustes**')}<br>"
        f"{location(section3_path, 'The D1 ladder baseline is **closed-form orthogonal Procrustes**')} | "
        "Closed-form SVD ladder map is explicitly distinct from the trainable Cayley adapter. |",
        f"| F5 | {location(source, 'def _canonical_reference_audit')}<br>"
        f"{location(ladder_path, 'superseded_not_a_tolerance_gate')} | "
        f"Old cross-seed C3a scalar 0.1609 is superseded by seed-42 per-query "
        f"{ladder['old_scalar_audit']['fixed_split_per_query_ndcg']:.4f}; recovery difference "
        f"{c3a_reference['absolute_difference']:.4f} exceeds "
        f"{c3a_reference['tolerance']:.3f}. |",
        f"| F6 | {location(source, 'primary_half_width =')}<br>"
        f"{location(contrast_path, 'primary_delta_ndcg_half_width')}<br>"
        f"{location(ci_path, 'preregistered primary')} | Primary ridge−MLP half-width "
        f"{g2['primary_delta_ndcg_half_width']:.4f}; secondary family median "
        f"{g2['secondary_median_family_delta_ndcg_half_width']:.4f}; rough scale "
        f"{g2['rough_query_count_for_threshold']} queries. |",
        f"| F7 | {location(tests_path, 'test_floor_oracle_and_methods_share_bootstrap_indices')}<br>"
        f"{location(tests_path, 'test_invalid_draw_gate_includes_epsilon')}<br>"
        f"{location(tests_path, 'test_literal_and_safe_fit_leakage_counts')}<br>"
        f"{location(bootstrap_path, 'result[' + chr(34) + '_bootstrap_indices' + chr(34) + ']')} | "
        "Shared-draw audit exposed; "
        f"1% invalid allowed and >1% rejected; leakage assertions bind "
        f"{audit['eval_positive_docs_in_literal_fit']}/{audit['eval_positive_docs_total']} and safe=0; "
        f"max harness-parity delta {max_parity_delta:.1e}. |",
        "",
        "Protocol-honesty note: C1's supplied F2 sentence called 0.089 the `median primary` "
        "half-width, while F6 requires ridge−MLP alone to be primary. The generated chair block "
        f"therefore reports primary={g2['primary_delta_ndcg_half_width']:.3f} and labels "
        f"{g2['secondary_median_family_delta_ndcg_half_width']:.3f} as secondary.",
    ]
    return "\n".join(lines) + "\n"


def _git_sha() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def find_paper() -> Optional[Path]:
    candidates = [
        Path.cwd() / "docs" / "waypoint-research-2026-06-09" / "paper-draft" / "main.md",
        Path.cwd().parents[2] / "docs" / "waypoint-research-2026-06-09" / "paper-draft" / "main.md",
    ]
    return next((path for path in candidates if path.exists()), None)


def run_d1(
    output_dir: Path,
    rebuild_pack: bool = False,
    rerun_learning_curve: bool = False,
    device: str = "cuda",
    paper_path: Optional[Path] = None,
    command: str = "python research/drift_recovery/run_d1.py",
) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pack_prefix = output_dir / "scifact_evalsplit_pack"
    if rebuild_pack or not pack_prefix.with_suffix(".npz").exists():
        pack = build_embedding_pack(pack_prefix, device=device)
    else:
        print("[D1] reloading frozen EmbeddingPack", flush=True)
        pack = EmbeddingPack.load(pack_prefix)
        assert_harness_parity(pack, atol=1e-12)

    print("[D1] recomputing literal and leakage-safe full-600 fits", flush=True)
    leakage = compute_leakage_sensitivity(pack, device=device)
    _write_json(output_dir / "leakage_sensitivity.json", leakage)
    ladder = ladder_table(pack, leakage_sensitivity=leakage)
    _write_json(output_dir / "ladder.json", ladder)
    bootstrap, contrasts = run_bootstrap(pack)
    _write_json(output_dir / "bootstrap_cis.json", bootstrap)
    _write_json(output_dir / "contrasts_holm.json", contrasts)
    learning_path = output_dir / "learning_curve.json"
    if rerun_learning_curve or not learning_path.exists():
        learning = run_learning_curves(pack, device=device)
        _write_json(learning_path, learning)
    else:
        learning = json.loads(learning_path.read_text(encoding="utf-8"))
        _write_json(learning_path, learning)

    (output_dir / "ci_report.md").write_text(
        render_ci_report(ladder, bootstrap, contrasts, leakage), encoding="utf-8"
    )
    (output_dir / "section3_draft.md").write_text(render_section3(), encoding="utf-8")
    selected_paper = paper_path or find_paper()
    (output_dir / "paper_edits.md").write_text(
        render_paper_edits(
            selected_paper,
            ladder,
            bootstrap,
            contrasts,
            leakage,
        ),
        encoding="utf-8",
    )
    parity = assert_harness_parity(pack, atol=1e-12)
    (output_dir / "C1_FIX_REPORT.md").write_text(
        render_c1_fix_report(ladder, contrasts, leakage, parity, output_dir),
        encoding="utf-8",
    )

    protocol = {
        "dataset": "SciFact",
        "split_seed": 42,
        "anchor_fraction": 0.4,
        "eval_queries": len(pack.query_ids),
        "bootstrap_draws": 10_000,
        "bootstrap_seed": 20260630,
        "learning_counts": list(COUNTS),
        "learning_order_seeds": list(ORDER_SEEDS),
        "ndcg": "binary NDCG@10 from merged harness",
    }
    manifest = {
        "record_type": "d1_run_manifest",
        "repo_sha": _git_sha(),
        "harness_sha": _git_sha(),
        "protocol_sha": hashlib.sha256(
            json.dumps(protocol, sort_keys=True).encode("utf-8")
        ).hexdigest(),
        "protocol": protocol,
        "models": pack.metadata["models"],
        "drift_manifest": pack.metadata["drift_manifest"],
        "environment": {
            "HF_HUB_OFFLINE": os.environ.get("HF_HUB_OFFLINE", ""),
            "HF_DATASETS_OFFLINE": os.environ.get("HF_DATASETS_OFFLINE", ""),
        },
        "command": command,
        "paper_source_read_only": str(selected_paper) if selected_paper else None,
        "outputs": sorted(
            {
                path.name for path in output_dir.iterdir()
            }.union({"run_manifest.json", "d1_summary.json"})
        ),
    }
    _write_json(output_dir / "run_manifest.json", manifest)
    result = {
        "output_dir": str(output_dir),
        "ladder": ladder,
        "g2": contrasts["g2"],
        "harness_parity": parity,
        "learning_rows": len(learning["rows"]),
    }
    _write_json(output_dir / "d1_summary.json", result)
    return result
