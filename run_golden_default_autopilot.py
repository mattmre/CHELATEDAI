"""Resumable autonomous search loop for a positive default candidate.

Contract-module scope note
--------------------------
``evidence_contract``, ``promotion_contract``, ``compute_budget_policy``, and
``evaluator_fabric`` are **intentionally not imported here**.  Those modules
define the shared evidence/promotion/compute infrastructure used by the
*Model-Scope campaign* (``run_model_scope_campaign.py``), which operates on
cross-run artifact bundles and promotion decisions.

This supervisor operates at a lower level: it collects Engine-Scope rows,
trains gate classifiers, runs per-iteration validation windows, and emits a
terminal decision JSON.  It does not produce evidence bundles, promotion
decisions, or compute-budget decisions — that aggregation belongs to the
Model-Scope layer that consumes this supervisor's output.  Importing those
contracts here would create a false dependency and could suggest they are
wired when they are not.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, Iterable, List, Sequence

from adaptive_overlay import build_overlay_report
from benchmark_utils import canonicalize_id, isolated_adapter_state, load_mteb_data
from embedding_backend import create_embedding_backend
from engine_scope import (
    ENGINE_SCOPE_SCHEMA_VERSION,
    build_engine_scope_rows,
    load_engine_scope_rows,
    summarize_engine_scope_rows,
)
from engine_scope_coverage import (
    select_task_offset_candidates,
    summarize_engine_scope_coverage,
)
from engine_scope_negatives import (
    build_hard_negative_replay_artifact,
    mine_hard_negative_families,
    select_query_subset_by_ids,
)
from learned_mask_gate import (
    enrich_mask_rows,
    filter_mask_rows,
    load_mask_example_rows,
    train_mask_gate,
    write_gate_config as write_mask_gate_config,
)
from learned_reformulation_gate import (
    enrich_query_rows,
    filter_reformulation_rows,
    load_query_attribution_rows,
    train_reformulation_gate,
    write_gate_config as write_reform_gate_config,
)
from run_road_course_campaign import RoadCourseProfile, evaluate_rankings
from run_road_course_tuning_loop import profile_summary
from run_thousand_query_tuning import (
    LoopSpec,
    PHASE_LOOP_TEMPLATES,
    build_query_attribution_rows,
    evaluate_profile_with_cache,
    run_thousand_query_cycle,
    select_query_window,
)
from static_mask_probe import (
    _mask_example_rows,
    _embed_mapping,
    _split_mapping,
    evaluate_conditional_mask,
    learn_harmful_dimension_mask,
    rank_by_cosine,
    run_static_mask_probe,
)


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_REFORM_ARTIFACTS = [
    Path("experiment_runs/roadcourse-small/attribution_probe_100.json"),
    Path("experiment_runs/roadcourse-small/selective_reform_probe_100.json"),
    Path("experiment_runs/roadcourse-small/reform_policy_probe_100.json"),
]


@dataclass(frozen=True)
class EvalSpec:
    task: str
    seed: int
    query_offset: int = 0


DEFAULT_EVAL_SPECS = [
    EvalSpec("SciFact", seed=170, query_offset=200),
    EvalSpec("NFCorpus", seed=171, query_offset=200),
    EvalSpec("FiQA2018", seed=172, query_offset=600),
]


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _now_iso() -> str:
    return _now_utc().isoformat()


def _timestamp_slug() -> str:
    return _now_utc().strftime("%Y%m%d-%H%M%S")


def _write_json_atomic(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    tmp_path.replace(path)


def _append_jsonl(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload) + "\n")


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_manifest(run_dir: Path) -> Dict[str, Any]:
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        return {}
    return _load_json(manifest_path)


def _update_manifest(run_dir: Path, manifest: Dict[str, Any]) -> None:
    _write_json_atomic(run_dir / "manifest.json", manifest)


def _record_event(run_dir: Path, kind: str, **fields: Any) -> None:
    _append_jsonl(
        run_dir / "events.jsonl",
        {
            "at": _now_iso(),
            "kind": kind,
            **fields,
        },
    )


def _phase_complete(info: Any) -> bool:
    return isinstance(info, dict) and info.get("status") in {"completed", "recovered", "recovered_partial"}


def _heartbeat(manifest: Dict[str, Any], run_dir: Path, *, iteration: int | None, phase: str | None) -> None:
    manifest["heartbeat_at"] = _now_iso()
    manifest["pid"] = os.getpid()
    manifest["current_iteration"] = iteration
    manifest["current_phase"] = phase
    _update_manifest(run_dir, manifest)


def _phase_output_status(path: Path) -> str | None:
    if not path.exists():
        return None
    try:
        payload = _load_json(path)
    except json.JSONDecodeError:
        return None
    status = payload.get("status")
    if isinstance(status, str):
        return status
    return "completed"


def _iteration_key(iteration: int) -> str:
    return f"iter_{iteration:03d}"


def _safe_default() -> Dict[str, Any]:
    return {
        "retrieval_profile": "baseline / FAST retrieval",
        "use_centering": False,
        "use_quantization": False,
        "chelation_p": 85,
        "chelation_threshold": 0.01,
        "adaptive_thresholding": False,
        "query_reformulation": "off by default",
        "query_masking": "off by default",
        "self_healing": "advisory only",
    }


def _judged_query_count(
    queries: Dict[Any, Any],
    qrels: Dict[Any, Dict[Any, float]],
) -> int:
    count = 0
    for query_id in queries:
        canonical_query_id = canonicalize_id(query_id)
        relevance = qrels.get(query_id, qrels.get(canonical_query_id, {}))
        if any(float(score) > 0 for score in relevance.values()):
            count += 1
    return count


def _resolve_eval_specs(
    specs: Sequence[EvalSpec],
    *,
    max_queries: int,
) -> List[EvalSpec]:
    if max_queries <= 0:
        raise ValueError("max_queries must be positive")

    resolved_specs: List[EvalSpec] = []
    for spec in specs:
        _corpus, queries, qrels = load_mteb_data(spec.task)
        judged_count = _judged_query_count(queries, qrels)
        if judged_count < max_queries:
            raise ValueError(
                f"{spec.task} has only {judged_count} judged queries, fewer than requested {max_queries}"
            )
        max_offset = max(0, judged_count - max_queries)
        resolved_specs.append(
            EvalSpec(
                task=spec.task,
                seed=spec.seed,
                query_offset=min(int(spec.query_offset), max_offset),
            )
        )
    return resolved_specs


def _mask_collection_specs(iteration: int, base_seed: int) -> List[EvalSpec]:
    sci_offsets = [0, 100]
    nfc_offsets = [0, 100]
    fiqa_offsets = [0, 200, 400]
    return [
        EvalSpec("SciFact", seed=base_seed + iteration, query_offset=sci_offsets[(iteration - 1) % len(sci_offsets)]),
        EvalSpec("NFCorpus", seed=base_seed + iteration, query_offset=nfc_offsets[(iteration - 1) % len(nfc_offsets)]),
        EvalSpec("FiQA2018", seed=base_seed + iteration, query_offset=fiqa_offsets[(iteration - 1) % len(fiqa_offsets)]),
    ]


def _reform_collection_specs(
    iteration: int,
    *,
    phase_queries: int,
    loop_queries: int,
    base_seed: int,
) -> List[LoopSpec]:
    """Rotate through the full road-course template set across iterations."""

    if phase_queries <= 0:
        raise ValueError("phase_queries must be positive")
    if loop_queries <= 0:
        raise ValueError("loop_queries must be positive")
    if phase_queries % loop_queries != 0:
        raise ValueError("phase_queries must divide evenly by loop_queries")

    loop_count = phase_queries // loop_queries
    start_index = (iteration - 1) * loop_count
    specs = []
    for local_index in range(loop_count):
        global_index = start_index + local_index
        task, offset = PHASE_LOOP_TEMPLATES[global_index % len(PHASE_LOOP_TEMPLATES)]
        seed = base_seed + (global_index // len(PHASE_LOOP_TEMPLATES))
        specs.append(LoopSpec(task=task, seed=seed, query_offset=offset))
    return specs


def _existing_reform_artifacts(run_dir: Path) -> List[Path]:
    artifacts = [path for path in DEFAULT_REFORM_ARTIFACTS if path.exists()]
    artifacts.extend(sorted((run_dir / "reformulation").glob("*.json")))
    return artifacts


def _existing_mask_artifacts(run_dir: Path) -> List[Path]:
    return sorted((run_dir / "mask").glob("*.json"))


def _existing_iteration_reports(run_dir: Path) -> List[Path]:
    return sorted((run_dir / "reports").glob("iteration_*.json"))


def _rows_within_run_dir(rows: Iterable[Dict[str, Any]], run_dir: Path) -> List[Dict[str, Any]]:
    run_root = run_dir.resolve()
    run_root_key = _path_scope_key(str(run_dir))
    scoped: List[Dict[str, Any]] = []
    for row in rows:
        artifact_path = row.get("_artifact_path")
        if not artifact_path:
            continue
        artifact_key = _path_scope_key(str(artifact_path))
        if artifact_key == run_root_key or artifact_key.startswith(f"{run_root_key}/"):
            scoped.append(dict(row))
            continue
        try:
            resolved = Path(str(artifact_path)).resolve()
        except (OSError, RuntimeError):
            continue
        if resolved == run_root or run_root in resolved.parents:
            scoped.append(dict(row))
    return scoped


def _path_scope_key(path_value: str) -> str:
    return path_value.replace("\\", "/").rstrip("/").lower()


def _deserialize_loop_specs(rows: Iterable[Dict[str, Any]]) -> List[LoopSpec]:
    return [
        LoopSpec(
            task=str(row["task"]),
            seed=int(row["seed"]),
            query_offset=int(row["query_offset"]),
        )
        for row in rows
    ]


def _campaign_reform_collection_rows(run_dir: Path) -> List[Dict[str, Any]]:
    reform_artifacts = sorted((run_dir / "reformulation").glob("*.json"))
    if not reform_artifacts:
        return []
    try:
        rows = load_engine_scope_rows(reform_artifacts)
    except ValueError:
        return []
    return [
        row
        for row in _rows_within_run_dir(rows, run_dir)
        if str(row.get("source_family") or "") == "reformulation_collection"
    ]


def _plan_reform_collection(
    run_dir: Path,
    *,
    iteration: int,
    phase_queries: int,
    loop_queries: int,
    base_seed: int,
    selection_mode: str,
) -> Dict[str, Any]:
    loop_count = phase_queries // loop_queries
    if selection_mode == "coverage_aware":
        campaign_rows = _campaign_reform_collection_rows(run_dir)
        if campaign_rows:
            coverage_summary = summarize_engine_scope_coverage(campaign_rows)
            ranked = select_task_offset_candidates(
                PHASE_LOOP_TEMPLATES,
                coverage_summary,
                count=len(PHASE_LOOP_TEMPLATES),
                rotation_offset=iteration - 1,
            )
            selected = ranked[:loop_count]
            if len(selected) == loop_count:
                seed = base_seed + iteration - 1
                loop_specs = [
                    LoopSpec(task=item["task"], seed=seed, query_offset=int(item["query_offset"]))
                    for item in selected
                ]
                return {
                    "selection_mode": "coverage_aware",
                    "coverage_row_count": len(campaign_rows),
                    "coverage_window_count": coverage_summary["window_count"],
                    "coverage_summary": {
                        "window_count": coverage_summary["window_count"],
                        "novel_window_count": coverage_summary["novel_window_count"],
                        "redundant_window_count": coverage_summary["redundant_window_count"],
                        "top_task_offsets": coverage_summary["task_offset_summary"][:5],
                    },
                    "candidate_scores": ranked,
                    "loop_specs": loop_specs,
                }
        selection_mode = "round_robin_fallback"

    loop_specs = _reform_collection_specs(
        iteration,
        phase_queries=phase_queries,
        loop_queries=loop_queries,
        base_seed=base_seed,
    )
    return {
        "selection_mode": selection_mode,
        "coverage_row_count": 0,
        "coverage_window_count": 0,
        "coverage_summary": None,
        "candidate_scores": [],
        "loop_specs": loop_specs,
    }


def _select_best_gate_attempt(
    rows: Iterable[Dict[str, Any]],
    *,
    trainer,
    attempt_count: int = 5,
    extra_fields: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    rows_list = [dict(row) for row in rows]
    attempts = []
    for remainder in range(attempt_count):
        config = trainer(rows_list, holdout_remainder=remainder)
        record = copy.deepcopy(config)
        record["holdout_remainder"] = remainder
        attempts.append(record)
    accepted = [item for item in attempts if item.get("gate") is not None]
    if accepted:
        accepted.sort(
            key=lambda item: (
                -float(item["accepted"][0]["holdout"]["mean_delta_ndcg_at_10"]),
                -int(item["accepted"][0]["holdout"]["matched"]),
                -float(item["accepted"][0]["train"]["mean_delta_ndcg_at_10"]),
            )
        )
        selected = copy.deepcopy(accepted[0])
    else:
        attempts.sort(
            key=lambda item: (
                -int(item.get("training_summary", {}).get("positive_examples", 0)),
                -int(item.get("training_summary", {}).get("train_rows", 0)),
            )
        )
        selected = copy.deepcopy(attempts[0]) if attempts else {"gate": None, "accepted": []}
    selected["attempts"] = attempts
    if extra_fields:
        selected.update(extra_fields)
    return selected


def _reform_validation_profiles(gate: Dict[str, Any] | None) -> List[RoadCourseProfile]:
    return [
        RoadCourseProfile("baseline"),
        RoadCourseProfile("guard_p85_t0.01", use_quantization=True, chelation_p=85, chelation_threshold=0.01),
        RoadCourseProfile("reform_rrf_v2", query_reformulation_variants=2, query_reformulation_policy="always"),
        RoadCourseProfile(
            "learned_reform_gate_v1",
            query_reformulation_variants=2,
            query_reformulation_policy=gate if gate is not None else "never",
        ),
        RoadCourseProfile(
            "guard_reform_rrf_v2",
            use_quantization=True,
            chelation_p=85,
            chelation_threshold=0.01,
            query_reformulation_variants=2,
            query_reformulation_policy="always",
        ),
        RoadCourseProfile(
            "guard_learned_reform_gate_v1",
            use_quantization=True,
            chelation_p=85,
            chelation_threshold=0.01,
            query_reformulation_variants=2,
            query_reformulation_policy=gate if gate is not None else "never",
        ),
    ]


def run_reform_validation(
    gate: Dict[str, Any] | None,
    *,
    model: str,
    specs: Sequence[EvalSpec],
    max_queries: int,
    sample_docs: int,
) -> Dict[str, Any]:
    windows = []
    for loop_index, spec in enumerate(specs, start=1):
        corpus, queries, qrels = load_mteb_data(spec.task)
        selected_corpus, selected_queries, selected_qrels = select_query_window(
            corpus,
            queries,
            qrels,
            query_offset=spec.query_offset,
            max_queries=max_queries,
            sample_docs=sample_docs,
            seed=spec.seed,
        )
        profiles = _reform_validation_profiles(gate)
        with isolated_adapter_state():
            engine_cache: Dict[str, Any] = {}
            try:
                profile_results = [
                    evaluate_profile_with_cache(
                        profile,
                        model,
                        selected_corpus,
                        selected_queries,
                        selected_qrels,
                        engine_cache,
                    )
                    for profile in profiles
                ]
            finally:
                for engine in engine_cache.values():
                    engine.close()
        summary = profile_summary(profile_results)
        windows.append({
            "loop": loop_index,
            "window": loop_index,
            "global_window": loop_index,
            "task": spec.task,
            "seed": spec.seed,
            "query_offset": spec.query_offset,
            "query_count": len(selected_queries),
            "corpus_size": len(selected_corpus),
            "queries": selected_queries,
            "profiles_evaluated": [asdict(profile) for profile in profiles],
            "summary": summary,
            "profile_results": profile_results,
        })

    baseline_guard_deltas = []
    always_deltas = []
    learned_deltas = []
    guard_always_deltas = []
    guard_learned_deltas = []
    learned_blockers = 0
    guard_learned_blockers = 0
    for window in windows:
        rows = {row["profile"]: row for row in window["summary"]["ranked_profiles"]}
        baseline_guard_deltas.append(float(rows["guard_p85_t0.01"]["delta_vs_baseline"]))
        always_deltas.append(float(rows["reform_rrf_v2"]["delta_vs_baseline"]))
        learned_deltas.append(float(rows["learned_reform_gate_v1"]["delta_vs_baseline"]))
        guard_always_deltas.append(
            float(rows["guard_reform_rrf_v2"]["ndcg_at_10"]) - float(rows["guard_p85_t0.01"]["ndcg_at_10"])
        )
        guard_learned_deltas.append(
            float(rows["guard_learned_reform_gate_v1"]["ndcg_at_10"]) - float(rows["guard_p85_t0.01"]["ndcg_at_10"])
        )
        learned_blockers += int(rows["learned_reform_gate_v1"]["fault_classification"]["promotion_blocker"])
        guard_learned_blockers += int(rows["guard_learned_reform_gate_v1"]["fault_classification"]["promotion_blocker"])
    query_attribution_rows = build_query_attribution_rows(windows)
    engine_scope_rows = build_engine_scope_rows(
        query_attribution_rows=query_attribution_rows,
        source_family="reformulation_validation",
    )
    return {
        "engine_scope_schema_version": ENGINE_SCOPE_SCHEMA_VERSION,
        "model": model,
        "max_queries": max_queries,
        "sample_docs": sample_docs,
        "windows": windows,
        "query_attribution_rows": query_attribution_rows,
        "engine_scope_rows": engine_scope_rows,
        "adaptive_overlay": build_overlay_report(engine_scope_rows),
        "aggregate": {
            "guard_mean_delta_vs_baseline": sum(baseline_guard_deltas) / len(baseline_guard_deltas) if baseline_guard_deltas else 0.0,
            "always_on_reform_mean_delta_vs_baseline": sum(always_deltas) / len(always_deltas) if always_deltas else 0.0,
            "learned_reform_mean_delta_vs_baseline": sum(learned_deltas) / len(learned_deltas) if learned_deltas else 0.0,
            "guard_always_reform_mean_delta_vs_guard": sum(guard_always_deltas) / len(guard_always_deltas) if guard_always_deltas else 0.0,
            "guard_learned_reform_mean_delta_vs_guard": sum(guard_learned_deltas) / len(guard_learned_deltas) if guard_learned_deltas else 0.0,
            "learned_reform_promotion_blockers": learned_blockers,
            "guard_learned_reform_promotion_blockers": guard_learned_blockers,
            "guard_learned_beats_guard": sum(delta > 0.001 for delta in guard_learned_deltas),
            "guard_learned_beats_guard_always": sum(
                learned > always
                for learned, always in zip(guard_learned_deltas, guard_always_deltas)
            ),
        },
    }


def _recommendation(
    reform_gate: Dict[str, Any] | None,
    reform_validation: Dict[str, Any],
    mask_gate: Dict[str, Any] | None,
    mask_validation: Dict[str, Any],
    *,
    reform_hard_negative_validation: Dict[str, Any] | None = None,
    coverage_summary: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    overlay_reports = [
        report
        for report in (
            reform_validation.get("adaptive_overlay") if isinstance(reform_validation, dict) else None,
            mask_validation.get("adaptive_overlay") if isinstance(mask_validation, dict) else None,
            reform_hard_negative_validation.get("adaptive_overlay")
            if isinstance(reform_hard_negative_validation, dict)
            else None,
        )
        if isinstance(report, dict)
    ]
    overlay_ready = any(
        bool((report.get("readiness") or {}).get("ready_for_broader_validation"))
        for report in overlay_reports
    )
    overlay_blockers = sorted({
        str(blocker)
        for report in overlay_reports
        for blocker in (report.get("readiness") or {}).get("blockers", [])
    })
    reform_stress_clean = True
    reform_hard_negative_family_count = 0
    if reform_hard_negative_validation:
        reform_hard_negative_family_count = int(reform_hard_negative_validation.get("family_count", 0))
        if reform_hard_negative_family_count > 0:
            reform_stress_clean = (
                int(reform_hard_negative_validation.get("guard_learned_negative_families", 0)) == 0
                and int(reform_hard_negative_validation.get("guard_learned_reform_promotion_blockers", 0)) == 0
                and float(reform_hard_negative_validation.get("guard_learned_reform_mean_delta_vs_guard", 0.0)) >= -0.001
            )
    reform_candidate = (
        reform_gate is not None
        and float(reform_validation.get("guard_learned_reform_mean_delta_vs_guard", 0.0)) > 0.001
        and int(reform_validation.get("guard_learned_reform_promotion_blockers", 1)) == 0
        and reform_stress_clean
    )
    mask_candidate = (
        mask_gate is not None
        and float(mask_validation.get("learned_gate_mean_delta_vs_baseline", 0.0)) > 0.001
        and int(mask_validation.get("learned_gate_negative_windows", 1)) == 0
    )
    coverage_guided_collection = bool(coverage_summary and int(coverage_summary.get("window_count", 0)) > 0)
    return {
        "default_change_allowed": bool(reform_candidate or mask_candidate),
        "safe_default_holds": not bool(reform_candidate or mask_candidate),
        "reform_gate_candidate_for_broader_validation": reform_candidate,
        "reform_gate_survived_hard_negative_replay": reform_stress_clean,
        "reform_hard_negative_family_count": reform_hard_negative_family_count,
        "mask_gate_candidate_for_broader_validation": mask_candidate,
        "adaptive_overlay_ready_for_broader_validation": overlay_ready,
        "adaptive_overlay_blockers": overlay_blockers,
        "coverage_guided_collection_recommended": coverage_guided_collection,
        "next_action": (
            "expand repeatability and transfer validation for the surviving gate candidate"
            if reform_candidate or mask_candidate
            else (
                "continue coverage-aware collection and fail-closed gate search"
                if coverage_guided_collection
                else "continue pooled data collection and fail-closed gate search"
            )
        ),
    }


def run_mask_validation(
    gate: Dict[str, Any] | None,
    *,
    model: str,
    specs: Sequence[EvalSpec],
    max_queries: int,
    train_queries: int,
    sample_docs: int,
    mask_fraction: float,
) -> Dict[str, Any]:
    windows = []
    for loop_index, spec in enumerate(specs, start=1):
        with isolated_adapter_state():
            corpus, queries, qrels = load_mteb_data(spec.task)
            selected_corpus, selected_queries, selected_qrels = select_query_window(
                corpus,
                queries,
                qrels,
                query_offset=spec.query_offset,
                max_queries=max_queries,
                sample_docs=sample_docs,
                seed=spec.seed,
            )
            query_ids = list(selected_queries)
            if not 0 < train_queries < len(query_ids):
                raise ValueError("train_queries must leave at least one holdout query")
            train_ids = query_ids[:train_queries]
            holdout_ids = query_ids[train_queries:]
            backend = create_embedding_backend(model)
            doc_embeddings = _embed_mapping(backend, selected_corpus)
            query_embeddings = _embed_mapping(backend, selected_queries)
            train_query_embeddings = _split_mapping(query_embeddings, train_ids)
            holdout_query_embeddings = _split_mapping(query_embeddings, holdout_ids)
            train_qrels = _split_mapping(selected_qrels, train_ids)
            holdout_qrels = _split_mapping(selected_qrels, holdout_ids)
            learned_mask = learn_harmful_dimension_mask(
                train_query_embeddings,
                doc_embeddings,
                train_qrels,
                mask_fraction=mask_fraction,
            )
            holdout_baseline = evaluate_rankings(rank_by_cosine(holdout_query_embeddings, doc_embeddings), holdout_qrels)
            holdout_masked = evaluate_rankings(
                rank_by_cosine(holdout_query_embeddings, doc_embeddings, mask=learned_mask["mask"]),
                holdout_qrels,
            )
            holdout_conditional = evaluate_conditional_mask(
                holdout_query_embeddings,
                doc_embeddings,
                holdout_qrels,
                learned_mask["mask"],
                gate,
                query_texts=_split_mapping(selected_queries, holdout_ids),
            )
        always_delta = holdout_masked["ndcg_at_10"] - holdout_baseline["ndcg_at_10"]
        learned_delta = holdout_conditional["metrics"]["ndcg_at_10"] - holdout_baseline["ndcg_at_10"]
        windows.append({
            "loop": loop_index,
            "task": spec.task,
            "seed": spec.seed,
            "query_offset": spec.query_offset,
            "query_count": len(selected_queries),
            "train_queries": len(train_ids),
            "holdout_queries": len(holdout_ids),
            "corpus_size": len(selected_corpus),
            "mask_count": learned_mask["mask_count"],
            "mask_example_rows": _mask_example_rows(
                holdout_conditional["query_examples"],
                task=spec.task,
                seed=spec.seed,
                query_offset=spec.query_offset,
                split="validation_holdout",
            ),
            "holdout": {
                "baseline": holdout_baseline,
                "always_masked": holdout_masked,
                "learned_gate": holdout_conditional,
                "always_masked_delta_vs_baseline": always_delta,
                "learned_gate_delta_vs_baseline": learned_delta,
            },
        })
    always_deltas = [float(window["holdout"]["always_masked_delta_vs_baseline"]) for window in windows]
    learned_deltas = [float(window["holdout"]["learned_gate_delta_vs_baseline"]) for window in windows]
    mask_example_rows = [
        row
        for window in windows
        for row in window.get("mask_example_rows", [])
    ]
    engine_scope_rows = build_engine_scope_rows(
        mask_example_rows=mask_example_rows,
        source_family="mask_validation",
    )
    return {
        "engine_scope_schema_version": ENGINE_SCOPE_SCHEMA_VERSION,
        "model": model,
        "max_queries": max_queries,
        "train_queries": train_queries,
        "sample_docs": sample_docs,
        "mask_fraction": mask_fraction,
        "windows": windows,
        "mask_example_rows": mask_example_rows,
        "engine_scope_rows": engine_scope_rows,
        "adaptive_overlay": build_overlay_report(engine_scope_rows),
        "aggregate": {
            "always_on_mask_mean_delta_vs_baseline": sum(always_deltas) / len(always_deltas) if always_deltas else 0.0,
            "learned_gate_mean_delta_vs_baseline": sum(learned_deltas) / len(learned_deltas) if learned_deltas else 0.0,
            "always_on_mask_negative_windows": sum(delta < -0.001 for delta in always_deltas),
            "learned_gate_negative_windows": sum(delta < -0.001 for delta in learned_deltas),
            "learned_gate_beats_always_on": sum(learned > always for learned, always in zip(learned_deltas, always_deltas)),
            "learned_gate_beats_baseline": sum(delta > 0.001 for delta in learned_deltas),
            "learned_gate_applied_queries": sum(
                int(window["holdout"]["learned_gate"]["applied_queries"])
                for window in windows
            ),
        },
    }


def run_reform_hard_negative_replay(
    gate: Dict[str, Any] | None,
    *,
    families: Sequence[Dict[str, Any]],
    model: str,
    sample_docs: int,
) -> Dict[str, Any]:
    windows = []
    for family_index, family in enumerate(families, start=1):
        query_ids = [str(query_id) for query_id in family.get("replay_query_ids", []) if str(query_id)]
        if not query_ids:
            continue
        corpus, queries, qrels = load_mteb_data(str(family["task"]))
        selected_corpus, selected_queries, selected_qrels = select_query_subset_by_ids(
            corpus,
            queries,
            qrels,
            query_ids,
            sample_docs=sample_docs,
            seed=1700 + family_index,
        )
        profiles = _reform_validation_profiles(gate)
        with isolated_adapter_state():
            engine_cache: Dict[str, Any] = {}
            try:
                profile_results = [
                    evaluate_profile_with_cache(
                        profile,
                        model,
                        selected_corpus,
                        selected_queries,
                        selected_qrels,
                        engine_cache,
                    )
                    for profile in profiles
                ]
            finally:
                for engine in engine_cache.values():
                    engine.close()
        summary = profile_summary(profile_results)
        windows.append({
            "loop": family_index,
            "window": family_index,
            "global_window": family_index,
            "task": str(family["task"]),
            "seed": 1700 + family_index,
            "query_offset": None,
            "query_count": len(selected_queries),
            "corpus_size": len(selected_corpus),
            "queries": selected_queries,
            "family_id": family.get("family_id"),
            "family_signature": family.get("signature"),
            "primary_fault_class": family.get("primary_fault_class"),
            "profiles_evaluated": [asdict(profile) for profile in profiles],
            "summary": summary,
            "profile_results": profile_results,
        })

    if not windows:
        return {
            "engine_scope_schema_version": ENGINE_SCOPE_SCHEMA_VERSION,
            "family_count": 0,
            "sample_docs": sample_docs,
            "windows": [],
            "query_attribution_rows": [],
            "engine_scope_rows": [],
            "aggregate": {
                "guard_always_reform_mean_delta_vs_guard": 0.0,
                "guard_learned_reform_mean_delta_vs_guard": 0.0,
                "guard_learned_negative_families": 0,
                "guard_learned_reform_promotion_blockers": 0,
                "guard_learned_beats_guard": 0,
                "guard_learned_beats_guard_always": 0,
            },
        }

    guard_always_deltas = []
    guard_learned_deltas = []
    guard_learned_blockers = 0
    for window in windows:
        rows = {row["profile"]: row for row in window["summary"]["ranked_profiles"]}
        guard_always_deltas.append(
            float(rows["guard_reform_rrf_v2"]["ndcg_at_10"]) - float(rows["guard_p85_t0.01"]["ndcg_at_10"])
        )
        guard_learned_deltas.append(
            float(rows["guard_learned_reform_gate_v1"]["ndcg_at_10"]) - float(rows["guard_p85_t0.01"]["ndcg_at_10"])
        )
        guard_learned_blockers += int(rows["guard_learned_reform_gate_v1"]["fault_classification"]["promotion_blocker"])
    query_attribution_rows = build_query_attribution_rows(windows)
    engine_scope_rows = build_engine_scope_rows(
        query_attribution_rows=query_attribution_rows,
        source_family="hard_negative_reform_validation",
    )
    return {
        "engine_scope_schema_version": ENGINE_SCOPE_SCHEMA_VERSION,
        "family_count": len(windows),
        "sample_docs": sample_docs,
        "windows": windows,
        "query_attribution_rows": query_attribution_rows,
        "engine_scope_rows": engine_scope_rows,
        "adaptive_overlay": build_overlay_report(engine_scope_rows),
        "aggregate": {
            "guard_always_reform_mean_delta_vs_guard": sum(guard_always_deltas) / len(guard_always_deltas),
            "guard_learned_reform_mean_delta_vs_guard": sum(guard_learned_deltas) / len(guard_learned_deltas),
            "guard_learned_negative_families": sum(delta < -0.001 for delta in guard_learned_deltas),
            "guard_learned_reform_promotion_blockers": guard_learned_blockers,
            "guard_learned_beats_guard": sum(delta > 0.001 for delta in guard_learned_deltas),
            "guard_learned_beats_guard_always": sum(
                learned > always
                for learned, always in zip(guard_learned_deltas, guard_always_deltas)
            ),
        },
    }


def _init_manifest(run_dir: Path, args: argparse.Namespace) -> Dict[str, Any]:
    deadline_at = _now_utc() + timedelta(hours=float(args.deadline_hours))
    return {
        "started_at": _now_iso(),
        "run_dir": str(run_dir),
        "pid": os.getpid(),
        "command": [sys.executable, *sys.argv],
        "status": "running",
        "deadline_at": deadline_at.isoformat(),
        "config": vars(args),
        "iterations": {},
    }


def _load_or_init_manifest(run_dir: Path, args: argparse.Namespace) -> Dict[str, Any]:
    manifest = _load_manifest(run_dir)
    if manifest:
        manifest.setdefault("resumed_at", _now_iso())
        manifest["pid"] = os.getpid()
        manifest["command"] = [sys.executable, *sys.argv]
        manifest["status"] = "running"
        return manifest
    return _init_manifest(run_dir, args)


def _recover_or_run_reform_collection(
    run_dir: Path,
    manifest: Dict[str, Any],
    iteration_info: Dict[str, Any],
    *,
    iteration: int,
    model: str,
    phase_queries: int,
    loop_queries: int,
    window_queries: int,
    sample_docs: int,
    base_seed: int,
    selection_mode: str,
) -> Dict[str, Any]:
    label = "collect_reformulation"
    output_path = run_dir / "reformulation" / f"reformulation_iter_{iteration:03d}.json"
    phase_info = iteration_info.setdefault("phases", {}).get(label)
    if _phase_complete(phase_info) and output_path.exists():
        return phase_info
    existing_status = _phase_output_status(output_path)
    if existing_status is not None:
        recovered = {
            "status": "recovered" if existing_status == "completed" else "recovered_partial",
            "output_path": str(output_path),
            "artifact_status": existing_status,
        }
        iteration_info["phases"][label] = recovered
        return recovered

    planned_specs = []
    persisted_phase = iteration_info.setdefault("phases", {}).get(label)
    if isinstance(persisted_phase, dict) and persisted_phase.get("planned_loop_specs"):
        planned_specs = _deserialize_loop_specs(persisted_phase["planned_loop_specs"])
        plan = {
            "selection_mode": str(persisted_phase.get("selection_mode") or selection_mode),
            "coverage_row_count": int(persisted_phase.get("coverage_row_count", 0)),
            "coverage_window_count": int(persisted_phase.get("coverage_window_count", 0)),
            "coverage_summary": persisted_phase.get("coverage_summary"),
            "candidate_scores": persisted_phase.get("candidate_scores", []),
            "loop_specs": planned_specs,
        }
    else:
        plan = _plan_reform_collection(
            run_dir,
            iteration=iteration,
            phase_queries=phase_queries,
            loop_queries=loop_queries,
            base_seed=base_seed,
            selection_mode=selection_mode,
        )
        planned_specs = plan["loop_specs"]
    iteration_info["phases"][label] = {
        "status": "running",
        "output_path": str(output_path),
        "started_at": _now_iso(),
        "selection_mode": plan["selection_mode"],
        "coverage_row_count": plan["coverage_row_count"],
        "coverage_window_count": plan["coverage_window_count"],
        "coverage_summary": plan["coverage_summary"],
        "candidate_scores": plan["candidate_scores"],
        "planned_loop_specs": [asdict(spec) for spec in planned_specs],
    }
    _heartbeat(manifest, run_dir, iteration=iteration, phase=label)
    result = run_thousand_query_cycle(
        model=model,
        loop_specs=planned_specs,
        loop_queries=loop_queries,
        window_queries=window_queries,
        sample_docs=sample_docs,
        checkpoint_path=output_path,
        strategy="reform_policy_search",
    )
    _write_json_atomic(output_path, result)
    completed = {
        "status": "completed",
        "output_path": str(output_path),
        "started_at": iteration_info["phases"][label]["started_at"],
        "finished_at": _now_iso(),
        "selection_mode": plan["selection_mode"],
        "coverage_row_count": plan["coverage_row_count"],
        "coverage_window_count": plan["coverage_window_count"],
        "coverage_summary": plan["coverage_summary"],
        "candidate_scores": plan["candidate_scores"],
        "loop_specs": [asdict(spec) for spec in planned_specs],
        "completed_queries": result["completed_queries"],
        "completed_windows": result["completed_windows"],
    }
    iteration_info["phases"][label] = completed
    return completed


def _recover_or_run_mask_collection(
    run_dir: Path,
    manifest: Dict[str, Any],
    iteration_info: Dict[str, Any],
    *,
    iteration: int,
    model: str,
    max_queries: int,
    train_queries: int,
    sample_docs: int,
    mask_fraction: float,
    base_seed: int,
) -> Dict[str, Any]:
    label = "collect_mask_probe"
    phase_info = iteration_info.setdefault("phases", {}).get(label)
    if _phase_complete(phase_info):
        return phase_info
    outputs = []
    specs = _mask_collection_specs(iteration, base_seed)
    for spec in specs:
        output_path = (
            run_dir / "mask"
            / f"mask_iter_{iteration:03d}_{spec.task}_seed{spec.seed}_offset{spec.query_offset}.json"
        )
        existing_status = _phase_output_status(output_path)
        if existing_status is not None:
            outputs.append({
                "task": spec.task,
                "seed": spec.seed,
                "query_offset": spec.query_offset,
                "status": "recovered" if existing_status == "completed" else "recovered_partial",
                "output_path": str(output_path),
            })
            continue
        _heartbeat(manifest, run_dir, iteration=iteration, phase=label)
        result = run_static_mask_probe(
            task=spec.task,
            model=model,
            query_offset=spec.query_offset,
            max_queries=max_queries,
            train_queries=train_queries,
            sample_docs=sample_docs,
            seed=spec.seed,
            mask_fraction=mask_fraction,
            classifier_gate=True,
        )
        _write_json_atomic(output_path, result)
        outputs.append({
            "task": spec.task,
            "seed": spec.seed,
            "query_offset": spec.query_offset,
            "status": "completed",
            "output_path": str(output_path),
            "conditional_promotion_candidate": result.get("conditional_promotion_candidate", False),
        })
    completed = {
        "status": "completed",
        "outputs": outputs,
        "finished_at": _now_iso(),
    }
    iteration_info["phases"][label] = completed
    return completed


def _run_iteration(
    run_dir: Path,
    manifest: Dict[str, Any],
    *,
    iteration: int,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    key = _iteration_key(iteration)
    iteration_info = manifest.setdefault("iterations", {}).setdefault(
        key,
        {
            "iteration": iteration,
            "started_at": _now_iso(),
            "status": "running",
            "phases": {},
        },
    )
    report: Dict[str, Any] = {
        "iteration": iteration,
        "started_at": iteration_info["started_at"],
        "engine_scope_schema_version": ENGINE_SCOPE_SCHEMA_VERSION,
        "golden_safe_default": _safe_default(),
    }
    validation_specs: List[EvalSpec] | None = None
    _record_event(run_dir, "iteration_started", iteration=iteration)

    try:
        report["reform_collection"] = _recover_or_run_reform_collection(
            run_dir,
            manifest,
            iteration_info,
            iteration=iteration,
            model=args.model,
            phase_queries=args.reform_phase_queries,
            loop_queries=args.reform_loop_queries,
            window_queries=args.reform_window_queries,
            sample_docs=args.reform_sample_docs,
            base_seed=args.reform_base_seed,
            selection_mode=args.reform_selection_mode,
        )
    except Exception as exc:
        report["reform_collection_error"] = {"type": type(exc).__name__, "message": str(exc)}
        _record_event(run_dir, "phase_error", iteration=iteration, phase="collect_reformulation", error=str(exc))

    try:
        reform_artifacts = _existing_reform_artifacts(run_dir)
        reform_rows = enrich_query_rows(load_query_attribution_rows(reform_artifacts))
        reform_filtered = filter_reformulation_rows(reform_rows, profile="reform_rrf_v2")
        reform_gate = _select_best_gate_attempt(
            reform_filtered,
            trainer=train_reformulation_gate,
            extra_fields={
                "profile": "reform_rrf_v2",
                "artifact_count": len(reform_artifacts),
                "pooled_rows": len(reform_filtered),
            },
        )
        reform_gate_output = run_dir / "reports" / f"learned_reform_gate_iter_{iteration:03d}.json"
        write_reform_gate_config(reform_gate, reform_gate_output)
        report["reform_training"] = {
            "output_path": str(reform_gate_output),
            **reform_gate,
        }
        iteration_info["phases"]["train_reform_gate"] = {
            "status": "completed",
            "output_path": str(reform_gate_output),
            "gate_present": reform_gate.get("gate") is not None,
            "finished_at": _now_iso(),
        }
    except Exception as exc:
        report["reform_training_error"] = {"type": type(exc).__name__, "message": str(exc)}
        _record_event(run_dir, "phase_error", iteration=iteration, phase="train_reform_gate", error=str(exc))
        reform_gate = {"gate": None}

    try:
        validation_specs = _resolve_eval_specs(
            DEFAULT_EVAL_SPECS,
            max_queries=max(args.validation_max_queries, args.mask_validation_max_queries),
        )
        report["validation_specs"] = [asdict(spec) for spec in validation_specs]
        reform_validation = run_reform_validation(
            reform_gate.get("gate"),
            model=args.model,
            specs=validation_specs,
            max_queries=args.validation_max_queries,
            sample_docs=args.validation_sample_docs,
        )
        report["reform_validation"] = reform_validation
        iteration_info["phases"]["validate_reform_gate"] = {
            "status": "completed",
            "finished_at": _now_iso(),
            "gate_present": reform_gate.get("gate") is not None,
            "guard_learned_mean_delta_vs_guard": reform_validation["aggregate"]["guard_learned_reform_mean_delta_vs_guard"],
        }
    except Exception as exc:
        report["reform_validation_error"] = {"type": type(exc).__name__, "message": str(exc)}
        _record_event(run_dir, "phase_error", iteration=iteration, phase="validate_reform_gate", error=str(exc))

    try:
        report["mask_collection"] = _recover_or_run_mask_collection(
            run_dir,
            manifest,
            iteration_info,
            iteration=iteration,
            model=args.model,
            max_queries=args.mask_max_queries,
            train_queries=args.mask_train_queries,
            sample_docs=args.mask_sample_docs,
            mask_fraction=args.mask_fraction,
            base_seed=args.mask_base_seed,
        )
    except Exception as exc:
        report["mask_collection_error"] = {"type": type(exc).__name__, "message": str(exc)}
        _record_event(run_dir, "phase_error", iteration=iteration, phase="collect_mask_probe", error=str(exc))

    try:
        mask_artifacts = _existing_mask_artifacts(run_dir)
        mask_rows = enrich_mask_rows(load_mask_example_rows(mask_artifacts))
        mask_filtered = filter_mask_rows(mask_rows, splits=("holdout",))
        mask_gate = _select_best_gate_attempt(
            mask_filtered,
            trainer=train_mask_gate,
            extra_fields={
                "artifact_count": len(mask_artifacts),
                "pooled_rows": len(mask_filtered),
                "splits": ["holdout"],
            },
        )
        mask_gate_output = run_dir / "reports" / f"learned_mask_gate_iter_{iteration:03d}.json"
        write_mask_gate_config(mask_gate, mask_gate_output)
        report["mask_training"] = {
            "output_path": str(mask_gate_output),
            **mask_gate,
        }
        iteration_info["phases"]["train_mask_gate"] = {
            "status": "completed",
            "output_path": str(mask_gate_output),
            "gate_present": mask_gate.get("gate") is not None,
            "finished_at": _now_iso(),
        }
    except Exception as exc:
        report["mask_training_error"] = {"type": type(exc).__name__, "message": str(exc)}
        _record_event(run_dir, "phase_error", iteration=iteration, phase="train_mask_gate", error=str(exc))
        mask_gate = {"gate": None}

    try:
        if validation_specs is None:
            validation_specs = _resolve_eval_specs(
                DEFAULT_EVAL_SPECS,
                max_queries=max(args.validation_max_queries, args.mask_validation_max_queries),
            )
            report["validation_specs"] = [asdict(spec) for spec in validation_specs]
        mask_validation = run_mask_validation(
            mask_gate.get("gate"),
            model=args.model,
            specs=validation_specs,
            max_queries=args.mask_validation_max_queries,
            train_queries=args.mask_validation_train_queries,
            sample_docs=args.mask_validation_sample_docs,
            mask_fraction=args.mask_fraction,
        )
        report["mask_validation"] = mask_validation
        iteration_info["phases"]["validate_mask_gate"] = {
            "status": "completed",
            "finished_at": _now_iso(),
            "gate_present": mask_gate.get("gate") is not None,
            "learned_gate_mean_delta_vs_baseline": mask_validation["aggregate"]["learned_gate_mean_delta_vs_baseline"],
        }
    except Exception as exc:
        report["mask_validation_error"] = {"type": type(exc).__name__, "message": str(exc)}
        _record_event(run_dir, "phase_error", iteration=iteration, phase="validate_mask_gate", error=str(exc))

    report["finished_at"] = _now_iso()
    report_path = run_dir / "reports" / f"iteration_{iteration:03d}.json"
    _write_json_atomic(report_path, report)
    pooled_engine_scope_rows: List[Dict[str, Any]] = []
    coverage_summary: Dict[str, Any] | None = None
    try:
        pooled_engine_scope_rows = load_engine_scope_rows([
            *_existing_reform_artifacts(run_dir),
            *_existing_mask_artifacts(run_dir),
            *_existing_iteration_reports(run_dir),
        ])
        pooled_engine_scope_output = run_dir / "reports" / f"engine_scope_pool_iter_{iteration:03d}.json"
        pooled_engine_scope_payload = {
            "engine_scope_schema_version": ENGINE_SCOPE_SCHEMA_VERSION,
            "iteration": iteration,
            "generated_at": _now_iso(),
            "source_artifacts": [
                str(path)
                for path in [
                    *_existing_reform_artifacts(run_dir),
                    *_existing_mask_artifacts(run_dir),
                    *_existing_iteration_reports(run_dir),
                ]
            ],
            "summary": summarize_engine_scope_rows(pooled_engine_scope_rows),
            "engine_scope_rows": pooled_engine_scope_rows,
        }
        _write_json_atomic(pooled_engine_scope_output, pooled_engine_scope_payload)
        report["engine_scope_pool"] = {
            "output_path": str(pooled_engine_scope_output),
            **pooled_engine_scope_payload["summary"],
        }
        coverage_output = run_dir / "reports" / f"engine_scope_coverage_iter_{iteration:03d}.json"
        coverage_rows = _rows_within_run_dir(pooled_engine_scope_rows, run_dir)
        coverage_scope = "campaign_only" if coverage_rows else "pooled_fallback"
        coverage_input_rows = coverage_rows if coverage_rows else pooled_engine_scope_rows
        coverage_summary = summarize_engine_scope_coverage(coverage_input_rows)
        _write_json_atomic(
            coverage_output,
            {
                "engine_scope_schema_version": ENGINE_SCOPE_SCHEMA_VERSION,
                "iteration": iteration,
                "generated_at": _now_iso(),
                "coverage_scope": coverage_scope,
                "input_row_count": len(coverage_input_rows),
                "bootstrap_row_count": len(pooled_engine_scope_rows) - len(coverage_rows),
                **coverage_summary,
            },
        )
        report["engine_scope_coverage"] = {
            "output_path": str(coverage_output),
            "coverage_scope": coverage_scope,
            "input_row_count": len(coverage_input_rows),
            "bootstrap_row_count": len(pooled_engine_scope_rows) - len(coverage_rows),
            "window_count": coverage_summary["window_count"],
            "novel_window_count": coverage_summary["novel_window_count"],
            "redundant_window_count": coverage_summary["redundant_window_count"],
            "top_overlap_pairs": coverage_summary["top_overlap_pairs"][:3],
            "top_task_offsets": coverage_summary["task_offset_summary"][:5],
        }
        _write_json_atomic(report_path, report)
        iteration_info["phases"]["pool_engine_scope"] = {
            "status": "completed",
            "output_path": str(pooled_engine_scope_output),
            "row_count": pooled_engine_scope_payload["summary"]["row_count"],
            "finished_at": _now_iso(),
        }
        iteration_info["phases"]["coverage_analysis"] = {
            "status": "completed",
            "output_path": str(coverage_output),
            "coverage_scope": coverage_scope,
            "window_count": coverage_summary["window_count"],
            "novel_window_count": coverage_summary["novel_window_count"],
            "redundant_window_count": coverage_summary["redundant_window_count"],
            "finished_at": _now_iso(),
        }
    except Exception as exc:
        report["engine_scope_pool_error"] = {"type": type(exc).__name__, "message": str(exc)}
        _write_json_atomic(report_path, report)
        _record_event(run_dir, "phase_error", iteration=iteration, phase="pool_engine_scope", error=str(exc))

    try:
        if pooled_engine_scope_rows:
            mining_output = run_dir / "reports" / f"engine_scope_negatives_iter_{iteration:03d}.json"
            mined = mine_hard_negative_families(
                pooled_engine_scope_rows,
                min_family_size=args.hard_negative_min_family_size,
                max_query_profile_families=args.hard_negative_max_families,
                max_mask_probe_families=args.hard_negative_max_families,
                max_queries_per_family=args.hard_negative_max_queries_per_family,
            )
            mining_payload = {
                "engine_scope_schema_version": ENGINE_SCOPE_SCHEMA_VERSION,
                "iteration": iteration,
                "generated_at": _now_iso(),
                "source_row_count": len(pooled_engine_scope_rows),
                "criteria": {
                    "min_family_size": args.hard_negative_min_family_size,
                    "max_families": args.hard_negative_max_families,
                    "max_queries_per_family": args.hard_negative_max_queries_per_family,
                },
                **mined,
            }
            _write_json_atomic(mining_output, mining_payload)
            replay_set_output = run_dir / "reports" / f"engine_scope_hard_negative_replay_set_iter_{iteration:03d}.json"
            replay_set_payload = {
                "engine_scope_schema_version": ENGINE_SCOPE_SCHEMA_VERSION,
                "iteration": iteration,
                "generated_at": _now_iso(),
                **build_hard_negative_replay_artifact(
                    pooled_engine_scope_rows,
                    [*mined["query_profile_families"], *mined["mask_probe_families"]],
                ),
            }
            _write_json_atomic(replay_set_output, replay_set_payload)
            report["hard_negative_mining"] = {
                "output_path": str(mining_output),
                "replay_set_output": str(replay_set_output),
                "query_profile_failure_count": mined["query_profile_failure_count"],
                "mask_probe_failure_count": mined["mask_probe_failure_count"],
                "replay_family_count": replay_set_payload["family_count"],
                "replay_row_count": replay_set_payload["summary"]["row_count"],
                "query_profile_families": [
                    {
                        "family_id": family["family_id"],
                        "task": family["task"],
                        "primary_fault_class": family.get("primary_fault_class"),
                        "query_count": family["query_count"],
                        "mean_delta_ndcg_at_10": family["mean_delta_ndcg_at_10"],
                    }
                    for family in mined["query_profile_families"]
                ],
            }
            iteration_info["phases"]["mine_hard_negatives"] = {
                "status": "completed",
                "output_path": str(mining_output),
                "replay_set_output": str(replay_set_output),
                "query_profile_failure_count": mined["query_profile_failure_count"],
                "mask_probe_failure_count": mined["mask_probe_failure_count"],
                "finished_at": _now_iso(),
            }

            replay = run_reform_hard_negative_replay(
                reform_gate.get("gate"),
                families=mined["query_profile_families"],
                model=args.model,
                sample_docs=args.hard_negative_sample_docs,
            )
            replay_output = run_dir / "reports" / f"hard_negative_reform_validation_iter_{iteration:03d}.json"
            _write_json_atomic(replay_output, replay)
            report["hard_negative_reform_validation"] = {
                "output_path": str(replay_output),
                "family_count": replay["family_count"],
                **replay["aggregate"],
            }
            iteration_info["phases"]["validate_hard_negatives"] = {
                "status": "completed",
                "output_path": str(replay_output),
                "family_count": replay["family_count"],
                "guard_learned_negative_families": replay["aggregate"]["guard_learned_negative_families"],
                "finished_at": _now_iso(),
            }
            _write_json_atomic(report_path, report)
    except Exception as exc:
        report["hard_negative_error"] = {"type": type(exc).__name__, "message": str(exc)}
        _write_json_atomic(report_path, report)
        _record_event(run_dir, "phase_error", iteration=iteration, phase="hard_negative_pipeline", error=str(exc))

    reform_validation = report.get("reform_validation", {}).get("aggregate", {})
    mask_validation = report.get("mask_validation", {}).get("aggregate", {})
    report["recommendation"] = _recommendation(
        reform_gate.get("gate"),
        reform_validation,
        mask_gate.get("gate"),
        mask_validation,
        reform_hard_negative_validation=report.get("hard_negative_reform_validation"),
        coverage_summary=report.get("engine_scope_coverage"),
    )
    _write_json_atomic(report_path, report)
    iteration_info["status"] = "completed"
    iteration_info["finished_at"] = report["finished_at"]
    iteration_info["report_path"] = str(report_path)
    iteration_info["recommendation"] = report["recommendation"]
    manifest["latest_report"] = str(report_path)
    manifest["latest_recommendation"] = report["recommendation"]
    manifest["current_phase"] = None
    _record_event(
        run_dir,
        "iteration_completed",
        iteration=iteration,
        report_path=str(report_path),
        recommendation=report["recommendation"],
    )
    return report


def _deadline_reached(manifest: Dict[str, Any]) -> bool:
    deadline_text = manifest.get("deadline_at")
    if not isinstance(deadline_text, str):
        return False
    return _now_utc() >= datetime.fromisoformat(deadline_text)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a resumable autonomous golden-default search loop")
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--resume-run-dir", default=None)
    parser.add_argument("--deadline-hours", type=float, default=48.0)
    parser.add_argument("--max-iterations", type=int, default=None)
    parser.add_argument("--sleep-seconds", type=float, default=5.0)
    parser.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--reform-phase-queries", type=int, default=300)
    parser.add_argument("--reform-loop-queries", type=int, default=100)
    parser.add_argument("--reform-window-queries", type=int, default=50)
    parser.add_argument("--reform-sample-docs", type=int, default=400)
    parser.add_argument("--reform-base-seed", type=int, default=260)
    parser.add_argument(
        "--reform-selection-mode",
        choices=["round_robin", "coverage_aware"],
        default="coverage_aware",
    )
    parser.add_argument("--mask-max-queries", type=int, default=100)
    parser.add_argument("--mask-train-queries", type=int, default=60)
    parser.add_argument("--mask-sample-docs", type=int, default=250)
    parser.add_argument("--mask-base-seed", type=int, default=360)
    parser.add_argument("--mask-fraction", type=float, default=0.02)
    parser.add_argument("--validation-max-queries", type=int, default=50)
    parser.add_argument("--validation-sample-docs", type=int, default=200)
    parser.add_argument("--mask-validation-max-queries", type=int, default=80)
    parser.add_argument("--mask-validation-train-queries", type=int, default=40)
    parser.add_argument("--mask-validation-sample-docs", type=int, default=220)
    parser.add_argument("--hard-negative-max-families", type=int, default=3)
    parser.add_argument("--hard-negative-min-family-size", type=int, default=2)
    parser.add_argument("--hard-negative-max-queries-per-family", type=int, default=8)
    parser.add_argument("--hard-negative-sample-docs", type=int, default=180)
    args = parser.parse_args()

    if args.resume_run_dir:
        run_dir = Path(args.resume_run_dir)
    elif args.run_dir:
        run_dir = Path(args.run_dir)
    else:
        run_dir = PROJECT_ROOT / "experiment_runs" / f"golden-default-autopilot-{_timestamp_slug()}"
    if not run_dir.is_absolute():
        run_dir = PROJECT_ROOT / run_dir
    run_dir.mkdir(parents=True, exist_ok=True)

    manifest = _load_or_init_manifest(run_dir, args)
    _update_manifest(run_dir, manifest)
    _record_event(run_dir, "autopilot_started", run_path=str(run_dir), config=manifest["config"])

    try:
        next_iteration = len(manifest.get("iterations", {})) + 1
        while True:
            if _deadline_reached(manifest):
                manifest["status"] = "completed"
                manifest["finished_at"] = _now_iso()
                break
            if args.max_iterations is not None and next_iteration > args.max_iterations:
                manifest["status"] = "completed"
                manifest["finished_at"] = _now_iso()
                break
            _heartbeat(manifest, run_dir, iteration=next_iteration, phase="iteration")
            _run_iteration(run_dir, manifest, iteration=next_iteration, args=args)
            _update_manifest(run_dir, manifest)
            next_iteration += 1
            if args.sleep_seconds > 0:
                time.sleep(float(args.sleep_seconds))
        _update_manifest(run_dir, manifest)
        terminal_decision: Dict[str, Any] = {
            "generated_at": _now_iso(),
            "run_dir": str(run_dir),
            "status": manifest["status"],
            "finished_at": manifest.get("finished_at"),
            "latest_report": manifest.get("latest_report"),
            "termination_reason": (
                "deadline_reached"
                if _deadline_reached(manifest)
                or (
                    isinstance(manifest.get("deadline_at"), str)
                    and _now_utc() >= datetime.fromisoformat(manifest["deadline_at"])
                )
                else "max_iterations_reached"
            ),
            "iteration_count": len(manifest.get("iterations", {})),
            "latest_recommendation": manifest.get("latest_recommendation"),
        }
        terminal_path = run_dir / "terminal_decision.json"
        _write_json_atomic(terminal_path, terminal_decision)
        _record_event(run_dir, "terminal_decision_written", path=str(terminal_path))
        print(json.dumps({
            "run_dir": str(run_dir),
            "status": manifest["status"],
            "finished_at": manifest.get("finished_at"),
            "latest_report": manifest.get("latest_report"),
            "terminal_decision": str(terminal_path),
        }, indent=2))
        return 0
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["failed_at"] = _now_iso()
        manifest["failure"] = {"type": type(exc).__name__, "message": str(exc)}
        _update_manifest(run_dir, manifest)
        _record_event(run_dir, "autopilot_failed", error=str(exc))
        raise


if __name__ == "__main__":
    raise SystemExit(main())
