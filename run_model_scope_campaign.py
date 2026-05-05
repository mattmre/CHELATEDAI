"""Replay/comparison/training campaign for Model-Scope observation artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping

from adaptive_overlay import (
    build_overlay_artifact_card,
    build_verifier_evidence_card,
    build_overlay_validation_report,
    decide_overlay_collection_budget,
)
from checkpoint_manager import CheckpointManager
from compute_budget_policy import decide_compute_budget, summarize_compute_budget_decisions
from config import ChelationConfig
from evaluator_fabric import detect_reward_overoptimization, grade_trace_events, summarize_evaluator_results
from evidence_contract import (
    build_episode_event,
    build_evidence_bundle,
    evidence_event_from_model_scope_artifact,
    summarize_evidence_bundle,
    write_evidence_bundle,
)
from expectation_comparator import ModelScopeExpectationComparator
from integrated_diagnostics_report import summarize_adaptive_overlay_report
from model_scope_artifacts import load_model_scope_artifact
from model_scope_features import build_feature_scorecard
from model_scope_memory import ModelScopeMemoryStore
from model_scope_trainer import ModelScopeShadowPolicyTrainer, ModelScopeTrainerConfig
from promotion_contract import PromotionGateConfig, evaluate_promotion_candidate


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    return str(value)


def _load_artifact_paths(input_path: str | Path) -> List[Path]:
    path = Path(input_path)
    if path.is_dir():
        return sorted(candidate for candidate in path.glob("*.json") if candidate.is_file())
    return [path]


def _artifact_query_key(artifact: Mapping[str, Any]) -> str:
    capture = artifact.get("capture", {})
    metadata = capture.get("metadata", {}) if isinstance(capture.get("metadata"), Mapping) else {}
    query_id = metadata.get("query_id")
    if query_id is not None:
        return f"query_id:{query_id}"
    prompt_hash = capture.get("prompt_hash")
    if prompt_hash:
        return f"prompt_hash:{prompt_hash}"
    output_path = artifact.get("output_path")
    return f"path:{output_path or 'unknown'}"


def _derive_holdout_report(records: List[Mapping[str, Any]]) -> Dict[str, Any] | None:
    holdout_records = [
        record
        for record in records
        if not bool(record.get("comparison", {}).get("reference"))
    ]
    if not holdout_records:
        return None
    scores = [
        float(record.get("comparison", {}).get("score", 0.0))
        for record in holdout_records
    ]
    failed = [
        record
        for record in holdout_records
        if not bool(record.get("comparison", {}).get("passed"))
    ]
    mean_score = sum(scores) / len(scores) if scores else 0.0
    return {
        "passed": not failed,
        "score": float(mean_score),
        "mean_score": float(mean_score),
        "holdout_count": len(holdout_records),
        "failed_count": len(failed),
        "failed_artifact_paths": [str(record.get("artifact_path")) for record in failed],
    }


def _load_optional_json_report(path_or_report: str | Path | Mapping[str, Any] | None) -> Dict[str, Any] | None:
    if path_or_report is None:
        return None
    if isinstance(path_or_report, Mapping):
        return dict(path_or_report)
    path = Path(path_or_report)
    return json.loads(path.read_text(encoding="utf-8"))


def run_model_scope_campaign(
    input_path: str | Path,
    *,
    output_dir: str | Path | None = None,
    max_rules: int = 8,
    min_alignment_score: float = 0.55,
    promotion_path: str | Path | None = None,
    holdout_report: Mapping[str, Any] | None = None,
    safety_report: Mapping[str, Any] | None = None,
    adaptive_overlay_report: str | Path | Mapping[str, Any] | None = None,
    adaptive_overlay_holdout_report: str | Path | Mapping[str, Any] | None = None,
    require_adaptive_overlay_readiness: bool = False,
) -> Dict[str, Any]:
    artifact_paths = _load_artifact_paths(input_path)
    artifacts = []
    for path in artifact_paths:
        artifact = load_model_scope_artifact(path)
        artifact = dict(artifact)
        artifact.setdefault("output_path", str(path))
        artifacts.append((path, artifact))

    comparator = ModelScopeExpectationComparator()
    memory = ModelScopeMemoryStore()
    profiles_by_query: Dict[str, Dict[str, Any]] = {}
    comparison_records = []
    evidence_events = []
    evaluator_records = []
    compute_budget_decisions = []
    for path, artifact in artifacts:
        query_key = _artifact_query_key(artifact)
        profile = profiles_by_query.get(query_key)
        if profile is None:
            profile = comparator.build_expectation_profile(artifact, profile_id=query_key, description="campaign_reference")
            profiles_by_query[query_key] = profile
            memory.store_expectation_profile(profile, metadata={"query_key": query_key, "source_path": str(path)})
            comparison = {
                "reference": True,
                "passed": True,
                "score": 1.0,
                "profile_id": profile["profile_id"],
                "candidate_id": path.stem,
                "reasons": [],
            }
            label = "positive"
        else:
            comparison = comparator.compare_to_profile(artifact, profile, candidate_id=path.stem)
            label = "positive" if comparison.get("passed") else "negative"
        artifact_with_comparison = dict(artifact)
        artifact_with_comparison["expectation_comparison"] = comparison
        model_event = evidence_event_from_model_scope_artifact(artifact_with_comparison)
        comparison_event = build_episode_event(
            event_type="expectation_comparison",
            surface="evaluator",
            query_id=query_key,
            artifact_id=str(path),
            action="compare_to_reference",
            decision="passed" if comparison.get("passed") else "failed",
            outcome_metrics={
                "score": comparison.get("score", 1.0 if comparison.get("passed") else 0.0),
            },
            provenance={"profile_id": profile["profile_id"], "source_path": str(path)},
            source_family="model_scope_expectation_comparator",
            payload=comparison,
        )
        evidence_events.extend([model_event, comparison_event])
        trace_grade = grade_trace_events(
            [model_event, comparison_event],
            required_surfaces={"model_scope", "evaluator"},
        )
        support_score = float(comparison.get("score", 1.0 if comparison.get("passed") else 0.0))
        compute_budget = decide_compute_budget(
            {
                "uncertainty_score": 1.0 - support_score,
                "retrieval_support_score": support_score,
                "evaluator_disagreement": 0.0 if comparison.get("passed") else 1.0,
                "safety_risk": 0.0 if trace_grade.get("passed") else 1.0,
                "hard_negative_risk": 0.0 if comparison.get("passed") else 1.0,
                "query_token_count": artifact.get("capture", {}).get("token_count", 0),
            }
        )
        compute_budget_decisions.append(compute_budget)
        evaluator_records.append(
            {
                "evaluator_id": "model_scope_expectation_comparator",
                "passed": bool(comparison.get("passed")),
                "score": float(comparison.get("score", 1.0 if comparison.get("passed") else 0.0)),
                "trace_grade_passed": bool(trace_grade.get("passed")),
                "compute_budget_units": int(compute_budget.get("budget_units", 1)),
            }
        )
        record = memory.record_observation(
            artifact_with_comparison,
            query_text=artifact.get("capture", {}).get("metadata", {}).get("query_text"),
            metadata={
                **dict(artifact.get("capture", {}).get("metadata", {})),
                "label": label,
                "query_key": query_key,
                "source_path": str(path),
            },
            expectation_profile_id=profile["profile_id"],
            evidence_event_ids=[model_event["event_id"], comparison_event["event_id"]],
        )
        comparison_records.append(
            {
                "artifact_path": str(path),
                "query_key": query_key,
                "label": label,
                "profile_id": profile["profile_id"],
                "episode_entry_id": record["episode_entry_id"],
                "comparison": comparison,
            }
        )

    replay_bundle = memory.build_replay_bundle(segment="episode", include_artifacts=True)
    evidence_bundle = build_evidence_bundle(
        evidence_events,
        campaign_id="model_scope_campaign",
        metadata={"input_path": str(input_path), "artifact_count": len(artifacts)},
    )
    replay_bundle["evidence_bundle_id"] = evidence_bundle["bundle_id"]
    replay_bundle["evaluator_report_id"] = "model_scope_evaluator_summary"
    replay_bundle["compute_budget_summary_id"] = "model_scope_compute_budget_summary"
    trace_grade = grade_trace_events(
        evidence_events,
        required_surfaces={"model_scope", "evaluator"},
    )
    resolved_holdout_report = dict(holdout_report) if holdout_report is not None else _derive_holdout_report(comparison_records)
    resolved_safety_report = dict(safety_report) if safety_report is not None else {
        "passed": bool(trace_grade.get("passed")),
        "failure_count": int(trace_grade.get("failure_count", 0)),
        "source": "trace_grade",
    }
    resolved_adaptive_overlay_report = _load_optional_json_report(adaptive_overlay_report)
    resolved_adaptive_overlay_holdout_report = _load_optional_json_report(adaptive_overlay_holdout_report)
    adaptive_overlay_summary = summarize_adaptive_overlay_report(resolved_adaptive_overlay_report)
    evaluator_summary = summarize_evaluator_results(evaluator_records)
    compute_budget_summary = summarize_compute_budget_decisions(compute_budget_decisions)
    replay_entries_for_scorecard = [
        {
            **entry,
            "label": str((entry.get("metadata") or {}).get("label") or "unknown"),
        }
        for entry in replay_bundle.get("entries", [])
    ]
    feature_scorecard = build_feature_scorecard(replay_entries_for_scorecard)
    trainer = ModelScopeShadowPolicyTrainer(
        ModelScopeTrainerConfig(
            max_rules=max_rules,
            min_alignment_score=min_alignment_score,
        )
    )
    candidate = trainer.train_shadow_policy(replay_bundle)
    hard_negative_report = {"blocker_count": int(sum(1 for item in comparison_records if item["label"] == "negative"))}
    reward_report = detect_reward_overoptimization(
        {"score": float(candidate.get("promotion_gate", {}).get("alignment_score", 0.0))},
        {"score": float(evaluator_summary.get("mean_score", 0.0))},
        max_divergence=0.20,
        min_training_gain=min_alignment_score,
    )
    promotion_decision = evaluate_promotion_candidate(
        candidate_id=str(candidate.get("candidate_id", "model_scope_shadow_policy_v1")),
        evidence_bundle=evidence_bundle,
        comparator_report={
            "passed": bool(candidate.get("promotion_gate", {}).get("promotion_ready")),
            "score": float(candidate.get("promotion_gate", {}).get("alignment_score", 0.0)),
        },
        holdout_report=resolved_holdout_report,
        safety_report=resolved_safety_report,
        hard_negative_report=hard_negative_report,
        evaluator_report=evaluator_summary,
        reward_report=reward_report,
        adaptive_overlay_report=resolved_adaptive_overlay_report,
        config=PromotionGateConfig(
            min_replay_score=min_alignment_score,
            min_evaluator_agreement=0.5,
            require_replay=True,
            require_holdout=True,
            require_safety=True,
            require_no_hard_negative_blockers=True,
            require_adaptive_overlay_readiness=require_adaptive_overlay_readiness,
        ),
    )
    adaptive_overlay_artifact_card = None
    adaptive_overlay_validation_report = None
    adaptive_overlay_collection_policy = None
    verifier_cards = []
    if resolved_adaptive_overlay_report is not None:
        verifier_cards = [
            build_verifier_evidence_card(
                verifier_id="model_scope_evaluator_summary",
                subject_id=str(candidate.get("candidate_id", "model_scope_shadow_policy_v1")),
                rubric={
                    "min_mean_score": 0.5,
                    "requires_trace_grade": True,
                    "default_runtime_controller": False,
                },
                result={
                    "passed": bool(evaluator_summary.get("majority_passed", False)),
                    "score": evaluator_summary.get("mean_score", 0.0),
                    "reasons": list(evaluator_summary.get("reasons", [])),
                    "blockers": (
                        []
                        if evaluator_summary.get("majority_passed", False)
                        else ["evaluator_summary_not_passed"]
                    ),
                },
                metadata={"source": "run_model_scope_campaign"},
            )
        ]
        adaptive_overlay_collection_policy = decide_overlay_collection_budget(
            resolved_adaptive_overlay_report,
            uncertainty_score=float(evaluator_summary.get("disagreement_rate", 0.0) or 0.0),
            coverage_novelty_score=float(feature_scorecard.get("feature_count", 0) > 0),
            blocker_history_count=int(hard_negative_report.get("blocker_count", 0)),
        )
        adaptive_overlay_validation_report = build_overlay_validation_report(
            candidate_id=str(candidate.get("candidate_id", "model_scope_shadow_policy_v1")),
            replay_overlay_report=resolved_adaptive_overlay_report,
            holdout_overlay_report=resolved_adaptive_overlay_holdout_report,
            metadata={
                "input_path": str(input_path),
                "adaptive_overlay_report": (
                    str(adaptive_overlay_report) if isinstance(adaptive_overlay_report, (str, Path)) else None
                ),
                "adaptive_overlay_holdout_report": (
                    str(adaptive_overlay_holdout_report)
                    if isinstance(adaptive_overlay_holdout_report, (str, Path))
                    else None
                ),
            },
        )
        adaptive_overlay_artifact_card = build_overlay_artifact_card(
            candidate_id=str(candidate.get("candidate_id", "model_scope_shadow_policy_v1")),
            overlay_report=resolved_adaptive_overlay_report,
            purpose="model-scope campaign supplied adaptive overlay evidence",
            source_path=str(adaptive_overlay_report) if isinstance(adaptive_overlay_report, (str, Path)) else None,
            promotion_decision=promotion_decision,
            validation_report=adaptive_overlay_validation_report,
            collection_policy=adaptive_overlay_collection_policy,
            verifier_cards=verifier_cards,
            replay_report={"entry_count": len(replay_bundle.get("entries", []))},
            holdout_report=resolved_holdout_report,
            hard_negative_report=hard_negative_report,
            evaluator_report=evaluator_summary,
            safety_report=resolved_safety_report,
            rollback_path=str(promotion_path) if promotion_path is not None else None,
            metadata={
                "input_path": str(input_path),
                "output_dir": str(output_dir) if output_dir is not None else None,
                "require_adaptive_overlay_readiness": require_adaptive_overlay_readiness,
            },
        )

    resolved_output_dir = Path(output_dir) if output_dir is not None else (
        ChelationConfig.MODEL_SCOPE_ARTIFACT_ROOT / "campaigns" / "latest"
    )
    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    memory_path = memory.save(resolved_output_dir / "memory_snapshot.json")
    evidence_path = write_evidence_bundle(resolved_output_dir / "evidence_bundle.json", evidence_bundle)
    replay_path = resolved_output_dir / "replay_bundle.json"
    replay_path.write_text(json.dumps(_json_safe(replay_bundle), indent=2), encoding="utf-8")
    comparison_path = resolved_output_dir / "comparison_report.json"
    comparison_path.write_text(
        json.dumps(
            {
                "artifact_count": len(artifacts),
                "profile_count": len(profiles_by_query),
                "records": _json_safe(comparison_records),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    candidate_path = resolved_output_dir / "shadow_policy_candidate.json"
    candidate_path.write_text(json.dumps(_json_safe(candidate), indent=2), encoding="utf-8")
    evaluator_path = resolved_output_dir / "evaluator_summary.json"
    evaluator_path.write_text(json.dumps(_json_safe(evaluator_summary), indent=2), encoding="utf-8")
    trace_grade_path = resolved_output_dir / "trace_grade.json"
    trace_grade_path.write_text(json.dumps(_json_safe(trace_grade), indent=2), encoding="utf-8")
    compute_budget_path = resolved_output_dir / "compute_budget_summary.json"
    compute_budget_path.write_text(json.dumps(_json_safe(compute_budget_summary), indent=2), encoding="utf-8")
    reward_report_path = resolved_output_dir / "reward_overoptimization_report.json"
    reward_report_path.write_text(json.dumps(_json_safe(reward_report), indent=2), encoding="utf-8")
    feature_scorecard_path = resolved_output_dir / "feature_scorecard.json"
    feature_scorecard_path.write_text(json.dumps(_json_safe(feature_scorecard), indent=2), encoding="utf-8")
    holdout_path = resolved_output_dir / "holdout_report.json"
    holdout_path.write_text(json.dumps(_json_safe(resolved_holdout_report), indent=2), encoding="utf-8")
    safety_path = resolved_output_dir / "safety_report.json"
    safety_path.write_text(json.dumps(_json_safe(resolved_safety_report), indent=2), encoding="utf-8")
    adaptive_overlay_path = None
    if resolved_adaptive_overlay_report is not None:
        adaptive_overlay_path = resolved_output_dir / "adaptive_overlay_report.json"
        adaptive_overlay_path.write_text(
            json.dumps(_json_safe(resolved_adaptive_overlay_report), indent=2),
            encoding="utf-8",
        )
    adaptive_overlay_holdout_path = None
    if resolved_adaptive_overlay_holdout_report is not None:
        adaptive_overlay_holdout_path = resolved_output_dir / "adaptive_overlay_holdout_report.json"
        adaptive_overlay_holdout_path.write_text(
            json.dumps(_json_safe(resolved_adaptive_overlay_holdout_report), indent=2),
            encoding="utf-8",
        )
    adaptive_overlay_validation_path = None
    if adaptive_overlay_validation_report is not None:
        adaptive_overlay_validation_path = resolved_output_dir / "adaptive_overlay_validation_report.json"
        adaptive_overlay_validation_path.write_text(
            json.dumps(_json_safe(adaptive_overlay_validation_report), indent=2),
            encoding="utf-8",
        )
    adaptive_overlay_collection_policy_path = None
    if adaptive_overlay_collection_policy is not None:
        adaptive_overlay_collection_policy_path = resolved_output_dir / "adaptive_overlay_collection_policy.json"
        adaptive_overlay_collection_policy_path.write_text(
            json.dumps(_json_safe(adaptive_overlay_collection_policy), indent=2),
            encoding="utf-8",
        )
    verifier_cards_path = None
    if verifier_cards:
        verifier_cards_path = resolved_output_dir / "verifier_evidence_cards.json"
        verifier_cards_path.write_text(json.dumps(_json_safe(verifier_cards), indent=2), encoding="utf-8")
    adaptive_overlay_artifact_card_path = None
    if adaptive_overlay_artifact_card is not None:
        adaptive_overlay_artifact_card_path = resolved_output_dir / "adaptive_overlay_artifact_card.json"
        adaptive_overlay_artifact_card_path.write_text(
            json.dumps(_json_safe(adaptive_overlay_artifact_card), indent=2),
            encoding="utf-8",
        )
    promotion_decision_path = resolved_output_dir / "promotion_decision.json"
    promotion_decision_path.write_text(json.dumps(_json_safe(promotion_decision), indent=2), encoding="utf-8")

    promotion = None
    if promotion_path is not None and promotion_decision.get("promotion_ready"):
        promotion = trainer.promote_candidate(
            candidate,
            promotion_path,
            checkpoint_manager=CheckpointManager(resolved_output_dir / "checkpoints"),
        )
    elif promotion_path is not None:
        promotion = {
            "promoted": False,
            "path": str(promotion_path),
            "checkpoint_id": None,
            "reasons": list(promotion_decision.get("reasons", ["promotion_decision_failed"])),
        }

    report = {
        "artifact_count": len(artifacts),
        "profile_count": len(profiles_by_query),
        "comparison_failure_count": sum(
            1 for item in comparison_records if not bool(item.get("comparison", {}).get("passed", True))
        ),
        "candidate": {
            "rule_count": int(len(candidate.get("policy", {}).get("rules", []))),
            "promotion_ready": bool(candidate.get("promotion_gate", {}).get("promotion_ready")),
            "alignment_score": float(candidate.get("promotion_gate", {}).get("alignment_score", 0.0)),
        },
        "evidence_summary": summarize_evidence_bundle(evidence_bundle),
        "evaluator_summary": evaluator_summary,
        "trace_grade": trace_grade,
        "compute_budget_summary": compute_budget_summary,
        "reward_overoptimization": reward_report,
        "feature_scorecard": {
            "entry_count": feature_scorecard["entry_count"],
            "feature_count": feature_scorecard["feature_count"],
            "top_features": feature_scorecard["features"][:5],
        },
        "holdout_report": resolved_holdout_report,
        "safety_report": resolved_safety_report,
        "hard_negative_report": hard_negative_report,
        "adaptive_overlay": resolved_adaptive_overlay_report,
        "adaptive_overlay_holdout": resolved_adaptive_overlay_holdout_report,
        "adaptive_overlay_summary": adaptive_overlay_summary,
        "adaptive_overlay_validation_report": adaptive_overlay_validation_report,
        "adaptive_overlay_collection_policy": adaptive_overlay_collection_policy,
        "adaptive_overlay_artifact_card": adaptive_overlay_artifact_card,
        "verifier_evidence_cards": verifier_cards,
        "promotion_decision": promotion_decision,
        "outputs": {
            "memory_snapshot": str(memory_path),
            "evidence_bundle": str(evidence_path),
            "replay_bundle": str(replay_path),
            "comparison_report": str(comparison_path),
            "shadow_policy_candidate": str(candidate_path),
            "evaluator_summary": str(evaluator_path),
            "trace_grade": str(trace_grade_path),
            "compute_budget_summary": str(compute_budget_path),
            "reward_overoptimization_report": str(reward_report_path),
            "feature_scorecard": str(feature_scorecard_path),
            "holdout_report": str(holdout_path),
            "safety_report": str(safety_path),
            "promotion_decision": str(promotion_decision_path),
        },
    }
    if adaptive_overlay_path is not None:
        report["outputs"]["adaptive_overlay_report"] = str(adaptive_overlay_path)
    if adaptive_overlay_holdout_path is not None:
        report["outputs"]["adaptive_overlay_holdout_report"] = str(adaptive_overlay_holdout_path)
    if adaptive_overlay_validation_path is not None:
        report["outputs"]["adaptive_overlay_validation_report"] = str(adaptive_overlay_validation_path)
    if adaptive_overlay_collection_policy_path is not None:
        report["outputs"]["adaptive_overlay_collection_policy"] = str(adaptive_overlay_collection_policy_path)
    if verifier_cards_path is not None:
        report["outputs"]["verifier_evidence_cards"] = str(verifier_cards_path)
    if adaptive_overlay_artifact_card_path is not None:
        report["outputs"]["adaptive_overlay_artifact_card"] = str(adaptive_overlay_artifact_card_path)
    if promotion is not None:
        report["promotion"] = promotion
    report_path = resolved_output_dir / "campaign_report.json"
    report_path.write_text(json.dumps(_json_safe(report), indent=2), encoding="utf-8")
    report["outputs"]["campaign_report"] = str(report_path)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a replay/comparison/training campaign over Model-Scope artifacts")
    parser.add_argument("input_path", help="Path to a Model-Scope artifact JSON file or a directory of artifacts")
    parser.add_argument(
        "--output-dir",
        default=str(ChelationConfig.MODEL_SCOPE_ARTIFACT_ROOT / "campaigns" / "latest"),
        help="Directory for memory, comparison, replay, and candidate outputs",
    )
    parser.add_argument("--max-rules", type=int, default=8, help="Maximum number of learned shadow-policy rules")
    parser.add_argument(
        "--min-alignment-score",
        type=float,
        default=0.55,
        help="Promotion threshold for replay alignment score",
    )
    parser.add_argument(
        "--promotion-path",
        type=str,
        default=None,
        help="Optional path to write a promoted shadow policy if the candidate passes the promotion gate",
    )
    parser.add_argument(
        "--adaptive-overlay-report",
        type=str,
        default=None,
        help="Optional adaptive overlay report JSON to include in promotion evidence",
    )
    parser.add_argument(
        "--adaptive-overlay-holdout-report",
        type=str,
        default=None,
        help="Optional holdout adaptive overlay report JSON to include in validation evidence",
    )
    parser.add_argument(
        "--require-adaptive-overlay-readiness",
        action="store_true",
        help="Require adaptive overlay readiness before promotion",
    )
    args = parser.parse_args()
    report = run_model_scope_campaign(
        args.input_path,
        output_dir=args.output_dir,
        max_rules=args.max_rules,
        min_alignment_score=args.min_alignment_score,
        promotion_path=args.promotion_path,
        adaptive_overlay_report=args.adaptive_overlay_report,
        adaptive_overlay_holdout_report=args.adaptive_overlay_holdout_report,
        require_adaptive_overlay_readiness=args.require_adaptive_overlay_readiness,
    )
    print(json.dumps(_json_safe(report), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
