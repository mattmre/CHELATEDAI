"""Meta-analysis helpers for ChelatedAI research pathways."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np

from learned_mask_policy import run_learned_mask_smoke
from synthetic_collapse_benchmark import run_synthetic_collapse_benchmark


def load_artifact(path: str | Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def summarize_query_attribution(rows: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    by_profile: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("profile") != "baseline":
            by_profile[str(row.get("profile"))].append(row)
    summary = {}
    for profile, profile_rows in by_profile.items():
        deltas = [float(row.get("delta_ndcg_at_10", 0.0)) for row in profile_rows]
        summary[profile] = {
            "queries": len(profile_rows),
            "mean_delta_ndcg_at_10": float(np.mean(deltas)) if deltas else 0.0,
            "positive_queries": sum(delta > 0 for delta in deltas),
            "negative_queries": sum(delta < 0 for delta in deltas),
            "unchanged_queries": sum(delta == 0 for delta in deltas),
            "top_doc_changed": sum(bool(row.get("top_doc_changed")) for row in profile_rows),
            "actions": _count_values(row.get("action") for row in profile_rows),
        }
    return summary


def _count_values(values: Iterable[Any]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for value in values:
        key = str(value)
        counts[key] = counts.get(key, 0) + 1
    return counts


def summarize_benchmark_families(artifacts: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    by_task_profile: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    for artifact in artifacts:
        for loop in artifact.get("loops", []):
            for window in loop.get("windows", []):
                task = window.get("task", loop.get("task", "unknown"))
                for row in window.get("summary", {}).get("ranked_profiles", []):
                    by_task_profile[str(task)][row["profile"]].append(float(row.get("delta_vs_baseline", 0.0)))
    family_summary = {}
    for task, profiles in by_task_profile.items():
        family_summary[task] = {
            profile: {
                "windows": len(deltas),
                "mean_delta_vs_baseline": float(np.mean(deltas)) if deltas else 0.0,
                "best_delta_vs_baseline": float(np.max(deltas)) if deltas else 0.0,
                "worst_delta_vs_baseline": float(np.min(deltas)) if deltas else 0.0,
            }
            for profile, deltas in profiles.items()
        }
    return family_summary


def propose_candidate_profiles(attribution_summary: Dict[str, Any]) -> List[Dict[str, Any]]:
    proposals = []
    for profile, row in attribution_summary.items():
        positives = int(row["positive_queries"])
        negatives = int(row["negative_queries"])
        mean_delta = float(row["mean_delta_ndcg_at_10"])
        if positives > 0 and mean_delta >= 0 and negatives <= positives:
            proposals.append({
                "profile": profile,
                "proposal": "retest_query_conditional",
                "mean_delta_ndcg_at_10": mean_delta,
                "positive_queries": positives,
                "negative_queries": negatives,
            })
        elif positives > 0:
            proposals.append({
                "profile": profile,
                "proposal": "use_as_training_data_only",
                "mean_delta_ndcg_at_10": mean_delta,
                "positive_queries": positives,
                "negative_queries": negatives,
            })
    proposals.sort(key=lambda item: (-item["mean_delta_ndcg_at_10"], item["negative_queries"], item["profile"]))
    return proposals


def run_meta_analysis(paths: List[str | Path]) -> Dict[str, Any]:
    artifacts = [load_artifact(path) for path in paths]
    attribution_rows = []
    for artifact in artifacts:
        attribution_rows.extend(artifact.get("query_attribution_rows", []))
    attribution_summary = summarize_query_attribution(attribution_rows)
    synthetic = run_synthetic_collapse_benchmark()
    learned_mask = run_learned_mask_smoke()
    candidates = propose_candidate_profiles(attribution_summary)
    return {
        "artifact_count": len(artifacts),
        "query_attribution_rows": len(attribution_rows),
        "query_attribution_summary": attribution_summary,
        "benchmark_family_summary": summarize_benchmark_families(artifacts),
        "candidate_profile_proposals": candidates,
        "synthetic_collapse": {
            "baseline_ndcg_at_3": synthetic["baseline"]["metrics"]["ndcg_at_3"],
            "masked_ndcg_at_3": synthetic["masked"]["metrics"]["ndcg_at_3"],
            "recovered": synthetic["recovered"],
        },
        "learned_mask": {
            "masked_dims": learned_mask["learned_mask"]["masked_dims"],
            "expected_collapse_dim": learned_mask["expected_collapse_dim"],
            "learned_ndcg_at_3": learned_mask["learned"]["metrics"]["ndcg_at_3"],
            "recovered": learned_mask["recovered"],
        },
        "golden_setting": None,
        "research_decision": (
            "controlled collapse and learned masking work on synthetic data; real attribution still requires "
            "query-conditional candidates with holdout validation"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze ChelatedAI research pathways from tuning artifacts")
    parser.add_argument("--artifact", action="append", required=True)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    result = run_meta_analysis(args.artifact)
    text = json.dumps(result, indent=2)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
