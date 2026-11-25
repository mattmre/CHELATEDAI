"""Unit tests for Wave-6 statistical quantiles, Holm-Bonferroni, and cross-family blind semantic judging."""

from __future__ import annotations

import pytest

from wave6_common import (
    BlindCrossFamilySemanticJudge,
    audit_schema_anomalies,
    holm_bonferroni_stepdown,
    paired_bootstrap_wave6,
    parse_strict_judge_verdict,
)


def test_paired_bootstrap_distinct_quantiles_and_cluster() -> None:
    scores_a = {f"q{i:03d}": (0.20 + 0.01 * (i % 17)) for i in range(80)}
    scores_b = {f"q{i:03d}": (0.10 + 0.005 * (i % 11)) for i in range(80)}
    clusters = {f"q{i:03d}": f"conv_{i // 8}" for i in range(80)}
    res = paired_bootstrap_wave6(scores_a, scores_b, n_bootstrap=10000, seed=7, cluster_by_id=clusters)
    assert res["p0125_bonf2"] < res["lcb_975"] < res["lcb_95"] < res["mean_delta"]
    assert res["n_clusters"] == 10
    assert res["cluster_p0125_bonf2"] <= res["cluster_lcb_975_p025"] <= res["cluster_lcb_95_p050"]


def test_paired_bootstrap_binary_tango_bonf2() -> None:
    scores_a = {f"q{i}": (1.0 if i < 65 else 0.0) for i in range(80)}
    scores_b = {f"q{i}": (1.0 if i < 45 else 0.0) for i in range(80)}
    res = paired_bootstrap_wave6(scores_a, scores_b, n_bootstrap=8000, seed=11)
    assert res["tango_score_lcb975_bonf2"] < res["tango_score_lcb95"]
    assert res["n10"] == 20.0
    assert res["n01"] == 0.0


def test_holm_bonferroni_stepdown_monotonicity() -> None:
    out = holm_bonferroni_stepdown({"a": 0.004, "b": 0.018, "c": 0.04, "d": 0.25}, alpha=0.05)
    assert out["a"]["reject_null_at_alpha"] is True
    assert out["b"]["reject_null_at_alpha"] is False
    assert out["c"]["reject_null_at_alpha"] is False
    assert out["d"]["reject_null_at_alpha"] is False
    assert out["a"]["holm_adjusted_p"] <= out["b"]["holm_adjusted_p"] <= out["c"]["holm_adjusted_p"]


def test_parse_strict_judge_verdict_rejects_dual_yes_no() -> None:
    assert parse_strict_judge_verdict("VERDICT: YES") == (True, True)
    assert parse_strict_judge_verdict("VERDICT: NO") == (False, True)
    assert parse_strict_judge_verdict("VERDICT: YES and VERDICT: NO") == (False, False)


def test_blind_cross_family_judge_enforces_distinct_family_and_schema_audit() -> None:
    with pytest.raises(ValueError, match="Cross-family judge violation"):
        BlindCrossFamilySemanticJudge(
            judge_family="nvidia-nemotron",
            generator_family="NVIDIA-Nemotron",
            completion_fn=lambda s, u: "VERDICT: YES",
        )

    calls = []

    def fake_judge(_sys: str, user: str) -> str:
        calls.append(user)
        if "I don't have information about Italian restaurants" in user:
            return "VERDICT: YES"
        if "ABSTAIN" in user:
            return "VERDICT: YES"
        return "VERDICT: NO"

    judge = BlindCrossFamilySemanticJudge(
        judge_family="alibaba-qwen2.5",
        generator_family="nvidia-nemotron",
        completion_fn=fake_judge,
    )
    graded = judge.grade_arms_blind(
        item_id="6aeb4375_abs",
        question="Which Italian restaurant did I prefer?",
        gold_reference="You mentioned Korean restaurants, not Italian restaurants.",
        arm_predictions={
            "chelated": "I don't have information about Italian restaurants you've tried.",
            "mem0_true": "I found 0 matching records.",
            "mem0_abs_bug": "ABSTRACT",
            "strict_abs": "ABSTAIN",
        },
        abstention_required=True,
    )
    assert graded["chelated"]["correct"] is True
    assert graded["strict_abs"]["correct"] is True
    assert graded["mem0_true"]["correct"] is False
    assert graded["mem0_true"]["schema_anomalies"]["found_n_matching_records_template"] is True
    assert graded["mem0_abs_bug"]["correct"] is False
    assert graded["mem0_abs_bug"]["schema_anomalies"]["bare_abstract_token"] is True
    # Ensure arm names were never leaked into the judge prompt
    assert all("chelated" not in c and "mem0" not in c for c in calls)
    assert audit_schema_anomalies("I found 3 matching records.")["found_n_matching_records_template"] is True
