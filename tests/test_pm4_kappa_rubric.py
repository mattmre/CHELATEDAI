"""Unit tests for Stage R6 PM-4 symmetric conversational rubric, stratified sampler, and Cohen's kappa bootstrap."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from scripts.run_r6_pm4_conversational_kappa_n200 import (
    SYMMETRIC_CONV_JUDGE_SYSTEM_PROMPT,
    _allocate_stratified_quotas,
    bootstrap_cohens_kappa,
    build_symmetric_judge_user_prompt,
    cluster_and_stratified_bootstrap_kappa,
    cohens_kappa_safe,
    parse_binary_verdict,
    sample_locomo_items,
    sample_longmemeval_items,
    sample_structured_r5b_items,
)


def test_symmetric_rubric_covers_core_equivalence_rules() -> None:
    prompt_lo = SYMMETRIC_CONV_JUDGE_SYSTEM_PROMPT.lower()
    assert "unanswerable" in prompt_lo
    assert "abstention" in prompt_lo
    assert "numeric" in prompt_lo
    assert "date" in prompt_lo
    assert "omits a required item" in prompt_lo
    assert "full sentences" in prompt_lo
    assert "yes or no" in prompt_lo


def test_build_symmetric_judge_user_prompt_is_deterministic() -> None:
    p1 = build_symmetric_judge_user_prompt(
        "October, 2023", "Oct 2023", question="When did the trip happen?", dataset="LoCoMo-300", question_type="cat_2"
    )
    p2 = build_symmetric_judge_user_prompt(
        "October, 2023", "Oct 2023", question="When did the trip happen?", dataset="LongMemEval-S", question_type="temporal"
    )
    assert p1 == p2  # Dataset-blind: identical prompt regardless of source dataset metadata
    assert "Question: When did the trip happen?" in p1
    assert "Reference Answer: October, 2023" in p1
    assert "Candidate Answer: Oct 2023" in p1
    assert "LoCoMo" not in p1


@pytest.mark.parametrize(
    ("raw", "expected_grade", "expected_clean"),
    [
        ("YES", 1, True),
        ("NO", 0, True),
        ("yes.", 1, True),
        ("No — missing item", 0, True),
        ("Verdict: YES", 1, True),
        ("Verdict: NO", 0, True),
        ("Reason: Candidate says Luna is the cat's name, matching Reference 'Luna'.\nVerdict: YES", 1, True),
        ("Reason: Reference is concrete fact but Candidate abstained.\nVerdict: NO", 0, True),
        ("<think>checking rules</think>\nYES", 1, True),
        ("<think>wrong entity</think>\nNO", 0, True),
        ("The final answer is YES because both match.", 1, True),
        ("YES and NO both apply here", 0, False),
        ("Verdict: YES\nVerdict: NO", 0, False),
        ("", 0, False),
        ("maybe uncertain", 0, False),
    ],
)
def test_parse_binary_verdict_handles_edge_cases(raw: str, expected_grade: int, expected_clean: bool) -> None:
    grade, clean = parse_binary_verdict(raw)
    assert grade == expected_grade
    assert clean == expected_clean


def test_cluster_and_stratified_bootstrap_keeps_pairs_and_computes_stratum_cis() -> None:
    rows = []
    for conv in range(4):
        for qi in range(5):
            g = 1 if (conv + qi) % 2 == 0 else 0
            rows.append({
                "dataset": "LoCoMo-300",
                "stratum": "cat_1",
                "qid": f"loc_{conv}_{qi}",
                "nemotron_grade": g,
                "qwen_grade": g,
            })
    for i in range(10):
        g = i % 2
        rows.append({
            "dataset": "LongMemEval-S",
            "stratum": "knowledge-update",
            "qid": f"lme_ku_{i}",
            "nemotron_grade": g,
            "qwen_grade": g if i < 9 else (1 - g),
        })
    res = cluster_and_stratified_bootstrap_kappa(rows, n_bootstrap=500, seed=7)
    assert res["locomo_n_conversation_clusters"] == 4
    assert res["conversational_n200_cluster_ci95"] is not None
    assert "LoCoMo-300::cat_1" in res["by_stratum_with_ci95"]
    assert "ci95" in res["by_stratum_with_ci95"]["LongMemEval-S::knowledge-update"]


def test_cohens_kappa_safe_and_bootstrap() -> None:
    # Exact match balanced -> kappa == 1.0
    k_perf, raw_perf, deg_perf = cohens_kappa_safe([1, 0, 1, 0, 1, 0], [1, 0, 1, 0, 1, 0])
    assert k_perf == pytest.approx(1.0)
    assert raw_perf == pytest.approx(1.0)
    assert not deg_perf

    # Degenerate all-ones -> flagged degenerate
    k_deg, raw_deg, deg_flag = cohens_kappa_safe([1, 1, 1, 1], [1, 1, 1, 1])
    assert k_deg == pytest.approx(1.0)
    assert raw_deg == pytest.approx(1.0)
    assert deg_flag is True

    # Realistic high-agreement sample (95/100 agree on 50/50 split -> kappa == 0.90)
    y1 = [1] * 50 + [0] * 50
    y2 = [1] * 47 + [0] * 3 + [0] * 48 + [1] * 2
    stats = bootstrap_cohens_kappa(y1, y2, n_bootstrap=3000, seed=42)
    assert stats["n_items"] == 100
    assert stats["n_agree"] == 95
    assert stats["n_disagree"] == 5
    assert stats["cohens_kappa"] == pytest.approx(0.90, abs=0.005)
    assert stats["ci95"][0] < stats["cohens_kappa"] < stats["ci95"][1]
    assert stats["ci95"][0] >= 0.80
    assert stats["confusion"]["both_yes"] == 47
    assert stats["confusion"]["both_no"] == 48
    assert stats["confusion"]["judge_a_yes_judge_b_no"] == 3
    assert stats["confusion"]["judge_a_no_judge_b_yes"] == 2


def test_stratified_quota_allocation_respects_caps() -> None:
    quotas = _allocate_stratified_quotas({"a": 10, "b": 100, "c": 100, "d": 100}, 100)
    assert sum(quotas.values()) == 100
    assert quotas["a"] == 10
    assert quotas["b"] == 30
    assert quotas["c"] == 30
    assert quotas["d"] == 30


def test_samplers_deterministic_and_stratified(tmp_path: Path) -> None:
    # Build synthetic LoCoMo CSV
    loc_csv = tmp_path / "locomo.csv"
    with loc_csv.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(
            fh,
            fieldnames=["regime", "arm", "qid", "conv_idx", "category", "is_abstention", "gold", "pred_raw", "is_correct", "api_error"],
        )
        w.writeheader()
        for cat in ("1", "2", "4", "5"):
            for i in range(10):
                for arm in ("M1_hybrid_flat", "M4f_soft_telemetry_fallback"):
                    w.writerow(
                        {
                            "regime": "regime_a_unmodified",
                            "arm": arm,
                            "qid": f"loc_{cat}_{i}",
                            "conv_idx": "0",
                            "category": cat,
                            "is_abstention": "1" if cat == "5" else "0",
                            "gold": "UNANSWERABLE" if cat == "5" else f"val_{i}",
                            "pred_raw": "UNANSWERABLE" if cat == "5" else f"val_{i}",
                            "is_correct": "1",
                            "api_error": "0",
                        }
                    )

    s1 = sample_locomo_items(loc_csv, n_target=20, seed="unit_seed")
    s2 = sample_locomo_items(loc_csv, n_target=20, seed="unit_seed")
    assert s1 == s2
    assert len(s1) == 20
    assert len({x["qid"] for x in s1}) == 20
    assert len({x["stratum"] for x in s1}) == 4

    # Build synthetic LongMemEval-S CSV across 6 question types
    lme_csv = tmp_path / "lme.csv"
    qtypes = (
        "knowledge-update",
        "multi-session",
        "single-session-assistant",
        "single-session-preference",
        "single-session-user",
        "temporal-reasoning",
    )
    with lme_csv.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(
            fh,
            fieldnames=["regime", "arm", "qid", "question_type", "gold", "pred_raw", "is_correct", "api_error"],
        )
        w.writeheader()
        for qt in qtypes:
            for i in range(6):
                w.writerow(
                    {
                        "regime": "regime_a_clean_n500",
                        "arm": "M4f_soft_telemetry_fallback",
                        "qid": f"lme_{qt}_{i}",
                        "question_type": qt,
                        "gold": f"gold_{i}",
                        "pred_raw": f"gold_{i}",
                        "is_correct": "1",
                        "api_error": "0",
                    }
                )

    l1 = sample_longmemeval_items(lme_csv, n_target=24, seed="unit_seed")
    assert len(l1) == 24
    assert len({x["qid"] for x in l1}) == 24
    assert {x["stratum"] for x in l1} == set(qtypes)

    # Build synthetic structured R5b matrix CSV
    mat_csv = tmp_path / "mat.csv"
    with mat_csv.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(
            fh,
            fieldnames=["case_id", "family", "arm", "generator", "gold", "pred", "judge_nemotron", "judge_qwen"],
        )
        w.writeheader()
        for fam in ("f1", "f2"):
            for arm in ("C0", "C4"):
                for gen in ("Nemotron_30B", "Qwen2.5_7B"):
                    for i in range(5):
                        w.writerow(
                            {
                                "case_id": f"{fam}_{i}",
                                "family": fam,
                                "arm": arm,
                                "generator": gen,
                                "gold": "65psi",
                                "pred": "65psi",
                                "judge_nemotron": "1",
                                "judge_qwen": "1",
                            }
                        )
    sampled_s, all_s = sample_structured_r5b_items(mat_csv, n_target=16, seed="unit_seed")
    assert len(all_s) == 40
    assert len(sampled_s) == 16
