"""Unit and artifact-contract tests for Wave 6 Track D (`Demo D4`, Guardrail `K4`)."""

from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest

from ek5_wfs_rlvr_env import ABSTAIN, LABEL_F, LABEL_T, LABEL_U
from prospective_realworld_drift_corpus import build_prospective_realworld_drift_suite
from scripts.run_wave6_d4_trained_epistemic_grpo import (
    EXACT_CORE_REWARD,
    FALSE_ABSTAIN_REWARD,
    FORMAT_BONUS_REWARD,
    PREREGISTERED_SEEDS,
    SPLIT_SIZES,
    WRONG_OR_MALFORMED_REWARD,
    RolloutTelemetryTracker,
    assert_four_way_disjoint_splits,
    build_all_four_disjoint_splits,
    derive_cot_from_prompt_text,
    extract_d1_final_answer,
    score_calibrated_wfs_completion,
    score_d1_prose_prediction,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
OSV_CACHE = REPO_ROOT / "artifacts" / "phase2_d1_4node_local_fleet_proof" / "_work" / "osv_cve_seeds.json"
ARTIFACT_DIR = REPO_ROOT / "artifacts" / "wave6_d4_trained_epistemic_grpo"


@pytest.fixture(scope="module")
def all_splits():
    return build_all_four_disjoint_splits()


def test_k4_four_way_splits_sizes_and_pairwise_disjointness_including_noise_retractor(all_splits) -> None:
    assert {k: len(v) for k, v in all_splits.items()} == SPLIT_SIZES
    d1_cases, _, _ = build_prospective_realworld_drift_suite(OSV_CACHE)
    proof = assert_four_way_disjoint_splits(all_splits, d1_cases=d1_cases)
    assert proof["noise_retractor_included_in_hash"] is True
    assert proof["d1_entity_overlap"] == 0
    assert proof["d1_value_overlap"] == 0
    assert len(proof["pairwise_checked_pairs"]) == 6
    # Every split must exercise retracted refutation withdrawals (`noise-retractor-*`) and all 3 labels T, F, U
    for s_name, meta in proof["splits"].items():
        assert len(meta["retractor_entities"]) == 1, f"Expected split-unique noise-retractor in {s_name}"
        assert set(meta["label_counts"].keys()) == {LABEL_T, LABEL_F, LABEL_U}


def test_k4_observable_cot_takes_only_prompt_string_and_never_leaks_store_status(all_splits) -> None:
    sig = inspect.signature(derive_cot_from_prompt_text)
    params = list(sig.parameters.keys())
    assert params == ["prompt_text"], f"CoT builder must accept ONLY prompt_text, got {params}"

    for split_name, eps in all_splits.items():
        for ep in eps:
            cot = derive_cot_from_prompt_text(ep.prompt)
            low = cot.lower()
            for forbidden in ("store_slot_status", "store_slot_reason", "audit_only", "unique_admitted"):
                assert forbidden not in low
            sc = score_calibrated_wfs_completion(cot, ep)
            assert sc["exact_match"] == 1.0, f"Oracle mismatch in {split_name} {ep.episode_id}: {sc}"
            assert sc["core_reward"] == EXACT_CORE_REWARD
            assert sc["format_bonus"] == FORMAT_BONUS_REWARD
            assert sc["total_reward"] == pytest.approx(1.05)


def test_k4_calibrated_reward_closes_025_loophole_and_penalizes_false_abstention(all_splits) -> None:
    eval_eps = all_splits["EVAL"]
    ep_t = next(ep for ep in eval_eps if ep.gold_label == LABEL_T)
    ep_f = next(ep for ep in eval_eps if ep.gold_label == LABEL_F)
    ep_u = next(ep for ep in eval_eps if ep.gold_label == LABEL_U)

    # 1. False abstention on determinate T or F MUST score -0.15 REGARDLESS of verdict (closing +0.25 loophole!)
    for ep_det in (ep_t, ep_f):
        for v_choice in ("T", "F", "U"):
            comp = f"<think>hedging</think>\nVERDICT: {v_choice}\nANSWER: {ABSTAIN}"
            sc = score_calibrated_wfs_completion(comp, ep_det)
            assert sc["core_reward"] == pytest.approx(FALSE_ABSTAIN_REWARD)
            assert sc["format_bonus"] == 0.0
            assert sc["total_reward"] == pytest.approx(FALSE_ABSTAIN_REWARD)
            assert sc["is_false_abstain"] is True

    # 2. Genuine U item keeps +1.00 core (+1.05 with clean <think>) for VERDICT: U + ANSWER: ABSTAIN
    sc_u_ok = score_calibrated_wfs_completion("<think>peer conflict</think>\nVERDICT: U\nANSWER: ABSTAIN", ep_u)
    assert sc_u_ok["core_reward"] == pytest.approx(1.0)
    assert sc_u_ok["format_bonus"] == pytest.approx(0.05)
    assert sc_u_ok["total_reward"] == pytest.approx(1.05)

    # 3. Unclosed <think>, negation, mixed commitment, extra trailing lines, and informal refusals -> -1.00
    bad_cases = [
        f"<think>unclosed reasoning\nVERDICT: {ep_t.gold_label}\nANSWER: {ep_t.gold_value}",
        f"<think>neg</think>\nVERDICT: {ep_t.gold_label}\nANSWER: not {ep_t.gold_value}",
        f"<think>mix</think>\nVERDICT: U\nANSWER: ABSTAIN or {ep_u.candidate_values[0]}",
        "<think>synonym</think>\nVERDICT: U\nANSWER: unknown",
        f"<think>extra</think>\nVERDICT: {ep_t.gold_label}\nANSWER: {ep_t.gold_value}\nEXTRA: commentary",
    ]
    for bad in bad_cases:
        sc_bad = score_calibrated_wfs_completion(bad, ep_t if "ABSTAIN" not in bad else ep_u)
        assert sc_bad["total_reward"] == pytest.approx(WRONG_OR_MALFORMED_REWARD)
        assert sc_bad["format_bonus"] == 0.0


def test_k4_keyword_only_reward_signature_rejects_positional_and_prompt_aliasing(all_splits) -> None:
    tracker = RolloutTelemetryTracker(num_generations=2)
    reward_fn = tracker.make_reward_fn()
    ep0 = all_splits["DEV"][0]
    prompts = [ep0.prompt, ep0.prompt]
    cots = [derive_cot_from_prompt_text(ep0.prompt), "<think>wrong</think>\nVERDICT: U\nANSWER: ABSTAIN"]
    ep_jsons = [ep0.to_json(), ep0.to_json()]

    # Positional call must fail immediately with TypeError
    with pytest.raises(TypeError):
        reward_fn(prompts, cots, ep_jsons)  # type: ignore[misc]

    # Aliasing `completions is prompts` must fail with AssertionError
    with pytest.raises(AssertionError):
        reward_fn(prompts=prompts, completions=prompts, episode=ep_jsons)

    rewards = reward_fn(prompts=prompts, completions=cots, episode=ep_jsons)
    assert len(rewards) == 2
    assert rewards[0] == pytest.approx(1.05)
    assert len(tracker.records) == 1
    assert tracker.records[0]["group_zero_std_frac"] == 0.0


def test_k4_d1_answer_extraction_and_scoring_invariants() -> None:
    ans, unclosed = extract_d1_final_answer("<think>Day 2 had 9.6p1, superseded day 15 by 9.8p1.</think>\nVERDICT: T\nANSWER: 9.8p1")
    assert not unclosed and ans == "9.8p1"
    sc = score_d1_prose_prediction(
        "<think>Day 2 had 9.6p1-grace-timer-env, superseded day 15 by 9.8p1.</think>\nVERDICT: T\nANSWER: 9.8p1",
        "9.8p1",
        ["9.6p1-grace-timer-env"],
    )
    assert sc["strict_exact_match"] == 1
    assert sc["frozen_d1_match"] == 1

    # Unclosed think must fail closed even if gold string appears inside <think>
    sc_trunc = score_d1_prose_prediction("<think>The verified answer is 9.8p1", "9.8p1", ["9.6p1-grace-timer-env"])
    assert sc_trunc["unclosed_think"] is True
    assert sc_trunc["strict_exact_match"] == 0
    assert sc_trunc["frozen_d1_match"] == 0


def test_k4_trained_artifact_bundle_and_receipt_integrity_if_present() -> None:
    manifest_path = ARTIFACT_DIR / "wave6_d4_epistemic_grpo_manifest.json"
    receipt_path = ARTIFACT_DIR / "report_receipt.json"
    if not manifest_path.is_file():
        pytest.skip("GPU artifact bundle not yet materialized")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))

    assert manifest["trained_grpo_checkpoint_present"] is True
    audit = manifest["k4_guardrail_audit"]
    assert audit["sft_completion_only_and_zero_oracle_leak"]["unmasked_prompt_token_count"] == 0
    assert audit["sft_completion_only_and_zero_oracle_leak"]["store_slot_status_leaked"] is False
    assert audit["sft_merge_and_unload_before_grpo"]["max_kl_ref_vs_frozen_sft_logit_diff"] < 1e-4
    assert audit["three_seed_training_telemetry"]["preregistered_seeds"] == list(PREREGISTERED_SEEDS)
    assert audit["three_seed_training_telemetry"]["frac_reward_zero_std_below_080"] is True
    assert audit["three_seed_training_telemetry"]["reload_logit_parity_below_1e4"] is True
    assert audit["peak_memory_within_34gb_cap"] is True

    for rel_name, expected_sha in receipt["artifact_hashes"].items():
        p = ARTIFACT_DIR / rel_name
        assert p.is_file(), f"Missing receipt-hashed file {p}"
        import hashlib

        assert hashlib.sha256(p.read_bytes()).hexdigest() == expected_sha
