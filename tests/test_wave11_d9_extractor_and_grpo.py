"""Fast CPU unit tests for Wave 11 Demo D9 (Guardrail `L4`):
  - Part A: Disjoint synthetic claim extractor corpus (`N=600` train / `N=120` dev),
            Suite-A disjointness + `< 0.25` 5-gram Jaccard, hard-negative two-value contrasts,
            `AuthorityDominatedExocortexV2` plug-in, and DEV-only model selection.
  - Part B: 5-way pairwise-disjoint WFS RLVR splits (`SFT_WARMUP N=128`, `GRPO_RL N=240`,
            `DEV N=80`, `EVAL_WAVE6_LOOK2 N=120`, `EVAL_VIRGIN_V2 N=120`, + `D1 N=60`),
            `0` WFS oracle mismatches, untruncated 4-step prompt-only CoT (`>= 95%`, zero
            truncation on `>6` nodes), and anti-collapse reward math (`E[R_always_abstain] < 0`).
"""

from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path

import numpy as np
import pytest

import ek5_wfs_rlvr_env as rlvr_env
from chelated_epistemic_proposer_v2 import compile_claims_to_ek5_v2
from chelated_fast_edge_extractor import (
    SYNTHETIC_DOMAINS,
    VALID_STANCES,
    SyntheticExtractorExample,
    assert_extractor_corpus_disjoint_from_suite_a,
    assert_train_dev_disjoint,
    build_disjoint_extractor_splits,
    build_exocortex_v2_from_claims,
    is_strict_valid_claim_json,
)
from chelated_raw_text_exocortex import RawDocumentEnvelope, parse_extracted_claim_json
from ek5_wfs_rlvr_env import ABSTAIN, LABEL_F, LABEL_T, LABEL_U, audit_episode
from prospective_realworld_drift_corpus import build_prospective_realworld_drift_suite
from scripts.run_wave11_d9_fast_extractor_and_u_grpo import (
    DEFAULT_D9B_AMENDMENT_PATH,
    EXACT_CORE_REWARD,
    FALSE_ABSTAIN_REWARD,
    FALSE_COMMIT_ON_U_REWARD,
    FORMAT_BONUS_REWARD,
    PARTIAL_VERDICT_MISMATCH_REWARD,
    PREREGISTERED_SEEDS,
    PRIMARY_TWO_SEED_SUBSET,
    PROTOCOL_V2_BATCH_SIZE,
    PROTOCOL_V2_BETA,
    PROTOCOL_V2_GRAD_ACCUM,
    PROTOCOL_V2_GRPO_STEPS,
    PROTOCOL_V2_LORA_ALPHA,
    PROTOCOL_V2_LORA_R,
    PROTOCOL_V2_LR,
    PROTOCOL_V2_NUM_GENERATIONS,
    PROTOCOL_V2_VERSION,
    RETIRED_DEV_DIAGNOSTIC_SEED,
    SLOT_POOL_VIRGIN_V2,
    WAVE6_EVAL_EXPECTED_SHA256,
    WAVE11_SPLIT_SIZES,
    WRONG_DETERMINATE_REWARD,
    BestDevSelector,
    Wave11RolloutTelemetryTracker,
    assemble_wave11_d9_manifest,
    assert_five_way_disjoint_splits,
    build_all_five_disjoint_splits,
    build_and_write_selection_frozen,
    can_resume_seed_checkpoint,
    compute_dev_eval_steps,
    compute_expected_always_abstain_reward,
    compute_grpo_seed_config_digest,
    compute_unscaled_group_advantages,
    derive_untruncated_wfs_cot_from_prompt,
    evaluate_always_abstain_wfs_baseline,
    hash_directory_files,
    load_wave6_pm3_suite_a_baselines,
    resolve_and_verify_trl_grpo_config_kwargs,
    run_part_b_u_grpo_on_spark2,
    run_verify_splits,
    score_epistemic_completion,
    select_best_extractor_on_dev,
    verify_selection_frozen_manifest,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
OSV_CACHE = REPO_ROOT / "artifacts" / "phase2_d1_4node_local_fleet_proof" / "_work" / "osv_cve_seeds.json"


@pytest.fixture(scope="module")
def extractor_splits():
    return build_disjoint_extractor_splits(n_train=600, n_dev=120, seed=20261123)


@pytest.fixture(scope="module")
def five_way_wfs_splits():
    return build_all_five_disjoint_splits()


@pytest.fixture(scope="module")
def suite_a_bundle():
    return build_prospective_realworld_drift_suite(OSV_CACHE)


# ---------------------------------------------------------------------------
# Part A Tests: Disjoint Synthetic Extractor Corpus & Exocortex V2 Plug-In
# ---------------------------------------------------------------------------
def test_part_a_extractor_corpus_balance_and_train_dev_disjoint(extractor_splits) -> None:
    train = extractor_splits["train"]
    dev = extractor_splits["dev"]
    assert len(train) == 600
    assert len(dev) == 120
    assert set(extractor_splits["domains"]) == set(SYNTHETIC_DOMAINS)
    assert extractor_splits["train_stance_counts"] == {s: 150 for s in VALID_STANCES}
    assert extractor_splits["dev_stance_counts"] == {s: 30 for s in VALID_STANCES}
    assert set(extractor_splits["train_domain_counts"].keys()) == set(SYNTHETIC_DOMAINS)
    assert set(extractor_splits["dev_domain_counts"].keys()) == set(SYNTHETIC_DOMAINS)

    td_proof = assert_train_dev_disjoint(train, dev)
    assert td_proof["train_dev_disjoint_passed"] is True
    assert td_proof["entity_overlap"] == 0
    assert td_proof["value_overlap"] == 0
    assert td_proof["text_overlap"] == 0


def test_part_a_assert_extractor_corpus_disjoint_from_suite_a_and_negative_control(
    extractor_splits, suite_a_bundle
) -> None:
    all_synth = extractor_splits["train"] + extractor_splits["dev"]
    proof = assert_extractor_corpus_disjoint_from_suite_a(
        all_synth,
        osv_cache_path=OSV_CACHE,
        max_5gram_jaccard_cap=0.25,
    )
    assert proof["disjoint_passed"] is True
    assert proof["n_suite_a_docs_checked"] == 1020
    assert proof["domain_overlap"] == 0
    assert proof["entity_overlap"] == 0
    assert proof["slot_overlap"] == 0
    assert proof["value_overlap"] == 0
    assert proof["max_5gram_jaccard"] < 0.25
    assert proof["max_echo_template_5gram_jaccard"] < 0.25

    # Negative control: injecting a Suite A entity/text MUST fail closed!
    suite_cases, suite_docs, _ = suite_a_bundle
    leaked = SyntheticExtractorExample(
        doc=suite_docs[0],
        target_entity=str(suite_cases[0]["subject"]),
        target_slot="leaked_slot",
        target_value=str(suite_cases[0]["gold"]),
        target_stance="ASSERT",
        domain="pharma_clinical",
        archetype="negative_control",
        target_json="{}",
    )
    with pytest.raises(AssertionError):
        assert_extractor_corpus_disjoint_from_suite_a([leaked], osv_cache_path=OSV_CACHE)


def test_part_a_hard_negative_contrasts_and_ek5_v2_conflict_registration(extractor_splits) -> None:
    train = extractor_splits["train"]
    archetypes = {ex.archetype for ex in train}
    assert "tier1_rumor_despite_denial_assert" in archetypes
    assert "two_value_authoritative_debunk" in archetypes
    assert "party_b_two_value_counter_filing_refute" in archetypes

    # Every synthetic target_json must pass strict JSON validation and round-trip through parse_extracted_claim_json
    for ex in train[:80]:
        ok, parsed = is_strict_valid_claim_json(ex.target_json)
        assert ok is True and parsed is not None
        rec = parse_extracted_claim_json(ex.doc.doc_id, ex.target_json)
        assert rec.stance == ex.target_stance
        assert rec.claim_value == ex.target_value

    # Verify Party-B counter-filing two-value contrast registers a genuine CONTESTED conflict in compile_claims_to_ek5_v2
    doc_a = RawDocumentEnvelope(
        doc_id="synth_pa",
        text="Primary adversarial submission (Side Alpha) on CLN-ONCO-9901-A: challenges rival findings and argues liver enzyme safety verdict equals GRADE1_TRANSIENT_ELEVATION_ONLY.",
        published_ts=12,
        ingested_ts=12,
        source_channel="sec_edgar_filing",
        domain="pharma_clinical",
    )
    doc_b = RawDocumentEnvelope(
        doc_id="synth_pb",
        text="Equal-standing rebuttal dossier (Side Beta) on CLN-ONCO-9901-A: rejects Side Alpha's GRADE1_TRANSIENT_ELEVATION_ONLY position and instead certifies GRADE3_DRUG_INDUCED_HEPATITIS.",
        published_ts=12,
        ingested_ts=13,
        source_channel="sec_edgar_filing",
        domain="pharma_clinical",
    )
    claims = [
        parse_extracted_claim_json(
            "synth_pa",
            '{"entity":"CLN-ONCO-9901-A","slot":"hepatotoxicity_signal_grade","value":"GRADE1_TRANSIENT_ELEVATION_ONLY","stance":"REFUTE"}',
        ),
        parse_extracted_claim_json(
            "synth_pb",
            '{"entity":"CLN-ONCO-9901-A","slot":"hepatotoxicity_signal_grade","value":"GRADE3_DRUG_INDUCED_HEPATITIS","stance":"REFUTE"}',
        ),
    ]
    store, _events, _atts = compile_claims_to_ek5_v2([doc_a, doc_b], claims)
    assert store.eligibility["synth_pa"] == "CONTESTED"
    assert store.eligibility["synth_pb"] == "CONTESTED"

    # Plug directly into AuthorityDominatedExocortexV2 -> deterministic CONTESTED_ABSTAIN!
    embs = np.array([[1.0, 0.0, 0.0, 0.0], [0.99, 0.1, 0.0, 0.0]], dtype=np.float32)
    exo = build_exocortex_v2_from_claims([doc_a, doc_b], claims, embs, embs)
    sel = exo.select_context_for_query(np.array([0.9, 0.9], dtype=np.float32), "What is CLN-ONCO-9901-A verdict?")
    assert sel["status"] == "CONTESTED_ABSTAIN"


def test_part_a_dev_only_selection_and_wave6_baselines() -> None:
    cands = [
        {
            "tag": "qwen2_5_0_5b",
            "base_model_id": "Qwen/Qwen2.5-0.5B-Instruct",
            "train_summary": {"total_params": 500_000_000, "merged_model_dir": "/tmp/m05"},
            "dev_metrics": {
                "json_valid_rate": 0.995,
                "stance_accuracy": 0.95,
                "value_exact_accuracy": 0.93,
                "full_tuple_accuracy": 0.92,
                "ms_per_17doc_batch_mean": 180.0,
            },
        },
        {
            "tag": "qwen2_5_1_5b",
            "base_model_id": "Qwen/Qwen2.5-1.5B-Instruct",
            "train_summary": {"total_params": 1_500_000_000, "merged_model_dir": "/tmp/m15"},
            "dev_metrics": {
                "json_valid_rate": 1.0,
                "stance_accuracy": 0.98,
                "value_exact_accuracy": 0.97,
                "full_tuple_accuracy": 0.96,
                "ms_per_17doc_batch_mean": 420.0,
            },
        },
    ]
    sel = select_best_extractor_on_dev(cands)
    assert sel["selected_model_id"] == "Qwen/Qwen2.5-0.5B-Instruct"

    baselines = load_wave6_pm3_suite_a_baselines()
    assert baselines["mem0_live_infer_true_accuracy"] == pytest.approx(0.8166666667, abs=1e-4)
    assert baselines["wave6_30b_extractor_accuracy"] == pytest.approx(0.9666666667, abs=1e-4)
    assert len(baselines["mem0_live_infer_true_per_id"]) == 60
    assert len(baselines["wave6_30b_extractor_per_id"]) == 60


# ---------------------------------------------------------------------------
# Part B Tests: 5-Way Disjoint Splits, Virgin V2, Untruncated CoT, Anti-Collapse
# ---------------------------------------------------------------------------
def test_part_b_five_way_disjoint_splits_oracle_zero_mismatches_and_wave6_sha(
    five_way_wfs_splits, suite_a_bundle
) -> None:
    assert {k: len(v) for k, v in five_way_wfs_splits.items()} == WAVE11_SPLIT_SIZES
    d1_cases, _, _ = suite_a_bundle
    proof = assert_five_way_disjoint_splits(five_way_wfs_splits, d1_cases=d1_cases, verify_wave6_eval_sha=True)

    assert len(proof["pairwise_checked_pairs"]) == 10  # C(5, 2) = 10 pairs
    assert proof["noise_retractor_included_in_hash"] is True
    assert proof["virgin_v2_vs_base_slot_pool_overlap"] == 0
    assert proof["d1_entity_overlap"] == 0
    assert proof["d1_value_overlap"] == 0
    assert proof["wave6_eval_sha256_matches_byte_identical"] is True
    assert proof["splits"]["EVAL_WAVE6_LOOK2"]["split_prompt_bundle_sha256"] == WAVE6_EVAL_EXPECTED_SHA256
    assert len(proof["eval_virgin_v2_presealed_sha256"]) == 64

    # Verify SLOT_POOL_VIRGIN_V2 has 4 brand-new tuples disjoint from all 8 ek5_wfs_rlvr_env.SLOT_POOL entries
    assert len(SLOT_POOL_VIRGIN_V2) == 4
    base_ents = {e.casefold() for e, _s, _vs in rlvr_env.SLOT_POOL}
    v2_ents = {e.casefold() for e, _s, _vs in SLOT_POOL_VIRGIN_V2}
    assert v2_ents.isdisjoint(base_ents)

    # Verify 0 WFS 3-way oracle mismatches across ALL 688 generated episodes + balanced T/F/U
    for s_name, eps in five_way_wfs_splits.items():
        lbls = proof["splits"][s_name]["label_counts"]
        assert set(lbls.keys()) == {LABEL_T, LABEL_F, LABEL_U}
        for ep in eps:
            aud = audit_episode(ep)
            mismatches = sum(v for k, v in aud.items() if k != "nodes_checked")
            assert mismatches == 0, f"WFS 3-way oracle mismatch in {s_name}/{ep.episode_id}: {aud}"


def test_part_b_untruncated_wfs_cot_prompt_only_accuracy_and_no_node_truncation(five_way_wfs_splits) -> None:
    sig = inspect.signature(derive_untruncated_wfs_cot_from_prompt)
    assert list(sig.parameters.keys()) == ["prompt_text"]

    total = 0
    exact = 0
    saw_gt6_target_nodes = False

    for split_name, eps in five_way_wfs_splits.items():
        for ep in eps:
            total += 1
            cot = derive_untruncated_wfs_cot_from_prompt(ep.prompt)
            low = cot.lower()
            for forbidden in (
                "store_slot_status",
                "store_slot_reason",
                "store_query_reason",
                "unique_admitted",
                "audit_only",
                "slot_contested",
            ):
                assert forbidden not in low

            # Verify all 4 structured steps are present
            assert "(1) Target" in cot
            assert "(2) Edges:" in cot
            assert "(3) WFS fixpoint:" in cot
            assert "(4) " in cot

            # Verify ZERO truncation on episodes with >6 target nodes!
            target_ev_ids = [
                ev.event_id
                for ev in ep.events
                if ev.entity == ep.entity
                and ev.slot == ep.slot
                and (ep.as_of is None or ev.tx_time <= ep.as_of[1])
            ]
            if len(target_ev_ids) > 6:
                saw_gt6_target_nodes = True
                for tid in target_ev_ids:
                    assert f"{tid}=" in cot, f"Target node {tid} truncated in {split_name}/{ep.episode_id}"

            sc = score_epistemic_completion(cot, ep)
            if sc["exact_match"] == 1.0:
                exact += 1
            assert sc["exact_match"] == 1.0, f"CoT mismatch in {split_name}/{ep.episode_id}: {sc}"
            assert sc["total_reward"] == pytest.approx(EXACT_CORE_REWARD + FORMAT_BONUS_REWARD)

    assert saw_gt6_target_nodes is True, "Expected episodes with >6 target nodes to verify untruncated trace"
    assert (exact / total) >= 0.95


def test_part_b_anti_collapse_reward_math_and_edge_cases(five_way_wfs_splits) -> None:
    # 0. Protocol v2 structural anti-collapse tie: FALSE_ABSTAIN_REWARD == WRONG_DETERMINATE_REWARD == -1.00
    assert FALSE_ABSTAIN_REWARD == pytest.approx(-1.00)
    assert WRONG_DETERMINATE_REWARD == pytest.approx(-1.00)
    assert FALSE_ABSTAIN_REWARD == WRONG_DETERMINATE_REWARD

    # 1. Analytical E[R_always_abstain] < 0 on balanced 1/3 T, 1/3 F, 1/3 U -> -0.95/3 = -0.316667
    e_abstain = compute_expected_always_abstain_reward(1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)
    assert e_abstain == pytest.approx(-0.95 / 3.0, abs=1e-6)
    assert e_abstain < 0.0

    # 2. Empirical Always_Abstain_Baseline on EVAL_VIRGIN_V2 (40 T, 40 F, 40 U) -> exact -0.3167 < 0
    abs_virgin = evaluate_always_abstain_wfs_baseline(five_way_wfs_splits["EVAL_VIRGIN_V2"])
    assert abs_virgin["abstain_rate"] == pytest.approx(1.0)
    assert abs_virgin["determinate_exact_match"] == pytest.approx(0.0)
    assert abs_virgin["u_abstention_exact_match"] == pytest.approx(1.0)
    assert abs_virgin["mean_total_reward"] < 0.0
    assert abs_virgin["mean_total_reward"] == pytest.approx(-0.95 / 3.0, abs=1e-3)

    # 3. Edge cases on T, F, U episodes
    v2_eps = five_way_wfs_splits["EVAL_VIRGIN_V2"]
    ep_t = next(ep for ep in v2_eps if ep.gold_label == LABEL_T)
    ep_f = next(ep for ep in v2_eps if ep.gold_label == LABEL_F)
    ep_u = next(ep for ep in v2_eps if ep.gold_label == LABEL_U)

    # Exact + clean <think> -> +1.05; exact without <think> -> +1.00 (no format bonus)
    sc_t_clean = score_epistemic_completion(f"<think>ok</think>\nVERDICT: T\nANSWER: {ep_t.gold_value}", ep_t)
    assert sc_t_clean["total_reward"] == pytest.approx(EXACT_CORE_REWARD + FORMAT_BONUS_REWARD)
    sc_t_bare = score_epistemic_completion(f"VERDICT: T\nANSWER: {ep_t.gold_value}", ep_t)
    assert sc_t_bare["total_reward"] == pytest.approx(EXACT_CORE_REWARD)
    assert sc_t_bare["format_bonus"] == 0.0

    # Right answer with wrong verdict -> +0.75 (0 format bonus, exact_match=0)
    sc_t_mismatch = score_epistemic_completion(f"<think>ok</think>\nVERDICT: F\nANSWER: {ep_t.gold_value}", ep_t)
    assert sc_t_mismatch["total_reward"] == pytest.approx(PARTIAL_VERDICT_MISMATCH_REWARD)
    assert sc_t_mismatch["format_bonus"] == 0.0
    assert sc_t_mismatch["exact_match"] == 0.0

    # False abstention on T or F -> -1.00 (== WRONG_DETERMINATE_REWARD) REGARDLESS of verdict!
    for ep_det in (ep_t, ep_f):
        for v_choice in ("T", "F", "U"):
            sc_fa = score_epistemic_completion(f"<think>hedging</think>\nVERDICT: {v_choice}\nANSWER: {ABSTAIN}", ep_det)
            assert sc_fa["core_reward"] == pytest.approx(-1.00)
            assert sc_fa["total_reward"] == pytest.approx(-1.00)
            assert sc_fa["format_bonus"] == 0.0
            assert sc_fa["is_false_abstain"] is True

    # 4. 0-correct determinate T/F groups mixing ABSTAIN, wrong determinate, and malformed MUST tie at -1.00 (std == 0.0)!
    for ep_det in (ep_t, ep_f):
        wrong_det_val = next(v for v in ep_det.candidate_values if v.casefold() != (ep_det.gold_value or "").casefold())
        zero_correct_group = [
            f"<think>hedge</think>\nVERDICT: U\nANSWER: {ABSTAIN}",
            f"<think>hedge2</think>\nVERDICT: T\nANSWER: {ABSTAIN}",
            f"<think>wrong1</think>\nVERDICT: {ep_det.gold_label}\nANSWER: {wrong_det_val}",
            f"<think>wrong2</think>\nVERDICT: F\nANSWER: {wrong_det_val}",
            "<think>unclosed\nVERDICT: T\nANSWER: bad",
            f"<think>neg</think>\nVERDICT: T\nANSWER: not {ep_det.gold_value}",
            "<think>syn</think>\nVERDICT: U\nANSWER: unknown",
            f"<think>wrong3</think>\nVERDICT: T\nANSWER: {wrong_det_val}",
        ]
        grp_tracker = Wave11RolloutTelemetryTracker(num_generations=8)
        grp_rfn = grp_tracker.make_reward_fn()
        grp_rewards = grp_rfn(
            prompts=[ep_det.prompt] * 8,
            completions=zero_correct_group,
            episode=[ep_det.to_json()] * 8,
        )
        assert all(r == pytest.approx(-1.00) for r in grp_rewards)
        assert float(np.std(grp_rewards)) == pytest.approx(0.0, abs=1e-12)
        assert grp_tracker.records[-1]["group_zero_std_frac"] == pytest.approx(1.0)
        adv_zero = compute_unscaled_group_advantages(grp_rewards, scale_rewards="none")
        assert np.allclose(adv_zero, np.zeros(8), atol=1e-12)

    # False commit on U -> -1.15!
    sc_u_fc = score_epistemic_completion(
        f"<think>guessing</think>\nVERDICT: T\nANSWER: {ep_u.candidate_values[0]}", ep_u
    )
    assert sc_u_fc["core_reward"] == pytest.approx(FALSE_COMMIT_ON_U_REWARD)
    assert sc_u_fc["total_reward"] == pytest.approx(FALSE_COMMIT_ON_U_REWARD)
    assert sc_u_fc["is_false_commit_on_u"] is True

    # Malformed / unclosed / negation / mixed / non-canonical / wrong determinate -> -1.00
    wrong_val = next(v for v in ep_t.candidate_values if v.casefold() != (ep_t.gold_value or "").casefold())
    bad_cases = [
        (f"<think>wrong</think>\nVERDICT: T\nANSWER: {wrong_val}", ep_t),
        (f"<think>unclosed\nVERDICT: T\nANSWER: {ep_t.gold_value}", ep_t),
        (f"<think>neg</think>\nVERDICT: T\nANSWER: not {ep_t.gold_value}", ep_t),
        (f"<think>mix</think>\nVERDICT: U\nANSWER: ABSTAIN or {ep_u.candidate_values[0]}", ep_u),
        ("<think>syn</think>\nVERDICT: U\nANSWER: unknown", ep_u),
        (f"<think>extra</think>\nVERDICT: T\nANSWER: {ep_t.gold_value}\nEXTRA: tail", ep_t),
    ]
    for txt, ep_obj in bad_cases:
        sc_bad = score_epistemic_completion(txt, ep_obj)
        assert sc_bad["total_reward"] == pytest.approx(WRONG_DETERMINATE_REWARD)
        assert sc_bad["format_bonus"] == 0.0

    # Keyword-only TRL reward wrapper rejects positional args and prompt aliasing
    tracker = Wave11RolloutTelemetryTracker(num_generations=2)
    rfn = tracker.make_reward_fn()
    with pytest.raises(TypeError):
        rfn([ep_t.prompt], ["x"], [ep_t.to_json()])  # type: ignore[misc]
    with pytest.raises(AssertionError):
        rfn(prompts=[ep_t.prompt], completions=[ep_t.prompt], episode=[ep_t.to_json()])
    rs = rfn(
        prompts=[ep_t.prompt, ep_u.prompt],
        completions=[
            derive_untruncated_wfs_cot_from_prompt(ep_t.prompt),
            f"<think>commit</think>\nVERDICT: T\nANSWER: {ep_u.candidate_values[0]}",
        ],
        episode=[ep_t.to_json(), ep_u.to_json()],
    )
    assert rs == [pytest.approx(1.05), pytest.approx(FALSE_COMMIT_ON_U_REWARD)]


# ---------------------------------------------------------------------------
# Protocol v2 Tests: TRL Startup Check, Config-Hash Resume, BestDev, Freeze
# ---------------------------------------------------------------------------
def test_protocol_v2_defaults_and_trl_scale_rewards_self_check() -> None:
    assert RETIRED_DEV_DIAGNOSTIC_SEED == 42
    assert PREREGISTERED_SEEDS == (137, 2026, 31415)
    assert PRIMARY_TWO_SEED_SUBSET == (137, 2026)
    assert PROTOCOL_V2_GRPO_STEPS == 60
    assert PROTOCOL_V2_LR == pytest.approx(5e-6)
    assert PROTOCOL_V2_BETA == pytest.approx(0.05)
    assert PROTOCOL_V2_LORA_R == 16
    assert PROTOCOL_V2_LORA_ALPHA == 32
    assert PROTOCOL_V2_BATCH_SIZE == 8
    assert PROTOCOL_V2_GRAD_ACCUM == 8
    assert PROTOCOL_V2_NUM_GENERATIONS == 8

    # 1. Spark #2 trl 1.18.2 signature (`scale_rewards: str = 'group'`, `loss_type: str = 'dapo'`)
    class _Trl1182GRPOConfig:
        def __init__(
            self,
            *,
            scale_rewards: str = "group",
            loss_type: str = "dapo",
            beta: float = 0.0,
            max_prompt_length: int = 512,
        ) -> None:
            self.scale_rewards = scale_rewards
            self.loss_type = loss_type
            self.beta = beta
            self.max_prompt_length = max_prompt_length

    class _Trl1182GRPOTrainer:
        def _generate_and_score_completions(self) -> None:
            _ = "scale_rewards"

    res_1182 = resolve_and_verify_trl_grpo_config_kwargs(_Trl1182GRPOConfig, _Trl1182GRPOTrainer)
    assert res_1182["scale_rewards"] == "none"
    assert res_1182["loss_type"] == "dr_grpo"
    assert res_1182["loss_type_supported"] is True
    assert res_1182["extra_kwargs"]["scale_rewards"] == "none"
    assert res_1182["extra_kwargs"]["loss_type"] == "dr_grpo"
    sc = res_1182["unscaled_advantage_self_check"]
    assert sc["passed"] is True
    assert sc["primary_probe_advantages"] == pytest.approx([1.5, -0.5, -0.5, -0.5])
    assert sc["trainer_source_checked"] is True

    # 2. Bool-typed legacy `scale_rewards: bool = True` -> passes `False`
    class _BoolGRPOConfig:
        def __init__(self, *, scale_rewards: bool = True, beta: float = 0.0) -> None:
            self.scale_rewards = scale_rewards
            self.beta = beta

    res_bool = resolve_and_verify_trl_grpo_config_kwargs(_BoolGRPOConfig)
    assert res_bool["scale_rewards"] is False
    assert res_bool["loss_type"] is None
    assert res_bool["loss_type_supported"] is False

    # 3. Missing `scale_rewards` -> fail closed!
    class _UnsupportedGRPOConfig:
        def __init__(self, *, beta: float = 0.0) -> None:
            self.beta = beta

    with pytest.raises(RuntimeError, match="scale_rewards"):
        resolve_and_verify_trl_grpo_config_kwargs(_UnsupportedGRPOConfig)


def test_protocol_v2_config_digest_resume_isolation(tmp_path: Path) -> None:
    base_kw = dict(
        seed=137,
        run_steps=60,
        lr=5e-6,
        beta=0.05,
        lora_r=16,
        lora_alpha=32,
        grad_accum=8,
        scale_rewards="none",
        loss_type="dr_grpo",
        false_abstain_reward=-1.00,
        script_sha256="a" * 64,
        merged_sft_sha256="b" * 64,
        amendment_sha256="c" * 64,
        grpo_split_sha256="d" * 64,
        dev_split_sha256="e" * 64,
    )
    d_base = compute_grpo_seed_config_digest(**base_kw)
    assert len(d_base) == 64

    # Every single Protocol v2 config field MUST change the digest!
    mutations = [
        {"seed": 42},
        {"run_steps": 160},
        {"lr": 2e-5},
        {"beta": 0.02},
        {"lora_r": 32},
        {"grad_accum": 2},
        {"scale_rewards": "group"},
        {"loss_type": "dapo"},
        {"false_abstain_reward": -0.55},
        {"script_sha256": "f" * 64},
        {"merged_sft_sha256": "1" * 64},
        {"amendment_sha256": "2" * 64},
    ]
    for mut in mutations:
        d_mut = compute_grpo_seed_config_digest(**{**base_kw, **mut})
        assert d_mut != d_base, f"Digest failed to change on mutation {mut}"

    # Verify stale checkpoint (e.g. wrong digest or legacy R0 schema) is NEVER resumed
    fin_dir = tmp_path / "final_step_adapter"
    best_dir = tmp_path / "best_dev_adapter"
    fin_dir.mkdir(parents=True)
    (fin_dir / "adapter_model.safetensors").write_bytes(b"weights-v2")
    fin_hashes = hash_directory_files(fin_dir)

    ckpt_p = tmp_path / "_ckpt_v2_seed_137.json"
    # Legacy R0 checkpoint without protocol_version/config_digest -> rejected
    ckpt_p.write_text(json.dumps({"seed_run": {"seed": 42}}), encoding="utf-8")
    ok, rec, reason = can_resume_seed_checkpoint(ckpt_p, d_base, fin_dir, best_dir)
    assert ok is False and rec is None and reason == "protocol_version_mismatch"

    # Mismatched config digest (e.g., lr=2e-5 stale checkpoint) -> rejected
    stale_digest = compute_grpo_seed_config_digest(**{**base_kw, "lr": 2e-5})
    ckpt_p.write_text(
        json.dumps(
            {
                "protocol_version": PROTOCOL_V2_VERSION,
                "config_digest": stale_digest,
                "seed_train_record": {
                    "fell_back_to_step0_sft": True,
                    "final_adapter_sha256": fin_hashes,
                },
            }
        ),
        encoding="utf-8",
    )
    ok, rec, reason = can_resume_seed_checkpoint(ckpt_p, d_base, fin_dir, best_dir)
    assert ok is False and reason == "config_digest_mismatch"

    # Exact digest + matching adapter hashes -> accepted
    ckpt_p.write_text(
        json.dumps(
            {
                "protocol_version": PROTOCOL_V2_VERSION,
                "config_digest": d_base,
                "seed_train_record": {
                    "fell_back_to_step0_sft": True,
                    "final_adapter_sha256": fin_hashes,
                },
            }
        ),
        encoding="utf-8",
    )
    ok, rec, reason = can_resume_seed_checkpoint(ckpt_p, d_base, fin_dir, best_dir)
    assert ok is True and rec is not None and reason == "ok"

    # Tampering 1 byte in adapter_model.safetensors -> rejected!
    (fin_dir / "adapter_model.safetensors").write_bytes(b"weights-v2-tampered")
    ok, rec, reason = can_resume_seed_checkpoint(ckpt_p, d_base, fin_dir, best_dir)
    assert ok is False and reason == "final_adapter_hash_mismatch"


def test_protocol_v2_best_dev_selector_and_step0_sft_fallback() -> None:
    assert compute_dev_eval_steps(60) == (10, 20, 30, 40, 50, 60)
    assert compute_dev_eval_steps(6) == (1, 2, 3, 4, 5, 6)

    sft_dev = {
        "exact_match": 0.65,
        "determinate_exact_match": 0.70,
        "u_abstention_exact_match": 0.55,
        "mean_total_reward": 0.35,
        "abstain_rate": 0.30,
        "false_abstain_rate_on_determinate": 0.05,
        "false_commit_rate_on_u": 0.45,
    }

    # Case 1: ALL step > 0 checkpoints collapse below `sft_det_em - 0.03` (0.67) even with higher overall EM
    sel_collapse = BestDevSelector(sft_dev, non_inferiority_margin=0.03)
    assert len(sel_collapse.history) == 1
    assert sel_collapse.history[0]["step"] == 0
    for step, det_em, ov_em, abst in ((10, 0.65, 0.72, 0.50), (20, 0.50, 0.66, 0.75), (60, 0.20, 0.45, 0.85)):
        rec, should_save = sel_collapse.record_step(
            step,
            {
                **sft_dev,
                "exact_match": ov_em,
                "determinate_exact_match": det_em,
                "abstain_rate": abst,
            },
        )
        assert rec["eligible"] is False
        assert should_save is False
    summ_c = sel_collapse.summary()
    assert summ_c["best_dev_step"] == 0
    assert summ_c["best_dev_status"] == "none_eligible_fell_back_to_step0_sft"
    assert summ_c["fell_back_to_step0_sft"] is True
    assert summ_c["eligible_grpo_steps_count"] == 0

    # Case 2: Step 20 clears det_em >= 0.67 and beats SFT exact_match -> promoted! Step 30 collapses -> rejected!
    sel_ok = BestDevSelector(sft_dev, non_inferiority_margin=0.03)
    rec10, save10 = sel_ok.record_step(10, {**sft_dev, "exact_match": 0.62, "determinate_exact_match": 0.68})
    assert rec10["eligible"] is True and save10 is False
    rec20, save20 = sel_ok.record_step(20, {**sft_dev, "exact_match": 0.74, "determinate_exact_match": 0.69})
    assert rec20["eligible"] is True and save20 is True
    rec30, save30 = sel_ok.record_step(30, {**sft_dev, "exact_match": 0.78, "determinate_exact_match": 0.60})
    assert rec30["eligible"] is False and save30 is False
    summ_ok = sel_ok.summary()
    assert summ_ok["best_dev_step"] == 20
    assert summ_ok["best_dev_status"] == "selected_eligible_grpo_step"
    assert summ_ok["fell_back_to_step0_sft"] is False
    assert summ_ok["eligible_grpo_steps_count"] == 2

    # Case 3: Eligible step > 0 exists, but none outperform step=0 SFT -> retains step=0 SFT
    sel_sft_wins = BestDevSelector(sft_dev, non_inferiority_margin=0.03)
    rec10_b, save10_b = sel_sft_wins.record_step(10, {**sft_dev, "exact_match": 0.60, "determinate_exact_match": 0.68})
    assert rec10_b["eligible"] is True and save10_b is False
    summ_sft = sel_sft_wins.summary()
    assert summ_sft["best_dev_step"] == 0
    assert summ_sft["best_dev_status"] == "step0_sft_retained_no_eligible_step_improved"
    assert summ_sft["fell_back_to_step0_sft"] is True


def test_protocol_v2_selection_frozen_out_of_loop_gate_and_tamper_rejection(tmp_path: Path) -> None:
    sft_dir = tmp_path / "sft_merged_model"
    sft_dir.mkdir(parents=True)
    (sft_dir / "config.json").write_text('{"model_type":"qwen2"}\n', encoding="utf-8")
    (sft_dir / "model.safetensors").write_bytes(b"sft-merged-bytes")

    amend_p = tmp_path / "d9b_amendment_v1.json"
    amend_p.write_text('{"schema_version":"wave11_d9b_amendment_v1"}\n', encoding="utf-8")

    script_p = tmp_path / "mock_script.py"
    script_p.write_text("# mock script\n", encoding="utf-8")

    seed_train_records: dict[str, dict[str, object]] = {}
    for seed_val, fell_back in ((137, False), (2026, True), (31415, False)):
        sk = f"seed_{seed_val}"
        fin_dir = tmp_path / sk / "final_step_adapter"
        fin_dir.mkdir(parents=True)
        (fin_dir / "adapter_model.safetensors").write_bytes(f"final-{seed_val}".encode("utf-8"))
        best_dir: Path | None = None
        if not fell_back:
            best_dir = tmp_path / sk / "best_dev_adapter"
            best_dir.mkdir(parents=True)
            (best_dir / "adapter_model.safetensors").write_bytes(f"best-{seed_val}".encode("utf-8"))
        seed_train_records[sk] = {
            "seed": seed_val,
            "config_digest": f"{seed_val:064x}",
            "final_adapter_dir": str(fin_dir),
            "best_dev_adapter_dir": None if best_dir is None else str(best_dir),
            "best_dev_step": 0 if fell_back else 30,
            "best_dev_status": "none_eligible_fell_back_to_step0_sft" if fell_back else "selected_eligible_grpo_step",
            "fell_back_to_step0_sft": fell_back,
        }

    out_dir = tmp_path / "out"
    frozen_p, frozen_payload = build_and_write_selection_frozen(
        output_dir=out_dir,
        merged_sft_dir=sft_dir,
        seed_train_records=seed_train_records,
        amendment_path=amend_p,
        script_path=script_p,
        presealed_splits_sha256={"EVAL_VIRGIN_V2": "a" * 64},
    )
    assert frozen_p.is_file()
    assert frozen_payload["in_loop_holdout_evaluations_performed"] == 0
    assert frozen_payload["seeds"]["seed_2026"]["fell_back_to_step0_sft"] is True
    assert frozen_payload["seeds"]["seed_2026"]["best_dev_source"] == "merged_sft_dir"
    assert frozen_payload["seeds"]["seed_2026"]["best_dev_adapter_dir"] is None

    # Clean verification passes
    verified = verify_selection_frozen_manifest(frozen_p)
    assert verified["schema_version"] == "wave11_d9b_selection_frozen_v2"

    # Tamper 1 byte in seed_137 final_step_adapter -> MUST refuse to score!
    fin_137_file = tmp_path / "seed_137" / "final_step_adapter" / "adapter_model.safetensors"
    orig_fin = fin_137_file.read_bytes()
    fin_137_file.write_bytes(orig_fin + b"X")
    with pytest.raises(AssertionError, match="final_step_adapter SHA-256 mismatch"):
        verify_selection_frozen_manifest(frozen_p)
    fin_137_file.write_bytes(orig_fin)

    # Tamper 1 byte in d9b_amendment_v1.json -> MUST refuse to score!
    orig_amend = amend_p.read_text(encoding="utf-8")
    amend_p.write_text(orig_amend + " ", encoding="utf-8")
    with pytest.raises(AssertionError, match="amendment SHA-256 mismatch"):
        verify_selection_frozen_manifest(frozen_p)
    amend_p.write_text(orig_amend, encoding="utf-8")

    # Tamper 1 byte in merged_sft_dir -> MUST refuse to score!
    sft_file = sft_dir / "model.safetensors"
    orig_sft = sft_file.read_bytes()
    sft_file.write_bytes(orig_sft + b"Y")
    with pytest.raises(AssertionError, match="merged_sft_dir SHA-256 mismatch"):
        verify_selection_frozen_manifest(frozen_p)
    sft_file.write_bytes(orig_sft)

    # Restored -> passes again
    verify_selection_frozen_manifest(frozen_p)


def test_d9b_amendment_v1_json_and_out_of_loop_source_audit(five_way_wfs_splits) -> None:
    assert DEFAULT_D9B_AMENDMENT_PATH.is_file()
    amend = json.loads(DEFAULT_D9B_AMENDMENT_PATH.read_text(encoding="utf-8"))
    proof = assert_five_way_disjoint_splits(five_way_wfs_splits, verify_wave6_eval_sha=True)

    assert amend["schema_version"] == "wave11_d9b_amendment_v1"
    assert amend["presealed_holdout_digests"]["EVAL_VIRGIN_V2_sha256"] == proof["eval_virgin_v2_presealed_sha256"]
    assert amend["presealed_holdout_digests"]["EVAL_WAVE6_LOOK2_sha256"] == WAVE6_EVAL_EXPECTED_SHA256
    assert amend["presealed_holdout_digests"]["eval_virgin_v2_scope"] == "virgin_for_grpo_arm_only"
    assert amend["prior_r0_run_and_abort_forensics"]["abort_location_and_cause"]["abort_line"] == 2628
    assert amend["prior_r0_run_and_abort_forensics"]["abort_location_and_cause"]["aborted_before_eval_virgin_v2"] is True
    assert amend["protocol_v2_bundled_specification"]["protocol_v2_single_knob_attribution_claimed"] is False
    knobs = amend["protocol_v2_bundled_specification"]["bundled_knobs"]
    assert knobs["false_abstain_reward"] == -1.0
    assert knobs["wrong_determinate_reward"] == -1.0
    assert knobs["scale_rewards"] == "none"
    assert knobs["loss_type"] == "dr_grpo"
    assert knobs["grpo_steps"] == 60
    assert knobs["gradient_accumulation_steps"] == 8
    assert knobs["prompt_groups_seen_per_seed"] == 480
    conf = amend["confirmatory_experimental_protocol"]
    assert conf["retired_dev_diagnostic_seed"] == 42
    assert conf["confirmatory_seeds"] == [137, 2026, 31415]
    assert conf["two_seed_sensitivity_subset"] == [137, 2026]
    assert conf["null_result_and_retune_precommitment"]["null_verdict_text"] == "K4 stands under Protocol v2"

    # Source-level out-of-loop holdout audit: ZERO holdout evaluations before `build_and_write_selection_frozen`!
    src = inspect.getsource(run_part_b_u_grpo_on_spark2)
    freeze_pos = src.index("build_and_write_selection_frozen(")
    pre_freeze = src[:freeze_pos]
    for forbidden_pre_freeze in (
        'eval_wfs(base_mdl, splits["EVAL_VIRGIN_V2"])',
        'eval_wfs(sft_reloaded, splits["EVAL_VIRGIN_V2"])',
        'eval_wfs(reloaded_final_peft, splits["EVAL_VIRGIN_V2"])',
        'eval_wfs(best_peft, splits["EVAL_VIRGIN_V2"])',
        'evaluate_always_abstain_wfs_baseline(splits["EVAL_VIRGIN_V2"])',
        'splits["EVAL_VIRGIN_V2"]',
        'splits["EVAL_WAVE6_LOOK2"]',
        "d1_user_prompts_k17",
        "d1_user_prompts_k5",
    ):
        assert forbidden_pre_freeze not in pre_freeze, f"Holdout reference leaked pre-freeze: {forbidden_pre_freeze}"
    assert "verify_selection_frozen_manifest(frozen_path)" in src[freeze_pos:]


def test_cli_verify_splits_and_assemble_manifest_roundtrip(tmp_path: Path) -> None:
    verify_res = run_verify_splits(tmp_path / "verify_out", OSV_CACHE)
    assert verify_res["status"] == "VERIFY_SPLITS_OK"
    assert verify_res["part_a_extractor_splits"]["suite_a_disjoint_proof"]["disjoint_passed"] is True
    assert verify_res["part_b_wfs_splits"]["five_way_disjoint_proof"]["wave6_eval_sha256_matches_byte_identical"] is True

    # Assemble-manifest roundtrip with realistic Part A (2,104 ms > 1,500 ms latency gate not met, accuracy/JSON pass)
    # and Part B null GRPO-vs-SFT (`K4 stands under Protocol v2`) -> receipt MUST remain COMPLETED!
    asm_dir = tmp_path / "assembled"
    asm_dir.mkdir(parents=True, exist_ok=True)
    (asm_dir / "wave11_d9_part_a_suite_a_n60_rows.csv").write_text("item_id,pred\nc1,ok\n", encoding="utf-8")
    (asm_dir / "wave11_d9_part_b_eval_virgin_v2_n120_rows.csv").write_text("episode_id,exact_match\ne1,1.0\n", encoding="utf-8")
    (asm_dir / "selection_frozen.json").write_text('{"schema_version":"wave11_d9b_selection_frozen_v2"}\n', encoding="utf-8")
    (asm_dir / "part_a_fast_extractor_results.json").write_text(
        json.dumps(
            {
                "synthetic_corpus_audit": {"suite_a_disjoint_proof": {"disjoint_passed": True}},
                "suite_a_evaluation_n60": {
                    "semantic_accuracy": 0.9667,
                    "extraction_benchmark": {"ms_per_case_mean": 2104.0, "json_valid_rate": 1.0},
                    "speedup_vs_wave6_30b_extractor": 4.6,
                    "l4_gates": {
                        "ms_per_case_below_1500": False,
                        "json_valid_rate_ge_099": True,
                        "beats_mem0_lcb975_gt_0": True,
                        "reload_parity_below_1e4": True,
                    },
                },
                "csv_artifact": {"filename": "wave11_d9_part_a_suite_a_n60_rows.csv"},
            }
        ),
        encoding="utf-8",
    )
    (asm_dir / "part_b_u_grpo_results.json").write_text(
        json.dumps(
            {
                "protocol_v2_hypothesis_test": {
                    "verdict": "K4 stands under Protocol v2",
                    "eval_virgin_v2_scope": "virgin_for_grpo_arm_only",
                    "protocol_v2_single_knob_attribution_claimed": False,
                },
                "csv_artifacts": {"wave11_d9_part_b_eval_virgin_v2_n120_rows.csv": "dummy"},
            }
        ),
        encoding="utf-8",
    )
    manifest = assemble_wave11_d9_manifest(output_dir=asm_dir)
    assert manifest["schema_version"] == "wave11_d9_fast_extractor_and_u_grpo_l4_v1"
    assert manifest["part_a_gate_diagnostics"]["part_a_ms_per_case_below_1500"] is False
    assert manifest["part_a_gate_diagnostics"]["part_a_latency_status"] == "latency_gate_not_met"
    assert "2104.0 ms" in str(manifest["part_a_gate_diagnostics"]["part_a_latency_disclosure"])
    assert manifest["part_b_hypothesis_summary"]["verdict"] == "K4 stands under Protocol v2"

    receipt = json.loads((asm_dir / "report_receipt.json").read_text(encoding="utf-8"))
    assert receipt["status"] == "COMPLETED"
    assert receipt["diagnostics"]["part_a_ms_per_case_below_1500"] is False
    assert receipt["diagnostics"]["part_a_latency_status"] == "latency_gate_not_met"
    assert receipt["diagnostics"]["part_b_hypothesis_verdict"] == "K4 stands under Protocol v2"
    assert "selection_frozen.json" in receipt["artifact_hashes"]
    for fname, expected_sha in receipt["artifact_hashes"].items():
        actual_sha = hashlib.sha256((asm_dir / fname).read_bytes()).hexdigest()
        assert actual_sha == expected_sha
