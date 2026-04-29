import unittest

from run_road_course_campaign import RoadCourseProfile
from run_road_course_tuning_loop import (
    CALIBRATED_PROFILE_GRID,
    MODULE_PROFILE_GRID,
    QUICK_PROFILE_GRID,
    classify_profile_behavior,
    profile_summary,
    propose_next_profiles,
)
from run_thousand_query_tuning import (
    LoopSpec,
    _profile_cache_key,
    build_gate_feature_rows,
    build_phase_loop_specs,
    build_query_attribution_rows,
    choose_window_profiles,
    evaluate_profile_with_cache,
    parse_loop_specs,
    query_metric_row,
    run_thousand_query_cycle,
    select_query_window,
    split_query_windows,
    summarize_gate_candidates,
    summarize_profile_outcomes,
)
from train_gate_from_tuning_artifact import train_gate_rules


class TestRoadCourseTuningLoop(unittest.TestCase):
    def test_profile_summary_detects_active_chelation_regression(self):
        summary = profile_summary([
            {
                "profile": "baseline",
                "metrics": {"ndcg_at_10": 0.8, "mrr": 0.7, "recall_at_10": 0.9},
                "action_mix": {"FAST": 100},
                "control_diagnostics": {"variance_mean": 0.002},
                "latency_ms_mean": 10.0,
            },
            {
                "profile": "adaptive_p85_t0.0004",
                "metrics": {"ndcg_at_10": 0.7, "mrr": 0.6, "recall_at_10": 0.8},
                "action_mix": {"CHELATE": 100},
                "control_diagnostics": {"variance_mean": 0.002, "mask_density_mean": 0.85},
                "latency_ms_mean": 12.0,
            },
        ])

        self.assertEqual(summary["best_profile"], "baseline")
        self.assertEqual(summary["active_chelate_regression_count"], 1)
        self.assertEqual(summary["fault_counts"]["actuator_active_negative"], 1)
        self.assertEqual(
            summary["ranked_profiles"][1]["fault_classification"]["fault_class"],
            "actuator_active_negative",
        )
        self.assertEqual(summary["ranked_profiles"][1]["control_diagnostics"]["mask_density_mean"], 0.85)

    def test_profile_behavior_classifies_noop_positive_negative_and_suspicious_paths(self):
        baseline = classify_profile_behavior({
            "profile": "baseline",
            "delta_vs_baseline": 0.0,
            "action_mix": {"FAST": 50},
            "control_diagnostics": {"jaccard_mean": 1.0, "mask_density_mean": 1.0},
        })
        no_op = classify_profile_behavior({
            "profile": "guard",
            "delta_vs_baseline": 0.0,
            "action_mix": {"FAST": 50},
            "control_diagnostics": {"jaccard_mean": 1.0, "mask_density_mean": 1.0},
        })
        positive = classify_profile_behavior({
            "profile": "adaptive",
            "delta_vs_baseline": 0.01,
            "action_mix": {"CHELATE": 10, "FAST": 40},
            "control_diagnostics": {"jaccard_mean": 0.8, "mask_density_mean": 0.95},
        })
        negative = classify_profile_behavior({
            "profile": "reform",
            "delta_vs_baseline": -0.01,
            "action_mix": {"REFORMULATE": 50},
            "control_diagnostics": {"reformulation_changed_count": 20},
        })
        suspicious = classify_profile_behavior({
            "profile": "hidden_change",
            "delta_vs_baseline": -0.01,
            "action_mix": {"FAST": 50},
            "control_diagnostics": {"jaccard_mean": 1.0, "mask_density_mean": 1.0},
        })

        self.assertEqual(baseline["fault_class"], "reference")
        self.assertEqual(no_op["fault_class"], "no_op_tied")
        self.assertEqual(positive["fault_class"], "actuator_active_positive")
        self.assertEqual(negative["fault_class"], "actuator_active_negative")
        self.assertEqual(suspicious["fault_class"], "metric_changed_without_actuator")
        self.assertTrue(suspicious["promotion_blocker"])

    def test_propose_next_profiles_moves_to_safer_guardrails_after_regression(self):
        profiles = propose_next_profiles({
            "active_chelate_regression_count": 1,
            "best_profile": "baseline",
        })
        names = {profile.name for profile in profiles}

        self.assertIn("baseline", names)
        self.assertIn("guard_p85_t0.01", names)
        self.assertIn("guard_p85_t0.01_reform_v2", names)
        self.assertIn("fast_guard_p85_t999", names)
        self.assertTrue(all(isinstance(profile, RoadCourseProfile) for profile in profiles))

    def test_quick_grid_covers_baseline_aggressive_guard_and_centered_profiles(self):
        names = {profile.name for profile in QUICK_PROFILE_GRID}

        self.assertIn("baseline", names)
        self.assertIn("adaptive_p85_t0.0004", names)
        self.assertIn("adaptive_p85_t0.01", names)
        self.assertIn("fast_guard_p85_t999", names)
        self.assertIn("centered_p85_temp1", names)

    def test_module_grid_covers_reformulation_guard_and_temperature_profiles(self):
        names = {profile.name for profile in MODULE_PROFILE_GRID}

        self.assertIn("reform_v2", names)
        self.assertIn("reform_v3", names)
        self.assertIn("guard_p85_t0.01_reform_v2", names)
        self.assertIn("fast_guard_p85_t999_reform_v2", names)
        self.assertIn("centered_p85_temp0.5", names)
        self.assertIn("centered_p85_temp2", names)

    def test_calibrated_grid_covers_threshold_transition_and_control_noop_detection(self):
        names = {profile.name for profile in CALIBRATED_PROFILE_GRID}

        self.assertIn("adaptive_p85_t0.0015", names)
        self.assertIn("adaptive_p85_t0.002", names)
        self.assertIn("adaptive_p50_t0.002", names)
        self.assertIn("adaptive_p95_t0.002", names)
        self.assertIn("adaptive_p85_t0.002_reform_rrf_v2", names)
        self.assertIn("reform_rrf_v3", names)

    def test_thousand_runner_parses_loop_specs(self):
        specs = parse_loop_specs("SciFact:46:0,NFCorpus:47:200")

        self.assertEqual(specs, [
            LoopSpec("SciFact", seed=46, query_offset=0),
            LoopSpec("NFCorpus", seed=47, query_offset=200),
        ])

    def test_thousand_runner_builds_phase_specs_by_cycling_available_windows(self):
        specs = build_phase_loop_specs(phase_queries=1600, loop_queries=200, base_seed=50)

        self.assertEqual(len(specs), 8)
        self.assertEqual(specs[0], LoopSpec("SciFact", seed=50, query_offset=0))
        self.assertEqual(specs[6], LoopSpec("FiQA2018", seed=50, query_offset=400))
        self.assertEqual(specs[7], LoopSpec("SciFact", seed=51, query_offset=0))

    def test_thousand_runner_selects_offset_query_window_and_preserves_relevant_docs(self):
        corpus = {f"d{i}": f"doc {i}" for i in range(6)}
        queries = {f"q{i}": f"query {i}" for i in range(5)}
        qrels = {f"q{i}": {f"d{i}": 1.0} for i in range(5)}

        selected_corpus, selected_queries, selected_qrels = select_query_window(
            corpus,
            queries,
            qrels,
            query_offset=2,
            max_queries=2,
            sample_docs=3,
            seed=42,
        )

        self.assertEqual(list(selected_queries), ["q2", "q3"])
        self.assertEqual(set(selected_qrels), {"q2", "q3"})
        self.assertIn("d2", selected_corpus)
        self.assertIn("d3", selected_corpus)

    def test_thousand_runner_reuses_cached_profile_engine(self):
        class FakeEngine:
            def __init__(self):
                self.calls = 0

            def run_inference(self, _query):
                self.calls += 1
                return [1], [1], [1.0], 1.0

            def get_last_runtime_diagnostics(self):
                return {"runtime": {"action": "FAST", "global_variance": 0.001}}

            def get_runtime_telemetry(self):
                return {"total_inferences": self.calls}

        engine = FakeEngine()
        cache = {_profile_cache_key(RoadCourseProfile("baseline")): engine}

        first = evaluate_profile_with_cache(
            RoadCourseProfile("baseline"),
            "unused",
            {"d1": "doc"},
            {"q1": "query"},
            {"q1": {"d1": 1.0}},
            cache,
        )
        second = evaluate_profile_with_cache(
            RoadCourseProfile("baseline"),
            "unused",
            {"d1": "doc"},
            {"q2": "query"},
            {"q2": {"d1": 1.0}},
            cache,
        )

        self.assertEqual(engine.calls, 2)
        self.assertEqual(first["action_mix"], {"FAST": 1})
        self.assertEqual(second["telemetry"]["total_inferences"], 2)
        self.assertIn("query_metrics", first)
        self.assertEqual(first["query_metrics"]["q1"]["action"], "FAST")

    def test_query_metric_row_scores_single_query(self):
        row = query_metric_row("q1", ["d2", "d1", "d3"], {"d1": 1.0})

        self.assertEqual(row["query_id"], "q1")
        self.assertEqual(row["first_relevant_rank"], 2)
        self.assertAlmostEqual(row["mrr"], 0.5)
        self.assertEqual(row["top_doc_id"], "d2")

    def test_thousand_runner_splits_query_windows(self):
        queries = {f"q{i}": f"query {i}" for i in range(4)}
        qrels = {f"q{i}": {f"d{i}": 1.0} for i in range(4)}

        windows = split_query_windows(queries, qrels, window_queries=2)

        self.assertEqual(len(windows), 2)
        self.assertEqual(list(windows[0][0]), ["q0", "q1"])
        self.assertEqual(set(windows[1][1]), {"q2", "q3"})

    def test_thousand_runner_adapts_profiles_after_active_regression(self):
        profiles = choose_window_profiles([
            {
                "ranked_profiles": [
                    {"profile": "baseline", "delta_vs_baseline": 0.0, "action_mix": {"FAST": 50}},
                    {"profile": "adaptive_p85_t0.002", "delta_vs_baseline": -0.02, "action_mix": {"CHELATE": 50}},
                    {"profile": "reform_rrf_v2", "delta_vs_baseline": -0.01, "action_mix": {"REFORMULATE": 50}},
                ]
            }
        ], window_index=2)
        names = {profile.name for profile in profiles}

        self.assertIn("baseline", names)
        self.assertIn("guard_p85_t0.01", names)
        self.assertIn("adaptive_p85_t0.003", names)

    def test_thousand_runner_prefers_same_task_history_for_profile_adaptation(self):
        profiles = choose_window_profiles(
            [
                {
                    "ranked_profiles": [
                        {"profile": "adaptive_p85_t0.002", "delta_vs_baseline": -0.02, "action_mix": {"CHELATE": 50}},
                    ]
                }
            ],
            window_index=2,
            task="FiQA2018",
            previous_windows=[
                {
                    "task": "FiQA2018",
                    "summary": {
                        "ranked_profiles": [
                            {
                                "profile": "adaptive_p85_t0.002",
                                "delta_vs_baseline": 0.01,
                                "action_mix": {"CHELATE": 5, "FAST": 45},
                            },
                            {
                                "profile": "reform_rrf_v2",
                                "delta_vs_baseline": 0.002,
                                "action_mix": {"REFORMULATE": 50},
                            },
                        ]
                    },
                }
            ],
        )
        names = {profile.name for profile in profiles}

        self.assertIn("adaptive_p85_t0.0025", names)
        self.assertIn("adaptive_p85_t0.0025_reform_rrf_v2", names)

    def test_thousand_runner_confirm_fiqa_strategy_retests_promising_branch(self):
        fiqa_profiles = choose_window_profiles([], window_index=1, task="FiQA2018", strategy="confirm_fiqa")
        scifact_profiles = choose_window_profiles([], window_index=2, task="SciFact", strategy="confirm_fiqa")
        nfcorpus_profiles = choose_window_profiles([], window_index=3, task="NFCorpus", strategy="confirm_fiqa")

        self.assertEqual(
            {profile.name for profile in fiqa_profiles},
            {
                "baseline",
                "guard_p85_t0.01",
                "adaptive_p85_t0.002",
                "adaptive_p85_t0.002_reform_rrf_v2",
            },
        )
        self.assertIn("adaptive_p95_t0.002", {profile.name for profile in scifact_profiles})
        self.assertIn("reform_rrf_v2", {profile.name for profile in nfcorpus_profiles})

    def test_thousand_runner_fault_aware_strategy_rotates_safer_probes(self):
        first = choose_window_profiles([], window_index=1, task="FiQA2018", strategy="fault_aware")
        third = choose_window_profiles([], window_index=3, task="FiQA2018", strategy="fault_aware")

        self.assertIn("adaptive_p99_t0.002", {profile.name for profile in first})
        self.assertIn("reform_rrf_v2", {profile.name for profile in third})

    def test_thousand_runner_fault_aware_strategy_prefers_positive_same_task_history(self):
        profiles = choose_window_profiles(
            [],
            window_index=2,
            task="SciFact",
            previous_windows=[
                {
                    "task": "SciFact",
                    "summary": {
                        "ranked_profiles": [
                            {
                                "profile": "adaptive_p95_t0.002",
                                "delta_vs_baseline": 0.01,
                                "action_mix": {"CHELATE": 5, "FAST": 45},
                                "control_diagnostics": {"jaccard_mean": 0.9},
                                "fault_classification": {"fault_class": "actuator_active_positive"},
                            },
                            {
                                "profile": "adaptive_p99_t0.002",
                                "delta_vs_baseline": -0.01,
                                "action_mix": {"CHELATE": 5, "FAST": 45},
                                "control_diagnostics": {"jaccard_mean": 0.9},
                                "fault_classification": {"fault_class": "actuator_active_negative"},
                            },
                        ]
                    },
                }
            ],
            strategy="fault_aware",
        )

        self.assertIn("adaptive_p95_t0.002", {profile.name for profile in profiles})

    def test_thousand_runner_gate_learning_strategy_collects_core_candidates(self):
        first = choose_window_profiles([], window_index=1, task="SciFact", strategy="gate_learning")
        sixth = choose_window_profiles([], window_index=6, task="SciFact", strategy="gate_learning")

        self.assertIn("baseline", {profile.name for profile in first})
        self.assertIn("guard_p85_t0.01", {profile.name for profile in first})
        self.assertIn("adaptive_p99_t0.0015", {profile.name for profile in first})
        self.assertIn("adaptive_p85_t0.002", {profile.name for profile in sixth})
        self.assertIn("reform_rrf_v2", {profile.name for profile in sixth})

    def test_thousand_runner_gate_feature_rows_and_candidate_report(self):
        windows = [
            {
                "loop": 1,
                "window": 1,
                "global_window": 1,
                "task": "SciFact",
                "queries": {"q1": "LDL cholesterol has no involvement in 10% risk"},
                "summary": {
                    "ranked_profiles": [
                        {
                            "profile": "baseline",
                            "ndcg_at_10": 0.8,
                            "delta_vs_baseline": 0.0,
                            "action_mix": {"FAST": 50},
                            "control_diagnostics": {"jaccard_mean": 1.0, "mask_density_mean": 1.0},
                            "fault_classification": {"fault_class": "reference", "promotion_blocker": False},
                        },
                        {
                            "profile": "adaptive_p99_t0.0015",
                            "ndcg_at_10": 0.81,
                            "delta_vs_baseline": 0.01,
                            "action_mix": {"CHELATE": 50},
                            "control_diagnostics": {
                                "variance_mean": 0.001,
                                "jaccard_mean": 0.95,
                                "mask_density_mean": 0.99,
                            },
                            "fault_classification": {
                                "fault_class": "actuator_active_positive",
                                "promotion_blocker": False,
                            },
                        },
                    ]
                },
            },
            {
                "loop": 1,
                "window": 2,
                "global_window": 2,
                "task": "SciFact",
                "summary": {
                    "ranked_profiles": [
                        {
                            "profile": "adaptive_p99_t0.0015",
                            "ndcg_at_10": 0.812,
                            "delta_vs_baseline": 0.012,
                            "action_mix": {"CHELATE": 50},
                            "control_diagnostics": {
                                "variance_mean": 0.0011,
                                "jaccard_mean": 0.96,
                                "mask_density_mean": 0.99,
                            },
                            "fault_classification": {
                                "fault_class": "actuator_active_positive",
                                "promotion_blocker": False,
                            },
                        },
                    ]
                },
            },
            {
                "loop": 1,
                "window": 3,
                "global_window": 3,
                "task": "SciFact",
                "summary": {
                    "ranked_profiles": [
                        {
                            "profile": "adaptive_p99_t0.0015",
                            "ndcg_at_10": 0.811,
                            "delta_vs_baseline": 0.011,
                            "action_mix": {"CHELATE": 50},
                            "control_diagnostics": {
                                "variance_mean": 0.0012,
                                "jaccard_mean": 0.97,
                                "mask_density_mean": 0.99,
                            },
                            "fault_classification": {
                                "fault_class": "actuator_active_positive",
                                "promotion_blocker": False,
                            },
                        },
                    ]
                },
            },
        ]

        rows = build_gate_feature_rows(windows)
        report = summarize_gate_candidates(rows)

        self.assertEqual(rows[1]["profile"], "adaptive_p99_t0.0015")
        self.assertEqual(rows[1]["task"], "SciFact")
        self.assertEqual(report["adaptive_p99_t0.0015"]["best_gate"]["gate"], "all")
        self.assertTrue(report["adaptive_p99_t0.0015"]["best_gate"]["shippable_gate_candidate"])

    def test_thousand_runner_builds_query_attribution_rows(self):
        windows = [
            {
                "loop": 1,
                "window": 1,
                "global_window": 1,
                "task": "SciFact",
                "queries": {"q1": "LDL cholesterol has no involvement in 10% risk"},
                "summary": {
                    "ranked_profiles": [
                        {
                            "profile": "baseline",
                            "delta_vs_baseline": 0.0,
                            "action_mix": {"FAST": 1},
                            "control_diagnostics": {},
                            "fault_classification": {"fault_class": "reference"},
                        },
                        {
                            "profile": "adaptive_p99_t0.0015",
                            "delta_vs_baseline": 0.1,
                            "action_mix": {"CHELATE": 1},
                            "control_diagnostics": {},
                            "fault_classification": {"fault_class": "actuator_active_positive"},
                        },
                    ]
                },
                "profile_results": [
                    {
                        "profile": "baseline",
                        "rankings": {"q1": ["d2", "d1"]},
                        "query_metrics": {
                            "q1": {
                                "ndcg_at_10": 0.5,
                                "mrr": 0.5,
                                "recall_at_10": 1.0,
                                "first_relevant_rank": 2,
                                "top_doc_id": "d2",
                                "action": "FAST",
                            }
                        },
                    },
                    {
                        "profile": "adaptive_p99_t0.0015",
                        "rankings": {"q1": ["d1", "d2"]},
                        "query_metrics": {
                            "q1": {
                                "ndcg_at_10": 1.0,
                                "mrr": 1.0,
                                "recall_at_10": 1.0,
                                "first_relevant_rank": 1,
                                "top_doc_id": "d1",
                                "action": "CHELATE",
                                "mask_density": 0.99,
                            }
                        },
                    },
                ],
            }
        ]

        rows = build_query_attribution_rows(windows)
        candidate = [row for row in rows if row["profile"] == "adaptive_p99_t0.0015"][0]

        self.assertEqual(candidate["delta_ndcg_at_10"], 0.5)
        self.assertEqual(candidate["rank_delta"], -1)
        self.assertTrue(candidate["top_doc_changed"])
        self.assertEqual(candidate["top10_overlap_with_baseline"], 2)
        self.assertEqual(candidate["fault_class"], "actuator_active_positive")
        self.assertEqual(candidate["query_numeric_token_count"], 1)
        self.assertEqual(candidate["query_negation_count"], 1)

    def test_gate_trainer_accepts_holdout_safe_rules(self):
        rows = []
        for window in range(1, 11):
            rows.append({
                "global_window": window,
                "task": "SciFact",
                "profile": "adaptive_p99_t0.0015",
                "delta_vs_baseline": 0.01,
                "fault_class": "actuator_active_positive",
                "jaccard_mean": 0.95,
                "mask_density_mean": 0.99,
                "variance_mean": 0.001,
            })
        config = train_gate_rules(rows, min_train_windows=3, min_holdout_windows=2)

        self.assertGreaterEqual(len(config["rules"]), 1)
        self.assertEqual(config["rules"][0]["profile"], "adaptive_p99_t0.0015")

    def test_gate_trainer_rejects_holdout_regressions(self):
        rows = []
        for window in range(1, 11):
            rows.append({
                "global_window": window,
                "task": "SciFact",
                "profile": "adaptive_p99_t0.0015",
                "delta_vs_baseline": -0.01 if window % 5 == 0 else 0.01,
                "fault_class": "actuator_active_negative" if window % 5 == 0 else "actuator_active_positive",
                "jaccard_mean": 0.95,
                "mask_density_mean": 0.99,
                "variance_mean": 0.001,
            })
        config = train_gate_rules(rows, min_train_windows=3, min_holdout_windows=2)

        self.assertEqual(config["rules"], [])

    def test_thousand_runner_learned_gate_uses_allowed_profiles(self):
        profiles = choose_window_profiles(
            [],
            window_index=1,
            task="SciFact",
            strategy="learned_gate",
            gate_config={
                "rules": [
                    {"profile": "adaptive_p99_t0.0015", "task": "SciFact", "feature": None},
                    {"profile": "adaptive_p85_t0.002", "task": "FiQA2018", "feature": None},
                ]
            },
        )
        names = {profile.name for profile in profiles}

        self.assertIn("baseline", names)
        self.assertIn("guard_p85_t0.01", names)
        self.assertIn("adaptive_p99_t0.0015", names)
        self.assertNotIn("adaptive_p85_t0.002", names)

    def test_thousand_runner_selective_reform_strategy_compares_reformulation_modes(self):
        profiles = choose_window_profiles([], window_index=1, task="SciFact", strategy="selective_reform")
        names = {profile.name for profile in profiles}

        self.assertEqual(names, {"baseline", "guard_p85_t0.01", "reform_rrf_v2", "selective_reform_rrf_v2"})

    def test_thousand_runner_reform_policy_search_compares_multiple_policies(self):
        profiles = choose_window_profiles([], window_index=1, task="SciFact", strategy="reform_policy_search")
        names = {profile.name for profile in profiles}

        self.assertEqual(names, {
            "baseline",
            "guard_p85_t0.01",
            "reform_rrf_v2",
            "selective_reform_rrf_v2",
            "reform_high_specificity_rrf_v2",
            "reform_claim_cue_rrf_v2",
        })

    def test_thousand_runner_rejects_unknown_strategy(self):
        with self.assertRaises(ValueError):
            run_thousand_query_cycle(
                model="unused",
                loop_specs=[],
                loop_queries=50,
                window_queries=50,
                sample_docs=10,
                strategy="unknown",
            )

    def test_thousand_runner_summarizes_profile_directionality(self):
        summary = summarize_profile_outcomes([
            {
                "task": "SciFact",
                "summary": {
                    "ranked_profiles": [
                        {
                            "profile": "baseline",
                            "delta_vs_baseline": 0.0,
                            "action_mix": {"FAST": 50},
                            "control_diagnostics": {"mask_density_mean": 1.0},
                        },
                        {
                            "profile": "adaptive_p85_t0.002",
                            "delta_vs_baseline": -0.01,
                            "action_mix": {"CHELATE": 10, "FAST": 40},
                            "control_diagnostics": {"mask_density_mean": 0.98},
                        },
                    ]
                },
            },
            {
                "task": "NFCorpus",
                "summary": {
                    "ranked_profiles": [
                        {
                            "profile": "adaptive_p85_t0.002",
                            "delta_vs_baseline": 0.002,
                            "action_mix": {"CHELATE": 5, "FAST": 45},
                            "control_diagnostics": {"mask_density_mean": 0.99},
                        },
                    ]
                },
            },
        ])

        adaptive = summary["adaptive_p85_t0.002"]
        self.assertEqual(adaptive["windows"], 2)
        self.assertEqual(adaptive["improved"], 1)
        self.assertEqual(adaptive["regressed"], 1)
        self.assertEqual(adaptive["action_mix"]["CHELATE"], 15)
        self.assertEqual(adaptive["fault_counts"]["actuator_active_negative"], 1)
        self.assertEqual(adaptive["fault_counts"]["actuator_active_positive"], 1)


if __name__ == "__main__":
    unittest.main()
