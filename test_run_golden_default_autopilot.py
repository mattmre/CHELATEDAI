import json
import tempfile
import unittest
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import patch

from run_golden_default_autopilot import (
    DEFAULT_EVAL_SPECS,
    EvalSpec,
    _plan_reform_collection,
    _recommendation,
    _reform_collection_specs,
    _resolve_eval_specs,
    _rows_within_run_dir,
    run_reform_validation,
)
from run_thousand_query_tuning import LoopSpec


class TestGoldenDefaultAutopilot(unittest.TestCase):
    def test_reform_collection_specs_rotate_across_templates_for_live_phase_shape(self):
        first = _reform_collection_specs(1, phase_queries=150, loop_queries=50, base_seed=260)
        second = _reform_collection_specs(2, phase_queries=150, loop_queries=50, base_seed=260)
        third = _reform_collection_specs(3, phase_queries=150, loop_queries=50, base_seed=260)

        self.assertEqual(first, [
            LoopSpec("SciFact", seed=260, query_offset=0),
            LoopSpec("SciFact", seed=260, query_offset=100),
            LoopSpec("NFCorpus", seed=260, query_offset=0),
        ])
        self.assertEqual(second, [
            LoopSpec("NFCorpus", seed=260, query_offset=100),
            LoopSpec("FiQA2018", seed=260, query_offset=0),
            LoopSpec("FiQA2018", seed=260, query_offset=200),
        ])
        self.assertEqual(third, [
            LoopSpec("FiQA2018", seed=260, query_offset=400),
            LoopSpec("SciFact", seed=261, query_offset=0),
            LoopSpec("SciFact", seed=261, query_offset=100),
        ])

    def test_default_eval_specs_use_offset_holdouts_outside_collection_windows(self):
        self.assertEqual(DEFAULT_EVAL_SPECS, [
            EvalSpec("SciFact", seed=170, query_offset=200),
            EvalSpec("NFCorpus", seed=171, query_offset=200),
            EvalSpec("FiQA2018", seed=172, query_offset=600),
        ])

    def test_resolve_eval_specs_clamps_offset_to_latest_full_window(self):
        queries = {f"q{i}": f"query {i}" for i in range(10)}
        qrels = {f"q{i}": {f"d{i}": 1.0} for i in range(10)}
        with patch(
            "run_golden_default_autopilot.load_mteb_data",
            return_value=({f"d{i}": f"doc {i}" for i in range(10)}, queries, qrels),
        ):
            resolved = _resolve_eval_specs(
                [EvalSpec("FiQA2018", seed=172, query_offset=8)],
                max_queries=4,
            )

        self.assertEqual(resolved, [EvalSpec("FiQA2018", seed=172, query_offset=6)])

    def test_run_reform_validation_uses_offset_aware_query_window_selection(self):
        observed_offsets = []

        def fake_select_query_window(_corpus, _queries, _qrels, *, query_offset, max_queries, sample_docs, seed):
            observed_offsets.append((query_offset, max_queries, sample_docs, seed))
            return {"d1": "doc"}, {"q1": "query"}, {"q1": {"d1": 1.0}}

        def fake_profile_summary(_profile_results):
            rows = []
            for profile, delta in [
                ("baseline", 0.0),
                ("guard_p85_t0.01", 0.0),
                ("reform_rrf_v2", -0.01),
                ("learned_reform_gate_v1", 0.0),
                ("guard_reform_rrf_v2", -0.005),
                ("guard_learned_reform_gate_v1", 0.0),
            ]:
                rows.append({
                    "profile": profile,
                    "ndcg_at_10": 0.5 + delta,
                    "delta_vs_baseline": delta,
                    "fault_classification": {"promotion_blocker": False},
                })
            return {"ranked_profiles": rows}

        with (
            patch("run_golden_default_autopilot.load_mteb_data", return_value=({"d1": "doc"}, {"q1": "query"}, {"q1": {"d1": 1.0}})),
            patch("run_golden_default_autopilot.select_query_window", side_effect=fake_select_query_window),
            patch("run_golden_default_autopilot.evaluate_profile_with_cache", side_effect=lambda profile, *_args, **_kwargs: {"profile": profile.name}),
            patch("run_golden_default_autopilot.profile_summary", side_effect=fake_profile_summary),
            patch("run_golden_default_autopilot.isolated_adapter_state", return_value=nullcontext()),
        ):
            result = run_reform_validation(
                None,
                model="unused",
                specs=[
                    EvalSpec("SciFact", seed=170, query_offset=200),
                    EvalSpec("NFCorpus", seed=171, query_offset=210),
                ],
                max_queries=20,
                sample_docs=120,
            )

        self.assertEqual(observed_offsets, [
            (200, 20, 120, 170),
            (210, 20, 120, 171),
        ])
        self.assertEqual(result["engine_scope_schema_version"], 1)
        self.assertIn("engine_scope_rows", result)
        self.assertEqual(result["windows"][0]["query_offset"], 200)
        self.assertEqual(result["windows"][1]["query_offset"], 210)

    def test_recommendation_requires_candidate_not_mere_gate_presence(self):
        recommendation = _recommendation(
            {"type": "linear_classifier"},
            {
                "guard_learned_reform_mean_delta_vs_guard": 0.0,
                "guard_learned_reform_promotion_blockers": 0,
            },
            None,
            {},
        )

        self.assertFalse(recommendation["reform_gate_candidate_for_broader_validation"])
        self.assertEqual(
            recommendation["next_action"],
            "continue pooled data collection and fail-closed gate search",
        )

    def test_recommendation_promotes_only_real_survivors(self):
        recommendation = _recommendation(
            None,
            {},
            {"type": "linear_classifier"},
            {
                "learned_gate_mean_delta_vs_baseline": 0.01,
                "learned_gate_negative_windows": 0,
            },
        )

        self.assertTrue(recommendation["mask_gate_candidate_for_broader_validation"])
        self.assertEqual(
            recommendation["next_action"],
            "expand repeatability and transfer validation for the surviving gate candidate",
        )

    def test_recommendation_surfaces_adaptive_overlay_readiness(self):
        recommendation = _recommendation(
            None,
            {
                "adaptive_overlay": {
                    "readiness": {
                        "ready_for_broader_validation": True,
                        "blockers": [],
                    }
                }
            },
            None,
            {
                "adaptive_overlay": {
                    "readiness": {
                        "ready_for_broader_validation": False,
                        "blockers": ["safe_pass_rate_below_threshold"],
                    }
                }
            },
        )

        self.assertTrue(recommendation["adaptive_overlay_ready_for_broader_validation"])
        self.assertEqual(recommendation["adaptive_overlay_blockers"], ["safe_pass_rate_below_threshold"])

    def test_recommendation_blocks_reform_candidate_after_bad_hard_negative_replay(self):
        recommendation = _recommendation(
            {"type": "linear_classifier"},
            {
                "guard_learned_reform_mean_delta_vs_guard": 0.01,
                "guard_learned_reform_promotion_blockers": 0,
            },
            None,
            {},
            reform_hard_negative_validation={
                "family_count": 2,
                "guard_learned_reform_mean_delta_vs_guard": -0.01,
                "guard_learned_negative_families": 1,
                "guard_learned_reform_promotion_blockers": 0,
            },
        )

        self.assertFalse(recommendation["reform_gate_candidate_for_broader_validation"])
        self.assertFalse(recommendation["reform_gate_survived_hard_negative_replay"])

    def test_plan_reform_collection_uses_campaign_coverage_for_selection(self):
        coverage_rows = [
            {
                "source_family": "reformulation_collection",
                "task": "SciFact",
                "query_offset": 0,
                "seed": 260,
                "row_type": "query_profile",
                "profile": "reform_rrf_v2",
                "delta_ndcg_at_10": 0.0,
                "query_token_count": 4,
                "query_char_count": 25,
                "query_stopword_ratio": 0.25,
                "query_numeric_token_count": 0,
                "query_negation_count": 0,
                "query_claim_cue_count": 0,
                "action": "FAST",
                "fault_class": "no_op_tied",
                "top10_overlap_with_baseline": 10,
                "top_doc_changed": False,
                "global_variance": 0.0,
                "jaccard": 1.0,
                "mask_density": 1.0,
                "reformulation_variant_count": 0,
                "reformulation_changed": False,
            }
        ]
        with patch("run_golden_default_autopilot._campaign_reform_collection_rows", return_value=coverage_rows):
            plan = _plan_reform_collection(
                Path(r"C:\GitHub\repos\CHELATEDAI\experiment_runs\campaign"),
                iteration=2,
                phase_queries=150,
                loop_queries=50,
                base_seed=260,
                selection_mode="coverage_aware",
            )

        self.assertEqual(plan["selection_mode"], "coverage_aware")
        self.assertEqual(len(plan["loop_specs"]), 3)
        self.assertNotIn(
            LoopSpec("SciFact", seed=261, query_offset=0),
            plan["loop_specs"],
        )
        self.assertEqual(plan["coverage_row_count"], 1)

    def test_rows_within_run_dir_filters_bootstrap_artifacts_from_coverage(self):
        rows = [
            {"_artifact_path": r"C:\GitHub\repos\CHELATEDAI\experiment_runs\campaign\reports\iteration_001.json", "query_id": "q1"},
            {"_artifact_path": r"experiment_runs\roadcourse-small\attribution_probe_100.json", "query_id": "q2"},
        ]

        scoped = _rows_within_run_dir(rows, Path(r"C:\GitHub\repos\CHELATEDAI\experiment_runs\campaign"))

        self.assertEqual(len(scoped), 1)
        self.assertEqual(scoped[0]["query_id"], "q1")

    # --- ENG-5 gap 1: default_change_allowed must reflect candidate gate state ---

    def test_recommendation_default_change_allowed_false_when_no_candidate(self):
        """default_change_allowed must be False when neither gate is a candidate.

        Guards against regression to the old hardcoded-False L2 escape: if both
        reform_candidate and mask_candidate are False the field should be False,
        but for the right reason — not because it was hardcoded.
        """
        rec = _recommendation(
            None,
            {},
            None,
            {},
        )
        self.assertFalse(rec["default_change_allowed"])
        self.assertTrue(rec["safe_default_holds"])

    def test_recommendation_default_change_allowed_true_when_reform_candidate(self):
        """default_change_allowed must be True when a reform gate is a real candidate.

        This directly guards against the ENG-5 gap 1 L2 escape: hardcoding
        'default_change_allowed': False regardless of gate outcome prevented the
        promotion path from ever firing.
        """
        rec = _recommendation(
            {"type": "linear_classifier"},
            {
                "guard_learned_reform_mean_delta_vs_guard": 0.005,
                "guard_learned_reform_promotion_blockers": 0,
            },
            None,
            {},
        )
        self.assertTrue(rec["reform_gate_candidate_for_broader_validation"])
        self.assertTrue(rec["default_change_allowed"])
        self.assertFalse(rec["safe_default_holds"])

    def test_recommendation_default_change_allowed_true_when_mask_candidate(self):
        """default_change_allowed must be True when a mask gate is a real candidate."""
        rec = _recommendation(
            None,
            {},
            {"type": "linear_classifier"},
            {
                "learned_gate_mean_delta_vs_baseline": 0.01,
                "learned_gate_negative_windows": 0,
            },
        )
        self.assertTrue(rec["mask_gate_candidate_for_broader_validation"])
        self.assertTrue(rec["default_change_allowed"])
        self.assertFalse(rec["safe_default_holds"])

    # --- ENG-5 gap 2: terminal_decision.json must be written at end of main() ---

    def test_main_writes_terminal_decision_json(self):
        """main() must write terminal_decision.json before returning.

        Guards against ENG-5 gap 2: acceptance criterion #3 (terminal decision
        artifact) was unimplemented.  We run main() with max-iterations=0 so
        it exits immediately after zero iterations, then verify the artifact
        exists and contains the required fields.
        """
        import argparse
        from run_golden_default_autopilot import main, _write_json_atomic, _now_iso

        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir) / "test_run"
            run_dir.mkdir()

            # Patch argument parsing so main() uses our controlled run directory
            # and exits after 0 iterations (max_iterations=0 → deadline already reached
            # semantics don't apply cleanly; use max_iterations=1 but stub _run_iteration).
            fake_args = [
                "--run-dir", str(run_dir),
                "--max-iterations", "1",
                "--deadline-hours", "48",
            ]

            # Stub _run_iteration to avoid any real computation
            def fake_run_iteration(run_dir, manifest, *, iteration, args):
                # Write a minimal iteration report so downstream code doesn't fail
                report = {
                    "iteration": iteration,
                    "started_at": _now_iso(),
                    "finished_at": _now_iso(),
                    "engine_scope_schema_version": 1,
                    "golden_safe_default": {},
                    "recommendation": {
                        "default_change_allowed": False,
                        "safe_default_holds": True,
                        "reform_gate_candidate_for_broader_validation": False,
                        "reform_gate_survived_hard_negative_replay": True,
                        "reform_hard_negative_family_count": 0,
                        "mask_gate_candidate_for_broader_validation": False,
                        "adaptive_overlay_ready_for_broader_validation": False,
                        "adaptive_overlay_blockers": [],
                        "coverage_guided_collection_recommended": False,
                        "next_action": "continue pooled data collection and fail-closed gate search",
                    },
                }
                report_path = run_dir / "reports" / f"iteration_{iteration:03d}.json"
                _write_json_atomic(report_path, report)
                manifest.setdefault("iterations", {})[f"iter_{iteration:03d}"] = {
                    "iteration": iteration,
                    "status": "completed",
                    "finished_at": _now_iso(),
                    "report_path": str(report_path),
                    "recommendation": report["recommendation"],
                }
                manifest["latest_report"] = str(report_path)
                manifest["latest_recommendation"] = report["recommendation"]
                return report

            with (
                patch("sys.argv", ["run_golden_default_autopilot.py", *fake_args]),
                patch("run_golden_default_autopilot._run_iteration", side_effect=fake_run_iteration),
                patch("run_golden_default_autopilot.time.sleep"),
            ):
                exit_code = main()

            self.assertEqual(exit_code, 0)
            terminal_path = run_dir / "terminal_decision.json"
            self.assertTrue(
                terminal_path.exists(),
                f"terminal_decision.json was not written to {terminal_path}",
            )
            payload = json.loads(terminal_path.read_text(encoding="utf-8"))
            # Required fields from acceptance criterion #3
            self.assertIn("generated_at", payload)
            self.assertIn("run_dir", payload)
            self.assertIn("status", payload)
            self.assertIn("termination_reason", payload)
            self.assertIn("iteration_count", payload)
            self.assertIn("latest_recommendation", payload)

    # --- ENG-5 gap 3: contract module absence must be intentional & documented ---

    def test_supervisor_module_docstring_explains_absent_contract_imports(self):
        """The supervisor module docstring must explain why contract modules are absent.

        Guards against ENG-5 gap 3: evidence_contract, promotion_contract,
        compute_budget_policy, and evaluator_fabric exist in the repo but are
        not imported by this supervisor.  The absence must be documented so it
        cannot be mistaken for an oversight.
        """
        import run_golden_default_autopilot
        doc = run_golden_default_autopilot.__doc__ or ""
        keywords = ["evidence_contract", "promotion_contract", "intentionally"]
        for keyword in keywords:
            self.assertIn(
                keyword,
                doc,
                f"Supervisor module docstring must mention '{keyword}' to document "
                "why contract modules are intentionally absent.",
            )

    def test_contract_modules_not_imported_by_supervisor(self):
        """The supervisor must NOT import contract modules at module level.

        The contract modules belong to the Model-Scope layer; importing them
        here would create a false dependency.
        """
        import run_golden_default_autopilot
        for module_name in (
            "evidence_contract",
            "promotion_contract",
            "compute_budget_policy",
            "evaluator_fabric",
        ):
            self.assertNotIn(
                module_name,
                dir(run_golden_default_autopilot),
                f"'{module_name}' must not be imported at module level in the supervisor.",
            )


if __name__ == "__main__":
    unittest.main()
