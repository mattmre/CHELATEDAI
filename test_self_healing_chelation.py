import json
import unittest
import os
from unittest.mock import MagicMock

from fitness_interfaces import FitnessEvaluation
from self_healing_chelation import (
    CandidateProvenanceLedger,
    SelfEditDirective,
    SelfHealingChelationConfig,
    SelfHealingChelationPlanner,
    build_self_healing_update_plan,
)


class TestSelfHealingChelationPlanner(unittest.TestCase):
    def test_generates_seal_and_eggroll_directives_from_diagnostics(self):
        planner = SelfHealingChelationPlanner(logger=MagicMock())
        directives = planner.generate_directives(
            ["new paper fact", "retrieval collapse observation"],
            {
                "runtime": {"status": "empty_results"},
                "quantization_gate": {"passed": False},
                "structural_health": {"score": 0.52},
            },
        )

        directive_ids = {directive.directive_id for directive in directives}
        self.assertIn("seal_implication_sft", directive_ids)
        self.assertIn("seal_retrieval_ttt", directive_ids)
        self.assertIn("eggroll_low_rank_self_edit", directive_ids)
        self.assertIn("quantization_survival_self_edit", directive_ids)
        self.assertTrue(all(directive.optimization_params for directive in directives))
        self.assertTrue(all(directive.synthetic_examples for directive in directives))

    def test_restem_filter_accepts_only_positive_retained_quantized_edits(self):
        planner = SelfHealingChelationPlanner(
            SelfHealingChelationConfig(baseline_fitness=0.50, min_retention_score=0.8),
            logger=MagicMock(),
        )
        good = SelfEditDirective("good", "implication_synthesis", "adapter_sft")
        forgetful = SelfEditDirective("forgetful", "implication_synthesis", "adapter_sft")
        quantized_loss = SelfEditDirective("quantized_loss", "low_rank_population_search", "eggroll_es")
        negative = SelfEditDirective("negative", "implication_synthesis", "adapter_sft")

        def fitness(directive):
            if directive.directive_id == "good":
                return 0.62
            if directive.directive_id == "forgetful":
                return FitnessEvaluation(directive.directive_id, 0.64, metrics={"retention_score": 0.70})
            if directive.directive_id == "quantized_loss":
                return FitnessEvaluation(
                    directive.directive_id,
                    0.63,
                    metrics={"fp32_fitness": 0.63, "quantized_fitness": 0.51},
                )
            return 0.49

        results = planner.evaluate_directives([good, forgetful, quantized_loss, negative], fitness)
        accepted = [result.directive.directive_id for result in results if result.accepted]
        reasons_by_id = {result.directive.directive_id: result.reasons for result in results}

        self.assertEqual(accepted, ["good"])
        self.assertIn("retention_below_threshold", reasons_by_id["forgetful"])
        self.assertIn("quantization_gate_failed", reasons_by_id["quantized_loss"])
        self.assertIn("reward_not_positive", reasons_by_id["negative"])

    def test_build_update_plan_is_json_safe_and_advisory_by_default(self):
        plan = build_self_healing_update_plan(
            context="Chelation should preserve factual retrieval under drift.",
            diagnostics={"retrieval_policy": {"high_variance_fast_path": True}},
            fitness=lambda directive: 1.0 if directive.directive_id == "seal_retrieval_ttt" else 0.0,
            config=SelfHealingChelationConfig(baseline_fitness=0.1, max_directives=3),
            logger=MagicMock(),
        )

        self.assertEqual(plan["mode"], "advisory")
        self.assertFalse(plan["safety"]["base_model_mutation_allowed"])
        self.assertEqual(plan["accepted_count"], 1)
        self.assertEqual(plan["best_directive_id"], "seal_retrieval_ttt")
        self.assertGreaterEqual(len(plan["self_generated_eval_probes"]), 1)
        self.assertEqual(plan["candidate_ledger"]["entry_count"], 3)
        self.assertEqual(plan["candidate_ledger"]["accepted_count"], 1)
        json.dumps(plan)

    def test_shadow_round_executes_directives_in_isolated_sandbox(self):
        planner = SelfHealingChelationPlanner(
            SelfHealingChelationConfig(baseline_fitness=0.2, max_directives=2),
            logger=MagicMock(),
        )
        plan = planner.execute_shadow_round(
            context=["adaptive retrieval collapse", "quantized adapter repair"],
            diagnostics={"runtime": {"status": "empty_results"}},
            fitness=lambda directive: FitnessEvaluation(
                candidate_id=directive.directive_id,
                fitness=0.3 if directive.directive_id == "seal_retrieval_ttt" else 0.2,
                metrics={"retention_score": 0.9},
            ),
        )

        self.assertTrue(plan["shadow_execution"]["isolated"])
        self.assertEqual(plan["shadow_execution"]["executed_count"], 2)
        self.assertFalse(plan["safety"]["base_model_mutation_allowed"])
        self.assertEqual(plan["accepted_count"], 1)

    def test_adaptive_validation_loop_records_debug_and_replay_actions(self):
        planner = SelfHealingChelationPlanner(
            SelfHealingChelationConfig(baseline_fitness=0.5, max_directives=2),
            logger=MagicMock(),
        )
        calls = {"count": 0}

        def fitness(directive):
            calls["count"] += 1
            score = 0.4 if calls["count"] <= 2 else 0.7
            return FitnessEvaluation(directive.directive_id, score, metrics={"retention_score": 0.9})

        loop = planner.run_adaptive_validation_loop(
            context="adaptive retrieval chelation",
            diagnostics={},
            fitness=fitness,
            rounds=2,
        )

        self.assertEqual(loop["rounds_completed"], 2)
        self.assertEqual(loop["rounds"][0]["status"], "debug")
        self.assertEqual(loop["rounds"][1]["status"], "pass")
        self.assertIn("increase_probe_coverage", loop["rounds"][0]["actions"])
        self.assertIn("retain_best_candidate_for_shadow_replay", loop["rounds"][1]["actions"])

    def test_candidate_ledger_records_hashes_and_gate_reasons(self):
        planner = SelfHealingChelationPlanner(SelfHealingChelationConfig(baseline_fitness=0.5), logger=MagicMock())
        directive = SelfEditDirective("candidate", "implication_synthesis", "adapter_sft")
        evaluation = planner.evaluate_directives([directive], lambda _directive: 0.4)[0]
        ledger = CandidateProvenanceLedger()
        entry = ledger.record(
            evaluation,
            baseline_fitness=0.5,
            context_hash="ctx",
            diagnostics_hash="diag",
            safety={"adapter_only": True},
        )

        self.assertEqual(entry.status, "rejected")
        self.assertIn("reward_not_positive", entry.reasons)
        self.assertEqual(len(entry.directive_hash), 16)
        self.assertEqual(ledger.to_dict()["rejected_count"], 1)

    def test_config_validation_rejects_unsafe_bounds(self):
        with self.assertRaises(ValueError):
            SelfHealingChelationConfig(min_retention_score=1.5)
        with self.assertRaises(ValueError):
            SelfHealingChelationConfig(max_directives=0)
        with self.assertRaises(ValueError):
            SelfHealingChelationConfig(min_structural_health=-0.1)


class TestSelfHealingPlannerResearch(unittest.TestCase):
    def tearDown(self) -> None:
        os.environ.pop("CHELATED_SHIM_RESEARCH", None)
        os.environ.pop("CHELATED_SHIM_PROMOTED", None)

    def test_research_meta_emits_on_plan_build(self) -> None:
        os.environ["CHELATED_SHIM_RESEARCH"] = "1"
        planner = SelfHealingChelationPlanner(
            logger=MagicMock(),
        )
        plan = planner.build_update_plan(
            context="retain retrieval facts",
            diagnostics={"structural_health": {"score": 0.52}},
            fitness=lambda directive: 1.0,
        )

        meta = planner.get_last_research_shim_meta()
        self.assertIsNotNone(meta)
        self.assertEqual(meta.get("sip_seam"), "SelfHealingChelationPlanner.build_update_plan")
        self.assertTrue(meta.get("research_shim_guard"))
        self.assertIn("research_shim_meta", plan)

    def test_research_meta_absent_by_default(self) -> None:
        planner = SelfHealingChelationPlanner(logger=MagicMock())
        plan = planner.build_update_plan(
            context="test context",
            diagnostics={"structural_health": {"score": 0.8}},
            fitness=lambda directive: 0.5,
        )

        meta = planner.get_last_research_shim_meta()
        self.assertIsNone(meta)
        self.assertIn("research_shim_meta", plan)
        self.assertIsNone(plan.get("research_shim_meta"))

    def test_research_meta_includes_promoted_sip_apply_when_promoted_enabled(self) -> None:
        os.environ["CHELATED_SHIM_RESEARCH"] = "1"
        os.environ["CHELATED_SHIM_PROMOTED"] = "1"
        planner = SelfHealingChelationPlanner(
            logger=MagicMock(),
        )
        plan = planner.build_update_plan(
            context="retain retrieval facts",
            diagnostics={"structural_health": {"score": 0.52}},
            fitness=lambda directive: 1.0,
        )

        meta = planner.get_last_research_shim_meta()
        self.assertIsNotNone(meta)
        self.assertIn("promoted_sip_apply", meta)
        self.assertIsInstance(meta["promoted_sip_apply"], dict)
        self.assertEqual(meta.get("sip_seam"), "SelfHealingChelationPlanner.build_update_plan")
        self.assertEqual(plan["research_shim_meta"], meta)


class TestSelfEditDirectiveOPSDIntegrator(unittest.TestCase):
    """Loop 1 tests for Agent 6 OPSD integration scaffold (SelfEditDirective -> on-policy distillation data).

    These tests validate the bridge from advisory SEAL/EGGROLL directives to
    teacher-student asymmetric self-distillation batches. They run without torch.
    Full objective + KL tests require a torch environment + ChelationAdapter instance.
    """

    def test_directive_to_onpolicy_batch_produces_privileged_teacher_data(self):
        from self_healing_chelation import SelfEditDirectiveOPSDIntegrator, OPSDTrainingBatch
        planner = SelfHealingChelationPlanner(logger=MagicMock())
        directives = planner.generate_directives(
            ["critical new retrieval fact under collapse"],
            {
                "runtime": {"status": "empty_results"},
                "structural_health": {"score": 0.55},
                "quantization_gate": {"passed": False},
            },
        )
        integrator = SelfEditDirectiveOPSDIntegrator()
        # Pick a directive that would be accepted in real flow (we test conversion regardless)
        directive = directives[0]
        batch = integrator.directive_to_onpolicy_batch(
            directive,
            context=["critical new retrieval fact under collapse"],
            diagnostics={
                "runtime": {"status": "empty_results"},
                "structural_health": {"score": 0.55},
                "quantization_gate": {"passed": False},
            },
        )

        self.assertIsInstance(batch, OPSDTrainingBatch)
        self.assertEqual(batch.directive_id, directive.directive_id)
        self.assertGreater(len(batch.student_inputs), 0)
        self.assertGreater(len(batch.teacher_targets), len(batch.student_inputs), "Teacher must have privileged extras")
        # Verify privileged injection happened
        privileged_hints = [t for t in batch.teacher_targets if "PRIVILEGED" in t]
        self.assertGreater(len(privileged_hints), 0, "Asymmetric privileged teacher context must be injected for OPSD")
        self.assertIn("quantization", " ".join(privileged_hints).lower() + " " + batch.directive_id.lower())
        self.assertGreater(batch.kl_weight, 0.0)
        self.assertIn("is_accepted_directive", batch.metadata)

    def test_kl_regularization_returns_sensible_value_or_scaffold(self):
        from self_healing_chelation import SelfEditDirectiveOPSDIntegrator
        integrator = SelfEditDirectiveOPSDIntegrator()
        # Without torch this returns diagnostic dict (test safe)
        result = integrator.compute_embedding_kl_regularization(
            base_embeddings=[0.1] * 8,  # fake
            adapted_embeddings=[0.12] * 8,
            weight=0.1,
        )
        # Either torch tensor or the scaffold dict
        if isinstance(result, dict):
            self.assertIn("note", result)
            self.assertIn("torch", result.get("note", "").lower())
        else:
            # If torch present, value should be small positive
            self.assertGreater(float(result), 0)

    def test_asymmetric_objective_handles_missing_torch_gracefully(self):
        from self_healing_chelation import SelfEditDirectiveOPSDIntegrator, OPSDTrainingBatch
        integrator = SelfEditDirectiveOPSDIntegrator()
        dummy_batch = OPSDTrainingBatch(
            directive_id="test",
            strategy="test",
            adaptation_mode="adapter_sft",
            student_inputs=["q1"],
            teacher_targets=["t1", "PRIVILEGED: extra"],
            positive_pairs=[],
            negative_pairs=[],
            privileged_diagnostics={},
            optimization_hints={},
            kl_weight=0.1,
        )
        # Pass fake non-tensor; method must not crash
        res = integrator.build_asymmetric_teacher_student_objective(
            dummy_batch, "fake_student", "fake_teacher"
        )
        self.assertIn("total", res)
        self.assertTrue(res["total"] == 0.0 or res.get("distill_loss") == "torch_unavailable")


if __name__ == "__main__":
    unittest.main()
