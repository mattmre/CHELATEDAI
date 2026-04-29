"""Stage 3 closed-course safety-testbed tests."""

import json
import unittest

from reproducibility_context import stable_hash
from run_safety_testbed import (
    build_closed_course_engine,
    closed_course_fixture,
    freeze_initial_values,
    run_closed_course_safety_testbed,
    _evaluate,
    _standard_rankings,
)


class TestStage3ClosedCourse(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = run_closed_course_safety_testbed()

    def test_non_saturated_fixture_baseline_is_frozen_and_reproducible(self):
        fixture = closed_course_fixture()
        engine, _logger, _fixture = build_closed_course_engine()
        rankings = _standard_rankings(engine, fixture["queries"], fixture["k"])
        evaluation = _evaluate(rankings, fixture["qrels"], fixture["k"])
        initial = freeze_initial_values(fixture, evaluation)

        self.assertGreater(evaluation.fitness, 0.0)
        self.assertLess(evaluation.fitness, 1.0)
        self.assertEqual(initial.dataset_hash, self.report["initial_values"]["dataset_hash"])
        self.assertEqual(initial.query_set_hash, stable_hash(fixture["queries"]))
        self.assertFalse(initial.metadata["saturated"])

    def test_chelation_only_loop_reports_actions_and_does_not_change_baseline(self):
        baseline = self.report["baseline"]
        chelation = self.report["chelation"]

        self.assertIn("CHELATE", chelation["action_mix"])
        self.assertGreaterEqual(sum(chelation["action_mix"].values()), 1)
        self.assertEqual(baseline["fitness"], self.report["initial_values"]["fitness"])
        self.assertEqual(len(chelation["diagnostics"]), self.report["fixture"]["query_count"])

    def test_reformulation_loop_emits_multi_variant_policy_metadata(self):
        reform = self.report["reformulation"]
        policies = [
            diagnostic["retrieval_policy"]["policy"]
            for diagnostic in reform["diagnostics"]
            if diagnostic.get("retrieval_policy")
        ]
        variant_counts = [
            diagnostic["query_reformulation"]["variant_count"]
            for diagnostic in reform["diagnostics"]
            if diagnostic.get("query_reformulation")
        ]

        self.assertTrue(policies)
        self.assertTrue(all(policy == "multi_variant_merge" for policy in policies))
        self.assertTrue(all(count >= 1 for count in variant_counts))

    def test_routing_loop_records_route_effectiveness_over_queries(self):
        effectiveness = self.report["routing"]["route_effectiveness"]

        self.assertEqual(effectiveness["total_routes_observed"], self.report["fixture"]["query_count"])
        self.assertGreaterEqual(len(effectiveness["routes"]), 1)
        for stats in effectiveness["routes"].values():
            self.assertIn("mean_jaccard", stats)
            self.assertGreaterEqual(stats["count"], 1)

    def test_quantization_loop_is_explicit_and_blocks_default_promotion(self):
        quantization = self.report["quantization"]
        decision = self.report["decision"]

        self.assertIn("passed", quantization)
        self.assertIn("retained_gain_ratio", quantization)
        self.assertFalse(decision["default_change_allowed"])
        if quantization["passed"]:
            self.assertGreaterEqual(quantization["retained_gain_ratio"], 0.8)
        else:
            self.assertTrue(quantization["reasons"])

    def test_structural_health_loop_triggers_advisory_gate_on_degradation(self):
        health = self.report["structural_health"]
        gate = self.report["adaptive_gate"]

        self.assertGreater(health["healthy"]["score"], health["degraded"]["score"])
        self.assertIn("reduce_optimization_aggression", gate["actions"])
        self.assertIn("enable_query_reformulation", gate["actions"])

    def test_diagnostics_loop_is_json_safe_and_dashboard_backed(self):
        report = self.report
        diagnostics = report["diagnostics_report"]
        dashboard = report["dashboard_summary"]

        json_payload = json.dumps(report, sort_keys=True)

        self.assertNotIn("tensor(", json_payload)
        self.assertNotIn("array(", json_payload)
        self.assertEqual(diagnostics["phase"], "closed_course")
        self.assertGreaterEqual(dashboard["runtime_diagnostics_count"], report["fixture"]["query_count"])
        self.assertIn("adaptive_gate", diagnostics)


if __name__ == "__main__":
    unittest.main(verbosity=2)
