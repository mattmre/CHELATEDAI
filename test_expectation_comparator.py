"""Tests for expectation_comparator — Slice-16 ComparatorRule API and ModelScopeExpectationComparator."""
from __future__ import annotations

import unittest

from expectation_comparator import (
    ComparisonResult,
    ExpectationComparator,
    ExpectationComparatorConfig,
    FeatureOverlapRule,
    InterventionCountRule,
    MeanActivationRule,
    ModelScopeExpectationComparator,
    ReplaySetGenerator,
)
from model_scope_memory import EpisodicMemory


def _artifact(prompt_hash: str, *, feature_id: str, value: float = 1.0, token_count: int = 3):
    return {
        "runtime": {"model_name": "Qwen/Qwen3.5-2B"},
        "capture": {
            "prompt_hash": prompt_hash,
            "token_count": token_count,
            "captured_layer_count": 1,
            "layer_indices": [0],
            "observations": [
                {
                    "layer_index": 0,
                    "feature_summary": {
                        "feature_space": "qwen_scope_sae",
                        "active_features": [{"feature_id": feature_id, "value": value}],
                    },
                }
            ],
        },
    }


# ---------------------------------------------------------------------------
# MeanActivationRule
# ---------------------------------------------------------------------------


class TestMeanActivationRule(unittest.TestCase):
    def test_zero_delta_same_values_passes(self):
        rule = MeanActivationRule(threshold=0.15)
        result = rule.evaluate({"mean_activation": 1.0}, {"mean_activation": 1.0})
        self.assertTrue(result.passed)
        self.assertAlmostEqual(result.delta, 0.0, places=6)

    def test_passes_when_delta_within_threshold(self):
        rule = MeanActivationRule(threshold=0.2)
        result = rule.evaluate({"mean_activation": 1.0}, {"mean_activation": 1.1})
        self.assertTrue(result.passed)

    def test_fails_when_delta_exceeds_threshold(self):
        rule = MeanActivationRule(threshold=0.05)
        result = rule.evaluate({"mean_activation": 1.0}, {"mean_activation": 1.5})
        self.assertFalse(result.passed)

    def test_threshold_boundary_exactly_at_threshold_passes(self):
        rule = MeanActivationRule(threshold=0.1)
        # delta = 0.1 / 1 = 0.1 which equals threshold → passes
        result = rule.evaluate({"mean_activation": 1.0}, {"mean_activation": 1.1})
        self.assertTrue(result.passed)

    def test_threshold_boundary_just_over_fails(self):
        rule = MeanActivationRule(threshold=0.1)
        result = rule.evaluate({"mean_activation": 1.0}, {"mean_activation": 1.2})
        self.assertFalse(result.passed)

    def test_zero_baseline_uses_epsilon(self):
        rule = MeanActivationRule(threshold=0.5)
        result = rule.evaluate({"mean_activation": 0.0}, {"mean_activation": 0.0})
        self.assertAlmostEqual(result.delta, 0.0, places=4)
        self.assertTrue(result.passed)

    def test_rule_name(self):
        self.assertEqual(MeanActivationRule().name, "mean_activation")

    def test_result_fields(self):
        rule = MeanActivationRule(threshold=0.3)
        result = rule.evaluate({"mean_activation": 2.0}, {"mean_activation": 2.0})
        self.assertIsInstance(result, ComparisonResult)
        self.assertEqual(result.rule_name, "mean_activation")
        self.assertAlmostEqual(result.threshold, 0.3)


# ---------------------------------------------------------------------------
# FeatureOverlapRule
# ---------------------------------------------------------------------------


class TestFeatureOverlapRule(unittest.TestCase):
    def test_full_overlap_passes(self):
        rule = FeatureOverlapRule(threshold=0.5)
        feats = {"a": 1.0, "b": 2.0}
        result = rule.evaluate({"features": feats}, {"features": feats})
        self.assertTrue(result.passed)
        self.assertAlmostEqual(result.delta, 1.0, places=4)

    def test_no_overlap_fails(self):
        rule = FeatureOverlapRule(threshold=0.5)
        result = rule.evaluate(
            {"features": {"a": 1.0}},
            {"features": {"b": 1.0}},
        )
        self.assertFalse(result.passed)

    def test_partial_overlap(self):
        rule = FeatureOverlapRule(threshold=0.4)
        result = rule.evaluate(
            {"features": {"a": 1.0, "b": 1.0, "c": 1.0}},
            {"features": {"b": 1.0, "c": 1.0, "d": 1.0}},
        )
        # intersection={b,c}=2, union={a,b,c,d}=4 → overlap≈0.5 → passes
        self.assertTrue(result.passed)
        self.assertGreater(result.delta, 0.4)

    def test_empty_features_edge_case(self):
        rule = FeatureOverlapRule(threshold=0.0)
        result = rule.evaluate({"features": {}}, {"features": {}})
        # Both empty → union=0 → overlap≈0/1e-9≈0 → passes at threshold=0
        self.assertTrue(result.passed)

    def test_zero_value_features_excluded(self):
        rule = FeatureOverlapRule(threshold=0.5)
        result = rule.evaluate(
            {"features": {"a": 0.0, "b": 1.0}},
            {"features": {"a": 0.0, "b": 1.0}},
        )
        # Only "b" is nonzero on both sides → full overlap
        self.assertTrue(result.passed)
        self.assertAlmostEqual(result.delta, 1.0, places=4)

    def test_threshold_boundary_exact(self):
        rule = FeatureOverlapRule(threshold=0.5)
        result = rule.evaluate(
            {"features": {"a": 1.0, "b": 1.0}},
            {"features": {"a": 1.0, "c": 1.0}},
        )
        # intersection={a}=1, union={a,b,c}=3 → 1/3 ≈ 0.333 < 0.5 → fails
        self.assertFalse(result.passed)

    def test_rule_name(self):
        self.assertEqual(FeatureOverlapRule().name, "feature_overlap")


# ---------------------------------------------------------------------------
# InterventionCountRule
# ---------------------------------------------------------------------------


class TestInterventionCountRule(unittest.TestCase):
    def test_within_max_passes(self):
        rule = InterventionCountRule(max_interventions=5)
        result = rule.evaluate({}, {"intervention_count": 3})
        self.assertTrue(result.passed)

    def test_at_max_passes(self):
        rule = InterventionCountRule(max_interventions=5)
        result = rule.evaluate({}, {"intervention_count": 5})
        self.assertTrue(result.passed)

    def test_exceeds_max_fails(self):
        rule = InterventionCountRule(max_interventions=5)
        result = rule.evaluate({}, {"intervention_count": 6})
        self.assertFalse(result.passed)

    def test_zero_count_passes(self):
        rule = InterventionCountRule(max_interventions=5)
        result = rule.evaluate({}, {"intervention_count": 0})
        self.assertTrue(result.passed)

    def test_delta_is_float_cast_of_count(self):
        rule = InterventionCountRule(max_interventions=10)
        result = rule.evaluate({}, {"intervention_count": 7})
        self.assertIsInstance(result.delta, float)
        self.assertAlmostEqual(result.delta, 7.0)

    def test_threshold_equals_max_interventions(self):
        rule = InterventionCountRule(max_interventions=3)
        result = rule.evaluate({}, {"intervention_count": 2})
        self.assertAlmostEqual(result.threshold, 3.0)

    def test_rule_name(self):
        self.assertEqual(InterventionCountRule().name, "intervention_count")


# ---------------------------------------------------------------------------
# ExpectationComparator
# ---------------------------------------------------------------------------


class TestExpectationComparator(unittest.TestCase):
    def _all_pass_comparator(self):
        ec = ExpectationComparator()
        ec.add_rule(MeanActivationRule(threshold=1.0))
        return ec

    def test_add_rule_increases_rule_count(self):
        ec = ExpectationComparator()
        ec.add_rule(MeanActivationRule())
        self.assertEqual(len(ec.rules), 1)

    def test_compare_returns_one_result_per_rule(self):
        ec = ExpectationComparator()
        ec.add_rule(MeanActivationRule())
        ec.add_rule(InterventionCountRule())
        results = ec.compare({"mean_activation": 1.0}, {"mean_activation": 1.0, "intervention_count": 1})
        self.assertEqual(len(results), 2)

    def test_all_passed_all_rules_pass(self):
        ec = self._all_pass_comparator()
        self.assertTrue(ec.all_passed({"mean_activation": 1.0}, {"mean_activation": 1.0}))

    def test_all_passed_false_when_any_fails(self):
        ec = ExpectationComparator()
        ec.add_rule(MeanActivationRule(threshold=0.0))
        self.assertFalse(ec.all_passed({"mean_activation": 1.0}, {"mean_activation": 2.0}))

    def test_summary_structure(self):
        ec = ExpectationComparator()
        ec.add_rule(MeanActivationRule())
        summary = ec.summary({"mean_activation": 1.0}, {"mean_activation": 1.0})
        self.assertIn("passed", summary)
        self.assertIn("results", summary)
        self.assertIn("rule_count", summary)
        self.assertEqual(summary["rule_count"], 1)

    def test_summary_passed_true_all_pass(self):
        ec = self._all_pass_comparator()
        summary = ec.summary({"mean_activation": 1.0}, {"mean_activation": 1.0})
        self.assertTrue(summary["passed"])

    def test_summary_passed_false_any_fail(self):
        ec = ExpectationComparator()
        ec.add_rule(MeanActivationRule(threshold=0.0))
        summary = ec.summary({"mean_activation": 1.0}, {"mean_activation": 5.0})
        self.assertFalse(summary["passed"])

    def test_empty_rules_all_passed_true(self):
        ec = ExpectationComparator()
        self.assertTrue(ec.all_passed({}, {}))

    def test_multiple_rules_all_pass(self):
        ec = ExpectationComparator()
        ec.add_rule(MeanActivationRule(threshold=0.5))
        ec.add_rule(InterventionCountRule(max_interventions=10))
        ec.add_rule(FeatureOverlapRule(threshold=0.0))
        self.assertTrue(
            ec.all_passed(
                {"mean_activation": 1.0, "features": {}},
                {"mean_activation": 1.1, "intervention_count": 2, "features": {}},
            )
        )


# ---------------------------------------------------------------------------
# ReplaySetGenerator
# ---------------------------------------------------------------------------


class TestReplaySetGenerator(unittest.TestCase):
    def _setup(self):
        em = EpisodicMemory()
        em.store("baseline", {"mean_activation": 1.0, "intervention_count": 0}, tags=["ep1"])
        em.store("cand1", {"mean_activation": 1.05, "intervention_count": 1}, tags=["ep1"])
        em.store("cand2", {"mean_activation": 1.2, "intervention_count": 3}, tags=["ep1"])
        em.store("other", {"mean_activation": 0.0}, tags=["ep2"])
        ec = ExpectationComparator()
        ec.add_rule(MeanActivationRule(threshold=0.5))
        ec.add_rule(InterventionCountRule(max_interventions=5))
        return em, ec

    def test_generate_returns_candidates_only(self):
        em, ec = self._setup()
        rsg = ReplaySetGenerator(em)
        results = rsg.generate("ep1", ec, "baseline")
        self.assertEqual(len(results), 2)

    def test_generate_result_has_entry_and_comparison(self):
        em, ec = self._setup()
        rsg = ReplaySetGenerator(em)
        results = rsg.generate("ep1", ec, "baseline")
        for item in results:
            self.assertIn("entry", item)
            self.assertIn("comparison", item)

    def test_generate_comparison_has_expected_keys(self):
        em, ec = self._setup()
        rsg = ReplaySetGenerator(em)
        results = rsg.generate("ep1", ec, "baseline")
        for item in results:
            comp = item["comparison"]
            self.assertIn("passed", comp)
            self.assertIn("results", comp)
            self.assertIn("rule_count", comp)

    def test_generate_no_baseline_returns_empty(self):
        em = EpisodicMemory()
        em.store("cand", {}, tags=["ep"])
        ec = ExpectationComparator()
        rsg = ReplaySetGenerator(em)
        self.assertEqual(rsg.generate("ep", ec, "missing_baseline"), [])

    def test_generate_empty_episode_returns_empty(self):
        em = EpisodicMemory()
        ec = ExpectationComparator()
        rsg = ReplaySetGenerator(em)
        self.assertEqual(rsg.generate("no_ep", ec, "baseline"), [])

    def test_generate_excludes_baseline_key_from_results(self):
        em, ec = self._setup()
        rsg = ReplaySetGenerator(em)
        results = rsg.generate("ep1", ec, "baseline")
        keys = [item["entry"].key for item in results]
        self.assertNotIn("baseline", keys)


# ---------------------------------------------------------------------------
# Legacy ModelScopeExpectationComparator tests (preserved)
# ---------------------------------------------------------------------------


class TestModelScopeExpectationComparator(unittest.TestCase):
    def test_compare_to_profile_passes_for_matching_artifact(self):
        comparator = ModelScopeExpectationComparator(
            ExpectationComparatorConfig(min_layer_overlap=1.0, min_feature_jaccard=1.0)
        )
        baseline = _artifact("q1", feature_id="11", value=1.5)
        profile = comparator.build_expectation_profile(baseline, profile_id="q1")

        result = comparator.compare_to_profile(_artifact("q1", feature_id="11", value=1.5), profile)

        self.assertTrue(result["passed"])
        self.assertAlmostEqual(result["score"], 1.0, places=6)
        self.assertEqual(result["profile_id"], "q1")

    def test_compare_to_profile_fails_for_feature_drift(self):
        comparator = ModelScopeExpectationComparator(
            ExpectationComparatorConfig(min_layer_overlap=1.0, min_feature_jaccard=0.8)
        )
        baseline = _artifact("q1", feature_id="11", value=1.5)
        profile = comparator.build_expectation_profile(baseline, profile_id="q1")

        result = comparator.compare_to_profile(_artifact("q1", feature_id="99", value=2.0), profile)

        self.assertFalse(result["passed"])
        self.assertIn("feature_jaccard_below_threshold", result["reasons"])


if __name__ == "__main__":
    unittest.main()

