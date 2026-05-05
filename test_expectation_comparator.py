import unittest

from expectation_comparator import ExpectationComparatorConfig, ModelScopeExpectationComparator


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
