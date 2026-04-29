import unittest

from synthetic_collapse_benchmark import (
    build_synthetic_collapse_fixture,
    evaluate_synthetic_collapse,
    run_synthetic_collapse_benchmark,
)


class TestSyntheticCollapseBenchmark(unittest.TestCase):
    def test_masked_collapse_dimension_recovers_rankings(self):
        result = run_synthetic_collapse_benchmark(topic_count=4, collapse_strength=4.0)

        self.assertTrue(result["recovered"])
        self.assertLess(result["baseline"]["metrics"]["ndcg_at_3"], result["masked"]["metrics"]["ndcg_at_3"])
        self.assertEqual(result["masked"]["metrics"]["ndcg_at_3"], 1.0)

    def test_without_mask_collapse_distractor_wins(self):
        fixture = build_synthetic_collapse_fixture(topic_count=3, collapse_strength=5.0)
        result = evaluate_synthetic_collapse(fixture)

        for query_id, ranking in result["rankings"].items():
            self.assertIn("collapse_distractor", ranking[0])
            self.assertNotEqual(ranking[0], fixture["qrels"][query_id])

    def test_fixture_requires_multiple_topics(self):
        with self.assertRaises(ValueError):
            build_synthetic_collapse_fixture(topic_count=1)


if __name__ == "__main__":
    unittest.main()
