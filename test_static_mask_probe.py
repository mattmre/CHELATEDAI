import unittest
from unittest.mock import patch

import numpy as np

from static_mask_probe import (
    build_conditional_mask_examples,
    evaluate_conditional_mask,
    learn_harmful_dimension_mask,
    rank_by_cosine,
    run_static_mask_probe,
    train_classifier_conditional_mask_gate,
    train_conditional_mask_gate,
    train_regularized_conditional_mask_gate,
)


class TestStaticMaskProbe(unittest.TestCase):
    def test_learns_harmful_dimension_from_training_queries(self):
        queries = {
            "q1": np.array([1.0, 4.0]),
            "q2": np.array([1.0, 4.0]),
        }
        docs = {
            "d_rel_1": np.array([1.0, 0.0]),
            "d_rel_2": np.array([1.0, 0.0]),
            "d_bad": np.array([0.0, 4.0]),
        }
        qrels = {"q1": {"d_rel_1": 1.0}, "q2": {"d_rel_2": 1.0}}

        learned = learn_harmful_dimension_mask(queries, docs, qrels, mask_fraction=0.5)

        self.assertEqual(learned["masked_dims"], [1])
        self.assertEqual(learned["training_examples"], 2)

    def test_mask_changes_ranking_when_collapse_dimension_removed(self):
        query = {"q": np.array([1.0, 4.0])}
        docs = {
            "relevant": np.array([1.0, 0.0]),
            "distractor": np.array([0.0, 4.0]),
        }

        baseline = rank_by_cosine(query, docs)
        masked = rank_by_cosine(query, docs, mask=np.array([1.0, 0.0]))

        self.assertEqual(baseline["q"][0], "distractor")
        self.assertEqual(masked["q"][0], "relevant")

    def test_mask_fraction_is_validated(self):
        with self.assertRaises(ValueError):
            learn_harmful_dimension_mask(
                {"q": np.array([1.0, 1.0])},
                {"d": np.array([1.0, 0.0])},
                {"q": {"d": 1.0}},
                mask_fraction=1.0,
            )

    def test_conditional_gate_selects_safe_mask_subset(self):
        examples = [
            {"query_id": "q1", "delta_ndcg_at_10": 0.2, "baseline_score_margin": 0.1, "baseline_top_score": 0.8,
             "query_norm": 1.0, "query_token_count": 3, "query_stopword_ratio": 0.0, "query_negation_count": 0,
             "query_claim_cue_count": 1},
            {"query_id": "q2", "delta_ndcg_at_10": 0.2, "baseline_score_margin": 0.2, "baseline_top_score": 0.8,
             "query_norm": 1.0, "query_token_count": 3, "query_stopword_ratio": 0.0, "query_negation_count": 0,
             "query_claim_cue_count": 1},
            {"query_id": "q3", "delta_ndcg_at_10": 0.0, "baseline_score_margin": 0.9, "baseline_top_score": 0.9,
             "query_norm": 1.0, "query_token_count": 10, "query_stopword_ratio": 0.0, "query_negation_count": 0,
             "query_claim_cue_count": 0},
        ]

        gate = train_conditional_mask_gate(examples, min_support=2, min_mean_delta=0.1)

        self.assertEqual(gate["gate"]["feature"], "baseline_score_margin")
        self.assertEqual(gate["gate"]["operator"], "<=")

    def test_conditional_mask_can_fail_closed_without_gate(self):
        query = {"q": np.array([1.0, 4.0])}
        docs = {
            "relevant": np.array([1.0, 0.0]),
            "distractor": np.array([0.0, 4.0]),
        }
        qrels = {"q": {"relevant": 1.0}}

        result = evaluate_conditional_mask(query, docs, qrels, np.array([1.0, 0.0]), None)

        self.assertEqual(result["applied_queries"], 0)
        self.assertEqual(result["metrics"]["mrr"], 0.5)

    def test_conditional_examples_include_mask_delta(self):
        query = {"q": np.array([1.0, 4.0])}
        docs = {
            "relevant": np.array([1.0, 0.0]),
            "distractor": np.array([0.0, 4.0]),
        }
        qrels = {"q": {"relevant": 1.0}}

        examples = build_conditional_mask_examples(query, docs, qrels, np.array([1.0, 0.0]))

        self.assertGreater(examples[0]["delta_ndcg_at_10"], 0.0)
        self.assertIn("baseline_score_margin", examples[0])

    def test_regularized_gate_rejects_train_only_pattern(self):
        examples = [
            {"query_id": "q1", "delta_ndcg_at_10": 0.2, "baseline_score_margin": 0.1, "baseline_top_score": 0.8,
             "query_norm": 1.0, "query_token_count": 3, "query_stopword_ratio": 0.0, "query_negation_count": 0,
             "query_claim_cue_count": 1},
            {"query_id": "q2", "delta_ndcg_at_10": 0.2, "baseline_score_margin": 0.2, "baseline_top_score": 0.8,
             "query_norm": 1.0, "query_token_count": 3, "query_stopword_ratio": 0.0, "query_negation_count": 0,
             "query_claim_cue_count": 1},
            {"query_id": "q3", "delta_ndcg_at_10": -0.2, "baseline_score_margin": 0.1, "baseline_top_score": 0.8,
             "query_norm": 1.0, "query_token_count": 3, "query_stopword_ratio": 0.0, "query_negation_count": 0,
             "query_claim_cue_count": 1},
            {"query_id": "q4", "delta_ndcg_at_10": 0.0, "baseline_score_margin": 0.9, "baseline_top_score": 0.9,
             "query_norm": 1.0, "query_token_count": 10, "query_stopword_ratio": 0.2, "query_negation_count": 0,
             "query_claim_cue_count": 0},
        ]

        gate = train_regularized_conditional_mask_gate(
            examples,
            validation_fraction=0.5,
            min_support=2,
            min_mean_delta=0.1,
        )

        self.assertIsNone(gate["gate"])

    def test_regularized_gate_accepts_validated_pattern(self):
        examples = [
            {"query_id": "q1", "delta_ndcg_at_10": 0.2, "baseline_score_margin": 0.1, "baseline_top_score": 0.8,
             "query_norm": 1.0, "query_token_count": 3, "query_stopword_ratio": 0.0, "query_negation_count": 0,
             "query_claim_cue_count": 1},
            {"query_id": "q2", "delta_ndcg_at_10": 0.2, "baseline_score_margin": 0.2, "baseline_top_score": 0.8,
             "query_norm": 1.0, "query_token_count": 3, "query_stopword_ratio": 0.0, "query_negation_count": 0,
             "query_claim_cue_count": 1},
            {"query_id": "q3", "delta_ndcg_at_10": 0.1, "baseline_score_margin": 0.1, "baseline_top_score": 0.8,
             "query_norm": 1.0, "query_token_count": 3, "query_stopword_ratio": 0.0, "query_negation_count": 0,
             "query_claim_cue_count": 1},
            {"query_id": "q4", "delta_ndcg_at_10": 0.0, "baseline_score_margin": 0.9, "baseline_top_score": 0.9,
             "query_norm": 1.0, "query_token_count": 10, "query_stopword_ratio": 0.2, "query_negation_count": 0,
             "query_claim_cue_count": 0},
        ]

        gate = train_regularized_conditional_mask_gate(
            examples,
            validation_fraction=0.5,
            min_support=2,
            min_mean_delta=0.1,
        )

        self.assertIsNotNone(gate["gate"])
        self.assertEqual(gate["accepted"][0]["validation"]["negative_examples"], 0)

    def test_classifier_gate_accepts_validated_pattern(self):
        examples = [
            {"query_id": "q1", "delta_ndcg_at_10": 0.2, "baseline_score_margin": 0.1, "baseline_top_score": 0.2,
             "query_norm": 1.0, "query_token_count": 3, "query_stopword_ratio": 0.0, "query_negation_count": 0,
             "query_claim_cue_count": 1},
            {"query_id": "q2", "delta_ndcg_at_10": 0.2, "baseline_score_margin": 0.2, "baseline_top_score": 0.2,
             "query_norm": 1.0, "query_token_count": 3, "query_stopword_ratio": 0.0, "query_negation_count": 0,
             "query_claim_cue_count": 1},
            {"query_id": "q3", "delta_ndcg_at_10": 0.0, "baseline_score_margin": 0.9, "baseline_top_score": 0.9,
             "query_norm": 1.0, "query_token_count": 10, "query_stopword_ratio": 0.2, "query_negation_count": 0,
             "query_claim_cue_count": 0},
            {"query_id": "q4", "delta_ndcg_at_10": 0.0, "baseline_score_margin": 0.8, "baseline_top_score": 0.8,
             "query_norm": 1.0, "query_token_count": 10, "query_stopword_ratio": 0.2, "query_negation_count": 0,
             "query_claim_cue_count": 0},
            {"query_id": "q5", "delta_ndcg_at_10": 0.1, "baseline_score_margin": 0.1, "baseline_top_score": 0.2,
             "query_norm": 1.0, "query_token_count": 3, "query_stopword_ratio": 0.0, "query_negation_count": 0,
             "query_claim_cue_count": 1},
            {"query_id": "q6", "delta_ndcg_at_10": 0.0, "baseline_score_margin": 0.9, "baseline_top_score": 0.9,
             "query_norm": 1.0, "query_token_count": 10, "query_stopword_ratio": 0.2, "query_negation_count": 0,
             "query_claim_cue_count": 0},
            {"query_id": "q7", "delta_ndcg_at_10": 0.1, "baseline_score_margin": 0.2, "baseline_top_score": 0.2,
             "query_norm": 1.0, "query_token_count": 3, "query_stopword_ratio": 0.0, "query_negation_count": 0,
             "query_claim_cue_count": 1},
            {"query_id": "q8", "delta_ndcg_at_10": 0.0, "baseline_score_margin": 0.8, "baseline_top_score": 0.8,
             "query_norm": 1.0, "query_token_count": 10, "query_stopword_ratio": 0.2, "query_negation_count": 0,
             "query_claim_cue_count": 0},
        ]

        gate = train_classifier_conditional_mask_gate(
            examples,
            validation_fraction=0.5,
            min_support=2,
            min_mean_delta=0.05,
            min_positive_examples=2,
        )

        self.assertIsNotNone(gate["gate"])
        self.assertEqual(gate["gate"]["type"], "linear_classifier")
        self.assertEqual(gate["accepted"][0]["validation"]["negative_examples"], 0)

    def test_classifier_gate_rejects_validation_negative_pattern(self):
        examples = [
            {"query_id": "q1", "delta_ndcg_at_10": 0.2, "baseline_score_margin": 0.1, "baseline_top_score": 0.2,
             "query_norm": 1.0, "query_token_count": 3, "query_stopword_ratio": 0.0, "query_negation_count": 0,
             "query_claim_cue_count": 1},
            {"query_id": "q2", "delta_ndcg_at_10": 0.2, "baseline_score_margin": 0.2, "baseline_top_score": 0.2,
             "query_norm": 1.0, "query_token_count": 3, "query_stopword_ratio": 0.0, "query_negation_count": 0,
             "query_claim_cue_count": 1},
            {"query_id": "q3", "delta_ndcg_at_10": 0.0, "baseline_score_margin": 0.9, "baseline_top_score": 0.9,
             "query_norm": 1.0, "query_token_count": 10, "query_stopword_ratio": 0.2, "query_negation_count": 0,
             "query_claim_cue_count": 0},
            {"query_id": "q4", "delta_ndcg_at_10": 0.0, "baseline_score_margin": 0.8, "baseline_top_score": 0.8,
             "query_norm": 1.0, "query_token_count": 10, "query_stopword_ratio": 0.2, "query_negation_count": 0,
             "query_claim_cue_count": 0},
            {"query_id": "q5", "delta_ndcg_at_10": -0.1, "baseline_score_margin": 0.1, "baseline_top_score": 0.2,
             "query_norm": 1.0, "query_token_count": 3, "query_stopword_ratio": 0.0, "query_negation_count": 0,
             "query_claim_cue_count": 1},
            {"query_id": "q6", "delta_ndcg_at_10": 0.0, "baseline_score_margin": 0.9, "baseline_top_score": 0.9,
             "query_norm": 1.0, "query_token_count": 10, "query_stopword_ratio": 0.2, "query_negation_count": 0,
             "query_claim_cue_count": 0},
            {"query_id": "q7", "delta_ndcg_at_10": -0.1, "baseline_score_margin": 0.2, "baseline_top_score": 0.2,
             "query_norm": 1.0, "query_token_count": 3, "query_stopword_ratio": 0.0, "query_negation_count": 0,
             "query_claim_cue_count": 1},
            {"query_id": "q8", "delta_ndcg_at_10": 0.0, "baseline_score_margin": 0.8, "baseline_top_score": 0.8,
             "query_norm": 1.0, "query_token_count": 10, "query_stopword_ratio": 0.2, "query_negation_count": 0,
             "query_claim_cue_count": 0},
        ]

        gate = train_classifier_conditional_mask_gate(
            examples,
            validation_fraction=0.5,
            min_support=2,
            min_mean_delta=0.05,
            min_positive_examples=2,
        )

        self.assertIsNone(gate["gate"])

    def test_run_static_probe_uses_regularized_gate_policy(self):
        corpus = {f"d{i}": f"doc{i}" for i in range(6)}
        queries = {f"q{i}": f"query{i}" for i in range(6)}
        qrels = {f"q{i}": {f"d{i}": 1.0} for i in range(6)}
        vectors = {
            **{f"doc{i}": np.array([1.0, 0.0]) if i % 2 == 0 else np.array([0.0, 1.0]) for i in range(6)},
            **{f"query{i}": np.array([1.0, 0.1]) if i % 2 == 0 else np.array([0.1, 1.0]) for i in range(6)},
        }

        class FakeBackend:
            def embed_raw(self, texts):
                return np.vstack([vectors[text] for text in texts])

        with (
            patch("static_mask_probe.load_mteb_data", return_value=(corpus, queries, qrels)),
            patch("static_mask_probe.create_embedding_backend", return_value=FakeBackend()),
        ):
            result = run_static_mask_probe(
                max_queries=6,
                train_queries=4,
                sample_docs=6,
                conditional=True,
                regularized_gate=True,
            )

        self.assertEqual(result["conditional"]["gate"]["policy"], "regularized_conditional_static_mask_gate")

    def test_run_static_probe_uses_classifier_gate_policy(self):
        corpus = {f"d{i}": f"doc{i}" for i in range(10)}
        queries = {f"q{i}": f"query{i}" for i in range(10)}
        qrels = {f"q{i}": {f"d{i}": 1.0} for i in range(10)}
        vectors = {
            **{f"doc{i}": np.array([1.0, 0.0]) if i % 2 == 0 else np.array([0.0, 1.0]) for i in range(10)},
            **{f"query{i}": np.array([1.0, 0.1]) if i % 2 == 0 else np.array([0.1, 1.0]) for i in range(10)},
        }

        class FakeBackend:
            def embed_raw(self, texts):
                return np.vstack([vectors[text] for text in texts])

        with (
            patch("static_mask_probe.load_mteb_data", return_value=(corpus, queries, qrels)),
            patch("static_mask_probe.create_embedding_backend", return_value=FakeBackend()),
        ):
            result = run_static_mask_probe(
                max_queries=10,
                train_queries=8,
                sample_docs=10,
                classifier_gate=True,
            )

        self.assertEqual(result["conditional"]["gate"]["policy"], "classifier_conditional_static_mask_gate")


if __name__ == "__main__":
    unittest.main()
