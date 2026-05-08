import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from learned_reformulation_gate import (
    enrich_query_rows,
    filter_reformulation_rows,
    load_attribution_pool,
    load_query_attribution_rows,
    predict_single,
    train_gate_from_pool,
    train_reformulation_gate,
)


class TestLearnedReformulationGate(unittest.TestCase):
    def test_load_query_attribution_rows_from_artifact(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            artifact_path = Path(tmpdir) / "artifact.json"
            artifact_path.write_text(
                '{"engine_scope_rows": [{"row_type": "query_profile", "profile": "reform_rrf_v2", "query_id": "q1", "task": "SciFact"}, {"row_type": "mask_probe", "query_id": "mask-q1"}]}',
                encoding="utf-8",
            )
            rows = load_query_attribution_rows([artifact_path])

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["profile"], "reform_rrf_v2")
        self.assertIn("_artifact_path", rows[0])
        self.assertEqual(rows[0]["row_type"], "query_profile")

    @patch("learned_reformulation_gate.load_mteb_data")
    def test_enrich_query_rows_backfills_lexical_features(self, mock_load_mteb_data):
        mock_load_mteb_data.return_value = (
            {"d1": "doc"},
            {"q1": "LDL cholesterol has no involvement in 10 percent risk"},
            {"q1": {"d1": 1.0}},
        )
        rows = enrich_query_rows([
            {"task": "SciFact", "query_id": "q1", "profile": "reform_rrf_v2", "delta_ndcg_at_10": 0.01}
        ])

        self.assertEqual(rows[0]["query_id"], "q1")
        self.assertGreater(rows[0]["query_token_count"], 0)
        self.assertGreaterEqual(rows[0]["query_negation_count"], 1)
        self.assertGreaterEqual(rows[0]["query_numeric_token_count"], 1)

    def test_filter_reformulation_rows_requires_features(self):
        rows = [
            {"profile": "baseline", "delta_ndcg_at_10": 0.0},
            {"profile": "reform_rrf_v2", "delta_ndcg_at_10": 0.1, "query_token_count": 3},
            {
                "profile": "reform_rrf_v2",
                "delta_ndcg_at_10": 0.1,
                "query_token_count": 3,
                "query_char_count": 20,
                "query_stopword_ratio": 0.4,
                "query_numeric_token_count": 0,
                "query_negation_count": 0,
                "query_claim_cue_count": 1,
            },
        ]
        filtered = filter_reformulation_rows(rows)

        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0]["profile"], "reform_rrf_v2")
        self.assertEqual(filtered[0]["row_type"], "query_profile")
        self.assertIn("top10_overlap_with_baseline", filtered[0])

    def test_train_reformulation_gate_fail_closed_without_positive_support(self):
        rows = []
        for index in range(6):
            rows.append({
                "task": "SciFact",
                "query_id": f"q{index}",
                "profile": "reform_rrf_v2",
                "delta_ndcg_at_10": 0.0,
                "query_token_count": 3,
                "query_char_count": 20,
                "query_stopword_ratio": 0.4,
                "query_numeric_token_count": 0,
                "query_negation_count": 0,
                "query_claim_cue_count": 1,
            })
        result = train_reformulation_gate(rows, min_positive_examples=1)

        self.assertIsNone(result["gate"])
        self.assertEqual(result["training_summary"]["positive_examples"], 0)
        self.assertEqual(result["feature_space"], "engine_scope")
        self.assertEqual(result["deployment_mode"], "advisory_only")
        self.assertFalse(result["runtime_compatible"])

    def test_train_reformulation_gate_accepts_simple_separable_case(self):
        rows = []
        for index in range(20):
            low_specificity = index < 10
            rows.append({
                "task": "SciFact",
                "query_id": f"q{index}",
                "profile": "reform_rrf_v2",
                "delta_ndcg_at_10": 0.02 if low_specificity else -0.01,
                "query_token_count": 3 if low_specificity else 12,
                "query_char_count": 24 if low_specificity else 88,
                "query_stopword_ratio": 0.45 if low_specificity else 0.05,
                "query_numeric_token_count": 0,
                "query_negation_count": 0,
                "query_claim_cue_count": 0,
            })
        result = train_reformulation_gate(
            rows,
            min_train_support=2,
            min_holdout_support=1,
            min_mean_delta=0.0,
            min_positive_examples=2,
        )

        self.assertIsNotNone(result["gate"])
        self.assertGreaterEqual(len(result["accepted"]), 1)
        self.assertEqual(result["gate"]["type"], "linear_classifier")
        self.assertEqual(result["deployment_mode"], "advisory_only")
        self.assertFalse(result["runtime_compatible"])


class TestTrainGateFromPool(unittest.TestCase):
    def _make_pool_rows(self, n_reform: int = 10, n_fast: int = 10) -> list:
        rows = []
        for i in range(n_reform):
            rows.append({
                "task": "SciFact",
                "query_id": f"r{i}",
                "action": "REFORMULATE",
                "delta_ndcg_at_10": -0.01,
                "fault_class": "reference",
                "global_variance": 0.3,
                "jaccard": 0.5,
                "mask_density": 0.6,
                "reformulation_variant_count": 3,
                "reformulation_changed": True,
                "query_token_count": 3,
                "query_char_count": 20,
                "query_stopword_ratio": 0.4,
                "query_numeric_token_count": 0,
                "query_negation_count": 0,
                "query_claim_cue_count": 1,
                "top10_overlap_with_baseline": 0.6,
                "top_doc_changed": False,
            })
        for i in range(n_fast):
            rows.append({
                "task": "SciFact",
                "query_id": f"f{i}",
                "action": "FAST",
                "delta_ndcg_at_10": 0.0,
                "fault_class": "no_op_tied",
                "global_variance": 0.05,
                "jaccard": 0.9,
                "mask_density": 0.1,
                "reformulation_variant_count": 0,
                "reformulation_changed": False,
                "query_token_count": 10,
                "query_char_count": 80,
                "query_stopword_ratio": 0.1,
                "query_numeric_token_count": 1,
                "query_negation_count": 0,
                "query_claim_cue_count": 2,
                "top10_overlap_with_baseline": 0.9,
                "top_doc_changed": False,
            })
        return rows

    def test_train_gate_from_pool_returns_config_structure(self):
        pool = {"query_attribution_rows": self._make_pool_rows()}
        result = train_gate_from_pool(pool, min_train_support=2, min_holdout_support=1)

        self.assertIn("version", result)
        self.assertEqual(result["version"], 1)
        self.assertEqual(result["policy"], "query_reformulation_action_classifier")
        self.assertEqual(result["feature_space"], "attribution_pool")
        self.assertIn("training_summary", result)
        self.assertEqual(result["training_summary"]["positive_action"], "REFORMULATE")

    def test_train_gate_from_pool_finds_gate_with_balanced_data(self):
        pool = {"query_attribution_rows": self._make_pool_rows(n_reform=15, n_fast=15)}
        result = train_gate_from_pool(pool, min_train_support=2, min_holdout_support=1)

        self.assertIsNotNone(result["gate"])
        self.assertGreater(len(result["accepted"]), 0)
        self.assertEqual(result["gate"]["type"], "linear_classifier")

    def test_train_gate_from_pool_fail_closed_no_reformulate(self):
        rows = self._make_pool_rows(n_reform=0, n_fast=10)
        pool = {"query_attribution_rows": rows}
        result = train_gate_from_pool(pool)

        self.assertIsNone(result["gate"])
        self.assertEqual(result["accepted"], [])

    def test_train_gate_from_pool_raises_on_empty_pool(self):
        with self.assertRaises(ValueError):
            train_gate_from_pool({"query_attribution_rows": []})

    def test_load_attribution_pool_raises_on_missing_file(self):
        with self.assertRaises(FileNotFoundError):
            load_attribution_pool("/nonexistent/pool.json")

    def test_load_attribution_pool_reads_json(self):
        pool_data = {"query_attribution_rows": [], "build_timestamp": "2026-01-01"}
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "pool.json"
            path.write_text(json.dumps(pool_data), encoding="utf-8")
            result = load_attribution_pool(path)
        self.assertEqual(result["build_timestamp"], "2026-01-01")


class TestPredictSingle(unittest.TestCase):
    def _make_gate_config(self, with_gate: bool = True) -> dict:
        """Return a minimal gate config for testing."""
        if not with_gate:
            return {"gate": None, "version": 1}
        gate = {
            "type": "linear_classifier",
            "threshold": 0.5,
            "features": ["query_token_count", "jaccard"],
            "means": [5.0, 0.5],
            "scales": [2.0, 0.3],
            "weights": [1.0, -1.0],
            "intercept": 0.0,
        }
        return {"gate": gate, "version": 1}

    def test_predict_single_fail_closed_when_no_gate(self):
        config = self._make_gate_config(with_gate=False)
        self.assertFalse(predict_single({"query_token_count": 3, "jaccard": 0.1}, config))

    def test_predict_single_returns_bool(self):
        config = self._make_gate_config(with_gate=True)
        result = predict_single({"query_token_count": 3, "jaccard": 0.1}, config)
        self.assertIsInstance(result, bool)

    def test_predict_single_missing_features_default_zero(self):
        config = self._make_gate_config(with_gate=True)
        result = predict_single({}, config)
        self.assertIsInstance(result, bool)

    def test_predict_single_high_score_predicts_true(self):
        # low token_count (3) is below mean (5) → normalized = -1 → logit = 10+5 = 15 → sigmoid ≈ 1
        gate = {
            "type": "linear_classifier",
            "threshold": 0.01,
            "features": ["query_token_count"],
            "means": [5.0],
            "scales": [2.0],
            "weights": [-10.0],
            "intercept": 5.0,
        }
        config = {"gate": gate, "version": 1}
        self.assertTrue(predict_single({"query_token_count": 3}, config))

    def test_predict_single_low_score_predicts_false(self):
        # low token_count (3) → normalized = -1 → logit = -10-5 = -15 → sigmoid ≈ 0
        gate = {
            "type": "linear_classifier",
            "threshold": 0.99,
            "features": ["query_token_count"],
            "means": [5.0],
            "scales": [2.0],
            "weights": [10.0],
            "intercept": -5.0,
        }
        config = {"gate": gate, "version": 1}
        self.assertFalse(predict_single({"query_token_count": 3}, config))


if __name__ == "__main__":
    unittest.main()