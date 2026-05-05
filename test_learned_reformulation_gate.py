import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from learned_reformulation_gate import (
    enrich_query_rows,
    filter_reformulation_rows,
    load_query_attribution_rows,
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


if __name__ == "__main__":
    unittest.main()
