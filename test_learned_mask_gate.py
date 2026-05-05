import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from learned_mask_gate import (
    enrich_mask_rows,
    filter_mask_rows,
    load_mask_example_rows,
    main,
    train_mask_gate,
)


def _mask_row(
    query_id: str,
    delta_ndcg_at_10: float,
    baseline_score_margin: float,
    baseline_top_score: float,
    query_norm: float,
    query_token_count: int,
    query_stopword_ratio: float,
    query_negation_count: int,
    query_claim_cue_count: int,
    *,
    task: str = "SciFact",
) -> dict:
    return {
        "task": task,
        "query_id": query_id,
        "delta_ndcg_at_10": delta_ndcg_at_10,
        "baseline_score_margin": baseline_score_margin,
        "baseline_top_score": baseline_top_score,
        "query_norm": query_norm,
        "query_token_count": query_token_count,
        "query_stopword_ratio": query_stopword_ratio,
        "query_negation_count": query_negation_count,
        "query_claim_cue_count": query_claim_cue_count,
    }


class TestLearnedMaskGate(unittest.TestCase):
    def test_load_mask_example_rows_discovers_nested_rows_and_backfills_task(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            artifact_path = Path(tmpdir) / "mask-artifact.json"
            artifact_path.write_text(
                json.dumps({
                    "engine_scope_rows": [
                        {
                            "row_type": "mask_probe",
                            "task": "SciFact",
                            "query_id": "q1",
                            "delta_ndcg_at_10": 0.02,
                            "baseline_score_margin": 0.1,
                            "baseline_top_score": 0.3,
                            "query_norm": 1.0,
                            "query_token_count": 4,
                            "query_stopword_ratio": 0.25,
                            "query_negation_count": 1,
                            "query_claim_cue_count": 1,
                        },
                        {
                            "row_type": "query_profile",
                            "task": "SciFact",
                            "query_id": "q-ignore",
                        }
                    ],
                }),
                encoding="utf-8",
            )

            rows = load_mask_example_rows([artifact_path])

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["task"], "SciFact")
        self.assertEqual(rows[0]["query_id"], "q1")
        self.assertIn("_artifact_path", rows[0])
        self.assertEqual(rows[0]["row_type"], "mask_probe")

    @patch("learned_mask_gate.load_mteb_data")
    def test_enrich_mask_rows_backfills_lexical_features(self, mock_load_mteb_data):
        mock_load_mteb_data.return_value = (
            {"d1": "doc"},
            {"q1": "LDL cholesterol has no involvement in 10 percent risk"},
            {"q1": {"d1": 1.0}},
        )
        rows = enrich_mask_rows([
            {
                "task": "SciFact",
                "query_id": "q1",
                "delta_ndcg_at_10": 0.01,
                "baseline_score_margin": 0.1,
                "baseline_top_score": 0.3,
                "query_norm": 1.0,
            }
        ])

        self.assertEqual(rows[0]["query_id"], "q1")
        self.assertGreater(rows[0]["query_token_count"], 0)
        self.assertGreater(rows[0]["query_char_count"], 0)
        self.assertGreaterEqual(rows[0]["query_negation_count"], 1)
        self.assertGreaterEqual(rows[0]["query_claim_cue_count"], 0)
        self.assertEqual(rows[0]["query_text"], "LDL cholesterol has no involvement in 10 percent risk")

    def test_filter_mask_rows_requires_numeric_features(self):
        rows = [
            {"task": "SciFact", "query_id": "q1", "delta_ndcg_at_10": 0.1},
            {
                "task": "SciFact",
                "query_id": "q2",
                "delta_ndcg_at_10": 0.1,
                "baseline_score_margin": 0.1,
                "baseline_top_score": 0.2,
                "query_norm": 1.0,
                "query_token_count": 3,
                "query_stopword_ratio": 0.0,
                "query_negation_count": 0,
                "query_claim_cue_count": 1,
            },
        ]

        filtered = filter_mask_rows(rows)

        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0]["query_id"], "q2")
        self.assertIn("query_char_count", filtered[0])
        self.assertIn("query_numeric_token_count", filtered[0])

    def test_train_mask_gate_fail_closed_without_positive_support(self):
        rows = []
        for index in range(10):
            rows.append(_mask_row(
                f"q{index}",
                0.0,
                baseline_score_margin=0.5,
                baseline_top_score=0.7,
                query_norm=1.0,
                query_token_count=5,
                query_stopword_ratio=0.2,
                query_negation_count=0,
                query_claim_cue_count=0,
            ))

        result = train_mask_gate(rows, min_positive_examples=1)

        self.assertIsNone(result["gate"])
        self.assertEqual(result["training_summary"]["positive_examples"], 0)
        self.assertEqual(result["feature_space"], "engine_scope")
        self.assertEqual(result["deployment_mode"], "advisory_only")

    def test_train_mask_gate_accepts_simple_separable_case(self):
        rows = []
        for index in range(60):
            helpful_mask = index < 30
            rows.append(_mask_row(
                f"q{index}",
                0.03 if helpful_mask else -0.01,
                baseline_score_margin=0.1 if helpful_mask else 0.9,
                baseline_top_score=0.2 if helpful_mask else 0.95,
                query_norm=1.0,
                query_token_count=3 if helpful_mask else 12,
                query_stopword_ratio=0.15 if helpful_mask else 0.55,
                query_negation_count=1 if helpful_mask else 0,
                query_claim_cue_count=1 if helpful_mask else 0,
            ))

        result = train_mask_gate(
            rows,
            min_train_support=3,
            min_holdout_support=1,
            min_mean_delta=0.001,
            min_positive_examples=4,
        )

        self.assertIsNotNone(result["gate"])
        self.assertGreaterEqual(len(result["accepted"]), 1)
        self.assertEqual(result["gate"]["type"], "linear_classifier")
        self.assertEqual(result["deployment_mode"], "advisory_only")

    def test_main_writes_gate_config_from_multiple_artifacts(self):
        positive_rows = [
            _mask_row(
                f"p{index}",
                0.03,
                baseline_score_margin=0.1,
                baseline_top_score=0.2,
                query_norm=1.0,
                query_token_count=3,
                query_stopword_ratio=0.15,
                query_negation_count=1,
                query_claim_cue_count=1,
            )
            for index in range(8)
        ]
        negative_rows = [
            _mask_row(
                f"n{index}",
                -0.01,
                baseline_score_margin=0.9,
                baseline_top_score=0.95,
                query_norm=1.0,
                query_token_count=12,
                query_stopword_ratio=0.55,
                query_negation_count=0,
                query_claim_cue_count=0,
            )
            for index in range(8)
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            artifact_one = Path(tmpdir) / "artifact-one.json"
            artifact_two = Path(tmpdir) / "artifact-two.json"
            output_path = Path(tmpdir) / "learned-mask-gate.json"
            artifact_one.write_text(json.dumps({"mask_example_rows": positive_rows[:4] + negative_rows[:4]}), encoding="utf-8")
            artifact_two.write_text(json.dumps({"results": {"mask_example_rows": positive_rows[4:] + negative_rows[4:]}}), encoding="utf-8")
            stdout = io.StringIO()

            with (
                patch("sys.argv", [
                    "learned_mask_gate.py",
                    "--artifact", str(artifact_one),
                    "--artifact", str(artifact_two),
                    "--output", str(output_path),
                    "--min-positive-examples", "2",
                    "--min-train-support", "2",
                    "--min-holdout-support", "1",
                ]),
                patch("sys.stdout", stdout),
            ):
                exit_code = main()

            config = json.loads(output_path.read_text(encoding="utf-8"))
            summary = json.loads(stdout.getvalue())

        self.assertEqual(exit_code, 0)
        self.assertEqual(config["artifact_count"], 2)
        self.assertEqual(config["pooled_rows"], 16)
        self.assertEqual(config["policy"], "query_mask_linear_classifier")
        self.assertEqual(summary["artifact_count"], 2)
        self.assertEqual(summary["pooled_rows"], 16)


if __name__ == "__main__":
    unittest.main()
