import json
from pathlib import Path
import tempfile
import unittest

from engine_scope import (
    ENGINE_SCOPE_SCHEMA_VERSION,
    build_engine_scope_rows,
    engine_scope_rows_to_evidence_events,
    load_engine_scope_rows,
    summarize_engine_scope_rows,
)


class TestEngineScope(unittest.TestCase):
    def test_build_engine_scope_rows_normalizes_query_and_mask_rows(self):
        rows = build_engine_scope_rows(
            query_attribution_rows=[{
                "task": "SciFact",
                "query_id": "q1",
                "query_text": "claim about vitamin b12",
                "query_token_count": 4,
                "query_char_count": 23,
                "query_stopword_ratio": 0.25,
                "query_numeric_token_count": 1,
                "query_negation_count": 0,
                "query_claim_cue_count": 1,
                "profile": "reform_rrf_v2",
                "delta_ndcg_at_10": 0.02,
                "baseline_rank": 5,
                "candidate_rank": 1,
                "rank_delta": -4,
                "action": "REFORMULATE",
                "reformulation_changed": True,
                "fault_class": "actuator_active_positive",
            }],
            mask_example_rows=[{
                "task": "SciFact",
                "query_id": "q2",
                "query_text": "plain mask query",
                "query_token_count": 3,
                "query_char_count": 16,
                "query_stopword_ratio": 0.0,
                "query_numeric_token_count": 0,
                "query_negation_count": 0,
                "query_claim_cue_count": 0,
                "delta_ndcg_at_10": 0.03,
                "baseline_rank": 4,
                "masked_rank": 1,
                "baseline_score_margin": 0.05,
                "baseline_top_score": 0.6,
                "query_norm": 1.2,
                "gate_applied": True,
                "selected_delta_ndcg_at_10": 0.03,
            }],
            source_family="unit_test",
        )

        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["schema_version"], ENGINE_SCOPE_SCHEMA_VERSION)
        self.assertEqual(rows[0]["row_type"], "query_profile")
        self.assertEqual(rows[0]["source_family"], "unit_test")
        self.assertEqual(rows[0]["baseline_rank"], 5)
        self.assertEqual(rows[0]["candidate_rank"], 1)
        self.assertEqual(rows[1]["row_type"], "mask_probe")
        self.assertTrue(rows[1]["gate_applied"])
        self.assertEqual(rows[1]["selected_delta_ndcg_at_10"], 0.03)

    def test_load_engine_scope_rows_supports_direct_and_legacy_artifacts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            direct_path = Path(tmpdir) / "direct.json"
            legacy_path = Path(tmpdir) / "legacy.json"
            direct_path.write_text(json.dumps({
                "engine_scope_rows": [
                    {
                        "row_type": "query_profile",
                        "source_family": "direct",
                        "_artifact_path": None,
                        "query_id": "q1",
                    }
                ]
            }), encoding="utf-8")
            legacy_path.write_text(json.dumps({
                "query_attribution_rows": [
                    {
                        "task": "SciFact",
                        "query_id": "q2",
                        "profile": "reform_rrf_v2",
                        "delta_ndcg_at_10": 0.1,
                    }
                ],
                "mask_example_rows": [
                    {
                        "task": "SciFact",
                        "query_id": "q3",
                        "delta_ndcg_at_10": 0.2,
                    }
                ],
            }), encoding="utf-8")

            rows = load_engine_scope_rows([direct_path, legacy_path])

        self.assertEqual(len(rows), 3)
        self.assertEqual(rows[0]["schema_version"], ENGINE_SCOPE_SCHEMA_VERSION)
        self.assertEqual(rows[0]["source_family"], "direct")
        self.assertEqual(rows[0]["_artifact_path"], str(direct_path))
        self.assertEqual(rows[1]["source_family"], "legacy_query_attribution")
        self.assertEqual(rows[2]["source_family"], "legacy_mask_probe")
        self.assertIsNotNone(rows[1]["_artifact_path"])

    def test_load_engine_scope_rows_prefers_direct_rows_without_legacy_duplication(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            artifact_path = Path(tmpdir) / "mixed.json"
            artifact_path.write_text(json.dumps({
                "engine_scope_rows": [
                    {
                        "row_type": "query_profile",
                        "source_family": "direct",
                        "query_id": "q1",
                    }
                ],
                "nested": {
                    "query_attribution_rows": [
                        {
                            "task": "SciFact",
                            "query_id": "q1",
                            "profile": "reform_rrf_v2",
                            "delta_ndcg_at_10": 0.1,
                        }
                    ]
                },
            }), encoding="utf-8")

            rows = load_engine_scope_rows([artifact_path])

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["source_family"], "direct")

    def test_summarize_engine_scope_rows_groups_by_type_and_source(self):
        summary = summarize_engine_scope_rows([
            {"row_type": "query_profile", "source_family": "reformulation_collection", "task": "SciFact"},
            {"row_type": "query_profile", "source_family": "reformulation_collection", "task": "SciFact"},
            {"row_type": "mask_probe", "source_family": "mask_validation", "task": "NFCorpus"},
        ])

        self.assertEqual(summary["schema_version"], ENGINE_SCOPE_SCHEMA_VERSION)
        self.assertEqual(summary["row_count"], 3)
        self.assertEqual(summary["source_families"]["reformulation_collection"], 2)
        self.assertEqual(summary["row_types"]["mask_probe"], 1)
        self.assertEqual(summary["tasks"], ["NFCorpus", "SciFact"])

    def test_engine_scope_rows_convert_to_evidence_events(self):
        rows = build_engine_scope_rows(
            query_attribution_rows=[{
                "task": "SciFact",
                "query_id": "q1",
                "profile": "reform_rrf_v2",
                "delta_ndcg_at_10": -0.02,
                "action": "REFORMULATE",
                "fault_class": "actuator_active_negative",
                "decision": "gate_block",
            }],
            source_family="unit_test",
        )

        events = engine_scope_rows_to_evidence_events(rows)

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["surface"], "engine_scope")
        self.assertEqual(events[0]["decision"], "gate_block")
        self.assertEqual(events[0]["outcome_metrics"]["delta_ndcg_at_10"], -0.02)


if __name__ == "__main__":
    unittest.main()
