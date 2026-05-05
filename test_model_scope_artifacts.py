import tempfile
import unittest
from pathlib import Path

from model_scope_artifacts import (
    MODEL_SCOPE_ARTIFACT_SCHEMA_VERSION,
    build_model_scope_artifact,
    load_model_scope_artifact,
    model_scope_artifact_to_evidence_event,
    summarize_model_scope_artifact,
    write_model_scope_artifact,
)


class TestModelScopeArtifacts(unittest.TestCase):
    def test_round_trip_and_summary(self):
        artifact = build_model_scope_artifact(
            runtime={
                "model_name": "Qwen/Qwen3.5-2B",
                "device": "cpu",
            },
            capture={
                "metadata": {"query_id": "q1"},
                "token_count": 4,
                "captured_layer_count": 2,
                "layer_indices": [0, 1],
                "observations": [
                    {"layer_index": 0},
                    {"layer_index": 1},
                ],
            },
            trace={"run_id": "unit"},
            evidence={"event_id": "evt_existing"},
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = write_model_scope_artifact(Path(tmpdir) / "artifact.json", artifact)
            loaded = load_model_scope_artifact(path)

        self.assertEqual(loaded["schema_version"], MODEL_SCOPE_ARTIFACT_SCHEMA_VERSION)
        self.assertEqual(loaded["runtime"]["model_name"], "Qwen/Qwen3.5-2B")
        summary = summarize_model_scope_artifact(loaded)
        self.assertEqual(summary["captured_layer_count"], 2)
        self.assertEqual(summary["token_count"], 4)
        self.assertEqual(summary["layer_indices"], [0, 1])
        self.assertEqual(summary["evidence_event_id"], "evt_existing")

        event = model_scope_artifact_to_evidence_event(loaded)
        self.assertEqual(event["surface"], "model_scope")
        self.assertEqual(event["query_id"], "q1")
        self.assertEqual(event["trace"]["run_id"], "unit")


if __name__ == "__main__":
    unittest.main()
