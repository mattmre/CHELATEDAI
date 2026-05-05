import tempfile
import unittest
from pathlib import Path

from model_scope_memory import MemorySegmentConfig, ModelScopeMemoryStore
from promotion_contract import evaluate_promotion_candidate
from evidence_contract import build_evidence_bundle, build_episode_event


def _artifact(prompt_hash: str, *, feature_id: str = "10", value: float = 1.0):
    return {
        "runtime": {"model_name": "Qwen/Qwen3.5-2B"},
        "capture": {
            "prompt_hash": prompt_hash,
            "token_count": 3,
            "captured_layer_count": 1,
            "layer_indices": [0],
            "metadata": {"query_id": prompt_hash},
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


class TestModelScopeMemoryStore(unittest.TestCase):
    def test_record_replay_promote_and_save(self):
        store = ModelScopeMemoryStore(
            segment_configs=[
                MemorySegmentConfig(name="working", max_entries=1),
                MemorySegmentConfig(name="episode", max_entries=4, allow_promotion=True),
                MemorySegmentConfig(name="expectation", max_entries=2),
                MemorySegmentConfig(name="persistent", max_entries=4),
            ]
        )

        store.store_expectation_profile({"profile_id": "q1", "layers": []})
        event_id = "evt_unit"
        first = store.record_observation(
            _artifact("q1", feature_id="10"),
            query_text="alpha",
            evidence_event_ids=[event_id],
            retention_reason="unit_replay",
            source_lineage=["raw_capture"],
        )
        second = store.record_observation(_artifact("q2", feature_id="20"), query_text="beta")

        self.assertEqual(store.segment_sizes()["working"], 1)
        self.assertEqual(store.segment_sizes()["episode"], 2)
        self.assertEqual(first["query_hash"], "q1")
        self.assertEqual(second["query_hash"], "q2")

        replay = store.build_replay_bundle(segment="episode", include_artifacts=True)
        self.assertEqual(replay["entry_count"], 2)
        first_replay = next(entry for entry in replay["entries"] if entry["query_hash"] == "q1")
        self.assertEqual(len(first_replay["artifact"]["capture"]["observations"]), 1)
        self.assertEqual(first_replay["evidence_event_ids"], [event_id])
        self.assertEqual(first_replay["retention_reason"], "unit_replay")
        self.assertEqual(first_replay["source_lineage"], ["raw_capture"])

        promoted = store.promote_episode(second["episode_entry_id"], reason="unit_test")
        self.assertTrue(promoted["persistent_entry_id"].startswith("persistent_"))
        self.assertEqual(store.segment_sizes()["persistent"], 1)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = store.save(Path(tmpdir) / "memory_snapshot.json")
            loaded = ModelScopeMemoryStore.load(path)

        self.assertEqual(loaded.segment_sizes()["episode"], 2)
        self.assertIsNotNone(loaded.get_expectation_profile("q1"))

    def test_promote_episode_respects_fail_closed_promotion_status(self):
        store = ModelScopeMemoryStore(
            segment_configs=[
                MemorySegmentConfig(name="working", max_entries=4),
                MemorySegmentConfig(name="episode", max_entries=4, allow_promotion=True),
                MemorySegmentConfig(name="persistent", max_entries=4),
            ]
        )
        entry = store.record_observation(_artifact("q1"), query_text="alpha")
        bundle = build_evidence_bundle(
            [build_episode_event(event_type="observe", surface="model_scope", query_id="q1")]
        )
        status = evaluate_promotion_candidate(
            candidate_id=entry["episode_entry_id"],
            evidence_bundle=bundle,
            comparator_report={"passed": True, "score": 1.0},
        )

        result = store.promote_episode(entry["episode_entry_id"], reason="unit_test", promotion_status=status)

        self.assertFalse(result["promoted"])
        self.assertEqual(store.segment_sizes()["persistent"], 0)
        self.assertIn("missing_holdout_report", result["reasons"])


if __name__ == "__main__":
    unittest.main()
