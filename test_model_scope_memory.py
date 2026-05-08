"""Tests for model_scope_memory — Slice-16 typed segmented memory API and ModelScopeMemoryStore."""
from __future__ import annotations

import tempfile
import time
import unittest
from pathlib import Path

from model_scope_memory import (
    EpisodicMemory,
    ExpectationStore,
    MemoryEntry,
    MemoryManager,
    MemorySegmentConfig,
    MemorySegmentType,
    ModelScopeMemoryStore,
    PersistentMemory,
    WorkingMemory,
)
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


# ---------------------------------------------------------------------------
# WorkingMemory
# ---------------------------------------------------------------------------


class TestWorkingMemory(unittest.TestCase):
    def test_store_and_retrieve(self):
        wm = WorkingMemory()
        entry = wm.store("k1", {"x": 1})
        self.assertIsInstance(entry, MemoryEntry)
        self.assertEqual(entry.key, "k1")
        self.assertEqual(entry.value["x"], 1)
        self.assertEqual(entry.segment, MemorySegmentType.WORKING)

    def test_retrieve_returns_most_recent(self):
        wm = WorkingMemory()
        wm.store("k1", {"v": 1})
        wm.store("k1", {"v": 2})
        retrieved = wm.retrieve("k1")
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.value["v"], 2)

    def test_retrieve_missing_returns_none(self):
        wm = WorkingMemory()
        self.assertIsNone(wm.retrieve("no_such_key"))

    def test_evicts_oldest_when_full(self):
        wm = WorkingMemory(max_entries=3)
        for i in range(4):
            wm.store(f"k{i}", {"i": i})
        entries = wm.list_entries()
        self.assertEqual(len(entries), 3)
        keys = [e.key for e in entries]
        self.assertNotIn("k0", keys)
        self.assertIn("k3", keys)

    def test_max_entries_one_keeps_last(self):
        wm = WorkingMemory(max_entries=1)
        wm.store("a", {"v": 1})
        wm.store("b", {"v": 2})
        self.assertEqual(len(wm.list_entries()), 1)
        self.assertEqual(wm.list_entries()[0].key, "b")

    def test_evict_expired_no_expiry_returns_zero(self):
        wm = WorkingMemory()
        wm.store("k", {"x": 1})
        self.assertEqual(wm.evict_expired(), 0)
        self.assertEqual(len(wm.list_entries()), 1)

    def test_clear_returns_count_and_empties(self):
        wm = WorkingMemory()
        wm.store("a", {})
        wm.store("b", {})
        count = wm.clear()
        self.assertEqual(count, 2)
        self.assertEqual(len(wm.list_entries()), 0)

    def test_list_entries_preserves_insertion_order(self):
        wm = WorkingMemory()
        for ch in ("x", "y", "z"):
            wm.store(ch, {})
        keys = [e.key for e in wm.list_entries()]
        self.assertEqual(keys, ["x", "y", "z"])

    def test_store_tags_attached(self):
        wm = WorkingMemory()
        entry = wm.store("k", {}, tags=["t1", "t2"])
        self.assertIn("t1", entry.tags)
        self.assertIn("t2", entry.tags)

    def test_invalid_max_entries_raises(self):
        with self.assertRaises(ValueError):
            WorkingMemory(max_entries=0)


# ---------------------------------------------------------------------------
# EpisodicMemory
# ---------------------------------------------------------------------------


class TestEpisodicMemory(unittest.TestCase):
    def test_store_and_retrieve(self):
        em = EpisodicMemory()
        em.store("ep1", {"a": 1})
        entry = em.retrieve("ep1")
        self.assertIsNotNone(entry)
        self.assertEqual(entry.value["a"], 1)
        self.assertEqual(entry.segment, MemorySegmentType.EPISODIC)

    def test_retrieve_missing_returns_none(self):
        em = EpisodicMemory()
        self.assertIsNone(em.retrieve("ghost"))

    def test_retrieve_respects_ttl_expiry(self):
        em = EpisodicMemory()
        em.store("k", {"v": 1}, ttl_seconds=0.01)
        time.sleep(0.05)
        self.assertIsNone(em.retrieve("k"))

    def test_evict_expired_removes_stale(self):
        em = EpisodicMemory()
        em.store("stale", {}, ttl_seconds=0.01)
        em.store("fresh", {})
        time.sleep(0.05)
        evicted = em.evict_expired()
        self.assertEqual(evicted, 1)
        self.assertIsNone(em.retrieve("stale"))
        self.assertIsNotNone(em.retrieve("fresh"))

    def test_query_by_tag_returns_matches(self):
        em = EpisodicMemory()
        em.store("e1", {}, tags=["run-1", "alpha"])
        em.store("e2", {}, tags=["run-1", "beta"])
        em.store("e3", {}, tags=["run-2"])
        matches = em.query_by_tag("run-1")
        self.assertEqual(len(matches), 2)

    def test_query_by_tag_excludes_expired(self):
        em = EpisodicMemory()
        em.store("e1", {}, tags=["ep"], ttl_seconds=0.01)
        em.store("e2", {}, tags=["ep"])
        time.sleep(0.05)
        matches = em.query_by_tag("ep")
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0].key, "e2")

    def test_replay_bundle_returns_tagged_entries(self):
        em = EpisodicMemory()
        em.store("obs1", {"n": 1}, tags=["ep-A"])
        em.store("obs2", {"n": 2}, tags=["ep-A"])
        em.store("other", {"n": 3}, tags=["ep-B"])
        bundle = em.replay_bundle("ep-A")
        self.assertEqual(len(bundle), 2)
        keys = {e.key for e in bundle}
        self.assertIn("obs1", keys)
        self.assertIn("obs2", keys)

    def test_replay_bundle_excludes_expired(self):
        em = EpisodicMemory()
        em.store("x", {}, tags=["ep"], ttl_seconds=0.01)
        em.store("y", {}, tags=["ep"])
        time.sleep(0.05)
        bundle = em.replay_bundle("ep")
        self.assertEqual(len(bundle), 1)
        self.assertEqual(bundle[0].key, "y")

    def test_replay_bundle_empty_when_no_match(self):
        em = EpisodicMemory()
        self.assertEqual(em.replay_bundle("nonexistent"), [])

    def test_entry_id_is_unique(self):
        em = EpisodicMemory()
        a = em.store("a", {})
        b = em.store("b", {})
        self.assertNotEqual(a.entry_id, b.entry_id)


# ---------------------------------------------------------------------------
# ExpectationStore
# ---------------------------------------------------------------------------


class TestExpectationStore(unittest.TestCase):
    def test_set_and_get_expectation(self):
        es = ExpectationStore()
        entry = es.set_expectation("model-v1", {"mean": 0.5})
        self.assertIsInstance(entry, MemoryEntry)
        self.assertEqual(entry.segment, MemorySegmentType.EXPECTATION)
        loaded = es.get_expectation("model-v1")
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded.value["baseline"]["mean"], 0.5)

    def test_get_missing_returns_none(self):
        es = ExpectationStore()
        self.assertIsNone(es.get_expectation("unknown"))

    def test_list_expectations(self):
        es = ExpectationStore()
        es.set_expectation("k1", {"a": 1})
        es.set_expectation("k2", {"b": 2})
        listed = es.list_expectations()
        self.assertEqual(len(listed), 2)

    def test_overwrite_updates_entry(self):
        es = ExpectationStore()
        es.set_expectation("k", {"v": 1})
        es.set_expectation("k", {"v": 99})
        entry = es.get_expectation("k")
        self.assertEqual(entry.value["baseline"]["v"], 99)

    def test_threshold_stored_in_value(self):
        es = ExpectationStore()
        entry = es.set_expectation("k", {}, threshold=0.25)
        self.assertAlmostEqual(entry.value["threshold"], 0.25)

    def test_expectation_has_expectation_tag(self):
        es = ExpectationStore()
        entry = es.set_expectation("k", {})
        self.assertIn("expectation", entry.tags)


# ---------------------------------------------------------------------------
# PersistentMemory
# ---------------------------------------------------------------------------


class TestPersistentMemory(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.base_dir = Path(self._tmpdir.name)

    def tearDown(self):
        self._tmpdir.cleanup()

    def test_save_and_load(self):
        pm = PersistentMemory(self.base_dir)
        pm.save("alpha", {"score": 42}, tags=["t1"])
        loaded = pm.load("alpha")
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded.value["score"], 42)
        self.assertIn("t1", loaded.tags)
        self.assertEqual(loaded.segment, MemorySegmentType.PERSISTENT)

    def test_load_missing_returns_none(self):
        pm = PersistentMemory(self.base_dir)
        self.assertIsNone(pm.load("no_such_key"))

    def test_list_keys(self):
        pm = PersistentMemory(self.base_dir)
        pm.save("x", {})
        pm.save("y", {})
        keys = pm.list_keys()
        self.assertIn("x", keys)
        self.assertIn("y", keys)

    def test_delete_returns_true_and_removes(self):
        pm = PersistentMemory(self.base_dir)
        pm.save("to_del", {})
        self.assertTrue(pm.delete("to_del"))
        self.assertIsNone(pm.load("to_del"))
        self.assertNotIn("to_del", pm.list_keys())

    def test_delete_missing_returns_false(self):
        pm = PersistentMemory(self.base_dir)
        self.assertFalse(pm.delete("ghost"))

    def test_save_overwrites_existing(self):
        pm = PersistentMemory(self.base_dir)
        pm.save("k", {"v": 1})
        pm.save("k", {"v": 2})
        loaded = pm.load("k")
        self.assertEqual(loaded.value["v"], 2)

    def test_key_path_sanitises_slashes(self):
        pm = PersistentMemory(self.base_dir)
        pm.save("a/b/c", {"ok": True})
        loaded = pm.load("a/b/c")
        self.assertIsNotNone(loaded)
        self.assertTrue(loaded.value["ok"])


# ---------------------------------------------------------------------------
# MemoryManager
# ---------------------------------------------------------------------------


class TestMemoryManager(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.base_dir = Path(self._tmpdir.name)

    def tearDown(self):
        self._tmpdir.cleanup()

    def test_snapshot_counts(self):
        mm = MemoryManager(base_dir=self.base_dir)
        mm.working.store("w1", {})
        mm.episodic.store("e1", {})
        mm.expectations.set_expectation("ex1", {})
        mm.persistent.save("p1", {})
        snap = mm.snapshot()
        self.assertEqual(snap["working"], 1)
        self.assertEqual(snap["episodic"], 1)
        self.assertEqual(snap["expectation"], 1)
        self.assertEqual(snap["persistent"], 1)

    def test_promote_to_persistent_copies_entry(self):
        mm = MemoryManager(base_dir=self.base_dir)
        entry = mm.working.store("promo_key", {"data": "hello"}, tags=["src"])
        promoted = mm.promote_to_persistent(entry)
        self.assertIsInstance(promoted, MemoryEntry)
        self.assertEqual(promoted.key, "promo_key")
        self.assertEqual(promoted.value["data"], "hello")
        loaded = mm.persistent.load("promo_key")
        self.assertIsNotNone(loaded)

    def test_working_max_entries_param(self):
        mm = MemoryManager(base_dir=self.base_dir, working_max_entries=2)
        for i in range(3):
            mm.working.store(f"k{i}", {})
        self.assertEqual(len(mm.working.list_entries()), 2)

    def test_snapshot_empty_all_zeros(self):
        mm = MemoryManager(base_dir=self.base_dir)
        snap = mm.snapshot()
        for v in snap.values():
            self.assertEqual(v, 0)


# ---------------------------------------------------------------------------
# Legacy ModelScopeMemoryStore tests (preserved)
# ---------------------------------------------------------------------------


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

        result = store.promote_episode(
            entry["episode_entry_id"], reason="unit_test", promotion_status=status
        )

        self.assertFalse(result["promoted"])
        self.assertEqual(store.segment_sizes()["persistent"], 0)
        self.assertIn("missing_holdout_report", result["reasons"])


if __name__ == "__main__":
    unittest.main()

