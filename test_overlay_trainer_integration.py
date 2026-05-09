"""Real-component integration tests for OverlayTrainer.

Uses real tempdir, real CheckpointManager, and real MemoryManager — no mocks for
checkpoint-save or memory-record paths.
"""

from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path

from expectation_comparator import ExpectationComparator, FeatureOverlapRule, MeanActivationRule
from model_scope_artifacts import ArtifactStore
from model_scope_features import SparseFeatureEvent
from model_scope_memory import MemoryManager
from model_scope_runtime import ActivationEvent
from model_scope_trainer import OverlayConfig, OverlayTrainer


def _make_activation() -> ActivationEvent:
    return ActivationEvent(
        schema_version="1.0",
        model_id="test_model",
        layer_id="layer_0",
        token_count=1,
        shape=(1, 1),
        mean_activation=0.0,
        norm_activation=0.0,
        captured_at="2026-01-01T00:00:00+00:00",
        run_id="test",
    )


def _make_sparse_event(features: dict) -> SparseFeatureEvent:
    return SparseFeatureEvent(
        source_activation=_make_activation(),
        feature_source="test",
        features=features,
        feature_count=len(features),
        nonzero_count=len(features),
        extracted_at="2026-01-01T00:00:00+00:00",
    )


def _make_trainer(base: Path, overlay_id: str = "ov_real") -> OverlayTrainer:
    config = OverlayConfig(overlay_id=overlay_id, policy_id="pol_real", learning_rate=0.1, max_epochs=5, patience=3)
    memory = MemoryManager(base_dir=base / "memory")
    artifact_dir = base / "artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    artifact_store = ArtifactStore(base_dir=artifact_dir)
    comparator = ExpectationComparator()
    comparator.add_rule(MeanActivationRule(threshold=0.0))
    comparator.add_rule(FeatureOverlapRule(threshold=0.0))
    checkpoint_dir = str(base / "checkpoints")
    return OverlayTrainer(
        config=config,
        memory=memory,
        comparator=comparator,
        artifact_store=artifact_store,
        checkpoint_dir=checkpoint_dir,
    )


class TestOverlayTrainerRealComponents(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    # ------------------------------------------------------------------
    # Checkpoint-save path: real disk writes
    # ------------------------------------------------------------------

    def test_checkpoint_saved_to_real_disk(self):
        trainer = _make_trainer(self.tmp)
        episodes = [
            ([_make_sparse_event({"a": 0.5, "b": 0.3})], [_make_sparse_event({"a": 0.8, "b": 0.7})]),
        ]
        records = trainer.run_campaign(episodes)
        improving = [r for r in records if r.improved]
        self.assertTrue(len(improving) > 0)
        for r in improving:
            self.assertIsNotNone(r.checkpoint_path)
            self.assertTrue(Path(r.checkpoint_path).exists())

    def test_checkpoint_file_contains_valid_json(self):
        trainer = _make_trainer(self.tmp)
        episodes = [([_make_sparse_event({"x": 0.4})], [_make_sparse_event({"x": 0.9})])]
        records = trainer.run_campaign(episodes)
        improving = [r for r in records if r.improved]
        self.assertTrue(len(improving) > 0)
        ck_path = Path(improving[0].checkpoint_path)
        data = json.loads(ck_path.read_text(encoding="utf-8"))
        self.assertIn("weights", data)
        self.assertIn("epoch", data)
        self.assertIn("overlay_id", data)

    def test_checkpoint_can_be_reloaded(self):
        trainer = _make_trainer(self.tmp, overlay_id="reload_ov")
        episodes = [([_make_sparse_event({"k": 1.0})], [_make_sparse_event({"k": 2.0})])]
        records = trainer.run_campaign(episodes)
        improving = [r for r in records if r.improved]
        self.assertTrue(len(improving) > 0)
        ck_path = improving[0].checkpoint_path
        weights_saved = json.loads(Path(ck_path).read_text(encoding="utf-8"))["weights"]

        trainer2 = _make_trainer(self.tmp, overlay_id="reload_ov_2")
        trainer2.load_weights(ck_path)
        for key, val in weights_saved.items():
            self.assertAlmostEqual(trainer2._weights[key], val, places=10)

    def test_checkpoint_weights_match_trained_weights(self):
        trainer = _make_trainer(self.tmp, overlay_id="wt_match")
        trainer.config.max_epochs = 1
        trainer.config.patience = 10
        episodes = [([_make_sparse_event({"f1": 0.5, "f2": 0.2})], [_make_sparse_event({"f1": 1.0, "f2": 0.8})])]
        records = trainer.run_campaign(episodes)
        improving = [r for r in records if r.improved]
        self.assertTrue(len(improving) > 0)
        ck_data = json.loads(Path(improving[0].checkpoint_path).read_text(encoding="utf-8"))
        for key, val in ck_data["weights"].items():
            self.assertAlmostEqual(val, trainer._weights[key], places=10)

    def test_rollback_resets_weights_and_memory_persists(self):
        trainer = _make_trainer(self.tmp, overlay_id="rb_real")
        inp = [_make_sparse_event({"q": 0.5})]
        tgt = [_make_sparse_event({"q": 1.0})]
        trainer.train_epoch(inp, tgt)
        self.assertTrue(len(trainer._weights) > 0)
        trainer.rollback()
        self.assertEqual(trainer._weights, {})
        # Real MemoryManager must have stored the rollback event
        entry = trainer.memory.working.retrieve("rollback")
        self.assertIsNotNone(entry)

    def test_rollback_after_save_weight_file_unchanged(self):
        trainer = _make_trainer(self.tmp, overlay_id="rb_file")
        episodes = [([_make_sparse_event({"m": 0.3})], [_make_sparse_event({"m": 0.9})])]
        records = trainer.run_campaign(episodes)
        improving = [r for r in records if r.improved]
        self.assertTrue(len(improving) > 0)
        ck_path = Path(improving[0].checkpoint_path)
        content_before = ck_path.read_text(encoding="utf-8")
        trainer.rollback()
        # Checkpoint file on disk is unchanged by rollback (rollback only resets in-memory state)
        self.assertEqual(ck_path.read_text(encoding="utf-8"), content_before)

    # ------------------------------------------------------------------
    # Memory-record path: real MemoryManager writes
    # ------------------------------------------------------------------

    def test_train_epoch_stores_in_real_episodic_memory(self):
        trainer = _make_trainer(self.tmp, overlay_id="ep_real")
        inp = [_make_sparse_event({"g": 0.7})]
        tgt = [_make_sparse_event({"g": 0.9})]
        trainer.train_epoch(inp, tgt)
        entry = trainer.memory.episodic.retrieve("epoch_0")
        self.assertIsNotNone(entry)

    def test_multiple_epochs_each_stored_in_memory(self):
        trainer = _make_trainer(self.tmp, overlay_id="multi_ep")
        inp = [_make_sparse_event({"h": 0.5})]
        tgt = [_make_sparse_event({"h": 0.8})]
        for _ in range(3):
            trainer.train_epoch(inp, tgt)
        for epoch_idx in range(3):
            entry = trainer.memory.episodic.retrieve(f"epoch_{epoch_idx}")
            self.assertIsNotNone(entry, f"Missing memory entry for epoch_{epoch_idx}")

    def test_evaluate_promotion_persists_to_real_persistent_memory(self):
        trainer = _make_trainer(self.tmp, overlay_id="promo_real")
        inp = [_make_sparse_event({"p": 0.4})]
        tgt = [_make_sparse_event({"p": 0.7})]
        trainer.train_epoch(inp, tgt)
        trainer.evaluate_promotion(inp, tgt)
        stored = trainer.memory.persistent.load("promotion_promo_real")
        self.assertIsNotNone(stored)

    # ------------------------------------------------------------------
    # Campaign + checkpoint lifecycle
    # ------------------------------------------------------------------

    def test_run_campaign_checkpoints_exist_on_disk(self):
        trainer = _make_trainer(self.tmp, overlay_id="camp_real")
        trainer.config.max_epochs = 4
        trainer.config.patience = 10
        episodes = [([_make_sparse_event({"a": 0.5})], [_make_sparse_event({"a": 0.9})])]
        records = trainer.run_campaign(episodes)
        improving = [r for r in records if r.improved]
        self.assertTrue(len(improving) > 0)
        for r in improving:
            self.assertIsNotNone(r.checkpoint_path)
            self.assertTrue(Path(r.checkpoint_path).exists(), f"Missing checkpoint: {r.checkpoint_path}")

    def test_run_campaign_non_improving_epochs_no_checkpoint_on_disk(self):
        trainer = _make_trainer(self.tmp, overlay_id="ni_real")
        trainer.config.min_improvement = 999.0
        trainer.config.patience = 100
        trainer.config.max_epochs = 3
        episodes = [([_make_sparse_event({"c": 0.5})], [_make_sparse_event({"c": 0.5})])]
        records = trainer.run_campaign(episodes)
        non_improving = [r for r in records if not r.improved]
        for r in non_improving:
            self.assertIsNone(r.checkpoint_path)

    def test_save_and_load_weights_roundtrip_real_file(self):
        trainer = _make_trainer(self.tmp, overlay_id="wt_rt")
        inp = [_make_sparse_event({"s": 0.3})]
        tgt = [_make_sparse_event({"s": 0.9})]
        trainer.train_epoch(inp, tgt)
        weights_before = dict(trainer._weights)

        weights_path = str(self.tmp / "weights_roundtrip.json")
        trainer.save_weights(weights_path)
        self.assertTrue(Path(weights_path).exists())

        trainer2 = _make_trainer(self.tmp, overlay_id="wt_rt_2")
        trainer2.load_weights(weights_path)
        self.assertEqual(trainer2._weights, weights_before)

    def test_checkpoint_dir_created_automatically(self):
        ck_dir = self.tmp / "auto_created_checkpoints"
        self.assertFalse(ck_dir.exists())
        trainer = _make_trainer(self.tmp, overlay_id="auto_dir")
        # Override checkpoint dir to one that does not exist yet
        trainer._checkpoint_dir = ck_dir
        # _save_checkpoint creates the dir on demand
        trainer._weights = {"v": 0.5}
        trainer._save_checkpoint(epoch=0)
        self.assertTrue(ck_dir.exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
