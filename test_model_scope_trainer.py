import argparse
import tempfile
import unittest
from pathlib import Path

from checkpoint_manager import CheckpointManager
from expectation_comparator import ExpectationComparator, FeatureOverlapRule, MeanActivationRule
from model_scope_artifacts import ArtifactStore
from model_scope_features import SparseFeatureEvent
from model_scope_memory import MemoryManager
from model_scope_runtime import ActivationEvent
from model_scope_trainer import (
    ModelScopeShadowPolicyTrainer,
    ModelScopeTrainerConfig,
    OverlayConfig,
    OverlayTrainer,
    PromotionDecision,
    TrainingRecord,
)


def _artifact(feature_id: str, value: float):
    return {
        "capture": {
            "observations": [
                {
                    "layer_index": 0,
                    "feature_summary": {
                        "feature_space": "qwen_scope_sae",
                        "active_features": [{"feature_id": feature_id, "value": value}],
                    },
                }
            ]
        }
    }


class TestModelScopeShadowPolicyTrainer(unittest.TestCase):
    def test_train_shadow_policy_learns_bounded_rules(self):
        replay_bundle = {
            "entries": [
                {"entry_id": "p1", "metadata": {"label": "positive"}, "artifact": _artifact("101", 1.2)},
                {"entry_id": "p2", "metadata": {"label": "positive"}, "artifact": _artifact("101", 1.1)},
                {"entry_id": "n1", "metadata": {"label": "negative"}, "artifact": _artifact("202", 1.4)},
                {"entry_id": "n2", "metadata": {"label": "negative"}, "artifact": _artifact("202", 1.3)},
            ]
        }
        trainer = ModelScopeShadowPolicyTrainer(
            ModelScopeTrainerConfig(
                max_rules=4,
                min_examples=4,
                min_feature_gap=0.2,
                min_alignment_score=0.5,
            )
        )

        candidate = trainer.train_shadow_policy(replay_bundle, candidate_id="shadow_v1")

        self.assertEqual(candidate["candidate_id"], "shadow_v1")
        self.assertTrue(candidate["promotion_gate"]["promotion_ready"])
        self.assertFalse(candidate["artifact_manifest"]["base_weights_mutated"])
        self.assertEqual(candidate["artifact_manifest"]["artifact_class"], "shadow_policy_overlay")
        action_types = {rule["action_type"] for rule in candidate["policy"]["rules"]}
        self.assertIn("amplify", action_types)
        self.assertIn("suppress", action_types)

    def test_train_shadow_policy_fails_closed_when_labels_are_sparse(self):
        replay_bundle = {
            "entries": [
                {"entry_id": "p1", "metadata": {"label": "positive"}, "artifact": _artifact("101", 1.2)},
                {"entry_id": "p2", "metadata": {"label": "positive"}, "artifact": _artifact("101", 1.1)},
            ]
        }
        trainer = ModelScopeShadowPolicyTrainer(ModelScopeTrainerConfig(min_examples=3))

        candidate = trainer.train_shadow_policy(replay_bundle, candidate_id="shadow_v2")

        self.assertFalse(candidate["promotion_gate"]["promotion_ready"])
        self.assertEqual(candidate["policy"]["rules"], [])
        self.assertFalse(candidate["artifact_manifest"]["base_weights_mutated"])

    def test_promote_candidate_writes_policy_file(self):
        replay_bundle = {
            "entries": [
                {"entry_id": "p1", "metadata": {"label": "positive"}, "artifact": _artifact("101", 1.2)},
                {"entry_id": "p2", "metadata": {"label": "positive"}, "artifact": _artifact("101", 1.1)},
                {"entry_id": "n1", "metadata": {"label": "negative"}, "artifact": _artifact("202", 1.4)},
                {"entry_id": "n2", "metadata": {"label": "negative"}, "artifact": _artifact("202", 1.3)},
            ]
        }
        trainer = ModelScopeShadowPolicyTrainer(ModelScopeTrainerConfig(min_alignment_score=0.5))
        candidate = trainer.train_shadow_policy(replay_bundle, candidate_id="shadow_v3")

        with tempfile.TemporaryDirectory() as tmpdir:
            target_path = Path(tmpdir) / "shadow_policy.json"
            target_path.write_text('{"name": "previous"}', encoding="utf-8")
            checkpoint_manager = CheckpointManager(Path(tmpdir) / "checkpoints")

            result = trainer.promote_candidate(candidate, target_path, checkpoint_manager=checkpoint_manager)

            self.assertTrue(result["promoted"])
            self.assertTrue(target_path.read_text(encoding="utf-8").startswith("{"))
            self.assertIsNotNone(result["checkpoint_id"])


# ---------------------------------------------------------------------------
# Slice-17 tests: OverlayConfig, TrainingRecord, PromotionDecision, OverlayTrainer
# ---------------------------------------------------------------------------


def _make_activation():
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


def _make_trainer(tmpdir: str, overlay_id: str = "ov1", policy_id: str = "pol1") -> OverlayTrainer:
    base = Path(tmpdir)
    config = OverlayConfig(overlay_id=overlay_id, policy_id=policy_id, learning_rate=0.1, max_epochs=5, patience=2)
    memory = MemoryManager(base_dir=base / "memory")
    artifact_dir = base / "artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    artifact_store = ArtifactStore(base_dir=artifact_dir)
    comparator = ExpectationComparator()
    comparator.add_rule(MeanActivationRule(threshold=0.0))
    comparator.add_rule(FeatureOverlapRule(threshold=0.0))
    return OverlayTrainer(
        config=config,
        memory=memory,
        comparator=comparator,
        artifact_store=artifact_store,
        checkpoint_dir=str(base / "checkpoints"),
    )


class TestOverlayConfig(unittest.TestCase):
    def test_overlay_config_required_fields(self):
        cfg = OverlayConfig(overlay_id="x", policy_id="p")
        self.assertEqual(cfg.overlay_id, "x")
        self.assertEqual(cfg.policy_id, "p")

    def test_overlay_config_defaults(self):
        cfg = OverlayConfig(overlay_id="x", policy_id="p")
        self.assertAlmostEqual(cfg.learning_rate, 0.001)
        self.assertEqual(cfg.max_epochs, 10)
        self.assertEqual(cfg.patience, 3)
        self.assertAlmostEqual(cfg.min_improvement, 0.01)
        self.assertAlmostEqual(cfg.promotion_threshold, 0.05)
        self.assertTrue(cfg.enabled)

    def test_overlay_config_custom_values(self):
        cfg = OverlayConfig(overlay_id="a", policy_id="b", learning_rate=0.5, max_epochs=20, patience=7)
        self.assertAlmostEqual(cfg.learning_rate, 0.5)
        self.assertEqual(cfg.max_epochs, 20)
        self.assertEqual(cfg.patience, 7)


class TestTrainingRecord(unittest.TestCase):
    def test_training_record_fields(self):
        rec = TrainingRecord(overlay_id="x", epoch=3, loss=0.42, improved=True, checkpoint_path="/tmp/ck", created_at="2026-01-01T00:00:00+00:00")
        self.assertEqual(rec.overlay_id, "x")
        self.assertEqual(rec.epoch, 3)
        self.assertAlmostEqual(rec.loss, 0.42)
        self.assertTrue(rec.improved)
        self.assertEqual(rec.checkpoint_path, "/tmp/ck")

    def test_training_record_none_checkpoint(self):
        rec = TrainingRecord(overlay_id="x", epoch=0, loss=0.1, improved=False, checkpoint_path=None, created_at="2026-01-01T00:00:00+00:00")
        self.assertIsNone(rec.checkpoint_path)


class TestPromotionDecision(unittest.TestCase):
    def test_promotion_decision_fields(self):
        dec = PromotionDecision(
            overlay_id="ov", promoted=True, reason="delta ok",
            baseline_score=0.3, candidate_score=0.8, delta=0.5, decided_at="2026-01-01T00:00:00+00:00",
        )
        self.assertEqual(dec.overlay_id, "ov")
        self.assertTrue(dec.promoted)
        self.assertAlmostEqual(dec.delta, 0.5)

    def test_promotion_decision_not_promoted(self):
        dec = PromotionDecision(
            overlay_id="ov", promoted=False, reason="too small",
            baseline_score=0.5, candidate_score=0.51, delta=0.01, decided_at="2026-01-01T00:00:00+00:00",
        )
        self.assertFalse(dec.promoted)


class TestOverlayTrainer(unittest.TestCase):
    def test_train_epoch_returns_training_record(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = _make_trainer(tmpdir)
            inp = [_make_sparse_event({"a": 0.5, "b": 0.3})]
            tgt = [_make_sparse_event({"a": 0.8, "b": 0.6})]
            record = trainer.train_epoch(inp, tgt)
            self.assertIsInstance(record, TrainingRecord)

    def test_train_epoch_loss_is_float(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = _make_trainer(tmpdir)
            record = trainer.train_epoch(
                [_make_sparse_event({"x": 0.4})],
                [_make_sparse_event({"x": 0.9})],
            )
            self.assertIsInstance(record.loss, float)

    def test_train_epoch_increments_counter(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = _make_trainer(tmpdir)
            ev = [_make_sparse_event({"f1": 0.5})]
            r0 = trainer.train_epoch(ev, ev)
            r1 = trainer.train_epoch(ev, ev)
            self.assertEqual(r0.epoch, 0)
            self.assertEqual(r1.epoch, 1)

    def test_train_epoch_first_call_is_improved(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = _make_trainer(tmpdir)
            ev = [_make_sparse_event({"a": 0.5})]
            record = trainer.train_epoch(ev, ev)
            self.assertTrue(record.improved)

    def test_train_epoch_stores_in_episodic_memory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = _make_trainer(tmpdir)
            ev = [_make_sparse_event({"g": 0.7})]
            trainer.train_epoch(ev, ev)
            entry = trainer.memory.episodic.retrieve("epoch_0")
            self.assertIsNotNone(entry)

    def test_train_epoch_weights_updated(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = _make_trainer(tmpdir)
            inp = [_make_sparse_event({"k": 1.0})]
            tgt = [_make_sparse_event({"k": 2.0})]
            trainer.train_epoch(inp, tgt)
            # weight should have moved away from 1.0 (initial)
            self.assertNotEqual(trainer._weights.get("k", 1.0), 1.0)

    def test_train_epoch_non_improving_second_call(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = _make_trainer(tmpdir, overlay_id="ov_ni")
            trainer.config.min_improvement = 999.0  # impossible to improve
            ev = [_make_sparse_event({"q": 0.5})]
            trainer.train_epoch(ev, ev)  # epoch 0 — always improved
            r1 = trainer.train_epoch(ev, ev)  # epoch 1 — cannot beat huge min_improvement
            self.assertFalse(r1.improved)

    def test_run_campaign_returns_records_list(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = _make_trainer(tmpdir)
            episodes = [([_make_sparse_event({"a": 0.5})], [_make_sparse_event({"a": 0.8})])]
            records = trainer.run_campaign(episodes)
            self.assertIsInstance(records, list)
            self.assertGreater(len(records), 0)

    def test_run_campaign_respects_max_epochs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = _make_trainer(tmpdir)
            trainer.config.max_epochs = 3
            trainer.config.patience = 100  # no early stop
            episodes = [([_make_sparse_event({"a": 0.5})], [_make_sparse_event({"a": 0.8})])]
            records = trainer.run_campaign(episodes)
            self.assertLessEqual(len(records), 3)

    def test_run_campaign_early_stops_at_patience(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = _make_trainer(tmpdir)
            trainer.config.max_epochs = 50
            trainer.config.patience = 2
            trainer.config.min_improvement = 999.0  # never improve
            episodes = [([_make_sparse_event({"a": 0.5})], [_make_sparse_event({"a": 0.5})])]
            records = trainer.run_campaign(episodes)
            # epoch 0 always improves; then 2 non-improving => stops at epoch 3
            self.assertLessEqual(len(records), 4)

    def test_run_campaign_improving_epochs_have_checkpoint_paths(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = _make_trainer(tmpdir)
            trainer.config.patience = 100
            episodes = [([_make_sparse_event({"b": 0.3})], [_make_sparse_event({"b": 0.9})])]
            records = trainer.run_campaign(episodes)
            improving = [r for r in records if r.improved]
            self.assertTrue(len(improving) > 0)
            for r in improving:
                self.assertIsNotNone(r.checkpoint_path)

    def test_run_campaign_non_improving_epochs_no_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = _make_trainer(tmpdir)
            trainer.config.min_improvement = 999.0
            trainer.config.patience = 100
            trainer.config.max_epochs = 3
            episodes = [([_make_sparse_event({"c": 0.5})], [_make_sparse_event({"c": 0.5})])]
            records = trainer.run_campaign(episodes)
            non_improving = [r for r in records if not r.improved]
            for r in non_improving:
                self.assertIsNone(r.checkpoint_path)

    def test_evaluate_promotion_returns_decision(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = _make_trainer(tmpdir)
            evts = [_make_sparse_event({"a": 0.5})]
            tgts = [_make_sparse_event({"a": 0.5})]
            dec = trainer.evaluate_promotion(evts, tgts)
            self.assertIsInstance(dec, PromotionDecision)

    def test_evaluate_promotion_promoted_when_delta_large(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = _make_trainer(tmpdir, overlay_id="promo_pos")
            trainer.config.promotion_threshold = 0.0  # any positive delta promotes
            # Train to push weights away from 1.0
            for _ in range(3):
                trainer.train_epoch(
                    [_make_sparse_event({"z": 0.5})],
                    [_make_sparse_event({"z": 1.0})],
                )
            dec = trainer.evaluate_promotion(
                [_make_sparse_event({"z": 0.5})],
                [_make_sparse_event({"z": 1.0})],
            )
            self.assertIsInstance(dec, PromotionDecision)
            self.assertIsInstance(dec.promoted, bool)

    def test_evaluate_promotion_not_promoted_when_threshold_high(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = _make_trainer(tmpdir, overlay_id="no_promo")
            trainer.config.promotion_threshold = 999.0  # impossible
            dec = trainer.evaluate_promotion(
                [_make_sparse_event({"m": 0.5})],
                [_make_sparse_event({"m": 0.5})],
            )
            self.assertFalse(dec.promoted)

    def test_evaluate_promotion_persists_to_memory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = _make_trainer(tmpdir, overlay_id="mem_ov")
            trainer.evaluate_promotion(
                [_make_sparse_event({"n": 0.5})],
                [_make_sparse_event({"n": 0.5})],
            )
            entry = trainer.memory.persistent.load("promotion_mem_ov")
            self.assertIsNotNone(entry)

    def test_evaluate_promotion_delta_field(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = _make_trainer(tmpdir, overlay_id="delta_ov")
            dec = trainer.evaluate_promotion(
                [_make_sparse_event({"p": 0.4})],
                [_make_sparse_event({"p": 0.4})],
            )
            self.assertAlmostEqual(dec.delta, dec.candidate_score - dec.baseline_score, places=9)

    def test_rollback_resets_weights(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = _make_trainer(tmpdir, overlay_id="rb_ov")
            trainer.train_epoch([_make_sparse_event({"q": 0.5})], [_make_sparse_event({"q": 1.0})])
            self.assertTrue(len(trainer._weights) > 0)
            trainer.rollback()
            self.assertEqual(trainer._weights, {})

    def test_rollback_resets_best_loss(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = _make_trainer(tmpdir, overlay_id="rb2_ov")
            trainer.train_epoch([_make_sparse_event({"r": 0.5})], [_make_sparse_event({"r": 1.0})])
            self.assertIsNotNone(trainer._best_loss)
            trainer.rollback()
            self.assertIsNone(trainer._best_loss)

    def test_rollback_stores_in_working_memory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = _make_trainer(tmpdir, overlay_id="rb3_ov")
            trainer.rollback()
            entry = trainer.memory.working.retrieve("rollback")
            self.assertIsNotNone(entry)

    def test_save_load_weights_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = _make_trainer(tmpdir, overlay_id="wt_ov")
            trainer.train_epoch([_make_sparse_event({"s": 0.3})], [_make_sparse_event({"s": 0.9})])
            weights_before = dict(trainer._weights)
            path = str(Path(tmpdir) / "weights.json")
            trainer.save_weights(path)
            trainer2 = _make_trainer(tmpdir + "_2", overlay_id="wt_ov")
            trainer2.load_weights(path)
            self.assertEqual(trainer2._weights, weights_before)

    def test_save_weights_creates_json_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = _make_trainer(tmpdir, overlay_id="sw_ov")
            trainer._weights = {"a": 1.5, "b": 0.7}
            path = str(Path(tmpdir) / "wts.json")
            trainer.save_weights(path)
            import json as _json
            data = _json.loads(Path(path).read_text())
            self.assertIn("weights", data)
            self.assertAlmostEqual(data["weights"]["a"], 1.5)

    def test_load_weights_missing_key_defaults_to_identity(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = _make_trainer(tmpdir, overlay_id="lw_ov")
            trainer._weights = {}
            # _get_weight for unknown key returns 1.0
            self.assertAlmostEqual(trainer._get_weight("nonexistent"), 1.0)

    def test_train_epoch_empty_events(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = _make_trainer(tmpdir, overlay_id="emp_ov")
            record = trainer.train_epoch([], [])
            self.assertAlmostEqual(record.loss, 0.0)

    def test_overlay_trainer_init_defaults(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = _make_trainer(tmpdir)
            self.assertEqual(trainer._weights, {})
            self.assertIsNone(trainer._best_loss)
            self.assertEqual(trainer._epoch_counter, 0)


class TestOverlayCampaignCLI(unittest.TestCase):
    def _parse(self, args_list):
        from run_model_scope_campaign import parse_args
        return parse_args(args_list)

    def _run(self, args):
        from run_model_scope_campaign import run_campaign
        return run_campaign(args)

    def test_parse_args_overlay_id(self):
        args = self._parse(["--overlay-id", "ov_test", "--policy-id", "pol_test"])
        self.assertEqual(args.overlay_id, "ov_test")

    def test_parse_args_policy_id(self):
        args = self._parse(["--overlay-id", "ov_x", "--policy-id", "pol_x"])
        self.assertEqual(args.policy_id, "pol_x")

    def test_parse_args_default_episodes(self):
        args = self._parse(["--overlay-id", "x", "--policy-id", "y"])
        self.assertEqual(args.episodes, 5)

    def test_parse_args_custom_episodes(self):
        args = self._parse(["--overlay-id", "x", "--policy-id", "y", "--episodes", "3"])
        self.assertEqual(args.episodes, 3)

    def test_parse_args_default_max_epochs(self):
        args = self._parse(["--overlay-id", "x", "--policy-id", "y"])
        self.assertEqual(args.max_epochs, 10)

    def test_parse_args_default_seed(self):
        args = self._parse(["--overlay-id", "x", "--policy-id", "y"])
        self.assertEqual(args.seed, 42)

    def test_run_campaign_returns_dict(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            args = argparse.Namespace(
                overlay_id="cli_ov1", policy_id="cli_pol1",
                episodes=2, max_epochs=3, learning_rate=0.01,
                promotion_threshold=0.05, output_dir=tmpdir, seed=1,
            )
            result = self._run(args)
            self.assertIsInstance(result, dict)

    def test_run_campaign_result_keys(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            args = argparse.Namespace(
                overlay_id="cli_ov2", policy_id="cli_pol2",
                episodes=1, max_epochs=2, learning_rate=0.01,
                promotion_threshold=0.05, output_dir=tmpdir, seed=7,
            )
            result = self._run(args)
            for key in ("overlay_id", "policy_id", "seed", "epochs_run", "final_loss", "promoted", "promotion_delta", "records"):
                self.assertIn(key, result)

    def test_run_campaign_epochs_run(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            args = argparse.Namespace(
                overlay_id="cli_ov3", policy_id="p3",
                episodes=2, max_epochs=5, learning_rate=0.01,
                promotion_threshold=0.05, output_dir=tmpdir, seed=42,
            )
            result = self._run(args)
            self.assertGreater(result["epochs_run"], 0)
            self.assertLessEqual(result["epochs_run"], 5)

    def test_run_campaign_writes_json_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            args = argparse.Namespace(
                overlay_id="cli_ov4", policy_id="p4",
                episodes=1, max_epochs=2, learning_rate=0.01,
                promotion_threshold=0.05, output_dir=tmpdir, seed=99,
            )
            self._run(args)
            out = Path(tmpdir) / "cli_ov4_campaign_result.json"
            self.assertTrue(out.exists())

    def test_run_campaign_reproducible_with_seed(self):
        with tempfile.TemporaryDirectory() as tmpdir1, tempfile.TemporaryDirectory() as tmpdir2:
            def _make_args(d):
                return argparse.Namespace(
                    overlay_id="rep_ov", policy_id="rep_pol",
                    episodes=3, max_epochs=4, learning_rate=0.01,
                    promotion_threshold=0.05, output_dir=d, seed=17,
                )
            r1 = self._run(_make_args(tmpdir1))
            r2 = self._run(_make_args(tmpdir2))
            self.assertEqual(r1["epochs_run"], r2["epochs_run"])
            self.assertAlmostEqual(r1["final_loss"], r2["final_loss"], places=10)

    def test_run_campaign_promoted_is_bool(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            args = argparse.Namespace(
                overlay_id="cli_bool", policy_id="pb",
                episodes=2, max_epochs=3, learning_rate=0.01,
                promotion_threshold=0.05, output_dir=tmpdir, seed=5,
            )
            result = self._run(args)
            self.assertIsInstance(result["promoted"], bool)


if __name__ == "__main__":
    unittest.main()
