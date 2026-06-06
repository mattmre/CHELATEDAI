from __future__ import annotations

import os
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from model_scope_features import SparseFeatureEvent
from model_scope_runtime import ActivationEvent
from model_scope_steering import (
    InterventionRecord,
    ModelScopeShadowSteerer,
    SteeringActuator,
    intervention_record_from_dict,
    intervention_record_to_dict,
    rollback_feature_event,
)
from steering_policy import (
    ModelScopeSteeringPolicy,
    PolicyRegistry,
    SteeringMode,
    SteeringPolicyConfig,
)


# ---------------------------------------------------------------------------
# Shared test fixtures
# ---------------------------------------------------------------------------


def _make_activation(run_id: str = "run_test_01") -> ActivationEvent:
    return ActivationEvent(
        schema_version="1.0",
        model_id="test_model",
        layer_id="layer.0",
        token_count=4,
        shape=(1, 4, 64),
        mean_activation=0.05,
        norm_activation=0.12,
        captured_at=datetime.now(timezone.utc).isoformat(),
        run_id=run_id,
    )


def _make_feature_event(features: dict = None) -> SparseFeatureEvent:
    return SparseFeatureEvent(
        schema_version="1.0",
        source_activation=_make_activation(),
        feature_source="raw_stats",
        features=dict(features) if features is not None else {"f0": 1.0, "f1": 2.0, "f2": 3.0},
        feature_count=3,
        nonzero_count=3,
        extracted_at=datetime.now(timezone.utc).isoformat(),
    )


def _make_policy(
    name: str = "p",
    mode: SteeringMode = SteeringMode.SHADOW,
    target_features: list = None,
    scale_factor: float = 1.0,
    max_interventions: int = 0,
) -> SteeringPolicyConfig:
    return SteeringPolicyConfig(
        name=name,
        mode=mode,
        target_features=target_features if target_features is not None else ["f0", "f1"],
        scale_factor=scale_factor,
        max_interventions=max_interventions,
    )


def _single_actuator(
    mode: SteeringMode = SteeringMode.SHADOW,
    scale_factor: float = 1.0,
    target_features: list = None,
    max_interventions: int = 0,
    max_total: int = 0,
):
    config = _make_policy(mode=mode, scale_factor=scale_factor, target_features=target_features, max_interventions=max_interventions)
    registry = PolicyRegistry()
    registry.register(config)
    actuator = SteeringActuator(registry, max_total_interventions=max_total)
    return config, actuator


# ---------------------------------------------------------------------------
# Existing shadow-steerer test (preserved)
# ---------------------------------------------------------------------------


class TestModelScopeShadowSteerer(unittest.TestCase):
    def test_shadow_policy_matches_features_without_applying_runtime_edits(self):
        steerer = ModelScopeShadowSteerer(
            ModelScopeSteeringPolicy.from_mapping(
                {
                    "name": "shadow_qwen_scope_test",
                    "deployment_mode": "shadow_mode",
                    "feature_space": "qwen_scope_sae",
                    "rules": [
                        {
                            "feature_id": "42",
                            "min_value": 0.8,
                            "layer_index": 3,
                            "action_type": "suppress",
                            "strength": 0.4,
                        }
                    ],
                }
            )
        )
        artifact = {
            "capture": {
                "observations": [
                    {
                        "layer_index": 3,
                        "feature_summary": {
                            "feature_space": "qwen_scope_sae",
                            "active_features": [
                                {"feature_id": 42, "value": 1.2},
                                {"feature_id": 7, "value": 0.3},
                            ],
                        },
                    }
                ]
            }
        }

        result = steerer.evaluate_capture(artifact)

        self.assertEqual(result["policy_name"], "shadow_qwen_scope_test")
        self.assertEqual(result["matched_rule_count"], 1)
        self.assertFalse(result["runtime_applied"])
        self.assertEqual(result["recommended_actions"][0]["feature_id"], "42")


class TestModelScopeShadowSteererResearchSeam(unittest.TestCase):
    def setUp(self) -> None:
        os.environ.pop("CHELATED_SHIM_RESEARCH", None)
        os.environ.pop("CHELATED_SHIM_PROMOTED", None)

    def tearDown(self) -> None:
        os.environ.pop("CHELATED_SHIM_RESEARCH", None)
        os.environ.pop("CHELATED_SHIM_PROMOTED", None)

    @staticmethod
    def _artifact(match: bool = True) -> dict:
        feature_id = "42" if match else "7"
        feature_value = 1.2 if match else 0.3
        return {
            "capture": {
                "observations": [
                    {
                        "layer_index": 3,
                        "feature_summary": {
                            "feature_space": "qwen_scope_sae",
                            "active_features": [
                                {"feature_id": feature_id, "value": feature_value},
                            ],
                        },
                    }
                ]
            }
        }

    @staticmethod
    def _steerer() -> ModelScopeShadowSteerer:
        return ModelScopeShadowSteerer(
            ModelScopeSteeringPolicy.from_mapping(
                {
                    "name": "shadow_qwen_scope_test",
                    "deployment_mode": "shadow_mode",
                    "feature_space": "qwen_scope_sae",
                    "rules": [
                        {
                            "feature_id": "42",
                            "min_value": 0.8,
                            "layer_index": 3,
                            "action_type": "suppress",
                            "strength": 0.4,
                        }
                    ],
                }
            )
        )

    def test_research_meta_emits_when_research_enabled(self) -> None:
        os.environ["CHELATED_SHIM_RESEARCH"] = "1"
        steerer = self._steerer()
        artifact = self._artifact(match=True)
        result = steerer.evaluate_capture(artifact)
        meta = steerer.get_last_research_shim_meta()
        self.assertIsNotNone(meta)
        self.assertTrue(meta.get("research_shim_guard"))
        self.assertEqual(meta.get("sip_seam"), "ModelScopeShadowSteerer.evaluate_capture")
        self.assertEqual(meta.get("research_stall_count"), 0)
        self.assertEqual(result.get("research_shim_meta"), meta)
        self.assertEqual(result["matched_rule_count"], 1)

    def test_research_meta_stall_count_increments_when_no_match(self) -> None:
        os.environ["CHELATED_SHIM_RESEARCH"] = "1"
        steerer = self._steerer()
        steerer.evaluate_capture(self._artifact(match=False))
        steerer.evaluate_capture(self._artifact(match=False))
        meta = steerer.get_last_research_shim_meta()
        self.assertIsNotNone(meta)
        self.assertEqual(meta.get("research_stall_count"), 2)

    def test_research_meta_absent_when_disabled(self) -> None:
        steerer = self._steerer()
        artifact = self._artifact(match=True)
        result = steerer.evaluate_capture(artifact)
        self.assertIsNone(steerer.get_last_research_shim_meta())
        self.assertIsNone(result.get("research_shim_meta"))

    def test_research_meta_includes_promoted_sip_apply_when_promoted_enabled(self) -> None:
        os.environ["CHELATED_SHIM_RESEARCH"] = "1"
        os.environ["CHELATED_SHIM_PROMOTED"] = "1"
        steerer = self._steerer()
        artifact = self._artifact(match=True)
        result = steerer.evaluate_capture(artifact)
        meta = steerer.get_last_research_shim_meta()
        self.assertIsNotNone(meta)
        self.assertIn("promoted_sip_apply", meta)
        self.assertIsInstance(meta["promoted_sip_apply"], dict)
        self.assertEqual(result["research_shim_meta"], meta)


# ---------------------------------------------------------------------------
# InterventionRecord construction
# ---------------------------------------------------------------------------


class TestInterventionRecord(unittest.TestCase):
    def test_default_schema_version(self):
        record = InterventionRecord(policy_id="p", policy_name="n")
        self.assertEqual(record.schema_version, "1.0")

    def test_record_id_assigned_on_construction(self):
        record = InterventionRecord(policy_id="p", policy_name="n")
        self.assertIsNotNone(record.record_id)
        self.assertGreater(len(record.record_id), 0)

    def test_record_id_unique_across_instances(self):
        r1 = InterventionRecord(policy_id="p", policy_name="n")
        r2 = InterventionRecord(policy_id="p", policy_name="n")
        self.assertNotEqual(r1.record_id, r2.record_id)

    def test_default_applied_is_false(self):
        record = InterventionRecord(policy_id="p", policy_name="n")
        self.assertFalse(record.applied)

    def test_default_decline_reason_is_none(self):
        record = InterventionRecord(policy_id="p", policy_name="n")
        self.assertIsNone(record.decline_reason)

    def test_default_mode_is_shadow(self):
        record = InterventionRecord(policy_id="p", policy_name="n")
        self.assertEqual(record.mode, SteeringMode.SHADOW)


# ---------------------------------------------------------------------------
# SHADOW mode
# ---------------------------------------------------------------------------


class TestSteeringActuatorShadow(unittest.TestCase):
    def test_shadow_applied_is_false(self):
        config, actuator = _single_actuator(SteeringMode.SHADOW)
        event = _make_feature_event()
        _, record = actuator.apply(event, config.policy_id)
        self.assertFalse(record.applied)

    def test_shadow_output_features_identical_to_input(self):
        config, actuator = _single_actuator(SteeringMode.SHADOW)
        event = _make_feature_event({"f0": 1.0, "f1": 2.0})
        out, _ = actuator.apply(event, config.policy_id)
        self.assertEqual(out.features, event.features)

    def test_shadow_does_not_mutate_input_event(self):
        config, actuator = _single_actuator(SteeringMode.SHADOW)
        event = _make_feature_event({"f0": 1.0, "f1": 2.0})
        original = dict(event.features)
        actuator.apply(event, config.policy_id)
        self.assertEqual(event.features, original)

    def test_shadow_decline_reason_is_none(self):
        config, actuator = _single_actuator(SteeringMode.SHADOW)
        event = _make_feature_event()
        _, record = actuator.apply(event, config.policy_id)
        self.assertIsNone(record.decline_reason)

    def test_shadow_features_modified_is_empty(self):
        config, actuator = _single_actuator(SteeringMode.SHADOW)
        event = _make_feature_event()
        _, record = actuator.apply(event, config.policy_id)
        self.assertEqual(record.features_modified, [])

    def test_shadow_returns_new_event_object(self):
        config, actuator = _single_actuator(SteeringMode.SHADOW)
        event = _make_feature_event({"f0": 1.0})
        out, _ = actuator.apply(event, config.policy_id)
        self.assertIsNot(out, event)
        self.assertIsNot(out.features, event.features)


# ---------------------------------------------------------------------------
# SOFT_SCALE mode
# ---------------------------------------------------------------------------


class TestSteeringActuatorSoftScale(unittest.TestCase):
    def test_soft_scale_applied_is_true(self):
        config, actuator = _single_actuator(SteeringMode.SOFT_SCALE, scale_factor=1.5, target_features=["f0"])
        event = _make_feature_event({"f0": 2.0, "f1": 3.0})
        _, record = actuator.apply(event, config.policy_id)
        self.assertTrue(record.applied)

    def test_soft_scale_targeted_feature_is_scaled(self):
        config, actuator = _single_actuator(SteeringMode.SOFT_SCALE, scale_factor=1.5, target_features=["f0"])
        event = _make_feature_event({"f0": 2.0, "f1": 3.0})
        out, _ = actuator.apply(event, config.policy_id)
        self.assertAlmostEqual(out.features["f0"], 3.0)

    def test_soft_scale_non_targeted_feature_unchanged(self):
        config, actuator = _single_actuator(SteeringMode.SOFT_SCALE, scale_factor=1.5, target_features=["f0"])
        event = _make_feature_event({"f0": 2.0, "f1": 3.0})
        out, _ = actuator.apply(event, config.policy_id)
        self.assertAlmostEqual(out.features["f1"], 3.0)

    def test_soft_scale_original_values_populated(self):
        config, actuator = _single_actuator(SteeringMode.SOFT_SCALE, scale_factor=1.5, target_features=["f0"])
        event = _make_feature_event({"f0": 2.0, "f1": 3.0})
        _, record = actuator.apply(event, config.policy_id)
        self.assertAlmostEqual(record.original_values["f0"], 2.0)

    def test_soft_scale_modified_values_populated(self):
        config, actuator = _single_actuator(SteeringMode.SOFT_SCALE, scale_factor=1.5, target_features=["f0"])
        event = _make_feature_event({"f0": 2.0, "f1": 3.0})
        _, record = actuator.apply(event, config.policy_id)
        self.assertAlmostEqual(record.modified_values["f0"], 3.0)

    def test_soft_scale_factor_zero_zeroes_feature(self):
        config, actuator = _single_actuator(SteeringMode.SOFT_SCALE, scale_factor=0.0, target_features=["f0"])
        event = _make_feature_event({"f0": 2.0, "f1": 3.0})
        out, _ = actuator.apply(event, config.policy_id)
        self.assertAlmostEqual(out.features["f0"], 0.0)

    def test_soft_scale_factor_two_doubles_feature(self):
        config, actuator = _single_actuator(SteeringMode.SOFT_SCALE, scale_factor=2.0, target_features=["f0"])
        event = _make_feature_event({"f0": 1.5})
        out, _ = actuator.apply(event, config.policy_id)
        self.assertAlmostEqual(out.features["f0"], 3.0)

    def test_soft_scale_does_not_mutate_input(self):
        config, actuator = _single_actuator(SteeringMode.SOFT_SCALE, scale_factor=2.0, target_features=["f0"])
        event = _make_feature_event({"f0": 1.0, "f1": 2.0})
        original = dict(event.features)
        actuator.apply(event, config.policy_id)
        self.assertEqual(event.features, original)


# ---------------------------------------------------------------------------
# SUPPRESSION mode
# ---------------------------------------------------------------------------


class TestSteeringActuatorSuppression(unittest.TestCase):
    def test_suppression_applied_is_true(self):
        config, actuator = _single_actuator(SteeringMode.SUPPRESSION, target_features=["f0"])
        event = _make_feature_event({"f0": 2.0, "f1": 3.0})
        _, record = actuator.apply(event, config.policy_id)
        self.assertTrue(record.applied)

    def test_suppression_targeted_feature_is_zeroed(self):
        config, actuator = _single_actuator(SteeringMode.SUPPRESSION, target_features=["f0"])
        event = _make_feature_event({"f0": 2.0, "f1": 3.0})
        out, _ = actuator.apply(event, config.policy_id)
        self.assertAlmostEqual(out.features["f0"], 0.0)

    def test_suppression_non_targeted_feature_unchanged(self):
        config, actuator = _single_actuator(SteeringMode.SUPPRESSION, target_features=["f0"])
        event = _make_feature_event({"f0": 2.0, "f1": 3.0})
        out, _ = actuator.apply(event, config.policy_id)
        self.assertAlmostEqual(out.features["f1"], 3.0)


# ---------------------------------------------------------------------------
# Declined interventions
# ---------------------------------------------------------------------------


class TestSteeringActuatorDecline(unittest.TestCase):
    def test_disabled_policy_declines_with_reason(self):
        config, actuator = _single_actuator(SteeringMode.SOFT_SCALE, scale_factor=1.5)
        actuator._registry.disable(config.policy_id)
        event = _make_feature_event()
        _, record = actuator.apply(event, config.policy_id)
        self.assertFalse(record.applied)
        self.assertIsNotNone(record.decline_reason)

    def test_blocked_policy_declines_with_reason(self):
        config, actuator = _single_actuator(SteeringMode.SOFT_SCALE, scale_factor=1.5)
        actuator._registry.block(config.policy_id, "safety")
        event = _make_feature_event()
        _, record = actuator.apply(event, config.policy_id)
        self.assertFalse(record.applied)
        self.assertIsNotNone(record.decline_reason)

    def test_unknown_policy_id_raises_key_error(self):
        registry = PolicyRegistry()
        actuator = SteeringActuator(registry)
        with self.assertRaises(KeyError):
            actuator.apply(_make_feature_event(), "completely_unknown_id")

    def test_max_total_interventions_exceeded_declines(self):
        config, actuator = _single_actuator(SteeringMode.SOFT_SCALE, scale_factor=1.5, max_total=1)
        event = _make_feature_event({"f0": 1.0, "f1": 2.0})
        _, r1 = actuator.apply(event, config.policy_id)
        self.assertTrue(r1.applied)
        _, r2 = actuator.apply(event, config.policy_id)
        self.assertFalse(r2.applied)
        self.assertEqual(r2.decline_reason, "max_total_interventions_exceeded")

    def test_policy_max_interventions_exceeded_declines(self):
        config, actuator = _single_actuator(SteeringMode.SOFT_SCALE, scale_factor=1.5, max_interventions=1)
        event = _make_feature_event({"f0": 1.0, "f1": 2.0})
        _, r1 = actuator.apply(event, config.policy_id)
        self.assertTrue(r1.applied)
        _, r2 = actuator.apply(event, config.policy_id)
        self.assertFalse(r2.applied)
        self.assertEqual(r2.decline_reason, "policy_max_interventions_exceeded")

    def test_max_interventions_zero_means_unlimited(self):
        config, actuator = _single_actuator(SteeringMode.SOFT_SCALE, scale_factor=1.1, max_interventions=0)
        event = _make_feature_event({"f0": 1.0, "f1": 2.0})
        for _ in range(5):
            _, record = actuator.apply(event, config.policy_id)
            self.assertTrue(record.applied)
        self.assertEqual(actuator.total_applied(), 5)


# ---------------------------------------------------------------------------
# apply_all_active
# ---------------------------------------------------------------------------


class TestApplyAllActive(unittest.TestCase):
    def test_chains_multiple_policies(self):
        c1 = _make_policy(name="p1", mode=SteeringMode.SOFT_SCALE, scale_factor=2.0, target_features=["f0"])
        c2 = _make_policy(name="p2", mode=SteeringMode.SOFT_SCALE, scale_factor=2.0, target_features=["f0"])
        registry = PolicyRegistry()
        registry.register(c1)
        registry.register(c2)
        actuator = SteeringActuator(registry)
        event = _make_feature_event({"f0": 1.0})
        out, records = actuator.apply_all_active(event)
        self.assertAlmostEqual(out.features["f0"], 4.0)
        self.assertEqual(len(records), 2)

    def test_no_active_policies_returns_original_and_empty_list(self):
        registry = PolicyRegistry()
        actuator = SteeringActuator(registry)
        event = _make_feature_event({"f0": 1.0})
        out, records = actuator.apply_all_active(event)
        self.assertEqual(out.features, event.features)
        self.assertEqual(records, [])

    def test_disabled_policy_excluded_from_apply_all(self):
        c1 = _make_policy(name="disabled_p", mode=SteeringMode.SOFT_SCALE, scale_factor=3.0, target_features=["f0"])
        registry = PolicyRegistry()
        registry.register(c1)
        registry.disable(c1.policy_id)
        actuator = SteeringActuator(registry)
        event = _make_feature_event({"f0": 1.0})
        out, records = actuator.apply_all_active(event)
        self.assertEqual(len(records), 0)
        self.assertAlmostEqual(out.features["f0"], 1.0)


# ---------------------------------------------------------------------------
# Record accumulation and counters
# ---------------------------------------------------------------------------


class TestRecordStats(unittest.TestCase):
    def test_get_records_accumulates_across_calls(self):
        config, actuator = _single_actuator(SteeringMode.SHADOW)
        event = _make_feature_event()
        actuator.apply(event, config.policy_id)
        actuator.apply(event, config.policy_id)
        self.assertEqual(len(actuator.get_records()), 2)

    def test_clear_records_resets_to_empty(self):
        config, actuator = _single_actuator(SteeringMode.SHADOW)
        event = _make_feature_event()
        actuator.apply(event, config.policy_id)
        actuator.clear_records()
        self.assertEqual(len(actuator.get_records()), 0)

    def test_total_applied_counts_applied_records(self):
        config, actuator = _single_actuator(SteeringMode.SOFT_SCALE, scale_factor=1.5)
        event = _make_feature_event({"f0": 1.0, "f1": 2.0})
        actuator.apply(event, config.policy_id)
        actuator.apply(event, config.policy_id)
        self.assertEqual(actuator.total_applied(), 2)

    def test_total_shadow_counts_shadow_records(self):
        config, actuator = _single_actuator(SteeringMode.SHADOW)
        event = _make_feature_event()
        actuator.apply(event, config.policy_id)
        actuator.apply(event, config.policy_id)
        self.assertEqual(actuator.total_shadow(), 2)
        self.assertEqual(actuator.total_applied(), 0)

    def test_declined_not_counted_as_shadow(self):
        config, actuator = _single_actuator(SteeringMode.SOFT_SCALE, scale_factor=1.5, max_interventions=1)
        event = _make_feature_event({"f0": 1.0, "f1": 2.0})
        actuator.apply(event, config.policy_id)  # applied
        actuator.apply(event, config.policy_id)  # declined
        self.assertEqual(actuator.total_applied(), 1)
        self.assertEqual(actuator.total_shadow(), 0)


# ---------------------------------------------------------------------------
# Rollback
# ---------------------------------------------------------------------------


class TestRollback(unittest.TestCase):
    def test_rollback_restores_original_values(self):
        config, actuator = _single_actuator(SteeringMode.SOFT_SCALE, scale_factor=2.0, target_features=["f0"])
        event = _make_feature_event({"f0": 1.0, "f1": 2.0})
        out, record = actuator.apply(event, config.policy_id)
        self.assertAlmostEqual(out.features["f0"], 2.0)
        rolled = rollback_feature_event(record, out)
        self.assertAlmostEqual(rolled.features["f0"], 1.0)

    def test_rollback_raises_value_error_on_unapplied_record(self):
        config, actuator = _single_actuator(SteeringMode.SHADOW)
        event = _make_feature_event()
        _, record = actuator.apply(event, config.policy_id)
        self.assertFalse(record.applied)
        with self.assertRaises(ValueError):
            rollback_feature_event(record, event)

    def test_rollback_non_targeted_features_preserved(self):
        config, actuator = _single_actuator(SteeringMode.SOFT_SCALE, scale_factor=2.0, target_features=["f0"])
        event = _make_feature_event({"f0": 1.0, "f1": 5.0})
        out, record = actuator.apply(event, config.policy_id)
        rolled = rollback_feature_event(record, out)
        self.assertAlmostEqual(rolled.features["f1"], 5.0)


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------


class TestInterventionRecordSerialization(unittest.TestCase):
    def test_round_trip_preserves_all_fields(self):
        config, actuator = _single_actuator(SteeringMode.SUPPRESSION, target_features=["f0"])
        event = _make_feature_event({"f0": 2.5, "f1": 1.0})
        _, record = actuator.apply(event, config.policy_id)
        d = intervention_record_to_dict(record)
        restored = intervention_record_from_dict(d)
        self.assertEqual(restored.record_id, record.record_id)
        self.assertEqual(restored.policy_id, record.policy_id)
        self.assertEqual(restored.mode, record.mode)
        self.assertEqual(restored.applied, record.applied)
        self.assertEqual(restored.features_targeted, record.features_targeted)
        self.assertEqual(restored.features_modified, record.features_modified)

    def test_round_trip_preserves_provenance_fields(self):
        config, actuator = _single_actuator(SteeringMode.SOFT_SCALE, scale_factor=1.2)
        event = _make_feature_event({"f0": 1.0, "f1": 2.0})
        _, record = actuator.apply(event, config.policy_id)
        restored = intervention_record_from_dict(intervention_record_to_dict(record))
        self.assertEqual(restored.run_id, "run_test_01")
        self.assertEqual(restored.model_id, "test_model")
        self.assertEqual(restored.layer_id, "layer.0")

    def test_serialise_mode_as_string(self):
        config, actuator = _single_actuator(SteeringMode.SHADOW)
        event = _make_feature_event()
        _, record = actuator.apply(event, config.policy_id)
        d = intervention_record_to_dict(record)
        self.assertEqual(d["mode"], "SHADOW")

    def test_round_trip_shadow_decline_reason_none(self):
        config, actuator = _single_actuator(SteeringMode.SHADOW)
        event = _make_feature_event()
        _, record = actuator.apply(event, config.policy_id)
        restored = intervention_record_from_dict(intervention_record_to_dict(record))
        self.assertIsNone(restored.decline_reason)


# ---------------------------------------------------------------------------
# SteeringActuator provenance persistence (MOD-3 fix)
# ---------------------------------------------------------------------------


class TestSteeringActuatorPersistence(unittest.TestCase):
    def _build_actuator_with_records(self, mode=SteeringMode.SOFT_SCALE, count=3):
        config, actuator = _single_actuator(mode, scale_factor=1.5, target_features=["f0"])
        event = _make_feature_event({"f0": 1.0, "f1": 2.0})
        for _ in range(count):
            actuator.apply(event, config.policy_id)
        return actuator

    def test_persist_records_returns_count(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            actuator = self._build_actuator_with_records(count=3)
            n = actuator.persist_records(Path(tmpdir) / "records.jsonl")
            self.assertEqual(n, 3)

    def test_persist_records_creates_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            actuator = self._build_actuator_with_records(count=2)
            path = Path(tmpdir) / "records.jsonl"
            actuator.persist_records(path)
            self.assertTrue(path.exists())

    def test_persist_records_writes_one_line_per_record(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            actuator = self._build_actuator_with_records(count=4)
            path = Path(tmpdir) / "records.jsonl"
            actuator.persist_records(path)
            lines = [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
            self.assertEqual(len(lines), 4)

    def test_persist_and_reload_preserves_record_count(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            actuator = self._build_actuator_with_records(count=5)
            path = Path(tmpdir) / "records.jsonl"
            actuator.persist_records(path)

            registry2 = PolicyRegistry()
            actuator2 = SteeringActuator(registry2)
            loaded = actuator2.load_records(path)
            self.assertEqual(loaded, 5)
            self.assertEqual(len(actuator2.get_records()), 5)

    def test_persist_and_reload_preserves_applied_status(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            actuator = self._build_actuator_with_records(SteeringMode.SOFT_SCALE, count=2)
            path = Path(tmpdir) / "records.jsonl"
            actuator.persist_records(path)

            registry2 = PolicyRegistry()
            actuator2 = SteeringActuator(registry2)
            actuator2.load_records(path)
            for r in actuator2.get_records():
                self.assertTrue(r.applied)

    def test_persist_and_reload_preserves_all_fields(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config, actuator = _single_actuator(SteeringMode.SUPPRESSION, target_features=["f0"])
            event = _make_feature_event({"f0": 2.5, "f1": 1.0})
            _, original = actuator.apply(event, config.policy_id)
            path = Path(tmpdir) / "records.jsonl"
            actuator.persist_records(path)

            registry2 = PolicyRegistry()
            actuator2 = SteeringActuator(registry2)
            actuator2.load_records(path)
            restored = actuator2.get_records()[0]

            self.assertEqual(restored.record_id, original.record_id)
            self.assertEqual(restored.policy_id, original.policy_id)
            self.assertEqual(restored.mode, original.mode)
            self.assertEqual(restored.applied, original.applied)
            self.assertEqual(restored.features_targeted, original.features_targeted)
            self.assertEqual(restored.original_values, original.original_values)
            self.assertEqual(restored.modified_values, original.modified_values)

    def test_reload_into_fresh_instance_across_new_actuator(self):
        """Records survive a process restart simulation: write then read into brand-new SteeringActuator."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Session A: build and persist
            actuator_a = self._build_actuator_with_records(count=2)
            path = Path(tmpdir) / "session_a.jsonl"
            actuator_a.persist_records(path)

            # Session B: fresh actuator with no prior state
            registry_b = PolicyRegistry()
            actuator_b = SteeringActuator(registry_b)
            self.assertEqual(len(actuator_b.get_records()), 0)
            actuator_b.load_records(path)
            self.assertEqual(len(actuator_b.get_records()), 2)

    def test_persist_empty_records_produces_empty_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = PolicyRegistry()
            actuator = SteeringActuator(registry)
            path = Path(tmpdir) / "empty.jsonl"
            n = actuator.persist_records(path)
            self.assertEqual(n, 0)
            self.assertEqual(path.read_text(encoding="utf-8").strip(), "")

    def test_load_records_returns_count(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            actuator = self._build_actuator_with_records(count=3)
            path = Path(tmpdir) / "records.jsonl"
            actuator.persist_records(path)

            registry2 = PolicyRegistry()
            actuator2 = SteeringActuator(registry2)
            n = actuator2.load_records(path)
            self.assertEqual(n, 3)

    def test_load_records_is_additive(self):
        """load_records appends, does not replace existing records."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config, actuator = _single_actuator(SteeringMode.SHADOW)
            event = _make_feature_event()
            actuator.apply(event, config.policy_id)

            path = Path(tmpdir) / "records.jsonl"
            actuator.persist_records(path)

            actuator.load_records(path)  # adds 1 more from file → total 2
            self.assertEqual(len(actuator.get_records()), 2)

    def test_load_records_skips_corrupt_line(self):
        """A corrupt JSON line between two valid records is skipped; valid records load."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Build two valid records and serialise them to a JSONL file.
            actuator_src = self._build_actuator_with_records(count=2)
            valid_lines = []
            for record in actuator_src.get_records():
                import json as _json_local
                from model_scope_steering import intervention_record_to_dict
                valid_lines.append(_json_local.dumps(intervention_record_to_dict(record)))

            # Insert a corrupt line between the two valid ones.
            path = Path(tmpdir) / "corrupt.jsonl"
            path.write_text(
                valid_lines[0] + "\n"
                + "THIS IS NOT VALID JSON {{{\n"
                + valid_lines[1] + "\n",
                encoding="utf-8",
            )

            registry2 = PolicyRegistry()
            actuator2 = SteeringActuator(registry2)
            loaded = actuator2.load_records(path)

            # Only the 2 valid records should be loaded; no exception raised.
            self.assertEqual(loaded, 2)
            self.assertEqual(len(actuator2.get_records()), 2)


if __name__ == "__main__":
    unittest.main()
