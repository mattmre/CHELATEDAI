"""Tests for steering_policy module — SteeringMode, PolicyStatus, SteeringPolicyConfig, PolicyRegistry."""

from __future__ import annotations

import unittest
from datetime import datetime

from steering_policy import (
    PolicyRegistry,
    PolicyStatus,
    SteeringMode,
    SteeringPolicyConfig,
)


def _make_config(
    name: str = "test_policy",
    mode: SteeringMode = SteeringMode.SHADOW,
    target_features: list = None,
    scale_factor: float = 1.0,
    max_interventions: int = 0,
) -> SteeringPolicyConfig:
    return SteeringPolicyConfig(
        name=name,
        mode=mode,
        target_features=target_features if target_features is not None else ["feat_0", "feat_1"],
        scale_factor=scale_factor,
        max_interventions=max_interventions,
    )


class TestSteeringPolicyConfig(unittest.TestCase):
    def test_schema_version_default(self):
        config = _make_config()
        self.assertEqual(config.schema_version, "1.0")

    def test_policy_id_assigned_on_construction(self):
        config = _make_config()
        self.assertIsNotNone(config.policy_id)
        self.assertGreater(len(config.policy_id), 0)

    def test_policy_id_unique_across_instances(self):
        a = _make_config()
        b = _make_config()
        self.assertNotEqual(a.policy_id, b.policy_id)

    def test_mode_shadow(self):
        config = _make_config(mode=SteeringMode.SHADOW)
        self.assertEqual(config.mode, SteeringMode.SHADOW)

    def test_mode_soft_scale(self):
        config = _make_config(mode=SteeringMode.SOFT_SCALE)
        self.assertEqual(config.mode, SteeringMode.SOFT_SCALE)

    def test_mode_suppression(self):
        config = _make_config(mode=SteeringMode.SUPPRESSION)
        self.assertEqual(config.mode, SteeringMode.SUPPRESSION)

    def test_scale_factor_default_is_one(self):
        config = _make_config()
        self.assertAlmostEqual(config.scale_factor, 1.0)

    def test_scale_factor_clamped_above_two(self):
        config = _make_config(scale_factor=5.0)
        self.assertAlmostEqual(config.scale_factor, 2.0)

    def test_scale_factor_clamped_below_zero(self):
        config = _make_config(scale_factor=-1.0)
        self.assertAlmostEqual(config.scale_factor, 0.0)

    def test_scale_factor_boundary_zero(self):
        config = _make_config(scale_factor=0.0)
        self.assertAlmostEqual(config.scale_factor, 0.0)

    def test_scale_factor_boundary_two(self):
        config = _make_config(scale_factor=2.0)
        self.assertAlmostEqual(config.scale_factor, 2.0)

    def test_scale_factor_mid_range_unchanged(self):
        config = _make_config(scale_factor=1.25)
        self.assertAlmostEqual(config.scale_factor, 1.25)

    def test_status_default_active(self):
        config = _make_config()
        self.assertEqual(config.status, PolicyStatus.ACTIVE)

    def test_created_at_is_iso8601(self):
        config = _make_config()
        dt = datetime.fromisoformat(config.created_at)
        self.assertIsInstance(dt, datetime)

    def test_description_default_empty(self):
        config = _make_config()
        self.assertEqual(config.description, "")

    def test_max_interventions_default_zero(self):
        config = _make_config()
        self.assertEqual(config.max_interventions, 0)

    def test_target_features_stored(self):
        config = _make_config(target_features=["alpha", "beta", "gamma"])
        self.assertEqual(config.target_features, ["alpha", "beta", "gamma"])

    def test_string_mode_coerced_in_post_init(self):
        config = SteeringPolicyConfig(
            name="coerce_test",
            mode="SOFT_SCALE",  # type: ignore[arg-type]
            target_features=["f"],
        )
        self.assertEqual(config.mode, SteeringMode.SOFT_SCALE)

    def test_string_status_coerced_in_post_init(self):
        config = SteeringPolicyConfig(
            name="coerce_status",
            mode=SteeringMode.SHADOW,
            target_features=["f"],
            status="DISABLED",  # type: ignore[arg-type]
        )
        self.assertEqual(config.status, PolicyStatus.DISABLED)


class TestPolicyRegistry(unittest.TestCase):
    def test_register_returns_policy_id(self):
        registry = PolicyRegistry()
        config = _make_config()
        returned = registry.register(config)
        self.assertEqual(returned, config.policy_id)

    def test_get_retrieves_correct_config(self):
        registry = PolicyRegistry()
        config = _make_config(name="my_policy")
        registry.register(config)
        retrieved = registry.get(config.policy_id)
        self.assertEqual(retrieved.name, "my_policy")

    def test_get_unknown_raises_key_error(self):
        registry = PolicyRegistry()
        with self.assertRaises(KeyError):
            registry.get("nonexistent_policy_id")

    def test_list_active_returns_only_active_policies(self):
        registry = PolicyRegistry()
        a = _make_config(name="a")
        b = _make_config(name="b")
        c = _make_config(name="to_disable")
        registry.register(a)
        registry.register(b)
        registry.register(c)
        registry.disable(c.policy_id)
        active = registry.list_active()
        names = [p.name for p in active]
        self.assertIn("a", names)
        self.assertIn("b", names)
        self.assertNotIn("to_disable", names)

    def test_disable_changes_status_to_disabled(self):
        registry = PolicyRegistry()
        config = _make_config()
        registry.register(config)
        result = registry.disable(config.policy_id)
        self.assertTrue(result)
        self.assertEqual(registry.get(config.policy_id).status, PolicyStatus.DISABLED)

    def test_disable_returns_false_for_unknown(self):
        registry = PolicyRegistry()
        self.assertFalse(registry.disable("nonexistent"))

    def test_block_changes_status_to_blocked(self):
        registry = PolicyRegistry()
        config = _make_config()
        registry.register(config)
        result = registry.block(config.policy_id, "safety check failed")
        self.assertTrue(result)
        self.assertEqual(registry.get(config.policy_id).status, PolicyStatus.BLOCKED)

    def test_block_returns_false_for_unknown(self):
        registry = PolicyRegistry()
        self.assertFalse(registry.block("nonexistent", "reason"))

    def test_block_stores_reason(self):
        registry = PolicyRegistry()
        config = _make_config()
        registry.register(config)
        registry.block(config.policy_id, "safety_violation")
        self.assertEqual(registry.get_block_reason(config.policy_id), "safety_violation")

    def test_get_block_reason_none_for_active_policy(self):
        registry = PolicyRegistry()
        config = _make_config()
        registry.register(config)
        self.assertIsNone(registry.get_block_reason(config.policy_id))

    def test_get_block_reason_none_for_unknown_id(self):
        registry = PolicyRegistry()
        self.assertIsNone(registry.get_block_reason("unknown_id"))

    def test_list_active_excludes_blocked_policies(self):
        registry = PolicyRegistry()
        config = _make_config()
        registry.register(config)
        registry.block(config.policy_id, "test")
        self.assertEqual(len(registry.list_active()), 0)

    def test_count_reflects_all_statuses(self):
        registry = PolicyRegistry()
        a = _make_config()
        b = _make_config()
        c = _make_config()
        registry.register(a)
        registry.register(b)
        registry.register(c)
        registry.disable(b.policy_id)
        registry.block(c.policy_id, "reason")
        self.assertEqual(registry.count(), 3)

    def test_to_dict_from_dict_round_trip_preserves_fields(self):
        registry = PolicyRegistry()
        config = SteeringPolicyConfig(
            name="roundtrip_policy",
            mode=SteeringMode.SOFT_SCALE,
            target_features=["f1", "f2"],
            scale_factor=1.5,
            description="A round-trip test",
            max_interventions=10,
        )
        registry.register(config)
        data = registry.to_dict()
        restored = PolicyRegistry.from_dict(data)
        retrieved = restored.get(config.policy_id)
        self.assertEqual(retrieved.name, "roundtrip_policy")
        self.assertEqual(retrieved.mode, SteeringMode.SOFT_SCALE)
        self.assertAlmostEqual(retrieved.scale_factor, 1.5)
        self.assertEqual(retrieved.target_features, ["f1", "f2"])
        self.assertEqual(retrieved.description, "A round-trip test")
        self.assertEqual(retrieved.max_interventions, 10)

    def test_from_dict_restores_disabled_status(self):
        registry = PolicyRegistry()
        config = _make_config()
        registry.register(config)
        registry.disable(config.policy_id)
        restored = PolicyRegistry.from_dict(registry.to_dict())
        self.assertEqual(restored.get(config.policy_id).status, PolicyStatus.DISABLED)

    def test_from_dict_restores_blocked_status_and_reason(self):
        registry = PolicyRegistry()
        config = _make_config()
        registry.register(config)
        registry.block(config.policy_id, "blocked_for_test")
        restored = PolicyRegistry.from_dict(registry.to_dict())
        self.assertEqual(restored.get(config.policy_id).status, PolicyStatus.BLOCKED)
        self.assertEqual(restored.get_block_reason(config.policy_id), "blocked_for_test")

    def test_to_dict_schema_version_present(self):
        registry = PolicyRegistry()
        config = _make_config()
        registry.register(config)
        d = registry.to_dict()
        self.assertEqual(d["policies"][config.policy_id]["schema_version"], "1.0")

    def test_to_dict_block_reasons_included(self):
        registry = PolicyRegistry()
        config = _make_config()
        registry.register(config)
        registry.block(config.policy_id, "my_reason")
        d = registry.to_dict()
        self.assertEqual(d["block_reasons"][config.policy_id], "my_reason")

    def test_max_interventions_zero_means_unlimited(self):
        """Verify max_interventions=0 is stored correctly; actuator test confirms unlimited."""
        config = _make_config(max_interventions=0)
        self.assertEqual(config.max_interventions, 0)

    def test_multiple_registrations_all_retrievable(self):
        registry = PolicyRegistry()
        configs = [_make_config(name=f"p{i}") for i in range(5)]
        for c in configs:
            registry.register(c)
        self.assertEqual(registry.count(), 5)
        for c in configs:
            self.assertEqual(registry.get(c.policy_id).name, c.name)


class TestModelScopeSteeringPolicyActivate(unittest.TestCase):
    """Tests for ModelScopeSteeringPolicy.activate() — R3 fix."""

    def setUp(self):
        from steering_policy import ModelScopeSteeringPolicy
        self.Policy = ModelScopeSteeringPolicy

    def test_default_deployment_mode_is_shadow_mode(self):
        policy = self.Policy()
        self.assertEqual(policy.deployment_mode, "shadow_mode")

    def test_activate_soft_scale_sets_mode(self):
        policy = self.Policy()
        policy.activate("soft_scale")
        self.assertEqual(policy.deployment_mode, "soft_scale")

    def test_activate_suppression_sets_mode(self):
        policy = self.Policy()
        policy.activate("suppression")
        self.assertEqual(policy.deployment_mode, "suppression")

    def test_activate_active_sets_mode(self):
        policy = self.Policy()
        policy.activate("active")
        self.assertEqual(policy.deployment_mode, "active")

    def test_activate_shadow_mode_raises_value_error(self):
        policy = self.Policy()
        with self.assertRaises(ValueError):
            policy.activate("shadow_mode")

    def test_activate_invalid_string_raises_value_error(self):
        policy = self.Policy()
        with self.assertRaises(ValueError):
            policy.activate("not_a_real_mode")


if __name__ == "__main__":
    unittest.main()
