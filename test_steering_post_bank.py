"""Tests for steering_post_bank.py — the living post-bank + prune/re-anneal
lifecycle (Phase II, H5a). numpy-only, stub corrections, no torch/GPU."""
from __future__ import annotations

import unittest

import numpy as np

from steering_post_bank import SteeringPost, SteeringPostBank


def _identity(v):
    return v


def _shift(amount):
    return lambda v: v + amount


def _three_post_bank(prune_below=0.5, min_posts=1, temperature=0.0):
    bank = SteeringPostBank(input_dim=3, prune_below=prune_below, min_posts=min_posts,
                            temperature=temperature)
    bank.register_post("e1", [1.0, 0.0, 0.0], _identity)
    bank.register_post("e2", [0.0, 1.0, 0.0], _identity)
    bank.register_post("e3", [0.0, 0.0, 1.0], _identity)
    return bank


class TestRoutingAndApply(unittest.TestCase):
    def test_route_picks_nearest_centroid_by_cosine(self):
        bank = _three_post_bank()
        self.assertEqual(bank.route([0.9, 0.1, 0.0]).key, "e1")
        self.assertEqual(bank.route([0.0, 0.2, 0.9]).key, "e3")

    def test_route_is_deterministic_tie_break_by_key(self):
        bank = SteeringPostBank(input_dim=2)
        bank.register_post("b", [1.0, 0.0], _identity)
        bank.register_post("a", [1.0, 0.0], _identity)  # identical centroid
        # Tie -> sorted-key order keeps the first seen ("a").
        self.assertEqual(bank.route([1.0, 0.0]).key, "a")

    def test_apply_corrects_and_logs_store_mutation(self):
        bank = _three_post_bank()
        bank.register_post("shift1", [1.0, 0.0, 0.0], _shift(0.5), fitness=1.0)
        # A vector near e1 actually routes to whichever centroid is closest; use a
        # bank where every post shifts so the correction norm is non-zero.
        b2 = SteeringPostBank(input_dim=3)
        b2.register_post("s", [1.0, 0.0, 0.0], _shift(0.25))
        vecs = np.array([[1.0, 0.0, 0.0], [0.9, 0.1, 0.0]])
        out = b2.apply(vecs)
        self.assertEqual(out.shape, vecs.shape)
        np.testing.assert_allclose(out, vecs + 0.25)
        log = [e for e in b2.lifecycle_log if e["action"] == "apply"][-1]
        self.assertEqual(log["n_vectors"], 2)
        self.assertEqual(log["per_post_hits"], {"s": 2})
        self.assertGreater(log["mean_correction_norm"], 0.0)

    def test_apply_identity_logs_zero_correction_norm(self):
        bank = _three_post_bank()
        out = bank.apply(np.eye(3))
        np.testing.assert_allclose(out, np.eye(3))
        log = [e for e in bank.lifecycle_log if e["action"] == "apply"][-1]
        self.assertEqual(log["mean_correction_norm"], 0.0)

    def test_apply_rejects_wrong_shape(self):
        bank = _three_post_bank()
        with self.assertRaises(ValueError):
            bank.apply(np.zeros((2, 4)))

    def test_route_validations(self):
        bank = SteeringPostBank(input_dim=3)
        with self.assertRaises(ValueError):
            bank.route([1.0, 0.0, 0.0])  # empty bank
        bank.register_post("e1", [1.0, 0.0, 0.0], _identity)
        with self.assertRaises(ValueError):
            bank.route([0.0, 0.0, 0.0])  # zero vector
        with self.assertRaises(ValueError):
            bank.route([1.0, 0.0])  # wrong dim


class TestLifecycle(unittest.TestCase):
    def test_record_fitness_set_and_ema(self):
        bank = _three_post_bank()
        self.assertEqual(bank.record_fitness("e1", 1.0), 1.0)
        # EMA: decay 0.5 -> 0.5*1.0 + 0.5*0.0 = 0.5
        self.assertAlmostEqual(bank.record_fitness("e1", 0.0, decay=0.5), 0.5)
        with self.assertRaises(KeyError):
            bank.record_fitness("nope", 1.0)

    def test_prune_removes_below_effective_threshold(self):
        bank = _three_post_bank(prune_below=0.5, min_posts=0)
        bank.record_fitness("e1", 0.9)
        bank.record_fitness("e2", 0.1)
        bank.record_fitness("e3", 0.0)
        pruned = bank.prune()
        self.assertEqual(set(pruned), {"e2", "e3"})
        self.assertEqual({p.key for p in bank.posts}, {"e1"})

    def test_prune_respects_min_posts_keeps_fittest(self):
        bank = _three_post_bank(prune_below=1.0, min_posts=1)  # all below -> all candidates
        bank.record_fitness("e1", 0.3)
        bank.record_fitness("e2", 0.2)
        bank.record_fitness("e3", 0.1)
        pruned = bank.prune()
        # min_posts=1 -> keep the single fittest (e1); prune the two weakest.
        self.assertEqual(set(pruned), {"e2", "e3"})
        self.assertEqual([p.key for p in bank.posts], ["e1"])

    def test_temperature_modulates_pruning(self):
        bank = _three_post_bank(prune_below=0.5, min_posts=0)
        for k in ("e1", "e2", "e3"):
            bank.record_fitness(k, 0.1)  # all below base threshold
        # Explore: T=1 -> effective threshold 0 -> nothing pruned.
        bank.anneal_step(1.0)
        self.assertEqual(bank.prune(), [])
        self.assertEqual(len(bank), 3)
        # Stabilize: T=0 -> effective threshold 0.5 -> all pruned.
        bank.anneal_step(0.0)
        self.assertEqual(set(bank.prune()), {"e1", "e2", "e3"})
        self.assertEqual(len(bank), 0)

    def test_re_anneal_recreates_posts_with_origin(self):
        bank = _three_post_bank(prune_below=1.0, min_posts=0)
        for k in ("e1", "e2", "e3"):
            bank.record_fitness(k, 0.0)
        pruned = bank.prune()
        self.assertEqual(len(bank), 0)

        def factory(key):
            return [1.0, 0.0, 0.0], _identity

        rebuilt = bank.re_anneal(factory, pruned)
        self.assertEqual(set(rebuilt), {"e1", "e2", "e3"})
        self.assertEqual(len(bank), 3)
        self.assertTrue(all(p.origin == "re-annealed" for p in bank.posts))
        self.assertTrue(all(p.fitness == 0.0 and p.age == 0 for p in bank.posts))

    def test_prune_and_reanneal_noop_when_drift_not_fired(self):
        bank = _three_post_bank(prune_below=1.0, min_posts=0)
        for k in ("e1", "e2", "e3"):
            bank.record_fitness(k, 0.0)

        def factory(key):
            return [1.0, 0.0, 0.0], _identity

        result = bank.prune_and_reanneal(factory, drift_fired=False)
        self.assertEqual(result, {"pruned": [], "reannealed": []})
        self.assertEqual(len(bank), 3)  # untouched

    def test_prune_and_reanneal_cycle_when_drift_fired(self):
        bank = _three_post_bank(prune_below=0.5, min_posts=0)
        bank.record_fitness("e1", 0.9)  # >= 0.5 -> survives
        bank.record_fitness("e2", 0.0)  # pruned + re-annealed
        bank.record_fitness("e3", 0.0)

        def factory(key):
            return [0.0, 1.0, 0.0], _shift(0.1)

        result = bank.prune_and_reanneal(factory, drift_fired=True)
        self.assertEqual(set(result["pruned"]), {"e2", "e3"})
        self.assertEqual(set(result["reannealed"]), {"e2", "e3"})
        self.assertEqual(len(bank), 3)  # restored
        self.assertEqual(bank.post("e2").origin, "re-annealed")
        self.assertEqual(bank.post("e1").origin, "registered")  # survivor unchanged
        # The lifecycle log records the disintegration + re-anneal.
        actions = [e["action"] for e in bank.lifecycle_log]
        self.assertIn("prune", actions)
        self.assertIn("re_anneal", actions)


class TestContractAndSerialization(unittest.TestCase):
    def test_register_validations(self):
        bank = SteeringPostBank(input_dim=3)
        with self.assertRaises(ValueError):
            bank.register_post("x", [1.0, 0.0], _identity)  # wrong dim
        with self.assertRaises(ValueError):
            bank.register_post("x", [1.0, 0.0, 0.0], "not callable")  # type: ignore[arg-type]

    def test_init_validations(self):
        with self.assertRaises(ValueError):
            SteeringPostBank(input_dim=0)

    def test_to_dict_is_json_safe_and_excludes_callables(self):
        import json

        bank = _three_post_bank()
        bank.record_fitness("e1", 0.7)
        bank.apply(np.eye(3))
        d = bank.to_dict()
        json.dumps(d)  # must not raise (no numpy / no callables leaked)
        self.assertEqual(d["record_type"], "steering_post_bank")
        self.assertEqual(len(d["posts"]), 3)
        self.assertTrue(any(p["key"] == "e1" and p["fitness"] == 0.7 for p in d["posts"]))

    def test_post_summary_has_no_vector_or_callable(self):
        post = SteeringPost("k", np.array([1.0, 0.0]), _identity, fitness=0.5)
        s = post.summary()
        self.assertEqual(set(s), {"key", "dim", "fitness", "age", "origin"})


if __name__ == "__main__":
    unittest.main()
