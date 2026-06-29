"""Tests for post_bank_conditions.py — C5/C5s/C5r orchestration + lifecycle
(Phase II, H5b / S2b). Pure orchestration is stub-tested; the real adapter factory
gets a light CPU test; run_post_bank_cycle uses a patched eval + fake store."""
from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

from annealing_schedule import AnnealingSchedule
from post_bank_conditions import (
    POSTBANK_CONDITIONS,
    evolve_or_build_bank,
    make_adapter_post_factory,
    run_post_bank_cycle,
)


def _anchor_pairs():
    return [
        {"doc_vector": [10.0, 0.0], "query_vector": [11.0, 0.0]},
        {"doc_vector": [10.1, 0.0], "query_vector": [11.1, 0.0]},
        {"doc_vector": [0.0, 10.0], "query_vector": [0.0, 11.0]},
        {"doc_vector": [0.0, 10.1], "query_vector": [0.0, 11.1]},
    ]


def _stub_make_post(d, q):
    centroid = d.mean(axis=0)
    return lambda v: v + centroid


def _const0_schedule():
    # temperature == 0 always -> effective prune threshold == prune_below.
    return AnnealingSchedule("constant", n_cycles=12, t_start=0.0)


class TestEvolveOrBuildBank(unittest.TestCase):
    def test_first_call_builds(self):
        out = evolve_or_build_bank(None, "C5", _anchor_pairs(), 2, 2, _stub_make_post,
                                   _const0_schedule(), cycle_index=1, seed=0)
        self.assertTrue(out["lifecycle"]["built"])
        self.assertEqual(len(out["bank"]), 2)
        self.assertEqual(out["state"]["built_cycle"], 1)

    def test_c5_evolves_prunes_and_reanneals(self):
        built = evolve_or_build_bank(None, "C5", _anchor_pairs(), 2, 2, _stub_make_post,
                                     _const0_schedule(), 1, 0, prune_below=0.5)
        # cycle 2: post:0 has fitness 0 (< effective 0.5) -> pruned + re-annealed;
        # post:1 has fitness 10 -> survives. Bank restored to 2 posts.
        out = evolve_or_build_bank(built["state"], "C5", _anchor_pairs(), 2, 2, _stub_make_post,
                                   _const0_schedule(), cycle_index=2, seed=0,
                                   post_fitness={"post:0": 0.0, "post:1": 10.0}, prune_below=0.5)
        self.assertEqual(out["lifecycle"]["pruned"], ["post:0"])
        self.assertEqual(out["lifecycle"]["reannealed"], ["post:0"])
        self.assertEqual(len(out["bank"]), 2)
        self.assertEqual(out["lifecycle"]["temperature"], 0.0)
        # the re-annealed post is fresh (origin reset)
        self.assertEqual(out["bank"].post("post:0").origin, "re-annealed")

    def test_c5s_static_bank_does_not_evolve(self):
        built = evolve_or_build_bank(None, "C5s", _anchor_pairs(), 2, 2, _stub_make_post,
                                     _const0_schedule(), 1, 0)
        out = evolve_or_build_bank(built["state"], "C5s", _anchor_pairs(), 2, 2, _stub_make_post,
                                   _const0_schedule(), 2, 0, post_fitness={"post:0": 0.0})
        self.assertEqual(out["lifecycle"]["pruned"], [])
        self.assertEqual(out["lifecycle"]["reannealed"], [])
        self.assertIs(out["bank"], built["bank"])  # same frozen bank object

    def test_c5r_one_shot_does_not_evolve(self):
        built = evolve_or_build_bank(None, "C5r", _anchor_pairs(), 2, 2, _stub_make_post,
                                     _const0_schedule(), 1, 0)
        out = evolve_or_build_bank(built["state"], "C5r", _anchor_pairs(), 2, 2, _stub_make_post,
                                   _const0_schedule(), 2, 0, post_fitness={"post:0": 0.0})
        self.assertEqual(out["lifecycle"]["pruned"], [])
        self.assertIs(out["bank"], built["bank"])

    def test_rejects_non_postbank_condition(self):
        with self.assertRaises(ValueError):
            evolve_or_build_bank(None, "C3a", _anchor_pairs(), 2, 2, _stub_make_post,
                                 _const0_schedule(), 1, 0)


class TestMakeAdapterPostFactory(unittest.TestCase):
    def test_trains_a_working_correction_callable(self):
        rng = np.random.RandomState(0)
        docs = rng.randn(3, 4).astype(np.float32)
        queries = rng.randn(3, 4).astype(np.float32)
        make_post = make_adapter_post_factory(vector_size=4, bounded=True, seed=0, steps=10)
        correct = make_post(docs, queries)
        # 1D input -> 1D output; 2D input -> 2D output; finite; same dim.
        out1d = correct(docs[0])
        self.assertEqual(out1d.shape, (4,))
        out2d = correct(docs)
        self.assertEqual(out2d.shape, (3, 4))
        self.assertTrue(np.all(np.isfinite(out1d)))
        self.assertTrue(np.all(np.isfinite(out2d)))

    def test_unbounded_variant_also_builds(self):
        make_post = make_adapter_post_factory(vector_size=3, bounded=False, seed=1, steps=5)
        correct = make_post(np.eye(3, dtype=np.float32), np.eye(3, dtype=np.float32))
        self.assertEqual(correct(np.ones(3, dtype=np.float32)).shape, (3,))


class _FakeQdrant:
    def __init__(self):
        self.upserted = None

    def upsert(self, collection_name, points):
        self.upserted = list(points)


class _FakeEngine:
    def __init__(self):
        self.vector_size = 2
        self.qdrant = _FakeQdrant()
        self.collection_name = "c"


def _points():
    return [
        {"id": "d1", "vector": [10.0, 0.0], "payload": None},
        {"id": "d2", "vector": [0.0, 10.0], "payload": None},
    ]


class TestRunPostBankCycle(unittest.TestCase):
    def _cfg(self):
        return {"k": 10, "cycles": 12, "seed": 0, "post_bank_clusters": 2}

    def test_no_correction_when_no_drop(self):
        engine = _FakeEngine()
        with patch("run_drift_recovery_experiment.evaluate_engine_with_query_vectors",
                   return_value=(0.9, {})):
            meta = run_post_bank_cycle(
                engine, "C5", self._cfg(), {"q": np.array([1.0, 0.0])}, {"q": {"d1": 1.0}},
                _anchor_pairs(), baseline_ndcg=0.9, original_doc_points=_points(),
                cycle_index=1, make_post_factory=_stub_make_post,
            )
        self.assertFalse(meta["should_correct"])
        self.assertFalse(meta["correction_applied"])
        self.assertIsNone(engine.qdrant.upserted)  # never applied

    def test_c5_builds_and_applies_on_drop(self):
        engine = _FakeEngine()
        with patch("run_drift_recovery_experiment.evaluate_engine_with_query_vectors",
                   return_value=(0.1, {})):
            meta = run_post_bank_cycle(
                engine, "C5", self._cfg(), {"q": np.array([1.0, 0.0])}, {"q": {"d1": 1.0}},
                _anchor_pairs(), baseline_ndcg=0.9, original_doc_points=_points(),
                cycle_index=1, make_post_factory=_stub_make_post,
            )
        self.assertTrue(meta["should_correct"])
        self.assertTrue(meta["correction_applied"])
        self.assertEqual(meta["store_updated"], 2)
        self.assertIsNotNone(getattr(engine, "_post_bank_state", None))
        self.assertEqual(meta["post_bank_kind"], "living")

    def test_c5r_one_shot_does_not_reapply(self):
        engine = _FakeEngine()
        with patch("run_drift_recovery_experiment.evaluate_engine_with_query_vectors",
                   return_value=(0.1, {})):
            run_post_bank_cycle(
                engine, "C5r", self._cfg(), {"q": np.array([1.0, 0.0])}, {"q": {"d1": 1.0}},
                _anchor_pairs(), 0.9, _points(), cycle_index=1, make_post_factory=_stub_make_post,
            )
            engine.qdrant.upserted = None  # reset to detect a re-apply
            meta2 = run_post_bank_cycle(
                engine, "C5r", self._cfg(), {"q": np.array([1.0, 0.0])}, {"q": {"d1": 1.0}},
                _anchor_pairs(), 0.9, _points(), cycle_index=2, make_post_factory=_stub_make_post,
            )
        self.assertFalse(meta2["correction_applied"])
        self.assertIsNone(engine.qdrant.upserted)  # one-shot: no second apply

    def test_postbank_conditions_constant(self):
        self.assertEqual(set(POSTBANK_CONDITIONS), {"C5", "C5s", "C5r"})


if __name__ == "__main__":
    unittest.main()
