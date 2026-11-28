"""Unit tests for W22-11 (unittest only; no pytest)."""

from __future__ import annotations

import json
import random
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import ek5_v3_stratified as ek5  # noqa: E402
import w22_11_wfs_deletion_stress as w  # noqa: E402

SMALL = w.TierConfig("t32", "stress", 32, 8, 3, 777, first_nodes=(8, 12), new_nodes_per_step=(0, 3), scc_size=(3, 16))


class TestHelpers(unittest.TestCase):
    def test_scc_membership_ring_and_selfloop(self):
        adj = {"a": {"b"}, "b": {"c"}, "c": {"a"}, "d": {"d"}, "e": {"a"}}
        self.assertEqual(w.nontrivial_scc_membership(list("abcde"), adj), {"a": 3, "b": 3, "c": 3, "d": 1})

    def test_scc_bucket(self):
        self.assertEqual([w.scc_bucket(s) for s in (0, 1, 2, 4, 8, 16, 32, 33)],
                         ["none", "self", "2", "3-4", "5-8", "9-16", "17-32", "33+"])

    def test_snapshot_scc_uses_only_dep_edges(self):
        snap = {"nodes": [{"id": "x"}, {"id": "y"}],
                "edges": [{"src": "x", "dst": "y", "type": "AMENDS"}, {"src": "y", "dst": "x", "type": "REFUTE"}]}
        self.assertEqual(w.snapshot_dv_scc(snap), {})
        snap["edges"][1]["type"] = "COPY_OF"
        self.assertEqual(w.snapshot_dv_scc(snap), {"x": 2, "y": 2})


class TestGenerator(unittest.TestCase):
    def test_stress_generator_plants_sccs_and_deletes(self):
        rng = random.Random(5)
        model = w.rc._Model(strict=True)
        withdrawn = removed = 0
        saw_scc = False
        cfg = w.TierConfig("g", "stress", 40, 12, 1, 0, first_nodes=(10, 12), scc_size=(4, 10), p_plant=1.0)
        for step in range(cfg.steps):
            b = w.gen_batch_stress(rng, model, step, cfg)
            withdrawn += len(b["withdraw"])
            removed += len(b["remove"])
            if w.nontrivial_scc_membership(model.nodes, w.model_dv_adj(model)):
                saw_scc = True
        self.assertTrue(saw_scc)
        self.assertGreater(withdrawn + removed, 0)
        self.assertLessEqual(len(model.nodes), 40)

    def test_strict_mode_never_closes_supersede_cycle(self):
        cfg = w.TierConfig("g", "stress", 24, 16, 1, 0)
        for seed in range(10):
            rng = random.Random(seed)
            model = w.rc._Model(strict=True)
            for step in range(cfg.steps):
                w.gen_batch_stress(rng, model, step, cfg)
            sup = model.edges("SUPERSEDE")
            self.assertEqual(w.nontrivial_scc_membership(model.nodes, sup), {})

    def test_baseline_tier_rejects_other_sizes(self):
        cfg = w.TierConfig("b", "baseline", 12, 2, 1, 0)
        with self.assertRaises(ValueError):
            w.run_sequence(cfg, 0, w._new_counts())


class TestComparators(unittest.TestCase):
    def _store(self):
        s = ek5.StratifiedEvidenceStore(strict_dag_supersede=False)
        s.apply_mutation(["a", "b", "c"], [("a", "b", "COPY_OF"), ("b", "a", "COPY_OF"), ("c", "a", "REFUTE")])
        return s

    def test_canonical_identical_stores_no_diff(self):
        t = w.canonical_dict(self._store())
        self.assertEqual(w.canonical_witness_diffs(t, w.canonical_dict(self._store())), [])
        self.assertEqual(json.loads(json.dumps(t)), t)

    def test_every_canonical_fault_kind_detected(self):
        t = w.canonical_dict(self._store())
        for k in range(6):
            bad = w.inject_canonical_fault(t, k)
            self.assertTrue(w.canonical_witness_diffs(bad, t), k)

    def test_canonical_pos_cyc_matches_cycle(self):
        nodes = {n["event_id"]: n for n in w.canonical_dict(self._store())["nodes"]}
        self.assertTrue(nodes["a"]["pos_cyc"] and nodes["b"]["pos_cyc"])
        self.assertFalse(nodes["c"]["pos_cyc"])
        self.assertEqual(nodes["a"]["reason"], "UNFOUNDED_CYCLE")


class TestEndToEnd(unittest.TestCase):
    def test_run_chunk_small_stress_tier(self):
        part = w.run_chunk((w.tier_dict(SMALL), 0, SMALL.sequences))
        self.assertEqual(part["sequences"], 3)
        self.assertEqual(part["checkpoints_mutation"], 3 * SMALL.steps)
        self.assertGreaterEqual(part["checkpoints_as_of"], 3)
        self.assertEqual(part["liveness"]["canonical_injected"], part["liveness"]["canonical_detected"])
        self.assertEqual(part["liveness"]["semantic_injected"], part["liveness"]["semantic_detected"])
        self.assertGreater(part["liveness"]["canonical_injected"], 0)
        self.assertEqual(json.loads(json.dumps(part)), part)

    def test_merge_counts(self):
        a = {"checkpoints": 2, "max_nodes_seen": 5, "semantic": {"val": 1}, "samples": {"semantic": [1]}}
        b = {"checkpoints": 3, "max_nodes_seen": 9, "semantic": {"val": 2, "cur": 1}, "samples": {"semantic": [2]}}
        m = w.merge_counts(w.merge_counts({}, a), b)
        self.assertEqual(m["checkpoints"], 5)
        self.assertEqual(m["max_nodes_seen"], 9)
        self.assertEqual(m["semantic"], {"val": 3, "cur": 1})
        self.assertEqual(m["samples"]["semantic"], [1, 2])

    def test_runner_smoke_writes_results(self):
        sys.path.insert(0, str(ROOT / "scripts"))
        import run_pv7_w23c_w22_11_wfs_deletion_stress as runner

        with tempfile.TemporaryDirectory() as td:
            rc = runner.main(["--out-dir", td, "--workers", "2", "--scale", "0.001"])
            self.assertEqual(rc, 0)
            res = json.loads((Path(td) / "w22_11_results.json").read_text(encoding="utf-8"))
            self.assertIn(res["verdict"]["verdict"].split("_")[0], {"INVALID", "BLOCKING", "CANONICALITY", "CLEAN"})
            self.assertTrue(all(v["ok"] for v in res["pinned_sources"].values()))
            self.assertEqual(set(res["per_tier"]), {"A0", "A1", "T1", "T2", "T3", "T4", "T5"})


if __name__ == "__main__":
    unittest.main()
