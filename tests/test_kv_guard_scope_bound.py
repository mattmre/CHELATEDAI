"""Unit tests for kv_guard_scope_bound (W22-10, #60 half). unittest only."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import kv_guard_scope_bound as kg  # noqa: E402
from chelated_ssd_kv_repo_cache import (  # noqa: E402
    CacheStats,
    SSDKVCache,
    StaleWeightRevisionError,
    _prompt_chain,
    build_model,
    build_repo,
    contract_for,
    select_repo_names,
    serve_chelated,
)
from ek5_v3_stratified import StratifiedEvent, StratifiedEvidenceStore  # noqa: E402


def _case(seed: int = 6001):
    name = select_repo_names(seed, 2)[0]
    files = build_repo(seed, name, 8).commits[5].files
    return files[:-1], files[-1]


class ForwardParityTests(unittest.TestCase):
    def test_traced_forward_matches_tinymodel_bitwise(self):
        m = build_model(6001)
        prefix, suffix = _case()
        run = kg.prefill_blocks(m, [b.tokens for b in prefix])
        past = None
        for b in prefix:
            _, kv = m.forward(b.tokens, past=past)
            past = kg.cat_kv(past, kv)
        for layer in range(2):
            self.assertTrue(np.array_equal(run.kv[layer][0], past[layer][0]))
            self.assertTrue(np.array_equal(run.kv[layer][1], past[layer][1]))
        lg_a, _ = m.forward(suffix.tokens, past=past)
        lg_b, _, _ = kg.forward_traced(m, suffix.tokens, past=past)
        self.assertTrue(np.array_equal(lg_a, lg_b))

    def test_disk_block_equals_instrumented_prefill_and_guard_refuses_revision(self):
        m = build_model(6001)
        prefix, _ = _case()
        with tempfile.TemporaryDirectory() as td:
            store = StratifiedEvidenceStore()
            store.apply_mutation([StratifiedEvent(f.version_id, authority_tier=1) for f in prefix])
            cache = SSDKVCache(Path(td), contract_for(m), store)
            serve_chelated(m, cache, prefix, CacheStats())
            chain = _prompt_chain(prefix)
            stale = None
            for key, _s, _e, ptoks in chain:
                stale = kg.cat_kv(stale, kg.read_block_unchecked(cache._path(key), ptoks))
            run = kg.prefill_blocks(m, [b.tokens for b in prefix])
            for layer in range(2):
                self.assertTrue(np.array_equal(stale[layer][0], run.kv[layer][0]))
            with self.assertRaises(StaleWeightRevisionError):
                cache.get(chain[0][0], chain[0][3], contract_for(m.with_adapter(77)))
            with self.assertRaises(ValueError):
                kg.read_block_unchecked(cache._path(chain[0][0]), chain[0][3][:-1])


class BoundTests(unittest.TestCase):
    def _measure(self, receiver, sender=None):
        sender = sender or build_model(6001)
        prefix, suffix = _case()
        s_run = kg.prefill_blocks(sender, [b.tokens for b in prefix])
        r_run = kg.prefill_blocks(receiver, [b.tokens for b in prefix])
        true, _, _ = kg.forward_traced(receiver, suffix.tokens, past=r_run.kv)
        obs, _, tr = kg.forward_traced(receiver, suffix.tokens, past=s_run.kv)
        toks = np.concatenate([b.tokens for b in prefix])
        B, _, flops = kg.receiver_activation_bound(kg.revision_norms(sender, receiver), toks, s_run.stats, s_run.kv, tr)
        return B, kg.max_abs(obs, true), flops

    def test_identity_bound_is_exactly_zero(self):
        B, err, _ = self._measure(build_model(6001))
        self.assertEqual(B, 0.0)
        self.assertEqual(err, 0.0)

    def test_last_layer_adapter_is_exact_reuse(self):
        sender = build_model(6001)
        receiver = sender.with_adapter(77)
        B, err, _ = self._measure(receiver, sender)
        self.assertEqual(B, 0.0)
        self.assertEqual(err, 0.0)
        prefix, _ = _case()
        self.assertTrue(kg.dependency_cone_unchanged(sender, receiver, np.concatenate([b.tokens for b in prefix])))

    def test_bound_is_sound_on_perturbations(self):
        sender = build_model(6001)
        for receiver in (
            kg.global_perturbation(sender, 1e-6, 1),
            kg.global_perturbation(sender, 1e-2, 2),
            kg.layer0_adapter(sender, 3),
            kg.fp16_round(sender),
            kg.cross_model(7001),
            sender.patched_base(),
        ):
            B, err, flops = self._measure(receiver, sender)
            self.assertTrue(kg.bound_is_sound(err, B), (B, err))
            self.assertGreater(B, 0.0)
            self.assertGreater(flops, 0)

    def test_cone_rule_detects_in_cone_change(self):
        sender = build_model(6001)
        toks = np.arange(64)
        self.assertFalse(kg.dependency_cone_unchanged(sender, kg.layer0_adapter(sender, 1), toks))
        self.assertFalse(kg.dependency_cone_unchanged(sender, sender.patched_base(), toks))
        self.assertTrue(kg.dependency_cone_unchanged(sender, sender.patched_base(), np.arange(1, 64)))

    def test_soft_lemma_and_soundness_slack(self):
        self.assertEqual(kg._soft(0.0), 0.0)
        self.assertEqual(kg._soft(10.0), 2.0)
        self.assertTrue(kg.bound_is_sound(1.0, 1.0))
        self.assertFalse(kg.bound_is_sound(1.0 + 1e-6, 1.0))

    def test_softmax_lemma_numerically(self):
        rng = np.random.default_rng(0)
        for _ in range(200):
            s = rng.standard_normal(30) * 5
            delta = float(rng.uniform(0, 0.5))
            s2 = s + rng.uniform(-delta, delta, 30)
            p = np.exp(s - s.max())
            p /= p.sum()
            p2 = np.exp(s2 - s2.max())
            p2 /= p2.sum()
            self.assertLessEqual(np.abs(p2 - p).sum(), kg._soft(delta) + 1e-12)


class BaselineTests(unittest.TestCase):
    def test_tlr_variants_exact_under_identity(self):
        m = build_model(6001)
        prefix, _ = _case()
        run = kg.prefill_blocks(m, [b.tokens for b in prefix])
        toks = np.concatenate([b.tokens for b in prefix])
        kv_h = kg.tlr_h_kv(m, run.kv, run.h_layer_inputs[1])
        kv_r = kg.tlr_r_kv(m, run.kv, toks)
        for kv in (kv_h, kv_r):
            self.assertLess(np.abs(kv[1][0] - run.kv[1][0]).max(), 1e-10)
            self.assertLess(np.abs(kv[1][1] - run.kv[1][1]).max(), 1e-10)

    def test_flop_model_ordering(self):
        for P in (50, 150, 300):
            self.assertLess(kg.flops_tlr_h(P), kg.flops_tlr_r(P))
            self.assertLess(kg.flops_tlr_r(P), kg.flops_full(P))

    def test_revisions_change_expected_weights(self):
        m = build_model(6001)
        self.assertTrue(np.array_equal(kg.fp16_round(m).wq[0], m.wq[0].astype(np.float16).astype(np.float64)))
        g = kg.global_perturbation(m, 1e-3, 1)
        rel = np.linalg.norm(g.wk[1] - m.wk[1]) / np.linalg.norm(m.wk[1])
        self.assertAlmostEqual(rel, 1e-3, places=9)
        self.assertTrue(np.array_equal(m.wo[0], build_model(6001).wo[0]))  # no aliasing mutation


if __name__ == "__main__":
    unittest.main()
