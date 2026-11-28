"""Unit tests for W22-07 blinded hidden-copy twins and priced provenance-check policies (unittest only)."""

import importlib.util
import os
import sys
import unittest
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import w22_07_copy_twins as W  # noqa: E402

REPO = Path(__file__).resolve().parent
SNAPSHOTS = REPO / "artifacts" / "pv7_w22e_b55_wild_live" / "frozen_snapshots"
SEED = W.FIXTURE_SEEDS[0]


def _row(label, decision, checks=0, pair="p0"):
    truth = W.SUPPORT if label == "I" else W.RETRACT
    return {"label": label, "decision": decision, "truth": truth, "checks": checks, "pair_key": pair}


def _summary(harm, coverage):
    return {"harm": {str(p): harm for p in W.PREVALENCES}, "coverage": {str(p): coverage for p in W.PREVALENCES}}


@unittest.skipUnless(SNAPSHOTS.is_dir(), "bundle #55 frozen snapshots not present")
class TestTwinFixture(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.roots = W.load_real_roots(SNAPSHOTS)
        cls.pairs0 = W.build_twin_pairs(cls.roots, SEED, 0.0)

    def test_one_pair_per_real_root_and_even_split(self):
        self.assertEqual(len(self.roots), 240)
        self.assertEqual(len(self.pairs0), 240)
        self.assertEqual(sum(p.split == "dev" for p in self.pairs0), 120)

    def test_deterministic(self):
        again = W.build_twin_pairs(self.roots, SEED, 0.0)
        self.assertEqual(
            [W.serialize_public(m) for p in self.pairs0 for m in p.members],
            [W.serialize_public(m) for p in again for m in p.members],
        )

    def test_oracle_matches_constructed_label(self):
        for pair in self.pairs0:
            labels = sorted(m.label for m in pair.members)
            self.assertEqual(labels, ["C", "I"])
            for member in pair.members:
                self.assertEqual(W.oracle_truth(member), member.truth)
                self.assertEqual(W.oracle_truth(member, W.PRE_CUT), W.SUPPORT)

    def test_exact_twins_have_identical_public_text_at_grade_zero(self):
        for pair in self.pairs0:
            c, i = pair.members
            self.assertEqual([w.text for w in c.witnesses], [w.text for w in i.witnesses])
            self.assertEqual([w.source for w in c.witnesses], [w.source for w in i.witnesses])
            self.assertEqual([w.declared for w in c.witnesses], [w.declared for w in i.witnesses])
            self.assertFalse(pair.leaky)

    def test_hidden_configuration_matches_label(self):
        for pair in self.pairs0:
            for member in pair.members:
                self.assertEqual(len(member.hidden), len(member.undeclared))
                if member.label == "C":
                    self.assertEqual(sum(member.hidden), 0)
                else:
                    self.assertEqual(sum(member.hidden), pair.j)

    def test_no_role_words_at_grade_zero(self):
        for pair in self.pairs0:
            for member in pair.members:
                self.assertEqual(W.role_word_violations(member), [])

    def test_leakage_grades_nest_and_keep_structure(self):
        leaky = {}
        for grade in W.LEAK_GRADES:
            pairs = W.build_twin_pairs(self.roots, SEED, grade)
            leaky[grade] = {p.pair_id for p in pairs if p.leaky}
            self.assertEqual([(p.k, p.d, p.j, p.split) for p in pairs], [(p.k, p.d, p.j, p.split) for p in self.pairs0])
        self.assertFalse(leaky[0.0])
        self.assertLessEqual(leaky[0.25], leaky[0.5])
        self.assertLessEqual(leaky[0.5], leaky[1.0])
        self.assertEqual(len(leaky[1.0]), 240)

    def test_grade_one_carries_cue(self):
        pairs = W.build_twin_pairs(self.roots, SEED, 1.0)
        cue_hits = sum(
            w.text.startswith(W.LEAK_CUE_PREFIX)
            for p in pairs
            for m in p.members
            if m.label == "C"
            for w in m.witnesses
            if not w.declared
        )
        self.assertGreater(cue_hits, 0)

    def test_ledger_without_checks_supports_both_twins(self):
        for pair in self.pairs0[:20]:
            decisions = {m.label: W.policy_ledger(m, 0).decision for m in pair.members}
            self.assertEqual(decisions, {"C": W.SUPPORT, "I": W.SUPPORT})

    def test_ledger_check_exposes_single_hidden_copy(self):
        found = False
        for pair in self.pairs0:
            c = next(m for m in pair.members if m.label == "C")
            if len(c.undeclared) == 1:
                outcome = W.policy_ledger(c, 1)
                self.assertEqual(outcome.decision, W.RETRACT)
                self.assertEqual(outcome.checks, 1)
                found = True
                break
        self.assertTrue(found)

    def test_family_level_classifier_is_chance_on_exact_twins(self):
        result = W.family_level_leakage(self.pairs0[:80])
        self.assertLessEqual(result["balanced_accuracy"], W.LEAKAGE_THRESHOLD)


class TestPolicies(unittest.TestCase):
    def setUp(self):
        self.model = W.CopyPosterior(
            prior={1: {(0,): 0.5, (1,): 0.5}, 2: {(0, 0): 0.5, (0, 1): 0.25, (1, 0): 0.25, (1, 1): 0.0}, 3: {}},
            bin_lik={0: [0.2] * 5, 1: [0.2] * 5},
        )

    def test_posterior_normalises_and_conditions(self):
        post = self.model.posterior((2, 2), {})
        self.assertAlmostEqual(sum(post.values()), 1.0)
        post = self.model.posterior((2, 2), {0: 1})
        self.assertAlmostEqual(W._p_support(post), 1.0)
        post = self.model.posterior((2, 2), {0: 0})
        self.assertAlmostEqual(W._p_support(post), 1 / 3)

    def test_stop_costs_and_tie_order(self):
        costs = W.stop_costs(0.5, W.PRIMARY_PRICES, True)
        self.assertEqual(W._argmin_decision(costs), W.ABSTAIN)
        costs = W.stop_costs(0.5, {"w_fs": 1.0, "w_fr": 1.0, "w_a": 0.9, "c": 0.1}, False)
        self.assertEqual(W._argmin_decision(costs), W.SUPPORT)

    def test_td_value_prefers_stopping_when_checks_expensive(self):
        prices = dict(W.PRIMARY_PRICES, c=5.0)
        value, action = W._td_value(self.model, (2, 2), {}, 2, prices, {})
        self.assertIsNone(action)
        self.assertAlmostEqual(value, 0.35)

    def test_td_value_checks_when_cheap(self):
        prices = dict(W.PRIMARY_PRICES, c=0.01)
        value, action = W._td_value(self.model, (2,), {}, 1, prices, {})
        self.assertEqual(action, 0)
        self.assertAlmostEqual(value, 0.01)

    def test_item_harm(self):
        p = W.PRIMARY_PRICES
        self.assertEqual(W.item_harm(W.SUPPORT, W.SUPPORT, 0, p), 0.0)
        self.assertAlmostEqual(W.item_harm(W.SUPPORT, W.RETRACT, 1, p), 1.1)
        self.assertAlmostEqual(W.item_harm(W.RETRACT, W.SUPPORT, 0, p), 1.0)
        self.assertAlmostEqual(W.item_harm(W.ABSTAIN, W.SUPPORT, 2, p), 0.55)

    def test_summarize_prevalence_weighting(self):
        rows = [_row("C", W.SUPPORT, pair="a"), _row("I", W.SUPPORT, pair="a"),
                _row("C", W.RETRACT, pair="b"), _row("I", W.ABSTAIN, pair="b")]
        s = W.summarize(rows, W.PRIMARY_PRICES)
        self.assertAlmostEqual(s["per_label"]["C"]["mean_harm"], 0.5)
        self.assertAlmostEqual(s["per_label"]["I"]["mean_harm"], 0.175)
        self.assertAlmostEqual(s["harm"]["0.25"], 0.25 * 0.5 + 0.75 * 0.175)
        self.assertAlmostEqual(s["coverage"]["0.5"], 0.5 * 1.0 + 0.5 * 0.5)
        self.assertEqual(s["pairs_both_correct"], 0)
        self.assertEqual(s["pairs_same_decision"], 1)

    def test_kill_rule(self):
        base = {arm: _summary(0.5, 1.0) for arm in W.CONVENTIONAL_ARMS}
        base["A6"] = _summary(0.2, 1.0)
        self.assertEqual(W.kill_rule_decision(base, True)["verdict"], "ADVANCE_T_D")
        base["A5"] = _summary(0.204, 0.996)
        decision = W.kill_rule_decision(base, True)
        self.assertEqual(decision["verdict"], "KILL_T_D")
        self.assertEqual(decision["matching_conventional_arms"], ["A5"])
        base["A5"] = _summary(0.2, 0.99)
        self.assertEqual(W.kill_rule_decision(base, True)["verdict"], "ADVANCE_T_D")
        self.assertEqual(W.kill_rule_decision(base, False)["verdict"], "FIXTURE_INVALID_LEAKAGE")

    def test_bootstrap_is_deterministic(self):
        rows = {arm: [] for arm in ("A6",) + W.CONVENTIONAL_ARMS}
        for n in range(10):
            for arm in rows:
                dec = W.SUPPORT if (arm == "A6" and n % 2) else W.RETRACT
                rows[arm].append(_row("C", dec, pair=f"p{n}"))
                rows[arm].append(_row("I", W.SUPPORT, pair=f"p{n}"))
        a = W.cluster_bootstrap_delta(rows, W.PRIMARY_PRICES)
        b = W.cluster_bootstrap_delta(rows, W.PRIMARY_PRICES)
        self.assertEqual(a, b)
        self.assertAlmostEqual(a["point"], 0.25)
        self.assertLessEqual(a["ci95_low"], a["point"])
        self.assertGreaterEqual(a["ci95_high"], a["point"])

    def test_text_helpers(self):
        self.assertAlmostEqual(W.tf_cosine("a b c", "c b a"), 1.0)
        self.assertEqual(W.tf_cosine("", "x"), 0.0)
        self.assertEqual(W.cosine_bin(1.0), 4)
        self.assertEqual(W.cosine_bin(0.0), 0)


class TestLeakageClassifier(unittest.TestCase):
    def test_separable_text_is_detected(self):
        texts = [f"alpha item {i}" if i % 2 else f"beta item {i}" for i in range(60)]
        labels = [i % 2 for i in range(60)]
        groups = [f"g{i // 2}" for i in range(60)]
        self.assertGreater(W.grouped_text_classifier(texts, labels, groups)["balanced_accuracy"], 0.95)

    def test_identical_text_is_chance(self):
        texts = ["same words here"] * 60
        labels = [i % 2 for i in range(60)]
        groups = [f"g{i // 2}" for i in range(60)]
        self.assertLessEqual(W.grouped_text_classifier(texts, labels, groups)["balanced_accuracy"], W.LEAKAGE_THRESHOLD)


@unittest.skipUnless(SNAPSHOTS.is_dir(), "bundle #55 frozen snapshots not present")
class TestRunnerLegacyAudit(unittest.TestCase):
    def test_legacy_55_fixture_is_leaky(self):
        spec = importlib.util.spec_from_file_location("w22_07_runner", REPO / "scripts" / "run_pv7_w23b_w22_07_copy_twins.py")
        runner = importlib.util.module_from_spec(spec)
        sys.modules["w22_07_runner"] = runner
        spec.loader.exec_module(runner)
        self.assertGreater(runner.legacy_55()["balanced_accuracy"], W.LEAKAGE_THRESHOLD)


if __name__ == "__main__":
    unittest.main()
