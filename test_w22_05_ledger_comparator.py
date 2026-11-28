"""Unit tests for w22_05_ledger_comparator (unittest only; no network, no models)."""

from __future__ import annotations

import ast
import unittest
from pathlib import Path

import w22_05_ledger_comparator as cmp
from chelated_raw_text_exocortex import RawDocumentEnvelope


def _doc(did, channel, ts):
    return RawDocumentEnvelope(doc_id=did, text=f"text {did}", published_ts=ts, ingested_ts=ts, source_channel=channel, tenant="prod", domain="cve")


def _item(conflict, ek5_ok, led_ok, ev=("a",), lv=("a",), ec=True, lc=True):
    return {
        "is_conflict": conflict,
        "ek5_verdict": ev,
        "ledger_verdict": lv,
        "ek5_correct": ek5_ok,
        "ledger_correct": led_ok,
        "ek5_covered": ec,
        "ledger_covered": lc,
    }


class TestArenaRoles(unittest.TestCase):
    def setUp(self):
        self.case = {
            "case_id": "c1",
            "regime": "supersession_under_echo_flood",
            "gold": "1.2.3",
            "stale_or_false_values": ["1.0.0"],
            "subject": "CVE-2024-0001",
        }

    def test_supersede_roles(self):
        self.assertEqual(cmp.arena_doc_role(self.case, "c1_d12_official_supersede"), ("SUPERSEDE", "1.2.3"))
        self.assertEqual(cmp.arena_doc_role(self.case, "c1_echo_03"), ("ASSERT", "1.0.0"))
        self.assertEqual(cmp.arena_doc_role(self.case, "c1_d14_late_archive"), ("ASSERT", "1.0.0"))

    def test_unknown_role_raises(self):
        with self.assertRaises(ValueError):
            cmp.arena_doc_role(self.case, "c1_mystery")
        with self.assertRaises(ValueError):
            cmp.arena_doc_role(self.case, "c2_echo_01")

    def test_contested_echo_b_before_echo(self):
        case = dict(self.case, regime="mutual_peer_conflict_abstain", gold="ABSTAIN", stale_or_false_values=["x", "y"])
        self.assertEqual(cmp.arena_doc_role(case, "c1_echo_b_02"), ("ASSERT", "y"))
        self.assertEqual(cmp.arena_doc_role(case, "c1_echo_02"), ("ASSERT", "x"))
        self.assertEqual(cmp.arena_doc_role(case, "c1_d00_party_a"), ("REFUTE", "x"))


class TestArenaCaseEndToEnd(unittest.TestCase):
    def test_supersession_case_both_arms_pick_official(self):
        case = {
            "case_id": "c1",
            "domain": "cve",
            "regime": "supersession_under_echo_flood",
            "gold": "1.2.3",
            "stale_or_false_values": ["1.0.0"],
            "subject": "CVE-2024-0001",
            "query_text": "What is the safe version for CVE-2024-0001?",
        }
        docs = [
            _doc("c1_d00_prelim", "vendor_advisory", 1),
            _doc("c1_echo_01", "community_forum", 2),
            _doc("c1_echo_02", "social_repost", 3),
            _doc("c1_d12_official_supersede", "official_nvd_advisory", 12),
            _doc("c1_d13_late_archive", "unverified_blog", 13),
        ]
        row = cmp.run_arena_case(case, docs)
        self.assertEqual(row["gold"], ["1.2.3"])
        self.assertEqual(row["ledger_verdict"], ["1.2.3"])
        self.assertTrue(row["ledger_correct"])
        self.assertEqual(row["ek5_correct"], row["ek5_verdict"] == ["1.2.3"])

    def test_contested_case_ledger_disputes(self):
        case = {
            "case_id": "c2",
            "domain": "cve",
            "regime": "mutual_peer_conflict_abstain",
            "gold": "ABSTAIN",
            "stale_or_false_values": ["exploitable", "intended"],
            "subject": "CVE-2026-0002",
            "query_text": "What is the verdict for CVE-2026-0002?",
        }
        docs = [
            _doc("c2_d00_party_a", "independent_security_lab", 1),
            _doc("c2_d01_party_b", "vendor_advisory", 2),
            _doc("c2_echo_01", "community_forum", 3),
            _doc("c2_echo_b_01", "community_forum", 4),
        ]
        row = cmp.run_arena_case(case, docs)
        self.assertEqual(row["ledger_status"], "DISPUTED")
        self.assertEqual(row["ledger_verdict"], [cmp.ABSTAIN])


class TestMetricsAndRule(unittest.TestCase):
    def test_harm_undefined_when_subset_empty(self):
        self.assertIsNone(cmp.harm_at(None, 0.0, 0.05))
        self.assertAlmostEqual(cmp.harm_at(1.0, 0.0, 0.25), 0.25)

    def test_retire_when_identical(self):
        items = [_item(True, True, True, (cmp.ABSTAIN,), (cmp.ABSTAIN,)), _item(False, True, True)]
        m = cmp.paired_metrics(items)
        d = cmp.kill_rule_decision({"x": m, "pooled": m}, {"x": True})
        self.assertEqual(d["verdict"], "RETIRE_EK5_CLEAN_GRAPH_PREMIUM")

    def test_advance_on_single_ek5_conflict_win(self):
        items = [_item(True, True, False, (cmp.ABSTAIN,), ("a",))] + [_item(False, True, True)] * 200
        m = cmp.paired_metrics(items)
        self.assertEqual(m["ek5_conflict_wins"], 1)
        d = cmp.kill_rule_decision({"x": m, "pooled": m}, {"x": True})
        self.assertEqual(d["verdict"], "ADVANCE_EK5_RETAINS_MEASURED_PREMIUM")
        self.assertFalse(d["checks"]["x"]["b_zero_ek5_conflict_wins"])

    def test_undefined_harm_blocks_retire(self):
        m = cmp.paired_metrics([_item(False, True, True)])
        d = cmp.kill_rule_decision({"x": m, "pooled": m}, {"x": True})
        self.assertTrue(d["verdict"].startswith("ADVANCE"))

    def test_failed_gate_partial_suffix(self):
        items = [_item(True, True, True, (cmp.ABSTAIN,), (cmp.ABSTAIN,)), _item(False, True, True)]
        m = cmp.paired_metrics(items)
        d = cmp.kill_rule_decision({"x": m, "y": m, "pooled": m}, {"x": True, "y": False})
        self.assertEqual(d["verdict"], "RETIRE_EK5_CLEAN_GRAPH_PREMIUM_PARTIAL_INPUTS")
        self.assertEqual(d["excluded_corpora"], ["y"])

    def test_coverage_drop_blocks_retire(self):
        items = [_item(True, True, True, (cmp.ABSTAIN,), (cmp.ABSTAIN,))] + [_item(False, False, False, ("a",), (), True, False)] * 10
        m = cmp.paired_metrics(items)
        self.assertFalse(cmp.kill_rule_checks(m)["c_coverage_equivalent"])


class TestCensus(unittest.TestCase):
    def test_mutual_refute_is_negation_cycle(self):
        c = cmp.negation_cycle_census(["a", "b", "c"], [("a", "b", "REFUTE"), ("b", "a", "REFUTE"), ("c", "a", "COPY_OF")])
        self.assertEqual(c["negation_cycle_sccs"], 1)
        self.assertEqual(c["mutual_refute_2cycles"], 1)
        self.assertEqual(c["nodes_in_negation_cycles"], 2)

    def test_positive_cycle_not_counted(self):
        c = cmp.negation_cycle_census(["a", "b"], [("a", "b", "COPY_OF"), ("b", "a", "COPY_OF")])
        self.assertEqual(c["negation_cycle_sccs"], 0)
        self.assertEqual(c["purely_positive_cycle_sccs"], 1)

    def test_dag_has_no_cycles(self):
        c = cmp.negation_cycle_census(["a", "b", "c"], [("a", "b", "SUPERSEDE"), ("b", "c", "REFUTE")])
        self.assertEqual(c["nontrivial_sccs"], 0)

    def test_negative_self_loop_counts(self):
        c = cmp.negation_cycle_census(["a"], [("a", "a", "REFUTE")])
        self.assertEqual(c["negation_cycle_sccs"], 1)

    def test_long_chain_iterative(self):
        n = 5000
        edges = [(str(i), str(i + 1), "COPY_OF") for i in range(n)] + [(str(n), "0", "SUPERSEDE")]
        c = cmp.negation_cycle_census([str(i) for i in range(n + 1)], edges)
        self.assertEqual(c["negation_cycle_sccs"], 1)
        self.assertEqual(c["max_negation_scc_size"], n + 1)


class TestLedgerDivergence(unittest.TestCase):
    def test_ordered_rebuild_matches_incremental(self):
        from chelated_pv7_spine.challenger_ledger import LedgerRecordEnvelope

        ops = [
            ("record", LedgerRecordEnvelope("t", "r1", "e", 1, 0, None, 1, "ASSERT", claim_value="x")),
            ("record", LedgerRecordEnvelope("t", "r2", "e", 2, 0, None, 2, "ASSERT", claim_value="y")),
            ("record", LedgerRecordEnvelope("t", "r2", "e", 2, 0, None, 3, "WITHDRAW")),
        ]
        out = cmp.ledger_rebuild_divergence(ops, [("t", "e", 5, 1), ("t", "e", 5, 2), ("t", "e", 5, 3)])
        self.assertEqual(out["inc_vs_fresh_ordered"]["digest_mismatches"], 0)
        self.assertEqual(out["queries"], 3)


class TestIndependence(unittest.TestCase):
    def test_challenger_ledger_imports_no_ek5(self):
        src = Path(cmp.__file__).resolve().parent / "chelated_pv7_spine" / "challenger_ledger.py"
        tree = ast.parse(src.read_text(encoding="utf-8"))
        mods = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                mods.update(a.name for a in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                mods.add(node.module)
        self.assertFalse(any("ek5" in m or "exocortex" in m or "epistemic" in m for m in mods), mods)


if __name__ == "__main__":
    unittest.main()
