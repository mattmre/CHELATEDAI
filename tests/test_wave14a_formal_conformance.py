"""Wave-14A formal package tests: EK5 Lean reference CLI conformance, comparator liveness, nanoda manifest.

unittest-style (pytest-compatible).  Tests that need the built Lean CLI skip LOUDLY (with the build command) when it is
absent; they never silently pass.  Everything else is pure Python.
"""

from __future__ import annotations

import importlib.util
import json
import re
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
FORMAL = REPO_ROOT / "formal"


def _load_runner():
    spec = importlib.util.spec_from_file_location("run_conformance_under_test", str(FORMAL / "run_conformance.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


rc = _load_runner()
CLI = rc.cli_path()
NEED_CLI = unittest.skipUnless(
    CLI.is_file(), "ek5_ref_cli not built: run `lake build ek5_ref_cli` in formal/ (conformance tests NOT exercised)"
)


def _snap(nodes, edges, missing=(), strict=True, sid="t"):
    return {
        "snapshot_id": sid,
        "strict_dag_supersede": strict,
        "nodes": [{"id": n, "tier": t, "domain": d} for n, t, d in nodes],
        "missing": list(missing),
        "edges": [{"src": s, "dst": d, "type": t} for s, d, t in edges],
    }


class TestGeneratorAndComparator(unittest.TestCase):
    def test_sequences_are_deterministic(self):
        a, _ = rc.run_sequence(12345)
        b, _ = rc.run_sequence(12345)
        self.assertEqual([c["snapshot"] for c in a], [c["snapshot"] for c in b])
        self.assertEqual([c["inc"] for c in a], [c["inc"] for c in b])

    def test_different_seeds_differ(self):
        a, _ = rc.run_sequence(1)
        b, _ = rc.run_sequence(2)
        self.assertNotEqual([c["snapshot"] for c in a], [c["snapshot"] for c in b])

    def test_sequence_has_mutation_and_asof_checkpoints(self):
        cps, _ = rc.run_sequence(99, steps=4)
        kinds = [c["kind"] for c in cps]
        self.assertEqual(kinds.count("mutation"), 4)
        self.assertGreaterEqual(kinds.count("as_of"), 1)

    def test_comparator_detects_every_fault_kind(self):
        cps, _ = rc.run_sequence(7)
        state = cps[0]["fresh"]
        self.assertEqual(rc.compare_states(state, state), [])
        for kind in range(9):
            bad = rc.inject_fault(state, kind)
            self.assertIsNotNone(bad)
            self.assertTrue(rc.compare_states(bad, state), "fault kind %d went undetected" % kind)

    def test_oracle_matches_python_store_on_samples(self):
        for seed in range(20):
            cps, _ = rc.run_sequence(seed)
            for cp in cps:
                got = {n: v["val"] for n, v in cp["inc"]["nodes"].items()}
                self.assertEqual(got, cp["oracle"]["python_inc"], cp["id"])

    def test_missing_parent_is_forced_false_by_oracle(self):
        cps = [c for s in range(40) for c in rc.run_sequence(s)[0] if c["snapshot"]["missing"]]
        self.assertTrue(cps, "generator never produced a missing-parent snapshot")
        for cp in cps[:20]:
            for m in cp["snapshot"]["missing"]:
                self.assertEqual(cp["oracle"]["python_inc"][m], "F")


@NEED_CLI
class TestLeanReferenceCli(unittest.TestCase):
    def run_one(self, snap):
        return rc.run_lean_cli(CLI, [snap])[0]

    def by_id(self, res):
        return {n["id"]: n for n in res["nodes"]}

    def test_positive_cycle_is_unfounded_false(self):
        res = self.run_one(_snap([("a", 3, "general"), ("b", 3, "general")], [("a", "b", "DERIVES_FROM"), ("b", "a", "DERIVES_FROM")]))
        self.assertTrue(res["ok"])
        for n in self.by_id(res).values():
            self.assertEqual((n["val"], n["reason"], n["pos_cyc"]), ("F", "UNFOUNDED_CYCLE", True))

    def test_mutual_refute_is_undetermined(self):
        res = self.run_one(_snap([("a", 3, "general"), ("b", 3, "general")], [("a", "b", "REFUTE"), ("b", "a", "REFUTE")]))
        for n in self.by_id(res).values():
            self.assertEqual((n["val"], n["eligibility"]), ("U", "CONTESTED"))

    def test_missing_parent_is_false(self):
        res = self.run_one(_snap([("a", 3, "general")], [], missing=["a"]))
        n = self.by_id(res)["a"]
        self.assertEqual((n["val"], n["reason"], n["eligibility"]), ("F", "MISSING_MANDATORY_PARENT", "AUDIT_ONLY"))

    def test_supersede_marks_old_noncurrent(self):
        res = self.run_one(_snap([("old", 3, "general"), ("new", 3, "general")], [("new", "old", "SUPERSEDE")]))
        ids = self.by_id(res)
        self.assertEqual(ids["old"]["cur"], "F")
        self.assertEqual(ids["new"]["cur"], "T")

    def test_strict_mode_rejects_supersede_cycle_nonstrict_quarantines(self):
        nodes = [("x", 3, "general"), ("y", 3, "general")]
        edges = [("x", "y", "SUPERSEDE"), ("y", "x", "SUPERSEDE")]
        self.assertFalse(self.run_one(_snap(nodes, edges, strict=True))["ok"])
        res = self.run_one(_snap(nodes, edges, strict=False))
        self.assertTrue(res["ok"])
        self.assertEqual(res["quarantined_supersede_cycles"], [["x", "y"]])

    def test_malformed_input_is_error_not_crash(self):
        res = self.run_one(_snap([("a", 3, "general")], [("a", "a", "NOT_A_TYPE")]))
        self.assertFalse(res["ok"])
        self.assertIn("error", res)

    def test_duplicate_node_id_is_error(self):
        res = self.run_one(_snap([("a", 3, "general"), ("a", 2, "general")], []))
        self.assertFalse(res["ok"])

    def test_cli_is_deterministic_and_batch_order_independent(self):
        cps = [c for s in range(5) for c in rc.run_sequence(s)[0]]
        snaps = [c["snapshot"] for c in cps]
        first = rc.run_lean_cli(CLI, snaps)
        again = rc.run_lean_cli(CLI, list(reversed(snaps)))
        self.assertEqual(first, list(reversed(again)))


@NEED_CLI
class TestDifferentialConformance(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = rc.run_conformance(sequences=1000, steps=rc.DEFAULT_STEPS, seed=rc.DEFAULT_SEED, cli=CLI)

    def test_all_gates_pass(self):
        failed = [g for g, ok in self.report["gates"].items() if not ok]
        self.assertEqual(failed, [], json.dumps(self.report["divergence_samples"], indent=1)[:2000])
        self.assertTrue(self.report["ok"])

    def test_scale_requirements(self):
        self.assertGreaterEqual(self.report["sequences"], 1000)
        self.assertGreaterEqual(self.report["checkpoints"]["total"], 5000)
        self.assertGreaterEqual(self.report["checkpoints"]["as_of_missing_parent"], 50)

    def test_lean_matches_fresh_python_on_every_field(self):
        self.assertEqual(self.report["divergences"]["lean_vs_fresh"]["total"], 0)

    def test_oracle_agrees_everywhere(self):
        self.assertEqual(sum(self.report["oracle"]["mismatch_counts"].values()), 0)

    def test_comparator_was_live(self):
        live = self.report["comparator_liveness"]
        self.assertGreater(live["faults_injected"], 100)
        self.assertEqual(live["faults_injected"], live["faults_detected"])

    def test_report_serialises_with_lf_endings(self):
        out = FORMAL / "conformance_report.json"
        rc.write_report(self.report, out)
        raw = out.read_bytes()
        self.assertNotIn(b"\r", raw)
        self.assertEqual(json.loads(raw.decode("utf-8"))["schema_version"], rc.SCHEMA_VERSION)


class TestManifestAndLeanSources(unittest.TestCase):
    def test_manifest_is_valid_for_verify_gate(self):
        m = rc.load_manifest()
        self.assertTrue(Path(m["nanoda_bin"]).name.startswith("nanoda_bin"))
        self.assertEqual(m["nanoda_args"], [".lake/nanoda/nanoda_config.json"])
        cfg = m["nanoda_config"]
        self.assertTrue(cfg["unpermitted_axiom_hard_error"])
        self.assertEqual(sorted(cfg["permitted_axioms"]), ["Classical.choice", "Quot.sound", "propext"])
        self.assertEqual(cfg["export_file_path"], m["export_file"])

    def test_manifest_declarations_exist_in_lean_source(self):
        decls = rc.theorem_declarations()
        for d in rc.load_manifest()["export_declarations"]:
            if d.split(".")[-1] in ("theorem3_ek5_cyclic_wfs", "evalStratum2Wfs_sound", "stratum1_noninterference", "theorem3_witness_nonvacuous"):
                self.assertIn(d, decls)

    def test_lean_sources_have_no_banned_constructs(self):
        banned = re.compile(r"\b(sorry|admit|native_decide|axiom)\b")
        for p in (FORMAL / "ChelatedFormal" / "Theorem3_EK5_CyclicWFS.lean", FORMAL / "Ek5RefCli.lean"):
            text = p.read_text(encoding="utf-8")
            # blank out block/doc comments (keeping newlines so line numbers stay right), then line comments
            text = re.sub(r"/-.*?-/", lambda m: re.sub(r"[^\n]", " ", m.group(0)), text, flags=re.S)
            for i, line in enumerate(text.splitlines(), 1):
                code = line.split("--", 1)[0]
                self.assertIsNone(banned.search(code), "%s:%d: %s" % (p.name, i, line))

    def test_owned_files_use_lf_endings(self):
        for p in (
            FORMAL / "run_conformance.py",
            FORMAL / "nanoda_manifest.json",
            FORMAL / "Ek5RefCli.lean",
            FORMAL / "ChelatedFormal" / "Theorem3_EK5_CyclicWFS.lean",
            Path(__file__),
        ):
            self.assertNotIn(b"\r", p.read_bytes(), p.name)


if __name__ == "__main__":
    unittest.main()
