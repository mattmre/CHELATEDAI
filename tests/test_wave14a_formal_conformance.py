"""Wave-14A formal package tests: EK5 Lean reference CLI conformance, comparator liveness, nanoda manifest.

unittest-style (pytest-compatible).  Tests that need the built Lean CLI skip LOUDLY (with the build command) when it is
absent; they never silently pass.  Everything else is pure Python.
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import os
import re
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

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
                self.assertEqual(got, cp["oracle"], cp["id"])
                fresh = {n: v["val"] for n, v in cp["fresh"]["nodes"].items()}
                self.assertEqual(fresh, cp["oracle"], cp["id"])

    def test_missing_parent_is_forced_false_by_oracle(self):
        cps = [c for s in range(40) for c in rc.run_sequence(s)[0] if c["snapshot"]["missing"]]
        self.assertTrue(cps, "generator never produced a missing-parent snapshot")
        for cp in cps[:20]:
            for m in cp["snapshot"]["missing"]:
                self.assertEqual(cp["oracle"][m], "F")


class TestIndependentSnapshotDerivation(unittest.TestCase):
    """The snapshot is derived from _Model's recorded history, not from the incremental store's tables."""

    def test_model_snapshot_equals_incremental_materialization(self):
        checked = 0
        for seed in range(30):
            cps, _ = rc.run_sequence(seed)
            for cp in cps:
                self.assertTrue(cp["inc_snapshot_matches_model"], cp["id"])
                self.assertIsNone(cp["inc_snapshot"], cp["id"])
                checked += 1
        self.assertGreater(checked, 100)

    def test_materialization_comparator_detects_every_injected_fault(self):
        cps, _ = rc.run_sequence(11)
        snap = cps[0]["snapshot"]
        self.assertTrue(rc.snapshots_equal(snap, snap))
        for kind in range(5):
            bad = rc.inject_snapshot_fault(snap, kind)
            self.assertIsNotNone(bad)
            self.assertFalse(rc.snapshots_equal(bad, snap), "snapshot fault kind %d went undetected" % kind)

    def test_snapshot_is_not_read_from_the_store(self):
        # Corrupt the store's own materialization: the model-derived snapshot is unaffected, the gate flags it.
        cps, _ = rc.run_sequence(5)
        snap = cps[-1]["snapshot"]
        corrupted = rc.inject_snapshot_fault(snap, 1)
        self.assertFalse(rc.snapshots_equal(corrupted, snap))

    def test_oracle_uses_raw_snapshot_only(self):
        # a REFUTE from a tier-1 node cannot defeat a tier-3 node (authority-qualified refs), computed from the snapshot
        snap = _snap([("hi", 3, "general"), ("lo", 1, "general")], [("lo", "hi", "REFUTE")])
        self.assertEqual(rc.snapshot_refs(snap)["hi"], set())
        self.assertEqual(rc.oracle_vals(snap)["hi"], "T")
        # a tier-3 REFUTE of a tier-1 node does count
        snap2 = _snap([("hi", 3, "general"), ("lo", 1, "general")], [("hi", "lo", "REFUTE")])
        self.assertEqual(rc.snapshot_refs(snap2)["lo"], {"hi"})
        self.assertEqual(rc.oracle_vals(snap2)["lo"], "F")

    def test_oracle_cross_domain_refute_is_blocked(self):
        snap = _snap([("a", 3, "legal"), ("b", 3, "medical")], [("a", "b", "REFUTE")])
        self.assertEqual(rc.snapshot_refs(snap)["b"], set())

    def test_oracle_cycle_and_missing_parent_authority(self):
        snap = _snap(
            [("a", 3, "general"), ("b", 3, "general"), ("c", 3, "general")],
            [("a", "b", "DERIVES_FROM"), ("b", "a", "DERIVES_FROM")],
            missing=["c"],
        )
        eff = rc.snapshot_eff_auth(snap)
        self.assertEqual((eff["a"], eff["b"], eff["c"]), (0, 0, 0))
        vals = rc.oracle_vals(snap)
        self.assertEqual((vals["a"], vals["b"], vals["c"]), ("F", "F", "F"))


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

    def test_stratum0_materialization_gate_and_liveness(self):
        self.assertTrue(self.report["gates"]["inc_stratum0_materialization_matches_model"])
        self.assertEqual(self.report["stratum0_materialization"]["mismatches"], 0)
        live = self.report["comparator_liveness"]
        self.assertGreater(live["materialization_faults_injected"], 100)
        self.assertEqual(live["materialization_faults_injected"], live["materialization_faults_detected"])

    def test_incremental_pitfall_b_disclosure(self):
        self.assertEqual(self.report["incremental_with_deletions_status"], "conformance_tested_unproven_in_lean")
        d = self.report["incremental_pitfall_b_pruning_disclosure"]
        self.assertEqual(d["status"], "conformance_tested_unproven_in_lean")
        self.assertIn("ek5_v3_stratified.py:665", d["explanation"])
        self.assertEqual(d["semantic_divergences_total"], 0)
        for field in ("eff_auth", "val", "cur", "eligibility", "node_set"):
            self.assertEqual(d["semantic_divergences_by_field"].get(field, 0), 0, field)
        self.assertGreater(d["nonsemantic_total"], 0)
        self.assertLessEqual(set(d["nonsemantic_differences_lean_vs_inc"]), {"pos_cyc", "reason"})

    def test_report_serialises_with_lf_endings(self):
        # Write outside formal/ so a test run cannot refresh elapsed_seconds in the sealed report.
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "conformance_report.json"
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

    def test_manifest_has_no_host_paths_and_pins(self):
        raw = (FORMAL / "nanoda_manifest.json").read_text(encoding="utf-8")
        for needle in ("C:\\\\", "C:/", "\\\\Users\\\\", "/Users/", "/home/", "antigravity"):
            self.assertNotIn(needle, raw, needle)
        m = json.loads(raw)
        self.assertEqual(m["nanoda_bin"], "nanoda_bin")
        self.assertEqual(m["lean4export_bin"], "lean4export")
        self.assertRegex(m["nanoda_git_rev"], r"^[0-9a-f]{40}$")
        self.assertRegex(m["lean4export_git_rev"], r"^[0-9a-f]{40}$")
        self.assertEqual(m["nanoda_version"], "0.4.32")

    def test_manifest_exports_27_declarations(self):
        decls = rc.load_manifest()["export_declarations"]
        self.assertEqual(len(decls), 27)
        self.assertEqual(len(set(decls)), 27)
        self.assertIn("ChelatedFormal.all_wave14a_formal_theorems_verified", decls)
        self.assertIn("ChelatedFormal.evalStratum2Wfs_sound", decls)

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


class TestBinaryResolutionAndNanodaParsing(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.home = Path(self._tmp.name)
        self.exe_nanoda = rc._exe_name("nanoda_bin")
        self.exe_l4e = rc._exe_name("lean4export")

    def _touch(self, *parts):
        p = self.home.joinpath(*parts)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"x")
        return str(p)

    def _env(self, **extra):
        env = {"HOME": str(self.home), "USERPROFILE": str(self.home), "PATH": ""}
        env.update(extra)
        return mock.patch.dict(os.environ, env, clear=True)

    def test_nanoda_resolution_order(self):
        cargo_bin = self._touch(".cargo", "bin", self.exe_nanoda)
        with self._env(), mock.patch.object(Path, "home", return_value=self.home):
            self.assertEqual(rc.resolve_nanoda_bin(), cargo_bin)
            target = self._touch(".cargo", "nanoda_target", "release", self.exe_nanoda)
            self.assertEqual(rc.resolve_nanoda_bin(), target)
            on_path = self._touch("pathdir", self.exe_nanoda)
            with mock.patch.object(rc.shutil, "which", return_value=on_path):
                self.assertEqual(rc.resolve_nanoda_bin(), on_path)
        env_bin = self._touch("envdir", self.exe_nanoda)
        with self._env(NANODA_BIN=env_bin), mock.patch.object(Path, "home", return_value=self.home):
            with mock.patch.object(rc.shutil, "which", return_value=self._touch("pathdir2", self.exe_nanoda)):
                self.assertEqual(rc.resolve_nanoda_bin(), env_bin)

    def test_nanoda_unresolved_is_none(self):
        with self._env(), mock.patch.object(Path, "home", return_value=self.home):
            self.assertIsNone(rc.resolve_nanoda_bin())

    def test_lean4export_resolution_order(self):
        cargo_bin = self._touch(".cargo", "bin", self.exe_l4e)
        with self._env(), mock.patch.object(Path, "home", return_value=self.home):
            self.assertEqual(rc.resolve_lean4export_bin(), cargo_bin)
            elan = self._touch(".elan", "bin", self.exe_l4e)
            self.assertEqual(rc.resolve_lean4export_bin(), elan)
            on_path = self._touch("pathdir", self.exe_l4e)
            with mock.patch.object(rc.shutil, "which", return_value=on_path):
                self.assertEqual(rc.resolve_lean4export_bin(), on_path)
        env_bin = self._touch("envdir", self.exe_l4e)
        with self._env(LEAN4EXPORT_BIN=env_bin), mock.patch.object(Path, "home", return_value=self.home):
            with mock.patch.object(rc.shutil, "which", return_value=self._touch("pathdir2", self.exe_l4e)):
                self.assertEqual(rc.resolve_lean4export_bin(), env_bin)

    def test_lean4export_unresolved_is_none(self):
        with self._env(), mock.patch.object(Path, "home", return_value=self.home):
            self.assertIsNone(rc.resolve_lean4export_bin())

    def test_prepare_nanoda_without_lean4export_raises(self):
        with self._env(), mock.patch.object(Path, "home", return_value=self.home):
            with self.assertRaises(FileNotFoundError):
                rc.prepare_nanoda()

    def test_run_nanoda_without_binary_raises(self):
        with self._env(), mock.patch.object(Path, "home", return_value=self.home):
            with self.assertRaises(FileNotFoundError):
                rc.run_nanoda()

    def test_parse_nanoda_output_requires_rc_zero_and_enough_declarations(self):
        good = "Checked 120 declarations with no typechecker errors\n"
        self.assertTrue(rc.parse_nanoda_output(0, good, 27)["accepted"])
        self.assertEqual(rc.parse_nanoda_output(0, good, 27)["declarations_checked"], 120)
        self.assertFalse(rc.parse_nanoda_output(1, good, 27)["accepted"])
        self.assertFalse(rc.parse_nanoda_output(0, "", 27)["accepted"])
        self.assertFalse(rc.parse_nanoda_output(0, "Checked 0 declarations with no typechecker errors", 27)["accepted"])
        self.assertFalse(rc.parse_nanoda_output(0, "Checked 26 declarations with no typechecker errors", 27)["accepted"])
        self.assertTrue(rc.parse_nanoda_output(0, "Checked 27 declarations with no typechecker errors", 27)["accepted"])

    def test_main_reports_not_run_rc2_when_tools_missing(self):
        with self._env(), mock.patch.object(Path, "home", return_value=self.home):
            err = io.StringIO()
            with contextlib.redirect_stderr(err):
                code = rc.main(["--run-nanoda", "--report", str(self.home / "not-the-sealed-report.json")])
            self.assertEqual(code, 2)
            self.assertIn("NOT_RUN", err.getvalue())


if __name__ == "__main__":
    unittest.main()
