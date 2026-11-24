"""Tests for scripts/verify_all_offline.py (unittest-compatible; no pytest-only fixtures)."""

from __future__ import annotations

import base64
import contextlib
import copy
import hashlib
import importlib.util
import io
import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parent.parent
MODULE_PATH = REPO_ROOT / "scripts" / "verify_all_offline.py"

_spec = importlib.util.spec_from_file_location("verify_all_offline_under_test", MODULE_PATH)
assert _spec is not None and _spec.loader is not None
V = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = V
_spec.loader.exec_module(V)

COPIED_FILES: Tuple[str, ...] = (
    "scripts/verify_non_nda_ip_clean.py",
    "docs/research/external-investor-teaser-non-nda.md",
    "docs/research/external-investor-teaser-deck-non-nda.html",
    "demo_workbench/compliance_copilot.html",
    "ek5_v3_stratified.py",
)
COPIED_DIRS: Tuple[str, ...] = (
    "artifacts/wave11_d6_opensearch_pgvector_qdrant",
    "artifacts/wave11_d8_compliance_copilot",
)
PRE_RELEASE = "PRE-RELEASE / HUMAN AUDIT & CLEAN-RERUN GATES PENDING"


def run_main(argv: List[str]) -> Tuple[int, str]:
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = V.main(argv)
    return rc, buf.getvalue()


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=1, sort_keys=True) + "\n", encoding="utf-8", newline="\n")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_bundle(root: Path, bid: str, extra_receipt: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Create artifacts/syn/<bid> with a manifest and a matching receipt; return the spec entry."""
    rel = f"artifacts/syn/{bid}"
    bdir = root / rel
    bdir.mkdir(parents=True, exist_ok=True)
    manifest = bdir / "m.json"
    write_json(manifest, {"bundle": bid, "value": 1})
    receipt: Dict[str, Any] = {"manifest_file": "m.json", "manifest_sha256": sha(manifest)}
    receipt.update(extra_receipt or {})
    write_json(bdir / "report_receipt.json", receipt)
    return {
        "id": bid,
        "part": "PART_TEST",
        "role": "TEST",
        "dir": rel,
        "manifest": "m.json",
        "receipt": "report_receipt.json",
        "notes": "synthetic",
    }


def make_repo(root: Path, n_bundles: int = 1) -> List[Dict[str, Any]]:
    """Build a minimal passing synthetic repository under ``root``."""
    for rel in COPIED_FILES:
        dst = root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPO_ROOT / rel, dst)
    for rel in COPIED_DIRS:
        shutil.copytree(REPO_ROOT / rel, root / rel)
    entries = [make_bundle(root, f"syn{i + 1}") for i in range(n_bundles)]
    write_json(root / V.SPEC_REL, {"schema_version": "test", "release_status": PRE_RELEASE, "bundles": entries})
    return entries


def check_status(payload: Dict[str, Any], name: str) -> str:
    for row in payload["checks"]:
        if row["name"] == name:
            return str(row["status"])
    raise AssertionError(f"check {name} not in report")


class SyntheticRepoCase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name) / "repo"
        self.root.mkdir()
        self.json_out = Path(self._tmp.name) / "out.json"

    def run_fast(self, *extra: str) -> Tuple[int, Dict[str, Any]]:
        rc, _ = run_main(["--fast", "--no-d8-replay", "--repo-root", str(self.root), "--json-out", str(self.json_out), *extra])
        return rc, json.loads(self.json_out.read_text(encoding="utf-8"))


class TestLiveRepo(unittest.TestCase):
    def test_fast_mode_against_real_repo(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "live.json"
            rc, text = run_main(["--fast", "--json-out", str(out)])
            payload = json.loads(out.read_text(encoding="utf-8"))
        failures = [c for c in payload["checks"] if c["status"] == "FAIL"]
        self.assertEqual(rc, 0, f"failures: {failures}\n{text[-2000:]}")
        self.assertLess(payload["wall_clock_seconds"], 45.0)
        self.assertEqual(payload["formal_tier"], "SKIPPED_USE_FULL_FORMAL")
        self.assertEqual(payload["skipped_checks"], list(V.FAST_SKIPPED_CHECKS))
        self.assertEqual(payload["mode"], "fast")
        self.assertEqual(payload["overall"], "PASS")
        self.assertEqual(check_status(payload, "d8.live_replay"), "PASS")
        self.assertEqual(check_status(payload, "d8.ed25519_and_hash_chain"), "PASS")
        self.assertEqual(payload["summary"]["bundles_failed"], 0)
        self.assertEqual(payload["summary"]["bundles_verified"], payload["summary"]["bundles_total"])
        for key in ("platform", "python", "wall_clock_seconds"):
            self.assertIn(key, payload)
        self.assertIn("formal_tier=SKIPPED_USE_FULL_FORMAL", text)


class TestSyntheticRepo(SyntheticRepoCase):
    def test_baseline_passes(self) -> None:
        make_repo(self.root)
        rc, payload = self.run_fast()
        failures = [c for c in payload["checks"] if c["status"] == "FAIL"]
        self.assertEqual(rc, 0, failures)
        self.assertEqual(check_status(payload, "d8.live_replay"), "SKIP")
        self.assertEqual(payload["summary"]["d8"]["replay"], "SKIPPED_BY_FLAG")

    def test_scales_to_46_bundles(self) -> None:
        make_repo(self.root, n_bundles=46)
        rc, payload = self.run_fast()
        self.assertEqual(rc, 0)
        self.assertEqual(payload["summary"]["bundles_total"], 46)
        self.assertEqual(payload["summary"]["bundles_verified"], 46)

    def test_tampered_manifest_fails(self) -> None:
        make_repo(self.root)
        manifest = self.root / "artifacts/syn/syn1/m.json"
        manifest.write_text(manifest.read_text(encoding="utf-8") + " ", encoding="utf-8", newline="\n")
        rc, payload = self.run_fast()
        self.assertNotEqual(rc, 0)
        self.assertEqual(check_status(payload, "bundle.syn1"), "FAIL")
        self.assertEqual(payload["overall"], "FAIL")

    def test_missing_receipt_fails(self) -> None:
        make_repo(self.root)
        (self.root / "artifacts/syn/syn1/report_receipt.json").unlink()
        rc, payload = self.run_fast()
        self.assertEqual(rc, 1)
        self.assertEqual(check_status(payload, "bundle.syn1"), "FAIL")

    def test_missing_ancillary_fails(self) -> None:
        entries = make_repo(self.root)
        entries[0]["ancillary_json_hashes"] = ["does_not_exist.json"]
        write_json(self.root / V.SPEC_REL, {"schema_version": "t", "release_status": PRE_RELEASE, "bundles": entries})
        rc, payload = self.run_fast()
        self.assertEqual(rc, 1)
        self.assertEqual(check_status(payload, "bundle.syn1"), "FAIL")

    def test_missing_standalone_part7_bundle_fails(self) -> None:
        make_repo(self.root)
        shutil.rmtree(self.root / "artifacts/wave11_d8_compliance_copilot")
        rc, payload = self.run_fast()
        self.assertEqual(rc, 1)
        self.assertTrue(any(c["status"] == "FAIL" and "d8" in c["name"] for c in payload["checks"]))

    def test_non_nda_leak_fails(self) -> None:
        make_repo(self.root)
        teaser = self.root / "docs/research/external-investor-teaser-non-nda.md"
        teaser.write_text(teaser.read_text(encoding="utf-8") + "\nWe use a Stiefel manifold.\n", encoding="utf-8", newline="\n")
        rc, payload = self.run_fast()
        self.assertNotEqual(rc, 0)
        self.assertEqual(check_status(payload, "non_nda.external-investor-teaser-non-nda.md"), "FAIL")
        self.assertEqual(check_status(payload, "non_nda.compliance_copilot.html"), "PASS")

    def test_non_nda_leak_in_copilot_html_fails(self) -> None:
        make_repo(self.root)
        page = self.root / "demo_workbench/compliance_copilot.html"
        page.write_text(page.read_text(encoding="utf-8") + "\n<!-- CSLS hubness -->\n", encoding="utf-8", newline="\n")
        rc, payload = self.run_fast()
        self.assertEqual(rc, 1)
        self.assertEqual(check_status(payload, "non_nda.compliance_copilot.html"), "FAIL")

    def test_raw_files_optional_when_absent_and_hashed_when_present(self) -> None:
        raw_name = "raw_rows.jsonl"
        sealed_name = "x_sealed_key_y.json"
        entries = make_repo(self.root)
        bdir = self.root / entries[0]["dir"]
        raw = bdir / raw_name
        raw.write_text('{"a":1}\n', encoding="utf-8", newline="\n")
        receipt = json.loads((bdir / "report_receipt.json").read_text(encoding="utf-8"))
        receipt["artifact_sha256"] = {raw_name: sha(raw), "other_raw.csv": "0" * 64, sealed_name: "1" * 64}
        write_json(bdir / "report_receipt.json", receipt)
        rc, payload = self.run_fast()
        self.assertEqual(rc, 0, [c for c in payload["checks"] if c["status"] == "FAIL"])
        self.assertIn("raw_hashed=1", next(c["detail"] for c in payload["checks"] if c["name"] == "bundle.syn1"))
        self.assertIn("raw_absent_optional=1", next(c["detail"] for c in payload["checks"] if c["name"] == "bundle.syn1"))
        self.assertIn("sealed_unread=1", next(c["detail"] for c in payload["checks"] if c["name"] == "bundle.syn1"))
        raw.write_text('{"a":2}\n', encoding="utf-8", newline="\n")
        rc2, payload2 = self.run_fast()
        self.assertEqual(rc2, 1)
        self.assertEqual(check_status(payload2, "bundle.syn1"), "FAIL")

    def test_missing_non_raw_sidecar_fails(self) -> None:
        entries = make_repo(self.root)
        bdir = self.root / entries[0]["dir"]
        receipt = json.loads((bdir / "report_receipt.json").read_text(encoding="utf-8"))
        receipt["artifact_hashes"] = {"extra.json": "0" * 64}
        write_json(bdir / "report_receipt.json", receipt)
        rc, payload = self.run_fast()
        self.assertEqual(rc, 1)
        self.assertEqual(check_status(payload, "bundle.syn1"), "FAIL")

    def test_pre_release_dropped_fails(self) -> None:
        entries = make_repo(self.root)
        write_json(self.root / V.SPEC_REL, {"schema_version": "t", "release_status": "RELEASED", "bundles": entries})
        rc, payload = self.run_fast()
        self.assertEqual(rc, 1)
        self.assertEqual(check_status(payload, "release_status.honesty"), "FAIL")

    def test_d6_stale_resurrection_tamper_fails(self) -> None:
        make_repo(self.root)
        bdir = self.root / "artifacts/wave11_d6_opensearch_pgvector_qdrant"
        manifest_path = bdir / "wave11_d6_multibackend_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        cell = next(c for *_, c in V.iter_d6_cases(manifest))
        cell["stale_resurrection_rate"] = 0.25
        write_json(manifest_path, manifest)
        receipt_path = bdir / "report_receipt.json"
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        receipt["manifest_sha256"] = sha(manifest_path)  # keep the receipt bound so only the metric gate trips
        write_json(receipt_path, receipt)
        rc, payload = self.run_fast()
        self.assertEqual(rc, 1)
        self.assertEqual(check_status(payload, "metrics.d6"), "FAIL")

    def test_registry_self_hash_tamper_fails(self) -> None:
        make_repo(self.root)
        registry = {"schema_version": "r", "release_status": PRE_RELEASE, "bundles": []}
        registry["registry_content_sha256"] = V.registry_content_digest(registry)
        write_json(self.root / V.REGISTRY_REL, registry)
        _, payload = self.run_fast()
        self.assertEqual(check_status(payload, "registry.content_sha256"), "PASS")
        registry["registry_content_sha256"] = "f" * 64
        write_json(self.root / V.REGISTRY_REL, registry)
        rc2, payload2 = self.run_fast()
        self.assertEqual(rc2, 1)
        self.assertEqual(check_status(payload2, "registry.content_sha256"), "FAIL")


class TestFullFormal(SyntheticRepoCase):
    def test_full_formal_not_run_when_formal_absent(self) -> None:
        make_repo(self.root)
        rc, payload = self._run_full()
        self.assertEqual(rc, 2)
        self.assertEqual(payload["formal_tier"], "NOT_RUN")
        self.assertEqual(payload["overall"], "NOT_RUN")
        self.assertEqual(payload["mode"], "full-formal")
        self.assertEqual(check_status(payload, "formal.prerequisites"), "NOT_RUN")
        self.assertFalse(any(c["status"] == "PASS" and c["name"].startswith("formal.") for c in payload["checks"]))

    def test_full_formal_failure_outranks_not_run(self) -> None:
        make_repo(self.root)
        (self.root / "artifacts/syn/syn1/report_receipt.json").unlink()
        rc, payload = self._run_full()
        self.assertEqual(rc, 1)
        self.assertEqual(payload["formal_tier"], "NOT_RUN")

    def test_full_formal_not_run_when_lakefile_missing(self) -> None:
        make_repo(self.root)
        (self.root / "formal").mkdir()
        reasons = V.formal_prerequisites(self.root)
        self.assertTrue(any("lakefile" in r for r in reasons))
        rc, payload = self._run_full()
        self.assertEqual(rc, 2)
        self.assertEqual(payload["formal_tier"], "NOT_RUN")

    def _run_full(self) -> Tuple[int, Dict[str, Any]]:
        rc, _ = run_main(["--full-formal", "--no-d8-replay", "--repo-root", str(self.root), "--json-out", str(self.json_out)])
        return rc, json.loads(self.json_out.read_text(encoding="utf-8"))


class TestFormalFailClosed(SyntheticRepoCase):
    """--full-formal must be NOT_RUN rc=2 (never PASS / PASS_PARTIAL) when the external kernel or conformance is missing."""

    DECLS = ["ChelatedFormal.a", "ChelatedFormal.b", "ChelatedFormal.c"]

    def setUp(self) -> None:
        super().setUp()
        self.home = Path(self._tmp.name) / "home"
        self.home.mkdir()
        make_repo(self.root)
        self.formal = self.root / "formal"
        (self.formal / "ChelatedFormal").mkdir(parents=True)
        (self.formal / "lakefile.lean").write_text("package chelated\n", encoding="utf-8", newline="\n")
        (self.formal / "ChelatedFormal" / "A.lean").write_text("theorem a : True := trivial\n", encoding="utf-8", newline="\n")
        write_json(
            self.formal / "nanoda_manifest.json",
            {
                "nanoda_bin": "nanoda_bin",
                "nanoda_args": [".lake/nanoda/nanoda_config.json"],
                "lean4export_bin": "lean4export",
                "export_file": ".lake/nanoda/chelated_formal_all.export",
                "export_declarations": self.DECLS,
            },
        )
        (self.formal / "run_conformance.py").write_text("raise SystemExit(0)\n", encoding="utf-8", newline="\n")

    def write_export(self, age_s: float) -> Path:
        """Create the export file with mtime = (newest source mtime) + age_s."""
        export = self.formal / ".lake" / "nanoda" / "chelated_formal_all.export"
        export.parent.mkdir(parents=True, exist_ok=True)
        export.write_bytes(b"{}\n")
        newest = max(f.stat().st_mtime for f in (self.formal / "ChelatedFormal" / "A.lean", self.formal / "nanoda_manifest.json"))
        os.utime(str(export), (newest + age_s, newest + age_s))
        return export

    def fake_exe(self, name: str) -> str:
        p = Path(self._tmp.name) / "bin" / (name + (".exe" if os.name == "nt" else ""))
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"x")
        return str(p)

    def env(self, **extra: str) -> "contextlib.ExitStack":
        stack = contextlib.ExitStack()
        stack.enter_context(mock.patch.dict(os.environ, {"HOME": str(self.home), "USERPROFILE": str(self.home)}))
        for key in ("NANODA_BIN", "LEAN4EXPORT_BIN"):
            os.environ.pop(key, None)
        os.environ.update(extra)
        stack.enter_context(mock.patch.object(V.Path, "home", return_value=self.home))
        stack.enter_context(mock.patch.object(V.shutil, "which", side_effect=lambda n, *a, **k: "lake" if n == "lake" else None))
        return stack

    def run_full(self) -> Tuple[int, Dict[str, Any]]:
        rc, _ = run_main(["--full-formal", "--no-d8-replay", "--repo-root", str(self.root), "--json-out", str(self.json_out)])
        return rc, json.loads(self.json_out.read_text(encoding="utf-8"))

    def status(self, report: "V.Report", name: str) -> str:
        return [c["status"] for c in report.checks if c["name"] == name][0]

    def test_missing_nanoda_bin_is_not_run_rc2(self) -> None:
        self.write_export(+5)
        with self.env():
            reasons = V.formal_prerequisites(self.root)
            rc, payload = self.run_full()
        self.assertTrue(any("nanoda_bin not found" in r for r in reasons), reasons)
        self.assertEqual(rc, 2)
        self.assertEqual(payload["formal_tier"], "NOT_RUN")
        self.assertEqual(payload["overall"], "NOT_RUN")

    def test_missing_run_conformance_is_not_run_rc2(self) -> None:
        self.write_export(+5)
        (self.formal / "run_conformance.py").unlink()
        with self.env(NANODA_BIN=self.fake_exe("nanoda_bin")):
            reasons = V.formal_prerequisites(self.root)
            rc, payload = self.run_full()
        self.assertTrue(any("run_conformance.py absent" in r for r in reasons), reasons)
        self.assertEqual((rc, payload["formal_tier"]), (2, "NOT_RUN"))

    def test_unconfigured_manifest_is_not_run_rc2(self) -> None:
        (self.formal / "nanoda_manifest.json").unlink()
        with self.env(NANODA_BIN=self.fake_exe("nanoda_bin")):
            reasons = V.formal_prerequisites(self.root)
            rc, payload = self.run_full()
        self.assertTrue(any("nanoda_manifest.json" in r for r in reasons), reasons)
        self.assertEqual((rc, payload["formal_tier"]), (2, "NOT_RUN"))

    def test_stale_or_missing_export_without_lean4export_is_not_run_rc2(self) -> None:
        with self.env(NANODA_BIN=self.fake_exe("nanoda_bin")):
            reasons_missing = V.formal_prerequisites(self.root)
            self.write_export(-100)
            reasons_stale = V.formal_prerequisites(self.root)
            rc, payload = self.run_full()
        for reasons in (reasons_missing, reasons_stale):
            self.assertTrue(any("lean4export" in r for r in reasons), reasons)
        self.assertEqual((rc, payload["formal_tier"]), (2, "NOT_RUN"))

    def test_fresh_export_needs_no_lean4export(self) -> None:
        self.write_export(+5)
        with self.env(NANODA_BIN=self.fake_exe("nanoda_bin")):
            self.assertEqual(V.formal_prerequisites(self.root), [])

    def test_export_staleness_tracks_lean_sources_and_manifest(self) -> None:
        manifest = V.load_nanoda_manifest(self.formal)
        self.assertTrue(V.export_is_stale(self.formal, manifest))
        export = self.write_export(+5)
        self.assertFalse(V.export_is_stale(self.formal, manifest))
        later = export.stat().st_mtime + 50
        os.utime(str(self.formal / "ChelatedFormal" / "A.lean"), (later, later))
        self.assertTrue(V.export_is_stale(self.formal, manifest))
        self.write_export(+100)
        self.assertFalse(V.export_is_stale(self.formal, manifest))
        latest = later + 500
        os.utime(str(self.formal / "nanoda_manifest.json"), (latest, latest))
        self.assertTrue(V.export_is_stale(self.formal, manifest))

    def test_run_formal_gates_returns_not_run_when_nanoda_unresolved(self) -> None:
        self.write_export(+5)
        report = V.Report()
        with self.env(), mock.patch.object(V, "_run", return_value=(0, "")):
            tier, _ = V.run_formal_gates(self.root, report, 5)
        self.assertEqual(tier, "NOT_RUN")
        self.assertEqual(self.status(report, "formal.nanoda_external_kernel"), "NOT_RUN")
        self.assertFalse(any(c["status"] == "SKIP" for c in report.checks))

    def gates_with_nanoda(self, out: str, rc_nanoda: int = 0) -> Tuple[str, "V.Report"]:
        self.write_export(+5)
        nanoda = self.fake_exe("nanoda_bin")

        def fake_run(cmd: List[str], cwd: Path, timeout_s: int) -> Tuple[int, str]:
            if cmd[0] == nanoda:
                return rc_nanoda, out
            return 0, ""

        report = V.Report()
        with self.env(NANODA_BIN=nanoda), mock.patch.object(V, "_run", side_effect=fake_run):
            tier, _ = V.run_formal_gates(self.root, report, 5)
        return tier, report

    def test_nanoda_accepts_rc0_with_enough_declarations(self) -> None:
        for out in (
            "Checked 3 declarations with no typechecker errors\n",
            "Checked 40 declarations with no typechecker errors\n",
        ):
            _, report = self.gates_with_nanoda(out)
            self.assertEqual(self.status(report, "formal.nanoda_external_kernel"), "PASS", out)

    def test_nanoda_rejects_zero_short_missing_line_and_nonzero_rc(self) -> None:
        for out, rc_n in (
            ("Checked 0 declarations with no typechecker errors\n", 0),
            ("Checked 2 declarations with no typechecker errors\n", 0),
            ("", 0),
            ("Checked 3 declarations with no typechecker errors\n", 1),
        ):
            tier, report = self.gates_with_nanoda(out, rc_n)
            self.assertEqual(self.status(report, "formal.nanoda_external_kernel"), "FAIL", (out, rc_n))
            self.assertEqual(tier, "FAIL")

    def test_stale_export_is_regenerated_via_prepare_nanoda(self) -> None:
        export = self.write_export(-100)
        nanoda = self.fake_exe("nanoda_bin")
        l4e = self.fake_exe("lean4export")
        calls: List[List[str]] = []

        def fake_run(cmd: List[str], cwd: Path, timeout_s: int) -> Tuple[int, str]:
            calls.append(list(cmd))
            if "--prepare-nanoda" in cmd:
                export.write_bytes(b"{}\n")
                newest = max(f.stat().st_mtime for f in (self.formal / "ChelatedFormal" / "A.lean", self.formal / "nanoda_manifest.json"))
                os.utime(str(export), (newest + 5, newest + 5))
                return 0, ""
            if cmd[0] == nanoda:
                return 0, "Checked 3 declarations with no typechecker errors\n"
            return 0, ""

        report = V.Report()
        with self.env(NANODA_BIN=nanoda, LEAN4EXPORT_BIN=l4e), mock.patch.object(V, "_run", side_effect=fake_run):
            V.run_formal_gates(self.root, report, 5)
        self.assertTrue(any("--prepare-nanoda" in c for c in calls))
        self.assertEqual(self.status(report, "formal.lean4export_prepare"), "PASS")
        self.assertEqual(self.status(report, "formal.nanoda_external_kernel"), "PASS")


class TestNanodaOutputAndResolvers(unittest.TestCase):
    def test_parse_nanoda_output(self) -> None:
        ok = "Checked 27 declarations with no typechecker errors"
        self.assertTrue(V.parse_nanoda_output(0, ok, 27)["accepted"])
        self.assertFalse(V.parse_nanoda_output(0, ok, 28)["accepted"])
        self.assertFalse(V.parse_nanoda_output(2, ok, 27)["accepted"])
        self.assertFalse(V.parse_nanoda_output(0, "", 1)["accepted"])
        self.assertFalse(V.parse_nanoda_output(0, "Checked 0 declarations with no typechecker errors", 0)["accepted"])

    def test_resolver_order(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            exe_n, exe_l = V._exe_name("nanoda_bin"), V._exe_name("lean4export")

            def touch(*parts: str) -> str:
                p = home.joinpath(*parts)
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_bytes(b"x")
                return str(p)

            with mock.patch.dict(os.environ), mock.patch.object(V.Path, "home", return_value=home), mock.patch.object(
                V.shutil, "which", return_value=None
            ):
                os.environ.pop("NANODA_BIN", None)
                os.environ.pop("LEAN4EXPORT_BIN", None)
                self.assertIsNone(V.resolve_nanoda_bin())
                self.assertIsNone(V.resolve_lean4export_bin())
                n_cargo = touch(".cargo", "bin", exe_n)
                l_cargo = touch(".cargo", "bin", exe_l)
                self.assertEqual(V.resolve_nanoda_bin(), n_cargo)
                self.assertEqual(V.resolve_lean4export_bin(), l_cargo)
                n_target = touch(".cargo", "nanoda_target", "release", exe_n)
                l_elan = touch(".elan", "bin", exe_l)
                self.assertEqual(V.resolve_nanoda_bin(), n_target)
                self.assertEqual(V.resolve_lean4export_bin(), l_elan)
                n_path, l_path = touch("p", exe_n), touch("q", exe_l)
                with mock.patch.object(V.shutil, "which", side_effect=lambda n, *a, **k: n_path if n == "nanoda_bin" else l_path):
                    self.assertEqual(V.resolve_nanoda_bin(), n_path)
                    self.assertEqual(V.resolve_lean4export_bin(), l_path)
                    n_env, l_env = touch("e", exe_n), touch("f", exe_l)
                    os.environ["NANODA_BIN"] = n_env
                    os.environ["LEAN4EXPORT_BIN"] = l_env
                    self.assertEqual(V.resolve_nanoda_bin(), n_env)
                    self.assertEqual(V.resolve_lean4export_bin(), l_env)


class TestAxiomAllowlist(unittest.TestCase):
    def test_positive_allowlisted_axioms_only(self) -> None:
        text = (
            "'Foo.bar' depends on axioms: [propext, Classical.choice, Quot.sound]\n"
            "'Foo.baz' does not depend on any axioms\n"
            "'Foo.multi' depends on axioms: [propext,\n Quot.sound]\n"
        )
        verdict = V.evaluate_axiom_report(text, expected=["Foo.bar", "Foo.baz", "Foo.multi"])
        self.assertTrue(verdict["ok"], verdict)
        self.assertEqual(verdict["declarations"], 3)
        self.assertEqual(V.parse_axiom_report(text)["Foo.multi"], ["propext", "Quot.sound"])

    def test_negative_sorry_ax_rejected(self) -> None:
        verdict = V.evaluate_axiom_report("'T' depends on axioms: [propext, sorryAx]\n")
        self.assertFalse(verdict["ok"])
        self.assertIn("T: sorryAx", verdict["violations"])

    def test_negative_reduce_bool_rejected(self) -> None:
        verdict = V.evaluate_axiom_report("'T' depends on axioms: [Lean.ofReduceBool, propext]\n")
        self.assertFalse(verdict["ok"])
        self.assertIn("T: Lean.ofReduceBool", verdict["violations"])

    def test_negative_custom_axiom_rejected(self) -> None:
        verdict = V.evaluate_axiom_report("'T' depends on axioms: [myCustomAxiom]\n")
        self.assertFalse(verdict["ok"])
        self.assertIn("T: myCustomAxiom", verdict["violations"])

    def test_negative_vacuous_and_missing_and_error(self) -> None:
        self.assertFalse(V.evaluate_axiom_report("")["ok"])
        missing = V.evaluate_axiom_report("'A' does not depend on any axioms\n", expected=["A", "B"])
        self.assertFalse(missing["ok"])
        self.assertTrue(any("B" in e for e in missing["errors"]))
        errored = V.evaluate_axiom_report("x.lean:3:4: error: unknown constant\n'A' does not depend on any axioms\n")
        self.assertFalse(errored["ok"])


class TestLeanScanners(unittest.TestCase):
    def test_forbidden_tokens_detected_outside_comments(self) -> None:
        src = "theorem a : True := by\n  trivial\ntheorem b : 1 = 1 := sorry\ntheorem c : 2 = 2 := by native_decide\n"
        self.assertEqual(V.scan_lean_forbidden(src), [(3, "sorry"), (4, "native_decide")])

    def test_comments_and_strings_ignored(self) -> None:
        src = (
            "-- sorry in a line comment\n"
            "/- block with native_decide\n   /- nested sorry -/ still comment sorry -/\n"
            'def s : String := "sorry native_decide"\n'
            "theorem ok : True := trivial\n"
        )
        self.assertEqual(V.scan_lean_forbidden(src), [])

    def test_collect_theorems_tracks_namespaces(self) -> None:
        src = (
            "namespace Foo\n"
            "theorem one : True := trivial\n"
            "private theorem hidden : True := trivial\n"
            "section\n"
            "lemma two : True := trivial\n"
            "end\n"
            "end Foo\n"
            "theorem top : True := trivial\n"
            "-- theorem commented : True := trivial\n"
        )
        self.assertEqual(V.collect_lean_theorems(src), ["Foo.one", "Foo.two", "top"])


class TestCryptoAndRelease(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        d8 = REPO_ROOT / "artifacts/wave11_d8_compliance_copilot/wave11_d8_copilot_manifest.json"
        cls.manifest = json.loads(d8.read_text(encoding="utf-8"))

    def test_real_d8_crypto_ok(self) -> None:
        self.assertEqual(V.verify_d8_receipt_crypto(self.manifest), [])

    def test_bad_signature_detected(self) -> None:
        m = copy.deepcopy(self.manifest)
        row = m["deterministic"]["receipts"]["receipts"][1]
        sig = bytearray(base64.b64decode(row["signature_b64"]))
        sig[0] ^= 0x01
        row["signature_b64"] = base64.b64encode(bytes(sig)).decode("ascii")
        faults = V.verify_d8_receipt_crypto(m)
        self.assertTrue(any("signature INVALID" in f for f in faults), faults)

    def test_broken_chain_detected(self) -> None:
        m = copy.deepcopy(self.manifest)
        m["deterministic"]["receipts"]["receipts"][1]["prev_hash"] = "0" * 64
        faults = V.verify_d8_receipt_crypto(m)
        self.assertTrue(any("prev_hash" in f for f in faults), faults)

    def test_wrong_head_detected(self) -> None:
        m = copy.deepcopy(self.manifest)
        m["deterministic"]["receipts"]["head_hash"] = "a" * 64
        self.assertTrue(any("head_hash" in f for f in V.verify_d8_receipt_crypto(m)))

    def test_replay_comparison_detects_drift(self) -> None:
        sealed = self.manifest["deterministic"]["receipts"]
        good = {
            "entry_hashes": [r["entry_hash"] for r in sealed["receipts"]],
            "ledger": {"intact": True, "length": len(sealed["receipts"]), "head_hash": sealed["head_hash"]},
            "public_key_b64": sealed["signing_key"]["public_key_b64"],
            "key_id": sealed["signing_key"]["key_id"],
        }
        self.assertEqual(V.compare_d8_replay(self.manifest, good), [])
        bad = copy.deepcopy(good)
        bad["entry_hashes"][0] = "b" * 64
        self.assertTrue(V.compare_d8_replay(self.manifest, bad))

    def test_release_status_rules(self) -> None:
        self.assertEqual(V.evaluate_release_status(PRE_RELEASE, None), [])
        self.assertTrue(V.evaluate_release_status("RELEASED", None))
        self.assertTrue(V.evaluate_release_status("PRE-RELEASE Ship-Ready", None))
        self.assertTrue(V.evaluate_release_status("RELEASED", {"status": "COMPLETE"}))
        self.assertEqual(
            V.evaluate_release_status("RELEASED", {"status": "COMPLETE", "human_grader_receipt": "r.json"}), []
        )


if __name__ == "__main__":
    unittest.main()
