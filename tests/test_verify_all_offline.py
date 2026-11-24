"""Tests for scripts/verify_all_offline.py (unittest-compatible; no pytest-only fixtures)."""

from __future__ import annotations

import base64
import contextlib
import copy
import hashlib
import importlib.util
import io
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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
