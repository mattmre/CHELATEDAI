"""Tamper-negative tests for the offline live-receipt verifiers of bundles #48, #57, #61 and #62.

Each verifier recomputes hashes, gate algebra and internal consistency from the committed live receipts without
contacting any server. These tests copy the receipts into a temporary root, corrupt one fact at a time and require
the verifier to notice. Nothing under ``artifacts/`` is modified. They prove the checks are not vacuous; they do NOT
prove the Spark-host measurements themselves are reproducible (they are not reproducible off that host).
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

SPINE = "32693f54e0ce7ad3d401ceb752ae4723b0eee251f6d49b95cdfb26a6201971e9"


def _load(name: str):
    path = REPO_ROOT / "scripts" / "verifiers.d" / f"{name}_verifier.py"
    spec = importlib.util.spec_from_file_location(f"pv7_live_{name}", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _failed(result) -> list:
    return [c["name"] for c in result["checks"] if not c["passed"]]


class _TempRoot(unittest.TestCase):
    rels: tuple = ()

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        for rel in self.rels:
            dest = self.root / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(REPO_ROOT / rel, dest)

    def read(self, rel: str):
        return json.loads((self.root / rel).read_bytes())

    def write(self, rel: str, obj) -> None:
        (self.root / rel).write_bytes(json.dumps(obj, indent=2, sort_keys=True).encode("utf-8"))


class Live57Tests(_TempRoot):
    rels = (
        "artifacts/pv7_w19_2b_cartridges/pv7_w19_2b_manifest.json",
        "artifacts/pv7_w19_2b_cartridges/report_receipt.json",
        "chelated_cartridge_array.py",
        "scripts/run_pv7_w19_2b_cartridges.py",
    )

    def setUp(self) -> None:
        super().setUp()
        self.v = _load("57")

    def verify(self):
        return self.v.verify_live_receipt(self.root, SPINE)

    def test_untampered_copy_verifies(self) -> None:
        result = self.verify()
        self.assertEqual(result["status"], "live-receipt-verified", _failed(result))

    def test_absent_receipt_is_reported_absent_not_passed(self) -> None:
        (self.root / self.v.LIVE_RECEIPT_REL).unlink()
        self.assertEqual(self.verify()["status"], "live-receipt-absent")

    def test_manifest_byte_change_breaks_hash_binding(self) -> None:
        manifest = self.read(self.v.LIVE_MANIFEST_REL)
        manifest["traced_peak_bytes"] = int(manifest["traced_peak_bytes"]) - 1
        self.write(self.v.LIVE_MANIFEST_REL, manifest)
        self.assertIn("live_manifest_sha256_binds_bytes", _failed(self.verify()))

    def test_rebound_gate_flip_is_caught_by_deeper_checks(self) -> None:
        manifest = self.read(self.v.LIVE_MANIFEST_REL)
        manifest["gates"][sorted(manifest["gates"])[0]] = False
        self.write(self.v.LIVE_MANIFEST_REL, manifest)
        receipt = self.read(self.v.LIVE_RECEIPT_REL)
        receipt["manifest_sha256"] = hashlib.sha256((self.root / self.v.LIVE_MANIFEST_REL).read_bytes()).hexdigest()
        self.write(self.v.LIVE_RECEIPT_REL, receipt)
        # deterministic_sha256 covers gate NAMES only (not values), so the gate-value algebra is what catches this.
        self.assertIn("live_gates_match_and_all_true", _failed(self.verify()))

    def test_revocation_post_enforcement_execution_is_caught(self) -> None:
        manifest = self.read(self.v.LIVE_MANIFEST_REL)
        manifest["revocation"]["post_enforcement_executions"] = 1
        self.write(self.v.LIVE_MANIFEST_REL, manifest)
        self.assertIn("live_revocation_zero_post_enforcement", _failed(self.verify()))

    def test_source_drift_is_reported_not_failed(self) -> None:
        path = self.root / "chelated_cartridge_array.py"
        path.write_bytes(path.read_bytes() + b"\n# drift\n")
        result = self.verify()
        self.assertEqual(result["source_drift_since_receipt"], ["chelated_cartridge_array.py"])
        self.assertEqual(result["status"], "live-receipt-verified")


class Live61Tests(_TempRoot):
    rels = (
        "artifacts/pv7_w20_d12_network_orbital_fabric/pv7_w20_d12_fabric_report.json",
        "artifacts/pv7_w20_d12_network_orbital_fabric/report_receipt.json",
    )

    def setUp(self) -> None:
        super().setUp()
        self.v = _load("61")

    def verify(self):
        return self.v.verify_live_report(self.root, SPINE)

    def test_untampered_copy_verifies_and_discloses_stand_ins(self) -> None:
        result = self.verify()
        self.assertEqual(result["status"], "live-report-verified", _failed(result))
        self.assertTrue(result["stand_in_encoders"], "stand-in encoders must be surfaced, never hidden")
        self.assertIn("informational", result["outcome_disclosure"])

    def test_report_byte_change_breaks_receipt_binding(self) -> None:
        report = self.read(self.v.LIVE_REPORT_REL)
        report["banner"] = str(report["banner"]) + " "
        self.write(self.v.LIVE_REPORT_REL, report)
        self.assertIn("live_receipt_binds_report_bytes", _failed(self.verify()))

    def test_rebound_gate_flip_is_caught(self) -> None:
        report = self.read(self.v.LIVE_REPORT_REL)
        report["gates"][0]["passed"] = False
        self.write(self.v.LIVE_REPORT_REL, report)
        receipt = self.read(self.v.LIVE_RECEIPT_REL)
        data = (self.root / self.v.LIVE_REPORT_REL).read_bytes()
        receipt["sha256"], receipt["bytes"] = hashlib.sha256(data).hexdigest(), len(data)
        self.write(self.v.LIVE_RECEIPT_REL, receipt)
        self.assertIn("live_gate_set_complete_and_all_pass", _failed(self.verify()))

    def test_rebound_scoreboard_edit_is_caught(self) -> None:
        report = self.read(self.v.LIVE_REPORT_REL)
        report["scoreboard"]["oracle"]["oracle_disagreements"] = 3
        self.write(self.v.LIVE_REPORT_REL, report)
        receipt = self.read(self.v.LIVE_RECEIPT_REL)
        data = (self.root / self.v.LIVE_REPORT_REL).read_bytes()
        receipt["sha256"], receipt["bytes"] = hashlib.sha256(data).hexdigest(), len(data)
        self.write(self.v.LIVE_RECEIPT_REL, receipt)
        failed = _failed(self.verify())
        self.assertIn("live_scoreboard_sha256_recomputed", failed)
        self.assertIn("live_oracle_and_register_zero_disagreements", failed)

    def test_honesty_flag_flip_is_caught(self) -> None:
        receipt = self.read(self.v.LIVE_RECEIPT_REL)
        receipt["hardware_asic_p4_silicon_in_loop_claimed"] = True
        self.write(self.v.LIVE_RECEIPT_REL, receipt)
        self.assertIn("live_receipt_honesty_flags_false", _failed(self.verify()))


class Live62Tests(_TempRoot):
    rels = (
        "artifacts/pv7_w20_d13_pooled_agent_loops/pv7_w20_d13_manifest.json",
        "artifacts/pv7_w20_d13_pooled_agent_loops/report_receipt.json",
        "chelated_pooled_agent_runtime.py",
        "scripts/run_pv7_w20_d13_pooled_agent_loops.py",
    )

    def setUp(self) -> None:
        super().setUp()
        self.v = _load("62")

    def verify(self):
        return self.v.verify_live_receipt(self.root, SPINE)

    def _rebind(self) -> None:
        receipt = self.read(self.v.LIVE_RECEIPT_REL)
        receipt["manifest_sha256"] = hashlib.sha256((self.root / self.v.LIVE_MANIFEST_REL).read_bytes()).hexdigest()
        self.write(self.v.LIVE_RECEIPT_REL, receipt)

    def test_committed_live_worker_failure_is_surfaced_not_converted_to_pass(self) -> None:
        result = self.verify()
        self.assertEqual(result["status"], "live-receipt-verified", _failed(result))
        self.assertEqual(result["live_worker_outcome"], "FAILED")
        self.assertEqual(result["failed_gates"], ["beats_full_restart_rebuild_jpm", "sham_reopen_control_valid"])
        self.assertEqual(result["on_time_cited_correct_jobs"], 0)
        self.assertIn("did NOT pass", result["outcome_disclosure"])

    def test_manifest_byte_change_breaks_hash_binding(self) -> None:
        manifest = self.read(self.v.LIVE_MANIFEST_REL)
        manifest["live_model_calls"] = int(manifest["live_model_calls"]) + 1
        self.write(self.v.LIVE_MANIFEST_REL, manifest)
        self.assertIn("live_manifest_sha256_binds_bytes", _failed(self.verify()))

    def test_rebound_forged_live_pass_is_caught(self) -> None:
        manifest = self.read(self.v.LIVE_MANIFEST_REL)
        for name in manifest["gates"]:
            manifest["gates"][name] = True
        manifest["overall_pass"] = True
        manifest["status"] = "local-drycheck-pass"
        self.write(self.v.LIVE_MANIFEST_REL, manifest)
        self._rebind()
        receipt = self.read(self.v.LIVE_RECEIPT_REL)
        receipt["gates"], receipt["overall_pass"], receipt["status"] = manifest["gates"], True, "local-drycheck-pass"
        self.write(self.v.LIVE_RECEIPT_REL, receipt)
        result = self.verify()
        failed = _failed(result)
        self.assertEqual(result["status"], "live-receipt-failed")
        self.assertIn("live_deterministic_sha256_recomputed", failed)
        self.assertIn("live_jpm_gate_matches_comparator", failed)
        self.assertIn("live_sham_gate_matches_control_valid", failed)

    def test_overall_pass_inconsistent_with_gates_is_caught(self) -> None:
        manifest = self.read(self.v.LIVE_MANIFEST_REL)
        manifest["overall_pass"] = True
        self.write(self.v.LIVE_MANIFEST_REL, manifest)
        self._rebind()
        self.assertIn("live_overall_pass_equals_all_gates", _failed(self.verify()))

    def test_served_model_mismatch_is_caught(self) -> None:
        receipt = self.read(self.v.LIVE_RECEIPT_REL)
        receipt["served_model"] = "some-other-model"
        self.write(self.v.LIVE_RECEIPT_REL, receipt)
        self.assertIn("live_served_model_matches_requested", _failed(self.verify()))


class Live48Tests(unittest.TestCase):
    REL = "artifacts/pv7_w17_6a_engine_closure_live/48-t6a-engine-closure.json"

    @classmethod
    def setUpClass(cls) -> None:
        cls.v = _load("48")
        cls.receipt = json.loads((REPO_ROOT / cls.REL).read_bytes())

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.path = Path(self._tmp.name) / "live.json"

    def run_verify(self, receipt):
        self.path.write_bytes(json.dumps(receipt).encode("utf-8"))
        return self.v.verify_live_receipt(self.path, {"ek5": self.receipt["ek5"]})

    def test_untampered_copy_verifies(self) -> None:
        self.assertEqual(self.run_verify(self.receipt)["status"], "live-receipt-verified")

    def test_recall_below_gap_is_rejected(self) -> None:
        bad = json.loads(json.dumps(self.receipt))
        bad["engines"][0]["cells"][0]["recall_at_10"] = 0.5
        with self.assertRaisesRegex(ValueError, "recall below preregistered gap"):
            self.run_verify(bad)

    def test_fixture_digest_forgery_is_rejected(self) -> None:
        bad = json.loads(json.dumps(self.receipt))
        bad["fixture"]["digest"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "fixture digest"):
            self.run_verify(bad)

    def test_failure_counter_is_rejected(self) -> None:
        bad = json.loads(json.dumps(self.receipt))
        bad["engines"][1]["cells"][0]["invalid_hits"] = 1
        with self.assertRaisesRegex(ValueError, "non-zero failure counters"):
            self.run_verify(bad)


if __name__ == "__main__":
    unittest.main()
