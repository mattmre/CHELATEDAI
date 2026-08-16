from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from rb13_wave0a_experiments import (
    FROZEN_PROTOCOL_SHA256,
    PROTOCOL_ID,
    Wave0ABudget,
    Wave0AResourceError,
    Wave0AValidationError,
    atomic_write_json,
    current_rss_bytes,
    make_stage_artifact,
    run_bil1,
    run_ek0,
    run_spu0,
    run_var1,
    sha256_payload,
)
from run_rb13_wave0a import (
    _archive_regular_file_set,
    _campaign_summary,
    _execute_worker,
    _stage_worker_reservation,
    _verify_archived_regeneration,
    run,
    verify_campaign,
    verify_archived_campaign,
    verify_archived_cross_phase,
    verify_cross_phase,
    verify_stage_artifact,
)
import run_rb13_wave0a


class RB13Wave0AExperimentsTests(unittest.TestCase):
    def setUp(self) -> None:
        # Unit fixtures exercise frozen scientific and failure-handling semantics,
        # not the aggregate RSS of whichever monolithic test process invokes them.
        # Keep production guards unchanged while giving both the in-process stage
        # functions and runner admission logic a deterministic, process-local
        # baseline. Individual resource tests override these patches explicitly.
        self._stage_rss_patch = patch(
            "rb13_wave0a_experiments.current_rss_bytes",
            return_value=64 * 1024**2,
        )
        self._runner_rss_patch = patch(
            "run_rb13_wave0a.current_rss_bytes",
            return_value=64 * 1024**2,
        )
        self._stage_rss_patch.start()
        self._runner_rss_patch.start()
        self.addCleanup(self._runner_rss_patch.stop)
        self.addCleanup(self._stage_rss_patch.stop)

    def test_budget_refuses_more_than_two_gib(self) -> None:
        with self.assertRaises(Wave0AValidationError):
            Wave0ABudget(max_estimated_bytes=2 * 1024**3 + 1)

    def test_budget_refuses_non_evidence_capable_one_byte_limits(self) -> None:
        with self.assertRaises(Wave0AValidationError):
            Wave0ABudget(max_estimated_bytes=1)
        with self.assertRaises(Wave0AValidationError):
            Wave0ABudget(max_output_bytes=1)

    def test_modeled_preflight_refuses_short_lived_spu_work_before_allocation(self) -> None:
        with self.assertRaises(Wave0AResourceError):
            run_spu0(
                seed=17, budget=Wave0ABudget(max_estimated_bytes=16 * 1024**2)
            )

    def test_live_rss_measurement_is_available(self) -> None:
        self.assertGreater(current_rss_bytes(), 0)

    def test_unit_fixture_rss_is_isolated_from_monolithic_process_baseline(self) -> None:
        contaminated_rss = 1200 * 1024**2
        with patch(
            "rb13_wave0a_experiments.current_rss_bytes",
            return_value=contaminated_rss,
        ):
            with self.assertRaises(Wave0AResourceError):
                run_bil1(seed=17)

        # The class-scoped unit seam restores the deterministic baseline after
        # the simulated monolithic-process contamination. Production calls have
        # no such patch and continue to fail closed on their measured RSS.
        self.assertEqual(run_bil1(seed=17)["stage_id"], "PRW-BIL1")

    def test_z_ek0_characterizes_current_contract_without_promoting_null(self) -> None:
        result = run_ek0()
        self.assertEqual(result["stage_id"], "PRW-EK0")
        self.assertFalse(result["null_survives"])
        self.assertTrue(result["observations"]["fifo_overflow"]["observed"])
        self.assertTrue(result["observations"]["annotation_mutates_payload"]["observed"])
        self.assertTrue(result["observations"]["direct_promote_bypasses_gate"]["observed"])
        self.assertTrue(result["observations"]["missing_status_promotes"]["observed"])
        self.assertTrue(result["observations"]["negative_status_fails_closed"]["observed"])
        self.assertEqual(result["scientific_claim_status"], "UNCONFIRMED")

    def test_bil1_is_permutation_lineage_and_fixed_point_invariant(self) -> None:
        result = run_bil1(seed=17)
        self.assertTrue(all(result["invariants"].values()))
        self.assertEqual(result["disposition"], "REDUCE_TO_PAIRED_BOOLEAN_UNION")
        self.assertEqual(
            [row["candidate"] for row in result["rows"]],
            ["SUPPORTED", "CONFLICTED", "SUPPORTED", "CONFLICTED", "CONFLICTED"],
        )

    def test_bil1_scalar_control_confuses_conflict_with_unknown(self) -> None:
        result = run_bil1(seed=17)
        conflict_row = result["rows"][1]
        self.assertEqual(conflict_row["candidate"], "CONFLICTED")
        self.assertEqual(conflict_row["controls"]["scalar"], "UNKNOWN")

    def test_spu0_fixed_stack_equals_flat_map(self) -> None:
        result = run_spu0(seed=17, dimension=8, factor_count=3, vector_count=12)
        self.assertEqual(result["disposition"], "ORDINARY_FACTORIZATION_EQUIVALENCE")
        self.assertLessEqual(result["output_max_abs_error"], 1e-10)
        self.assertLessEqual(result["factorized_output_max_abs_error"], 1e-10)
        self.assertLessEqual(result["distance_max_abs_error"], 1e-10)
        self.assertTrue(result["rank_equal"])
        self.assertGreater(
            result["resource_frontier"]["factorized_multiplications"],
            result["resource_frontier"]["dense_multiplications"],
        )
        self.assertGreater(
            result["resource_frontier"]["factorized_additions"],
            result["resource_frontier"]["dense_additions"],
        )
        self.assertLessEqual(result["output_max_relative_error"], 1e-10)

    def test_spu0_is_deterministic_for_unit_fixture(self) -> None:
        first = run_spu0(seed=17, dimension=8, factor_count=3, vector_count=12)
        second = run_spu0(seed=17, dimension=8, factor_count=3, vector_count=12)
        first.pop("resource_estimate")
        second.pop("resource_estimate")
        self.assertEqual(first, second)

    def test_var1_graph_cut_matches_exact_oracle(self) -> None:
        result = run_var1(seed=17, n=8)
        for row in result["rows"]:
            self.assertAlmostEqual(
                row["methods"]["graph_cut"]["objective_gap"], 0.0, places=9
            )
            self.assertTrue(row["methods"]["graph_cut"]["feasible"])

    def test_var1_reports_all_frozen_controls_and_boundaries(self) -> None:
        result = run_var1(seed=17, n=8)
        self.assertEqual(len(result["rows"]), 4)
        for row in result["rows"]:
            self.assertEqual(
                set(row["methods"]),
                {"smooth_rounding", "proximal_graph_tv", "graph_cut", "submodular_greedy"},
            )
            for metrics in row["methods"].values():
                self.assertGreaterEqual(metrics["objective_gap"], -1e-9)
                self.assertGreaterEqual(metrics["selection_jaccard"], 0.0)
                self.assertLessEqual(metrics["selection_jaccard"], 1.0)
                self.assertTrue(metrics["feasible"])
        self.assertEqual(result["scientific_claim_status"], "UNCONFIRMED")

    def test_stage_artifact_binds_protocol_phase_and_digest(self) -> None:
        artifact = make_stage_artifact(
            "PRW-SPU0",
            seed=17,
            phase="UNIT-REDUCED",
            protocol_digest=FROZEN_PROTOCOL_SHA256,
            budget=Wave0ABudget(),
        )
        self.assertEqual(artifact["protocol_id"], PROTOCOL_ID)
        self.assertEqual(artifact["phase"], "UNIT-REDUCED")
        self.assertEqual(artifact["protocol_digest"], FROZEN_PROTOCOL_SHA256)
        self.assertTrue(artifact["artifact_digest"])
        verify_stage_artifact(artifact, stage_id="PRW-SPU0", seed=17, phase="UNIT-REDUCED")

    def test_forged_claim_is_rejected_even_with_recomputed_digest(self) -> None:
        artifact = make_stage_artifact(
            "PRW-SPU0", seed=17, phase="UNIT-FORGED",
            protocol_digest=FROZEN_PROTOCOL_SHA256, budget=Wave0ABudget(),
        )
        artifact["result"]["novelty_status"] = "CONFIRMED"
        unsigned = dict(artifact)
        unsigned.pop("artifact_digest")
        artifact["artifact_digest"] = sha256_payload(unsigned)
        with self.assertRaisesRegex(Wave0AValidationError, "novelty"):
            verify_stage_artifact(artifact, stage_id="PRW-SPU0", seed=17, phase="UNIT-FORGED")

    def test_resealed_spu_false_disposition_is_rejected(self) -> None:
        artifact = make_stage_artifact(
            "PRW-SPU0", seed=17, phase="UNIT-SPU-FORGE",
            protocol_digest=FROZEN_PROTOCOL_SHA256, budget=Wave0ABudget(),
        )
        artifact["result"]["output_max_abs_error"] = 999.0
        unsigned = dict(artifact)
        unsigned.pop("artifact_digest")
        artifact["artifact_digest"] = sha256_payload(unsigned)
        with self.assertRaisesRegex(Wave0AValidationError, "regeneration"):
            verify_stage_artifact(
                artifact, stage_id="PRW-SPU0", seed=17, phase="UNIT-SPU-FORGE"
            )

    def test_resealed_var1_flipped_null_is_rejected(self) -> None:
        artifact = make_stage_artifact(
            "PRW-VAR1", seed=17, phase="UNIT-VAR-FORGE",
            protocol_digest=FROZEN_PROTOCOL_SHA256, budget=Wave0ABudget(),
        )
        artifact["result"]["null_survives"] = not artifact["result"]["null_survives"]
        unsigned = dict(artifact)
        unsigned.pop("artifact_digest")
        artifact["artifact_digest"] = sha256_payload(unsigned)
        with self.assertRaisesRegex(Wave0AValidationError, "null/disposition"):
            verify_stage_artifact(
                artifact, stage_id="PRW-VAR1", seed=17, phase="UNIT-VAR-FORGE"
            )

    def test_resealed_bil1_false_invariants_are_rejected(self) -> None:
        artifact = make_stage_artifact(
            "PRW-BIL1", seed=17, phase="UNIT-BIL-FORGE",
            protocol_digest=FROZEN_PROTOCOL_SHA256, budget=Wave0ABudget(),
        )
        artifact["result"]["invariants"] = {
            key: False for key in artifact["result"]["invariants"]
        }
        artifact["result"]["disposition"] = "KILL_CANDIDATE_SEMANTICS"
        unsigned = dict(artifact)
        unsigned.pop("artifact_digest")
        artifact["artifact_digest"] = sha256_payload(unsigned)
        with self.assertRaisesRegex(Wave0AValidationError, "frozen invariants"):
            verify_stage_artifact(
                artifact, stage_id="PRW-BIL1", seed=17, phase="UNIT-BIL-FORGE"
            )

    def test_resealed_var1_rewritten_smooth_fixture_is_rejected(self) -> None:
        artifact = make_stage_artifact(
            "PRW-VAR1", seed=19, phase="UNIT-VAR-REWRITE",
            protocol_digest=FROZEN_PROTOCOL_SHA256, budget=Wave0ABudget(),
        )
        for row in artifact["result"]["rows"]:
            smooth = row["methods"]["smooth_rounding"]
            smooth.update({
                "objective": row["exact"]["objective"], "objective_gap": 0.0,
                "selection_jaccard": 1.0, "feasible": True,
                "downstream_recall": row["exact"]["downstream_recall"],
                "mask": list(row["exact"]["mask"]),
            })
        artifact["result"]["null_survives"] = True
        artifact["result"]["disposition"] = "RETAIN_SMOOTH_HARD_MASK_NULL"
        unsigned = dict(artifact)
        unsigned.pop("artifact_digest")
        artifact["artifact_digest"] = sha256_payload(unsigned)
        with self.assertRaisesRegex(Wave0AValidationError, "regeneration"):
            verify_stage_artifact(
                artifact, stage_id="PRW-VAR1", seed=19, phase="UNIT-VAR-REWRITE"
            )

    def test_resealed_negative_budget_is_rejected(self) -> None:
        artifact = make_stage_artifact(
            "PRW-SPU0", seed=17, phase="UNIT-BUDGET-FORGE",
            protocol_digest=FROZEN_PROTOCOL_SHA256, budget=Wave0ABudget(),
        )
        artifact["budget"]["max_estimated_bytes"] = -1
        unsigned = dict(artifact)
        unsigned.pop("artifact_digest")
        artifact["artifact_digest"] = sha256_payload(unsigned)
        with self.assertRaises(Wave0AValidationError):
            verify_stage_artifact(
                artifact, stage_id="PRW-SPU0", seed=17, phase="UNIT-BUDGET-FORGE"
            )

    def test_resealed_artifact_budget_above_frozen_bounds_is_rejected(self) -> None:
        attacks = {
            "max_estimated_bytes": 3 * 1024**3,
            "max_seconds_per_stage": 999999.0,
            "max_output_bytes": 999999999,
        }
        for field, value in attacks.items():
            with self.subTest(field=field):
                artifact = make_stage_artifact(
                    "PRW-SPU0", seed=17, phase="UNIT-BUDGET-BOUNDS",
                    protocol_digest=FROZEN_PROTOCOL_SHA256, budget=Wave0ABudget(),
                )
                artifact["budget"][field] = value
                unsigned = dict(artifact)
                unsigned.pop("artifact_digest")
                artifact["artifact_digest"] = sha256_payload(unsigned)
                with self.assertRaisesRegex(Wave0AValidationError, "frozen bounds"):
                    verify_stage_artifact(
                        artifact, stage_id="PRW-SPU0", seed=17,
                        phase="UNIT-BUDGET-BOUNDS",
                    )

    def test_resealed_artifact_budget_rejects_integer_deadline_type(self) -> None:
        artifact = make_stage_artifact(
            "PRW-SPU0", seed=17, phase="UNIT-BUDGET-TYPE",
            protocol_digest=FROZEN_PROTOCOL_SHA256, budget=Wave0ABudget(),
        )
        artifact["budget"]["max_seconds_per_stage"] = 120
        unsigned = dict(artifact)
        unsigned.pop("artifact_digest")
        artifact["artifact_digest"] = sha256_payload(unsigned)
        with self.assertRaisesRegex(Wave0AValidationError, "deadline budget"):
            verify_stage_artifact(
                artifact, stage_id="PRW-SPU0", seed=17, phase="UNIT-BUDGET-TYPE"
            )

    def test_v7_resource_failure_is_below_exact_v8_ek0_reservation(self) -> None:
        v7_sampled_peak = 590_852_096
        v7_sampled_backstop = 570_822_656
        v8_aggregate_reservation = 768 * 1024**2
        self.assertGreater(v7_sampled_peak, v7_sampled_backstop)
        self.assertLess(v7_sampled_peak, v8_aggregate_reservation)
        parent_rss = 64 * 1024**2
        worker, sampled_backstop = _stage_worker_reservation(
            "PRW-EK0", parent_rss, 768 * 1024**2
        )
        self.assertEqual(sampled_backstop, v8_aggregate_reservation)
        self.assertEqual(worker, v8_aggregate_reservation - parent_rss)
        with self.assertRaises(Wave0AResourceError):
            _stage_worker_reservation("PRW-EK0", parent_rss, 512 * 1024**2)
        protocol = Path(
            "docs/research/rb13-wave0a-method-dev-protocol-2026-08-15.md"
        ).read_text(encoding="utf-8")
        self.assertIn("`COMPLETE` EK0 stage artifact for seed", protocol)
        self.assertIn("no valid or published campaign result", protocol)
        self.assertIn("`status=INVALID_RUN`", protocol)
        self.assertIn("`failure_category=RESOURCE_OR_DEADLINE`", protocol)
        self.assertNotIn("INVALID_RUN_RESOURCE", protocol)
        self.assertIn("stage metrics were not inspected or used", protocol)

    def test_nonfinite_artifact_is_rejected_recursively(self) -> None:
        artifact = make_stage_artifact(
            "PRW-SPU0", seed=17, phase="UNIT-NAN",
            protocol_digest=FROZEN_PROTOCOL_SHA256, budget=Wave0ABudget(),
        )
        artifact["result"]["output_max_abs_error"] = float("nan")
        with self.assertRaisesRegex(Wave0AValidationError, "nonfinite"):
            verify_stage_artifact(artifact, stage_id="PRW-SPU0", seed=17, phase="UNIT-NAN")

    def test_atomic_json_has_no_temporary_residue(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "artifact.json"
            atomic_write_json(output, {"value": 1}, 1024)
            self.assertEqual(json.loads(output.read_text(encoding="utf-8")), {"value": 1})
            self.assertFalse(list(output.parent.glob("*.tmp")))

    def test_runner_rejects_protocol_digest_mismatch_before_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            protocol = root / "protocol.md"
            protocol.write_text("frozen", encoding="utf-8")
            attacker_digest = hashlib.sha256(protocol.read_bytes()).hexdigest()
            output = root / "official"
            with self.assertRaises(Wave0AValidationError):
                run(
                    output_directory=output,
                    phase="SELECT",
                    protocol_path=protocol,
                    expected_protocol_sha256=attacker_digest,
                )
            self.assertFalse(output.exists())

    def test_repository_protocol_matches_embedded_digest(self) -> None:
        protocol = Path("docs/research/rb13-wave0a-method-dev-protocol-2026-08-15.md")
        self.assertEqual(hashlib.sha256(protocol.read_bytes()).hexdigest(), FROZEN_PROTOCOL_SHA256)

    def test_actual_copied_linux_archives_verify_portably(self) -> None:
        root = Path("artifacts/method-dev/rb13-wave0a")
        select = verify_archived_campaign(root / "select", expected_phase="SELECT")
        report = verify_archived_campaign(root / "report", expected_phase="REPORT")
        cross = verify_archived_cross_phase(root / "select", root / "report")
        self.assertEqual(select["status"], "ARCHIVED_CAMPAIGN_VERIFIED")
        self.assertEqual(report["status"], "ARCHIVED_CAMPAIGN_VERIFIED")
        self.assertEqual(cross["status"], "ARCHIVED_CROSS_PHASE_VERIFIED")
        self.assertEqual(
            select["manifest_file_sha256_custody_root"],
            "f17defbae0c6ac7a49a5eba295604ead4a70e66ee5dae749c7f2289153339ab2",
        )
        self.assertEqual(
            report["manifest_file_sha256_custody_root"],
            "e03f090d632307f58f5316f2dbec9a2b214ecc8bcbcff5cc8910bbd17a371a2c",
        )
        select_is_exact = select["maximum_absolute_regeneration_difference"] == 0.0
        report_is_exact = report["maximum_absolute_regeneration_difference"] == 0.0
        cross_is_exact = cross["maximum_absolute_regeneration_difference"] == 0.0
        self.assertEqual(select["current_platform_exact_regeneration"], select_is_exact)
        self.assertEqual(report["current_platform_exact_regeneration"], report_is_exact)
        self.assertEqual(cross["current_platform_exact_regeneration"], cross_is_exact)
        self.assertEqual(cross_is_exact, select_is_exact and report_is_exact)
        for phase_name, phase_is_exact in (
            ("SELECT", select_is_exact),
            ("REPORT", report_is_exact),
        ):
            if phase_is_exact:
                self.assertEqual(
                    verify_campaign(
                        root / phase_name.lower(), expected_phase=phase_name
                    )["status"],
                    "VERIFIED",
                )
            else:
                with self.assertRaisesRegex(Wave0AValidationError, "regeneration"):
                    verify_campaign(
                        root / phase_name.lower(), expected_phase=phase_name
                    )
        if not cross_is_exact:
            self.assertGreater(cross["maximum_absolute_regeneration_difference"], 0.0)
        self.assertLessEqual(
            cross["maximum_absolute_regeneration_difference"], 2.0e-15
        )

    def test_archived_campaign_rejects_coherently_resealed_numeric_tamper(self) -> None:
        source = Path("artifacts/method-dev/rb13-wave0a/select")
        with tempfile.TemporaryDirectory() as directory:
            copied = Path(directory) / "select"
            shutil.copytree(source, copied)
            manifest_path = copied / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            entry = next(
                item for item in manifest["entries"]
                if item["stage_id"] == "PRW-VAR1" and item["seed"] == 1301
            )
            artifact_path = copied / entry["path"]
            artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
            row = artifact["result"]["rows"][0]
            delta = 1.0e-12
            row["exact"]["objective"] += delta
            row["integrality_gap"] += delta
            for metrics in row["methods"].values():
                metrics["objective"] += delta
            unsigned_artifact = dict(artifact)
            unsigned_artifact.pop("artifact_digest")
            artifact["artifact_digest"] = sha256_payload(unsigned_artifact)
            artifact_path.write_bytes(
                (json.dumps(
                    artifact, sort_keys=True, separators=(",", ":")
                ) + "\n").encode("utf-8")
            )
            entry["artifact_digest"] = artifact["artifact_digest"]
            entry["file_sha256"] = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
            unsigned_manifest = dict(manifest)
            unsigned_manifest.pop("manifest_digest")
            manifest["manifest_digest"] = sha256_payload(unsigned_manifest)
            manifest_path.write_bytes(
                (json.dumps(
                    manifest, sort_keys=True, separators=(",", ":")
                ) + "\n").encode("utf-8")
            )
            tampered_root = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
            with patch.dict(
                run_rb13_wave0a.ARCHIVE_MANIFEST_FILE_SHA256,
                {"SELECT": tampered_root},
            ):
                with self.assertRaisesRegex(Wave0AValidationError, "numeric drift"):
                    verify_archived_campaign(copied, expected_phase="SELECT")

    def test_archived_campaign_rejects_noncanonical_manifest_bytes(self) -> None:
        source = Path("artifacts/method-dev/rb13-wave0a/select")
        with tempfile.TemporaryDirectory() as directory:
            copied = Path(directory) / "select"
            shutil.copytree(source, copied)
            manifest_path = copied / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest_path.write_text(
                json.dumps(manifest, sort_keys=True, indent=2) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(Wave0AValidationError, "not canonical"):
                verify_archived_campaign(copied, expected_phase="SELECT")

    def test_archived_campaign_custody_root_rejects_coherent_runtime_reseal(self) -> None:
        source = Path("artifacts/method-dev/rb13-wave0a/select")
        with tempfile.TemporaryDirectory() as directory:
            copied = Path(directory) / "select"
            shutil.copytree(source, copied)
            manifest_path = copied / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            entry = next(
                item for item in manifest["entries"]
                if item["stage_id"] == "PRW-VAR1" and item["seed"] == 1301
            )
            artifact_path = copied / entry["path"]
            artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
            artifact["result"]["rows"][0]["runtime_seconds"] += 1.0
            unsigned_artifact = dict(artifact)
            unsigned_artifact.pop("artifact_digest")
            artifact["artifact_digest"] = sha256_payload(unsigned_artifact)
            artifact_path.write_bytes(
                (json.dumps(
                    artifact, sort_keys=True, separators=(",", ":")
                ) + "\n").encode("utf-8")
            )
            entry["artifact_digest"] = artifact["artifact_digest"]
            entry["file_sha256"] = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
            unsigned_manifest = dict(manifest)
            unsigned_manifest.pop("manifest_digest")
            manifest["manifest_digest"] = sha256_payload(unsigned_manifest)
            manifest_path.write_bytes(
                (json.dumps(
                    manifest, sort_keys=True, separators=(",", ":")
                ) + "\n").encode("utf-8")
            )
            with self.assertRaisesRegex(Wave0AValidationError, "custody root"):
                verify_archived_campaign(copied, expected_phase="SELECT")

    def test_archived_campaign_rejects_unexpected_regular_file(self) -> None:
        source = Path("artifacts/method-dev/rb13-wave0a/select")
        with tempfile.TemporaryDirectory() as directory:
            copied = Path(directory) / "select"
            shutil.copytree(source, copied)
            (copied / "unexpected.txt").write_text("not in manifest", encoding="utf-8")
            with self.assertRaisesRegex(Wave0AValidationError, "regular-file set"):
                verify_archived_campaign(copied, expected_phase="SELECT")

    def test_archived_campaign_rejects_unsafe_manifest_path_after_root_check(self) -> None:
        source = Path("artifacts/method-dev/rb13-wave0a/select")
        with tempfile.TemporaryDirectory() as directory:
            copied = Path(directory) / "select"
            shutil.copytree(source, copied)
            manifest_path = copied / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["entries"][0]["path"] = "../escape.json"
            unsigned = dict(manifest)
            unsigned.pop("manifest_digest")
            manifest["manifest_digest"] = sha256_payload(unsigned)
            manifest_path.write_bytes(
                (json.dumps(
                    manifest, sort_keys=True, separators=(",", ":")
                ) + "\n").encode("utf-8")
            )
            tampered_root = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
            with patch.dict(
                run_rb13_wave0a.ARCHIVE_MANIFEST_FILE_SHA256,
                {"SELECT": tampered_root},
            ):
                with self.assertRaisesRegex(Wave0AValidationError, "unsafe"):
                    verify_archived_campaign(copied, expected_phase="SELECT")

    def test_archived_campaign_rejects_symlink_file_or_directory(self) -> None:
        source = Path("artifacts/method-dev/rb13-wave0a/select")
        with tempfile.TemporaryDirectory() as directory:
            copied = Path(directory) / "select"
            shutil.copytree(source, copied)
            target = copied / "manifest.json"
            link = copied / "linked-manifest.json"
            try:
                os.symlink(target, link)
            except OSError as error:
                self.skipTest(f"symlink creation unavailable: {error}")
            with self.assertRaisesRegex(Wave0AValidationError, "linked/nonregular"):
                verify_archived_campaign(copied, expected_phase="SELECT")
            link.unlink()
            directory_link = copied / "linked-seed"
            os.symlink(copied / "seed-1301", directory_link, target_is_directory=True)
            with self.assertRaisesRegex(Wave0AValidationError, "linked directory"):
                verify_archived_campaign(copied, expected_phase="SELECT")

    def test_archived_regeneration_rejects_every_nonfinite_float_case(self) -> None:
        for observed, expected in (
            (float("nan"), 0.0), (float("inf"), 0.0),
            (float("-inf"), 0.0), (0.0, float("nan")),
            (0.0, float("inf")), (1.0e308, -1.0e308),
        ):
            with self.subTest(observed=observed, expected=expected):
                with self.assertRaisesRegex(Wave0AValidationError, "nonfinite"):
                    _verify_archived_regeneration(
                        observed, expected, absolute_tolerance=2.0e-15,
                        path="hostile-float",
                    )

    def test_portable_reparse_detection_rejects_mocked_root(self) -> None:
        with patch("run_rb13_wave0a._path_is_reparse_point", return_value=True):
            with self.assertRaisesRegex(Wave0AValidationError, "root cannot be a link"):
                _archive_regular_file_set(Path("portable-mocked-root"))

    @unittest.skipUnless(sys.platform == "win32", "Windows junction regression")
    def test_windows_mklink_junction_root_and_nested_are_rejected(self) -> None:
        source = Path("artifacts/method-dev/rb13-wave0a/select").resolve()
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory).resolve()
            root_link = base / "root-junction"
            created = subprocess.run(
                ["cmd", "/c", "mklink", "/J", str(root_link), str(source)],
                capture_output=True, text=True, check=False,
            )
            if created.returncode != 0:
                self.skipTest(f"mklink /J unavailable: {created.stderr or created.stdout}")
            try:
                with self.assertRaisesRegex(Wave0AValidationError, "root cannot be a link"):
                    verify_archived_campaign(root_link, expected_phase="SELECT")
            finally:
                os.rmdir(root_link)

            copied = base / "select"
            shutil.copytree(source, copied)
            empty_target = base / "empty-target"
            empty_target.mkdir()
            nested_link = copied / "nested-junction"
            created = subprocess.run(
                ["cmd", "/c", "mklink", "/J", str(nested_link), str(empty_target)],
                capture_output=True, text=True, check=False,
            )
            self.assertEqual(created.returncode, 0, created.stderr or created.stdout)
            try:
                with self.assertRaisesRegex(Wave0AValidationError, "linked directory"):
                    verify_archived_campaign(copied, expected_phase="SELECT")
            finally:
                os.rmdir(nested_link)

    def test_runner_rejects_existing_output_before_work(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            protocol = root / "protocol.md"
            protocol.write_text("frozen", encoding="utf-8")
            digest = hashlib.sha256(protocol.read_bytes()).hexdigest()
            output = root / "official"
            output.mkdir()
            with self.assertRaises(Wave0AValidationError):
                run(
                    output_directory=output,
                    phase="REPORT",
                    protocol_path=protocol,
                    expected_protocol_sha256=digest,
                )

    def test_reduced_runner_writes_content_bound_manifest_atomically(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            protocol = Path("docs/research/rb13-wave0a-method-dev-protocol-2026-08-15.md")
            output = root / "reduced"
            with patch.object(run_rb13_wave0a, "SELECT_SEEDS", (17,)), patch.object(
                run_rb13_wave0a, "STAGE_IDS", ("PRW-SPU0",)
            ):
                manifest = run(
                    output_directory=output,
                    phase="SELECT",
                    protocol_path=protocol,
                    expected_protocol_sha256=FROZEN_PROTOCOL_SHA256,
                )
            self.assertEqual(manifest["entry_count"], 1)
            self.assertEqual(manifest["config"]["rss_sample_interval_seconds"], 0.05)
            self.assertEqual(
                manifest["config"]["rss_sampling_limitation"],
                "short_lived_touched_memory_overages_may_evade_sampling",
            )
            self.assertTrue((output / "manifest.json").is_file())
            self.assertFalse(list(root.glob("*.partial-*")))
            entry = manifest["entries"][0]
            artifact_path = output / entry["path"]
            self.assertEqual(hashlib.sha256(artifact_path.read_bytes()).hexdigest(), entry["file_sha256"])
            with patch.object(run_rb13_wave0a, "SELECT_SEEDS", (17,)), patch.object(
                run_rb13_wave0a, "STAGE_IDS", ("PRW-SPU0",)
            ):
                self.assertEqual(verify_campaign(output)["status"], "VERIFIED")
                artifact_path.write_text("{}", encoding="utf-8")
                with self.assertRaisesRegex(Wave0AValidationError, "file SHA"):
                    verify_campaign(output)

    def test_runner_quarantines_partial_directory_on_worker_failure(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            protocol = Path("docs/research/rb13-wave0a-method-dev-protocol-2026-08-15.md")
            output = root / "failed"
            with patch.object(run_rb13_wave0a, "SELECT_SEEDS", (17,)), patch.object(
                run_rb13_wave0a, "STAGE_IDS", ("PRW-SPU0",)
            ), patch("run_rb13_wave0a._execute_worker", return_value=(2, "", "frozen failure", 1)):
                with self.assertRaisesRegex(RuntimeError, "frozen failure"):
                    run(
                        output_directory=output,
                        phase="SELECT",
                        protocol_path=protocol,
                        expected_protocol_sha256=FROZEN_PROTOCOL_SHA256,
                    )
            self.assertFalse(output.exists())
            quarantines = list(root.glob("failed.QUARANTINED-*"))
            self.assertEqual(len(quarantines), 1)
            invalid = json.loads((quarantines[0] / "invalid_run.json").read_text(encoding="utf-8"))
            self.assertEqual(invalid["status"], "INVALID_RUN")
            self.assertEqual(invalid["failure_category"], "WORKER_NONZERO")

    def test_hostile_unicode_worker_failure_is_byte_bounded_and_original_is_raised(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            protocol = Path(
                "docs/research/rb13-wave0a-method-dev-protocol-2026-08-15.md"
            )
            hostile = ("😀💥\x00\x01\"\\\n" * 4000) + "TAIL-SENTINEL"
            output = root / "hostile"
            with patch.object(run_rb13_wave0a, "SELECT_SEEDS", (17,)), patch.object(
                run_rb13_wave0a, "STAGE_IDS", ("PRW-SPU0",)
            ), patch(
                "run_rb13_wave0a._execute_worker",
                return_value=(2, "", hostile, 1),
            ):
                with self.assertRaises(RuntimeError) as raised:
                    run(output_directory=output, phase="SELECT", protocol_path=protocol)
            self.assertIn("TAIL-SENTINEL", str(raised.exception))
            quarantine = next(root.glob("hostile.QUARANTINED-*"))
            invalid_path = quarantine / "invalid_run.json"
            self.assertTrue(invalid_path.is_file())
            self.assertGreater(invalid_path.stat().st_size, 0)
            self.assertLessEqual(invalid_path.stat().st_size, 16 * 1024)
            invalid = json.loads(invalid_path.read_text(encoding="utf-8"))
            self.assertTrue(invalid["detail_truncated"])
            self.assertTrue(invalid["detail"].startswith("...[UTF8_BUDGET_TRUNCATED]"))
            self.assertEqual(
                invalid["detail_full_sha256"],
                hashlib.sha256(hostile.encode("utf-8")).hexdigest(),
            )

    def test_evidence_serialization_failure_does_not_replace_worker_error(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            protocol = Path(
                "docs/research/rb13-wave0a-method-dev-protocol-2026-08-15.md"
            )
            output = root / "evidence-failure"
            with patch.object(run_rb13_wave0a, "SELECT_SEEDS", (17,)), patch.object(
                run_rb13_wave0a, "STAGE_IDS", ("PRW-SPU0",)
            ), patch(
                "run_rb13_wave0a._execute_worker",
                return_value=(2, "", "ORIGINAL-WORKER-ERROR", 1),
            ), patch(
                "run_rb13_wave0a._invalid_run_payload",
                side_effect=ValueError("secondary evidence failure"),
            ):
                with self.assertRaisesRegex(RuntimeError, "ORIGINAL-WORKER-ERROR"):
                    run(output_directory=output, phase="SELECT", protocol_path=protocol)
            self.assertTrue(next(root.glob("evidence-failure.QUARANTINED-*"), None))

    def test_runner_quarantines_immediate_parent_rss_failure(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            protocol = Path(
                "docs/research/rb13-wave0a-method-dev-protocol-2026-08-15.md"
            )
            output = root / "rss-immediate"
            with patch(
                "run_rb13_wave0a.current_rss_bytes",
                side_effect=RuntimeError("immediate rss failure"),
            ):
                with self.assertRaisesRegex(RuntimeError, "immediate rss failure"):
                    run(output_directory=output, phase="SELECT", protocol_path=protocol)
            self.assertFalse(output.exists())
            quarantine = next(root.glob("rss-immediate.QUARANTINED-*"))
            invalid = json.loads(
                (quarantine / "invalid_run.json").read_text(encoding="utf-8")
            )
            self.assertEqual(invalid["status"], "INVALID_RUN")
            self.assertEqual(invalid["failure_category"], "CAMPAIGN_FAILURE")
            self.assertEqual(invalid["stage_id"], "CAMPAIGN")

    def test_resource_failure_persists_invalid_run_identity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            protocol = Path("docs/research/rb13-wave0a-method-dev-protocol-2026-08-15.md")
            with patch.object(run_rb13_wave0a, "SELECT_SEEDS", (17,)), patch.object(
                run_rb13_wave0a, "STAGE_IDS", ("PRW-SPU0",)
            ), patch("run_rb13_wave0a._execute_worker", side_effect=Wave0AResourceError("rss stop")):
                with self.assertRaises(Wave0AResourceError):
                    run(output_directory=root / "rss", phase="SELECT", protocol_path=protocol)
            quarantine = next(root.glob("rss.QUARANTINED-*"))
            invalid = json.loads((quarantine / "invalid_run.json").read_text(encoding="utf-8"))
            self.assertEqual(invalid["failure_category"], "RESOURCE_OR_DEADLINE")
            self.assertEqual((invalid["stage_id"], invalid["seed"]), ("PRW-SPU0", 17))

    def test_campaign_rejects_resealed_sampled_peak_over_modeled_ceiling(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "reduced"
            protocol = Path(
                "docs/research/rb13-wave0a-method-dev-protocol-2026-08-15.md"
            )
            with patch.object(run_rb13_wave0a, "SELECT_SEEDS", (17,)), patch.object(
                run_rb13_wave0a, "STAGE_IDS", ("PRW-SPU0",)
            ):
                manifest = run(
                    output_directory=output, phase="SELECT", protocol_path=protocol
                )
                manifest["entries"][0]["sampled_peak_parent_plus_worker_rss_bytes"] = (
                    manifest["entries"][0]["sampled_rss_backstop_ceiling_bytes"] + 1
                )
                unsigned = dict(manifest)
                unsigned.pop("manifest_digest")
                manifest["manifest_digest"] = sha256_payload(unsigned)
                (output / "manifest.json").write_text(
                    json.dumps(manifest), encoding="utf-8"
                )
                with self.assertRaisesRegex(Wave0AValidationError, "telemetry"):
                    verify_campaign(output)

    def test_campaign_rejects_platform_inconsistent_monitor_flags(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "reduced"
            protocol = Path(
                "docs/research/rb13-wave0a-method-dev-protocol-2026-08-15.md"
            )
            with patch.object(run_rb13_wave0a, "SELECT_SEEDS", (17,)), patch.object(
                run_rb13_wave0a, "STAGE_IDS", ("PRW-SPU0",)
            ):
                manifest = run(
                    output_directory=output, phase="SELECT", protocol_path=protocol
                )
                config = manifest["config"]
                config["linux_sampled_process_tree_monitor"] = not config[
                    "linux_sampled_process_tree_monitor"
                ]
                config["windows_sampled_direct_child_monitor_only"] = not config[
                    "windows_sampled_direct_child_monitor_only"
                ]
                manifest["config_digest"] = sha256_payload(config)
                unsigned = dict(manifest)
                unsigned.pop("manifest_digest")
                manifest["manifest_digest"] = sha256_payload(unsigned)
                (output / "manifest.json").write_text(
                    json.dumps(manifest), encoding="utf-8"
                )
                with self.assertRaisesRegex(Wave0AValidationError, "monitor scope"):
                    verify_campaign(output)

    def test_zz_resealed_empty_ek0_witness_is_rejected(self) -> None:
        artifact = make_stage_artifact(
            "PRW-EK0", seed=17, phase="UNIT-EK-FORGE",
            protocol_digest=FROZEN_PROTOCOL_SHA256, budget=Wave0ABudget(),
        )
        artifact["result"]["observations"] = {}
        artifact["result"]["null_survives"] = True
        artifact["result"]["disposition"] = "NULL_SURVIVES"
        unsigned = dict(artifact)
        unsigned.pop("artifact_digest")
        artifact["artifact_digest"] = sha256_payload(unsigned)
        with self.assertRaisesRegex(Wave0AValidationError, "frozen store witness"):
            verify_stage_artifact(
                artifact, stage_id="PRW-EK0", seed=17, phase="UNIT-EK-FORGE"
            )

    def test_var1_seed_variance_aggregates_corresponding_cells_only(self) -> None:
        artifacts = [
            make_stage_artifact(
                "PRW-VAR1", seed=seed, phase="UNIT-SUMMARY",
                protocol_digest=FROZEN_PROTOCOL_SHA256, budget=Wave0ABudget(),
            )
            for seed in (17, 19)
        ]
        summary = _campaign_summary(artifacts, "UNIT-SUMMARY")
        self.assertTrue(summary["cross_cell_values_not_used_as_seed_variance"])
        self.assertEqual(
            {cell["seed_count"] for cell in summary["var1_corresponding_cell_seed_summary"].values()},
            {2},
        )
        self.assertNotIn("seed_variance", artifacts[0]["result"])

    def test_cross_phase_summary_keeps_synthetic_label_and_corresponding_cells(self) -> None:
        select = {
            "status": "VERIFIED", "phase": "SELECT", "entry_count": 1,
            "summary": {
                "var1_corresponding_cell_seed_summary": {"cell-1": {"mean_objective_gap": 0.2}},
                "var1_null_survives_all_seeds_and_cells": False,
            },
        }
        report = {
            "status": "VERIFIED", "phase": "REPORT", "entry_count": 1,
            "summary": {
                "var1_corresponding_cell_seed_summary": {"cell-1": {"mean_objective_gap": 0.3}},
                "var1_null_survives_all_seeds_and_cells": False,
            },
        }
        with patch("run_rb13_wave0a.verify_campaign", side_effect=(select, report)):
            result = verify_cross_phase(Path("select"), Path("report"))
        self.assertEqual(result["fixture_scope"], "synthetic_non_independent_labels")
        self.assertAlmostEqual(result["corresponding_cell_mean_gap_delta"]["cell-1"], 0.1)
        self.assertFalse(result["var1_cross_phase_null_survives"])
        self.assertEqual(
            result["var1_cross_phase_disposition"],
            "ROUTE_SMOOTH_METHOD_TO_DECLARED_STATE_SUBPROBLEM_ONLY",
        )

    @unittest.skipIf(sys.platform == "win32", "Linux process-group smoke")
    def test_linux_monitor_kills_descendant_after_leader_exit(self) -> None:
        code = (
            "import subprocess,sys;"
            "subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)'],"
            "stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)"
        )
        returncode, _, _, _ = _execute_worker(
            [sys.executable, "-c", code], dict(os.environ),
            deadline_seconds=2.0, effective_ceiling_bytes=512 * 1024**2,
        )
        self.assertEqual(returncode, 0)

    def test_numpy_nonfinite_outputs_are_not_present(self) -> None:
        result = run_spu0(seed=17, dimension=8, factor_count=3, vector_count=12)
        self.assertTrue(np.isfinite(result["output_max_abs_error"]))
        self.assertTrue(np.isfinite(result["distance_max_abs_error"]))


if __name__ == "__main__":
    unittest.main()
