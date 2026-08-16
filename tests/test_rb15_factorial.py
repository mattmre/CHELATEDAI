import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

import rb15_factorial_experiments as factorial
import run_rb15_factorial as runner


class RB15FactorialCoreTests(unittest.TestCase):
    def test_preflight_freezes_matched_factorial_and_work_count(self):
        result = factorial.resource_preflight()
        self.assertTrue(result["design_validation_passed"])
        self.assertEqual(result["factorial_cell_count"], 136)
        self.assertEqual(result["report_only_control_cell_count"], 34)
        self.assertEqual(result["total_cell_count"], 170)
        self.assertEqual(result["modeled_rk4_step_count"], 2_040_000)
        totals = result["configuration_totals"]
        for suffix in ("linear", "duffing"):
            single = totals[f"single_{suffix}"]
            distributed = totals[f"distributed_{suffix}"]
            for field in (
                "auxiliary_mass_total",
                "attachment_linear_stiffness_total",
                "attachment_relative_damping_total",
                "attachment_cubic_stiffness_total",
            ):
                self.assertAlmostEqual(single[field], distributed[field], places=15)
        self.assertEqual(totals["single_duffing"]["sidecar_count"], 1)
        self.assertEqual(totals["distributed_duffing"]["sidecar_count"], 2)

    def test_design_preflight_rejects_host_and_ordered_attachment_drift(self):
        configurations = factorial._configurations()
        changed_host = factorial._expected_host_stiffness()
        changed_host[0, 0] += 0.01
        host_drift = dict(configurations)
        host_drift["single_linear"] = factorial.GraphSystem(
            changed_host, configurations["single_linear"].attachments
        )
        with patch.object(factorial, "_configurations", return_value=host_drift):
            with self.assertRaises(factorial.FactorialValidationError):
                factorial.resource_preflight()

        order_drift = dict(configurations)
        distributed = configurations["distributed_linear"]
        order_drift["distributed_linear"] = factorial.GraphSystem(
            distributed.stiffness, tuple(reversed(distributed.attachments))
        )
        with patch.object(factorial, "_configurations", return_value=order_drift):
            with self.assertRaises(factorial.FactorialValidationError):
                factorial.resource_preflight()

    def test_initial_states_use_fixed_ids_and_zero_local_mismatch(self):
        for run_id in factorial.RUN_IDS:
            for system in factorial._configurations().values():
                state = factorial._initial_state(system, run_id)
                offset = 2 * system.node_count
                for index, attachment in enumerate(system.attachments):
                    self.assertEqual(state[offset + index], state[attachment.node])
        with self.assertRaises(factorial.FactorialValidationError):
            factorial._initial_state(factorial._configurations()["single_linear"], 8)

    def test_resource_guard_fails_closed_on_deadline_and_rss(self):
        with patch.object(factorial, "_self_rss_bytes", return_value=(1, 1, "fake")):
            deadline = factorial.ResourceGuard(
                started=time.perf_counter() - 2.0, deadline_seconds=1.0
            )
            with self.assertRaises(factorial.ResourceLimitExceeded) as caught:
                deadline.check("fixture")
            self.assertEqual(caught.exception.code, "COOPERATIVE_DEADLINE_EXCEEDED")
        with patch.object(
            factorial, "_self_rss_bytes", return_value=(101, 101, "fake")
        ):
            rss = factorial.ResourceGuard(
                started=time.perf_counter(), rss_ceiling_bytes=100
            )
            with self.assertRaises(factorial.ResourceLimitExceeded) as caught:
                rss.check("fixture")
            self.assertEqual(caught.exception.code, "SELF_PROCESS_RSS_CEILING_EXCEEDED")

    def test_initial_rss_breach_is_failed_runtime_admission_not_design_preflight(self):
        over_ceiling = factorial.MAX_RSS_BYTES + 1
        with patch.object(
            factorial,
            "_self_rss_bytes",
            return_value=(over_ceiling, over_ceiling, "fixture"),
        ):
            payload = factorial.run_factorial(7)
        self.assertTrue(payload["design_preflight"]["design_validation_passed"])
        self.assertFalse(payload["resource_usage"]["runtime_admission"]["passed"])
        self.assertFalse(payload["resource_usage"]["rss_guard_passed"])
        self.assertEqual(
            payload["status"], "FACTORIAL_KILLED_ON_FROZEN_SYNTHETIC_GATES"
        )

    @staticmethod
    def _synthetic_cell(frequency, gain, settling=0.01):
        return {
            "frequency": frequency,
            "finite_trajectory": True,
            "endpoint_fundamental_transfer_gain": gain,
            "response": {"relative_settling_delta": settling},
        }

    def test_endpoints_encode_distribution_by_nonlinearity_interaction(self):
        frequencies = np.asarray((0.8, 1.0), dtype=np.float64)
        branches = {}
        gains = {
            "single_linear": 1.0,
            "distributed_linear": 1.0,
            "single_duffing": 1.0,
            "distributed_duffing": 0.8,
            factorial.CONTROL_CONFIGURATION: 1.2,
        }
        for name, gain in gains.items():
            branches[name] = {
                direction: [
                    self._synthetic_cell(float(frequency), gain)
                    for frequency in frequency_values
                ]
                for direction, frequency_values in (
                    ("forward", frequencies),
                    ("reverse", frequencies[::-1]),
                )
            }
        with patch.object(factorial, "FREQUENCIES", frequencies):
            endpoints = factorial.compute_endpoints(branches)
        self.assertAlmostEqual(endpoints["aggregate_interaction_ratio"], 0.2)
        self.assertAlmostEqual(endpoints["distributed_duffing_mean_gain_ratio"], 0.8)
        self.assertAlmostEqual(
            endpoints["worst_case_distributed_duffing_excess_ratio"], -0.2
        )
        self.assertEqual(endpoints["maximum_relative_hysteresis"], 0.0)
        self.assertTrue(all(gate["passed"] for gate in factorial.evaluate_gates(endpoints)))

    def test_reduced_fixture_is_strict_json_and_preserves_policy(self):
        frequencies = np.asarray((0.8,), dtype=np.float64)
        with patch.multiple(
            factorial,
            FREQUENCIES=frequencies,
            TOTAL_PERIODS=2,
            MEASURE_PERIODS=1,
            STEPS_PER_PERIOD=20,
            MAX_COMPUTATION_SECONDS=30.0,
        ):
            payload = factorial.run_factorial(7, reduced_fixture=True)
        encoded = json.dumps(payload, allow_nan=False, sort_keys=True)
        self.assertIsInstance(encoded, str)
        self.assertEqual(
            payload["phase_and_continuation_policy"], factorial.FORCING_PHASE_POLICY
        )
        self.assertEqual(payload["evidence_mode"], "REDUCED_TEST_FIXTURE_NOT_OFFICIAL_PROTOCOL")
        self.assertEqual(
            sum(
                len(cells)
                for directions in payload["branches"].values()
                for cells in directions.values()
            ),
            10,
        )
        self.assertIsNotNone(payload["endpoints"])
        self.assertTrue(payload["resource_usage"]["deadline_enforced_during_integration"])
        self.assertTrue(payload["resource_usage"]["rss_enforced_during_integration"])


class RB15FactorialArtifactTests(unittest.TestCase):
    @staticmethod
    def _harmonic(amplitude):
        return {
            "amplitude": amplitude,
            "offset": 0.0,
            "sin_coefficient": amplitude,
            "cos_coefficient": 0.0,
            "residual_rms": 0.0,
        }

    @classmethod
    def _response(cls, amplitude, settling=0.01):
        return {
            "fundamental": cls._harmonic(amplitude),
            "third_harmonic": cls._harmonic(0.01),
            "previous_half_fundamental_amplitude": amplitude * (1.0 + settling),
            "final_half_fundamental_amplitude": amplitude,
            "relative_settling_delta": settling,
        }

    @staticmethod
    def _payload(run_id=7):
        over_ceiling = factorial.MAX_RSS_BYTES + 1
        with patch.object(
            factorial,
            "_self_rss_bytes",
            return_value=(over_ceiling, over_ceiling, "fixture"),
        ):
            return factorial.run_factorial(run_id)

    @classmethod
    def _survivor_payload(cls, run_id=7):
        payload = cls._payload(run_id)
        gains = {
            "single_linear": 1.0,
            "distributed_linear": 1.0,
            "single_duffing": 1.0,
            "distributed_duffing": 0.8,
            factorial.CONTROL_CONFIGURATION: 1.2,
        }
        branches = {}
        steps = factorial.TOTAL_PERIODS * factorial.STEPS_PER_PERIOD
        systems = factorial._configurations()
        for configuration, gain in gains.items():
            branches[configuration] = {}
            for direction, frequency_values in (
                ("forward", factorial.FREQUENCIES),
                ("reverse", factorial.FREQUENCIES[::-1]),
            ):
                branches[configuration][direction] = [
                    {
                        "configuration": configuration,
                        "direction": direction,
                        "frequency": float(frequency),
                        "forcing_amplitude": factorial.FORCING_AMPLITUDE,
                        "forcing_node": factorial.FORCING_NODE,
                        "measurement_node": factorial.MEASUREMENT_NODE,
                        "finite_trajectory": True,
                        "failure": None,
                        "steps_completed": steps,
                        "rhs_evaluations": 4 * steps,
                        "endpoint_fundamental_transfer_gain": gain,
                        "endpoint_third_harmonic_transfer_gain": 0.05,
                        "response": cls._response(
                            gain * factorial.FORCING_AMPLITUDE
                        ),
                        "attachment_local_mismatch_responses": [
                            {
                                "attachment_index": index,
                                "attachment_node": attachment.node,
                                "fundamental": cls._harmonic(0.1),
                                "third_harmonic": cls._harmonic(0.01),
                                "fundamental_phase_relative_to_forcing_radians": 0.0,
                                "descriptive_first_harmonic_effective_stiffness": (
                                    attachment.linear_stiffness
                                    + 0.75 * attachment.cubic_stiffness * 0.1**2
                                ),
                            }
                            for index, attachment in enumerate(
                                systems[configuration].attachments
                            )
                        ],
                        "final_state": [0.0] * systems[configuration].state_size,
                    }
                    for frequency in frequency_values
                ]
        endpoints = factorial.compute_endpoints(branches)
        payload.update(
            {
                "status": "FACTORIAL_SURVIVES_FROZEN_SYNTHETIC_GATES",
                "branches": branches,
                "endpoints": endpoints,
                "gates": [
                    {
                        "name": "complete_finite_grid",
                        "operator": "==",
                        "threshold": True,
                        "observed": True,
                        "passed": True,
                    },
                    *factorial.evaluate_gates(endpoints),
                ],
                "failure_count": 0,
                "failures": [],
            }
        )
        payload["resource_usage"].update(
            {
                "deadline_guard_passed": True,
                "rss_guard_passed": True,
                "maximum_observed_current_rss_bytes": 1,
                "maximum_observed_peak_rss_bytes": 1,
                "initial_current_rss_bytes": 1,
                "initial_peak_rss_bytes": 1,
                "rss_measurement_method": "fixture",
                "resource_check_count": 8332,
                "last_checked_checkpoint": "finalize",
                "computation_wall_seconds": 1.0,
                "last_guard_elapsed_seconds": 0.9,
                "runtime_admission": {
                    "checkpoint": "runtime_admission",
                    "current_rss_bytes": 1,
                    "peak_rss_bytes": 1,
                    "rss_measurement_method": "fixture",
                    "passed": True,
                },
            }
        )
        return payload

    @classmethod
    def _finalize_resource_failure_payload(cls, reason, run_id=7):
        """Return a complete-grid payload interrupted by the final guard check."""
        payload = cls._survivor_payload(run_id)
        usage = payload["resource_usage"]
        failure = {
            "check": "resource_guard",
            "reason": reason,
            "checkpoint": "finalize",
        }
        payload.update(
            {
                "status": "FACTORIAL_KILLED_ON_FROZEN_SYNTHETIC_GATES",
                "failure_count": 1,
                "failures": [failure],
            }
        )
        if reason == "COOPERATIVE_DEADLINE_EXCEEDED":
            usage.update(
                {
                    "deadline_guard_passed": False,
                    "resource_check_count": 8331,
                    "last_guard_elapsed_seconds": 301.0,
                    "computation_wall_seconds": 302.0,
                }
            )
        elif reason == "SELF_PROCESS_RSS_CEILING_EXCEEDED":
            over_ceiling = factorial.MAX_RSS_BYTES + 1
            usage.update(
                {
                    "rss_guard_passed": False,
                    "maximum_observed_current_rss_bytes": over_ceiling,
                    "maximum_observed_peak_rss_bytes": over_ceiling,
                    "resource_check_count": 8332,
                }
            )
        elif reason == "RSS_MEASUREMENT_METHOD_CHANGED":
            failure.update(
                {
                    "previous_rss_measurement_method": "fixture",
                    "observed_rss_measurement_method": "changed-fixture",
                }
            )
            usage.update(
                {
                    "rss_guard_passed": False,
                    "resource_check_count": 8332,
                }
            )
        else:
            raise ValueError(reason)
        return payload

    @classmethod
    def _terminal_nonfinite_payload(cls, run_id=7):
        """Return the production-shaped history for a final-cell numerical failure."""
        payload = cls._survivor_payload(run_id)
        configuration = factorial.CONTROL_CONFIGURATION
        direction = "reverse"
        failed = payload["branches"][configuration][direction][-1]
        frequency = failed["frequency"]
        failed.clear()
        failed.update(
            {
                "frequency": frequency,
                "finite_trajectory": False,
                "failure": "NONFINITE_TRAJECTORY",
                "steps_completed": 1,
                "rhs_evaluations": 4,
                "configuration": configuration,
                "direction": direction,
            }
        )
        payload.update(
            {
                "status": "FACTORIAL_KILLED_ON_FROZEN_SYNTHETIC_GATES",
                "endpoints": None,
                "gates": [
                    {
                        "name": "complete_finite_grid",
                        "operator": "==",
                        "threshold": True,
                        "observed": False,
                        "passed": False,
                    }
                ],
                "failure_count": 1,
                "failures": [
                    {
                        "check": "finite_trajectory",
                        "configuration": configuration,
                        "direction": direction,
                        "frequency": frequency,
                        "reason": "NONFINITE_TRAJECTORY",
                    }
                ],
            }
        )
        payload["resource_usage"].update(
            {
                "resource_check_count": 8284,
                "last_checked_checkpoint": (
                    f"{configuration}:{direction}:{frequency:.17g}:step-0"
                ),
            }
        )
        return payload

    @classmethod
    def _frozen_gate_failure_payload(cls, run_id=7):
        """Return an all-finite trace with multiple frozen failures in order."""
        payload = cls._survivor_payload(run_id)
        for direction in ("forward", "reverse"):
            for cell in payload["branches"]["distributed_duffing"][direction]:
                cell["response"] = cls._response(
                    1.2 * factorial.FORCING_AMPLITUDE
                )
                cell["endpoint_fundamental_transfer_gain"] = 1.2
        endpoints = factorial.compute_endpoints(payload["branches"])
        gates = [payload["gates"][0], *factorial.evaluate_gates(endpoints)]
        failures = [
            {
                "check": "frozen_gate",
                "reason": "FROZEN_THRESHOLD_FAILED",
                "gate": gate["name"],
            }
            for gate in gates
            if not gate["passed"]
        ]
        payload.update(
            {
                "status": "FACTORIAL_KILLED_ON_FROZEN_SYNTHETIC_GATES",
                "endpoints": endpoints,
                "gates": gates,
                "failure_count": len(failures),
                "failures": failures,
            }
        )
        return payload

    @staticmethod
    def _reseal(output, artifact):
        artifact_path = output / runner.FACTORIAL_FILENAME
        artifact_bytes = runner._canonical_json_bytes(artifact)
        artifact_path.write_bytes(artifact_bytes)
        manifest_path = output / runner.MANIFEST_FILENAME
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for field in (
            "protocol_id",
            "run_id",
            "status",
            "scientific_claim_status",
            "novelty_claim_status",
            "failure_count",
        ):
            manifest[field] = artifact[field]
        manifest["resource_guards"] = runner._resource_guards(artifact)
        manifest["factorial_artifact"].update(
            {
                "bytes": len(artifact_bytes),
                "sha256": runner._sha256(artifact_bytes),
            }
        )
        manifest_path.write_bytes(runner._canonical_json_bytes(manifest))

    def test_runner_writes_and_verifies_atomic_canonical_artifacts(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory) / "run-7"
            payload = self._survivor_payload()
            with patch.object(runner, "run_factorial", return_value=payload):
                written = runner.write_run(output, 7)
            manifest, artifact = runner.verify_manifest(output)
            self.assertEqual(written, manifest)
            self.assertEqual(artifact, payload)
            self.assertTrue((output / runner.FACTORIAL_FILENAME).read_bytes().endswith(b"\n"))
            self.assertFalse(list(output.glob("*.tmp")))

    def test_transaction_failure_leaves_no_partial_result_and_allows_rerun(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory) / "run-7"
            payload = self._payload()
            original_write = runner._atomic_write
            calls = 0

            def fail_second_write(path, data):
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise OSError("forced manifest write failure")
                original_write(path, data)

            with patch.object(runner, "run_factorial", return_value=payload):
                with patch.object(runner, "_atomic_write", side_effect=fail_second_write):
                    with self.assertRaises(OSError):
                        runner.write_run(output, 7)
                self.assertFalse(output.exists())
                self.assertFalse(list(Path(temporary_directory).glob(".run-7.stage-*")))
                runner.write_run(output, 7)
            runner.verify_manifest(output)

    def test_publication_boundaries_leave_no_partial_final(self):
        payload = self._payload()
        for boundary in ("first_write", "stage_verify", "stage_fsync", "rename"):
            with self.subTest(boundary=boundary):
                with tempfile.TemporaryDirectory() as temporary_directory:
                    output = Path(temporary_directory) / "run-7"
                    original_write = runner._atomic_write
                    original_verify = runner.verify_manifest
                    original_fsync = runner._fsync_directory
                    original_replace = runner.os.replace

                    def write_side_effect(path, data):
                        if boundary == "first_write":
                            raise OSError("forced first write failure")
                        return original_write(path, data)

                    def verify_side_effect(path):
                        if boundary == "stage_verify":
                            raise runner.FactorialArtifactError("forced stage verify")
                        return original_verify(path)

                    def fsync_side_effect(path):
                        if boundary == "stage_fsync":
                            raise OSError("forced stage fsync failure")
                        return original_fsync(path)

                    def replace_side_effect(source, destination):
                        if boundary == "rename" and Path(source).is_dir():
                            raise OSError("forced directory rename failure")
                        return original_replace(source, destination)

                    with patch.object(runner, "run_factorial", return_value=payload):
                        with patch.object(runner, "_atomic_write", side_effect=write_side_effect):
                            with patch.object(runner, "verify_manifest", side_effect=verify_side_effect):
                                with patch.object(runner, "_fsync_directory", side_effect=fsync_side_effect):
                                    with patch.object(runner.os, "replace", side_effect=replace_side_effect):
                                        with self.assertRaises(
                                            (OSError, runner.FactorialArtifactError)
                                        ):
                                            runner.write_run(output, 7)
                    self.assertFalse(output.exists())
                    self.assertFalse(
                        list(Path(temporary_directory).glob(".run-7.stage-*"))
                    )

    def test_parent_fsync_failure_commits_and_retry_is_idempotent(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory) / "run-7"
            payload = self._payload()
            calls = 0

            def fail_parent_fsync(path):
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise OSError("forced parent fsync failure after commit")

            with patch.object(runner, "run_factorial", return_value=payload) as run:
                with patch.object(
                    runner, "_fsync_directory", side_effect=fail_parent_fsync
                ):
                    committed = runner.write_run(output, 7)
                runner.verify_manifest(output)
                run.reset_mock()
                recovered = runner.write_run(output, 7)
                run.assert_not_called()
            self.assertEqual(committed, recovered)

    def test_dangling_symlink_is_rejected_before_payload_execution(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory) / "dangling"
            with patch.object(type(output), "is_symlink", return_value=True):
                with patch.object(type(output), "exists", return_value=False):
                    with patch.object(runner, "run_factorial") as run:
                        with self.assertRaises(runner.FactorialArtifactError):
                            runner.write_run(output, 7)
                        run.assert_not_called()

    def test_runner_rejects_nonempty_output_before_execution(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory) / "occupied"
            output.mkdir()
            (output / "keep.txt").write_text("preserve", encoding="utf-8")
            with patch.object(runner, "run_factorial") as run:
                with self.assertRaises(runner.FactorialArtifactError):
                    runner.write_run(output, 7)
                run.assert_not_called()
            self.assertEqual((output / "keep.txt").read_text(encoding="utf-8"), "preserve")

    def test_verifier_rejects_tampering_and_path_escape(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory) / "run-7"
            with patch.object(runner, "run_factorial", return_value=self._payload()):
                runner.write_run(output, 7)
            artifact_path = output / runner.FACTORIAL_FILENAME
            artifact_path.write_bytes(artifact_path.read_bytes() + b" ")
            with self.assertRaises(runner.FactorialArtifactError):
                runner.verify_manifest(output)

    def test_verifier_rejects_canonical_digest_valid_semantic_contradictions(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory) / "run-7"
            payload = self._payload()
            with patch.object(runner, "run_factorial", return_value=payload):
                runner.write_run(output, 7)
            artifact = json.loads(
                (output / runner.FACTORIAL_FILENAME).read_text(encoding="utf-8")
            )
            artifact["status"] = "FACTORIAL_SURVIVES_FROZEN_SYNTHETIC_GATES"
            self._reseal(output, artifact)
            with self.assertRaises(runner.FactorialArtifactError):
                runner.verify_manifest(output)

    def test_verifier_rejects_resealed_negative_and_zero_check_admission_attacks(self):
        attacks = ("negative_gain", "negative_settling", "zero_check_admission")
        for attack in attacks:
            with self.subTest(attack=attack):
                with tempfile.TemporaryDirectory() as temporary_directory:
                    output = Path(temporary_directory) / "run-7"
                    payload = (
                        self._payload()
                        if attack == "zero_check_admission"
                        else self._survivor_payload()
                    )
                    with patch.object(runner, "run_factorial", return_value=payload):
                        runner.write_run(output, 7)
                    artifact = json.loads(
                        (output / runner.FACTORIAL_FILENAME).read_text(
                            encoding="utf-8"
                        )
                    )
                    if attack == "negative_gain":
                        artifact["branches"]["single_linear"]["forward"][0][
                            "endpoint_fundamental_transfer_gain"
                        ] = -1.0
                    elif attack == "negative_settling":
                        artifact["branches"]["single_linear"]["forward"][0][
                            "response"
                        ]["relative_settling_delta"] = -0.1
                    else:
                        usage = artifact["resource_usage"]
                        usage["resource_check_count"] = 0
                        usage["initial_current_rss_bytes"] = None
                        usage["initial_peak_rss_bytes"] = None
                        usage["rss_measurement_method"] = None
                        usage["runtime_admission"] = {
                            "checkpoint": "runtime_admission",
                            "current_rss_bytes": None,
                            "peak_rss_bytes": None,
                            "rss_measurement_method": None,
                            "passed": True,
                        }
                    self._reseal(output, artifact)
                    with self.assertRaises(runner.FactorialArtifactError):
                        runner.verify_manifest(output)

        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory) / "run-7"
            payload = self._survivor_payload()
            with patch.object(runner, "run_factorial", return_value=payload):
                runner.write_run(output, 7)
            artifact = json.loads(
                (output / runner.FACTORIAL_FILENAME).read_text(encoding="utf-8")
            )
            artifact["endpoints"]["aggregate_interaction_ratio"] = 0.9
            aggregate_gate = next(
                gate
                for gate in artifact["gates"]
                if gate["name"] == "aggregate_interaction_ratio"
            )
            aggregate_gate["observed"] = 0.9
            aggregate_gate["passed"] = True
            self._reseal(output, artifact)
            with self.assertRaises(runner.FactorialArtifactError):
                runner.verify_manifest(output)

    def test_verifier_rejects_resealed_guard_trace_attacks(self):
        attacks = {
            "survivor_wrong_count": (
                self._survivor_payload,
                lambda artifact: artifact["resource_usage"].update(
                    {"resource_check_count": 8331}
                ),
            ),
            "survivor_wrong_terminal": (
                self._survivor_payload,
                lambda artifact: artifact["resource_usage"].update(
                    {
                        "last_checked_checkpoint": (
                            "no_sidecar_control:reverse:0.80000000000000004:complete"
                        )
                    }
                ),
            ),
            "wall_precedes_guard": (
                self._survivor_payload,
                lambda artifact: artifact["resource_usage"].update(
                    {"computation_wall_seconds": 0.8}
                ),
            ),
            "deadline_not_exceeded": (
                lambda: self._finalize_resource_failure_payload(
                    "COOPERATIVE_DEADLINE_EXCEEDED"
                ),
                lambda artifact: artifact["resource_usage"].update(
                    {
                        "last_guard_elapsed_seconds": 300.0,
                        "computation_wall_seconds": 301.0,
                    }
                ),
            ),
            "deadline_failed_check_counted": (
                lambda: self._finalize_resource_failure_payload(
                    "COOPERATIVE_DEADLINE_EXCEEDED"
                ),
                lambda artifact: artifact["resource_usage"].update(
                    {"resource_check_count": 8332}
                ),
            ),
            "rss_not_over_ceiling": (
                lambda: self._finalize_resource_failure_payload(
                    "SELF_PROCESS_RSS_CEILING_EXCEEDED"
                ),
                lambda artifact: artifact["resource_usage"].update(
                    {
                        "maximum_observed_current_rss_bytes": 1,
                        "maximum_observed_peak_rss_bytes": 1,
                    }
                ),
            ),
            "rss_failed_sample_not_counted": (
                lambda: self._finalize_resource_failure_payload(
                    "SELF_PROCESS_RSS_CEILING_EXCEEDED"
                ),
                lambda artifact: artifact["resource_usage"].update(
                    {"resource_check_count": 8331}
                ),
            ),
            "both_guards_failed": (
                lambda: self._finalize_resource_failure_payload(
                    "SELF_PROCESS_RSS_CEILING_EXCEEDED"
                ),
                lambda artifact: artifact["resource_usage"].update(
                    {"deadline_guard_passed": False}
                ),
            ),
            "method_change_at_admission": (
                lambda: self._finalize_resource_failure_payload(
                    "RSS_MEASUREMENT_METHOD_CHANGED"
                ),
                lambda artifact: (
                    artifact["failures"][0].update(
                        {"checkpoint": "runtime_admission"}
                    ),
                    artifact["resource_usage"].update(
                        {"last_checked_checkpoint": "runtime_admission"}
                    ),
                ),
            ),
            "method_change_previous_not_retained": (
                lambda: self._finalize_resource_failure_payload(
                    "RSS_MEASUREMENT_METHOD_CHANGED"
                ),
                lambda artifact: artifact["failures"][0].update(
                    {"previous_rss_measurement_method": "other-fixture"}
                ),
            ),
        }
        for name, (payload_factory, mutate) in attacks.items():
            with self.subTest(attack=name):
                with tempfile.TemporaryDirectory() as temporary_directory:
                    output = Path(temporary_directory) / "run-7"
                    payload = payload_factory()
                    with patch.object(runner, "run_factorial", return_value=payload):
                        runner.write_run(output, 7)
                    artifact = json.loads(
                        (output / runner.FACTORIAL_FILENAME).read_text(
                            encoding="utf-8"
                        )
                    )
                    mutate(artifact)
                    self._reseal(output, artifact)
                    with self.assertRaises(runner.FactorialArtifactError):
                        runner.verify_manifest(output)

    def test_verifier_rejects_resealed_continuation_and_failure_history_attacks(self):
        def make_first_cell_nonfinite(artifact):
            cell = artifact["branches"]["single_linear"]["forward"][0]
            frequency = cell["frequency"]
            cell.clear()
            cell.update(
                {
                    "frequency": frequency,
                    "finite_trajectory": False,
                    "failure": "NONFINITE_TRAJECTORY",
                    "steps_completed": 1,
                    "rhs_evaluations": 4,
                    "configuration": "single_linear",
                    "direction": "forward",
                }
            )

        def duplicate_trajectory_failure(artifact):
            artifact["failures"].append(dict(artifact["failures"][0]))
            artifact["failure_count"] = len(artifact["failures"])

        def append_gate_after_finalize_failure(artifact):
            artifact["failures"].append(
                {
                    "check": "frozen_gate",
                    "reason": "FROZEN_THRESHOLD_FAILED",
                    "gate": "aggregate_interaction_ratio",
                }
            )
            artifact["failure_count"] = len(artifact["failures"])

        attacks = {
            "finite_after_nonfinite_same_sweep": (
                self._survivor_payload,
                make_first_cell_nonfinite,
            ),
            "duplicate_finite_trajectory_failure": (
                self._terminal_nonfinite_payload,
                duplicate_trajectory_failure,
            ),
            "frozen_gate_after_finalize_resource_failure": (
                lambda: self._finalize_resource_failure_payload(
                    "SELF_PROCESS_RSS_CEILING_EXCEEDED"
                ),
                append_gate_after_finalize_failure,
            ),
            "frozen_gate_failures_reordered": (
                self._frozen_gate_failure_payload,
                lambda artifact: artifact["failures"].reverse(),
            ),
        }
        for name, (payload_factory, mutate) in attacks.items():
            with self.subTest(attack=name):
                with tempfile.TemporaryDirectory() as temporary_directory:
                    output = Path(temporary_directory) / "run-7"
                    payload = payload_factory()
                    with patch.object(runner, "run_factorial", return_value=payload):
                        runner.write_run(output, 7)
                    artifact = json.loads(
                        (output / runner.FACTORIAL_FILENAME).read_text(
                            encoding="utf-8"
                        )
                    )
                    mutate(artifact)
                    self._reseal(output, artifact)
                    with self.assertRaises(runner.FactorialArtifactError):
                        runner.verify_manifest(output)

        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory) / "run-7"
            payload = self._payload()
            with patch.object(runner, "run_factorial", return_value=payload):
                runner.write_run(output, 7)
            artifact = json.loads(
                (output / runner.FACTORIAL_FILENAME).read_text(encoding="utf-8")
            )
            artifact["failure_count"] = -1
            self._reseal(output, artifact)
            with self.assertRaises(runner.FactorialArtifactError):
                runner.verify_manifest(output)

        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory) / "run-7"
            with patch.object(runner, "run_factorial", return_value=self._payload()):
                runner.write_run(output, 7)
            manifest_path = output / runner.MANIFEST_FILENAME
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["factorial_artifact"]["path"] = "../factorial.json"
            manifest_path.write_bytes(runner._canonical_json_bytes(manifest))
            with self.assertRaises(runner.FactorialArtifactError):
                runner.verify_manifest(output)


if __name__ == "__main__":
    unittest.main()
