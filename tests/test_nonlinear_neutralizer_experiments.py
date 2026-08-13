import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

import nonlinear_neutralizer_experiments as nln
import run_rb15_nonlinear_neutralizer as runner


class NonlinearNeutralizerCoreTests(unittest.TestCase):
    def test_attachment_and_graph_contracts_reject_invalid_inputs(self):
        with self.assertRaises(nln.StageAValidationError):
            nln.Attachment(True, 0.2, 0.2, 0.03, 0.5)
        with self.assertRaises(nln.StageAValidationError):
            nln.Attachment(0, 0.0, 0.2, 0.03, 0.5)
        with self.assertRaises(nln.StageAValidationError):
            nln.GraphSystem(np.asarray([[1.0, 1.0], [0.0, 1.0]]))

    def test_invalid_run_ids_fail_closed(self):
        for run_id in (0, 8, True, 7.0):
            with self.subTest(run_id=run_id):
                with self.assertRaises(nln.StageAValidationError):
                    nln.run_linear_limit_check(run_id)

    def test_linear_limit_matches_independent_oracle(self):
        result = nln.run_linear_limit_check(7)
        self.assertTrue(result["passed"])
        self.assertLessEqual(result["final_state_l2_error"], nln.LINEAR_TOLERANCE)
        self.assertLess(result["oracle_imaginary_residual"], 1.0e-12)

    def test_unforced_storage_and_protected_block_pass_frozen_tolerances(self):
        energy = nln.run_energy_check(7)
        protected = nln.run_protected_channel_check(7)
        self.assertTrue(energy["passed"])
        self.assertLessEqual(
            energy["maximum_positive_energy_step"], nln.ENERGY_STEP_TOLERANCE
        )
        self.assertTrue(protected["passed"])
        self.assertLessEqual(
            protected["final_relative_leakage"], nln.PROTECTED_LEAKAGE_TOLERANCE
        )
        self.assertEqual(protected["off_block_operator_norm"], 0.0)

    def test_harmonic_fit_recovers_known_components(self):
        frequency = 1.3
        times = np.linspace(0.0, 40.0 * math.pi / frequency, 4000, endpoint=False)
        values = 0.4 + 2.0 * np.sin(frequency * times) - 3.0 * np.cos(
            frequency * times
        )
        result = nln.fit_harmonic(times, values, frequency, 1)
        self.assertAlmostEqual(result["offset"], 0.4, places=12)
        self.assertAlmostEqual(result["sin_coefficient"], 2.0, places=12)
        self.assertAlmostEqual(result["cos_coefficient"], -3.0, places=12)
        self.assertAlmostEqual(result["amplitude"], math.sqrt(13.0), places=12)
        with self.assertRaises(nln.StageAValidationError):
            nln.fit_harmonic(times, values, frequency, 2)

    @staticmethod
    def _peak_cell(frequency, amplitude, finite=True):
        return {
            "frequency": frequency,
            "finite_trajectory": finite,
            "response": {"fundamental": {"amplitude": amplitude}},
        }

    def test_peak_censor_uses_declared_grid_not_finite_subset(self):
        cells = [
            self._peak_cell(0.70, 0.0, finite=False),
            self._peak_cell(0.75, 2.0),
            self._peak_cell(0.80, 1.0),
        ]
        interior = nln._peak_summary(cells, (0.70, 0.75, 0.80))
        self.assertTrue(interior["resolved"])
        self.assertFalse(interior["boundary_censored"])
        boundary = nln._peak_summary(cells[1:], (0.75, 0.80))
        self.assertFalse(boundary["resolved"])
        self.assertEqual(boundary["failure"], "BOUNDARY_CENSORED_ARGMAX")

    def test_frozen_protocol_cell_counts_are_explicit(self):
        self.assertEqual(len(nln.SCALAR_FREQUENCIES), 37)
        self.assertEqual(len(nln.SCALAR_AMPLITUDES), 2)
        self.assertEqual(37 * 2 * 2, 148)
        self.assertEqual(len(nln.GRAPH_FREQUENCIES), 17)
        self.assertEqual(len(nln._graph_configurations()), 4)
        self.assertEqual(17 * 4 * 2, 136)
        self.assertEqual(nln.TOTAL_PERIODS, 60)
        self.assertEqual(nln.STEPS_PER_PERIOD, 200)
        self.assertEqual(nln.MEASURE_PERIODS, 20)

    def test_process_rss_probe_returns_finite_self_measurement(self):
        current, peak, method = nln._self_rss_bytes()
        self.assertGreater(current, 0)
        self.assertGreaterEqual(peak, current)
        self.assertIsInstance(method, str)
        self.assertTrue(method)

    def test_reduced_sweeps_retain_cells_and_local_self_grading_fields(self):
        scalar_grid = np.asarray((0.8, 1.0), dtype=np.float64)
        graph_grid = np.asarray((0.8, 1.0), dtype=np.float64)
        with patch.multiple(
            nln,
            TOTAL_PERIODS=2,
            MEASURE_PERIODS=1,
            STEPS_PER_PERIOD=20,
            SCALAR_FREQUENCIES=scalar_grid,
            GRAPH_FREQUENCIES=graph_grid,
        ):
            scalar = nln.run_scalar_sweeps()
            graph = nln.run_graph_sweeps()
        scalar_cells = [
            cell
            for amplitude in scalar["branches"].values()
            for branch in amplitude.values()
            for cell in branch
        ]
        self.assertEqual(len(scalar_cells), 8)
        self.assertTrue(
            all(cell["convergence_assessment"] == "UNASSESSED_NO_FROZEN_GATE" for cell in scalar_cells)
        )
        self.assertEqual(
            set(graph["branches"]),
            {
                "no_sidecar",
                "single_full_physical_coefficients",
                "two_distributed_half_parameters",
                "two_colocated_half_parameters",
            },
        )
        graph_cells = [
            cell
            for configuration in graph["branches"].values()
            for branch in configuration.values()
            for cell in branch
        ]
        self.assertEqual(len(graph_cells), 16)
        distributed = graph["branches"]["two_distributed_half_parameters"]["forward"][0]
        self.assertEqual(len(distributed["attachment_local_mismatch_responses"]), 2)
        local = distributed["attachment_local_mismatch_responses"][0]
        self.assertIn("fundamental_phase_relative_to_forcing_radians", local)
        self.assertIn("third_harmonic", local)
        self.assertIn("descriptive_first_harmonic_effective_stiffness", local)
        equivalence = graph["colocated_two_half_vs_single_full_equivalence"]
        self.assertTrue(equivalence["passed"])
        self.assertLessEqual(
            equivalence["maximum_fundamental_amplitude_delta"],
            equivalence["tolerance"],
        )

    def test_reduced_complete_payload_is_strict_json_serializable(self):
        grid = np.asarray((0.8,), dtype=np.float64)
        with patch.multiple(
            nln,
            TOTAL_PERIODS=2,
            MEASURE_PERIODS=1,
            STEPS_PER_PERIOD=20,
            SCALAR_FREQUENCIES=grid,
            GRAPH_FREQUENCIES=grid,
        ):
            payload = nln.run_stage_a(7)
        encoded = json.dumps(payload, allow_nan=False, sort_keys=True)
        self.assertIsInstance(encoded, str)


class NonlinearNeutralizerArtifactTests(unittest.TestCase):
    @staticmethod
    def _payload(run_id=7):
        return {
            "protocol_id": nln.PROTOCOL_ID,
            "run_id": run_id,
            "status": "FROZEN_FAILURES_RETAINED",
            "scientific_claim_status": nln.SCIENTIFIC_CLAIM_STATUS,
            "novelty_claim_status": nln.NOVELTY_CLAIM_STATUS,
            "failure_count": 1,
            "resource_usage": {
                "rss_gate_passed": True,
                "wall_gate_passed": True,
            },
            "failures": [{"reason": "BOUNDARY_CENSORED_PEAK"}],
        }

    def test_runner_writes_and_verifies_canonical_atomic_artifacts(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory) / "run-7"
            payload = self._payload()
            with patch.object(runner, "run_stage_a", return_value=payload):
                written = runner.write_run(output, 7)
            manifest, stage = runner.verify_manifest(output)
            self.assertEqual(written, manifest)
            self.assertEqual(stage, payload)
            stage_bytes = (output / runner.STAGE_FILENAME).read_bytes()
            self.assertTrue(stage_bytes.endswith(b"\n"))
            self.assertEqual(manifest["stage_artifact"]["bytes"], len(stage_bytes))
            self.assertFalse(list(output.glob("*.tmp")))

    def test_verifier_rejects_stage_tampering(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory) / "run-7"
            with patch.object(runner, "run_stage_a", return_value=self._payload()):
                runner.write_run(output, 7)
            stage_path = output / runner.STAGE_FILENAME
            stage_path.write_bytes(stage_path.read_bytes() + b" ")
            with self.assertRaises(runner.ArtifactIntegrityError):
                runner.verify_manifest(output)

    def test_verifier_rejects_path_escape_and_status_mismatch(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory) / "run-7"
            with patch.object(runner, "run_stage_a", return_value=self._payload()):
                runner.write_run(output, 7)
            manifest_path = output / runner.MANIFEST_FILENAME
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["stage_artifact"]["path"] = "../stage_a.json"
            manifest_path.write_bytes(runner._canonical_json_bytes(manifest))
            with self.assertRaises(runner.ArtifactIntegrityError):
                runner.verify_manifest(output)

            manifest["stage_artifact"]["path"] = runner.STAGE_FILENAME
            manifest["status"] = "EXECUTION_CONSISTENT_ON_FROZEN_FIXTURES"
            manifest_path.write_bytes(runner._canonical_json_bytes(manifest))
            with self.assertRaises(runner.ArtifactIntegrityError):
                runner.verify_manifest(output)

    def test_runner_rejects_invalid_run_id_without_creating_output(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory) / "invalid"
            with self.assertRaises(runner.ArtifactIntegrityError):
                runner.write_run(output, 8)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
