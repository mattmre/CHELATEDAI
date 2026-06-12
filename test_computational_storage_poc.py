import os
import sys
import tempfile
import unittest

import numpy as np

POC_DIR = os.path.join(os.path.dirname(__file__), "computational_storage_poc")
if POC_DIR not in sys.path:
    sys.path.insert(0, POC_DIR)

from block_graph import BLOCK_SIZE, build_graph_payload, run_block_graph  # noqa: E402
from block_graph import clear_last_research_shim_meta, get_last_research_shim_meta  # noqa: E402
from mock_array import ArraySimulation  # noqa: E402
from mock_nvme import MockNVMeDrive, traditional_host_inference  # noqa: E402
from validation_config import DEFAULT_RANDOM_SEED, MIN_REFERENCE_ACCURACY  # noqa: E402

try:
    from train_and_compile import (  # noqa: E402
        DIGITS_DEPENDENCIES_AVAILABLE,
        compile_model,
        evaluate_torch_model,
        load_digits_split,
        train_digit_classifier,
    )
    from test_real_model import evaluate_storage_model, validate_storage_metrics  # noqa: E402
except ModuleNotFoundError:
    DIGITS_DEPENDENCIES_AVAILABLE = False

    # Optional dependency path (torch/sklearn) not installed in lightweight test
    # environments. The digit-roundtrip test is decorated with skipUnless and
    # should remain skipped when unavailable.
    compile_model = None
    evaluate_torch_model = None
    load_digits_split = None
    train_digit_classifier = None
    evaluate_storage_model = None
    validate_storage_metrics = None


class TestComputationalStorageBlockGraph(unittest.TestCase):
    def test_offset_zero_processes_first_block_and_matches_expected_values(self):
        w1 = np.array(
            [
                [1.0, -0.5],
                [-1.0, 2.0],
                [0.5, 0.5],
            ],
            dtype=np.float32,
        )
        w2 = np.array(
            [
                [2.0, 1.0],
                [1.0, -1.0],
            ],
            dtype=np.float32,
        )
        payload = build_graph_payload([w1, w2])

        input_act = np.zeros((1, BLOCK_SIZE), dtype=np.float16)
        input_act[0, :3] = [1.0, -2.0, 3.0]

        expected_hidden = np.maximum(np.array([[1.0, -2.0, 3.0]], dtype=np.float32) @ w1, 0)
        expected_output = expected_hidden @ w2

        graph_output, blocks_processed = run_block_graph(payload, input_act, trigger_offset=0, hidden_activation="relu")

        self.assertEqual(blocks_processed, 2)
        np.testing.assert_allclose(graph_output[:, :2], expected_output, rtol=1e-4, atol=1e-4)
        self.assertFalse(np.allclose(graph_output[0, :3], input_act[0, :3]))

    def test_storage_and_host_paths_share_the_same_semantics(self):
        payload = build_graph_payload(
            [
                np.eye(4, dtype=np.float32),
                np.full((4, 2), 0.5, dtype=np.float32),
            ]
        )

        input_act = np.zeros((1, BLOCK_SIZE), dtype=np.float16)
        input_act[0, :4] = [1.0, 2.0, 3.0, 4.0]

        with tempfile.TemporaryDirectory() as temp_dir:
            binary_path = os.path.join(temp_dir, "graph.bin")
            with open(binary_path, "wb") as f:
                f.write(payload)

            with MockNVMeDrive(binary_path) as drive:
                storage_output, storage_latency = drive.computational_inference(0, input_act)
                host_output, host_latency = traditional_host_inference(drive, 0, input_act)

        np.testing.assert_allclose(storage_output, host_output, rtol=1e-6, atol=1e-6)
        self.assertLess(storage_latency, host_latency)


class TestComputationalStorageBlockGraphResearch(unittest.TestCase):
    def setUp(self) -> None:
        clear_last_research_shim_meta()
        os.environ.pop("CHELATED_SHIM_RESEARCH", None)
        os.environ.pop("CHELATED_SHIM_PROMOTED", None)

    def tearDown(self) -> None:
        os.environ.pop("CHELATED_SHIM_RESEARCH", None)
        os.environ.pop("CHELATED_SHIM_PROMOTED", None)

    @staticmethod
    def _sample_payload() -> bytes:
        return build_graph_payload(
            [
                np.eye(2, dtype=np.float32),
                np.ones((2, 2), dtype=np.float32),
            ]
        )

    def test_research_meta_emits_with_research_flag(self) -> None:
        import os

        os.environ["CHELATED_SHIM_RESEARCH"] = "1"
        payload = self._sample_payload()
        input_act = np.zeros((1, BLOCK_SIZE), dtype=np.float16)
        input_act[0, :2] = [1.0, 3.0]

        run_block_graph(payload, input_act, trigger_offset=0, hidden_activation="identity")

        meta = get_last_research_shim_meta()
        self.assertIsNotNone(meta)
        self.assertTrue(meta.get("research_shim_guard"))
        self.assertEqual(meta.get("sip_seam"), "computational_storage_poc.block_graph.run_block_graph")
        self.assertEqual(meta.get("research_stall_count"), 0)
        self.assertGreaterEqual(meta.get("blocks_processed", 0), 1)

    def test_research_meta_stall_count_increments_on_failure(self) -> None:
        import os

        os.environ["CHELATED_SHIM_RESEARCH"] = "1"
        input_act = np.zeros((1, BLOCK_SIZE), dtype=np.float16)
        input_act[0, :2] = [1.0, 3.0]

        with self.assertRaises(ValueError):
            run_block_graph(b"", input_act, trigger_offset=0)

        first_meta = get_last_research_shim_meta()
        self.assertIsNotNone(first_meta)
        self.assertEqual(first_meta.get("research_stall_count"), 1)
        self.assertIn("error", first_meta)

        with self.assertRaises(ValueError):
            run_block_graph(b"", input_act, trigger_offset=0)

        second_meta = get_last_research_shim_meta()
        self.assertIsNotNone(second_meta)
        self.assertEqual(second_meta.get("research_stall_count"), 2)

    def test_research_meta_absent_when_flag_off(self) -> None:
        payload = self._sample_payload()
        input_act = np.zeros((1, BLOCK_SIZE), dtype=np.float16)
        input_act[0, :2] = [2.0, -1.0]
        run_block_graph(payload, input_act)
        self.assertIsNone(get_last_research_shim_meta())

    def test_research_meta_includes_promoted_sip_apply_when_promoted_enabled(self) -> None:
        os.environ["CHELATED_SHIM_RESEARCH"] = "1"
        os.environ["CHELATED_SHIM_PROMOTED"] = "1"
        payload = self._sample_payload()
        input_act = np.zeros((1, BLOCK_SIZE), dtype=np.float16)
        input_act[0, :2] = [1.0, 3.0]

        run_block_graph(payload, input_act, trigger_offset=0, hidden_activation="identity")

        meta = get_last_research_shim_meta()
        self.assertIsNotNone(meta)
        self.assertIn("promoted_sip_apply", meta)
        self.assertIsInstance(meta["promoted_sip_apply"], dict)
        self.assertEqual(meta.get("sip_seam"), "computational_storage_poc.block_graph.run_block_graph")


class TestComputationalStorageLatencyModel(unittest.TestCase):
    def test_speculative_racing_beats_sequential_for_unique_drive_dispatch(self):
        simulation = ArraySimulation(num_drives=4)
        metrics = simulation.compare_execution_modes([10, 22, 5, 99], [10, 22, 5, 99])

        self.assertLess(metrics["speculative_time_ms"], metrics["sequential_time_ms"])
        self.assertGreater(metrics["latency_hidden_pct"], 0.0)

    def test_drive_contention_increases_speculative_latency(self):
        balanced = ArraySimulation(num_drives=4)
        contended = ArraySimulation(num_drives=2)

        balanced_time = balanced.speculative_multipath_racing([10, 22, 5, 99])
        contended_time = contended.speculative_multipath_racing([10, 22, 5, 99])

        self.assertGreater(contended_time, balanced_time)


@unittest.skipUnless(DIGITS_DEPENDENCIES_AVAILABLE, "scikit-learn not installed")
class TestComputationalStorageDigitsRoundTrip(unittest.TestCase):
    def test_storage_round_trip_matches_reference_model_accuracy(self):
        X_train, X_test, y_train, y_test = load_digits_split()
        model = train_digit_classifier(X_train, y_train, epochs=20, seed=DEFAULT_RANDOM_SEED)
        torch_accuracy = evaluate_torch_model(model, X_test, y_test)

        with tempfile.TemporaryDirectory() as temp_dir:
            binary_path = os.path.join(temp_dir, "real_model.bin")
            compile_model(model, binary_path)
            storage_metrics = evaluate_storage_model(binary_path, X_test, y_test)

        validate_storage_metrics(storage_metrics, expected_torch_accuracy=torch_accuracy)
        self.assertGreaterEqual(torch_accuracy, MIN_REFERENCE_ACCURACY)
        self.assertAlmostEqual(storage_metrics["accuracy"], torch_accuracy, delta=0.02)


if __name__ == "__main__":
    unittest.main()
