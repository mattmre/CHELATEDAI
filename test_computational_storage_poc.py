import os
import sys
import tempfile
import unittest

import numpy as np

POC_DIR = os.path.join(os.path.dirname(__file__), "computational_storage_poc")
if POC_DIR not in sys.path:
    sys.path.insert(0, POC_DIR)

from block_graph import BLOCK_SIZE, build_graph_payload, run_block_graph  # noqa: E402
from mock_array import ArraySimulation  # noqa: E402
from mock_nvme import MockNVMeDrive, traditional_host_inference  # noqa: E402
from test_real_model import evaluate_storage_model, validate_storage_metrics  # noqa: E402
from train_and_compile import (  # noqa: E402
    DIGITS_DEPENDENCIES_AVAILABLE,
    compile_model,
    evaluate_torch_model,
    load_digits_split,
    train_digit_classifier,
)
from validation_config import DEFAULT_RANDOM_SEED, MIN_REFERENCE_ACCURACY  # noqa: E402


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

            drive = MockNVMeDrive(binary_path)
            storage_output, storage_latency = drive.computational_inference(0, input_act)
            host_output, host_latency = traditional_host_inference(drive, 0, input_act)

        np.testing.assert_allclose(storage_output, host_output, rtol=1e-6, atol=1e-6)
        self.assertLess(storage_latency, host_latency)


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
