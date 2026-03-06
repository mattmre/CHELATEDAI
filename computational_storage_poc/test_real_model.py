import numpy as np

from mock_nvme import MockNVMeDrive
from train_and_compile import (
    DIGITS_DEPENDENCIES_AVAILABLE,
    load_digits_split,
    train_and_compile,
)
from validation_config import (
    DEFAULT_RANDOM_SEED,
    DEFAULT_TRAIN_EPOCHS,
    MAX_ACCURACY_GAP,
    MIN_STORAGE_ACCURACY,
)


def _prepare_storage_input(sample: np.ndarray) -> np.ndarray:
    input_act = np.zeros((1, 512), dtype=np.float16)
    input_act[0, : sample.shape[0]] = sample
    return input_act


def evaluate_storage_model(binary_path: str, X_test, y_test):
    drive = MockNVMeDrive(binary_path)
    correct = 0
    total_comp_latency = 0.0

    for sample, expected_label in zip(X_test, y_test):
        input_act = _prepare_storage_input(sample)
        out_comp, latency = drive.computational_inference(0x0, input_act)
        total_comp_latency += latency

        pred = int(np.argmax(out_comp[0, :10]))
        if pred == int(expected_label):
            correct += 1

    total = len(y_test)
    return {
        "samples": total,
        "accuracy": correct / total,
        "total_latency_ms": total_comp_latency * 1000,
    }


def validate_storage_metrics(metrics, expected_torch_accuracy=None):
    if metrics["accuracy"] < MIN_STORAGE_ACCURACY:
        raise AssertionError(
            f"Computational-storage accuracy regressed to {metrics['accuracy']:.3f}; "
            f"minimum expected is {MIN_STORAGE_ACCURACY:.3f}"
        )

    if expected_torch_accuracy is not None:
        accuracy_gap = abs(expected_torch_accuracy - metrics["accuracy"])
        if accuracy_gap > MAX_ACCURACY_GAP:
            raise AssertionError(
                f"Computational-storage accuracy gap {accuracy_gap:.3f} exceeds "
                f"maximum allowed {MAX_ACCURACY_GAP:.3f}"
            )


def test_real_model(
    binary_path: str = "real_model.bin",
    expected_torch_accuracy: float | None = None,
):
    if not DIGITS_DEPENDENCIES_AVAILABLE:
        raise RuntimeError("scikit-learn is required for the real-model storage validation")

    if expected_torch_accuracy is None:
        compile_metrics = train_and_compile(
            output_path=binary_path,
            epochs=DEFAULT_TRAIN_EPOCHS,
            seed=DEFAULT_RANDOM_SEED,
        )
        expected_torch_accuracy = compile_metrics["torch_accuracy"]

    print("Loading test dataset...")
    _, X_test, _, y_test = load_digits_split()

    print(f"Initializing Mock NVMe Array with {binary_path}...")
    metrics = evaluate_storage_model(binary_path, X_test, y_test)
    validate_storage_metrics(metrics, expected_torch_accuracy=expected_torch_accuracy)

    print(f"\nResults over {metrics['samples']} test samples:")
    print(f"Accuracy via Computational SSD: {metrics['accuracy'] * 100:.2f}%")
    print(f"Reference PyTorch Accuracy: {expected_torch_accuracy * 100:.2f}%")
    print(f"Total Theoretical Latency: {metrics['total_latency_ms']:.2f} ms")
    return metrics


if __name__ == "__main__":
    test_real_model()
