from __future__ import annotations

import numpy as np

from block_graph import BLOCK_SIZE, BYTES_PER_PARAM, TOTAL_BLOCK_BYTES, run_block_graph
from validation_config import MAX_PARITY_DIFF, MIN_SPEEDUP_FACTOR

# Theoretical latencies in seconds
PCIE_ROUND_TRIP_LATENCY = 10e-6
PCIE_BANDWIDTH_BPS = 7.88e9
INTERNAL_FLASH_LATENCY = 100e-6
CPU_MACS_PER_SECOND = 50e9
SSD_MACS_PER_SECOND = 1e12


def _transfer_latency(num_bytes: int) -> float:
    return PCIE_ROUND_TRIP_LATENCY + (num_bytes / PCIE_BANDWIDTH_BPS)


def _matrix_compute_latency(macs_per_second: float) -> float:
    return (BLOCK_SIZE * BLOCK_SIZE) / macs_per_second


class MockNVMeDrive:
    def __init__(self, binary_path: str):
        with open(binary_path, "rb") as f:
            self.flash_memory = f.read()

    def host_read_block(self, offset: int, size: int):
        transfer_time = size / PCIE_BANDWIDTH_BPS
        latency_cost = PCIE_ROUND_TRIP_LATENCY + INTERNAL_FLASH_LATENCY + transfer_time
        return self.flash_memory[offset : offset + size], latency_cost

    def computational_inference(
        self,
        trigger_offset: int,
        input_activations: np.ndarray,
        hidden_activation: str | None = "relu",
    ):
        current_activations, blocks_processed = run_block_graph(
            self.flash_memory,
            input_activations,
            trigger_offset=trigger_offset,
            hidden_activation=hidden_activation,
        )
        total_latency = (
            _transfer_latency(input_activations.nbytes)
            + blocks_processed * (INTERNAL_FLASH_LATENCY + _matrix_compute_latency(SSD_MACS_PER_SECOND))
            + _transfer_latency(current_activations.nbytes)
        )
        return current_activations, total_latency


def traditional_host_inference(
    drive: MockNVMeDrive,
    trigger_offset: int,
    input_activations: np.ndarray,
    hidden_activation: str | None = "relu",
):
    current_activations, blocks_processed = run_block_graph(
        drive.flash_memory,
        input_activations,
        trigger_offset=trigger_offset,
        hidden_activation=hidden_activation,
    )
    total_latency = blocks_processed * (
        drive.host_read_block(0, TOTAL_BLOCK_BYTES)[1] + _matrix_compute_latency(CPU_MACS_PER_SECOND)
    )
    return current_activations, total_latency


def profile_sample_inference(binary_path: str = "model.bin", seed: int = 42):
    rng = np.random.default_rng(seed)
    drive = MockNVMeDrive(binary_path)
    inputs = np.zeros((1, BLOCK_SIZE), dtype=np.float16)
    inputs[0, :256] = rng.normal(size=256).astype(np.float32) * 0.1

    out_trad, trad_time = traditional_host_inference(drive, 0x0, inputs, hidden_activation="relu")
    out_comp, comp_time = drive.computational_inference(0x0, inputs, hidden_activation="relu")
    diff = float(np.max(np.abs(out_trad - out_comp)))

    return {
        "traditional_time_us": trad_time * 1_000_000,
        "computational_time_us": comp_time * 1_000_000,
        "max_abs_diff": diff,
        "speedup_factor": trad_time / comp_time,
    }


if __name__ == "__main__":
    metrics = profile_sample_inference()

    print("\n--- Theoretical Latency Profiling (2 Layers) ---")
    print(f"Traditional Inference: {metrics['traditional_time_us']:.1f} microseconds")
    print(f"Computational Storage: {metrics['computational_time_us']:.1f} microseconds")
    print(
        f"\nVerification: Outputs match? {'YES' if metrics['max_abs_diff'] < MAX_PARITY_DIFF else 'NO'} "
        f"(Max diff: {metrics['max_abs_diff']})"
    )
    print(f"Speedup Factor: {metrics['speedup_factor']:.2f}x\n")

    layers = 10000
    print(
        f"--- Extrapolated to 10,000 layers "
        f"({min(layers * BLOCK_SIZE * BLOCK_SIZE * BYTES_PER_PARAM / 1e9, 9999):.2f} GB model) ---"
    )
    print(f"Traditional Inference: {(metrics['traditional_time_us'] / 1_000_000 / 2) * layers * 1000:.1f} ms")
    print(f"Computational Storage: {(metrics['computational_time_us'] / 1_000_000 / 2) * layers * 1000:.1f} ms")

    if metrics["max_abs_diff"] >= MAX_PARITY_DIFF:
        raise SystemExit("Storage and host outputs diverged")
    if metrics["speedup_factor"] <= MIN_SPEEDUP_FACTOR:
        raise SystemExit("Theoretical computational-storage speedup regressed")
