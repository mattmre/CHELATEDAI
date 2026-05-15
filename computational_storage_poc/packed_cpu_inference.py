from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from block_graph import apply_hidden_activation
from cpu_backends import CPUInferenceBackend, NumpyFloat32Backend
from packed_graph import DiskBackedPackedGraph, INT8_STORAGE_DTYPE


@dataclass(frozen=True)
class PackedCPUInferenceResult:
    output: np.ndarray
    backend_name: str
    blocks_processed: int
    bytes_read: int


def run_packed_graph_with_backend(
    graph: DiskBackedPackedGraph,
    input_activations: np.ndarray,
    backend: CPUInferenceBackend | None = None,
    *,
    trigger_block: int = 0,
    hidden_activation: str | None = "relu",
) -> PackedCPUInferenceResult:
    inference_backend = backend or NumpyFloat32Backend()
    current_block = int(trigger_block)
    current_activations = input_activations.astype(np.float32)
    blocks_processed = 0
    bytes_read = 0

    while True:
        metadata = graph.blocks[current_block]
        if metadata.storage_dtype == INT8_STORAGE_DTYPE:
            quantized_block = graph.read_quantized_block(current_block)
            backend_result = inference_backend.matmul_quantized_weights(
                current_activations[:, : metadata.rows],
                quantized_block.quantized_matrix,
                quantized_block.scale,
            )
            next_block = quantized_block.next_block
        else:
            block = graph.read_block(current_block)
            backend_result = inference_backend.matmul(current_activations[:, : metadata.rows], block.matrix)
            next_block = block.next_block

        current_activations = backend_result.output
        blocks_processed += 1
        bytes_read += metadata.matrix_nbytes

        if next_block is None:
            break

        current_activations = apply_hidden_activation(current_activations, hidden_activation)
        current_block = next_block

    return PackedCPUInferenceResult(
        output=current_activations,
        backend_name=inference_backend.name,
        blocks_processed=blocks_processed,
        bytes_read=bytes_read,
    )
