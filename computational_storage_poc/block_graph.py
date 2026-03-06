import struct
from dataclasses import dataclass

import numpy as np

BLOCK_SIZE = 512
PARAM_TYPE = np.float16
BYTES_PER_PARAM = np.dtype(PARAM_TYPE).itemsize
MATRIX_BYTES = BLOCK_SIZE * BLOCK_SIZE * BYTES_PER_PARAM
POINTER_BYTES = 8
TOTAL_BLOCK_BYTES = MATRIX_BYTES + POINTER_BYTES


@dataclass(frozen=True)
class BlockRecord:
    matrix: np.ndarray
    next_offset: int


def create_block(matrix: np.ndarray, next_block_offset: int) -> bytes:
    if matrix.ndim != 2:
        raise ValueError("Matrix must be 2D")

    rows, cols = matrix.shape
    if rows > BLOCK_SIZE or cols > BLOCK_SIZE:
        raise ValueError(f"Matrix exceeds {BLOCK_SIZE}x{BLOCK_SIZE}")

    padded = np.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=PARAM_TYPE)
    padded[:rows, :cols] = matrix.astype(PARAM_TYPE)
    return padded.tobytes() + struct.pack("<Q", int(next_block_offset))


def build_graph_payload(layer_matrices) -> bytes:
    matrices = list(layer_matrices)
    if not matrices:
        raise ValueError("At least one layer matrix is required")

    blocks = []
    for index, matrix in enumerate(matrices):
        next_offset = 0 if index == len(matrices) - 1 else (index + 1) * TOTAL_BLOCK_BYTES
        blocks.append(create_block(matrix, next_offset))
    return b"".join(blocks)


def read_block(flash_memory: bytes, offset: int) -> BlockRecord:
    if offset < 0:
        raise ValueError("Block offsets must be non-negative")

    end_offset = offset + TOTAL_BLOCK_BYTES
    if end_offset > len(flash_memory):
        raise ValueError(
            f"Offset {offset} points outside the graph payload "
            f"({len(flash_memory)} bytes available)"
        )

    block_data = flash_memory[offset:end_offset]
    matrix_bytes = block_data[:MATRIX_BYTES]
    pointer_bytes = block_data[MATRIX_BYTES:]
    matrix = np.frombuffer(matrix_bytes, dtype=PARAM_TYPE).reshape((BLOCK_SIZE, BLOCK_SIZE)).astype(np.float32)
    next_offset = struct.unpack("<Q", pointer_bytes)[0]
    return BlockRecord(matrix=matrix, next_offset=next_offset)


def apply_hidden_activation(values: np.ndarray, activation: str | None) -> np.ndarray:
    if activation in (None, "identity"):
        return values
    if activation == "relu":
        return np.maximum(values, 0)
    raise ValueError(f"Unsupported hidden activation: {activation}")


def run_block_graph(
    flash_memory: bytes,
    input_activations: np.ndarray,
    trigger_offset: int = 0,
    hidden_activation: str | None = "relu",
):
    current_offset = int(trigger_offset)
    current_activations = input_activations.astype(np.float32)
    blocks_processed = 0

    while True:
        block = read_block(flash_memory, current_offset)
        current_activations = current_activations @ block.matrix
        blocks_processed += 1

        if block.next_offset == 0:
            break

        current_activations = apply_hidden_activation(current_activations, hidden_activation)
        current_offset = block.next_offset

    return current_activations, blocks_processed
