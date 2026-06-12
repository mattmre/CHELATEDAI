from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import Any

import numpy as np

BLOCK_SIZE = 512
PARAM_TYPE = np.float16
BYTES_PER_PARAM = np.dtype(PARAM_TYPE).itemsize
MATRIX_BYTES = BLOCK_SIZE * BLOCK_SIZE * BYTES_PER_PARAM
POINTER_BYTES = 8
TOTAL_BLOCK_BYTES = MATRIX_BYTES + POINTER_BYTES
try:
    from chelated_shim_research import (
        promoted_sip_apply,
        bump_stall_counter,
        research_enabled,
        research_preflight_metadata,
    )
except ModuleNotFoundError:
    def promoted_sip_apply(v):
        return np.array(v, dtype=float).copy(), None

    def research_enabled() -> bool:
        return False

    def bump_stall_counter(counter: int, *, has_work: bool) -> int:
        return counter

    def research_preflight_metadata(
        *,
        seam: str,
        stall_count: int,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return {
            "research_shim_guard": False,
            "research_stall_count": stall_count,
            "sip_seam": seam,
            **(dict(extra or {})),
        }

_research_block_graph_stall_count = 0
_last_research_shim_meta: dict[str, Any] | None = None


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
    global _research_block_graph_stall_count, _last_research_shim_meta
    current_offset = int(trigger_offset)
    current_activations = input_activations.astype(np.float32)
    blocks_processed = 0
    dim = int(current_activations.shape[-1]) if current_activations.ndim else 8
    if dim <= 0:
        dim = 8

    try:
        while True:
            block = read_block(flash_memory, current_offset)
            current_activations = current_activations @ block.matrix
            blocks_processed += 1

            if block.next_offset == 0:
                break

            current_activations = apply_hidden_activation(
                current_activations, hidden_activation
            )
            current_offset = block.next_offset
    except Exception as exc:
        if research_enabled():
            _, promoted_meta = promoted_sip_apply(np.zeros(dim, dtype=float))
            _research_block_graph_stall_count = bump_stall_counter(
                _research_block_graph_stall_count,
                has_work=False,
            )
            _last_research_shim_meta = research_preflight_metadata(
                seam="computational_storage_poc.block_graph.run_block_graph",
                stall_count=_research_block_graph_stall_count,
                extra={
                    "error": exc.__class__.__name__,
                    "error_message": str(exc),
                    "blocks_processed": blocks_processed,
                    "payload_bytes": len(flash_memory),
                    **({"promoted_sip_apply": promoted_meta} if promoted_meta else {}),
                },
            )
        raise

    if research_enabled():
        flat = current_activations.reshape(-1)
        sample_dim = min(8, flat.size)
        sample = flat[:sample_dim].astype(float)
        applied, promoted_meta = promoted_sip_apply(sample)
        if applied is not None and len(applied):
            flat = flat.astype(float).copy()
            flat[:sample_dim] = applied[:sample_dim]
            current_activations = flat.reshape(current_activations.shape)
        _research_block_graph_stall_count = bump_stall_counter(
            _research_block_graph_stall_count,
            has_work=True,
        )
        _last_research_shim_meta = research_preflight_metadata(
            seam="computational_storage_poc.block_graph.run_block_graph",
            stall_count=_research_block_graph_stall_count,
            extra={
                "blocks_processed": blocks_processed,
                "payload_bytes": len(flash_memory),
                "input_shape": tuple(current_activations.shape),
                "hidden_activation": hidden_activation,
                "promoted_sip_apply": promoted_meta,
            },
        )

    return current_activations, blocks_processed


def get_last_research_shim_meta() -> dict[str, Any] | None:
    """Return metadata from the most recent research-seam preflight path."""
    return _last_research_shim_meta


def clear_last_research_shim_meta() -> None:
    """Reset research metadata for deterministic tests."""
    global _last_research_shim_meta, _research_block_graph_stall_count
    _last_research_shim_meta = None
    _research_block_graph_stall_count = 0
