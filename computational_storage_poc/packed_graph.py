from __future__ import annotations

import json
import mmap
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

import numpy as np

from block_graph import PARAM_TYPE, apply_hidden_activation

# Research-stage / POC module. No production code path in this repo consumes it.
EXPERIMENTAL = True

PACKED_GRAPH_MAGIC = b"CSPG01"
HEADER_LENGTH_BYTES = 4
FLOAT16_STORAGE_DTYPE = "float16"
INT8_STORAGE_DTYPE = "int8"


@dataclass(frozen=True)
class PackedBlockMetadata:
    rows: int
    cols: int
    matrix_offset: int
    matrix_nbytes: int
    next_block: int | None
    storage_dtype: str
    scale: float


@dataclass(frozen=True)
class PackedGraphBlock:
    matrix: np.ndarray
    next_block: int | None


@dataclass(frozen=True)
class PackedGraphChunk:
    matrix: np.ndarray
    row_start: int
    row_end: int
    bytes_read: int


@dataclass(frozen=True)
class QuantizedPackedGraphBlock:
    quantized_matrix: np.ndarray
    scale: float
    next_block: int | None


@dataclass(frozen=True)
class QuantizedPackedGraphChunk:
    quantized_matrix: np.ndarray
    scale: float
    row_start: int
    row_end: int
    bytes_read: int


def _quantize_matrix_to_int8(matrix: np.ndarray) -> tuple[np.ndarray, float]:
    max_abs = float(np.max(np.abs(matrix)))
    if max_abs == 0.0:
        return np.zeros_like(matrix, dtype=np.int8), 1.0

    scale = max_abs / 127.0
    quantized = np.clip(np.rint(matrix / scale), -127, 127).astype(np.int8)
    return quantized, scale


def _build_manifest(layer_matrices: list[np.ndarray], storage_dtype: str) -> tuple[dict, bytes]:
    blocks: list[dict[str, int | None]] = []
    blob_parts: list[bytes] = []
    current_offset = 0

    for index, matrix in enumerate(layer_matrices):
        if matrix.ndim != 2:
            raise ValueError("Packed graph layers must be 2D matrices")

        rows, cols = matrix.shape
        if storage_dtype == FLOAT16_STORAGE_DTYPE:
            packed = matrix.astype(PARAM_TYPE, copy=False).tobytes(order="C")
            scale = 1.0
        elif storage_dtype == INT8_STORAGE_DTYPE:
            quantized, scale = _quantize_matrix_to_int8(matrix.astype(np.float32, copy=False))
            packed = quantized.tobytes(order="C")
        else:
            raise ValueError(f"Unsupported storage dtype: {storage_dtype}")

        blocks.append(
            {
                "rows": int(rows),
                "cols": int(cols),
                "matrix_offset": current_offset,
                "matrix_nbytes": len(packed),
                "next_block": None if index == len(layer_matrices) - 1 else index + 1,
                "storage_dtype": storage_dtype,
                "scale": scale,
            }
        )
        blob_parts.append(packed)
        current_offset += len(packed)

    manifest = {
        "version": 1,
        "param_dtype": str(np.dtype(PARAM_TYPE)),
        "storage_dtype": storage_dtype,
        "blocks": blocks,
    }
    return manifest, b"".join(blob_parts)


def build_packed_graph_artifact(layer_matrices, storage_dtype: str = FLOAT16_STORAGE_DTYPE) -> bytes:
    matrices = [np.asarray(matrix, dtype=np.float32) for matrix in layer_matrices]
    if not matrices:
        raise ValueError("At least one layer matrix is required")

    manifest, blob = _build_manifest(matrices, storage_dtype=storage_dtype)
    manifest_bytes = json.dumps(manifest, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return PACKED_GRAPH_MAGIC + struct.pack("<I", len(manifest_bytes)) + manifest_bytes + blob


def write_packed_graph_artifact(
    path: str | Path,
    layer_matrices,
    storage_dtype: str = FLOAT16_STORAGE_DTYPE,
) -> dict[str, int | str]:
    matrices = [np.asarray(matrix, dtype=np.float32) for matrix in layer_matrices]
    payload = build_packed_graph_artifact(matrices, storage_dtype=storage_dtype)
    target = Path(path)
    target.write_bytes(payload)
    return {
        "artifact_bytes": len(payload),
        "block_count": len(matrices),
        "storage_dtype": storage_dtype,
    }


class DiskBackedPackedGraph:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._file: BinaryIO = self.path.open("rb")
        self._mapping = mmap.mmap(self._file.fileno(), 0, access=mmap.ACCESS_READ)

        header = self._mapping[: len(PACKED_GRAPH_MAGIC)]
        if header != PACKED_GRAPH_MAGIC:
            self.close()
            raise ValueError("Not a packed graph artifact")

        manifest_size_start = len(PACKED_GRAPH_MAGIC)
        manifest_size_end = manifest_size_start + HEADER_LENGTH_BYTES
        manifest_size = struct.unpack("<I", self._mapping[manifest_size_start:manifest_size_end])[0]
        manifest_start = manifest_size_end
        manifest_end = manifest_start + manifest_size
        manifest_bytes = self._mapping[manifest_start:manifest_end]
        manifest = json.loads(manifest_bytes.decode("utf-8"))

        self.manifest = manifest
        self.data_start = manifest_end
        self.blocks = [
            PackedBlockMetadata(
                rows=int(block["rows"]),
                cols=int(block["cols"]),
                matrix_offset=int(block["matrix_offset"]),
                matrix_nbytes=int(block["matrix_nbytes"]),
                next_block=None if block["next_block"] is None else int(block["next_block"]),
                storage_dtype=str(block.get("storage_dtype", manifest.get("storage_dtype", FLOAT16_STORAGE_DTYPE))),
                scale=float(block.get("scale", 1.0)),
            )
            for block in manifest["blocks"]
        ]

    def close(self) -> None:
        mapping = getattr(self, "_mapping", None)
        if mapping is not None:
            mapping.close()
            self._mapping = None
        file_obj = getattr(self, "_file", None)
        if file_obj is not None:
            file_obj.close()
            self._file = None

    def __enter__(self) -> "DiskBackedPackedGraph":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def __del__(self) -> None:
        self.close()

    @staticmethod
    def _dtype_for_storage(storage_dtype: str):
        if storage_dtype == FLOAT16_STORAGE_DTYPE:
            return PARAM_TYPE
        if storage_dtype == INT8_STORAGE_DTYPE:
            return np.int8
        raise ValueError(f"Unsupported storage dtype: {storage_dtype}")

    def _resolve_chunk_bounds(self, metadata: PackedBlockMetadata, row_start: int, row_end: int) -> tuple[int, int]:
        if row_start < 0 or row_end < 0:
            raise ValueError("Row chunk bounds must be non-negative")
        if row_start >= row_end:
            raise ValueError("Row chunk must contain at least one row")
        if row_end > metadata.rows:
            raise ValueError(f"Row chunk [{row_start}, {row_end}) exceeds block rows {metadata.rows}")
        return row_start, row_end

    def _read_chunk_bytes(self, metadata: PackedBlockMetadata, row_start: int, row_end: int) -> tuple[bytes, int]:
        storage_dtype = self._dtype_for_storage(metadata.storage_dtype)
        bytes_per_row = metadata.cols * np.dtype(storage_dtype).itemsize
        start = self.data_start + metadata.matrix_offset + (row_start * bytes_per_row)
        byte_count = (row_end - row_start) * bytes_per_row
        end = start + byte_count
        return self._mapping[start:end], byte_count

    def read_block(self, block_index: int) -> PackedGraphBlock:
        if block_index < 0 or block_index >= len(self.blocks):
            raise ValueError(f"Block index {block_index} is out of range")

        metadata = self.blocks[block_index]
        start = self.data_start + metadata.matrix_offset
        end = start + metadata.matrix_nbytes
        matrix_bytes = self._mapping[start:end]
        if metadata.storage_dtype == FLOAT16_STORAGE_DTYPE:
            matrix = np.frombuffer(matrix_bytes, dtype=PARAM_TYPE).reshape((metadata.rows, metadata.cols)).astype(np.float32)
        elif metadata.storage_dtype == INT8_STORAGE_DTYPE:
            quantized_matrix = np.frombuffer(matrix_bytes, dtype=np.int8).reshape((metadata.rows, metadata.cols))
            matrix = quantized_matrix.astype(np.float32) * metadata.scale
        else:
            raise ValueError(f"Unsupported storage dtype: {metadata.storage_dtype}")
        return PackedGraphBlock(matrix=matrix, next_block=metadata.next_block)

    def read_quantized_block(self, block_index: int) -> QuantizedPackedGraphBlock:
        if block_index < 0 or block_index >= len(self.blocks):
            raise ValueError(f"Block index {block_index} is out of range")

        metadata = self.blocks[block_index]
        if metadata.storage_dtype != INT8_STORAGE_DTYPE:
            raise ValueError("Quantized block reads require int8 packed storage")

        start = self.data_start + metadata.matrix_offset
        end = start + metadata.matrix_nbytes
        matrix_bytes = self._mapping[start:end]
        quantized_matrix = np.frombuffer(matrix_bytes, dtype=np.int8).reshape((metadata.rows, metadata.cols))
        return QuantizedPackedGraphBlock(
            quantized_matrix=quantized_matrix,
            scale=metadata.scale,
            next_block=metadata.next_block,
        )

    def read_block_chunk(self, block_index: int, row_start: int, row_end: int) -> PackedGraphChunk:
        if block_index < 0 or block_index >= len(self.blocks):
            raise ValueError(f"Block index {block_index} is out of range")

        metadata = self.blocks[block_index]
        row_start, row_end = self._resolve_chunk_bounds(metadata, row_start, row_end)
        matrix_bytes, byte_count = self._read_chunk_bytes(metadata, row_start, row_end)

        if metadata.storage_dtype == FLOAT16_STORAGE_DTYPE:
            matrix = np.frombuffer(matrix_bytes, dtype=PARAM_TYPE).reshape((row_end - row_start, metadata.cols)).astype(
                np.float32
            )
        elif metadata.storage_dtype == INT8_STORAGE_DTYPE:
            quantized_matrix = np.frombuffer(matrix_bytes, dtype=np.int8).reshape((row_end - row_start, metadata.cols))
            matrix = quantized_matrix.astype(np.float32) * metadata.scale
        else:
            raise ValueError(f"Unsupported storage dtype: {metadata.storage_dtype}")

        return PackedGraphChunk(
            matrix=matrix,
            row_start=row_start,
            row_end=row_end,
            bytes_read=byte_count,
        )

    def read_quantized_block_chunk(self, block_index: int, row_start: int, row_end: int) -> QuantizedPackedGraphChunk:
        if block_index < 0 or block_index >= len(self.blocks):
            raise ValueError(f"Block index {block_index} is out of range")

        metadata = self.blocks[block_index]
        if metadata.storage_dtype != INT8_STORAGE_DTYPE:
            raise ValueError("Quantized chunk reads require int8 packed storage")

        row_start, row_end = self._resolve_chunk_bounds(metadata, row_start, row_end)
        matrix_bytes, byte_count = self._read_chunk_bytes(metadata, row_start, row_end)
        quantized_matrix = np.frombuffer(matrix_bytes, dtype=np.int8).reshape((row_end - row_start, metadata.cols))
        return QuantizedPackedGraphChunk(
            quantized_matrix=quantized_matrix,
            scale=metadata.scale,
            row_start=row_start,
            row_end=row_end,
            bytes_read=byte_count,
        )

    @property
    def artifact_size_bytes(self) -> int:
        return len(self._mapping)

    @property
    def block_count(self) -> int:
        return len(self.blocks)


def run_packed_graph(
    graph: DiskBackedPackedGraph,
    input_activations: np.ndarray,
    trigger_block: int = 0,
    hidden_activation: str | None = "relu",
) -> tuple[np.ndarray, int, int]:
    current_block = int(trigger_block)
    current_activations = input_activations.astype(np.float32)
    blocks_processed = 0
    bytes_read = 0

    while True:
        block = graph.read_block(current_block)
        metadata = graph.blocks[current_block]
        current_activations = current_activations[:, : metadata.rows] @ block.matrix
        blocks_processed += 1
        bytes_read += metadata.matrix_nbytes

        if block.next_block is None:
            break

        current_activations = apply_hidden_activation(current_activations, hidden_activation)
        current_block = block.next_block

    return current_activations, blocks_processed, bytes_read
