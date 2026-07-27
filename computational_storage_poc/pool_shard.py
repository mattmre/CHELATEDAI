"""Lossless retrieval-pool shards carried by the block-graph payload format.

The block graph stores matrix cells as float16, so directly passing arbitrary
float32 document vectors to ``build_graph_payload`` would quantize them.  This
module instead stores each byte of the vectors' C-order float32 representation
as one float16 cell.  Integer values 0..255 are exactly representable in
float16, allowing ``read_block`` to reconstruct the original bytes exactly.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from _experimental import mark_experimental
from block_graph import BLOCK_SIZE, TOTAL_BLOCK_BYTES, build_graph_payload, read_block


EXPERIMENTAL = True
mark_experimental(__name__, EXPERIMENTAL)

FORMAT_NAME = "chelatedai.pool-shard.block-graph"
FORMAT_VERSION = 1
BYTE_ENCODING = "float32-c-order-bytes-as-exact-fp16-u8"
CELLS_PER_BLOCK = BLOCK_SIZE * BLOCK_SIZE


def _manifest_path(shard_path: Path) -> Path:
    return Path(f"{shard_path}.manifest.json")


def _vector_bytes(vectors: np.ndarray) -> bytes:
    return np.ascontiguousarray(vectors).tobytes(order="C")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _validate_vectors_and_ids(vectors: np.ndarray, ids: Sequence[str]) -> list[str]:
    if not isinstance(vectors, np.ndarray):
        raise TypeError("vectors must be a numpy.ndarray")
    if vectors.dtype != np.dtype(np.float32):
        raise TypeError(f"vectors must have dtype float32, got {vectors.dtype}")
    if vectors.ndim != 2:
        raise ValueError(f"vectors must be 2D [n, d], got shape {vectors.shape}")
    if vectors.shape[0] == 0 or vectors.shape[1] == 0:
        raise ValueError(f"vectors must have non-zero n and d, got shape {vectors.shape}")
    if isinstance(ids, (str, bytes)) or not isinstance(ids, Sequence):
        raise TypeError("ids must be a non-text sequence of strings")
    doc_ids = list(ids)
    if any(not isinstance(doc_id, str) for doc_id in doc_ids):
        raise TypeError("ids must be a non-text sequence of strings")
    if len(doc_ids) != vectors.shape[0]:
        raise ValueError(f"ids length {len(doc_ids)} does not match vector rows {vectors.shape[0]}")
    return doc_ids


def _encode_vector_bytes(raw_vector_bytes: bytes) -> bytes:
    raw_u8 = np.frombuffer(raw_vector_bytes, dtype=np.uint8)
    matrices = []
    for start in range(0, raw_u8.size, CELLS_PER_BLOCK):
        chunk = raw_u8[start : start + CELLS_PER_BLOCK]
        rows = (chunk.size + BLOCK_SIZE - 1) // BLOCK_SIZE
        matrix = np.zeros((rows, BLOCK_SIZE), dtype=np.float32)
        matrix.reshape(-1)[: chunk.size] = chunk
        matrices.append(matrix)
    return build_graph_payload(matrices)


def write_pool_shard(path: str | Path, vectors: np.ndarray, ids: Sequence[str]) -> dict[str, Any]:
    """Write a block-graph payload and adjacent JSON manifest.

    ``path`` contains only block-graph blocks.  The manifest is written to
    ``<path>.manifest.json``.
    """
    shard_path = Path(path)
    doc_ids = _validate_vectors_and_ids(vectors, ids)
    contiguous_vectors = np.ascontiguousarray(vectors)
    raw_vector_bytes = _vector_bytes(contiguous_vectors)
    payload = _encode_vector_bytes(raw_vector_bytes)
    payload_blocks = len(payload) // TOTAL_BLOCK_BYTES

    manifest = {
        "byte_encoding": BYTE_ENCODING,
        "doc_ids": doc_ids,
        "dtype": "float32",
        "format": FORMAT_NAME,
        "payload_blocks": payload_blocks,
        "raw_vector_bytes": len(raw_vector_bytes),
        "shape": [int(contiguous_vectors.shape[0]), int(contiguous_vectors.shape[1])],
        "vector_sha256": _sha256(raw_vector_bytes),
        "version": FORMAT_VERSION,
    }

    shard_path.parent.mkdir(parents=True, exist_ok=True)
    shard_path.write_bytes(payload)
    _manifest_path(shard_path).write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    return manifest


def _load_manifest(shard_path: Path) -> dict[str, Any]:
    manifest_path = _manifest_path(shard_path)
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Unable to read pool-shard manifest {manifest_path}: {exc}") from exc
    if not isinstance(manifest, dict):
        raise ValueError("Pool-shard manifest must be a JSON object")

    required = {
        "byte_encoding",
        "doc_ids",
        "dtype",
        "format",
        "payload_blocks",
        "raw_vector_bytes",
        "shape",
        "vector_sha256",
        "version",
    }
    missing = sorted(required - manifest.keys())
    if missing:
        raise ValueError(f"Pool-shard manifest is missing fields: {missing}")
    if manifest["format"] != FORMAT_NAME or manifest["version"] != FORMAT_VERSION:
        raise ValueError(f"Unsupported pool-shard format/version: {manifest['format']!r}/{manifest['version']!r}")
    if manifest["byte_encoding"] != BYTE_ENCODING or manifest["dtype"] != "float32":
        raise ValueError(f"Unsupported pool-shard encoding/dtype: {manifest['byte_encoding']!r}/{manifest['dtype']!r}")
    return manifest


def _decode_payload_via_block_graph(payload: bytes, manifest: dict[str, Any]) -> bytes:
    block_count = manifest["payload_blocks"]
    raw_byte_count = manifest["raw_vector_bytes"]
    if not isinstance(block_count, int) or block_count <= 0:
        raise ValueError(f"Invalid payload_blocks: {block_count!r}")
    if not isinstance(raw_byte_count, int) or raw_byte_count <= 0:
        raise ValueError(f"Invalid raw_vector_bytes: {raw_byte_count!r}")
    expected_payload_bytes = block_count * TOTAL_BLOCK_BYTES
    if len(payload) != expected_payload_bytes:
        raise ValueError(f"Pool-shard payload length mismatch: expected {expected_payload_bytes}, got {len(payload)}")
    if raw_byte_count > block_count * CELLS_PER_BLOCK:
        raise ValueError(f"Manifest raw_vector_bytes {raw_byte_count} exceed {block_count} block capacity")

    decoded_chunks = []
    offset = 0
    for block_index in range(block_count):
        block = read_block(payload, offset)
        cells = block.matrix.reshape(-1)
        valid_u8_cells = np.isfinite(cells) & (cells >= 0) & (cells <= 255) & (cells == np.floor(cells))
        if not np.all(valid_u8_cells):
            bad_index = int(np.flatnonzero(~valid_u8_cells)[0])
            raise ValueError(f"Invalid encoded byte in block {block_index}, cell {bad_index}: {cells[bad_index]!r}")
        decoded_chunks.append(cells.astype(np.uint8).tobytes())

        expected_next = 0 if block_index == block_count - 1 else (block_index + 1) * TOTAL_BLOCK_BYTES
        if block.next_offset != expected_next:
            raise ValueError(
                f"Invalid block-graph link at block {block_index}: expected {expected_next}, "
                f"got {block.next_offset}"
            )
        offset = block.next_offset

    decoded_bytes = b"".join(decoded_chunks)
    if any(decoded_bytes[raw_byte_count:]):
        first_nonzero_padding = next(
            index for index, value in enumerate(decoded_bytes[raw_byte_count:], start=raw_byte_count) if value != 0
        )
        raise ValueError(
            f"Non-zero block padding at decoded byte {first_nonzero_padding}: "
            f"{decoded_bytes[first_nonzero_padding]}"
        )
    return decoded_bytes[:raw_byte_count]


def _read_pool_shard_with_manifest(path: str | Path) -> tuple[np.ndarray, list[str], dict[str, Any]]:
    shard_path = Path(path)
    manifest = _load_manifest(shard_path)
    try:
        payload = shard_path.read_bytes()
    except OSError as exc:
        raise ValueError(f"Unable to read pool-shard payload {shard_path}: {exc}") from exc

    raw_vector_bytes = _decode_payload_via_block_graph(payload, manifest)
    shape = manifest["shape"]
    if not isinstance(shape, list) or len(shape) != 2 or any(not isinstance(size, int) or size <= 0 for size in shape):
        raise ValueError(f"Invalid vector shape in manifest: {shape!r}")
    expected_raw_bytes = shape[0] * shape[1] * np.dtype(np.float32).itemsize
    if len(raw_vector_bytes) != expected_raw_bytes:
        raise ValueError(
            f"Vector byte count does not match shape {shape}: expected {expected_raw_bytes}, "
            f"got {len(raw_vector_bytes)}"
        )

    doc_ids = manifest["doc_ids"]
    if not isinstance(doc_ids, list) or any(not isinstance(doc_id, str) for doc_id in doc_ids):
        raise ValueError("Manifest doc_ids must be a list of strings")
    if len(doc_ids) != shape[0]:
        raise ValueError(f"Manifest doc_ids length {len(doc_ids)} does not match vector rows {shape[0]}")

    vectors = np.frombuffer(raw_vector_bytes, dtype=np.float32).copy().reshape(shape)
    return vectors, doc_ids, manifest


def read_pool_shard(path: str | Path) -> tuple[np.ndarray, list[str]]:
    """Reconstruct vectors and ids by traversing the block-graph payload."""
    vectors, doc_ids, _manifest = _read_pool_shard_with_manifest(path)
    return vectors, doc_ids


def _first_id_diff(expected: list[str], actual: list[str]) -> str:
    if len(expected) != len(actual):
        return f"length expected={len(expected)} actual={len(actual)}"
    for index, (expected_id, actual_id) in enumerate(zip(expected, actual)):
        if expected_id != actual_id:
            return f"index={index} expected={expected_id!r} actual={actual_id!r}"
    return "none"


def _first_vector_diff(expected: np.ndarray, actual: np.ndarray) -> str:
    if expected.shape != actual.shape:
        return f"shape expected={expected.shape} actual={actual.shape}"
    if expected.dtype != actual.dtype:
        return f"dtype expected={expected.dtype} actual={actual.dtype}"

    expected_bytes = _vector_bytes(expected)
    actual_bytes = _vector_bytes(actual)
    if expected_bytes == actual_bytes:
        return "none"

    first_byte_offset = next(
        index
        for index, (expected_byte, actual_byte) in enumerate(zip(expected_bytes, actual_bytes))
        if expected_byte != actual_byte
    )
    itemsize = expected.dtype.itemsize
    flat_index = first_byte_offset // itemsize
    index = tuple(int(value) for value in np.unravel_index(flat_index, expected.shape))
    element_start = flat_index * itemsize
    element_end = element_start + itemsize
    expected_element_bytes = expected_bytes[element_start:element_end].hex()
    actual_element_bytes = actual_bytes[element_start:element_end].hex()
    return (
        f"index={index} byte_offset={first_byte_offset} "
        f"expected={expected[index]!r} actual={actual[index]!r} "
        f"expected_bytes=0x{expected_element_bytes} actual_bytes=0x{actual_element_bytes}"
    )


def verify_pool_shard_parity(
    in_memory_vectors: np.ndarray,
    in_memory_ids: Sequence[str],
    shard_path: str | Path,
) -> dict[str, Any]:
    """Fail closed unless manifest, disk-read, and in-memory data match exactly."""
    expected_ids = _validate_vectors_and_ids(in_memory_vectors, in_memory_ids)
    disk_vectors, disk_ids, manifest = _read_pool_shard_with_manifest(shard_path)

    in_memory_bytes = _vector_bytes(in_memory_vectors)
    disk_bytes = _vector_bytes(disk_vectors)
    in_memory_hash = _sha256(in_memory_bytes)
    disk_hash = _sha256(disk_bytes)
    manifest_hash = manifest["vector_sha256"]
    manifest_hash_match = manifest_hash == disk_hash
    in_memory_hash_match = in_memory_hash == disk_hash
    ids_match = expected_ids == disk_ids
    # Byte-exact comparison (not np.array_equal / IEEE value equality): a bit-exact
    # parity check must treat byte-identical NaN payloads as matching, and this is
    # the same semantics as the SHA256 check above.
    vectors_match = in_memory_vectors.shape == disk_vectors.shape and in_memory_bytes == disk_bytes

    if not (manifest_hash_match and in_memory_hash_match and ids_match and vectors_match):
        raise AssertionError(
            "Pool-shard parity mismatch: "
            f"manifest_hash={manifest_hash} disk_hash={disk_hash} in_memory_hash={in_memory_hash}; "
            f"manifest_hash_match={manifest_hash_match} in_memory_hash_match={in_memory_hash_match}; "
            f"ids_match={ids_match} ({_first_id_diff(expected_ids, disk_ids)}); "
            f"vectors_match={vectors_match} ({_first_vector_diff(in_memory_vectors, disk_vectors)})"
        )

    n, d = in_memory_vectors.shape
    return {
        "d": int(d),
        "disk_sha256": disk_hash,
        "ids_match": ids_match,
        "in_memory_sha256": in_memory_hash,
        "in_memory_hash_match": in_memory_hash_match,
        "manifest_hash_match": manifest_hash_match,
        "manifest_sha256": manifest_hash,
        "n": int(n),
        "vectors_match": vectors_match,
    }


def retrieve_topk(query: np.ndarray, vectors: np.ndarray, ids: Sequence[str], k: int) -> list[str]:
    """Return deterministic top-k ids using float32 dot-product scores."""
    doc_ids = _validate_vectors_and_ids(vectors, ids)
    query_array = np.asarray(query, dtype=np.float32)
    if query_array.ndim != 1 or query_array.shape[0] != vectors.shape[1]:
        raise ValueError(f"query must have shape ({vectors.shape[1]},), got {query_array.shape}")
    if not isinstance(k, int) or isinstance(k, bool) or k <= 0 or k > vectors.shape[0]:
        raise ValueError(f"k must be an integer in [1, {vectors.shape[0]}], got {k!r}")

    scores = vectors @ query_array
    top_indices = np.argsort(-scores, kind="stable")[:k]
    return [doc_ids[int(index)] for index in top_indices]
