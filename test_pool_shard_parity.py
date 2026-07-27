import hashlib
import json
import os
import struct
import sys
import tempfile
import unittest
from collections.abc import Sequence
from pathlib import Path

import numpy as np

POC_DIR = os.path.join(os.path.dirname(__file__), "computational_storage_poc")
if POC_DIR not in sys.path:
    sys.path.insert(0, POC_DIR)

from block_graph import POINTER_BYTES, TOTAL_BLOCK_BYTES  # noqa: E402
from pool_shard import (  # noqa: E402
    CELLS_PER_BLOCK,
    read_pool_shard,
    retrieve_topk,
    verify_pool_shard_parity,
    write_pool_shard,
)


FIXTURE_VECTORS = np.array(
    [
        [0.1, -1.25, 3.1415927, 0.0],
        [2.5, 0.33333334, -0.75, 1.0],
        [-1.0, 4.25, 0.2, -2.0],
        [0.5, 0.5, 0.5, 0.5],
        [8.0, -3.0, 1.5, 0.125],
        [-0.25, 0.75, 2.0, 3.5],
    ],
    dtype=np.float32,
)
FIXTURE_IDS = ["doc-alpha", "doc-beta", "doc-gamma", "doc-delta", "doc-epsilon", "doc-zeta"]
FIXTURE_QUERIES = [
    np.array([1.0, 0.0, 0.5, -0.25], dtype=np.float32),
    np.array([-0.5, 1.0, 0.25, 0.75], dtype=np.float32),
    np.array([0.0, -1.0, 1.0, 1.0], dtype=np.float32),
]


class DocIdSequence(Sequence[str]):
    """Non-list sequence used to exercise the public Sequence[str] contract."""

    def __init__(self, values):
        self._values = tuple(values)

    def __getitem__(self, index):
        return self._values[index]

    def __len__(self):
        return len(self._values)


class TestPoolShardParity(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.shard_path = Path(self.temp_dir.name) / "fixture.pool"

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_round_trip_and_manifest_sha256_are_bit_identical(self):
        manifest = write_pool_shard(self.shard_path, FIXTURE_VECTORS, FIXTURE_IDS)

        disk_vectors, disk_ids = read_pool_shard(self.shard_path)
        disk_hash = hashlib.sha256(disk_vectors.tobytes(order="C")).hexdigest()
        disk_manifest = json.loads(Path(f"{self.shard_path}.manifest.json").read_text(encoding="utf-8"))

        self.assertEqual(disk_vectors.dtype, np.dtype(np.float32))
        self.assertTrue(np.array_equal(disk_vectors, FIXTURE_VECTORS))
        self.assertEqual(disk_ids, FIXTURE_IDS)
        self.assertEqual(disk_hash, manifest["vector_sha256"])
        self.assertEqual(disk_hash, disk_manifest["vector_sha256"])

        parity = verify_pool_shard_parity(FIXTURE_VECTORS, FIXTURE_IDS, self.shard_path)
        self.assertTrue(parity["manifest_hash_match"])
        self.assertTrue(parity["in_memory_hash_match"])
        self.assertTrue(parity["ids_match"])
        self.assertTrue(parity["vectors_match"])
        self.assertEqual((parity["n"], parity["d"]), FIXTURE_VECTORS.shape)

    def test_disk_read_retrieval_matches_in_memory_for_fixture_queries(self):
        write_pool_shard(self.shard_path, FIXTURE_VECTORS, FIXTURE_IDS)
        disk_vectors, disk_ids = read_pool_shard(self.shard_path)

        for query in FIXTURE_QUERIES:
            with self.subTest(query=query.tolist()):
                in_memory_topk = retrieve_topk(query, FIXTURE_VECTORS, FIXTURE_IDS, k=3)
                disk_topk = retrieve_topk(query, disk_vectors, disk_ids, k=3)
                self.assertEqual(disk_topk, in_memory_topk)

    def test_write_accepts_non_list_sequence_of_strings(self):
        sequence_ids = DocIdSequence(FIXTURE_IDS)

        write_pool_shard(self.shard_path, FIXTURE_VECTORS, sequence_ids)
        disk_vectors, disk_ids = read_pool_shard(self.shard_path)

        self.assertEqual(disk_vectors.tobytes(order="C"), FIXTURE_VECTORS.tobytes(order="C"))
        self.assertEqual(disk_ids, FIXTURE_IDS)

    def test_write_rejects_text_scalars_and_non_string_members(self):
        invalid_ids = (
            "doc-alpha",
            b"doc-alpha",
            [*FIXTURE_IDS[:-1], 7],
        )

        for ids in invalid_ids:
            with self.subTest(ids=ids):
                with self.assertRaisesRegex(TypeError, "non-text sequence of strings"):
                    write_pool_shard(self.shard_path, FIXTURE_VECTORS, ids)

    def test_read_rejects_non_object_manifest_roots(self):
        manifest_path = Path(f"{self.shard_path}.manifest.json")
        non_object_roots = (None, True, 7, "manifest", ["manifest"])

        for root in non_object_roots:
            with self.subTest(root=root):
                manifest_path.write_text(json.dumps(root), encoding="utf-8")
                with self.assertRaisesRegex(
                    ValueError,
                    "Pool-shard manifest must be a JSON object",
                ):
                    read_pool_shard(self.shard_path)

    def test_manifest_scalar_integer_fields_reject_bool_and_float(self):
        base_manifest = write_pool_shard(self.shard_path, FIXTURE_VECTORS, FIXTURE_IDS)
        manifest_path = Path(f"{self.shard_path}.manifest.json")

        for field in ("version", "payload_blocks", "raw_vector_bytes"):
            invalid_values = (True, float(base_manifest[field]))
            for invalid_value in invalid_values:
                with self.subTest(field=field, invalid_value=invalid_value):
                    candidate = dict(base_manifest)
                    candidate[field] = invalid_value
                    manifest_path.write_text(json.dumps(candidate), encoding="utf-8")
                    with self.assertRaisesRegex(
                        ValueError,
                        rf"Manifest {field} must be .*integer",
                    ):
                        read_pool_shard(self.shard_path)

    def test_manifest_shape_components_require_exact_positive_integers(self):
        base_manifest = write_pool_shard(self.shard_path, FIXTURE_VECTORS, FIXTURE_IDS)
        manifest_path = Path(f"{self.shard_path}.manifest.json")
        n, d = base_manifest["shape"]
        invalid_shapes = (
            [True, d],
            [n, False],
            [float(n), d],
            [n, float(d)],
        )

        for invalid_shape in invalid_shapes:
            with self.subTest(invalid_shape=invalid_shape):
                candidate = dict(base_manifest)
                candidate["shape"] = invalid_shape
                manifest_path.write_text(json.dumps(candidate), encoding="utf-8")
                with self.assertRaisesRegex(
                    ValueError,
                    "Manifest shape must be a two-element list of positive integers",
                ):
                    read_pool_shard(self.shard_path)

    def test_read_rejects_noncanonical_extra_zero_block(self):
        manifest = write_pool_shard(self.shard_path, FIXTURE_VECTORS, FIXTURE_IDS)
        payload = bytearray(self.shard_path.read_bytes())
        payload[-POINTER_BYTES:] = struct.pack("<Q", TOTAL_BLOCK_BYTES)
        payload.extend(bytes(TOTAL_BLOCK_BYTES))
        self.shard_path.write_bytes(payload)

        manifest["payload_blocks"] += 1
        Path(f"{self.shard_path}.manifest.json").write_text(
            json.dumps(manifest),
            encoding="utf-8",
        )

        with self.assertRaisesRegex(ValueError, "Non-canonical payload_blocks"):
            read_pool_shard(self.shard_path)
        with self.assertRaisesRegex(ValueError, "Non-canonical payload_blocks"):
            verify_pool_shard_parity(FIXTURE_VECTORS, FIXTURE_IDS, self.shard_path)

    def test_round_trip_one_float_past_single_block_capacity(self):
        float_count = CELLS_PER_BLOCK // np.dtype(np.float32).itemsize + 1
        vectors = np.arange(float_count, dtype=np.float32).reshape(1, float_count)
        ids = ["doc-multi-block"]

        manifest = write_pool_shard(self.shard_path, vectors, ids)
        disk_vectors, disk_ids = read_pool_shard(self.shard_path)
        parity = verify_pool_shard_parity(vectors, ids, self.shard_path)

        self.assertEqual(manifest["raw_vector_bytes"], CELLS_PER_BLOCK + 4)
        self.assertEqual(manifest["payload_blocks"], 2)
        self.assertEqual(self.shard_path.stat().st_size, 2 * TOTAL_BLOCK_BYTES)
        self.assertEqual(disk_vectors.tobytes(order="C"), vectors.tobytes(order="C"))
        self.assertEqual(disk_ids, ids)
        self.assertTrue(parity["vectors_match"])

    def test_nan_payload_round_trips_byte_identically_and_parity_passes(self):
        # Byte-exact parity must treat a byte-identical NaN payload as matching
        # (np.array_equal would wrongly reject it since NaN != NaN). Also covers
        # +inf/-inf which are byte-stable but not IEEE-equal to themselves under
        # value comparison only for NaN — included for completeness.
        vectors = np.array(
            [[np.nan, 1.0, -2.0, 0.5], [np.inf, -np.inf, 0.0, np.nan]],
            dtype=np.float32,
        )
        ids = ["doc-nan-a", "doc-nan-b"]
        write_pool_shard(self.shard_path, vectors, ids)
        disk_vectors, disk_ids = read_pool_shard(self.shard_path)

        # Byte-identical round-trip (NaN payloads preserved bit-for-bit).
        self.assertEqual(disk_vectors.tobytes(order="C"), vectors.tobytes(order="C"))
        self.assertEqual(disk_ids, ids)
        parity = verify_pool_shard_parity(vectors, ids, self.shard_path)
        self.assertTrue(parity["vectors_match"])
        self.assertTrue(parity["in_memory_hash_match"])

    def test_nan_diagnostic_skips_matching_nan_before_later_difference(self):
        disk_vectors = np.array([[np.nan, 1.0]], dtype=np.float32)
        in_memory_vectors = np.array([[np.nan, 2.0]], dtype=np.float32)
        ids = ["doc-nan"]
        write_pool_shard(self.shard_path, disk_vectors, ids)

        with self.assertRaises(AssertionError) as raised:
            verify_pool_shard_parity(in_memory_vectors, ids, self.shard_path)

        diagnostic = str(raised.exception)
        self.assertIn("index=(0, 1)", diagnostic)
        self.assertNotIn("index=(0, 0) expected=np.float32(nan) actual=np.float32(nan)", diagnostic)

    def test_nan_diagnostic_distinguishes_different_nan_payload_bits(self):
        disk_vectors = np.array([[0x7FC00001]], dtype=np.uint32).view(np.float32)
        in_memory_vectors = np.array([[0x7FC00002]], dtype=np.uint32).view(np.float32)
        ids = ["doc-nan"]
        write_pool_shard(self.shard_path, disk_vectors, ids)

        with self.assertRaises(AssertionError) as raised:
            verify_pool_shard_parity(in_memory_vectors, ids, self.shard_path)

        diagnostic = str(raised.exception)
        self.assertIn("index=(0, 0)", diagnostic)
        self.assertIn(
            f"expected_bytes=0x{in_memory_vectors.tobytes(order='C').hex()}",
            diagnostic,
        )
        self.assertIn(
            f"actual_bytes=0x{disk_vectors.tobytes(order='C').hex()}",
            diagnostic,
        )

    def test_verify_raises_on_mutated_vector_byte(self):
        write_pool_shard(self.shard_path, FIXTURE_VECTORS, FIXTURE_IDS)
        mutated = FIXTURE_VECTORS.copy()
        mutated[0, 0] = np.float32(FIXTURE_VECTORS[0, 0] + 1.0)
        with self.assertRaisesRegex(AssertionError, "Pool-shard parity mismatch"):
            verify_pool_shard_parity(mutated, FIXTURE_IDS, self.shard_path)

    def test_verify_raises_on_corrupted_payload(self):
        write_pool_shard(self.shard_path, FIXTURE_VECTORS, FIXTURE_IDS)
        payload = bytearray(self.shard_path.read_bytes())
        encoded_byte = np.frombuffer(payload[:2], dtype=np.float16)[0]
        payload[:2] = np.float16(encoded_byte + 1).tobytes()
        self.shard_path.write_bytes(payload)

        with self.assertRaisesRegex(AssertionError, "Pool-shard parity mismatch"):
            verify_pool_shard_parity(FIXTURE_VECTORS, FIXTURE_IDS, self.shard_path)

    def test_verify_raises_on_corrupted_padding(self):
        manifest = write_pool_shard(self.shard_path, FIXTURE_VECTORS, FIXTURE_IDS)
        payload = bytearray(self.shard_path.read_bytes())
        first_padding_cell = manifest["raw_vector_bytes"]
        padding_offset = first_padding_cell * np.dtype(np.float16).itemsize
        payload[padding_offset : padding_offset + 2] = np.float16(1).tobytes()
        self.shard_path.write_bytes(payload)

        with self.assertRaisesRegex(ValueError, "Non-zero block padding"):
            verify_pool_shard_parity(FIXTURE_VECTORS, FIXTURE_IDS, self.shard_path)


if __name__ == "__main__":
    unittest.main()
