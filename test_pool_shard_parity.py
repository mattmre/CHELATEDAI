import hashlib
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

POC_DIR = os.path.join(os.path.dirname(__file__), "computational_storage_poc")
if POC_DIR not in sys.path:
    sys.path.insert(0, POC_DIR)

from pool_shard import (  # noqa: E402
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
