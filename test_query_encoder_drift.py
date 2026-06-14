from __future__ import annotations

import hashlib
import json
import unittest

import numpy as np

from query_encoder_drift import QueryEncoderDrift

STORE_DIM = 384
SWAP_DIM = 96


def _deterministic_unit_vector(text: str, dim: int) -> np.ndarray:
    """Stable per-text pseudo-random unit vector, mimicking a frozen second encoder."""
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    seed = int.from_bytes(digest[:8], "big")
    rng = np.random.default_rng(seed)
    vec = rng.standard_normal(dim).astype(np.float32)
    norm = float(np.linalg.norm(vec))
    return (vec / norm).astype(np.float32) if norm > 0 else vec


class _StubSwapBackend:
    """Deterministic 'different encoder': text -> stable SWAP_DIM unit vector."""

    def __init__(self, dim: int = SWAP_DIM):
        self.dim = dim

    def embed_raw(self, texts):
        return np.vstack([_deterministic_unit_vector(t, self.dim) for t in texts]).astype(np.float32)


class _StubOriginalBackend:
    """The ORIGINAL frozen encoder: doc/query 'item-{d}' -> one-hot[d] in STORE_DIM.

    Re-embedding the same text always yields the same vector, so this is what
    makes condition C2 (re-embed-with-original) a genuine, observable code path
    rather than a copy of the cached store.
    """

    def __init__(self, dim: int = STORE_DIM):
        self.dim = dim

    def embed_raw(self, texts):
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            idx = int(text.split("-")[-1])
            out[row, idx % self.dim] = 1.0
        return out


def _original_doc_vectors(doc_count: int, backend=None):
    """Original-encoder doc store built by actually calling the original encoder."""
    backend = backend or _StubOriginalBackend()
    embedded = backend.embed_raw([f"item-{d}" for d in range(doc_count)])
    return {d: embedded[d].astype(np.float32) for d in range(doc_count)}


def _retrieval_accuracy(query_vectors, doc_vectors):
    """Fraction of queries whose nearest doc (cosine) is the matching doc id."""
    doc_ids = sorted(doc_vectors.keys())
    doc_mat = np.vstack([doc_vectors[i] for i in doc_ids]).astype(np.float32)
    doc_norms = np.linalg.norm(doc_mat, axis=1, keepdims=True)
    doc_norms = np.where(doc_norms > 0, doc_norms, 1.0)
    doc_unit = doc_mat / doc_norms
    correct = 0
    for d, qv in query_vectors.items():
        qn = float(np.linalg.norm(qv))
        qu = qv / qn if qn > 0 else qv
        sims = doc_unit @ qu
        if doc_ids[int(np.argmax(sims))] == d:
            correct += 1
    return correct / len(query_vectors)


class TestQueryEncoderDrift(unittest.TestCase):
    def test_drifted_queries_are_seed_deterministic(self):
        texts = [f"query text {i}" for i in range(8)]
        a = QueryEncoderDrift(STORE_DIM, seed=13, swap_backend=_StubSwapBackend())
        b = QueryEncoderDrift(STORE_DIM, seed=13, swap_backend=_StubSwapBackend())
        emb_a = a.embed_queries(texts)
        emb_b = b.embed_queries(texts)
        np.testing.assert_array_equal(emb_a, emb_b)
        self.assertEqual(a.manifest(), b.manifest())

    def test_output_dim_and_normalized(self):
        drift = QueryEncoderDrift(STORE_DIM, seed=1, swap_backend=_StubSwapBackend())
        emb = drift.embed_queries([f"q{i}" for i in range(5)])
        self.assertEqual(emb.shape, (5, STORE_DIM))
        norms = np.linalg.norm(emb, axis=1)
        np.testing.assert_allclose(norms, np.ones(5), atol=1e-5)

    def test_query_encoder_drift_defeats_frozen_reembed_oracle(self):
        """The load-bearing oracle-breaker proof.

        Baseline (queries+docs in original space) retrieves perfectly. After a
        query-encoder upgrade, retrieval degrades. C2 (re-embed docs with the
        ORIGINAL model -> docs unchanged) does NOT recover. Only re-embedding the
        docs into the NEW query space (the expensive oracle) recovers -- which is
        exactly why C2 is not a fair baseline here.
        """
        doc_count = 40
        original_backend = _StubOriginalBackend()
        original_docs = _original_doc_vectors(doc_count, backend=original_backend)
        # Baseline: query d == doc d (one-hot). Perfect alignment.
        baseline_queries = {d: original_docs[d].copy() for d in range(doc_count)}
        baseline_acc = _retrieval_accuracy(baseline_queries, original_docs)

        # Query-encoder upgrade: queries now come from the swapped encoder. Doc
        # text == query text per id, so the swap embedding is shared by id.
        drift = QueryEncoderDrift(STORE_DIM, seed=7, swap_backend=_StubSwapBackend())
        drifted_q_mat = drift.embed_queries([f"item-{d}" for d in range(doc_count)])
        drifted_queries = {d: drifted_q_mat[d] for d in range(doc_count)}
        drifted_acc = _retrieval_accuracy(drifted_queries, original_docs)

        # C2 maintenance: actually RE-EMBED the doc text with the ORIGINAL encoder
        # (a real backend call, not a copy of the cached store). We first OBSERVE
        # that this reproduces the cached vectors exactly -- i.e. C2 is genuinely a
        # no-op for query-side drift, demonstrated rather than assumed -- then
        # measure that retrieval stays degraded.
        c2_reembed = original_backend.embed_raw([f"item-{d}" for d in range(doc_count)])
        c2_docs = {d: c2_reembed[d].astype(np.float32) for d in range(doc_count)}
        for d in range(doc_count):
            np.testing.assert_array_equal(c2_docs[d], original_docs[d])  # C2 no-op is observed
        c2_acc = _retrieval_accuracy(drifted_queries, c2_docs)

        # Expensive oracle: re-embed docs with the NEW encoder+projection (same
        # text per id -> same drifted vector). Now docs align with drifted queries.
        oracle_drift = QueryEncoderDrift(STORE_DIM, seed=7, swap_backend=_StubSwapBackend())
        oracle_doc_mat = oracle_drift.embed_queries([f"item-{d}" for d in range(doc_count)])
        oracle_docs = {d: oracle_doc_mat[d] for d in range(doc_count)}
        oracle_acc = _retrieval_accuracy(drifted_queries, oracle_docs)

        # Baseline retrieves perfectly; the upgrade degrades it materially.
        self.assertEqual(baseline_acc, 1.0)
        self.assertLess(drifted_acc, 0.5)
        # The oracle-breaker: C2 re-embed-with-original is a demonstrated no-op
        # (asserted above) and therefore does NOT recover -- retrieval stays low.
        self.assertLess(c2_acc, 0.5)
        # ...while re-embedding into the NEW space fully recovers, proving recovery
        # is possible and that C2's failure is structural (a no-op), not noise.
        self.assertEqual(oracle_acc, 1.0)
        self.assertGreater(oracle_acc, c2_acc)  # recovery exists but C2 does not achieve it

    def test_manifest_round_trips_and_has_checksum(self):
        drift = QueryEncoderDrift(STORE_DIM, seed=5, swap_backend=_StubSwapBackend())
        drift.embed_queries(["a", "b"])
        manifest = drift.manifest()
        decoded = json.loads(json.dumps(manifest, sort_keys=True))
        self.assertEqual(decoded, manifest)
        self.assertEqual(manifest["drift"], "query_encoder_swap")
        self.assertEqual(len(manifest["projection_checksum"]), 64)
        self.assertEqual(manifest["swap_dim"], SWAP_DIM)
        self.assertEqual(manifest["store_dim"], STORE_DIM)

    def test_manifest_before_embed_raises(self):
        drift = QueryEncoderDrift(STORE_DIM, seed=5, swap_backend=_StubSwapBackend())
        with self.assertRaises(ValueError):
            drift.manifest()

    def test_empty_texts_raises(self):
        drift = QueryEncoderDrift(STORE_DIM, seed=5, swap_backend=_StubSwapBackend())
        with self.assertRaises(ValueError):
            drift.embed_queries([])


@unittest.skipUnless(
    __import__("importlib").util.find_spec("sentence_transformers") is not None,
    "sentence-transformers not installed",
)
class TestQueryEncoderDriftRealModelSmoke(unittest.TestCase):
    def test_real_swap_model_path_produces_store_dim_unit_vectors(self):
        try:
            drift = QueryEncoderDrift(STORE_DIM, swap_model_name="all-mpnet-base-v2", seed=3)
            emb = drift.embed_queries(["a scientific claim about cells", "another query"])
        except Exception as exc:  # model not cached / offline fetch blocked
            self.skipTest(f"swap model unavailable: {type(exc).__name__}: {exc}")
        self.assertEqual(emb.shape, (2, STORE_DIM))
        np.testing.assert_allclose(np.linalg.norm(emb, axis=1), np.ones(2), atol=1e-4)
        self.assertEqual(drift.manifest()["swap_dim"], 768)


if __name__ == "__main__":
    unittest.main()
