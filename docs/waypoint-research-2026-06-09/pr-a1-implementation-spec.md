# PR-A1 Implementation Spec — model-swap drift injector

Local-only. Grounded in confirmed APIs (2026-06-13). Build this the moment Track 0 (#267)
merges; branch `feat/drift-injector-model-swap` off updated main.

## Confirmed APIs
- `DimensionProjection(teacher_dim, student_dim, hidden_dim=None)` — `teacher_distillation.py:24`;
  has `.project_numpy(embeddings) -> np.ndarray` (`:60`). Linear-ish near-identity init module.
- `create_embedding_backend(model_name, logger=None) -> EmbeddingBackend` —
  `embedding_backend.py:304`; `.embed_raw(texts) -> np.ndarray` (`:45`). Local model path
  is `LocalEmbeddingBackend` (SentenceTransformer). Second model: `all-mpnet-base-v2` (768-dim).
- `DriftInjector` current state: `_load_records()` returns records with `id`, `vector`,
  `vector_name`, `payload`. With `store_full_text_payload=True`, `payload["text"]` holds doc text.
- Manifest already carries `injection_index`; validation precedes RNG consumption.

## Method to add: `inject_model_swap_drift`

```python
def inject_model_swap_drift(self, fraction, swap_model_name="all-mpnet-base-v2",
                            swap_backend=None, projection_seed=None) -> dict:
    """Re-embed a deterministic fraction of stored docs with a DIFFERENT encoder,
    project to the store's dim via a frozen seeded DimensionProjection, renormalize,
    upsert. Defeats the C2 frozen-original re-embed oracle: re-embedding the text with
    the ORIGINAL model no longer reproduces this (projected-other-model) vector."""
```

Implementation order (validation-before-RNG contract preserved):
1. `records = self._load_records()`; `store_dim = self._vector_size(records)`.
2. Validate: `_validate_fraction(fraction)`; require each affected record has
   `payload["text"]` (raise ValueError naming the first missing id — model-swap needs text).
   These raise before any RNG / embedding.
3. `affected = self._select_affected(records, fraction)` (consumes RNG once, as today).
4. Build/accept the swap backend: `backend = swap_backend or create_embedding_backend(swap_model_name)`.
   Read `swap_dim` from one embed of a probe text OR `backend` shape on first embed.
5. Build the FROZEN projection: `proj = DimensionProjection(swap_dim, store_dim)`; seed it
   deterministically from `projection_seed or self.seed` (set torch manual seed around its
   construction, OR re-init its weights from a seeded numpy RNG — must be reproducible and
   NOT trained). Record `projection_checksum = sha256(concatenated proj state_dict tensors)`.
6. For affected records: `new = proj.project_numpy(backend.embed_raw([text]))[0]`; L2-normalize;
   assign as the record vector (float32).
7. `self._upsert_records(affected)`; manifest via `_manifest(...)` plus model-swap fields:
   `swap_model`, `projection_seed`, `projection_checksum`, `swap_dim`, `store_dim`.
   Add `mode="model_swap"`, `angle_degrees=None`, `sigma=None`, `rotation_pairs=[]`.

Note `_manifest` increments `injection_index` — keep that single increment site.

## Tests (`test_drift_injector.py`, all must run WITHOUT mpnet via a stub backend)

Stub: `class _StubSwapBackend: def embed_raw(self, texts): return <deterministic seeded
vectors of swap_dim per text>` — deterministic function of text hash so re-embedding the
same text twice is identical (mirrors a real frozen encoder).

1. `test_model_swap_drift_is_seed_deterministic` — same seed + same stub → byte-identical
   post-drift vectors and manifests (two fresh engines).
2. `test_model_swap_affects_expected_fraction_leaves_rest_identical` — exact count, unaffected
   bit-identical.
3. **`test_model_swap_defeats_frozen_reembed_oracle` (THE load-bearing EVIDENCE test).**
   On a 50-doc in-memory corpus: capture pre-drift vector for an affected doc; apply
   model-swap; assert the post-drift vector differs from pre-drift by a measurable margin;
   then simulate C2 (re-embed the SAME text with the ORIGINAL stub/engine model) and assert
   that the C2 result is NOT equal to the post-drift (model-swap) vector — i.e. the frozen
   re-embed cannot reconstruct the drifted state. This is the oracle-breaker proof.
4. `test_model_swap_manifest_round_trips_and_has_projection_checksum` — JSON round-trip;
   `projection_checksum` is 64 hex chars; `swap_model`/`swap_dim`/`store_dim` present.
5. `test_model_swap_requires_doc_text` — record without `payload["text"]` raises ValueError
   before RNG (replay-preserving), consistent with the validate-before-RNG contract.
6. `test_model_swap_real_model_smoke` — `@unittest.skipUnless` mpnet is locally importable;
   real `all-mpnet-base-v2` path on ~5 docs; asserts shape store_dim and finite norms.

## EVIDENCE line for the PR
Run the real-model path here (3090, HF_HUB_OFFLINE if cached): measured NDCG drop on a
small SciFact slice after model-swap drift, AND the oracle-breaker numeric (C2-reembed
cosine-to-pre-drift vs model-swap cosine-to-pre-drift) showing C2 cannot invert it. If
mpnet is not cached and cannot be fetched, ship with the stub-backed oracle-breaker test as
floor evidence and disclose the real-model run as deferred to PR-A4 calibration (do not fake it).

## Scope discipline
No new adapter types, no new loss functions (reuses DimensionProjection + existing backend).
The actuator/supervision (anchor InfoNCE) and the P0/P1 mechanism fixes (use_quantization,
NDCG-drop trigger) belong to PR-A2, not here — A1 is drift generation + the oracle-breaker proof only.
