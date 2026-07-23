# Codex build — Rung 17: disk pool slice (block-graph read with host parity)

Implement Phase-II rung 17: one precomputed **retrieval pool shard** (a block of document vectors +
their ids) is encoded to disk via the computational-storage block-graph payload and read back with a
**host parity check**. CPU-only, no GPU. Deterministic. Branch is checked out
(`lattice/rung17-diskpool-20260714`).

## Read first (match real APIs — do not invent)
- `computational_storage_poc/block_graph.py` — `BlockRecord`, `create_block(matrix, next_block_offset)`,
  `build_graph_payload(layer_matrices)`, `read_block(flash_memory, offset)`, `run_block_graph(...)`.
  Understand exactly how a matrix is encoded into a payload and reconstructed by `read_block` (dtype,
  shape, offsets, checksums if any).
- `computational_storage_poc/` for existing tests/conventions (`test_computational_storage_*`), and
  the storage-track docs (`COMPUTATIONAL_STORAGE_*`, `docs/computational-storage-*`).

## Build
1. **Pool-shard format** on top of the block graph. A shard = (doc_vectors: float32 [n, d], doc_ids:
   list[str]). Encode the doc-vector matrix via `build_graph_payload` (or `create_block`), and carry
   the ids + shape + dtype + a SHA256 of the raw vector bytes in a small manifest (sidecar JSON or a
   header block). Write a `write_pool_shard(path, vectors, ids)` and a `read_pool_shard(path)` that
   reconstructs `(vectors, ids)` by reading the block(s) back through `read_block` / `run_block_graph`
   — NOT by just re-loading the raw npz (the point is the block-graph read path is exercised).
2. **Host parity check.** `verify_pool_shard_parity(in_memory_vectors, in_memory_ids, shard_path)`:
   recomputes SHA256 over the disk-read vector bytes and compares to the manifest hash AND to the
   in-memory hash; asserts ids match exactly; asserts `np.array_equal` (bit-identical) between
   in-memory and disk-read vectors. Returns a parity record (hashes, match booleans, n, d).
3. **Retrieval equivalence.** `retrieve_topk(query, vectors, ids, k)` (cosine or dot); prove the top-k
   ids from the **disk-read** shard are bit-identical to the top-k from the **in-memory** pool for a
   fixture query set. This is the "one end-to-end read of a pool shard with parity" the exit names.
4. **Fail-closed = a parity mismatch is a BUG, not a negative.** If hashes/ids/vectors differ, raise
   loudly with the diff; do not "handle" it.

## Deliverables
1. `computational_storage_poc/pool_shard.py` (or a clearly-named module) with write/read/verify/retrieve.
2. `test_pool_shard_parity.py` (plain unittest, offline): round-trip write→read reconstructs vectors+ids
   bit-identically; SHA256 manifest matches disk-read; top-k retrieval identical in-memory vs disk-read
   on a fixture; a corrupted-shard fixture makes `verify_pool_shard_parity` raise (parity catches
   corruption). Run it + the existing `test_computational_storage_poc` (no regression); paste results.
3. `docs/rung17-disk-pool-slice.md` — the shard format, the read path through `block_graph`, and the
   parity + retrieval-equivalence result for one concrete fixture shard (n, d, hash, top-k match).

Be brutally honest: the read path MUST go through the block-graph `read_block`/`run_block_graph`, not a
shortcut that re-loads the original array (that would make the parity trivially true and the block-graph
integration fake). If `block_graph.py`'s payload format cannot round-trip a float32 pool matrix exactly
(e.g. it quantizes or reshapes), say so with file:line and either use the lossless path it does support
or document the exact fidelity limit — do NOT claim bit-identical parity if the format is lossy.
