# Rung 17: Disk Pool Slice

## Scope

This Phase-II slice writes one precomputed retrieval-pool shard to a
computational-storage block-graph payload, reconstructs it through the
block-graph reader, and compares that disk-read result with the host's
in-memory vectors and document ids. It is deterministic, offline, CPU-only
POC code. It is not a production retrieval path or physical-device result.

## Shard format

A shard has two files:

- `<path>` is one or more fixed-size block-graph blocks produced by
  `build_graph_payload`.
- `<path>.manifest.json` is deterministic compact JSON containing the format
  name and version, byte encoding, `float32` dtype, `[n, d]` shape, ordered
  document ids, raw-vector byte count, block count, and SHA256 of the logical
  C-order float32 vector bytes.

The underlying block format is unchanged: each block is a 512 x 512 FP16
matrix followed by an eight-byte little-endian next-block offset. The final
offset is zero.

### Why the vectors are byte-lane encoded

Directly passing a float32 pool matrix to `create_block` is lossy.
`computational_storage_poc/block_graph.py:30-32` casts the matrix to FP16, and
`read_block` reconstructs that FP16 payload as float32 at
`computational_storage_poc/block_graph.py:58-63`. Only values exactly
representable in FP16 would survive that direct path bit-identically.

Rung 17 therefore uses the lossless path that the same block format supports:

1. Serialize the contiguous vector matrix as C-order float32 bytes.
2. Store each byte as a numeric FP16 cell with value 0 through 255. Every such
   integer is exactly representable in FP16.
3. Split at 512 x 512 cells and pass those matrices to
   `build_graph_payload`.
4. On read, convert the decoded, validated integer cells back to bytes, crop
   to the manifest byte count, and reshape as manifest-declared float32.

This trades space for fidelity: one raw byte occupies one two-byte FP16 cell,
and every block is padded to the fixed block size. One block carries up to
262,144 raw vector bytes (65,536 float32 values).

## Actual read and parity path

`read_pool_shard` reads the payload file as opaque flash bytes and traverses
its graph links using `read_block`. It validates the payload length, each
next-block offset, each decoded byte lane, the manifest byte count, shape, and
id count, and zero-valued block padding before reconstructing the matrix. It does not use NPZ, NPY, mmap of a
raw vector matrix, or any other array reload shortcut. `run_block_graph` is
not appropriate here because it executes matrix multiplication; `read_block`
is the block-graph API that reconstructs stored matrices.

`verify_pool_shard_parity` then computes the SHA256 of the disk-read logical
vector bytes and requires all of the following:

- manifest SHA256 equals disk-read SHA256;
- in-memory SHA256 equals disk-read SHA256;
- document ids match exactly and in order; and
- the C-order float32 vector bytes are exactly equal.

Any mismatch raises with hashes, match flags, and the first id/vector diff.
Structural payload corruption also raises. A parity failure is never converted
to a negative result record. The byte-exact comparison treats an identical NaN
payload as equal while still distinguishing different NaN payload bits; mismatch
diagnostics report both the array index and the differing element bytes.

`retrieve_topk` uses float32 dot-product scores and stable descending sorting.
Stable sorting makes tied scores preserve shard order deterministically.

## Concrete fixture evidence

The offline fixture in `test_pool_shard_parity.py` contains six documents with
four-dimensional float32 vectors, including values not exactly representable
in FP16, and three fixture queries.

| Field | Result |
|---|---|
| Shape | `n=6`, `d=4` |
| Raw vector bytes | `96` |
| Payload blocks | `1` |
| Manifest / disk / host SHA256 | `637bf79f46866a5458b80bc521bd6db2ae9cf543e7036ab950668e3fe8c80973` |
| Id equality | exact |
| Byte equality | exact |
| Query 1 top 3 | `doc-epsilon`, `doc-beta`, `doc-alpha` |
| Query 2 top 3 | `doc-zeta`, `doc-gamma`, `doc-delta` |
| Query 3 top 3 | `doc-zeta`, `doc-epsilon`, `doc-alpha` |
| All disk vs host top-k comparisons | exact |
| Corrupted logical payload byte | parity raises `AssertionError` |
| Corrupted padding byte lane | shard read raises `ValueError` |

Validation re-run on 2026-07-27 after review hardening:

```text
python -m unittest -v test_pool_shard_parity test_computational_storage_poc
Ran 16 tests in 8.345s
OK
```
