# Grok Tier B — rung 17 disk pool slice (commit e12b7668). Try to disprove it.

Official Tier B scorer for commit `e12b7668` on branch `lattice/rung17-diskpool-20260714` (cwd = repo
root). This adds a block-graph-carried retrieval pool shard with host parity. Try to DISPROVE that it
honestly delivers "one precomputed pool shard readable via block_graph.py with host parity" (rung-17
exit criteria) at BHS 100. Read `computational_storage_poc/pool_shard.py`,
`computational_storage_poc/block_graph.py`, `test_pool_shard_parity.py`, `docs/rung17-disk-pool-slice.md`.

## Attack — the two ways this could be fake or wrong
1. **Is the read a real block-graph traversal or a shortcut?** Confirm `read_pool_shard` reconstructs
   the vectors by calling `block_graph.read_block` over the payload and following `next_offset` links —
   NOT by re-loading the original array / an npz / the manifest bytes. If the "disk read" bypasses
   `read_block`, the block-graph integration is theater → FAIL.
2. **Is "bit-identical parity" real given block_graph quantizes to FP16?** The claim: each float32 byte
   (0..255) is stored as one FP16 cell, and integers ≤255 are exactly representable in FP16, so the
   round-trip is lossless. VERIFY this is actually true (float16 exactly represents 0..255) and that the
   decode path validates cells are finite integers in [0,255]. Construct a random float32 matrix
   yourself, round-trip it through write/read, and assert `np.array_equal` + SHA256 match. If ANY
   float32 bit pattern fails to round-trip, the "bit-identical" claim is false → FAIL.

## Also check
3. **Parity fail-closed:** does `verify_pool_shard_parity` actually raise on a corrupted payload / wrong
   ids / mutated vector, or can a mismatch slip through? Try to make it pass on non-matching data.
4. **Manifest integrity:** the manifest carries the SHA256 + shape + ids separately from the payload.
   Can a manifest/payload mismatch (e.g. edited manifest hash) pass verify? Is the manifest hash checked
   against the DISK-read bytes (not just echoed)?
5. **Honesty of the doc/commit:** is the FP16 limitation + 4x overhead disclosed, or is it sold as a
   native float32 store? Any overclaim that this is wired into live retrieval (it is EXPERIMENTAL/POC)?
6. **Regression:** run `python -m unittest test_pool_shard_parity` and `python test_computational_storage_poc.py`.

Deliver `BHS_TIER_B: <0-100>`, `BHS_TIER_B_SEVERITY`, PASS/PASS-WITH-FIXES/FAIL, a defect table
(severity, file:line, fix), and a one-line bottom line: is this a genuine parity-checked block-graph
pool-shard read, or does the read bypass the format / is the bit-exact claim false? Run the round-trip
yourself. Fresh-agent: you = grok-4.5, implementer = codex/Fable — independent.
