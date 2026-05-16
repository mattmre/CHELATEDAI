# Phase 7 Promotion Review

Date: 2026-03-29
Status: completed
Recommendation: promote the current system as a **research baseline**, defer any **production-ready** promotion

## Scope

This review covers the disk-first CPU / retrieval baseline implemented across:

- packed storage substrate
- CPU inference substrate
- sparse selective-loading runtime
- repo graph memory
- integrated repository-Q&A prototype
- repo-memory compression path

This review does **not** promote the system as a disk-resident LLM or as a production alternative to a GPU inference stack.

## Final Call

### Promote

Promote the current implementation as the active **research baseline** for ChelatedAI's disk-first CPU / retrieval architecture work.

### Defer

Defer any broader promotion as:

- a production-ready local inference system
- a true long-context generator
- a general replacement for GPU-backed model serving

## Why The Baseline Is Promotable

The full stack passed the intended Phase 7 checks:

- storage reduction and real disk-backed reads are in place
- CPU-only packed inference has a usable low-bit baseline
- sparse selective loading reduces streamed bytes materially
- repo-local graph memory works independently
- the integrated runtime is runnable end-to-end on CPU
- the int8 repo-memory compression path reduces mapped bytes materially without observed hit-rate loss on the current benchmark set

## Why Production Promotion Is Deferred

The strongest remaining reasons are:

1. The integrated runtime is still a retrieval-and-reranking prototype, not a generative system.
2. The reranker is deterministic and hand-authored, not trained.
3. Repo-memory quality is still benchmarked on a small internal query set.
4. The int8 compression path still shows optimization headroom in latency and heap behavior.
5. No MoE / REAP path has been integrated yet.
6. No KV-cache or long-context memory path exists yet because the runtime is not a long-context generator.
7. The architecture is not yet demonstrably simpler operationally than a GPU-backed alternative for general workloads.

## Benchmark Summary

### Storage

- packed read reduction: about `94.71%`

### CPU

- prequantized int8 vs float32 speedup: about `1.19x`

### Sparse Runtime

- streamed byte reduction: about `61.98%`

### Repo Memory

- top-3 hit rate: `75%`

### Integrated Runtime

- average total latency: about `3.33 ms`
- top-3 hit rate: `75%`

### Compression

- repo-memory mapped-byte reduction: about `75%`
- integrated-runtime mapped-byte reduction: about `74.71%`

## Recommended Presets

For the current baseline, prefer:

- packed artifact storage: `packed_int8`
- repo-memory embedding storage: `int8`
- sparse chunk rows: `4`
- sparse stream start block: `1`
- integrated runtime top-k: `3`
- benchmark ingest filter: exclude tests, benchmark files, docs, and cache directories from the repo-memory surface

## Next Recommended Branches

Primary:

- `Phase 3B`: MoE / REAP branch, if a concrete MoE target is selected

or

- targeted optimization passes on:
  - int8 repo-memory query path
  - integrated runtime ranking quality
  - larger and more realistic evaluation sets

Secondary:

- revisit long-context / KV-cache compression only after a generator-style runtime exists

## No-Go Conditions

Reset or defer the branch if future measurements show any of the following:

- integrated retrieval quality drops materially below the current top-3 baseline
- compression reduces mapped bytes but creates unacceptable latency or heap regressions
- future integration work collapses the current phase-scoped boundaries and mixes MoE, retrieval, and runtime changes into the same seam
