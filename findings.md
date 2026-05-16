# Findings & Decisions

## 2026-03-28 Disk-Resident LLM Feasibility Findings
- The paper referenced in the user request was not attached in-session. The closest primary-source fit to the request is `LLM in a Flash`, so the current feasibility pass uses that as the working assumption.
- ChelatedAI's current computational-storage path is still a transport and replay proof, not a transformer runtime.
- `computational_storage_poc/block_graph.py` uses fixed `512 x 512` FP16 dense blocks with zero padding, which is far too inefficient for realistic transformer storage.
- `computational_storage_poc/mock_nvme.py` preloads the full binary into RAM, so the current "NVMe" path does not measure real SSD behavior.
- A credible near-term architecture is SSD-resident compressed weights plus CPU execution, not end-to-end transformer compute inside a commodity SSD controller.
- `LLM in a Flash` is the right architectural template for the storage side:
  - keep attention weights resident
  - stream only a small active FFN slice
  - prefer larger contiguous reads and reuse windows
- `T-MAC` is the right architectural template for the CPU side:
  - execute low-bit kernels directly
  - avoid dequantize-then-matmul overhead
- The new estimator at `computational_storage_poc/disk_llm_estimator.py` shows:
  - dense full-model streaming from SSD is not viable
  - sparse flash-style streaming makes 7B-70B plausible
  - 405B is only marginal even on workstation-class hardware
- Highest-value repo improvements are:
  - manifest-driven quantized packing
  - real disk-backed reads
  - sparse FFN predictor and cache
  - CPU low-bit kernel path
  - transformer microbenchmark acceptance test

## 2026-03-28 Addendum Findings: REAP / TurboQuant / CPU-Disk Systems
- `REAP` is directly relevant for disk-first deployment of coding MoE models because it reduces the stored expert bank without destroying router behavior.
- `TurboQuant` is not a weight-footprint method; it is primarily a KV-cache and vector-search compression method.
- The strongest published CPU-only inference evidence found in primary sources is currently:
  - `T-MAC` for low-bit CPU inference
  - `bitnet.cpp` / `1-bit AI Infra` for native 1-bit CPU inference
  - `Gemma.cpp` as an official CPU runtime surface from Google
- The strongest published evidence for external memory replacing parametric scale is:
  - `kNN-LM`
  - `RETRO`
  - graph-structured retrieval for code generation such as PKG / GraphSkill
- The practical architecture answer is not "SSD instead of GPU" in isolation; it is "small CPU-native core + disk-backed memory and retrieval + optional disk-resident compressed weights."

## 2026-03-28 Revised Roadmap Findings
- The old roadmap framing was too passive: it assumed the main remaining work was evaluation, not a new architecture program.
- The roadmap now needs to be treated as a multi-phase program with explicit dependency ordering:
  - storage substrate
  - CPU inference substrate
  - compression branches
  - retrieval / graph memory
  - runtime integration
  - long-context compression
  - end-to-end promotion review
- ARCH-AEP needs a new mandatory loop for architecture-led work:
  - scope lock
  - implementation
  - ARCH-AEP review
  - code analysis / hardening
  - promote / defer
- The safest way to minimize scope drift is to prohibit cross-phase PRs and require explicit phase non-goals.
- Retrieval / graph memory should be developed as a first-class phase, not mixed into low-level storage or CPU kernel phases.

## 2026-03-29 Phase 1 Storage Substrate Findings
- The first safe implementation slice for `Phase 1` is compatible with the new roadmap:
  - add a manifest-driven packed artifact path
  - switch the mock NVMe path to real file-backed access
  - add a benchmark surface for size/read reduction
- `packed_graph.py` now demonstrates the right substrate direction:
  - explicit manifest
  - exact matrix-shape storage
  - no mandatory `512 x 512` zero padding
  - `mmap`-backed reads
- The legacy padded path remains intact, which keeps the phase bounded and avoids premature runtime integration.
- Moving `MockNVMeDrive` from eager file reads to `mmap` surfaced an important Windows-specific lifecycle concern: tests and call sites must explicitly close the mapping before tempdir cleanup.
- The new storage benchmark showed the intended direction clearly on the current test graph:
  - artifact size reduction: about `94.69%`
  - read-byte reduction: about `94.71%`
  - parity maintained (`max_abs_diff = 0.0`)
- The packed substrate is now wired into the standard compile helpers:
  - `compiler.py --format packed`
  - `train_and_compile.compile_model(..., artifact_format="packed")`
- This phase should stop here for now. The next phase should build on the new packed substrate rather than mixing in CPU kernel work immediately.

## 2026-03-29 Phase 2 CPU Inference Substrate Findings
- The correct next bounded slice after the packed storage substrate is a CPU execution seam that consumes packed artifacts without introducing sparse routing, MoE logic, or retrieval memory yet.
- `cpu_backends.py` now provides a minimal backend abstraction with two baseline implementations:
  - `NumpyFloat32Backend` as the numerical reference path
  - `NumpyInt8DynamicBackend` as a correctness-oriented low-bit baseline
- `packed_cpu_inference.py` proves that packed disk-backed artifacts can be executed through the CPU backend seam without falling back to the legacy padded graph path.
- The current int8 implementation is numerically close enough to serve as a Phase 2 substrate:
  - benchmark `max_abs_diff` is about `0.002349`
  - unit tolerance remained within the current test thresholds
- The current int8 implementation is not yet a performance win:
  - float32 latency was about `0.1624 ms`
  - int8 latency was about `0.3110 ms`
  - reported speedup was about `0.52x`
- That performance result is expected for this implementation shape because it dynamically quantizes both activations and weights on every call and still uses generic NumPy matmul kernels rather than packed low-bit kernels.
- ARCH-AEP promotion decision for this phase:
  - promote as a correctness and interface baseline
  - do not promote as a CPU-performance claim
- The clean dependency-preserving next steps are:
  - `Phase 2b`: packed quantized artifacts plus reusable scales and lower-overhead CPU kernels
  - `Phase 3A`: sparse/dense FFN selective loading on top of the packed storage and CPU seam

## 2026-03-29 Phase 2b Prequantized Artifact Findings
- The cleanest way to improve the initial CPU substrate was to keep the same packed container and add a second packed storage mode rather than inventing a separate artifact family.
- `packed_graph.py` now supports:
  - FP16 packed weights
  - INT8 packed weights with per-block scales
- `packed_cpu_inference.py` now exploits that storage mode by letting the int8 backend consume prequantized weights directly instead of requantizing them on each call.
- `compiler.py` and `train_and_compile.py` now expose `packed_int8` as a first-class artifact format, which keeps the compile surface aligned with the roadmap.
- Validation results for the current microbenchmark are good enough to promote this slice as the first low-bit performance baseline:
  - float32 latency: about `0.1818 ms`
  - dynamic int8 latency: about `0.5022 ms`
  - prequantized int8 latency: about `0.1524 ms`
  - prequantized int8 speedup vs dynamic int8: about `3.30x`
  - prequantized int8 speedup vs float32: about `1.19x`
  - prequantized int8 read bytes: `41600` vs `83200` for the FP16-packed path
  - prequantized int8 max absolute diff: about `0.002579`
- ARCH-AEP promotion decision for this phase:
  - promote the prequantized INT8 path as the current CPU baseline for packed artifacts
  - still treat it as a prototype kernel path, not a final answer for large-model CPU inference
- Remaining technical constraints:
  - activations are still dynamically quantized per call
  - execution still relies on generic NumPy integer matmul rather than packed low-bit kernels
  - no sparse FFN loading, MoE routing, or retrieval memory is included yet
- The next bounded choices remain:
  - `Phase 2c`: deeper CPU-kernel and activation-path optimization
  - `Phase 3A`: selective FFN loading / sparse runtime behavior

## 2026-03-29 Phase 3A Dense / Sparse FFN Findings
- The most bounded way to implement `Phase 3A` in the current POC is to keep the packed artifact format stable and add selective row-chunk reads for streamed blocks.
- `packed_graph.py` now exposes row-chunk readers for both:
  - dequantized float storage
  - prequantized INT8 storage
- `sparse_cpu_inference.py` implements the first selective-loading runtime:
  - resident-vs-streamed split via `stream_from_block`
  - activation-driven routing heuristic using nonzero rows
  - `SparseChunkCache` for chunk reuse across repeated calls
- The current runtime preserves exact linear equivalence by summing only the row chunks whose activations are nonzero, rather than approximating the output.
- The benchmark harness was tuned so the streamed FFN block materially dominates the byte budget; otherwise the always-resident first block hides the gain signal.
- Current Phase 3A benchmark results:
  - avg dense latency: about `0.2478 ms`
  - avg sparse latency: about `0.2253 ms`
  - dense bytes per token: `98304`
  - sparse bytes per token: `37376`
  - streamed byte reduction: about `61.98%`
  - cache hits: `28`
  - cache misses: `36`
  - max absolute diff: about `0.000806`
- ARCH-AEP promotion decision for this phase:
  - promote the selective-loading runtime as the baseline dense/sparse FFN prototype
  - keep the scope explicitly limited to row-chunk streaming on the current toy packed-graph path
- Remaining constraints:
  - routing is still a simple activation sparsity heuristic, not a learned predictor
  - the cache is a small in-memory prototype, not yet a full runtime policy
  - the benchmark is still a layer-execution harness, not transformer token generation
- The clean next bounded choices are now:
  - `Phase 3B`: MoE / REAP-compatible storage and execution
  - `Phase 4`: retrieval and graph memory substrate

## 2026-03-29 Phase 4 Retrieval / Graph Memory Findings
- The cleanest `Phase 4` slice is a standalone repo-memory substrate that does not couple to the inference runtime yet.
- `repo_graph_memory.py` now provides:
  - a disk-backed node / edge / embedding bundle
  - file and symbol nodes
  - local-import and containment edges
  - a memory-mapped embedding surface
  - a query API that blends vector, lexical, and graph signals
- The current embedding approach is intentionally lightweight:
  - stable hashed token embeddings
  - no dependency on external embedding services
  - good enough for local architecture validation
- `repo_graph_memory_benchmark.py` measures this layer independently of the model runtime and currently reports:
  - node count: `111`
  - edge count: `220`
  - ingest latency: about `93.65 ms`
  - average query latency: about `0.4328 ms`
  - top-1 hit rate: `75%`
  - top-3 hit rate: `75%`
- An ingestion filter was necessary to keep the benchmark honest:
  - benchmark files, tests, docs, and cache directories can otherwise dominate repo-local code retrieval with artificial lexical matches
- ARCH-AEP promotion decision for this phase:
  - promote the repo-memory layer as the current disk-backed code-memory substrate
  - keep it decoupled from execution until `Phase 5`
- Remaining constraints:
  - embedding quality is still hash-based rather than model-based
  - graph edges are limited to containment and local-import relationships
  - reranking is heuristic, not learned
- The clean next bounded choices are now:
  - `Phase 5`: integrate storage, CPU, sparse runtime, and repo memory into one runnable prototype path
  - `Phase 3B`: keep MoE / REAP as a separate parallel branch if a concrete MoE target is chosen

## 2026-03-29 Phase 5 Runtime Integration Findings
- The cleanest `Phase 5` slice is a single CPU-only repository Q&A style prototype rather than a fake token generator.
- `integrated_repo_runtime.py` now integrates:
  - the disk-backed repo-memory query surface
  - retrieval-result featureization
  - a packed INT8 reranker artifact
  - the sparse CPU execution path
- The current end-to-end runtime uses a deterministic reranker graph rather than a trained language model, which keeps the phase honest:
  - it proves orchestration and metric capture
  - it does not overclaim generative capability
- `integrated_runtime_benchmark.py` now measures the integrated path and currently reports:
  - average retrieval latency: about `1.3089 ms`
  - average inference latency: about `1.9976 ms`
  - average total latency: about `3.3251 ms`
  - queries per second: about `28`
  - average bytes read per query: `1152`
  - mapped bytes: `121313`
  - peak Python heap: about `55.91 KB`
  - top-1 hit rate: `50%`
  - top-3 hit rate: `75%`
- A code-aware tokenization fix was necessary in the repo-memory substrate so snake_case and CamelCase repository identifiers become searchable in a repo-local way.
- ARCH-AEP promotion decision for this phase:
  - promote the integrated repository-Q&A prototype as the current end-to-end local baseline
  - keep the claim boundary narrow: this is orchestration and retrieval-aware reranking, not a full disk-resident LLM
- Remaining constraints:
  - the reranker is hand-authored, not trained
  - the runtime still targets repository retrieval/reranking rather than free-form code generation
  - no KV-cache or memory compression exists yet
  - MoE / REAP remains intentionally separate
- The clean next bounded choices are now:
  - `Phase 6`: long-context and memory compression on top of the integrated baseline
  - `Phase 3B`: MoE / REAP path as a parallel branch if a concrete MoE target is selected

## 2026-03-29 Phase 6 Memory Compression Findings
- The cleanest `Phase 6` slice on the current baseline is compressed repo-memory embeddings rather than KV-cache work, because the runtime is still a retrieval/reranking prototype rather than a long-context generator.
- `repo_graph_memory.py` now supports:
  - float32 embedding storage
  - int8-compressed embedding storage with a saved scale
- The compression path was threaded through both the standalone repo-memory benchmark and the integrated runtime so memory savings and quality impact can be measured at two levels.
- Current repo-memory compression benchmark results:
  - float32 mapped bytes: `122880`
  - int8 mapped bytes: `30720`
  - mapped-byte reduction: about `75%`
  - float32 top-1 / top-3 hit rate: `50%` / `75%`
  - int8 top-1 / top-3 hit rate: `50%` / `75%`
  - int8 average query latency is slightly higher in the current path
- Current integrated-runtime compression benchmark results:
  - float32 mapped bytes: `123361`
  - int8 mapped bytes: `31201`
  - mapped-byte reduction: about `74.71%`
  - float32 average total latency: about `3.8535 ms`
  - int8 average total latency: about `4.0467 ms`
  - float32 top-1 / top-3 hit rate: `50%` / `75%`
  - int8 top-1 / top-3 hit rate: `50%` / `75%`
  - int8 peak Python heap is higher in the current implementation because query-time quantization introduces temporary allocations
- ARCH-AEP promotion decision for this phase:
  - promote the int8 repo-memory path as the current compression experiment baseline
  - do not claim it is fully optimized yet, because the latency and heap tradeoff still need review
- Remaining constraints:
  - compression currently targets the repo-memory embeddings only
  - no KV-cache compression exists because the runtime is not a long-context generator yet
  - query-time quantization overhead still leaves optimization headroom
- The clean next bounded choices are now:
  - `Phase 7`: end-to-end evaluation and promotion review
  - `Phase 3B`: MoE / REAP as a separate parallel branch

## 2026-03-29 Phase 7 Evaluation And Promotion Findings
- `phase7_system_evaluation.py` now aggregates the benchmark pack across storage, CPU, sparse runtime, repo memory, integrated runtime, and compression.
- The current benchmark thresholds all pass for the intended research-baseline scope:
  - storage reduction
  - CPU baseline
  - sparse runtime
  - repo memory
  - integrated runtime
  - standalone compression
  - integrated compression
- The current promotion call is:
  - promote as `research baseline`
  - do not promote as `production-ready`
- The strongest reason for the defer remains architectural scope, not a failing benchmark:
  - the system is still retrieval-and-reranking oriented
  - it is not a full generative runtime
  - it is not yet clearly simpler operationally than a GPU-backed alternative
- The promotion memo at `docs/phase7-promotion-review-2026-03-29.md` now records:
  - promote/defer decision
  - recommended presets
  - no-go conditions
  - next-branch recommendations
- The clean next bounded choices after the closed review are:
  - `Phase 3B`: MoE / REAP branch if a concrete target model exists
  - targeted optimization and evaluation expansion under the current research baseline

## 2026-03-29 Phase 3B MoE / REAP Branch Findings
- The cleanest post-review parallel branch was to keep MoE / REAP isolated from the integrated baseline and land it as a separate artifact/runtime seam.
- `moe_reap.py` now provides:
  - a disk-backed MoE artifact format
  - expert-bank metadata with preserved router ids
  - routed-expert CPU execution
  - a REAP-like pruning compatibility function based on expert weight norms
- The current benchmark proves the intended branch property:
  - full artifact bytes: `2025`
  - pruned artifact bytes: `1122`
  - artifact reduction: about `44.59%`
  - full bytes read: `704`
  - pruned bytes read: `384`
  - read reduction: about `45.45%`
  - full experts evaluated: `4`
  - pruned experts evaluated: `2`
- The branch is intentionally not integrated into the Phase 5 prototype yet.
- ARCH-AEP promotion decision for this branch:
  - promote as a valid parallel MoE / REAP compatibility branch
  - defer runtime integration until a concrete MoE target model is selected

## 2026-03-29 Targeted Optimization And Evaluation Expansion Findings
- The cleanest post-Phase-7 follow-up was a bounded optimization and benchmark-hardening pass on the existing research baseline, not more architecture churn.
- `repo_graph_memory.py` now scores int8-compressed embeddings in row chunks instead of materializing an `int32` copy of the full memory-mapped matrix for every query.
- `repo_graph_memory.py` also now:
  - favors precise path matches over helper-path spillover
  - returns unique paths instead of repeated file/symbol duplicates for the same path
- That change materially improved the compressed integrated-runtime heap profile:
  - previous int8 peak Python heap: about `124.55 KB`
  - current int8 peak Python heap: about `58.45 KB`
- `integrated_repo_runtime.py` now uses retrieval-dominant fusion for final ranking, so the packed reranker refines candidate order instead of overwhelming stronger retrieval evidence.
- `cpu_inference_benchmark.py` now uses warmup plus median-of-trials timing so the Phase 7 promotion call is not driven by one noisy microbenchmark sample.
- `retrieval_eval_suite.py` now defines a shared 10-query repo-local benchmark suite covering:
  - packed storage
  - sparse loading
  - CPU backends
  - training / compile flow
  - repo memory
  - integrated runtime
  - MoE / REAP
  - payload transport
  - emulation
  - disk sizing / feasibility estimation
- The standalone repo-memory benchmark now reports on that wider suite:
  - query count: `10`
  - top-1 hit rate: `100%`
  - top-3 hit rate: `100%`
- The integrated runtime benchmark now reports on the same wider suite:
  - query count: `10`
  - average total latency: about `3.13 ms`
  - top-1 hit rate: `100%`
  - top-3 hit rate: `100%`
- The int8 compression tradeoff is now clearer after the optimization pass:
  - mapped-byte reduction is still about `75%`
  - quality stayed unchanged on the shared suite
  - standalone int8 query latency is now slightly better than float32 on the shared suite
  - integrated int8 total latency remains higher than float32, so compression is still a memory win first and an end-to-end latency win second
- ARCH-AEP decision for this pass:
  - promote as a bounded optimization and evaluation-hardening pass
  - keep the recommendation unchanged: the stack remains a promoted research baseline, not a production-ready system

## Requirements
- Proceed on the active computational-storage follow-up items overnight.
- Use implementation-session style orchestration with research, architecture, implementation, and validation phases.
- Keep track of session log and agent work.
- Create a PR for each reviewable item so the user can review via PR workflow tomorrow.
- Do not jeopardize work running in other repos.

## Research Findings
- Latest active handoff is the computational-storage post-merge follow-up from Session 25.
- The active queue is limited to four items: hardware evidence, emulator-path CI decision, transport-path scope decision, and retention policy review.
- The old Session 21/22 “top 15” is historical and already completed.
- `.github/workflows/test.yml` currently runs global `unittest` plus a dedicated computational-storage fundamentals job, but neither job provides a distinct emulator-path gate.
- `.github/workflows/build_firmware.yml` separately validates RP2040 firmware compilation and artifact generation.
- `test_computational_storage_payload.py` validates the deterministic payload contract, virtual-disk trigger-sector injection, and host-reader decoding from a file path.
- `computational_storage_poc/usb_host_inference.py` is the host-side raw-sector reader for physical or file-backed devices.
- A local hardware probe for RP2040 / Pico / TinyUSB devices returned no present device, so authentic real-hardware evidence cannot be captured tonight unless a device appears later.
- `computational_storage_poc/emulation/fuse_block_emulator.py` already contains emulator semantics, but importing it in CI currently requires `fusepy`; the core read behavior can be extracted into a dependency-light module and tested directly.
- `computational_storage_poc/emulation/docker-compose.yml` requires privileged FUSE (`/dev/fuse`), which is a poor fit for a stable default GitHub Actions gate.
- The existing docs already state that firmware scope is transport correctness, not on-device digits inference, but that boundary can be made more explicit and centralized.
- Backup refs currently present locally include `backup/retired-*` refs plus older local backup refs; remote backup refs from February are also still present.
- The hardware-evidence prep implementation can be validated tonight with file-backed trigger-sector images, which gives strong confidence in tomorrow’s physical capture workflow without claiming hardware success.
- The emulator-CI implementation can stay branch-independent from the hardware-evidence branch by avoiding the optional host-reader verbosity refinement.
- A dedicated emulator job can remain lightweight because it only needs `numpy`, the virtual controller, and the host-reader path.
- The scope decision is best enforced through one canonical doc referenced by the POC and firmware docs, rather than duplicated prose in multiple places.
- The safest retention decision tonight is a timed manual review, not deletion, because the computational-storage follow-up is still active and the March rollback refs are fresh.
- The planning-with-files catchup helper path in the installed skill is stale on this machine (`.claude` path missing), so session recovery has to be done from repo files directly.
- PR `#90` had a real correctness gap during review: `resolve_drive_path()` on Windows rewrote already-formed device paths like `\\.\PhysicalDrive2` into malformed paths. The branch now preserves explicit device paths and has a regression test.
- Local hardware inspection on 2026-03-06 still shows no RP2040 / Raspberry Pi Pico / TinyUSB mass-storage device. The only removable USB disk currently visible is a SanDisk drive, so physical evidence remains blocked.
- PR `#94` should not be merged until after the implementation PRs land or it is refreshed, because its handoff state would otherwise become stale as soon as `#90`-`#93` merge.

## Technical Decisions
| Decision | Rationale |
|----------|-----------|
| Separate active roadmap from historical top-15 work | Prevents wasting time reopening already completed items |
| Use documentation artifacts to preserve “fresh agent” outputs | Maintains auditable orchestration in a single-session environment |
| Treat hardware evidence as blocked by actual device availability | Prevents fabricated or software-only claims being mislabeled as physical validation |
| Slice the work into four item PRs plus one wrap PR | Minimizes conflicts while still giving the user reviewable units tomorrow |
| Prefer a reusable evidence-capture tool over a one-off manual checklist | Lets hardware evidence be captured immediately once a device is available |
| Extend `usb_host_inference.py` with a path resolver and optional verbosity control | Keeps existing CLI behavior while making automation and tests cleaner |
| Keep the emulator-CI branch independent of the host-reader refinement | Avoids hidden coupling between review branches |
| Express the current transport boundary through a canonical decision doc | Makes future scope promotion auditable and reduces overstatement risk |
| Use a tiered retention policy with a dated manual review window | Protects rollback paths now while preventing indefinite artifact sprawl |
| Preserve explicit Windows device paths in `resolve_drive_path()` | Matches the documented CLI contract for hardware evidence capture and avoids mangling valid raw-device arguments |
| Treat current hardware evidence as blocked despite a visible USB disk | The visible removable storage is a SanDisk drive, not an RP2040/Pico-class target |
| Merge the session-wrap PR last | Keeps `next-session.md`, `tracker-pointer.md`, and `CLAUDE.md` aligned with post-merge reality |
| Refresh the existing wrap PR instead of opening a sixth PR | Preserves the existing review thread while replacing the stale pre-merge handoff with the accurate Session 27 state |

## Issues Encountered
| Issue | Resolution |
|-------|------------|
| No evidence yet that physical RP2040 hardware is available | Verify connected devices before attempting hardware evidence capture |
| `computational_storage_poc/fuse_fs.py` does not exist at the probed path | Inspect actual POC file layout before planning emulator-path CI changes |
| `resolve_drive_path()` mis-handled explicit Windows device paths in PR `#90` | Fix on branch and add regression coverage before merge |
| `ruff check` was mistakenly run against `.github/workflows/test.yml` | Restrict lint to Python files; use other tooling for YAML if needed |

## Resources
- `CLAUDE.md`
- `docs/ARCH AGENTIC ENGINEERING AND PLANNING/next-session.md`
- `docs/ARCH AGENTIC ENGINEERING AND PLANNING/session-log-2026-03-06-session25.md`
- `computational_storage_poc/README.md`
- `computational_storage_poc/firmware/README_FIRMWARE.md`
- `.github/workflows/test.yml`
- `.github/workflows/build_firmware.yml`
- `test_computational_storage_payload.py`
- `computational_storage_poc/usb_host_inference.py`

## Visual/Browser Findings
- None yet.

## 2026-03-06 Roadmap Audit Findings
- `docs/ARCH AGENTIC ENGINEERING AND PLANNING/tracker-index.md` still shows only `AEP-2026-03-06` as active, and that cycle is limited to real RP2040 hardware evidence capture plus the dated retention review.
- `docs/ARCH AGENTIC ENGINEERING AND PLANNING/next-session.md` aligns with the tracker: no non-hardware engineering phase is listed as active.
- `README.md` says Phases 1-4 are complete.
- `REFACTORING_PLAN.md`, `COMPLETION_SUMMARY.md`, and `PR_DESCRIPTION.md` still contain older "Deferred to Phase 4" or "Future Development" language, so they cannot be treated as authoritative roadmap sources without checking the live code.
- The next audit step is to verify those older deferred items against the current codebase and experiment scripts before declaring the development roadmap complete.
- Live-code verification shows the previously deferred research features are already present:
  - `antigravity_engine.py` implements `ingest_streaming()` and `enable_adaptive_threshold()`.
  - `teacher_distillation.py` supports configurable `batch_size`, chunked encoding, and ensemble parallelism.
  - `cross_lingual_distillation.py` exists with language-aware teacher routing.
  - `online_updater.py` supports `triplet_margin`, `infonce`, and `cosine_similarity` losses plus diagnostics/scheduling.
  - `benchmark_beir.py`, `benchmark_multitask.py`, `dashboard_server.py`, `run_sweep.py`, and `run_large_sweep.py` all exist.
- Historical docs that still list missing features are stale relative to the code. Session 22 and Session 23 logs show the old top-15 implementation items were delivered and merged.
- The notable non-hardware gap is experimental execution, not implementation: `run_large_sweep.py` exists, but `large_sweep_results.json` and `large_sweep_results.csv` do not.
- `docs/phase4-experiment-protocol.md` is useful background for Phase 4 feature usage, but it is not a current post-development roadmap or weight-refinement plan.
- A new canonical current-state doc now exists at `docs/roadmap-audit-and-weight-refinement-plan-2026-03-06.md`.
- `docs/INDEX.md` now links that audit/test-plan doc so future sessions do not need to reconstruct the conclusion from session logs.

## 2026-03-06 Experiment Campaign Findings
- Runtime prerequisites are currently available locally:
  - `torch 2.9.1+cpu`
  - `sentence_transformers 5.2.0`
  - `mteb 2.6.1`
  - `qdrant_client` import succeeded
  - `numpy 2.3.5`
- A `requests` dependency warning is emitted during imports, but it is not a hard blocker.
- Existing local experiment artifacts include `adapter_weights.pt` and `sweep_results.json`.
- No completed large-sweep artifact exists yet.
- `antigravity_engine.py` already exposes `enable_online_updates(...)`, so Phase 5 does not require new engine hooks.
- Historical docs indicate `benchmark_comparative.py` already has an `online_updates` configuration, which may be sufficient for online-ablation work without adding a brand-new benchmark script.
- `benchmark_comparative.py` and `benchmark_beir.py` were not safe to use as-is for real evaluation because their CLIs did not wire in a real `engine_factory`; they defaulted to dummy retrieval. This session patched them to use real engines from the CLI path.
- Real evaluation also required mapping Qdrant point IDs back to original document IDs. A shared `map_predicted_ids()` helper is now in `benchmark_utils.py`.
- `run_sweep.py` and `run_large_sweep.py` originally reused the shared SciFact Qdrant path and had no query cap. This session added:
  - `--max-queries`
  - `--db-path`
- The first bounded campaign failed Phase 1 because an orphaned `run_sweep.py` process from an earlier launch still held `db_scifact_evolution`. That stale process was terminated.
- The campaign runner now:
  - passes `-u` to child Python benchmark commands for unbuffered logging on future runs
  - snapshots/restores `adapter_weights.pt`
  - writes an on-disk manifest
  - uses UTF-8 for child processes
  - launches sweep phases against isolated per-run Qdrant directories
- The current active campaign run is `experiment_runs/weight-refinement-20260306-session28-isolated`.
- During the current isolated run, `phase1_standard_sweep.log` stayed quiet after initialization, but the child `run_sweep.py` process continued consuming CPU and writing to the isolated Qdrant folder, indicating active execution rather than an early crash.
- The worktree currently has uncommitted documentation/planning files from the roadmap audit:
  - `docs/INDEX.md`
  - `docs/roadmap-audit-and-weight-refinement-plan-2026-03-06.md`
  - `task_plan.md`
  - `findings.md`
  - `progress.md`
- Local branch inventory includes many historical `backup/*`, `feat/*`, `feature/*`, and `pr/*` branches. They will be treated as historical context only unless this campaign needs recovery from them.
- The remaining contamination bug was broader than the runner-level snapshot/restore: `benchmark_distillation.py` reused the shared root adapter across baseline/offline/hybrid in a single process, `benchmark_multitask.py` could reuse it across tasks, and `benchmark_beir.py` inherited the same risk through `ComparativeTestbed`.
- The fix is now centralized in `benchmark_utils.isolated_adapter_state()`. `benchmark_comparative.py` uses it per configuration, `benchmark_distillation.py` uses it per mode, and `benchmark_multitask.py` uses it per task.
- `benchmark_distillation.py` now accepts `--max-eval-queries`, and `run_weight_refinement_campaign.py` passes the campaign query budget through to that phase.
- Targeted validation passed after the fix:
  - `python -m py_compile benchmark_utils.py benchmark_comparative.py benchmark_distillation.py benchmark_multitask.py run_weight_refinement_campaign.py test_benchmark_comparative.py`
  - `python -m ruff check benchmark_utils.py benchmark_comparative.py benchmark_distillation.py benchmark_multitask.py run_weight_refinement_campaign.py test_benchmark_comparative.py`
  - `python -m unittest test_benchmark_comparative.py test_benchmark_beir.py -v`
  - a synthetic real-engine smoke covering baseline/offline/hybrid, with the root adapter checksum unchanged before and after
- The previous run at `experiment_runs/weight-refinement-20260306-session28-isolated` was intentionally abandoned after the contamination diagnosis.
- The fresh clean relaunch is `experiment_runs/weight-refinement-20260306-session28-clean`.
- The clean run started at `2026-03-06T13:42:14` with wrapper PID `50404`, runner PID `86244`, and Phase 1 child PID `72276`.
- Early clean-run evidence:
  - `manifest.json` created successfully with `baseline_adapter_snapshot: null`
  - `phase1_standard_sweep.log` is updating normally
  - Phase 1 is ingesting SciFact into the per-run Qdrant path `experiment_runs/weight-refinement-20260306-session28-clean/qdrant/phase1_scifact_db`
- Completed short-run outputs before the interruption:
  - Phase 1 sweep winner: `learning_rate=0.01`, `threshold=1`, `noise_scale=0.2`, `epochs=5`, improving SciFact NDCG from `0.6289` to `0.6766` (`+0.0477`)
  - Phase 2 distillation produced no retrieval delta across teacher weights `0.3`, `0.5`, and `0.7`; baseline, offline, and hybrid all stayed at mean NDCG `0.7553`, while offline pretraining time ranged from `171.7s` to `216.3s`
  - Phase 3 multitask results were stable (`avg_jaccard=1.0`) but showed zero learning gain; aggregate NDCG was `0.6782` for the small suite and `0.6265` for the medium suite
  - Phase 4 BEIR small favored `baseline` and `random_mask_50pct` (`mean_ndcg_at_10=0.6839`) over `chelation` (`0.5745`) and `online_updates` (`0.5746`), with `online_updates` also incurring materially higher latency (`87.75ms` mean)
- The clean campaign then stranded during `phase4_beir_medium`:
  - existing log content stopped mid-SciFact `online_updates`
  - no `phase4_beir_medium.json`, `phase5_online_ablation.json`, or `SUMMARY.md` was written
  - `manifest.json` stayed frozen at `phase4_beir_small`
- `run_weight_refinement_campaign.py` now supports `--resume-run-dir`:
  - resume mode loads the existing manifest
  - recovers already-completed phases from existing output files
  - executes only missing phases
  - can continue into Phase 5 and launch Phase 6 without replaying the entire campaign
- Resume validation passed with:
  - `python -m py_compile run_weight_refinement_campaign.py test_run_weight_refinement_campaign.py`
  - `python -m ruff check run_weight_refinement_campaign.py test_run_weight_refinement_campaign.py`
  - `python -m unittest test_run_weight_refinement_campaign.py -v`
- Active resumed run state:
  - resume wrapper PID `35364`
  - resumed child PID `47792`
  - child command: `python -u benchmark_beir.py --tier medium --model sentence-transformers/all-MiniLM-L6-v2 --max-queries 50 --output ...\\phase4_beir_medium.json`
  - `logs/phase4_beir_medium.log` was recreated at `2026-03-06 23:39:53`, confirming the resume path re-entered the missing BEIR medium phase

## 2026-03-06 Backlog Triage Findings
- The current repo state still supports the earlier roadmap-audit conclusion: there is no unfinished product-implementation phase outside the computational-storage follow-through.
- `docs/roadmap-audit-and-weight-refinement-plan-2026-03-06.md` is still aligned with `README.md`: the remaining non-hardware work is experiment execution, result analysis, and optional documentation cleanup.
- The strongest actionable non-test work tonight is cleanup of stale roadmap language in:
  - `REFACTORING_PLAN.md`
  - `COMPLETION_SUMMARY.md`
  - `PR_DESCRIPTION.md`
- Those files still describe deferred "Phase 4" work even though the corresponding capabilities are already present in the codebase, so they now create confusion rather than plan useful implementation work.
- No additional active engineering phase was found via repository-wide backlog-marker search; the remaining concrete product backlog is still:
  - real RP2040 hardware evidence capture when hardware is available
  - the dated retention review window on or after `2026-04-05`

## 2026-03-07 Overnight Orchestration Findings
- The resumed `phase4_beir_medium` process was still live after the priority shifted away from overnight execution. It was intentionally stopped to avoid burning CPU while the session moved to documentation and reviewable PR prep.
- Three isolated PRs were opened instead of mixing code, doc cleanup, and wrap artifacts:
  - PR `#96` `feat: harden weight refinement campaign recovery`
  - PR `#97` `docs: normalize stale roadmap documents`
  - PR `#98` `docs: wrap session28 overnight orchestration`
- PR `#96` contains:
  - real-engine evaluation wiring for comparative and BEIR entrypoints
  - adapter isolation to avoid cross-configuration checkpoint contamination
  - resume support for interrupted campaign runs
  - durable docs for the partial Session 28 campaign results
- PR `#97` converts stale "Phase 4" roadmap language in the old hardening docs into explicit historical notes instead of active backlog wording.
- PR `#98` records the overnight orchestration in the session log, verification log, phase summaries, next-session handoff, and `CLAUDE.md`.
- No test suite was rerun in the overnight session by user direction. Only lightweight non-test validation was recorded:
  - `python -m py_compile ...` for PR `#96`
  - `git diff --check` for PRs `#96`, `#97`, and `#98`
