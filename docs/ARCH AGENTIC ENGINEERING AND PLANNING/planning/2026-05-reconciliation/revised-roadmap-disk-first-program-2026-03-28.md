# Revised Roadmap: Disk-First CPU And Retrieval Program

Date: 2026-03-28
Status: Proposed active program roadmap

## Why This Replaces The Old Framing

The older roadmap audit correctly concluded that the prior retrieval feature backlog was largely implemented.
That conclusion is now too narrow for the current direction.

ChelatedAI is no longer just deciding how to run more evaluation on the existing retrieval stack.
It is now deciding whether to evolve into a **disk-first, CPU-native, retrieval-augmented system** that combines:

- selective SSD-resident model storage
- CPU-native low-bit inference
- optional MoE expert pruning
- optional KV / vector compression
- graph and embedding memory for code

That requires a new program roadmap, not another benchmark-only plan.

## Program Goal

Build a credible non-GPU-first architecture for coding and retrieval workloads that:

1. keeps the model small enough for CPU-native execution
2. shifts large or sparse artifacts onto disk
3. uses retrieval and graph memory to reduce parametric burden
4. preserves tight scope control between phases and PRs
5. adds explicit review and hardening loops after each implementation phase

## Operating Principles

1. **CPU first, not controller fantasy**
   - Treat SSD as a storage and bandwidth surface.
   - Treat CPU as the real execution surface unless and until hardware evidence proves otherwise.

2. **External memory is a first-class capability**
   - Code graphs, embeddings, and retrieved artifacts are part of the architecture, not an afterthought.

3. **Do not solve all bottlenecks at once**
   - Weight footprint, KV cache, CPU kernel efficiency, and retrieval memory are separate concerns.

4. **Every phase must end with review loops**
   - Architecture review
   - code analysis / hardening review
   - promotion or defer decision

5. **PRs stay small and phase-scoped**
   - No cross-phase implementation PRs.
   - No multi-surface PRs unless the interface itself is the artifact.

## Standard Phase Loop

Every roadmap phase uses the same loop:

1. **ARCH**
   - define scope lock
   - write or update architecture brief / ADR
   - define entry and exit gates

2. **IMPLEMENT**
   - deliver the smallest coherent slice of code
   - keep PRs aligned to one component boundary or one integration seam

3. **ARCH-AEP REVIEW**
   - run architecture and dependency review on the delivered slice
   - log risks, drift, and follow-up findings before next-phase expansion

4. **CODE ANALYSIS / HARDENING**
   - targeted code review
   - test expansion
   - performance or correctness validation
   - doc synchronization

5. **PROMOTE / DEFER**
   - either mark the phase as a baseline for the next phase
   - or defer with a written unblock plan

This loop is mandatory for every phase below.

## Program Dependency Graph

```text
P0 Scope Lock + Architecture Reset
  -> P1 Storage Substrate
      -> P2 CPU Inference Substrate
          -> P3 Compression Branches
              -> P3A Dense/Sparse FFN path
              -> P3B MoE + REAP path
          -> P4 Retrieval / Graph Memory
      -> P5 Runtime Integration
          -> P6 Long-Context And Memory Compression
              -> P7 End-to-End Evaluation And Promotion
```

Key dependency rule:

- `P4 Retrieval / Graph Memory` may start once `P1` is stable, but it may not merge end-to-end runtime bindings until `P2` is stable.

## Phase 0: Scope Lock And Architecture Reset

### Goal

Replace the old passive evaluation plan with a program plan, interfaces, and success metrics.

### Deliverables

- this roadmap
- updated ARCH-AEP workflow guidance
- explicit phase entry / exit criteria
- initial component boundaries

### Implementation scope

- docs only

### PR shape

- one docs PR

### Exit gates

- active roadmap points to this document
- ARCH-AEP docs include the mandatory review loops
- no unresolved ambiguity about whether the target system is:
  - CPU-native
  - disk-assisted
  - retrieval-augmented

## Phase 1: Storage Substrate

### Goal

Build a real disk-backed artifact layer instead of the current dense toy block payload path.

### Deliverables

- manifest-driven model artifact format
- quantized packed blocks or bundles
- `mmap` or equivalent disk-backed reader
- benchmark harness for read patterns and chunk reuse

### Must not include

- transformer runtime integration
- CPU kernel work
- retrieval graph work

### Recommended PR split

1. artifact manifest schema
2. packed block writer / reader
3. disk-backed benchmark harness

### Exit gates

- no full-file preload in the "real" storage path
- deterministic parity tests on packed artifacts
- measured storage metrics, not only theoretical constants

### Critical risks

- overfitting format to the current toy MLP path
- mixing storage format decisions with runtime compute decisions too early

## Phase 2: CPU Inference Substrate

### Goal

Establish a credible CPU-native inference path for small models and packed weights.

### Deliverables

- low-bit kernel abstraction
- at least one practical CPU inference backend
- benchmarkable token-generation or layer-execution harness

### Recommended direction

- prioritize low-bit / LUT-friendly execution
- treat T-MAC / BitNet-style CPU paths as design inspiration

### Recommended PR split

1. backend abstraction and capability model
2. first CPU execution backend
3. benchmark and regression harness

### Exit gates

- measured CPU baseline exists
- packed storage artifacts from Phase 1 can feed CPU execution
- no dependency on GPU presence for baseline validation

### Critical risks

- trying to support too many backends at once
- importing a research kernel without a stable abstraction boundary

## Phase 3: Compression Branches

This phase is intentionally split into two branches because dense and MoE paths have different dependencies.

### Phase 3A: Dense / Sparse FFN Path

#### Goal

Add a sparse selective-loading path for dense or semi-sparse models.

#### Deliverables

- resident-vs-streamed tensor split
- sparse predictor or routing heuristic
- sliding window or chunk reuse cache

#### Exit gates

- measurable reduction in streamed bytes per token
- correctness preserved against dense baseline on the test model

### Phase 3B: MoE + REAP Path

#### Goal

Support MoE-targeted disk reduction before packing and execution.

#### Deliverables

- MoE artifact model
- expert-bank layout
- pruning pipeline or compatibility layer for REAP-like preprocessing

#### Exit gates

- expert bank can be pruned before packing
- routing and expert metadata survive serialization

#### Critical risks

- coupling MoE pruning to a dense-only storage format
- mixing router changes with runtime changes in one PR

## Phase 4: Retrieval And Graph Memory

### Goal

Externalize part of coding capability into a disk-backed knowledge layer.

### Deliverables

- code graph schema
- embedding index / retrieval surface
- graph-aware or hybrid reranking
- repo-local memory ingestion pipeline

### Why this phase exists

This is where the plan explicitly uses "stored knowledge on disk" to reduce parametric burden.

### Recommended PR split

1. schema and storage model
2. ingestion and update pipeline
3. retrieval and reranking API

### Exit gates

- memory layer works without runtime coupling
- retrieval quality and latency are measured independently

### Critical risks

- turning this into a general knowledge platform instead of a code-memory layer
- mixing memory ingestion and model execution in the same phase

## Phase 5: Runtime Integration

### Goal

Combine the storage substrate, CPU inference substrate, and retrieval memory into a single runnable prototype.

### Deliverables

- scheduler for resident vs streamed tensors
- retrieval-aware execution orchestration
- one end-to-end prototype path for a small target model

### Recommended scope

- choose a single reference model class
- choose a single reference workload:
  - code completion
  - retrieval-augmented code editing
  - repository Q&A

### Exit gates

- end-to-end local run succeeds on CPU without GPU
- runtime metrics are captured:
  - token/sec
  - SSD bytes/sec
  - RAM footprint
  - retrieval latency

### Critical risks

- integrating before the lower layers stabilize
- adding multiple workload types at once

## Phase 6: Long-Context And Memory Compression

### Goal

Reduce runtime memory pressure once the integrated baseline exists.

### Deliverables

- KV-cache compression experiments
- compressed retrieval payload or vector-store experiments
- memory-pressure benchmarks

### Recommended direction

- treat TurboQuant-like methods as phase-6 work, not phase-1 work

### Exit gates

- memory savings are measured
- quality impact is measured
- compression does not destabilize baseline execution

### Critical risks

- optimizing KV cache before the base runtime exists
- treating vector compression as a substitute for weight compression

## Phase 7: End-To-End Evaluation And Promotion

### Goal

Decide whether the architecture is promotable, deferrable, or needs a branch reset.

### Deliverables

- benchmark pack
- promotion memo
- no-go memo if required
- preset / config recommendations

### Required review loops

- ARCH-AEP post-implementation review
- code analysis / hardening review
- documentation audit

### Promotion questions

1. Is CPU-only execution practically usable on the chosen workload?
2. Does disk assistance improve deployability without killing throughput?
3. Does retrieval / graph memory reduce the need for a larger resident model?
4. Is the combined system simpler than the operational cost of using a GPU?

## What Changes In ARCH-AEP

The current ARCH-AEP workflow is remediation-oriented.
For this program, it also needs a **program loop**.

Additional required loop after each implementation phase:

1. architecture conformance review
2. dependency drift review
3. code analysis / hardening pass
4. phase close checklist
5. next-phase scope lock

This is not optional.
Without it, the later phases will accumulate hidden coupling and scope drift.

## PR And Scope Discipline

### Rules

1. One phase at a time.
2. One integration seam per PR.
3. No merging future-phase scaffolding "just to save time."
4. Every phase needs explicit non-goals.
5. Every phase needs a defer path for unresolved research risk.

### Recommended PR size rule

- docs / architecture PRs: one sitting
- code PRs: one component or one seam
- integration PRs: one prototype path only

### Context minimization rule

Before each phase:

- refresh only the current roadmap phase
- current ADR / architecture brief
- directly touched component docs
- current phase tracker

Do not pull the full historical archive into every implementation loop.

## Immediate Next Steps

1. Promote this roadmap as the active program plan.
2. Update ARCH-AEP docs to reflect the new mandatory review loops.
3. Start `Phase 1: Storage Substrate` as the first real implementation phase.
4. Keep retrieval-memory work in backlog preparation, but do not integrate it until the storage substrate is stable.

## Active Recommendation

The strongest architecture program for ChelatedAI right now is:

1. **Storage substrate**
2. **CPU inference substrate**
3. **Compression branch**
4. **Retrieval / graph memory**
5. **Runtime integration**
6. **Long-context compression**
7. **Promotion review**

That order is the best balance between ambition and control.
