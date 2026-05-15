# Disk-Resident LLM Addendum: REAP, TurboQuant, And CPU/Disk-First Systems

Date: 2026-03-28

## Why This Addendum Exists

This note extends the earlier feasibility memo with two new directions:

- **REAP** from Cerebras
- **TurboQuant** from Google Research

It also answers the follow-up question:

> Is anyone actually building competitive CPU / RAM / disk-first systems without relying on GPU inference?

## Short Answer

Yes, but the important distinction is:

1. **Weight footprint reduction**
2. **KV-cache reduction**
3. **CPU kernel efficiency**
4. **External-memory retrieval instead of parametric compute**

These are not interchangeable.

For ChelatedAI's storage-first direction:

- **REAP** is directly relevant if the target model is an MoE, especially a coding MoE.
- **TurboQuant** is relevant for long-context memory pressure and attention throughput, but it does **not** shrink the base model weight footprint.
- CPU-first systems are now credible, but mostly through **aggressive low-bit kernels**, not through generic dense CPU inference.
- Retrieval / graph-memory systems can substitute for some parametric compute, but they change the architecture into a memory-augmented or RAG-style system rather than a pure disk-hosted model runtime.

## REAP: Why It Matters

Primary sources:

- Cerebras blog: https://www.cerebras.ai/blog/reap
- Paper: https://arxiv.org/abs/2510.13999

What REAP does:

- targets **sparsely activated MoE models**
- prunes low-impact experts
- preserves router control instead of merging experts into static averages

Why that matters for ChelatedAI:

- a disk-resident MoE system is bottlenecked by the total parameter footprint of all experts, even though only a few experts are active per token
- REAP attacks exactly that mismatch

What the sources claim:

- the paper reports strong results from `20B` to `1T` parameter SMoE models
- it says REAP achieves near-lossless compression on code-generation tasks even after pruning `50%` of experts
- the Cerebras blog says that on `Qwen3-480B-Coder-FP8`, `50%` pruning retained `97.6%` of baseline non-agentic coding ability and `96.7%` on SWE-Bench Verified

Why this is more relevant than generic pruning:

- for coding-oriented MoEs, REAP is not just shrinking a model
- it is shrinking the **disk-resident expert bank**
- that is exactly the storage burden ChelatedAI would otherwise have to stream or shard

### Practical implication for the disk-first design

If the future target is an MoE coder model:

- **add REAP before disk packing**
- then quantize
- then pack experts into shards or bundles on disk

That order is important.

Doing quantization first without expert pruning leaves too much dead storage.

### What REAP does not solve

- CPU kernel efficiency
- KV cache growth
- dense-model footprint

It is a highly relevant component, not a full solution.

## TurboQuant: Why It Matters

Primary sources:

- Google Research blog: https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/
- Paper: https://arxiv.org/abs/2504.19874

What TurboQuant does:

- compresses **vectors**
- especially useful for **KV cache** and **vector search**
- aims to reduce distortion while avoiding the usual memory-overhead tax of vector quantization

What the Google sources claim:

- TurboQuant is targeted at KV cache compression and vector search
- the blog says it can reduce KV memory by at least `6x`
- the blog also says `4-bit` TurboQuant can achieve up to `8x` speedup in attention-logit computation versus `32-bit` unquantized keys on H100
- the paper states absolute quality neutrality around `3.5 bits per channel` and marginal degradation around `2.5 bits per channel` for KV cache quantization

### Why TurboQuant matters for ChelatedAI

TurboQuant does **not** reduce the base model weight size on disk.

It helps with:

- long-context inference
- DRAM pressure from KV cache
- attention-side bandwidth
- vector-store compression for retrieval systems

That means it is relevant in **two** ChelatedAI tracks:

1. future transformer runtime work
2. the repo's existing retrieval / vector-store work

### Where to use it

Use TurboQuant-like ideas for:

- KV cache if ChelatedAI adds a transformer inference path
- compressed vector indices or retrieval payloads for large external memory

Do **not** treat TurboQuant as a substitute for disk-hosted weight compression.

It solves a different bottleneck.

## Combined Strategy: What To Add And In What Order

If the goal is a practical disk-first coding system:

1. Choose an MoE or sparse-friendly model family.
2. Apply **REAP** if the model is MoE.
3. Apply low-bit weight compression for the remaining weights.
4. Use **CPU LUT kernels** or native 1-bit / low-bit kernels for inference.
5. Add **TurboQuant-like KV compression** only when long context becomes the next bottleneck.
6. Add retrieval / graph memory for reusable coding knowledge and code patterns.

This yields a coherent stack:

- REAP shrinks stored experts
- low-bit kernels make CPU inference viable
- TurboQuant shrinks runtime memory
- retrieval reduces the amount of parametric reasoning needed

## Who Is Actually Doing CPU / RAM / Disk-First Work?

### 1. Apple: LLM in a Flash

Source:

- https://arxiv.org/abs/2312.11514

Why it matters:

- explicit flash / DRAM / compute cost model
- selective FFN loading from flash
- designed for limited-memory devices

This is still the closest primary-source match to ChelatedAI's disk-first hypothesis.

### 2. T-MAC: CPU LUT-based low-bit inference

Source:

- https://arxiv.org/abs/2407.00088

What it shows:

- low-bit CPU inference is no longer just a toy idea
- the paper reports:
  - `30 tok/s` on a single core of M2 Ultra for BitNet-b1.58-3B
  - `71 tok/s` on eight cores
  - `11 tok/s` on Raspberry Pi 5

That is one of the clearest pieces of evidence that CPU-only inference can be operationally useful.

### 3. BitNet CPU stack / bitnet.cpp

Sources:

- `1-bit AI Infra: Part 1.1`: https://arxiv.org/abs/2410.16144
- `BitNet b1.58 2B4T Technical Report`: https://arxiv.org/abs/2504.12285

What they show:

- CPU inference stacks are being built specifically for native low-bit models
- the CPU paper reports `2.37x` to `6.17x` speedups on x86 and `1.37x` to `5.07x` on ARM
- the 2B4T report says official CPU implementations are available

This matters because it means the CPU path now has both:

- native models
- native runtimes

### 4. Google Gemma.cpp

Sources:

- https://ai.google.dev/gemma/docs/gemma_cpp
- https://ai.google.dev/gemma/docs/integrations/ollama

What it shows:

- Google explicitly ships a lightweight pure C++ CPU runtime for Gemma
- Google also documents Gemma running through `llama.cpp` and `Ollama` on systems without a GPU

This is not proof of frontier CPU-only competitiveness by itself, but it is clear evidence that major labs are treating CPU execution as a real deployment surface.

## Adjacent But Not Pure CPU-Only

These are relevant because they use RAM / disk / I/O hierarchies, but they still depend on a GPU:

### FlexGen

Source:

- https://arxiv.org/abs/2303.06865

What it shows:

- GPU + CPU + disk offloading can run very large models on small GPU memory
- useful as a planning reference for tensor placement

### HeteGen

Source:

- https://arxiv.org/abs/2403.01164

What it shows:

- heterogeneous CPU + GPU + I/O scheduling can reduce offload latency

These systems are useful architectural references, but they do not satisfy your “without GPU inference” requirement.

## External Memory And Retrieval Instead Of More Compute

This is the closest match to your idea of:

> predetermined coding knowledge stored on disk and recalled through embeddings / graph structure

### kNN-LM

Source:

- https://arxiv.org/abs/1911.00172

What it shows:

- a pretrained language model can be augmented with a nearest-neighbor datastore
- no retraining is required for the improvement

This is the classic example of replacing some parametric burden with a disk-backed memory.

### RETRO

Source:

- https://arxiv.org/abs/2112.04426

What it shows:

- retrieval from a huge external database can let a much smaller model compete with much larger dense models
- the paper reports GPT-3-comparable performance with a model using `25x` fewer parameters when backed by a `2T` token database

This is extremely relevant to your direction because it says:

- model competitiveness can come from **external memory**
- not only from bigger resident weights

### Programming Knowledge Graphs for code generation

Source:

- https://arxiv.org/abs/2601.20810

What it shows:

- graph-structured retrieval can improve code generation accuracy by improving retrieval granularity and reranking
- the abstract reports up to `20%` pass@1 gains and a `34%` improvement over baselines on MBPP

This is much closer to the “coding subgraphs on disk” idea than plain chunk-based RAG.

### GraphSkill

Source:

- https://arxiv.org/abs/2603.06620

What it shows:

- hierarchical retrieval from structured documentation can improve graph-reasoning code generation while lowering inference cost

This is early-stage research, but it supports the direction of:

- store structured knowledge externally
- retrieve only the relevant subgraph
- let the model synthesize over it

## What This Means For ChelatedAI

### Strong recommendations

1. **If targeting MoE coder models, add REAP-style expert pruning to the roadmap.**
   - It is one of the most directly applicable additions for a disk-first coding assistant.

2. **Treat TurboQuant as a second-stage optimization.**
   - Use it after the base weight-footprint problem is solved.
   - It is most valuable for long context and retrieval-heavy flows.

3. **Do not pursue commodity SSD-controller execution as the main path.**
   - CPU kernels plus disk bandwidth are the credible near-term path.

4. **Add a retrieval-memory layer for code.**
   - store code idioms, dependency usage, API patterns, and repo-specific subgraphs on disk
   - retrieve them with embeddings or graph-aware search
   - use the model as a synthesizer over retrieved code knowledge

### Practical architecture split

ChelatedAI should likely evolve into two coupled engines:

#### A. Parametric core

- small enough to run on CPU
- low-bit or 1-bit friendly
- sparse / MoE-friendly if possible

#### B. Disk memory system

- code graph store
- API usage graph
- project embeddings
- retrieval index
- optional execution templates or canonical solutions

That split is more realistic than trying to make the parametric model alone do everything.

## Feasibility Summary

### Most useful additions

- **REAP**: yes, if MoE
- **TurboQuant**: yes, for KV / vector-store compression
- **CPU low-bit runtimes**: mandatory
- **graph / retrieval memory for code**: strongly recommended

### Best overall design direction

The strongest path now looks like:

1. compressed low-bit CPU-native model
2. disk-resident expert / weight store if needed
3. retrieval / graph memory for code
4. KV compression for long context

That is the closest thing to a competitive non-GPU coding system that still has a defensible technical story.
