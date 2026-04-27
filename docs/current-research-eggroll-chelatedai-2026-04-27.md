# Current Research Validation for EGGROLL-Inspired ChelatedAI Work

## Scope

This note records a targeted current-research scan for the EGGROLL-inspired ChelatedAI implementation plan. The scan focused on work from roughly the last two months where available, plus immediately adjacent 2025-2026 work that directly informs the architecture.

## Findings

### Low-rank Evolution Strategies and zeroth-order optimization

Recent work around **Evolution Strategies at the Hyperscale / EGGROLL** supports using low-rank perturbations to scale zeroth-order optimization to billion-parameter models. The key implementation lesson for ChelatedAI is that low-rank perturbations can make black-box optimization look like efficient batched inference while preserving scalar-fitness optimization.

Related low-rank ZO work, including LOREN/LOZO-style methods, reinforces the same design choice: constrain search into low-rank or curvature-aware subspaces rather than perturbing all parameters naively.

Implementation consequence:

- Add a low-rank ES optimizer for adapter-only parameters.
- Keep deterministic seed replay and avoid storing full perturbation matrices.
- Make scalar retrieval fitness the optimization target.

### Quantized zeroth-order tuning

Recent quantized ZO work such as QuZO/QZO and Quantized Evolution Strategies points toward directly optimizing low-precision or quantization-constrained models with forward-pass-only feedback. This aligns with the Session 31 BoundedAdapter finding that small MLP corrections can be invisible after INT8 quantization.

Implementation consequence:

- Add quantization-aware scoring to the ES fitness path.
- Reject candidates that only improve pre-quantization vectors.
- Keep bounded correction floors active for quantized retrieval deployments.

### Geometry-aware LoRA/adapter optimization

RoZO-style work on geometry-aware zeroth-order optimization for low-rank adapters supports the decision to target adapters rather than base model weights. ChelatedAI should not mutate base embedding models in this track; it should optimize bounded adapter or mask-correction surfaces.

Implementation consequence:

- Start with `LowRankAffineAdapter`, `BoundedAdapter`, and mask/correction parameters.
- Keep current Adam pathways as defaults.
- Add ES behind explicit config/API flags only.

### Adaptive retrieval and self-correcting RAG

Recent agentic RAG and self-correcting retrieval systems support closed feedback loops around retrieval quality, query reformulation, reranking, and failure-driven updates. ChelatedAI's chelation logs, topology health, isomer drift, and candidate-specific transfer gates fit this trend.

Implementation consequence:

- Treat retrieval metrics and structural health as scalar ES fitness.
- Preserve Session 33 repeatability and transfer gates before any preset promotion.

### Computational storage and near-data retrieval

Recent near-storage processing work for LLM inference and vector search, including SmartSSD/FPGA-style LLM offload and near-data graph ANN search, supports the storage-node research direction. It does not support a passive-SSD replacement for GPU compute. The defensible path is near-data retrieval filtering, candidate scoring, KV/cache-style memory operations, and sharded fitness evaluation.

Implementation consequence:

- Add a storage-sharded population evaluation simulation.
- Keep RP2040 and SSD claims scoped until real hardware evidence exists.
- Frame storage as a near-data candidate-evaluation substrate, not a proven full LLM runtime.

## Design decision

Proceed with a full non-default implementation:

1. Low-rank ES adapter optimizer.
2. Quantization-aware fitness.
3. Kalman-controlled perturbation scale.
4. Online ES micro-population updater.
5. Storage-sharded population scoring simulation.
6. Tests and docs.

Defaults remain unchanged until a future candidate passes repeatability, transfer, and quantized retrieval gates.
