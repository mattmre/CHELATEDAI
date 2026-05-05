# AttnRes Adapter Implementation (2026-05-04)

## Source Paper

MoonshotAI "Attention Residuals" (2025) — Block-AttnRes variant.  
Repository: https://github.com/MoonshotAI/Attention-Residuals

## Core Concept

Standard transformer residuals apply a fixed `h = h + f(h)` at every layer, which causes
two pathologies as models deepen: (1) early-layer signal gets diluted, and (2) hidden-state
magnitudes grow unboundedly with PreNorm architectures.

AttnRes replaces fixed accumulation with **learned softmax attention over all prior block
outputs**. The Block variant (practical) partitions layers into ~8 blocks, keeps standard
residuals within each block, and inserts cross-block attention aggregation between blocks.
The final representation is a content-aware weighted sum of all block outputs — not just
the last one.

Benchmark results on Kimi Linear 48B / 1.4T tokens:
- GPQA-Diamond: +7.5%
- Mathematics: +3.6%
- HumanEval: +3.1%
- BBH: +1.7%
- MMLU: +1.1%
- Only 1.25× additional training compute vs. baseline

## Mapping to ChelatedAI

ChelatedAI adapters already use the same `x + delta` residual pattern as a standard
transformer block — but as a single-shot correction. The pathology is the same: if the
adapter's single correction is wrong (too strong, too weak, wrong direction), there is no
mechanism to recover short of retraining.

The Block-AttnRes design maps directly:
- `num_blocks` correction blocks = layers within a transformer block
- Input embedding as block-0 = pre-transformer representation
- Cross-block softmax attention = the AttnRes aggregation selecting correction depth

## What Was Implemented

### `BlockAttnResAdapter` (`chelation_adapter.py`)

New `"attnres"` adapter type, accessible via `create_adapter("attnres", ...)`.

Architecture:
```
Input x
  ├── block_states[0] = x
  ├── block_states[1] = x + MLP_1(x)         # within-block residual
  ├── block_states[2] = h1 + MLP_2(h1)
  ├── ...
  └── block_states[N] = h_{N-1} + MLP_N(h_{N-1})

query = query_proj(h_N)                       # final state as query
keys  = key_proj(stack(block_states))         # all states as keys
attn  = softmax(query · keys^T / sqrt(proj))  # cross-block attention
out   = normalize(attn · stack(block_states)) # weighted aggregation
```

Key properties:
- Near-identity initialization (std=0.001 weights) — safe base model preservation
- L2-normalized output — preserves cosine similarity metric
- Compatible with `BoundedAdapter` wrapping and existing sedimentation loss
- `regularization_loss()` returns 0.0 — no extra terms needed

### `LayerAttentionAggregator` (`chelation_adapter.py`)

Companion module for the Model-Scope runtime. Takes `[batch, num_layers, hidden_size]`
mean-pooled transformer layer embeddings and replaces the "use last layer" heuristic with
a learned softmax aggregation.

This is the bridge between AttnRes and the model hook bus: once `ModelHookBus` is extended
to capture raw mean-pooled embeddings per layer (not just statistical summaries),
`LayerAttentionAggregator` can be trained on top to produce better retrieval embeddings
than any single-layer extraction.

### Config Presets (`config.py`)

```python
# ADAPTER_TYPE_PRESETS — "attnres" entry with num_blocks=4
ChelationConfig.get_preset("attnres", "adapter_type")

# ATTNRES_ADAPTER_PRESETS
ChelationConfig.get_preset("shallow",  "attnres_adapter")  # num_blocks=2
ChelationConfig.get_preset("balanced", "attnres_adapter")  # num_blocks=4
ChelationConfig.get_preset("deep",     "attnres_adapter")  # num_blocks=8 (paper scale)
```

### Tests (`test_attnres_adapter.py`)

37 new tests, all passing:
- `TestBlockAttnResAdapterBasics` — shape, normalization, identity init, save/load
- `TestBlockAttnResAdapterBlocks` — num_blocks variants, gradient flow through blocks and attention
- `TestBlockAttnResAdapterFactory` — `create_adapter("attnres")`, bounded wrapping, error messages
- `TestAttnResConfigPresets` — all three presets, copy isolation
- `TestLayerAttentionAggregator` — shape, normalization, gradient flow, error guards

## Next Steps (AttnRes track)

1. **Wire `LayerAttentionAggregator` into `model_scope_runtime.py`** — done. The hook bus
   can optionally capture raw mean-pooled embeddings per layer via
   `HookObservationConfig.capture_raw_embeddings`, and `ModelScopeRuntime` can aggregate
   those tensors before final embedding normalization.
2. **Benchmark `attnres` vs. `mlp` on the road-course harness** — done for the initial
   deterministic SciFact slice. See `docs/attnres-road-course-benchmark-2026-05-04.md`.
   The balanced AttnRes path tied the MLP baseline and did not justify a default change.
3. **Evaluate num_blocks sensitivity** — engine and road-course profiles now expose
   explicit AttnRes block/projection controls, including `--profile-set attnres_num_blocks`
   for near-identity shallow/balanced/deep comparison and
   `--profile-set attnres_trained_num_blocks` for the same adapter-depth comparison after
   an opt-in sedimentation warmup/training pass. The next handoff is to run the trained
   grid on SciFact and NFCorpus to find the right default for embedding-size adapters
   (vs. full transformers).
