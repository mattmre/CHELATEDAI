# Evolution Strategies at the Hyperscale: ChelatedAI Comparison

## Purpose

This report compares the paper **"Evolution Strategies at the Hyperscale"** and its EGGROLL method with the ChelatedAI research direction, especially prior sessions around chelation, in-process tuning, low-rank/bounded adapters, distillation, quantized correction, repeatability gates, and computational-storage/drive-node experiments.

Source reviewed:

- Paper: `https://eshyperscale.github.io/imgs/paper.pdf`
- Project site: `https://eshyperscale.github.io/`

The paper includes an NVIDIA AI Technology Center affiliation, but its concrete technical claims are about GPU-efficient, low-rank Evolution Strategies. It does not claim that SSDs replace GPUs for full model training or inference.

## Executive conclusion

The paper strongly validates ChelatedAI's strategic direction around:

- black-box or scalar-fitness optimization
- low-rank correction surfaces
- quantization-aware/int8-compatible tuning
- bounded parameter-efficient updates
- inference-like batched optimization
- large-population candidate evaluation
- task-level fitness instead of purely differentiable losses

The paper does **not** directly validate the strongest version of the storage hypothesis: "AI can run as a stored database on SSDs instead of GPUs." EGGROLL still depends on GPU-class dense math throughput. What it does support is a more defensible intermediate thesis:

> ChelatedAI can evolve toward black-box, low-rank, quantization-aware, fitness-driven adaptation, and parts of candidate evaluation, retrieval fitness, vector filtering, and graph dispatch may eventually move closer to storage nodes. Dense neural math still requires GPU-class or accelerator-class compute unless future computational-storage devices provide real vector/matrix acceleration.

## Paper summary

EGGROLL stands for **Evolution Guided GeneRal Optimisation via Low-rank Learning**. It is a scalable Evolution Strategies method for high-dimensional neural-network optimization.

The paper's central claim is:

1. Classical Evolution Strategies are attractive because they are black-box, parallelizable, and can optimize noisy or non-differentiable objectives.
2. Naive ES does not scale well on GPUs for large neural networks because full-rank perturbations create low arithmetic intensity and heavy memory movement.
3. EGGROLL replaces full-rank random perturbations with low-rank perturbations of the form `E = A B^T / sqrt(r)`.
4. Each individual perturbation is low-rank, but the aggregate weighted update across a large population can become full-rank.
5. The hardware advantage comes from preserving the efficient base matrix multiply and adding cheap low-rank batched work, making training look more like batched inference.
6. The paper reports large speedups over naive ES and up to 91% of pure batch-inference throughput for relevant regimes.
7. The paper demonstrates pure integer language-model pretraining, RL benchmarks, foundation-model fine-tuning, and int8 quantized RWKV distillation/fine-tuning.

## ChelatedAI conceptual alignment matrix

| EGGROLL / paper point | ChelatedAI analogue | Alignment | Analysis |
|---|---|---:|---|
| Black-box scalar fitness | NDCG@10, hybrid gain, topology health, isomer drift, collapse severity | High | ChelatedAI already measures outcomes that can score candidate perturbations without differentiating through the full retrieval stack. |
| Low-rank perturbations | `LowRankAffineAdapter`, adapter factory, bounded correction wrappers | High | EGGROLL gives a stronger optimization method for the same family of parameter-efficient correction surfaces. |
| Aggregate update from many low-rank samples | Sedimentation over many collapse records and hard negatives | Medium-high | ChelatedAI accumulates many local corrective signals; EGGROLL gives theory for combining many low-rank directions into richer updates. |
| No gradient through objective required | Retrieval-quality optimization after vector search/reranking | High | Retrieval ranking, task score, and user-feedback metrics are often non-differentiable. ES is naturally suited to this. |
| Quantized/int8 model adaptation | Session 31 BoundedAdapter and INT8 correction-floor work | Very high | The repo already identified correction invisibility under INT8 quantization. EGGROLL's quantized/int8 experiments strengthen this track. |
| Inference-like optimization | Online updater and in-process correction | Medium-high | EGGROLL suggests replacing some online gradient steps with micro-population search around a query or batch. |
| Population-based exploration | Weight-refinement campaigns and candidate gates | High | Sessions 32 and 33 already test populations of adapter/teacher-weight candidates at the campaign level. EGGROLL brings this into the optimizer itself. |
| Perturbation-scale stability theory | Kalman LR and convergence monitoring | Medium | ChelatedAI has empirical stability tools; EGGROLL adds a theoretical reason to make perturbation scale adaptive. |
| Hardware-efficient batched perturbations | Computational-storage branch dispatch and drive-node racing | Medium | The overlap is indirect: both try to move work into parallel candidate evaluation. EGGROLL is GPU-centric, not storage-centric. |
| SSDs replace GPUs | Computational-storage vision | Low | Not supported directly. The paper uses GPUs heavily and optimizes GPU throughput. |

## Relevant ChelatedAI session evidence

### Session 31: strongest paper alignment

Session 31 added the most relevant architecture for an EGGROLL-inspired ChelatedAI path:

- `BoundedAdapter` wrapper with correction bounds, per-dimension scaling, and INT8 correction-floor handling.
- `SedimentationInfoNCELoss`, hybrid loss, and hard-negative mining.
- `KalmanLRScheduler` for confidence-aware learning-rate modulation.
- `DimensionProjection` fixes so teacher-student projection remains trainable.

Repo evidence:

- `docs/ARCH AGENTIC ENGINEERING AND PLANNING/session-log-2026-03-12-session31.md`
- `docs/ARCH AGENTIC ENGINEERING AND PLANNING/research-2026-03-12-session31-scaler-constrainer.md`

Key implication:

> Session 31 already moved ChelatedAI toward bounded, quantization-aware, low-rank, parameter-efficient adaptation. EGGROLL suggests replacing or augmenting the current gradient optimizer for those correction layers with low-rank population search.

### Sessions 32 and 33: evidence gates remain mandatory

Session 32 found one locally positive candidate:

- adapter: `mlp`
- teacher: `sentence-transformers/all-mpnet-base-v2`
- teacher weight: `0.3`
- hybrid final NDCG@10: `0.6239`
- baseline final NDCG@10: `0.6012`

But adjacent teacher weights collapsed, transfer was not proven, and no preset promotion occurred.

Session 33 repeated the positive SciFact candidate, then rejected it on candidate-specific transfer:

| Task | Baseline final NDCG@10 | Hybrid final NDCG@10 | Gain | Gate |
|---|---:|---:|---:|---|
| SciFact | 0.6012 | 0.6239 | +0.0226 | pass |
| NFCorpus | 0.4893 | 0.4847 | -0.0046 | fail |

Repo evidence:

- `docs/weight-refinement-campaign-results-2026-04-25-session32.md`
- `docs/weight-refinement-follow-up-results-2026-04-25-session33.md`
- `run_repeatability_check.py`
- `run_candidate_transfer_gate.py`

Key implication:

> EGGROLL-inspired candidates should not bypass ChelatedAI's repeatability and transfer gates. A strong method can still overfit one retrieval task or degrade adjacent datasets.

## Computational-storage comparison

ChelatedAI's storage track is serious but intentionally scope-limited.

Repo evidence:

- `docs/COMPUTATIONAL_STORAGE_DRIVE_NODES.md`
- `docs/computational-storage-transport-scope-decision.md`

Current validated storage claims:

- flash-friendly block-graph format
- software validation of storage-style block replay
- mock NVMe latency model
- speculative multi-drive node-racing simulation
- deterministic firmware/emulator/host transport surface

Current disallowed claims:

- full LLM running directly from a hard drive
- full RP2040 on-device digits inference already proven
- real hardware latency or throughput beyond software/theoretical validation
- SSD replacing GPU for dense model training

EGGROLL does not change that boundary. It strengthens the case for **near-data candidate evaluation** and **storage-resident retrieval/fitness infrastructure**, not passive SSD-only model execution.

## Correct storage thesis after EGGROLL

The most defensible thesis is:

> Storage can become an active memory and candidate-evaluation substrate for retrieval and low-rank adaptation, while dense model math remains on GPU-class or accelerator-class compute until storage devices expose sufficient vector/matrix execution.

Potential storage-resident responsibilities:

- vector shard storage
- local nearest-neighbor filtering
- collapse-log storage
- candidate perturbation seed storage
- retrieval-fitness scoring over local shards
- block-graph traversal
- speculative branch dispatch
- scalar fitness return to host

Responsibilities that still require GPU/accelerator-class compute:

- dense embedding model inference
- large matrix multiplications
- LLM forward passes at meaningful scale
- high-throughput population evaluation unless computational storage has equivalent math hardware

## Recommended ChelatedAI roadmap

### Phase 1: Add an experimental EGGROLL-style optimizer

Add a non-default optimizer path for adapter-only experiments:

```text
optimizer = "adam" | "eggroll_es"
```

Initial constraints:

- no base embedding model mutation
- update adapters only
- rank 1 perturbations first
- deterministic seeds
- bounded corrections
- scalar retrieval fitness
- no preset changes from first runs

Best first targets:

1. `LowRankAffineAdapter`
2. `BoundedAdapter.dim_scale`
3. final correction layer of `ChelationAdapter`
4. `DimensionMaskPredictor` mask logits

### Phase 2: Define ChelatedAI-native ES fitness

Candidate fitness should combine retrieval quality and structural safety:

| Fitness component | Purpose |
|---|---|
| NDCG@10 gain | Direct retrieval quality |
| Positive-negative rank margin | Local robust proxy |
| Collapse reduction | Core ChelatedAI objective |
| Topology cohesion | Structural health |
| Isomer drift penalty | Prevents structure-preserving failure |
| Quantized retrieval score | Ensures corrections survive INT8/Qdrant quantization |
| Teacher agreement | Useful when task labels are sparse |

### Phase 3: Adapt Kalman logic to ES perturbation scale

Extend the Session 31 Kalman idea:

| Current Kalman LR | EGGROLL-style extension |
|---|---|
| loss variance modulates learning rate | fitness variance modulates ES sigma |
| high variance lowers LR | high variance lowers perturbation scale or increases population |
| low variance raises LR | low variance permits larger exploratory perturbations |
| confidence-aware gradient training | confidence-aware black-box population search |

This could become a distinct ChelatedAI contribution: **Kalman-controlled low-rank Evolution Strategies for retrieval adaptation**.

### Phase 4: Replace some online gradient steps with micro-population search

Current online correction can be reframed:

```text
query arrives
-> retrieve baseline
-> generate small population of bounded low-rank/mask perturbations
-> score each candidate with local retrieval fitness
-> apply best weighted update or no-op
-> rollback if structure worsens
```

Conservative starting values:

| Parameter | Initial value |
|---|---:|
| population size | 8-32 |
| rank | 1 |
| updated params | mask logits or `dim_scale` only |
| persistence | ephemeral unless repeated wins |
| safety rule | no update if collapse/isomer risk rises |

### Phase 5: Quantization-first validation

Because Session 31 found that small MLP corrections can be invisible after INT8 quantization, every ES candidate should be scored both before and after quantization.

Promotion blocker:

> If a candidate only improves unquantized vectors but loses the gain after Qdrant/INT8 simulation, reject it.

### Phase 6: Candidate-specific promotion stack

Reuse the Session 33 evidence hierarchy:

1. Single-run quality gate.
2. Independent repeatability gate.
3. Candidate-specific small transfer gate.
4. Candidate-specific medium transfer gate.
5. BEIR/task-level validation.
6. Quantized retrieval validation.
7. Stability and adjacent-hyperparameter checks.
8. Preset promotion only after all gates pass.

### Phase 7: Storage-node experiment

Use the computational-storage mock path to simulate storage-sharded population evaluation:

```text
host:
  embeds query
  sends query vector + perturbation seeds to storage shards

storage shard:
  evaluates local vector candidates
  scores perturbation against local retrieval fitness
  returns scalar fitness + top-k

host:
  aggregates scalar fitness
  applies bounded low-rank update
```

Deliverable:

- latency comparison
- retrieval-quality comparison
- per-shard scalar fitness logs
- explicit statement that this is simulation unless run on real computational-storage hardware

### Phase 8: Hardware escalation

RP2040 remains appropriate for transport proof, not serious AI compute. For a real "AI as stored database" proof, evaluate:

| Hardware class | Why |
|---|---|
| FPGA with PCIe/NVMe | Can implement custom vector kernels near storage |
| computational-storage SSD SDK | Closest to the thesis |
| DPU/IPU-style device | Stronger local compute and networking |
| GPU-direct storage prototype | Transitional architecture |
| multi-SSD host simulation | Cheap pre-hardware validation |

## How EGGROLL could learn from ChelatedAI

ChelatedAI also offers ideas that could strengthen EGGROLL-like systems:

| ChelatedAI idea | Benefit to EGGROLL-like optimization |
|---|---|
| Structural health reporting | Avoid reward wins that damage representation topology |
| Isomer detection | Detect same-answer/different-structure failures |
| Sedimentation | Reuse historical failure clusters to bias future perturbations |
| BoundedAdapter | Keep black-box updates within safe correction budgets |
| INT8 correction floors | Ensure updates survive quantized deployment |
| Candidate-specific transfer gates | Reduce single-benchmark overfitting |
| AEP severity/effort model | Prioritize expensive population search where it matters |
| Drive-node dispatch | Explore near-data population evaluation |

## Implementation priorities

Recommended order:

1. Add an experimental ES optimizer around adapter-only correction surfaces.
2. Start with `LowRankAffineAdapter` plus `BoundedAdapter`.
3. Use scalar retrieval fitness rather than MSE-only loss.
4. Add quantized scoring before any promotion decision.
5. Run the Session 33 repeatability and transfer gates.
6. Only after a candidate passes software gates, connect the storage-node simulation.
7. Keep all SSD/database claims explicitly scoped until hardware evidence exists.

## Final position

The paper is highly relevant to ChelatedAI. It supports the repo's direction more strongly than it supports a direct GPU-replacement claim.

The strongest updated strategy is:

> Use EGGROLL-style low-rank Evolution Strategies as an experimental optimizer for ChelatedAI's bounded, quantization-aware adapter and mask-correction layers. Treat retrieval quality and structural health as scalar fitness. Preserve Session 33's strict repeatability and transfer gates. Continue computational-storage work as a near-data retrieval and candidate-evaluation track, not as a proven SSD-only LLM runtime.

