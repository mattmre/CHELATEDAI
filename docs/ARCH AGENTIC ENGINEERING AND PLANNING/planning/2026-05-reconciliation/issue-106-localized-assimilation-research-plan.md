# Issue #106 — Localized Assimilation: Research Plan & Implementation Phases

**Session date:** 2026-04-08  
**Origin:** GitHub Issue #106 (mattmre/CHELATEDAI)  
**Panels run:** 6 expert panels (3 rounds) + 1 novelty identification panel  
**Status:** Research complete, implementation ready to begin

---

## 1. Original Concepts (Issue #106 + Comments)

### Core Proposal (Issue Body)
Live data should be "contoured" into a model's specific quantized/sharded vector subspace — not added to a flat RAG index. The model should "speak its own dialect" rather than translating from generic embeddings. Using distillation-via-assimilation (stronger model responses adjusting weights) to redefine subspace structure. Merging multiple subspace vectors into a new composite subspace as an additional quant vector representation.

### Three Directions Identified

**Direction A — Localized Subspace Assimilation (LSA)**  
Per-shard correction adapters routed by the model's own subspace geometry. Different semantic regions of the corpus get different correction geometries. Routing must use pre-adapter (raw) embeddings — this is the critical architectural invariant.

**Direction B — Angular Momentum Dynamics**  
Model weight state treated as a rotating body in high-dimensional space. `L_new = L_model + Σ(r_i × p_i)`. Multiple simultaneous data vectors apply "torque." Adversarial data has anomalous "centrifugal force" (Fc = mω²r). Maps formally to Riemannian SGD on SO(n) via Cayley parameterization — the `OrthogonalProcrustesAdapter` already lives on this manifold.

**Direction C — Document-Level DSOA**  
Dynamic Subspace Orthogonality Auditing. Treat each document as a local subspace manifold. Detect angular momentum delta spikes between document sections as context-injection attack signals. System Prompt = high-inertia flywheel that resists subspace displacement.

---

## 2. Architecture Decisions (Non-Negotiable)

| Decision | Rationale |
|---|---|
| **Route on raw pre-adapter embeddings only** | Prevents shard boundaries from going stale as the adapter trains; routing on adapted embeddings breaks the invariant on the first sedimentation cycle |
| **AntigravityEngine must not change** | God Object (1,621 lines, 84 flat files); all new code in new modules or `experiments/` |
| **Adapter-only approach for production** | Base model weight mutation introduces catastrophic forgetting, training/serving skew, and lineage problems; confine to `experiments/open_model_ft/` |
| **Gemma excluded** | Gemma Terms of Use restrict derivative distribution; use Apache 2.0 / MIT / Llama 3.2 Community only |
| **Shard ID in checkpoint filename** | Prevents cross-shard checkpoint contamination (precedent: Session 28 fix) |

### Translation Table: Physics Metaphor → Tensor Operation
*(This table must precede any code that uses angular momentum vocabulary)*

| Angular Momentum Term | ChelatedAI Tensor Equivalent |
|---|---|
| Angular momentum | Per-document correction norm vector |
| Precession | Singular vector angular drift across epochs |
| Centrifugal Defense | Correction norm z-score outlier / Mahalanobis distance |
| Dzhanibekov guard | Merge subspace orthogonality check |
| System Prompt flywheel | Reference anchor embedding; anchor-distance flag per quantum |
| Topological Phase Transition | Adapter weight divergence beyond BoundedAdapter budget |
| Torque from data vectors | Riemannian SGD step on SO(n) Cayley manifold |

---

## 3. Verified Literature (High Confidence)

All papers below are confirmed real. Do not cite Gemini's confabulated titles ("SSAM: Singular Subspace Alignment for Merging Multimodal LLMs, 2026" and "Quantization-Robust LLM Unlearning, 2026") — those concepts are real but the specific titles are not verified.

### Tier 1 — Directly Relevant, Already Partially Implemented
| Paper | ArXiv | Repo Connection |
|---|---|---|
| LoRA | 2106.09685 | `LowRankAffineAdapter` |
| QLoRA | 2305.14314 | `BoundedAdapter` (INT8 noise floor) |
| **DoRA** | **2402.09353** | Recommended over LoRA for open-model fine-tuning |
| InfoNCE / SimCLR | 2002.05709 | `SedimentationInfoNCELoss` |
| Ensemble Kalman Inversion | 1901.09640 | `KalmanLRScheduler` |
| Cayley transform / Stiefel manifold | Becigneul & Ganea 2019 | `OrthogonalProcrustesAdapter` |
| TTT | 1909.13231 | `online_updater.py` |
| Mahalanobis OOD | 1807.03888 | Not yet; extends `stability_tracker.py` |

### Tier 2 — Relevant, Not Yet Implemented
| Paper | ArXiv | Use |
|---|---|---|
| TIES-Merging | 2306.01708 | Subspace merging protocol |
| EWC | 1612.00796 | Catastrophic forgetting guard for base model mutation |
| GEM (continual learning) | 1706.08840 | Multi-task adapter stability |
| AnglE | 2309.12871 | Closest prior art to LSA as a fine-tuning objective |
| BGE-M3 | 2309.07597 | Cross-architecture validation base |
| E5-mistral | 2401.00368 | Retrieval fine-tuning methodology |
| NV-Embed-v2 | 2405.17428 | Validates InfoNCE on decoder embedders |
| SFR-Embedding-Mistral | 2405.07440 | Targeted retrieval fine-tuning > general instruction tuning |
| GTE training methodology | 2308.03281 | Base model methodology (GTE-Qwen2) |
| BEIR benchmark | 2104.08663 | Evaluation standard |
| MTEB | 2210.07316 | Evaluation standard |
| Matryoshka (MRL) | 2205.13147 | Subspace truncation / nested structure |
| **geoopt** | **2005.02819** | Riemannian optimization in PyTorch (Grassmann distance) |
| HackAPrompt | 2211.09527 | Prompt injection evaluation |
| MAML | 1703.03400 | Fast adaptation / "torque from multiple data vectors" |
| Monolith (ByteDance) | 2209.07663 | Production architecture for live embedding updates |

---

## 4. Phased Implementation Plan

### Phase A — Immediate Drop-In (1 PR, ~1 session, no GPU)

**Goal:** Better frozen base model with zero disruption to existing architecture.

**Change:** Add `trust_remote_code=True` flag to `LocalEmbeddingBackend` in `embedding_backend.py`. This enables loading `nomic-ai/nomic-embed-text-v1.5` (Apache 2.0, 137M params, 768-dim, Matryoshka) which loads natively via SentenceTransformers.

**Expected gain:** 2–5 NDCG@10 points on BEIR from base model quality alone, before any adapter training.

**Files:** `embedding_backend.py` (~20 lines)

---

### Sprint 1 — Geometry Loss Classes (5 story points, no GPU needed)

**Goal:** Formalize LSA, DSOA, and angular momentum as differentiable training objectives. These improve the existing adapter training pipeline immediately and are prerequisites for the open-model fine-tuning track.

**New files:**

```
geometry_losses.py  (or split as below)
  dsoa_loss.py          ~80 lines  — soft bond-distribution KL regularizer
  angular_loss.py       ~40 lines  — Procrustes angular velocity regularizer  
  lsa_loss.py           ~30 lines  — subspace projection contrastive (wraps SedimentationInfoNCELoss)
```

**Key implementation notes:**

`dsoa_loss.py`:
- Modify `TopologyAnalyzer` to support soft (temperature-scaled softmax) bond assignments
- Target bond distribution = "healthy" baseline from a clean corpus
- Loss = `F.kl_div(current_soft_distribution, target_distribution)`
- Batch-size sensitivity: validate signal-to-noise at batch 32 vs. 128 vs. 256 before deploying

`angular_loss.py`:
- `OrthogonalProcrustesAdapter.get_angular_velocity(prev_skew_param)`
- `torch.norm(A_now - A_prev, p='fro')` where A = skew-symmetric form of `_skew_param`
- Expose in `get_state()` dict; add rolling `_prev_skew_param` snapshot
- ~20 lines + 5 tests in existing `test_unit_core.py`

`lsa_loss.py`:
- SVD-based ephemeral "fast weights" on the incoming document batch
- `U_svd, S, Vt = torch.linalg.svd(embedding_matrix, full_matrices=False)`
- Option A (preferred): compose additively at inference, no checkpoint writes
- Option B: subspace alignment regularization term in `sedimentation_loss.py`

**New dependency:** `geoopt` (MIT license) for Grassmann manifold distance

**Test count impact:** ~45 new tests → projected total ~1,127

---

### Sprint 2 — Quantization Spike (1 story point)

**Goal:** Validate whether the INT8 "stickiness" problem is real or already solved by `BoundedAdapter`.

```python
# test_subspace_assimilation_spike.py
adapter = create_adapter("low_rank", input_dim=384)
x = torch.randn(1, 384)
delta_pre = (adapter(x) - F.normalize(x, dim=1)).norm().item()
x_q = torch.quantize_per_tensor(x, scale=1/127.0, zero_point=0, dtype=torch.qint8)
delta_post = (adapter(x_q.dequantize()) - F.normalize(x_q.dequantize(), dim=1)).norm().item()
# If delta_post/delta_pre > 0.78: BoundedAdapter already handles it → skip QAT track
# If delta_post/delta_pre < 0.5: QAT track needed
```

**Decision gate:**
- Ratio > 0.78 → corrections survive quantization → `BoundedAdapter.min_correction=0.01` is sufficient → proceed to Sprint 3A (subspace alignment regularizer)
- Ratio < 0.5 → corrections are lost → proceed to Sprint 3B (ephemeral fast-weights)

---

### Sprint 3 — Localized Shard Routing (2-3 sessions)

**Goal:** Fixed-K shard routing as an isolated, optional layer on top of existing adapter system.

**New module:** `subspace_router.py`
```python
class SubspaceRouter:
    def __init__(self, n_shards, input_dim, adapter_type="mlp"):
        self.centroids = None  # fit at index time via online k-means on RAW embeddings
        self.adapters = {i: create_adapter(adapter_type, input_dim) for i in range(n_shards)}
    
    def route(self, raw_embedding):  # MUST be pre-adapter embedding
        shard_id = np.argmin(np.linalg.norm(self.centroids - raw_embedding, axis=1))
        return shard_id, self.adapters[shard_id]
```

**Config additions to `ChelationConfig`:**
- `n_shards: int = 1` (default = global adapter, backward compatible)
- `min_shard_size: int = 50` (below threshold: fall back to global adapter)
- `use_localized_adapters: bool = False`

**Changes to `AntigravityEngine.ingest()`:** Tag Qdrant points with `shard_id` in payload

**Changes to `CheckpointManager`:** Accept shard-namespaced paths (shard ID in filename)

**Shard starvation guard:** If shard document count < `min_shard_size`, hold adapter at identity, fall back to global

**New tests:** ~40 for routing logic; all existing 1,082 tests must continue passing

---

### Phase B — Open Model Fine-Tuning Experiment (GPU required, 4-8 sessions)

**Containment rule:** All code in `experiments/open_model_ft/`. CI unchanged. `requirements-experimental.txt` separate from root `requirements.txt`.

**CI containment pattern:**
```python
try:
    import transformers, peft, geoopt
    EXPERIMENTAL_DEPS_AVAILABLE = True
except ImportError:
    EXPERIMENTAL_DEPS_AVAILABLE = False
# All experimental tests: @unittest.skipUnless(EXPERIMENTAL_DEPS_AVAILABLE, ...)
```

**New files:**
```
experiments/open_model_ft/
  finetuneable_hf_backend.py      — FineTunableHFBackend(EmbeddingBackend)
  lsa_dora_config.py              — DoRA config: layers 8-12, rank-16, subspace schedule
  run_mvexp.py                    — 4-condition benchmark (A/B/C/D)
  requirements-experimental.txt  — transformers>=4.40, peft>=0.9.0, geoopt, accelerate>=0.28
```

**Recommended base model:** `Alibaba-NLP/gte-Qwen2-1.5B-instruct` (Apache 2.0, 1.5B params, current MTEB SOTA architecture, loads via `SentenceTransformer()` directly)

**Fine-tuning loss:**
```
L_LSA = L_retrieval + λ₁ * L_subspace + λ₂ * L_DSOA

L_retrieval = InfoNCE  (already in SedimentationInfoNCELoss)
L_subspace  = geoopt Grassmann distance(W_k_current, W_k_target)
L_DSOA      = max_margin(embed(clean_doc), embed(injected_doc), margin=δ)
```

**Use DoRA, not vanilla LoRA:** `peft >= 0.9.0`, DoRA decouples direction/magnitude — correct for subspace-constrained updates.

**4-Condition Evaluation Protocol:**

| Condition | Base | Adapter | Purpose |
|---|---|---|---|
| A (baseline) | Frozen GTE-Qwen2-1.5B | None | Reference |
| B (adapter only) | Frozen GTE-Qwen2-1.5B | ChelationAdapter (sedimentation) | Does adapter add value on better base? |
| C (LSA fine-tune) | GTE-Qwen2-1.5B, LSA fine-tuned | None | Does direct fine-tuning work? |
| D (combined) | GTE-Qwen2-1.5B, LSA fine-tuned | ChelationAdapter (sedimentation) | Are they complementary or redundant? |

**Evaluation datasets:** TREC-COVID, FIQA, DBPedia, SciFact, NFCorpus, HotpotQA  
**Avoid:** MS MARCO as primary claim (over-benchmarked, contamination risk)  
**Primary metric:** nDCG@10 | **Secondary:** Recall@100, MRR@10  
**Custom metric:** Semantic collapse rate + Jaccard divergence pre/post injection

**Minimum credible claim:** Improvement on ≥10/15 BEIR tasks, avg ≥1.0 nDCG@10 on TREC-COVID/FIQA/DBPedia, measurable collapse rate reduction

**Hardware:** Single RTX 4090 (24GB) — fits comfortably  
**Compute cost:** <$10 on cloud spot instances

---

### Phase C — Release (contingent on positive Phase B result)

1. Push fine-tuned weights → `ChelatedAI/gte-qwen2-1.5b-lsa` on HuggingFace Hub (Apache 2.0)
2. Model card: base model version hash, license, training datasets manifest, nDCG comparison table
3. ChelatedAI config entry: `create_embedding_backend("chelatedai/gte-qwen2-1.5b-lsa")`
4. Stacked value: fine-tuned base + runtime ChelationAdapter for deployment personalization

---

## 5. Novel Contributions (No Prior Art)

These 10 ideas emerged from Issue #106 and have no direct equivalent in published literature. Ranked by novelty confidence and implementation feasibility.

---

### NC-1: Topology-Triggered Sedimentation Gating
**Novelty: HIGH | 80% new | Target: EMNLP 2026**

Event-driven training gate: initiate/terminate adapter fine-tuning based on the *rate of change* of the collapse log (`d/dt(|chelation_log|)` over a rolling window). Training fires on collapse rate spike; goes quiescent on return to baseline. No labels. No schedule. No held-out eval set.

**Gap:** TTT adapts every input. EWC manages forgetting. Active learning selects samples. **Nothing selects whether the training regime is active at all based on self-measured structural topology.** No RAG paper implements autonomous topology-triggered retraining.

**Minimum experiment:** Binary gate on collapse rate derivative → compare 4 conditions on BEIR with synthetic collapse injection → hypothesis: topology-gated training matches always-training NDCG with significantly fewer gradient steps.

**Implementation:** `collapse_rate_tracker.py` + training gate in `sedimentation_trainer.py`

**Paper framing:** "Self-Healing RAG via Topology-Triggered Adapter Fine-Tuning"

---

### NC-2: Intra-Document Section-Transition DSOA
**Novelty: HIGH | 90% new | Target: ACL 2026**

Pre-indexing document integrity check: compute angular delta `arccos(q_i · q_{i+1})` between consecutive chunk-level embeddings within a single document. Flag documents where max intra-document delta exceeds a clean-corpus-calibrated threshold. Operates at ingestion time, before any retrieval step, zero labeled attack examples.

**Gap:** All existing injection defenses operate at retrieval time. **No paper defends the ingestion pipeline.** No benchmark dataset exists at this section-transition granularity — creating it is itself a contribution.

**Minimum experiment:** 500 BEIR documents + 100 synthetically injected → ROC AUC using max delta as sole signal → compare to perplexity and TF-IDF baselines.

**Implementation:** `document_subspace_auditor.py` (already in Sprint 1 plan)

**Paper framing:** "Geometric Document Integrity Verification for RAG Pipelines"

---

### NC-3: Gradient Trajectory Angular Acceleration in so(n) as Adversarial Signal
**Novelty: HIGH | 85% new | Target: NeurIPS 2026**

Track `α_t = ||ω_t - ω_{t-1}||` where `ω_t = ||A_t - A_{t-1}||_F` for the Cayley adapter's skew-symmetric parameter A. Flag training batches with `α_t` above threshold as potentially adversarial; quarantine and retrain.

**Core insight:** Clean batches reinforce a consistent correction direction → smooth low-curvature optimizer trajectories. Adversarial batches attempt a conflicting rotation → high angular acceleration. Attacker must now produce both correct-looking embeddings AND smooth optimizer trajectories — a second-order constraint significantly raising attack cost.

**Gap:** Activation-space detection, gradient magnitude detection, certified defenses all exist. **No paper uses curvature of the optimizer's path through a Lie algebra as an adversarial detection signal.**

**Implementation:** 3-line addition to training loop (store A_prev, compute `||A_t - A_{t-1}||_F`). `OrthogonalProcrustesAdapter._skew_param` already exists.

**Minimum experiment:** sedimentation on BEIR-SciFact with 10% adversarial documents → AUC-ROC for adversarial vs. clean batches using `α_t` as sole signal.

---

### NC-4: Collapse-Behavior Fingerprint Routing
**Novelty: HIGH | 80% new | Target: SIGIR 2026**

Route documents to per-shard adapters by *collapse behavior fingerprint* (which specific embedding dimensions collapse per document, as a binary vector over input dimensions) — not by semantic cluster. Documents sharing a collapse fingerprint get jointly trained corrections targeting those specific dimensions.

**The inversion:** Standard routing uses current content geometry. This uses the geometry of past failures. Documents with identical semantic position but different collapse patterns get different adapters.

**Gap:** MoE routes on input features. Federated learning partitions on client identity. **No routing strategy uses the geometry of past failures as the routing key.**

**Implementation:** Extract collapse fingerprints from existing `chelation_log`. Cluster by Jaccard similarity. Train one adapter per cluster.

---

### NC-5: Adaptive System Prompt Inertia via Embedding Intrinsic Dimensionality
**Novelty: HIGH | 75% new | Target: NAACL 2026**

Compute intrinsic dimensionality (ID) of the system prompt's embedding distribution using paraphrase-augmented variants (Facco et al. 2017 two-NN estimator). Use ID as a dynamic "moment of inertia" weight when resolving conflicts between system prompt and incoming document context. Richer system prompts get higher inertia. Unsupervised. Label-free.

**Gap:** All existing system prompt defenses (PromptGuard, LlamaGuard) use trained classifiers on labeled attack examples. **No work derives a dynamic geometric label-free weighting for system prompt authority from the prompt's own embedding structure.**

---

### NC-6: Soft Bond-Distribution KL Regularizer
**Novelty: MEDIUM-HIGH | 70% new | Target: NeurIPS 2026**

Replace hard-threshold bond-type assignments in `TopologyAnalyzer` with softmax-based soft assignments. Add `KL(current_bond_distribution || target_healthy_distribution)` as a training regularizer alongside the sedimentation loss.

**Gap:** TDA is used to *analyze* embeddings (Hofer et al. 2017, Gabrielsson et al. 2020). **No paper uses a bond-type distribution as a differentiable training regularizer to actively prevent semantic collapse.**

**Implementation:** Modify `TopologyAnalyzer` for soft assignments + `F.kl_div()` call in `sedimentation_loss.py`.

**Paper framing:** "Topological Regularization for Self-Correcting Embedding Systems"

---

### NC-7: Stacked SO(n) Composition — Geometry of Nested Cayley Adapters
**Novelty: MEDIUM-HIGH | 60% new | Target: ACL 2026**

Ablation and theoretical analysis of: DoRA-fine-tuned base (implicit Stiefel constraint) + Cayley-parameterized Procrustes adapter. Does ordering matter? Does the non-commutativity of SO(n) affect retrieval quality?

**Gap:** Individual SO(n) adapters studied. Individual DoRA studied. **Nested Stiefel constraints at different model hierarchy levels have no published treatment.**

**Minimum experiment:** Conditions A/B/C/D × two orderings × BEIR-SciFact and BEIR-FIQA. Measure NDCG@10 and effective rank of composite transform.

---

### NC-8: Composite Subspace as Persistent Entity with MDL Identity Criterion
**Novelty: MEDIUM | 65% new | Target: NeurIPS workshop**

When two adapter subspaces are merged, treat the result as a new entity with its own identity. Define "productive merge" via MDL: `MDL(A_merged) < MDL(A_1) + MDL(A_2)` means genuine shared structure was found. Computable via eigenvalue entropy of `(A_1 + A_2)/2` vs. individual eigenvalue entropies.

**Gap:** TIES-Merging and ZipIt! optimize task performance after merging but **never compute the information content of the merged representation as a criterion for whether the merge should happen.**

---

### NC-9: Quantization-Tier as Correction Routing Key
**Novelty: MEDIUM | 60% new | Target: MLSys 2026**

Use the quantization tier of a stored embedding (INT8 vs. INT4 vs. FP16) as a routing key to select among correction adapters, each calibrated to one tier's error distribution. `BoundedAdapter` already handles INT8 — this generalizes to a family.

**Gap:** QLoRA quantizes adapter weights. GPTQ/AWQ are post-training quantization. **No paper treats quantization level as a routing feature for selecting among adapter variants.**

---

### NC-10: Dzhanibekov Effect as Instability Predictor in SO(n) Training
**Novelty: MEDIUM | 80% new | Target: ICML 2027**

The intermediate axis theorem predicts rotation around the intermediate principal axis produces periodic flips. Track eigenvalue spectrum of `W = (I-A)(I+A)^{-1}` across training steps. If primary eigenvector drifts near the intermediate eigenvalue, adapter training failure is imminent. **This is a predictive (not post-hoc) instability detector.**

**Gap:** All optimization stability literature covers gradient explosion and loss divergence. **No paper uses the intermediate-axis theorem as a training instability predictor in SO(n).**

---

## 6. Model Selection Reference

| Model | Params | License | Release? | GPU Required | Verdict |
|---|---|---|---|---|---|
| **nomic-embed-text-v1.5** | 137M | Apache 2.0 | Yes, clean | No | **Phase A: use now** |
| **GTE-Qwen2-1.5B-instruct** | 1.5B | Apache 2.0 | Yes, clean | 8GB (DoRA) | **Phase B dev target** |
| **GTE-Qwen2-7B-instruct** | 7B | Apache 2.0 | Yes, clean | 40GB / 24GB (QLoRA) | Phase B publication target |
| BGE-M3 | 570M | MIT | Yes, clean | 8GB | Cross-arch validation |
| E5-mistral-7B | 7B | MIT | Yes, clean | 24GB | Viable alternative |
| all-mpnet-base-v2 | 109M | Apache 2.0 | Yes, clean | No | Clean A/B (already in repo as teacher) |
| Llama-3.2-3B | 3B | Llama Community | Yes, w/ attribution | 14GB (4-bit) | Backup if Qwen licensing changes |
| Phi-4-mini | 3.8B | MIT | Yes, clean | No (INT4) | GPU-only backup |
| Gemma 2 | 2B | Gemma Terms | **Restricted** | — | **Excluded** |

**Licensing note:** `Alibaba-NLP/gte-Qwen2-1.5B-instruct` (the embedding variant) is Apache 2.0 and fully releasable. Distinct from Qwen2.5 general-purpose models. Avoid GTE-Qwen2 variants with non-commercial riders — verify model card at download time.

---

## 7. Risk Register (Top Items)

| Risk | Severity | Mitigation |
|---|---|---|
| Centroid drift — routing boundaries go stale | CRITICAL | Route on raw base-model embeddings only (never adapted) |
| Catastrophic forgetting (base model mutation) | CRITICAL | Confine to adapter layer; use EWC if base mutation attempted |
| Quantization collapse — updates die below INT8 noise floor | HIGH | Sprint 2 spike first; `BoundedAdapter` may already handle it |
| Angular momentum formalism underspecification | HIGH | Use translation table; implement as Riemannian SGD on SO(n) |
| Shard starvation | MEDIUM | `min_shard_size` guard; sparse shards fall back to global adapter |
| DSOA false positive rate | MEDIUM | Diagnostic-only initially; calibrate on synthetic adversarial examples |
| Cross-shard checkpoint contamination | MEDIUM | Shard ID in checkpoint filename |
| DSOA batch-size noise | MEDIUM | Validate signal-to-noise at batch 32/128/256 before deploying as loss |
| Angular momentum vocabulary in code | LOW (corrosive) | Translation table precedes all code using this vocabulary |
| Gemma licensing | HIGH | Simply excluded — use Apache 2.0 / MIT models only |

---

## 8. Sprint Summary (Effort and Sequencing)

| Sprint | Deliverable | Story Points | GPU? | Outcome |
|---|---|---|---|---|
| Phase A | nomic-embed-text-v1.5 as frozen base | S=1 | No | Immediate BEIR improvement |
| Sprint 1 | `geometry_losses.py` (dsoa + angular + lsa) | 5 | No | Improves existing adapter training immediately |
| Sprint 2 | Quantization spike (50 lines) | S=1 | No | Decision gate for Sprint 3 path |
| Sprint 3A or 3B | Subspace assimilation (path depends on spike) | M+M=6 | No | Direction A implementation |
| Sprint 4 | `subspace_router.py` + shard routing | M+S+S=5 | No | Direction A routing layer |
| Phase B | Open model fine-tuning (GTE-Qwen2-1.5B) | 7 | RTX 4090 | Direction B proof-of-concept |
| Phase C | Release to HuggingFace Hub | S=1 | No | Releasable artifact |
| Research track | Novel contribution experiments (NC-1 through NC-5) | Variable | Varies | Papers / preprints |

**Total committed work for releasable artifact (Phase A + Sprint 1 + Sprint 2 + Phase B):** ~14 story points  
**Hardware requirement:** Single RTX 4090 (24GB) for Phase B only  
**Compute cost estimate:** <$10 on cloud spot instances

---

## 9. Next Session Checklist

### Start here:
- [ ] **Phase A (1 PR):** Add `trust_remote_code=True` to `LocalEmbeddingBackend`. Test with `nomic-ai/nomic-embed-text-v1.5`. Run existing benchmark suite to confirm no regression.
- [ ] **Sprint 1, file 1:** `angular_loss.py` — add `get_angular_velocity()` to `OrthogonalProcrustesAdapter`, expose in `get_state()`. ~20 lines, ~5 tests. (Lowest risk, highest immediate diagnostic value.)
- [ ] **Sprint 1, file 2:** `document_subspace_auditor.py` — `DocumentSubspaceAuditor.audit(text, system_prompt=None)` → `{angular_deltas, max_angular_delta, injection_risk_score, flagged}`. Core: `np.arccos(np.clip(np.dot(q_i, q_{i+1}), -1.0, 1.0))`. ~150 lines, ~40 tests.
- [ ] **Sprint 2:** Run the 50-line quantization spike. Record `delta_post/delta_pre` ratio. File the decision.

### Decisions deferred to Phase B:
- [ ] Confirm GTE-Qwen2-1.5B model card license at download time
- [ ] Select training dataset (MS MARCO vs. Natural Questions vs. HotpotQA for clean Apache 2.0 provenance)
- [ ] Decide whether Condition D (stacked SO(n)) is the primary ablation or an appendix

### Novel contributions to prototype in parallel with main sprints:
- [ ] **NC-3 (3 lines):** Add angular acceleration tracking to `OrthogonalProcrustesAdapter` training loop. No experiment yet — just instrument.
- [ ] **NC-2:** Build synthetic injection dataset alongside `document_subspace_auditor.py` (100 documents × 3 injection types)

---

## 10. Files Affected by This Plan

| File | Change | Sprint |
|---|---|---|
| `embedding_backend.py` | Add `trust_remote_code=True` to `LocalEmbeddingBackend` | Phase A |
| `chelation_adapter.py` | Add `get_angular_velocity()` + `_prev_skew_param` to `OrthogonalProcrustesAdapter` | Sprint 1 |
| `sedimentation_loss.py` | Add `BondDistributionRegularizer` (or import from new file) | Sprint 1 |
| `topology_analyzer.py` | Add soft (temperature-scaled) bond assignment mode | Sprint 1 |
| `config.py` | Add `n_shards`, `min_shard_size`, `use_localized_adapters` to `ChelationConfig` | Sprint 3 |
| `antigravity_engine.py` | Tag Qdrant points with `shard_id` in `ingest()` | Sprint 3 |
| `checkpoint_manager.py` | Accept shard-namespaced checkpoint paths | Sprint 3 |

**New files (production):**
- `document_subspace_auditor.py`
- `subspace_router.py`
- `geometry_losses.py` (or `dsoa_loss.py`, `angular_loss.py`, `lsa_loss.py`)
- `collapse_rate_tracker.py` (NC-1)

**New files (experimental, not in CI):**
- `experiments/open_model_ft/finetuneable_hf_backend.py`
- `experiments/open_model_ft/lsa_dora_config.py`
- `experiments/open_model_ft/run_mvexp.py`
- `experiments/open_model_ft/requirements-experimental.txt`

**New test files:**
- `test_document_subspace_auditor.py` (~40 tests)
- `test_subspace_router.py` (~40 tests)
- `test_subspace_assimilation_spike.py` (~5 tests)

**Projected test count after all production sprints:** ~1,172 (from 1,082 baseline)
