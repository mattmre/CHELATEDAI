# ChelatedAI Documentation Corpus — Claim Extraction

Scope: top-level `docs/` research/strategy docs (excluded: `chelation_opsd_research`, `brutal-honesty-kit`, `conventions`, `ARCH AGENTIC ENGINEERING AND PLANNING` contents, `computational_storage_poc`, and the `research-500-ai-agent-tuning` corpus).

Tracks: adaptive-retrieval, chelation, dimension-masking, spectral-reranking, sedimentation, self-healing, drift-detection, distillation, online-correction, quantization, evaluation, lattice, computational-storage, cross-lingual.

---

## 1. evolution-strategies-hyperscale-chelatedai-analysis.md

1. CLAIM: evo-1 :: distillation :: EGGROLL replaces full-rank random perturbations with low-rank perturbations E = A·Bᵀ/√r; the aggregate weighted update across a large population can become full-rank, and training then resembles batched inference. [Verify against the cited paper source.]
2. CLAIM: evo-2 :: quantization :: The EGGROLL paper demonstrates pure-integer (int8) language-model pretraining, RL benchmarks, foundation-model fine-tuning, and int8-quantized RWKV distillation/fine-tuning, and reports up to 91% of pure batch-inference throughput; only 1.25× additional compute for AttnRes-style gains via low-rank search. [Verify against the cited paper source.]
3. CLAIM: evo-3 :: online-correction :: Session 31's BoundedAdapter with correction bounds and an INT8 correction floor, SedimentationInfoNCELoss, KalmanLRScheduler, and DimensionProjection moved ChelatedAI toward bounded, quantization-aware, low-rank, parameter-efficient adaptation and is the strongest existing analogue to a low-rank ES path. [Verify in repo code/config.]
4. CLAIM: evo-4 :: evaluation :: Session 32 found a locally positive candidate (MLP adapter, all-mpnet-base-v2 teacher, teacher_weight 0.3; hybrid NDCG@10 0.6239 versus baseline 0.6012 on SciFact), but Session 33 rejected it on candidate-specific transfer (NFCorpus hybrid 0.4847 versus baseline 0.4893, -0.0046).
5. CLAIM: evo-5 :: computational-storage :: The defensible storage thesis is near-data candidate evaluation (vector shards, local nearest-neighbor filtering, sharded fitness scoring, scalar return to host), not SSD-only model execution; dense neural math still requires GPU/accelerator-class compute.

Research needs / gaps / questions:
- Is a Kalman-controlled low-rank Evolution Strategies optimizer for adapter-only retrieval fitness viable (population 8–32, rank 1)?
- Do ES candidates survive INT8/Qdrant quantization, with promotion blocked if gain disappears?
- Do ES candidates pass the Session-33 repeatability/transfer gate hierarchy before preset promotion?
- Can storage-sharded population scoring match host-side latency/quality in a deterministic mock?

## 2. frontier-adaptive-overlay-research-2026-05-05.md

1. CLAIM: fow-1 :: adaptive-retrieval :: ODAR (arXiv:2602.23681) routes between fast and slow reasoning agents using difficulty estimation and free-energy/risk-sensitive answer fusion; the doc adapts it as a difficulty-gated overlay-collection policy, not a copied agent harness.
2. CLAIM: fow-2 :: adaptive-retrieval :: Adaptive test-time compute allocation (arXiv:2604.14853) frames budgeted compute allocation as constrained optimization with a lightweight classifier imitating oracle allocation; the doc accepts it as a future overlay budget policy.
3. CLAIM: fow-3 :: drift-detection :: TIDE (arXiv:2602.02196) evaluates test-time improvement via trajectory dynamics, looping, and memory burden; the doc adopts trajectory-health diagnostics (loop/repeated-branch count, oracle-gap trend, blocker recurrence by channel, and budget per safe pass).
4. CLAIM: fow-4 :: self-healing :: DeepVerifier (arXiv:2601.15808) uses rubric-guided verification as a plug-in test-time module; the doc accepts it only as verifier/rubric evidence cards, not a default runtime controller, and defers automatic self-refinement loops.
5. CLAIM: fow-5 :: evaluation :: The cycle implemented `build_overlay_artifact_card`, requiring replay and holdout readiness and failing closed on missing or regressing holdout, plus verifier evidence cards and a budget-aware advisory collection policy; the next target is a Model-Scope smoke campaign with a supplied overlay report.

Research needs / gaps / questions:
- Which budget-aware overlay collection thresholds make observe-only, standard, and broadened collection safe?
- Do overlay trajectory-health metrics predict retrieval regressions before promotion?
- Is token-level routing (TARo, arXiv:2603.18411) worth revisiting once verifier signals are fine-grained?
- What provenance and rollback schema should a promoted overlay carry?

## 3. hybrid-distillation-research.md

1. CLAIM: hd-1 :: distillation :: SDFT (arXiv:2601.19897) proposes a sequential two-stage pipeline (pre-distillation on teacher embeddings, then task fine-tuning with contrastive loss) requiring labeled query-document pairs; the doc treats this as the reference teacher-distillation baseline.
2. CLAIM: hd-2 :: distillation :: ChelatedAI implements an adaptive homeostatic target, `target = (1−α)·homeostatic_target + α·teacher_embedding`, then normalizes it; `DEFAULT_TEACHER_WEIGHT = 0.5` and `DEFAULT_OFFLINE_EPOCHS = 15`, with α=0 equivalent to baseline and α=1 to offline teacher mode.
3. CLAIM: hd-3 :: online-correction :: ChelatedAI baselines are self-supervised via collapse detection (chelation log) rather than qrels or contrastive labels; the baseline homeostatic push is `current_vec + normalize(current_vec − avg_noise)·push_magnitude`.
4. CLAIM: hd-4 :: evaluation :: Teacher distillation presumes `teacher_dim == student_dim`; dimension mismatch is a hard failure with error logging and fallback to baseline, with a learnable projection layer left as future work.
5. CLAIM: hd-5 :: evaluation :: The hypothesis that information-sparse domains benefit from teacher_weight 0.7–0.9 while rich domains prefer 0.3–0.5 remains untested; later Session-28 evidence shows teacher-weight changes (0.3/0.5/0.7) produced no NDCG movement in a bounded run.

Research needs / gaps / questions:
- What is the optimal teacher_weight per domain?
- Does teacher guidance reduce measured collapse frequency?
- Does distillation lift hold across identical-dimension teachers only?
- What are the risks of teacher-bias amplification, offline overfitting, and runtime dependence on teacher availability?

## 4. current-research-eggroll-chelatedai-2026-04-27.md

1. CLAIM: es2-1 :: distillation :: Low-rank perturbations make black-box or zeroth-order optimization scale to billion-parameter models while looking like efficient batched inference; LOREN/LOZO-style work reinforces constraining search to low-rank or curvature-aware subspaces rather than all-parameter perturbations.
2. CLAIM: es2-2 :: quantization :: Recent quantized zeroth-order work (QuZO/QZO and Quantized Evolution Strategies) optimizes low-precision or quantization-constrained models with forward-pass-only feedback, aligning with the Session-31 finding that small MLP corrections can be invisible after INT8 quantization.
3. CLAIM: es2-3 :: online-correction :: Zero-order or ES updates should target adapter or mask-correction surfaces only (`LowRankAffineAdapter`, `BoundedAdapter`, and mask logits); base embedding models must not be mutated and current Adam paths stay defaults behind config/API flags.
4. CLAIM: es2-4 :: computational-storage :: Near-storage processing work for LLM inference and vector search (SmartSSD/FPGA-style offload and near-data graph ANN search) supports near-data retrieval filtering, candidate scoring, and sharded fitness evaluation, not passive-SSD replacement for GPU compute.
5. CLAIM: es2-5 :: evaluation :: The accepted design is a non-default low-rank ES adapter optimizer with quantization-aware fitness, Kalman-controlled perturbation scale, online micro-population updating, and storage-sharded population scoring, all gated by repeatability, transfer, and quantized-retrieval checks.

Research needs / gaps / questions:
- Which scalar fitness composition (NDCG gain, rank margin, collapse reduction, topology cohesion, isomer drift penalty, quantized score, and teacher agreement) best drives ES?
- What deterministic seed replay requirements and population-size/rank hyperparameters are needed?

## 5. llm-architecture-ai-engineering-adaptation-review-2026-04-27.md

1. CLAIM: la-1 :: adaptive-retrieval :: The cross-architecture review concludes frontier LLM gains come from adaptive sparsity, compressed memory, local/global context schedules, routing, normalization, and inference controls; it maps these to ChelatedAI as shared-plus-routed adapters, compact retrieval traces/masks, and local/global retrieval policy.
2. CLAIM: la-2 :: quantization :: DeepSeek-V3/R1/V3.2 (256 experts with shared expert and MLA KV compression), Qwen3/Qwen3-Next (Gated DeltaNet plus gated attention), and Kimi Linear (channel-wise memory gating with NoPE in MLA layers) are documented as archetypes for dense bases with routed specialist adapters and adaptive memory weighting.
3. CLAIM: la-3 :: drift-detection :: Recommended P0 gates are norm-drift metrics (query norm, retrieved-vector norm distribution, adapted-vector norm ratio, mask entropy, and route confidence) feeding `IntegratedDiagnosticsReport`, with online updates gated on norm stability.
4. CLAIM: la-4 :: evaluation :: Required operational gates are p50/p95/p99 latency, stage timings, cache hit rate, cost/query, peak memory, and quantization-retained gain; required quality gates are NDCG@K, MRR, Recall@K, structural health, isomer score, topology drift, and route-level fitness delta.
5. CLAIM: la-5 :: self-healing :: LLM-backed query reformulation, learned adapter routing, online adapter mutation from weak labels, persistent semantic-cache answers, storage-aware candidate promotion, speculative retrieval acceptance, and automatic global-anchor injection should not be enabled by default.

Research needs / gaps / questions:
- Is a local/global retrieval trigger based on health, isomer, or confidence degradation implementable without latency regression?
- Do attention-sink-style global anchors bias retrieval, requiring opt-in benchmarking?
- Is speculative retrieval acceptable without a verifier-fitness contract?

## 6. attnres-adapter-implementation-2026-05-04.md

1. CLAIM: ar-1 :: adaptive-retrieval :: MoonshotAI Attention Residuals (Block-AttnRes) replaces fixed residual accumulation with learned softmax attention over prior block outputs; the cited paper reports on Kimi Linear 48B/1.4T: GPQA-Diamond +7.5%, Mathematics +3.6%, HumanEval +3.1%, BBH +1.7%, and MMLU +1.1% at only 1.25× additional training compute.
2. CLAIM: ar-2 :: adaptive-retrieval :: `BlockAttnResAdapter` (`create_adapter("attnres")`, with 2/4/8 block presets) implements cross-block softmax attention over block states with near-identity initialization (std 0.001) and L2-normalized output, preserving cosine-metric and BoundedAdapter compatibility; the doc reports 37 tests passing.
3. CLAIM: ar-3 :: adaptive-retrieval :: `LayerAttentionAggregator` replaces last-layer extraction with learned softmax aggregation over mean-pooled per-layer embeddings and is wired into `model_scope_runtime` via optional raw-embedding capture.
4. CLAIM: ar-4 :: evaluation :: The road-course benchmark shows balanced AttnRes ties the deterministic SciFact MLP baseline (NDCG@10 = 0.816574 for baseline and all AttnRes profiles); no default promotion occurred and the quantization gate failed closed with retained gain 0.0.
5. CLAIM: ar-5 :: quantization :: Trained AttnRes profiles underperformed the untrained baseline in the failed-closed road-course benchmark; NFCorpus transfer improved balanced AttnRes NDCG, but the quantization-survival gate still failed.

Research needs / gaps / questions:
- Does `num_blocks` sensitivity change after sedimentation warmup?
- Can contrastive rather than InfoNCE sedimentation targets unlock AttnRes gains?
- Why do quantized-vector gains fail to survive, and what is repeat-seed variance?

## 7. attnres-road-course-benchmark-2026-05-04.md

1. CLAIM: arc-1 :: evaluation :: On a deterministic SciFact slice (20 queries, 1200 sampled docs, all-MiniLM-L6-v2), baseline, adaptive_p85_t0.01, attnres_baseline, and attnres_balanced_p85_t0.01 all produced NDCG@10 = 0.816574, MAP@10 = 0.771726, MRR = 0.770833, and Recall@10 = 0.95; the AttnRes path is live and non-regressing but non-improving before training.
2. CLAIM: arc-2 :: quantization :: The quantization-survival gate failed closed for the best profile: quantized NDCG@10 = 0.815590 with retained gain ratio 0.0, due to `fp32_gain_below_minimum`, `retained_gain_below_threshold`, and `quantized_fitness_below_baseline`.
3. CLAIM: arc-3 :: evaluation :: The `baseline_remains_best` promotion decision keeps default change disabled; the run is harness and safety validation, not evidence to promote AttnRes.

Research needs / gaps / questions:
- Does explicit shallow/balanced/deep `num_blocks` profiling after sedimentation warmup change the no-lift outcome?

## 8. model-scope-steering-architecture-2026-05-01.md

1. CLAIM: ms-1 :: adaptive-retrieval :: As of 2026-05-01, official Qwen-Scope artifacts were publicly visible for Qwen3.5-9B, Qwen3-8B, Qwen3.5-2B, and Qwen3-1.7B; verified open-weight Qwen3.6 releases were 2026-04-15 (Qwen3.6-35B-A3B) and 2026-04-22 (Qwen3.6-27B), while verified sparse-feature hooking remained on Qwen3/Qwen3.5, making Qwen3.5-9B the primary pilot.
2. CLAIM: ms-2 :: self-healing :: The proposed Model-Scope architecture is a base runtime plus hook bus, SAE feature codec, steering actuators, segmented memory (working/episode/persistent/expectation), comparator, and promotion/rollback; direct base-weight mutation is excluded from the first tranche.
3. CLAIM: ms-3 :: evaluation :: The claim boundary is local model hooking, sparse feature extraction, fail-closed steering overlays, segmented memory/replay, and bounded iterative training; it does not claim autonomous base-model rewriting, always-on self-modification, unrestricted long-horizon memory, or model-family-agnostic sparse steering.

Research needs / gaps / questions:
- Does SAE activation on Qwen3.5 transfer to retrieval bias in embedding space?
- What comparator thresholds justify persistence?
- What budgets and promotion/rollback semantics should steering artifacts use?

## 9. self-adapting-chelation-seal-eggroll-analysis-2026-04-28.md

1. CLAIM: seal-1 :: self-healing :: SEAL (arXiv:2506.10943) frames LLMs as static systems needing directives that transform new context into update data and update directives; training is a nested loop in which outer RL samples self-edits scored by downstream performance after an inner update loop, with ReSTEM-style reinforcement of useful edits.
2. CLAIM: seal-2 :: self-healing :: The paper reports knowledge incorporation improving no-context SQuAD from the low-30s percent to 47.0% after RL-trained self-edits; self-generated data outperformed GPT-4.1 synthetic data in the single-passage setting, and few-shot ARC-style adaptation improved self-edit success over no-adaptation baselines.
3. CLAIM: seal-3 :: quantization :: EGGROLL's int8/quantized training transfers as a `QuantizationPromotionGate` rule: reject any self-edit or accepted candidate whose FP32 improvement disappears under simulated quantization; a retention gate rejects edits that risk catastrophic forgetting.
4. CLAIM: seal-4 :: online-correction :: `SelfHealingChelationPlanner` and `build_self_healing_update_plan` produce JSON-safe plans with accepted/rejected directives, reasons, best fitness, and safety metadata; default mode is advisory and does not mutate base weights.
5. CLAIM: seal-5 :: evaluation :: `run_live_fire_diagnostics.py` scores self-healing updates with real retrieval fitness from `RetrievalFitnessEvaluator.evaluate_engine()` over generated probes instead of fixed synthetic reward deltas.

Research needs / gaps / questions:
- Does the current deterministic synthetic fixture approximate SEAL-quality generated data?
- Which adaptation mode (adapter_sft, online_contrastive_update, or eggroll_es) shows real lift under road-course gates?
- What is the cost of repeated self-edit reward loops?

## 10. seal-eggroll-multipanel-architecture-2026-04-28.md

1. CLAIM: mpan-1 :: self-healing :: The panel consensus is a gated, adapter-only self-healing loop: diagnostics plus context → self-edit directives → sandboxed adapter-only execution → retrieval reward, retention, quantization, health, and latency gates → ReSTEM filtering → ledger → road-course promotion; base weights stay frozen.
2. CLAIM: mpan-2 :: self-healing :: The first `self_healing_chelation.py` implementation is safe scaffolding but shallow proof; it adds `CandidateProvenanceLedger`, `SelfGeneratedEvalProbe`, `SelfEditSandboxExecutor`, `execute_shadow_round()`, and `run_adaptive_validation_loop()`.
3. CLAIM: mpan-3 :: evaluation :: Live-fire self-healing reward is real engine retrieval fitness over generated probes via `RetrievalFitnessEvaluator.evaluate_engine()`, recorded in a two-round adaptive validation loop with quantification gates preserved in ledger entries.
4. CLAIM: mpan-4 :: computational-storage :: The storage boundary remains scope-limited to sharded ANN, candidate metadata, deterministic seed/config replay, scalar local scoring, and latency-aware routing, not SSD-hosted LLM execution or GPU replacement.
5. CLAIM: mpan-5 :: evaluation :: The docs do not yet claim a learned SEAL self-edit policy, broad retrieval lift, safe persistent self-adaptation, storage-resident LLM training/inference, or default promotion beyond the road-course-supported 0.01 chelation guardrail.

Research needs / gaps / questions:
- Run multi-seed BEIR/multitask road-course campaigns comparing baseline, Adam, ES, and SEAL-guided ES.
- Add real retention-replay qrels and hard-negative probes before enabling persistent adapter updates.

## 11. road-course-results-2026-04-27.md

1. CLAIM: rc-1 :: evaluation :: Always-on centered chelation regresses retrieval on small-model SciFact: NDCG@10 0.4928 versus baseline 0.6745 (BEIR quick); the aggressive 0.0004 adaptive threshold fires CHELATE on every query and regresses NDCG by about 0.03–0.07 across 1k/5k-query loops.
2. CLAIM: rc-2 :: chelation :: The default-safe profile is `guard_p85_t0.01`, which ties baseline by staying on the FAST path (0/100 improved in every 100-window aggregate); `DEFAULT_CHELATION_THRESHOLD` changed from 0.0004 to 0.01 with `use_centering=False` and `use_quantization=False`, while 0.0004 is experimental.
3. CLAIM: rc-3 :: dimension-masking :: Calibrated controls work mechanically but every active percentile regressed NDCG; the transition band is about 0.002–0.003, p50 masks more dimensions than p85/p95, and temperature scaling cannot rescue centered chelation.
4. CLAIM: rc-4 :: adaptive-retrieval :: Reformulation (`reform_rrf_v2`, reciprocal-rank fusion) reliably changes rankings (78/100 SciFact and 38/100 NFCorpus) but not reliably toward improvement: window mean −0.0079 across 41 windows and negative fault counts; it is non-promotable except for weak task-conditional retests.
5. CLAIM: rc-5 :: evaluation :: A synthetic collapse fixture with one noisy dimension shows collapsed baseline NDCG@3/MRR/Recall@3 = 0.0 and known-mask recovery = 1.0; learned-mask smoke recovers collapse dimension id 4, but real static masks overfit and regularized conditional masks fail closed or show non-repeating holdout gains of only +0.001–0.0014.
6. CLAIM: rc-6 :: evaluation :: Learned/holdout gate training accepted 0 of 140 candidate rules, a fail-closed negative result; no safe global gate emerged from current window-level features.

Research needs / gaps / questions:
- Is a task/confidence-gated adaptive_p85_t0.002 branch repeatable outside seed noise?
- Do query-level attribution features enable a supervised gate with holdout windows and false-positive penalties?
- Does a reconstruction/mask objective provide denser positive supervision for classifier gates?

## 12. live-fire-diagnostics-2026-04-27.md

1. CLAIM: lf-1 :: evaluation :: A dependency-light deterministic harness with in-memory embeddings and fake Qdrant runs the full engine path end-to-end: baseline and live-fire retrieval fitness 1.0, structural health 1.0, quantization retained-gain 1.0, 17 runtime diagnostic events, 61 total events, and an adaptive-gate pass; overall result is warning by design.
2. CLAIM: lf-2 :: chelation :: The synthetic fixture forces CHELATE rate = 1.00, so production chelation thresholds must not be tuned from this fixture; the later road-course decision moved the default threshold to 0.01 and this calibration drift must be recorded.
3. CLAIM: lf-3 :: evaluation :: Calibration guidance sets quantization retention hard gate at 0.8; norm-drift watch at 0.75–1.33 with hard 0.5–2.0; route effectiveness warning below 0.5 and disable below 0.25 with at least 20 samples; structural health preferred at least 0.7 and no promotion below 0.6; promotion requires transfer and quantization survival of a positive gain.

Research needs / gaps / questions:
- Does chelation improve retrieval on non-saturated query sets?
- What threshold bands produce useful CHELATE rates?
- Do routed adapters improve retrieval or merely preserve baseline?
- Do FP32 gains survive INT8, and can online updates pass norm/drift/health gates?

## 13. research-2026-02-21-molecular-structure-comparison.md

1. CLAIM: mol-1 :: lattice :: The Molecular Structure of Thought paper (arXiv:2601.06002v2) models effective long CoT as a stable behavioral topology over covalent/deep-reasoning, hydrogen/self-reflection, and van-der-Waals/self-exploration bonds; it reports a transition-probability graph invariant across models/tasks and 81.72% of reflection steps reconnecting to previously formed semantic clusters.
2. CLAIM: mol-2 :: dimension-masking :: Independent literature validates the core premise: Dimension-Mask-Layer (2510.15308) cuts dimensions 40–50% with minimal loss; Random Dimension Removal (2508.17744) removes 50% with minimal impact; Query-Aware Adaptive Dimension Selection (2602.03306) learns per-query masking.
3. CLAIM: mol-3 :: drift-detection :: Semantic-collapse theory papers (entropic drift, length-induced collapse as a low-pass filter, and single-vector capacity limits) provide theoretical grounding for variance-based collapse detection and correction beyond static embeddings.
4. CLAIM: mol-4 :: online-correction :: Online-Optimized RAG (2509.20415) performs online gradient updates at inference; Drift-Adapter (EMNLP 2025, 2509.23471) reports 95–99% recall recovery with <10µs latency using Orthogonal-Procrustes residual transforms; TTARAG (2601.11443) performs test-time adaptation from retrieved passages; the doc judges ChelatedAI's offline sedimentation behind this direction.
5. CLAIM: mol-5 :: evaluation :: Iceberg benchmark (2512.12980) shows retrieval metrics can diverge from task-level metrics, motivating task-centric evaluation; the survey identifies the integration of dimension masking, spectral reranking, offline adapter training, and recursive decomposition as the distinctive contribution, with complexity overhead as the main risk.

Research needs / gaps / questions:
- Benchmark ChelatedAI head-to-head against Drift-Adapter and TempScale.
- Add convergence detection for sedimentation training.
- Compare learned importance-predictor masks with variance thresholds.
- Measure mask stability across queries, variance-distribution convergence, and SAE signal dimensions.

## 14. research-2026-02-27-sweep-analysis.md

1. CLAIM: sw-1 :: sedimentation :: An 81-configuration sedimentation sweep on SciFact (nomic-embed-text 768d, MLP adapter, chelation_p=85, baseline NDCG@10 0.58669) found no configuration above baseline; the best retained 99.9% of baseline (NDCG 0.58605) and 0/81 configurations improved.
2. CLAIM: sw-2 :: sedimentation :: Learning rate explains 61.5% of post-sedimentation NDCG variance; LR ≥ 0.1 is catastrophic, with 51/81 configurations collapsing to fixed-point NDCG 0.18696 and LR 0.1 without noise falling below 0.005.
3. CLAIM: sw-3 :: sedimentation :: Threshold explains 5.8% of variance; threshold ≥ 2 degrades quality 5–54% and threshold 3 causes destructive interference; noise_scale 0.05 is the most stable regularizer and fewer epochs are safer.
4. CLAIM: sw-4 :: chelation :: The safe operating envelope is narrow (LR≈0.01, threshold=1); near-identity adapter initialization (weight std 0.001) is the key safety property and high LR destroys it in the first epoch.
5. CLAIM: sw-5 :: evaluation :: The evidence-supported preset changes are threshold 1, epochs 5, noise_scale 0.05/0.2, and aggressive threshold capped at 2; `sweep_optimized` is LR 0.01, epochs 5, threshold 1, noise 0.2.

Research needs / gaps / questions:
- Locate the exact LR collapse boundary in 0.02–0.08.
- Test whether Procrustes or low-rank adapters resist collapse.
- Sweep push_magnitude and validate NFCorpus, FiQA, and TREC-COVID.
- Add patience-based convergence detection.

## 15. weight-refinement-campaign-results-2026-03-06-session28.md

1. CLAIM: wr-1 :: evaluation :: A bounded Session-28 campaign (MiniLM-L6-v2, 50 queries, three distillation cycles ×30 queries, three epochs, LR 0.001) found a positive local SciFact region: LR 0.01 / threshold 1 / noise 0.2 / epochs 5 gives NDCG@10 0.6289 → 0.6766 (+0.0477), but this is not promotable alone.
2. CLAIM: wr-2 :: distillation :: A Phase-2 teacher-weight study (0.3/0.5/0.7) produced identical mean NDCG@10 0.7553 for baseline, offline, and hybrid in every arm; offline pretraining added 171.7–216.3 seconds per arm without observed retrieval benefit.
3. CLAIM: wr-3 :: evaluation :: Phase-4 BEIR-small comparison found baseline mean NDCG@10 0.6839 versus chelation-family configurations at 0.5745–0.5748; online updates added latency (87.75 ms versus 23.67 ms baseline) without quality gains, so no chelation-family configuration is a promotion candidate.
4. CLAIM: wr-4 :: evaluation :: The campaign was interrupted during BEIR-medium; later phases were not completed. The conclusion is that the benchmark stack needed reproducibility hardening more than new features, and stable behavior alone is not transferable quality improvement.

Research needs / gaps / questions:
- Resume BEIR-medium/large phases.
- Do not repeat teacher-weight sweeps before recovering measurable gain.
- Identify reproducibility-hardening changes required before rerunning.

## Top-level index/blueprint docs

`docs/INDEX.md`, `docs/SYSTEM_BLUEPRINT.md`, and `docs/MODULE_GUIDE.md` are navigational/architectural guides, not research-claim documents. They repeat claims already extracted above (the 0.01 guardrail default, claim-boundary notes, and track inventory) and contain no novel verifiable research statements. `docs/next-session.md` contains carried-debt items but was outside the requested list and was not analyzed.

---

## CROSS-TRACK SYNTHESIS

The 10 strongest, most verifiable claims that synthesize across the corpus:

- S1 :: chelation :: Always-on or aggressive chelation is not a promotion candidate: centered chelation regresses NDCG@10 from 0.6745 to 0.4928 (BEIR quick, MiniLM/SciFact), the 0.0004 adaptive threshold regresses about 0.03–0.07 across 1k–5k-query loops, and every active percentile mask regressed.
- S2 :: evaluation :: The only safe default-setting decision supported by evidence so far is `guard_p85_t0.01` (`DEFAULT_CHELATION_THRESHOLD` changed 0.0004 to 0.01 with `use_centering=False`); it ties baseline by staying on the FAST path, and no golden/default-promotable profile emerged.
- S3 :: dimension-masking :: The chelation premise is validated only on controlled fixtures, not real slices: synthetic one-dimension collapse recovers from NDCG@3/MRR/Recall@3 = 0.0 to 1.0 when the known noisy dimension is masked, while real masks overfit, fail closed, or show non-repeating +0.001–0.0014 holdout gains.
- S4 :: sedimentation :: Sedimentation is a quality-preservation mechanism rather than an improvement mechanism: the best of 81 swept configurations retains 99.9% of baseline, 63% collapse to fixed-point NDCG 0.18696 at LR ≥ 0.1, and LR explains 61.5% of outcome variance.
- S5 :: distillation :: Teacher-weight strength (0.3/0.5/0.7) had zero measured NDCG impact in bounded runs (0.7553 in every arm), and distillation produced no lift; the normalized hybrid target defaults to α=0.5 and offline epochs 15.
- S6 :: self-healing :: SEAL plus EGGROLL is the clearest implementation route for self-healing chelation: SEAL contributes the self-edit outer/inner loop and ReSTEM filtering; EGGROLL contributes low-rank scalar-fitness ES; ChelatedAI contributes retrieval-fitness, retention, and quantization gates; base weights stay frozen and plans default to advisory mode.
- S7 :: quantization :: Quantization survival is a recurring hard fail-closed gate: the AttnRes road course retained-gain ratio is 0.0, no FP32 gain blocks every candidate, and Session-31/EGGROLL evidence shows small MLP corrections can be invisible after INT8.
- S8 :: online-correction :: The proposed non-default ES path is concrete and gated: low-rank ES adapter optimization, quantization-aware scalar fitness, Kalman-controlled perturbation scale, online micro-population search (population 8–32, rank 1), and storage-sharded scoring are promoted only after repeatability, transfer, and quantized-retrieval gates.
- S9 :: adaptive-retrieval :: Frontier overlay and Model-Scope directions are staged as budgeted, evidence-card-based, fail-closed additions rather than runtime defaults: replay-plus-holdout overlay cards, verifier/rubric cards, trajectory-health diagnostics, and Qwen3.5-9B SAE steering with bounded promotion/rollback and no base-weight mutation.
- S10 :: evaluation :: The corpus's most reliable finding is fail-closed promotion culture: trained AttnRes underperformed the untrained baseline, 0/140 learned-gate rules were accepted, reform_rrf_v2 changed 78/100 rankings without reliable improvement, and the gate hierarchy (repeatability → transfer → quantization → health/latency) rejected every candidate except the 0.01 guardrail.
