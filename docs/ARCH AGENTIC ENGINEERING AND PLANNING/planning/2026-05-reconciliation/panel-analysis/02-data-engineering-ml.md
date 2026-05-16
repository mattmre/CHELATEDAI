# Panel of Experts: Data Engineering & ML Review
## ChelatedAI Repository — Panel Report 02

**Date:** 2026-04-04
**Panel:** Data Engineering & ML
**Files Analyzed:**
- `teacher_distillation.py`
- `teacher_weight_scheduler.py`
- `cross_lingual_distillation.py`
- `language_detector.py`
- `sedimentation_loss.py`
- `sedimentation_trainer.py`
- `sedimentation.py`
- `kalman_lr_scheduler.py`
- `dimension_mask_predictor.py`
- `embedding_quality.py`
- `isomer_detector.py`
- `topology_analyzer.py`
- `docs/RESEARCH_TRACKS.md`
- `docs/disk-resident-llm-feasibility-2026-03-28.md`

---

## CONVENE PHASE

**Panel Mandate:** Identify every ML correctness issue, training instability risk, evaluation methodology flaw, reproducibility gap, and improvement opportunity across the adaptive vector search and self-correcting embedding pipeline.

**Core concern framing (pre-solo):** The system trains a lightweight adapter on top of a frozen embedding model, using a "sedimentation" metaphor to push embeddings away from noisy neighborhoods. The panel must assess whether the adapter is learning correct signal, whether the training loop is numerically stable, whether evaluation methodology is sound, and whether the distillation pipeline (teacher→student, cross-lingual, ensemble) introduces biases or correctness failures.

---

## SOLO REVIEWS

---

### Dr. Priya Lakshman — Principal ML Engineer

**Focus: ML architecture soundness, training stability, loss function design, convergence behavior, numerical stability.**

**F-ML-001 [CRITICAL] — InfoNCE: temperature=0.07 is dangerously small for typical batch sizes**
*File: `sedimentation_loss.py`, lines 26, 159*
The default InfoNCE temperature of 0.07 was designed for contrastive learning with large batches (512–65536 in SimCLR/MoCo). The sedimentation loop processes small batches of collapse targets — potentially 10–100 items. At temperature=0.07 with a batch of 10, the sim_matrix values are amplified ~14x, pushing softmax into near-one-hot territory. This means gradients become extremely sparse: only the near-duplicate pairs receive gradient signal, and all others are near-zero. This does not improve retrieval ranking; it causes mode collapse into the hardest pair in the batch. No batch-size-aware temperature scaling exists anywhere in the codebase.

**F-ML-002 [CRITICAL] — DimensionProjection `_init_near_identity` is broken for the Sequential (hidden) path**
*File: `teacher_distillation.py`, lines 42–46*
When `hidden_dim` is provided, the initialization applies `xavier_uniform_` then multiplies by `0.001`. For the two-layer path (teacher_dim → hidden_dim → student_dim), xavier initialization is applied independently to each linear layer — this is semantically correct. However, multiplying xavier weights by 0.001 does not produce a near-identity mapping; it produces a near-zero mapping. The composition of two near-zero matrices is near-zero-squared. On first forward pass, teacher embeddings will be projected to approximately zero, making the teacher provide zero (or near-zero) signal. The system silently falls back to student-only mode without logging this pathology.

**F-ML-003 [CRITICAL] — `generate_distillation_targets` detaches projection from computation graph**
*File: `teacher_distillation.py`, line 340*
`projected = self._projection.project_tensor(teacher_tensor)` returns a grad-carrying tensor, but then `teacher_embeds = projected.detach().numpy()` is called on line 340. The comment on `project_tensor` (line 68) explicitly states it "keeps the projection in the computation graph so its parameters receive gradients," but the detach immediately after defeats this purpose. The `DimensionProjection` parameters will never receive gradients through this path. The projection is an untrained identity approximation forever.

**F-ML-004 [HIGH] — `SedimentationHybridLoss`: loss scale mismatch between MSE and InfoNCE**
*File: `sedimentation_loss.py`, lines 81–93*
`loss_mse` is a mean-squared error in raw embedding space (typical range: 0.001–0.1 for unit-sphere embeddings). `loss_infonce` is cross-entropy over a softmaxed similarity matrix (typical range: log(batch_size), i.e., 2.3 for N=10, 4.6 for N=100). With `contrastive_weight=0.5` and `mse_weight=0.5`, the InfoNCE term dominates by 10–100x in practice. The blending is numerically well-formed but semantically incorrect: the "hybrid" is effectively pure InfoNCE. No normalization or loss-scale calibration exists.

**F-ML-005 [HIGH] — Adaptive scheduler: `decrease_factor` applied on improvement is semantically inverted**
*File: `teacher_weight_scheduler.py`, lines 83–90*
The scheduler reduces `current_weight` by `decrease_factor` when loss improves (`loss < _best_loss`). The rationale seems to be "if the student is learning well, rely less on the teacher." But this creates an unstable feedback loop: improvement causes less teacher guidance, which may degrade the student, causing loss to stop improving, which triggers `increase_factor`, restoring teacher weight. This oscillation was observed empirically (Session 29 notes). The schedule is not tested with loss curves that have natural valley shapes.

**F-ML-006 [HIGH] — Hierarchical sedimentation uses only MSE loss, ignoring the configurable loss**
*File: `sedimentation.py`, lines 134–135*
`criterion = torch.nn.MSELoss()` is hardcoded. The engine's `set_sedimentation_loss()` capability (supporting InfoNCE and hybrid) is bypassed in the hierarchical path. This creates a training/loss inconsistency: the main engine can train with InfoNCE, but the hierarchical variant always falls back to MSE. The `engine.sedimentation_loss` attribute is never consulted.

**F-ML-007 [HIGH] — No gradient clipping anywhere in any training loop**
*Files: `sedimentation.py` line 150, `dimension_mask_predictor.py` line 182*
Neither the hierarchical sedimentation optimizer nor the MaskPreTrainer optimizer applies gradient clipping. For the InfoNCE loss at small temperature (finding F-ML-001), this compounds: near-one-hot softmax outputs produce large gradient magnitudes on the dominant pair. Without clipping, a single pathological batch can cause weight explosion.

**F-ML-008 [HIGH] — MaskPreTrainer uses BCELoss on targets computed from `percentile(cluster_var, chelation_p)`**
*File: `dimension_mask_predictor.py`, lines 150–153, 179*
The teacher mask `(cluster_var < threshold).astype(float)` produces hard binary labels (0.0 or 1.0). BCELoss with a Sigmoid-activated predictor on hard binary targets creates flat gradient regions when the predictor is confidently correct (outputs near 0 or 1). This is the classic vanishing gradient problem with BCE on hard labels. Label smoothing (e.g., target = 0.9/0.1) or soft teacher labels would be more stable.

**F-ML-009 [MEDIUM] — `compute_homeostatic_target` push direction is not scale-invariant**
*File: `sedimentation_trainer.py`, lines 37–66*
`push_magnitude=0.1` is an absolute offset in the normalized vector space. After normalization, the function returns `homeostatic_target / target_norm`. But the push is applied in the pre-normalized space: `current_vec + (diff_norm * 0.1)`. The effective angular displacement depends on `|current_vec|`, which varies if the input is not unit-normalized. For a unit vector, pushing by 0.1 gives ~5.7° displacement. For a vector of norm 2, the same push gives ~2.9°. This inconsistency makes homeostatic strength dependent on input scale.

**F-ML-010 [MEDIUM] — No warmup for sedimentation optimizer**
*Files: `sedimentation.py` line 134, main engine training loop*
The Adam optimizer is initialized fresh every sedimentation cycle. No warmup schedule is applied. For a near-identity adapter (small std=0.001 initialization), the first few gradient steps may be large relative to the weight magnitude, causing initial instability before the loss landscape is understood.

**F-ML-011 [MEDIUM] — `DimensionProjection` near-identity init is wrong when `teacher_dim > student_dim`**
*File: `teacher_distillation.py`, lines 48–55*
For the non-sequential path, `self.projection.weight[:min_dim, :min_dim] = torch.eye(min_dim) * 0.999`. When `teacher_dim=768, student_dim=384`, the weight shape is `[384, 768]`. The init sets `weight[:384, :384] = eye(384)*0.999` and `weight[:384, 384:] ≈ 0.001*randn`. This means the first 384 teacher dims map approximately as identity to student, while the last 384 teacher dims are mapped with tiny near-zero weights. This is an implicit feature selection that discards half the teacher's signal without any justification.

**F-ML-012 [MEDIUM] — Global refinement uses `learning_rate * 0.1` but starts a new Adam optimizer**
*File: `sedimentation.py`, line 164*
Starting a fresh Adam optimizer resets momentum and second-moment estimates to zero. The `* 0.1` LR reduction is therefore not equivalent to learning rate decay in a continuous optimizer — it is a cold restart at lower LR. The first few global-refinement steps will have artificially inflated effective LR because Adam's denominator (second moment) is near-zero until it accumulates.

**F-ML-013 [LOW] — `EmbeddingQualityAssessor` quality formula is not calibrated**
*File: `embedding_quality.py`, lines 62–68*
`quality = 1.0 / (1.0 + raw_score)` where `raw_score = sum(decay^i for i in range(n))`. For `n=1, decay=0.95`: `raw_score=1.0, quality=0.5`. For `n=2`: `raw_score=1.95, quality=0.34`. A single chelation event immediately drops a document to quality=0.5, below the `high_threshold=0.8`. This means almost all documents in the chelation log are immediately "medium" quality after a single collision, which may cause over-aggressive adaptive threshold application.

**F-ML-014 [LOW] — KalmanLRScheduler uses population variance not sample variance**
*File: `kalman_lr_scheduler.py`, line 85*
`np.var(self._loss_history)` computes population variance (ddof=0). For a window of `window_size=10`, the first few steps have high variance not because loss is unstable but because the estimator has few samples. Should use `np.var(..., ddof=1)` for unbiased estimation, or apply a burn-in guard.

**F-ML-015 [LOW] — `TeacherWeightScheduler.step_decay` can decay to zero for large `num_decays`**
*File: `teacher_weight_scheduler.py`, lines 73–77*
`gamma=0.5, step_size=10` means weight halves every 10 steps. After 100 steps, weight = `initial_weight * 0.5^10 = initial_weight * 0.001`. The `min_weight=0.01` guard prevents complete collapse, but this effective floor may conflict with scenarios where `initial_weight * 0.5^k < 0.01` but the user intended the decay to stop naturally. The min_weight should be documented as an intentional design choice, not just an overflow guard.

**F-ML-016 [MEDIUM] — Training happens on ALL collapse targets, but Qdrant update is only for the trained subset**
*File: `sedimentation.py`, lines 71–83*
Vectors are retrieved only for `targets` (items with `len(v) >= threshold`). But `chelation_log.clear()` is called unconditionally at line 217, clearing entries for documents that were NOT trained on (because they had fewer than `threshold` collisions). This discards valid signal from sub-threshold documents without training on it.

**F-ML-017 [LOW] — No learning rate schedule for MaskPreTrainer**
*File: `dimension_mask_predictor.py`, line 178*
`optimizer = torch.optim.Adam(..., lr=self.learning_rate)`. Fixed LR for all epochs. For `buffer_size=1000` examples trained for `epochs=10`, this is likely fine, but for the online learning use case where buffer accumulates continuously, a decaying LR would prevent overfitting to stale examples.

**F-ML-018 [MEDIUM] — Topology distance normalization is incorrect**
*File: `topology_analyzer.py`, lines 308–312*
`max_change = 3.0 * np.sqrt(diff.size)`. The Frobenius norm of a matrix with all entries being 3 (max bond code difference) is `3 * sqrt(n*n) = 3*n`, not `3*sqrt(n*n)`. Wait — `sqrt(diff.size)` where `diff.size = n*n`, so `max_change = 3*n`. The Frobenius norm of the all-3 matrix is `3*n`. This is correct. However, the topology_distance value includes the diagonal (self-similarity pairs), which are always 0 change. The diagonal contributes to `diff.size` denominator but not numerator, biasing the metric downward.

**F-ML-019 [LOW] — `_simple_partition` can create empty clusters**
*File: `sedimentation.py`, line 239*
The recursion stops when `len(vectors) <= 1`, but the `n_clusters // 2` split can be 1 for small `n_clusters`. When a cluster has a single unique value in the split dimension, both halves have the same median, causing the `not np.any(left_mask) or not np.any(right_mask)` guard to fire — returning the whole set as one cluster. This means `n_clusters` is not guaranteed, and subsequent code iterates over whatever clusters exist (line 139 checks `len(cluster_indices) == 0`).

**F-ML-020 [MEDIUM] — No `torch.manual_seed` or `numpy.random.seed` in any training path**
*Files: `sedimentation.py`, `dimension_mask_predictor.py`, `teacher_distillation.py`*
All training loops are non-deterministic. Adam's initialization, dropout (if added later), and any stochastic augmentation will vary across runs. The CLAUDE.md states 1082 tests passing, but benchmark reproducibility is not guaranteed. There is no seed management utility, no experiment ID generator, and no hash-based reproducibility check.

---

### Carlos Mendez — MLOps Lead

**Focus: ML lifecycle, reproducibility, experiment tracking, model versioning, training/serving parity, data quality.**

**F-ML-021 [CRITICAL] — No experiment tracking: hyperparameters, loss curves, and adapter states are not logged to any artifact store**
*Files: `sedimentation.py`, `sedimentation_trainer.py`, `dimension_mask_predictor.py`*
The only logging is JSON events via `chelation_logger.py`. There is no MLflow, W&B, Neptune, or even CSV/JSON artifact logging of training runs. Given that Session 29 notes document a "Phase 1-validated sweet spot" of LR=0.01, threshold=1, this was discovered empirically with no recorded experiment trail. If this is wrong, there is no audit trail to identify what changed. The loss curves from `log_training_epoch` are emitted as log events only — not persisted as plottable artifacts.

**F-ML-022 [CRITICAL] — Adapter checkpoint naming has no experiment ID, creating overwrite collisions**
*Files: `antigravity_engine.py` (referenced), `sedimentation.py` line 180*
`self.engine.adapter.save(self.engine.adapter_path)` saves to a fixed path. Multiple training runs (with different thresholds, LRs, or loss functions) overwrite the same file. The CLAUDE.md Session 28 note mentions "adapter checkpoint contamination risk" was resolved by "isolated checkpoints per config," but `HierarchicalSedimentationEngine` still calls `save(self.engine.adapter_path)` directly with no config-derived path suffix. If two training sessions run sequentially, the second overwrites the first's best checkpoint.

**F-ML-023 [CRITICAL] — Training/serving parity broken: adapter trained on sedimentation targets, but inference uses raw adapter output**
*Files: `sedimentation.py` lines 186–187, main engine inference path*
`new_vectors_np = self.engine.adapter(input_tensor).numpy()` is called without normalization before upserting to Qdrant. But during inference, cosine similarity is used. If the adapter output is not normalized (which MSE loss does not enforce), stored vectors may have varying norms, making cosine similarity comparisons inconsistent between old (unit-norm) and new (adapter-output, non-unit-norm) vectors.

**F-ML-024 [HIGH] — `EnsembleTeacherHelper` parallel encoding has a silent race condition**
*File: `teacher_distillation.py`, lines 503–528*
The `ThreadPoolExecutor` collects results via `executor.map(self._encode_single_teacher, work_items)`. If a teacher's `get_teacher_embeddings` modifies shared state (e.g., `self.teacher_model` lazy loading inside `load_teacher_model()`), two threads could both enter `load_teacher_model()` simultaneously for the same teacher object, potentially creating two model instances or corrupting the `self.teacher_model` reference. The lazy-load check `if self.teacher_model is not None: return` at line 126 is not protected by a lock.

**F-ML-025 [HIGH] — No versioning of the DimensionProjection layer alongside the adapter**
*File: `teacher_distillation.py`, lines 282–295*
`TeacherDistillationHelper._projection` is lazily created and stored in memory. It is never saved to disk. If training runs across sessions with teacher-student dimension mismatch, each new session creates a freshly-initialized (untrained) projection, discarding any knowledge the previous projection accumulated. This is a complete re-initialization bug for any stateful distillation workflow.

**F-ML-026 [HIGH] — `MaskPreTrainer._buffer` is never serialized**
*File: `dimension_mask_predictor.py`, lines 153–157*
The buffer accumulates training examples (`cluster_mean, cluster_var, teacher_mask` tuples) but is an in-memory list with no persistence. If the process restarts, all accumulated examples are lost. For online learning with `buffer_size=1000`, it could take many inference calls to re-accumulate training signal. No checkpoint/resume mechanism exists.

**F-ML-027 [HIGH] — Teacher model is loaded on the wrong device relative to adapter**
*File: `teacher_distillation.py`, lines 132–143*
`device = 'cuda' if torch.cuda.is_available() else 'cpu'`. The adapter (in `chelation_adapter.py`) is likely on CPU by default. If CUDA is available, teacher embeddings are computed on GPU, then `.numpy()` is called (which requires CPU tensor), causing an implicit GPU→CPU transfer at line 65 (`projected.detach().numpy()`). For GPU environments, this transfer happens on every call to `project_numpy`. The proper pattern is to ensure both model and adapter are on the same device and only move to CPU at the final output boundary.

**F-ML-028 [HIGH] — `sync_vectors_to_qdrant` `payload_map` optimization can silently serve stale payloads**
*File: `sedimentation_trainer.py`, lines 101–118*
When `payload_map` is provided (the F-031 optimization), the function uses it directly without refreshing from Qdrant. If any external process updates payload metadata between the initial `retrieve` (lines 73–82 of `sedimentation.py`) and the `upsert` (line 123 of `sedimentation_trainer.py`), those updates are silently overwritten. In production RAG systems, payload metadata may be updated frequently (e.g., document timestamps, relevance feedback). No staleness check is performed.

**F-ML-029 [MEDIUM] — Teacher model default is `all-MiniLM-L6-v2` (same as student in tests)**
*File: `teacher_distillation.py`, line 85*
Session 29 notes explicitly document that "same-model distillation is a no-op." The default `teacher_model_name="sentence-transformers/all-MiniLM-L6-v2"` matches the test model `model_name="all-MiniLM-L6-v2"`. Any test or production setup that relies on defaults will silently run zero-signal distillation. The correct default should be the `ChelationConfig.DEFAULT_TEACHER_MODEL` (which should be a different model).

**F-ML-030 [MEDIUM] — `CrossLingualTeacherRouter._teachers` dictionary grows without bound**
*File: `cross_lingual_distillation.py`, lines 123–136*
`_get_or_create_teacher()` creates a new `TeacherDistillationHelper` for each unique model name encountered. There is no eviction policy or maximum teacher count. In an adversarial or misconfigured environment where many different model names are mapped, this could load an unbounded number of sentence transformer models into memory. Each model is ~100MB+.

**F-ML-031 [MEDIUM] — No data quality checks on teacher embeddings before blending**
*File: `teacher_distillation.py`, lines 353–360*
The blend `(1 - alpha) * current_embeddings + alpha * teacher_embeds` proceeds without checking for NaN or Inf in `teacher_embeds`. If the teacher model produces degenerate outputs (e.g., empty string input, tokenizer truncation artifacts, or OOM-induced partial results), the blend silently incorporates them. The post-blend normalization (`norms = np.maximum(norms, 1e-9)`) masks NaN propagation rather than detecting it.

**F-ML-032 [MEDIUM] — No checkpoint integrity check after save in hierarchical sedimentation**
*File: `sedimentation.py`, lines 180–197*
`self.engine.adapter.save(self.engine.adapter_path)` is called, but there is no reload-and-verify step (e.g., load and check weight hash, compare output on a test input before/after). `SafeTrainingContext` (F-043) provides rollback semantics, but the save itself is not verified. A disk I/O failure during write could leave a corrupt partial checkpoint that the rollback logic would not detect.

**F-ML-033 [LOW] — `KalmanLRScheduler` is not applied to any optimizer parameter group**
*File: `kalman_lr_scheduler.py`, referenced from `antigravity_engine.py`*
`KalmanLRScheduler.step(loss)` returns a new LR but does not modify any optimizer in-place. The caller must manually call `for param_group in optimizer.param_groups: param_group['lr'] = new_lr`. This is a common PyTorch pattern, but there is no integration test confirming the LR is actually applied. If the caller forgets to apply the returned LR, the scheduler runs silently without effect.

**F-ML-034 [LOW] — Teacher encoding is always done with `normalize_embeddings=True`**
*File: `teacher_distillation.py`, line 184*
Forcing normalization at encoding time means teacher embeddings always have unit norm. Student embeddings (from the base model) may or may not be normalized depending on the backend. If the student uses raw un-normalized sentence-transformers output, the blend `(1-alpha)*student + alpha*teacher` mixes different-scale vectors, making the blend ratio `alpha` scale-sensitive.

**F-ML-035 [LOW] — No batching in `LanguageDetector.detect_batch`**
*File: `language_detector.py`, lines 158–168*
`detect_batch` loops over texts one at a time: `return [self.detect(text) for text in texts]`. `langdetect` is a statistical model that could benefit from batch processing or at least skip cache-hit texts. The cache is per-text string, so repeated texts benefit, but the first-pass detection is entirely serial.

---

### Dr. Adaeze Okonkwo — Senior Data Scientist

**Focus: Statistical validity, benchmark design, evaluation methodology, data leakage, metric selection.**

**F-ML-036 [CRITICAL] — Isomer detection uses Jaccard on ranked lists without position weighting**
*File: `isomer_detector.py`, lines 53–72*
`compute_jaccard(set_a, set_b)` treats all retrieved documents as a set. Position matters: a swap between rank-1 and rank-2 has the same Jaccard impact as a swap between rank-1 and rank-100. For retrieval quality assessment, rank-biased overlap (RBO) or normalized discounted cumulative gain (NDCG) would be far more appropriate. Jaccard measures set difference, not quality degradation. A system that shifts the same documents from rank-1 to rank-10 shows zero isomer signal.

**F-ML-037 [CRITICAL] — `EmbeddingQualityAssessor` conflates retrieval noise with document quality**
*File: `embedding_quality.py`, lines 54–70*
A document appearing frequently in chelation logs (noise clusters) does not mean the document is low quality — it means its embedding is numerically close to other embeddings. This conflation is fundamental: an important document that happens to share semantic space with many others will receive a low "quality" score and be subjected to stricter retrieval thresholds, effectively penalizing relevant documents in dense semantic neighborhoods. There is no validation against human relevance judgments.

**F-ML-038 [HIGH] — Topology bond thresholds (covalent=0.90, hydrogen=0.70) are not calibrated to the actual embedding distribution**
*File: `topology_analyzer.py`, lines 46–63*
The thresholds are hardcoded domain analogies from chemistry. In practice, embedding space cosine similarity distributions vary dramatically by model, domain, and corpus. For `all-MiniLM-L6-v2` on typical RAG corpora, mean pairwise cosine similarity may be 0.3–0.5, meaning most pairs are "vdw" or "none" bonds by this classification. The "covalent" category may be nearly empty for healthy embeddings, making it useless as a collapse signal. No calibration against actual embedding distributions is documented.

**F-ML-039 [HIGH] — `compute_topology_change` computes `collapse_pressure` on full corpus, not on chelation-flagged subset**
*File: `topology_analyzer.py`, lines 276–323*
`collapse_pressure = post_ratios[BOND_COVALENT] - pre_ratios[BOND_COVALENT]` is computed over all embeddings. The signal of interest is specifically in the collapse-flagged neighborhoods. Computing over the full corpus dilutes the signal from problematic regions. A corpus with 1000 documents and 10 collapsing documents would show near-zero corpus-level collapse_pressure even if the 10-document neighborhood has total collapse.

**F-ML-040 [HIGH] — No held-out evaluation set for sedimentation training**
*Files: `sedimentation.py`, `sedimentation_trainer.py`*
The sedimentation loop trains on ALL collapse targets and updates ALL their vectors. There is no train/validation split to monitor for overfitting or regression. The concept of "training" on the very documents that will be retrieved creates a direct data leakage path: the adapter learns to minimize loss on exactly the vectors it will be evaluated on.

**F-ML-041 [HIGH] — Chelation log is the sole training signal source with no de-duplication**
*File: `sedimentation.py`, lines 52–53*
`targets = {k: v for k, v in self.engine.chelation_log.items() if len(v) >= threshold}`. The chelation log is a dict of `doc_id → [noise_vectors]`. If the same document is retrieved many times in similar queries, its collision count grows, giving it disproportionate weight in training. No deduplication of identical noise vectors is performed. A single query repeated 100 times would make one document appear as the most "collapsed," regardless of actual semantic health.

**F-ML-042 [HIGH] — Alignment metric in `compute_alignment_metric` uses cosine similarity between student and projected teacher — circular evaluation**
*File: `teacher_distillation.py`, lines 372–418*
The projection itself is trained (or initialized near-identity) to map teacher→student space. Measuring cosine similarity between student and projected-teacher embeddings measures how well the projection was trained, not how aligned the student is with the teacher's actual semantic structure. A projection that maps everything to the same vector would achieve perfect alignment score while destroying semantic content.

**F-ML-043 [HIGH] — `find_similar_query_isomers` has O(n²) complexity with no warning**
*File: `isomer_detector.py`, lines 206–244*
`combinations(query_list, 2)` iterates over all pairs. For a query set of size 1000, this is ~500K comparisons. For 10K queries, ~50M. The function is called without size warnings or sampling. This is a silent O(n²) computational bomb in production or large-scale evaluation.

**F-ML-044 [MEDIUM] — `detect_sedimentation_isomers` compares only common queries, silently discarding queries unique to one set**
*File: `isomer_detector.py`, lines 150–153*
`common_queries = set(results_a.keys()) & set(results_b.keys())`. Queries in `pre_results` but not `post_results` (or vice versa) are silently discarded. In a real pre/post sedimentation comparison, query coverage differences may indicate queries that were pathologically affected by sedimentation (e.g., returning no results post-sedimentation). These should be surfaced, not silently dropped.

**F-ML-045 [MEDIUM] — `get_strength_distribution` computes std with numpy default (population std, ddof=0)**
*File: `isomer_detector.py`, line 273*
`float(np.std(arr))` uses population standard deviation. For small isomer batches (10–50 queries), sample std (ddof=1) would be more appropriate. This is a minor statistical bias but affects reported spread.

**F-ML-046 [MEDIUM] — Quality score computation uses all chelation events with equal temporal position assumed**
*File: `embedding_quality.py`, lines 62–67*
`sum(decay_factor^i for i in range(n))` assumes events are ordered from most-recent (i=0) to oldest (i=n-1). But the chelation log is a list that may not be time-ordered. If events are appended in arbitrary order, the decay weighting is meaningless. No timestamp is stored with each event.

**F-ML-047 [MEDIUM] — Topology change analysis does not control for corpus size changes**
*File: `topology_analyzer.py`, lines 276–323*
If sedimentation changes the total number of embeddings (e.g., documents are added or removed between pre/post snapshots), the bond ratio comparison is invalid. `bond_ratios` are normalized by the number of pairs (n*(n-1)), which changes with corpus size. The function does not check that `pre_embeddings.shape[0] == post_embeddings.shape[0]`.

**F-ML-048 [LOW] — Isomer history accumulation could cause memory leak over long running deployments**
*File: `isomer_detector.py`, lines 190–196*
`self._isomer_history.append({...})` is unbounded. After 1M inference calls, the history list could contain millions of entries. No maximum history length or rolling window is implemented. `reset()` is the only mitigation and must be called manually.

**F-ML-049 [LOW] — `compute_cluster_connectivity` reports `cluster_cohesion[label] = 1.0` for singleton clusters**
*File: `topology_analyzer.py`, lines 246–258*
Assigning cohesion=1.0 to a singleton cluster is mathematically undefined (a single element has no pairwise similarity with anything). Treating it as 1.0 inflates the mean cluster cohesion metric when many singletons exist. In high-dimensional sparse spaces (common in RAG), many documents may end up in singleton clusters post-sedimentation.

**F-ML-050 [LOW] — No statistical significance testing on isomer ratio or topology change metrics**
*Files: `isomer_detector.py`, `topology_analyzer.py`*
All reported metrics are point estimates with no confidence intervals. Comparing isomer ratios across runs (e.g., "isomer_ratio went from 0.15 to 0.12") has no statistical test to determine if the change is meaningful. For a corpus of 100 queries, a change from 15/100 to 12/100 isomers is not statistically significant (p≈0.5 by binomial test).

---

### Kenji Watanabe — ML Infrastructure Engineer

**Focus: Computational efficiency, memory usage, batch processing, GPU/CPU utilization, model serialization.**

**F-ML-051 [HIGH] — `TopologyAnalyzer.build_bond_matrix` computes full N×N similarity matrix for N embeddings**
*File: `topology_analyzer.py`, lines 92–110*
The full pairwise similarity matrix computation is O(N²) memory and O(N²·D) compute. For N=10,000 embeddings of D=384 dimensions, this allocates a 10K×10K float64 matrix (~800MB) plus intermediate computations. No chunked or approximate nearest-neighbor approach is provided. This will OOM for any meaningful corpus called at inference time.

**F-ML-052 [HIGH] — `EnsembleTeacherHelper` parallel encoding creates N model inference threads simultaneously**
*File: `teacher_distillation.py`, lines 503–513*
With `max_workers=4` and 4 teachers, all 4 SentenceTransformer models are encoding simultaneously. Each model may use PyTorch's internal thread pool. Nested thread pool contention (executor threads × PyTorch internal threads) can cause CPU oversubscription. For CPU-only inference, this may be slower than sequential encoding. GIL behavior in CPython partially mitigates this, but PyTorch's C++ layer releases the GIL.

**F-ML-053 [HIGH] — `_simple_partition` in hierarchical sedimentation is O(N log N) calls to `np.var` per level**
*File: `sedimentation.py`, lines 221–262*
Each recursive call computes `np.var(vectors, axis=0)` over the current partition. This is O(N×D) per level, with O(N log N) total across all levels (assuming balanced splits). For large collapse target sets (N=1000, D=384), this sums to ~10M floating point operations just for partitioning. No caching of variance computations is done.

**F-ML-054 [HIGH] — Teacher models are loaded once but never unloaded; model references persist after use**
*File: `cross_lingual_distillation.py`, lines 123–136; `teacher_distillation.py`, lines 124–167*
`TeacherDistillationHelper` holds a strong reference to `self.teacher_model` (a SentenceTransformer). Once loaded, it is never released even if the router/helper is no longer used for training. In long-running inference servers, this creates permanent memory occupation. No `unload()` or `__del__` method exists.

**F-ML-055 [MEDIUM] — `LanguageDetector` cache uses raw text strings as keys**
*File: `language_detector.py`, line 135*
`self._cache[text] = result` stores full text strings as dictionary keys. For long documents (paragraphs or whole passages), this creates large memory usage per cache entry. The cache key should be a hash (e.g., `hashlib.md5(text.encode()).hexdigest()`) to reduce memory per key by 10–100x.

**F-ML-056 [MEDIUM] — `_evict_half_cache` iterates over dictionary keys to evict, which is O(N/2) per eviction**
*File: `language_detector.py`, lines 148–156*
The eviction strategy collects keys to evict by iterating and checking index — this is O(N). Python dicts in 3.7+ maintain insertion order, so this evicts the oldest half, which is correct. However, the loop logic is fragile: it breaks when `idx >= len(self._cache) // 2`, but `len(self._cache)` is evaluated once outside the loop. Concurrent modifications (not thread-safe anyway) would break this. A `collections.OrderedDict` with a proper LRU policy would be more efficient.

**F-ML-057 [MEDIUM] — `get_teacher_embeddings` returns `np.array([])` for empty input but downstream code may not handle shape `(0,)`**
*File: `teacher_distillation.py`, line 201*
`return np.array([])` has shape `(0,)` not `(0, teacher_dim)`. Callers that check `len(teacher_embeds) == 0` (line 321) correctly identify this case, but callers that try to slice `teacher_embeds.shape[1]` (e.g., any code that does `if teacher_embeds.shape[1] != student_dim`) will raise `IndexError` on a 1D array. This is a latent shape mismatch bug.

**F-ML-058 [MEDIUM] — `input_tensor` and `target_tensor` for hierarchical training are created on CPU without device awareness**
*File: `sedimentation.py`, lines 108–109*
`torch.tensor(np.array(training_inputs), dtype=torch.float32)` creates CPU tensors. If CUDA is available and the adapter parameters are on GPU (possible if `eager_load=True` moves the teacher to GPU), the optimizer will fail with a device mismatch. No `.to(device)` call exists anywhere in the hierarchical training path.

**F-ML-059 [LOW] — `HierarchicalSedimentationEngine` creates a new `CheckpointManager` on every instantiation**
*File: `sedimentation.py`, line 37*
If the engine is created and destroyed repeatedly (e.g., in tests or batch processing), each instance creates a fresh `CheckpointManager`. This is wasteful if the checkpoint manager allocates file system resources on construction, and it prevents checkpoint reuse across engine instances.

**F-ML-060 [LOW] — `MaskPreTrainer.train` creates `means`, `variances`, `targets` tensors from full buffer every call**
*File: `dimension_mask_predictor.py`, lines 174–176*
Building the full tensor from the list of tuples on every call is O(buffer_size) memory allocation per training call. For `buffer_size=1000` with `input_dim=384`, this allocates 3 tensors of shape `(1000, 384)` = ~4.4MB each. This is minor but could be a `TensorDataset` with incremental updates instead.

---

### Sofia Reyes — Research Engineer

**Focus: Algorithm correctness, mathematical soundness, implementation vs. paper fidelity, theoretical guarantees.**

**F-ML-061 [CRITICAL] — InfoNCE implementation is not symmetric: outputs vs. targets**
*File: `sedimentation_loss.py`, lines 42–56*
Standard InfoNCE loss in contrastive learning is typically applied symmetrically: `L(a→b) + L(b→a)`. This implementation only computes `L(outputs→targets)`. For embedding alignment (distillation), the targets are fixed and do not receive gradients, so asymmetric InfoNCE is theoretically valid. However, the loss function name and docstring do not clarify this asymmetry. The actual InfoNCE guarantee (lower bound on mutual information) requires symmetric treatment for tight bounds. The current implementation provides weaker guarantees.

**F-ML-062 [CRITICAL] — Kalman filter analogy is mathematically unsound**
*File: `kalman_lr_scheduler.py`, lines 84–92*
The standard Kalman gain is `K = P·H^T · (H·P·H^T + R)^{-1}`. The implementation simplifies to `K = Q / (Q + R)` where `Q` is a scalar process noise and `R` is the empirical loss variance. This is a scalar 1D Kalman filter. The fundamental problem is that `R` (measurement noise) is computed from the *loss value sequence*, but in the Kalman filter, `R` should represent the variance of the *measurement equation noise*, not the variance of the quantity being tracked. Using loss variance as measurement noise is an incorrect analogy: high loss variance may indicate large true gradient signals (which should drive aggressive updates), not noisy measurements (which should drive conservative updates). The mathematical justification is inverted.

**F-ML-063 [CRITICAL] — `DimensionProjection._init_near_identity` contradicts itself for teacher_dim < student_dim**
*File: `teacher_distillation.py`, lines 48–55*
For teacher_dim < student_dim (e.g., teacher=128, student=384): `weight.shape = [384, 128]`. `weight[:128, :128] = eye(128)*0.999`. But `weight[128:, :128]` are initialized via `torch.randn * 0.001`. This means the projection maps the 128-dim teacher to the FIRST 128 dimensions of the student, leaving the remaining 256 student dimensions driven by near-zero random noise. The initialization silently pads with noise rather than zero-initializing the unmapped dimensions — making the initial projected teacher embeddings have small but non-zero components in dimensions with no teacher signal.

**F-ML-064 [HIGH] — Homeostatic push does not account for the unit sphere constraint**
*File: `sedimentation_trainer.py`, lines 52–53*
`homeostatic_target = current_vec + (diff_norm * push_magnitude)`. This is a Euclidean push in ambient space, not a geodesic push on the unit sphere. After normalization, the effective angular displacement is `arcsin(push_magnitude / |current_vec + diff_norm * push_magnitude|)`, which depends on the current vector's magnitude and direction relative to the push direction. For unit vectors, the actual angular displacement varies between 0 and 90° depending on the angle between `current_vec` and `diff_norm`. The system claims to control the "push magnitude" but actually controls a Euclidean offset, which translates to inconsistent angular changes.

**F-ML-065 [HIGH] — `SedimentationInfoNCELoss` batch acts as both positive pairs and negative set — contaminated negatives**
*File: `sedimentation_loss.py`, lines 42–56*
The similarity matrix `torch.mm(outputs_norm, targets_norm.t())` treats `target[j]` for `j≠i` as a negative for `output[i]`. But the targets themselves are derived from sedimentation homeostatic pushes — meaning `target[j]` is the "desired position" for document j. If documents i and j are semantically similar (both in the same semantic cluster), then `target[j]` is a semantically valid neighbor for `output[i]`. Using it as a hard negative is incorrect and will push semantically similar documents apart. This is the core "false negative problem" in contrastive learning, and it is unaddressed.

**F-ML-066 [HIGH] — Teacher weight blending is not theoretically sound for non-compatible embedding spaces**
*File: `teacher_distillation.py`, lines 353–360*
`blended = (1 - alpha) * current_embeddings + alpha * teacher_embeds`. Linear interpolation between two unit vectors produces a vector that is NOT on the unit sphere: `|u + v| = sqrt(2 + 2*cos(θ))`, which is ≤ 2 (it is only 1 when u=v). The post-blend normalization (line 358-360) correct the magnitude but does not account for the fact that the blend point is not the geodesic midpoint on the unit sphere. For embeddings that are far apart in angular distance (e.g., from different domains), the Euclidean blend point is deep inside the sphere, and normalization will project it outward to a semantically arbitrary location. Slerp (spherical linear interpolation) would be mathematically correct.

**F-ML-067 [HIGH] — `_simple_partition` uses median split, which does not guarantee balanced partitions for skewed distributions**
*File: `sedimentation.py`, lines 241–249*
The median split guarantees equal-sized left and right sets only for continuous distributions with a unique median. For embeddings with many identical values in the split dimension (common after normalization), many points may exactly equal the median, all going to the left side. The guard `if not np.any(left_mask) or not np.any(right_mask)` catches the complete-imbalance case, but highly skewed splits (e.g., 95/5) are not detected and can produce very unequal clusters.

**F-ML-068 [HIGH] — Language detection → teacher routing: language code "zh" maps to same teacher as "zh-cn" / "zh-tw"**
*File: `cross_lingual_distillation.py`, lines 41–51*
`LanguageTeacherMapping.get_teacher_for_language(lang)` does exact-match on language codes. `langdetect` may return `zh-cn` or `zh-tw` for Chinese text. If the mapping only has `"zh"` as a key, Simplified and Traditional Chinese will both fall through to the default teacher rather than a Chinese-capable model. No prefix-matching or normalization of language codes is implemented.

**F-ML-069 [MEDIUM] — `EmbeddingQualityAssessor.get_adaptive_threshold` returns un-bounded thresholds**
*File: `embedding_quality.py`, lines 89–107*
`scale = 1.0 + (1.0 - quality_score) * 2.0` ranges from 1.0 to 3.0. For `quality_score=0`, threshold is `3 * base_threshold`. There is no cap on the returned threshold, and the formula is not derived from any empirical calibration. Using an adaptive threshold of `3x base_threshold` for low-quality documents means they require extremely strong chelation evidence before triggering — the opposite of the desired behavior (low-quality documents should be chelated more aggressively, not less).

**Wait — re-reading:** Scale of 3x means `returned_threshold = 3 * base_threshold`. A higher threshold means the engine requires MORE collisions to trigger chelation. But the comment says "more aggressive chelation." This is semantically inverted: higher chelation threshold = less sensitive chelation = less frequent intervention. This is a logic bug.

**F-ML-070 [MEDIUM] — `TeacherWeightScheduler.step` for `cosine_annealing`: no warm restart support despite documentation**
*File: `teacher_weight_scheduler.py`, lines 65–70*
The docstring says "Cosine decay with optional warm restarts" but the implementation only does `progress = min(effective_step / effective_total, 1.0)` — a single cosine decay to `final_weight`. There is no restart mechanism. The scheduler will plateau at `final_weight` after `total_steps` without restarting.

**F-ML-071 [MEDIUM] — `IsomerDetector.compute_jaccard` returns 1.0 for two empty sets — misleading**
*File: `isomer_detector.py`, lines 67–72*
Two empty result sets are considered perfectly similar (Jaccard=1.0, isomer_strength=0.0). But two queries that both return empty results likely indicate a system failure, not stable retrieval. The caller should detect empty results as a distinct failure case rather than treating them as non-isomers.

**F-ML-072 [MEDIUM] — `_heuristic_detect` for Latin script: confidence is `latin_ratio * 0.6`**
*File: `language_detector.py`, lines 331–334*
Latin script is shared by dozens of languages (English, French, German, Spanish, Portuguese, etc.). The heuristic always returns `"en"` (via `default_language="en"`) for Latin-script text, with a capped confidence of 0.6. This means all Latin-script European languages (which represent a large fraction of real-world multilingual corpora) will be routed to the English teacher. Cross-lingual routing for European languages is fundamentally broken in heuristic mode.

**F-ML-073 [LOW] — `compute_topology_change` "collapse_pressure" is signed but may be negative for healthy sedimentation**
*File: `topology_analyzer.py`, line 299*
`collapse_pressure = post_ratios[BOND_COVALENT] - pre_ratios[BOND_COVALENT]`. If sedimentation succeeds in separating embeddings, this value is negative (fewer covalent bonds post-training). The metric name "pressure" implies a magnitude but it is signed. Callers comparing `collapse_pressure > threshold` will fail to detect that negative values mean improvement, not neutral behavior.

**F-ML-074 [LOW] — `variance-based partition` split dimension is fixed at argmax(var) — could be pathological**
*File: `sedimentation.py`, line 241*
`dim = int(np.argmax(np.var(vectors, axis=0)))` always splits on the highest-variance dimension. If one dimension has extreme outliers driving high variance but is otherwise uninformative for clustering (e.g., a constant dimension with one outlier), the split is misleading. PCA-based splitting (using the first principal component direction) would be more robust but adds sklearn dependency.

---

### Devil's Advocate — Contrarian Expert

**F-ML-075 [Dissent on F-ML-001] — Temperature=0.07 may be appropriate for this use case**
InfoNCE with temperature=0.07 in SimCLR is used with batches of 512–65536 AUGMENTED views of the same image. In this system, the InfoNCE batch consists of DISTINCT documents — the probability that any two documents in a sedimentation batch are semantically equivalent is low. Therefore, low temperature (sharper softmax) is not "wrong" — it's appropriate for a classification task where negatives are genuinely unrelated. The real problem is not temperature but batch size: small batches create a "limited negatives" problem regardless of temperature.

**F-ML-076 [Dissent on F-ML-062] — Kalman analogy may be "inspired by" not "identical to"**
The implementation is clearly labeled "Kalman-gain-inspired" — it does not claim to be a rigorous Kalman filter. The scalar gain K = Q/(Q+R) as a learning rate modulator has been used empirically in multiple online learning papers. The question is whether it works empirically, not whether it matches the derivation. The sign of the effect (high loss variance → lower LR) is defensible even if the theoretical basis is weak.

**F-ML-077 [Dissent on F-ML-037] — EmbeddingQualityAssessor is not claiming to measure semantic quality**
The class name says "quality" but the actual measure is "embedding stability" — how often this document appears in collision-causing neighborhoods. This is a distinct signal from semantic quality, and using it to adjust adaptive thresholds is a reasonable engineering choice. The concern should be named more precisely: the threshold adjustment direction (F-ML-069) is the bug, not the signal itself.

**F-ML-078 [Dissent on F-ML-036] — Jaccard is appropriate for boolean retrieval membership**
For the specific use case (detecting when sedimentation fundamentally changes which documents appear in top-k), Jaccard on the set is sufficient. The question "did the result set change?" is answered by Jaccard. The question "how much did quality change?" requires NDCG. These are different questions. The isomer detector is detecting structural change, not quality change — and for that purpose Jaccard is defensible.

**F-ML-079 [Dissent on F-ML-064] — Euclidean push is fine for near-unit-norm vectors**
In practice, sentence transformers normalize output embeddings. For vectors on or near the unit sphere, Euclidean push + re-normalization approximates geodesic push for small magnitudes (push_magnitude=0.1 << 1.0). The error is second-order in push_magnitude. This is a theoretical concern but probably has minimal practical impact for the 0.1 magnitude used.

**F-ML-080 [NEW CONCERN] — The entire sedimentation mechanism is circular: training targets are derived from the model's own outputs**
*File: `sedimentation_trainer.py`, lines 37–66*
The homeostatic target is computed from `current_vec` (the adapter's current output for this document) and `noise_vectors` (other documents' embeddings that are currently similar). The system trains the adapter to make `current_vec` dissimilar to its current neighbors. But after training, the adapter changes all vectors — meaning the neighbors may no longer be similar, and the training target was based on a stale neighborhood. This is a non-stationary optimization target problem: the landscape shifts under the gradient. The system could oscillate (push A away from B, then B away from A, ad infinitum) without convergence.

---

## CHALLENGE PHASE

### Challenge Log

**Challenge C-01: Dr. Lakshman challenges Dr. Okonkwo on F-ML-037**
Lakshman: F-ML-037 is correct but misses the deeper issue. The quality score formula `1/(1+raw_score)` means that once a document appears even once in the chelation log, its quality permanently degrades unless `clear()` is called. There is no forgetting mechanism proportional to document health improvement post-sedimentation. This is not just semantic conflation — it's a ratchet that guarantees monotonically degrading quality scores with no recovery path.

**Challenge C-02: Sofia Reyes challenges Kenji Watanabe on F-ML-051**
Reyes: F-ML-051 correctly identifies the O(N²) memory issue, but the proposed fix ("approximate nearest-neighbor") changes the semantics of topology analysis. Bond classification requires exact pairwise similarity; approximate methods would introduce classification errors at bond boundaries. The real fix is either restricting analysis to subsets (e.g., top-k nearest neighbors only) or using sparse matrix representations for the bond matrix after classification.

**Challenge C-03: Carlos Mendez challenges Dr. Lakshman on F-ML-003**
Mendez: F-ML-003 is correct about the detach breaking the gradient flow. But the actual intended use case is `generate_distillation_targets`, which produces numpy targets. These targets are then used in MSE loss computation with the adapter outputs. The projection gradients are not needed because the projection is a fixed preprocessing step (teacher embeddings → student-compatible space). The confusion arises from `project_tensor` existing at all — if the projection is never trained, `project_numpy` should be the only public method. The "co-training" comment in `project_tensor` misleads readers into thinking gradients flow when they don't.

**Challenge C-04: Devil's Advocate challenges Dr. Reyes on F-ML-065**
Devil's Advocate: The "false negative" problem in InfoNCE for the sedimentation context is mitigated by the fact that sedimentation batches are specifically selected for HIGH collision probability. Documents in the same sedimentation batch are known to be in the same noisy neighborhood, which means they ARE semantically similar and SHOULD NOT be pushed apart. F-ML-065 may be identifying a feature (within-batch positives are handled as in-batch contrastive learning) rather than a bug. However, if targets are the homeostatic push destinations (post-separation), then two targets that were previously collapsing should be pushed to DIFFERENT locations — making them valid negatives for each other. The correctness depends on whether the targets are the desired post-separation positions or the current (pre-separation) positions.

Response from Dr. Reyes: This challenge reveals a deeper issue: the system does not document whether targets represent desired post-separation embeddings or current embeddings adjusted by a push. From the code, `compute_homeostatic_target` returns a vector pushed AWAY from the noise cluster — this IS a post-separation target. Two documents in the same noisy cluster should have targets pointing in DIFFERENT directions (away from their respective noise centers). Using these diverging targets as negatives for each other is correct. Challenge withdrawn for the batch-contrastive concern, but F-ML-065 stands for the case where batch contains documents from different semantic domains.

**Challenge C-05: Dr. Okonkwo challenges Kenji Watanabe on F-ML-052**
Okonkwo: The thread pool contention concern (F-ML-052) is valid, but there is a larger evaluation concern: the ensemble is producing a weighted average of embeddings from different models with different semantic spaces. A weighted average of two vectors from non-aligned semantic spaces is not semantically meaningful. If `all-MiniLM-L6-v2` maps "dog" to direction A and `all-mpnet-base-v2` maps "dog" to direction B, averaging A and B produces a vector that may be closer to neither "dog" concept. The ensemble should use late fusion (re-rank on individual teacher scores) not early fusion (average teacher embeddings) for this to be semantically valid.

**Challenge C-06: Dr. Lakshman challenges Devil's Advocate on F-ML-080**
Lakshman agrees with Devil's Advocate F-ML-080 as a new finding and elevates it: this non-stationary target problem is the most fundamental theoretical issue in the entire system. The sedimentation cycle is a fixed-point iteration: it converges only if the map `T(v) = adapter(v)` has a stable fixed point. The homeostatic push explicitly tries to move embeddings AWAY from their current positions — meaning the desired fixed point may not be a fixed point of the adapter map. The system could be doing work (consuming compute, modifying weights) without converging to any stable state. This deserves CRITICAL classification.

---

## CONVERGE PHASE

**Stack-ranked findings by severity:**

---

## FULL FINDINGS LIST BY SEVERITY

### CRITICAL (11 findings)

| ID | Description | File | Lines |
|----|-------------|------|-------|
| F-ML-001 | InfoNCE temperature=0.07 unsafe for small sedimentation batches | `sedimentation_loss.py` | 26, 159 |
| F-ML-002 | DimensionProjection hidden-path init produces near-zero mapping | `teacher_distillation.py` | 42–46 |
| F-ML-003 | project_tensor gradient immediately detached — DimensionProjection never trains | `teacher_distillation.py` | 340 |
| F-ML-021 | No experiment tracking; no artifact store for hyperparameters and loss curves | `sedimentation.py` / pipeline-wide | — |
| F-ML-022 | Adapter checkpoint overwritten by each run — no experiment ID in path | `sedimentation.py` | 180 |
| F-ML-023 | Training/serving parity broken: adapter output not normalized before Qdrant upsert | `sedimentation.py` | 186–187 |
| F-ML-036 | Isomer Jaccard ignores position weighting — rank changes are invisible | `isomer_detector.py` | 53–72 |
| F-ML-061 | InfoNCE asymmetry undocumented; weaker mutual-information guarantees | `sedimentation_loss.py` | 42–56 |
| F-ML-062 | Kalman gain analogy is mathematically inverted: high variance should increase LR | `kalman_lr_scheduler.py` | 84–92 |
| F-ML-065 | InfoNCE false negative problem: semantically similar docs pushed apart | `sedimentation_loss.py` | 42–56 |
| F-ML-080 | Non-stationary optimization target: sedimentation may oscillate without convergence | `sedimentation_trainer.py` | 37–66 |

### HIGH (28 findings)

| ID | Description | File | Lines |
|----|-------------|------|-------|
| F-ML-004 | Hybrid loss: InfoNCE dominates MSE by 10–100x due to scale mismatch | `sedimentation_loss.py` | 81–93 |
| F-ML-005 | Adaptive scheduler: decrease_factor on improvement creates oscillation loop | `teacher_weight_scheduler.py` | 83–90 |
| F-ML-006 | Hierarchical sedimentation hardcodes MSE, ignores engine's configured loss | `sedimentation.py` | 134–135 |
| F-ML-007 | No gradient clipping in any training loop | `sedimentation.py`, `dimension_mask_predictor.py` | 150, 182 |
| F-ML-008 | BCELoss on hard binary targets → vanishing gradients in MaskPreTrainer | `dimension_mask_predictor.py` | 150–153, 179 |
| F-ML-024 | ThreadPoolExecutor race condition on lazy teacher model loading | `teacher_distillation.py` | 124–128, 503–513 |
| F-ML-025 | DimensionProjection layer not saved to disk — reinitializes each session | `teacher_distillation.py` | 282–295 |
| F-ML-026 | MaskPreTrainer._buffer never persisted; lost on restart | `dimension_mask_predictor.py` | 153–157 |
| F-ML-027 | Teacher on GPU, adapter on CPU — implicit device mismatch | `teacher_distillation.py` | 132–143 |
| F-ML-028 | payload_map optimization can silently serve stale Qdrant metadata | `sedimentation_trainer.py` | 101–118 |
| F-ML-037 | EmbeddingQualityAssessor conflates retrieval noise with document quality | `embedding_quality.py` | 54–70 |
| F-ML-038 | Topology bond thresholds not calibrated to actual embedding distribution | `topology_analyzer.py` | 46–63 |
| F-ML-039 | Collapse pressure diluted by computing over full corpus, not collapse subset | `topology_analyzer.py` | 276–323 |
| F-ML-040 | No train/validation split in sedimentation — direct data leakage | `sedimentation.py` / training loop | — |
| F-ML-041 | Chelation log not deduplicated — repeated queries inflate training weights | `sedimentation.py` | 52–53 |
| F-ML-042 | Alignment metric measures projection quality, not semantic alignment | `teacher_distillation.py` | 372–418 |
| F-ML-043 | find_similar_query_isomers is O(n²) with no warning or sampling | `isomer_detector.py` | 206–244 |
| F-ML-051 | TopologyAnalyzer builds full N×N similarity matrix — OOM for large corpora | `topology_analyzer.py` | 92–110 |
| F-ML-052 | Parallel teacher encoding causes nested thread pool contention | `teacher_distillation.py` | 503–513 |
| F-ML-053 | _simple_partition calls np.var O(N log N) times without caching | `sedimentation.py` | 221–262 |
| F-ML-054 | Teacher models loaded but never unloaded — permanent memory occupation | `cross_lingual_distillation.py` | 123–136 |
| F-ML-063 | DimensionProjection init for teacher_dim < student_dim pads with noise | `teacher_distillation.py` | 48–55 |
| F-ML-064 | Homeostatic push uses Euclidean displacement, not geodesic sphere push | `sedimentation_trainer.py` | 52–53 |
| F-ML-066 | Teacher weight blending is Euclidean, not slerp — arbitrary on-sphere result | `teacher_distillation.py` | 353–360 |
| F-ML-067 | Median split can produce highly skewed partitions for degenerate distributions | `sedimentation.py` | 241–249 |
| F-ML-068 | Language code exact-match fails for zh-cn/zh-tw variants | `cross_lingual_distillation.py` | 41–51 |
| F-ML-069 | get_adaptive_threshold direction is inverted — low quality means less chelation | `embedding_quality.py` | 89–107 |
| F-ML-072 | Heuristic detector routes all Latin-script languages to English teacher | `language_detector.py` | 331–334 |

### MEDIUM (22 findings)

| ID | Description | File | Lines |
|----|-------------|------|-------|
| F-ML-009 | Homeostatic push magnitude not scale-invariant to input norm | `sedimentation_trainer.py` | 37–66 |
| F-ML-010 | No optimizer warmup for near-identity adapter initialization | `sedimentation.py` | 134 |
| F-ML-011 | DimensionProjection init discards top-half teacher dims when teacher > student | `teacher_distillation.py` | 48–55 |
| F-ML-012 | Global refinement uses fresh Adam optimizer — momentum cold restart | `sedimentation.py` | 164 |
| F-ML-016 | chelation_log.clear() discards sub-threshold entries never trained on | `sedimentation.py` | 217 |
| F-ML-018 | Topology distance normalization includes diagonal — biased metric | `topology_analyzer.py` | 308–312 |
| F-ML-020 | No random seed management — training is non-deterministic | `sedimentation.py`, `dimension_mask_predictor.py` | — |
| F-ML-029 | Default teacher matches test student — silent zero-signal distillation | `teacher_distillation.py` | 85 |
| F-ML-030 | CrossLingualTeacherRouter._teachers grows without bound | `cross_lingual_distillation.py` | 123–136 |
| F-ML-031 | No NaN/Inf check on teacher embeddings before blending | `teacher_distillation.py` | 353–360 |
| F-ML-032 | Adapter save not verified after write | `sedimentation.py` | 180 |
| F-ML-044 | detect_sedimentation_isomers silently drops queries unique to one set | `isomer_detector.py` | 150–153 |
| F-ML-046 | Quality score decay assumes temporal ordering of chelation events | `embedding_quality.py` | 62–67 |
| F-ML-047 | Topology change analysis does not check corpus size consistency | `topology_analyzer.py` | 276–323 |
| F-ML-055 | Language cache uses full text strings as keys — high memory per entry | `language_detector.py` | 135 |
| F-ML-056 | Cache eviction implementation is fragile and O(N) | `language_detector.py` | 148–156 |
| F-ML-057 | get_teacher_embeddings returns shape (0,) not (0, dim) for empty input | `teacher_distillation.py` | 201 |
| F-ML-058 | Hierarchical training creates CPU tensors without device awareness | `sedimentation.py` | 108–109 |
| F-ML-070 | cosine_annealing schedule docstring claims warm restarts; not implemented | `teacher_weight_scheduler.py` | 65–70 |
| F-ML-071 | compute_jaccard returns 1.0 for two empty sets — masks query failures | `isomer_detector.py` | 67–72 |
| F-ML-073 | collapse_pressure signed metric — negative improvement values mislead callers | `topology_analyzer.py` | 299 |
| F-ML-074 | variance-based split dimension can be pathological outlier-driven | `sedimentation.py` | 241 |

### LOW (11 findings)

| ID | Description | File | Lines |
|----|-------------|------|-------|
| F-ML-013 | Quality formula: single chelation event immediately drops doc below high_threshold | `embedding_quality.py` | 62–68 |
| F-ML-014 | KalmanLRScheduler uses population variance (ddof=0), biased for small windows | `kalman_lr_scheduler.py` | 85 |
| F-ML-015 | step_decay gamma cascade undocumented interaction with min_weight floor | `teacher_weight_scheduler.py` | 73–77 |
| F-ML-017 | No LR schedule for MaskPreTrainer — fixed LR for all epochs | `dimension_mask_predictor.py` | 178 |
| F-ML-019 | _simple_partition cluster count not guaranteed for degenerate inputs | `sedimentation.py` | 239 |
| F-ML-033 | KalmanLRScheduler.step returns new LR but does not apply it to optimizer | `kalman_lr_scheduler.py` | — |
| F-ML-034 | Teacher encoding always normalize_embeddings=True; student may not be normalized | `teacher_distillation.py` | 184 |
| F-ML-035 | detect_batch is fully serial — no batching for langdetect | `language_detector.py` | 158–168 |
| F-ML-045 | Isomer strength_distribution uses population std | `isomer_detector.py` | 273 |
| F-ML-048 | isomer_history accumulates without bound — memory leak in long deployment | `isomer_detector.py` | 190–196 |
| F-ML-049 | singleton cluster cohesion=1.0 inflates mean cohesion metric | `topology_analyzer.py` | 246–258 |
| F-ML-059 | CheckpointManager created per-HierarchicalSedimentationEngine-instance | `sedimentation.py` | 37 |
| F-ML-060 | MaskPreTrainer.train allocates full buffer tensors each call | `dimension_mask_predictor.py` | 174–176 |

---

## DISSENT LOG

**Dissent D-01 (Devil's Advocate, from F-ML-075):** Temperature=0.07 may be appropriate when negatives are genuinely hard (different semantic topics in same batch). The batch composition matter more than the temperature value. *Consensus response: Partially accepted. The finding is reframed to emphasize batch size sensitivity rather than temperature being universally wrong. Recommendation is batch-size-aware temperature or a documented justification.*

**Dissent D-02 (Devil's Advocate, from F-ML-076):** Kalman analogy need not be rigorous to be useful. Empirical validation should be the arbiter. *Consensus response: Dissent noted but not accepted as mitigating the CRITICAL classification. The inversion of the effect direction (F-ML-062) is empirically testable and likely wrong: high loss variance often signals large gradient magnitude, which should favor higher LR, opposite to the current behavior.*

**Dissent D-03 (Devil's Advocate, from F-ML-077):** EmbeddingQualityAssessor measures embedding stability, not semantic quality. *Consensus response: Partially accepted. The class should be renamed to `EmbeddingStabilityAssessor`. But F-ML-069 (threshold direction inversion) remains a HIGH severity bug regardless of nomenclature.*

**Dissent D-04 (Devil's Advocate, from F-ML-078):** Jaccard measures structural change, not quality change — appropriate for isomer detection. *Consensus response: Accepted for the isomer detection use case. F-ML-036 downgraded from CRITICAL to HIGH for isomer_detector.py specifically, but CRITICAL classification stands for any downstream evaluation using isomer signal as a quality proxy.*

**Dissent D-05 (Kenji Watanabe, on F-ML-066):** Slerp is more expensive than Euclidean blend + normalize and the error is small for typical embedding angles. *Consensus response: Accepted as a practical tradeoff for small push magnitudes. HIGH classification maintained because cross-lingual embeddings from different model families can have large angular distances where the approximation error is non-negligible.*

---

## FEASIBILITY GATE

### Critical Findings: Impact / Effort / Risk / Dependencies

| ID | Finding | Impact | Effort | Risk | Deps |
|----|---------|--------|--------|------|------|
| F-ML-001 | InfoNCE temperature scaling | HIGH — could improve InfoNCE training significantly | LOW — add `temperature = max(0.07, math.log(batch_size) / 100)` or expose `auto_temperature` flag | LOW | None |
| F-ML-002 | DimensionProjection hidden-path init | HIGH — teacher provides near-zero signal through hidden path | LOW — change `module.weight.mul_(0.001)` to proper near-identity for each layer | LOW | None |
| F-ML-003 | project_tensor detach bug | HIGH — projection never learns | MEDIUM — decide if projection is trained or fixed; if fixed, remove `project_tensor`; if trained, remove detach at line 340 and add projection to optimizer | MEDIUM — behavior change | None |
| F-ML-021 | No experiment tracking | HIGH — cannot reproduce results | MEDIUM — add CSV/JSON run logging at minimum; MLflow integration preferred | LOW | MLflow or similar |
| F-ML-022 | Checkpoint overwrite | HIGH — previous runs lost | LOW — generate checkpoint path from `f"{adapter_path}_{hash(config)[:8]}.pt"` | LOW | None |
| F-ML-023 | Parity: upserted vectors not normalized | CRITICAL — cosine similarity broken post-training | LOW — add `new_vectors_np = new_vectors_np / np.linalg.norm(new_vectors_np, axis=1, keepdims=True)` before upsert | LOW | None |
| F-ML-062 | Kalman gain inversion | HIGH — LR may be wrong direction | MEDIUM — invert the logic: K = 1 - Q/(Q+R) or use a different uncertainty-to-LR mapping | MEDIUM — behavior change; needs ablation | None |
| F-ML-065 | InfoNCE false negatives | HIGH — pushes semantically similar docs apart | HIGH — requires mining-aware negative construction or hard negative masks | HIGH — architectural change | chelation_log structure |
| F-ML-080 | Non-stationary target convergence | CRITICAL — system may not converge | HIGH — requires convergence analysis, fixed-point theory, or convergence criterion | HIGH — fundamental design | All sedimentation code |
| F-ML-069 | Adaptive threshold direction inverted | HIGH — low quality docs under-chelated | LOW — change scale formula: `scale = 1.0 / (0.5 + quality_score)` or equivalent | LOW | None |
| F-ML-072 | Latin-script → English teacher routing | HIGH — EU languages misdirected | MEDIUM — add prefix matching for language codes; add explicit mappings for common EU languages | LOW | langdetect |

---

## PRIORITIZED REMEDIATION ROADMAP

### Phase 1: Immediate Correctness Fixes (1–2 days each, no architectural changes)

**P1-A: Fix upserted vector normalization (F-ML-023)**
In `sedimentation.py`, line 186, before `sync_vectors_to_qdrant`:
```python
norms = np.linalg.norm(new_vectors_np, axis=1, keepdims=True)
norms = np.maximum(norms, 1e-9)
new_vectors_np = new_vectors_np / norms
```
This is the most operationally dangerous bug: all Qdrant vectors after sedimentation have incorrect norms for cosine similarity.

**P1-B: Fix adaptive threshold direction (F-ML-069)**
In `embedding_quality.py`, line 104:
```python
# BEFORE: scale = 1.0 + (1.0 - quality_score) * 2.0  # Low quality → 3x threshold → less chelation
# AFTER: scale = 1.0 / (0.5 + quality_score * 0.5)    # Low quality → smaller threshold → more chelation
```

**P1-C: Fix default teacher model (F-ML-029)**
In `teacher_distillation.py`, line 85:
```python
# BEFORE: teacher_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
# AFTER:  teacher_model_name: str = ChelationConfig.DEFAULT_TEACHER_MODEL
```
And ensure `ChelationConfig.DEFAULT_TEACHER_MODEL` is set to a model different from the test student model.

**P1-D: Fix InfoNCE temperature for small batches (F-ML-001)**
In `sedimentation_loss.py`, add a batch-size-aware temperature method:
```python
def auto_temperature(self, batch_size: int) -> float:
    """Return recommended temperature for a given batch size."""
    return max(self.temperature, math.log(max(batch_size, 2)) / 50.0)
```
And use it in forward:
```python
temp = max(self.temperature, math.log(outputs.size(0)) / 50.0) if self.auto_scale else self.temperature
```

**P1-E: Fix hierarchical sedimentation using engine's configured loss (F-ML-006)**
In `sedimentation.py`, line 134–135:
```python
# BEFORE: criterion = torch.nn.MSELoss()
# AFTER:  criterion = getattr(self.engine, '_sedimentation_loss', torch.nn.MSELoss())
```

**P1-F: Add lock for thread-safe lazy loading (F-ML-024)**
In `TeacherDistillationHelper`, add a `threading.Lock`:
```python
import threading
self._load_lock = threading.Lock()

def load_teacher_model(self):
    with self._load_lock:
        if self.teacher_model is not None:
            return
        # ... existing load code
```

### Phase 2: Training Quality Improvements (1 week, needs validation runs)

**P2-A: Add gradient clipping to all training loops (F-ML-007)**
```python
# In sedimentation.py after loss.backward():
torch.nn.utils.clip_grad_norm_(self.engine.adapter.parameters(), max_norm=1.0)
```

**P2-B: Add label smoothing to MaskPreTrainer (F-ML-008)**
```python
# In dimension_mask_predictor.py, replace BCELoss:
criterion = nn.BCELoss()
# With:
smooth_eps = 0.1
targets_smooth = targets * (1 - smooth_eps) + smooth_eps / 2
criterion = nn.BCELoss()
loss = criterion(predictions, targets_smooth)
```

**P2-C: Add seed management utility**
Create `training_utils.py` with `set_all_seeds(seed: int)` that sets `torch.manual_seed`, `np.random.seed`, and Python `random.seed`. Call at the start of every training function.

**P2-D: Use `np.var(..., ddof=1)` in KalmanLRScheduler (F-ML-014)**
In `kalman_lr_scheduler.py`, line 85:
```python
R = float(np.var(self._loss_history, ddof=1))
```

**P2-E: Persist DimensionProjection to disk (F-ML-025)**
Add `save_projection(path)` and `load_projection(path)` methods to `TeacherDistillationHelper`. Call `save_projection` after training completes.

**P2-F: Add hash-based cache keys in LanguageDetector (F-ML-055)**
```python
import hashlib
cache_key = hashlib.md5(text.encode('utf-8', errors='replace')).hexdigest()
```

### Phase 3: Architecture Quality Improvements (1–2 weeks, may need ablation)

**P3-A: Address DimensionProjection training decision (F-ML-003)**
Decide and document: is `DimensionProjection` trained end-to-end or is it a fixed preprocessor?
- If fixed: remove `project_tensor`, keep only `project_numpy`. Document in class docstring.
- If trained: remove `detach()` at line 340, add projection to optimizer parameter group.

**P3-B: Address Kalman gain inversion (F-ML-062)**
Empirically test both formulations (current vs. inverted) on a controlled benchmark. If the inverted formulation (K = 1 - Q/(Q+R), or equivalently K = R/(Q+R)) improves training stability, switch and document the theoretical basis.

**P3-C: Add language code normalization in CrossLingualTeacherRouter (F-ML-068)**
```python
def _normalize_lang(lang: str) -> str:
    """Normalize language codes: zh-cn → zh, zh-tw → zh-hant, etc."""
    if '-' in lang:
        base = lang.split('-')[0].lower()
        return base  # Or use a full BCP-47 normalization table
    return lang.lower()
```
Apply in `_group_by_language` before routing.

**P3-D: Add variance-aware normalization to HybridLoss (F-ML-004)**
```python
# In SedimentationHybridLoss.forward:
with torch.no_grad():
    mse_scale = loss_mse.detach().item() + 1e-8
    nce_scale = loss_infonce.detach().item() + 1e-8
    ratio = mse_scale / nce_scale
loss = self.mse_weight * loss_mse + self.contrastive_weight * (loss_infonce * ratio)
```

**P3-E: Add slerp blending option for teacher targets (F-ML-066)**
```python
def _slerp(v0: np.ndarray, v1: np.ndarray, t: float) -> np.ndarray:
    """Spherical linear interpolation."""
    v0 = v0 / np.linalg.norm(v0)
    v1 = v1 / np.linalg.norm(v1)
    dot = np.clip(np.sum(v0 * v1, axis=-1, keepdims=True), -1.0, 1.0)
    theta = np.arccos(dot)
    sin_theta = np.sin(theta)
    if np.all(sin_theta < 1e-10):
        return v0
    return np.sin((1 - t) * theta) / sin_theta * v0 + np.sin(t * theta) / sin_theta * v1
```

**P3-F: Refactor to add held-out validation in sedimentation (F-ML-040)**
Reserve 20% of collapse targets as a validation set. Compute validation loss after each epoch. Implement early stopping if validation loss increases. This is essential to detect overfitting to the observed chelation events.

### Phase 4: Research-Level Improvements (Weeks to months)

**P4-A: Address InfoNCE false negative problem (F-ML-065)**
Implement an in-batch negative mask using the `chelation_log` structure:
```python
# Before computing InfoNCE loss, build a mask of valid negatives
valid_negatives = build_negative_mask(batch_doc_ids, chelation_log)
sim_matrix = sim_matrix * valid_negatives  # Zero out known false negatives
```

**P4-B: Investigate non-stationary convergence (F-ML-080)**
Implement a convergence test: measure the L2 distance between consecutive adapter weight states after each sedimentation cycle. If weights are still changing substantially after N cycles, the system has not found a stable fixed point. Expose this as a health metric in the dashboard.

**P4-C: Topology bond threshold calibration (F-ML-038)**
Run a corpus-level analysis to find the empirical similarity distribution percentiles (5th, 25th, 50th, 75th, 95th percentile pairwise cosine similarity) for the target embedding model. Set bond thresholds at calibrated percentile boundaries rather than fixed magic numbers.

**P4-D: Replace Jaccard with RBO for isomer strength measurement (F-ML-036)**
Implement Rank-Biased Overlap (Webber et al., 2010) as an alternative to Jaccard for position-sensitive isomer detection. Expose `similarity_metric="jaccard"|"rbo"` parameter in `IsomerDetector.__init__`.

**P4-E: Add MLflow or W&B experiment tracking (F-ML-021)**
Instrument all training paths with structured logging of: run ID, config hash, hyperparameters, per-epoch loss, final loss, adapter weight norm changes, validation loss if available. At minimum, produce per-run JSON summaries in `runs/` directory.

**P4-F: Ensemble embedding fusion fix (from C-05)**
Replace early-fusion (weighted average of teacher embeddings) with late-fusion (average cosine scores from individual teachers). This preserves the semantic geometry of each teacher's embedding space:
```python
# In EnsembleTeacherHelper.generate_distillation_targets:
# Instead of: blended = teacher_weight * teacher_embs + ...
# Compute individual teacher alignment scores and weight the final retrieval rankings
```

---

## EXECUTIVE SUMMARY: TOP 10 ML FINDINGS

1. **[CRITICAL] Training/serving parity broken (F-ML-023):** Adapter outputs written to Qdrant are not normalized before upsert. All post-sedimentation cosine similarity comparisons use unnormalized vectors mixed with normalized pre-sedimentation vectors. This is an active correctness bug affecting every sedimentation cycle. Fix: add 2 lines of normalization before `sync_vectors_to_qdrant`.

2. **[CRITICAL] Non-stationary optimization target (F-ML-080):** Sedimentation trains the adapter to push vectors away from their CURRENT neighbors, but training changes the neighbor structure. The optimization target shifts with every weight update. There is no theoretical guarantee the system converges — it may oscillate indefinitely. Mitigation: convergence monitoring on adapter weight stability across sedimentation cycles.

3. **[CRITICAL] Kalman gain direction inverted (F-ML-062):** High loss variance causes the Kalman scheduler to LOWER the learning rate. But high loss variance in gradient descent typically indicates a large, reliable gradient signal — exactly when aggressive updates are appropriate. The scheduler may be systematically fighting the gradient signal. Needs empirical ablation and likely sign reversal.

4. **[CRITICAL] InfoNCE false negative contamination (F-ML-065):** Semantically similar documents in the same sedimentation batch are used as negatives for each other. This causes the InfoNCE loss to actively push semantically similar documents apart, which is the opposite of the desired retrieval behavior. Requires in-batch negative masking using the chelation log.

5. **[CRITICAL] DimensionProjection never trains (F-ML-003):** `project_tensor` is called with `detach().numpy()` immediately after — gradients are discarded. The projection is described as trainable but is in practice a fixed, poorly-initialized preprocessor. Either train it properly by adding to optimizer groups, or document it as fixed and simplify the API.

6. **[HIGH] InfoNCE temperature unsafe for small batches (F-ML-001):** Temperature=0.07 produces near-one-hot softmax for batches of 10–100 documents typical in sedimentation. This creates extremely sparse gradients: only the hardest pair in the batch receives meaningful gradient signal. Recommendation: implement auto-temperature scaling with batch size.

7. **[HIGH] Hybrid loss scale mismatch (F-ML-004):** The InfoNCE term is 10–100x larger in magnitude than the MSE term in typical sedimentation scenarios. Despite 50/50 weighting, the hybrid loss is functionally pure InfoNCE. Dynamic loss-scale normalization is needed for the blend to have the intended effect.

8. **[HIGH] No train/validation split in sedimentation (F-ML-040):** The adapter is trained on the same documents it will later retrieve. This is direct data leakage. Without a held-out set, there is no way to distinguish generalization from memorization. Any reported benchmark improvements after sedimentation may reflect overfitting to the observed collision events.

9. **[HIGH] Adaptive threshold direction inverted (F-ML-069):** `EmbeddingQualityAssessor.get_adaptive_threshold` increases the chelation threshold for low-quality documents, making them LESS likely to be chelated. The documented intent is "more aggressive chelation for low-quality documents" — the implementation is inverted. This is a logic bug where low-quality embeddings are protected from the correction mechanism.

10. **[HIGH] Default teacher = student (F-ML-029):** The default `teacher_model_name` in `TeacherDistillationHelper` is `all-MiniLM-L6-v2`, the same model used as the student in tests and typical usage. Same-model distillation produces zero signal (Session 29 confirms this empirically). Any setup relying on defaults silently runs a no-op distillation loop, wasting compute without learning.

---

## PANEL METADATA

**Total findings produced:** 83 (not counting 6 challenge/dissent sub-findings)
- Critical: 11
- High: 28
- Medium: 22
- Low: 13 (inclusive of F-ML-059, F-ML-060)

**Files with most findings:**
- `teacher_distillation.py`: 15 findings
- `sedimentation.py`: 14 findings
- `sedimentation_loss.py`: 5 findings
- `topology_analyzer.py`: 8 findings
- `isomer_detector.py`: 7 findings
- `dimension_mask_predictor.py`: 6 findings
- `embedding_quality.py`: 5 findings
- `language_detector.py`: 5 findings
- `kalman_lr_scheduler.py`: 4 findings
- `cross_lingual_distillation.py`: 3 findings
- `teacher_weight_scheduler.py`: 3 findings
- `sedimentation_trainer.py`: 4 findings

**Panel consensus:** The most critical quality issue is the training/serving parity break (F-ML-023), which is a 2-line fix with immediate correctness impact. The most fundamental research issue is the non-stationary target problem (F-ML-080), which requires theoretical investigation. The most widespread category of issues is reproducibility (no seeds, no experiment tracking, no checkpoint versioning), which should be addressed as a coordinated infrastructure improvement rather than file-by-file.
