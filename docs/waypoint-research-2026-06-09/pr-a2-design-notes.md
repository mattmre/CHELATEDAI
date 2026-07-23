# PR-A2 Design Notes — anchor-supervised closed loop + mechanism fixes (query-encoder arena)

Local-only. Build AFTER PR-A1 (query_encoder_drift) and the flaky-test fix both merge, so
the harness base is stable. Branch off updated main.

## Harness facts (confirmed in run_drift_recovery_experiment.py)
- `CONDITIONS = ("C0","C1","C2","C3","C4")` (line 24).
- `_build_engine` (155): builds `AntigravityEngine(...)` WITHOUT `use_quantization` → defaults
  False → the P0 bug (chelation_log never populated → sedimentation early-returns → no store
  write). C3/C4 attach a bounded/unbounded MLP adapter.
- `_run_condition_cycle` (185): per-condition action. C3/C4 enable annealing controller,
  `_prime_correction_log`, `observe_annealing_drift`, and if `should_correct` run
  `run_sedimentation_cycle`; `correction_applied = checksum changed`.
- Drift today is store-mutation via `_inject_drift` (rotation/noise).

## The arena change (query-encoder upgrade)
Query-encoder drift is NOT a store mutation — it changes how EVAL QUERIES are embedded.
So the harness must, for the drifted run, embed eval queries via `QueryEncoderDrift`
(2nd model → frozen projection → store dim) instead of the engine's own model, and measure
NDCG of those drifted query vectors against the (unchanged, cached) doc store.

Implementation approach: add a drift mode `"query_encoder_swap"`. Rather than mutating the
store, it constructs a `QueryEncoderDrift` and the harness measures retrieval by scoring the
drifted query vectors against stored doc vectors directly (cosine / engine search by vector).
The "baseline" NDCG uses engine-native query embeddings; the "drifted" NDCG uses
QueryEncoderDrift query embeddings. Document this clearly — the drift manifest is the
QueryEncoderDrift.manifest().

## New conditions
- **C0** frozen: drifted queries, no correction (lower bound).
- **C2** maintenance: re-embed docs with the ORIGINAL engine model. PROVEN no-op for this
  arena (docs already in original space) → cannot recover. Keep as the honest "cheap
  maintenance fails" baseline. ALSO add **C2-oracle**: re-embed ALL docs with the NEW model
  (full re-embed) = expensive upper bound that DOES recover.
- **C3a** anchor-supervised closed loop (PRIMARY, OURS): bounded MLP adapter on doc vectors,
  trained so adapted doc vectors align with the drifted query space, supervised by held-out
  pre-drift (query, relevant-doc) anchor pairs via InfoNCE
  (`engine.set_sedimentation_loss("infonce")`). Anchor queries embedded with the NEW
  (drifted) encoder; eval queries DISJOINT from anchors (split by query id, seeded).
- **C4a** unbounded ablation of C3a.

## Mechanism fixes (P0/P1) — make the actuator able to fire
- P0: build engine with `use_quantization=True` (and likely `use_centering=True`) so
  chelation_log populates and sedimentation has candidates → store actually mutates.
- P1: drive `observe_annealing_drift(drift_magnitude=...)` from the measured NDCG-drop vs
  baseline (the method accepts an explicit override) so `should_correct` tracks real drift,
  instead of the drift-insensitive global_variance.

## The supervision target (the load-bearing piece)
C3a's adapter must learn from a signal that encodes the PRE-DRIFT relevance, available at
maintenance time WITHOUT re-embedding all docs with the new model (that's C2-oracle).
Anchor InfoNCE: for held-out anchor query q' (NEW-encoder embedding) and its known-relevant
doc d, train adapter A so cos(q', A(v_d)) > cos(q', A(v_other)) in-batch. v_d is the cached
(original-space) doc vector. This gives the adapter a real target: rotate cached doc vectors
toward the new query space using only a small labeled anchor set + the new query encoder.

## Acceptance / EVIDENCE
- A real 2-cycle tiny run where C3a has `should_correct=True` AND `correction_applied=True`
  AND correction norm > floor — i.e. the actuator FINALLY fires (the v1 failure is fixed).
  If it won't fire after 2 honest attempts → ship that as the finding + escalate (do NOT
  hand-force). EVIDENCE line: the firing counts + a real ΔNDCG for C3a vs C0.
- Determinism: same seed → same trajectory; anchor/eval disjointness asserted.
- This is PR-A2's scope; the full 3-seed campaign is PR-A4.

## Measurement-path detail (confirmed in run_drift_recovery_experiment.py)
- `evaluate_engine(engine, queries, qrels, k)` (line 133) calls `engine.run_inference(query_text)`
  per query — it embeds queries with the ENGINE's OWN model. So baseline NDCG uses native
  query embeddings.
- For the drifted (query-encoder-upgrade) condition, queries must be embedded with the SWAPPED
  encoder. `run_inference` embeds internally and can't be swapped cleanly → add an alternate
  eval path `evaluate_engine_with_query_vectors(engine, query_vectors, qrels, k)` that searches
  the store by each precomputed QueryEncoderDrift vector (qdrant search-by-vector) and computes
  NDCG the same way (reuse `ndcg_at_k`, `map_predicted_ids`). Baseline uses native; drifted +
  all post-drift cycles use the drifted query vectors against the (cached, unchanged) doc store.
- C3a correction: the bounded adapter corrects DOC vectors so adapted docs realign with the
  drifted query space; measure by searching adapted doc vectors with drifted query vectors.
  `_correction_norm_stats` already applies `engine.adapter` to stored vectors (line 306-309) —
  reuse that pattern for the corrected-doc retrieval.
- C2 (re-embed docs with original model) is the proven no-op; C2-oracle (re-embed docs with the
  NEW swap model+projection) is the expensive upper bound that recovers.

## PR-A2b refined design (confirmed against sedimentation_loss.py)

`SedimentationInfoNCELoss(outputs, targets)` exists (sedimentation_loss.py:15) — a clean
contrastive objective. So C3a does NOT need the unsupervised run_sedimentation_cycle path
(homeostatic targets / chelation_log / use_quantization). Instead, a DEDICATED supervised
correction step:

1. **Anchor split:** before drift, hold out a seeded subset of (query_id, relevant_doc_id)
   pairs from qrels, DISJOINT from the eval query set (split by query id). These are the
   only labels the corrector sees.
2. **Anchor target:** embed anchor queries with the SAME QueryEncoderDrift used for eval
   (the new/drifted query space). For each anchor pair, positive = (cached doc vector v_d,
   drifted anchor query q'). 
3. **Train the bounded adapter** A so cos(q', A(v_d)) is high vs in-batch negatives:
   outputs = A(stack(v_d for anchor docs)), targets = stack(q' for anchor queries),
   loss = SedimentationInfoNCELoss(outputs, targets); a few gradient steps. C4a = unbounded.
4. **Fire trigger (P1):** should_correct driven by measured NDCG-drop vs baseline (the
   annealing controller's observe_annealing_drift accepts an explicit magnitude). The
   corrector runs only while drop exceeds threshold.
5. **Apply (the actuator FIRES):** apply trained A to ALL stored doc vectors and upsert →
   store mutates → correction_applied=True. Measure NDCG with the drifted EVAL queries
   (disjoint from anchors) → genuine generalization, not memorization.

This bypasses the P0 use_quantization issue (no sedimentation candidate machinery needed).
EVIDENCE: a real run where should_correct=True, correction_applied=True, norm>floor, and a
real ΔNDCG for C3a vs C0 — win, lose, or partial, reported honestly. If C3a can't beat C0
even supervised, that bounds bounded-adapter capacity (a key finding). C2O remains the
oracle upper bound; C2 the no-op; C3a the cheap learned realignment between them.

## Slice boundary
PR-A2 = query_encoder_swap drift mode in harness + C3a/C4a conditions + P0/P1 fixes + the
"actuator fires" evidence test. PR-A3 = teacher-supervised C3b. PR-A4 = campaign + results.
Keep PR-A2 shippable + BHS-100-able; reuse existing InfoNCE loss + adapter + controller
(no new adapter types / loss functions per scope freeze).
