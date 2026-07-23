# PR-A2b Implementation Spec — supervised closed loop (C3a/C4a) on the query-encoder arena

Local-only. Build AFTER #270 (PR-A2a) merges. Branch off updated main. This is the crux
slice: the actuator must FIRE and the loop must get a real test. Confirmed APIs below.

## Confirmed building blocks
- `SedimentationInfoNCELoss(temperature=...)(outputs, targets)` — sedimentation_loss.py:15;
  contrastive: pulls each output toward its row-matched target, away from in-batch others.
- `QueryEncoderDrift(store_dim, swap_model_name, seed, swap_backend=None).embed_queries(texts)`
  — returns (n, store_dim) L2-normalized drifted query vectors (the new query space).
- `create_adapter("mlp", input_dim, bounded=, min_correction=, max_correction=)` — used by
  C3/C4 already (run_drift_recovery_experiment.py:163-170).
- Harness: `run_experiment` slices via `select_road_course_slice` (line 69) → sliced_queries,
  sliced_qrels; `evaluate_engine_with_query_vectors` (A2a) measures with drifted query vectors;
  `_run_condition_cycle(engine, condition, queries, manifest, run_config, query_drift)`.
- `_build_query_encoder_drift(engine, queries, config)` (A2a) builds the drift + drifted vecs.

## Conditions to add
- **C3a** — supervised closed loop, bounded adapter (the thesis test).
- **C4a** — same, unbounded adapter (ablation: does the bound cost recovery?).
Add both to CONDITIONS. C3a/C4a are only valid with drift=query_encoder_swap (validate).

## Anchor split (in run_experiment, only when drift==query_encoder_swap and condition in C3a/C4a)
Before drift, split sliced_queries by query id into ANCHOR (held-out supervision) and EVAL
(disjoint), seeded by config.seed. E.g. anchor_fraction=0.4. Anchors carry their qrels.
EVAL queries are what every NDCG measurement uses (via drifted query vectors). The anchor
queries are embedded in the DRIFTED space (same QueryEncoderDrift) to supervise the adapter;
they must NOT appear in the eval set — assert disjointness. Record anchor/eval counts in the
run config/manifest for provenance.

## C3a/C4a correction (dedicated supervised step — NOT run_sedimentation_cycle)
In _run_condition_cycle for C3a/C4a:
1. Detection/trigger (P1): compute current drifted-eval NDCG drop vs baseline; call
   `engine.observe_annealing_drift(drift_magnitude=<ndcg_drop>)` (the method accepts an
   explicit magnitude) → controller.should_correct(). Only correct while should_correct.
2. Build supervision batch: for each anchor (query_id, relevant_doc_id) pair, positive =
   (cached stored vector v_d of the relevant doc, drifted anchor query vector q'). Gather
   doc vectors from the store by id; embed anchor queries via the run's QueryEncoderDrift.
3. Train the engine.adapter (already bounded/unbounded per condition) for a few steps:
   `loss = SedimentationInfoNCELoss()(adapter(doc_tensor), query_tensor)`; Adam, small lr,
   N steps (e.g. lr=0.01, 20 steps — tune small). torch.manual_seed(config.seed) before the
   loop for determinism (note: the #269 flaky-test isolation pins torch globals in TESTS; the
   production training must itself be deterministic under a fixed seed).
4. Apply (actuator FIRES): apply the trained adapter to ALL stored doc vectors and upsert →
   store mutates → correction_applied = (store checksum changed) = True. Record
   correction_norm_stats (reuse _correction_norm_stats pattern).
5. Metadata: action="supervised_anchor_infonce_correction", bounded flag, should_correct,
   correction_applied, anchor_count, eval_count, ndcg_drop_observed, correction_norm_stats.

## Acceptance / EVIDENCE (the slice's whole point)
- A real tiny run (stub swap encoder, like A2a tests) where for C3a: should_correct=True AND
  correction_applied=True AND mean correction norm > floor — the actuator FIRES (v1 never did).
- C3a final NDCG vs C0 (frozen) and vs C2O (oracle) reported. Honest outcomes, all shippable:
  C3a recovers some fraction of the C0→C2O gap (positive); or C3a≈C0 despite firing (bounded
  capacity bottleneck — a real finding); or C3a≈C2O (cheap learned realignment matches the
  expensive re-embed — strong positive). Report what the numbers say.
- Determinism: same seed → same trajectory. Anchor/eval disjointness asserted.
- Do NOT break A2a (C0/C2/C2O) or rotation/noise/C1/C3/C4. Keep #269 flaky isolation intact.
- If the actuator will NOT fire after 2 honest attempts, ship that as the documented finding
  and escalate — do NOT hand-force the trigger.

## Scope
C3a/C4a + anchor split + supervised step only. No new adapter types / loss functions (reuse
MLP adapter + SedimentationInfoNCELoss). No mpnet network dependency in tests (stub swap).
unittest, py3.9. PR-A3 = teacher-supervised C3b; PR-A4 = real-model campaign with the headline
numbers. Tight diff; full BHS; fresh adversarial Tier B to a genuine 100.

## Also fold in (cheap, prevents recurring CI hang)
Harden the real-model smoke test in test_query_encoder_drift.py: gate
TestQueryEncoderDriftRealModelSmoke on an opt-in env var (e.g.
`os.environ.get("CHELATED_RUN_REAL_MODEL_TESTS")=="1"`) in addition to the
sentence_transformers check, so normal CI skips WITHOUT a network attempt (the 3.11 job on
#270 hung ~17 min trying to reach huggingface.co). If folding here is out of scope, do it as a
tiny separate fix first.
