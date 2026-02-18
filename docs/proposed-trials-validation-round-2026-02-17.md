# Proposed Trials and Validation Round (2026-02-17)

## Purpose

Capture and preserve all strategy ideas raised in this session, then convert them into a concrete comparative testing and validation program for iterative model improvement.

---

## Captured Session Ideas (No Loss)

### User-proposed ideas captured

1. **Vector weight balancing of teacher components by request vector-space context**
   - Concern: static teacher blending can add noise for short requests.
   - Concern: static blending can dull or over-smooth very verbose requests.

2. **Use historical manifold signal tied to vector/quantization impression**
   - Incorporate manifold behavior from retrieval history, not only single-query state.
   - Include quantization-related signal where possible.

3. **Add Eagan value based on delta-z over time**
   - Use time-aware drift (`delta z`) from chelated/RAG-bearing vector log entries.
   - Apply this temporal drift signal to adaptation decisions.

### Assistant strategy captured

1. Replace fixed hybrid scalar (`teacher_weight`) with **request-adaptive teacher blend** `alpha(q)`.
2. Build an **Eagan temporal drift score** from logs and integrate it into weighting/trigger control.
3. Extend logging so each event carries enough fields for offline validation, dashboards, and ablation studies.
4. Run fixed-vs-adaptive comparative rounds with stratified reporting (not aggregate-only metrics).

---

## Proposed Technical Framework

## 1) Request-adaptive teacher weighting

Current state: hybrid training uses a global scalar `teacher_weight` for all requests in a run.

Proposed:

- Compute per-request features:
  - `query_len_tokens`
  - `conjunction_count` (proxy for compositionality)
  - `global_variance` from local neighborhood
  - optional `quantization_residual_proxy`
  - optional decomposition indicator

- Compute bounded adaptive weight:

```text
alpha_raw(q) = w0
             + w_len * f_len(q)
             + w_var * f_var(q)
             + w_quant * f_quant(q)
             + w_comp * f_comp(q)

alpha(q) = clip(EMA(alpha_raw(q)), alpha_min, alpha_max)
```

- Intended behavior:
  - lower `alpha(q)` for short/atomic low-drift requests
  - higher `alpha(q)` for verbose/high-variance/high-drift requests

---

## 2) Eagan value from temporal delta-z drift

Define vector drift over time for each document/vector neighborhood:

- `z_i(t)`: state vector for item `i` at time `t` (from chelation-bearing events)
- `delta_z_i(t) = ||z_i(t) - z_i(t-1)||_2`

Time-decayed drift energy (Eagan value):

```text
E_i(t_now) = sum_k exp(-(t_now - t_k)/tau) * delta_z_i(t_k)
```

Request-level aggregate (over retrieved neighborhood):

```text
E_query = aggregate(E_i for top-k neighborhood)
```

Optional impact-weighted variant:

```text
E_query_weighted = E_query * (1 - jaccard_top10) * variance_percentile
```

Control usage:

- low `E_query` + short query: reduce teacher influence, prefer fast/stable path
- high `E_query` + verbose/compositional query: increase teacher influence and chelation sensitivity

---

## 3) Required logging/data-model upgrades

Current in-memory chelation log stores only center vectors per doc.

Proposed per-event fields:

- `timestamp`
- `doc_id`
- `center_vector_ref` (or compact representation)
- `query_len_tokens`
- `global_variance`
- `action` (`FAST`, `CHELATE`, etc.)
- `jaccard_top10`
- `quantization_residual_proxy` (if available)
- `delta_z`
- `eagan_value`
- `alpha_used`
- `training_mode`

Persist in JSONL for:

- longitudinal analysis
- dashboard overlays
- reproducible ablations

---

## Comparative Testing and Validation Round

## Round 0: Environment and regression gate

- Run repository-local tests:
  - `python -m pytest (Get-ChildItem -Name test_*.py) -q`
- Gate: must pass before and after experimental changes.

## Round 1: Fixed-weight baselines

Run distillation benchmark with fixed teacher weights:

- `w in {0.0, 0.2, 0.5, 0.8, 1.0}`
- same dataset/task splits and randomization controls
- isolated checkpoints per run (no shared adapter carryover)

Primary outputs:

- final-cycle NDCG@10
- per-cycle trend
- latency cost

## Round 2: Adaptive alpha without Eagan

Enable `alpha(q)` from request features only; compare against best fixed-weight baseline.

## Round 3: Adaptive alpha + Eagan temporal drift

Enable both adaptive alpha and Eagan drift modulation.

## Round 4: Ablations

Disable one signal at a time:

- no length feature
- no variance feature
- no quantization proxy
- no Eagan temporal term

## Round 5: Recursive retrieval interaction

Evaluate whether Eagan/alpha improves recursive path quality:

- SciFact with `rrf`, `union`, `intersection`
- compare decomposition-triggered queries separately from atomic queries

---

## Metrics to Report (Stratified)

Global:

- NDCG@10 mean/std
- learning gain
- stability (Jaccard)
- query latency and overhead

Stratified:

- short vs medium vs verbose query bins
- atomic vs decomposed queries
- low vs high drift (`E_query`) bins

Failure-focused:

- short-query degradation rate
- verbose-query recovery gain
- percentage of queries improved/degraded/unchanged

---

## Acceptance Criteria for Next Iteration

1. No regression in repository-local test suite.
2. Adaptive strategy beats best fixed baseline on at least one core task while not materially degrading short-query bins.
3. Eagan-enhanced strategy reduces degraded-query fraction vs fixed baseline.
4. Latency overhead remains within predefined operational tolerance for interactive retrieval.

---

## Current Evidence Snapshot (from this session)

- Local test gate: `266 passed, 1 warning` (repo-local command above).
- Expanded multitask:
  - small suite NDCG mean `0.6766`
  - medium suite NDCG mean `0.6166` (FiQA is hardest in current run)
- Distillation:
  - `teacher_weight=0.5` underperformed baseline in expanded run
  - `teacher_weight=0.8` outperformed baseline in expanded run
- Recursive benchmark (SciFact, mock decomposition):
  - RRF/Union/Intersection all below standard baseline in this run

These results justify moving from fixed scalar teacher blending to adaptive + temporal drift-aware control.

---

## Immediate Build Plan for Implementation Cycle

1. Add experimental config flags for adaptive alpha and Eagan drift term.
2. Extend log schema and in-memory chelation event structure.
3. Implement request feature extraction + alpha controller + smoothing/clamp.
4. Implement delta-z and Eagan accumulation over time-decayed window.
5. Add tests:
   - controller bounds and monotonic behavior
   - temporal decay correctness
   - log-field presence and serialization
   - fallback behavior when data is sparse
6. Run comparative rounds and publish result table in follow-up doc.
