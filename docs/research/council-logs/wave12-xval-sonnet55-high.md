# Wave 12 Cross-Model Validation — Sonnet 5.5 (high) non-author audit of D6 + D7

Reviewer: Claude Sonnet 5.5 (non-author of D6 / D7). Date: 2026-11-25.
Scope: Whitepaper 6.0 Part VII (Cross-Model Validation), Demo D6 (OpenSearch / pgvector / Qdrant) and Demo D7 (live Zep-Graphiti / Mem0 arena).
Method: read-only. Every number below was recomputed from the committed artifacts, not copied from receipts. CPU only; no GPU job, no edit to code, tests, receipts or manifests, no sealed-key or blind-followup file opened.

## Verdict

**PASS_WITH_MANDATORY_DISCLOSURES**

Integrity and the recomputable statistics are sound, and there are no blocking defects in D6 or D7. Several headline-adjacent quantities can be misread, and 6.0 prose must carry the disclosures in section (c). Without them the draft would overclaim, and I would move the verdict to BLOCK.

## (a) Verified checks

### SHA-256 and provenance

| Item | Receipt value | Recomputed from disk | Result |
|---|---|---|---|
| D6 `wave11_d6_multibackend_manifest.json` | `0578d56b715d44dfa71d499217e61ccece91c291d60eefdbf9ff5bd59393612b` | same | MATCH |
| D7 `wave11_d7_zep_graphiti_manifest.json` | `d6a12534fa7c0dd456ee2717220ec3b49c0ee7887989e5c5a001fd2b2a761440` | same | MATCH |
| D7 `suite_a.jsonl` | `35bf134f52d69da8d45efc095271039bf0757b1775f61ee86e05df309c306ee3` | same | MATCH |
| D7 `suite_b.jsonl` | `151e90c3a85df176628831913dc7356e28cc8e677aa76c5571c12dd400b1e7ac` | same | MATCH |
| D6 `implementation_sha256` (appliance, runner) | `b61df62a…a771`, `c5b32561…bbae` | same on disk | MATCH |

- D7 files have 0 CR bytes and 0 occurrences of `<REDACTED>` inside the sidecars. A regex scan for password, api key, `sk-`, bearer, `hf_` and secret found 0 hits in `suite_a.jsonl` and `suite_b.jsonl`.
- The only credential-bearing field is `docker_run_argv` in the manifest, and it reads `NEO4J_AUTH=neo4j/<REDACTED>`.

### D6 — cross-engine parity, stale resurrection, fallback

- The manifest has 70 cells: 14 FiQA2018 cells (retraction 0% only) and 56 SciFact cells (0/10/50/90%).
  - Engines: OpenSearch 3.9.0, pgvector 0.8.7 on PG 17.12, Qdrant 1.20.2.
  - Qdrant is pinned by digest `sha256:d27759…` and image id, not only by the `:latest` tag.
  - `failed_gates = []` and `status = COMPLETED`.
- **nDCG@10 parity.** At retraction 0%, all 14 FiQA2018 and 14 SciFact (`Chelated_ZeroMu_CSLS_8B`) cells have identical nDCG dicts across engines within each (corpus, dim, direction). SciFact values: 0.62079 (1024→384), 0.63676 (384→1024), 0.64205 (384→4096), 0.63731 (4096→384). This includes pgvector `halfvec4000_hnsw_cte` and `exact_vector4096`.
- **Stale resurrection.** `stale_resurrection_rate = 0.0` and `raw_engine_stale_hit_rate = 0.0` in all 70 cells.
  - `admitted_recall_at_10 = 1.0` in all 56 SciFact cells, measured against the exhaustive NumPy float64 oracle.
  - The positive control does discriminate. The simulated stale-replica path (259 lagged docs, about 5% of the corpus) returns stale hits at 5.2–5.6% (10% retraction), 7.6–8.2% (50%) and 31.3–36.3% (90%) before control-plane admission.
  - These cells were recomputed. They come from `lagged_replica_only_stale_hit_rate`.
- **Canaries.** Score-inversion canaries pass in all 10 configurations at 1040 hits checked (fewer for some ANN modes). The maximum absolute error is below 1e-5 everywhere except pgvector `halfvec4000` (see (c)).
- **Exact-fallback fraction (`exact_fallback_fraction_of_shard_queries`).**
  - It is 1.0 in 63 of 70 cells. Seven cells are below 1.0.
  - SciFact, pgvector `exact_vector4096`, retractions 0/10/50% (6 cells): 0.5.
  - FiQA2018, pgvector `exact_vector4096`: 0.8447 (384→4096) and 0.9342 (4096→384).
  - **`served_routes` are exact scans in every cell.** For example `opensearch_knn_score_exact`, `qdrant_dot_exact`, `pgvector_vector_exact_fallback`, `pgvector_halfvec4096_exact_fallback`, and `pgvector_exact_vector4096` (exact by design on the 4096-d shard).
  - ANN appears only as `initial_routes` (the first shortlist). The certified served path never ended on ANN in this evidence.
- **Retraction mechanism.** Retraction is an admission allowlist applied at query time (`point_id = ANY(allowed)`, OpenSearch `terms`, Qdrant `HasIdCondition`). It is not physical deletion.
- **Reproducibility.** The D6 and D7 test files pass offline on CPU: `CUDA_VISIBLE_DEVICES="" pytest tests/test_wave11_d6_multibackend.py tests/test_wave11_d7_zep_graphiti.py -q` gives **41 passed, 1 warning in 15.99 s**.

### D7 — recomputation from the per-item JSONL

- Re-ran `suite_a_horizon_inference(rows, horizon, n_bootstrap=20000, seed=42)` on `suite_a.jsonl` for both horizons. **0 mismatches** against `suite_a_inference` in the manifest (every CI, p-value, Holm entry and clock field).
- Recomputed accuracies match the manifest.

| Suite A (N=60) | Chelated | Graphiti | Mem0 |
|---|---|---|---|
| delta0 | 0.9667 | 0.0333 | 0.1000 |
| settled | 0.9667 | 0.4667 | 0.4500 |

- Paired counts, Chelated minus the baseline (`n10` = Chelated right and baseline wrong; `n01` = the reverse):
  - delta0: Graphiti `n10=56, n01=0`; Mem0 `n10=52, n01=0`.
  - settled: Graphiti `n10=31, n01=1`; Mem0 `n10=31, n01=0`.
- Holm-Bonferroni (family of 2): `holm_adjusted_p = 0.0001` for both comparisons at both horizons. Both reject at alpha = 0.05.
- Clopper-Pearson and Tango CIs reproduce exactly (0 numeric mismatches against the manifest). Settled Chelated minus Graphiti: delta 0.50, Tango 95% interval [0.358, 0.627], CP-Bonferroni [0.265, 0.664].
- **Clocks (Suite A).**

| Arm | Write-ack p50 | Consolidation / settle wait (mean) | End-to-end write+background (mean) | Mean write prompt tokens |
|---|---|---|---|---|
| Chelated | 144.6 ms | 10,033.6 ms | **10,373.3 ms** | 6,883 |
| Graphiti | 0.009 ms | 7,524.4 ms | 7,524.4 ms | 85,727 |
| Mem0 | 0.012 ms | 2,755.1 ms | 2,755.1 ms | 152,483 |

  The manifest `write_clock_disclosure` strings match the code.
- **Stale leakage (Suite A, 60 stale items).** Answer-stale rate at settled: Chelated 0.0167, Graphiti 0.4833, Mem0 0.55. Stale substring in retrieved context at settled: Chelated 0.7833, Graphiti 0.9833, Mem0 0.9667.
- **Judge and generator.** The judge family (`alibaba-qwen2.5`) differs from the generator family (`nvidia-nemotron`). The code enforces this with `BlindCrossFamilySemanticJudge`. All 80 rows have `judge_complete=True`, 0 unclean judge parses and 0 arm failures. Graphiti graph audit is `OK` for all 80 items.
- **Suite B** (N=20, LongMemEval-S knowledge-update): `inferential_claim: false`, `label: descriptive-only`, `unpooled: true`, `holm_bonferroni: null`. The code guard at `scripts/run_wave11_d7_live_zep_graphiti_arena.py:361` rejects a pooled or Holm-bearing Suite B. Recomputed accuracies match the manifest.

| Suite B (N=20) | delta0 | settled |
|---|---|---|
| Chelated | 0.90 | 0.90 |
| Graphiti | 0.40 | 0.60 |
| Mem0 | 0.30 | 0.55 |

## (b) Defects

### Blocking
None. D6 and D7 manifests, sidecars and receipts reconcile, and the stats reproduce.

### Non-blocking (fix or disclose)

1. **D7 p-values are bootstrap-resolution floors.**
   - `paired_bootstrap_wave6` returns `p = min(1, 2·min(P(boot≤0), P(boot≥0)) + 1/n_boot)`. With 20,000 resamples the minimum reported p is 5e-5, and Holm gives 1e-4.
   - Exact McNemar on the same discordant counts:

| Comparison | Exact McNemar p | Reported |
|---|---|---|
| Suite A delta0 vs Graphiti | 2.8e-17 | 5e-5 |
| Suite A delta0 vs Mem0 | 4.4e-16 | 5e-5 |
| Suite A settled vs Graphiti | 1.5e-8 | 5e-5 |
| Suite A settled vs Mem0 | 9.3e-10 | 5e-5 |
| U_separate (n=15) vs Graphiti | 4.9e-4 | 5e-5 |
| U_separate vs Mem0 (delta0) | 3.9e-3 | 5e-5 |
| U_separate vs Mem0 (settled) | 2.0e-3 | 5e-5 |
| Suite B, Mem0, settled | 0.039 | 5e-5 |

   - For the Holm family (N=60) the floor is conservative. For `U_separate` (n=15) and Suite B it is **anti-conservative by 10× to 780×**. `U_separate` is outside the Holm family, and Suite B is declared descriptive.
2. **"Clopper-Pearson" in D7 is not a paired CP interval.** `exact_paired_binary_ci` takes `[CP_low(n10) − CP_high(n01), CP_high(n10) − CP_low(n01)]` at alpha = 0.025 each side. This is a Bonferroni difference of two marginal binomial intervals, which is conservative. The Tango score interval is the appropriate paired interval.
3. **D7 has no `implementation_sha256`.** D6 binds code hashes in the manifest; D7 binds only `git_sha = 23305244…`.
   - `chelated_graphiti_arena_v1.py` is unchanged since that SHA.
   - `scripts/run_wave11_d7_live_zep_graphiti_arena.py` changed afterwards. Commit `f7a515db` added `redacted_docker_run_argv()`, and `46e8f558` added `manifest_file` and `artifact_hashes` to the receipt. Both were committed 2026-11-25 08:49–08:54.
   - The D7 artifact was first committed in the redaction commit `f7a515db`. It is therefore a redacted post-run seal and not the raw live-run bytes. I can only attest to the bytes on disk and to the Suite A recomputation, not to the live run itself.
4. **D6 retraction cells record no nDCG.**
   - `quality` is `None` for every retraction above 0% (`quality_metrics(...) if snapshot.epoch == 1 else None`). Only admitted-recall vs the exact oracle is recorded there (1.0 in all 56 SciFact cells).
   - D6 has no end-to-end quality-under-retraction measurement. Cross-engine nDCG parity is evidenced at 0% retraction only.
   - FiQA2018 has no retraction cells and no oracle recall (`admitted_recall_at_10 = None`).
   - The failed-gate loop (`run_live`, around line 636) iterates only SciFact. FiQA2018 is ungated.
5. **D6 `raw_engine_stale_hit_rate = 0` is partly by construction.** The engines receive the admitted-only allowlist, so zero raw stale hits is expected. The informative measurements are the positive control (`lagged_replica_only_stale_hit_rate`, 5–36%) and the egress check.
6. **D6 `exact_fallback_fraction` of 0.5 understates exactness.** For `pgvector_exact_vector4096`, the other 50% of shard queries are served by an exact route by design, so 100% of served queries are exact scans in every cell.
7. **D6 halfvec canary tolerance.** pgvector `halfvec4000` uses a relaxed limit of 2e-3. Its observed maximum error is 1.108e-5, which would fail the 1e-5 limit used for every other configuration.
8. **D7 naming ambiguity.** In the bootstrap block, `lcb_95` is the one-sided 5th percentile, `lcb_975` is the two-sided lower bound, and `tango_score_ucb95` is the upper end of a two-sided 95% interval. Both are labelled "95".
9. **D7 test coverage.** The D7 test file does not open the committed artifacts, so artifact drift is caught only by `verify_all_offline`, not by the unit tests. The D6 test checks the receipt-manifest round trip on synthetic data.

## (c) Mandatory Brutal-Honesty disclosures for Whitepaper 6.0

**D6**

1. **No pure-ANN claim.** The certified served path was exact (full scan or verified exhaustion) in all 70 cells. `exact_fallback_fraction_of_shard_queries = 1.0` in 63 of 70 cells, 0.5 for `pgvector_exact_vector4096` at SciFact 0/10/50%, and 0.84 / 0.93 for the FiQA2018 `pgvector_exact_vector4096` cells. The sub-1.0 value there reflects an exact-by-design route, so the served path is 100% exact in every cell. Latencies reported (service RTT p50 about 10–36 ms on SciFact, up to about 1 s on FiQA2018 OpenSearch) are exact-scan plus ANN-request RTT, not ANN latency.
2. **ANN only as first shortlist.** `initial_admitted_recall_at_10 = 1.0` on SciFact. On FiQA2018 the initial ANN nDCG sits slightly below the served nDCG in some cells (for example OpenSearch 0.2867 vs 0.2871; Qdrant 4096→384 0.3046 vs 0.3068). Do not attribute served quality to ANN recall.
3. **Parity scope.** Cross-engine nDCG parity (identical to full double precision across 3–4 engine variants per cell) is demonstrated at 0% retraction on SciFact and FiQA2018 only. Parity is expected because every served path is an exact scan over the same stored operands.
4. **Stale resurrection scope.**
   - The 0.00% claim covers the control-plane admission filter on a **simulated** stale replica (an allowlist over a live backend, 259 lagged docs, about 5% of the corpus), on **SciFact only** (about 5.2k docs, 86 REPORT queries, 2 directions, 2 dims).
   - It is not a measurement of real replica lag, real deletes or tombstones in OpenSearch, pgvector or Qdrant.
   - FiQA2018 was run at 0% retraction only.
   - The raw-engine 0.0 is trivially zero under an allowlist; the unfiltered replica control gives 5–36%.
5. **No retraction-quality number.** nDCG under 10/50/90% retraction was not recorded. Only admitted recall vs the exhaustive oracle was (1.0).
6. **Environment.** One arm64 Spark host (`10.0.0.49`), memory limited to 8 GB per engine, 1,253 s wall, and single-run RTT percentiles with no replicate runs.
7. **Halfvec tolerance** is 2e-3, not 1e-5; observed 1.1e-5.
8. **Qdrant** image tag is `latest`; the pinned digest is recorded.

**D7**

1. **Primary horizon is settled.**
   - Lead with settled accuracy: 0.9667 vs 0.4667 (Graphiti) and 0.45 (Mem0), deltas +0.50 and +0.517.
   - The delta0 figures (0.9667 vs 0.033 and 0.10) are an asymmetric read-immediately comparison. They are not a like-for-like accuracy gap, because baselines are read before background consolidation while the Chelated write waits for its synchronous extraction.
2. **Write-clock honesty.**
   - The 144.6 ms figure is the post-extraction EK5 + index ack only.
   - The end-to-end clock is 10,373 ms for Chelated vs 7,524 ms Graphiti and 2,755 ms Mem0.
   - Chelated is the slowest end to end (about 1.4× Graphiti, 3.8× Mem0) in Suite A.
   - The ack figures for Graphiti and Mem0 (0.009 ms and 0.012 ms) are call-return times and are not comparable to Chelated's.
   - Chelated's advantage on write tokens (6.9k prompt vs 85.7k and 152.5k) must not be presented as a latency advantage.
3. **P-value language.**
   - Write "p < 1e-4 (bootstrap-resolution limited; 20,000 resamples)", not "p = 0.0001". Never quote 5e-5 or 0.0001 as a point estimate.
   - Exact McNemar p-values may be cited instead (table in (b)1).
4. **Interval language.**
   - Call the CP interval a "Bonferroni difference of two exact binomial intervals". Call Tango the paired score interval.
   - Do not cite the bootstrap `lcb_95` as a two-sided 95% bound.
5. **Sample scope.** Suite A is N=60 (4 regimes of 15, synthetic enterprise-memory regimes). The `U_separate` abstention subset (n=15) is outside the Holm family and its bootstrap p is anti-conservative; report it as descriptive. One seed (42), one reader model (`nemotron-lightning`), one judge model (Qwen2.5-7B). No replication across seeds or models.
6. **Suite B is descriptive only.**
   - N=20 LongMemEval-S knowledge-update, unpooled, `inferential_claim: false`.
   - Settled accuracy 0.90 vs 0.60 vs 0.55. Exact McNemar p = 0.07 (Graphiti) and 0.04 (Mem0), which would not survive a two-comparison correction.
   - No Holm and no pooling with Suite A.
7. **Suite B arms are not the same pipeline as Suite A.**
   - The Chelated arm in Suite B is GPU embed + rerank retrieval over raw conversation messages (0 write tokens, no LLM extraction).
   - Graphiti and Mem0 ingest the same conversation through LLM extraction (about 5.65M and 0.44M mean write prompt tokens).
   - The retraction message is chosen from the dataset's `has_answer` annotation, which is used to order the baseline ingestion (prior sessions first, retraction last).
   - Do not present Suite B write-cost columns as a like-for-like comparison.
8. **Stale-in-context caveat.** Chelated's answer-stale rate is 1.7%, but stale substrings appear in 78% of its retrieved contexts. Do not claim stale values never reach the model.
9. **Provenance.**
   - The committed D7 manifest was sealed by a later redaction commit (`f7a515db`), after the live run's `git_sha` (23305244), and carries no implementation hashes.
   - Credentials are redacted (`NEO4J_AUTH=neo4j/<REDACTED>`; the Neo4j instance was ephemeral and loopback-only).
   - The arena module is unchanged since the run SHA; the runner was changed only for redaction and receipt fields.
10. **Evidence tier.** Both demos are `spark1-live` single-host runs. `local_check_is_not_evidence: true` applies to D7.

## (d) Verdict

**PASS_WITH_MANDATORY_DISCLOSURES.**

- D6: integrity, cross-engine parity at 0% retraction, 0.00% stale resurrection (simulated lag, SciFact), and oracle recall 1.0 all verify. The 6.0 prose must not claim ANN latency or ANN-served quality.
- D7: integrity and the full Suite A inference reproduce exactly. The 6.0 prose must lead with the settled horizon, state the end-to-end write clock, treat p-values as bootstrap-floor-limited, and keep Suite B descriptive and non-comparable on write cost.
- Escalate to BLOCK if 6.0 text states any of the following: an ANN-served latency or quality figure for D6, a "p = 0.0001" point value, the delta0 gap as the headline accuracy gap, the 144.6 ms ack as the write latency, or an inferential Suite B claim.
