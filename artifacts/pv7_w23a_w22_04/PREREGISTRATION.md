# W22-04 Preregistration — Real-data bridge x merge factorial at equal total cost

Frozen and committed before any W22-04 result was computed. Base commit `2c408925`, branch `pv7/w23a-w22-04-gpu-cells`.
Written 2026-11-26.

## 0. Question and kill rule

The queue entry is `docs/research/wave22_queue_additions_2026-11-25.json`, W22-04. Its kill rule is reproduced verbatim:

> Kill unique ZeroMu bridge claim if centered/ordinary Procrustes + CSLS lands within +/-0.005 nDCG@10 (or overlapping CI) on >=10/12 domains at equal/lower cost; retain CSLS merge as the core finding.

## 1. Data, encoders, setup (unchanged from the prior W22A / closure runs)

### Domains

There are 12 unburned public domains, loaded by the original loaders with seed `20261125`.

- **prior7**: `trec-covid`, `touche-2020`, `cqadupstack-{android,physics,programmers,stats,unix}`. Loaded by `scripts/run_colab_l4_pm6_and_pv7_closure.py::load_beir_slice_from_hf` with `UNBURNED_DOMAIN_PLAN`:
  - qrels capped at top-8 per query, eval split 0.72;
  - 280 anchor docs and 160 dev-ref docs;
  - all anchors are used to fit.
- **remaining5**: `dbpedia`, `climate-fever`, `cqadupstack-{english,gis,tex}`. Loaded by `scripts/run_colab_l4_w22a_b54_remaining5_and_poisoning.py::load_beir_slice_from_hf` with `REMAINING5_DOMAIN_PLAN`:
  - the first 500 anchors are used to fit (the original `anchors_per_draw=500`).

Source files are `mteb/*` HF dataset files (`qrels/test.tsv`, `queries.jsonl`, `corpus.jsonl`). Their sha256 is checked against the `hf_dataset_file_sha256` / `hf_dataset_digests` values recorded in the two source manifests. A mismatch is recorded as a departure and is not silently accepted.

### Encoders

- old index: `sentence-transformers/all-MiniLM-L6-v2` (384-d);
- new encoder: `Qwen/Qwen3-Embedding-8B` (4096-d) at revision `1d8ad4ca9b3dd8059ad90a75d4983776a23d44af`, run in bf16.

Both use `max_seq_length=256` and `normalize_embeddings=True`, through the original `encode_all_texts_on_gpu`. The other three encoders of the original 5-encoder runs (bge-small, mpnet, bge-m3) are not used by the factorial and are **not** encoded. This is a known departure, covered under section 7.

### Mode B (20% backfill), identical to the source runs

- Queries are Qwen-4096.
- 80% of eval docs are served from the MiniLM-384 index through the bridge.
- 20% are native Qwen-4096, selected by the mask `default_rng(int.from_bytes(sha256("bf20:{domain}")[:4],"big")).choice(n_docs, round(0.2 n_docs))`.
- DEV calibration uses the Qwen dev queries and the MiniLM dev-ref docs (disjoint from eval; strict-disjoint guard on).
- k = 10 for nDCG@10, `csls_k=10`, ridge reg `1e-2`, ZeroMu `dual_adap` with causal EMA alpha `0.05`.

## 2. Arms

### Primary bridges (4), from `chelated_pv7_spine/bridge_merge_harness.py::BRIDGE_REGISTRY`

- `CenteredProcrustes`
- `UncenteredProcrustes` (the "ordinary Procrustes" of the kill rule)
- `RidgeDriftAdapter` (the "DriftAdapter")
- `ZeroMuPolarBridge` (the "ZeroMu")

### Primary merges (4)

- `RawCosineMerge`
- `CalibratedZScoreMerge`
- `CSLSMerge`
- `RRFMerge`. This merge is new in this slice and is defined below.

### Secondary (reported, not used by the decision rule)

- `Rank1QueryResidual` x the same 4 merges.

Total: 12 domains x 5 bridges x 4 merges = 240 cells, of which 192 are primary.

### `RRFMerge` definition (frozen)

Reciprocal-rank fusion of the two shards:

1. Within each shard (bridged columns `~mask`, native columns `mask`) and for each query row, rank the shard's documents by descending raw score.
   - Ranks are 1-based.
   - Within-shard ties are broken by ascending column index, using a stable sort.
2. The fused score is `1 / (rrf_k + rank)` with `rrf_k = 60`.
3. The final top-k uses the harness's canonical `(-score, doc_id)` tie-break, so equal-rank documents from the two shards are ordered by doc_id.

RRF uses no DEV reference, which means it costs no more than the other merges. It is added to `MERGE_REGISTRY`. The default `MERGE_NAMES` stays the original three names so existing callers and tests are unchanged, and W22-04 passes `merge_names` explicitly.

## 3. Cost accounting ("equal total cost")

All bridges in a domain share exactly the same paid inputs:

1. the same anchor pairs (500 for remaining5, 280 for prior7), each encoded by both encoders;
2. the same 20% re-embedded backfill;
3. the same DEV queries and DEV-ref docs.

No bridge receives more anchors, more backfill, or extra supervision.

Per-bridge state:

| Bridge | Persisted state | Extra per-doc state | Query-time state |
|---|---|---|---|
| UncenteredProcrustes | W (4096x384) | none | none |
| CenteredProcrustes | W, mu_x, mu_y | none | none |
| RidgeDriftAdapter | W | none | none |
| ZeroMuPolarBridge | Q, mu_x, mu_d (from DEV docs) | 4-byte float32 centered norm / doc | causal query EMA mu_q |

Both Procrustes variants therefore cost **less than or equal to** ZeroMu on every axis, so the kill rule's "at equal/lower cost" clause is satisfied by construction for the Procrustes comparators. The runner records this table in the results JSON.

## 4. Per-domain statistic and CI method

For domain d and comparator B in {CenteredProcrustes, UncenteredProcrustes}:

- `delta_d(B) = mean_q[ nDCG@10(ZeroMu+CSLS, q) - nDCG@10(B+CSLS, q) ]` over the domain's eval queries that have positives. The query set is identical across cells.
- **Primary CI:** a 95% paired percentile bootstrap of the per-query differences, using `bridge_merge_harness.paired_bootstrap_ci` with `n_resamples=2000` and `seed=1601`.
- **Secondary CI** (sensitivity only): the marginal 95% percentile bootstrap CI of each arm's mean (same resample indices, seed 1601), and whether the two marginal CIs overlap.

## 5. Decision rule (frozen; applies the kill rule literally)

A domain d is **WITHIN** if, for at least one comparator B in {CenteredProcrustes, UncenteredProcrustes}, one of these holds:

- (a) `|delta_d(B)| <= 0.005`, or
- (b) the primary paired-difference 95% CI of `delta_d(B)` contains 0. This is how "overlapping CI" is operationalized: no statistically resolved difference between ZeroMu+CSLS and B+CSLS.

The cost condition holds by construction (section 3).

- **Verdict KILL** (the unique ZeroMu bridge claim is killed): WITHIN on >= 10 of 12 domains.
- **Verdict NOT KILLED:** WITHIN on <= 9 of 12 domains.

If fewer than 12 domains can be run, the verdict is computed only when the outcome is already decided whatever the missing domains do. KILL needs at least 10 WITHIN among those run. NOT KILLED needs more than (n_run - 10) NOT-WITHIN, i.e. at least 3 NOT-WITHIN. Otherwise the verdict is `UNDECIDED (DEFERRED_SCOPE)`.

Reported, not gating:

- **Secondary sensitivity verdict:** the same count with (b) replaced by marginal-CI overlap (section 4).
- **Direction breakdown** of the NOT-WITHIN domains: either ZeroMu is significantly better (CI > 0 and delta > 0.005) or the comparator is significantly better (CI < 0 and delta < -0.005). The kill rule counts only WITHIN domains toward the kill. Domains where Procrustes is resolved better are reported separately and are **not** relabelled as WITHIN.

**CSLS retention check** (descriptive, not a gate). For each bridge and domain, report `nDCG(B+CSLS) - max(nDCG(B+Raw), nDCG(B+ZScore), nDCG(B+RRF))` with a paired bootstrap CI (seed 1602) against the best non-CSLS merge, which is selected on the point estimate. Also report the count of domains where CSLS is the best merge for that bridge.

## 6. Fidelity check against the source manifests

The 15 original cells (5 bridges x 3 old merges) per domain are compared with the values recorded in:

- `artifacts/pv7_colab_l4_closure_20261125/colab_l4_closure_manifest.json` (prior7);
- `artifacts/pv7_w22a_b54_12dom_complete/w22a_b54_12dom_manifest.json` (remaining5).

Report the per-domain max |diff|, and the eval query and doc counts compared with the manifests.

- Exact equality is not expected: the hardware is an RTX 3090 rather than an L4, the torch version differs, and only 2 of the 5 encoders are encoded. Encoding batches are therefore composed differently from the original runs, and bf16 padding can change embeddings slightly.
- A domain with any cell |diff| > 0.02, or with count mismatches, is **flagged** in the report.
- Flagged domains are not excluded. The decision uses this run's freshly computed numbers for all domains.

## 7. Known departures, stated before running

1. Hardware and stack: a local RTX 3090 with torch 2.5.1+cu121, sentence-transformers 5.2.0 and transformers 4.57.3. The original ran on a Colab L4 with torch 2.14.1.
2. Only MiniLM and Qwen3 are encoded. The global row-collision scrub therefore runs over 2 encoders instead of 5. Both source manifests recorded 0 scrubbed rows, and this run reports its own scrub count.
3. Encoding batch size is 32 for Qwen3, against 64 originally, to stay within the 3090's memory. Other encoders keep their original batch sizes.
4. Datasets and model were fetched once each from the same public HF sources the original code used, into the local HF cache. All computation runs with `HF_HUB_OFFLINE=1` and `HF_DATASETS_OFFLINE=1`.
5. The remaining5 domains are loaded and encoded as one group, and the prior7 domains as another, as in the source runs. For remaining5, the Qwen prompt-shifted query texts are also encoded so that the batch composition matches the source run. They are not used in the analysis.

## 8. Seeds

- data slice: `20261125`;
- backfill mask: sha256(`bf20:{domain}`);
- primary paired bootstrap: `1601`;
- CSLS retention bootstrap: `1602`;
- `n_resamples = 2000`, `alpha = 0.05`.

## 9. Outputs

- `artifacts/pv7_w23a_w22_04/w22_04_results.json`: all cell means, per-domain deltas and CIs, the decision table, fidelity, cost table and provenance.
- `artifacts/pv7_w23a_w22_04/report.md`.
- Embedding caches go in `artifacts/pv7_w23a_w22_04/scratch/` (gitignored by `artifacts/**/scratch/`).
