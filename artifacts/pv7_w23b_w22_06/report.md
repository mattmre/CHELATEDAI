# W22-06 Report: Migration Information Frontier (CPU-only)

(Report text written by the W22-06 implementer sub-agent; transcribed into this file by the orchestrator because the harness blocks sub-agents from writing `.md` files.)

Commits on `pv7/w23b-w22-06-migration-frontier` (based on the W22-04 tip `91c5cc7a`):
- `336f280a`: preregistration, committed alone
- `793a4f29`: module, runner and tests
- `37be1330`: `w22_06_results.json`

## Verdicts

| Sub-test | Verdict |
|---|---|
| (a) observation-identical flip pairs | **SUPPORTED** (phenomenon PRESENT), but close to automatic under this definition of "margin" |
| (b) chain vs direct Procrustes refit | **DEFERRED_SCOPE**: not run, no verdict |
| (c) centering law (leave-one-domain-out) | **KILL**: R2_LODO = -0.476, below 0.5 |
| (d) whole-query conformal containment | **KILL**: median K* is 329 in-domain and 393 out-of-domain (limit 200), and OOD coverage drops 6.0 pp (limit 5) |
| (e) #140 bit ledger | descriptive only; the #140 `d log(1/eps)` bound was not tested |

**Run**
- Settings: `CUDA_VISIBLE_DEVICES=""`, `HF_HUB_OFFLINE=1`, `HF_DATASETS_OFFLINE=1`, run at `793a4f29`.
- Wall time 314.5 s, exit 0, CPU only.
- No encoding. Every embedding was read from the W22-04 caches after checking their text digests.

**Reproduction check (reported, not used as a gate)**
- nDCG@10 for Centered+CSLS and Uncentered+CSLS matches W22-04's cells exactly (difference 0.0) on all 12 domains.
- The row-collision scrub removed 0 rows everywhere.

## (a) Flip pairs: SUPPORTED, PRESENT
- **Pooled over 2549 queries:** 12038 of 12093 missing native docs can be explained by the margin (0.9955).
- **Per query:**
  - Share with at least one flip pair: 0.9969.
  - Share with any top-10 error: 0.9973.
  - Native docs missing per query: 4.74 on average.
- **Per domain:** all 12 domains are at or above 0.5. touche-2020 has the lowest share of queries with a flip, at 0.853.
- **Why the pass is weak:**
  - Margins run 0.096–0.153, about 2–3x the median residual.
  - The calibration slopes are small (0.07–0.30), which squeezes the calibrated scores together.
  - So nearly every boundary pair falls within the margin, and the pass comes close to following from the definition.
  - It must not be cited as a real finding that migration errors are margin-level.
- **trec-covid:**
  - Its calibration slope is negative (b = -0.097): served scores rank the DEV docs in reverse of the native order.
  - It misses all 10 native docs on every query.
  - Native nDCG there (0.388) is below served (0.484), so the native ranking is not a quality ceiling for that domain.

## (c) Centering law: KILL
- **Centering gain g:** 0.0325–0.0796, positive on all 12 domains.
- **Predictor x:** -0.0035 to 0.0421.
- **Leave-one-domain-out fit:** R2_LODO = -0.4763, sign accuracy = 1.00.
- **In-sample fit:**
  - R² = 0.0132.
  - Slope beta = 0.130.
  - Intercept alpha = 0.0548.
  - Spearman = 0.210.
- **Reading:**
  - The preregistration disclosed in advance that the sign test would almost certainly pass, so R² was the deciding criterion.
  - Centering helps on every domain, but the predictor does not say how much.
  - The fit is carried by the intercept alone.

## (d) Conformal containment at alpha = 0.10, k = 10: KILL
- **Set size K\*:**
  - Median in-domain K* = 329; median out-of-domain K* = 393.
  - The largest is 602 in-domain (trec-covid) and 402 out-of-domain.
- **Coverage:**
  - In-domain 0.9217, out-of-domain 0.8615, a drop of 6.0 pp.
  - Nominal minus out-of-domain is 3.85 pp, which does not trip rule 4.
- **Rules tripped:** 1, 2 and 3. The secondary check using the maximum instead of the median also gives KILL.
- **Largest coverage drops by domain:**
  - trec-covid 47.4 pp
  - english 27.1 pp
  - climate-fever 17.4 pp
  - dbpedia 9.2 pp
- **Small calibration sets:** trec-covid and touche-2020 have only 16 calibration queries each, so their numbers are noisy.
- **Group transfer:**
  - prior7 to remaining5: K* = 301, coverage 0.7955.
  - remaining5 to prior7: K* = 448, coverage 0.9615.
- **Practical reading:** to contain the native top-10, the served list has to be about 300–600 docs long out of 750–954.

## (e) Ledger, descriptive (per domain in `per_domain[*].ledger`)
- **Bridge parameter bits relative to `d_src*d_tgt*32`:** Uncentered 1.000, Centered 1.00285, ZeroMu 1.00545.
- **Centered bridge size:** 6.31 MB at fp32.
- **Retained bytes vs a full re-index (fp32, Centered):** 0.7137 (0.6041 for dbpedia, which has 954 docs).
- **New-encoder forward passes vs a full re-index:**
  - 0.5933 for trec-covid and touche-2020.
  - 0.68 for the other prior7 domains (280 anchors).
  - 0.9733 for the remaining5 domains (500 anchors); dbpedia is 0.8082.
- **Samples:**
  - Anchors: 280 or 500.
  - DEV queries: 80 (15 for trec-covid and touche).
  - Dev-ref docs: 160 or 180.
- **Reading:**
  - At 750 docs the 4096x384 bridge matrix alone is about half the size of a full re-index.
  - The migration saves little storage and little compute at this scale.

## Departures from the preregistration and the queue item
1. **(b) not run.** It needs bge-small, mpnet and bge-m3 embeddings, none of which are cached, and re-encoding was forbidden. Frozen in prereg §5.
2. **(c) uses leave-one-domain-out instead of leave-one-encoder-family-out.** Only one encoder pair is cached. Frozen in prereg §3.
3. **(a)'s definition is the implementer's own.** The A1/F4 source memos were not found in the repo.
4. **Dataset sha256 manifests were never compared.** The digests from `_load_group` were ignored; text identity rests only on the cache text digests and row counts. This is not in the preregistration.
5. **DEV margin calibration fuses DEV queries using DEV itself as the CSLS reference.** The preregistration does not state this explicitly.
6. **The log shows "[download] Fetching ... via HuggingFace Hub" lines.** The offline flags were set, so the data should have come from the local cache, but no network monitor was run to confirm it.
7. **No bootstrap was run.** The preregistration allowed one for description only.

## Brutal Honesty
- **L1 (stubs):** none.
- **L2 (escape conditionals):** `conformal_k` falls back to the whole corpus when the calibration set is too small. This is preregistered and did not fire: with n = 16, ceil(17 × 0.9) = 16, which is not greater than 16.
- **L3 (mocks in production paths):** none.
- **L4 (partial presented as complete):**
  - (b) and the encoder-family version of (c) are deferred.
  - The SUPPORTED verdict for (a) is close to automatic and should not be cited as substantive.
- **L5/L8/L12 (untested production paths):** the runner has no unit test of its own; only the single real run exercised it. The path where it refuses to start without CPU/offline flags is untested.
- **L11 (broad-catch swallowing):** none.
- **Other:** the exact reproduction shows this matched W22-04's pipeline, not that W22-04's pipeline is correct.

**Evidence and smoke**
- EVIDENCE: `w22_06_results.json` (commit `37be1330`) and scratch `run.log`.
- SMOKE: floor tier only, meaning the 17 unit tests pass and `python -m ruff check` is clean on the new files, plus one real CPU run. There was no ceiling-tier `scripts/smoke.sh` run.

**BHS lines**
- BHS_SELF_DRAFT: 90
- BHS_SELF_DRAFT_AGENT: claude-opus-5-5 (W22-06 implementer)
- BHS_TIER_B: PENDING. BHS_TIER_B_AGENT: PENDING. BHS_TIER_B_SEVERITY: PENDING.
- BHS_OFFICIAL: PENDING. It is min(90, Tier B), so it is at most 90 and this cannot merge as it stands.
- CARRY_FORWARD: none.
- DEFERRED_SCOPE: (b) chain vs direct refit; encoder-family LOFO for (c); the #140 `d log(1/eps)` bound.
- LOOP_ITERATIONS: 1.
- OPERATOR_OVERRIDE: none.

**Why 90:** (b) was not run, the runner has no test of its own, the dataset manifest checks were skipped, and the (a) pass is close to automatic.

## Scratch receipt (removable)
- **Path:** `C:\Users\mattm\AppData\Local\Temp\claude\439e695c92b40043\c9f4318c-455f-427a-844a-07562a0a3c42\scratchpad\w22-06\artifacts\pv7_w23b_w22_06\scratch\`
- **Size:** 415,771 bytes in 2 files: `run.log` (3,885 B) and `w22_06_per_query.jsonl` (411,886 B).
- **Status:** gitignored via `artifacts/**/scratch/`. Nothing was deleted.
