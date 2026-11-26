# W22-06 Report: Migration Information Frontier (CPU-only)

(Report text written by the W22-06 implementer sub-agent; transcribed into this file by the orchestrator because the harness blocks sub-agents from writing `.md` files.)

- **Preregistration:** `artifacts/pv7_w23b_w22_06/PREREGISTRATION.md` (`336f280a`, committed alone).
- **Code:** `migration_frontier_w22_06.py`, `scripts/run_pv7_w22_06_migration_frontier.py` and `tests/test_migration_frontier_w22_06.py`. First written in `793a4f29`; the Tier B round-1 fix is `8ff93b0f`.
- **Results:** `artifacts/pv7_w23b_w22_06/w22_06_results.json`.
  - `37be1330`: first run, at `793a4f29`.
  - `07a073c9`: re-run at `8ff93b0f`. This is the current version.
- **Run conditions:** `CUDA_VISIBLE_DEVICES=""`, `HF_HUB_OFFLINE=1`, `HF_DATASETS_OFFLINE=1`, CPU only. The current run took 196.2 s and exited 0.
- **Inputs:** nothing was encoded. All embeddings were read from the W22-04 scratch caches after their text digests were checked.

## Verdicts

| Sub-test | Bundle(s) | Verdict |
|---|---|---|
| (a) top-k flip pairs on identical observations | #7/#15 | **SUPPORTED** (the phenomenon is PRESENT). A random-ranking null also passes the pooled criterion and PRESENT; see (a). |
| (b) chain vs direct Procrustes refit | #54 | **DEFERRED_SCOPE**: not run, no verdict |
| (c) centering-gain law, leave-one-domain-out | #16 | **KILL**: R2_LODO = -0.476, below 0.5; sign accuracy 1.00 |
| (d) whole-query conformal containment | #37 | **KILL**: median K* is 329 in-domain and 393 out-of-domain, above the limit of 200. The coverage-drop rule depends on trec-covid. |
| (e) bit/sample ledger | OpenAI #140 | descriptive only; the #140 bound is NOT tested |

**Reproduction check (reported, not used as a gate):** for `CenteredProcrustes x CSLSMerge` and `UncenteredProcrustes x CSLSMerge`, nDCG@10 matches W22-04's recorded cells exactly (absolute difference 0.0) on all 12 domains. The row-collision scrub removed 0 rows everywhere.

## (a) Flip pairs (current numbers, after the Tier B fix)

**Method.** The served top-k S_k is ranked by the raw served (fused) scores. The DEV affine calibration c = a + b*fused is used only in the margin test |c_i - c_j| <= m.

**Pooled results over 2549 eval queries**
- 11994 native docs are missing from the served top-10. The margin explains 11939 of them (**0.99541**).
- Share of queries with at least one flip pair: 0.99686.
- Share of queries with any top-10 error: 0.99725.
- Mean missing docs per query: 4.705.
- Domains at or above 0.50: 12/12.

**Calibration.** The margins m range from 0.096 to 0.153. The OLS slopes b range from 0.07 to 0.30, except trec-covid, where b = -0.097.

**trec-covid**
- The calibration slope is negative: on DEV, the served scores order the documents in reverse of the native order.
- It misses 7.17 of the 10 native docs per query.
- Native nDCG there (0.388) is below served (0.484), so the native ranking is not a quality ceiling for that domain.

**touche-2020** has the lowest share of queries with a flip pair, 0.853.

**Null baseline (Tier B; `scratch/tierb/attack_a.py`, not re-run by the author).** Each null keeps the same a, b and m per domain.

| Null | Pooled share explainable | Share of queries with a flip pair |
|---|---|---|
| Randomly permuted served ranking | 0.534 | 0.989 |
| Random served scores | 0.581 | 0.997 |

- **A random served ranking passes both the pooled criterion (≥ 0.50) and PRESENT.** PRESENT is therefore uninformative.
- Only the per-domain rule fails under the permuted null: touche 0.488, android 0.497 and programmers 0.374.
- The real pipeline's 0.995 is well above both nulls. The defensible claim is narrow: the served pipeline's top-10 errors are far more often margin-level than chance would produce.
- The SUPPORTED verdict must not be cited as evidence for a sharp margin law. m is the 0.90 quantile of |residual|, about 2–3x the median residual, and the small slopes compress the calibrated scores.

## (c) Centering law (leave-one-domain-out over 12 domains): KILL

**Variables**
- Gain g, Centered minus Uncentered nDCG@10: 0.0325 to 0.0796, positive on all 12 domains.
- Predictor x = e_Unc - e_Cen: -0.0035 to 0.0421.

**Fit**
- Leave-one-domain-out: R2_LODO = **-0.4763**, sign accuracy = 1.00.
- In-sample: R² 0.0132, beta 0.130, alpha 0.0548.
- Spearman correlation of x and g: 0.210.

**Why sign accuracy proves nothing.** The prereg (§0) disclosed in advance that the sign test would almost certainly pass: Centered beat Uncentered on all 12 domains in W22-04. Every prediction is carried by the intercept.

**Why the achievable R² is capped.** g covers a narrow range (spread about 0.047) and carries sampling noise from per-domain nDCG.
- A constant predictor, the leave-one-out mean, scores R² = -0.190. That value is an identity, 1-(12/11)², whatever the data.
- The law scores -0.476, worse than that constant.
- gis has high leverage: hat value 0.453, against 0.09–0.24 for the other domains. Its large x (0.042) comes with an ordinary g (0.044).

**Reading.** Centering helps on every domain, but the held-out alignment-error gap does not predict by how much.

## (d) Conformal containment (alpha = 0.10, k = 10): KILL

**Rule results**
- Median K* = **329** in-domain and **393** out-of-domain, against a limit of 200. The maximum is 602 in-domain (trec-covid) and 402 out-of-domain.
- Mean coverage is 0.9217 in-domain and 0.8615 out-of-domain, a drop of **6.03 pp** (unweighted mean over domains).
- Nominal minus out-of-domain coverage is 3.85 pp, which does not trip rule 4.
- Rules 1, 2 and 3 trip. The secondary check that uses the maximum instead of the median also returns KILL.

**Rule 3 depends on trec-covid.** The 6.03 pp drop is an unweighted mean over domains.
- Weighted by query, it is 3.29 pp.
- Without trec-covid (a 47.4 pp drop with n_cal = 16), it is about 2.27 pp.
- Both of these are under 5 pp.
- **The KILL therefore rests on rules 1 and 2, which are robust.** Tier B reports a median K* of 213.5 even at alpha = 0.20.

**Other per-domain coverage drops:** english 27.1 pp, climate-fever 17.4 pp, dbpedia 9.2 pp. trec-covid and touche-2020 have only 16 calibration queries each.

**Group transfer**

| Direction | K* | Coverage |
|---|---|---|
| prior7 to remaining5 | 301 | 0.7955 |
| remaining5 to prior7 | 448 | 0.9615 |

**Practical reading.** To contain the native top-10 for 90% of queries, the served list would need 300–600 of the 750–954 docs. That is useless as a re-rank budget.

## (e) Ledger (descriptive; per-domain values in `per_domain[*].ledger`)

Each domain has 750 docs (dbpedia has 954). d_src = 4096 (Qwen3) and d_tgt = 384 (MiniLM).

| Quantity | Value |
|---|---|
| Bridge parameter bits ÷ (d_src*d_tgt*32) | Uncentered 1.000, Centered 1.00285, ZeroMu 1.00545 |
| Centered bridge parameters (fp32) | 6.31 MB (Q is 6.29 MB, plus the means and scale) |
| Added bytes ÷ full re-index bytes (fp32), Centered | 0.7137 (dbpedia 0.6041) |
| New-encoder forward passes ÷ full re-index | 0.5933 (trec-covid, touche); 0.68 (other prior7, 280 anchors); 0.9733 (remaining5, 500 anchors); 0.8082 (dbpedia) |
| Samples | 280 or 500 anchors; 80 DEV queries (15 for trec-covid and touche); 160 or 180 dev-ref docs |

- **Storage:** at this corpus size, the 4096x384 bridge matrix alone is about half the size of a full fp32 re-index.
- **Compute:** the anchors, DEV queries and backfill together need 59–97% of the forward passes of a full re-index.
- **Reading:** the migration saves little storage and little compute here. The #140 `d log(1/eps)` lower bound is not tested.

## Departures from the prereg and the queue item

1. **(b) not run.** The bge-small, mpnet and bge-m3 encoders are in the local HF cache, but their embeddings of this corpus are not cached. The prereg and the task barred any encoding in this run, so the chain kill rule is untested (prereg §5).
2. **(c) uses leave-one-domain-out instead of leave-one-encoder-family-out.** Only one encoder pair is cached (prereg §3).
3. **(a)'s definition is the author's own.** The A1/F4 Round 1/2 source memos were not found.
4. **The dataset-file sha256 manifests were not compared.** The digests from `_load_group` were ignored. Text identity rests only on the W22-04 cache text digests and the row counts. This is not in the prereg.
5. **DEV margin calibration fuses the DEV queries with DEV itself as the CSLS reference.** The prereg does not state this explicitly.
6. **The log contains "[download] Fetching ... via HuggingFace Hub" lines.** With the offline flags set, these were served from the local cache. That the run stayed offline rests on the environment flags only; no packet capture was taken.
7. **No bootstrap was run.** The prereg allowed seed 2206 for description only.
8. **A bug fix made after the results, found by Tier B round 1.** (a) built S_k from the calibrated scores a + b*fused instead of from the served ranking. With b < 0 (trec-covid), it compared the served *bottom* 10.
   - Fixed in `8ff93b0f`, together with a b<0 regression test. The run was repeated and the results regenerated in `07a073c9`.

     | Quantity | Before the fix | After the fix |
     |---|---|---|
     | trec-covid missing natives | 350 | 251 |
     | trec-covid mean missing per query | 10.00 | 7.17 |
     | Pooled missing natives | 12093 | 11994 |
     | Pooled share explainable | 0.99545 | 0.99541 |

   - The verdict did not change.
9. **Prereg §3 misquotes the kill rule.** It cites "sign holds on <80% of domains" as the queue's wording. The queue actually says "LOFO sign <80% / R^2 <0.5", with no "of domains"; the phrase came from the task handoff's paraphrase. The test applied (sign accuracy over held-out domains) is unaffected.

## Tier B round 1

- **Reviewer and score:** `ad3b89dcbf0ee5912`; BHS_TIER_B 88, severity important.
- **IMPORTANT (fixed):** the served top-k in (a) was ranked by calibrated rather than served scores, which reverses the ranking when b < 0. Fixed in `8ff93b0f` and re-run in `07a073c9` (departure 8). The verdict did not change.
- **Nit 2 (folded into (a)):** a random-ranking null passes the pooled criterion and PRESENT. Only the per-domain rule separates it from the real pipeline. The real 0.995 beats the null.
- **Nit 3 (folded into (d)):** rule 3 depends on trec-covid (3.29 pp query-weighted, about 2.27 pp without trec-covid). The KILL rests on rules 1 and 2.
- **Nit 4 (folded into (c) and departure 9):** the "of domains" misquote is disclosed. R² is capped by the narrow range of g: the constant leave-one-out mean scores -0.190 against the law's -0.476. gis has high leverage.
- **Nit 5 (fixed in `8ff93b0f`):** the (b) docstring now says the encoders are cached, their embeddings are not, and encoding was barred.

## Brutal Honesty

- **L1 (stubs):** none.
- **L2 (escape conditionals):** `conformal_k` returns `n_docs` when ceil((n+1)(1-alpha)) > n. This branch is preregistered and never fired: the smallest calibration set has n = 16, and ceil(17*0.9) = 16, which is ≤ 16.
- **L3 (mocks in production paths):** none.
- **L4 (partial presented as complete):**
  - (b) and the encoder-family LOFO are deferred.
  - The SUPPORTED verdict for (a) is nearly built in: a random ranking passes both the pooled criterion and PRESENT.
  - A real ranking bug in (a) survived self-review and was caught only by Tier B.
- **L5/L8/L12 (untested production paths):** the runner has no unit test of its own; only the two real runs exercised it. The path where it refuses to run without the CPU/offline settings is untested.
- **L11 (broad-catch swallowing):** none.
- **Other:**
  - The exact reproduction shows the pipeline matches W22-04's, not that W22-04's pipeline is correct.
  - trec-covid and touche-2020 have only 16 conformal calibration queries each.

**Evidence and smoke**
- EVIDENCE: `w22_06_results.json` at `07a073c9` (run at `8ff93b0f`), plus the scratch `run.log`.
- SMOKE: floor tier only. That means 18/18 unit tests pass (`python -m unittest tests.test_migration_frontier_w22_06`), ruff is clean, and there were two real CPU runs. There was no ceiling-tier `scripts/smoke.sh`.

**BHS lines**
- BHS_SELF_DRAFT: 90
- BHS_SELF_DRAFT_AGENT: claude-opus-5-5 (W22-06 implementer sub-agent)
- BHS_TIER_B: 88
- BHS_TIER_B_AGENT: fresh general-purpose sub-agent ad3b89dcbf0ee5912
- BHS_TIER_B_SEVERITY: important
- BHS_OFFICIAL: 88, which is min(90, 88). This does not merge.
- CARRY_FORWARD: none created by this slice.
- DEFERRED_SCOPE:
  - (b) chain vs direct: needs bge-small, mpnet and bge-m3 embeddings of the corpus, and encoding was barred.
  - (c) encoder-family LOFO: needs at least 3 encoder families.
  - The #140 `d log(1/eps)` bound: needs a learning task and an error target.
- LOOP_ITERATIONS: 2
- OPERATOR_OVERRIDE: none

**Why the self-score is 90:**
- (b) was not run.
- The runner has no test of its own.
- The dataset manifest sha checks were skipped.
- The positive (a) verdict is largely built into the definition, as the random-ranking null shows.

## Scratch receipt (removable)

- **Path:** `C:\Users\mattm\AppData\Local\Temp\claude\439e695c92b40043\c9f4318c-455f-427a-844a-07562a0a3c42\scratchpad\w22-06\artifacts\pv7_w23b_w22_06\scratch\`
- **Size:** 844,248 bytes in 7 files.

| File | Size | What it is |
|---|---|---|
| `run.log` | 3,885 B | current run |
| `w22_06_per_query.jsonl` | 411,782 B | current run |
| `run_r1_prefix.log` | 3,885 B | first run, before the fix |
| `w22_06_per_query_r1_prefix.jsonl` | 411,886 B | first run, before the fix |
| `tierb/attack_a.py` | 4,035 B | Tier B script |
| `tierb/attack_a.log` | 4,604 B | Tier B output |
| `tierb/recompute_cd.py` | 4,171 B | Tier B script |

- Gitignored through `artifacts/**/scratch/`. Nothing was deleted.
