# W22-06 Preregistration — Migration Information Frontier (CPU-only)

Frozen before any W22-06 result is computed. This file is committed alone, before
the analysis module and runner are written.

- Queue item: `W22-06` in `docs/research/wave22_queue_additions_2026-11-25.json`
- Branch: `pv7/w23b-w22-06-migration-frontier`, based on the W22-04 tip `91c5cc7a`
- Bundles tested: #7, #15, #16, #37, #54. Math extension folded in: OpenAI #140 (bit ledger).
- Kill rules, quoted from the queue: "Kill chain claim if direct refit >= chain everywhere;
  kill centering law if LOFO sign <80% / R^2 <0.5; kill practical coverage policy if
  set size >20x k at alpha=0.10 or OOD coverage drops >5 pp."

## 0. What the author had seen before freezing

The W22-04 results (`artifacts/pv7_w23a_w22_04/w22_04_results.json`) were already
committed, and the author had read them. That includes the per-domain
`CenteredProcrustes__CSLSMerge` and `UncenteredProcrustes__CSLSMerge` nDCG@10. Centered
was higher than Uncentered on all 12 domains there. The sign part of the centering law
in §3 is therefore nearly guaranteed to pass, and R^2 is the binding criterion.

The author had NOT computed any of the following before freezing:
- the §3 predictor;
- any score-margin, flip-pair or containment quantity;
- any ledger number.

W22-04's KILL verdict is not used anywhere. Only its cached embeddings and data loaders
are used.

## 1. Data, compute and the frozen serving pipeline

- **Domains.** The 12 W22-04 domains, loaded with W22-04's loaders.
  - prior7: `trec-covid`, `touche-2020`, `cqadupstack-{android,physics,programmers,stats,unix}`.
    All 280 anchors are used to fit.
  - remaining5: `dbpedia`, `climate-fever`, `cqadupstack-{english,gis,tex}`. The first
    500 anchors are used to fit.
  - Data seed 20261125.
  - The global row-collision scrub is applied exactly as in W22-04.
- **Embeddings.** Read only from the W22-04 scratch caches: MiniLM-384 and Qwen3-4096.
  Each cache is checked against the text digest before use.
  - No encoding of any kind is done.
  - Environment: `CUDA_VISIBLE_DEVICES=""`, `HF_HUB_OFFLINE=1`, `HF_DATASETS_OFFLINE=1`.
  - The runner refuses to start if any of these is not set.
- **Serving pipeline P** (Mode B, identical to the W22-04 cell `CenteredProcrustes x CSLSMerge`):
  - Queries are Qwen3-4096.
  - 80% of eval docs are served from MiniLM-384 through
    `CenteredProcrustes.fit(Qwen anchors, MiniLM anchors)` (ambient scale).
  - The other 20% are served natively in Qwen3-4096. These are chosen by the `bf20:{domain}` mask.
  - The two shards are fused with `CSLSMerge(csls_k=10)`. The DEV references are Qwen DEV
    queries scored against the eval corpus, with `strict_disjoint_dev=True`.
  - k = 10.
  - The fused matrix is computed with the harness classes directly. As a reproduction
    check, its nDCG@10 is compared to W22-04's recorded cell. This check is reported, not gated.
- **Reference ranking ("native", the full re-index ceiling).** Cosine of Qwen3-4096
  queries against Qwen3-4096 eval docs, over ALL eval docs. The native top-k is N_k(q).
  The served top-k is S_k(q). Both use the harness canonical tie-break `(-score, doc_id)`.

## 2. Sub-test (a) — observation-identical top-k flip pairs on empirical margins

The source memos (A1/F4, Round 1/2) were not found in the repo. The operationalization
below is the author's own and is disclosed as such.

- **Margin calibration (per domain, DEV only).**
  1. Compute fused DEV scores F_dev by applying the same pipeline P to the Qwen DEV
     queries over the eval corpus. CSLS r_d comes from the same DEV reference, as in
     the serving path.
  2. Compute native DEV scores G_dev.
  3. Restrict to the boundary region B_dev: (dev query, doc) pairs whose doc is in the
     native top-50 or the fused top-50 for that query (50 = 5k).
  4. Fit OLS `G ≈ a + b*F` on B_dev.
  5. Residual `r = G - (a + b*F)`. The margin is `m = quantile_0.90(|r|)` over B_dev
     (alpha = 0.10).
- **Eval.** The calibrated served score is `c = a + b*F_eval`.
  - Intruders I(q) = S_k \ N_k.
  - Missing natives M(q) = N_k \ S_k.
  - A missing native j is **margin-explainable** if some intruder i satisfies |c_i - c_j| <= m.
  - An **observation-identical flip pair** is any (i, j) in I x M with |c_i - c_j| <= m.
- **Metrics (per domain and pooled over all eval queries):**
  - `frac_queries_with_flip`: share of queries with at least one flip pair.
  - `frac_missing_margin_explainable`: margin-explainable missing natives divided by all
    missing natives, pooled.
  - Mean |M(q)|.
  - The margin m and the coefficients (a, b).
- **Decision (descriptive claim; the queue gives no kill rule for (a)).** "Migration
  top-k errors are margin-level" is SUPPORTED iff both of these hold:
  1. pooled `frac_missing_margin_explainable >= 0.50`;
  2. at least 10 of 12 domains have `frac_missing_margin_explainable >= 0.50`.

  Otherwise it is NOT_SUPPORTED.

  Separately, the margin-vulnerability phenomenon is reported as PRESENT iff pooled
  `frac_queries_with_flip >= 0.10`.

## 3. Sub-test (c) — centering-gain law (LOFO)

**Departure frozen here.** The handoff defines LOFO as leave-one-*encoder-family*-out.
Only 2 encoders are cached, a single source/target family pair, so encoder-family LOFO
is impossible without re-encoding. It is DEFERRED_SCOPE (see §5).

The law is tested as **leave-one-domain-out (LODO) over the 12 domains**. This matches
the kill rule's wording, "sign holds on <80% of domains".

- **Outcome.** For each domain d, `g_d = nDCG@10(Centered x CSLS) - nDCG@10(Uncentered x CSLS)`.
  Both are computed fresh with the pipeline in §1, with the bridge swapped.
- **Predictor (single, label-free, held-out).**
  - Use the dev-ref docs, which have both encodings and were never used to fit.
  - For each bridge B, compute `e_B = mean(1 - cos(B.transform(Qwen_devref), MiniLM_devref))`.
  - `x_d = e_Uncentered - e_Centered`.
- **Law.** `g = alpha + beta * x`, fitted by OLS.
  - LODO: fit on 11 domains and predict the held-out domain, giving ĝ_d.
  - `R2_LODO = 1 - sum(g - ĝ)^2 / sum(g - mean(g))^2`. The mean is over all 12 domains.
  - `sign_acc = mean[sign(ĝ_d) == sign(g_d)]`. A zero counts as a mismatch unless both are zero.
- **Kill rule (verbatim threshold).** KILL the centering law iff `sign_acc < 0.80` OR
  `R2_LODO < 0.5`. Otherwise NOT_KILLED.
- **Secondary (reported, not decision-bearing):** in-sample R^2, Spearman(x, g), and the
  per-domain table.

## 4. Sub-test (d) — whole-query top-k conformal containment

- **Event.** The native top-k is contained in the served top-K.
  - Nonconformity score: `s(q) = max over j in N_k(q) of rank_P(q, j)`.
  - rank_P is the 1-indexed served rank under P, with the canonical tie-break.
  - `s(q) <= K` if and only if `N_k(q) ⊆ S_K(q)`.
  - No qrels are used.
- **Split.**
  - Each eval query goes to CAL or TEST by the parity of the first byte of
    `sha256("w22-06-cal:{domain}:{canonical_qid}")`: even goes to CAL, odd to TEST.
  - The split is deterministic. No random seed is needed.
- **Split-conformal quantile.** `K* = the ceil((n+1)(1-alpha))`-th smallest CAL score,
  with alpha = 0.10. If `ceil((n+1)(1-alpha)) > n`, then `K* = n_docs` (the trivial set).
  - **In-domain:** K*_d is computed from domain d's CAL half. `cov_in_d` is the share
    of d's TEST half with `s <= K*_d`.
  - **OOD (leave-one-domain-out):** K*_{-d} is computed from the pooled CAL halves of
    the other 11 domains. `cov_ood_d` is the share of d's TEST half with `s <= K*_{-d}`.
- **Kill rule (practical coverage policy).** KILL iff ANY of the following holds:
  1. median over domains of K*_d > 20k = 200;
  2. median over domains of K*_{-d} > 200;
  3. `mean_d(cov_in_d) - mean_d(cov_ood_d) > 0.05`;
  4. `(1 - alpha) - mean_d(cov_ood_d) > 0.05`, i.e. OOD coverage falls more than 5 pp
     below nominal.

  Otherwise NOT_KILLED.
- **Secondary (reported, not decision-bearing):**
  - max over domains of K*_d and K*_{-d};
  - per-domain coverage drops;
  - the verdict that would result if "median" were replaced by "max";
  - prior7 to remaining5 group-transfer (calibrate on one group's CAL, test on the other
    group's TEST), both directions.

## 5. Sub-test (b) — chain vs direct Procrustes refit: DEFERRED_SCOPE

The #54 chain is MiniLM-384 -> bge-small-384 -> mpnet-768 -> bge-m3-1024 -> Qwen3-4096.
It needs bge-small, mpnet and bge-m3 embeddings. None of them is cached on this branch.

The task instruction says: "If a part truly needs re-encoding, scope it out as
DEFERRED_SCOPE instead of using the GPU."

Sub-test (b) is therefore NOT RUN, and no chain verdict is issued. The kill rule
"direct refit >= chain everywhere" remains untested. Encoder-family LOFO for (c) is
deferred for the same reason.

## 6. Sub-test (e) — retained-bit / sample ledger (OpenAI #140)

This is a descriptive accounting with no kill rule. For each domain, and for each of
`UncenteredProcrustes`, `CenteredProcrustes` and `ZeroMuPolarBridge` (+CSLS), plus two
reference points (full re-index and legacy-only), count the items below.

- **Retained artifacts:** bridge parameters (matrix, means, scale), CSLS r_d vectors,
  ZeroMu sidecar and state, the backfill index set, the native backfill embeddings, and
  the legacy index.
  - Bytes are reported at fp32 and fp64.
  - Precision metadata is counted as 8 bytes per array (dtype + shape tag).
  - The legacy index is reported separately as "already retained".
- **Work to recover:** the number of new-encoder forward passes, which is anchors + DEV
  queries + backfilled docs. This is compared with full re-index (all docs).
- **Sample counts:** anchors, DEV queries, dev-ref docs.
- **#140 lower bound.** Only the ratio of bridge-parameter bits to `d_src * d_tgt * 32`
  is reported.
  - The `d log(1/eps)` memory–sample lower bound is NOT tested. It needs a learning task
    and an error target. Only the Gaussian synthetic control in #140 has those, and it is
    out of scope here.

## 7. Seeds and constants

| Constant | Value |
|---|---|
| Data seed | 20261125 |
| Backfill mask | `bf20:{domain}` sha256 (W22-04) |
| Conformal split | sha256 parity (§4) |
| Bootstrap | none in decision rules; any descriptive bootstrap CI uses seed 2206 with 2000 resamples |
| k | 10 |
| csls_k | 10 |
| alpha (conformal) | 0.10 |
| Margin quantile | 0.90 |
| Boundary region | top-50 |
| remaining5 anchors used to fit | 500 |

## 8. Outputs

- Results JSON: `artifacts/pv7_w23b_w22_06/w22_06_results.json`. Committed.
- Bulk per-query dumps: `artifacts/pv7_w23b_w22_06/scratch/`. Gitignored via
  `artifacts/**/scratch/`.
- No GPU, no network, no deletion of any input.
