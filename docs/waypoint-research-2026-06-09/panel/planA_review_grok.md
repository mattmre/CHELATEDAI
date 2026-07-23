# ADVERSARIAL PLAN REVIEW — D1/D2/D3 (Grok 4.5)
# Role: attack the build brief + panel protocol; propose leanest rigorous plan
# Date: 2026-07-09
# Grounding: paper-draft/main.md, scripts/ladder_evalsplit.py, agent-build harness (#291),
#            out_codex.txt protocol, roundtable-synthesis.md. No code built.

---

## Executive verdict

**Ship D1 hard. Scope D3 to a calibrated predictor + one-pager. Run only a stripped D2
existence screen — do not implement Codex's full Quora × 7-corrector × 6-baseline matrix.**

The build brief as written will not finish honestly on one 3090. Worse: a faithful full-D2
run is structured to produce **false negatives on chelation and false positives on
"local > global"** unless several design defects are fixed first. D1's bootstrap fix is
necessary but **not sufficient** to rehabilitate the ceiling claim. D3 as a "theorem"
is hand-wavy; as a **margin-based calibrated estimator** it is the only high-ceiling
adjacent idea and should be protected from D2 thrash.

Odds if we execute the brief literally:
- D1 ships publishable: **high** (~70%) *if* language is downgraded and bootstrap lands
- Full D2 finishes with binding kill criteria: **low** (~25%) — will be cut mid-flight
- D3 becomes a real certificate rather than a blog formula: **low-mid** (~20%) under full scope
- Chelation revival via D2: **~10–15%** (panel prior; I do not revise upward)

---

## 1. Attacks on D2 (crossover) — false positives / false negatives

### 1.1 False POSITIVE paths (declare a win that is not real)

**(FP-A) Synthetic Regime-C is labeled with the corruption that creates it.**
Corruption = "collapse 8/16 clusters toward centroid." If the detector is scored with
cluster labels derived from the *same* k-means that defined the corruption, AUPRC ≥ 0.80
is almost free: cohesion/variance features literally recover the injection mask.
**This is not prediction of retrieval failure; it is recovery of the simulation knob.**

Mitigation (binding): detector features must be computed **without** access to the
injection mask *and* without the k-means that generated it (or use a held-out clustering
seed / different K). Score AUPRC on **retrieval-loss clusters** (docs whose corruption
actually moves NDCG for queries that touch them), not on injection labels alone.

**(FP-B) Local ridge beats global ridge by overfitting the synthetic piecewise map.**
Regime-C is *literally* piecewise constant toward centroids. Independent per-cluster
ridge with K matching the injection is the MLE of the data-generating process. A ≥5-pt
local win here does **not** transfer to natural stores and does not vindicate chelation
(paired local maps are not chelation). Drift-Adapter already showed metadata-routed locals
help on synthetic heterogeneous drift (0.85→0.94 ARR). Replicating that is confirmation,
not discovery.

Mitigation: (i) select K on validation with a **parameter-matched** baseline (global +
low-rank local residual, shared A); (ii) require the win to survive **soft routing** and
**cluster-boundary queries**; (iii) never claim chelation-win from paired local ridge.

**(FP-C) "Beat CBIE/hubness" on a corruption that CBIE was designed for.**
Centroid-collapse *is* a cluster-isotropy / hubness toy. Established local scaling and
CBIE are the correct nulls. If chelation barely edges them after hyperparameter fishing
on test, that is selection bias, not a mechanism win.

Mitigation: freeze all chelation knobs (α, detector thresholds, anneal schedule) from
**paper §3 / harness defaults** before Regime-C; one validation pass max; Holm vs the
full baseline family, not vs a cherry-picked weak baseline.

**(FP-D) Recovery R with tiny oracle gap inflates "points."**
If clean−corrupt NDCG gap is 0.03, a 5-pt recovery is 0.0015 NDCG — below the 0.02
absolute NDCG bar in spirit, even if both are "met" on a lucky seed. The dual threshold
helps only if both are enforced **jointly on the same bootstrap CI**, not as alternative
readings of the point estimate.

Mitigation: pre-register that **both** ΔR ≥ 5 pts **and** ΔNDCG ≥ 0.02 must have
paired-bootstrap 95% CIs excluding 0 (not just point estimates), and oracle−floor gap
must be ≥ 0.05 NDCG or the cell is marked underpowered / excluded.

**(FP-E) Same-sign-in-5-seeds without multiplicity discipline.**
5 seeds × many correctors × 3 β × K-grid × datasets ≈ dozens of tests. Holm only vs
global ridge is under-correction. "Same sign in 5 seeds" is weak when effect is noise-
centered near zero (all seeds slightly positive by chance with small n_queries).

Mitigation: primary comparison list ≤ 4, pre-registered; everything else exploratory.
Hierarchical bootstrap over (seed, query). Report family-wise error for the primary list.

### 1.2 False NEGATIVE paths (kill something that deserved a fair test)

**(FN-A) Bounded chelation α=0.05 is set up to fail even "home turf."**
At α=0.05 the max rotation is ~2.86°. Regime-C with β=0.20 can move points far more than
that toward the centroid. A near-identity bound will fail a large synthetic collapse **by
construction**, same geometry as the upgrade arena. That kills the *bound*, not the
detection or local-routing idea.

Mitigation: Regime-C primary kill should be on **unbounded local/topology-gated repair**
OR on a β schedule where the required displacement is ≤ α (e.g. β∈{0.02,0.05,0.10} with
measured mean ‖x′−x‖). Separately report bounded α=0.05 as a stress test of the design
premise, not as the sole kill arm.

**(FN-B) Detector API mismatch.**
`IsomerDetector` compares *query result lists* (Jaccard / isomer strength), not
"which document clusters are corrupted." `TopologyAnalyzer.compute_cluster_connectivity`
needs labels you already have. Neither currently outputs a calibrated per-cluster score
for AUPRC against an injection mask. Wiring them naively will yield AUPRC ~ chance →
false kill of "detection."

Mitigation: define one explicit `score_clusters(doc_embs) -> (n_clusters,)` using only
pre-declared features (cohesion drop vs snapshot, bond-ratio shift, hubness). Preregister
features. Do not invent post-hoc scores after seeing AUPRC.

**(FN-C) SciFact/NFCorpus underpower the absolute NDCG bar.**
Canonical ladder uses ~60 eval queries after anchor split (max_queries=100, fraction 0.4).
Paired bootstrap CIs will be **wide**. Requiring ΔNDCG ≥ 0.02 with CI excluding 0 can
reject true 0.03 effects. Full Quora is the opposite problem (won't finish).

Mitigation: **power analysis first** on existing SciFact query-level NDCG vectors. If
median CI half-width for ridge−MLP already > 0.02, SciFact cannot host the win rule.
Then either raise query count (full SciFact test, not road-course 100) or admit the
0.02 bar only on Quora-subset ≥2k queries.

**(FN-D) Paired global/local regression as "privileged oracles" still leak into narrative.**
If the write-up leads with "local ridge recovered 90%," readers will hear chelation won.
Chelation is unpaired + bounded + detector-triggered. Keep paired methods in a
**diagnostic lane** only; kill criteria apply only to unsupervised / detector-gated arms.

**(FN-E) Isotropy≠quality landmine (panel).**
Even a true AUPRC=0.9 detector of "collapsed clusters" can fail to improve NDCG after
repair (ACL 2023 anisotropy literature). Detection success ≠ retrieval success. A plan
that kills only on detector AUPRC or only on NDCG will mis-fire.

Mitigation: **two independent kill axes**, both required for "alive":
1. Detection: AUPRC ≥ 0.80 on retrieval-harmful clusters (not injection cosplay).
2. Correction: beats CBIE **and** hubness on NDCG with the dual CI rule.
Fail either → mechanism dead. Pass detection / fail correction → detectors might be
systems-useful for *budgeted re-embed*, not for chelation repair (Grok panel P3 residual).

### 1.3 Structural overscope in the brief's D2

Full Codex matrix (Quora 522k docs × exact retrieval × 5 seeds × correctors
{ridge, Proc, MLP×3 widths, K∈{2,4,8,16}, global+LR local, graph k∈{16,64,256},
chelation} × Regime-C baselines {ZCA, AbT, CBIE, hubness×2} × β×3) is a **campaign**,
not a deliverable. On one 3090:

| Work item | Order-of-magnitude cost |
|---|---|
| Embed Quora 522k × 2 encoders | multi-hour to multi-day + disk; feasible once if cached |
| Exact cosine 10k queries × 522k | heavy unless FAISS/IVF; "exact" is the budget killer |
| MLP 400-epoch style × seeds × widths | dominates GPU if naively looped |
| Graph kNN on 64k anchors | RAM/time; doable at subset scale only |
| 10k bootstrap × all cells | CPU trivial if query-level scores cached; catastrophic if re-retrieve |

**Will not finish if treated as one atomic milestone.** The brief already says "feasible
subset first" — that must be the *only* D2 until a go gate, not a warm-up before Quora.

### 1.4 Deeper protocol hole: Regime-C is not "within-model collapse"

Grok panel E★ asked for **progressive** fixed-encoder corpus stress (inserts, near-dupes,
query-shift). Codex Regime-C is a **one-shot synthetic collapse**. Passing or failing it
does **not** answer "does the disease exist in nature?"

- Regime-C pass + no natural disease = lab curiosity.
- Regime-C fail = mechanism fails even when we gift it the pathology → strong kill.
- Natural disease without Regime-C = you never built the toy correctly.

Honest sequencing: **Regime-C is a necessary mechanism screen, not a sufficiency proof.**
Do not advertise D2 alone as settling within-model chelation for production stores.

---

## 2. Attacks on D1 — where the statistical fix falls short

### 2.1 What bootstrap CIs actually buy

10k paired-query bootstrap on the **fixed** SciFact eval-split ladder buys:
- uncertainty on NDCG and on Δ(NDCG_ridge − NDCG_MLP), Δ(ridge − C3a);
- honest error bars in Table 1 / Fig 1.

It does **not** buy:
1. **Anchor-split uncertainty** — fit half is one `RNG.permutation` (seed 42). Bootstrap
   queries conditional on a fixed W. Need outer bootstrap or multi-seed refits of W.
2. **Projection-seed uncertainty** — `QueryEncoderDrift` projection is one seed.
3. **MLP optimization uncertainty** — one training run ≠ capacity ceiling.
4. **A mathematical ceiling** — CIs around an observed plateau still do not make a ceiling.
5. **Held-out reconstruction** — ranking can plateau while MSE still falls (or vice versa).

If the plan stops at "bootstrap CIs + rename ceiling → plateau," a competent reviewer still
asks: *how many independent maps?* One learning curve over anchor count on **one** split
is the minimum extra; without multi-seed W refits, "plateau" is still fragile.

### 2.2 Learning-curve traps

Anchor-count curve for ridge vs MLP vs C3a:
- **C3a is not the same supervision** (anchor-InfoNCE on queries vs doc-paired regression).
  Plotting them on one x-axis labeled "anchors" confounds sample size with supervision type.
  The paper's own +59 pt story is supervision-type, not n. A curve that re-learns that is
  fine only if axes are honest: for ridge/MLP, n_doc_pairs; for C3a, n_query_anchors and
  steps/lr held to campaign values (or swept and reported).
- **x-axis range**: SciFact road-course has 1200 docs → 600 fit pairs. You cannot claim
  saturation toward Drift-Adapter's 20k regime. At n≈600, d=384, ridge's win over lstsq
  is expected regularization; MLP may still be under-optimized.
- **Selection on the curve**: picking best λ / width on the same eval queries reintroduces
  the ceiling fallacy. λ and MLP early-stop on a **validation doc reconstruction** or
  held-out query split, then freeze for the reported eval bootstrap.

### 2.3 C3a comparison integrity

Campaign C3a NDCG is often **imported as a constant** (`C3A = 0.1609` in
`ladder_evalsplit.py`) while maps are re-fit in a self-contained NDCG. Pipeline note in
§5.5 already admits absolute NDCG differs across harness vs self-contained. For CIs on
ridge−C3a you need **query-level scores from the same retrieval implementation**, not a
scalar campaign mean. Otherwise the 4.2× claim's CI is theater.

Minimum honest path: re-run C3a once on the matched eval-split with per-query NDCG
vectors saved; bootstrap paired against ridge on those vectors.

### 2.4 Language risk after the "fix"

Paper currently says "linear post-hoc ceiling" in abstract, intro, §5.5, figures.
Downgrade must be **global**, not one sentence in §5.5. Remaining landmines:
- "by construction" affine language (if any residual)
- Procrustes bounds cited as *explaining* 84% NDCG (they don't; margins do)
- "irreducible final ~16%" without residual decomposition (capacity vs text-only info)

§3 is still a stub ("will detail…"). Shipping without §3 math for bound α and adapter
forward maps is a methodology-paper fail. LaTeX/arXiv is optional; **§3 + stats + language
pass** is not.

### 2.5 Reproducibility gap

Scripts live git-excluded under `docs/waypoint-research-.../scripts/` with hardcoded
worktree paths (`gpu-campaigns`). "One repro command" is false until either:
- scripts are moved to the importable harness with CLI flags, or
- a frozen artifact pack (embeddings + per-query scores + manifest SHAs) allows
  bootstrap/learning-curve **without** re-embedding.

On a 3090, prefer **artifact-first D1**: serialize Do, Dor, Qd, qrels, fit indices once;
all stats offline. That is faster and more honest than re-running ST encodes for every
paper edit.

---

## 3. Attacks on D3 — is the recoverability certificate well-posed?

### 3.1 What is well-posed

A **sufficient condition** in the spirit of Codex's margin inequality is well-posed:

For each query q and relevant/irrelevant pair (r,j) with oracle margin
m = q·y_r − q·y_j > 0, the mapped pair preserves order if
m > ‖q‖ (‖e_r‖ + ‖e_j‖) (loose) or a tighter residual form after an affine fit on
held-out pairs.

From **paired** (X, Y_oracle) calibration docs + query vectors, one can estimate:
- Gram / Procrustes distortion: residual after best orthogonal/affine map on held-out pairs;
- residual norms ‖e_i‖ distribution;
- empirical margin distribution on a **calibration query set**.

Then predict expected pairwise inversion rate → proxy for NDCG@10 recovery bands
{<70, 70–90, >90, >99}.

That is a **predictor**, not yet a certificate.

### 3.2 What is hand-wavy if left as briefed

| Phrase in brief | Failure mode |
|---|---|
| "theorem" | No free theorem from Gram distortion alone → NDCG; needs margin model + independence assumptions that fail for top-k |
| "BEFORE fitting" | Gram distortion for affine recoverability **is** the fit residual of an affine map (or a CV residual). "Before deployment" ≠ "without any paired algebra." Clarify: before **full corpus** re-embed / before committing to a corrector class |
| "~70/90/99%" | Discrete bins imply calibration data across many regimes; you currently have ~few encoder pairs × 2 datasets. Bins will be overfit unless leave-one-regime-out |
| "validate on our existing runs" | Training the calibrator on SciFact mpnet and "validating" on the same run is circular. Need held-out **regime** (bge-large, NFCorpus, different anchor n) |
| Point-map residual only | Ignores that ranking uses only top-k competitors; average residual can be large while top-k margins are safe (or reverse) |

### 3.3 Minimum non-hand-wavy D3

1. **Define features (frozen list):**
   - held-out affine residual: mean/p90 cosine error, Frobenius after ridge CV;
   - Procrustes disparity / Gram skew indicators;
   - query margin quantiles (p10/p50 of oracle top1−top2, topk gaps) under floor and under
     predicted mapped docs using **cross-fitted** residuals (no test-query leakage).
2. **Output:** predicted recovery R̂ and a conservative lower band R̂_lo (e.g. 10% quantile
   over residual bootstrap).
3. **Calibration protocol:** fit monotone map or isotonic regression from features → R on
   **training regimes**; evaluate MAE / bin accuracy on held-out regimes
   (at least: SciFact-mpnet train → NFCorpus and/or bge-large test).
4. **Success bar (honest):** Spearman(R̂, R) ≥ 0.7 across ≥6 distinct (dataset, encoder,
   n_anchor) cells **or** 80% correct three-way binning {<0.7, 0.7–0.9, ≥0.9} on held-out
   cells. If bar fails, ship as **negative methodology note**: "margin residual proxy
   insufficient under our sample."
5. **Do not claim theorem** until a proved bound matches empirical coverage
   (e.g. R ≥ 1 − f(residual, margins) with empirical coverage ≥95% on held-out).
   Default deliverable name: **Recoverability Estimator (prototype)**, certificate only if
   coverage validates.

### 3.4 D3 dependency on D1, not on D2

D3 needs query-level margins and residual vectors from the upgrade arena — the same
artifacts D1 should freeze. D2's synthetic collapse is a **different** recoverability
question (unpaired). Do not block D3 on D2. Do not dilute D3 into "also runs on Regime-C"
in v1.

---

## 4. What is over-scoped and will not finish

| Item | Verdict |
|---|---|
| Full BEIR-Quora 522k exact retrieval matrix | **Cut** from critical path; optional after subset go |
| MLP width sweep 256/512/1024 × 5 seeds | **Cut** to one residual MLP, fixed width, val-early-stop |
| Independent full per-cluster ridge K∈{2,4,8,16} | **Cut** to K∈{4,16} or replace with global+LR-local only |
| Graph residual k∈{16,64,256} | **Cut** from D2 v1 (P2 novelty low; expensive) |
| All-but-the-top + ZCA + CBIE + dual hubness + chelation | Keep **CBIE + one hubness + ZCA**; AbT optional |
| β∈{0.05,0.10,0.20} full factorial | Start **β=0.10 only**; add β only if under/over powered |
| D1 LaTeX/arXiv conversion | **Optional** after markdown stats pass |
| D3 "theorem" | **Defer**; estimator only |
| Living-bank / anneal revival work | **Out of scope** (panel: dead) |
| Natural longitudinal E★ corpus growth | **Not in D1–D3 brief**; if chelation still wanted after D2 fail, E★ is the *next* program — do not stuff into D2 |

---

## 5. Leanest plan that is still rigorous

### 5.1 Architecture (minimal modules)

Reuse (agent-build harness, import-only):
- `run_drift_recovery_experiment.py`: `load_mteb_data`, `_split_anchor_eval`, evaluate paths
- `query_encoder_drift.py`: `QueryEncoderDrift`
- `run_road_course_campaign.py`: `select_road_course_slice`
- `drift_recovery_metrics.py`: `ndcg_at_k`
- `chelation_adapter.py` / bound path for C3a + α-shrink
- `topology_analyzer.py` (features only; do not pretend full product loop)

New (prefer under local research tree or a single `research/drift_recovery/` package —
**do not** sprawl 15 root scripts):

```
research/drift_recovery/   # or docs/.../scripts/_lib/ if staying local-only
  artifacts.py             # save/load EmbeddingPack, QueryScorePack, manifests
  maps.py                  # ridge, procrustes, residual_mlp.fit/apply
  stats.py                 # paired_bootstrap, holm, recovery_R, learning_curve
  regime_u.py              # upgrade arena builder from pack
  regime_c.py              # centroid-collapse injector + baselines (ZCA, CBIE, hubness)
  detect_score.py          # preregistered cluster scores for AUPRC
  recoverability.py        # residual + margin features → R̂
  preregister_d2.md        # frozen comparisons, seeds, kill rules (text)
runners/
  d1_bootstrap_curve.py    # offline from packs → tables/figs for paper
  d1_refit_c3a_query_scores.py
  d2_subset_crossover.py   # SciFact+NFCorpus only
  d3_calibrate.py
paper-draft/main.md        # language + §3 + results integration
```

Interfaces (sketch):

```text
EmbeddingPack: Do, Dor|None, Q_eval, doc_ids, q_ids, qrels, manifest
Map: fit(X,Y)->State; apply(X)->Xhat
bootstrap_paired(per_query_scores_a, per_query_scores_b, n=10000, seed)->CI
predict_recoverability(pack_cal, map_family="ridge")-> {R_hat, R_lo, features}
```

### 5.2 Sequencing

```
Phase 0 (0.5–1 day): Artifact freeze
  Re-encode SciFact+NFCorpus packs (mpnet, bge if already in figures).
  Save per-query scores for ridge/MLP/C3a on matched split.
  GO if packs reload and reproduce ladder point estimates within 1e-3 NDCG.

Phase 1 — D1 core (parallelizable writing + stats): 2–4 days
  stats.bootstrap on frozen query scores
  learning curve n ∈ {50,100,200,400,600} docs for ridge/MLP (val-selected λ)
  re-run C3a once for paired vectors if missing
  §3 write bound + adapter maps from code
  global language pass: ceiling → observed plateau; affine-by-construction removed
  GO-D1-SHIP: CIs in main table; no "ceiling" left; one repro path from packs;
              ridge−C3a CI excludes 0 (expected); ridge−MLP CI reported honestly
              (may INCLUDE 0 — that is a feature, not a bug)

Phase 2 — D3 prototype (after Phase 0; parallel to D1 writing): 2–3 days
  residual+margin features; leave-one-regime-out calibration
  one-pager framing
  GO-D3: held-out regime MAE/bin bar met OR explicit failure write-up

Phase 3 — D2 lean screen (only after D1 language frozen / preregister file committed):
  Datasets: SciFact full-eval queries if power requires; else NFCorpus; optional Quora-5k docs subset
  Regime-U: ridge vs 1 MLP vs global+LR-local vs chelation α=0.05 (diagnostic)
  Regime-C: β=0.10, K=16 collapse 8 clusters; baselines ZCA, CBIE, hubness;
            chelation detector AUPRC + repair NDCG; paired ridge diagnostic only
  5 seeds, 10k bootstrap, ≤4 primary tests, Holm
  GO/NO-GO chelation:
    NO-GO if (repair fails dual CI vs CBIE and hubness) OR (AUPRC < 0.80 on harm labels)
    SOFT-GO detection-only if AUPRC ok but repair fails → park as re-embed router idea
    Hard GO only if both axes pass on 2 datasets — then consider Quora scale-up
```

**Parallelism:** Phase 1 writing §3 ∥ Phase 1 stats ∥ Phase 2 feature code after packs.
**Do not** start Phase 3 corrector zoo before preregister markdown exists.

### 5.3 Scope cut: minimum honest vs full

| Deliverable | MINIMUM honest shippable | FULL (only if minimum succeeds) |
|---|---|---|
| **D1** | Packs + 10k paired bootstrap on ridge/MLP/C3a + 1 learning curve + language downgrade + §3 + repro from packs | Multi-seed W refits, bge CIs, LaTeX, residual capacity decomposition |
| **D2** | SciFact+NFCorpus, β=0.10, 4 primary methods, CBIE+hubness+ZCA, detector AUPRC, dual kill | Quora 522k, full K-grid, graph residual, β sweep, MLP widths |
| **D3** | Cross-fitted residual+margin → R̂; leave-one-regime-out; one-pager | Proved bound + coverage certificate + multi-benchmark release |

### 5.4 Top risks and mitigations

| Risk | Mitigation |
|---|---|
| Stats theater (CI on wrong randomness) | Outer multi-seed for maps; query bootstrap for ranking; declare both |
| Overfitting D2 | Preregister file + parameter-matched locals + freeze chelation knobs |
| Scale blowup | Artifact-first; subset-only D2; no exact-522k on critical path |
| Detector cosplay | Harm-based labels; no injection-k-means features |
| Bound false kill | Separate bounded stress from primary unpaired repair arm |
| Isotropy≠NDCG | Dual kill axes |
| D3 circularity | Leave-one-regime-out; rename certificate→estimator until coverage |
| C3a scalar import | Require query-level re-score |
| Scope creep back to living-bank | Explicit out-of-scope list in preregister |

### 5.5 Explicit go/no-go gates

1. **G0 Artifacts:** reload packs → ladder within 1e-3. Fail → stop, fix encode pipeline.
2. **G1 D1 ship:** plateau language + CIs + §3 + repro. Fail → do not open D2.
3. **G2 Power:** median bootstrap half-width for primary ΔNDCG ≤ 0.015 on chosen eval set,
   or increase queries. Fail → change dataset, do not claim kill criteria.
4. **G3 D2 chelation kill (binding):**
   - If Regime-C repair does not beat CBIE **and** hubness under dual CI rule on both
     SciFact and NFCorpus → **terminate chelation corrector in both regimes.**
   - If AUPRC < 0.80 on harm labels → detectors are not predictive; no "mis-benchmarked"
     defense from detection.
5. **G4 D2 optional scale-up:** only if G3 hard-GO; then Quora-subset (not necessarily 522k).
6. **G5 D3:** held-out calibration bar; else ship negative note, keep D1 as main product.

---

## 6. Build-plan recommendation (final)

### Prioritized portfolio

| Priority | Work | Rationale |
|---|---|---|
| **P0** | **D1 minimum** (artifacts, bootstrap, learning curve, §3, language) | Highest EV product; closes the one hole reviewers will hit; almost done |
| **P1** | **D3 estimator prototype** on D1 packs | Only adjacent idea with exceptional upside; small code surface; does not need chelation |
| **P2** | **D2 lean Regime-C (+ thin Regime-U)** with binding kill | Honest last test; deliberately cheap enough to finish; accepts ~10–15% survival odds |
| **P3** | Quora / graph / full local zoo / LaTeX | Only after P0–P2; likely never if G3 NO-GO |

### What I recommend you actually schedule (1–2 engineer-weeks)

**Week structure (single 3090, Windows):**
1. Days 1–2: Artifact packs + bootstrap + learning curve tables/figs; start §3.
2. Days 2–4: Paper language pass + C3a query-level scores + repro command; D1 "shippable."
3. Days 3–5 (overlap): D3 recoverability.py + leave-one-regime-out + one-pager.
4. Days 5–8: Preregister D2; implement regime_c + CBIE/hubness/ZCA + detector scores;
   5-seed lean crossover; apply G3 kill without negotiation.
5. Day 9: Write D2 outcome into paper Limitations / §7 (negative expected and valuable).
6. Do **not** rebuild living-bank. Do **not** start Quora exact-522k.

### Recommendation in one line

**Execute D1 fully, D3 as a calibrated estimator (not a theorem), and a ruthlessly reduced D2
whose only job is a binding chelation kill — treat the full Codex matrix as a research
wishlist, not a build plan.**

### If forced to cut one deliverable entirely

Cut **full D2** before cutting D1 or D3. A negative methodology paper with honest CIs and a
recoverability estimator is a coherent arXiv unit. An unfinished crossover with 12 methods
and no power is a second failed campaign.

### If forced to cut to a single deliverable

**D1 only.** It is the panel's ~65–75% publishable product. D3 can be "future work" with
the margin inequality already sketched in the panel math. D2 without D1 language fixes
risks publishing the unsound ceiling then contradicting it mid-review.

---

## 7. Concrete module/file architecture summary (for implementers)

**Reuse:** harness loaders, QueryEncoderDrift, road-course slice, ndcg, create_adapter /
bound, topology cohesion stats.

**Do not reuse as-is:** isomer result-list API for cluster AUPRC; campaign scripts' hardcoded
worktrees; scalar C3a constants; paper's "ceiling" wording.

**Build first:** `artifacts.py` + `stats.py` (unblocks D1 and D3).
**Build second:** `maps.py` + `d1_bootstrap_curve.py` + paper edits.
**Build third:** `recoverability.py` + `d3_calibrate.py`.
**Build fourth:** `regime_c.py` + `detect_score.py` + `d2_subset_crossover.py` + `preregister_d2.md`.

**Dependency graph:**

```
artifacts ──┬── stats ── D1 paper tables
            ├── maps ───┬── D1 learning curve
            │           ├── D3 residuals
            │           └── D2 Regime-U thin
            ├── recoverability ── D3
            └── regime_c + detect_score ── D2 (after preregister)
```

---

## 8. Brutal honesty on this review itself

- I did not re-run experiments; costs are order-of-magnitude, not measured wall-clock.
- Codex planA file in-tree did not complete (sandbox); critique targets the **build brief +
  Codex panel protocol**, not a finished peer build plan.
- Detector "harm labels" need a precise operational definition at implement time; that is a
  residual ambiguity I flag rather than pretend resolved.
- Lean D2 still might be underpowered on SciFact; G2 (power) is load-bearing — if skipped,
  the kill criteria become cosplay.

END OF ADVERSARIAL PLAN REVIEW
