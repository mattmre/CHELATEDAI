# Agreed build plan v1 (chair synthesis of Grok + Codex Phase-A) — 2026-06-30

## 🔒 LOCKED — sign-off addendum v2 (Grok + Codex both RATIFY-WITH-CHANGES, 2026-06-30)
Binding changes folded in before Phase B:
1. **Option (a)** locked: track `research/drift_recovery/` + submission artifacts; strategy/waypoint notes stay
   git-excluded. **Build LOCAL-ONLY (agent-build worktree, no commits) until shipment** (defers the git decision).
2. **G2 is a HARD gate before G3:** if median bootstrap half-width for primary ΔNDCG > ~0.015, do NOT interpret
   G3 — grow the query set or **preregister the cell as underpowered/negative** (Codex+Grok).
3. **⚠️ α=0.05 DECOUPLED from the G3 kill arm (Grok, load-bearing):** the primary chelation kill uses a corrector
   that can *actually move mass* (unbounded, or α calibrated so mean ‖x′−x‖ under β=0.10 is feasible). α=0.05 is
   reported as a SEPARATE design-premise stress test, NOT the G3 correction. (Fixes the rigged-to-fail FN-A.)
4. **Detector = an explicit cluster-level HARM scorer** (features → P(retrieval loss)), NOT an `IsomerDetector`
   Jaccard relabel. AUPRC measured against harm labels, not the injection knob.
5. **Holm family FROZEN to: chelation-vs-CBIE, chelation-vs-hubness** (± one preregistered contrast); all else exploratory.
6. Rename `certificate/` → `estimator/`.
Ordering **P0=D1 → P1=D3 → P2=lean-D2** ratified and binding. Quora / full matrix / LaTeX are post-gate only. Do not expand.

---


Converged from Grok (lean/adversarial) + Codex (rigorous engineering). Grok's scope discipline + Codex's
contracts/leakage guards. AGY seat was dead. **This is the spec Codex will build against after the sign-off round.**

## Priority & scope discipline (binding)
- **P0 = D1** (paper fix; highest EV, nearly done). **P1 = D3 estimator** (exceptional-upside; small surface).
  **P2 = lean D2 kill-screen** (honest last chelation test). **P3 = Quora-522k / graph-k sweep / LaTeX** — ONLY
  after gates pass; likely never. If forced to one deliverable: **D1**.
- **Full Codex matrix (Quora × 7 correctors × 6 baselines) is a wishlist, NOT the build.** Start feasible
  subset (SciFact + NFCorpus), β=0.10 only, ≤4 primary methods.

## Architecture (one package: `research/drift_recovery/`)
- `harness_bridge.py` — the ONLY module allowed to import the merged harness (`load_mteb_data`,
  `_split_anchor_eval`, `QueryEncoderDrift`, `select_road_course_slice`, `evaluate_*`, `ndcg`). Behind it:
  `assert_harness_parity()` proves packs reproduce the merged aggregate NDCG to ≤1e-12. Do NOT reimplement NDCG.
- `contracts.py` — `RetrievalCase / ExperimentSplit / AnchorPairs / PerQueryRecord / RunManifest` (repo+harness
  SHAs, protocol SHA, model revisions, every seed, exact command).
- **Two method interfaces** (Codex): `QueryAdapter.fit/transform` (operates on query vectors) and
  `ScoreTransform.fit/rank` (hubness operates on SCORES, not vectors). `MethodRunner` unifies output.
- `artifacts.py` / EmbeddingPack — **artifact-first**: freeze `Do, Dor, Qd, qrels, fit_idx, per-query scores`
  once; ALL stats run offline on packs.
- `stats/` — `paired_bootstrap.py` (10k deterministic paired-query draws; recovery = ratio of resampled query
  MEANS, not mean of per-query ratios; same draw for floor/oracle/all methods; invalid if oracle−floor≤ε, block
  if >1% invalid), `multiple_testing.py` (Holm over preregistered families), `learning_curve.py`.
- `regimes/` — `upgrade.py` (Regime-U), `synthetic_collapse.py` (Regime-C centroid collapse).
- `methods/` — `affine.py` (ridge, Procrustes), `local_adapters.py` (per-cluster ridge K, global+low-rank-local
  soft-routed — parameter-matched), `isotropy.py` (ZCA, all-but-top, **CBIE**), `hubness.py` (local+global scaling),
  `bounded_chelation.py` (α + topology/isomer detector).
- `d1/paper_stats.py` + `render_paper_assets.py` (machine-readable tables → paper; NO hand-entered numbers).
- `d2/` — `crossover.py`, `detector_evaluation.py`, `decisions.py`. `certificate/` — `features.py`,
  `margin_bound.py`, `calibration.py`, `predictor.py`, `validation.py`.
- `protocols/*.yaml` preregistered; `tests/` incl. `test_harness_parity`, `test_method_leakage`,
  `test_paired_bootstrap`, `test_synthetic_collapse`.

## LEAKAGE CONTRACT (binding, both experts)
No method receives evaluation qrels during `fit`. HP selection uses a separate **anchor-validation** partition,
frozen in the preregistration. qrel-positive docs excluded from the anchor pool. All docs pass through the learned
map (never substitute known targets into the index).

## D1 (paper) — specifics
- 10k paired-query bootstrap CIs on NDCG / recovery / ΔNDCG (ridge−MLP, ridge−C3a). Fix **C3a-as-scalar**: score
  C3a per-query through the SAME NDCG implementation (else the bootstrap is theater — Grok FP).
- Learning curve counts {8,16,32,64,128} × 5 anchor-order seeds, nested prefixes, same IDs for ridge/MLP/C3a.
  **Guard the confound (Grok):** the +59-pt story is supervision *type* (doc-pairs vs query-InfoNCE), NOT sample
  size — do not put them on a shared fake n-anchors axis.
- **Downgrade "84% linear ceiling" → "observed plateau under this protocol" EVERYWHERE** (abstract, §1, §5.5, §6,
  §7, §8 — not one edit). Write **§3** fully (adapter maps + exact bound formula + floor + controller schedule).
- Fix repro (scripts git-excluded w/ hardcoded gpu-campaigns paths). Relocate S1 to §5.6 (single-seed, dominated).

## D2 (lean kill-screen) — specifics + Grok's defect fixes
- Datasets: SciFact + NFCorpus (Quora post-gate only). Corruption: collapse 8/16 clusters toward centroid, **β=0.10**.
- Methods (≤4 primary): global ridge; global+low-rank-local (param-matched) as the "local" diagnostic; **CBIE**;
  one **hubness** scaling. Baselines: ZCA, all-but-top. Chelation: bounded α=0.05 **+ detector**.
- **Detector must predict HARM labels (actual retrieval loss per cluster), NOT the k-means injection knob (Grok FP-A
  "detector cosplay").** Fix `IsomerDetector` API mismatch (it scores result-list Jaccard, not doc clusters — FN-B).
- **Dual kill rule:** a "win" = ≥5 recovery pts AND ≥0.02 absolute NDCG AND 95% CI excludes 0 AND same sign in all 5
  seeds AND **min oracle gap ≥0.05 in the cell** (Grok FP-D: 5 recovery pts on a 0.03 gap = 0.0015 NDCG = noise).
  Cap primary tests ≤4, Holm the family (FP-E multiplicity).
- Kill criteria apply ONLY to unpaired / detector-gated arms (privileged paired local ridge is a diagnostic, not a
  chelation win — FN-D). Require BOTH detection (AUPRC≥0.80 on harm labels) AND correction axes (FN-E: isotropy≠NDCG).

## D3 (recoverability ESTIMATOR — not "theorem/certificate")
- Cross-fitted **residual + query-margin** features → R̂ + lower band. Core: order preserved when
  m = qᵀy_r − qᵀy_j > ‖q‖(‖e_r‖+‖e_j‖); predict inversion rate → recovery bands.
- **Leave-one-regime-out** validation (held-out: bge / NFCorpus / different n_anchor — NOT same-run, which is circular).
- Success bar: Spearman(R̂, R) ≥ 0.7 OR 3-way bin accuracy ≥ 80%; else ship an explicit negative note. Depends on D1
  packs, NOT D2. Rename "certificate/theorem" → **calibrated estimator** (Gram distortion alone ≠ NDCG; top-k breaks free theorems).

## Gates (binding)
G0 packs reload & reproduce ladder ≤1e-3 · G1 D1 shippable (CIs + plateau language + §3 + repro) — NO D2 before G1 ·
G2 power: median bootstrap half-width for primary ΔNDCG ≤ ~0.015 (else grow query set) · G3 chelation kill: fail if
unpaired repair doesn't beat BOTH CBIE and one hubness under dual-CI rule on 2 datasets, OR detector AUPRC<0.80 on
harm labels → terminate mechanism (detection-only pass → park as re-embed router, not corrector) · G5 D3 calibration bar or negative note.

## OPEN DECISION for the operator (flagged by Codex)
Codex insists final paper source/tables/figures/protocols/provenance be **git-tracked** before shipment — this
**conflicts with the standing "never commit `docs/waypoint-research-2026-06-09/`" guardrail.** Options: (a) keep
strategy folder git-excluded, create a NEW tracked `research/drift_recovery/` package + a tracked paper repo at
submission time; (b) keep everything local until arXiv. **Recommend (a): code package tracked, strategy notes stay excluded.**
