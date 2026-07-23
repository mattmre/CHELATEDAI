ROLE — ADVERSARIAL PLAN REVIEWER (Grok): You adversarially critique the proposed build plan for D1-D3. Attack the weak points: where will the crossover experiment (D2) give a false positive/negative? Where does D1's statistical fix fall short? Is D3's recoverability certificate actually well-posed or hand-wavy? What is over-scoped and will not finish? Propose the leanest plan that is still rigorous. Then give your build-plan recommendation.

# Build brief — 3 deliverables from the chelation roundtable (2026-06-30)

We ran an adversarial panel (Grok 4.5 + Codex gpt-5.6-sol + Claude) on a drift-recovery research program.
Consensus: the elaborate "chelation" corrector is DEAD as a drift/encoder-upgrade method; the within-model
claim is untested (~10-15% odds); the honest products are a negative/methodology paper + one decisive test +
one genuinely-new adjacent idea. We are now BUILDING three deliverables. Your job (this phase) is to analyze
the ARCHITECTURE and produce/critique an IMPLEMENTATION PLAN — not to build yet.

## Established facts (canonical, real runs — do not re-derive)
- Arena: query-encoder-upgrade drift. Cached docs in MiniLM-L6 (384-d); queries re-embedded with mpnet-base
  (768-d) → seeded projection to 384-d. Oracle = re-embed docs with new encoder. Metric: NDCG@10, recovery
  R = (NDCG(corr)−NDCG(floor))/(NDCG(oracle)−NDCG(floor)).
- SciFact eval-split canonical: ridge 84.4%, C3a (elaborate corrector) 20% → 4.2×; MLP 81% (≈ridge);
  unregularized lstsq 67%; bound α=0.1 → 2.1%. Generalizes to bge-large + NFCorpus.
- Panel's decisive corrections: (i) "encoder-upgrade is affine BY CONSTRUCTION" is FALSE — it's an empirical
  fit, not a theorem (encoders are nonlinear; Procrustes dot-product preservation is an ASSUMPTION). (ii) The
  "84% ceiling" is statistically UNSOUND as stated — ridge-vs-MLP is 0.027 NDCG with NO confidence intervals;
  it is an *observed plateau*, not a ceiling. (iii) isotropy ≠ retrieval quality ("Is Anisotropy Truly
  Harmful?" ACL 2023) — landmine under the within-model pivot. (iv) The near-identity bound is genuinely
  disqualifying (α=0.05 → max 2.86° rotation; doc-side-only detector cannot observe a query-space rotation).

## Existing assets
- Merged harness (import-only) in a worktree at `D:\GITHUB\CHELATEDAI\.claude\worktrees\agent-build`
  (origin/main, #291): `run_drift_recovery_experiment.py` (load_mteb_data, _split_anchor_eval, evaluate_*,
  ndcg), `query_encoder_drift.py` (QueryEncoderDrift), `run_road_course_campaign.py` (select_road_course_slice).
  Also topology_analyzer.py, isomer_detector.py, stability_tracker.py (chelation's detection machinery).
- The paper + 19 experiment scripts + figures live (git-EXCLUDED, LOCAL-ONLY) in
  `D:\GITHUB\CHELATEDAI\docs\waypoint-research-2026-06-09\` (`paper-draft/main.md`, `scripts/`, `figures/`).
  Cached models: MiniLM-L6, mpnet-base, bge-large-en-v1.5, Qwen2.5-0.5B-Instruct. HF offline env available.

## THE THREE DELIVERABLES TO PLAN

### D1 — Fix + ship the negative/methodology paper
- Add **10k paired-query bootstrap CIs** + an **anchor-count learning curve** for ridge vs MLP vs C3a; report
  CIs everywhere; **downgrade "84% linear ceiling" → "observed plateau under this protocol."**
- Write **§3 (System Under Test)** fully (adapter forward maps, the exact bound formula + floor, controller schedule).
- Apply remaining panel corrections; relocate S1 to its own subsection (single-seed, dominated); strengthen
  Reproducibility (commit SHAs, seeds, one repro command). Optional: LaTeX conversion for arXiv cs.IR.

### D2 — The one honest crossover experiment (Codex's design)
- Preregistered **Regime-U (real upgrade) × Regime-C (favorable synthetic heterogeneous collapse) crossover.**
- Correctors: ridge, orthogonal Procrustes, residual MLP, per-cluster ridge K∈{2,4,8,16}, global+low-rank-local
  residual w/ soft routing, graph-residual (kNN kernel-smoothed), bounded chelation α=0.05.
- Regime-C baselines that MUST be beaten: global centering/ZCA, all-but-the-top, **CBIE (cluster-based isotropy)**,
  local+global hubness scaling, plus the chelation topology/isomer detector (must PREDICT corrupted clusters,
  AUPRC ≥0.80). Corruption: collapse 8/16 clusters toward centroid, β∈{0.05,0.10,0.20}.
- Stats: 5 seeds, 10k paired bootstrap, Holm correction. **Win = ≥5 recovery pts AND ≥0.02 NDCG AND 95% CI
  excludes 0 AND same sign in all 5 seeds.** Data scale: FEASIBLE subset first (SciFact/NFCorpus/Quora-subset),
  full BEIR-Quora (522k) only if it runs. Kill criterion binding: if chelation can't beat CBIE/hubness in
  Regime-C, terminate the mechanism.

### D3 — Scope the recoverability certificate (the exceptional-potential idea; NOT chelation)
- A calibrated estimator/theorem: from paired **Gram-distortion** + **query-margin distributions**, predict
  BEFORE fitting whether an affine adapter will recover ~70/90/99% of retrieval. Prototype the estimator +
  validate its predicted-vs-actual recovery on our existing runs. One-pager framing + a working predictor.

## Your Phase-A deliverable (PLAN/ARCHITECTURE ONLY — do not build)
1. A concrete module/file architecture for D1-D3 (what to reuse from the harness; new files; interfaces).
2. Dependency graph + sequencing (what must precede what; what's parallelizable).
3. Top risks + how the plan mitigates each (esp. the statistical-rigor + overfitting + scale risks the panel raised).
4. A realistic scope cut: what's the MINIMUM that makes each deliverable honest & shippable vs the full version.
5. Explicit go/no-go gates tied to the panel's kill criteria.
Be concrete and buildable. Assume a single 3090 GPU, Windows, Python, the merged harness above.

