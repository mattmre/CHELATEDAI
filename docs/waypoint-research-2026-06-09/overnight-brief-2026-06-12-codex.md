# Overnight Brief — 2026-06-12 → 06-13 (Codex, full machine + RTX 3090)

You are the implementing agent for the June drift-recovery campaign. Tonight's mission:
**PR-4 (experiment harness) and PR-5 (campaign runs)** per the saved plan. You have the
whole machine and an RTX 3090 — use the GPU for embedding encoding, and use the long
unattended window for the full campaign matrix.

## Read first (in this order)

1. `docs/waypoint-research-2026-06-09/implementation-plan-june-2026-drift-paper.md`
   — §3 PR-4 and PR-5 specs are your contract. §2 repo conventions. §7 BHS. §8 escalation.
2. `docs/waypoint-research-2026-06-09/review-notes-cleanup-pass.md` — current state +
   standing checks applied to every slice.
3. `git pull origin main` BEFORE branching. The injection-index fix
   (`feat/drift-injector-injection-index`) should be merged; if its PR is still open,
   rebase your PR-4 branch on that branch and say so in your PR body.

## Status going in

- PR-1/2/3 are MERGED at BHS 100 (#258 drift injector, #259 recovery metrics,
  #260 annealing controller). `enable_annealing_controller` is at antigravity_engine.py:824.
- DriftInjector manifests now carry `injection_index`; validation precedes all RNG
  consumption. **Contract for your harness: one fresh `DriftInjector` per injection
  event; record seed AND injection_index in the run config JSON.**
- PRs #256 and #257 are parked drafts — DO NOT touch or merge them tonight.

## Tonight's work

### Slice 1 — PR-4: `run_drift_recovery_experiment.py` (plan §3 PR-4, follow exactly)

- CLI and conditions C0–C4 exactly as specified. One shared skeleton:
  ingest → baseline NDCG@10 → inject drift (save manifest) → N cycles of
  condition-specific action → measure each cycle via `RecoveryTracker`.
- C3 uses `enable_annealing_controller` + `create_adapter(..., bounded=True)` +
  detection-triggered `run_sedimentation_cycle`; C4 = C3 with `bounded=False`.
- Output JSON must contain: full config (including seed + injection_index), drift
  manifest, per-cycle trajectory, recovery_cycle, post-recovery stability,
  correction-norm stats for C3/C4, wall-clock.
- Tests: tiny-scale end-to-end through the REAL engine path (all five conditions,
  2 cycles, well-formed JSON) + determinism (same seed twice → identical trajectory).
- Full BHS loop: self-score, fresh adversarial Tier B agent told to DISPROVE, iterate
  until genuine 100, open PR with the §7 template, wait for CI green, merge.

### Slice 2 — PR-5: campaign execution (plan §3 PR-5)

Run AFTER PR-4 is merged. The matrix:
conditions {C0,C1,C2,C3,C4} × drift {rotation(0.5, 25°), noise(0.5, σ=0.05)} × seeds
{42, 1337, 7} on SciFact, `--max-queries 100 --sample-docs 1200 --cycles 12`.

- With the 3090 you should NOT need to shrink the matrix. If a single run still exceeds
  ~45 min, document why before reducing anything.
- All JSONs + matplotlib plots (NDCG-vs-cycle per condition, mean±std across seeds)
  under `experiment_runs/drift-recovery/`. Results table in
  `docs/drift-recovery-results-2026-06.md` per the plan's column spec.
- **Report whatever the numbers say.** If C3 loses to C1/C2, that is the result —
  no tuning-until-win, no selective reporting (L4 score-gaming). The pre-registered
  primary comparison is C3 vs C1 and C3 vs C2 on recovery_cycle and final NDCG.

### If both slices finish — stretch (plan §6, in order)

1. NFCorpus transfer (same matrix, seed 42 only).
2. Masking ablation C5 (C3 + `enable_learned_masking`).
3. Do NOT start model-swap drift (D3) tonight.

## Host-specific warnings

- **HF SSL failure**: `scripts/smoke_pipeline.py` previously failed on this Windows host
  verifying Hugging Face SSL certs while downloading `all-MiniLM-L6-v2`. Prior campaigns
  ran locally, so the model is likely in the HF cache. FIRST ACTION of slice 2: a 1-doc
  model-load sanity check. If download fails and cache is absent, try
  `HF_HUB_OFFLINE=1`; NEVER globally disable SSL verification. If still blocked, log to
  `docs/waypoint-research-2026-06-09/blockers.md` and continue with whatever runs work.
- GPU: sentence-transformers will pick up CUDA automatically; confirm with a quick
  `torch.cuda.is_available()` check and note device in run configs (device affects
  nothing about determinism claims for retrieval, but record it).
- Qdrant `:memory:` per run; do not share collections across runs.
- If a run crashes mid-matrix: save partial JSON, log it, continue remaining runs.
  Never hand-edit artifacts.

## Hard rules tonight

1. No stubs, no placeholder data, no fabricated or extrapolated numbers — every value in
   every artifact comes from an actual completed run.
2. Each slice fully done (tests + Tier B 100 + PR + CI green + merged) before the next.
3. Do not push or commit `docs/waypoint-research-2026-06-09/` (git-excluded; leave it so).
4. Do not touch parked PRs #256/#257, the SHIM/steering code, or anything outside plan scope.
5. Same gap surviving 2 Tier B iterations → stop looping, write it to blockers.md, move on.
6. End of night: append a status section ("Overnight 2026-06-13") to
   `docs/waypoint-research-2026-06-09/review-notes-cleanup-pass.md` with: PRs opened/merged,
   runs completed/failed, headline numbers (C3 vs baselines), and open blockers.

## Definition of a successful night

Minimum: PR-4 merged at BHS 100 + at least the rotation-drift third of the matrix
(15 runs) committed with plots. Target: full 30-run matrix + results doc. Stretch: §6
items 1–2. The paper draft consumes these artifacts tomorrow — completeness and honesty
beat speed.
