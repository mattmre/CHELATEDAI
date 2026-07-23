# Overnight Goal — 2026-06-12 → 06-13 (Codex run instructions)

You are the implementing agent for the ChelatedAI June drift-recovery campaign, running
overnight with the full machine and the RTX 3090. Objective: complete PR-4 and PR-5 of
the saved plan, each fully implemented, genuinely Tier-B-reviewed to BHS 100, and merged,
with all campaign artifacts committed.

## Start here, in this exact order

1. `git -C D:\GITHUB\CHELATEDAI pull origin main` — main must be at or past commit
   `a445b2b` (PR #261, injection_index contract). Confirm before any branching.
2. Read `docs/waypoint-research-2026-06-09/overnight-brief-2026-06-12-codex.md` in full.
   It is your mission contract for tonight.
3. Read `docs/waypoint-research-2026-06-09/implementation-plan-june-2026-drift-paper.md`
   §2 (repo conventions), §3 PR-4 and PR-5 specs, §7 (BHS PR template), §8 (escalation).
4. Read `docs/waypoint-research-2026-06-09/review-notes-cleanup-pass.md` — apply the
   "Standing checks for every slice review" section to your own work before each PR.

## Slice 1 — PR-4: `run_drift_recovery_experiment.py`

Build exactly to plan §3 PR-4 spec. Conditions C0–C4 over one shared skeleton
(ingest → baseline NDCG@10 → inject drift, saving the manifest → N cycles of
condition-specific action → measure each cycle with RecoveryTracker).

Hard requirements:
- One fresh `DriftInjector` per injection event; every run-config JSON records seed
  AND injection_index.
- C3 uses `enable_annealing_controller` + `create_adapter(..., bounded=True)` +
  detection-triggered `run_sedimentation_cycle`. C4 is C3 with `bounded=False`.
- Tests: tiny-scale end-to-end through the REAL engine path covering all five
  conditions producing well-formed JSON, plus same-seed-twice → identical trajectory.
- Full BHS loop: self-score, spawn a fresh adversarial Tier B agent instructed to
  DISPROVE (not verify), iterate until a genuine 100, open the PR with the complete §7
  template body, wait for CI green, merge. Use admin merge only if the branch policy
  blocks with all checks green — documented repo precedent.

## Slice 2 — PR-5: campaign execution (only after PR-4 is merged)

Run the 30-run matrix: conditions {C0,C1,C2,C3,C4} × drift {rotation(0.5, 25°),
noise(0.5, σ=0.05)} × seeds {42, 1337, 7} on SciFact with
`--max-queries 100 --sample-docs 1200 --cycles 12`.

- Before the matrix: 1-doc model-load sanity check (known HF SSL issue on this host —
  if download fails use the local HF cache or `HF_HUB_OFFLINE=1`; NEVER disable SSL
  verification; if truly blocked, log to
  `docs/waypoint-research-2026-06-09/blockers.md` and run what works).
- Confirm `torch.cuda.is_available()` and record the device in run configs.
- All result JSONs and matplotlib plots (NDCG-vs-cycle per condition, mean±std across
  seeds) under `experiment_runs/drift-recovery/`; results table per the plan's column
  spec in `docs/drift-recovery-results-2026-06.md`.
- If a run crashes: save partial JSON, log it, continue the remaining runs.

## Hard rules

- Every number in every artifact comes from an actual completed run. Report whatever
  the numbers say — if C3 loses to C1/C2, that IS the result. No tuning-until-win, no
  selective reporting, no hand-edited artifacts.
- No stubs, no placeholders, no fabricated Tier B scores. A gap surviving 2 Tier B
  iterations goes to `blockers.md`, not a third loop.
- Do not touch parked draft PRs #256/#257, the SHIM/steering code, or anything outside
  plan scope. Do not start model-swap drift (D3).
- Never commit or push anything under `docs/waypoint-research-2026-06-09/`
  (git-excluded; leave it that way).
- If both slices finish with night remaining, stretch goals in order:
  (1) NFCorpus transfer, same matrix, seed 42 only;
  (2) masking ablation C5 = C3 + `enable_learned_masking`. Nothing else.

## End of night (regardless of progress)

Append an "Overnight 2026-06-13" section to
`docs/waypoint-research-2026-06-09/review-notes-cleanup-pass.md` with: PRs
opened/merged with commit hashes, runs completed vs failed, headline numbers (C3
recovery_cycle and final NDCG vs C0/C1/C2 per drift mode), and open blockers.

## Success criteria

Floor: PR-4 merged at BHS 100 + the 15 rotation-drift runs committed with plots.
Target: full 30-run matrix + results doc. The paper draft consumes your artifacts
tomorrow — completeness and honesty beat speed.
