# Grok Tier B adversarial review — L1 commit (H2 re-run, closes CD-H1-01 + CD-A2-01)

You are the official Tier B scorer for commit `5e95d97b` on branch
`lattice/phase2-continue-20260713` (cwd = repo root). Try to DISPROVE that it honestly closes the two
debt rows. This is a docs+evidence commit (no production code changed).

Inspect: `git show 5e95d97b --stat` and the diff; `docs/next-session.md` CD-H1-01 + CD-A2-01 rows;
`docs/drift-recovery-swap-results-2026-06.md` + `docs/drift-recovery-swap-nfcorpus-results-2026-06.md`;
`experiment_runs/drift-recovery/swap/swap-campaign-manifest-2026-06.json` +
`experiment_runs/drift-recovery/swap-nfcorpus/swap-campaign-manifest-2026-06.json`.

## Attack these
1. **CD-H1-01 regression gate.** The commit claims C3a baseline now == C0/C2/C2O/C4a per seed. Recompute
   from the manifests' `main_rows` (baseline_ndcg per condition per seed) for BOTH datasets. If any seed
   has C3a baseline != the others, the H1 fix regressed and the closure is FALSE.
2. **Numbers vs docs.** Do the regenerated docs' Main Matrix numbers match the manifest `main_summary`
   rows exactly? Any hand-edited/stale number = FAIL.
3. **CD-A2-01 evidence.** Does the SciFact manifest actually show `config.swap_model ==
   "all-mpnet-base-v2"` and a query_encoder_swap arena? Is the closure's cited evidence real, or is it
   leaning only on the opt-in smoke (which the debt row explicitly says is not enough on its own)?
4. **Hygiene honesty.** The commit untracks 22 NFCorpus per-run JSONs (25 MB) and keeps only the
   manifests. Verify: (a) the manifests still contain the per-condition/per-seed rows the docs are
   generated from (so no evidence was actually lost by untracking), (b) the untracked files are covered
   by .gitignore (so they won't silently re-appear), (c) SciFact per-run files were already gitignored
   (the stated consistency rationale is true).
5. **Block flag.** Confirm `scripts/check_block_flag.py` returns CLEAR with 0 carried debt rows AFTER
   this commit, and that no OPEN blocking row was closed without evidence.
6. **BHS body honesty.** Is anything in the commit's Brutal Honesty section an overclaim, or is a real
   L-class issue omitted?

Deliver `BHS_TIER_B: <0-100>` + PASS/PASS-WITH-FIXES/FAIL, a defect table (severity, where, fix), and a
one-line bottom line. Recompute the parity and the doc/manifest number match yourself — do not trust the
commit message. Per the rulebook only BHS_OFFICIAL=100 merges; if you find a real gap, score it under
100 and say exactly what must change.
