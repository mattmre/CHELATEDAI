# Grok Tier B — final combined review of the 3-commit lattice slice (before push/PR)

You are the official Tier B scorer for the branch `lattice/phase2-continue-20260713` (cwd = repo
root), 3 commits ahead of origin/main:
- `2b0fa044` results(h2) — H2 re-run, closes CD-H1-01 + CD-A2-01 (already Tier-B'd at 96→fixed; re-check the fixes stuck)
- `7e6c9118` results(h5) — living-bank verdict + H4 ablation
- `c6a698e4` docs(lattice) — Phase II docs-truth sync

Review the COMMITTED diff (`git diff origin/main..HEAD`), not the panel drafts. Try to DISPROVE that
this is a clean, honest, BHS-100 PR. This is a docs+evidence slice (no production code changed — verify
that claim with `git diff --stat`; any `.py` change under repo root that is not a test/doc is a finding).

## Attack
1. **L2 verdict numbers.** Every NDCG in CHANGELOG.md + docs/next-session.md Disposition + the three
   result docs must match the committed manifests
   (`experiment_runs/drift-recovery/post-bank-headtohead*/…manifest…json` main_rows/summary). Recompute
   the C5/C5s/C5r means and the LIVING BANK WINS booleans yourself. C5==C5s must be bit-identical on
   both datasets; gate False on both. H4 0.236297 vs 0.005258 must match the two committed h4 JSONs.
2. **L3 docs-truth.** For every "DONE (#NNN)" in the ROADMAP Status column, confirm the PR is on
   origin/main (git log). For 13 PARTIAL and 15/16/17 OPEN, confirm they are genuinely not delivered
   (grep). Any DONE without a merged PR, or any OPEN that is actually shipped, = FAIL.
3. **L1 fixes stuck.** Confirm the amended L1 commit no longer says "C4a unchanged" or "from history",
   and CD-H1-01/CD-A2-01 are CLOSED with the NFCorpus-was-the-contamination nuance.
4. **Block flag + hygiene.** `scripts/check_block_flag.py` -> CLEAR, 0 rows. Confirm no evidence was
   lost by the L1 NFCorpus-blob untrack (manifests still hold the rows). Confirm the H5/H4 tracked
   evidence is small (manifests + <1MB H4 JSONs), per-run H5 JSONs gitignored.
5. **BHS bodies.** Any overclaim across the three commit messages, or an omitted L-class issue?

Deliver `BHS_TIER_B: <0-100>` + PASS / PASS-WITH-FIXES / FAIL, a defect table (severity, commit,
where, fix), and a one-line bottom line. Recompute; do not trust the commit prose. Only
BHS_OFFICIAL=100 merges — if any real gap remains, score under 100 and state exactly what must change
(prefer amend-only fixes, no re-run).
