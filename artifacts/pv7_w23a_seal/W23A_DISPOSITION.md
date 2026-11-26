# Wave 23A disposition (W22-01..W22-05, W22-17)

Date: 2026-11-26. Spine branch `pv7/w16-core`, head `7ffbd0df` when this was written.
Source of the queue: `docs/research/waves-handoff-wave22-breakthrough-audit-and-queue-additions-2026-11-25.md`
in the main checkout.

**Status of every verdict here: PROPOSED.** Nothing in this file has changed a scorecard tier yet.

**Tier B round 1 has been done.** A fresh adversarial reviewer was run on W22-02, W22-03 and W22-16. Each returned
concrete corrections, and all of them were checked against the artifacts and folded into the reports on
2026-11-26:
- **W22-02:**
  - The tamper tests do load the D6/D8 manifests.
  - `verify_part7` and `verify_pristine_mirror` are hash-only too.
  - Raw rows are untracked in git.
- **W22-03:** #41 moves to PARTIAL, the #41 counts are fixed, the Qwen ≤1/17 row is fixed, #62 is a tie, and each
  claim now cites where it is made.
- **W22-16:** SELECT and REPORT are separate splits, and the fixture scope is k=4.

No tier should move until a round-2 reviewer has seen the corrected reports.

CI note: GitHub Actions has not run any job since 2026-11-22 because of an account billing block. No CI result
below counts as evidence; every result is a local command run.

## W22-01 — seal the verifier at the current head: PASS

- Command: `python scripts/verify_all_offline.py --full-formal --verifier-jobs 4` at `7ffbd0df`.
- Result: `PASS`, `formal_tier=PASS`, `skipped_checks=none`, 548 s wall time. The external nanoda kernel checked
  25,473 declarations. Log: `verify_all_offline_7ffbd0df.log`.
- Fixes it needed (commit `7ffbd0df`):
  - The SDK cold-start import-barrier test failed only when run with other tests, because a module already in
    `sys.modules` skips meta-path finders.
  - `ruff` had 8 errors: 2 were fixed, and 6 were in a retained evidence worker that is now excluded by exact path.
- Remaining nits, none of which hide a failure:
  - **#54 threshold wording.** `registry.d/54-t6b-12domain-4hop.json:15` says `>=10/12` domains must be positive.
    The code (`chelated_12domain_4hop_gauntlet.py:146`) and the scorecard both use the stricter `>=11/12`, and the
    live result of 11/12 clears it. The registry fragment is under the spine digest, so it is left unedited and the
    mismatch is disclosed here. The claim is not inflated by it.
  - **#62 `ABSENT`.** `62_verifier.py:38` returns `live_worker_outcome=ABSENT` only when the live receipts are
    missing. Both receipts are tracked in git. `tests/test_pv7_w21_live_receipts.py:80` asserts that a missing
    receipt is reported as absent, not passed. The live outcome is reported as `FAILED`, and no consumer turns it
    into a pass.
  - **Scorecard digests.** The docs cite the scorecard's embedded content digest, `scorecard_sha256`, not the
    sha256 of the file. The file's sha256 is `799aebd83896`.
    - The current embedded digest is `7954f9fcf0c7`, and that is what `WAVES_1_TO_21_MASTER_EVIDENCE_AND_PR_INDEX.md:4`
      cites, so the index is correct.
    - The PR #366 body cited the stale `192855cdfd2d`. It was updated to `7954f9fcf0c7…` on 2026-11-26.
    - `6a2a254b49e6` appears only in historical council logs, which are left as written.

## W22-02 — claim-coverage audit: proposed downgrades

Full report: `w22_02_claim_coverage_audit.md`.

- Of the 31 bundles marked verified-sealed:
  - 0 are fully recomputed from raw rows;
  - 1 is partially recomputed (#43 D8);
  - 30 are only hash-replayed.
- Proposed new tier: `hash-sealed-unrecomputed`.
- Ordinals 1–12 have raw rows on disk, so they can be recovered with a recompute verifier and a re-seal. Those rows
  are gitignored and untracked, though, so this works only on this machine and not from a fresh checkout.
- Ordinals 16–42 have no raw rows.
- Legacy nDCG headlines:
  - The faulty `ndcg_at_k` is still on `origin/main`, in `benchmark_utils.py:161` and `drift_recovery_metrics.py`.
  - 105 of 113 artifacts can be recovered.
  - H4, the base matrix, calibration and the knob sweep need restating.
  - A4 C3a and H5 cannot be restated, because their raw runs are missing.
- PR #364 cannot be merged as it stands:
  - it is a draft;
  - the code-owner ruleset applies;
  - its commits are unsigned;
  - a merge commit breaks linear history;
  - it is about 209k lines;
  - CI has never run on it.
- Recommendation for #364: split the mlr-v1 fix (`09184c83`) into its own small PR.

## W22-03 — cost-matched re-score (corrected after Tier B round 1)

Full report: `w22_03_cost_matched_rescore.md`.

| Bundle | Verdict | Finding | New vs scorecard? |
|---|---|---|---|
| #41 D6 | **PARTIAL** | Merge quality survives: Chelated beats raw cosine and RRF in all 8 direction groups, and DEV z-score in 7 of 8. DEV z-score significantly beats Chelated in SciFact 4096 d384→d4096 (UCB95 −0.0036). Latency is not shown, because the timers are unmatched and the scorecard already says "no pure ANN latency claim". | New: the DEV z-score loss. |
| #45 D10-B | SUSPEND the deployment reading | Bare 70B at k17 ties Chelated exactly (0.98333 EM, p=1.0) and is about 5× faster. | Already disclosed. |
| #46 D11 + #53 | SUSPEND the "better suppressor" reading | At matched FPR, raw cosine has equal or higher TPR than Arm 9 at every level. #53's best arm is ZeroMu with raw cosine (0.50% FPR, 18.2% TPR), not ZeroMu with CSLS (0.67%, 12.7%). | #46 already disclosed. #53 new: which arm the headline FPR belongs to. |
| #62 D13 | **TIE** with conventional arms | Chelated scores 59.38 JPM vs 62.87 and 62.97 for the conventional arms, with 8 vs 7 on-time jobs and no CI. Only the full-restart win is real. The sham control is invalid. | New: the anti-L4 note omits the conventional arms. |

## W22-04 — 12-domain bridge comparison: tie with the conventional baseline

These numbers are reconciled from `artifacts/pv7_w22a_b54_12dom_complete/w22a_b54_12dom_manifest.json`. No new
compute was run.

- **ZeroMu vs CenteredProcrustes+CSLS:** −0.32 pp, 95% CI [−1.57, +0.68], 6 domains won and 6 lost. This is a
  statistical tie, so a claim that ZeroMu is a uniquely better bridge is not supported against the conventional
  baseline.
- **ZeroMu vs UncenteredProcrustes+CSLS:** +5.28 pp, LCB +3.35 pp, 11/12 domains positive.
- **CSLS merge (H6B_merge):** +2.45 pp, LCB +0.92 pp, 11/12 domains positive. This result holds.
- **4-hop chain:** mixed, 7/12 domains positive.
- **Missing cells:**
  - the DriftAdapter bridge;
  - Raw, ZScore and RRF merges for each bridge;
  - per-domain CIs, which the strict kill rule needs.
- No embeddings are cached locally, so the missing cells need GPU re-encoding on the 3090 or Colab.

## W22-05 — bitemporal SQL ledger comparator and negation-cycle census: NOT STARTED

This is about 10 hours of engineering. An implementer agent is working on it on branch
`pv7/w23a-w22-05-ledger-comparator`. It is not done until it has its own PR and a Tier B review.

## W22-17 — core-math and prime-ring fixes: PRs open, Tier B pending

- **PR #369** (`fix/w23a-d03-d08-core-math`, base `main`):
  - D-03, D-04 and D-07 fixed.
  - D-05 deliberately unchanged, because of the documented F-023 guard.
  - D-06 confirmed, but the default is unchanged. Changing it would alter the C3/C3b golden trajectories, so that
    is an owner decision.
  - D-08 exists only on the #364 branch.
  - It conflicts with #364's own versions of D-03, D-04, D-06 and D-07.
  - Tier B round 1 scored it 85 (MEDIUM). Its fixes are applied at `da21b6f9`:
    - beta must be in the open interval (0, 1);
    - B=0 raises;
    - the rerank-path change and the absolute 1e-12 tie threshold are disclosed, with a pinned test.
  - The self-score stays at 85 until the owner decides D-06. BHS_OFFICIAL = 85, LOOP_ITERATIONS = 3. Not mergeable.
- **PR #368** (`fix/w23a-prime-ring-tie-contract`, base `codex/prime-ring-onion-method-dev`):
  - Before the fix, 117/336 cases matched. After the fix, 336/336 match, and adjacent unequal cases are 672/672.
  - Contract `PRW-NUMERICAL-TIE-CONTRACT-001` covers it.
  - Iteration 3 (`fb7d6e94`):
    - adds an exact snapped-spectrum reference guard;
    - corrects the Bluestein wording;
    - counts the guard's memory.
  - 123 prime-ring tests pass locally.
  - Scores: self 94, Tier B round 2 = 90 (LOW-MEDIUM), so BHS_OFFICIAL = 90. LOOP_ITERATIONS = 3. Not mergeable.
  - Still open (all disclosed in the PR body):
    - the constant 64 in the bound is not proven;
    - three float tie sites remain: `_decode_sparse_oppw`, `_decode_native_oppw` and `_payload_only_scores`;
    - FAILED wording remains at `prime_ring_leading_shell.py:441/717/810`;
    - old artifacts have not been regenerated;
    - smoke ran at floor tier only.
- Not done in W23A:
  - the PR #293 (CD-R13-01) re-probe;
  - closing out PR #295 (CD-R16-01). Closing a PR is outward-facing, so it waits for the owner.
