# Wave 23A disposition (W22-01..W22-05, W22-17)

Date: 2026-11-26. Spine branch `pv7/w16-core`, head `7ffbd0df` when this was written.
Source of the queue: `docs/research/waves-handoff-wave22-breakthrough-audit-and-queue-additions-2026-11-25.md`
in the main checkout.

**Status of every verdict here: PROPOSED.** Nothing in this file has changed a scorecard tier yet. The rulebook
needs a fresh adversarial reviewer to try to disprove W22-02 and W22-03 before any tier moves.

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
  - **Stale scorecard hashes.**
    - The scorecard blob is `799aebd83896`, both at `34125969` and at the head.
    - `192855cdfd2d` is still in the PR #366 body.
    - `7954f9fcf0c7` is in `WAVES_1_TO_21_MASTER_EVIDENCE_AND_PR_INDEX.md:4`.
    - `6a2a254b49e6` appears only in historical council logs, which are left as written.

## W22-02 — claim-coverage audit: proposed downgrades

Full report: `w22_02_claim_coverage_audit.md`.

- Of the 31 bundles marked verified-sealed:
  - 0 are fully recomputed from raw rows;
  - 1 is partially recomputed (#43 D8);
  - 30 are only hash-replayed.
- Proposed new tier: `hash-sealed-unrecomputed`.
- Ordinals 1–12 have raw rows on disk, so they can be recovered with a recompute verifier and a re-seal.
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

## W22-03 — cost-matched re-score: four SUSPEND verdicts

Full report: `w22_03_cost_matched_rescore.md`.

| Bundle | Finding |
|---|---|
| #41 D6 | The 940 ms figure is the chelated p95 itself, and every served path is an exact scan. Only the merge-quality win over raw cosine survives. |
| #45 D10-B | Bare 70B at k17 ties Chelated exactly (0.98333 EM, p=1.0) and is about 5× faster. |
| #46 D11 + #53 | At matched false refusals, raw cosine and the linear bridge beat Chelated. #53's best arm is ZeroMu with raw cosine (0.50% FPR, 18.2% TPR), not ZeroMu with CSLS (0.67%, 12.7%). |
| #62 D13 | The conventional arms (62.97 and 62.87 JPM) beat Chelated (59.38). ZeroMu costs 3.59 JPM, and the sham control is invalid. |

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

This is about 10 hours of engineering. It is carried to the next cycle.

## W22-17 — core-math and prime-ring fixes: PRs open, Tier B pending

- **PR #369** (`fix/w23a-d03-d08-core-math`, base `main`):
  - D-03, D-04 and D-07 fixed.
  - D-05 deliberately unchanged, because of the documented F-023 guard.
  - D-06 confirmed, but the default is unchanged. Changing it would alter the C3/C3b golden trajectories, so that
    is an owner decision.
  - D-08 exists only on the #364 branch.
  - Self-score 85. It conflicts with #364's own versions of D-03, D-04, D-06 and D-07.
- **PR #368** (`fix/w23a-prime-ring-tie-contract`, base `codex/prime-ring-onion-method-dev`):
  - Before the fix, 117/336 cases matched. After the fix, 336/336 match, and adjacent unequal cases are 672/672.
  - Contract `PRW-NUMERICAL-TIE-CONTRACT-001` covers it.
  - Self-score 92.
  - Still open:
    - the constant 64 in the bound is not proven;
    - two other float tie sites remain;
    - old artifacts have not been regenerated.
- Not done in W23A:
  - the PR #293 (CD-R13-01) re-probe;
  - closing out PR #295 (CD-R16-01). Closing a PR is outward-facing, so it waits for the owner.
