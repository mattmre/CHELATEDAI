# Wave 23A disposition (W22-01..W22-05, W22-17)

Date: 2026-11-26. Spine branch `pv7/w16-core`. First written at `7ffbd0df`. Round-1 corrections are in `2c125bef`, and round-2 corrections are in the commit that adds this line.
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
- **W22-16:** SELECT and REPORT are separate splits. Round 1 also asked to scope the kill to k=4; round 2 showed
  that was wrong.

**Tier B round 2 has been done too**, by a fresh reviewer on `2c125bef`:

| Report | Round 1 | Round 2 | What round 2 found |
|---|---|---|---|
| W22-02 | not scored | 98 | Nits only. |
| W22-03 | not scored | 90 | Six omissions, all fixed: SciFact stale rate is 31–36% at 90% retraction and FiQA has none; REDOSONLY differs vs k17 too; a third Holm survivor (phase3:de c_vs_a, p 0.029); Arm 5 ≡ Arm 9; the UncCSLS cells do exist (recomputed); the FiQA DEV win is marginal (p 0.091). |
| W22-16 | 93 | 85 | The round-1 "kill is only at k=4" caveat was wrong, because the identical selection also holds at k=1. The SAE wrong-sign excuse applies only to Qwen3.5. A line citation was off by two. All fixed. |
| This file | not scored | 88 | The defects above, plus a stale head line and missing round-1 scores. Fixed. |

I re-checked every round-2 finding against the artifacts myself before editing. None of these scores is 100, so
every verdict stays PROPOSED. No tier moves until a round-3 reviewer clears the corrected text.

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
| #41 D6 | **PARTIAL** | Merge quality survives: Chelated beats raw cosine and RRF in all 8 direction groups, and DEV z-score in 7 of 8 one-sided (6 of 8 robustly; FiQA 4096 d384→d4096 has p 0.091). DEV z-score significantly beats Chelated in SciFact 4096 d384→d4096 (UCB95 −0.0036). Latency is not shown, because the timers are unmatched and the scorecard already says "no pure ANN latency claim". | New: the DEV z-score loss. |
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

## W22-05 — bitemporal SQL ledger comparator and negation-cycle census: preregistered rule says RETIRE (PROPOSED)

- **Branch:** `pv7/w23a-w22-05-ledger-comparator`. Preregistration `cfd40d6d` was committed before any result;
  the implementation is `1c9176f7`.
- **Verdict under the preregistered rule:** `RETIRE_EK5_CLEAN_GRAPH_PREMIUM`. EK5 and an ordinary SQLite bitemporal
  ledger agree on all 1,020 items: arena 60/60 for both arms, and #55 420/960 for both, failing on the same cuts.
  Harm is identical at every p.
- **Limits:**
  - The arena cannot tell the arms apart. An authority-only ledger also scores 60/60.
  - So this shows "no measured clean-graph premium on these inputs", not that one could never exist.
- **Negation-cycle census:**
  - Arena: 30/60 graphs have a mutual-REFUTE 2-cycle (every contested and rumor case).
  - #55: 0 of 240.
- **Not reproduced:** the 962 explanation-diff figure. At 6 steps the count is 153, which matches the citation. At 12
  steps it is 825, not 962, and I don't know why.
- **Shuffled-order rebuild:** 81/960 `winner_record_id` mismatches. These are tie-break by insertion order; status and
  value never change.
- **Tier B round 1 (fresh agent):** 80, severity important. The numbers reproduce exactly and the rule matches the
  preregistration. The reviewer found the verdict has little power:
  - On #55, EK5's graphs hold only supersede edges, with 0 copy edges and 0 cycles, so its machinery never runs.
  - Every #55 conflict item sits in one stratum, which both arms get fully right.
  - The #55 tie was already known before the preregistration.
  - These limits are folded into `report.md` (`f757103d`).
- **Draft PR #370.** BHS_OFFICIAL = 80. Not mergeable.
- **DEFERRED_SCOPE:**
  - the PostgreSQL 17 arm;
  - a matched Conv120 comparison;
  - the gap between 825 and 962.

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

## W23B early result: W22-08 capture-recapture — KILLED (PROPOSED)

- **Branch:** `pv7/w23b-w22-08-capture-recapture`. Preregistration `3911ad1c`; implementation `c105b6fe`.
- **Result:** none of the five reviewer-list families passes all four kill rules (K1 independence, K2 overlap,
  K3 finite estimate, K4 bound).
  - F-PV7 S_A and S_B die on K1. The final judges read the peer memos, so only one list in each stratum is
    independent.
  - F-W8 and F-R5 die only on K2. Their overlaps are 0.85 and 0.82, against a 0.80 cutoff, so a different defect-ID
    granularity might change this.
  - F-W14 dies on K2 and K3.
  - F-W4 dies on K1.
- **What this means:** a Chao-style estimate of "how many defects all reviewers missed" cannot be supported from the
  existing council logs.
- **A separate finding:** the W23A audit found 4 S_A defects that no F-PV7 list had caught (B53:C08, B54:C08,
  B62:C08, B62:C09).
- **Tier B round 1 (fresh agent):** 85, severity important. The result reproduces byte for byte. F-PV7 is dead for
  structural reasons. The overall KILLED is **fragile**:
  - F-R5 passes all four rules if r5_grok is held to its own declared scope.
  - F-W8 passes after three plausible recodes under the coder's own tie-break rule.
  - The fragility note is in the generated report (`fcaffde6`).
- **Draft PR #371.** BHS_OFFICIAL = 85. Not mergeable.
- **Carried forward:** a second independent coder for F-W8 and F-R5.

## W23B/W23C: W22-07, W22-09 and W22-11 after Tier B round 1

All three are draft PRs against `pv7/w16-core`. None can merge, because BHS is below 100.

| Item | PR | Verdict (PROPOSED) | Self | Tier B | Official |
|---|---|---|---|---|---|
| W22-07 copy twins | #372 | `KILL_T_D`, zero-provenance-signal regime only | 92 | 88 (important) | 88 |
| W22-09 retraction boundary | #373 | KILLED, carried by K2 on #60 | 90 | 85 (important) | 85 |
| W22-11 WFS deletion stress | #374 | `CLEAN_1M_CANONICAL_FROZEN` | 92 | 88 (important) | 88 |

- **W22-07:**
  - The λ=0 twins are genuinely blinded. Text-only balanced accuracy is at most 0.513, and the legacy fixtures
    #55/#20/#38 leak at 1.000.
  - A5 (Bayesian copy model) is at least as good as A6 (T-D): H(0.5) 0.172 vs 0.176, CI on the difference
    [−0.021, +0.028].
  - **Tier B limits:**
    - The primary cell cannot exercise A6's targeting or its pricing.
    - Seed 20261127 on its own gives ADVANCE.
    - A5 and A6 differ on 77/720 families, a net of 3.
    - At λ=0.25 A6 wins all 18 B=1 cells.
  - Read the kill as "no T-D premium without provenance signal".
- **W22-09:**
  - A flat KV checkpoint with suffix replay matches #60's retraction. Three Tier B re-runs gave latency ratios of
    0.39–0.78×.
  - The K1 firings are demoted:
    - `P51-DIGEST` is a coverage gap in the `checkpoint_replay_exact` metric. #51's answers come from `inc_store`, so
      no wrong answer can be served through it.
    - `PEK5-ASOF` is a defect only under bitemporal semantics.
  - The titled "compressed neural state" does not exist in any bundle.
- **W22-11:**
  - 1,266,000 mutation checkpoints, with 0 semantic and 0 C1 mismatches. CW-1 is frozen.
  - The raw `pos_cyc`/`reason` telemetry is history-dependent: 3.08M one-way diffs.
  - Only about 0.53M checkpoints are on graphs larger than 11 nodes.
  - **New carried debt `CD-W22-11-01`:** `demo_workbench/compliance_copilot_engine.py:785/929/1086` shows raw
    `store.reason` to users. A stale `pos_cyc` therefore reads "retracted by a higher-authority notice" where the
    canonical answer is "circular support only".
  - The Lean precondition is met by the numbers. Funding it is the owner's call.

## W22-10: what runs

- **N3 swarm-threshold part: NOT RUN.** The queue makes it conditional on W22-07 passing. W22-07 killed T-D in the
  zero-signal regime, so the condition is not met.
- **#60 KV-guard scope-bound part:** not conditional on W22-07. It is CPU-only and is queued next.
