# Round 3 Tier 2 open PR index

Tier 2 began only after all eight dirty worktrees were dispositioned in Tier 1. All ten PR bodies, exact heads/bases, exact changed-path sets, diffs, reviews, issue comments, paginated review threads, accumulated checks, newest failed/cancelled logs, and required-check queries were opened. GitHub's file API capped three lists at 100; exact local object diffs established 213/#296, 351/#257, and 168/#256 paths.

| PR / exact head | Live state and checks | Tier 2 disposition |
|---|---|---|
| #308 `8ce7feb987490e8d39479673cba18604e376568d` | draft; MERGEABLE/BLOCKED; newest §4 pass and block-flag fail; 17 success/7 historical failures; 0 threads | body now truthfully records Tier B/official 0 and no functional result; retain draft |
| #297 `53a86f0fbdaa1ebb220023e3787d9231b332eaa8` | draft; MERGEABLE/BLOCKED; 11 success/1 body-validator fail; 0 threads | net-new Medium `PR297-001`: duplicate TSV primary keys are accepted and metrics inflate; retain record draft |
| #296 `65c9085cd048e8a7351a53e87666fdd5639e612b` | draft; CONFLICTING/DIRTY; 10 success/1 block fail/1 neutral; 0 threads | Round 1 `PR296-001` controls completion contradiction; local dirty overlay is not public-head evidence |
| #295 `730b305e8352b1c7f41e8b77062c4c7cba543dc6` | non-draft; MERGEABLE/CLEAN; GitGuardian only; 2 UI-unresolved threads | thread mechanisms fixed/non-actionable at head; dimension mismatch remains prior duplicate `CD-R16-01`/Round 2 `PR295-001` |
| #294 `6e78cf419c59ada7d4dd3ad2dace8f8721552451` | non-draft; MERGEABLE/CLEAN; GitGuardian only; 4 UI-unresolved threads | net-new High `PR294-001`: interrupted two-file rewrite silently joins new vectors to old IDs; thread items otherwise substantively fixed |
| #293 `454e4a32453b29040eda9c286ed120a158a265b4` | non-draft; MERGEABLE/CLEAN; GitGuardian only; 2 outdated/unresolved threads | both thread mechanisms fixed at head; later BHS rejection remains Round 1 aggregate |
| #292 `eb7509583e81b6a62d13b82587398562e4bba09a` | non-draft; CONFLICTING/REVIEW_REQUIRED; 11 success/1 neutral; 0 threads | stale public-head quantitative claims remain Round 1 `PR296-001` aggregate |
| #278 `9473e9f7d192299e6333608d91f3a659c63eda8d` | non-draft; MERGEABLE/BLOCKED; 11 success/1 §4 failure; 7 outdated newline threads | docs-only; body has all 12 required fields absent and control-character corruption; evidence-not-ready, reserved for Round 4 L9 contradiction pass |
| #257 `663be66e3292aae654a0f615e7c5e2b4dd7a98d4` | draft; CONFLICTING/DIRTY; 10 success/1 lint fail/1 Python 3.9 cancellation; 4 current threads | four mechanisms are Round 1 packets plus Round 2 `PR257-001`; owner parked; use clean replacement |
| #256 `4379b9cdeaa54b1b53f043f034ffc68bb60494d4` | draft; MERGEABLE/BLOCKED; 10 success/2 failures; 8 current threads | F821 and mutable-state packets carried; remaining research-stub comments are parked/non-production; use clean replacement |

Every `gh pr checks <n> --required` query reported no required checks, and REST reports the relevant bases are unprotected. This is `SKIPPED_GATE: REQUIRED_PR_CHECK_ENFORCEMENT`, not evidence that the visible checks are sufficient.

No open PR changes a served frontend/client path. Exact base/head blobs for all four known FE roots are equal across every PR; Round 3 Tier 2 therefore records `NO_NEW_FE_FINDINGS`.
