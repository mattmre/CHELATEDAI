# Tier 2 — open PR index

Live snapshot on 2026-08-29. `MERGEABLE/CLEAN` below is Git graph state, never acceptance. Every PR returned `gh pr checks --required`: **no required checks reported**.

| PR | State / head | Opened evidence | Adversarial disposition |
|---:|---|---|---|
| [#308](https://github.com/mattmre/CHELATEDAI/pull/308) | draft; `8ce7feb`; MERGEABLE/BLOCKED | §4 failed twice: 12 BHS fields absent, 16 violations; §6.3 block failed twice; no reviews/threads | BHS 0 until body/gates exist; block flag remains authoritative |
| [#297](https://github.com/mattmre/CHELATEDAI/pull/297) | draft; `53a86f0`; merge state unknown | §4 failed; BHS fields absent; no reviews/threads | BHS 0 |
| [#296](https://github.com/mattmre/CHELATEDAI/pull/296) | draft; `65c9085`; CONFLICTING/DIRTY | §4 passes; §6.3 fails; body says 100 while carrying three merge blockers | at most 70/Critical; PR296-001 |
| [#295](https://github.com/mattmre/CHELATEDAI/pull/295) | non-draft; `730b305`; MERGEABLE/CLEAN | body says 100; only GitGuardian; two current unresolved threads | later exact probes score 60/Critical; withdraw after operator approval |
| [#294](https://github.com/mattmre/CHELATEDAI/pull/294) | non-draft; `6e78cf4`; MERGEABLE/CLEAN | body says 100; only GitGuardian; four unresolved threads (three outdated; remaining code issue fixed at head) | prior 100 bound to pre-transplant stack; replacement + fresh Tier B required; current unscored/0 |
| [#293](https://github.com/mattmre/CHELATEDAI/pull/293) | non-draft; `454e4a3`; MERGEABLE/CLEAN | body says 100; only GitGuardian; two outdated/fixed threads | later exact probes score 70/Critical; withdraw after operator approval |
| [#292](https://github.com/mattmre/CHELATEDAI/pull/292) | `eb75095`; CONFLICTING/DIRTY | latest §4 passes; one COMMENTED review; public head lacks local reconditioning | at most 70/Critical; stale claims/no hosted replacement runs |
| [#278](https://github.com/mattmre/CHELATEDAI/pull/278) | `9473e9f`; MERGEABLE/BLOCKED | §4 failed twice; no BHS fields; seven outdated cosmetic threads | BHS 0 |
| [#257](https://github.com/mattmre/CHELATEDAI/pull/257) | draft; `663be66`; CONFLICTING/DIRTY | lint fails (41); Python 3.9 cancelled; four current backend threads; “operator-review-deferred” is not Tier B | at most 70/Critical; parked; PR257 findings |
| [#256](https://github.com/mattmre/CHELATEDAI/pull/256) | draft; `4379b9c`; MERGEABLE/BLOCKED | §4 fails; lint fails (16); eight current threads | BHS 0; parked; PR256 findings |

All bodies, file lists, checks, reviews, issue comments, GraphQL review threads, failed-run logs, and diffs were opened. PR #257 exceeded GitHub's diff endpoint limit (406; 351 files), so the exact local base..head diff was opened instead. No PR has approval evidence.
