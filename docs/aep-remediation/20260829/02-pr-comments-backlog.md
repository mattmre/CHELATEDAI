# Tier 2 — PR review/comments backlog

## Counts and truth boundary

There are 27 unresolved review threads: #295 2 current; #294 4 (3 outdated, 1 current but fixed at head); #293 2 outdated/fixed; #278 7 outdated cosmetic; #257 4 current; #256 8 current. None has an author reply. `COMMENTED` Gemini reviews are summaries, not approvals.

| Priority | PR | Thread / issue | Disposition |
|---|---:|---|---|
| stop | #257 | [fallback config TypeError](https://github.com/mattmre/CHELATEDAI/pull/257#discussion_r3367630939) | Critical PR257-001; current |
| high | #257 | [nested isolation leak](https://github.com/mattmre/CHELATEDAI/pull/257#discussion_r3367630942) | PR257-002; current |
| medium | #257 | [callback serialization escapes envelope](https://github.com/mattmre/CHELATEDAI/pull/257#discussion_r3367630945) | PR257-003; current |
| high | #256 | [registry copy leak 1](https://github.com/mattmre/CHELATEDAI/pull/256#discussion_r3314775287), [copy leak 2](https://github.com/mattmre/CHELATEDAI/pull/256#discussion_r3314775289) | PR256-002; current |
| high | #256/#257 | hosted lint F821, `corrective` before assignment | PR256-001; duplicated by ancestry in #257 |
| review | #295 | [router](https://github.com/mattmre/CHELATEDAI/pull/295#discussion_r3610493332) | concurrency part fixed; norm cache is performance only |
| review | #295 | [Git environment](https://github.com/mattmre/CHELATEDAI/pull/295#discussion_r3610493334) | current thread; later rejection supersedes merge posture |
| normalize | #293/#294/#278 | outdated/fixed or cosmetic threads | resolve only on the correct retained/replacement PR; do not use resolution as acceptance |

Owner parking comments: [#257](https://github.com/mattmre/CHELATEDAI/pull/257#issuecomment-4686474160), [#256](https://github.com/mattmre/CHELATEDAI/pull/256#issuecomment-4686473987).

## Cross-PR completion contradiction

`docs/next-session.md:63-65,542` on PR #296 records later disproof: #292 public claims stale; #293 70/Critical after six iterations; #295 60/Critical after five; #294's 100 applies only to the pre-transplant stack. Live bodies still expose 100/mergeable language. See `AEP-20260829-PR296-001`. Canonical v3.7.1 treats legacy “L6 aggregated-claim drift” as L4; L6 is retired. The audit preserves “legacy L6” only when quoting the repository's older record.
