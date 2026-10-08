# T2 A2 PR/check receipt

Snapshot: `2026-08-29T22:20:05-04:00`; read-only. Commands opened `gh pr view` bodies/head/base/files/status rollups, REST issue/review/review-comment endpoints, GraphQL review threads, `gh pr checks --required`, failed/cancelled job logs, and exact local `git diff BASE...HEAD` objects when GitHub omitted large patches.

```text
#308 8ce7feb <- 3d94141  draft BLOCKED      paths=1   checks=17S/7F historical; latest body PASS/block FAIL; 0/0/0
#297 53a86f0 <- 34ce4b5  draft BLOCKED      paths=6   checks=11S/1F body; 0/0/0
#296 65c9085 <- 34ce4b5  draft DIRTY        paths=213 checks=10S/1F/1N; 0/0/0
#295 730b305 <- 12aa1be  open  CLEAN        paths=19  checks=1S; 0/1/2
#294 6e78cf4 <- 98e9dec  open  CLEAN        paths=5   checks=1S; 0/1/4
#293 454e4a3 <- eb75095  open  CLEAN        paths=6   checks=1S; 0/1/2
#292 eb75095 <- 34ce4b5  open  DIRTY        paths=38  checks=11S/1N; 0/1/0
#278 9473e9f <- 34ce4b5  open  BLOCKED      paths=8   checks=11S/1F body; 0/1/7
#257 663be66 <- 3595283  draft DIRTY        paths=351 checks=10S/1F/1C; 1/1/4
#256 4379b9c <- 3595283  draft BLOCKED      paths=168 checks=10S/2F; 1/1/8
```

All ten required-check queries returned no configured required checks. #257 Python 3.9 was cancelled after about six hours. #257 lint reports 41 issues including the existing F821 `corrective` path; #256 lint reports the inherited F821 path and its body validator misses 16 fields. #278's validator reports all twelve required fields absent. #308's newest block failure reads `BLOCKED`; its accumulated seven failures are repeated historical schema/block runs at the same head, not seven new test-suite failures.

Two net-new candidates survived normalization: `PR294-001` and `PR297-001`. PR #257's provisional fourth candidate was deduplicated to Round 2 `PR257-001`; PR #256's research-stub notes remain parked/non-production.
