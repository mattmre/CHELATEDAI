# Cross-panel challenge log

| ID | Challenge and resolution | Findings |
|---|---|---|
| CH-01 | **FE → BE:** Bearer comparison is sound but the browser has no credential path. Either implement a complete browser session contract or remove the remote-browser implication. | FE001, FE006 |
| CH-02 | **Reliability → FE/BE:** 67 unit tests passed while runtime reproduced 401 page failure. Add one production-path browser smoke and report unit/runtime evidence separately. | FE001, FE006 |
| CH-03 | **FE → BE:** Correct JSON 4xx/5xx is insufficient when clients render it as zero. Version the envelope and assert the rendered state. | FE006 |
| CH-04 | **Reliability → data owners:** Artifact presence is not freshness, reproducibility, or lifecycle. Require schema/source SHA/config/timestamp/state or show unknown. | FE002, FE005, WT003-001 |
| CH-05 | **Security → FE:** DOM injection is not Low merely because localhost is default; checkout/artifact authors are a trust boundary and non-loopback mode exists. | FE003 |
| CH-06 | **Implementation → FE:** Fixing only `dashboard/index.html` leaves inline fallback sinks. Change both or reduce to one canonical client without an L16 rewrite. | FE003 |
| CH-07 | **Accessibility → prioritization:** Click-only primary tabs remove functionality, not polish. Retain Medium and browser-keyboard ACs. | FE004 |
| CH-08 | **BE → PR audit:** Passing 60/15/49 unit tests cannot override later 70/60 Critical runtime disproof. Preserve both evidence sets. | PR296-001 |
| CH-09 | **Worktree → BE:** Source SHA written by the producer is self-attestation. RHPC must receive trusted expected identity externally. | WT020-002 |
| CH-10 | **Reliability → RHPC:** Hash-then-reread and check-then-replace are not immutable snapshot or atomic no-replace guarantees. | WT020-003, WT020-004 |
| CH-11 | **BE → Worktree:** Locally reconditioned code does not rehabilitate a stale public PR. Bind every claim to branch, SHA, dirty state, and fresh-checkout evidence. | PR296-001, WT001-001, WT020-001 |
| CH-12 | **Implementation → all:** `MERGEABLE/CLEAN` with no required checks is not a go signal. Gate state must be named independently. | all PRs |
| CH-13 | **Normalizer → worktree severities:** Recoverable uncommitted hazards are Medium/Low until demonstrated product loss; do not inflate them. | WT001-001, WT003-001, WT020-001, WT019-001 |
| CH-14 | **Sofia ↔ James dissent:** forgeable RHPC official provenance is Critical by decision-plane impact versus High by current uncommitted exposure. Root retains Critical; automatic stop-line applies before any official run. | WT020-002 |
| CH-15 | **Canonical taxonomy → older repo labels:** v3.7.1 retires L6 and folds aggregated drift into L4. Preserve “legacy L6” only as source history, never as current taxonomy. | PR296-001, WT003-001 |
