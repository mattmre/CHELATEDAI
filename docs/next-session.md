# Next Session

<!--
  This file is the Tier C (cross-PR / cross-session) state surface for the
  Brutal Honesty Rulebook v3.2. It is read by:
    - scripts/check_block_flag.py  (the merge gate for new feature work)
    - the Tier B reviewer of the next PR you open
    - whoever picks the work back up after a session boundary

  Schema is fixed. Do not rename headings. Do not collapse the tables. Empty
  state is honest state — empty rows do not get deleted, they get filled with
  the "_none yet_" placeholder which the validator recognizes.

  Cycle definition (per rulebook §6.1): 1 cycle = the next operator-initiated
  session OR 5 calendar days, whichever comes FIRST. Items with TTL=1 cycle
  must be cleared before the next cycle starts or the block flag flips to
  BLOCKED automatically.
-->

## Block flag

**Current**: `CLEAR` — no Carried Debt items have expired.

When the flag is `BLOCKED`, no new feature work may merge until Carried Debt
is empty. The flag is set automatically by `scripts/check_block_flag.py`:
- `CLEAR` if no Carried Debt rows OR all open rows are still in their first
  cycle (TTL not yet expired).
- `BLOCKED` if any open Carried Debt row has survived a full cycle without
  being closed.

The script does not know cycle age — that is set by the operator at session-
wrap by inspecting the TTL column. The `**Current**:` line above is the
authoritative source; everything else is advisory.

## Carried Debt

| ID | Item | Source | TTL | Blocking | Status |
|----|------|--------|-----|----------|--------|
| CD-001 | smoke_pipeline.py ceiling-tier not yet implemented; floor-tier only (`run_ceiling_smoke()` returns sentinel 2). Ceiling gap = no real end-to-end fixture exercise of AntigravityEngine | kit install 2026-05-10 | 1 cycle | NO — honestly disclosed per Rule 5 | OPEN — implement ceiling smoke once a lightweight fixture path is identified |
| CD-002 | `scripts/smoke.sh` Stage 1 exits non-zero: `tests/test_e2e_smoke.py` does not exist; smoke.sh is the `bash`-mode entry point but the repo has no e2e smoke test file. The Python `smoke_pipeline.py` path (used by CI and operator) is unaffected. | kit v3.3 upgrade 2026-05-12 | 1 cycle | NO — CI uses `smoke_pipeline.py` directly; gap is only in the `bash scripts/smoke.sh` code path | OPEN — add `tests/test_e2e_smoke.py` surface-boot test or re-route smoke.sh Stage 1 to smoke_pipeline.py |

**Schema**:
- `ID`: stable identifier, prefix `CD-` + sequential number (CD-001, CD-002, ...).
- `Item`: one short sentence stating the gap. Reference file:line where useful.
- `Source`: PR number or session ID that opened the debt.
- `TTL`: `1 cycle`, `expired`, or `—` (placeholder row only).
- `Blocking`: `YES` (forbids new feature work in next cycle if not cleared)
  or `NO — <reason>` (cosmetic / honestly-disclosed gap that does not block).
- `Status`: `OPEN — <note>`, `**CLOSED** by PR #N`, or `_—_` (placeholder).
  Rows starting with `CLOSED` (case-insensitive, stripping `**` markdown bold)
  are filtered out of the active-debt count by `check_block_flag.py`.

## Deferred Scope

Items from this cycle's PRs whose `DEFERRED_SCOPE:` field captured ≥25% of
original scope. These are NOT debt — they are honestly-bounded
not-in-scope items. They appear here so the next planner sees them.

| ID | Item | Source PR | Why deferred |
|----|------|-----------|--------------|
| _none yet_ | _—_ | _—_ | _—_ |

## Aggregate BHS trend

Track `BHS_OFFICIAL` per merged PR over the last 5 cycles. Falling trend = the
remediation loop is succeeding. Flat-at-100 trend = either real progress or the
loop is being gamed (Tier B agents not adversarial enough). Spot-check the
Tier B reports if the trend looks suspicious.

| Cycle | PRs merged | Avg BHS_OFFICIAL | Avg LOOP_ITERATIONS | OPERATOR_OVERRIDE count |
|-------|------------|------------------|---------------------|------------------------|
| _current_ | _—_ | _—_ | _—_ | _—_ |

## Operator overrides log

Every PR merged at `BHS_OFFICIAL < 100` (i.e. with `OPERATOR_OVERRIDE:`
populated) gets a permanent row here. The override creates an automatic top-
priority Carried Debt entry; this log is the audit trail.

| PR | BHS_OFFICIAL at merge | Override reason | Override author | Out-of-band ref |
|----|----------------------|-----------------|-----------------|-----------------|
| _none yet_ | _—_ | _—_ | _—_ | _—_ |

---

**Last session**: 2026-05-10 — kit install (v3.2 baseline)
**Last validated by `check_block_flag.py`**: run after this PR merges
