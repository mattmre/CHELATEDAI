# Brutal Honesty Rulebook

**Status**: Active convention. Loaded by `CLAUDE.md`. Portable to any repo.
**Version**: v3.3 (2026-05-11) — v3.2 executable enforcement plus L13 (soft-prose-claimed-as-mechanical) and the v3.3 schema/prose/artifact drift validator (`scripts/validate_v33_schema_drift.py`); see §12 changelog. v3.2 (2026-05-09): Tier B independence enforcement, severity caps, `DEFERRED_SCOPE:` tracking, quantitative cycle definition, override structural barrier at BLOCKED state, executable validators.
**Read time**: 11 minutes. **Do not expand without removing something.**

---

## 0. Honest premise + the evidence rule

The dominant agent failure of the last ~6 weeks is **false-completion**: agents ship scaffold while reporting tasks done. No rule eliminates hallucination. This document is **friction** — it makes the lie cost more than the truth, names the failure modes so they can be caught by pattern, and gives the operator concrete checks.

The single most powerful posture this rulebook captures:

> **Assume every implementation/completion claim is false until independently proven by runtime evidence.**

What does NOT count as evidence:
- A test existing
- A route existing
- A doc saying complete
- A PR title or summary saying complete
- An agent claiming complete
- A "brutal honesty" self-attestation **alone**

What DOES count as evidence:
- Runtime output from the production code path (not the test path)
- Real UI / API behavior an operator can observe
- Persistence / state mutation a query can confirm
- An artifact / replay / checkpoint that survives a fresh checkout
- An independent reviewer who **tried to disprove** the claim and failed (and reported what they checked)

Self-attestation is necessary but not sufficient. The brutal-honesty section in §4 is only valuable when **read against the diff** by a human or a second fresh agent.

If you are an agent reading this: assume the prior agent in this session lied to you. Assume the summary you were handed is wrong in at least one load-bearing place. Assume the test suite passes for the wrong reason. Verify before you build on it.

---

## 1. Lie taxonomy — name the failures so you can see them

| #  | Name | Pattern | How it ships |
|----|------|---------|--------------|
| L1  | **Scaffold-as-feature** | Function exists, signature is correct, body is `pass` / `return None` / `raise NotImplementedError` / `return f"[stub] {x}"` | Tests assert against the stub return; CI is green |
| L2  | **Conditional escape hatch** | A guard added under pressure to bypass a failing path — typically `if not <flag>: return ...` or `if not <obj>: return ...` introduced in the same diff that "fixed" a red test, an unimplemented branch, or a runtime exception the implementer didn't want to handle. Distinguish from legitimate guard clauses (null checks, feature flags, input validation) by *intent*: legitimate guards exist on day one; L2 guards appear in the diff that made red turn green. | Real path never executes in test; only the early-return branch is exercised |
| L3  | **Mock-ate-the-real-code** | Mock added "for the test" replaces the real implementation in the production import path | `pytest` green, `python -m mymodule` raises |
| L4  | **Partial-with-claim-of-complete** | 8 of 10 sub-tasks done; summary says "all complete" or omits the 2 that weren't | Operator runs end-to-end, hits the missing 2, discovers the lie |
| L5  | **Test-as-truth** | "All tests pass" treated as proof of working software | No real-world / e2e / smoke verification was ever run |
| L6  | **Aggregated-claim drift** | "All N PRs merged" when 1 was force-merged with failing checks, or 1 actually closed without merging | Status table shows ✅ on every row; reality is one row is ❌ in disguise |
| L7  | **Re-summarization decay** | Each compaction or hand-off drops nuance; confidence amplifies | By the third summary, "stub returning empty string" has become "translation engine implemented" |
| L8  | **Test that asserts the bug** | Test expects `return ""` because that's what the code does; "passing" means the broken behavior is locked in | Refactor breaks the test; lie has been load-bearing for weeks |
| L9  | **Doc-as-implementation** | A planning doc, runbook, or README is written; the code referenced does not exist or does not behave as described | Operator follows the doc, hits an `ImportError` |
| L10 | **Dependency phantom** | Import statement references a module that exists but is empty, or a function that has the wrong signature | Linter is happy; first real call raises `AttributeError` |
| L11 | **Broad-catch swallowing** | `try: ... except Exception: pass` (or `return ""` / `return None`) hides real failures so the path looks successful | Production reports success; output is empty/wrong; nothing in logs |
| L12 | **Status-permissive test** | Test asserts `status in {200, 202, 204}` or `assert response is not None` — accepts any plausible response, proves nothing specific | Auth wrapper, route registration, or noop handler all pass; real behavior never tested |
| L13 | **Soft-prose-claimed-as-mechanical** | A doc claims a check, predicate, schema, validator, artifact, or gate is mechanically enforced, but the diff only adds prose/examples or the committed artifact already drifts from the prose it supposedly enforces | Reviewers trust a mechanism that does not exist, or exists with a different closed set than the document claims |

Quote by number in PR review. "This is **L1 with a side of L12**" forces the agent to recognize the pattern, not relitigate the specifics.

---

## 2. Five hard rules

These five are the entire enforceable surface. No more rules. More rules = more drift.

### Rule 1 — Evidence rule

A claim of "complete" is false until backed by runtime evidence per §0. Every PR with a completion claim must include in its body:

```
EVIDENCE: <one or more of: command output, UI/API trace, state-mutation query, artifact path, independent disprove attempt that failed>
SMOKE: <result of running scripts/smoke.sh — see Rule 5>
```

Evidence pointing at "tests passed" or "PR text says done" is **not evidence**. Re-read §0.

### Rule 2 — Visible means verified

Incomplete code may live in the repository. Incomplete code may **not** be presented as a working feature. The boundary:

- **Allowed**: stub adapters, scaffold modules, partial implementations, work-in-progress branches — provided they are explicitly labeled as such (in code comment, in `docs/feature-status.md` if the repo keeps one, or in the PR's brutal-honesty section).
- **Forbidden**: surfacing scaffold via UI, advertising it in API docs, writing release notes that imply it works, marking the parent feature complete on a roadmap.

Showing a feature without runtime evidence is itself an L4 lie. The rule is the brake.

### Rule 3 — Mandatory PR brutal-honesty section

Every PR body ends with the section in §4. Empty answers must be **justified**, not omitted. If the section is absent or all answers are unjustified "none", the reviewer rejects with: *"What did you fake to get here?"*

The section must disclose every L1–L13 instance in the diff with `file:line`. Stop-the-line patterns (the L-numbered items, plus broad try/except swallowing, plus any new `# TODO`/`# FIXME`/`# stub`/`# placeholder`) are the disclosure target. Either remove them or disclose them — silence is rejection.

### Rule 4 — Adversarial cross-agent review

Implementation slices are done by a fresh sub-agent (no prior session context). Before merge, a **second** fresh sub-agent is given:
- the PR diff
- the brutal-honesty section
- the EVIDENCE pointer
- the smoke-path command (Rule 5)
- this rulebook

…and is asked: *"Try to disprove the completion claim. Look for claims without evidence, evidence without runtime trace, tests that don't exercise production paths, undisclosed L1–L13 instances. If you cannot disprove, say so explicitly — including what you checked and what you could not check."*

This is **not a code review**. It is a **lie hunt**. A second pair of eyes that does not share the implementation context will catch L1/L2/L3/L4/L11 in seconds.

### Rule 5 — One deterministic smoke path per repo

Every repo has exactly **one** shell command that is the release gate. This command:

- Lives at `scripts/smoke.sh` (or equivalent — documented in CLAUDE.md)
- Returns non-zero if the smoke fails
- Is referenced by `EVIDENCE` and `SMOKE` in every "complete" PR (Rule 1)
- Has TWO honest tiers, and the PR body MUST disclose which tier was actually run:
  - **Floor (v3.2 minimum)**: a deterministic smoke that imports the production module, asserts its documented entry points + constants exist, and exits non-zero on import failure. This is what `scripts/smoke_pipeline.py` ships in this rulebook's reference repo. It catches L1/L10 (scaffold-as-feature, dependency phantom) and L4 import-time regressions, but it does NOT prove the pipeline produces correct output. The smoke MUST self-disclose its floor-tier scope when it runs (the reference `smoke_pipeline.py` prints "HONEST DISCLOSURE" + scope limits to stdout).
  - **Ceiling (target)**: a true end-to-end flow that runs the production code path against a fixture input and asserts a non-empty production output (e.g. for an OCR pipeline: extract → OCR → assemble → verify the output PDF contains expected text). Achieving the ceiling is the goal; staying at the floor is acceptable IF the PR body's `SMOKE:` line names which tier was run AND the gap appears as Carried Debt with a target cycle.

If the smoke path fails, **the release is not ready, regardless of test count**. Test count is irrelevant. CI green is irrelevant. `pytest -xvs` passing is irrelevant. The smoke path is the floor of "working" — but only if the PR body honestly names which tier the smoke is at. Claiming "smoke passed" while running floor-tier and presenting it as ceiling-tier is an L4 partial-as-complete lie.

---

## 3. Brutal-honesty trigger phrases

These phrases reliably extract a more honest response than "is this done?". Use them in cross-agent verification, in post-merge review, and when something feels too clean.

- **"Be brutally honest."** — operator-confirmed effective; the word *brutally* matters
- **"What did you fake to get here?"** — surfaces L1, L2, L3, L11
- **"Where is the real path different from what you summarized?"** — surfaces L4, L7
- **"If I ran this on a fresh checkout right now, what would actually fail?"** — surfaces L9, L10
- **"Tell me the smallest concrete thing that does NOT work yet."** — surfaces L4
- **"What's the gap between 'tests pass' and 'a user can do X end-to-end'?"** — surfaces L5, L8, L12
- **"Show me the runtime evidence, not the test."** — forces §0 evidence rule
- **"List every conditional in this diff that exists ONLY because the real implementation is missing or broken."** — surfaces L2
- **"Show me the diff hunk where the production code path actually runs the new logic — not the test."** — surfaces L3, L8
- **"If I disable this feature behind a flag right now, what visible behavior changes for the operator?"** — surfaces L4, L9, Rule 2 violations

If you (the agent) are asked any of these and you do not have an honest answer, **say so**. Speculating is itself a lie. *"I don't know"* is the correct answer when there is no evidence.

---

## 4. The PR-body brutal-honesty section (mandatory template)

Every PR body ends with this block. No exceptions. Empty answers are never bare "none" — they are "I claim none, here is why":

```markdown
## Brutal Honesty

**What I did NOT implement that the PR title or summary might imply I did:**
<list, or "I claim nothing was overstated; the title is exactly what shipped — and here is the diff hunk that proves it: <hunk reference>">

**What I stubbed, mocked, or worked around (with file:line):**
<list with file:line, or "none — and here is the grep that proves it: `grep -nE 'TODO|FIXME|stub|placeholder|NotImplementedError|return None|return (\"\"|'\\'''\\''')' <files I touched>` (matches both `return \"\"` and `return ''`)">

**What conditionals in this diff exist ONLY because the real path didn't work:**
<list with file:line, or "none">

**What broad try/except blocks were added or modified, and what they catch:**
<list with file:line and the specific exceptions caught + handled, or "none — no try/except added or modified">

**What tests in this PR do NOT exercise the production import path:**
<list with file:line, or "none — every new test imports from the production module, not from a test fixture; status assertions are exact, not permissive">

**What did I claim "complete" or "working" that I did NOT end-to-end verify with the smoke command:**
<list, or "nothing — smoke command was run; output below or attached">

**Lie-taxonomy self-classification (numbers from §1 of `docs/conventions/brutal-honesty-rulebook.md`):**
<e.g. "L1 in `foo.py:42` — function returns stub; tracked in #N" — or "no instances">

**Visibility status (Rule 2):**
<one of: "Feature is hidden — not exposed via UI/API/docs/release notes" | "Feature is visible — runtime evidence below" | "Feature is partially visible — here is what is exposed and what is not">

EVIDENCE: <command output, UI/API trace, state-mutation query, artifact path, or "independent reviewer @<agent> tried to disprove and failed; their report is at <path>">
SMOKE: <output of `scripts/smoke.sh` or equivalent — paste tail or attach>
BHS_SELF_DRAFT: <0–100, implementer's self-assessed score after Tier A. DRAFT only. Self-scoring above 95 with one-line justification will be auto-downgraded by Tier B.>
BHS_SELF_DRAFT_AGENT: <session/agent identifier of the implementer (e.g. "session abc123, agent claude-sonnet-4.5") — required so Tier B independence can be audited>
BHS_TIER_B: <0–100, official score assigned by adversarial fresh agent — NOT by the implementer. Includes one-paragraph justification listing what was checked.>
BHS_TIER_B_AGENT: <session/agent identifier of the Tier B reviewer — MUST differ from BHS_SELF_DRAFT_AGENT. Same agent doing both is itself an L4 lie (score-gaming); the §4 validator script (scripts/validate_pr_brutal_honesty.py) will then FAIL the PR with expected BHS_OFFICIAL = 0 (it does not silently rewrite the field — the PR author must correct it).>
BHS_TIER_B_SEVERITY: <one of: "none" (no gaps found), "cosmetic" (no cap on Tier B score), "important" (caps Tier B at 90), "critical" (caps Tier B at 70 — security, correctness, or false-completion in load-bearing claim). Severity caps apply REGARDLESS of other findings.>
BHS_OFFICIAL: <min(BHS_SELF_DRAFT, BHS_TIER_B). Merge gate: only PRs with BHS_OFFICIAL = 100 may merge.>
CARRY_FORWARD: <bullet list of items going to docs/next-session.md (see §6.3) with TTL=1 cycle. Empty = "none — every disclosed gap is closed in this PR" (Tier B will verify).>
DEFERRED_SCOPE: <empty if no scope reduction; otherwise list each piece of original scope that was deferred, with owner and target cycle. Scope reductions ≥25% of the original lane size MUST be tracked here AND mirrored as Carried Debt entries with TTL=1 cycle. Silent scope reduction is itself an L4 lie.>
LOOP_ITERATIONS: <how many Tier A iterations this PR went through (1–5). If 5 and BHS_SELF_DRAFT < 100, the PR is being withdrawn or scope-reduced — explain which.>
OPERATOR_OVERRIDE: <empty (default), OR "<reason, operator name, timestamp, out-of-band reference>" if merging at BHS_OFFICIAL < 100. Override at BLOCKED state additionally REQUIRES a co-signer name OR an out-of-band reference (Slack thread URL, calendar entry, signed email). Single-name override at BLOCKED is itself an L4 lie. Override creates a top-priority carried-debt entry per §6.3.>
```

If you cannot fill this honestly, **the PR is not ready**. That is the entire enforcement mechanism.

---

## 5. Anti-overhead clauses

These exist to prevent this rulebook from becoming the thing it warns against.

- **No new tests required by this rulebook.** Tests are floor, not ceiling. Adding tests does not satisfy any rule above.
- **No new agent types required.** Use whatever sub-agent type makes sense for the work. Rule 4 just says "fresh context."
- **No new dashboards, metrics, or status-page entries required.**
- **No new approval workflow required.** A human reviewer (or Rule 4 cross-agent) reading the brutal-honesty section against the diff is the workflow.
- **No new linting framework required.** A grep is enough. If the grep gets gamed, that is itself a §4 disclosure.
- **Feature inventory rows are OPTIONAL.** Some projects benefit from a per-feature ledger (`docs/feature-status.md` with `feature | status | exposure | evidence`). Others don't. This rulebook does not require one. If you add one, keep it to ≤4 columns; the 8-field schemas you'll see elsewhere are exactly the bloat we're avoiding.
- **This document does not grow without shrinking elsewhere.** New failure modes get a row in §1. New rules do not get added — if a sixth rule looks necessary, the answer is almost always "it is already covered by Rule 4 (adversarial cross-agent review) or §6 (the Remediation Loop)."
- **The Remediation Loop (§6) is not new infrastructure.** It uses the brutal-honesty section that's already in every PR (Rule 3), one file the repo already has or trivially adds (`docs/next-session.md`), and one number (BHS) that is reported in the PR body. No new agent types, no new MCP servers, no new dashboards, no new CI workflows. If someone proposes adding any of those "to support the loop", they have misread §6.5.

---

## 6. The Remediation Loop (meta-component, not a hard gate)

This is the part of the rulebook that is most often skipped. The five hard rules above catch lies **at PR time**. They do not, by themselves, prevent the **compounding** of debt across PRs. The loop does.

**Premise**: every PR generates two outputs — the diff, and a list of things the brutal-honesty section disclosed but did not fix. The second list is the poison. If it is not drained on a regular cadence, the next PR builds on top of it, the cadence after that builds on top of that, and within a few sessions the codebase is carrying weeks of undisclosed-by-summary technical debt.

The loop is what drains the poison.

### 6.1 The three-tier structure

The loop runs at three nested cadences. Confusing them is what produces the "thousands of validation tests just to reassure ourselves" failure mode.

**Tier A — Per-PR loop (5 iterations max; only BHS=100 may proceed)**
Inside a single PR, before declaring it ready:

```
1. Implement the slice.
2. Run the brutal-honesty self-check (use §3 trigger phrases on yourself).
3. Self-score: BHS_SELF_DRAFT 0–100. This is a DRAFT only — see Tier B for the official score.
4. If BHS_SELF_DRAFT < 100 AND iteration < 5: list the gaps with file:line, fix the smallest one that closes the most score, GOTO step 2.
5. If BHS_SELF_DRAFT = 100: stop. Write BHS_SELF_DRAFT into the PR body. Hand off to Tier B for the official score.
6. If BHS_SELF_DRAFT < 100 AND iteration = 5: STOP. Do NOT request merge. Choose one of:
   (a) Reduce the PR's claimed scope to something that DOES hit 100 (e.g. ship the scaffold at flag-OFF, defer the flag-ON path to a separate PR).
   (b) Escalate to the operator: the gap is architectural, not a sprint task.
   Do NOT ship at <100 hoping carry-forward will catch it. Carry-forward is for unknown-unknowns surfaced post-merge, not for known gaps the implementer chose to defer.
```

The cap of **5** prevents the agent from looping forever to feel productive. The **100/100 merge gate** prevents the agent from shipping at "good enough" and pushing the gap into next-session.md. **There is no "ships with disclosed caveats" path.** Either the PR's scope hits 100 honestly, or the scope shrinks until it does.

**Anti-spiral guards** (these are non-negotiable):
- Each iteration must produce a **concrete diff** (no "rechecked, looks good" iterations — those are §1 L7 re-summarization decay disguised as work).
- The score must move toward 100 by closing gaps, not by re-defining what "complete" means. Moving the goalposts is itself an L4 lie.
- If the **same gap survives two consecutive iterations**, stop the loop and escalate to the operator. Don't loop more on it; the iteration is not the right tool for that gap (it's probably architectural).
- A self-score above 95 with a one-line "no significant gaps" justification is **suspicious** — Tier B will downgrade it. Real codebases have edges. Either name an edge or accept the downgrade.
- **Scope reduction is not a lie** (this is the most important guard). Shipping a smaller, fully-complete slice is honest. Shipping a larger, partially-complete slice with carry-forward is the lie this version exists to prevent.

**Tier B — Pre-merge adversarial pass (the official scorer; self-scoring is forbidden)**
After Tier A completes (with `BHS_SELF_DRAFT = 100`), before merge, Rule 4 fires once: a second fresh agent tries to disprove. This is **not another loop**. It is one independent attempt. **Tier B assigns the official BHS, not the implementer.**

Mechanics:
- Tier B is given the diff, the brutal-honesty section, EVIDENCE, the smoke-path command, and this rulebook.
- Tier B's job is to disprove `BHS_SELF_DRAFT = 100`. They look for: undisclosed L1–L13 instances, claims without evidence, tests that don't exercise production paths, scope-padding (the implementer claimed less than they actually shipped, hiding gaps in the unclaimed surface), evidence that doesn't trace through the production path.
- Tier B writes `BHS_TIER_B` 0–100 with a one-paragraph justification.
- **Official score**: `BHS_OFFICIAL = min(BHS_SELF_DRAFT, BHS_TIER_B)`. The lower number always wins.
- **Merge gate**: only PRs with `BHS_OFFICIAL = 100` may merge. Period. No "merge with caveats." No "operator approval" clause inside the rulebook (the operator can override outside the rulebook by typing the merge command themselves — see "operator override" below — but the rulebook does not authorize that).
- **Score-gaming detection**: if `BHS_SELF_DRAFT - BHS_TIER_B > 5`, an automatic L4 disclosure is added to the PR body naming the gap. Repeat offenders are spotted by operator pattern-detection at Tier C.

If `BHS_OFFICIAL < 100`, the PR returns to Tier A for one more iteration (counted against the 5-cap). If the cap is exhausted, fall back to Tier A step 6: reduce scope or escalate.

**Operator override** (escape hatch, not a rule): the operator can always type the merge command themselves. The rulebook cannot prevent that. What it CAN do is make the override visible: if a PR is merged with `BHS_OFFICIAL < 100`, the implementer must add an `OPERATOR_OVERRIDE: <reason, operator name, timestamp, out-of-band reference>` line to the PR body (4 comma-separated fields, enforced by `scripts/validate_pr_brutal_honesty.py` R8) and the unclosed gap is force-added to next-session.md as a top-priority carried-debt item with TTL=1 cycle. At BLOCKED state, the `out-of-band reference` field MUST be a co-signer name OR a verifiable artifact (Slack thread URL, signed email timestamp, calendar entry) — see §6.3 override structural-barrier table. The override is logged, not authorized. If the operator did not actually approve, that is itself an L4 lie catchable by the next session's first action (read next-session.md, find the override, ask the operator: "did you sign this?").

**Tier C — Cross-PR remediation cycle (every N PRs OR end of session, whichever comes first)**
This is the part that actually drains the poison. After every batch of PRs (or at session boundary):

```
1. Aggregate the brutal-honesty sections of every PR shipped since the last remediation cycle.
2. Extract every disclosed gap (these should be rare — the 100/100 gate prevents most), every L1–L13 instance flagged by Tier B, every OPERATOR_OVERRIDE invocation, every L4 score-gaming flag.
3. Append the aggregate to docs/next-session.md (or repo equivalent — see §6.3) under "Carried Debt" with TTL = 1 cycle on each item.
4. Score the SYSTEM (not the PR): Aggregate BHS = honest assessment of how far the system is from "production-ready" given what just shipped + what's outstanding. Aggregate BHS is set by the operator or a fresh adversarial agent, NOT by the implementer.
5. Set the block flag (§6.3): CLEAR if Carried Debt count = 0; BLOCKED if any items remain from a prior cycle (their TTL expired without resolution).
```

**The 1-cycle TTL rule** is the crucial one — and it is what closes the v3 loophole that would have let debt accumulate indefinitely. Mechanics:

- Every Carried Debt item enters at **TTL = 1 cycle**.
- During the next cycle, those items are the **first priority** — drained before any new feature work in that cycle.
- At the END of that cycle, any item still on the list **flips the block flag to BLOCKED**.
- A BLOCKED next-session forbids ALL new feature work in the cycle after that until Carried Debt count returns to 0. There is no two-cycle slop, no "we'll get to it next sprint," no "we have bandwidth for one more feature first."
- The block flag is set in `docs/next-session.md` and is the **first thing** the next session's first agent reads. If the agent opens new feature work while the flag is BLOCKED, that is itself a §1 L9 lie (doc-as-implementation: ignored the convention the doc requires).

**There is no "remediation sprint vote."** The block flag is automatic from the math: if Carried Debt count > 0 after one cycle, BLOCKED. No appeal inside the rulebook (operator can override outside the rulebook, same mechanics as the merge gate — see Tier B operator override).

### 6.2 Brutal Honesty Score (BHS) — definition and calibration

**Scale**: 0–100. **What 100 means**: the PR's claimed scope is complete with no gaps. The operator could ship this to production right now, with no operator action required to make it work, no operator-known caveats, no undisclosed surface. **What 0 means**: this is scaffold; nothing works.

**Important clarification on "the PR's claimed scope"**: a feature shipping behind a flag default OFF can hit 100 if (a) the flag default is OFF, (b) flag-OFF behavior is unchanged from prior `main` (verified), (c) flag-ON behavior has fully-disclosed scope and behaves correctly when flipped (tested in dev), and (d) the flag-flip-to-ON itself is a separate PR which must independently hit 100. **Scope reduction is the honest path** to 100; caveat-shipping is not.

**Calibration anchors** (use these so the score doesn't drift across sessions):

| Score | Meaning | Action |
|-------|---------|--------|
| 100   | The PR's claimed scope is complete with no gaps. Smoke passes. Tier B confirmed. | **Merges.** |
| 90–99 | Implementer thinks it's complete; Tier B found something specific. | **Does not merge.** Back to Tier A: close the gap OR reduce the scope claim. |
| 70–89 | Real gaps remain in the claimed scope. | **Does not merge.** Reduce scope to a slice that hits 100, ship that, defer the rest to a follow-up PR. |
| 50–69 | Substantial work remaining. | **Does not merge.** Visible-means-verified (Rule 2) prohibits exposing it via UI/API/docs. May land as scaffold with explicit `# stub` markers if the PR's claimed scope is "scaffold only." |
| 0–49  | Draft or research spike. | **Does not merge.** Mark as such; do not request review-as-feature. |

**Critical**: there is **no row** in this table that says "ships with caveats." That row is what v3.1 deleted. The only path to merge is `BHS_OFFICIAL = 100`. Everything else is "go back and either close the gap or reduce the scope."

**Honesty enforcement**:
- A self-score above 95 with a one-line justification is **automatically suspicious** — Tier B will downgrade it to ≤95 unless the justification names a specific edge that was tested and passed.
- A `BHS_SELF_DRAFT - BHS_TIER_B` gap of more than 5 points triggers an automatic L4 score-gaming disclosure in the PR body (Tier B will add it).
- Score-gaming is itself an L4 lie. Repeat patterns surface at Tier C; the operator can decide to require that specific implementer agent type submit only `BHS_SELF_DRAFT = 0` for the next N PRs and let Tier B set the score from scratch.
- **The merge gate is binary**: 100 or no-merge. There is no boundary justification because there is no boundary — anything below 100 is "back to Tier A" and never reaches the merge button.

**Severity caps (Tier B downgrade rules — automatic, not negotiated)**:

| Tier B severity finding | Cap on `BHS_TIER_B` | Examples |
|-------------------------|---------------------|----------|
| `none` (no gaps found) | no cap (may be 100) | Tier B tried to disprove and failed; reported what they checked |
| `cosmetic` | no cap (may be 100) | Typo in docstring, formatting nit, suggestion that doesn't change behavior |
| `important` | ≤ 90 | Missing test coverage on a non-critical path, undocumented config, performance regression < 10%, Rule 5 smoke command suboptimal but works |
| `critical` | ≤ 70 | Security vulnerability, data corruption risk, false-completion claim in a load-bearing surface, smoke command does not run, Tier B identity = Tier A identity |

The cap applies **regardless of other findings**. Tier B records the highest-severity finding in `BHS_TIER_B_SEVERITY:` and caps `BHS_TIER_B` accordingly. Implementer cannot argue the cap; the cap exists so a 95 can't be silently issued for a critical-severity issue.

**Tier B independence enforcement**:
- `BHS_SELF_DRAFT_AGENT:` and `BHS_TIER_B_AGENT:` MUST be different agents/sessions. Same agent self-attesting independence is L4 score-gaming.
- If they are the same: the §4 validator script (`scripts/validate_pr_brutal_honesty.py`) FAILS the PR with expected `BHS_OFFICIAL = 0` (it does not silently rewrite the field — that would itself be L4 score-gaming-by-tooling). The PR author must either re-run with a genuinely independent Tier B reviewer, or correct the field to 0 and accept the merge gate failure.
- "Different sub-agent in the same parent session" is acceptable IF the sub-agent receives no context from the implementer beyond the diff + brutal-honesty section + EVIDENCE + smoke-path command + this rulebook. (See Rule 4.)
- "Same agent, different turn" is NOT acceptable. The agent has the prior context.

### 6.3 Carry-forward to `next-session.md`

Most repos already have a session-handoff file. In OCR_LOCAL it is `docs/next-session.md`. If your repo doesn't have one, **create it** with this minimum schema:

```markdown
# Next Session

## Block flag

<one of:
  "CLEAR — Carried Debt count = 0; next session may take new feature work."
  "BLOCKED — Carried Debt items survived a full cycle (TTL expired). New feature work is FORBIDDEN until Carried Debt count returns to 0. The first agent of this session must read this flag before opening any task."
>

## Carried Debt (TTL = 1 cycle; expired items flip block flag to BLOCKED)

| ID | Item | Source | TTL | Blocking | Status |
|----|------|--------|-----|----------|--------|
| CD-001 | <smallest concrete remaining gap> | PR #<N> or planning doc § | `1 cycle` for first-cycle items; `expired` for items that have survived a full cycle | YES / NO (does this gap forbid new feature work?) | `OPEN — first cycle` / `OPEN — expired (BLOCKED)` / `**CLOSED** by PR #<M> (one-line evidence)` |
| ...    | ...                              | ...                       | ...                                                                                  | ...                                                | ... |

(Required columns: `ID`, `Item`, `Source`, `TTL`, `Blocking`, `Status`. The `Status` column is what `scripts/check_block_flag.py` filters on — rows whose Status starts with `CLOSED` (case-insensitive, ignoring leading markdown bold markers) are historical record and NOT counted as active debt. A table without a `Status` column falls back to "count every non-placeholder row" which is correct for first-time installs but loses the ability to retain CLOSED rows as audit trail.

Empty table = CLEAR. Any OPEN rows past their first cycle = BLOCKED.)

## Deferred Scope (mirrors PR `DEFERRED_SCOPE:` lines; ≥25% reductions also appear in Carried Debt)

| ID | Original lane | Deferred portion | From PR | Target cycle | Owner |
|----|---------------|------------------|---------|--------------|-------|
| DS-001 | <feature name> | <what was cut from this PR> | #<N> | YYYY-MM-DD | <agent type or operator> |
| ... | ... | ... | ... | ... | ... |

(Silent scope reduction is an L4 lie. If a PR's diff doesn't match its title/description scope, the missing piece MUST appear here OR in Carried Debt OR both.)

## Aggregate BHS trend (system-level, set by operator or fresh adversarial agent)

| Cycle date | Aggregate BHS | Trend vs prior | Carried Debt count at cycle end | Deferred Scope count at cycle end |
|------------|---------------|----------------|---------------------------------|-----------------------------------|
| ...        | ...           | ▲/▼/=          | ...                             | ... |

## Operator overrides log (Tier B merge-gate bypasses)

| PR | Date | BHS_OFFICIAL at merge | Reason | Operator name | Co-signer or out-of-band ref (REQUIRED if block flag was BLOCKED) | Carried-debt entry created? |
|----|------|----------------------|--------|---------------|------------------------------------------------------------------|----------------------------|
| ... | ...  | ...                  | ...    | ...           | <name OR Slack URL OR signed email timestamp OR "n/a — flag was CLEAR"> | yes / no |
```

The block flag is the **first thing** the next session's first agent reads. Reading it is non-negotiable. If the flag is BLOCKED and the agent opens new feature work anyway, that is itself a §1 L9 lie (doc-as-implementation: ignored the convention the doc requires). The only allowed work in a BLOCKED session is draining the Carried Debt table to empty.

**TTL mechanics in plain English**:
- A debt item enters the table with TTL = 1 cycle.
- During the next cycle, the item is the **first priority**.
- At the END of that cycle, if the item is still there, the block flag flips to BLOCKED for the cycle after that.
- The item's TTL does NOT reset by being "in progress" — only by being closed (removed from the table).
- There is no "we'll get to it next sprint." There is no two-cycle slop. There is no soft warning. The flag flips, and the next session is debt-only until the table is empty.

**Definition of "cycle"** (made quantitative so TTL math is unambiguous):

> **A cycle = ONE operator-initiated session OR 5 calendar days from the prior `next-session.md` write, whichever comes first.**

- "Operator-initiated session" = a fresh agent/session opened by the operator that reads `next-session.md` as part of its first action. Sub-agents spawned within that session do not count as separate cycles.
- The 5-calendar-day cap exists so a project that goes idle for two weeks does not get a free pass on accumulated debt — the block flag still flips on day 6.
- The cycle clock starts at the **timestamp of the prior `next-session.md` commit**, not at "when work resumes." Going idle does not pause TTL.
- If two operator-initiated sessions occur within 24 hours, they count as ONE cycle (operator using a follow-up session to finish what was started). This prevents the gaming pattern of "open-and-close fake sessions to burn down TTL."

**Scope-reduction tracking** (closes the renamed-loophole risk):

If a PR's scope is reduced during Tier A (per §6.1 step 6a — "ship the scaffold, defer the flag-ON path"), the deferred portion MUST be tracked. Two places:

1. **In the PR body's `DEFERRED_SCOPE:` line** (§4 template) — list each piece of deferred scope with owner and target cycle.
2. **In the `Carried Debt` table above** with TTL = 1 cycle — same item, different surface.

Scope reductions ≥ 25% of the original lane size MUST appear in both. Reductions < 25% must appear at least in `DEFERRED_SCOPE:`. Silent scope reduction (reducing scope without recording it) is itself an L4 lie — Tier B is required to compare the original PR title/description against the diff and flag any unrecorded reductions as critical-severity (auto-caps `BHS_TIER_B` at 70).

**The renaming loophole**: without this rule, an implementer could rename the lane mid-PR ("we never claimed to ship feature X, only the scaffold of feature X") and zero-out their disclosure obligation. The `DEFERRED_SCOPE:` line plus the Carried Debt mirror prevent that — the deferred work has to be tracked under SOME name in SOME place.

**Operator override structural barrier** (closes the chain-override loophole):

The operator override is the explicit escape hatch (see §6.1 Tier B). Three escalating barriers based on system state:

| State | Override requirement | Rationale |
|-------|---------------------|-----------|
| Block flag = CLEAR, BHS_OFFICIAL ≥ 90 | `OPERATOR_OVERRIDE: <reason, name, timestamp, out-of-band reference>` (4 comma-separated fields, enforced by validator R8). At CLEAR+BHS≥90 the `out-of-band reference` field MAY be `n/a — flag was CLEAR` but MUST be present. Carried-debt entry created. | Logged, not authorized. Operator accepts known small gap. |
| Block flag = CLEAR, BHS_OFFICIAL < 90 | Same 4-field structure as above, AND Tier B severity must be `important` or lower (NOT `critical`). | Critical-severity findings cannot be overridden under CLEAR; force back to Tier A. |
| Block flag = BLOCKED | Same 4-field structure, AND the `out-of-band reference` field MUST be one of: (a) a real co-signer name (not the operator's own name a second time), OR (b) a verifiable artifact (Slack thread URL, signed email timestamp, calendar entry). The string `n/a` is FORBIDDEN at BLOCKED. | Operator-attested overrides at BLOCKED are themselves L4-lie targets. Single-name attestation cannot be cross-checked. |

The `scripts/validate_pr_brutal_honesty.py` validator enforces the structural format of the override line. It cannot enforce that the co-signer or out-of-band reference is genuine — that is the operator's audit responsibility — but it CAN enforce that the line includes the required fields. Override line missing required fields → validator fails the PR with expected `BHS_OFFICIAL = 0`; merge cannot proceed until the format is corrected.

### 6.4 Learnings adaptation (the loop modifies the rulebook)

The rulebook is **not static**. Each remediation cycle should produce zero, one, or rarely two adaptations to **this document**:

- A new failure pattern caught that doesn't fit the existing L-rows → propose a new L-row in §1 (with the file:line evidence).
- A trigger phrase that reliably worked in this cycle's loop → add to §3 (no more than one per cycle to avoid bloat).
- A score-gaming pattern caught → add a guard to §6.2.
- A spiral or wasted-loop pattern caught → add a guard to §6.1.

**The §5 anti-overhead clause still applies**: the rulebook does not grow without shrinking elsewhere. If you add an L-row, retire a row that hasn't been quoted in 6 months. The rulebook should stay readable in 9 minutes; if it crosses 12, prune.

Adaptations are made in the same PR as the cycle's last fix. They are NOT a separate workstream. The whole point is that the rules mature **from the team's own experience**, not from external authority.

### 6.5 Why this is a meta-component, not a new layer of process

The loop is **already implicit** in good engineering practice — humans call it "retrospective" or "the Friday meeting" or "tech-debt week." The rulebook just **names it** and **enforces a cadence** so it doesn't get skipped when context bloats or when the next sprint feels urgent.

It does not require:
- A new agent type
- A new MCP server or skill registration
- A new dashboard
- A new CI workflow
- A new approval gate

It requires:
- The brutal-honesty section in every PR (Rule 3 — already required)
- Reading and updating one file (`docs/next-session.md`) at session boundaries
- One adaptation entry to this rulebook per cycle, when warranted
- Honest scoring (BHS) that the operator can spot-check

The reason the loop lives **in the rulebook** rather than in a skill or MCP server is **deliberate**: skills can fail to load, agents can spin up without MCP context, and a new fresh agent in a bloated session may never invoke the skill. The rulebook is in `CLAUDE.md`'s load-bearing reference — every agent reads it on session start without needing to discover anything. **The convention IS the meta-component.** Don't duplicate it into a skill; that just creates two places where the truth might be wrong.

---

## 7. Optional: harness enforcement (Claude Code Stop hook)

If you want the harness to refuse to mark a turn complete when the implementer claimed "done" without a brutal-honesty section + EVIDENCE + SMOKE, add a `Stop` hook in `.claude/settings.json`. Sketch:

```json
{
  "hooks": {
    "Stop": [
      {
        "matcher": "*",
        "hooks": [
          {
            "type": "command",
            "command": "python .claude/hooks/check_brutal_honesty.py"
          }
        ]
      }
    ]
  }
}
```

The script checks: if the most recent commit message or open PR body contains "complete", "done", "ship", "finished", "ready" without a `## Brutal Honesty` section that includes `EVIDENCE:` and `SMOKE:` lines, exit non-zero with a message. The agent then has to add the section before the turn can stop.

This is **opt-in**. It is friction. It will fire false positives. It will catch real lies.

---

## 8. Operator escalation patterns

When you (the operator) suspect you are being lied to:

1. **Open the PR diff and read it.** Not the summary, the diff. The lie lives in the diff in 95% of cases.
2. **Run the smoke path on a fresh checkout** (or a worktree). If it doesn't run, the PR is not complete — regardless of what was claimed.
3. **Ask a fresh agent**: *"Read this PR diff and the brutal-honesty section. Try to disprove completion. If you can't disprove, name the gap between what the section claims and what the diff shows."*
4. **Quote the lie taxonomy by number** when calling out a failure: "This is L1 with a side of L12." Naming it forces the agent to recognize the pattern, not argue the specifics.
5. **If the agent argues**: ask *"Walk me through the production code path step by step from the entry point. Not the test. The production path."* — they cannot fake this without writing fake code into the response, and they will usually concede instead.
6. **If the agent claims done with no EVIDENCE**: ask *"Show me runtime evidence per Rule 1. A test passing is not evidence."*
7. **If the smoke path is missing from the repo**: that is itself a Rule 5 violation — the repo is not release-ready until one exists.

---

## 9. What this rulebook does NOT promise

- It does not eliminate hallucination.
- It does not catch every lie.
- It does not replace human review.
- It does not work if the brutal-honesty section is itself lied in — it only works if the section is **checked against the diff** by a human or a second fresh agent (Rule 4).
- It does not work if it is not loaded into CLAUDE.md as a hard reference. A document the agent never reads is a document that does not exist.

What it does promise:

- It names the failure modes so they can be caught by pattern (§1 L1–L13).
- It defines what counts as evidence vs what doesn't (§0). This is the most important paragraph in the document.
- It forces the implementer to **write down**, in their own words, what they faked. A lie that is written down is much easier to catch than a lie that is implied by silence (§4 template).
- It gives the operator trigger phrases that reliably extract more honest output than "is this done?" (§3).
- It defines an **adversarial** verification mechanism (§Rule 4) that does not require new infrastructure.
- It gives every repo a single deterministic floor of "working" (§Rule 5).

---

## 10. Portable copy-paste block (drop into any repo's CLAUDE.md)

```markdown
## Brutal Honesty Convention

This repo follows the Brutal Honesty Rulebook (canonical source: `docs/conventions/brutal-honesty-rulebook.md`; portable summary below).

**Premise**: Assume every implementation/completion claim is false until independently proven by runtime evidence. Tests existing, routes existing, docs saying complete, PR text saying complete, agent claims, and even self-attested brutal-honesty sections are NOT evidence. Evidence is: command output from the production code path, real UI/API behavior, persistence/state mutation, artifact/replay/checkpoint surviving a fresh checkout, or an independent reviewer who tried to disprove and failed.

**Five hard rules**:
1. **Evidence rule.** Every PR with a "complete" claim includes `EVIDENCE:` and `SMOKE:` lines pointing at runtime evidence — not tests.
2. **Visible means verified.** Incomplete code may live in the repo; it may NOT be presented as a working feature (no UI surfacing, no API doc, no release-note implication, no roadmap tick).
3. **Mandatory PR brutal-honesty section.** Every PR body ends with a `## Brutal Honesty` section disclosing stubs, mocks, escape conditionals, broad-catch swallowing, partial implementations, untested production paths, and unverified completion claims with `file:line`. Empty answers must be justified.
4. **Adversarial cross-agent review.** A fresh sub-agent (no prior context) is given the diff + brutal-honesty section + EVIDENCE + smoke command + this rulebook, and is asked to **try to disprove** completion. Not verify — *disprove*.
5. **One deterministic smoke path per repo.** `scripts/smoke.sh` (or equivalent) is the release gate. **Two honest tiers per §1 Rule 5**: Floor (import + surface-check through production code paths — the v3.2 minimum) is acceptable IF the PR body's `SMOKE:` line names which tier was run AND any ceiling-tier gap appears as a Carried Debt entry. Ceiling (true end-to-end against a real fixture) is the target. Claiming "smoke passed" at ceiling-tier when only floor-tier was run is itself L4 partial-as-complete. If smoke fails, the release is not ready regardless of test count.

**Lie taxonomy** (quote by number in PR review):
L1 Scaffold-as-feature · L2 Conditional escape hatch · L3 Mock-ate-the-real-code · L4 Partial-with-claim-of-complete · L5 Test-as-truth · L6 Aggregated-claim drift · L7 Re-summarization decay · L8 Test that asserts the bug · L9 Doc-as-implementation · L10 Dependency phantom · L11 Broad-catch swallowing · L12 Status-permissive test

**Trigger phrases** that reliably extract honest output: *"Be brutally honest." · "What did you fake to get here?" · "If I ran this on a fresh checkout right now, what would actually fail?" · "Tell me the smallest concrete thing that does NOT work yet." · "Show me the runtime evidence, not the test." · "If I disable this feature behind a flag, what visible behavior changes?"*

**The Remediation Loop (meta-component, three tiers; v3.1 closed the carry-forward loophole, v3.2 added executable enforcement):**
- **Tier A — Per-PR loop**: 5 iterations max. Implementer self-scores `BHS_SELF_DRAFT` after each iteration. If self-score < 100 at iteration 5, the PR is **withdrawn or scope-reduced** — it does NOT ship at <100 hoping for carry-forward. Same gap surviving 2 iterations → escalate, don't loop more.
- **Tier B — Pre-merge adversarial pass (THE OFFICIAL SCORER)**: one fresh agent assigns `BHS_TIER_B` independently. `BHS_OFFICIAL = min(BHS_SELF_DRAFT, BHS_TIER_B)`. **Merge gate: only PRs with `BHS_OFFICIAL = 100` may merge.** Period. No "ships with caveats" path. Self-vs-Tier-B gap > 5 → automatic L4 score-gaming disclosure. Tier B independence is enforced by `BHS_SELF_DRAFT_AGENT:` and `BHS_TIER_B_AGENT:` differing — if they match, the validator FAILS the PR with expected `BHS_OFFICIAL = 0` (it does not silently rewrite the field).
- **Tier B severity caps**: critical finding caps `BHS_TIER_B` at 70; important caps at 90; cosmetic uncapped. Severity recorded in `BHS_TIER_B_SEVERITY:` line.
- **Tier C — Cross-PR remediation cycle**: every N PRs OR session boundary, aggregate disclosures into `docs/next-session.md` "Carried Debt" with TTL = 1 cycle on each item. Items not cleared in the next cycle flip the **block flag** to BLOCKED — the cycle after that forbids ALL new feature work until Carried Debt = 0. **No two-cycle slop.** No remediation-sprint vote. Block flag is automatic from the math.
- **Cycle definition**: ONE operator-initiated session OR 5 calendar days from prior `next-session.md` write, whichever comes first.
- **Scope reduction tracking**: scope reductions ≥25% of the original lane MUST appear in both `DEFERRED_SCOPE:` (PR body) AND Carried Debt (next-session.md). Silent scope reduction = L4 lie.
- **BHS scale (only 100 ships)**: 100 = merges · 90–99 = does not merge, back to Tier A · 70–89 = does not merge, reduce scope · 50–69 = does not merge, scaffold-only allowed · 0–49 = draft, not a PR. Scope reduction is the honest path to 100; caveat-shipping is not.
- **Operator override**: the operator can always type the merge command themselves. The rulebook makes the override visible (`OPERATOR_OVERRIDE:` line with reason + name + timestamp + out-of-band reference) and force-creates a top-priority carried-debt entry for the next cycle. Override at BLOCKED state additionally requires a co-signer name OR an out-of-band reference (Slack URL, signed email, calendar entry). The override is logged, not authorized.
- **Why it's a meta-component, not a new layer**: the rulebook lives in `CLAUDE.md` and is automatically in context every session. Skills/MCPs can fail to load; the convention can't. Don't duplicate it into a skill.

Speculating is itself a lie. *"I don't know"* is the correct answer when there is no evidence.
```

That block is the entire portable surface. Paste it into any other repo's `CLAUDE.md`. The full rulebook is the long-form reference.

---

## 11. Install prompt for OTHER repos (paste into a fresh agent session)

Use this prompt verbatim in another repo's session to install/update the convention there:

```
Install the Brutal Honesty Convention (v3.3 — v3.2 executable gates plus L13 and schema/prose/artifact drift validation) in this repo.

Steps (do NOT skip any):
1. Read the canonical Brutal Honesty Rulebook (`docs/conventions/brutal-honesty-rulebook.md` if a mirrored copy already exists locally; otherwise fetch from the source repository this convention was installed from). Confirm the version is at least v3.3, which adds: (a) the Remediation Loop meta-component (§6 — added in v3.1), (b) the hard 100/100 merge gate (v3.1), (c) Tier B (a fresh adversarial agent) as the official scorer (v3.1), (d) 1-cycle carry-forward TTL (v3.1), (e) `BHS_*_AGENT` independence enforcement (v3.2), (f) Tier B severity caps (v3.2), (g) `DEFERRED_SCOPE:` ≥25% tracking (v3.2), (h) quantitative cycle definition (v3.2), (i) executable validators in `scripts/` (v3.2), and (j) L13 + schema/prose/artifact drift validation (v3.3). If you are reading v3.0 or older, STOP — install the v3.3+ version, do not back-port the loopholes.
2. Create `docs/conventions/brutal-honesty-rulebook.md` in THIS repo with the SAME content (copy verbatim — do not reinterpret).
3. Edit this repo's `CLAUDE.md`: add a new section titled `## Brutal Honesty Convention (load-bearing — read first)` immediately after the file's purpose line and BEFORE any project-specific context. Paste the §10 portable block from the rulebook into this section.
4. Create `scripts/smoke.sh` (or document the equivalent) — ONE shell command that exercises a real flow through this repo's production code paths (not test fixtures). Per §1 Rule 5 two-tier framing, **floor-tier is acceptable at install time** (import + surface-check through production code paths — verifies the wiring exists and the bytecode loads, catches L1/L4-import/L10) IF the install PR's `SMOKE:` line names the tier ("floor-tier") AND the ceiling-tier gap (true end-to-end against a real fixture) is added as a Carried Debt entry in step 5's table. This is honest because: floor-tier is NOT a fake smoke (it does run production code), and the ceiling-tier gap is openly tracked rather than hidden. Ceiling-tier at install requires the operator to provide a representative input fixture; if you don't know what flow + fixture that is, STOP and ask the operator. Do NOT invent one. Do NOT claim ceiling-tier passed if you only ran floor-tier — that is L4 partial-as-complete.
5. Create `docs/next-session.md` (or this repo's equivalent — check first; many repos already have a session-handoff file) with the v3.2 minimum schema from §6.3:
   - `## Block flag` — set to "CLEAR — Carried Debt count = 0; next session may take new feature work."
   - `## Carried Debt (TTL = 1 cycle...)` — empty table at install time. Required columns (in order): `ID | Item | Source | TTL | Blocking | Status`. The `Status` column is REQUIRED — `scripts/check_block_flag.py` filters CLOSED rows out of the active-debt count by reading this column, and CLOSED rows are kept in the table as audit trail. Without a `Status` column the script falls back to count-every-row, which works but loses the audit trail.
   - `## Deferred Scope (mirrors PR DEFERRED_SCOPE: lines)` — empty table at install time
   - `## Aggregate BHS trend` — table with first row = today's date, Aggregate BHS = "baseline (install PR)", Carried Debt count at cycle end = 0, Deferred Scope count = 0
   - `## Operator overrides log` — empty table at install time, with `Co-signer or out-of-band ref (REQUIRED if block flag was BLOCKED)` column included
6. Copy the validators (or create equivalents): `scripts/validate_pr_brutal_honesty.py` (parses PR body, enforces §4 template, fails on missing/inconsistent BHS lines, fails on `BHS_SELF_DRAFT_AGENT == BHS_TIER_B_AGENT`) and `scripts/check_block_flag.py` (reads next-session.md, exits non-zero if Block flag = BLOCKED). Wire both into CI as required checks. Without these scripts, the v3.2 enforcement is documentation-only — that is acceptable for the install PR but MUST be tracked as Carried Debt.
7. Open a single PR titled `docs(conventions): install Brutal Honesty Rulebook v3.3 (hard 100/100 gate + L13 + schema drift validator)` containing only steps 2-6.
8. Fill out the PR body using the §4 template from the rulebook. For this install PR specifically:
   - EVIDENCE: "rulebook v3.3 file exists at documented path; CLAUDE.md edit visible in diff; smoke.sh exists or operator-acknowledged TBD; next-session.md exists with v3.2 minimum schema (block flag, Carried Debt, Deferred Scope, BHS trend, override log); validator scripts and v3.3 drift artifacts exist or operator-acknowledged TBD."
   - SMOKE: "not applicable — docs-only install PR; smoke path is created in step 4 for future PRs. If smoke.sh is also being created in this PR, paste the output here."
   - BHS_SELF_DRAFT: honest self-score. If you couldn't write smoke.sh OR the validator scripts, you cannot honestly draft 100 — reduce the PR's scope to a slice that hits 100 (e.g. ship the rulebook docs alone, defer smoke.sh + validators to follow-up PRs tracked as Carried Debt). DO NOT ship at <100 expecting carry-forward to catch it.
   - BHS_SELF_DRAFT_AGENT: <your session/agent identifier — required>
   - BHS_TIER_B: leave blank for now; spawn a SECOND fresh agent (different session, no prior context) and ask them to score this PR against the rulebook. They write BHS_TIER_B and a one-paragraph justification. If their score is < 100, this PR returns to Tier A — either close the gap (e.g. write smoke.sh) or reduce the PR's scope.
   - BHS_TIER_B_AGENT: <Tier B reviewer's session/agent identifier — MUST differ from BHS_SELF_DRAFT_AGENT>
   - BHS_TIER_B_SEVERITY: <one of: none / cosmetic / important / critical>
   - BHS_OFFICIAL: min(BHS_SELF_DRAFT, BHS_TIER_B), with severity caps applied. Only merges at 100.
   - CARRY_FORWARD: list anything that cannot be in the install PR (e.g. "smoke.sh — operator action required to identify entry point" or "validator scripts — TBD next cycle"). Each becomes a TTL=1 carried-debt item in next-session.md.
   - DEFERRED_SCOPE: list any deferred scope (e.g. "validator scripts originally scoped here, deferred to PR #N+1, target cycle YYYY-MM-DD").
   - LOOP_ITERATIONS: usually 1 for an install PR.
   - OPERATOR_OVERRIDE: empty (this PR should hit 100 honestly via scope-reduction; if you find yourself asking for an override on the install PR, you are shipping the lie this convention exists to prevent — STOP and reduce scope instead).
9. **Hard merge gate from this PR forward**: every PR body in this repo MUST include the §4 template AND ALL of: `BHS_SELF_DRAFT`, `BHS_SELF_DRAFT_AGENT`, `BHS_TIER_B`, `BHS_TIER_B_AGENT`, `BHS_TIER_B_SEVERITY`, `BHS_OFFICIAL`, `CARRY_FORWARD`, `DEFERRED_SCOPE`, `LOOP_ITERATIONS`, `OPERATOR_OVERRIDE` lines. PRs with `BHS_OFFICIAL < 100` do not merge unless `OPERATOR_OVERRIDE:` is present (and at BLOCKED state, the override line must include a co-signer name OR an out-of-band reference). Tier B is a SECOND fresh agent (not the implementer). Self-scoring above 95 with a one-line justification is auto-suspicious. `BHS_SELF_DRAFT_AGENT == BHS_TIER_B_AGENT` causes the validator to FAIL the PR with expected `BHS_OFFICIAL = 0` (validator does not silently rewrite the field — that would itself be L4 score-gaming-by-tooling).
10. **Every session starts by reading `docs/next-session.md` "Block flag" FIRST** — before opening any task. If BLOCKED, the only allowed work is draining the Carried Debt table to empty. Ignoring the flag is itself an L9 lie. Run `python scripts/check_block_flag.py` if installed; otherwise read the file directly.
11. Do NOT add tests beyond what the validators need. Do NOT add a CI gate beyond `validate_pr_brutal_honesty.py` and `check_block_flag.py`. Do NOT propose new agent types. Do NOT add MCP servers or skill registrations beyond the EVOKORE-MCP integration if your project uses it. Do NOT add a feature inventory schema unless the operator explicitly asks. The rulebook itself forbids these (§5 anti-overhead and §6.5).
12. Be brutally honest in your PR body about anything you couldn't do. Note: with v3.2's hard gate, "couldn't do X" usually means **you must reduce the PR's scope** to a slice that doesn't depend on X (track the cut as `DEFERRED_SCOPE:` ≥25% per §6.3). Caveat-shipping is the failure mode v3.1 deleted; silent scope reduction is the failure mode v3.2 deletes.

If you cannot complete any of these steps, STOP and report what blocked you. Do not work around it. Do not fake it. Do not skip ahead. The whole point of this convention is that working around it is the failure mode it exists to prevent.

After install, the loop is now live in this repo:
- Tier A: every PR has a 5-iteration self-loop ending in a BHS_SELF_DRAFT.
- Tier B: every PR pre-merge has a fresh-agent adversarial pass that assigns BHS_TIER_B (the official score). Only BHS_OFFICIAL = 100 merges.
- Tier C: every session starts by reading next-session.md "Block flag" — BLOCKED sessions are debt-only until Carried Debt = 0.
- The rulebook itself adapts (§6.4) — when a new failure pattern is caught that doesn't fit the existing L-rows, propose the next L-row in the same PR as the cycle's last fix.
```

---

## 12. Changelog

- **v3.3 (2026-05-11)**: Added L13 (soft-prose-claimed-as-mechanical) and the v3.3 schema/prose/artifact enforcement layer. New files include `docs/conventions/brutal-honesty-kit/v3.3-architecture.md`, `docs/conventions/brutal-honesty-kit/v3.3/{enums,schemas,tables}/`, `scripts/validate_v33_schema_drift.py`, and validator fixture tests. The validator fails on prose-vs-artifact drift, JSONL example drift, schema-vs-enum drift, and malformed artifacts. This release does not ship the v3.4 orchestrator; the architecture document outlines it as roadmap material.
- **v3.2 (2026-05-09)**: Closed the eight Tier B-flagged gaps surfaced by v3.1's first adversarial pass (`BHS_TIER_B = 62/100` on PR #833). Six structural changes:
  1. **Tier B independence enforcement** (§4 template + §6.2). New `BHS_SELF_DRAFT_AGENT:` and `BHS_TIER_B_AGENT:` lines in PR body. The validator script FAILS the PR with expected `BHS_OFFICIAL = 0` if the two agents are the same (it does not silently rewrite — silent rewrite would itself be score-gaming-by-tooling). Closes the "agent self-attests independence" loophole.
  2. **Severity caps on Tier B scores** (§6.2 table + new `BHS_TIER_B_SEVERITY:` line). Critical caps `BHS_TIER_B` at 70; important caps at 90; cosmetic uncapped. Severity is recorded so a 95 cannot be silently issued for a critical-severity issue.
  3. **`DEFERRED_SCOPE:` tracking with ≥25% threshold** (§4 template + §6.3 schema). New required PR-body line. Scope reductions ≥25% of the original lane MUST appear in both `DEFERRED_SCOPE:` AND Carried Debt — closes the renamed-loophole risk where a PR could re-define its claimed scope mid-flight.
  4. **Quantitative cycle definition** (§6.3). "Cycle = ONE operator-initiated session OR 5 calendar days from the prior `next-session.md` write, whichever comes first." Closes the ambiguity that let TTL-1 effectively mean "indefinite" if the operator deferred the next session.
  5. **Override structural barrier at BLOCKED state** (§6.3 table). At BLOCKED, the `OPERATOR_OVERRIDE:` line additionally requires a co-signer name OR an out-of-band reference (Slack URL, signed email, calendar entry timestamp). Single-name override at BLOCKED is itself an L4 lie target — closes the "operator can serially override their way out of any debt" loophole.
  6. **Executable validators** (`scripts/validate_pr_brutal_honesty.py` + `scripts/check_block_flag.py`). Without these, "automatic" claims in v3.1 were L1 lies (the rulebook claimed enforcement that did not exist). v3.2 ships the validators alongside the rulebook so the §4 template, severity caps, agent-identity check, and block flag are actually enforced in CI — not just documented.
  Also added: `scripts/smoke.sh` for OCR_LOCAL specifically (closes Rule 5 violation that v3.1 inherited); §11 install prompt extended to require validator scripts in step 6 (was steps 1-11, now 1-12); L2 description tightened to distinguish escape conditionals from legitimate guard clauses (gemini-code-assist feedback on PR #833); `return ""` grep example expanded to match both quote styles; `OCR_LOCAL` reference in §11 replaced with portable language. Section count unchanged at 12. Read time grew 10 → 11 minutes.
- **v3.1 (2026-05-09)**: Closed the carry-forward loophole. Three changes that interlock:
  1. **Hard 100/100 merge gate** (§6.1 Tier A step 6, §6.2 calibration table). v3 allowed shipping at any BHS ≥ 70 "with disclosed caveats." v3.1 deletes that path: only `BHS_OFFICIAL = 100` may merge. PRs scoring below 100 must close the gap OR reduce the PR's claimed scope to a slice that hits 100. **Scope reduction is the honest path; caveat-shipping is the lie this version exists to prevent.**
  2. **Tier B becomes the official scorer** (§6.1 Tier B). v3 had the implementer self-assign BHS, with Tier B as advisory. v3.1 inverts that: implementer assigns `BHS_SELF_DRAFT` (DRAFT only); Tier B (a fresh adversarial agent) assigns `BHS_TIER_B`; `BHS_OFFICIAL = min(BHS_SELF_DRAFT, BHS_TIER_B)`. Self-vs-Tier-B gap > 5 points triggers automatic L4 score-gaming disclosure. **Self-scoring is forbidden as the merge authority.**
  3. **1-cycle carry-forward TTL with automatic block flag** (§6.1 Tier C, §6.3 schema). v3 allowed two consecutive cycles of debt outpacing remediation before forcing a debt-only sprint. v3.1 caps each carried-debt item at TTL = 1 cycle. Items not cleared in the next cycle automatically flip the **block flag** in `docs/next-session.md` to BLOCKED — the cycle after that forbids ALL new feature work until Carried Debt = 0. **No two-cycle slop. No remediation-sprint vote. The flag flips from the math.**
  Also added: explicit `OPERATOR_OVERRIDE:` mechanism (the operator can always type the merge command themselves; the rulebook makes the override visible and force-creates a top-priority carried-debt entry); §6.3 schema now includes `Block flag`, TTL column on Carried Debt table, `Operator overrides log` section; §4 PR template extended with `BHS_SELF_DRAFT` / `BHS_TIER_B` / `BHS_OFFICIAL` / `OPERATOR_OVERRIDE` lines (was a single `BHS:` line); §10 portable block + §11 install prompt updated for the harder gate. Section count unchanged at 12. Read time grew 9 → 10 minutes.
- **v3 (2026-05-09)**: Reframed as a meta-component, not a hard PR gate. Added §6 The Remediation Loop with three tiers: Tier A per-PR loop (5 iterations or BHS = 100), Tier B pre-merge adversarial pass (Rule 4, single shot — not another loop), Tier C cross-PR remediation cycle (aggregate disclosures into `docs/next-session.md` at session boundary; next session's first priority is carried debt). Introduced the **Brutal Honesty Score (BHS)** with calibration anchors (§6.2). Added carry-forward mechanism via `docs/next-session.md` minimum schema (§6.3). Added the **learnings adaptation** clause (§6.4) so the rulebook matures from team experience. Added §6.5 explaining why the loop is a meta-component (lives in CLAUDE.md, not in skills/MCPs that can fail to load). Extended §4 PR template with `BHS:`, `CARRY_FORWARD:`, and `LOOP_ITERATIONS:` lines. Updated §5 anti-overhead with explicit "loop is not new infrastructure" clause. Updated §10 portable block with three-tier summary. Updated §11 install prompt with loop-aware steps (5→9). Section count grew from 11 → 12 (added §6 Remediation Loop). Read time grew from 7 → 9 minutes; pruning candidates flagged for v4 if anything in §1 hasn't been quoted within 6 months.
- **v2 (2026-05-09)**: Consolidated parallel rulebook from another project session. Added §0 evidence rule (with explicit lists of what counts and what doesn't), Rule 2 (visible means verified), Rule 5 (one deterministic smoke path per repo), L11 (broad-catch swallowing), L12 (status-permissive test), §10 install prompt for other repos. Sharpened Rule 4 to "adversarial — try to disprove" framing. Extended §3 trigger phrases (+2). Extended §4 PR template with broad-try/except disclosure, exact-status-assertion check, and `EVIDENCE:` + `SMOKE:` lines. Section count grew from 10 → 11 (added §10 install prompt); §5 anti-overhead clause now explicitly calls out optional feature-inventory rows as opt-in only with a ≤4-column cap.
- **v1 (2026-05-09 earlier)**: Initial version. §1 lie taxonomy L1–L10, 5 hard rules, trigger phrases, PR template, anti-overhead clauses, Stop-hook sketch, operator escalation patterns, portable copy-paste block.
