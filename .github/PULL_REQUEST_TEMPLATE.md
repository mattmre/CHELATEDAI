<!--
  Brutal Honesty Rulebook v3.3 §4 PR body template.

  Copy this whole file as your PR body. Replace every <placeholder>. Empty
  fields are NOT acceptable — write `none` or `n/a — <reason>` and let the
  Tier B reviewer judge whether the justification is honest.

  scripts/validate_pr_brutal_honesty.py parses this template structurally —
  do not rename fields, do not change the colon format, do not add blockquote
  prefixes. The validator IS the merge gate.
-->

## Summary

<1–3 sentences. What does this PR do? Cite the rulebook section / planning
doc / Carried Debt entry it closes.>

## Test plan

- [ ] <test 1>
- [ ] <test 2>
- [ ] <smoke command and expected exit code>

---

## Brutal Honesty (rulebook v3.3 §4)

EVIDENCE: <1–3 lines pointing at runtime evidence. Command output, file path,
URL — NOT a test name. Test existing is not evidence. Example:
"`python scripts/smoke_pipeline.py` exits 0; output asserts 'invoice_total'
extracted from tests/fixtures/invoice.pdf at line 47">

SMOKE: <command + tier ran. Examples:
"`bash scripts/smoke.sh` PASS, floor tier (ceiling deferred — see CARRY_FORWARD)"
or "`bash scripts/smoke.sh` PASS, ceiling tier against tests/fixtures/sample.pdf">

BHS_SELF_DRAFT: <0–100, integer. Implementer's self-score for this PR.>
BHS_SELF_DRAFT_AGENT: <agent identifier, e.g. "claude-opus-4-7@2026-05-10T14:32Z"
or operator name. MUST differ from BHS_TIER_B_AGENT or BHS_OFFICIAL forces to 0.>

BHS_TIER_B: <0–100, integer. Independent adversarial scorer.>
BHS_TIER_B_AGENT: <agent identifier, fresh sub-agent NOT used for implementation.>
BHS_TIER_B_SEVERITY: <none | cosmetic | important | critical. Severity of the
worst issue Tier B found. Caps Tier B score: critical ≤ 70, important ≤ 90,
cosmetic / none uncapped.>

BHS_OFFICIAL: <integer = min(BHS_SELF_DRAFT, capped(BHS_TIER_B)). Validator
recomputes; mismatch is a FAIL. Only 100 may merge without OPERATOR_OVERRIDE.>

CARRY_FORWARD: <comma-separated list of CD-### IDs added or extended by this
PR. Empty if none. Example: "CD-013, CD-014" or "none">

DEFERRED_SCOPE: <comma-separated list of DS-### IDs for scope items deferred
out of this PR (only required when ≥25% of original scope deferred). Empty
otherwise. Example: "DS-001 (CI workflow split out)" or "none">

LOOP_ITERATIONS: <1–5. How many Tier A iterations were run. If = 5 and
BHS_SELF_DRAFT < 100, this PR must be withdrawn or scope-reduced — not merged
hoping for carry-forward. State which path you chose in this section.>

OPERATOR_OVERRIDE: <write `none` if BHS_OFFICIAL = 100 — DO NOT leave this
blank or the validator's regex will greedily consume the next non-empty line.
Otherwise four comma-separated fields: reason, operator name, ISO-8601 timestamp
UTC, out-of-band reference (URL/ticket/Slack permalink). Example:
"hotfix for prod outage, alex, 2026-05-10T14:32:00Z, https://github.com/.../issues/999">

### Disclosures

<Use the lie taxonomy L1–L13 (rulebook §1). Quote by number. Empty answers
must be JUSTIFIED, not omitted. Example:
"L1 (scaffold-as-feature): none — production module exposes real entry point
verified by smoke_pipeline.py:124.
L2 (escape conditional): none.
L3 (mock in production path): see foo.py:47 — TODO replace stub call to
external API with real client; tracked as CD-015 (this PR).
L4 (partial-with-claim-of-complete): none.
L9 (doc-as-implementation): none.
L11 (broad-catch swallowing): see bar.py:88 — except Exception around
network call is intentional, falls through to retry; documented inline.
Other L1–L13: none.">

L1 (scaffold-as-feature): <none | file:line + brief description>
L2 (escape conditional): <none | ...>
L3 (mock in production path): <none | ...>
L4 (partial-with-claim-of-complete): <none | ...>
L5 (untested production path): <none | ...>
L8 (test that asserts the bug): <none | ...>
L9 (doc-as-implementation): <none | ...>
L11 (broad-catch swallowing): <none | ...>
L12 (untested error path): <none | ...>
Other L1–L13: <none | ...>
