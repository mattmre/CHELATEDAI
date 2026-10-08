# Round 2 lesser fix-agent playbook

This cycle is plan-only. A fix agent may begin only from a `ready-for-fix-agent` row in `04-master-backlog.md`, or after the operator records and resolves every dependency on a `blocked` row. Each finding is designed to stand alone; if its target SHA, dirty state, source lines, or defect probe no longer match, stop with `EVIDENCE_DRIFT` rather than adapting silently.

## Cold intake

1. Read repository guidance, `docs/next-session.md`, this playbook, `04-master-backlog.md`, the assigned finding, and every named dependency.
2. Record remote, worktree path, branch, exact HEAD/upstream, and full porcelain before editing. Preserve all unrelated dirty and untracked state.
3. Confirm the controlling target: root exact HEAD, dirty owner overlay, or a clean replacement for a retained PR. Never repair a superseded public head in place unless the operator explicitly selects it.
4. Run the packet's current defect probe before modification. Capture command, exit, and exact salient output. A passing unit module does not replace the defect probe.
5. Restrict edits to the packet's named atomic contract. Do not stage, commit, push, retarget, close, merge, reset, clean, delete, deploy, restart services, or change product claims without the corresponding authority.

## Maximum-five Tier A loop

1. Reproduce the current defect.
2. Make the smallest complete contract repair; do not mix adjacent cleanup or dependency upgrades.
3. Execute AC1–AC3 mechanically and preserve their outputs.
4. Run focused regressions, then the canonical repository gates appropriate to the changed path.
5. Test the actual boundary named by the packet: process terminal state, persistence failure, concurrency order, page selection, response discriminant, or vector/config invariant.
6. Inspect the exact diff and intended-file manifest. Prove unrelated dirt is unchanged.
7. Self-score with `BHS_SELF_DRAFT`, agent identity, severity, iteration, every skipped gate, deferred scope, and exact candidate SHA. Only 100 advances as shippable.
8. Two consecutive identical failure sets are a plateau: stop and escalate with both receipts. At five Tier A iterations below 100, reduce/withdraw scope; never replace evidence with a caveat.

## Required handoff

- finding ID, controlling baseline, exact candidate SHA, and intended-file manifest;
- before/after production-path probe outputs;
- AC1–AC3 table with executable commands and exact results;
- focused, full/canonical, browser/process/concurrency/persistence gates as applicable;
- explicit `SKIPPED_GATES`, remaining dependencies, and evidence boundary;
- final porcelain/diff showing unrelated user state preserved;
- Tier A score fields and request for a fresh different-agent Tier B review.

## Atomic repair cohorts

- **Checkpoint cohort:** REPO-001 and REPO-002 must share one no-overwrite identity plus durable catalog commit design. Validate same-name/frozen-time, persistence failure, restart restore, and concurrent creators.
- **AEP lifecycle:** REPO-003 requires typed candidate/verifier receipts; REPO-004 requires an execution boundary with revocable result authority. Do not solve either by status renaming or a longer timeout.
- **Dashboard history:** REPO-005 must preserve unreadable newest state as unknown/error across validation, campaign, preflight, and evidence loaders. The frontend contract must render that discriminant.
- **Dashboard client:** REPO-008 is an operator decision. Only after one canonical surface is selected may REPO-009/010 be fixed or closed as superseded. Add a checked-in served-page gate; loader/handler units are subordinate evidence.
- **Runtime configuration:** REPO-006 validates centroid batches against engine dimension before live assignment; REPO-007 validates all threshold fields/bounds before one atomic enable commit.
- **QSCCI supervisor:** WT-001 is blocked until service ownership defines recovery for unknown child/group state. No service restore may occur before terminal-and-reaped proof.
- **PR #257:** use a clean replacement target. Preserve the primary block exception and record secondary-hook failure without replacing it.

## BHS v3.7.1 L1–L16 taxonomy

| ID | Canonical operational reading |
|---|---|
| L1 | Scaffold-as-feature — infrastructure or shape is presented as delivered behavior. |
| L2 | Conditional escape hatch — an unsupported condition silently weakens the claimed path. |
| L3 | Mock-ate-the-real-code — a mock path replaces the production dependency being claimed. |
| L4 | Partial-with-claim-of-complete — incomplete work or aggregated drift is presented as complete. |
| L5 | Test-as-truth — test existence or pass substitutes for production-path evidence. |
| L6 | Retired — assign no new findings; historical aggregated drift maps to L4. |
| L7 | Re-summarization decay — later summaries erase material limits or failures. |
| L8 | Test that asserts the bug — a regression encodes incorrect behavior as success. |
| L9 | Doc-as-implementation — prose, status, or command stands in for a missing mechanism. |
| L10 | Dependency phantom — claimed wiring relies on absent or non-durable implementation. |
| L11 | Broad-catch swallowing — failure is hidden or converted into false success. |
| L12 | Status-permissive test — broad response acceptance lets incorrect behavior pass. |
| L13 | Soft-prose-claimed-as-mechanical — prose says a gate exists while enforcement differs or is absent. |
| L14 | Unit-substitution-as-completion — a unit substitute is presented as integration/system completion. |
| L15 | Gate-skipped-via-missing-config — missing configuration converts a required gate into skip/pass. |
| L16 | Wholesale-rewrite-as-edit — broad replacement is framed as a bounded change, hiding scope and risk. |

Authority: BHS v3.7.1 rulebook from the Brutal Honesty Kit `v3.5/` implementation path. These are operational paraphrases for packet review; the fresh scorer must open the authority and may remove any tag not supported by the packet's actual mechanism.

## Prompt wording for reliable agent execution

Describe ordinary FE/BE remediation as local contract verification: loading/empty/stale/error/success, process ownership, persistence, concurrency, validation, and component wiring. Use bounded malformed fixtures and local tests. Keep any separate security review explicitly scoped, owner-authorized, and isolated from general functional assignments. This wording preserves the audit while avoiding classifier ambiguity caused by broad discovery language combined with access-control and active network-probing terms.
