# Lesser fix-agent playbook

Each finding file is intended to be executable cold. If source state contradicts its recorded SHA/lines, stop and report drift; do not improvise on a different branch.

Cycle control is `08-cycle-control.md`; `04-master-backlog.md` is authoritative for owner, exact target, status, verification ID, and dependencies. Only a packet whose **status line** says `ready-for-fix-agent` may be assigned cold. Every `blocked` packet must first receive the named operator/contract/harness decision and an updated exact target; prose mentioning a future ready state does not change its status.

## Mandatory intake

1. Read `CLAUDE.md`, `docs/CLAUDE.md`, `docs/next-session.md`, this playbook, the assigned finding, and its dependencies.
2. Record `git remote -v`, branch, exact HEAD, upstream, `git status --porcelain=v2 --branch`, and worktree path before editing.
3. Confirm whether the target is a public PR head, clean replacement worktree, or dirty owner lane. Never treat local uncommitted bytes as PR evidence.
4. Copy the finding's defect probe before changing code. If it does not reproduce, label `EVIDENCE_DRIFT` with command/output; do not declare fixed.
5. Work only on the finding's named files and atomic contract. Preserve all unrelated dirt; never bulk stage, reset, clean, restore, delete, force-push, retarget, close, or merge without operator authority.

## Fix loop (maximum five Tier A iterations per slice)

1. **Disprove first:** run the failing production-path probe and capture the exact failure.
2. **Make the smallest contract repair:** no adjacent rewrite, formatting sweep, dependency upgrade, or public claim edit.
3. **Run AC1–AC3:** each must have a mechanical check, not a prose statement.
4. **Run focused regressions, then canonical smoke:** name skipped/unavailable gates exactly. Tests do not substitute for browser, process, concurrency, publication, or provenance probes.
5. **Inspect the diff and fresh checkout:** `git diff --check`; intended file manifest; build/install/runtime from immutable candidate when required.
6. **Tier A self-score:** list `BHS_SELF_DRAFT`, agent identity, severity, iteration, carry-forward/deferred scope, and exact evidence. Only 100 is a merge candidate.
7. Same failure set twice is a plateau: stop looping and escalate with the two artifacts. At iteration five below 100, scope-reduce/withdraw; never “ship with caveats.”

## Required handoff packet

- finding ID and exact candidate SHA;
- before/after defect probe outputs;
- AC1–AC3 result table and commands;
- focused tests, production-path evidence, full/canonical gates, and every SKIPPED_GATE;
- `git status` and intended-file manifest proving unrelated dirt was preserved;
- remaining risks/dependencies and no broader completion claim;
- Tier A fields and a request for a fresh different-agent Tier B.

## Slice-specific traps

- **PR truth:** closing/replacing PRs is operator-only. Preserve rejected SHAs and do not force-push history into a false continuity claim.
- **RHPC:** source SHA must enter from a trusted authorization outside produced artifacts; snapshot/hash/verify the same bytes; absent-target publication needs atomic no-replace. Test concurrent hostile mutations.
- **Dashboard:** handler units are insufficient. Use a real browser against a real ephemeral server. Test token lifecycle, 401/500, valid empty, stale data, hostile artifact strings, and keyboard tabs. Cover both clients.
- **State/registry:** use nested, exception, concurrency, aliasing, and mutation probes. A copy test must mutate nested arrays/objects and compare authoritative hashes.
- **Artifacts/reports:** presence is not provenance. Require schema version, candidate SHA/config, generated time, lifecycle, and atomic publication.
- **BHS migration:** upgrade rulebook, enums, schema, prose, validator, fixtures, PR template, and workflow as one cohort; preserve canonical L6 retirement and legacy-record mapping.

## Canonical BHS v3.7.1 taxonomy to quote in reviews

| ID | Canonical label / operational reading |
|---|---|
| L1 | Scaffold-as-feature — infrastructure or shape presented as delivered behavior. |
| L2 | Conditional escape hatch — unsupported condition silently weakens the claimed path. |
| L3 | Mock-ate-the-real-code — mock path replaces the production dependency being claimed. |
| L4 | Partial-with-claim-of-complete — includes aggregated completion drift from older L6 usage. |
| L5 | Test-as-truth — test existence/pass substituted for production-path evidence. |
| L6 | Retired — do not assign to new findings; map historical aggregated drift to L4. |
| L7 | Re-summarization decay — later summaries erase material limits/failures. |
| L8 | Test that asserts the bug — regression encodes incorrect behavior as success. |
| L9 | Doc-as-implementation — prose or command stands in for missing mechanism. |
| L10 | Dependency phantom — wiring/claims depend on absent or undurable implementation. |
| L11 | Broad-catch swallowing — failure is caught and hidden or converted to false success. |
| L12 | Status-permissive test — broad status acceptance lets incorrect response pass. |
| L13 | Soft-prose-claimed-as-mechanical — prose says a gate exists when enforcement differs/does not. |
| L14 | Unit-substitution-as-completion — a unit substitute is presented as integration/system completion. |
| L15 | Gate-skipped-via-missing-config — absent configuration turns a required gate into a skip/pass. |
| L16 | Wholesale-rewrite-as-edit — broad replacement is framed as a bounded edit, hiding scope/risk. |

Authority: canonical [BHS v3.7.1 rulebook](https://github.com/mattmre/Brutal-Honesty-Kit/blob/main/v3.5/rulebook/brutal-honesty-rulebook.md). The descriptions above are audit-operational paraphrases; reviewers should open the authority before scoring.
