# Research publication and cross-machine audit handoff

**Date:** 2026-10-08. **Scope:** publish the new public research material requested by the user, preserve private/prospective fixture boundaries, and audit the laptop repository folder.

## Publication scope

The desktop checkout is `codex/prime-ring-onion-method-dev` at `65c9085cd048e8a7351a53e87666fdd5639e612b`. Its committed history is already published in draft [PR #296](https://github.com/mattmre/CHELATEDAI/pull/296). The new publication branch is `codex/publish-research-backlog-20261008`, based on that exact public tip; the PR is a draft stacked on the preservation branch.

The [inventory](publication-inventory-2026-10-08.json) accounts for 200 locally modified/untracked files at the preparation snapshot. The reviewed payload contains 165 of those files, approximately 15.1 MB, plus this handoff and its inventory:

- the [OpenAI math applicability review](openai-math-applicability-review-2026-10-07.md), 45-file external-source receipt, [MA0–MA7 task packet](innovation-test-queue-2026-09-12/math-findings-integration-2026-10-07.md), and [24 specified cases](innovation-test-queue-2026-09-12/math-test-cases-2026-10-07.csv);
- the existing local innovation queue, recurrent-design review, portfolio/schedule, and their updated issue #106 / next-session / prime-ring pointers;
- unpublished paired-intervention v2/v3 and QSCCI v4 source, focused tests, protocol/schema documents, and retained method-development archives;
- the local ARCH-AEP remediation planning/review records and fixed-graph one-pager;
- the curated `.codex-public-safe/` aggregate evidence and its verifier, with exact existing bytes preserved.

These are retained source/planning/evidence records. Publishing them does not rerun their campaigns, independently authenticate model-produced measurements, establish a new utility result, or override any existing debt, negative, protocol, or merge gate. Historical documents retain their original dates and evidence classifications.

## Material deliberately retained locally

The inventory includes a disposition and SHA-256 for every remaining file:

- 27 local tooling/private-context files, including the operator handoff that explicitly says not to publish;
- four prospective QSCCI fixture JSON files, including the all-partition authoring source and SELECT/REPORT partitions;
- four fixture-boundary authoring/freezing/validation test tools whose complete fixture dependency remains private.

The prospective fixture is described by its public review/draft contract and hash bindings; its texts/labels are not included in this publication payload. The user's broad publication request is recorded here without silently consuming the planned blind-evaluation boundary. Those fixtures are not deleted or changed. A later protocol decision can change their disclosure policy explicitly before any campaign uses them.

## Fresh publication validation

The [validation receipt](publication-validation-2026-10-08.json) records the exact checks. Staged-byte verification preserved 164 of 165 source files byte for byte; the remaining external-source receipt JSON differs only by Git's CRLF-to-LF metadata normalization. All frozen protocols/artifacts and verbatim imported sources retain their exact bytes. The staged 167-file payload, before adding the validation receipt itself, passed Gitleaks with zero findings and `git diff --cached --check` with zero findings.

- Integration audit: **42 tasks, eight MA tasks, 24 specified cases, 45 dependency edges, 128 resolving local links, zero structural errors**. The audit preserved all 23 pre-existing document bodies/files in the preparation snapshot.
- Focused code/archive suite: `python -m unittest tests.test_paired_intervention_experiments test_qscci_frozen_contract test_qscci_numerics test_qscci_selection_gates test_qscci_supervisor test_qscci_v4_archive` completed **144 tests, four skipped, zero failures** in 44.162 seconds. Skips are conditional platform/CUDA coverage, not positive evidence for those paths.
- Curated archive: `python .codex-public-safe/validate_artifacts.py` returned `PASS`, exact hashes/canonical JSON/closed schemas, zero forbidden privacy hits, and zero Gitleaks findings.
- Exact 166-file payload scan, before this newly written handoff: Gitleaks returned exit zero with **zero findings**. Final staged hygiene/scan is repeated only to cover the added handoff and publication metadata.
- `git diff --check` passed on tracked changes. The complete staged delta is checked before commit.
- `python scripts/check_block_flag.py` returned **BLOCKED**, three carried-debt rows, as expected. This is an honest draft preservation PR, not merge admission. No operator override or independent Tier-B score is claimed.

The new 24 mathematical cases remain `SPECIFIED_NOT_IMPLEMENTED`; the 144 executed tests concern the older retained code/archive payload. No Lean/Comparator proof, new model experiment, GPU campaign, or REPORT-based evaluation was executed for the math integration.

## Laptop audit status

The online `MRE-LAPTOP` peer was located. Its SSH server required a fresh Tailscale sign-in before returning even the remote hostname. The requested repo-folder inventory/diff audit is therefore **PENDING_SSH_AUTHENTICATION**, not complete and not a clean-repository finding. No laptop repository, service, branch, or file was changed.

After authentication, inspect the confirmed repo root read-only: repository/worktree paths, remotes, branch/HEAD, tracked changes, Git-eligible untracked files, ahead/behind state, unpublished refs, and matching CHELATEDAI path hashes against GitHub and the desktop snapshot. Preserve every dirty tree and private fixture. Record per-repo dispositions before proposing any additional publication; unrelated laptop repositories need their own context and review rather than being copied into this CHELATEDAI PR.

## First next action

For research, continue MA0's source-to-operator ledger and MA7's numerical-receipt design in the current portfolio order, then C0/MA1's pre-outcome descriptor specification. For the cross-machine audit, complete the requested Tailscale sign-in and confirm the laptop repo root. Scientific execution and merge readiness remain separate from this publication.
