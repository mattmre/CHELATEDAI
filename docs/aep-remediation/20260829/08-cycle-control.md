# ARCH-AEP cycle control — AEP-20260829-1

- **Cycle ID:** `AEP-20260829-1`
- **Mode:** plan-only; no product, PR, branch, or worktree mutation
- **Scope lock:** frontend, backend, and FE/BE boundary defects only: component wiring, client state, routes, API/authz/validation/data/workers, contracts, L1–L16, and user-affecting accessibility. Cosmetic nits and unrelated research validity are excluded.
- **Baseline:** root `codex/prime-ring-onion-method-dev@65c9085cd048e8a7351a53e87666fdd5639e612b`; dirty state is inventoried, not normalized.
- **Authoritative tracker:** `04-master-backlog.md`; the durable cycle pointer is `../README.md`.
- **Verification namespace:** each finding carries a literal `VER-AEP-20260829-...` identifier and the tracker binds that ID to target state and evidence disposition.
- **Tier order:** T0 inventory → T1 dirty worktrees → T2 open PRs/comments/checks/diffs → T3 full-repo fallback. The verification log records this chronology.
- **Control owner:** `/root` for audit assembly only. Product repair owners are deliberately unassigned or operator-gated in the tracker.
- **Close rule:** an audit-artifact cycle is closed only after manifest verification and fresh different-agent Tier B=100. Product remains blocked until finding acceptance criteria are independently satisfied on immutable candidates.

## Gate state

`SKIPPED_GATES: none` for the plan-only audit artifact after Tier A iteration 4. Corrected cold commands and controlling raw runtime probes are sealed in `cold-command-verification.md`, `http-auth-probe-receipt.txt`, and `be002-stage-probe-receipt.txt`. Product/hardware/deployment/operator-only actions remain explicitly deferred scope and are not represented as passing product gates.

This file does not itself declare completion. The manifest-excluded scorecard is the only final audit verdict surface.
