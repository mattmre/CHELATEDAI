# Architecture -- Session 26 Retention Policy

## Objective
Define a deletion policy that is conservative enough to protect rollback paths while specific enough to prevent indefinite branch/artifact sprawl.

## Design
1. Inventory current local refs, remote refs, and retired artifact bundles.
2. Group them by recency and current usefulness.
3. Record a manual review date and sequence for deletion candidates.
4. Log the decision in `change-log.md` so future sessions inherit it explicitly.

## Acceptance Criteria
- There is one inventory-backed retention document in `docs/`.
- `change-log.md` records the defer decision and review window.
- The policy distinguishes the payload backup from the older February safety snapshots.
