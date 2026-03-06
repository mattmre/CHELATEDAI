# Computational Storage Retention Policy -- 2026-03-06

## Decision Summary

Do **not** delete any backup refs or retired branch artifacts tonight.

Instead, use the following retention tiers:

## Tier A -- Active rollback aid

Keep `backup/feat-computational-storage-payload` until the real-hardware evidence item is closed and the transport follow-up work has stabilized.

Reason:
- It is the most directly relevant rollback/reference point for the recently merged payload path.
- The hardware-evidence task is still open, so this backup may still be useful for comparison.

## Tier B -- Recently retired computational-storage refs and artifacts

Keep the following until a **manual review on or after 2026-04-05**:

- `backup/retired-computational-storage-poc-2026-03-06`
- `backup/retired-computational-storage-poc-rescue-2026-03-06`
- `backup/retired-session22-online-correction-2026-03-06`
- `.claude/retired-branch-artifacts/2026-03-06-session22-online-correction/`

Reason:
- These were created during the stale-branch retirement and rescue flow on 2026-03-06.
- They are still close enough to the active follow-up work to be useful if rollback or forensic comparison is needed.

## Tier C -- Older February safety snapshots

Mark the following as **delete candidates** for the same 2026-04-05 review window:

- `backup/local-main-ahead-2026-02-18`
- `backup/local-session8-ahead-2026-02-18`
- `backup/wip-local-snapshot-2026-02-18`
- `origin/backup/local-main-ahead-2026-02-18`
- `origin/backup/local-session8-ahead-2026-02-18`
- `origin/backup/wip-local-snapshot-2026-02-18`

Reason:
- They predate the current computational-storage split by more than two weeks.
- The earlier cleanup report already treated these as temporary safety snapshots rather than long-term artifacts.

## Current Inventory

### Local backup refs

| Ref | Created | Commit | Notes |
| --- | --- | --- | --- |
| `backup/feat-computational-storage-payload` | 2026-03-05 | `69e9282` | Payload rollback aid |
| `backup/local-main-ahead-2026-02-18` | 2026-02-16 | `9ab462c` | Older safety snapshot |
| `backup/local-session8-ahead-2026-02-18` | 2026-02-17 | `799ec36` | Older safety snapshot |
| `backup/retired-computational-storage-poc-2026-03-06` | 2026-03-05 | `b291e14` | Stale branch retirement |
| `backup/retired-computational-storage-poc-rescue-2026-03-06` | 2026-03-05 | `20ac79e` | Rescue rollback ref |
| `backup/retired-session22-online-correction-2026-03-06` | 2026-02-28 | `1f58d79` | Session 22 stale-line retirement |
| `backup/wip-local-snapshot-2026-02-18` | 2026-02-18 | `337cb24` | Older work-in-progress snapshot |

### Remote backup refs

| Ref | Created | Commit | Notes |
| --- | --- | --- | --- |
| `origin/backup/local-main-ahead-2026-02-18` | 2026-02-16 | `9ab462c` | Older safety snapshot |
| `origin/backup/local-session8-ahead-2026-02-18` | 2026-02-17 | `799ec36` | Older safety snapshot |
| `origin/backup/wip-local-snapshot-2026-02-18` | 2026-02-18 | `337cb24` | Older work-in-progress snapshot |

### Retired branch artifacts

| Artifact bundle | Last updated | Files | Approx. size |
| --- | --- | --- | --- |
| `.claude/retired-branch-artifacts/2026-03-06-session22-online-correction/` | 2026-03-05 | 4 | 10,663 bytes |

Contained files:
- `findings.md`
- `progress.md`
- `session-log-2026-02-27-session23.untracked.md`
- `task_plan.md`

## Review Trigger For 2026-04-05

At the 2026-04-05 or later review:

1. Confirm `main` has remained stable and no rollback comparison is still in use.
2. Confirm the transport follow-up items have progressed enough that `backup/feat-computational-storage-payload` is no longer uniquely valuable.
3. Delete Tier C items first if still unneeded.
4. Delete Tier B items only after confirming the stale-branch retirement path is no longer needed for forensic comparison.

Deletion remains manual. This document authorizes review, not automatic cleanup.
