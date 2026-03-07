# Architecture — Session 28 Stale Roadmap Cleanup

## Goal

Normalize legacy planning documents without erasing their historical context.

## Approach

### Preserve historical intent

The old documents still matter as records of what was known and prioritized on 2026-01-06. They should not be rewritten into new planning documents.

### Add explicit supersession notes

Each document should clearly state:

- it is historical
- old "Phase 4" wording is no longer authoritative
- the current state lives in the 2026-03-06 roadmap audit

### Rename misleading sections

The most confusing headings are:

- `Deferred to Phase 4`
- `Future Development (Phase 4)`
- `What Was NOT Done (Deferred to Phase 4)`

These should become historical labels rather than active-plan labels.

## Scope

This cleanup is documentation-only. It does not:

- reopen feature implementation
- change test expectations
- alter the current computational-storage follow-through

## Output

One documentation-only PR that makes legacy docs safe to read again while keeping the repo's historical timeline intact.
