# BHS v3.3 — schema, table, and enum artifacts

This directory contains the **mechanically-validated** artifacts that v3.3
points at. The prose architecture document
(`../v3.3-architecture.md`) describes *what* and *why*; the files here
define *exactly* what is permitted and rejected at the byte level.

## Why this directory exists

Three consecutive Tier B iterations (iter-5, iter-6, iter-7) returned the
same finding family: prose in `v3.3-architecture.md` drifted from its own
referenced grammars and predicates. The drift was structural, not
typographical:

- An enumerated suffix list named only some of the suffixes the prose
  introduced later in the same document (Finding 1, iter-7).
- A cross-reference pointed at a section of `docs/next-session.md` that
  the same v3.3 plan was deliberately retiring (Finding 2, iter-7).
- A post-merge-audit predicate (the "(β) predicate") contradicted the
  forward-walk semantics of the very records it was supposed to evaluate
  (Finding 3, iter-7).
- A trigger condition referenced "cycle boundary" without defining it as
  a first-class observable event (Finding 4, iter-7).

The shared root cause is that prose-as-grammar is unenforceable. A reader
(human or agent) cannot mechanically check that the prose in §6.5 matches
the prose in §6.7, and the doc has no compile step.

The artifacts in this directory replace that prose-grammar with:

- **`enums/`** — flat text files, one identifier per line. The single
  source of truth for every closed-set string that appears in a record
  field (event kinds, cause suffixes, authority names, etc.).
- **`schemas/`** — JSON Schema Draft 2020-12 documents for each
  append-only JSONL file the runtime writes. Reject-on-invalid.
- **`tables/`** — YAML files for structured data that does not fit
  cleanly in a flat enum or a JSON Schema: state machines, clock
  triggers, audit predicates.

`scripts/validate_v33_schema_drift.py` (Phase 1 deliverable) is the
compile step. It runs at commit time, extracts every identifier from
`v3.3-architecture.md` prose, and cross-checks them against the artifacts
here. Any drift fails the commit.

## How to extend v3.3

To add a new event kind, cause, authority, or state:

1. Add the identifier to the appropriate file in `enums/`.
2. Add or update the matching `oneOf` branch in the relevant
   `schemas/*.schema.json`.
3. If the new identifier participates in a state machine or has a
   triggered side-effect, update the relevant file in `tables/`.
4. Add or update prose in `v3.3-architecture.md` that references the
   new identifier.
5. Update `scripts/validate_v33_schema_drift.py` test fixtures so the
   validator's known-good corpus includes the new identifier.
6. Run the validator locally; commit only when it passes.

The order matters. Adding prose first and updating the artifacts later
is exactly the failure mode this directory exists to prevent.

## File-by-file index

```
enums/
  lane-history-kinds.txt          12 event kinds for lane-history.jsonl
  cd-record-kinds.txt              4 record kinds for carried-debt.jsonl
  lane-pause-causes.txt            5 cause namespaces
  plan-faults.txt                  6 plan_fault suffixes (incl. digest_emit_overdue)
  merge-authority-suffixes.txt    15 rulebook/charter authority paths
  state-recovery-reasons.txt       5 reasons for state_recovery events
  pivot-kinds.txt                  3 kinds of approved pivot

schemas/
  lane-history.schema.json        Draft-2020-12 schema, oneOf-per-kind
  carried-debt.schema.json        Draft-2020-12 schema, oneOf-per-kind

tables/
  cycle-clock.yaml                Defines cycle_boundary event, cd_tick trigger
  cd-state-machine.yaml           CD lifecycle, transitions, (β) predicate
```

## Authority precedence

When prose in `v3.3-architecture.md` and an artifact in this directory
disagree, the artifact is authoritative. The schema-drift validator will
fail the commit, surfacing the disagreement; the resolution is to fix
the prose, not the artifact.

If an artifact itself is wrong, the fix is a dedicated PR titled
`docs(bhs): fix v3.3 artifact <path> — <reason>`, with a Tier B review
that verifies the change does not silently invalidate already-written
records in `out/lane-history.jsonl` or `out/carried-debt.jsonl`. A
breaking schema change requires a schema_version bump and a coordinated
migration of all existing records.
