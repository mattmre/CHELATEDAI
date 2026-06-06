# Core execution track — goal charter

**Policy:** SHIM blocking rows are **deferred last** (`docs/ROADMAP_EXECUTION.md`).  
**BHS target:** 100/100 on every module touched in the active step.  
**Merge gate:** `CLEAR` while following the core queue.

## Goal

Execute `docs/ROADMAP_EXECUTION.md` **one step at a time**. Do not start step N+1 until step N exit criteria pass.

## Worker mandate

- Iteratively improve code toward BHS 100/100 on the **current step only**.
- No doc-only slices; no new shim backlog while core queue is active.
- Re-run regression tests after each change.

## Stop

- Current step exit criteria met, or wall clock / operator halt.
- SHIM work resumes only after queue step 8.