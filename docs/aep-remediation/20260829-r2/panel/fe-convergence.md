# Round 2 FE panel convergence — Alex Rivera

## Converged position

The local UI correctness pass found two High and one Medium net-new frontend defects. All three arise from a split dashboard architecture rather than visual polish: the normal page cannot reach three implemented panels, concurrent resource completion can erase a real failure, and three render catches write errors into hidden nodes. The panel does not claim browser execution because the repository has no checked-in browser gate.

## Findings and implementation meaning

1. **Canonical-surface split — High / blocked.** `serve_dashboard()` chooses the checked-in static client whenever it exists, while Phase C, Model Scope, and TTS exist only in the inline client. File absence is acting as a feature flag. The operator must choose one canonical client before a lesser agent ports or removes anything (REPO-008).
2. **Shared concurrent error state — High / blocked.** Summary and events start together and independently clear one global banner on success. The later completion wins even when the other resource failed. If the inline client is retained, request state must be keyed per resource and the aggregate derived from active failures (REPO-009).
3. **Hidden render failures — Medium / blocked.** TTS, Model Scope, and Phase C hide their loading node before formatting, then catches write errors into that hidden node. Use validated response discriminants and dedicated visible state containers if those panels remain (REPO-010).

## Carry-forward boundaries

Source reinspection confirms prior FE002, FE004, FE005, and FE006 mechanisms remain present; they are pointer-carried rather than counted again. The revised local-only pass did not reassess prior FE001 credential/session behavior or FE003 executable-markup behavior. That restraint is deliberate: no runtime/browser statement is made without the corresponding reproducible harness.

## Positive behavior preserved

- Inline summary/events requests check response status and distinguish valid empty events.
- Ordinary inline event rendering uses DOM/text nodes.
- External campaign, validation, preflight, and evidence loaders use the status-aware helper and explicit empty/error rows.
- Handler/loader tests remain useful beneath the missing client/served-page layer: `Ran 67 tests; OK (skipped=2)`.

## Acceptance shape

Choose one client, make every retained panel reachable in the normal checkout, model each resource as loading/empty/success/stale/error, validate response discriminants before rendering, and add a checked-in served-page gate. Until that exists, source and unit evidence cannot be upgraded to browser correctness.
