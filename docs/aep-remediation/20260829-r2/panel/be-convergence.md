# Round 2 BE/reliability convergence — Sofia Andersson and James Okafor

## Converged position

The backend and boundary review found two net-new Critical, six High, and one Medium defects outside the frontend-only packets. The dominant failure is premature publication of state: a service is restored before child termination is proved, lifecycle marks work verified without receipts, checkpoints are acknowledged without unique/durable catalog state, and configuration is announced enabled before invariants hold.

## Reliability lenses

1. **Termination before restoration.** Sending a terminal signal is not proof of a terminal/reaped process. The QSCCI supervisor must keep the displaced GPU service stopped when child/group state is unknown (WT-001).
2. **Receipts before closure.** Phase-return bookkeeping is not remediation or independent verification. Analysis-only runs cannot emit `VERIFIED`; immutable candidate and verifier receipts must be typed and linked (REPO-003).
3. **Timeout before revocation.** A join timeout does not cancel a Python thread. Workers require immutable inputs, generation-scoped output acceptance, and an execution boundary whose terminal state can be established (REPO-004).
4. **Identity and metadata as one checkpoint commit.** Collision-resistant no-replace IDs and atomic metadata durability must be one transaction boundary; either packet fixed alone leaves false recovery guarantees (REPO-001/002).
5. **Newest evidence cannot disappear.** Parse/read failure is an explicit current state. Skipping it and showing older green is operator misinformation (REPO-005).
6. **Validate before enable.** Router dimensions and adaptive-threshold bounds must be validated as complete batches before live state or enabled events change (REPO-006/007).
7. **Preserve the primary failure.** PR #257 must keep the original block-processing exception authoritative when a secondary research hook also fails (PR257-001).

## Unit/runtime boundary

Focused suites passed while all of these paths remained reproducible. That is not contradictory: existing tests omit same-second collisions, persistence failure, absent callbacks, late worker mutation, malformed-newest evidence, incompatible router geometry, and non-atomic threshold failure. The result is an L5 boundary, not a reason to distrust unit tests generally.

## Dissent and resolution

- James initially treats the service-restoration defect as operationally bounded by an untracked supervisor overlay; Sofia retains Critical because it can create simultaneous GPU ownership and corrupt the official run boundary. The finding stays blocked and Critical.
- The validation-history repair is cold-ready on the backend response contract, but frontend rendering must later consume its unknown/error discriminant.
- The adapter-router regression module does not exist at root. The current engine suite is only regression context; adding a focused tracked module is acceptance work.

## Implementation constraint

Fix agents receive atomic packets, preserve dirty state, and bind results to immutable candidates. No passing suite, enabled event, returned identifier, or phase status is accepted as completion without the corresponding production-path invariant.
