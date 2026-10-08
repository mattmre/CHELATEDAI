# BE/reliability/implementation convergence

Sofia Andersson led backend correctness, challenged by James Okafor on reliability and Margaret Chen on implementation quality. Exact PR heads, live metadata, comments, checks/logs, selected unit slices, dirty worktree code, and production-path probes were separated explicitly.

## Evidence lenses

1. **Which rulebook is authoritative?** Canonical BHS v3.7.1 was independently opened from the kit's `v3.5/` implementation path. The repo still automates v3.3/L1–L13; L14–L16 in this audit are manual, not mechanically enforced. This becomes BE001-001 rather than a false claim that the requested gate ran.
2. **Can live #292–#295 claims survive later disproof?** No. `docs/next-session.md:63-65,542` supersedes bodies: #293/#295 failed 70/60 Critical, #292 is stale, #294 needs replacement review. Git mergeability is not correctness; PR296-001 is stop-line Critical.
3. **What did clean unit slices prove?** #293 60, #294 15, and #295 49 tests pass. They prove bounded regressions only, not concurrency, publication, qrels completeness, engine activation, base compatibility, or BHS acceptance. Later disprove evidence remains authoritative.
4. **Does PR257's fallback execute?** `{runtime_config_base}` deterministically constructs a set with an unhashable dict. This supported fallback branch is a Critical runtime crash even though the edit is one line.
5. **Are state/error boundaries reversible?** Nested adapter isolation protects only the outer context; callback result serialization escapes its error envelope. Require exception-path restoration and unpicklable-result probes.
6. **Can official RHPC prove trusted source?** No expected reviewed SHA enters admission; a fabricated SHA verifies. Dissent: James caps exposure High while uncommitted; Sofia calls the official decision plane Critical. The audit retains Critical and blocks any official claim.
7. **Does sound bearer comparison make a working UI?** No. Browser credential acquisition/propagation is a shared FE/BE contract; acceptance is a browser loading protected data without token exposure.
8. **Can backend errors become plausible UI success or executable markup?** Yes. Typed server failures are laundered by clients and persistent artifact text enters `innerHTML`. Both boundary halves must be tested together.
9. **Do report/lifecycle consumers have producers?** No current test report producer exists and sweep state is inferred. Define versioned producer and lifecycle contracts before patching labels.
10. **Can fixes be accepted from mixed worktrees?** No. Every lesser agent needs an explicit branch/SHA/file cohort; retained work must reproduce from fresh checkout. Do not fix superseded public heads in place when the later record requires replacement.

## Strengths to preserve

- Deterministic unit slices remain useful and several substantive #293/#294 review concerns are fixed at those heads.
- Negative verdicts are often explicit; #256/#257 are parked instead of merged.
- The later next-session record candidly preserves exact failed scores and unsafe boundaries.
- FE supplied live HTTP evidence, exposing a gap unit tests missed.
- RHPC artifact generation already uses canonical encoding, fsync, internal regeneration, and exhaustive inventory; hardening should preserve these properties.

## Convergence and dissent

Critical: live PR truth drift, PR257 deterministic fallback crash, and forgeable RHPC official provenance. High: RHPC snapshot/no-replace races, nested isolation, browser auth, error-as-empty, unsafe sinks, missing report producer, registry ownership leak, and BHS enforcement drift. Medium/Low: callback diagnostics, lifecycle inference, and recoverable local worktree hygiene. Margaret's implementation constraint is atomic replacement PRs with one runtime contract each, not repeated polishing of superseded branches.
