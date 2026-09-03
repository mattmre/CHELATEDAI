# Session 2026-08-25 — EGV held-out production closeout

Status: `FINAL_CANDIDATE`

## Research disposition

The commissioning campaign reached a valid terminal negative result. All 80
coordinates reached `BUDGET_EXHAUSTED` after 960 attempts. Generation produced
617 response-contract failures and 343 evaluator-admitted candidates; all 343
were incorrect, comprising 321 wrong answers and 22 runtime failures. Zero
candidates were correct and zero were promoted.

The training freezer excluded all 343 candidates, admitted zero rows across zero
represented tasks, and returned `NO_ADMISSIBLE_TRAINING_SET`. The R5 diagnostic
repaired formatting for 387 responses but produced zero semantic passes. No
research adapter was created, and trained arms plus matched trained-versus-base
evaluation remain `UNEVALUATED`.

This is a completed negative experiment, not a stalled training run. Creating a
LoRA artifact would require inventing or relaxing the frozen admission evidence
and is therefore outside the accepted protocol.

## Production implementation

PR #306 is based on merged PR #305. Production code was frozen at head
`fe40b1b9b442ab9e9af38765aa0266b4f9b3843f`, tree
`1a92b331570afcc359208fc47c7eaf53f9a68079`. The accepted implementation/test
identity is head `9345ad3c16ed23623fba1255b1de5106e1c17a8e`, tree
`852823ad6c5a8297d8c566d30eb0b3e0be728396`. The intervening code change is
test-only: one correction-shock test now uses its writable temporary root rather
than requesting a directory at the filesystem root.

## Validation

- The focused portability regression passed locally in 38.902 seconds.
- Hosted Python 3.9, 3.10, 3.11, and 3.12 jobs each ran 3,255 tests and passed.
- Lint, smoke, schema, PR-body, computational-storage, and security checks
  passed. The §6.3 block-flag check remained truthfully failed under the
  authenticated, PR #305/#306-only owner override.
- Each accelerator independently ran the same 16-module exact-head suite in an
  ephemeral Python 3.12 CUDA container with no network and a read-only source
  mount. Each ran 366 tests: 354 passed, 12 expected platform skips, zero
  failures, and zero errors. Durations were 144.352 and 149.119 seconds.
- The exact source archive SHA-256 was
  `54b76ccdb5ecc4b6609471d876bec0a32d80998c33c4ba06a52731ed9b92c391`.
- An independent read-only exact-head review returned GO with zero Critical,
  High, Medium, or Low findings. It was not represented as a submitted GitHub
  approval.

The final documentation commit that carries this log must receive the same
hosted checks and a fresh bounded exact-head documentation review before merge.

## Operator service restoration

After accelerator acceptance, the operator-owned inference service was restored
and verified. Public-safe health, expected model-list, metrics, and minimal chat
probes pass, and zero active campaign workloads remain. Deployment identity,
runtime layout, and recovery details are recorded only in the private operator
handoff. No private endpoint, host, path, credential, or deployment topology is
included here.

## Claim boundaries

- No Qwen training trajectory completed under this protocol.
- No research adapter or matched model evaluation exists.
- Independent execution on two accelerators is not live cross-host
  trainer/evaluator transport.
- The results do not establish model utility, novelty, cost reduction, or
  production readiness.
- Public files exclude private prompts, model outputs, credentials, endpoints,
  paths, host labels, and deployment topology.

## Governance and disposition

PR #305 is merged. PR #306 may merge only after its final documentation head is
frozen, checked, independently reviewed, and disposed under the bounded owner
override already recorded for PRs #305 and #306. If this file is present on
`main`, use the live PR #306 record as the authoritative merge identity.

Historical CD-A2-01 and CD-H1-01 remain open and expired. CD-305-01 remains the
audit guard for the scoped override. The repository block flag remains
truthfully `BLOCKED`; this campaign does not clear or conceal that debt.
