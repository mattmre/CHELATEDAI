# Drift-recovery research recovery manifest

Status: recovered unpublished research; not merged acceptance evidence

Recovery date: 2026-07-22

Recovery branch: `codex/recover-drift-research-20260722`

Base commit: `34ce4b5632e0d9cd2a16c29e0e1acc42e645b9c2`
Original worktree: `.claude/worktrees/agent-build`

## Preserved scope

This branch attaches the previously untracked `research/drift_recovery/` program and its three top-level contract suites to a durable Git history. The recovered program contains the D1 translation ladder, D2 crossover study, D2b sparse-local preflight, D3 recoverability estimator, preregistrations, frozen embedding packs, per-query evidence, and generated reports.

Before this recovery, none of these files was reachable from a local branch, tag, stash, or remote ref. The original detached worktree was not deleted, moved, or regenerated.

The pre-manifest recovery set contains 206 non-bytecode files and 119,126,236 bytes. Its aggregate manifest digest is:

`sha256(sorted(relative_path + " " + file_sha256), UTF-8 with LF separators) = 7d6be8da286775fe022893526849d37e4757f9df1019896f8f457adc973f11eb`

The largest preserved file is 16,072,097 bytes.

## Validation at recovery

- `python -m unittest discover -s research\drift_recovery\tests -p "test_*.py" -v`: 49 passed.
- `python -m unittest tests.test_objA tests.test_objA_v2 tests.test_sparse_local_preflight -v`: 22 passed.
- `python -m ruff check research tests\test_objA.py tests\test_objA_v2.py tests\test_sparse_local_preflight.py`: passed.

One lint-only repair was made during recovery in `estimator/baseline_analysis.py`: an unused import was removed and three semicolon-separated statements were split. No numerical operation, artifact, preregistration, or reported result was changed.

## Scientific interpretation boundary

- D1 is evidence that a leakage-safe global document-space ridge bridge can recover much of a model-swap retrieval gap when enough same-text document pairs are available.
- D2 is inconclusive because its synthetic collapse produced an insufficient or negative oracle gap; it is not positive evidence for chelation.
- D2b closes its tested sparse-local CPU preflight regime; it is not proof of cross-model routing.
- D3 and powered Objective A v2 are negative for the tested simple recoverability estimators.
- The packs do not contain old-space embeddings for the same anchor and evaluation query texts. They therefore cannot test a new-query-to-old-store bridge without a new, preregistered pack version.

Do not promote this recovery branch as a positive scientific verdict merely because its files and tests are now durable.
