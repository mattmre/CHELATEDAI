# Phase I steps 1–3 re-verification (2026-09-23)

Checked on `42c2cff` and the follow-up in this change. Steps 1 and 2 are not closed. Step 3's isolation behavior is locked by a new test. The CI torch-cache note that step 4 left open is recorded here from the workflow file, not from a new cache system.

## Step 1 — projection

`DimensionProjection.project_tensor` does not detach. `test_unit_core.py` already checks that a backward through `project_tensor` changes the weights, and that `project_numpy` does not record gradients.

`TeacherDistillationHelper.generate_distillation_targets` returns a numpy array. A grad-carrying tensor cannot be converted with `.numpy()` unless it is detached first. The previous teacher path called `project_tensor` and then `.detach().numpy()`, which discarded the graph and left the "parameters can be trained" comment false. The panel note F-ML-003 (`docs/ARCH AGENTIC ENGINEERING AND PLANNING/planning/2026-05-reconciliation/panel-analysis/02-data-engineering-ml.md`) says to either train the projection with its own loss or treat it as a fixed preprocessor.

This change treats that target path as a fixed preprocessor and calls `project_numpy`. It does not add a new projection objective. The sedimentation optimizer can still list projection parameters; the adapter loss against numpy targets does not produce a gradient for them. Step 1's exit, "the projection trains on the teacher path," is not claimed.

## Step 2 — InfoNCE

`SedimentationInfoNCELoss.forward` still builds a similarity matrix of every output against every target and uses cross-entropy on the diagonal. Other documents' targets in the batch are negatives. There is no label that says which of those pairs are false negatives, so this change does not add a mask that would zero every off-diagonal entry. That mask would leave a one-class problem and a loss that no longer measures alignment. `test_other_in_batch_targets_are_negatives` locks the current coupling. Step 2 stays open.

## Step 3 — adapter checkpoint isolation

`benchmark_utils.isolated_adapter_state` copies an existing `adapter_weights.pt` aside, unlinks it, and on exit deletes whatever the block wrote and moves the copy back. `.gitignore` already ignores `adapter_weights.pt` and `adapter_weights.benchmark-backup-*.pt`. `test_isolated_adapter_state.py` checks restore, a file created inside an empty block, and a nested block. No stray backup remains after the tests.

## CI torch cache

`.github/workflows/test.yml` sets `cache: pip` on `actions/setup-python@v5` for the lint, test, and computational-storage jobs. The test jobs then run `pip install torch --index-url https://download.pytorch.org/whl/cpu` before `requirements.txt`. There is no separate `actions/cache` step whose key is the torch wheel. This note does not claim a cold runner restores that wheel.

## Not done here

Dashboard FM-3 is already on this tree: `loadSummary` and `loadEvents` call `fetchJson`, which throws on a non-OK response, and the catch paths do not write a zero event count. A browser DOM-fire of that 401 was not run. FM-2's old 501 was not reproduced: `do_OPTIONS` returns 405 without a bearer token and 204 when authorized. A browser preflight was not run. Spark BEIR was not run. Rung 15 was not started.
