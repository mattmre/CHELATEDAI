# 2026-09-26 projection step

Worktree `phase-i/step1-projection-loss` from `origin/main` `2bac5d3f88941cf3d3034909b796e811daf31c9f`.

`TeacherDistillationHelper.generate_distillation_targets` keeps a live projection tensor. `run_sedimentation_cycle` and `run_offline_distillation`, including the `eggroll_es` branch, train that projection. The ndarray return is a detached copy for array callers. The loss uses `recompute_live_targets`, not that copy.

Step 2 stays open. `SedimentationInfoNCELoss.forward` is still full-batch cross-entropy. Other in-batch targets are negatives. There are no false-negative labels. `test_other_in_batch_targets_are_negatives` was not rewritten. H5 was not replayed. Rung 15 was not started.

`EnsembleTeacherHelper.generate_distillation_targets` (`teacher_distillation.py`) and `CrossLingualTeacherRouter.generate_distillation_targets` (`cross_lingual_distillation.py`) still call `project_numpy`. They are not the engine training path.

Browser proofs and CONTRACT-01 stay unpaid. No new carried-debt row was added.
