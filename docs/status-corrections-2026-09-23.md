# Status corrections — 2026-09-23

Commit checked before the edits: `49804ae3ddcebf4e4060fae95ba812331299f757` (`origin/main` and `HEAD`). Each record is one false present-tense sentence (or one bullet that contained one), the replacement, and the command or line opened before the sentence was changed. Sentences without an opened source were not changed; those are listed at the end.

Ancestor checks (all exit 0): `git merge-base --is-ancestor` of `c148f7b3c1e3858b461f6c08ddb3e58cd7e99a85` (#292), `782ab62ddbaf6ab6c40085255e48ff9a21562d34` (#293), and `8e6e83b7c30bae34015ad12996314b5eea2d1c64` (#294) against `HEAD` and `origin/main`. `gh pr view` showed #292, #293, and #294 `MERGED` at those commits, and #295 `OPEN` with `mergedAt: null`. `git merge-base --is-ancestor d2393942119a69a79ee30a1807eef5d5694906da HEAD` (PR #295 head) exited 1.

## Corrections

### 1. `docs/ROADMAP_EXECUTION.md` block-flag bullet

- False sentence: `Merge gate: **CLEAR** for this sequence (`docs/next-session.md`). SHIM debts stay **OPEN** but **non-blocking** until the queue below finishes.`
- Replacement: `Merge gate: **CLEAR** on this commit. `python3 scripts/check_block_flag.py` reports zero OPEN carried-debt rows in `docs/next-session.md` (22 data rows, every Status cell starts with CLOSED). CHANGELOG names seven historical ids, not eight: SHIM-CD-01, SHIM-CD-02, SHIM-CD-06, SHIM-CD-08, SHIM-CD-09, SHIM-CD-03, and SHIM-CD-07. SHIM-CD-05 is already marked CLOSED in CHANGELOG. SHIM-CD-04 does not occur. Those seven names are not rows in the carried-debt table, so the checker does not count them. This sentence does not close them and does not add debt rows.`
- Opened: `python3 scripts/check_block_flag.py` (CLEAR, row count 0, exit 0). Row parse of `docs/next-session.md` Carried Debt: 22 data rows, 22 closed, 0 not closed. `rg -n "SHIM-CD-"` and `rg SHIM-CD-04` (no matches).

### 2. `docs/ROADMAP_EXECUTION.md` automation commands

- False sentence: the Automation section tells the operator to run `bash scripts/run_step_with_checks.sh` and `bash scripts/loop_core_10m.sh`, and says the phase loop prefers CORE-SLICE handlers, as if those programs are in this tree.
- Replacement: added `Those commands are not runnable on this commit: scripts/run_step_with_checks.sh, scripts/loop_core_10m.sh, and scripts/phase_development_loop.py are absent.`
- Opened: `os.path.exists` for those three paths (all ABSENT).

### 3. `docs/ROADMAP_EXECUTION.md` "what we are not doing"

- False sentence: `Rung 17's disk-pool slice is in this PR (#294), not already merged.` The same bullet also says disintegration on main is `#279 PARTIAL` (post-bank only).
- Replacement: Rung 17 is on main via merged PR #294 (`8e6e83b7c30bae34015ad12996314b5eea2d1c64`). Disintegration on main includes #279 post-bank prune/re-anneal. Rung 13 detector-to-DAG prune is on main via #293. Rung 15 is OPEN, not done, and not refused. Rung 16 is not on this commit.
- Opened: ancestor checks above. `evidence_dag.py:290` `def prune_edges`. `evidence_dag_disintegration.py:201` `def reanneal_edges`. `git merge-base --is-ancestor 06f4f5cced82fb37547b1a7a36ee34f0dc9451f6 HEAD` (#279) exit 0. `gh pr list --state merged --search "GNN prototype"` returned no rows.

### 4. `docs/ROADMAP_EXECUTION.md` status snapshot

- False sentence: `Rung **17** (disk pool slice) is **in this PR (#294), not already merged**.` The snapshot header still says it was rebased onto `origin/main` `782ab62`, and it groups rungs 15 and 16 as "not done".
- Replacement: snapshot is for this commit `49804ae3ddcebf4e4060fae95ba812331299f757`. Rung 17 is on main via #294. Rung 15 is OPEN, not done, and not refused. Rung 16 is not on this commit (PR #295 open; head not an ancestor). H5 is on main via #292. Rung 13 detector-to-DAG prune is on main via #293.
- Opened: same ancestor checks. Additional ancestors checked exit 0: #260 `f0c643ae7bf5a7c7ca612ceaa878b7cb15412540`, #280 `9db098ea0d2f7df91b54d29e5ee1c6eb9ea65bb8`, #277 `81e3614bec7d5e5b9cce57af8e7126c6b97775dc`, #284 `efe6d1be55b5c702eb3a1cccde3169feb04b6c34`, #285 `49b36046b6d59a554633f631924005130a05cd1f`, #286 `d8181f64161522eb39f8b0e1fd9e48aa0fa536b2`, #289 `d1bfc0903d7de5c2b81d41a7d081e7da8502ad40`, #290 `6d3af3009c29b31ce72244c13b53c02170bfbd61`, #291 `34ce4b5632e0d9cd2a16c29e0e1acc42e645b9c2`.

### 5. `docs/ROADMAP_EXECUTION.md` rung 15 status cell

- False sentence: none that called rung 15 refused or done. The cell said only `OPEN — not done` and did not say it is not refused.
- Replacement: `OPEN — not done, not refused.` No merged GNN-prototype PR in the search above.
- Opened: `gh pr list --state merged --search "GNN prototype"` empty; `gh pr list --state open --search "GNN"` empty. This is a search, not a proof that no alias exists.

### 6. `docs/ROADMAP_EXECUTION.md` rung 16 status cell

- False sentence: `**OPEN — not on main** (open PR #295). Not done.` "Not done" is more than this commit can say. The PR title was not copied.
- Replacement: `Not on this commit.` PR #295 is open and not merged. Its head is not an ancestor of this commit.
- Opened: `gh pr view 295 --json state,mergedAt,mergeCommit` (`OPEN`, null, null). Ancestor check of `d2393942119a69a79ee30a1807eef5d5694906da` exited 1.

### 7. `docs/ROADMAP_EXECUTION.md` rung 17 status and exit cell

- False sentence: `**IN THIS PR (#294), NOT ALREADY MERGED.** Not on main before this PR.` and `Do not call rung 17 already merged.`
- Replacement: on main via merged PR #294. This commit's `computational_storage_poc/pool_shard.py` defines `write_pool_shard` (line 100), `read_pool_shard` (line 241), and `verify_pool_shard_parity` (line 286), plus `retrieve_topk` (line 331). Callers in `*.py` are only that module and `test_pool_shard_parity.py`. `docs/rung17-disk-pool-slice.md` lines 8–9 say it is not a production retrieval path.
- Opened: those line numbers; `rg` of the four names under `*.py`; rung-17 doc lines 8–9; module lines 3–7 and 29 (`BYTE_ENCODING`), 297–306 (three SHA256 compares and the NaN byte-identity comment).

### 8. `docs/next-session.md` H5 disposition

- False sentence: `This H5 verdict is on this branch (PR #292), not previously on main. Do not promote the living bank. Rung 13 detector-to-DAG prune, rung 15, rung 16, and rung 17 are not done.`
- Replacement: H5 is on main via merged PR #292 (`c148f7b3c1e3858b461f6c08ddb3e58cd7e99a85`). Do not promote the living bank. Rung 13 detector-to-DAG prune is on main via #293. Rung 17 is on main via #294. Rung 15 is OPEN, not done, and not refused. Rung 16 is not on this commit. The numeric table under that paragraph was left as written.
- Opened: ancestor checks. H5 numbers kept because `docs/drift-recovery-post-bank-headtohead-results-2026-06.md` lines 13–15 and 19–24 and `docs/drift-recovery-post-bank-headtohead-nfcorpus-results-2026-06.md` lines 13–15 and 19–24 contain 0.131135, 0.180862, 0.046389, and 0.045817, and both say `LIVING BANK WINS (beats both): False`. No new metrics were added.

### 9. `docs/next-session.md` rebase note

- False sentence: `Rung 13 detector-to-DAG prune (#293), rung 17 disk pool (#294), and rung 16 quant routing (#295) stay unmerged.`
- Replacement: #293 and #294 are merged and are ancestors of this commit. #295 is not merged and is not on this commit.
- Opened: ancestor checks and `gh pr view` states above. Carried Debt rows were not edited.

### 10. `CHANGELOG.md` Unreleased Phase II bullet

- False sentence: `On this branch, not previously on main: the H5 living-bank verdict` and `detector-to-DAG prune is not on main. Not done: rungs 13, 15, 16, and 17.`
- Replacement: H5 is on main via #292. Rung 13 detector-to-DAG prune is on main via #293. Rung 17 is on main via #294. Rung 15 is OPEN, not done, and not refused. Rung 16 is not on this commit. The existing H5 figures in the later verdict bullet were not rewritten.
- Opened: ancestor checks and the two H5 result docs named above.

### 11. `CHANGELOG.md` Unreleased file bullets

- False sentence: the Added bullets name `scripts/phase_development_loop.py`, `scripts/run_10min_priority_bhs_loop.py`, `scripts/loop_core_10m.sh`, `scripts/loop_10m.sh`, `scripts/chelated_loop_timer.py`, `docs/loop_workers/`, the six `scripts/record_shim_*_evidence.py` files, `chelated_shim_research.py`, `shim_node_promoted.py`, `shim_collapse_benchmark_extension_promoted.py`, `scripts/run_five_worker_shim_gate.py`, `scripts/verify_shim_development.sh`, `scripts/run_step_with_checks.sh`, `reports/ARCH_AEP_REMEDIATION_FINDINGS*.md`, `tests/test_shim_*`, `tests/test_learning_loop_e2e.py`, `tests/test_phase_development_loop_scheduler_handler.py`, and `tests/test_run_10min_priority_bhs_loop.py` as if they are in this tree.
- Replacement: each of those paths is not in this commit. `test_model_scope_runtime.py` and `test_model_scope_steering.py` are present at the repository root and were not re-run. `tests/test_model_scope_runtime.py` is not in this commit.
- Opened: `os.path.exists` for each path (ABSENT, except the two root Model-Scope tests). `glob` of `tests/test_shim_*` and `reports/*` was empty. `docs/loop_workers` ABSENT.

### 12. `CHANGELOG.md` bounded `run_large_sweep` claim

- False sentence: `` `run_large_sweep.py` bounded persistence `` in the Unreleased Changed bullet, plus the 2026-06-03 bullets `Bound the large sweep persistence path (run_large_sweep.py) to avoid per-iteration full JSON rewrites`, `Added run_large_sweep to package py-modules in pyproject.toml`, and the `test_run_large_sweep.py` checkmark.
- Replacement: the `run_large_sweep` claims are ahead of this tree. `run_large_sweep.py` lines 125–134 still read the whole JSON, append one object, and write the whole JSON back. `run_large_sweep` is absent from `pyproject.toml` `py-modules` (the list starts at line 61). `test_run_large_sweep.py` is not in this commit. The same Unreleased bullet still names `benchmark_utils.py`, `sedimentation_loss.py`, `checkpoint_manager.py`, and `.gitignore`; those clauses were not re-verified and were not rewritten. `CLAUDE.md` was not edited and that 2026-06-03 clause was not re-verified.
- Opened: `run_large_sweep.py` lines 125–134. `pyproject.toml` line 61 and a search that found no `run_large_sweep` in that file. `os.path.exists("test_run_large_sweep.py")` ABSENT.

### 13. `CHANGELOG.md` core-queue bullet

- False sentence: `Core queue steps 1–6 and 8 are implemented and regression-tested on this branch`.
- Replacement: that claim is ahead of this tree for step 8 because `tests/test_learning_loop_e2e.py` is absent. Steps 1–4 were not re-verified in this PR (not declared done or not done). The `run_large_sweep` claims are ahead of this tree, as in row 12. This bullet does not say Phase I is complete. Step 7 was not re-verified in this PR (the previous fixture/integration-gated clause was not kept, because it was not re-opened).
- Opened: absence of `tests/test_learning_loop_e2e.py`; the `run_large_sweep.py` and `pyproject.toml` lines in row 12. Steps 1–4 code and tests were not opened.

### 14. `CHANGELOG.md` SHIM-CD findings

- False sentence: `partial promoted SIP and registry probes exist` under the OPEN historical ids, read as files on this commit. The "8 open rows" count is not in this file; the seven names are.
- Replacement: SHIM-CD-01/02/06/08/09 stay not-closed, and the promoted-probe clause is withdrawn because those two modules are absent. SHIM-CD-03 and SHIM-CD-07 were left unchanged in that pass. Row 34 replaces them because those sentences were not re-opened. The seven names are not carried-debt table rows, so `check_block_flag.py` does not count them. SHIM-CD-05 stays the changelog's CLOSED id, with the note that `chelated_shim_research.py` is not in this commit. SHIM-CD-04 does not occur. No debt rows were added.
- Opened: Carried Debt parse (0 not-closed). `rg SHIM-CD-04` no matches. `os.path.exists` for the two modules ABSENT. `rg "promoted_sip_apply|promoted_registry_probe|CHELATED_SHIM_RESEARCH"` hits only CHANGELOG and README.

### 15. `CHANGELOG.md` phase-loop turn bullet

- False sentence: the phase loop is at turn 3171+ with a next slice, as a state of this commit.
- Replacement: `scripts/phase_development_loop.py` is not in this commit, so that turn count was not re-verified here.
- Opened: `os.path.exists("scripts/phase_development_loop.py")` ABSENT. The number 3171 was not re-derived.

### 16. `CHANGELOG.md` lattice bullet that says H5 is on this branch

- False sentence: `Rungs 9–12 DONE, 13 PARTIAL (post-bank only), 14 apparatus DONE. The H5 living-bank verdict is on this branch, not previously on main.`
- Replacement: rung 13 detector-to-DAG prune is on main via #293; #279 post-bank prune/re-anneal is also an ancestor. The H5 verdict is on main via #292. The earlier PR list in that bullet was checked via the ancestor commits in row 4, not re-executed.
- Opened: ancestor checks for #292, #293, and #279.

### 17. `CHANGELOG.md` "not done, and not on main" bullet

- False sentence: `Not done, and not on main: rung 13 detector-to-DAG prune (open PR #293); ... rung 17 disk pool (open PR #294).`
- Replacement: rung 13 and rung 17 are on main via the merged commits above. Rung 15 is OPEN, not done, and not refused. Rung 16 is not on this commit. No arena result was written.
- Opened: `gh pr view` for #293, #294, and #295.

### 18. `CHANGELOG.md` 2026-06-05 learning-loop lines

- False sentence: `Executed end-to-end learning-loop regression on fixture corpus (tests/test_learning_loop_e2e.py)` and `` `python tests/test_learning_loop_e2e.py` ✅ ``.
- Replacement: that path is not in this commit, so those lines are not a result for this tree. The parked PR #257 test was not ported and was not opened.
- Opened: `os.path.exists("tests/test_learning_loop_e2e.py")` ABSENT. `gh pr view 257` state OPEN, `mergedAt` null. The test body was not read.

### 19. `CHANGELOG.md` 2026-05-28 `promoted_sip_apply` line

- False sentence: `` `promoted_sip_apply()` partial wiring `` as code on this commit.
- Replacement: `promoted_sip_apply` does not occur in `*.py` on this commit.
- Opened: `rg "promoted_sip_apply"` — CHANGELOG.md only, besides the README module name.

### 20. `README.md` Phase I / Phase II status table

- False sentence: `Steps 1–6 and 8 largely complete on live branch` and `Phase II ... starts after Phase I exit`, plus `Open, env-guarded` for a shim module that is not here.
- Replacement: Phase I is not complete. Step 8's test is absent. Steps 1–4 were not re-verified. `chelated_shim_research.py` is not in this commit. Rungs 13 and 17 are on main. Rung 15 is OPEN, not done, not refused. Rung 16 is not on this commit.
- Opened: the same absence check, ancestor checks, and PR #295 state.

### 21. `README.md` lattice surface table and primary entrypoints

- False sentence: next milestones still list rung 11, rung 12, rung 13, and rung 17 as not-yet, and `chelated_shim_research.py` as a surface that exists today.
- Replacement: #260, #280, #277, #293, and #294 are ancestors, with the file lines cited in rows 3 and 7. `chelated_shim_research.py` is not in this commit. Rung 10 merge commits #284, #285, #286, and #289 are ancestors; that does not close historical SHIM-CD ids.
- Opened: those ancestor checks and `os.path.exists`. Present and not behavior-tested: `model_scope_steering.py`, `self_healing_chelation.py`, `build_attribution_pool.py`, `vector_store.py`, `sedimentation.py`, `chelation_adapter.py`, `online_updater.py`, `isomer_detector.py`, `antigravity_engine.py`, and `computational_storage_poc/block_graph.py`.

### 22. `README.md` primary-path command

- False sentence: the validation fence runs `python scripts/phase_development_loop.py --once` as a current step.
- Replacement: that command was removed from the fence. The path is not in this commit.
- Opened: `os.path.exists("scripts/phase_development_loop.py")` ABSENT. `scripts/check_block_flag.py` was run and exists.

### 23. `README.md` live-branch status rows

- False sentence: `Progress branch: feat/live-progress-tracker-20260606` as this commit; `Done on branch` for the sweep row and for `tests/test_learning_loop_e2e.py`; SHIM and phase-loop rows naming absent files; `CLEAR` with `8 open SHIM carried-debt rows`.
- Replacement: this commit is `49804ae3ddcebf4e4060fae95ba812331299f757`. PR #257 is open and is not this commit. The learning-loop test, shim modules, recorders, five-worker gate, phase-loop script, `scripts/loop_core_10m.sh`, and `reports/` are not in this commit. The block-flag sentence matches row 1. Steps 1–3 in the ML row are marked not re-verified, not done and not not-done. Phase I is not called complete.
- Opened: `gh pr view 257` OPEN. Absence checks. Block-flag script and the 22-row parse. `run_large_sweep.py` lines 125–134 and `pyproject.toml` line 61.

### 24. `README.md` "new surfaces" table

- False sentence: the table lists `chelated_shim_research.py`, `shim_node_promoted.py`, `scripts/record_shim_*_evidence.py`, `scripts/run_five_worker_shim_gate.py`, `scripts/phase_development_loop.py`, and `reports/ARCH_AEP_REMEDIATION_FINDINGS*.md` as surfaces on this tree.
- Replacement: those paths are not in this commit.
- Opened: `os.path.exists` and empty `reports/` / `tests/test_shim_*` globs.

## Refused — evidence was not opened, so the sentence stays

- `README.md` Model-Scope pilot row (`In progress`) was not changed. The pilot code and tests were not opened far enough to call step 7 done or not done. `test_model_scope_runtime.py` exists at the repo root; it was not run.
- `CHANGELOG.md` 2026-06-03 claims that InfoNCE masking and `benchmark_utils.py` adapter isolation were fixed were not changed. The files exist (`sedimentation_loss.py`, `test_sedimentation_loss.py`, `benchmark_utils.py`, `test_benchmark_utils.py`). The specific fixes and the checkmarks were not re-run. Steps 1–4 were not declared done or not done.
- `CHANGELOG.md` `python -m unittest -q test_benchmark_utils.py` and `test_sedimentation_loss.py` checkmarks were not changed. Those commands were not re-run.
- `docs/next-session.md` Carried Debt row text was not changed, including CLOSED rows that still say "on this branch". Status cells were not edited. No OPEN row was added.
- `docs/next-session.md` Session Evidence (2026-09-03) leftovers line was not rewritten. It is a dated log. The present-tense merge correction is the rebase note.
- The parked PR #257 learning-loop test was not opened and not ported. Its assertions were not written into this tree.
- PR #295's title, arena deltas, and body were not copied. Rung 16 is described only as not on this commit.
- `CLAUDE.md` was not edited.
- H5 and H4 numbers already printed in `CHANGELOG.md` and `docs/next-session.md` were not replaced and no new metrics were invented. The H4 ablation file `docs/drift-recovery-h4-compound-cycles-ablation-2026-07.md` exists; its figures were not re-derived in this pass beyond confirming the H5 source files exist for the numbers that were kept.
- `CHANGELOG.md` "Model-Scope stack ... hardened" was not re-opened.
- `CHANGELOG.md` 2026-06-05 `CLAUDE.md` entrypoint sentence was not re-opened. `CLAUDE.md` was not edited.
- Harness PR numbers #258–#276 were not re-checked. Sentences that used to treat that whole span as freshly git-verified now say they were not re-checked.
- `docs/ROADMAP_EXECUTION.md` rows 9, 11, 12, and 14 were not rewritten except where a rung-16 overlap phrase or a "this PR" phrase was opened. Row 9's DONE claim was not re-verified.

### 25. `CHANGELOG.md` Unreleased heading

- False sentence: `## [Unreleased] — live progress branch feat/live-progress-tracker-20260606` as the identity of this commit.
- Replacement: the section discusses `49804ae3ddcebf4e4060fae95ba812331299f757`. PR #257 is open and is not this commit. Row 33 supersedes the earlier wording that the section was checked on that commit.
- Opened: `git rev-parse HEAD` and `gh pr view 257` (`OPEN`, `mergedAt` null) in the prior pass. Follow-up `git rev-parse HEAD` is `081a5ccb781634dc8145427e9e0dc1e446852625`. `gh pr view 257` was not re-run.

### 26. `CHANGELOG.md` core engine seams

- False sentence: those five modules are "wired for optional SHIM preflight metadata".
- Replacement: the five files are present and a search for `shim` / `SHIM` in them returned no matches, so that wiring sentence is ahead of this tree.
- Opened: `os.path.exists` for the five paths (PRESENT). `rg shim|SHIM` on those files returned no matches.

### 27. `CHANGELOG.md` lattice ancestor sentence

- False sentence: the rewritten bullet briefly said every merge commit in the earlier PR list had been ancestor-checked.
- Replacement: only the hashes actually checked are listed. #281 `65d3bbf1f3898ae5ef408570fce4bf688b13113b`, #282 `cc580302344f29149dd922447753b1d9701c4bb9`, #283 `7e8443f924593ba3afbef720fd3877f14715c1ab`, #287 `44ded13eafb2b29805ac8fd9743452f14e4b51d1`, and #288 `2ee132573b6b5ecc936be342000624b65b5e923f` were checked after the first ledger draft and are ancestors (exit 0). #258–#276 were not re-checked.
- Opened: `git merge-base --is-ancestor` for those five oids, exit 0.

### 28. `CHANGELOG.md` Model-Scope test path

- False sentence: `tests/test_model_scope_runtime.py` and `` `python tests/test_model_scope_runtime.py` ✅ `` as a path on this commit.
- Replacement: that path is absent. `test_model_scope_runtime.py` is at the repository root and was not re-run. `test_model_scope_steering.py` is present and was not re-run.
- Opened: `os.path.exists` for both paths.

### 29. `docs/ROADMAP_EXECUTION.md` example command and Phase II start line

- False sentence: the example runs `scripts/run_step_with_checks.sh` and `tests/test_model_scope_runtime.py` as if both exist, and "Starts after step 8" is the only status of Phase II.
- Replacement: those two paths are absent. `tests/test_check_block_flag.py` is present and was not re-run. Step 8's learning-loop test is absent, so Phase I is not complete, while the snapshot records Phase II rungs already on main.
- Opened: `os.path.exists` results above.

### 30. `docs/ROADMAP_EXECUTION.md` rung 10 status cell

- False sentence: `live routes overlap rung 16` can be read as rung 16 routes existing on this commit.
- Replacement: `docs/rung10-shim-substrate-dod.md` line 55 says introducing live routes is a future feature that overlaps rung 16, and rung 16 is not on this commit.
- Opened: that doc line 55. PR #295 state as in row 6.

### 31. `docs/ROADMAP_EXECUTION.md` rung 13 "this PR" tail

- False sentence: `Not introduced by this PR.`
- Replacement: the prune landed in #293 and is already on this commit. This docs edit does not add that code.
- Opened: #293 ancestor check. `evidence_dag.py:290` and `evidence_dag_disintegration.py:201`.

### 32. `README.md` section heading

- False sentence: `## Live Branch Status` as the name of this commit's status.
- Replacement: `## Status on this commit`.
- Opened: `git rev-parse HEAD` equals `49804ae3ddcebf4e4060fae95ba812331299f757`, which is `origin/main`.

### 33. `CHANGELOG.md` Unreleased heading and Validation (2026-06-06)

- False sentence: the Unreleased heading said the section was checked on `49804ae3ddcebf4e4060fae95ba812331299f757`. Under it, the Validation (2026-06-06) lines were `107 tests OK`, `2684 tests OK (10 skipped)`, and `python scripts/check_block_flag.py` PASS (CLEAR), readable as a run on that commit.
- Replacement: the heading names that commit as `origin/main` and says it is not a check that every sentence in the section was opened or re-run on that commit. The validation block is prefixed: those three lines are the 2026-06-06 log and were not re-run on that commit. The historical numbers are unchanged. This supersedes row 25's "the section is checked" wording.
- Opened: `CHANGELOG.md` line 5 and the validation command lines (lines 49–51 before the prefix). `git rev-parse origin/main` printed `49804ae3ddcebf4e4060fae95ba812331299f757`. `git merge-base --is-ancestor 49804ae3ddcebf4e4060fae95ba812331299f757 HEAD` exited 0 before this follow-up commit. The two unittest commands were not executed. After these edits, `python3 scripts/check_block_flag.py` printed CLEAR, carried-debt row count 0, and exited 0. That run is the PR gate. It is not the source of the dated PASS line.

### 34. `CHANGELOG.md` SHIM-CD-03 and SHIM-CD-07

- False sentence: the SHIM-CD-03 bullet stated a present-tense simulator claim, and the SHIM-CD-07 bullet stated a present-tense metric-lift claim. Those sentences were not re-opened.
- Replacement: each bullet now reads "not re-verified on this commit (`49804ae3ddcebf4e4060fae95ba812331299f757`). This note does not close the id and does not add a debt row." The earlier simulator wording and the earlier metric-lift wording are not repeated. Row 14's "keep their previous sentences" clause is superseded for these two ids only.
- Opened: `CHANGELOG.md` lines 37–38 before the rewrite. No simulator module and no metrics section were opened. `docs/next-session.md` Carried Debt status cells were not edited.

### 35. `docs/attnres-adapter-implementation-2026-05-04.md` paper year, five scores, and 1.25x sentence

- False sentence: the paper is dated 2025; GPQA-Diamond +7.5%, Mathematics +3.6%, HumanEval +3.1%, BBH +1.7%, and MMLU +1.1% are percent; and "Only 1.25× additional training compute vs. baseline."
- Replacement: Chen et al., Attention Residuals, 2026, arXiv:2603.15031. GPQA-Diamond 36.9 to 44.4 (+7.5 points), Math 53.5 to 57.1 (+3.6 points), HumanEval 59.1 to 62.2 (+3.1 points), BBH 76.3 to 78.0 (+1.7 points), MMLU 73.5 to 74.6 (+1.1 points). Separate scaling-law sentence: Block AttnRes matches the loss of a baseline trained with 1.25x more compute. That is not a claim that this run used 1.25x more training compute than the baseline. Benchmarks the file did not already list were not added.
- Opened: that file's source line and the benchmark list (lines 5 and 20–26 before the edit). MoonshotAI Attention Residuals README, read 2026-09-23, https://github.com/MoonshotAI/Attention-Residuals and https://raw.githubusercontent.com/MoonshotAI/Attention-Residuals/master/README.md. Scaling Laws paragraph: "Block AttnRes matches the loss of a baseline trained with 1.25x more compute." Downstream table: MMLU 73.5 / 74.6, GPQA-Diamond 36.9 / 44.4, BBH 76.3 / 78.0, Math 53.5 / 57.1, HumanEval 59.1 / 62.2. Citation bibtex `year = {2026}` and `eprint = {2603.15031}`. README prose says "+7.5 on GPQA-Diamond" and "+3.1 on HumanEval" with no percent sign.
