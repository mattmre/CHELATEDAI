# Phase I step 8 — learning loop (2026-09-23)

Phase I step 8's exit, from `docs/ROADMAP_EXECUTION.md`, is one test: ingest, sedimentation, and a measurable metric delta on a fixture corpus. This change adds `tests/test_learning_loop_e2e.py`. It is not a claim that Phase I is complete. Steps 1–4 were not re-verified here, and they are not declared done or not done. Rungs 15 and 16 were not touched.

The test ingests six documents into a real `AntigravityEngine` (`qdrant_location=":memory:"`, `use_centering=True`, `training_mode="baseline"`, `store_full_text_payload=True`), calls real `run_inference("query-seed")`, then real `run_sedimentation_cycle(threshold=1, learning_rate=0.5, epochs=8)`. The metric is the chelated id list (`run_inference` result index 1, the final ranking when centering is on) plus the L2 change of `embed("query-seed")`. A second test uses `epochs=0` on a fresh engine after the same ingest and one inference. That ranked-id list does not change, so a zero-epoch path fails the first test's inequality. The adapter asserted in both tests is `ChelationAdapter`. Qdrant is the real in-memory client. `qdrant.retrieve` is not patched.

`model_name` is `floor-dim8-fixture`, which does not start with `ollama:`, so `embed` takes the local path and applies the adapter. `create_embedding_backend` is patched to a dim-8 floor-tier stand-in: for text `i` in that call, component `i % 8` is set to 1.0 and then component 0 is set to 0.25. That stand-in is not MiniLM and not a BEIR encoder. `get_logger` is patched to `MagicMock`. `torch.manual_seed(0)` and `np.random.seed(0)` run before the constructor because `ChelationAdapter` initializes with `std=0.001` and the exact permutation depends on that draw. The assertions do not lock a permutation. They require the chelation log to be non-empty, the ranked ids to differ after eight epochs, a finite query-embedding L2 greater than 0, and both id lists to contain ids 0..5.

## Commands and numbers

Spark unittest, from this worktree, after the test file was in its final form:

```text
CUDA_VISIBLE_DEVICES= PYTHONPATH=/tmp/qc \
  /home/mattmre/research/CHELATEDAI-spark/repo/.venv/bin/python -u -m unittest tests.test_learning_loop_e2e -v
```

Python was 3.12.3. Exit 0. `Ran 2 tests in 2.414s` and `OK`. Printed lines:

```text
Created checkpoint: before_sedimentation_cycle_threshold_1_20260923_135440
Operation 'sedimentation_cycle_threshold_1' completed successfully
LEARNING_LOOP_POSITIVE ranked_before=[0, 1, 5, 3, 4, 2] ranked_after=[0, 1, 4, 2, 3, 5] l2=0.84740625 log_ids=[0, 1, 2, 3, 4, 5] log_counts=[1, 1, 1, 1, 1, 1] adapter=ChelationAdapter
LEARNING_LOOP_ZERO ranked_before=[0, 1, 5, 3, 4, 2] ranked_after=[0, 1, 5, 3, 4, 2]
```

The positive chelated ranking moved from `[0, 1, 5, 3, 4, 2]` to `[0, 1, 4, 2, 3, 5]`. Query-embedding L2 was `0.84740625`. Before sedimentation the log held ids 0..5 with one entry each. The zero-epoch test printed the same list twice and did not print `Created checkpoint` (epochs 0 returns before `SafeTrainingContext`). An earlier Spark run of the same assertions, before unused scout-id bindings were removed and without `-u`, also exited 0 and printed the same two `LEARNING_LOOP_*` lines (`Ran 2 tests in 2.162s`).

A handoff figure of `[0, 1, 2, 4, 5, 3]` to `[0, 1, 2, 3, 4, 5]` was not printed by these runs. Eight earlier unseeded draws on the same fixture, before the seed was fixed in the test, all changed the chelated list. The L2 values printed on those draws were `0.870574`, `0.728467`, `0.731247`, `0.804877`, `0.878045`, `0.706649`, `0.753084`, and `0.719285`. Those draws are not the unittest evidence above.

`.venv-egv` (`/home/mattmre/repos/CHELATEDAI-EGV/.venv-egv/bin/python`, also Python 3.12.3):

```text
/home/mattmre/repos/CHELATEDAI-EGV/.venv-egv/bin/python -c 'import numpy; print(numpy.__version__)'
```

printed `2.5.2`. `import torch` in that interpreter raised `ModuleNotFoundError: No module named 'torch'` (an `ImportError`). The unittest command with that interpreter failed at import:

```text
File "/home/mattmre/repos/CHELATEDAI-EGV-STEP8/antigravity_engine.py", line 11, in <module>
    import torch
ModuleNotFoundError: No module named 'torch'
```

`Ran 1 test` / `FAILED (errors=1)`. Nothing was installed into either venv.

Ruff is present at `/home/mattmre/research/CHELATEDAI-spark/repo/.venv/bin/ruff`. `ruff check tests/test_learning_loop_e2e.py` printed `All checks passed!`.

`python3 scripts/check_block_flag.py` from this worktree printed `Block flag state: CLEAR`, `Carried Debt row count: 0`, `RESULT: PASS`, exit 0, once before this note existed and again after the roadmap sentence was saved. `docs/next-session.md` was not edited.

An import check under `PYTHONPATH=/tmp/qc` on the Spark interpreter printed `numpy 2.5.3` from `/tmp/qc/numpy/__init__.py` and `torch 2.13.0+cu130`. `/tmp/qc` is where `qdrant_client` was imported from (`qdrant_client-1.19.1.dist-info` is in that directory). The unittest command used that `PYTHONPATH`.

## Why this is floor-tier

The encoder is the dim-8 fixture above, not MiniLM and not a BEIR run. No BEIR dataset, query set, or official retrieval metric was executed. Ceiling BEIR was not run. `scripts/smoke.sh` was not run. `.venv-egv` cannot import the engine because torch is missing, so this loop was not run there. In-memory Qdrant is not a live cross-host retrieval deployment.

## Cwd checkpoint hazard

`CheckpointManager` defaults to `Path("./checkpoints")` and creates that directory in its constructor. `AntigravityEngine.__init__` constructs a `CheckpointManager`, and `run_sedimentation_cycle` writes a checkpoint through `SafeTrainingContext` before it trains. The adapter file is separate: `ChelationConfig.ADAPTER_WEIGHTS_PATH` is `PROJECT_ROOT / "adapter_weights.pt"` (absolute, under this repo), and sedimentation saves the real adapter there.

The test changes the process cwd to a `TemporaryDirectory` before `AntigravityEngine()` is constructed, which is before sedimentation, so `./checkpoints` is created in that temp directory. `benchmark_utils.isolated_adapter_state()` hides and then deletes the repo adapter file for the duration of the run, restoring a previous file if one was present. After the Spark unittest, `git status --porcelain` showed only the untracked test file. `checkpoints`, `adapter_weights.pt`, and `chelation_events.jsonl` were not in the repo root. Nothing else in the repo was deleted.

## Docs not edited

`README.md` and `CHANGELOG.md` still say `tests/test_learning_loop_e2e.py` is absent. They were left unchanged on purpose. The only roadmap edit is the sentence in `docs/ROADMAP_EXECUTION.md` that used to say the test is not in this commit. That sentence now says the test is added by this change, that it fails when the ranked-id list does not change, and that it does not use a production encoder. It still says Phase I is not complete. The carried-debt table was not edited.
