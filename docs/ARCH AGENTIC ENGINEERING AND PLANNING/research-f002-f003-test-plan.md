# Research: Test Plan for F002 (benchmark_rlm.py) and F003 (checkpoint_manager.py)

**Date:** 2026-02-13
**Researcher:** Researcher Agent
**Status:** Complete

## Overview

Two files with zero test coverage need test suites:
1. `benchmark_rlm.py` -- Pure metric functions and dict search helpers (F002)
2. `checkpoint_manager.py` -- CheckpointManager class and SafeTrainingContext (F003)

## Part 1: benchmark_rlm.py (~15 tests)

### dcg_at_k(r, k) -- Pure numpy function
- Empty list -> 0.0
- k=0 -> 0.0
- [1] at k=1 -> 1.0 (1/log2(2))
- [1,1,1,1,1] at k=5 -> 2.9485
- [1,0,1,0,0] at k=5 -> 1.5
- [3,2,1] at k=3 -> 4.7619
- k > len(r) uses all elements

### ndcg_at_k(r, k) -- Calls dcg_at_k internally
- All zeros -> 0.0
- Perfect ranking -> 1.0
- [0,0,1] at k=3 -> 0.5
- [1,0,1] at k=3 -> 0.9197
- [1,2,3] at k=3 -> 0.7901

### find_keys(obj, target_keys) -- Recursive dict search
- Non-dict -> None
- Keys at top level -> returns that dict
- Keys nested -> finds them
- Keys split across levels -> None

### find_payload(obj, key) -- Recursive value search
- Top-level key -> value
- Nested key -> value
- Missing key -> None
- KNOWN BUG: falsy nested values return None (line 66: `if res:` not `if res is not None:`)

### map_predicted_ids(engine, pred_ids) -- Needs MagicMock engine
- With original_id payload -> mapped strings
- Without original_id -> str(point.id) fallback
- Exception -> str(pid) fallback

## Part 2: checkpoint_manager.py (~17 tests)

### CheckpointManager
- create_checkpoint: returns ID, copies file, saves metadata, tracks hash
- restore_checkpoint: latest, specific ID, invalid ID, hash mismatch warning
- delete_checkpoint: existing, nonexistent, updates latest pointer
- cleanup_old_checkpoints: keeps N most recent, noop under limit

### SafeTrainingContext
- Success: mark_success() prevents rollback
- Failure: exception triggers rollback
- Failure: no mark_success() triggers rollback
- auto_rollback=False: no rollback on failure

## Conventions
- unittest-style, no pytest classes
- Temp dirs via tempfile.mkdtemp(), cleanup in tearDown
- assertAlmostEqual for floats (places=3)
- MagicMock for engine in map_predicted_ids
- No logger mocking needed (neither file uses get_logger)

## Numerical Reference Values
```
log2(2)=1.0, log2(3)=1.585, log2(4)=2.0, log2(5)=2.322, log2(6)=2.585
dcg([1,1,1,1,1],5) = 2.9485
dcg([1,0,1,0,0],5) = 1.5
dcg([3,2,1],3) = 4.7619
ndcg([0,0,1],3) = 0.5
ndcg([1,0,1],3) = 0.9197
ndcg([1,2,3],3) = 0.7901
```
