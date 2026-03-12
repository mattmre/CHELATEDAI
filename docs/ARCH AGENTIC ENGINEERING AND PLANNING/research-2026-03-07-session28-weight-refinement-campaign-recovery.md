# Research — Session 28 Weight-Refinement Campaign Recovery

## Context

Session 27 closed the remaining non-hardware implementation backlog and reduced the active work to:

1. RP2040 hardware evidence capture when a real device is available
2. the dated computational-storage retention review
3. evaluation and weight-refinement research on the retrieval stack

The next useful engineering work was not a new product feature. It was controlled evaluation of the already-shipped benchmark surfaces and elimination of experiment contamination risks.

## Research Questions

1. Are the benchmark CLIs evaluating the real retrieval engine, or are some still using dummy fallback behavior?
2. Can sequential benchmark configurations mutate the shared `adapter_weights.pt` checkpoint and contaminate later measurements?
3. Can the weight-refinement campaign resume after an interrupted phase without replaying all prior work?
4. Which completed results from the partial run are durable enough to document now, even though the full campaign did not finish?

## Code And Artifact Review

### Real evaluation wiring

- `benchmark_comparative.py` already had a comparative abstraction, but the CLI path did not build a real engine against the corpus.
- `benchmark_beir.py` could run dataset loops, but needed a CLI-exposed real engine factory and bounded query support so campaign runs would produce real retrieval metrics instead of synthetic fallback behavior.

### Cross-configuration contamination risk

- The repo keeps a shared root checkpoint at `adapter_weights.pt`.
- Benchmark modes that instantiate and train adapters in the same process can accidentally:
  - consume an existing root checkpoint when a clean run was intended
  - leave behind a mutated checkpoint that contaminates subsequent configurations
- This risk applied across:
  - comparative benchmark configurations
  - distillation baseline/offline/hybrid mode transitions
  - multitask loops
  - any resumed campaign phase that expected clean per-phase state

### Long-running campaign failure mode

- The clean run directory `experiment_runs/weight-refinement-20260306-session28-clean` completed:
  - Phase 1 standard sweep
  - Phase 2 distillation weight study
  - Phase 3 multitask suites
  - Phase 4 BEIR small
- It then stranded during `phase4_beir_medium`:
  - log output stopped mid-run
  - no `phase4_beir_medium.json` was written
  - no Phase 5 ablation output or final summary was produced
  - the manifest remained frozen before the interrupted phase

### Resume requirement

- Replaying the full campaign after an interruption is expensive and unnecessary when prior phase outputs already exist on disk.
- The runner therefore needed a native resume path that could:
  - trust completed output artifacts
  - rebuild manifest state from those artifacts
  - continue only the missing phases

## Findings

1. Benchmark hardening was justified by correctness, not optimization.
   The real risk was invalid or contaminated evaluation results.
2. Adapter isolation needed to be centralized.
   A shared helper is safer than ad hoc checkpoint cleanup in each benchmark.
3. Partial results from the completed phases are already informative enough to steer the next research iteration.
4. The current winning signal is narrow:
   the standard sweep found a SciFact improvement region, but the distillation, multitask, and BEIR results do not support promoting the broader chelation-heavy configurations yet.

## Durable Results Worth Preserving

- Phase 1 sweep winner:
  - `learning_rate=0.01`
  - `threshold=1`
  - `noise_scale=0.2`
  - `epochs=5`
  - SciFact `NDCG@10` improved from `0.6289` to `0.6766`
- Phase 2 distillation:
  - teacher weights `0.3`, `0.5`, and `0.7` all held mean `NDCG@10` at `0.7553`
  - offline pretraining cost remained material without observed retrieval gain in this bounded run
- Phase 3 multitask:
  - small suite mean `NDCG@10 = 0.6782`
  - medium suite mean `NDCG@10 = 0.6265`
  - mean learning gain remained `0.0`
  - mean stability remained `1.0`
- Phase 4 BEIR small:
  - `baseline` and `random_mask_50pct` tied for the best aggregate `mean_ndcg_at_10 = 0.6839`
  - `chelation` family variants clustered near `0.5745` to `0.5748`
  - `online_updates` was slower (`87.75 ms` mean latency) without quality improvement

## Decision

The benchmark infrastructure and the partial campaign results should be preserved in one PR:

- code changes that make future evaluation reproducible and resumable
- research/architecture docs that explain why those changes were necessary
- a durable analysis memo that records what the completed phases already tell us

The interrupted run itself should remain local evidence, not be committed as raw transient logs or Qdrant data.
