# Weight Refinement Follow-up Results (Session 33)

**Primary run directories:**  
- `experiment_runs/repeatability-20260425-085209-212460-session33-mlp-tw03`  
- `experiment_runs/repeatability-20260425-122553-342045-multitask-small-session33-candidate-small`

**Follow-up date:** 2026-04-25

## Status

This focused follow-up is complete.

Completed:

- stale-roadmap cleanup review fix on PR `#111`
- focused repeatability helper and hardening on PR `#112`
- canonical SciFact repeatability rerun for `mlp` + `teacher_weight=0.3`
- candidate-specific transfer gate tooling via `run_candidate_transfer_gate.py`
- candidate-specific small multitask gate (`SciFact` reused, `NFCorpus` rerun)

Not completed intentionally:

- candidate-specific medium multitask gate
- candidate-specific BEIR gate
- large sweep expansion
- RP2040 hardware evidence capture

Those remaining phases were stopped because the corrected small multitask gate failed.

## Tooling Correction

During the follow-up, the existing generic transfer scripts were audited before using them as promotion evidence:

- `benchmark_multitask.py` does not evaluate the specific Session 33 distillation candidate
- `benchmark_beir.py` evaluates the generic comparative configuration matrix rather than the repeatability candidate

Because of that, the earlier generic multitask and BEIR outputs from this session are useful only as auxiliary repo-level benchmarking. They are **not** valid promotion evidence for the `mlp` + `teacher_weight=0.3` candidate.

To correct that gap, this session added `run_candidate_transfer_gate.py`, which reuses the exact repeatability candidate configuration and aggregates per-task `benchmark_distillation.py` results into a transfer-gate summary.

## Focused Repeatability Result

Candidate:

- adapter: `mlp`
- teacher: `sentence-transformers/all-mpnet-base-v2`
- teacher weight: `0.3`
- cycles / queries / epochs: `5 / 50 / 5`
- learning rate: `0.01`

SciFact repeatability outcome:

- baseline final `NDCG@10`: `0.6012277113567815`
- hybrid final `NDCG@10`: `0.6238711568667803`
- hybrid gain: `+0.02264344550999875`
- repeatability gate: **pass**

Interpretation:

- The only locally positive Session 32 candidate did reproduce on a clean second SciFact run.
- That was enough to justify one candidate-specific transfer check, but not a preset update by itself.

## Candidate-Specific Small Multitask Gate

Scope:

- `SciFact` reused from the canonical repeatability artifact
- `NFCorpus` rerun with the exact same candidate settings

Per-task outcome:

| task | baseline final NDCG@10 | hybrid final NDCG@10 | gain | gate |
| --- | ---: | ---: | ---: | --- |
| `SciFact` | `0.6012` | `0.6239` | `+0.0226` | pass |
| `NFCorpus` | `0.4893` | `0.4847` | `-0.0046` | fail |

Aggregate:

- mean baseline final `NDCG@10`: `0.545275298457266`
- mean hybrid final `NDCG@10`: `0.5542922980982423`
- mean hybrid gain: `+0.009016999640976353`
- positive gains: `1`
- non-positive gains: `1`
- transfer gate: **fail**

Important detail:

- The aggregate mean stayed slightly positive only because the reused SciFact result was strong.
- The strict task-level gate rejected the candidate because `NFCorpus` regressed relative to its frozen baseline.

`NFCorpus` detailed outcome:

- baseline final `NDCG@10`: `0.4893228855577504`
- offline final `NDCG@10`: `0.09563766566662998`
- hybrid final `NDCG@10`: `0.48471343932970434`
- hybrid gain: `-0.0046094462280460435`

## Explicit Outcome

**No preset promotion from Session 33 follow-up.**

Why the candidate stopped here:

1. Repeatability was satisfied on SciFact.
2. Cross-dataset preservation was **not** satisfied on the corrected small multitask gate.
3. `NFCorpus` regressed under the exact candidate that had looked positive on SciFact.
4. Because the first corrected transfer gate already failed, medium multitask and BEIR expansion were not justified.

## Recommended Next Research Step

If additional work is still desired, do **not** continue medium multitask or BEIR for this same candidate.

Instead:

1. Keep current presets and defaults unchanged.
2. Keep `run_candidate_transfer_gate.py` as the required transfer-evidence path for future single-candidate follow-ups.
3. Start any future evaluation from a new candidate or a materially changed hypothesis rather than re-running this same `mlp` + `teacher_weight=0.3` path deeper into the transfer stack.
