# Road-Course Results And Default Threshold Decision

This run used the smallest standard local embedding model path already used by the repository: `sentence-transformers/all-MiniLM-L6-v2`.

## Campaigns run

### BEIR quick road-course

Command:

```bash
python benchmark_beir.py --tier quick --max-queries 20 --model sentence-transformers/all-MiniLM-L6-v2 --output experiment_runs\roadcourse-small\beir_quick_minilm_20.json
```

Dataset: SciFact, 5,183 documents, first 20 evaluated queries.

| Profile | NDCG@10 | MAP@10 | MRR | Recall@10 | Mean latency ms | Outcome |
|---|---:|---:|---:|---:|---:|---|
| baseline | 0.6745 | 0.6264 | 0.6458 | 0.8000 | 22.50 | best |
| random_mask_50pct | 0.6745 | 0.6264 | 0.6458 | 0.8000 | 32.25 | tied quality, slower |
| chelation | 0.4928 | 0.4188 | 0.4271 | 0.6750 | 22.47 | rejected |
| chelation+tempscale_0.5 | 0.4928 | 0.4188 | 0.4271 | 0.6750 | 23.72 | rejected |
| chelation+tempscale_2.0 | 0.4928 | 0.4188 | 0.4271 | 0.6750 | 27.87 | rejected |
| procrustes | 0.4928 | 0.4188 | 0.4271 | 0.6750 | 29.05 | rejected |
| low_rank_16 | 0.4928 | 0.4188 | 0.4271 | 0.6750 | 28.87 | rejected |
| online_updates | 0.4928 | 0.4188 | 0.4271 | 0.6750 | 32.80 | rejected |

Conclusion: always-on centered chelation is not safe as a default on this small-model SciFact road course.

### Targeted profile grid

Command pattern:

```bash
python run_road_course_campaign.py --task SciFact --max-queries 20 --sample-docs 1200 --seed <seed> --output experiment_runs\roadcourse-small\roadcourse_profile_grid_q20_docs1200_seed<seed>.json
```

Seeds: 42, 43, 44.

| Profile | Mean NDCG@10 | Min | Max | Action mix | Decision |
|---|---:|---:|---:|---|---|
| baseline | 0.8332 | 0.8166 | 0.8421 | FAST | best/tied |
| adaptive_p85_t0.01 | 0.8332 | 0.8166 | 0.8421 | FAST | best/tied |
| fast_guard_p85_t999 | 0.8332 | 0.8166 | 0.8421 | FAST | best/tied |
| adaptive_p50_t0.0004 | 0.7318 | 0.7175 | 0.7490 | CHELATE | rejected |
| adaptive_p75_t0.0004 | 0.7318 | 0.7175 | 0.7490 | CHELATE | rejected |
| adaptive_p85_t0.0004 | 0.7318 | 0.7175 | 0.7490 | CHELATE | rejected |
| adaptive_p95_t0.0004 | 0.7318 | 0.7175 | 0.7490 | CHELATE | rejected |
| adaptive_p85_t0.001 | 0.7318 | 0.7175 | 0.7490 | CHELATE | rejected |
| centered_p85_temp1 | 0.7318 | 0.7175 | 0.7490 | CHELATE_ALWAYS | rejected |
| centered_p50_temp1 | 0.7318 | 0.7175 | 0.7490 | CHELATE_ALWAYS | rejected |

Conclusion: the previous `0.0004` adaptive threshold is too aggressive for this model/dataset. A `0.01` guard threshold preserves baseline quality by staying on the FAST path for this road-course distribution.

### First-hundred adaptive tuning loops

Command pattern:

```bash
python run_road_course_tuning_loop.py --task <task> --max-queries 100 --sample-docs 1200 --seed <seed> --rounds 2 --initial-grid quick --output experiment_runs\roadcourse-small\<task>_hundred_tuning_loop_seed<seed>_docs1200.json
```

Runs completed:

| Task | Seed | Queries | Corpus docs | Round-1 baseline NDCG@10 | Round-1 aggressive `0.0004` NDCG@10 | Round-2 guard range | Recommendation |
|---|---:|---:|---:|---:|---:|---|---|
| SciFact | 42 | 100 | 1200 | 0.82 | 0.79 | `0.005-0.02` tied baseline | keep baseline / `0.01` guard |
| SciFact | 43 | 100 | 1200 | 0.82 | 0.77 | `0.005-0.02` tied baseline | keep baseline / `0.01` guard |
| NFCorpus | 42 | 100 | 2063 | 0.62 | 0.56 | `0.005-0.02` tied baseline | keep baseline / `0.01` guard |

Interpretation:

- The monitoring numbers are stable across two SciFact seeds and one transfer task.
- The old aggressive `0.0004` threshold causes active `CHELATE` on every query and regresses NDCG@10 by about `0.03-0.07`.
- Always-centered chelation follows the same regression pattern.
- Safer guard thresholds (`0.005`, `0.01`, `0.02`) stay on the FAST path and preserve baseline quality, but do not produce lift.
- No default/profile promotion is justified from these loops. The correct setting decision remains: keep the conservative `0.01` threshold guardrail and keep aggressive chelation opt-in only.

### Module-aware first-hundred tuning loops

Command pattern:

```bash
python run_road_course_tuning_loop.py --task <task> --max-queries 100 --sample-docs 1200 --seed <seed> --rounds <rounds> --initial-grid modules --output experiment_runs\roadcourse-small\<task>_hundred_module_tuning_seed<seed>_docs1200.json
```

Runs completed:

| Task | Seed | Rounds | Queries | Corpus docs | Baseline NDCG@10 | Best module profile | Active/centered chelation | Quantization gate | Recommendation |
|---|---:|---:|---:|---:|---:|---|---|---|---|
| SciFact | 44 | 2 | 100 | 1200 | 0.8242 | baseline / reformulation / guard ties | `0.7600` (`-0.0642`) | fail: no FP32 gain; quantized below baseline | keep baseline / guarded FAST path |
| NFCorpus | 43 | 1 | 100 | 1200 | 0.6205 | baseline / reformulation / guard ties | `0.5554` (`-0.0651`) | fail: no FP32 gain | keep baseline / guarded FAST path |

Module interpretation:

- Query reformulation (`reform_v2`, `reform_v3`) and guard-plus-reformulation profiles are now included in the adaptive tuning loop and exercised through the `REFORMULATE` path on every evaluated query.
- Reformulation is wiring-positive but quality-neutral on these first-hundred runs: it preserves baseline NDCG/MRR/Recall but does not create measurable lift.
- Guarded quantization at threshold `0.01` remains quality-neutral and stays on the FAST path.
- Temperature-centered profiles (`temp0.5`, `temp2`) match the active chelation regression, so temperature scaling does not rescue always-centered chelation on these fixtures.
- The quantization promotion gate now fails closed when there is no FP32 lift, even if a quantized score is slightly above baseline; no profile has evidence for promotion.

### Calibrated actuator first-hundred loops

The next loop addressed the "does it work?" question directly. Two actuator gaps were fixed before running it:

- Query reformulation now uses reciprocal-rank fusion across all generated variants instead of stopping after the original query fills the top 10.
- Active chelation reranking now applies the `chelation_p` percentile mask during centered reranking, so percentile changes alter mask density instead of being telemetry-only.

Command pattern:

```bash
python run_road_course_tuning_loop.py --task <task> --max-queries 100 --sample-docs 1200 --seed <seed> --rounds <rounds> --initial-grid calibrated --output experiment_runs\roadcourse-small\<task>_hundred_calibrated_tuning_seed<seed>_docs1200.json
```

Runs completed:

| Task | Seed | Rounds | Baseline NDCG@10 | Best active candidate | Reformulation result | Threshold transition | Recommendation |
|---|---:|---:|---:|---|---|---|---|
| SciFact | 45 | 2 | 0.8332 | `adaptive_p85_t0.002`: 0.8324 (`-0.0008`) | `reform_rrf_v2`: 0.8296 (`-0.0037`), changed 78/100 rankings | `0.0015` mostly CHELATE; `0.002` mixed 9/100 CHELATE; `0.003+` FAST | keep baseline / guarded FAST path |
| NFCorpus | 44 | 1 | 0.6205 | `adaptive_p95_t0.002`: 0.5991 (`-0.0214`) | `reform_rrf_v2`: 0.6164 (`-0.0041`), changed 38/100 rankings | `0.0015` mostly CHELATE; `0.002` mixed 12/100 CHELATE; `0.003+` FAST | keep baseline / guarded FAST path |

Actuator interpretation:

- The threshold control works and the useful transition band for these slices is around `0.002-0.003`; below that, chelation fires often and harms retrieval; above that, the profile becomes equivalent to baseline.
- The percentile control now works mechanically: at threshold `0.002`, mask density changes with `p` (`p50` masks more dimensions than `p85`, `p95` masks fewer). However, every active percentile still regressed NDCG.
- Reformulation now works mechanically: reciprocal-rank fusion changed 78/100 SciFact rankings and 38/100 NFCorpus rankings for `v2`. Those changes reduced quality, so reformulation should stay advisory/diagnostic rather than a promoted retrieval path.
- `reform_rrf_v3` is too aggressive on SciFact (`-0.0504`) and should be treated as rejected for this small-model road course.
- Temperature scaling remains a non-promotable control for ranking because positive scalar scaling of centered cosine scores is monotonic; it cannot rescue the centered-chelation ordering by itself.
- Quantization promotion still fails because the best candidate has no FP32 lift, even when simulated quantized embeddings are numerically above baseline. This is the intended fail-closed behavior.

### Adaptive 1,000-query cycle with 50-query checkpoints

Command:

```bash
python run_thousand_query_tuning.py --loop-queries 200 --window-queries 50 --sample-docs 400 --output experiment_runs\roadcourse-small\adaptive_thousand_query_tuning.json
```

Cycle design:

- Five loops of 200 judged queries each: SciFact, NFCorpus, and three FiQA2018 windows.
- Twenty validation windows total, each with 50 queries.
- Each window evaluated a compact adaptive profile set: baseline, conservative guard, one active-chelation probe, and one reformulation/combined probe.
- The next window's active probe was selected from prior action mix and delta behavior: strong active regression moves toward safer thresholds; small active regression probes percentile changes; reformulation is retested/combined only as evidence warrants.

Aggregate outcomes:

| Profile | Windows | Mean delta vs baseline | Best | Worst | Improved / tied / regressed | Main interpretation |
|---|---:|---:|---:|---:|---|---|
| `baseline` | 20 | 0.0000 | 0.0000 | 0.0000 | 0 / 20 / 0 | reference |
| `guard_p85_t0.01` | 20 | 0.0000 | 0.0000 | 0.0000 | 0 / 20 / 0 | safe no-op FAST guard |
| `adaptive_p85_t0.003` | 4 | 0.0000 | 0.0000 | 0.0000 | 0 / 4 / 0 | mostly/entirely FAST, no lift |
| `adaptive_p95_t0.002` | 2 | 0.0005 | 0.0011 | 0.0000 | 1 / 1 / 0 | small positive directional signal, too sparse |
| `adaptive_p85_t0.002` | 5 | 0.0033 | 0.0181 | -0.0071 | 2 / 1 / 2 | FiQA-positive, cross-task unstable |
| `adaptive_p85_t0.002_reform_rrf_v2` | 11 | -0.0037 | 0.0124 | -0.0335 | 3 / 2 / 6 | occasional FiQA lift, too many regressions |
| `reform_rrf_v2` | 9 | -0.0083 | 0.0062 | -0.0247 | 2 / 1 / 6 | SciFact-positive windows, FiQA-negative overall |
| `adaptive_p85_t0.0015` | 4 | -0.0090 | 0.0145 | -0.0339 | 2 / 0 / 2 | too aggressive/unstable |
| `adaptive_p50_t0.002` | 5 | -0.0122 | 0.0119 | -0.0424 | 1 / 1 / 3 | p50 mask too destructive overall |

Per-task directional signal:

- SciFact: `reform_rrf_v2` was positive in the two windows where it was selected (mean `+0.0041`), while active chelation remained negative or neutral.
- NFCorpus: no adaptive profile produced useful lift; guard/baseline remained best.
- FiQA2018: `adaptive_p85_t0.002` had the clearest directional lift (2 windows, mean `+0.0128`, best `+0.0181`), but combined reformulation/chelation and lower thresholds were unstable.

Decision:

- No default or general profile promotion. The result is directional, not promotable.
- The next candidate worth isolating is a task- or confidence-gated `adaptive_p85_t0.002` branch for FiQA-like windows, with repeatability and transfer gates.
- `reform_rrf_v2` should be treated as query/task-conditional only; it can help SciFact windows but regresses FiQA and should not be globally enabled.
- Guard threshold `0.01` remains the safe default because it preserves baseline by staying FAST.

### Adaptive 5,000-query phase with 50-query checkpoints

Command:

```bash
python run_thousand_query_tuning.py --phase-queries 5000 --loop-queries 200 --window-queries 50 --sample-docs 250 --base-seed 60 --output experiment_runs\roadcourse-small\adaptive_fivek_query_tuning.json
```

Phase design:

- 5,000 judged query evaluations across 25 loops and 100 validation windows.
- The available judged queries in SciFact, NFCorpus, and FiQA2018 are fewer than 5,000 unique questions, so this phase cycles query windows with different corpus-sampling seeds. Treat this as repeated robustness probing, not 5,000 unique qrels.
- The runner is task-aware: each new window adapts from same-task history when available, so SciFact/NFCorpus/FiQA do not blindly inherit the prior task's setting.
- The optimized runner reuses per-profile engines inside each 200-query loop while preserving 50-query checkpoint decisions.

Aggregate outcomes:

| Profile | Windows | Mean delta vs baseline | Best | Worst | Improved / tied / regressed | Main interpretation |
|---|---:|---:|---:|---:|---|---|
| `baseline` | 100 | 0.0000 | 0.0000 | 0.0000 | 0 / 100 / 0 | reference |
| `guard_p85_t0.01` | 100 | 0.0000 | 0.0000 | 0.0000 | 0 / 100 / 0 | safe no-op FAST guard |
| `adaptive_p85_t0.0025` | 15 | 0.0000 | 0.0000 | 0.0000 | 0 / 15 / 0 | too high; becomes FAST/no-op |
| `adaptive_p85_t0.003` | 14 | 0.0000 | 0.0000 | 0.0000 | 0 / 14 / 0 | too high; becomes FAST/no-op |
| `adaptive_p50_t0.002` | 10 | -0.0012 | 0.0000 | -0.0087 | 0 / 8 / 2 | mostly no-op, occasional harm |
| `adaptive_p85_t0.002` | 41 | -0.0024 | 0.0181 | -0.0325 | 17 / 4 / 20 | repeat FiQA lift, unstable globally |
| `adaptive_p85_t0.002_reform_rrf_v2` | 50 | -0.0038 | 0.0126 | -0.0281 | 9 / 11 / 30 | FiQA-positive, cross-task damaging |
| `adaptive_p90_t0.002` | 8 | -0.0054 | 0.0015 | -0.0192 | 1 / 3 / 4 | no useful global signal |
| `adaptive_p85_t0.0025_reform_rrf_v2` | 9 | -0.0068 | 0.0004 | -0.0203 | 0 / 3 / 6 | rejected |
| `adaptive_p95_t0.002` | 12 | -0.0079 | 0.0074 | -0.0299 | 5 / 1 / 6 | SciFact occasional lift, NFCorpus damage |
| `reform_rrf_v2` | 41 | -0.0079 | 0.0015 | -0.0247 | 1 / 11 / 29 | generally harmful |

Task-specific signal:

- SciFact: no robust winner. `adaptive_p90_t0.002` had only a tiny mean positive delta (`+0.0004`) and `adaptive_p95_t0.002` had occasional windows up to `+0.0074`, but both are too small/sparse for promotion.
- NFCorpus: no useful adaptive path. Guard/baseline stayed best; active chelation and reformulation mostly tied or regressed.
- FiQA2018: the only apparent prospect in this first phase. `adaptive_p85_t0.002_reform_rrf_v2` averaged `+0.0076` over 9 FiQA windows with 6 improvements and 3 regressions. `adaptive_p85_t0.002` produced 12 positive FiQA windows but still had 5 regressions, with mean `-0.0003` over all FiQA selections because the negative windows were larger. The confirmation phase below retested this branch more aggressively and rejected it.

Decision:

- No global promotion.
- `guard_p85_t0.01` remains the default-safe setting because it exactly preserves baseline behavior.
- `reform_rrf_v2`, `adaptive_p50_t0.002`, `adaptive_p85_t0.0025_reform_rrf_v2`, `adaptive_p90_t0.002`, and global `adaptive_p95_t0.002` should be treated as rejected for broad use.
- The only next research branch worth isolating from this pass was a FiQA-like, confidence-gated candidate around `adaptive_p85_t0.002` and `adaptive_p85_t0.002_reform_rrf_v2`. The confirmation phase below retested that hypothesis and found the signal did not hold.

### Confirmatory 5,000-query FiQA-focused phase

Command:

```bash
python run_thousand_query_tuning.py --phase-queries 5000 --loop-queries 200 --window-queries 50 --sample-docs 250 --base-seed 80 --strategy confirm_fiqa --output experiment_runs\roadcourse-small\adaptive_fivek_confirm_fiqa_tuning.json
```

Confirmation design:

- 5,000 judged query evaluations across another 25 loops and 100 validation windows.
- `--strategy confirm_fiqa` forces every FiQA2018 window to retest `adaptive_p85_t0.002` and `adaptive_p85_t0.002_reform_rrf_v2` instead of allowing the adaptive selector to drift away from the candidate.
- SciFact and NFCorpus windows continue to challenge generalization with alternating reformulation or active-chelation probes.
- This is still repeated robustness probing over the available judged query windows, not 5,000 unique qrels.

Aggregate outcomes:

| Profile | Windows | Mean delta vs baseline | Best | Worst | Improved / tied / regressed | Main interpretation |
|---|---:|---:|---:|---:|---|---|
| `baseline` | 100 | 0.0000 | 0.0000 | 0.0000 | 0 / 100 / 0 | reference |
| `guard_p85_t0.01` | 100 | 0.0000 | 0.0000 | 0.0000 | 0 / 100 / 0 | safe FAST/no-op guard |
| `adaptive_p85_t0.002` | 52 | -0.0039 | 0.0181 | -0.0325 | 22 / 4 / 26 | active control works mechanically, but mean quality is negative |
| `adaptive_p85_t0.002_reform_rrf_v2` | 36 | -0.0108 | 0.0126 | -0.0345 | 12 / 0 / 24 | forced FiQA retest rejects the earlier combined-signal hypothesis |
| `adaptive_p95_t0.002` | 16 | -0.0026 | 0.0109 | -0.0170 | 5 / 1 / 10 | sparse SciFact lift, not robust |
| `reform_rrf_v2` | 32 | -0.0022 | 0.0018 | -0.0110 | 3 / 17 / 12 | reformulation changes many rankings but does not improve them reliably |

Task-specific confirmation:

- FiQA2018: `adaptive_p85_t0.002` averaged `-0.0022` over 36 FiQA windows with 18 improvements and 18 regressions. `adaptive_p85_t0.002_reform_rrf_v2` averaged `-0.0108` over 36 FiQA windows with 12 improvements and 24 regressions. The prior positive combined branch did not survive confirmation.
- NFCorpus: still baseline/guard-only. `adaptive_p85_t0.002` averaged `-0.0076`, and reformulation averaged `-0.0034`.
- SciFact: no robust candidate. `adaptive_p95_t0.002` averaged `-0.0026`; reformulation averaged `-0.0009`.

Confirmation decision:

- No global promotion.
- No FiQA-specific route promotion from these hand-tuned inference controls.
- `guard_p85_t0.01` remains the only safe default-preserving profile because it stays on the FAST path and ties baseline exactly.
- The implementation answer to "does it work?" is now split: the actuators work and move rankings (`CHELATE`/`REFORMULATE` action mixes, mask densities, and reformulation change counts are non-zero), but the tested untrained inference-time controls usually move rankings in the wrong direction.
- The next improvement path should shift away from hand-tuned active chelation/reformulation thresholds and toward trained or learned route candidates with explicit promotion gates.

### Fault-classification implementation for future introspection

The tuning summaries now classify every evaluated profile/window so future artifacts can answer whether a module is inactive, active-positive, active-negative, or suspicious:

| Classification | Meaning | Promotion impact |
|---|---|---|
| `reference` | baseline row | neutral |
| `no_op_tied` | no ranking-changing actuator and metric ties baseline | safe sentinel, not an improvement |
| `metric_changed_without_actuator` | metric moved while diagnostics report no active control | high-priority implementation/instrumentation fault |
| `actuator_active_positive` | ranking-changing control improved the window | candidate evidence only |
| `actuator_active_negative` | ranking-changing control regressed the window | promotion blocker |
| `actuator_active_neutral` | ranking-changing control moved results without meaningful metric delta | hold/retest |

Implementation surfaces:

- `run_road_course_tuning_loop.profile_summary()` adds `fault_classification`, `fault_counts`, and `promotion_blockers` to each round summary.
- `run_thousand_query_tuning.summarize_profile_outcomes()` aggregates `fault_counts` across 1,000/5,000-query phases.
- This turns future runs from score-only sweeps into fault attribution: a profile that ties baseline because it stayed FAST is separated from a profile that actively changed rankings and failed.

Improvement guideline from the introspection pass:

- Treat fixed-threshold active chelation and fixed deterministic reformulation as diagnostic probes, not promotion candidates.
- Prioritize learned or evaluated controls: qrels-aware mask learning, variant confidence/weighting before RRF, route effectiveness based on NDCG/MRR deltas, and campaign gates that stop repeatedly negative active profiles.
- Promotion candidates must show `actuator_active_positive` counts without repeated `actuator_active_negative` blockers, then pass repeatability, transfer, quantization-retained-gain, latency, and norm/health gates.

### Fault-aware 5,000-query golden-setting search

Command:

```bash
python run_thousand_query_tuning.py --phase-queries 5000 --loop-queries 200 --window-queries 50 --sample-docs 250 --base-seed 90 --strategy fault_aware --output experiment_runs\roadcourse-small\adaptive_fivek_fault_aware_tuning.json
```

Search design:

- 5,000 judged query evaluations across 25 loops and 100 validation windows.
- `--strategy fault_aware` uses the new fault classifications to rotate away from repeated active-negative branches and test less destructive `p99` chelation probes.
- The run still keeps `baseline` and `guard_p85_t0.01` as sentinels in every window.
- The runner recommendation now reports `default_promotable_profiles` and `golden_candidate_profiles`; a golden profile requires cross-task coverage, meaningful mean lift, no regressions, and no active-negative or suspicious metric-movement faults.

Aggregate outcomes:

| Profile | Windows | Mean delta vs baseline | Best | Worst | Improved / tied / regressed | Fault interpretation |
|---|---:|---:|---:|---:|---|---|
| `baseline` | 100 | 0.0000 | 0.0000 | 0.0000 | 0 / 100 / 0 | reference |
| `guard_p85_t0.01` | 100 | 0.0000 | 0.0000 | 0.0000 | 0 / 100 / 0 | safe no-op sentinel |
| `adaptive_p85_t0.0025` | 51 | 0.0000 | 0.0005 | 0.0000 | 0 / 51 / 0 | no-op/tied; not an improvement |
| `adaptive_p99_t0.0015` | 19 | -0.0024 | 0.0263 | -0.0441 | 9 / 2 / 8 | strongest positive windows, but unstable and promotion-blocked |
| `reform_rrf_v2` | 33 | -0.0053 | 0.0037 | -0.0318 | 3 / 13 / 17 | mostly active-negative or neutral |
| `adaptive_p85_t0.002` | 5 | -0.0016 | 0.0124 | -0.0185 | 2 / 0 / 3 | insufficient and blocked by regressions |
| `adaptive_p95_t0.002` | 3 | -0.0067 | 0.0109 | -0.0193 | 1 / 0 / 2 | insufficient and blocked by regressions |
| `adaptive_p99_t0.002` | 4 | -0.0105 | 0.0000 | -0.0292 | 0 / 1 / 3 | rejected |
| `adaptive_p99_t0.0025` | 10 | 0.0000 | 0.0000 | 0.0000 | 0 / 10 / 0 | no-op/tied; not an improvement |

Task-specific findings:

- SciFact: `adaptive_p99_t0.0015` produced the largest observed positive window (`+0.0263`) and 9 positive windows, but also 7 active-negative windows and a worse negative tail (`-0.0441`), so it is not shippable. `reform_rrf_v2` was slightly positive on average (`+0.0006`) but too small and still had an active-negative blocker.
- FiQA2018: no candidate repeated the earlier weak signal. `adaptive_p85_t0.002` averaged `-0.0016`; `reform_rrf_v2` averaged `-0.0099`.
- NFCorpus: remained baseline/guard-only. Active candidates either tied by staying FAST or regressed.

Golden-setting decision:

- `golden_candidate_profiles`: none.
- `default_promotable_profiles`: none.
- No dramatic default-safe improvement was found.
- The best information from this run is diagnostic: `adaptive_p99_t0.0015` can create large positive SciFact windows, but the same mechanism creates larger negative windows. This is exactly the pattern that argues for a learned/query-conditional mask gate rather than a global threshold default.
- `guard_p85_t0.01` remains the only default-safe profile, but it is a preservation guard, not an improvement.

### Gate-learning 5,000-query campaign

Command:

```bash
python run_thousand_query_tuning.py --phase-queries 5000 --loop-queries 200 --window-queries 50 --sample-docs 250 --base-seed 100 --strategy gate_learning --output experiment_runs\roadcourse-small\adaptive_fivek_gate_learning_tuning.json
```

Implementation and search design:

- Added `--strategy gate_learning` to keep the high-upside `adaptive_p99_t0.0015` probe visible while rotating supporting active/reformulation probes by task and prior fault history.
- Added `gate_feature_rows` to the artifact: one flattened row per profile/window with task, profile, delta, fault class, action counts, variance, Jaccard, mask density, and reformulation-change diagnostics.
- Added `gate_candidate_report` to build simple post-hoc diagnostic subsets such as task-only gates, Jaccard bands, mask-density bands, variance bands, and reformulation-changed/unchanged subsets.
- Added `shippable_gate_candidates` to the recommendation payload. A gate candidate must have enough windows, positive mean delta, no meaningful regressions, and no active-negative or suspicious metric-movement faults.

Aggregate outcomes:

| Profile | Windows | Mean delta vs baseline | Best | Worst | Improved / tied / regressed | Fault interpretation |
|---|---:|---:|---:|---:|---|---|
| `baseline` | 100 | 0.0000 | 0.0000 | 0.0000 | 0 / 100 / 0 | reference |
| `guard_p85_t0.01` | 100 | 0.0000 | 0.0000 | 0.0000 | 0 / 100 / 0 | safe no-op sentinel |
| `adaptive_p99_t0.0015` | 100 | -0.0254 | 0.0473 | -0.0926 | 33 / 0 / 67 | large upside windows, but overwhelmingly unsafe |
| `adaptive_p85_t0.002` | 55 | -0.0055 | 0.0136 | -0.0325 | 15 / 5 / 35 | active-negative dominated |
| `reform_rrf_v2` | 33 | -0.0054 | 0.0035 | -0.0318 | 3 / 12 / 18 | mostly neutral/negative |
| `adaptive_p95_t0.002` | 10 | -0.0044 | 0.0053 | -0.0140 | 2 / 1 / 7 | insufficient and unsafe |
| `adaptive_p99_t0.002` | 11 | -0.0010 | 0.0199 | -0.0234 | 5 / 0 / 6 | balanced positive/negative, no gate |
| `adaptive_p85_t0.0025` | 17 | 0.0000 | 0.0000 | 0.0000 | 0 / 17 / 0 | no-op/tied |
| `adaptive_p99_t0.0025` | 15 | 0.0000 | 0.0000 | 0.0000 | 0 / 15 / 0 | no-op/tied |

Best post-hoc diagnostic gates:

| Profile | Best gate | Windows | Mean delta | Best | Worst | Improved / tied / regressed | Shippable |
|---|---|---:|---:|---:|---:|---|---|
| `adaptive_p85_t0.002` | `mask_density_mean>=0.98` | 13 | 0.0012 | 0.0071 | -0.0011 | 5 / 4 / 4 | no |
| `adaptive_p99_t0.0015` | `task:FiQA2018` | 36 | -0.0038 | 0.0225 | -0.0642 | 21 / 0 / 15 | no |
| `adaptive_p99_t0.002` | `jaccard_mean<0.9` | 8 | 0.0010 | 0.0199 | -0.0214 | 4 / 0 / 4 | no |
| `reform_rrf_v2` | `task:SciFact` | 10 | 0.0003 | 0.0035 | -0.0028 | 2 / 6 / 2 | no |

Gate-learning decision:

- `shippable_gate_candidates`: none.
- `golden_candidate_profiles`: none.
- `default_promotable_profiles`: none.
- The strongest single upside remains `adaptive_p99_t0.0015`, now with an even larger observed positive window (`+0.0473`), but it also produced a severe negative tail (`-0.0926`) and 67 active-negative windows. It is not a setting; it is labeled training data.
- The simple post-hoc gates are not expressive enough to isolate safe wins. The best weak subset, `adaptive_p85_t0.002` with `mask_density_mean>=0.98`, averaged `+0.0012` but still had 4 regressions and 4 active-negative faults.
- The next implementation should be a supervised gate trained from `gate_feature_rows`, with explicit holdout windows and false-positive penalties. A hand-written global threshold or one-dimensional diagnostic gate is not sufficient.

### Conservative learned-gate implementation

Implemented follow-up tooling:

- Added `train_gate_from_tuning_artifact.py` and the `chelatedai-train-gate` console entry point.
- The trainer reads `gate_feature_rows`, proposes candidate profile/task/diagnostic rules, splits by held-out `global_window` modulo, and only accepts rules that pass both train and holdout support, mean-delta, regression, active-negative, and suspicious-metric-movement gates.
- Added `--strategy learned_gate` plus `--gate-artifact` / `--gate-config` to `run_thousand_query_tuning.py`. Learned-gate runs always keep `baseline` and `guard_p85_t0.01`; accepted gate profiles are added only when their task scope applies. If no rule applies, the runner falls back to sentinels rather than activating an unsafe profile.

Training command:

```bash
python train_gate_from_tuning_artifact.py --input experiment_runs\roadcourse-small\adaptive_fivek_gate_learning_tuning.json --output experiment_runs\roadcourse-small\learned_gate_v1.json
```

Training outcome:

| Artifact | Accepted rules | Rejected rules | Decision |
|---|---:|---:|---|
| `experiment_runs\roadcourse-small\learned_gate_v1.json` | 0 | 140 | fail closed |

This is a useful negative result. The supervised/holdout gate did not find a safe profile in the current feature rows, so a 5,000-query learned-gate campaign would only run the baseline and guard sentinels. The next improvement path is not another threshold sweep; it is richer training data, likely query-level or learned-mask features, plus candidates whose active-positive behavior survives holdout without active-negative tails.

### Six-path alternative validation tracks implemented

The follow-up implementation moved all six alternative pathways from planning into code:

1. **Query-level attribution rows.** `run_thousand_query_tuning.py` now records `query_metrics` for every evaluated profile/query and flattens them into `query_attribution_rows` in running and completed artifacts. Each row includes task, query, profile, per-query NDCG/MRR/Recall deltas against baseline, first-relevant-rank delta, top-10 overlap, top-document change, action, variance, Jaccard, mask density, reformulation flags, and profile fault class. This gives future gates examples at the level where the actuator actually helps or hurts instead of forcing them to infer from 50-query averages.
2. **Synthetic collapse benchmark.** `synthetic_collapse_benchmark.py` and the `chelatedai-synthetic-collapse` entry point create deterministic semantic-collapse fixtures where a single noisy dimension makes distractors outrank relevant documents. Masking the known collapse dimension recovers retrieval, which gives the project a controlled "does the chelation premise work at all?" test independent of noisy real benchmark slices.
3. **Learned mask smoke.** `learned_mask_policy.py` and `chelatedai-learned-mask-smoke` learn the harmful dimension in the synthetic collapse fixture by pairwise query/relevant/distractor alignment and recover the known collapse dimension without being handed it directly.
4. **Selective reformulation policy.** `query_reformulator.should_apply_reformulation()` and `AntigravityEngine.enable_query_reformulation(policy=...)` add an opt-in `selective_low_specificity` policy. `run_thousand_query_tuning.py --strategy selective_reform` compares always-on RRF reformulation with the selective policy.
5. **Benchmark-family meta-analysis.** `research_pathway_analyzer.py` / `chelatedai-research-pathways` summarizes profile outcomes by task/family across one or more artifacts.
6. **Candidate-profile proposals.** The same meta-analyzer proposes candidate profiles from query attribution, distinguishing retestable query-conditional signals from profiles that should remain training data only.

Synthetic collapse smoke result:

```bash
python synthetic_collapse_benchmark.py
```

| Condition | NDCG@3 | MRR | Recall@3 |
|---|---:|---:|---:|
| Collapsed baseline | 0.0 | 0.0 | 0.0 |
| Known noisy dimension masked | 1.0 | 1.0 | 1.0 |

This does not prove the current production chelation thresholds are shippable. It proves the core dimension-masking premise can recover a controlled collapse case, while the road-course evidence says the real system still needs better query-level features and learned candidates before any default/profile promotion.

Compact real-data probes:

```bash
python run_thousand_query_tuning.py --phase-queries 100 --loop-queries 50 --window-queries 25 --sample-docs 160 --base-seed 130 --strategy gate_learning --output experiment_runs\roadcourse-small\attribution_probe_100.json
python run_thousand_query_tuning.py --phase-queries 100 --loop-queries 50 --window-queries 25 --sample-docs 160 --base-seed 131 --strategy selective_reform --output experiment_runs\roadcourse-small\selective_reform_probe_100.json
python research_pathway_analyzer.py --artifact experiment_runs\roadcourse-small\attribution_probe_100.json --artifact experiment_runs\roadcourse-small\selective_reform_probe_100.json --output experiment_runs\roadcourse-small\research_pathway_meta_200.json
```

Meta-probe outcome:

| Pathway/profile | Evidence from compact probes | Decision |
|---|---|---|
| Synthetic collapse | baseline NDCG@3/MRR/Recall@3 `0.0`; known-mask recovery `1.0` | core premise works in controlled collapse |
| Learned mask smoke | learned masked dimension `4`, matching expected collapse dimension `4`; recovered NDCG@3 `1.0` | learned masking is promising on controlled fixtures |
| `adaptive_p99_t0.0015` | 100 query rows, mean per-query delta `-0.0267`, 3 positive / 11 negative | training data only |
| `adaptive_p85_t0.002` | 50 query rows, mean per-query delta `-0.0057`, 3 positive / 2 negative | training data only |
| `reform_rrf_v2` | 125 query rows across probes, mean per-query delta `+0.0002`, 4 positive / 1 negative; window mean approximately neutral | retest query-conditionally, not shippable |
| `selective_reform_rrf_v2` | 100 query rows, mean per-query delta `-0.0005`, 1 positive / 1 negative; skipped most reformulations but did not improve safety | not better than always-on in this policy |

Research decision: no golden setting emerged. The most useful result is that controlled collapse and learned masking validate the premise, while real road-course attribution says current chelation thresholds are unsafe and reformulation is the only weak retest candidate. The next productive branch is query-conditional reformulation and learned mask candidates trained on richer attribution, not another global threshold sweep.

### Follow-on reformulation and static-mask probes

The next autopilot loop expanded the weak reformulation lead and then pivoted to learned/static masking:

```bash
python run_thousand_query_tuning.py --phase-queries 100 --loop-queries 50 --window-queries 25 --sample-docs 160 --base-seed 132 --strategy reform_policy_search --output experiment_runs\roadcourse-small\reform_policy_probe_100.json
python static_mask_probe.py --task SciFact --max-queries 100 --train-queries 50 --sample-docs 200 --seed 140 --mask-fraction 0.02 --output experiment_runs\roadcourse-small\static_mask_probe_scifact_100.json
```

Reformulation-policy outcome:

| Profile | Windows | Mean delta | Best | Worst | Fault outcome |
|---|---:|---:|---:|---:|---|
| `reform_rrf_v2` | 4 | -0.0006 | 0.0000 | -0.0012 | 2 neutral / 2 active-negative |
| `selective_reform_rrf_v2` | 4 | -0.0006 | 0.0000 | -0.0012 | skipped most reformulations but kept the same negative windows |
| `reform_high_specificity_rrf_v2` | 4 | 0.0000 | 0.0000 | 0.0000 | neutral/no-op relative to baseline |
| `reform_claim_cue_rrf_v2` | 4 | -0.0006 | 0.0000 | -0.0012 | active-negative on the same windows |

This removes `reform_rrf_v2` as a current retest lead in this slice. It remains useful as attribution data, but not a candidate setting.

Static-mask outcome:

| Mask fraction | Masked dims | Train delta NDCG@10 | Holdout delta NDCG@10 | Decision |
|---:|---:|---:|---:|---|
| 0.005 | 2 | -0.0005 | -0.0060 | reject |
| 0.010 | 4 | -0.0005 | -0.0055 | reject |
| 0.020 | 8 | +0.0024 | -0.0061 | overfit; reject |
| 0.050 | 19 | -0.0054 | -0.0046 | reject |

The learned-mask premise still holds in the controlled collapse fixture, but the first real supervised static-mask transfer probe overfits or damages holdout retrieval. The next mask branch should be query-conditional and regularized, not a global static mask learned from a small train slice.

### Query-conditional mask follow-up

The next loop added fail-closed conditional masking to `static_mask_probe.py --conditional`. The probe learns a global harmful-dimension mask from train queries, searches simple one-feature query gates on train examples, and applies the mask to holdout queries only when the selected gate matches. If no gate passes train criteria, it falls back to baseline.

Compact SciFact conditional-mask outcomes:

| Artifact | Mask fraction | Selected gate | Global holdout delta | Conditional holdout delta | Decision |
|---|---:|---|---:|---:|---|
| `conditional_mask_probe_scifact_100.json` | 0.020 | `query_stopword_ratio <= 0.0852` | -0.0063 | 0.0000 | damage avoided, no lift |
| `conditional_mask_probe_scifact_100_f0005.json` | 0.005 | `query_stopword_ratio <= 0.0852` | -0.0060 | 0.0000 | damage avoided, no lift |
| `conditional_mask_probe_scifact_100_f001.json` | 0.010 | `query_stopword_ratio <= 0.0852` | -0.0052 | +0.0009 | weak positive, below threshold |
| `conditional_mask_probe_scifact_100_f001_seed144.json` | 0.010 | `query_stopword_ratio <= 0.0852` | -0.0040 | -0.0040 | repeated branch regressed |
| `conditional_mask_probe_scifact_100_f001_seed145.json` | 0.010 | `query_stopword_ratio <= 0.0852` | -0.0001 | +0.0005 | weak positive, below threshold |
| `conditional_mask_probe_scifact_100_f005.json` | 0.050 | `baseline_score_margin <= 0.0475` | -0.0128 | -0.0114 | reject |

This is more informative than the global static mask because it can fail closed or reduce harm, but it is not a golden path. The repeated low-stopword-ratio gate is a weak research lead for future query-conditional/regularized masking, not a setting: holdout gains are small and inconsistent, and one repeat regressed.

### Regularized conditional-mask follow-up

The next loop added `static_mask_probe.py --regularized-gate`, which splits train examples again before selecting a conditional mask gate. A gate must first pass train criteria, then survive the internal validation slice with support, non-negative mean delta, and no negative examples before it can be applied to the external holdout. If no gate passes, the probe fails closed to baseline rankings.

Commands:

```bash
python static_mask_probe.py --task SciFact --max-queries 100 --train-queries 50 --sample-docs 200 --seed 146 --mask-fraction 0.01 --conditional --regularized-gate --output experiment_runs\roadcourse-small\regularized_conditional_mask_scifact_100_seed146.json
python static_mask_probe.py --task SciFact --max-queries 100 --train-queries 50 --sample-docs 200 --seed 147 --mask-fraction 0.01 --conditional --regularized-gate --output experiment_runs\roadcourse-small\regularized_conditional_mask_scifact_100_seed147.json
python static_mask_probe.py --task SciFact --max-queries 100 --train-queries 50 --sample-docs 200 --seed 148 --mask-fraction 0.01 --conditional --regularized-gate --output experiment_runs\roadcourse-small\regularized_conditional_mask_scifact_100_seed148.json
```

Compact SciFact regularized-mask outcomes:

| Artifact | Selected gate | Global holdout delta | Conditional holdout delta | Applied holdout queries | Decision |
|---|---|---:|---:|---:|---|
| `regularized_conditional_mask_scifact_100_seed146.json` | `baseline_score_margin <= 0.0721` | -0.0013 | +0.0014 | 14/50 | local candidate only; needs repeat |
| `regularized_conditional_mask_scifact_100_seed147.json` | `query_token_count >= 15` | -0.0044 | 0.0000 | 10/50 | damage avoided, no lift |
| `regularized_conditional_mask_scifact_100_seed148.json` | none | -0.0007 | 0.0000 | 0/50 | fail closed |

This is the strongest mask-path evidence so far, but it is still not a golden setting. Regularization did what we wanted operationally: it prevented broad static-mask damage and failed closed on one seed. However, the only external holdout lift was small and did not repeat across adjacent seeds. The next mask branch should move from one-feature thresholds to a small regularized classifier over query features and baseline score margins, still with internal validation and repeat/transfer gates before any candidate promotion.

### Classifier-gated conditional-mask sweep

The next branch implemented the small learned gate suggested by the regularized one-feature loop. `static_mask_probe.py --classifier-gate` trains a compact logistic classifier over baseline score margins and query lexical features, then converts the score into a conditional-mask gate. The implementation is still fail-closed: it needs train support, internal validation support, positive internal validation delta, no negative examples, and a minimum count of positive train examples before the external holdout can be touched.

Three 50-loop compact SciFact sweeps were run over seeds 200-249 at mask fraction 0.01:

| Sweep artifact | Gate criteria change | Mean conditional delta | Best | Worst | Positive > 0.001 | Negative < -0.001 | Fail-closed loops | Decision |
|---|---|---:|---:|---:|---:|---:|---:|---|
| `classifier_conditional_mask_sweep_scifact_50.json` | initial classifier validation accepted non-negative internal validation | -0.0004 | 0.0000 | -0.0084 | 0 | 3 | 41/50 | reject; leaked harm |
| `classifier_conditional_mask_strict_sweep_scifact_50.json` | validation mean had to clear `min_mean_delta` | -0.0004 | 0.0000 | -0.0084 | 0 | 3 | 47/50 | reject; still leaked sparse-positive harm |
| `classifier_conditional_mask_posfloor_sweep_scifact_50.json` | added minimum positive train-example floor | 0.0000 | 0.0000 | 0.0000 | 0 | 0 | 50/50 | safe inert; no lift |

The useful finding is not a candidate profile; it is a diagnosis. Real compact SciFact train slices rarely contain enough positive mask examples for a learned classifier to generalize. Sparse-positive classifier gates can look acceptable on internal validation and still harm external holdout. Requiring at least four positive train examples makes the branch safe but completely inert on this setup. The next research branch should change the data/learning surface rather than loosen gates: larger train windows, query-level examples pooled across tasks/seeds, or a different mask objective that produces denser positive supervision before classifier gates are retested.

## Default decision

Changed `ChelationConfig.DEFAULT_CHELATION_THRESHOLD` from `0.0004` to `0.01`.

This is not a promotion of always-on chelation. It is a guardrail update: the runtime still defaults to `use_centering=False` and `use_quantization=False`, and the higher threshold prevents opt-in adaptive quantization from over-chelating on the tested small-model road course.

Updated presets:

- `conservative`: threshold `0.01`
- `balanced`: threshold `0.01`
- `aggressive`: threshold `0.0004`, now explicitly marked experimental because it regressed on MiniLM/SciFact

## Remaining evidence boundary

This result is enough to set a safer default threshold for small-model SciFact behavior. It is not enough to claim chelation lift broadly. Larger BEIR tiers, multitask transfer, trained-adapter candidates, and non-SciFact tasks still need separate campaign evidence before promoting any more aggressive adaptive profile.
