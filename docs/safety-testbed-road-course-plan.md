# Safety Testbed Road-Course Campaign Plan

The safety testbed now separates wiring proof from roadworthiness proof. Stage 0 through Stage 5 cover instrumentation, component benches, control-surface dyno sweeps, a non-saturated closed course, calibration profile loops, and failure-injection ravine tests. The first small-model road-course campaign is captured in [Road-Course Results And Default Threshold Decision](road-course-results-2026-04-27.md).

## Default promotion gate

- Default change allowed now: `True` only for the conservative threshold guardrail captured in `ChelationConfig.DEFAULT_CHELATION_THRESHOLD = 0.01`
- Minimum campaigns: `beir_small_tier`, `multitask_transfer`, `repeatability_matrix`, and `quantization_survival`
- Retrieval lift: at least `+0.01`, target `+0.03`, against frozen baseline values for any quality-improving profile promotion
- Quantized retained gain: at least `0.80`
- Structural health: at least `0.70`, with hard reject below `0.60`
- Norm ratio: hard band `0.50-2.00`, watch band `0.75-1.33`
- Diagnostics: every warning/failure must be explicit and JSON-safe

## Campaigns

### beir_small_tier

Freeze a BEIR small-tier baseline and compare adaptive retrieval, query reformulation, adapter routing, and quantized candidates. The first quick SciFact run rejected always-on chelation and supported only a threshold guardrail. Required evidence for future promotion: baseline NDCG/MRR/Recall, adaptive metrics, quantization retained gain, structural-health score, runtime latency percentiles, and warning/failure counts.

### multitask_transfer

Validate that candidate settings transfer across held-out task/query families. Required evidence: source-task metrics, transfer-task metrics, route effectiveness by task, and query reformulation action mix.

### repeatability_matrix

Run repeated seeds and reject settings that only work once. Required evidence: seed matrix, mean/std/min/max score, tolerance decision, and artifact hash.

### quantization_survival

Ensure candidate gains survive INT8 simulation before promotion. Required evidence: FP32 fitness, quantized fitness, retained gain ratio, and gate reasons on failure.

### structural_health_ablation

Confirm structural-health penalties correlate with retrieval outcomes and catch collapse/isomer/topology regressions. Required evidence: collapse ratio ramp, isomer drift ramp, topology cohesion drift, and adaptive gate actions.

## Documentation refresh requirements

Every road-course campaign summary should record baseline metrics, adaptive metrics, transfer metrics, quantization retention, health/norm/route telemetry, latency, warnings/failures, setting decisions, and rollback conditions. `docs/RESEARCH_TRACKS.md` should continue to distinguish implemented safety harnesses from actual benchmark proof.
