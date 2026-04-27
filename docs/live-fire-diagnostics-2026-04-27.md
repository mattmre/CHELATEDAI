# Live-Fire Diagnostics And Calibration

## Purpose

This document captures the first deterministic live-fire validation round for the current ChelatedAI runtime. The goal was not to replace BEIR or multitask benchmark campaigns; it was to verify that the engine, controls, telemetry, integrated diagnostics, adaptive gates, and dashboard summaries are wired and JSON-serializable in one end-to-end run.

## Harness

Run:

```powershell
python run_live_fire_diagnostics.py --output live_fire_results.json
```

The harness uses an in-memory deterministic embedding backend and a fake Qdrant-compatible vector store so it can run without model downloads, Ollama, Docker, or network services. It exercises:

- `AntigravityEngine.run_inference()`
- query reformulation
- adapter routing and route outcome recording
- stability tracker norm/variance/mask diagnostics
- retrieval-fitness scoring
- fitness composition with structural health, quantization, and storage metadata
- integrated diagnostics reports
- advisory adaptive gates
- dashboard event summarization

## Observed Results

| Signal | Observed value | Status |
|---|---:|---|
| Baseline retrieval fitness | `1.0` | pass |
| Baseline NDCG@10 / MRR / Recall@K | `1.0 / 1.0 / 1.0` | pass |
| Live-fire retrieval fitness | `1.0` | pass |
| Final composed fitness | `1.0` | pass |
| Structural health score | `1.0` | pass |
| Quantization retained-gain ratio | `1.0` | pass |
| Adapter norm ratio latest/mean | `1.0 / 1.0` | pass |
| Runtime diagnostics events | `17` | pass |
| Total harness events | `61` | pass |
| Adaptive gate status | `pass` | pass |
| Overall harness result | `warning` | expected |

Expected warnings:

1. The deterministic corpus is intentionally tiny and saturated, so baseline and live-fire retrieval both scored `1.0`. This validates wiring and reporting but does not prove retrieval lift.
2. The fixture forces a noisy/chelation-heavy path, so the CHELATE rate was `1.00`; do not tune production thresholds from this synthetic fixture alone.

## Current Calibration Guidance

| Control | Recommended start | Watch band / gate | Adjustment signal |
|---|---:|---:|---|
| Chelation threshold | `0.0004` | explore `0.0002-0.001`; target 20-40% CHELATE rate | Adjust when CHELATE rate, retrieval fitness, or threshold oscillation moves out of band |
| Quantization retention | `0.8-0.9` | hard gate currently `0.8` | Raise only after FP32 gains are reliably above baseline |
| Norm drift | `1.0` target | watch `0.75-1.33`; hard `0.5-2.0` | Investigate repeated hard-band exits |
| Route effectiveness | collect at least 20 samples | warn below `0.5`; disable below `0.25` | Disable or retrain routes with low Jaccard plus no retrieval lift |
| Structural health | `>=0.7` preferred | do not promote below `0.6` | Penalize candidates with persistent collapse, isomer, or topology drift |
| Retrieval fitness | baseline or better | prefer +1-3% before promotion | Promote only if gain survives transfer and quantization gates |

## Testing Priorities

1. Keep `run_live_fire_diagnostics.py` as the first smoke gate before longer campaigns.
2. Add live BEIR or multitask campaigns that compute `InitialChelatedValues` on frozen corpus/query/qrels.
3. Compare baseline, adaptive chelation, ES candidate, and quantized candidate rankings.
4. Require no structural-health, norm-drift, route-effectiveness, or latency regression before promotion.
5. Capture each campaign in a dated docs artifact and verification-log entry.

## Research Priorities

1. Determine whether chelation improves retrieval metrics on non-saturated query sets.
2. Identify threshold ranges that produce useful CHELATE rates without excessive reranking.
3. Test whether routed adapters improve retrieval or merely preserve baseline rankings.
4. Quantify whether INT8 simulation preserves any measured FP32 gains.
5. Evaluate online updates with norm/drift and structural-health controls enabled.

## Interpretation

The current stack is functionally wired: engine controls, submodule telemetry, integrated reports, adaptive gates, and dashboard summaries operate in a deterministic end-to-end run. The next proof step is empirical rather than architectural: run larger frozen baseline campaigns to see whether chelation produces measurable retrieval lift and which thresholds should move from defaults to promoted presets.
