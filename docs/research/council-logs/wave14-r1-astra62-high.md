---
panel_type: panel-of-experts
artifact: wave14-r0-master-roadmap-and-pm6-formal-spec
run_date: 2026-11-24
run_id: wave14-r1-astra62-high
challenge_depth: 'Round 1 of 3; source review only'
active_experts: ['gpt-6.2-astra (high): systems, streaming security, formal methods']
persist_version: 1
---

# Decision: separate the releases; repair before relaunch

Finish Waves 11–13 first. Do not call v6 sealed or advertise zero-token leakage. `bundle_registry_spec.json:3` still requires human audit and clean rerun. Reviewed R0 SHA-256 `0e88f8112f359a2e3663e808c285304e85567634d804aa91f7497809f0718c25` at HEAD `4fae0b77`; the three D7/D9/D10 implementation files below are untracked, so HEAD alone cannot identify the Spark payload. No runtime jobs, builds or protected-file reads.

## Q1 — Relaunch decisions

- **D7:** `chelated_graphiti_arena_v1.py:617` now consults record keys. But line 371 returns a repaired dictionary *before* schema validation; line 396 forces `json_schema` on retry even after fallback to `json_object`. A parseable object is not necessarily a valid extraction. Route every success through one validator and retry only with a supported format. Before the full arena, replay malformed-but-repairable wrong-field output and a schema-unsupported endpoint; require valid typed output or recorded arm failure.
- **D9:** `scripts/run_wave11_d9_fast_extractor_and_u_grpo.py:2091` already reloads the saved SFT base as FP32 and compares active versus merged on CPU. The BF16 GPU comparison at 2631 is separately **active versus reloaded active**, not versus merged. R0’s historical `0.002645` is not explained by this source alone. `autocast_adapter_dtype=False` is not proof of FP32 adapter weights; one prompt’s last eight positions cannot establish broad parity. Record actual parameter dtypes, checkpoint hashes, device and multiple valid positions; explicitly reject nonfinite differences (`NaN >= tolerance` is false). Replay the saved seed before retraining. Lines 2542/2562/2565/3103 still select by EM/reward and default to `2e-5/0.02/160`; enforce the determinate floor, record “none eligible,” keep final-step primary, and label inspected evaluation data a repeat look.
- **D10:** `scripts/run_wave11_d10_5m_real_cutover_and_70b_scale.py:1092` counts forbidden hits but does not require ten survivors: empty output passes the zero-leak assertion. Require positive unmasked exposure, fill@10 and admitted recall against an exact sampled oracle. At 1570 the manifest reads supplied paths; at 1590 the receipt hashes optional, conventionally named files instead. Bind mandatory digests to the actual input bytes; test distinct input/output directories and a stale same-name file. Serialize D9 and the 70B sweep on Spark #2 until measured unified-memory headroom covers resident services.

## Q2 — What Lean can certify

Use Euclidean spaces, not an unqualified function-space norm. Induct from each map’s Gram identity to `‖Bx‖ = (∏ αᵢ) ‖x‖`; for nonzero domain and positive scales, extremal stretch ratio is one. No rectangular inverse or SVD existence proof required. `spherical_migration_bridge.py:411` also centers/translates and optionally normalizes: the theorem concerns the linear factor. The adjoint identity does not recover components orthogonal to the learned range; CSLS cannot reconstruct a discarded distinction. For M2, include denominator rounding and CSLS terms before claiming a production rank margin; score error alone cannot certify an omitted ANN candidate. Anchor count without a spectral gap/noise bound does not prove convergence.

`ek5_v3_stratified.py:189` defines alternating reducts; 717 handles currency and 1095 rejects native supersede cycles. Imported cycles can instead be quarantined. A unique *least* fixed point does not mean only one fixed point. Prove a named restricted fragment first. Compare executable Lean and independent full recomputation against Python after **each mutation**, including rejection, authority changes, audit-only edges and imported cycles. Report excluded semantics. Pin Lean, mathlib, exporter and `nanoda_bin` revisions; check exported named theorem dependencies with an explicit axiom allowlist and a negative control. Upstream `ammkrn/nanoda_lib` README documents configurable permitted axioms. No compatible local tuple was demonstrated; `lake build` alone is not independent checking.

## Q3/Q4 — The client boundary and the useful benchmark

With stride one, suppose a match first becomes decidable on token 64. After token 63, a 32-token holdback has released **31**. Overlap supplies detector context, not confidentiality. For bounded detection delay require holdback ≥ maximum tokens *after onset* until verdict, including scheduling lag, or buffer the entire response. Unbounded semantic delay has no finite guarantee. Test actual client bytes across every boundary offset, Unicode normalization, split encodings, short final flush, timeout and cancellation; never flush unchecked bytes. Distinguish transport safety from undetected content even under full buffering.

Benchmark exact scoring of 480 restricted nodes separately from million-vector retrieval: one dense FP32 pass over `1M × 384` reads 1.536 GB before overhead. D10’s shortlist scorer reranks candidates, not the vault. Compare ANN misses with exact restricted scores and calibrate CSLS on the same reference population. Isolate the 8B embedder or measure co-residency under load: publish p95/p99 client delay, queue depth, memory and timeout behavior. Lock thresholds on DEV; report multilingual recall, benign refusals and best-of-ten **per-work** success. Zero refusals among 60 independent works still permits ~4.9% at one-sided 95% confidence.

Ship one offline replay command checking exact inputs, emitted bytes and gate receipts. Prioritize it and PM-1b closure over extra model arms; keep full `/formal` and D11 in v7. Round 2 should challenge these counterexamples, not repeat unchanged checks.
