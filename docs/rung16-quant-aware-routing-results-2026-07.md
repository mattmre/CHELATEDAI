# Rung 16 — Quant-Aware Routing Results (July 2026)

Preregistration: `prereg_rung16.json`.
File SHA256: `7061f1f20e720a653172e9ae995989549d5d07b4ac603215ab3cbdc4ffc5ba15`; canonical JSON SHA256:
`48bad48fe78bc205b1e9c0abae8cc504ba067c27f2868548107b67a39f5c069a`.
GPU: `NVIDIA GeForce RTX 3090`.

Promotion was decided on SELECT only. REPORT CIs below are descriptive and cannot rescue a failed SELECT gate.

Artifact note: the campaign wrote the Arena A and Arena B selection locks before their REPORT calls
(observed at 08:21:07 and 08:22:39 respectively). A post-run metadata-only edit at 08:24 added the
raw preregistration file hash alongside the already-recorded canonical JSON hash, so the current lock
file modification times reflect that hash annotation rather than their original pre-REPORT writes.
No decision, metric, query ID, route usage, or verdict field changed.
The completed manifest was then used to backfill per-arena REPORT-consumed markers at 12:31:28Z;
they record the frozen-report and current selection-lock hashes and make subsequent default-runner
processes refuse the already-consumed preregistration/arena pair. These markers strengthen future
cross-process enforcement but are not presented as contemporaneous proof of the original REPORT call.

Final adversarial review then hardened the implementation without re-reading REPORT: the router now
freezes and checks route membership, centroids, margin, configuration, qrels, split, documents,
oracle, and preregistration after SELECT; every retained adapter is quant-gated even if SELECT did
not route a query to it; encoder-swap serving requires an explicit query-vector provider; and future
selection locks record source-file hashes plus Git state. The final runner also defers REPORT-query
embedding until after the durable SELECT lock. Both recorded arenas used every retained
specialist and the global fallback on SELECT, so the broadened quant-gate scope covers the same four
adapters and cannot change either recorded verdict. The original campaign manifest did not record a
contemporaneous code/Git hash, however, and the runner was hardened after the run. Therefore the
numeric artifacts are internally recomputable from their stored per-query rows but are not
cryptographically attributable to the final source bytes. REPORT was not rerun to paper over that
lineage limitation.

## Arena A — default SciFact swap

Verdict: **FAIL-CLOSED**. Corpus 1200; queries 100; split 40/30/30.

### Frozen SELECT promotion gate

- Plane vs single-global delta: 0.000000 NDCG
- Paired 95% CI: [0.000000, 0.000000]
- Required lower bound: > 0.005000
- Quant pass-rate: 0.250
- No-route floor passed: True
- SELECT gate passed: **False**

Per-used-adapter SELECT quant gates (baseline is no-route on the same routed-query subset):

| Adapter | Queries | No-route | FP32 | INT8-sim | Retained | Pass | Reasons |
|---|---:|---:|---:|---:|---:|:---:|---|
| cluster:0 | 6 | 0.000000 | 0.000000 | 0.000000 | 1.0000 | False | fp32_gain_below_minimum |
| cluster:1 | 1 | 0.000000 | 0.000000 | 0.000000 | 1.0000 | False | fp32_gain_below_minimum |
| cluster:2 | 5 | 0.000000 | 0.000000 | 0.000000 | 1.0000 | False | fp32_gain_below_minimum |
| global | 18 | 0.000000 | 0.100472 | 0.083748 | 0.8335 | True | none |

SELECT route usage:

| Route | Count | p_k |
|---|---:|---:|
| cluster:0 | 6 | 0.2000 |
| cluster:1 | 1 | 0.0333 |
| cluster:2 | 5 | 0.1667 |
| global | 18 | 0.6000 |

Entropy (nats): 1.040383; n_used: 4.

### One frozen REPORT evaluation

- Plane FP32 NDCG: 0.098804
- Plane INT8-sim NDCG: 0.099826
- Single-global NDCG: 0.102136
- No-route NDCG: 0.000000
- C2O oracle NDCG: 0.753579
- Single-best-route NDCG: 0.059563
- Plane − single-global: -0.003332
- Descriptive paired 95% CI: [-0.051022, 0.041025]
- Multi-route binding passed: True (qualifying: ['cluster:0', 'cluster:1', 'cluster:2'])

REPORT route usage:

| Route | Count | p_k |
|---|---:|---:|
| cluster:0 | 4 | 0.1333 |
| cluster:1 | 4 | 0.1333 |
| cluster:2 | 9 | 0.3000 |
| global | 13 | 0.4333 |

Entropy (nats): 1.260873; n_used: 4.

## Arena B — mixed SciFact + NFCorpus + FiQA2018

Verdict: **FAIL-CLOSED**. Corpus 4511; queries 300; split 120/90/90.

> **Scope correction (Tier B, important).** Arena B was intended as routing's "fair-chance home turf"
> (clusters = distinct domains). It did **not** cleanly exercise that hypothesis: under the
> encoder swap, serve-time centroid assignment is largely **anti-domain**, so most queries were routed
> to a specialist trained on a *different* domain. What this arena actually falsified is the
> **preregistered centroid-margin domain plane**, not "domain-specialized routing" in general.
>
> REPORT domain → home-route purity (recomputed from the frozen manifest rows):
>
> | Domain | Home-routed | Purity |
> |---|---:|---:|
> | FiQA2018 | 9/30 | 30.0% |
> | NFCorpus | 1/30 | **3.3%** |
> | SciFact | 11/30 | 36.7% |
>
> REPORT loss attribution (plane − single-global, per query):
>
> | Bucket | n | mean Δ | sum Δ |
> |---|---:|---:|---:|
> | Home-correct specialist | 21 | **+0.0257** | +0.5391 |
> | Cross-domain (misrouted) specialist | 39 | **−0.0309** | −1.2035 |
> | Global fallback | 30 | 0.0000 | 0.0000 |
> | **Total** | 90 | **−0.007383** | — |
>
> **Read this carefully:** specialists *helped* when they were routed to their own domain (+0.026);
> the plane lost because misroutes (65% of specialist-served queries) dominated. The binding
> constraint measured here is **route assignment under encoder-swap drift**, not specialist capacity.
> This does not rescue the verdict — the preregistered plane-level SELECT gate still failed
> (δ = −0.000899, CI [−0.026005, 0.024604], required lower bound > 0.005) and home-correct wins do not
> promote under a plane-level rule — but it means the multi-domain negative is **weaker evidence
> against domain routing per se** than a naive reading of "fail-closed on its home turf" suggests.

### Frozen SELECT promotion gate

- Plane vs single-global delta: -0.000899 NDCG
- Paired 95% CI: [-0.026005, 0.024604]
- Required lower bound: > 0.005000
- Quant pass-rate: 0.750
- No-route floor passed: True
- SELECT gate passed: **False**

Per-used-adapter SELECT quant gates (baseline is no-route on the same routed-query subset):

| Adapter | Queries | No-route | FP32 | INT8-sim | Retained | Pass | Reasons |
|---|---:|---:|---:|---:|---:|:---:|---|
| domain:FiQA2018 | 16 | 0.029330 | 0.092709 | 0.077713 | 0.7634 | False | retained_gain_below_threshold |
| domain:NFCorpus | 16 | 0.000000 | 0.039433 | 0.039433 | 1.0000 | True | none |
| domain:SciFact | 29 | 0.004988 | 0.047712 | 0.045600 | 0.9506 | True | none |
| global | 29 | 0.000000 | 0.052151 | 0.052151 | 1.0000 | True | none |

SELECT route usage:

| Route | Count | p_k |
|---|---:|---:|
| domain:FiQA2018 | 16 | 0.1778 |
| domain:NFCorpus | 16 | 0.1778 |
| domain:SciFact | 29 | 0.3222 |
| global | 29 | 0.3222 |

Entropy (nats): 1.343965; n_used: 4.

### One frozen REPORT evaluation

- Plane FP32 NDCG: 0.067189
- Plane INT8-sim NDCG: 0.069242
- Single-global NDCG: 0.074573
- No-route NDCG: 0.000000
- C2O oracle NDCG: 0.614455
- Single-best-route NDCG: 0.008472
- Plane − single-global: -0.007383
- Descriptive paired 95% CI: [-0.027989, 0.012247]
- Multi-route binding passed: True (qualifying: ['domain:FiQA2018', 'domain:NFCorpus', 'domain:SciFact'])

REPORT route usage:

| Route | Count | p_k |
|---|---:|---:|
| domain:FiQA2018 | 19 | 0.2111 |
| domain:NFCorpus | 15 | 0.1667 |
| domain:SciFact | 26 | 0.2889 |
| global | 30 | 0.3333 |

Entropy (nats): 1.351904; n_used: 4.

## Validation pasted from the final working tree

- `python -m unittest -v test_adapter_router test_quant_aware_routing` — **24 tests, OK**.
- `python -m unittest -v test_eggroll_strategic_platform test_safety_component_controls test_safety_instrumentation` — **41 tests, OK**.
- `python -m ruff check adapter_router.py quant_aware_routing.py antigravity_engine.py run_quant_aware_routing_campaign.py test_adapter_router.py test_quant_aware_routing.py` — **All checks passed**.
- `python -m py_compile ...` for the six changed Python modules above — **PASS**.
- `git diff --check` — **PASS**.
- `python scripts/check_block_flag.py` — **PASS; block flag CLEAR**.
- A guarded Arena-A runner invocation was attempted after the campaign and refused with
  `REPORT is one-shot across processes`; this is the expected cross-process one-shot behavior.

At this handoff the Rung-16 files and evidence remain working-tree changes, not committed/pushed
review evidence. Publication must preserve the disclosure above and must not rerun either REPORT.
