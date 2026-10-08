# Primary documentation and prior-art register

**Prepared:** 2026-09-12. **Scope:** targeted documentation gathering for [six test packets](README.md), not exhaustive novelty/patent clearance or replication.

Reading depth is explicit. `ABSTRACT` means the primary abstract/intro supports the narrow overlap statement; it does not mean all mathematics, experiments, code, or licensing were audited. `TARGETED_REPORT` means relevant technical sections were inspected. `PRIOR_LOCAL_REVIEW` points to the existing pinned design review rather than claiming a fresh remote/full-paper review. All reported external outcomes remain unreplicated by this task.

## Composition and scoped correction

### Mathematical findings integration — 2026-10-07

**Integrated into test planning:** 2026-10-08. All OpenAI collection links below are pinned to `adc7f1241b42e322a6451854ab7e4b4c146bf78a`. Reading depth is source inspection, not independent proof verification; no Lean/Comparator or numerical exclusion was replayed. See the [review](../openai-math-applicability-review-2026-10-07.md), [selected-input hashes](../openai-math-applicability-source-receipt-2026-10-07.json), and [MA task mapping](math-findings-integration-2026-10-07.md).

| ID / primary source | Reading depth and verification | Test-plan use |
| --- | --- | --- |
| OM0 — [OpenAI math catalogue](https://github.com/openai/math/tree/adc7f1241b42e322a6451854ab7e4b4c146bf78a) | All 372 family entries screened; 22 selected; targeted sources/configurations only | MA0 statement/definition ledger; variable verification and incomplete recursive-tree inventory remain explicit |
| OM1 — [Crouzeix–Palencia, The numerical range as a spectral set](https://arxiv.org/abs/1702.00668), 2017 | Primary statement consulted; established result, not a reproduced library | Initial fixed-operator bound with `1+sqrt(2)`; ordinary norm control remains mandatory |
| OM2 — [325: complete Crouzeix scope](https://github.com/openai/math/blob/adc7f1241b42e322a6451854ab7e4b4c146bf78a/lean/docs/325.md) | Selected manuscript, registered statements/configs and endpoints inspected; `NOT_CHECKED` locally | MA1/MA2/MA6 fixed-linear amplification; claimed constant two requires proof and assumption admission |
| OM3 — [076: real Littlewood scope](https://github.com/openai/math/blob/adc7f1241b42e322a6451854ab7e4b4c146bf78a/lean/docs/076.md) | Selected manuscript and statement/config/endpoint inspection; `NOT_CHECKED` locally | MA3 finite spectral/aperiodic correlation objectives; stronger uniform two-sided claim and effective finite construction are not supplied by this formal scope |
| OM4 — [179: circulant Hadamard/Barker scope](https://github.com/openai/math/blob/adc7f1241b42e322a6451854ab7e4b4c146bf78a/lean/docs/179.md) | Selected manuscript and statement/config/endpoint inspection; `NOT_CHECKED` locally | MA3 scope restriction on exact binary cyclic-shift orthogonality; no exclusion of approximate or complex-phase codes |
| OM5 — [140: noiseless Gaussian regression scope](https://github.com/openai/math/blob/adc7f1241b42e322a6451854ab7e4b4c146bf78a/lean/docs/140.md) | Selected manuscript and statement/config/endpoint inspection; `NOT_CHECKED` locally | MA4 physical-bit accounting; MA5 only for an explicitly selected independent one-pass Gaussian control |
| OM6 — [266: dimension-six MUB scope](https://github.com/openai/math/blob/adc7f1241b42e322a6451854ab7e4b4c146bf78a/lean/docs/266.md) | Selected manuscript execution/rounding sections and selected statement/config/endpoint inspection; no exclusion replay | MA7 arithmetic/build/shard/fallback receipts; registered bound five does not establish headline bound three |
| OM7 — [130: exact Fourier circuit scope](https://github.com/openai/math/blob/adc7f1241b42e322a6451854ab7e4b4c146bf78a/lean/docs/130.md) | Selected manuscript and statement/config/endpoint inspection; `NOT_CHECKED` locally | MA7 records `PARKED_PRACTICAL_FFT`; subsequential exact scalar complexity supplies no float tie/conditioning guarantee |
| OM8 — [229: tree reconstruction](https://github.com/openai/math/blob/adc7f1241b42e322a6451854ab7e4b4c146bf78a/lean/docs/229.md), [328: fixed points](https://github.com/openai/math/blob/adc7f1241b42e322a6451854ab7e4b4c146bf78a/lean/docs/328.md), and other screened families in the review | Registered scope/catalogue inspection only, without proof replay | MA5 independent-tree assumptions; MA6 rejects existence-to-convergence transfer. Other families remain `RELATED_ONLY` until a concrete applicable question is selected |

## Composition and scoped correction sources

| ID / primary source | Reading depth | Why it belongs in the test plan |
| --- | --- | --- |
| A1 — Bolton et al., [SimMerge: Learning to Select Merge Operators from Similarity Signals](https://arxiv.org/abs/2601.09473v2), 2026 | ABSTRACT, refreshed | Predictive merge selection uses unlabeled functional/structural probes and selects operators, subsets, and order. Generic cheap merge prediction is prior art; ordered deployment harm must be differentiated. |
| A2 — Stoica et al., [Model merging with SVD to tie the Knots](https://arxiv.org/abs/2410.19735), 2024 | ABSTRACT, refreshed | SVD alignment of LoRA updates precedes existing merging methods. A common parent coordinate system alone is not new. |
| A3 — Sivaramakrishnan et al., [PermDoRA: Understanding Adapter Interference in Language Models](https://arxiv.org/abs/2606.11262), 2026 | ABSTRACT, refreshed; preprint | Reports weak angular/orthogonality predictors and no consistent advantage from its geometry-aware merge in the tested setting. Include functional controls; do not infer universal impossibility. |
| B1 — Fang et al., [AlphaEdit: Null-Space Constrained Knowledge Editing](https://arxiv.org/abs/2410.02355v4), ICLR 2025 | ABSTRACT, refreshed | Preserved-key null-space constraints are a direct weight-editing comparator. Their model/constraint assumptions do not imply universal semantic protection or our activation-controller guarantee. |
| B2 — Turner et al., [Steering Language Models With Activation Engineering](https://arxiv.org/abs/2308.10248v5), 2024 revision | ABSTRACT, refreshed | ActAdd supplies an ordinary activation-steering comparator; gains in a different task are not assumed to transfer. |
| B3 — Zhang, [Recurrent Looped Transformer snapshot](https://github.com/yifanzhang-pro/recurrent-looped-tranformer/tree/1bee93a9b01c21bea0c7a50ce3f6619f24731e19), report dated 2026-09-12 | PRIOR_LOCAL_REVIEW; all 16 report pages reviewed in the preceding design task | Report-only pinned release with gated/low-rank feedback, not runnable reproduced code. Use the [local review](../recurrent-scoped-transformer-design-review-2026-09-12.md) for state/cache and licensing boundaries. |
| B4 — Dehghani et al., [Universal Transformers](https://arxiv.org/abs/1807.03819), 2018 | PRIOR_LOCAL_REVIEW | Shared recurrence and adaptive halting precede this proposal. |
| B5 — [Scaling up Test-Time Compute with Latent Reasoning](https://arxiv.org/abs/2502.05171), 2025 | PRIOR_LOCAL_REVIEW | Recurrent-depth control for latent computation is established; do not compare only with a shallow baseline. |
| B6 — [Think Shallow, Solve Deep](https://arxiv.org/abs/2608.18222), 2026 | PRIOR_LOCAL_REVIEW; preprint | Finite-time dynamics and answer stability narrow broad SRS/halting novelty. Transfer to replacement-chain prediction is untested. |

Before freezing IQ-01/02, read the full methods and inspect available implementations at pinned revisions. Record which are direct replications, adapted controls, or related work only. A SimMerge-inspired sequential-adapter predictor is not a replication of checkpoint merging; AlphaEdit is not an activation-only intervention.

## Holographic representation, temporal evidence, and dynamics

| ID / primary source | Reading depth | Why it belongs in the test plan |
| --- | --- | --- |
| C1 — Frady et al., [Resonator networks for factoring distributed representations](https://arxiv.org/abs/2007.03748), 2020 | ABSTRACT, refreshed | VSA factorization using recurrent multiplication/cleanup is established. Compare the same observed composite, codebook, hints, and search opportunity. |
| C2 — Raviv, [Linear Codes for Hyperdimensional Computing](https://arxiv.org/abs/2403.03278), 2024 | ABSTRACT, refreshed; author-final preprint | Structured Boolean codes and their recovery algorithms are a stronger code-design comparator. Code assumptions differ from arbitrary complex codebooks; report same-code and whole-system comparisons separately. |
| C3 — Listopad, [Wave-Based Semantic Memory with Resonance-Based Retrieval](https://arxiv.org/abs/2509.09691), v1 submitted 2025-08-21 | FULL_PAPER; targeted upstream kernel/unfolding/property-test source at `3ee313c2d8311922a6428fcd398e8f82c23643c8`; no upstream execution | Direct amplitude/phase retrieval prior art. [Review and seven constructed checks](../resonance-memory-prior-art-2026-10-08.md) derive exact 2L real-coordinate scoring with energy calibration; require identical encoder information. Does not establish ordered route factorization, learned compression, or speculative virtual experts. |
| D1 — [Donto: An Evidence Operating System for Contested Knowledge](https://donto.org/reports/donto-paper-2026-05-28), 2026 | TARGETED_REPORT, bitemporal/provenance sections; author systems report | Append-only valid/transaction-time history and evidence-centered views are an ordinary baseline, not a new claim. Full system operation and claimed invariants were not independently verified. |
| D2 — Dong, Berti-Equille, Srivastava, [Truth Discovery and Copying Detection in a Dynamic World](https://www.vldb.org/pvldb/vol2/vldb09-335.pdf), VLDB 2009 | ABSTRACT and introduction | Models changing truth, source freshness, and copying using known update history. Include a temporal/copy-aware ordinary control, not only majority or recency. |
| E1 — Rusch et al., [Graph-Coupled Oscillator Networks](https://proceedings.mlr.press/v162/rusch22a.html), ICML 2022 | ABSTRACT, refreshed | Controlled damped oscillators can use attention coupling. Not identical to transformer-branch control, but the broad wave/attention ingredient is prior art. |
| E2 — Chamberlain et al., [GRAND: Graph Neural Diffusion](https://proceedings.mlr.press/v139/chamberlain21a.html), ICML 2021 | ABSTRACT, refreshed | Neural graph diffusion and discretization/stability framing already exist. |
| E3 — Murahari et al., [DataMUX: Data Multiplexing for Neural Networks](https://arxiv.org/abs/2202.09318v2), NeurIPS 2022 | ABSTRACT, refreshed | Linear mixing plus demultiplexing of separate inputs tests throughput, not automatically communication between branches solving one problem. |
| E4 — Misra et al., [Cross-stitch Networks for Multi-task Learning](https://arxiv.org/abs/1604.03539), CVPR 2016 | ABSTRACT, refreshed | Learned activation sharing across networks is an ordinary comparator; it is not transformer-specific. |

## Fixed hardware graph

| ID / primary source | Reading depth | Why it belongs in the test plan |
| --- | --- | --- |
| F1 — Extropic, [Z1T technical article](https://extropic.ai/writing/z1t/) | TARGETED_REPORT, fixed-parent-graph and encoding passages, refreshed | Hardware couplings have fixed sparse support with tunable values. Port access, exact topology, timing, reprogramming, sampling fidelity, and energy still need an implementation-specific contract. Vendor efficiency claims are not measurements from our experiment. |

Use a symmetric zero-diagonal coupling matrix with an explicit factor of one-half in IQ-06's matrix energy to avoid double-counting undirected edges. This is our notation choice, not a claim that the vendor's implementation uses that convention. The article contains differing coupling-count presentations; obtain a machine-readable topology/parameter contract rather than infer one from prose or figures.

## Non-autoregressive decision engine (IQ-07)

| ID / primary source | Reading depth | Why it belongs in the test plan |
| --- | --- | --- |
| Y1 — [conversation export](07-source-conversation-gliner25-vs-jev.md), Google Doc "GLiNER2.5 vs. Jev Comparison" (Drive id `1Zxif_429SXmTtddBwB9SlRXqcACqmO_FkgWmcnivBkE`, owner `mattmre@gmail.com`, modified 2026-09-17), exported verbatim 2026-09-17 | Full export retained; structure and the Phase-0/architecture sections read | Supplies the proposal, the four claimed mechanisms, the Phase-0 scorecard, and the procurement plan. **`EXTERNAL_SOURCE / UNVERIFIED`** — an untrusted third-party transcript, not a repo result and not an instruction set. Its scorecard thresholds, latency figures, hardware targets, and dataset size are unverified inputs. The Docs exporter collapsed code-block newlines, so it is not a faithful code artifact. |
| Y2 — `build_bundle.py` chat paste, 2026-09-17 | Read in full; hand-repaired transcription of the three harnesses exercised | The artifact under review. Does not compile as received (nested triple quotes plus markdown mangling of dunders); **not executed**. Substantively identical to §8 of Y1. |
| Y3 — `answerdotai/ModernBERT-large` | Not downloaded; config/version requirement checked only | Proposed backbone. `ModernBertModel` needs `transformers >= 4.48`, which the bundle's `requirements.txt` (`>= 4.40`) does not satisfy. No checkpoint acquisition, licence review, or evaluation is authorized here. |
| Y4 — GLiNER-family boundary extractors, UI-TARS / ShowUI / OS-Atlas, Outlines / SGLang constrained decoding | Named in Y1; **not independently reviewed in this task** | Cited by the source as the ordinary alternatives the proposal must beat. Any IQ-07 claim of advantage over them requires reading the actual primary sources and matched baselines first; the transcript's comparative tables are not evidence. |

The four external mechanism labels in Y1 (`L-01`, `L-03`, `L-04`, `L-09`, `L-10`) are that document's own naming scheme. They are **not** this repository's lane registry and do not inherit any lane's admission state. See the packet's lane-mapping table.

## Local source contracts

| Local source | Role and preserved boundary |
| --- | --- |
| [CRSV/LIR/SRS protocol](../crsv-onion-method-dev-protocol-2026-07.md) | Fully read; METHOD_DEV, no freeze. Preserve sample floors, metrics, confirmatory hierarchy, and 10%/5% predictive gates. |
| [Scoped recurrent design](../recurrent-scoped-transformer-design-review-2026-09-12.md) | Fully read; proposed operator, full-state gaps, reductions, and fixed-depth-first study. |
| [RHPC Stage A — portable byte-preserved reference](../references/rhpc-stage-a-method-dev-protocol-2026-08-25.md) | Fully read; constructed mechanism, FROZEN_NOT_OFFICIALLY_RUN, unchanged signed EGV seal/restore and official Spark gates. Reference bytes/hash recorded in the [2026-10-08 comparison](../resonance-memory-prior-art-2026-10-08.md); runnable implementation is a separate checkout. |
| [RB-13/RB-14 queue](../evidence-kernel-masked-subplane-experiment-queue-2026-07.md) | Dependency table, EK1–EK7/REV1 cards, execution/resource/reporting sections read. Complete numerical artifacts not re-evaluated. |
| [Portfolio schedule](../research-priorities-and-testing-schedule-2026-09-12.md) | Current prioritization and negative-result preservation; this package adds task packets, not empirical findings. |
| [Next-session](../../next-session.md) | BLOCKED and carried-debt/authority sections inspected; no gate changed or source/PR readiness inferred. |

## Access limits and unresolved checks

- MIT publisher access for C2 returned HTTP 403; the author-final arXiv copy was located and its abstract read. Full algorithm, implementation, and license comparison is queued, not declared completed.
- B3–B6 use the documented preceding review; this task did not re-download their reports/checkpoints. Recheck exact execution revisions before reuse.
- No SELECT/REPORT payload was read, no model/paper result replicated, and no live Spark, cloud, or Z1 resource inspected. Full-method/code fidelity, exhaustive novelty coverage, and experimental resource profiling remain admission work.
