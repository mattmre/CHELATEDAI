# ChelatedAI Related-Works Research Plan

Ownership: this plan lives in the repo and is updated continuously.
Status: active — iteration 14 continuation after segment 1 cap; current catalog is 74 works / 63 strong / 100% claim coverage.

## Objective

Identify and curate related works — relevant, cousins, structural duplicates, and additive/directional/mathematical additions — for the CHELATEDAI research portfolio, grounded in the repo's md files and reports, and continuously update/adapt this plan to capture strong supporting and supplementary works.

## Source Map (repo documents surveyed)

- README.md — portfolio overview, tracks, mains, quick start
- docs/VISION_LIQUIFIED_LATTICE.md — north star lattice vision
- docs/ROADMAP_EXECUTION.md — Phase I queue + Phase II steps 9-17
- docs/RESEARCH_TRACKS.md — nine research tracks
- docs/INDEX.md — documentation inventory
- REFERENCES.md — existing formal attribution list
- docs/drift-recovery-{results,diagnostics,calibrated-results,knob-sweep}-2026-06.md — drift recovery experiments
- docs/chelation_opsd_research/loop_01/01_literature_deep_dive.md — OPSD/SDPO/MIS-PO
- docs/chelation_opsd_research/loop_01/09_related_research_scan.md — existing self-improvement scan with 12 upgrade patterns
- docs/chelation_opsd_research/CHELATION_OPSD_RESEARCH_PLAN.md — 10-loop program
- research_processing/extract_claims_1.md — compact claim extraction report for attnres/, evolution-strategies/, SEAL/EGGROLL, drift, storage, and evaluation docs

## Related-work categories and matching criteria

| category | definition |
|---|---|
| relevant | directly applicable to a repo claim — same problem, compatible approach |
| cousin | neighboring problem/approach with transferable idea |
| structural_duplicate | near-identical mechanism in another project we must account for |
| additive | extends our research in a new direction (fills a gap, adds a mathematical layer) |
| math_foundation | formal/mathematical basis for a repo technique |

Strength rating: strong / supporting / weak.
Coverage rule: a claim is covered only by a strong work whose claims field lists it.

> Note: works in the seed catalog flagged `VERIFY` (or `as-cited-in-repo`) must be verified against primary sources before they stay strong; unverifiable strong entries get demoted. This keeps the metric honest.

## Method (per iteration)

1. Scan repo docs for claims and new research-surface changes; update claims.tsv if a materially new claim appears (keep corpus stable unless docs change).
2. Research external works per priority queue, classify into categories above, add/update rows in related_works.tsv.
3. Verify uncertain arXiv IDs and strong claims against primary sources (arXiv abstract pages, project pages) — search web, read abstract.
4. Update this plan: mark category coverage, list new findings, adjust next-iteration priorities.
5. Run harness, record metrics (log per experiment run).
6. Adapt plan per results for the next iteration.

## Metrics and direction

Primary: `claim_coverage` (percent) — fraction of claim corpus covered by at least one strong work. Direction: higher.
Secondaries: `related_works_total`, `related_works_strong`, `structural_duplicates_considered`.

Rationale: objective is identification of strong supporting works; a fixed, doc-extracted claim corpus gives a deterministic coverage proxy that grows only with verified strong entries. Harness workload is fully file-based, no network, fixed seeds: deterministic.

## Verification standard

- strong requires author + arXiv id / URL and a notes entry tracing the mapping (repo doc reference or primary-source check).
- arXiv ids marked `VERIFY`/`tbd` demoted or removed unless primary-source check confirms.
- No fabricated works — every catalog entry has a traceable source.

## Related-work categories to target next (gaps, ascending priority)

1. Verify remaining supporting/seed rows (RDR, LaBSE, DPR, and related entries) against primary sources before promotion.
2. Compare additional structural duplicates across post-hoc correction, whitening/flow, corrective retrieval, and routed-compute mechanisms; document near-duplicate boundaries.
3. Add drift-detection benchmarks and continual-evaluation standards around c06/c13, with explicit threshold calibration.
4. Extract concrete end-to-end latency/energy values from computational-storage works and distinguish near-storage scoring from storage-resident model compute.
5. Compare mBERT-KD, BGE-M3, and LaBSE recipes against the repo's `cross_lingual_distillation` module (c10).


## Existing priorities and open questions

See loop_01 papers and REFERENCES.md for the embedded attribution. Key open research questions the plan drives toward:

| Open question | Track | Deliverable |
|---|---|---|
| Does post-hoc correction beat re-embedding (oracle re-embed C2)? | adaptive | proof/refutation study — compare to BERT-whitening/flow |
| What formal guarantees bound residual linear-transform drift? | chelation | stability + retention math |
| Which temperature schedules unify anneal-explore/stabilize? | lattice | annealing controller theory |
| Evidence DAG + optional GNN training data, sparse+recall tradeoff? | lattice | drift benchmark first |
| Quant-aware adapter co-design? | quantization | quant-KL integrated loss |
| Beyond mock NVMe: which works have latency data? | storage | real-latency evidence search |
| Multi-cycle self-correction retention proofs? | self-healing | SDFT/OPSD-era loop evidence |

## First iteration backlog (post-init_experiment)

- Verify w25-w29, w31, w32 arXiv IDs (or demote them)
- Research lattice/graph-rag annealing works (fills c11-c14)
- Research computational-storage evidence works (fills c15)
- Research cross-lingual teacher routing works (fills c10)
- Update this plan with findings per iteration

## Iteration log

- Iter 0 (baseline, 2026-08-15): claim corpus 16; seed catalog 37 works; harness `autoresearch.sh` written and validated.
- Iter 2 (2026-08-15): verified 10 arXiv IDs (OPSD 2601.18734, SDPO 2601.20802, SDFT 2601.19897, MIS-PO 2602.10604, PRM survey 2510.08049, TTARAG 2601.11443, SEAL 2506.10943) and corrected wrong IDs (MRL->2205.13147, BEIR->2104.08663, MTEB->2210.07316, DPR->2004.04906, GPTQ->2210.17323, SimCSE->2104.08821). Added 20 works: EAD annealing (2510.05251, c11), evidence-DAG/GNN cluster (DualG-MRAG 2607.28580, MemGraphRAG 2606.00610, LogicRAG 2508.06105, GNN-RAG 2405.05539, MSoT 2601.06002), cross-lingual cluster (mBERT-KD 2004.09813, BGE-M3 2402.03216, LaBSE 2012.03464), storage cluster (CXL BW 2509.03377, CXL-KV 2511.00549, near-storage 2502.09921, HyMCache 2607.18141, CXL pooling 2606.12556), SAE steering (2309.08600), temperature scaling (1706.04599), Neural-ODE flows (1806.07366). claim_coverage 81.2% -> 100%; related_works_total 37 -> 57; related_works_strong 28 -> 38.
- Iter 3 (2026-08-15): verified 18+ arXiv abs via primary-source reads. Promoted to strong: DualG-MRAG 2607.28580, MemGraphRAG 2606.00610, LogicRAG 2508.06105, GNN-RAG 2405.20139, HILOS 2502.09921, CXL TRACE 2509.03377, CXL PNM 2511.00321, HyMCache 2607.18141, ITME 2606.12556. Corrected ids: LaBSE 2012.03464 -> 2007.01852, QAADS 2602.03306 (fixed REFERENCES MRL mislabel), GNN-RAG 2405.05539 -> 2405.20139, STaR 2203.14565 -> 2203.14465, anisotropy 1907.12009, DML 2506.14878 -> 2510.15308, SAE authors Cunningham et al. strong count 38 -> 47; coverage remains 100%.
- Iter 4 (2026-08-15): replaced inaccurate VectorQ c06 mapping with two verified drift-detection works: DriftLens (2406.17813, unsupervised real-time representation-space drift detection + per-label characterization) and Gupta et al. (2312.02337, LLM-embedding drift measurement, drift-sensitivity metric); fixed GKD id (2311.17031 -> 2306.13649), Online-Optimized RAG authors (Pan/Li/Wang), DML authors; VectorQ (2502.03771) recategorized cousin (adaptive semantic caching thresholds) with honest notes. coverage 93.8 -> 100.0; strong 47 -> 49.

- Iter 5 (2026-08-15): added depth works verified via abs reads: DAPO (2503.14476; clip-higher + dynamic sampling; c09/c11/c05), Mixture-of-Depths (2404.02258; capacity-limited top-k routing; c14), CSSD survey (2304.01666; semantic shift taxonomy; c06). Catalog 60 -> 63 works; strong 49 -> 51.
- Iter 6 (2026-08-15): added GPL (2112.07577; unsupervised domain adaptation of dense retrieval via pseudo-labeling; c05/c16); generated human-readable RELATED_WORKS_CATALOG.md from the TSV. Catalog 64 works; strong 52.
- Iter 7 (2026-08-15): added LoRA Learns Less and Forgets Less (2405.09673; retention evidence for low-rank correction; c07/c09); regenerated catalog. Catalog 65 works; strong 53. Coverage steady 100.0.
- Iter 8 (2026-08-15): added LightRAG (2410.05779; graph+vector dual-level retrieval with incremental updates; lattice evidence-graph cousin; c12/c13). Catalog 66 works; strong 54. Coverage steady 100.0.
- Iter 9 (2026-08-15): added verified test-time adaptation cluster: Test-Time Training 1909.13231 (self-supervised online updates), Tent 2006.10726 (entropy-minimization affine adaptation with collapse caution), CoTTA 2203.13591 (continual adaptation with source restoration/retention); catalog 69 works; strong 57; coverage steady 100.0.
- Iter 10 (2026-08-15): added verified math-foundation works: Deep CORAL (1607.01719; covariance alignment; c04/c06), DANN (1505.07818; domain-invariant routing; c06/c09), and MMD (0805.2368; RKHS two-sample drift test; c06). After removing a duplicated pre-existing DualG-MRAG row, the catalog has 71 unique works; strong 59; coverage 100.0%.
- Iter 11 (2026-08-15): removed a duplicated DualG-MRAG TSV row after uniqueness audit, then added Switch Transformers (2101.03961; capacity-constrained top-1 routing, load balancing, lower-precision stability; c14) after primary-source verification. Catalog target 72 unique works; strong 60; coverage target 100.0%.
- Iter 12 (2026-08-15): added Ragas (2309.15217; reference-free retrieval/context/faithfulness evaluation; c16) after primary-source verification. Catalog 73 unique works; strong 61; coverage 100.0%.
- Iter 13 (2026-08-15): added AdapterFusion (2005.00247; non-destructive adapter composition and retention across 16 NLU tasks; c07/c09) after primary-source verification. Catalog target 74 unique works; strong 62; coverage target 100.0%.
- Iter 14 (2026-08-15): verified and promoted Randomly Removing 50% of Dimensions in Text Embeddings (2508.17744; random removal up to 50% across 6 encoders and 26 retrieval/classification tasks; c03) from supporting to strong. Catalog remains 74 unique works; strong 63; coverage 100.0%.
