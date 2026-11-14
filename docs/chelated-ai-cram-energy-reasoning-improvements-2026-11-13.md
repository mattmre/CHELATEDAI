# Chelated AI × CRAM energy reasoning: what to improve

Ben Hoover's thread (Prefrontal, Nov 13 2026) shows a tiny learned Hopfield–Tank energy that memorizes valid patterns and settles into them. This document maps that to the beach/footprint vision and to CHELATEDAI as it stands, and ranks what to change.

**Provenance.** Faithful Markdown conversion of the Page [Chelated AI × CRAM energy reasoning: what to improve](https://cursor.com/canvas/store-78ce16e3-367f-4281-9424-10208f183fd0/8e47e5b4-0dcf-4019-ba22-300fd27d03eb), filed into the repo on 2026-11-13. Analysis is against `mattmre/CHELATEDAI` `main` @ `e40c088`. Numbers are taken from that write-up; none were invented here.

**Sources used for the original write-up:**

- [x.com/Ben_Hoov/status/2121294692968993068](https://x.com/Ben_Hoov/status/2121294692968993068) (thread of 10 + author replies, read via X)
- [arxiv.org/abs/2611.17309](https://arxiv.org/abs/2611.17309) and [prefrontal.ai/news/cram](https://prefrontal.ai/news/cram)
- github.com/mattmre/CHELATEDAI `main` @ `e40c088` (2026-11-07), read-only clone
- Desktop `D:\HARK_CHELATEDAI` status notes 2026-10-24 and 2026-11-09 (read-only)
- [arxiv.org/abs/2605.15422](https://arxiv.org/abs/2605.15422) (DualKV)

## Contents

- [The tweet](#the-tweet)
- [Matt's vision, mapped](#matts-vision-mapped)
- [Where CHELATEDAI is today](#where-chelatedai-is-today)
- [What the evidence says so far](#what-the-evidence-says-so-far)
- [Gaps](#gaps)
- [Prioritized improvements](#prioritized-improvements)
- [What doesn't fit](#what-doesnt-fit)
- [Open questions](#open-questions)
- [Sources](#sources)

## Bottom line

The thread is about energy-based reasoning on constraint puzzles, not retrieval, but its core move fits the beach image well: learn a landscape where well-trodden (valid) patterns are deep basins, complete partial inputs by settling into a basin, and use the basin depth to know when to stop. CHELATEDAI has none of that machinery today. Its own evidence also points at "consolidate by rewriting from source" over "patch vectors in place". Re-embedding from text recovered retrieval after drift; the adaptive bank did not.

## The tweet

| Claim | Detail |
|-------|--------|
| 1,000× faster training vs looped models | 42 ms Sudoku, 132 ms Maze on one H200 |
| 282× fewer inference FLOPs | vs EBR at matched 99.86% Sudoku accuracy |
| 7.7K nonzero weights learned | of 263,169 available (Sudoku) |
| 30 puzzles to reach ~100% | vs 700+ for larger looped models |

Ben Hoover, Hugh Blayney and James Whittington (Prefrontal, London) posted "Classical Energies Learn to Reason 1,000× Faster than Looped Models" ([arXiv 2611.17309](https://arxiv.org/abs/2611.17309)). The model is called CRAM, short for Compact Reasoning Associative Memory.

| Topic | What CRAM does |
|-------|----------------|
| Energy | A mean-field Hopfield–Tank (1985) energy: per-cell beliefs `p_i` (softmax), learned pairwise cost matrices `K_ij` between every pair of cells, a per-cell bias `B_i`, plus an entropy term. It is a pairwise Markov random field. |
| Training | No backprop-through-time. Pseudolikelihood lowers the energy of each observed solution and raises the energy of every single-cell corruption of it (the negatives). Weights start at zero. An L1 soft-threshold drives most costs to exactly zero, so the weights become the learned rule set. |
| Inference | Clamp the clues and start 4 chains from random fills. Do noisy single-cell descent (focused Gibbs: update cells whose local energy is high). Stop when any chain's energy falls below a threshold fitted from the training solutions. |
| Why it's fast | Learning the constraints is easier than learning reasoning trajectories. Compute adapts per puzzle because the energy says when it's done. |
| Limits (authors' own) | Not a general reasoner. The four benchmarks are pairwise-constraint puzzles that 1–3 hand-set parameters solve exactly, so they shouldn't be used to test whether looped reasoning scales. Maze-Hard needed a hidden layer (29 params total). Learning hidden quantities without task knowledge is open. |
| Author replies | Sudoku constraints weren't hard-wired; each cell could see every cell, and most links learned zero weight. Maze used a locality prior (one shared 3×3 kernel). EBR differs: it is a looped TRM with an energy on top, while CRAM is a physical system. |

> Learning constraints is a lot easier than learning reasoning trajectories.
>
> — Ben Hoover, thread post 5/10

## Matt's vision, mapped

| Beach / footprint idea | Closest thing in CRAM | Fit |
|------------------------|-----------------------|-----|
| Well-traveled zones where human knowledge left its imprint | Low-energy basins, carved by pulling down the energy of observed valid solutions | Strong |
| Recall a zone and recreate it perfectly | Associative-memory pattern completion: clamp the partial input, descend to the stored attractor | Strong |
| Quantize and consolidate | L1 sparsity: 263K free weights collapse to about 7.7K nonzero. Hand-set versions need 1–3 unique values. It compresses to rules, not low-bit numbers | Partial |
| Consolidation driven by rewrites | Nothing like text rewriting. The nearest analog is negatives built as single-cell rewrites of each solution | Weak |
| Machine enriches itself from subregions of the vector space | No self-enrichment. Each task is trained separately from labeled solutions | Absent |
| Pristine knowledge centers of known knowns and known responses | The stopping threshold fitted on training solutions is a learned definition of settled, known-good state | Partial |
| Interplay across disciplines | Pairwise `K_ij` interactions between cells; across zones this would be the same idea one level up | Analogy only |

Your picture already has the idea that matters most: knowledge as a landscape whose deep places are the well-trodden ones, plus a test for having arrived. Energy models make that precise. The rewrite-and-enrich half of the vision isn't in this paper and needs other machinery (below).

## Where CHELATEDAI is today

```mermaid
flowchart LR
  q["Query"] --> scout["Gravity sensor<br/>top-K neighbors from Qdrant"]
  scout --> chel["Chelation<br/>variance mask + spectral centering"]
  chel --> log["chelation_log<br/>noise centroids per doc"]
  log --> sed["Sedimentation (sleep cycle)<br/>adapter pushed away from noise"]
  sed --> gates["Gates<br/>annealing, quant survival, DAG prune"]
```

| Concept | What the repo does | Where |
|---------|--------------------|-------|
| Zone sensing | Pulls the query's top-K neighborhood and masks the highest-variance dimensions (keeps the stable bottom P%, or inverts) | `antigravity_engine.py` `_gravity_sensor` L544, `_chelate_toxicity` L565 |
| Re-centering | Subtracts the neighborhood mean from query and candidates, applies the mask and re-ranks. Logs that centroid per doc | `antigravity_engine.py` `_spectral_chelation_ranking` L1634 |
| Consolidation (sleep) | Trains a residual adapter so a doc's vector moves away from the mean of its noise centroids | `run_sedimentation_cycle` L1696; `sedimentation_trainer.py` `compute_homeostatic_target` |
| Segmentation | Hierarchical sedimentation splits the corpus into about n/5 variance-based clusters, trains each one, then refines globally | `sedimentation.py` L40–240 (`_simple_partition`) |
| Structure / drift | Cosine bond tiers (covalent ≥0.90, hydrogen, VdW), isomer strength = 1 − Jaccard, evidence DAG with fitness-gated prune and re-anneal | `topology_analyzer.py`, `isomer_detector.py`, `evidence_dag.py` (#277, #293) |
| Quantization | FP32 vs quantized fitness gate for promotion; FP16 disk pool shard with parity check | `quantization_promotion_gate.py`; `computational_storage_poc/pool_shard.py` (#294) |
| Rewrites | Deterministic, LLM-free query variants fused with RRF. No document rewriting | `query_reformulator.py`; `enable_query_reformulation` L1069 |
| Self-distillation plan | OPSD/SDPO research plan to make self-healing real (currently advisory only) | `docs/chelation_opsd_research/loop_01/` |

Code search on main: no Hopfield or energy function, no product quantization or IVF, and "consolidat" appears in 0 Python files (only in docs).

## What the evidence says so far

| Result | Number | Source |
|--------|--------|--------|
| Encoder swap drift, SciFact: frozen | NDCG 0.819 → 0.006 | `docs/drift-recovery-post-bank-headtohead-results-2026-06.md` |
| … living annealed bank (C5) | 0.131 | same |
| … one-shot router (C5r) | 0.181 | same |
| … oracle re-embed from raw text (C2O) | 0.813 | same |
| Same on NFCorpus: C5 vs C2O | 0.046 vs 0.612 | `docs/drift-recovery-post-bank-headtohead-nfcorpus-results-2026-06.md` |
| Full-strength chelation vs raw (MiniLM; replicated on bge-small) | −0.059 to −0.145 nDCG@10; bge-small −0.088 SciFact | Desktop status notes 10-24 and 11-09 (`factory/c1`) |
| Dense + BM25 hybrid vs flat | +0.065 nDCG@10 | Status 10-24 |
| Reformulation gain on held-out queries | +0.0001 pooled (didn't replicate) | Status 10-24 |
| Earlier +0.081 adapter gain | Traced to train/test overlap | Status 11-09 |

**This is the key signal for the rewrite idea.** In your own drift experiments, rebuilding vectors from source text (C2O) recovers almost fully. Every in-place correction (living bank, router, adapters) recovers a fraction. That's direct support for consolidating by rewriting from source rather than nudging stored vectors. The factory notes from Nov 9 say the evidence currently supports a methodology / negative-results paper, not a positive one.

## Gaps

1. No notion of settled. Chelation fires on a fixed variance percentile, with no learned threshold for when a neighborhood is already well-known and should be left alone. CRAM's stopping threshold is exactly this, fitted from known-good examples.
2. Zones are not footprints. `_simple_partition` uses corpus variance with n/5 clusters. Nothing measures how heavily a region is used (query traffic, clicks, citations), so the system can't tell a well-trodden beach from an empty dune.
3. Consolidation patches vectors instead of rewriting knowledge. Sedimentation moves vectors away from noise centroids. Nothing produces a canonical rewritten artifact per zone, which is what your own C2O result says works.
4. Training signal is weak and leaky. The OPSD audit flags an InfoNCE temperature of 0.07, sparse thresholded targets and chelation_log training on eval data. CRAM gets dense signal from structured negatives (every one-step corruption) and a sparsity prior.
5. No interaction model between zones. Cross-discipline links exist only as evidence-DAG edges with prune/re-anneal. Rung 15 (GNN) is still open, and a much cheaper pairwise model hasn't been tried.
6. Baselines are beating the mechanism. Hybrid BM25 + dense beats flat retrieval, and chelation loses to raw. Any new idea has to beat hybrid first.

## Prioritized improvements

The table below keeps the original write-up numbering. The suggested implementation order is 6, then 2, then 1, then 3, 4, and 5.

| # | Change | Why (tweet / evidence link) | Effort | How to test |
|---|--------|------------------------------|--------|-------------|
| 1 | Consolidate by rewriting from source: per dense zone, periodically re-embed from raw text and generate a canonical zone card (summary + Q&A pairs with provenance), indexed beside the raw docs. Retire in-place vector patching as the default consolidation path | Your C2O vs C5 result (0.813 vs 0.131). This is the rewrites half of your vision; RAPTOR, GraphRAG community summaries and WRAP-style rephrasing are the prior art | M–L | Drift arena C0/C2O/C5 plus a new rewrite arm; answer correctness and citation accuracy, not just nDCG; strict train/test doc split (X-15 lesson) |
| 2 | Learned settledness gate: compute a local energy for each query neighborhood (e.g. modern-Hopfield log-sum-exp over the top-K) and fit a threshold on training queries. Chelate or re-rank only above it, and pass through untouched below it | CRAM's stopping threshold fitted from training solutions. Extends your one verified positive (L01-C2: the 0.01 guard preserves baseline) | S–M | Holm-corrected vs raw, CSLS and hybrid on SciFact / NFCorpus / ArguAna × MiniLM and bge-small; report the fire rate; equivalence via the existing TOST module |
| 3 | Footprint-weighted zones: replace `_simple_partition` with k-means/IVF zones and keep per-zone traffic counts (queries, hits, citations). Rank zones by footprint density and consolidate the top ones first | The beach metaphor made measurable. IVF partitioning is standard (FAISS) | M | Per-zone nDCG before/after; check that dense-zone gains don't cost sparse zones; hubness stats per zone |
| 4 | Contrastive energy training for adapters without BPTT: positives are the labeled relevant docs; negatives are structured one-step corruptions (swap one doc in the ranked list, perturb one dimension block). Add an L1 soft-threshold so most weights go to zero | CRAM's pseudolikelihood + L1. Sparse weights are interpretable and should survive quantization better, which ties into the existing QuantizationPromotionGate | M | Same split as current sedimentation; compare nDCG, nonzero count and INT8 retained-gain ratio; mutation test that the negatives matter |
| 5 | Zone-to-zone pairwise energy as a lightweight alternative to rung 15's GNN: learn sparse `K` between zone-relevance variables over the evidence DAG and re-rank by low joint energy | CRAM shows a pairwise MRF with learned sparse costs can carry structure cheaply. This is the cross-discipline interplay in your vision | M | Must beat the flat pool and hybrid on the drift fixture, or fail closed (the rung 15 rule already says so) |
| 6 | Hand-set floor for every claim: before crediting a learned mechanism, run a 1–3 parameter hand-set version (e.g. fixed CSLS, fixed BM25 weight) | The authors' warning: if 3 hand-set parameters solve the benchmark, the benchmark proves nothing | S | Add the floor row to every factory card; a claim passes only if it beats the floor by the MDE |

Suggested order: 6 and 2 first (cheap, low risk, protect baseline), then 1 (highest expected value given your own data), then 3, 4 and 5. Effort is a rough estimate for one engineer on the existing harness.

## What doesn't fit

**Transfers**

- Energy as a built-in "done" signal and compute budget
- Training by pull-down / push-up, no unrolled trajectories
- Sparsity turns weights into readable rules
- Tiny models plus a strong prior beat big generic ones on structured tasks

**Doesn't transfer (or contradicts)**

- CRAM works because puzzles have exact pairwise constraints; language knowledge doesn't
- The authors say it is not a general reasoner, and Maze-Hard needed hand-chosen hidden units
- No rewriting, enrichment or self-improvement anywhere in the paper
- No retrieval experiments; energy gains on RAG are unproven
- Your own self-annealing bank (H5) is a hard negative, so self-organizing consolidation has already failed once here

## Open questions

1. By rewrites, do you mean LLM text rewrites of documents (zone cards, paraphrases) or rewriting the stored vectors? The evidence favors the first.
2. Where should footprints come from: real query logs (which corpus/app?), or research-folder usage (bookmarks, ledgers)?
3. Is the target retrieval quality (nDCG) or answer quality? Zone cards mostly pay off on answers and citations.
4. Branch `factory/c1` (HEAD `84d5cef`, 162 commits past `2bac5d3` per the Nov 9 status) is not on the GitHub remote, and Spark work is off-limits, so this write-up did not read its code. Should improvements target `main` or `factory/c1`?
5. Issue [#106](https://github.com/mattmre/CHELATEDAI/issues/106) (Localized assimilation) is the closest existing write-up of this idea. Fold this page into it, or keep it separate?

## Sources

| Source | Detail | Status |
|--------|--------|--------|
| X thread | Main post, 9 follow-ups and the author's replies to 5 questions, read via X | Read |
| Paper and blog | arXiv 2611.17309 (method, Tables 1/2/5, App. D) and prefrontal.ai/news/cram | Read |
| CHELATEDAI on GitHub | `main` @ `e40c088`: README, VISION_LIQUIFIED_LATTICE, ROADMAP_EXECUTION, next-session, drift results, core modules; 9 open issues, 6+ open PRs (all review/preservation/fix work, none on this topic) | Read |
| `D:\HARK_CHELATEDAI` | Not a git repo: dated status notes plus archive/log zips. Read the two status notes only | Partial |
| `D:\RESEARCH` | No chelated-specific research notes beyond an X-analysis copy of the README. The ledger lists 2605.15422 as "Agent orchestration", NEW | Searched |
| arXiv 2605.15422 | This is DualKV, a shared-prompt FlashAttention kernel for GRPO/DAPO training. It's unrelated to segmentation or consolidation; it only matters if the EGV Qwen RL training lane resumes | Tangential |
| `factory/c1` branch code | Not on the remote; not accessed | Not accessed |

- [Ben Hoover thread](https://x.com/Ben_Hoov/status/2121294692968993068) — x.com, 2026-11-13
- [Classical Energies Learn to Reason 1,000× Faster than Looped Models](https://arxiv.org/abs/2611.17309) — arxiv.org
- [Reasoning at Hyperspeed (blog + live demo)](https://prefrontal.ai/news/cram) — prefrontal.ai
- [mattmre/CHELATEDAI](https://github.com/mattmre/CHELATEDAI) — github.com
- [Issue #106: Localized assimilation](https://github.com/mattmre/CHELATEDAI/issues/106) — github.com
