# Verified Claims and Sources — Deep-Research Run 2026-06-09

Raw output of the adversarial verification stage. Each confirmed claim survived a 3-vote
panel instructed to *refute* it (vote shown as confirm-refute). Run stats: 5 search
angles, 24 sources fetched, 120 claims extracted, top 25 verified → 19 confirmed,
1 genuinely refuted, 5 killed by abstention (session limit). The synthesis agent also hit
the limit; synthesis was done manually in [novelty-assessment-five-concepts.md](novelty-assessment-five-concepts.md).

## Confirmed claims (19)

### Search-Adaptor — arXiv 2310.08750 (ACL 2024)

1. **(3-0)** Performs post-hoc embedding correction with the base model fully frozen — no
   weights/gradients needed, works on API-only embedding models.
   > "Search-Adaptor does not require access to the LLM weights or gradients and can even
   > be applied to the LLMs that are only accessible via prediction APIs."
2. **(3-0)** The adapter is an additive residual MLP via skip connection — structurally
   the same "small correction added to a frozen embedding" design as chelation adapters.
   > "A learnable adaptation function is defined as f:ℝᵈ→ℝᵈ ... with a skip connection:
   > q̂eᵢ=qeᵢ+f(qeᵢ)"
3. **(3-0)** Supervised-only (labeled query-corpus pairs, ranking loss); no collapse/drift
   detection, no per-query masking, no continual/online adaptation — the
   detection-triggered, self-correcting parts of Concept 1 are NOT covered.

### DIME — ACM 10.1145/3626772.3657691 (SIGIR 2024)

4. **(3-0)** Defines query-dependent dimension importance estimators selecting per-query
   dimension subsets — per-query dynamic dimension masking is established prior art.
5. **(3-0)** Masking zeroes dimensions of existing embeddings at retrieval time, no
   retraining of the base encoder.
6. **(3-0)** Formalizes the Manifold Clustering hypothesis: queries + relevant documents
   lie in a query-dependent lower-dimensional manifold.

### Query-aware dimension masking — arXiv 2602.03306 (Feb 2026)

7. **(3-0)** Trains a lightweight predictor (single FC layer + softmax) mapping a frozen
   query embedding to per-dimension importance; selects query-specific subsets at
   inference. Directly overlaps `dimension_mask_predictor.py`.
8. **(3-0)** Central finding: query-aware masking improves retrieval *effectiveness* —
   many dimensions are actively harmful per query. Published prior work as of Feb 2026.

### Semantic Shift — arXiv 2603.21437 (Mar 2026)

9. **(3-0)** Formalizes embedding-concentration detection with metric
   `Shift(k) = Local(k) · Disp(k)` — collapse detection via geometric statistics is an
   active, formalized research area.
10. **(3-0)** Its remedy is text segmentation (Semantic Shift Splitter), NOT post-hoc
    embedding correction — no adapters, no bounded corrections, no masking.

### Quake — arXiv 2506.03437 (2025)

11. **(3-0)** Adaptive vector-search indexing for dynamic/skewed workloads with evolving
    distributions — direct prior art for vector-index lifecycle under drift.
12. **(3-0)** Restructures via multi-level partitioning guided by a predictive cost
    model — functionally overlaps "disintegration + re-anneal" but framed as cost
    optimization, not annealing.

### Ada-IVF — arXiv 2411.00970 (2024)

13. **(3-0)** Adaptive maintenance policy decides which IVF partitions are degraded and
    repartitions them — drift-triggered, targeted teardown of stale structure.
14. **(3-0)** Local re-clustering mechanism rebuilds targeted partitions — a
    teardown-then-rebuild cycle analogous to disintegration→re-anneal, with no
    temperature/annealing controller and no coupling to embedding-correction training.
15. **(3-0)** Scope is index-structure maintenance only; no claim about correcting or
    retraining embeddings — the annealing-coupled correction training is NOT covered.

### FreshDiskANN — arXiv 2105.09613 (2021)

16. **(3-0)** First graph-based ANN index reflecting corpus updates in real time without
    full rebuilds — lifecycle management of vector indexes solved at systems level
    pre-2024 (framed as update consolidation, not annealing).
17. **(3-0)** Pre-FreshDiskANN graph indices were static with prohibitive rebuilds — the
    underlying pain point is well-trodden.

### SmartVector — arXiv 2604.20598 (Apr 2026)

18. **(3-0)** Five-stage lifecycle for vectors under drift (encoding, consolidation,
    retrieval/reconsolidation, decay, supersession), brain-inspired — lifecycle management
    of vector stores is actively published, via metadata/scoring not retraining.
19. **(2-0, 1 abstain)** Handles stale vectors by deprioritization, not deletion: below
    0.15 confidence → DORMANT (still indexed, archived) — "disintegration" as deliberate
    pruning is NOT what this closest lifecycle work does.

## Genuinely refuted (1)

- **(0-3)** ~~"Search-Adaptor explicitly bounds corrections via a recovery-regularization
  L1 penalty, so bounded/near-identity correction is already published prior art."~~
  REFUTED — bounded-correction (BoundedAdapter / INT8-safe budget) is NOT covered by the
  closest prior work. This strengthens the bounded-correction novelty claim.

## Killed by abstention (session limit — treat as UNVERIFIED, not false)

- Zep/Graphiti (arXiv 2501.13956): typed bi-temporal dynamic KG for agent memory;
  stale-edge invalidation (non-lossy, no pruning); classical retrieval steering, no GNN.
- HippoRAG 2 (arXiv 2502.14802): typed graph over corpus (phrase/passage nodes; relation,
  synonym, contains edges); no query nodes or correction-actuator nodes.
- SmartVector: "explicitly avoids any retraining/correction of embeddings" (1-0, 2 abstain).

## Full source list (24 fetched)

| Source | Quality | Notes |
|---|---|---|
| [arXiv 2310.08750](https://arxiv.org/abs/2310.08750) | primary | Search-Adaptor |
| [ACM 10.1145/3626772.3657691](https://dl.acm.org/doi/10.1145/3626772.3657691) | primary | DIME |
| [arXiv 2602.03306](https://arxiv.org/html/2602.03306v2) | primary | Query-aware masking |
| [arXiv 2603.21437](https://arxiv.org/html/2603.21437v1) | primary | Semantic Shift |
| [arXiv 2506.03437](https://arxiv.org/abs/2506.03437) | primary | Quake |
| [arXiv 2411.00970](https://arxiv.org/abs/2411.00970) | primary | Ada-IVF |
| [arXiv 2105.09613](https://arxiv.org/abs/2105.09613) | primary | FreshDiskANN |
| [arXiv 2604.20598](https://arxiv.org/html/2604.20598v1) | primary | SmartVector |
| [arXiv 2501.13956](https://arxiv.org/abs/2501.13956) | primary | Zep/Graphiti |
| [arXiv 2502.14802](https://arxiv.org/abs/2502.14802) | primary | HippoRAG 2 |
| [arXiv 2405.20139](https://arxiv.org/abs/2405.20139) | primary | GNN-RAG |
| [arXiv 2603.11768](https://arxiv.org/html/2603.11768) | primary | (Concept 3/4 angle) |
| [arXiv 2604.12285](https://arxiv.org/html/2604.12285v1) | primary | (Concept 3/4 angle) |
| [Awesome-GraphRAG](https://github.com/DEEP-PolyU/Awesome-GraphRAG) | secondary | GraphRAG survey repo |
| [arXiv 2509.23471](https://arxiv.org/abs/2509.23471) | primary | (Concept 4 angle) |
| [arXiv 2510.13406](https://arxiv.org/html/2510.13406v1) | primary | (Concept 4 angle) |
| [arXiv 2505.12540](https://arxiv.org/html/2505.12540v2) | primary | (Concept 4 angle) |
| [arXiv 2209.15430](https://arxiv.org/abs/2209.15430) | primary | (model-compat angle) |
| [ACM 10.1145/3736589](https://dl.acm.org/doi/10.1145/3736589) | primary | (Concept 5 angle) |
| [arXiv 2312.03141](https://arxiv.org/abs/2312.03141) | primary | (in-storage ANN angle) |
| [arXiv 2601.01937](https://arxiv.org/pdf/2601.01937) | primary | (Concept 5 angle) |
| [arXiv 2312.11514](https://arxiv.org/abs/2312.11514) | primary | LLM in a Flash |
| [arXiv 2312.04257](https://arxiv.org/pdf/2312.04257) | primary | (Concept 5 angle) |
| [arXiv 2603.01779](https://arxiv.org/pdf/2603.01779) | primary | (Concept 5 angle) |

Concept 4/5 sources were fetched but their claims never reached verification — the
"(angle)" rows are inputs to a future verification pass, not assessed evidence.

## Shim deep-dive sources (direct search 2026-06-09, separate from the run)

- [Steering Vector Fields — arXiv 2602.01654](https://arxiv.org/abs/2602.01654)
- [FLAS — arXiv 2605.05892](https://arxiv.org/abs/2605.05892)
- [LD-MoLE — arXiv 2509.25684](https://arxiv.org/abs/2509.25684)
- [Queryable LoRA — arXiv 2605.08423](https://arxiv.org/html/2605.08423)
- [Activation Steering Field Guide 2026](https://subhadipmitra.com/blog/2026/activation-steering-field-guide/)
- [GrAInS — arXiv 2507.18043](https://arxiv.org/pdf/2507.18043)
