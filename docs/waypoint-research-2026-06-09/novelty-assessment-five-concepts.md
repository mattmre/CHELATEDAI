# Novelty Assessment: Five Core Concepts vs. 2024–2026 Literature

**Date:** 2026-06-09 · **Method:** deep-research harness (5 search angles, 24 primary
sources fetched, 120 claims extracted, top 25 adversarially verified by 3-vote panels)
plus synthesis. Verification coverage caveats in [README.md](README.md).

**Concepts assessed** (from `docs/VISION_LIQUIFIED_LATTICE.md`, `docs/RESEARCH_TRACKS.md`,
Session 30 dual-hemisphere notes):

1. Semantic collapse detection + post-hoc bounded adapter correction + per-query dimension masking
2. Self-annealing vector pools — temperature controller + drift-triggered "disintegration" + re-anneal
3. Evidence DAG/GNN linking queries, documents, and correction-actuators
4. Correction-geometry meta-space (congruent/parallel subspaces, invariant intercepts, persistence across base-model swaps)
5. Computational-storage / disk-resident retrieval and model subgraphs

---

## Concept 1 — Semantic collapse + adapter correction + masking

**Verdict: mostly trodden piecewise; the closed detect→correct loop and bounded corrections remain open.**

Verified prior art (all 3-0):

- **Search-Adaptor (Google, ACL 2024)** — [arXiv 2310.08750](https://arxiv.org/abs/2310.08750).
  Additive residual MLP on frozen embeddings: `new = original + f(original)`. Works on
  API-only embedding models (no weights/gradients needed). Structurally the same design
  as chelation adapters. BUT: supervised-only (labeled query-corpus pairs, ranking loss);
  no collapse/drift detection, no masking, no continual/online adaptation.
- **DIME (SIGIR 2024)** — [ACM DL 10.1145/3626772.3657691](https://dl.acm.org/doi/10.1145/3626772.3657691).
  Query-dependent Dimension IMportance Estimators; zero out dimensions per query at
  retrieval time, frozen encoder. Formalizes the Manifold Clustering hypothesis (query +
  relevant docs lie in a query-dependent lower-dimensional manifold) — the theoretical
  version of the "some dimensions are noise for this query" intuition.
- **Query-aware dimension masking (Feb 2026)** — [arXiv 2602.03306](https://arxiv.org/html/2602.03306v2).
  Trains a lightweight predictor (single FC layer + softmax) mapping a frozen query
  embedding to per-dimension importance; selects query-specific dimension subsets at
  inference. **This is nearly exactly `dimension_mask_predictor.py`.** Central finding:
  masking improves retrieval *effectiveness*, not just efficiency — many dimensions are
  actively harmful per query.
- **Semantic Shift detection (Mar 2026)** — [arXiv 2603.21437](https://arxiv.org/html/2603.21437v1).
  Formalizes embedding-concentration detection with metric `Shift(k) = Local(k) · Disp(k)`.
  Remedy is *text segmentation* (Semantic Shift Splitter), NOT embedding correction — no
  adapters, no masking, no bounded corrections.

**What remains uncovered (verified):**

- The claim that Search-Adaptor already covers bounded/near-identity corrections was
  **refuted 0-3** — BoundedAdapter-style INT8-safe correction budgets are not in the
  closest prior work.
- No single published work couples detection → triggered bounded correction → continual
  online adaptation as one loop. Detection papers stop at detection or segment text;
  correction papers are supervised one-shot.

---

## Concept 2 — Self-annealing pools / disintegration

**Verdict: the problem is old and well-solved at the index-structure level; coupling
lifecycle to embedding-correction training under one controller is unpublished.**

Verified prior art (all 3-0):

- **FreshDiskANN (Microsoft, 2021)** — [arXiv 2105.09613](https://arxiv.org/abs/2105.09613).
  First graph-based ANN index with real-time inserts/deletes, no periodic full rebuilds.
  The pain point (static indices requiring expensive rebuilds) is the exact pain point
  disintegration+re-anneal targets — solved at systems level pre-2024.
- **Ada-IVF (2024)** — [arXiv 2411.00970](https://arxiv.org/abs/2411.00970). **Closest
  single match for "disintegration."** Adaptive maintenance policy monitors which IVF
  partitions have degraded and triggers targeted teardown + local re-clustering. That is
  functionally drift-triggered disintegration followed by re-anneal — framed as database
  index maintenance, no temperature controller, and explicitly does NOT touch the
  embeddings themselves (index structure only).
- **Quake (2025)** — [arXiv 2506.03437](https://arxiv.org/abs/2506.03437). Adaptive
  vector indexing for dynamic/skewed workloads; multi-level partitioning adjusted by a
  predictive cost model. Cost-model-driven restructure ≈ disintegration/re-anneal framed
  as cost optimization.
- **SmartVector (Apr 2026)** — [arXiv 2604.20598](https://arxiv.org/html/2604.20598v1).
  Five-stage brain-inspired vector lifecycle: encoding → consolidation →
  retrieval/reconsolidation → decay → supersession. Handles stale vectors by
  *deprioritization* (DORMANT below 0.15 confidence — still indexed, archived for audit),
  NOT deletion/pruning (verified 2-0).

**What remains uncovered (verified):** none of these retrain or correct embeddings; none
couples index lifecycle to correction training; no unified annealing/temperature schedule
spans both. Honest warning: "annealing" is metaphor until it demonstrably does something a
cost model doesn't — a reviewer will ask exactly that.

---

## Concept 3 — Evidence DAG / GNN with correction-actuators

**Verdict: typed dynamic graph memory is crowded territory; correction-actuators as graph
nodes appear to be unclaimed. (Partially verified — votes cut off by session limit.)**

Likely prior art (extracted, unverified):

- **Zep / Graphiti (Jan 2025)** — [arXiv 2501.13956](https://arxiv.org/abs/2501.13956).
  Typed, hierarchical, bi-temporal dynamic knowledge graph for agent memory (episodes /
  semantic entities / communities). Stale knowledge handled by *edge invalidation*
  (non-lossy), not pruning. Retrieval steered by cosine/BM25/BFS + rerankers — **no GNN**.
- **HippoRAG 2 (ICML 2025)** — [arXiv 2502.14802](https://arxiv.org/abs/2502.14802).
  Typed graph over corpus via LLM OpenIE: phrase + passage nodes, three edge types
  (relation, synonym, contains). No query nodes, no correction-actuator nodes.
- **GNN-RAG** — [arXiv 2405.20139](https://arxiv.org/abs/2405.20139) — GNN-based
  retrieval over KGs exists as a line of work.
- GraphRAG ecosystem is large; see [Awesome-GraphRAG](https://github.com/DEEP-PolyU/Awesome-GraphRAG).

**What appears uncovered:** linking queries → documents → *the correction machinery that
fixed them* (adapters, masks, steering routes) in one typed graph, and using that graph to
steer/prune. No found system has actuator nodes. Drift-triggered *pruning* of graph
structure (vs. invalidation) also appears open in this space.

---

## Concept 4 — Correction-geometry meta-space

**Verdict: most conceptually novel of the five; least developed; verification never
reached this concept (model knowledge + fetched-source list only).**

Adjacent territory:

- Procrustes/embedding-space alignment, backward-compatible training (BCT), forward-
  compatible training (FCT) — cheap re-sync across model upgrades is established.
- **Drift-Adapter (EMNLP 2025)** — already cited in Session 30 notes; validates the three
  adapter types for bridging model swaps.
- **Task arithmetic / model editing** — task vectors compose additively; orthogonality
  between edits ≈ "parallel subspaces"; equivalent edits ≈ "congruent subspaces." This
  literature rhymes strongly with the subspace taxonomy and must be engaged before
  claiming the framing. (Compositional steering work has independently found that naive
  vector addition causes interference — supporting the need for structured composition.)

**What appears uncovered:** modeling the space of *retrieval corrections* as a structured
manifold (congruent/parallel subspaces, invariant intercepts, fiber-bundle framing) where
the *structure* persists across base-model swaps and only coordinates re-sync. No found
work does this for retrieval correction. Caveat: hardest to make rigorous — fiber-bundle
language without theorems or experiments reads as decoration. The testable kernel: **do
learned correction subspaces transfer across a base-model swap with only cheap
re-alignment?** That is a runnable experiment.

---

## Concept 5 — Computational storage / disk-resident retrieval

**Verdict: well-trodden conceptually; novelty is low; the repo's scope-locked claim
posture is correct. (Verification never reached this concept.)**

Known territory: DiskANN lineage (SSD-resident ANN), Apple's "LLM in a Flash"
([arXiv 2312.11514](https://arxiv.org/abs/2312.11514)) for flash-resident inference,
SmartSSD/in-storage-processing ANN papers (fetched sources included
[arXiv 2312.03141](https://arxiv.org/abs/2312.03141), [arXiv 2312.04257](https://arxiv.org/pdf/2312.04257),
[ACM 10.1145/3736589](https://dl.acm.org/doi/10.1145/3736589)). The conceptual claim
(run ANN/model subgraphs near storage) is established research. Value here is integration
with the lattice control plane, not the storage idea itself.

---

## Overall verdict and recommendations

1. **Combination novelty is real but is the weakest form of novelty** — it only counts
   with end-to-end evidence. The drift-injection recovery experiment (Phase II step 14)
   is the paper: inject drift, show the closed loop recovers NDCG within N anneal cycles,
   ablate against maintenance-only (Ada-IVF-style) and correction-only (Search-Adaptor-style).
2. **Benchmark against the five closest works**: Search-Adaptor, DIME, Ada-IVF, Quake,
   SmartVector. They are now the baselines; engaging them honestly is what separates a
   layman's writeup from a credible one.
3. **The window is closing on components** — Feb–Apr 2026 papers landed independently on
   masking and lifecycle. Prioritize demonstrating the integrated loop, not the parts.
4. The repo's existing gate (GNN only after DAG schema + drift benchmark) is the right
   sequencing.
