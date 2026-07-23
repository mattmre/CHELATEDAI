# Rung 15 (GNN prototype) — CUT decision and ROADMAP edit

**Date:** 2026-07-19. **Decision:** **CUT** (do not build, do not spend GPU). **Phase II closes as
complete-with-one-documented-cut.**

**Verdict confidence:** high, and it does *not* rest on the fail-closed track record. It rests on a
structural property of `evidence_dag.py` that I verified by running the actual builder. If that
property changes, the cut re-opens — see §5.

---

## 1. The honest CUT justification

Three grounds, ordered by strength. I am ranking them deliberately because the weakest one is the one
most likely to be used as a rationalization, and it should not carry the decision.

### Ground A (decisive, verified) — the GNN is not merely "within-span," it is *identical* to the flat pool on this graph

Rung 15's exit criteria bind the GNN to *this* Evidence DAG. I built a DAG at Arena-B scale (300
attribution rows, 3 tasks × 3 profiles × 2 strategies × 4 actions) through the real
`from_attribution_pool` in
`D:\GITHUB\CHELATEDAI\.claude\worktrees\relaxed-wozniak-271e04\evidence_dag.py` and measured its
topology. Result (validator clean, 0 violations):

| Property | Measured |
|---|---:|
| QUERY nodes | 300 |
| CLUSTER nodes | 9 |
| ACTUATOR nodes | 6 |
| Edges | 730 |
| **QUERY in-degree (max)** | **0** |
| QUERY out-degree (max) | 2 |
| CLUSTER out-degree (max) | 0 |
| **query→query edges** | **0** |

Two consequences follow with no hand-waving:

1. **Under message passing along edge direction, a GNN over this DAG is exactly an MLP on node
   features — which is the flat-pool baseline.** Query nodes have in-degree 0 (they are pure
   sources: `from_attribution_pool` only ever emits queries as `src`, lines 419/430 of
   `evidence_dag.py`). A query node therefore receives no message at any depth, so its representation
   after *k* rounds equals its own features for all *k*. "GNN beats flat pool" is not a hypothesis
   here; the two models are the same function class. The measurement would return a difference of
   exactly zero plus optimizer noise.

2. **Under symmetrized/reversed edges (the only way to make queries receive), the GNN's extra
   capacity over the flat pool is exactly "group means over ≤15 buckets."** A query's neighborhood is
   its one `c:{task}:{profile}` node and its one `a:{strategy}:{action}` node. There are 9 and 6 of
   those respectively — 15 total, versus 300 queries. There are **zero query→query edges**, so there
   is no relational signal between queries at all; every path between two queries goes through a
   bucket node they share by construction. The message a query receives is thus a function of two
   categorical ids plus within-bucket feature means. A flat-pool baseline reproduces that span with 15
   one-hot columns and a group-mean column — one line of feature engineering, no graph, no GPU.

So the GNN's hypothesis class over this DAG is contained in
`{node features} ∪ {group means over 15 buckets}`. This is a within-span reparameterization in the
strictest sense, and it is the class the deep-research pass says is dominated.

**Scope discipline — do not invoke the superposition wedge to rescue this.** The Garg–Kleinberg–Peng
quadratic gap (arXiv:2602.11246) is real and proven, and it *is* a genuine argument that linear
readout is not span-exhausting. But it is about nonlinear decoding of *k*-sparse features **within one
embedding space**, and it requires rich superposed feature structure to decode. A depth-2 DAG whose
non-query vertex set has cardinality 15 has no such structure. Citing that result as support for
rung 15 would repeat exactly the scope error the HI-1 red-team flagged (its item C.8: the proven wedge
is within-space feature readout; HI-1 tested cross-space alignment). Rung 15 is a third thing —
graph-topological aggregation over a near-bipartite fan-in — and neither the superposition results nor
the vec2vec results are evidence for or against it.

**This also makes rung 15 unfalsifiable in the HI-1 sense.** A negative result carries information only
if the experiment could have come out otherwise. Here the outcome is fixed by construction. The
endgame plan's consolation — "a clean fail-closed GNN is itself a publishable result" — does not hold
when the fail-closed is structurally guaranteed: "graph learning did not beat the flat pool on a graph
with no exploitable topology" is a statement about our schema, not about graph learning. Decision
value ≈ 0, which is HI-1 red-team finding (d) reached independently.

### Ground B (strong, arithmetic) — the power problem that killed HI-1, computed from this fixture's own observed CIs

I did not simulate this. I took the **observed paired 95% CIs from the rung-16 REPORT evaluations**
(`docs/rung16-quant-aware-routing-results-2026-07.md`), which are per-query paired NDCG differences on
the same drift/swap fixture rung 15 would use.

| Arena | Held-out n | Observed paired 95% CI | Half-width *h* | Implied per-query paired SD |
|---|---:|---|---:|---:|
| A (SciFact swap) | 30 | [−0.051022, 0.041025] | 0.046024 | 0.1286 |
| B (multi-domain) | 90 | [−0.027989, 0.012247] | 0.020118 | 0.0974 |

(SD back-solved from *h* = 1.96·SD/√n.)

Achievable held-out n on this fixture is **30** (Arena A, 40/30/30 split of 100 queries) or **90**
(Arena B, 120/90/90 of 300). Under rung-16's own promotion bar (paired CI lower bound > 0.005), the
true effect the GNN must have to clear:

- **n = 30:** δ > 0.005 + 0.046 = **0.051 absolute NDCG**
- **n = 90:** δ > 0.005 + 0.020 = **0.025 absolute NDCG**

Now compare against the largest structural effect this program has ever measured on this fixture:
**+0.0257** — and that is the *favorable post-hoc subset* (home-correct specialists, n = 21, Arena B),
not a preregistered plane-level quantity. The plane-level effects actually measured were **−0.0033**
(Arena A) and **−0.0074** (Arena B).

So, inverting to required n (SD = 0.0974, bar = lower bound > 0.005):

| Assumed true GNN effect | Required held-out n | Available |
|---|---:|---:|
| 0.050 (≈2× best-ever post-hoc subset) | 18 | 90 ✓ |
| 0.0257 (best-ever post-hoc subset) | 86 | 90 — coin flip |
| 0.010 | 1,457 | 90 ✗ |
| 0.0074 (magnitude of observed plane effect) | 6,323 | 90 ✗ |

Read the middle row carefully: **even if the GNN achieved the single most favorable effect size this
program has ever measured anywhere — a post-hoc-selected subset effect, on the arena that most flatters
it — it would land essentially exactly on the bar at n = 90.** That is a coin flip purchased with a GPU
campaign. If its true effect is instead the size of anything measured at *plane* level, clearing the
bar needs ~70× more held-out queries than the fixture contains.

Two honest caveats on this ground, which is why it is B and not A:
- The implied SD is from rung-16's *routing* contrast, not a GNN contrast. A GNN-vs-flat-pool contrast
  could have lower paired variance (both models see the same features). I have **not** measured that;
  treat the SD as an assumption carried from the nearest available measurement, not a fact about
  rung 15.
- Under Ground A the point is partly moot: if the two models are the same function class, the true
  effect is 0 and no n suffices.

**Do not rescue this by counting edges instead of queries.** At Arena-B scale the DAG has 730 edges,
which would look like n = 730. But those edges are deterministic functions of the same 300 rows (each
row emits 1 or 3 edges), so edge-level n is pseudo-replication. Inflating n that way is precisely the
unit-of-analysis error that produced the recoverability estimator's inventory overfit — block-LOO
Spearman +0.886 at 6 cells inverting to −0.706 at 12. Same failure mode, one rung later.

### Ground C (weakest — a prior, not evidence) — the six-fail-closed pattern

Six consecutive results in which elaborate structure lost to a trivial baseline is a strong *prior*. It
is **not** evidence about this specific mechanism, and I want it on the record that **this ground alone
would not justify the cut.** A track record can justify raising the evidentiary bar before spending GPU;
it cannot substitute for an argument about the mechanism. If Grounds A and B did not hold, the correct
call would be to run rung 15 despite the track record. They do hold, so C only sets the budget posture:
given six negatives, we do not spend a GPU campaign on an experiment whose outcome is determined by
construction.

### Would I recommend RUN instead? Honestly considered, and no — but with one live lead preserved

The strongest pro-RUN case is the Tier B nuance from Arena B: domain specialists *helped* when routed to
their own domain (+0.0257, n = 21); the plane lost because centroid routing misrouted 65% of specialist-
served queries under encoder swap. The binding constraint was **route assignment**, not specialist
capacity. That is a genuinely live, unfalsified question, and it is the one thing in this arc that
still looks winnable.

But **a GNN over the Evidence DAG cannot address it.** Route assignment is a function that must map a
*new, unseen* query to a route. In this DAG a new query is an isolated source node with in-degree 0 and
no query→query edges — it has no neighborhood to aggregate over, so message passing supplies it with
literally nothing. The mechanism that would attack the Arena-B finding is a better query→route
assignment function operating on **embeddings** (e.g. supervised routing, or kNN/soft routing in the
swapped space instead of frozen centroid margin), which is a different experiment on a different
substrate. Cutting rung 15 must not bury that lead — §5 records it as the named successor.

---

## 2. Exact ROADMAP edits

File: `docs/ROADMAP_EXECUTION.md`.

### Edit 1 — step-15 row (Phase II table, line 71)

**OLD:**
```
| 15 | **GNN prototype** | **OPEN** (no merged PR; no PyG/DGL code — only a docstring forward-ref in `evidence_dag.py`) | Lightweight GNN over evidence DAG (PyG or DGL); only after steps 12–14 green; must beat flat-pool baseline on drift fixture or fail closed |
```

**NEW:**
```
| 15 | **GNN prototype** | **CUT** (2026-07-19, pre-GPU; no PyG/DGL code — only a docstring forward-ref in `evidence_dag.py`). Cut on a **structural** ground, not on the fail-closed track record: on the rung-12 Evidence DAG a GNN is not merely within-span, it is *identical* to the flat-pool baseline. Measured through the real `from_attribution_pool` at Arena-B scale (300 rows → 300 query / 9 cluster / 6 actuator nodes, 730 edges): **QUERY in-degree = 0** and **zero query→query edges**, so under directed message passing a query never receives a message and the GNN reduces exactly to an MLP on node features. Symmetrized, its only extra capacity is group means over ≤15 bucket nodes — reproducible by 15 one-hots in the flat pool. The comparison is therefore not a hypothesis and a fail-closed would carry ~0 bits. Power is independently prohibitive: achievable held-out n is 30/90, and from rung-16's own observed paired CIs on this fixture (SD ≈ 0.097–0.129) the bar needs a true effect ≥0.025 (n=90) / ≥0.051 (n=30) — larger than the best effect ever measured in this program (+0.0257, a post-hoc subset). Full reasoning + re-open condition: `docs/waypoint-research-2026-06-09/panel/rung15-cut-decision.md`. | Lightweight GNN over evidence DAG (PyG or DGL); only after steps 12–14 green; must beat flat-pool baseline on drift fixture or fail closed. **Not evaluated — cut before GPU as unwinnable-by-construction. NOT a claim the GNN was tested and lost.** |
```

### Edit 2 — status snapshot (lines 54–61)

**OLD:**
```
**Status snapshot (2026-07, git-verified):** rungs **9–13 DONE** (13 completed by the detector-driven
Evidence-DAG edge-prune loop, rung-13 PR); **14 apparatus DONE** with the **H5 living-bank question
closed as a hard negative** (LIVING BANK WINS = False on SciFact + NFCorpus); **16 DONE locally as
an honest non-promotion** (integrated quant-aware plane; both preregistered arenas FAIL-CLOSED; change
set/evidence not yet published); **17 DONE** (block-graph pool-shard read with host parity, rung-17
PR); **15 OPEN**. Lattice apparatus PRs: #260, #277, #279–#291 + rung-13 + rung-17. Remaining
executable feature work is **15 (GNN)** — the endgame program (see
`docs/waypoint-research-2026-06-09/panel/lattice-endgame-plan-2026-07-14.md`).
```

**NEW:**
```
**Status snapshot (2026-07-19, git-verified): Phase II is CLOSED — complete with one documented cut.**
Rungs **9–13 DONE** (13 completed by the detector-driven Evidence-DAG edge-prune loop, rung-13 PR);
**14 apparatus DONE** with the **H5 living-bank question closed as a hard negative** (LIVING BANK WINS
= False on SciFact + NFCorpus); **16 DONE locally as an honest non-promotion** (integrated quant-aware
plane; both preregistered arenas FAIL-CLOSED; change set/evidence not yet published); **17 DONE**
(block-graph pool-shard read with host parity, rung-17 PR); **15 CUT pre-GPU** — the rung-12 Evidence
DAG has no exploitable topology (query in-degree 0, zero query→query edges, 15 bucket nodes), so a GNN
over it collapses to the flat-pool baseline it was supposed to beat; the comparison is not a hypothesis
and was cut rather than run as a foregone fail-closed
(`docs/waypoint-research-2026-06-09/panel/rung15-cut-decision.md`). Lattice apparatus PRs: #260, #277,
#279–#291 + rung-13 + rung-17. **There is no remaining executable Phase II feature work.** The one
live lead surfaced by this program is *route assignment under encoder-swap drift* (rung-16 Tier B:
home-correct specialists +0.0257/n=21 vs misroutes −0.0309/n=39) — that is a new embedding-space
question, not a Phase II rung, and is not carried as Phase II debt.
```

### Edit 3 — "What we are not doing" bullet (line 43)

**OLD:**
```
- No GNN layer or disk-pool integration until Phase II steps 12–14 are honestly closed (schema **#277 DONE**; drift apparatus **DONE** with the H5 living-bank verdict a hard negative; disintegration **DONE** — detector-driven Evidence-DAG edge prune, rung-13 PR). Rungs 16 and 17 are now implemented on their feature branches; rung 15 is the remaining endgame feature.
```

**NEW:**
```
- No GNN layer over the Evidence DAG. Steps 12–14 are honestly closed (schema **#277 DONE**; drift apparatus **DONE** with the H5 living-bank verdict a hard negative; disintegration **DONE** — detector-driven Evidence-DAG edge prune, rung-13 PR), and rungs 16 and 17 are implemented on their feature branches. Rung 15 was then **CUT pre-GPU** (2026-07-19): the rung-12 DAG gives a GNN no topology to exploit (query in-degree 0, zero query→query edges, 15 bucket nodes), so it is identical to — not an improvement on — the flat-pool baseline. Re-open only if the Evidence DAG gains genuine multi-hop or query↔query structure (see `panel/rung15-cut-decision.md` §5). **Phase II is closed; do not open new lattice rungs against it.**
```

### Edit 4 (required for consistency) — Phase II dependency block (lines 81–82)

**OLD:**
```
12 → 15 (GNN needs DAG schema)
13 + 14 → 15 (GNN needs drift benchmark)
```

**NEW:**
```
12 → 15 (GNN needs DAG schema) — MOOT: 15 CUT 2026-07-19
13 + 14 → 15 (GNN needs drift benchmark) — MOOT: 15 CUT 2026-07-19
```

---

## 3. CHANGELOG sentence

```
Cut Phase II rung 15 (GNN over the Evidence DAG) before any GPU spend: measured through the real `from_attribution_pool` builder, the rung-12 DAG has query in-degree 0, zero query→query edges, and only 15 bucket nodes, so a GNN over it is identical to the flat-pool baseline it was required to beat — Phase II closes complete-with-one-documented-cut rather than carrying a foregone fail-closed.
```

---

## 4. What this cut does *not* claim

Stated explicitly so no future reader over-reads it:

- **Not** "GNNs don't work." It is a claim about *this schema*, verified on *this builder*.
- **Not** "the GNN was tested and lost." Nothing was run. The ROADMAP row says so in terms.
- **Not** that the superposition/compressed-sensing wedge is refuted. That wedge is real and proven; it
  is simply out of scope for a 15-vertex depth-2 aggregation graph.
- **Not** that graph structure is worthless to this program. A DAG with real multi-hop structure would
  be a different object entirely.

## 5. Re-open condition (this cut is itself falsifiable)

Rung 15 becomes live again, without further argument, if **any** of these becomes true:

1. `from_attribution_pool` (or a successor builder) emits **query→query edges** — e.g. shared-document,
   shared-cluster-membership, or temporal-succession links — so queries acquire nonzero in-degree and a
   real neighborhood.
2. The DAG acquires **paths longer than 2 hops** that are not determined by a single attribution row.
3. The cluster/actuator vertex count grows to the same order as the query count (i.e. buckets stop
   being a ≤15-way categorical), so group-mean encoding is no longer a cheap exact substitute.
4. A drift fixture with **≥1,500 held-out queries** becomes available *and* a pilot shows a
   GNN-vs-flat-pool paired SD materially below the 0.097 carried here.

**Named successor (the live lead, deliberately not buried):** route assignment under encoder-swap drift.
Rung 16 Arena B falsified the *preregistered centroid-margin plane*, not domain routing — specialists
helped on their home domain (+0.0257, n = 21) and lost only to 65% misrouting (−0.0309, n = 39). A
supervised or kNN/soft router in the swapped embedding space is the experiment that actually tests
that, and it is **not** rung 15 and **not** a GNN over the Evidence DAG. If the program continues, that
is where the GPU should go — and per the lesson that governs this whole arc, its power math must be
done *first*, with the win bar derived from the achievable n rather than asserted ahead of it.
