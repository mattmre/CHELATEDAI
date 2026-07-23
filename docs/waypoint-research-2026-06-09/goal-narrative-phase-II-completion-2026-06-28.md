# Goal Narrative — Complete the Liquified Lattice, Then Earn the Paper

**Date:** 2026-06-28 · **Status:** Active north-star execution narrative · **Scope:** local-only strategy doc (not committed)
**Supersedes the publish-now plan:** the operator decision (2026-06-28) is to **complete the nine-rung Phase II program first**, then formalize the research as a capstone paper — not to ship the Step-14 methodology note now.

---

## 0. The narrative — why now, and what changed

For almost the entire life of this program, the central mechanism did not work. The "self-annealing lattice" had detectors that fired, adapters that initialized, an annealing schedule that ran — and through all of it, **the correction loop never moved a single stored vector.** The synthetic-drift sprint proved it cold: `correction_applied = 0/36`, the store byte-identical before and after. Every forward rung of the roadmap — disintegration (13), the GNN (15) — was gated behind one unmet condition: *a correction that actually changes retrieval.*

That condition is now met. In the encoder-upgrade arena, with an explicit pre-drift supervision signal, the actuator fired on every cycle of every seed across two datasets, and retrieval *partially* recovered — ~19% of the oracle gap on SciFact, ~2.25% on NFCorpus. **The loop moved its first vector.** That is the pivot this narrative is built on.

But be honest about the size of the win. It is a **hairline crack, not an open door**:

- It is **partial** — the loop closes ~19% of the oracle gap on SciFact and ~2.25% on NFCorpus. Never full recovery.
- It is **one-shot** — the "12-cycle loop" is a flat step function; the correction lands at cycle 1 and is re-applied identically. There is no living dynamic yet.
- It is **one arena** — query/encoder upgrade only. The synthetic regime still never engages.
- It uses the **established shim family** — a single bounded near-identity adapter, which is prior art (LoRA/Search-Adaptor lineage), not our novelty.

So the mission of this phase is not to celebrate the crack. It is to **widen it into a real, living mechanism**, then build the rest of the lattice on top of that mechanism — so that when we finally write the paper, it is a capstone over a working system, not a methodology note over a stub.

**The two things that are already ours and already proven** — the *C2-oracle hazard* and the *supervision-signal mechanism* — are bankable methodology findings, and the field is converging fast (steering-post banks, flow steering, masking all landed Feb–May 2026). Completing the program before publishing is the operator's call and it is a sound one, **but it carries scoop risk.** The mitigation is non-blocking and runs in parallel from day one (see §5, Priority Insurance): timestamp the two findings on Zenodo now so finishing the program costs us nothing in priority.

---

## 1. Where the nine rungs actually stand

The Phase II program is steps 9–17 of `VISION_LIQUIFIED_LATTICE.md`. Honest current status
(updated 2026-06-29 **POST-VERDICT**): the rung-14 head-to-head VERDICT campaign RAN on the 3090
and **FALSIFIED the load-bearing thesis** — the living bank does NOT beat the static bank
(C5 ≡ C5s to 15 s.f., 3 seeds, both datasets; `living_bank_wins=False`). Worse, the whole corrector
line is beaten ~4.5× by a one-line `np.linalg.lstsq` baseline (real, exact arena). **Rungs 13/15
(which were *gated on a hardened 14*) are therefore moot — there is no living structure to build on.**
The program does NOT continue as a rung-ladder; the surviving asset is a methodology/hazards paper
(3 runtime-verified hazards incl. the new trivial-baseline finding). Rungs 9 & 11 done · 10
substrate-done · 12 partial-but-orphaned · **14 RAN → thesis falsified** · 13/15/16/17 moot/untouched:

| # | Rung | Status | Honest reality |
|---|---|---|---|
| 9 | Model-Scope shadow + provenance | **done** | bounded steering policy, persist/load round-trip exercised |
| 10 | SHIM substrate DoD | **substrate done** | A1a executable rollback + A1b quant+rollback promotion gate + A1c default-safe control plane + DoD (#284–286); one run-script adoption step deferred per the DoD |
| 11 | Annealing controller | **done** | H6 temperature schedule (#280) drives the C5 living-bank explore/stabilize/prune lifecycle (#283) |
| 12 | Evidence DAG schema | **partial** | typed schema + validator + JSON-schema + deterministic pool builder landed (PR #277, `81e3614`); live wiring into the loop is rung 13, gated |
| 13 | Disintegration loop | **MOOT** | was gated on a hardened rung-14 living bank; 14 falsified → no living structure to operate on. Dead unless the thesis is re-conceived. |
| 14 | Concept-drift experiment | **RAN → THESIS FALSIFIED** | verdict campaign executed (3 seeds, both datasets): C5 living ≡ C5s static to 15 s.f., `living_bank_wins=False`; lifecycle is a no-op. Corrector beaten ~4.5× by trivial `lstsq`. The experiment is done; the thesis it tested is disproven. |
| 15 | GNN prototype | gated · not started | needs 12 + 13 + 14 green; must beat a flat-pool baseline |
| 16 | Quant-aware shim routing | not started | adapter router + quantization promotion gate not built |
| 17 | Disk pool slice | not started | block-graph transport proof exists; pool-shard read integration pending |

**Banked:** rung 9, the harness, the arena, and now THREE runtime-verified methodology hazards
(C2-oracle, supervision-signal, and the 2026-06-29 trivial-baseline/`lstsq`-beats-corrector finding).
**Real work remaining:** NONE as a rung-ladder — rung 14's falsification removes the living structure
rungs 13/15 depend on. The honest path forward is a **methodology/hazards paper** (experiments done,
real), or a re-conception of the correction mechanism (the chelation bounded-near-identity premise is
the wrong tool for catastrophic drift; supervision signal dominates — see `review-notes-cleanup-pass.md`).
Operator decision pending; papers deferred per 2026-06-29.

---

## 2. Completing the current phase — harden Step 14 into a foundation

Rung 14 is "partial." It does not get to stay partial, and it does not get to be declared complete by writing the word. Completion means the loop becomes a mechanism the rest of the lattice can stand on. Five deliverables, each with a runtime definition-of-done:

**14.1 — Close the five invariant/honesty fixes (paper + harness).**
The NFCorpus C3a eval-baseline divergence (bound-floor contamination of the "same eval subset" claim), the "~25× not 26×" correction, stating the 2.25% oracle-gap explicitly, disclosing the one-shot trajectory, and clarifying the 60-query scored subset.
*DoD:* the NFCorpus campaign re-run so all five conditions share one per-seed baseline (or an explicit, evidenced disclosure of why they don't); paper numbers match manifests exactly; every "×N over frozen" multiplier is reported alongside the absolute oracle-gap fraction (19.0% SciFact / 2.25% NFCorpus), because C0 ≈ 0.006 is floor noise and the multiplier alone misleads; and the inverted budget effect is stated plainly (more training steps raise SciFact recovery but not NFCorpus; the unbounded variant beats bounded only on SciFact).
> **Status 2026-06-28:** harness eval-split fix DONE — **H1 merged (PR #276, `d5fc7df`), Tier-B 100/none**; runtime evidence shows all 5 conditions share one NFCorpus baseline (C3a−C0=0.0). The paper-side honesty edits + the full 2-dataset re-run that refreshes the C3a numbers are **H2 (in progress, branch `feat/drift-h2-swap-rerun`)**, tracked as CD-H1-01.

**14.2 — Teacher-supervised C3b.**
A second supervision source (teacher re-embedding of anchor docs) alongside the held-out InfoNCE pairs, to probe whether richer supervision raises the capacity ceiling.
*DoD:* C3b run on both datasets, 3 seeds; one of three pre-registered outcomes reported honestly (raises ceiling / matches C3a / degrades), with the correction norm proving it is learning, not bound-floor.

**14.3 — Resolve the one-shot question.**
Either make the loop genuinely iterate (cycle N consumes cycle N-1's corrected state and improves), or characterize honestly and mechanistically why one-shot is the correct fixed point in this arena. We do not ship a "closed loop" framing over a step function without one of these.
*DoD:* a trajectory that either improves across cycles, or a written mechanism + the decision to call it one-shot supervised realignment in all surfaces.

**14.4 — The post-bank proving experiment (the bridge rung).**
This is the load-bearing deliverable. Replace the single adapter with a **bank of steering posts** and show an **annealed-route bank recovers retrieval where a frozen SVF-style static bank degrades AND a one-shot LD-MoLE-style router degrades.** The baselines are handed to us by the literature.
*DoD:* head-to-head on the drift fixture, 3 seeds; the living bank beats both static baselines on final NDCG with the lifecycle (prune + re-anneal) actually exercised — store mutation logged, not asserted. **This single experiment is what converts the headline thesis novelty from speculation to evidence, and it is what gives rungs 13 and 15 a living structure to operate on.**

**14.5 — Finish the annealing controller (rung 11) as the controller for the bank.**
Promote rung 11 from "gates one adapter cycle" to "owns one temperature schedule across the post-bank's explore/stabilize/prune phases."
*DoD:* one module drives the bank's anneal/disintegrate temperature; sedimentation, online update, and ES read from it; idempotent under re-seed.

**Phase-complete gate (all must hold, with runtime evidence):** a non-zero applied correction that changes retrieval, reproduced across seeds; the post-bank beating static baselines; the annealing controller driving the bank; carried debt held at zero (CD-247-01/02 were already closed by PR #267 on 2026-06-13 — the gate is that no *new* debt has accrued, not that these re-open); and `VISION_LIQUIFIED_LATTICE.md` + roadmap claims downgraded from "self-healing" to "partial recovery, characterized ceiling, encoder-upgrade arena."

---

## 3. Implementing the next phase — the remaining rungs, in dependency order

With 14 hardened into a living bank, the rest of the lattice has something real to build on. Four tracks, gated, not parallelized past their dependencies.

**Track A — Control plane (rungs 10, 11-finalized).**
Rung 10 SHIM substrate: move the shim seams out of env-only research guards into production-wired control planes with a rollback test. Rung 11 lands fully in §2.5.
*Unlocks:* hot-swappable, promotable correction routes — the substrate the bank routes over.
*Gate:* quantization-survival check passes for a promoted shim route.

**Track B — The living structure (rungs 12, 13).** *Gated on 14-hardened.*
Rung 12 Evidence DAG schema: a typed query ↔ cluster ↔ **correction-actuator** graph (adapters/masks/routes as first-class nodes), JSON schema + validator, no GNN yet. Rung 13 Disintegration loop: detector-triggered prune of low-fitness edges/pool entries, re-anneal recording fitness before/after.
*Unlocks:* the graph over which drift-triggered pruning and re-annealing operate — the structural form of "disintegration."
*Gate:* edge/node count stable or decreasing under sparsification **without recall collapse** on the drift fixture.

**Track C — Intelligence (rung 15).** *Gated on 12 + 13 + 14 all green.*
GNN prototype over the evidence DAG.
*Unlocks:* learned structure over the lattice.
*Gate:* the GNN must **beat a flat-pool baseline** on the drift fixture, or it does not promote. No GNN for its own sake.

**Track D — Scale and production (rungs 16, 17).** *Parallelizable with B/C once A is done.*
Rung 16 quant-aware shim routing: adapter router + quantization promotion gate as a steering plane. Rung 17 disk pool slice: one precomputed pool shard readable via block-graph with host parity.
*Unlocks:* the production and disk-scale endgame.
*Gate:* shim survival pass rate ≥ documented threshold; one end-to-end shard read with parity check.

**Critical path:** 14-hardened → (11, 12) → 13 → 15. Tracks A and D run alongside where dependencies allow. Honor the execution policy: one track promotes at a time; nothing clears its gate without runtime evidence that changes retrieval.

---

## 4. The capstone — formalizing the research

Only when the program is whole does the paper change shape. It graduates from a **methodology note** (two hazards + a harness + a partial worked example) into the **full Liquified-Lattice system paper**:

- The two proven hazards (oracle baseline, supervision signal) become the *framing and motivation*.
- The headline result becomes **"a living, annealed bank of correction posts recovers retrieval under drift where static banks and one-shot routers fail"** — backed by Track B/C evidence.
- The DAG/GNN and the disintegration lifecycle become *results*, not future work.

**What the capstone paper requires before it can be written:** rungs 12–15 green with their gates met, the post-bank head-to-head reproduced, and drift-recovery measured against the SVF and LD-MoLE baselines across ≥2 datasets. Until then, the paper stays a draft and the findings stay timestamped (§5).

---

## 5. Priority insurance (non-blocking, starts day one)

The two methodology findings are publishable today and the surrounding field is converging monthly. Completing the program first must not cost us priority. Therefore, in parallel and without blocking program work: **publish a versioned Zenodo DOI of the harness + the two hazard findings now.** A versioned artifact makes no "finished result" promise, so it costs zero credibility while timestamping the contributions. If a competing paper lands mid-program, we have the date.

---

## 6. Operating principles (carried from the conventions, non-negotiable)

1. **Frozen base weights.** Promotion targets adapters, overlays, routes, pool shards, shim registries — never base weights.
2. **Evidence before promotion.** No rung clears its gate on docs, tests, or assertions alone — only runtime evidence that changes retrieval on the fixture.
3. **Fresh-adversarial Tier-B per slice.** Every slice is implemented by one agent and a second fresh agent tries to *disprove* completion. `BHS_OFFICIAL = 100` to merge.
4. **One track at a time.** Per execution policy; no parallel promotion across tracks.
5. **Honest scope on every surface.** Partial is called partial; one-shot is called one-shot; speculation stays in "future work." The vision doc tells the truth about the loop's current reach.
6. **The gate is retrieval, not the actuator firing.** "Correction applied" is not the bar. "Retrieval changed, measurably, reproducibly, and better than the static baseline" is the bar.

---

## 7. Success metrics (program-level, from the vision doc, made concrete)

| Metric | Target signal |
|---|---|
| Drift recovery | NDCG/MRR recovers a growing fraction of the oracle gap within N anneal cycles; living bank > static bank > one-shot router |
| Graph health | Edge/node count stable or decreasing under sparsification without recall collapse |
| Shim survival | Quantization gate pass rate for promoted shim routes ≥ documented threshold |
| Pool freshness | Lazy update latency and staleness bounds documented per shard |
| Disk path | One end-to-end read of a pool shard via block-graph payload with parity check |

---

## 8. The one-line version

> The loop finally moved a vector. Now we *attempt* to turn that one-shot crack into a living, annealed bank of correction posts (hardening rung 14) — the bank-beats-static-bank-and-router result is the experiment to run, not a foregone conclusion. Then we build the typed graph and disintegration lifecycle on top of it (rungs 12–13), let a GNN learn over it only if it beats the flat baseline (rung 15), and wire the control and disk planes around it (rungs 10, 16, 17). Only when the lattice is whole — each rung's gate met by runtime evidence — do we write the capstone paper, with our two already-proven hazards as its frame. We Zenodo-timestamp those two findings today so finishing the program never costs us the priority date.
