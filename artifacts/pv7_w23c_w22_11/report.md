# W22-11 Report: Large-Graph Incremental WFS Deletion and Canonical Witness Stress Test

- Preregistration: `PREREGISTRATION.md`, committed in `ea70796b` before any result.
- Results: `w22_11_results.json`.
- Post-hoc mechanism probe: `w22_11_mechanism_probe.json`.

**Verdict (preregistered mapping): `CLEAN_1M_CANONICAL_FROZEN`**

- 1,266,000 mutation checkpoints; 1,370,901 checkpoints including `as_of` rewinds.
- **Semantic mismatches: 0.**
- **Canonical witness (C1) mismatches: 0.**
- Comparator liveness: 100%.
- Pinned source hashes: OK.
- The frozen invariant `CW-1` is recorded in the results JSON.
- **The raw engine telemetry is NOT canonical.** It has 3,077,470 raw `pos_cyc`/`reason` differences. They
  are characterized below and were not hidden by weakening any comparison.

Wall time: 945 s on 16 local cores (CPU only), plus 102 s for the post-hoc probe. Every preregistered chunk
completed (1,140/1,140), and the run did not hit the deadline.

## Reproduce

    python scripts/run_pv7_w23c_w22_11_wfs_deletion_stress.py --workers 16   # main run, ~16 min
    python scripts/w22_11_mechanism_probe.py --sequences 200                  # post-hoc probe, ~2 min
    python -m unittest test_w22_11_wfs_deletion_stress -v                    # unit tests

## Counts

| Tier | nodes max | steps | seqs | mutation cps | as_of cps | cps after a D_V-edge deletion | max D_V SCC | semantic | raw diffs (pos_cyc / reason) | canonical diffs |
|---|---|---|---|---|---|---|---|---|---|---|
| A0 anchor | 11 | 6 | 1,000 | 6,000 | 1,722 | 837 | 4 | 0 | 104 / 49 | 0 |
| A1 anchor | 11 | 12 | 1,000 | 12,000 | 1,881 | 1,802 | 5 | 0 | 548 / 277 | 0 |
| T1 long | 11 | 24 | 30,000 | 720,000 | 59,502 | 116,289 | 11 | 0 | 120,002 / 75,372 | 0 |
| T2 stress-32 | 32 | 24 | 12,000 | 288,000 | 23,996 | 196,580 | 32 | 0 | 387,806 / 315,074 | 0 |
| T3 stress-64 | 64 | 24 | 6,000 | 144,000 | 12,000 | 100,409 | 64 | 0 | 514,465 / 402,837 | 0 |
| T4 stress-128 | 128 | 32 | 2,500 | 80,000 | 5,000 | 58,883 | 128 | 0 | 577,442 / 411,848 | 0 |
| T5 stress-256 | 256 | 40 | 400 | 16,000 | 800 | 12,221 | 255 | 0 | 161,733 / 109,913 | 0 |
| **Total** | | | 52,900 | **1,266,000** | 104,901 | 487,021 | 255 | **0** | **1,762,100 / 1,315,370** | **0** |

- Node-level comparisons: 32,903,661.
- Deletions applied:
  - 1,071,859 attestation withdrawals;
  - 354,349 triple removals;
  - 796,284 of those withdrawals and removals hit D_V edges, including the targeted cycle-breaks.

**Semantic class (the kill rule): 0 across all classes.**
- inc vs fresh on `eff_auth`, `val`, `cur`, `eligibility` and the node set;
- inc `val` vs the textbook alternating-fixpoint WFS oracle, computed from the raw snapshot;
- inc Stratum-0 materialization vs the independent model snapshot;
- the canonicalizer's independent `cur`/`eligibility` re-derivation, checked against the inc, fresh and
  fresh2 stores.

**Canonical witness C1 (`export_canonical_ek5_audit_trace`, unmodified): 0 across all classes.**
- C1(inc) vs C1(fresh) over the full trace: per node `val/cur/eligibility/reason/cause/pos_cyc/witness`,
  plus `positive_cycles`, `supersede_cycles` and `lineage_cycles`.
- C1(fresh) vs C1(fresh2). fresh2 is rebuilt in a different insertion order.
- C1 `reason`/`pos_cyc` vs the fresh store's raw telemetry. This is the canonicalizer's documented claim,
  never previously tested at scale.
- C1 `pos_cyc` vs D_V non-trivial-SCC membership computed from the raw model snapshot.

**Comparator liveness.**

| Class | Faults injected | Faults detected |
|---|---|---|
| Canonical-trace | 151,920 | 151,920 |
| Semantic-state | 126,600 | 126,600 |

**Anchors (P2).** Both anchors reproduced W22-05 exactly:
- A0: 7,722 checkpoints, 153 diffs = 104 `pos_cyc` + 49 `reason`;
- A1: 13,881 checkpoints, 825 diffs = 548 + 277.

So the harness is the same generator and comparator, and the larger tiers extend it rather than replace it.

## Characterization of the raw witness diffs

The task asked which field differs, on which graph shapes, and whether the cause is order dependence.

All 3,077,470 raw diffs share one shape:

| Property | Result |
|---|---|
| Field and direction | 1,762,100 `pos_cyc` diffs, **all** `inc=False, fresh=True` ("missed cycle"). 1,315,370 `reason` diffs, **all** `REFUTED -> UNFOUNDED_CYCLE`. Zero `inc=True, fresh=False` ("stale cycle"), even under 796,284 D_V-edge deletions and targeted cycle-breaks. Zero `blocked`/`quarantined` diffs. |
| Value | 100% on `val = F` nodes. |
| Checkpoint kind | 100% at `mutation` checkpoints. Zero at `as_of`, which are themselves fresh replays. |
| Graph shape | 100% on nodes inside a non-trivial D_V SCC of the final graph. Zero on acyclic nodes. By the node's SCC size: 33+ 1,606,391; 17-32 753,921; 9-16 378,586; 5-8 135,542; 3-4 115,219; 2 76,403; self-loop 11,408. |
| Deletion in the batch | 2,142,956 diffs (70%) occur in a batch that deleted a D_V edge; 934,514 (30%) occur in a batch without one. |
| Order dependence | **None.** fresh vs fresh2 raw telemetry: 0 diffs, and none of the inc-vs-fresh diff nodes is order-dependent. The diffs come from **incremental history**, not insertion order. |

### Post-hoc mechanism probe

This probe was **not preregistered**. It wraps the instance's `_solve_stratum2_region` and records each
batch's dirty region, without changing what the engine computes. It replayed the first 200 sequences of
T1-T3 and the first 25 of T4-T5:

| Tier | missed cycle: node re-solved, but its SCC extends past the dirty region | missed cycle: node not re-solved this batch | unexplained | reason diffs on a node that also has a `pos_cyc` diff |
|---|---|---|---|---|
| T1 | 50 | 822 | 0 | 668 / 668 |
| T2 | 1,817 | 4,983 | 0 | 5,521 / 5,521 |
| T3 | 3,567 | 13,388 | 0 | 13,512 / 13,512 |
| T4 | 848 | 4,557 | 0 | 3,417 / 3,417 |
| T5 | 1,545 | 6,041 | 0 | 4,814 / 4,814 |

Two mechanisms account for every diff in the probe sample:

- **(a) Dirty-restricted Tarjan.** About 21% of the sample. `cyc_nodes = _nontrivial_scc_nodes(dirty,
  deps_v, subset=dirty)` at `ek5_v3_stratified.py:380` only sees cycles that lie wholly inside the dirty
  region. When part of a cycle sits outside the region on a frozen F boundary, the node still gets `val=F`
  through its F dependency, but with `pos_cyc=False`.
- **(b) Pruned nodes keep an old flag.** About 79%. A node that is already F and shielded is skipped by the
  Pitfall-A/B seed and cone pruning (`ek5_v3_stratified.py:641-701`). It is skipped even when this batch
  closes a D_V cycle through it, so its `pos_cyc=False` is left over from an earlier batch.
- **`reason` follows `pos_cyc`.** Every `reason` diff sits on a node that also has a `pos_cyc` diff. The
  Stratum-3 reason only re-reads `pos_cyc`.

**Reading.** The incremental `pos_cyc` behaves as a **sound under-approximation** of "on a positive D_V
cycle". It never claimed a cycle absent from the final graph in 1.27 M mutation checkpoints. But it is
**history-dependent**, so it is not a canonical witness. This is consistent with the Wave-14A disclosure
that blamed Pitfall-B pruning. It sharpens that disclosure in two ways: mechanism (a) also contributes, and
the direction was strictly one-way in this run.

## Canonical witness outcome

C1, the existing spine canonicalizer, used unmodified, is a deterministic normal form. Its outputs:
- `pos_cyc` is non-trivial D_V-SCC membership on the final graph;
- `reason` is re-derived bottom-up;
- each node gets a lexicographically minimal witness tuple;
- the cycle lists are sorted.

At all 1,370,901 checkpoints it was invariant across:
- the incremental store;
- the fresh rebuild;
- a second fresh rebuild with a different insertion order.

It also matched the fresh store's raw telemetry and the snapshot-derived SCC membership.

The frozen invariant, as preregistered:

> **CW-1**: `sha256(json.dumps(export_canonical_ek5_audit_trace(S).to_dict(), sort_keys=True, separators=(",",":")))`
> is equal for the incremental store and every from-scratch rebuild of the same visible log under any
> insertion order. The canonicalizer is pinned at sha256
> `b8440b62b8c20bf28d35a8c961ab671d3ea8ce90c45fd916c53e8b5a3d8dc432`.

**Scope of the freeze.** CW-1 makes the canonical *export* reproducible. It does **not** make the engine's
own `store.reason`/`store.pos_cyc` canonical; those still differ, as shown above. Any audit trail or
byte-level diff must read the C1 export, not the store fields.

**How strong the canonical result is.** It is weaker than the raw count suggests:
- C1 reads only `val` plus the store's edge tables. Both are already required to be equal by the semantic
  and Stratum-0 gates. So C1(inc) == C1(fresh) is **close to implied by construction**.
- The checks that carry real information are three, and all were 0:
  - C1 vs the engine's independently coded fresh-path telemetry;
  - C1 `pos_cyc` vs snapshot SCC membership;
  - C1's own `cur`/`eligibility` re-derivation vs the stores.

## Predictions

| # | Prediction | Outcome |
|---|---|---|
| P1 | semantic 0 | held |
| P2 | anchors 153 / 825 | held, exactly |
| P3 | raw diffs non-zero, only on `val=F` nodes, only at mutation checkpoints | held |
| P4 | fresh-vs-fresh raw diffs 0 | held |
| P5 | C1 diffs 0 and both contract checks 0 | held |

## Lean gate status

The queue item's precondition for funding Lean was 10^6 clean checkpoints plus a frozen invariant. That
precondition is **met by the numbers** (1,266,000 mutation checkpoints, semantic 0, CW-1 frozen). Lean was
**not attempted**, as instructed. The funding decision belongs to the operator after a Tier B review.

## Limits (what this does NOT show)

1. **Same-engine differential.** "fresh" is the same `StratifiedEvidenceStore` code run from scratch. The
   only engine-independent checks are:
   - the WFS oracle, which covers `val` only;
   - the Stratum-0 snapshot model;
   - the canonicalizer re-derivation of `cur`/`eligibility`/`pos_cyc`.

   `eff_auth` is therefore checked inc-vs-fresh only, which is the same code. The independent
   `run_conformance.snapshot_eff_auth` exists, but I did not wire it into the semantic class. That is a gap.
2. **No Lean CLI comparison.** `ek5_ref_cli` was not built or run, so the Wave-14A `lean_vs_inc` /
   `lean_vs_fresh` arms are not extended to the larger graphs.
3. **Engine configuration.** Only the default configuration was exercised: `exact_min_rank=False` (FIFO
   path), `max_dirty_edges=None` and `buffer_pending=False`. The heap path and the dense-fallback circuit
   breaker were **not exercised**.
4. **Generator.**
   - Nodes are never deleted, because the engine has no event-deletion API. "Arbitrary deletions" means
     edge withdrawals and removals plus `as_of` rewinds.
   - In T4/T5, random D_V additions merge most nodes into one giant SCC (max 255 of 256). Large-graph
     checkpoints are therefore dominated by "everything on one positive cycle, so F". That is semantically
     easy for WFS, although it is exactly where the raw-telemetry diffs concentrate.
   - SCC sizes 9-64 are well covered (T2/T3). Mid-size SCCs at 128/256 nodes are thin.
5. **`as_of` checkpoints** (104,901) compare two from-scratch replays. They exercise no incremental
   deletion maintenance. This is why the 10^6 gate was evaluated on mutation checkpoints only.
6. **Mechanism probe.** It is post-hoc, covers a sample (650 sequences), and is a classification, not a
   proof. Its "not re-solved this batch" bucket does not separately confirm *which* shield pruned each node.
7. **Not adversarial.** The run is randomized with fixed seeds. There is no search for graphs that maximise
   divergence, so 0 semantic mismatches is evidence at this coverage, not a proof.

## Departures from the preregistration

- **Compute.** 16 local cores instead of the queue item's 60 Spark cores. This was declared in the
  preregistration and needed no deadline truncation.
- **Added after the run (post-hoc):** `scripts/w22_11_mechanism_probe.py` and `w22_11_mechanism_probe.json`.
  They do not affect the verdict.
- No other departures. The tiers, seeds, comparison classes, kill rule and verdict mapping were run as
  frozen. The module `w22_11_wfs_deletion_stress.py` is unchanged since the preregistration commit.

## Brutal Honesty (rulebook v3.3, L1-L13)

| # | Category | Disclosure |
|---|---|---|
| L1 | Stubs | None. Every comparison path runs; there are no placeholder bodies or TODOs. |
| L2 | Escape conditionals | `w22_11_wfs_deletion_stress.check_checkpoint` skips C1-vs-fresh-raw for a node absent from fresh (`if fn is None: continue`). This cannot hide a mismatch, because a node-set difference is already counted as semantic, and that count is 0. |
| L3 | Mocks in production paths | None. The real engine, canonicalizer and oracle are used. The probe wraps an instance method only to record dirty sets, and returns the original result unchanged. |
| L4 | Partial-as-complete | "Canonical witness frozen" is scoped to the C1 export. Raw engine telemetry remains non-canonical, and the verdict section says so. The canonical-equality result is partly by construction; see "How strong the canonical result is". |
| L5/L8/L12 | Untested production paths | Not exercised: `exact_min_rank=True`, the `max_dirty_edges` dense fallback, `buffer_pending=True`, the Lean CLI arm, and an independent `eff_auth` check (limits 1-3). |
| L6/L7 | Fake data, inflated metrics | None. `as_of` checkpoints are reported separately and excluded from the 10^6 gate. |
| L9 | Hidden failures | None. The raw diffs (3,077,470) are reported in the verdict block. |
| L10 | Speculation | The probe's mechanism split is post-hoc, labelled as such, and a classification rather than a proof. |
| L11 | Broad-catch swallowing | None. The only `try/except` is `except StopIteration` around the chunk iterator in the runner (`submit_next`), which is iterator exhaustion. |
| L13 | Soft prose claimed as mechanical | The verdict, gates and anchor checks are computed in code (`verdict()`, `anchor_checks()` in the runner). This prose restates `w22_11_results.json`. |

**Process disclosures:**
- A timing-only pilot was run before the preregistration, on disjoint seeds, and was disclosed in it.
- The unit test's `--scale 0.001` smoke run touched the first preregistered sequences after the
  preregistration was committed. Nothing was tuned from it.

**SMOKE tier:** ceiling for this item. The real engine ran end-to-end on generated fixtures at the target
scale. Tier B has not been run.

**BHS_SELF_DRAFT: 92.** This is not 100 because of three gaps:
- independent `eff_auth` is not checked (limit 1), and the non-default engine paths are untested (limit 3);
- canonical equality is partly by construction;
- there is no fresh adversarial Tier B review yet.

Per the rulebook, this does not ship at under 100. A Tier B reviewer should attack limits 1 and 3 first.
