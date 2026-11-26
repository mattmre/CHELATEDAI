# W22-11 Preregistration: Large-Graph Incremental WFS Deletion Stress Test and Canonical Witness

Status: **frozen before any W22-11 result was computed**. This file is committed by itself, before the
results JSON and report. The module that will be run, `w22_11_wfs_deletion_stress.py`, is committed with it
so that the exact comparison code is pinned too. Any later change to either file shows up in `git log`.

- Queue item: `W22-11` in `docs/research/wave22_queue_additions_2026-11-25.json`. The item is named "Large-Graph
  Incremental WFS Deletion & Canonical Witness Stress Test (10^6 Checkpoints)". It depends on `W22-05`.
- Spine commit: `e5e0b037`. Branch: `pv7/w23c-w22-11-wfs-deletion-stress`.
- Compute: CPU only, on a 16-core local machine (the queue item assumed 60 Spark cores). The wall budget for
  the whole run is 45 minutes or less. No LLM calls, no models, no Lean. Lean mechanization is explicitly
  out of scope.

## Pinned sources (sha256 at e5e0b037)

| File | sha256 |
|---|---|
| `ek5_v3_stratified.py` (incremental engine and fresh-rebuild engine) | `b54dadb51e1f54576f39e22808fa9c7cd9209fe3de86bf62139eded10b13dfd0` |
| `formal/run_conformance.py` (model, baseline generator, snapshot, fresh rebuild, oracle, comparator) | `49c56c3344d635e026284f9f44eb7e62d19a95339924ea6a3c84bd37d2ce271e` |
| `chelated_pv7_spine/ek5_reason_canonicalizer.py` (canonical witness candidate C1) | `b8440b62b8c20bf28d35a8c961ab671d3ea8ce90c45fd916c53e8b5a3d8dc432` |

The results JSON re-hashes these three files. A changed hash invalidates the run.

## What I already knew before freezing (not blind)

- **The Wave-14A conformance report** (`formal/conformance_report.json`) covers 1000 sequences x 6 steps =
  7,722 checkpoints at 11 nodes or fewer.
  - Semantic divergences: 0.
  - inc_vs_fresh telemetry divergences: 153 = 104 `pos_cyc` + 49 `reason`.
  - The report attributes these to Pitfall-B pruning at `ek5_v3_stratified.py:665`.
- **W22-05** (sibling worktree, `w22_05_results.json` key `ek5_conformance_inc_vs_fresh_python`) re-ran that
  generator in Python with seed 20261124 and per-sequence seed `seed*100003+i`:
  - 6 steps: 7,722 checkpoints, 153 diffs (104 `pos_cyc`, 49 `reason`), semantic 0;
  - 12 steps: 13,881 checkpoints, 825 diffs (548 `pos_cyc`, 277 `reason`), semantic 0.
- **A canonicalizer already exists in the spine** (`chelated_pv7_spine/ek5_reason_canonicalizer.py`, Wave 16).
  - It claims to derive `reason`/`pos_cyc`/witness lazily from the final semantic state and edge tables.
  - It claims the result equals a fresh rebuild.
  - Its own tests cover small stores only. Its claim has never been run against the conformance generator or
    at scale. I use it as candidate C1 below and do not modify it.
- **Code reading (a mechanism hypothesis, not evidence).**
  - `_solve_stratum2_region` computes `cyc_nodes = _nontrivial_scc_nodes(dirty, deps_v, subset=dirty)`
    (line 380) and sets `pos_cyc[u] = u in cyc_nodes` for dirty nodes only (line 550). So incremental
    `pos_cyc` means "on a D_V cycle that lies wholly inside this batch's dirty region". Nodes pruned from the
    cone by the Pitfall-A/B shields keep their old flag.
  - In a fresh rebuild every node that has dependencies is seeded, so every D_V cycle lies wholly inside the
    dirty region.
- **Timing-only pilot.**
  - Before this file was written I ran a pilot on seed offsets 9001–9005. Those seeds are disjoint from every
    preregistered seed below.
  - It printed only checkpoint throughput, maximum node count and maximum SCC size. No mismatch counts were
    printed or inspected.
  - Throughput was about 577 checkpoints/s per core at 11 nodes, 274 at 32 nodes, 131 at 64 nodes, 47 at
    128 nodes and 12.5 at 256 nodes.
  - The maximum D_V SCC was 121 of 128 nodes and 241 of 256.
  - The tier sizes below are chosen from these numbers.

## Generators

- **`baseline`**: `run_conformance._gen_batch`, used **unchanged**, with `MAX_NODES = 11`.
  - Each step has random withdrawals (p=0.35, 1–2 ids) and a random triple removal (p=0.15).
  - It also makes random additions with weighted edge types. In non-strict mode it plants imported
    SUPERSEDE 2-cycles.
- **`stress`**: `w22_11_wfs_deletion_stress.gen_batch_stress`. On every step after step 0:
  - (a) random withdrawals with p=0.6, 1–3 ids;
  - (b) a **targeted cycle-break** with p=0.5: an active D_V edge (`COPY_OF`, `DERIVES_CURRENT`,
    `DERIVES_FROM`, `AMENDS` or `DERIVES_SNAPSHOT`) whose two endpoints share a non-trivial D_V SCC is
    deleted, by withdrawal or triple removal at 50/50;
  - (c) a random triple removal with p=0.3;
  - (d) a **planted D_V SCC** with p=0.3: a ring of k nodes with random D_V edge types. Rings of k>=4 get
    1..k/4 chords with p=0.3, and with p=0.5 the ring also gets a REFUTE from an outside node;
  - (e) random additions under the same legality rules as `_gen_batch` (strict mode never closes a
    SUPERSEDE or lineage cycle; non-strict mode plants imported SUPERSEDE cycles; 10% of additions
    re-activate a withdrawn attestation id).
- Random additions also merge planted rings into larger SCCs.
- Events (nodes) are never deleted, because the engine has no event-deletion API. "Arbitrary deletions" here
  means edge-attestation withdrawals and triple removals. Bitemporal `as_of` rewinds hide nodes.

## Tiers (seed of sequence i = `(20261124 + seed_offset) * 100003 + i`)

| Tier | Generator | max nodes | steps | sequences | seed_offset | first-step nodes | new nodes/step | adds first / per step | planted SCC k |
|---|---|---|---|---|---|---|---|---|---|
| A0 anchor | baseline | 11 | 6 | 1,000 | 0 | (3,5) | (0,2) | (2,6)/(0,4) | — |
| A1 anchor | baseline | 11 | 12 | 1,000 | 0 | (3,5) | (0,2) | (2,6)/(0,4) | — |
| T1 long | baseline | 11 | 24 | 30,000 | 1 | (3,5) | (0,2) | (2,6)/(0,4) | — |
| T2 stress-32 | stress | 32 | 24 | 12,000 | 2 | (8,12) | (0,3) | (2,6)/(0,4) | 3–16 |
| T3 stress-64 | stress | 64 | 24 | 6,000 | 3 | (16,24) | (1,4) | (8,16)/(2,8) | 4–32 |
| T4 stress-128 | stress | 128 | 32 | 2,500 | 4 | (32,48) | (2,6) | (16,32)/(4,12) | 8–64 |
| T5 stress-256 | stress | 256 | 40 | 400 | 5 | (64,96) | (2,8) | (32,64)/(6,16) | 16–128 |

- A0 and A1 use exactly the W22-05 seeds and generator, so they are **reproduction anchors**.
- Planned checkpoints:
  - A0 7,722 (known), A1 13,881 (known);
  - T1 780,000, T2 312,000, T3 156,000, T4 85,000, T5 16,800 (two `as_of` checkpoints per sequence where
    the grid permits, else one);
  - total about 1.37 x 10^6, of which about 1.27 x 10^6 are mutation checkpoints.

## Checkpoint definition

A checkpoint is one materialized graph state at which every comparison below is evaluated, over every node.
There are two kinds:
- **`mutation`**: the live incremental store right after an atomic `apply_mutation` batch. These are the
  only checkpoints that exercise incremental deletion maintenance.
- **`as_of`**: a bitemporal rewind via `store.as_of(vc, tc)`. This is itself a from-scratch replay, so the
  comparison is fresh-vs-fresh. These are counted (matching the Wave-14A definition) but **reported
  separately**.

The 10^6 target is evaluated on **mutation checkpoints only**, which is the stricter reading. The total
including `as_of` is reported next to it.

At each checkpoint the code computes:
- `snapshot`: the Stratum-0 snapshot from the independent model;
- `inc`: the incremental store;
- `fresh`: a rebuild from the model in a shuffled order, using the generation RNG exactly as
  `run_conformance` does;
- `fresh2`: a second rebuild with an independent RNG, so a different insertion order;
- `oracle`: the textbook alternating-fixpoint WFS, computed from the raw snapshot only;
- `C(x) = export_canonical_ek5_audit_trace(x).to_dict()` for x in {inc, fresh, fresh2}.

## Comparison classes

1. **Semantic.** Every item here is a kill-rule mismatch:
   - inc vs fresh on `eff_auth`, `val`, `cur`, `eligibility` and the node set;
   - inc `val` vs `oracle`;
   - inc Stratum-0 materialization vs `snapshot`;
   - every entry of `C(x).semantic_mismatches` for x in {inc, fresh, fresh2}, where the canonicalizer's
     independent `cur`/`eligibility` re-derivation disagrees with the store.
2. **Raw witness (characterized, not the canonicality gate).**
   - inc vs fresh on the engine telemetry: `reason`, `pos_cyc`, `blocked`, `quarantined`.
   - Each node diff is tagged with: tier, checkpoint kind, field, direction (`inc_missed_cycle` = inc False /
     fresh True, `inc_stale_cycle` = inc True / fresh False, or the reason pair), the node's D_V SCC size
     bucket in the snapshot, its `val`, whether the batch deleted a D_V edge, and whether fresh-vs-fresh2
     also differs at that node.
3. **Canonical witness (the canonicality gate).**
   - `C(inc)` vs `C(fresh)` over the whole trace: per node `val/cur/eligibility/reason/cause/pos_cyc/witness`
     (the witness is the lexicographically minimal contributing-id tuple), plus `positive_cycles`,
     `supersede_cycles` and `lineage_cycles`.
   - Also `C(fresh)` vs `C(fresh2)`.
   - Contract checks, reported and gating:
     - `C(inc)` `reason`/`pos_cyc` equals the fresh store's raw `reason`/`pos_cyc` (the canonicalizer's
       documented claim);
     - `C(inc)` `pos_cyc` equals D_V non-trivial-SCC membership computed from the raw snapshot.
4. **Order dependence.** Fresh-vs-fresh2 raw telemetry diffs, reported. A non-zero count would mean the
   engine's raw telemetry depends on insertion order even with no incremental history.

Before reading the predictions below: canonical inc-vs-fresh equality is **close to implied by
construction**. C1 reads only `val` plus the store's edge tables, and both are already required to be equal
by class 1 and the Stratum-0 gate. So the non-trivial content of class 3 is three things:
- (i) C1 agrees with the engine's fresh-path telemetry, which is independently coded;
- (ii) C1 agrees with snapshot-derived SCC membership;
- (iii) C1's cur/eligibility cross-check holds.
The report must state this.

## Comparator liveness (must be 100%)

Every 50th sequence of every tier injects deterministic faults at each of its mutation checkpoints:
- six canonical-trace corruption kinds (reason, pos_cyc, witness, cause, positive_cycles,
  supersede_cycles);
- five semantic-state corruption kinds (`run_conformance.inject_fault` kinds 0, 1, 2, 3, 8).

Each corrupted copy is compared against the clean fresh object. Liveness requires detected == injected for
both classes. Liveness below 100% invalidates the run.

## Predictions (stated in advance; the report says whether each held)

- P1: semantic mismatches = 0 in every tier.
- P2: A0 reproduces raw 153 (104 `pos_cyc` + 49 `reason`), and A1 reproduces 825 (548 + 277). These are
  reproduction gates. A miss means the harness does not replicate W22-05, and the run is flagged.
- P3: raw witness diffs are non-zero in T1–T5. They occur only on `val = F` nodes, and only in `mutation`
  checkpoints.
- P4: raw fresh-vs-fresh2 diffs = 0. If so, the raw diffs come from incremental history, not from insertion
  order.
- P5: canonical (C1) inc-vs-fresh diffs = 0, fresh-vs-fresh2 canonical diffs = 0, and both contract checks
  = 0.

## Kill rule and verdict mapping (exact)

The headline is set by the first matching row:

| Condition | Verdict |
|---|---|
| Any liveness miss, or a pinned-source hash mismatch | `INVALID_RUN` |
| semantic total > 0 | `BLOCKING_BUG` (semantic mismatch; Lean stays unfunded) |
| canonical C1 diff total > 0 (inc-vs-fresh, fresh-vs-fresh2, or either contract check) | `CANONICALITY_NOT_ESTABLISHED` |
| mutation checkpoints >= 1,000,000, semantic 0, canonical 0 | `CLEAN_1M_CANONICAL_FROZEN`. The C1 invariant is frozen; this meets the queue item's precondition for funding Lean, but does not fund it. |
| mutation checkpoints < 1,000,000, semantic 0, canonical 0 | `CLEAN_PARTIAL_<n>`. Not enough for the Lean precondition. |

Anchor-reproduction misses (P2) do not change the verdict, but they are reported as a harness defect.

**Raw witness diffs never pass or fail the run.** They are expected under the known pruning design. They are
characterized and reported, and they establish that the **raw** engine telemetry is **not** canonical. They
are never re-labelled as clean, and the comparison is never weakened to hide them.

### If C1 fails

I may design a C2 normal form, but only after seeing results. C2 would be labelled post-hoc, and it counts
only if it is re-run clean on **new seeds** (seed_offset + 100 for every tier), at the same scale and with
the same gates. Every other part of the run stays as frozen here.

## Frozen invariant (if the run reaches the clean verdict)

The frozen canonical witness invariant `CW-1` is defined as follows:

`CW-1 := sha256(json.dumps(export_canonical_ek5_audit_trace(S).to_dict(), sort_keys=True,
separators=(",",":")))` must be equal for the incremental store S_inc and every from-scratch rebuild of the
same visible log, under any insertion order. The canonicalizer is pinned at the sha256 above.

The results JSON records this definition and the pinned hash.

## Runtime and stopping

- 16 worker processes run chunks of 50 sequences (25 for T4, 10 for T5).
- Chunks from all tiers are submitted interleaved round-robin, so an early stop truncates every tier
  evenly.
- At a hard wall deadline of 40 minutes, no new chunks are started. The count actually reached is reported
  exactly, and the verdict uses that count.
- Outputs: `artifacts/pv7_w23c_w22_11/w22_11_results.json` and `report.md`. Per-chunk raw JSON goes to
  `artifacts/pv7_w23c_w22_11/scratch/`, which is gitignored.
