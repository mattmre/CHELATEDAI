# H4 — Resolving the "one-shot" question (characterization)

**Decision: characterize (the loop is one-shot *by design*), do NOT make cycles
compound.** Trajectory evidence and mechanism below.

## The evidence

Every per-cycle NDCG trajectory in the encoder-upgrade arena is a **flat step
function**: the value is set at cycle 1 and is byte-identical for all 12 cycles, for
every condition (C0/C2 flat-low, C2O flat-recovered, C3a/C4a flat-partial). This was
verified directly from the run artifacts (the per-cycle `recovery.trajectory[].ndcg`
arrays are constant) and is what the paper's §5.4 "one-shot" disclosure refers to.

## The mechanism (why it is one-shot)

`_supervised_anchor_cycle` (run_drift_recovery_experiment.py):

1. snapshots the original (pre-correction) cached doc vectors **once**, before the
   cycle loop (`original_doc_points`);
2. at each cycle, re-seeds (`torch.manual_seed(seed)`) and **re-creates the adapter
   from scratch**, then trains it on the **same fixed anchor pairs**, then applies it
   to the **same fixed snapshot**.

So each cycle is a pure deterministic function of `(snapshot, anchor_pairs, seed)` —
inputs that do not change across cycles. Every cycle therefore reproduces the identical
adapter, hence the identical correction, hence the flat trajectory. This is not a bug:
it is **intentional idempotency**, asserted by the existing regression test
`test_c3a_multi_cycle_is_idempotent` (cycles=3 produce identical trajectories).

## Why NOT make cycles compound

"Compounding" (feeding cycle N−1's corrected store into cycle N) was considered and
rejected:

- It would **break the tested idempotency guarantee** that the supervised correction is
  a reproducible function of the pre-drift snapshot.
- It re-introduces the exact failure mode the snapshot was added to avoid: applying the
  adapter to an already-mutated store across cycles **compounds** the correction and
  drifts away from the supervised target (the original code comment at the snapshot call
  flags this).
- There is no evidence it would help: the supervised target is fixed (the drifted anchor
  queries); re-deriving the same optimum repeatedly cannot improve it. Genuine
  iterative gain would require a *changing* target (e.g. progressive re-anchoring), which
  is a different experiment (belongs with the post-bank / annealed-route work, H5), not a
  reinterpretation of this loop.

## Conclusion (for the paper)

Frame the encoder-upgrade result honestly as a **one-shot supervised realignment**, not an
iterating closed loop: the detector triggers, one bounded correction is computed from the
held-out pre-drift anchors and written to the store, and subsequent cycles re-apply the
identical correction without further gain. The "12-cycle" apparatus exercises the
detection/idempotency machinery but adds no recovery beyond cycle 1 in this arena. This
matches §5.4 and §8 of the draft. H4 is resolved by characterization; no code change.
