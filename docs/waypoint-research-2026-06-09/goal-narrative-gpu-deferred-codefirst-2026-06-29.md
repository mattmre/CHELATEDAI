# Goal Narrative — Build the Lattice Machinery Code-First (GPU-deferred session)

**Date:** 2026-06-29 · **Mode:** CODE-FIRST, GPU-DEFERRED · **Local-only strategy doc.**
Supersedes the campaign-led pace while the 3090 is reserved for the concurrent Grok
documentation waves. Operator constraint: **do not use the 3090** except optional
micro-smoke validation; build and unit-test everything else now; resume campaigns the
moment the GPU is free.

---

## 0. The mode shift (why now)

The 3090 is busy (Grok doc-gen, in waves, owning the main working tree on
`feat/brain-file-map-b0-b1`). So we flip the cadence: instead of campaign → code, this
session is **code → (deferred) campaign**. We build the entire remaining lattice machinery —
above all the living post-bank that is the thesis's load-bearing test — fully under
**stub-backed unit tests**, merge each slice at BHS 100, and **defer every heavy GPU
campaign** until the operator frees the card.

**Honest framing, stated up front:** the *code* can all land now; the *thesis verdict* —
does a living annealed bank of correction posts beat a frozen static bank AND a one-shot
router? — is a runtime-evidence claim and is **gated on the GPU campaign**. This session
builds the apparatus to completion so the verdict is *one command away* when the 3090 is
free. We will not claim the bank "works" or "beats" anything until that campaign runs.

The payoff: convert GPU-idle time into "everything-but-the-numbers done."

## 1. State going in

- **Merged on main:** H1 (#276, eval-split fix), B1 (#277, rung 12 Evidence DAG schema).
- **H2 PAUSED (numbers deferred, code ready):** the H1 + telemetry harness fixes are
  written and verified; the regenerated swap numbers wait for the GPU. CD-H1-01 stays open.
  Partial valid cells live in worktree `.claude/worktrees/h2-rerun` for resume.
- **Concurrent:** Grok doc-gen owns the main working tree. **Every slice this session runs
  in its own isolated git worktree** so the two never collide.

## 2. The code-first slices (this session), in dependency order

Each: fresh implementer → fresh adversarial Tier-B (instructed to disprove) → PR with full
BHS body → only BHS 100 merges → checkpoint review-notes + goal narrative. In an isolated
worktree. **Stub backends (`TinyEmbeddingBackend` / `StubSwapBackend`) are the evidence;**
the GPU campaign is explicitly disclosed as deferred (CARRY_FORWARD).

**S1 — H5a: the steering-post bank module (the novel core).**
A standalone `steering_post_bank.py`: a bank of correction *posts* (each a bounded
near-identity adapter) keyed by cluster centroid; register / select / route (reuse
`adapter_router.AdapterRouter` + `create_adapter`/`BoundedAdapter`); and a **prune-and-
re-anneal lifecycle** — posts pruned when a drift/fitness signal fires, re-annealed from a
fixed anchor snapshot — with store-mutation logging. Stub-tested: posts register, routes
select by centroid, the lifecycle prunes + re-anneals deterministically, mutations are
logged. **No GPU.**

**S2 — H5b: three harness conditions + head-to-head wiring (no campaign yet).**
Wire into `run_drift_recovery_experiment.py`, mirroring `_supervised_anchor_cycle`, three
conditions all scored on the same held-out eval subset:
- `C5` living annealed-route post-bank,
- `C5s` frozen SVF-style static bank (register once, no re-anneal),
- `C5r` one-shot LD-MoLE-style router (route once, no lifecycle).
Plus the campaign-driver entry. Stub-tested: all three fire, mutate the store, and the
prune/re-anneal lifecycle is exercised for `C5` only; deterministic; same eval subset.
**The campaign that produces the actual `C5 vs C5s vs C5r` numbers is DEFERRED to the GPU.**

**S3 — H3: teacher-supervised C3b condition.**
New condition supervising the adapter with a teacher (new-encoder) re-embedding of anchor
docs as the distillation target. Stub-tested (fires; learns a non-floor correction on the
stub geometry). Campaign deferred.

**S4 — H6: annealing controller (rung 11).**
A real temperature-schedule module driving the bank's explore / stabilize / prune phases —
genuinely new, NOT a wrapper over the existing scalar score-scaler `set_temperature`.
Stub-tested; wires into S1's bank.

**S5 — A1: rung 10 SHIM substrate DoD + rollback test.** (If the session lasts.)

## 3. What this session explicitly does NOT do (gated on the 3090)

- H2 re-run completion + commit (the regenerated numbers).
- The H5 / H3 head-to-head **campaigns** — the recovery numbers and the thesis verdict.
- Any claim that the living bank beats the baselines (runtime-evidence claim → waits).

Each PR's Brutal-Honesty section says exactly this: "code + stub-tested; the real-data
campaign is deferred to the 3090 (CARRY_FORWARD: run C5/C5s/C5r + C3b campaigns)."

## 4. Operating discipline

- **Worktree isolation for every slice** (Grok owns the main tree; never commit/switch it).
- **No 3090** except *optional* micro-smoke: a single tiny cell (~20 docs / ~5 queries /
  1 cycle) to confirm a new condition runs end-to-end on real models without erroring —
  seconds, only if the GPU is momentarily idle, never a campaign. If unsure, skip it; the
  stub tests are the gate.
- ≤4 concurrent agents. BHS 100 per slice. Frozen base weights. Every eventual number from
  a real run.
- Never commit/push `docs/waypoint-research-2026-06-09/`.

## 5. One-line version

> The 3090 is on doc duty, so we build the whole lattice apparatus — the living post-bank
> (the thesis test), its two baselines, the teacher variant, the annealing controller —
> fully stub-tested and merged at 100, deferring only the GPU numbers. When the card frees,
> we fire the campaigns and get the verdict in one step. We do not claim the bank works
> until then.
