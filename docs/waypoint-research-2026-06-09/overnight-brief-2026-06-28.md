# Overnight Brief — Harden Step 14 & Open Phase II (feed this to kick off the run)

Paste the block below to start the autonomous overnight session. It is self-contained; on
kickoff, re-read `goal-narrative-phase-II-completion-2026-06-28.md` and
`review-notes-cleanup-pass.md` first.

---

You are continuing ChelatedAI's **Liquified Lattice — Phase II** execution autonomously, overnight. Use `/workflows` to orchestrate and oversee. Work through the slices below in order; at each gate, checkpoint and continue to the next *independent* slice if blocked. Don't stop to ask unless a slice needs an operator decision you cannot resolve from the docs.

**Read first:** `docs/waypoint-research-2026-06-09/goal-narrative-phase-II-completion-2026-06-28.md` (the plan) and `docs/waypoint-research-2026-06-09/review-notes-cleanup-pass.md` (running log). Your north star: *the loop moved its first vector; widen that one-shot crack into a living, annealed bank of correction posts, then build the lattice on it.*

**Non-negotiable guardrails:**
- **Never commit or push `docs/waypoint-research-2026-06-09/`** — it is git-excluded strategy (paper + plans). Paper/strategy edits stay local; only code/harness/tests go in public PRs.
- **Every number comes from a real run.** No fabricated metrics, ever (BHS Rule 3). If a run didn't happen, say so.
- **GPU is shared overnight.** Run GPU campaigns *sequentially*, modest batches, offline (`HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1`); `nvidia-smi` before each; abort if memory is tight. Other work shares the machine — do not OOM it.
- **Max 4 concurrent agents** to avoid throttling.
- **Frozen base weights.** Corrections target adapters/masks/routes/posts only.
- **BHS per slice:** fresh implementer → fresh adversarial Tier-B agent instructed to *disprove* completion → open PR with the full Brutal-Honesty body (`BHS_SELF_DRAFT`, `BHS_TIER_B`, `BHS_OFFICIAL`, etc.). Only `BHS_OFFICIAL = 100` merges. No dirty branches accumulated (Session Rule 2).
- **The gate is retrieval, not the actuator firing.** "Correction applied" is not the bar; "retrieval changed, measurably, reproducibly, and better than the static baseline" is.

**Work order — Part A: harden Step 14 (complete the current phase).**

1. **H1 — Harness eval-split fix (public PR).** Diagnose and fix the NFCorpus C3a baseline divergence in `run_drift_recovery_experiment.py` (C3a/bounded is the only condition whose per-seed baseline differs; C4a/unbounded matches C0 — points at the bounded-adapter floor being live in the retrieval path at baseline-measurement time). Make all five conditions share one per-seed baseline. Add a regression test asserting identical baseline across conditions on a tiny NFCorpus slice. Confirm with a small offline GPU re-run.
2. **H2 — Paper invariant/honesty fixes (local waypoint: paper-draft/main.md + main.tex).** "~25× not 26×"; state the 2.25% NFCorpus oracle-gap explicitly; disclose the flat/one-shot trajectory; clarify NDCG is on the 60-query eval subset; state the inverted budget effect plainly. Use H1's re-run numbers if they shift.
3. **H3 — Teacher-supervised C3b (public PR).** Implement teacher re-embedding supervision of anchor docs alongside the InfoNCE pairs. Run both datasets × 3 seeds, sequential offline GPU. Report one of the three pre-registered outcomes honestly (raises ceiling / matches C3a / degrades), with the correction norm proving it learns (not bound-floor). Fold results into the paper draft locally.
4. **H4 — Resolve the one-shot question (public PR or analysis slice).** Either make cycle N consume cycle N-1's corrected state and improve, or characterize mechanistically why one-shot is the correct fixed point here. Trajectory evidence either way.
5. **H5 — The post-bank proving experiment (public PR — the load-bearing slice).** Implement a **bank of steering posts** + **annealed routes** + **prune/re-anneal lifecycle**, plus two baselines handed to us by the literature: a **frozen SVF-style static bank** and a **one-shot LD-MoLE-style router**. Head-to-head on the drift fixture, 3 seeds, offline GPU. DoD: the living bank beats both static baselines on final NDCG with the lifecycle actually exercised (store mutation logged, not asserted). This converts the headline thesis from speculation to evidence and gives rungs 13/15 a structure to operate on.
6. **H6 — Finalize the annealing controller, rung 11 (public PR).** Promote it from "gates one adapter cycle" to "owns one temperature schedule across the bank's explore/stabilize/prune phases," read by sedimentation + online-update + ES; idempotent under re-seed.

**Work order — Part B: open Phase II (begin what's unblocked; code-first so GPU isn't the bottleneck).**

7. **A1 — Rung 10 SHIM substrate DoD (public PR):** move shim seams out of env-only guards into production-wired control planes + rollback test.
8. **B1 — Rung 12 Evidence DAG schema (public PR):** typed query/cluster/**correction-actuator** node graph, JSON schema + validator, no GNN yet.
   *(Rungs 13 disintegration and 15 GNN stay gated until H5 + B1 land. Rungs 16/17 are Track D, later.)*

**Sequencing for the night (GPU-aware):** land H1+H2 first (cheap, locks honesty, mostly no GPU). Then run GPU campaigns sequentially: H3 (C3b), then H5 (post-bank head-to-head). Interleave the code-only slices (H5/H6 module code, A1, B1) while campaigns run. **If the night is short, priority order is H1 → H2 → H3 → H5** — those carry the most value.

**Checkpoint protocol:** after each merged slice, append a dated entry to `review-notes-cleanup-pass.md` (what landed, Tier-B verdict, runtime evidence) and tick the status table in `goal-narrative-phase-II-completion-2026-06-28.md`. Keep `docs/waypoint-research-2026-06-09/` local. Self-pace across the night; in the morning leave a concise status: slices merged, GPU campaigns run with real numbers, what's blocked, and the next slice to pick up.

**Priority insurance (non-blocking, do once early):** prepare — do not publish without operator confirmation — a Zenodo-ready bundle of the harness + the two proven hazard findings, so finishing the program first doesn't cost the priority date. Stage it; flag it for the operator; do not upload autonomously.
