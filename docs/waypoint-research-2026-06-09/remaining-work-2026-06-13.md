# Remaining Work & Roadmap State — 2026-06-13

Local-only (git-excluded). Synthesizes a 7-agent analysis + 2 adversarial verifications
of the repo state after the June drift-recovery sprint (PRs #258–#266 all merged).
Every claim below is backed by file:line evidence in the workflow transcript; the
load-bearing diagnosis was adversarially verified (could not be disproved).

## The headline (read this first)

The drift-recovery sprint executed cleanly and **honestly produced a negative result** —
but the deeper finding from verification is that **the experiment cannot test its own
thesis as built**. The closed-loop corrector (C3) never mutated the vector store:
`correction_applied = 0/36` on the main matrix and `0/12` across the knob sweep
(`run_drift_recovery_experiment.py:220` defines applied = store-checksum-changed; it never
changed). This is "the actuator never engaged," not "the actuator engaged and
underperformed." Three stacked, confirmed causes:

1. **Deep design gap (load-bearing, verified):** the sedimentation target is built
   entirely from *post-drift* geometry (`compute_homeostatic_target` on the already-drifted
   vector, `sedimentation_trainer.py:14-66`; `antigravity_engine.py:1806-1811`). Nothing
   encodes the *pre-drift* geometry — no teacher, no held-out relevance pairs, no clean
   snapshot. So the correct optimum for the unbounded adapter is ~zero correction (C4's
   measured norm 0.000189 confirms it). C3's 0.010000 is the BoundedAdapter floor
   *manufacturing* a delta out of a near-zero correction (`chelation_adapter.py:402-411`),
   not a recovered correction. **No wiring fix repairs this — it needs a supervision
   signal.** This is the same class as the repo's known "same-model distillation is a
   no-op" finding (Session 29).
2. **Shallow wiring bug (fixable):** the harness builds the engine with
   `use_quantization=False` (`run_drift_recovery_experiment.py:158-162`), so the CHELATE
   branch never runs (`antigravity_engine.py:2687-2693`), `chelation_log` stays empty,
   `run_sedimentation_cycle` finds zero candidates and early-returns
   (`antigravity_engine.py:1746-1749`). Same `use_quantization` bug class as Session 29.
3. **Wrong trigger signal (fixable):** the controller's drift scalar reads only
   `global_variance` — mean per-dim variance of the *current* top-K neighbor cluster
   (`antigravity_engine.py:2671-2675`) — which the injected drift does not move.
   Correction to an earlier note: detection fires **36/36** in the default campaign
   (trigger_threshold 0.0); the `0/12` is only the knob sweep at threshold 0.05/0.15. The
   underlying point (drift-insensitive signal) stands; the symptom is "fires always or
   never depending on threshold," not "never fires."

**And the only condition that "recovers" (C2) is an oracle, not a baseline** — it
re-embeds the unchanged raw text with the frozen original model, inverting synthetic
drift by construction (`max baseline−final diff = 0`). The diagnostics already relabeled
it correctly. Net: zero honest positive data points across 78 artifacts.

## Publish decision (adversarially verified: do NOT publish to arXiv today)

A single-dataset, single-model, never-engaged-actuator null plus a tautological upper
bound is an under-powered null that risks a layman first-author's credibility. arXiv will
*accept* it — that's the trap, not the safety net. Two honest paths:

- **Path A (recommended):** one more sprint (supervised-correction variant + model-swap
  arena), then publish as a **methodology/harness paper** with an honest mixed result.
  §§1–4, §6, and the C2-oracle insight can be drafted **now in parallel** — not blocked on
  runs. The C2-oracle catch ("the obvious baseline cheats by construction; here's the
  inversion identity") is genuinely citable methodology.
- **Path B (bank the timestamp):** release the harness + negative diagnosis as a
  versioned Zenodo DOI now (the plan's own fallback), upgrade to the arXiv preprint after
  the sprint. A versioned artifact carries no "finished result" promise, so it costs no
  credibility.

Either way, do NOT dress `correction_applied = 0/36` as a "rigorous negative" — the
mechanism never operated, so it is a *non-engagement observation plus a conjecture about
cause*, not a demonstrated mechanism. The diagnosis becomes real only once a supervised
variant makes the actuator fire and we observe what it does.

## What the sprint actually advanced (Phase II map)

| Step | Title | Status |
|---|---|---|
| 1–9 | Phase I + Model-Scope shadow (step 9) | DONE |
| 10 | SHIM substrate DoD | NOT STARTED (gated, stays parked) |
| 11 | Annealing controller | IN-PROGRESS (minimal): wired to sedimentation + `set_temperature` only, NOT `online_updater`/ES. Exit criterion (high-T increases perturbation on an observable-correction path) unmet. |
| 12 | Evidence DAG schema | NOT STARTED |
| 13 | Disintegration loop | NOT STARTED (never touched despite "11+13+14" framing) |
| 14 | Concept-drift experiment | IN-PROGRESS / blocked: apparatus complete, exit criterion (documented recovery in CHANGELOG) **unmet** — result is a negative. |
| 15 | GNN prototype | NOT STARTED (correctly gated on 12+13+14) |
| 16 | Quant-aware shim routing | NOT STARTED |
| 17 | Disk pool slice | NOT STARTED |

**Genuinely next**, in order: fix the mechanism (P0 store-write/quantization, P1
drift-sensitive trigger, P2 supervision target) → re-run step 14 at calibrated severity →
only then steps 12/13/15 (DAG, disintegration, GNN), which all assume a loop that changes
retrieval outcomes.

### Roadmap doc items now stale (need a docs PR)
- `VISION_LIQUIFIED_LATTICE.md:128` lists "adapter-only self-healing with measurable
  recovery" as in-scope/achievable — the sprint's confirmed result is *no* recovery;
  downgrade to "open question, negative first result."
- `VISION_LIQUIFIED_LATTICE.md:152` "drift recovery" success metric needs a caveat that
  the inaugural attempt failed.
- `ROADMAP_EXECUTION.md` step 13/14 gates should require a *non-zero applied correction
  that changes retrieval*, not mere controller existence.
- The June plan's C2 = "maintenance-only (Ada-IVF analog)" is now known to be an oracle;
  any reuse of that definition is stale.

## Debt & hygiene punch list

- **Block-flag latent violation (important):** `next-session.md` declares CLEAR, but
  CD-247-01 (stale BHS placeholder comments at `aep_orchestrator.py:678,966`) and
  CD-247-02 (6 broad `except Exception` swallow sites) have been OPEN since ~2026-05-16 —
  ~28 days, multiple cycles. By the rulebook TTL math the flag should be BLOCKED; the 8
  drift PRs merged under an incorrectly-CLEAR flag. Both are <30-min fixes. **Close both
  in one hygiene PR** to make the flag honest (strictly better than flipping to BLOCKED).
- **Prune 9 merged remote branches** (the `feat/drift-*` + `feat/annealing-controller-pr3`
  branches for #258–#266), then `git fetch --prune`.
- **Remove stale worktree** `serene-aryabhata-3ed9a9` (holds dead branch, work merged
  via #260).
- **Close PR #257** (live-progress umbrella) — superseded by direct-to-main commits
  `bf23a47`/`f402175`; it's 11-behind with BHS 82 + deferred Tier B, can't merge anyway.
  Salvage any unique content first, then close + prune.
- **Keep PR #256 parked** (SHIM SIP probe) — precondition (Phase I step 8 E2E loop) still
  unmet; rebase before any revival; operator should confirm its `OPERATOR_OVERRIDE.md`
  "ACTIVE" delegation is still intended.
- **Clarity nit (not a contradiction):** `drift-recovery-results-2026-06.md:5` says
  `Device: cuda` while the honesty note mentions `HF_HUB_OFFLINE=1`. These are orthogonal
  (offline = no network download; cuda = compute device) — a cached model runs on GPU
  offline. Add one clarifying clause so a reviewer doesn't misread it as a CPU/GPU
  contradiction.

## Next experiment design (the path to a real test)

Promoted from the plan's §6 stretch to primary, because v1 proved the synthetic-drift
arena cannot test the thesis. Full PR breakdown in [goal-2026-06-14.md](goal-2026-06-14.md).

- **Slice A — Model-swap drift + supervised C3 (highest priority).** Drift = re-embed a
  fraction of the corpus with `all-mpnet-base-v2` through a frozen seeded
  `DimensionProjection(768→384)`; queries stay on MiniLM. C2's frozen-MiniLM re-embed can
  no longer invert this → **the oracle dies and C2 becomes an honest maintenance
  baseline** (verify `C2_final − baseline ≠ 0` as a *gating test*, not an assumption).
  C3a gets **anchor-pair InfoNCE supervision** from held-out pre-drift (query, relevant-doc)
  pairs disjoint from the eval set — a real target without re-embedding. C3b (secondary)
  uses the MiniLM-as-teacher path (shares info with C2; label it clearly).
- **Slice B — Supervised variant on the *existing* synthetic drift.** Give C3 the same
  anchor-InfoNCE target on calibrated noise (σ=0.035). Separates "can a *supervised*
  bounded adapter recover known drift?" (should be answerable) from "can the *unsupervised*
  loop self-heal?" (v1 answered: no). Turns the v1 negative into a scoped positive + named
  open problem.

Pre-stated honest risks (reporting them later is not failure): the model-swap drift may be
too severe for a bounded adapter (C3a ≈ C0 → report the severity threshold); the detector
may still not fire (→ contribution narrows to "recovers when manually triggered; auto-detection
of cross-model drift is open"); C3b/C3s may merely approach C2 (→ "cheaper-not-better
maintenance," still a legitimate systems result). All publishable; none require tune-to-win.
