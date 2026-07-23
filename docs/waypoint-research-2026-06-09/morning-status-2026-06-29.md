# Morning status — overnight Phase II execution (2026-06-28 → 29)

> ## 🌅 TONIGHT (2026-06-29, GPU FREED — campaigns RAN, real numbers) — READ THIS FIRST
>
> The 3090 freed; I ran the full-scale GPU campaigns and then chased the result to ground with
> operator-authorized fair-baseline experiments. **The H5 thesis is falsified, and we found out
> mechanistically WHY — a clean, publishable result. Full detail: top of `review-notes-cleanup-pass.md`.**
>
> **H5 (load-bearing) — FALSIFIED, real, 3 seeds, both datasets:** C5 living == C5s static to 15 s.f.
> (`living_bank_wins=False` both). The prune/re-anneal lifecycle is a no-op. H3 C3b also ran
> (teacher-distill helps on NFCorpus 0.127 but is oracle-adjacent; still beaten — see below). H2
> crashed on RAM (lowest value, skipped).
>
> **Then the real discovery (Rank 4 → Rank 1 → sweep → MLP, all real runs, exact arena, sanity-matched):**
> | corrector | SciFact recovery of oracle gap | NFCorpus |
> |---|---|---|
> | our entire line (C3a/C5/living bank) | ~16% | ~30% |
> | **trivial regularized linear map (ridge/Procrustes)** | **~85%** | **~80%** |
> | residual MLP (nonlinear) | 82% (no gain) | — |
> | oracle (re-embed) | 100% | 100% |
>
> **THE MECHANISTIC FINDING — our corrector fails on TWO axes vs a trivial paired-least-squares map
> (~16% vs ~85%), PRIMARILY weak supervision, secondarily the bound.** Real-number decomposition (our
> bounded/unbounded anchor-InfoNCE corrector = campaign C3a/C4a): supervision effect (unbounded:
> anchor-InfoNCE C4a 25% → paired-least-squares 85%) ≈ **+60pts, DOMINANT**; bound effect (C4a 25% →
> C3a 16%) ≈ **−9pts, secondary**. An α-shrink ablation separately shows a *tight* bound alone wrecks
> even a good map (α=0.05 → 2.8%) — so the chelation "small bounded near-identity correction" premise
> IS genuinely ill-suited to catastrophic drift, but the honest headline is "wrong supervision signal
> (relevance-InfoNCE vs paired-embedding regression) + bounded, vs the trivial baseline," not "the bound
> did it." The drift is ~85% recoverable with an unbounded regularized linear map; nonlinearity doesn't
> help (85% = post-hoc ceiling; last 15% = oracle's re-embed advantage). *(Self-corrected an earlier
> over-strong "the bound is THE cause" framing — see review-notes.)*
>
> **What survives (and is genuinely publishable):** a methodology/hazards paper, now with FOUR claims —
> (1) the query-encoder-upgrade arena; (2) the degenerate re-embed-oracle hazard; (3) the unsupervised
> identity-collapse hazard; (4) **the bound/trivial-baseline hazard** — elaborate bounded correctors
> underperform a one-line `lstsq` 4.5×; the bound→recovery curve is the spine. Novelty pass (2 workflows):
> the corrector mechanisms are all anticipated (Drift-Adapter, NoisyNet, MoE-LoRA); the hazards are not.
>
> **What's blocked:** nothing technical. The mechanism investigation is DONE and conclusive. The only
> open item is a direction call (operator deferred papers): write the methodology paper, or not. No
> further corrector variants worth running — `lstsq` settled it.
>
> **Caveats found tonight:** the campaign's 0.819/0.598 "baselines" are 1200-doc-SUBSET inflations
> (real MTEB MiniLM: SciFact 0.648, NFCorpus 0.397, both reproduced); C3b's teacher is oracle-adjacent.
> Experiments: `scratchpad/rank{4_direction,1_inarena,1_sweep,1_mlp}.py`. Zenodo bundle should be updated
> with hazard #4 before any upload.

---


> **HEADLINE (end of session): 12 PRs merged at BHS 100 — EVERY code slice in the plan is done.**
> The full H5 post-bank apparatus (rungs 11 + 14), the rung-10 SHIM substrate (A1a/A1b/A1c),
> rung-12 (B1), and H3 (teacher-supervised C3b — module #287 + harness wiring #288) are all
> CODE-COMPLETE on `main`, each adversarially reviewed to a genuine 100. The ONLY remaining work
> is GPU-gated — the H5 head-to-head VERDICT campaign, H2's re-run, the C3b/H3 campaign, H4's
> trajectory evidence — all parked for the operator's 3090; plus rungs 13/15/16/17 gated on the
> H5 verdict, and rung-10's one run-script adoption step (honestly deferred per
> `docs/rung10-shim-substrate-dod.md`). PRs: #276 #277 #279 #280 #281 #282 #283 #284 #285 #286
> #287 #288. ⚠️ C3b's teacher is the C2O ORACLE signal (disclosed) — its campaign must frame it
> as not-oracle-free.

> **🚩 PRIORITY-INSURANCE FLAG (Zenodo bundle — STAGED, NOT UPLOADED):**
> `docs/waypoint-research-2026-06-09/zenodo-staging/` is a complete pre-upload bundle banking the
> date on the two runtime-verified hazard findings (C2-oracle hazard; supervision-signal
> mechanism) — README + FILES.md manifest + `.zenodo.json`. Verified all manifest files exist on
> `main` EXCEPT `test_engine_telemetry_cuda_guard.py` (lives on the H2 branch; manifest now flags
> it to omit pre-H2). **Operator actions before any upload:** (1) set the license + author/ORCID
> in `.zenodo.json` (currently `TBD`); (2) pull final §5 numbers from the REGENERATED result docs
> after the H2 GPU re-run; (3) zip FILES.md from a clean tag. Nothing was uploaded (guardrail).

Plan: `goal-narrative-phase-II-completion-2026-06-28.md`. Sequencing target: land H1+H2,
then H3/H5 campaigns, interleaving code slices.

---

## ☀️ FINAL SESSION SUMMARY (2026-06-29, code-first GPU-deferred mode)

**Operator re-plan mid-session:** 3090 reserved for the concurrent Grok doc workstream → I
flipped to code-first, building the H5 post-bank machinery under stub tests, deferring every
GPU campaign. Governing doc: `goal-narrative-gpu-deferred-codefirst-2026-06-29.md`.

**6 PRs merged on `main`, each at BHS_OFFICIAL 100 (fresh adversarial Tier-B per slice):**
| PR | Slice | What |
|----|-------|------|
| #276 | H1 | harness eval-split fix (all conditions share per-seed baseline) |
| #277 | B1 | rung 12 typed Evidence DAG schema |
| #279 | S1 / H5a | steering-post bank + prune/re-anneal lifecycle |
| #280 | H6 / rung 11 | annealing temperature schedule (explore→stabilize) |
| #281 | S2a / H5b | post-bank-from-anchors builder (clustering + assembly) |
| #282 | S2b-runtime | build+apply glue (route → correct → write-back) |

**Post-bank plumbing is COMPLETE** — bank + schedule + builder + glue all on main. The
adversarial loop caught + fixed real bugs pre-merge in B1 (fictional-token clustering) and
S2a (non-reentrant empty-cluster re-seed).

**✅ S2b-conditions MERGED (#283)** — `C5`/`C5s`/`C5r` + the anneal/prune/re-anneal lifecycle
wired into the harness, end-to-end swap-arena tested (all three fire + mutate the store; C5's
prune/re-anneal forced + asserted). **The H5 post-bank experiment is now CODE-COMPLETE on
`main` (7 PRs):** bank #279 + schedule #280 + builder #281 + glue #282 + conditions #283.

**Remaining — all GPU-gated, parked for the operator's 3090. The DRIVERS NOW EXIST (#290), so
each is one command. Run offline + sequentially, `nvidia-smi` first, abort if tight:**

```python
# (from a clean checkout of main @ #290+, with HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1)
from run_drift_recovery_swap_campaign import (
    SwapCampaignConfig, run_swap_campaign, run_condition_head_to_head)

# 1. H5 head-to-head (the load-bearing verdict) — C0/C2O/C5/C5s/C5r, per dataset:
run_condition_head_to_head(SwapCampaignConfig(task="SciFact",
    output_dir="experiment_runs/drift-recovery/postbank-scifact",
    report_md="docs/postbank-headtohead-scifact-2026-06.md"))
run_condition_head_to_head(SwapCampaignConfig(task="NFCorpus", sample_docs=1200,
    output_dir="experiment_runs/drift-recovery/postbank-nfcorpus",
    report_md="docs/postbank-headtohead-nfcorpus-2026-06.md"))
# verdict["living_bank_wins"] (C5 > C5s AND C5 > C5r) is the H5 gate.

# 2. H3 C3b head-to-head (same driver, C0/C2O/C3a/C3b) — both datasets:
run_condition_head_to_head(SwapCampaignConfig(task="SciFact",
    output_dir="experiment_runs/drift-recovery/c3b-scifact",
    report_md="docs/c3b-headtohead-scifact-2026-06.md"),
    conditions=("C0","C2O","C3a","C3b"))
# (repeat task="NFCorpus")  — report C3b correction-norm + vs C3a, per the oracle-teacher caveat.

# 3. H2 NFCorpus re-run (existing A4 driver) — regenerates §5 numbers:
run_swap_campaign(SwapCampaignConfig(task="NFCorpus",
    output_dir="experiment_runs/drift-recovery/swap-nfcorpus",
    report_md="docs/drift-recovery-swap-nfcorpus-results-2026-06.md"))
# verify C3a baseline == C0 in the regenerated cells before committing (CD-H1-01).
```

H4's `compound_cycles` ablation is the only remaining sweep knob not yet added to the harness
(noted in the paper as a small future experiment; the one-shot characterization stands without it).

**Blocked / notes:** GPU campaigns blocked ONLY by the operator's 3090 reservation (not by
code). Subagent **Bash tool is broken session-wide** (`line 108: unexpected EOF`) — Tier-B
agents verify code by trace; implementer captures the green runs (main-session shell works).
Grok doc workstream owns the main working tree (`feat/brain-file-map-b0-b1`); all my work was
worktree-isolated, zero collisions.

**Next slice:** S2b-conditions. Then fire the campaigns the moment the 3090 frees.

---

## Slices merged / landed
- **H1 — MERGED, PR #276 (`d5fc7df`), Tier-B 100/none.** Eval-split baseline fix: all five
  conditions now share one per-seed baseline (C3a−C0 = 0.0, runtime-verified on NFCorpus
  seed 42). Root cause: build-time bounded adapter contaminated C3a's ingest/baseline.
- **Telemetry CUDA-guard fix — committed on H2 branch (`6c3e184`)** + 3 tests. Real CPU/CI
  crash bug (`get_device_name(0)` when CUDA is available-but-hidden). Ships with H2.
- **B1 — Evidence DAG schema (rung 12) — MERGED, PR #277 (`81e3614`), Tier-B 100/none.** Typed
  QUERY/CLUSTER/ACTUATOR graph + validator + JSON schema + deterministic pool builder;
  schema-first, no GNN. Tier-B loop: 72/critical (builder matched a fictional `REFORM`
  token, masked by a fictional-token test → silently dropped real REFORMULATE/CHELATE_ALWAYS
  corrections) → fixed to the canonical `{CHELATE, CHELATE_ALWAYS, REFORMULATE}` vocabulary
  with a discriminating test → 100/none. (Rung 12 → partial: schema landed; live wiring is
  rung 13, gated.)

## Campaigns run with real numbers
- **SciFact swap campaign — REGENERATED on the H1-fixed harness (GPU).** Manifest + result
  doc rebuilt. Headline barely moved (confirms H1 was a cleanliness fix, not a result
  change): C0/C2 0.006 · **C3a 0.161** (was 0.159) · C4a 0.206 · C2O 0.813 · baseline 0.819.
  Budget sweep 0.157 → 0.181. Paper §5.1 updated to these (and "~25×" made consistent,
  60-query eval clarified).
- **NFCorpus swap campaign — RE-RUN IN PROGRESS on GPU** (`b24ehsq01`), the machine having
  freed (~23 GB RAM, GPU ~23 GB free). Resumes the regeneration deferred during a high-load
  window. On completion: commit regenerated docs/manifests + telemetry fix → H2 PR → clear
  CD-H1-01 → finish paper §5.2/§5.3 (the only stale numbers; §5.2 carries a "pending
  regeneration" marker).

## What's blocked
- Nothing hard-blocked right now (machine freed). Earlier window: ~8 GB RAM / GPU full with
  the operator's other overnight work crashed campaigns (OOM + access violations); handled by
  stopping, deferring, and resuming when free.

## Done (light, no PR needed)
- **H4 — one-shot question RESOLVED by characterization** (`h4-one-shot-characterization-2026-06-28.md`):
  flat trajectory is intentional idempotency (per-cycle re-init from a fixed pre-drift
  snapshot); compounding rejected. Paper frames it as one-shot supervised realignment.
- **Zenodo priority-insurance bundle STAGED (not uploaded)** — `zenodo-staging/`, flagged for
  operator (confirm license/author + pull final numbers before upload).

## Next slice (in order)
1. Merge B1 (#277) → checkpoint.
2. Finish H2: let NFCorpus re-run complete → commit regenerated docs/manifests + telemetry
   fix → H2 PR → clear CD-H1-01 → finish paper §5.2/§5.3.
3. **H3 (teacher-supervised C3b)** then **H5 (post-bank proving experiment — load-bearing)** —
   GPU now free. H5 design recon done (`adapter_router.AdapterRouter` = the bank basis for
   the SVF-static + LD-MoLE-router baselines).
4. Then rungs 13/15 (gated on H5 + B1), then the capstone paper.

## Honest note
The work so far hardens the FOUNDATION (Step 14) and lands the DAG substrate — it does not
yet test the thesis. The pivot remains **H5**: does a living annealed bank beat a static bank
and a one-shot router? Until that runs, "the plan is viable" stays a hypothesis.

---

## UPDATE (2026-06-29, later) — H2 BLOCKED on data integrity + concurrent-session collision

**Do NOT trust or commit the H2 re-run numbers.** Verified problems:

- The NFCorpus re-run cells (written 14:24) show a **DIVERGED** C3a baseline (0.556340) vs
  C0 (0.555004) — the *pre-H1* value. The H1 fix makes these EQUAL (confirmed earlier:
  C3a−C0=0.0). So those cells were computed by a harness **without** the H1 fix, even though
  the harness file on disk now has it. The whole H2 re-run (both datasets) is therefore
  suspect and must be re-run in isolation and re-verified (C3a baseline must == C0) before
  any commit.
- **Root cause: a concurrent process is operating on this repo.** The main working tree is
  now on branch **`feat/brain-file-map-b0-b1`** — not created by this session (my H2 work is
  on `feat/drift-h2-rerun-c3a`). An external session switching the shared working-tree branch
  mid-campaign explains the invalid cells, plus the RAM/GPU contention and throttling. A probe
  re-run just OOM'd (host RAM contended again).

**What's safe:** H1 (PR #276) and B1 (PR #277) are merged on `main` and unaffected.

**Resolution (autonomous, no operator action required):** the collision is fixed by
**worktree isolation** — a branch can only be checked out in one worktree, so a dedicated
worktree's branch cannot be switched by the other session. Created
`.claude/worktrees/h2-rerun` on branch `feat/h2-swap-rerun-clean` (off `feat/drift-h2-rerun-c3a`
= H1 fix + telemetry fix; both verified present in the worktree harness). Cleared all swap
cells there and launched a fresh re-run of BOTH datasets inside the worktree (bg `b1o9ql62d`,
GPU, RAM ~21 GB free). It writes only to the worktree's `experiment_runs/` + `docs/`, immune to
the foreign branch. The main working tree (on `feat/brain-file-map-b0-b1`) is left untouched —
that is the other session's; I am not committing the invalid main-tree H2 cells.

**Gate before any H2 commit:** assert `C3a baseline == C0` in the worktree cells (the H1
invariant). Only then commit the regenerated docs/manifests + telemetry fix → H2 PR → clear
CD-H1-01 → paper §5.2/§5.3.

**Next slice:** verify + commit H2 (on re-run completion), then H3 (C3b), then H5 (post-bank,
load-bearing). All remaining campaign work stays in the isolated worktree to prevent recurrence.
