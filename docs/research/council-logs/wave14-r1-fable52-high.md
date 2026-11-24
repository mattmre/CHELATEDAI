---
panel_type: panel-of-experts (4-model council, Round 1 of 3)
artifact: wave14-r0-master-roadmap-and-pm6-formal-spec
run_date: 2026-11-24
run_id: wave14-r1-fable52-high
challenge_depth: round-1 independent review (cross-examination pending R2)
active_experts: [diligence-reviewer, formal-proof-auditor, statistical-methodologist] — simulated lenses, one model (claude-fable-5-2, effort high)
persist_version: 1.1.0
---

# Wave 14 R1 — Fable 5.2

**Coverage.** R0 in full; D9 script (`d9`) 2080–2109, 2484–2669, 2740–2834; `arena` 118–413; `ek5` 1–56, 186–237; `bridge` 845–904. D10 and registry grepped only. Nothing executed. **[V]** verified in source, **[I]** inference.

## Q1: D9 Part B, D7

**F1 [V] Ordering claim holds, narrower than stated.** `d9:2627–2629` raises before `:2639–2640`; no GRPO checkpoint touched `EVAL_VIRGIN_V2`. Always-abstain, base and SFT were already scored on it (`:2172`, `:2335`, `:2423`). Say "virgin for the GRPO arm".

**F2 [V] The retune contaminates `seed_42` DEV; VIRGIN stays clean only if the loop changes.** The new config rests on four DEV points (`:2487`, steps 40/80/120/160), so "peaked at 40" means "first look". `:2640` scores VIRGIN inside the seed loop: `seed_42`'s number is visible before 137/2026 train. Protocol:
1. Commit an amendment record before launch (old/new config, DEV history, crash line, script SHA, VIRGIN SHA).
2. Train all seeds with DEV-only selection, hash the six adapters into `selection_frozen.json`, then run one eval-only VIRGIN pass.
3. Primary: `final_step@60`; `best_dev` is descriptive.
4. No retune once any GRPO VIRGIN number exists; otherwise mint `EVAL_VIRGIN_V3` (generator seed, `:285`).
5. Report the 2-seed (137, 2026) mean beside the 3-seed mean.

**F3 [V] The guard is unimplemented and noise-level.** `is_better` (`:2542–2545`) has no guard. DEV has 54 determinate items (0.6852=37/54), so 0.03 is 1.6 items against SE≈0.063. No step-0 candidate exists, so `best_dev` can never pick "SFT only".

**F4 [V/I] The parity diagnosis does not match disk.** `:2090–2105` already loads the base in `float32` and compares fp32-PEFT to fp32-merged. Either the fix is applied (R0 stale) or 0.002645 came from this code and "bf16 base" is wrong. **[I]** Candidates: bf16-stored adapter tensors under `autocast_adapter_dtype=False`, or fp32 round-off making `1e-4` unattainable at 7B. Check adapter dtypes plus one float64 probe; no evals.

**F5 [V] Pre-registration is unverifiable.** The D9 script and `arena` are untracked; "pre-registered" carries no commit timestamp. Resume key (`:2494`) omits lr/beta, so a same-step retune resumes stale results.

**F6 [V/I]** The NI gate (`:2819`) is a point estimate; the bootstrap (`:2740`) ignores seed variance. **[I]** 120 paired items give about 50% power for +10 points, and K4's prior was p=0.45. Pre-commit to reporting a null as "K4 stands".

**F7 [V] D7 retry bug.** The `model_validate` retry (`arena:397–403`) hardcodes `json_schema` even after the client locked `json_object` (`:262`, `:365`) on a server 400, re-sending the rejected format.

**F8 [V] D7 fairness.** `max_tokens` clamps to 4096 (`:306`) with no `finish_reason` check (`:344`); truncated Graphiti output goes to repair, which reads as a sandbagged baseline. `:371` returns repaired JSON unvalidated. Add truncation and retry counters.

## Q2: /formal

**F9 Theorem 1 holds for any semi-orthogonal matrix, including a random one.** It proves conditioning, not alignment. State `∀x, ‖Bx‖=α‖x‖` (κ₂=1, no Mathlib SVD). The deployed map `normalize((x−μ)Q)` is affine and nonlinear, outside the hypothesis; say so in `Statement.lean`.

**F10 Theorems 2 and 3 are near-definitional** (adjoint; linearity of the inner product). The non-trivial content is within-shard rank invariance under cosine, with `‖Wᵀq‖` as the cross-shard factor. Theorem 3 assumes `‖d‖=1` (`bridge:884`) and omits CSLS.

**F11 [I] `Δ_cert` may be vacuous:** worst-case SQ8 error at d=384 is near 10⁻², against top-k gaps near 10⁻³. Report the share of real queries clearing it.

**F12 [V] Theorem 4 as worded is false for the shipped engine.** Stratum 2 is well-founded semantics with negation: Γ is antimonotone, Tarski applies to Γ² (`ek5:23–24, 199`). Mutual `REFUTE` yields U, so the program is not stratified; the "K=4 strata" are pipeline layers. Restricting to DAGs removes every cyclic case and trivializes the result. Honest: Γ² monotone, unique three-valued model, total when REFUTE is acyclic.

**F13 [I]** Also needed: stratum non-interference (suspect: Phase D, `ek5:1169`) and incremental = full **with deletions**.

**F14 Hygiene.** Ban `native_decide`; add satisfiability witnesses. M4 is testing: publish cycle/U-state coverage and add the in-repo oracle as a third party.

**F15 Cut M3 from Lean** (Mathlib lacks Wedin and Marchenko–Pastur). Keep prose plus an n-sweep against tuned ridge; untuned OLS is a strawman.

## Q3: D11/PM-6

**F16 Phase 0 selects on the outcome.** Arm 1 on R1 leaks 100% by construction; label it. Confirm memorization per model under test, not "7B or 30B". **[I]** 7B yield may fall below 80 works; pre-register a fallback and assign DEV/test by hash.

**F17 N=20 DEV does not carry the claim.** 0/20 misses bounds the miss rate at 14% (95%); 0/60 at 4.9%. R0 gives DEV no benign negatives, yet τ is a two-class threshold. Add 20 DEV siblings and pre-register the operating point. The 300 evals per arm are 60 clusters; bootstrap by work.

**F18 R3 vs R5 is the crux and is mis-framed.** [I, not legal advice] Plot facts are not protected expression; blocking summaries is a policy choice. Frame R3 as privilege/opt-out derivative suppression, and report one R3-vs-R5 ROC with CI at fixed FPR.

**F19 Holdback.** "0 leaked tokens" requires holdback ≥ detection latency; Tier 2B's 64-token window exceeds the 32-token holdback.

**F20 Missing arms:** n-gram filter alone (likeliest to match R1/R2), Gate-1-only, Gate-2-only, cosine without ZeroMu/CSLS, a real steering/unlearning arm. Arm 5 is Phase-3-only, so the grid is not crossed.

**F21 Claims.** Cite cases by court, docket, date, posture. Replace "Parametric Recall Neutering" and "unlearning" with inference-time suppression. Suppression with retained `CITES_AUDIT` is not erasure (D8 too).

## Q4: Adds and cuts

- **Add:** one-command CPU `verify-offline` (rehash bundles, re-derive headline statistics from row CSVs); a committed pre-registration ledger; a budgeted adaptive red-team wave for D11.
- **Sequence:** seal v6.0 first; PM-4b human grading is the release blocker. Close PM-1b or retire it as a stated negative. Theorem 4 precedes D11.
- **Cut:** Lean M3; standalone Theorem 3; cross-lingual Phase 3; D10 Part B 70B sweep.

**Unknowns.** Whether F4's function predates the crash; stratum non-interference; Phase 0 yield.