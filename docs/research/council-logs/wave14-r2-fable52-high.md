---
panel_type: panel-of-experts (4-model council, Round 2 of 3)
artifact: wave14-r0-master-roadmap-and-pm6-formal-spec
run_date: 2026-11-24
run_id: wave14-r2-fable52-high
challenge_depth: round-2-of-3
active_experts: [diligence-reviewer, formal-proof-auditor, statistical-methodologist] — simulated lenses, one model (claude-fable-5-2, effort high)
persist_version: 1.1.0
---

# Wave 14 R2 — Fable 5.2: adjudication of C1–C6

**Coverage.** R0 and all four R1 memos in full; new reads in `d9`, `d10`, `ek5`, `bridge` at the lines cited. Nothing executed. **[V]** verified in source, **[I]** inference, **[U]** unverifiable here.

## C1 — D10 Part B: I withdraw the cut
My R1 cut cited no evidence. The run is small [V] (`d10:1351–1362`) and answers "a bigger reader fixes it". Keep it, on three conditions:
1. **The grid is not crossed [V].** Chelated arms get Nemotron-30B extraction and `gated_decisions` (`d10:1189, 1331–1332`); no `70B + Chelated` arm exists. Claim "small reader + governance vs bare 70B", never "governance beats scale".
2. Say "Llama-3.3-70B 4-bit NF4", not "frontier".
3. Run after D9-B with headroom measured, one attempt. A load failure is recorded NOT RUN and never blocks the v6.0 seal.

## C2 — Holdback: Carbon's conditional design fails
Zero pre-verdict release needs `H ≥ decision_length+s+L−2` (s: stride; L: verdict latency in tokens).
- **Tier 2A (10-gram), all traffic:** `H₀ ≥ 8+s+L`. With `H=0` the match is decidable only after nine tokens are out, so "arm on Tier 2A" arms after the leak. Carbon's `<2 ms` benign overhead and a zero-leak verbatim claim cannot both hold.
- **Tier 2B (W=64), armed streams:** `H₁ ≥ 62+s+L`. Arm on a soft prompt signal (entity mention, proximity at `τ_arm < τ_gov`, translate/summarize intent). Arming costs latency, not refusals, so "never block entity mentions" survives. Unarmed exposure is at most `62+s+L−H₀` tokens.
- **Latency.** Scoring 480 vectors is negligible [I, unmeasured]; two costs are missing. The encoder forward per stride dominates, so `s=1` is infeasible and L must be measured. CSLS `r_q` is online, from the query's own top-k (`d10:416–417`) [V]; over 480 restricted vectors that is a different population from calibration. Freeze a reference sample and lock τ on it.
- **Wording:** "0 client bytes of any flagged window on armed streams (k/N, one-sided 95% bound); ≤ K tokens unarmed; arming recall on R3 and detector miss rate reported separately."

## C3 — R3 framing stands; I withdraw the Phase 3 cut
The four-way ablation turns my objection into a test. Pre-register the nulls ("bridge adds nothing over monolingual MiniLM-384"; "native 4096-d wins"). DEV ≥ 40–50 with paired benign siblings, bootstrapped by work: agreed. No memo contests the privilege/opt-out framing.

## C4 — Unified D9-B protocol
- **Reward tie: adopt.** `d9:155–157` [V]: −0.55 > −1.00, so Neon's mechanism is real. It is derivable without data, hence not DEV-tuned. K4 is then tested "under reward v2", ledgered.
- **`scale_rewards=False`: [U].** `trl` is not installed here, so 1.18.2 is unconfirmed. From older releases I recall `scale_rewards` and a separate `loss_type="dr_grpo"`; Neon merges two knobs. Reward-side emulation is impossible: z-scoring is affine-invariant, so the tie is the only reward-side lever. `d9:2552` omits an unsupported field silently [V]; assert the field on Spark #2, record the resolved value, fail closed.
- **Seven changes at once** ship as "protocol v2", with no claim about which cured collapse. Freeze β at 0.05.
- **Seeds: concede to Carbon** (137, 2026, 31415; `seed_42` retired). My F2 loop stands: train all, hash adapters, one VIRGIN pass.
- **Reject as acceptance criteria:** Neon's "GRPO > SFT, p<0.05" (that is the hypothesis) and Carbon's `<1e-4` on bf16. The `[0.25, 0.45]` abstain band is an abort tripwire only.
- **Parity:** the orchestrator's note confirms my F4 candidate. Cast adapters `.float()`, gate fp32–fp32 at 1e-4, disclose the bf16 figure ungated, reject NaN.

## C5 — /formal
- **Procrustes as stated is false for the direction PM-6 uses.** `bridge:315–345` fits `Q = UVᵀ`, shape `(d_x, d_y)` [V]. For 4096→384, Q is tall and `‖XQ‖_F` varies with Q (unbalanced Procrustes). Counterexample: `X=diag(1,10)`, `y=(1,1)ᵀ`. `UVᵀ = (1,10)/√101` gives squared residual ≈ 80.9; `w=(1,0)` gives 1.
- **True and provable:** trace optimality, `tr(WᵀXᵀY) ≤ tr Σ` for all Stiefel W, both directions. Frobenius optimality is a corollary only when `‖XW‖_F` is constant (384→4096, or whitened X). Taking U, Σ, V as hypotheses avoids SVD existence; add a satisfiability witness.
- **Does it cure F9? Partly.** It separates fitted from random W, but in-sample, on centered anchors, for the plain branch only (not soft-spec or rank-truncated, `bridge:322–345`). Publish the residual.
- **Theorem 4:** Carbon and I converge on the cyclic alternating fixpoint (`ek5:189–237`). Add Astra's "least ≠ unique" wording and per-mutation conformance with deletions.
- **Theorem 5:** true but one line (projection is the bound). For PM-6, Neon's own "freeze `μ_q = μ₀` in gates" suffices.
- **Floor:** axiom allowlist, `native_decide` banned; `nanoda` where buildable.

## C6 — Baselines, PM-1b, Spark #3, verifier
- **Arms:** adopt Tier-2A-only, Gate-1-only, Gate-2-only, raw cosine. If n-gram-only matches Arm 6 on R1/R2, the contribution is R3/R4 only. One steering arm, timeboxed, labeled "untuned reference".
- **R2b:** target the arming rule (no entity mention, multi-turn), fixed budget, after threshold freeze.
- **PM-1b:** 95.00 vs 95.83 on N=120 is one item, and no margin was registered before the look. Say "no detectable difference on clean chat (underpowered); 2.42× compaction; gate retired as a stated negative".
- **Spark #3 [U]:** acceptable if shards are byte-hashed after transfer and free memory covers the 44 GiB RSS gate (`d10:914`); otherwise Spark #1 after D7.
- **Verifier:** `<45 s` fits hashes and signatures; `lake build` and 1,000-graph parity need a slower target.

**Wave order.** (1) D7 patch, replay tests, Spark #1. (2) D9-B v2, Spark #2. (3) D10-A, Spark #3 if hashed. (4) D10-B, Spark #2. (5) v6.0 seal, verifier, human PM-4b. (6) v7: Theorem 4, then 1/1b/2, then PM-6 Phase 0.

**Unknowns.** trl 1.18.2 fields; encoder latency L; bnb 4-bit load on GB10; Phase 0 yield.
