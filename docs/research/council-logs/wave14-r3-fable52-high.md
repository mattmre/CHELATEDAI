---
panel_type: panel-of-experts (4-model council, Round 3 of 3)
artifact: docs/research/council-logs/wave14-r2-synthesis-draft-plan.md
run_date: 2026-11-24
run_id: wave14-r3-fable52-high
challenge_depth: round-3-of-3-final
active_experts: [diligence-reviewer, formal-proof-auditor, statistical-methodologist] — simulated lenses, one model (claude-fable-5-2, effort high)
persist_version: 1.1.0
---

# Wave 14 R3 — Fable 5.2: final binding audit

**Verdict: APPROVED WITH FINAL AMENDMENTS.** Lock once A1–A6 are applied. None reorders a wave.

**Coverage.** Synthesis draft and all four R2 memos in full; new reads at `d10:1184–1363` and `d9:2300–2340`. Nothing built or executed, so every Lean remark is inference. **[V]**: verified in source.

## Amendments (exact deltas)

**A1 — §2.3: D10-B is not "fully crossed" [V].**
- `d10:1291–1294`: a gated arm returns `ABSTAIN` without calling the reader, so on contested cases every Chelated arm is identical whatever the reader size.
- `d10:1249`: every Chelated arm contains a Nemotron-30B extractor.
- `d10:1342–1363`: the 70B runs inline, outside `_eval_hf_reader`. Appending `chelated_ctx` there skips the gate.

Delta: replace "fully crossed" with "reader size varied under a fixed 30B-extractor governance pipeline". Route 70B+Chelated through the same `gated_decisions` short-circuit. Report gate-decided and reader-decided cases separately. Add extraction time to Chelated latency. Never write "governance beats scale". If Part B is `NOT_RUN`, v6.0 documents drop the scale claim.

**A2 — §4.3: holdback.**
1. Release is verdict-gated: a token leaves only after every window containing it has a verdict, and a late verdict withholds. `L_p99` sizes TTFT; as a safety bound it leaks the 1% tail.
2. Profile B's `≤ H₁−H₀` bound holds only if Tier 2B scores every stream. State that, or report unarmed exposure as unbounded.
3. Holdback covers every text-bearing SSE field (logprobs, tool-call arguments, reasoning deltas), or the gateway rejects those options.
4. Compute the Clopper-Pearson bound per work: 0/60 gives ≤ 4.9%.

**A3 — §2.2, §4.2: holdout hygiene.**
1. Astra's "fit centers without test works" was dropped. Bridge `W`, `μ_en`, `μ_lang` and the 4,096-vector `r_q` sample exclude all TEST works and the 480 restricted nodes. Hash those lists before DEV calibration.
2. Split and bootstrap by entity or series unless one work per entity is asserted. Commit the split salt before Phase 0 yields are seen. Set `τ_arm` on DEV. Add a fallback for a 30B yield below 110.
3. D9-B: the VIRGIN pass is one-shot. Any later config or selection change mints `EVAL_VIRGIN_V3`.
4. `d9:2305–2320` resumes SFT VIRGIN scores from `_ckpt_stage1_w11.json` [V]. Bind that file and the merged-SFT hash in the amendment. Add `r`, `grad_accum` and script SHA to the resume key. Name the test statistic, α and seed aggregation before launch.

**A4 — §3, §5: formal.**
1. T1b mixes orientations: the witness is `XᵀY`, the corollary uses `X_leg`, `Y_new`. Fix: `A ∈ ℝ^{n×d₁}` (new), `B ∈ ℝ^{n×d₀}` (legacy), witness `AᵀB = UΣVᵀ`. Then (i) `tr(WᵀAᵀB) ≤ tr Σ`; (ii) `‖BVUᵀ − A‖_F ≤ ‖BWᵀ − A‖_F`, unconditional; (iii) downward optimality only under `AᵀA = cI`, as a separate lemma.
2. Counterexample: use `X = diag(3,4)`, `y = (1,1)ᵀ`. Fitted `(3/5, 4/5)` leaves 137/25; `e₁` leaves 5. No `Real.sqrt`.
3. T3: `ek5_ref_cli` must call a function with a kernel-checked proof that it equals the `lfp(Γ²)` specification. Otherwise the parity test compares three unverified programs.
4. Hash `Statement.lean` before proofs. Timebox incremental-with-deletions; on a miss, omit it and label it "conformance-tested, unproven". No statement weakening.
5. Offline `lake build` needs a pre-populated Mathlib cache and a pinned `lean4export`. A missing cache reports `NOT_RUN`, never pass. `--fast` prints its skipped checks. "4-Module" lists five modules.

**A5 — §4.4: baselines.** Astra's "match operating point" was dropped. Calibrate arms 3, 5, 7 and 8 on DEV at the same benign FPR and report full ROC. Hash-freeze arm 2's prompt before TEST. Compare arm 6 on Qwen2.5-7B only, with no superiority claim. Arm 4 is an expected-null control. Pre-register: if arm 3 matches arm 8 on R1/R2, the claim narrows to R3/R4. R2b reports per-work any-of-K success.

**A6 — §1, §2.2: wording.** Retained sealed bundles (D6, D8, D9-A) predate `v6.0-rc1`, so say "each bundle records its producing SHA". D9-A latency is "gate not met (2,104 vs 1,500 ms)", not a "variance". Assert Spark #1 `MemAvailable ≥ 48 GiB` immediately before D10-A.

## Material dissent retained
- **PM-1b:** Astra holds SCOPED/OPEN; the draft retires it. Registry text: "retired negative; non-inferiority not established, no margin registered".
- **Verifier timing:** Fable and Astra place `--fast` at the v6.0 seal (Wave 13), Neon and Carbon at v7.0.
- **D9-B attribution:** seven changes ship together; none is credited with curing collapse.

## Irreducible limitations
- Lean proves statements over ℝ about a Lean model. fp32 behaviour and the Python implementation are covered only by empirical conformance.
- "Zero client bytes" is a transport property of flagged windows. It does not bound detector misses.
- 60 TEST clusters cannot confirm FPR ≤ 0.02, and `τ_gov` rests on 50 benign works.
- Works are selected for memorization, so arm 1's ceiling holds by construction.
- SFT VIRGIN scores were known when Protocol v2 was designed. With 3 seeds and N=120, a null is weak evidence.
- Unverified here: case citations, Lean compilation, encoder latency `L`, Phase 0 yield, 70B load.

**Reviewer:** claude-fable-5-2 (high), one model. Round 3 of 3.
