# Wave-4 R1 advisory audit — Grok 4.7

Read-only audit of Sol PM1b, Opus RLVR+D3, and Neon PM4+GRPO. Counts recomputed from the published tables. No sealed keys or runners.

## 1. Independent verification

**PM4 κ identities hold.** 92/92/8/8 on n=200: p_o=0.92, yes-rates 0.50, p_e=0.50, κ=0.8400. LoCoMo 41/54/2/3 → 0.8983. LongMemEval 51/38/6/5 → 0.7762. Structured 136/183/0/1 → 0.9936. Pooled 520 → 0.9341; pooled 1000 → 0.9554. Delta 0.84−0.7414=0.0986. CI [0.7587, 0.9099] covers 0.7414 and sits under 0.80; LongMemEval is [0.6401, 0.8973]. κ=2p_o−1 at these margins; 16 errors at yes-rate 0.90 would be κ≈0.56. `single-session-user` is κ=0.4783 on n=16, no CI.

**PM1b paired counts hold, and `assignment_gates_pass` is false.** Clean LME gated−flat=(82−43)/500=0.078 (LCB 0.036). Clean LoCoMo=(30−42)/300=−0.040. R5 knowledge-update gated−flat=(23−3)/78=0.2564; bootstrap LCB 0.1410 and Tango LCB 0.1409 miss ≥0.15. Hybrid−flat is +21.79 pp; gated−hybrid is +3.85 pp (10 vs 7, LCB −6.4 pp). Zero-discordance CP ±0.0145 on n=300 matches the Bonferroni difference of two 97.5% marginal CPs (`scripts/night1_common.py:500-503`). `lcb_975` copies `lcb_95`.

**RLVR algebra holds.** Labels 354/354/292. Always-abstain = p_U = 0.292. Collapse = 0.416. Oracle−collapse = 0.584 = 2 p_U. Binary-exact collapse−oracle = p_U. Unique argmax 1000/1000, margin 0.25 (1.00 vs 0.75). D3 0/80 ASR upper 0.04506 = beta(1, 80). B4 = 77 correct + 3 abstain.

## 2. Findings

### CRITICAL

**C1. GRPO eval rows are train rows.** `build_ek5_verifiable_rlvr_dataset` (`scripts/colab_h100_70b_and_epistemic_grpo.py:352-440`) builds text from `idx`; the seed only permutes (`:439-440`). Train is idx 0..119 (`:511`); eval is idx 0..39 (`:512`). All 120 eval items are copies of train items.

**C2. The GRPO reward is a substring test.** `:472` pays +1.0 when `g_lo in ans`, so `not 6.1.52` is correct. On U (`:466-468`) any of `abstain|contested|unresolved|cannot determine` pays +1.0 unless a listed stale string appears. U stales are only `(v_old, v_new)` (`:434`), so `cannot determine, use {v_mid}` is perfect. `format_epistemic_tags_reward` (`:483-489`) adds +0.25 with no truth condition (formatted hack 1.25 vs bare correct 1.0). `run_offline_verify` (`:664-666`) never tries these strings. Eval (`:574`) uses the same `in` test (`:109-111`).

**C3. Pooled κ mixes an old rubric with a self-judge.** `sample_structured_r5b_items` (`scripts/run_r6_pm4_conversational_kappa_n200.py:415-416`) loads stored R5b bits and `:734-750` concatenates them. Structured κ=0.9936 is 320/520 of the pool the gate at `:814` treats as one number. Judge A is Nemotron (`:7-8`, `:625`); PM1b's answer model is `nemotron-lightning`.

**C4. D3 cannot see a cross-value copy, and ASR=0 is not success.** `infer_copy_edges` clusters only inside one asserted value (`scripts/run_phase2_d3_multiagent_poisoning_bus.py:387-389`). `calibrate_taus` (`:460-475`) maximizes root-anchored attacker groups at FP=0 on the same templates (seeds 777001 vs 20261122). Single linkage (`:350-351`) chains. τ_cross=0.15 is that recall-max point. B3/F2 is ASR 0, accuracy 0, abstain 1.0 because 384-d and 768-d vectors are never compared (`:380-381`). B2's missing 0.25 is a third concrete value. B4's residual is F4 reader-abstain 3/20 with `llm_calls=80`. Co-design has to govern the abstract.

### MAJOR

**M1. PM1b's lift is the ungated hybrid; abstention gets worse as recall rises.** Clean gated−hybrid Δ=0 (LoCoMo gate 8/300; LME gate 10/500, same prompt 493/500). `retrieve` (`chelated_conversational_v3.py:284-345`) has no empty-context action. Echo collapse needs cosine ≥0.80 and Jaccard ≥0.55 (`:196-206`). LoCoMo type 5 is 0.13 gated vs 0.39 flat, evidence-any on the other 200 is 0.855 vs 0.63, and prompts are ~1666 vs ~413 tokens. `r5_grade` (`scripts/run_r6_pm1b_conversational_sota_closure.py:390-392`) scores abstention by cue (`:81-84`) and skips the judge; a cue inside a non-abstention under 90 characters fails. LME gold of length ≥4 auto-passes as a substring within gold+70 characters (`:384-385`). One Nemotron endpoint answers, writes echoes, and judges.

**M2. Three-valued RLVR pays for withholding the value.** Over-abstention scores 0 (`ek5_wfs_rlvr_env.py:58`); a correct verdict plus `ABSTAIN` adds 0.25 (`:850-851`). The runner-up does not answer. Both val checks call `compute_alternating_wfs_oracle` (`:634`, `:944`). Scenario `want` matched the store on 1000/1000. p_U is 5/17 by round-robin, so 0.584 tracks that mix. Binary-exact makes abstention strictly dominated on all 292 U episodes.

**M3. Rubric v8 and the parser.** Rules 1 and 4 (`scripts/run_r6_pm4_conversational_kappa_n200.py:62-73`) mark extra context YES, a dropped list item NO, and preference as "clearly follows." All 16 preview disagreements are partial credit, elaboration, or date paraphrase. `parse_binary_verdict` (`:167-168`) treats YES and NO in one string as a clean parse. Stratum κ has no CI. The bootstrap (`:206-207`) ignores LoCoMo conversation clusters.

### MINOR

Quote D3's Clopper–Pearson interval, not the [0, 0] bootstrap (`scripts/run_phase2_d3_multiagent_poisoning_bus.py:740-746`). `grade_answer` (`:597-601`) takes one substring. Qwen truncates at 2048 (`scripts/run_r6_pm4_conversational_kappa_n200.py:541`).

## 3. Refinements and v4.0 disclosures

1. Exact-match one answer span. Reject negation. Closed abstain lexicon. Cap format reward at 0.05. Eval on new entities, new values, and a regime absent from SFT and GRPO. Publish reward and exact-match separately.
2. Pre-register the LongMemEval κ lower bound, with a conversation-cluster bootstrap, stratum intervals, and a human slice. Regrade structured rows under v8 or delete the pooled gate. Judge with a family that did not write the answers. State κ=2p_o−1 at these margins.
3. Publish D3 {attack, correct, abstain, other} per family. Fit τ off the eval templates. Forbid chaining and allow a copy hypothesis across different strings. Put co-design in the abstract.
4. Attribute clean LME +7.8 pp to dual-granularity hybrid retrieval. State the clean EK5 answer delta as 0. Treat LoCoMo type 5 and LongMemEval preference (gated 0.10) as primary harms. Add retrieve-nothing and a token-matched flat control. Print `assignment_gates_pass=false` next to the +25.6 pp knowledge-update mean.
5. Disclose the 0.25 verdict margin, the 0 reward for over-abstention, the shared WFS function, and the balanced 17-way mixture.

κ=0.84, D3 ASR=0, and any GRPO delta are not confirmatory until those five are in the paper.
