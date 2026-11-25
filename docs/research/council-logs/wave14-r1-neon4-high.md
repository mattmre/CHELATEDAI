---
panel_type: architecture-diligence-r1
artifact: docs/research/council-logs/wave14-r0-master-roadmap-and-pm6-formal-spec.md
run_date: 2026-11-24
run_id: wave14-council-r1-neon4-high
challenge_depth: round-1-of-3
active_experts:
  - post-training-rlvr-specialist
  - representation-unlearning-researcher
  - high-dimensional-geometry-lead
persist_version: 1.1.0
---

# Wave 14 Council — Round 1 Review (`gemini-4-neon`, High)

**Reviewed Scope:** `wave14-r0-master-roadmap-and-pm6-formal-spec.md`, cross-checked against `scripts/run_wave11_d9_fast_extractor_and_u_grpo.py`, `spherical_migration_bridge.py`, `scripts/run_wave11_d10_5m_real_cutover_and_70b_scale.py`, `scripts/run_wave6_d2_4096d_both_modes_3beir.py`, `ek5_v3_stratified.py`, `chelated_graphiti_arena_v1.py`, and `docs/research/bundle_registry_spec.json`. Read-only audit; no GPU jobs run.

## Findings (Ordered by Consequence)

### 1. [CRITICAL] `D9` Part B `U`-GRPO Collapse Is Structural; R0's `lr/beta/60-step` Patch Only Masks It
- **Location:** `scripts/run_wave11_d9_fast_extractor_and_u_grpo.py:L152-158, L1085-1160, L2080-2109, L2477-2484, L2553-2575, L2627-2640`.
- **Evidence:** Although `E[R_always_abstain] = -0.017 < 0` (`L151`), `FALSE_ABSTAIN_REWARD = -0.55 > WRONG_DETERMINATE_REWARD = -1.00` (`L155-157`). Under TRL `GRPOTrainer` group standardization $A_i = (R_i - \bar{R})/\sigma_R$ (`num_generations=8`), **any hard `T/F` prompt with 0/8 exact rollouts gives strict positive unit-scale advantage to `ABSTAIN`** (1 abstain + 7 wrong $\Rightarrow A_{\text{abstain}} = +\sqrt{7} \approx +2.65$, matching a unique correct answer). Once abstention rises, `U` groups saturate (`frac_reward_zero_std → 1`, zero gradient) while hard `T/F` groups keep reinforcing `ABSTAIN`. Worse, `per_device_train_batch_size=8, num_generations=8, grad_accum=2` yields only **2 prompt-groups/step** (high class-mix variance) at `r=32, lr=2e-5, beta=0.02`. Silver lining: `seed_42` aborted at `L2628` before `L2640`, so `EVAL_VIRGIN_V2` remains uncontaminated by GRPO.
- **Impact:** Lowering `lr=5e-6, beta=0.06` and stopping at `60` steps merely pauses before the `-0.55 > -1.00` attractor dominates; collapse will recur across seeds.
- **Correction:** (1) Set `FALSE_ABSTAIN_REWARD = WRONG_DETERMINATE_REWARD = -1.00` (`E[R_always_abstain] = -0.317`; 0-correct `T/F` groups tie at `-1.00` with $\sigma_R=0 \Rightarrow$ **zero abstention gradient**). (2) Disable std division (`scale_rewards=False` / Dr.GRPO) to stop amplifying small gaps and short `ABSTAIN` traces. (3) Use `grad_accum=8` (8 prompts/step), `r=16`, `lr=5e-6`, `beta=0.05`, `60` steps; keep pre-registered `final_step` PRIMARY (`best_dev` with `det_em >= sft_det_em - 0.03` secondary). (4) Cast adapter weights `.float()` alongside `base_fp32` in `_verify_fp32_merge_and_unload_parity` (`L2090-2106`).
- **Verification:** Across all 3 seeds, `dev_abstain_rate ∈ [0.25, 0.45]` through step 60, `det_em >= sft_det_em - 0.03`, and `GRPO_3SeedMean > SFT_7B` ($p < 0.05$ Holm) on virgin `EVAL_VIRGIN_V2`.

### 2. [HIGH] `CausalEMA` Query-Stream Poisoning (`M6`) & `/formal` Theorem Hardening
- **Location:** `scripts/run_wave11_d10_5m_real_cutover_and_70b_scale.py:L83, L275, L309-327, L390-394`; `scripts/run_wave6_d2_4096d_both_modes_3beir.py:L70, L149-168`; `spherical_migration_bridge.py:L845-886`.
- **Evidence:** `causal_stream_bridge_queries` updates $\mu_q^{(t)} = 0.95\mu_q^{(t-1)} + 0.05 q_t$ unconditionally (`half-life ≈ 13.5` queries). 100 identical/clustered queries $q_{\text{adv}}$ yield $(0.95)^{100}=0.0059$, dragging $\mu_q^{(100)} \approx 0.994 q_{\text{adv}}$. This induces $\|q_{\text{adv}}-\mu_q\|\to 0$ noise blow-up, benign retrieval rotation, and a **direct `PM-6` Arm-6 bypass**: dragging $\mu_q \to v_{\text{restricted}}$ collapses `ZeroMu` similarity on the real extraction query. In `/formal`: Theorem 3's $\varepsilon_{\text{quant}}/\gamma$ ignores near-centroid hubs (`L275` clamps `c_norms` at `1e-6`, blowing up $\Delta_{\text{cert}}$) and `fp32` sidecar rounding; Theorem 2's $\langle q,Wd\rangle=\langle W^\top q,d\rangle$ is trivial unless paired with scale-distortion proof.
- **Correction:** (1) Replace raw EMA with **Anchored-Clipped EMA (Theorem 5 / `M6`)**: $\mu_q^{(t)} = \Pi_{B(\mu_0,\rho)}((1-\alpha-\lambda)\mu_q^{(t-1)} + \alpha q_t + \lambda \mu_0)$ plus duplicate skip ($\cos(q_t,q_{t-1})>0.95$). Prove $\|\mu_q^{(t)}-\mu_0\|_2 \le \rho$ inductively in Lean 4 (bounding angular distortion by $\arcsin(\rho/\gamma_q)$). **Freeze $\mu_q=\mu_0$ (stateless) in `PM-6` Gate 1/2.** (2) Prove Theorem 1 via norm-isometry $\|Bx\|_2=(\prod\alpha_t)\|x\|_2$ (no Mathlib4 SVD tarpit) + `2->3->2` rank-1 counterexample; prove Theorem 2's query-dependent factor $1/\sqrt{1-\eta(q)}$ ($\eta(q)=\|(I-WW^\top)q\|_2^2/\|q\|_2^2$); state Theorem 3 per-doc over $\gamma_d \ge \gamma_{\min}$ with denominator error $\delta_\gamma$. Keep `M3` (Wedin/Marchenko-Pastur) out of Lean 4 as an analytical appendix + empirical $n$-sweep.
- **Verification:** `lake build` zero-`sorry` + clean `#print axioms`; 100-query burst test shows unguarded EMA collapse vs `<0.5 pp` drift under Anchored-Clipped EMA.

### 3. [HIGH] `PM-6` (`D11`) Strawman Baselines & Falsifiable Cross-Lingual `4096→384` Risk
- **Location:** `wave14-r0-master-roadmap-and-pm6-formal-spec.md` §3 (Phases 2–3).
- **Evidence:** (a) Arms 1–4 lack any output filter or white-box steering baseline, leaving `R0`'s critique of *"activation-steering benign over-refusal"* untested. (b) *Inference (unmeasured):* Projecting multilingual `Qwen3-8B` (`4096-d`) via English-fitted `W ∈ St(4096,384)` into English-only `MiniLM` under English $\mu_{\text{en}}$ injects language bias $W^\top(\mu_{\text{lang}}-\mu_{\text{en}})$, miscalibrating `DEV` $\tau_{\text{gov}}$ and conflating `R3` plot paraphrases/translations with `R5` same-entity QA. (c) 64-token/32-holdback windows risk token-interleaving dilution.
- **Correction:** (1) Replace redundant Arm 4 (`mem0ai`, covered in `D7/PM-3`) with **(i) Tier-2A N-gram/MinHash-only** and **(ii) White-Box `Qwen2.5-7B` Activation Steering (CAA) or Contrastive Logit Suppression (WHP)**. (2) Frame Phase 3 cross-lingual carry-forward as a falsifiable test: log $\|W^\top(\mu_{\text{es/de}}-\mu_{\text{en}})\|_2$, ablate $\mu_{\text{en}}$ vs unlabeled per-language $\mu_{\text{lang}}$, and report `4096-d native` vs `4096→384` `ROC-AUC` separately for `R3-paraphrase`, `R3-DE/ES`, and `R5`. (3) Add interleaving attacks to `R2`, decouple verbatim vs synopsis thresholds with `min_infringing_tokens`, and require `TPR(R1..R4) >= 0.90` at `FPR(R5) <= 0.05`.
- **Verification:** Full `ROC-AUC(R3 vs R5)` and leak-vs-false-refusal Pareto table across all 6 arms on held-out `N=60` works.

### 4. [MEDIUM] Sequencing, `D7` Retry Bug, `PM-1b` Re-Scope, & Cuts
- **Location:** `chelated_graphiti_arena_v1.py:L262, L382-404`; `docs/research/bundle_registry_spec.json:L26, L361-367`.
- **Evidence:** `L397` hardcodes `"type": "json_schema"` during `ValidationError` retry even after `json_object` fallback (`L262`), risking HTTP 400. In `wave6_pm1b_v4_reranked_sota`, `Chelated_v4` already hits `95.00%` at `2.42x` compaction vs `DualGran_NoEK5` (`95.83%`); clean chat lacks retractions.
- **Correction:** Use `self._build_response_format(response_model)` at `L397`. Seal `Waves 11–13` (`v6.0`, `45/45` bundles) before `v7.0` (`/formal` + `D11`). Re-scope `PM-1b` as clean non-inferiority + compaction instead of chasing +1% clean QA. Cut Lean formalization of `M3`/`A2g`/`A3`; add a `<60s` 1-command CPU proofpack (`lake build` + `ek5_ref_cli` 1,000-graph parity + `M6` burst test + `45/45` SHA-256 audit).
- **Verification:** `D7` unit test passes under `json_object` fallback + `ValidationError` retry; `v6.0` seals `45/45` before `v7.0` begins.

## Unknowns & Runtime Gaps
- Exact Spark #2 `seed_42` per-step `frac_reward_zero_std`/`kl` logs are absent locally; GRPO fix awaits 3-seed GPU proof.
- Cross-lingual shift norm $\|W^\top(\mu_{\text{es/de}}-\mu_{\text{en}})\|_2$ is unmeasured until Phase 3 runs.

**Reviewer:** `gemini-4-neon` (High) | **Phase:** Round 1 of 3 | **Record:** `docs/research/council-logs/wave14-r1-neon4-high.md`.
