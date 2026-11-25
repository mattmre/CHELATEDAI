---
panel_type: panel-of-experts (4-model council, Round 3 of 3)
artifact: docs/research/council-logs/wave14-r2-synthesis-draft-plan.md
run_date: 2026-11-24
run_id: wave14-r3-neon4-high
challenge_depth: round-3-of-3-final
active_experts:
  - post-training-rlvr-specialist
  - representation-unlearning-researcher
  - high-dimensional-geometry-lead
persist_version: 1.1.0
---

# Wave 14 Council — Round 3 Final Convergence & Sign-Off (`gemini-4-neon`, High)

**Final Binding Verdict:** **`APPROVED WITH FINAL AMENDMENTS`** (lock Waves 11–15 upon applying `Δ1–Δ6`).

**Scope:** Audited `wave14-r2-synthesis-draft-plan.md` against all four R2 peer memos (`fable52`, `astra62`, `neon4`, `carbon4`) and source code in `d9`, `d10`, `chelated_graphiti_arena_v1.py`, `ek5_v3_stratified.py`, and `spherical_migration_bridge.py` (`HEAD 4fae0b77`). Read-only; zero GPU/SSH/Docker runs; zero reads of `*sealed_key*` or `*blind-followup*`. `[V]` = source-verified; `[U]` = unexecuted unknown.

## Last-Mile Stress Check & Required Delta Edits (`Δ1–Δ6`)

1. **`Δ1` — `Theorem1b` Shape & Corollary Bug (Draft L76–78 [V], `bridge:315-349`):**
   - *Flaw:* L76 defines $X^\top Y = U\Sigma V^\top$ with $U\in\operatorname{St}(d_1,d_0), V\in O(d_0)$ ($X\in\mathbb{R}^{n\times d_1}, Y\in\mathbb{R}^{n\times d_0}$), then L78 switches to $\|X_{\text{leg}} V U^\top - Y_{\text{new}}\|_F$ *when $d_0\le d_1$ (or $X^\top X=I$)*, breaking Lean types and conflating the isotropic condition.
   - *Delta:* Fix $Y_{\text{new}}\in\mathbb{R}^{n\times d_1}$, $X_{\text{leg}}\in\mathbb{R}^{n\times d_0}$ ($d_0\le d_1$), $Y_{\text{new}}^\top X_{\text{leg}} = U\Sigma V^\top$, $W^* = U V^\top$. Prove: **(a)** inner-product optimality $\operatorname{tr}(W^\top Y_{\text{new}}^\top X_{\text{leg}})\le\operatorname{tr}(\Sigma)$ over $\operatorname{St}(d_1,d_0)$; **(b)** unconditional lifting Frobenius optimality $\|X_{\text{leg}}(W^*)^\top - Y_{\text{new}}\|_F \le \|X_{\text{leg}} W^\top - Y_{\text{new}}\|_F$; **(c)** downward Frobenius optimality only under isotropic $Y_{\text{new}}^\top Y_{\text{new}} = c I_{d_1}$, falsified at $d_1=2, d_0=1$ by $Y_{\text{new}}=\operatorname{diag}(1,10)$, $X_{\text{leg}}=(1,1)^\top$.

2. **`Δ2` — `Theorem2` `ZeroMu` Sidecar Premise (Draft L79 [V], `d10:390-394`):**
   - *Flaw:* L79 says `(given unit-norm d or stored centered norm ||d - μ_d||_2)`. Unit-norm $d$ alone cannot determine $\|d-\mu_d\|_2$ without a per-doc scalar; `d10:390-394` divides by stored `leg_norms_f32` ($\|d-\mu_d\|_2$) using unit-norm centered query $\hat{q}_c$, requiring no unit-norm assumption on $d$.
   - *Delta:* State `(given unit-norm centered query q̂_c and stored 4 B/doc centered norm ||d - μ_d||_2; unit-norm d not required)`.

3. **`Δ3` — `D9-B` Surviving Code Traps (Draft L59–63 [V], `d9:2557, 2604-2605, 2636, 2640-2663, 2672`):**
   - *Flaw:* (i) `d9:2604-2605` saves `final_step` into `best_dev_dir` if missing, overwriting `step=0`/`"none_eligible"`; (ii) `d9:2636` hard-asserts `bf16` reload diff `< 1e-4`; (iii) `d9:2640-2663` scores `EVAL_VIRGIN_V2`, `EVAL_WAVE6_LOOK2`, and `D1_k17/k5` inside the seed loop; (iv) `d9:2672` hardcodes `prompt_groups_seen = run_steps * 2`.
   - *Delta:* Remove `d9:2604-2605` fallback; freeze `merged_sft_dir` + up to 6 distinct GRPO adapter SHAs in `selection_frozen.json`; downgrade `d9:2636` to ungated telemetry (gating `fp32`-vs-`fp32` `< 1e-4` across `>=8` probes); move all test-split evals to one post-freeze pass; compute `prompt_groups_seen = run_steps * grad_accum`.

4. **`Δ4` — `D10-B` `70B + Chelated` Fairness & Judging (Draft L67 [V], `d10:1189, 1291-1294, 1419-1423, 1486`):**
   - *Flaw:* Reusing `chelated_ctx` on `70B` without `gated_decisions` breaks parity with `1.5B/7B + Chelated` (`d10:1291-1294`), omits `70B + Chelated` from `qwen_judge_scores` (`d10:1419-1423`), and overclaims a fully crossed grid (Nemotron-30B extracts for all Chelated arms; `Bare 7B/30B` lack `k=17`).
   - *Delta:* Pass `chelated_decisions` to `70B + Chelated`, add it to `qwen_judge_scores`, label it **`Llama-3.3-70B (4-bit NF4) Reader + Chelated (Nemotron-30B Extractor)`**, and include `70B+Chelated − Bare_70B` in the Holm family (`d10:1486`) or mark it descriptive.

5. **`Δ5` — `PM-6` `r_q` Contamination, Latency, & Stride Barrier (Draft L96, L99–102):**
   - *Flaw:* The frozen `4,096`-vector `r_q` background pool lacks a disjointness rule against `DEV`/`TEST` restricted works and `Q_benign` siblings; `<0.2 ms` covers only the `480`-node GEMV, omitting the `4,096`-vector `r_q` GEMV (`6.3 MB` at `384-d`; `67 MB` at `4096-d`); Profile B L102 leaves stride `s` unspecified.
   - *Delta:* Enforce strict disjointness on the `4,096`-vector `r_q` pool; measure `r_q` GEMV separately from `480`-node scoring and window embedding; pin `s_2A = 1` (`H_0 >= 9 + ceil(L_2A)`, emit-after-verdict barrier) for Profile B and `s_2B = 16` (`H = 64 + 16 + ceil(L_p99)`) for Profile A.

6. **`Δ6` — Lean-4 Compilation Guard (Draft L73, L80):**
   - *Delta:* Prove `Theorem3` `lfp(Γ_P^2)` via finite-set Knaster-Tarski iteration over `Finset (Fin n)` (avoiding heavy Mathlib order-topology imports), and verify clean Windows `lake build` + `nanoda_bin` on pinned commits before Wave 14B.

## Surviving Irreducible Limitations & Retained Material Dissent

- **D9-B Confounding & SFT Prior Look:** Protocol v2 bundles 7 knobs (no single-knob causal claim); `EVAL_VIRGIN_V2` is virgin for GRPO only (SFT observed).
- **`PM-1b` Dissent (`Astra R2`):** Astra retained `SCOPED/OPEN` (`95.00%` vs `95.83%`, `N=120`, no pre-registered margin). Draft's `RETIRED_AS_STATED_NEGATIVE` stands only while strictly prohibiting statistical non-inferiority claims.
- **Runtime Unknowns `[U]`:** Human `PM-4b` blind audit (release gate), `bnb 0.50.2` 4-bit `70B` load on GB10 (`NOT_RUN` fallback), `D11` Phase-0 `7B` yield, live `Qwen3-8B` `L_p99`, and cross-lingual `R3-ES/DE` AUC vs $\|W^\top(\mu_{\text{lang}}-\mu_{\text{en}})\|_2$.

**Next Check:** Apply `Δ1–Δ6`, commit `v6.0-rc1` patches + offline replay tests, and execute Waves 11–15 in order.

**Reviewer:** `gemini-4-neon` (High) | **Phase:** Round 3 of 3 (Final) | **Record:** `docs/research/council-logs/wave14-r3-neon4-high.md`.
