---
panel_type: panel-of-experts (4-model council, Round 2 of 3)
artifact: docs/research/council-logs/wave14-r0-master-roadmap-and-pm6-formal-spec.md
run_date: "2026-11-24"
run_id: wave14-r2-carbon4-high
challenge_depth: round-2-of-3
active_experts:
  - gemini-4-carbon (Architecture, Release Governance, & Cross-Model Synthesis)
persist_version: "1.1.0"
---

# Wave 14 Council — Round 2 Adjudication (`gemini-4-carbon`, High)

**Scope:** Static cross-examination of Round-1 memos (`fable52`, `astra62`, `neon4`, `carbon4`) against `R0` (`4fae0b77`), `d9`, `d10`, `ek5_v3_stratified.py`, `spherical_migration_bridge.py`, `chelated_graphiti_arena_v1.py`, and `wave6_d4_epistemic_grpo_manifest.json`. Zero R1 replay.

## Adjudication of Material Conflicts (`C1–C6`)

### `C1` (`D10 Part B` `70B` Sweep): **KEEP** (Serialized + Patched)
- **Ruling:** Reject Fable's (`F21/Q4`) cut. `d10:1161-1534` is already coded, `Llama-3.3-70B-Instruct-bnb-4bit` is cached on Spark #2, and ~25 GPU minutes directly falsifies the primary diligence objection (*"70B with full context `k=17` resolves staleness without EK5"*) via dual cross-family blind judges (`Nemotron-30B` + `Qwen2.5-7B`, `d10:1456-1486`).
- **Patches:** Serialize `D10-B` strictly *after* `D9-B` exits on Spark #2. Apply Astra's `D10` fixes: assert `unmasked_leaks > 0`, `len(fnc) == 10` (`fill@10`), and sampled exact-oracle `admitted_recall@10` (`d10:1093-1116`), and hash `args.part_a_json` / `args.part_b_json` directly (`d10:1570-1594`). In `D7` (`arena:344, 371, 397`), validate repaired JSON before returning, reuse `_build_response_format` on retry, and count `finish_reason == "length"` truncations (`F7/F8`).

### `C2` (`PM-6` Holdback & Gate-2 Latency): **CONCEDE Carbon R1; `72`-Token Holdback**
- **Ruling:** Carbon R1's conditional `0`-holdback is **withdrawn**: `R3` oblique/translated prompts trigger neither Gate 1 nor Tier 2A, and Astra (`Q3`) + Fable (`F19`) proved `holdback=32 < window=64` leaks up to `31` client tokens before token 64 fires. Conversely, Astra (`Q3`) proved Tier 2B scores only the **`480` restricted vectors** (`480x384` = `737 KB` fp32 + frozen `1M`-corpus `mu_d`/`norms_f32`/`r_d_f32` scalars) in **`<0.2 ms` GEMV**—never scanning `1M` vectors; latency is solely window embedding (`<2 ms` MiniLM; `<10 ms` dedicated `Qwen3-8B` on Spark #3).
- **Resolution & Wording:** Set unconditional `holdback = W_detect + B_sched = 72` tokens (`W=64`, stride `16` after Tier 2A+2B clear, full EOS check for `<64` tokens, drop buffer on cancel/timeout; send `<5 ms` SSE `governance_stream_open` keepalive). Claim **"zero unscreened client bytes (`0` post-detection leakage for spans crossing $\tau_{\text{gov}}$ within `W=64`, bounded by the `R3-vs-R5` ROC curve)"**, retiring *"unlearning"* and *"Parametric Recall Neutering"* (`F21`).

### `C3` (`PM-6` `R3` Framing & Phase 3 Cross-Lingual): **Reframe + 4-Way Falsifiable Ablation**
- **Ruling:** Adopt Fable `F18`: reframe `R3–R4` as **Enterprise Privilege Clawback, Embargo, & Contractual Derivative Suppression** (never claiming plot facts are copyrightable expression; cite *In re OpenAI* / *GEMA* by court, docket, date, and posture). Per `F16/F17`, confirm Phase-0 memorization per generator (`7B` and `30B` separately), SHA-split works pre-tuning, expand `DEV` to **`50` memorized + `50` `Q_benign` works** (`N_test = 60 + 60`), and cluster-bootstrap ROCs by work.
- **Keep Phase 3:** Reject Fable's cut; test `ES/DE -> EN 384-d` as Neon's **4-way falsifiable ablation**: `(a)` monolingual `MiniLM-384`, `(b)` `4096->384` ($\mu_{\text{en}}$), `(c)` `4096->384` (unlabeled per-language $\mu_{\text{lang}}$ cancelling $W^\top(\mu_{\text{lang}}-\mu_{\text{en}})$ with zero vault re-embed), and `(d)` native `4096-d` `480`-doc oracle, logging $\|W^\top(\mu_{\text{lang}}-\mu_{\text{en}})\|_2$.

### `C4` (`D9-B` Unified Protocol & `trl 1.18.2`): **Reward Tie + Dr.GRPO + Out-of-Loop Virgin Eval**
- **Ruling:** `F4` is resolved (`d9:2090-2106` `bf16` LoRA vs `fp32` merge); cast `peft_fp32` params `.float()`, probe `>=8` prompts, and assert `isfinite(diff) and diff < 1e-4`. Adopt Neon's fix: `FALSE_ABSTAIN_REWARD = -1.00` (`= WRONG_DETERMINATE_REWARD`) ties 0-correct `T/F` groups at `-1.00` ($\sigma_R = 0 \implies A_i = 0$), killing the $+\sqrt{7}$ abstention attractor (`grad_accum=8`, `r=16`, `lr=5e-6`, `beta=0.05`, `60` steps, `eval_steps=10`, adding `step=0` SFT to `best_dev` per `F3`).
- **`trl 1.18.2` (`wave6_d4:516`) Check:** `GRPOConfig` natively exposes `scale_rewards=False` (`"none"`) and `loss_type="dr_grpo"` [inference from upstream `trl 1.x`; Spark venv unexecuted locally]. **Reward-side scaling cannot substitute** because $(c r - c\bar{r})/(c\sigma_r)$ is scale-invariant; assert `"scale_rewards" in grpo_cfg_params` (`d9:2140`) at startup. Commit `d9b_amendment_v1.json` + hyperparam-keyed resume paths (`F5`), retire `seed_42` as DEV-diagnostic, train fresh seeds `(137, 2026, 31415)` **DEV-only**, freeze 6 adapter SHAs in `selection_frozen.json` (`F2`), and run one locked `EVAL_VIRGIN_V2` pass (`final_step@60` primary; null = *"K4 stands"*).

### `C5` (`/formal` Procrustes Trap, Cyclic $\Gamma^2$ WFS, & `Anchored-Clipped EMA`)
- **Algebraic Catch:** For tall $W \in \operatorname{St}(d_1,d_0)$ ($d_1 > d_0$, $W^\top W = I_{d_0}$), **downward $\|X U V^\top - Y\|_F \le \|X W - Y\|_F$ is FALSE** ($W W^\top \ne I_{d_1}$ makes $\operatorname{tr}(W^\top X^\top X W)$ $W$-dependent; counterexample: $X=\operatorname{diag}(10,1), Y=(1,2)^\top \implies \|X e_2 - Y\|_F^2 = 2 < \|X U V^\top - Y\|_F^2 = 80.8$). Given witness $X^\top Y = U\Sigma V^\top$ ($\Sigma \ge 0$), prove via Cauchy-Schwarz (**zero Mathlib SVD**): **(i) maximal inner-product alignment** $\langle X W, Y\rangle_F \le \langle X U V^\top, Y\rangle_F = \operatorname{tr}(\Sigma)$ and **(ii) upward Frobenius optimality** $\|Y V U^\top - X\|_F \le \|Y W^\top - X\|_F$, curing `F9` alongside affine/normalization disclosure (`bridge:411`).
- **WFS & EMA:** Merge Theorems 2–3 (`\delta_\gamma`, CSLS, empirical $\Delta_{\text{cert}}$ pass rate); cut Lean `M3`. Prove cyclic antimonotone $\Gamma$ / monotone $\Gamma^2$ WFS (`ek5:189-237`), Phase C/D authority non-interference (`ek5:1134-1189`, `F12/F13`), and **Theorem 5 (`Anchored-Clipped EMA` $\|\mu_q^{(t)}-\mu_0\|_2 \le \rho$)** (freezing $\mu_q=\mu_0$ in `PM-6`). Ban `native_decide`/`sorryAx`, add satisfiability witnesses, pin `nanoda_bin` under NDA, and enforce `1,000` per-mutation (insert/delete/authority/cycle) 3-way parity (`Lean` vs `oracle` vs `EK5`).

### `C6` (Baselines, `PM-1b`, Fleet Wave Table, & `verify_all_offline.py`)
- **`PM-6` & `PM-1b`:** Replace `mem0ai` with a crossed 6-arm matrix (`Bare`, `Tier-2A N-Gram-Only`, `Gate-1+RawDelete`, `Gate-2 RawCosine [No ZeroMu/CSLS/EK5]`, `White-Box CAA/Logit-Suppression [P2] / Linear-CAR [P3]`, `Full Chelated`) plus `R2b` adaptive interleaving/homoglyph red-team (`32`-token sub-spans). Retire `PM-1b` (`K2`) as clean-chat non-inferiority + `2.42x` compaction (`+29.5 pp` `KU` drift).

| Wave | Fleet Assignment | Gate |
|---|---|---|
| **`v6.0-rc1`** (Commit D7/D9/D10 fixes) | Spark #1: `D7`; Spark #3 (`200GbE`): `D10-A` (`9.47M`); Spark #2: `D9-B` `(137,2026,31415)` $\to$ `D10-B` (`70B`) | Seal `45/45` `v6.0` bundles; human `PM-4b` |
| **`v7.0`** (`feat/v7-formal-pm6`) | Local CPU: `/formal` (`Thms 1–5`, `nanoda_bin`) $\to$ Spark #1–#3: `D11/PM-6` (`P0–P3` + `R2b`) | **`verify_all_offline.py`** (`<45s`: SHAs, CSV CI/p replay, `0/56` Non-NDA, `1,000`-mutation parity) |

## Residual Unknowns
- Live Spark #2 `trl 1.18.2` `GRPOConfig` check, Phase-0 `7B` yield, $\|W^\top(\mu_{\text{lang}}-\mu_{\text{en}})\|_2$, and human `PM-4b` remain unexecuted.
