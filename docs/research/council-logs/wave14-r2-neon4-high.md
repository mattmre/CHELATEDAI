---
panel_type: panel-of-experts (4-model council, Round 2 of 3)
artifact: docs/research/council-logs/wave14-r0-master-roadmap-and-pm6-formal-spec.md
run_date: 2026-11-24
run_id: wave14-r2-neon4-high
challenge_depth: round-2-of-3
active_experts:
  - post-training-rlvr-specialist
  - representation-unlearning-researcher
  - high-dimensional-geometry-lead
persist_version: 1.1.0
---

# Wave 14 Council — Round 2 Adjudication (`gemini-4-neon`, High)

**Scope:** Adjudicates `C1–C6` across all four R1 memos (`fable52`, `astra62`, `neon4`, `carbon4`) plus `d9:2090-2106, 2140, 2552-2585` and `d10:1161`. Read-only; no GPU/SSH/Docker jobs.

## C1 — `D10 Part B` (`70B` Sweep): **KEEP (Serialized in `v6.0`)**
Overrule Fable's cut (`F21/Q4`). `cmd_run_part_b_70b_sweep` (`scripts/run_wave11_d10_5m_real_cutover_and_70b_scale.py:1161`) is coded and `unsloth/Llama-3.3-70B-Instruct-bnb-4bit` is cached on Spark #2. Marginal cost is tiny while diligence value is high: buyers ask whether `70B` reads past stale chunks without epistemic governance. Disclose `4-bit NF4` everywhere, verify `:8890` headroom post-`release_native_heap()`, and run strictly **after** `D9-B`. Patch `D10-A` first (`fill@10`, positive unmasked exposure, mandatory input-byte digests at `:1092, 1570-1590`).

## C2 — `PM-6` Holdback & Gate-2 Latency: **REJECT `0`-Holdback Bypass; Ship Two Modes**
Astra and Fable refute R0's `32`-token holdback (`32 < 64` leaks up to `31` tokens on stride-1 onset) and Carbon's conditional `0`-holdback bypass (`R3` paraphrases/translations lack literal Gate-1/Tier-2A triggers). Astra also corrects the latency math: exact `ZeroMu+CSLS` scores only the **`480` restricted vault vectors** (`480x384` / `480x4096` + precomputed `1M`-corpus $r_d$ sidecar, `<0.2 ms`), **not** a `1M` scan; window embedding dominates latency.
- **Profile A (Strict Zero-Leak Transport):** Enforce $H \ge W + \delta_{\text{sched}}$ (`H=64` for `W=64`, or full-response buffer on short outputs). Claim **zero unverified client-byte leakage for detector-caught windows** (tested across offsets, Unicode, split encodings, flush, cancel)—strictly separate from detector recall. Publish `64*TPOT` TTFT tax and p95/p99 latency.
- **Profile B (Conditional Low-Latency SLA):** Separate Pareto point (benign TTFT `<5 ms`, bounded exposure when Gate 1/Tier 2A misses `R3`). Never blend Profile A safety with Profile B latency.

## C3 — `PM-6` `R3` Framing & Phase 3 Cross-Lingual: **ADOPT Framing; KEEP Falsifiable Ablation**
Adopt Fable's `F18/F21` split: copyright covers expression (`R1/R2`), while `R3` plot/translation blocking is **privilege-clawback / contractual opt-out derivative suppression**. Replace *"Parametric Recall Neutering"* / *"unlearning"* with **inference-time suppression & governance carry-forward**, cite cases by court/docket/date/posture, and disclose that retained `CITES_AUDIT` is suppression, not erasure.

Overrule Fable's Phase 3 cut: zero-reembed `384->4096` carry-forward is the differentiator. Run a **pre-registered 4-way falsifiable ablation** on paired `R3-ES/DE` vs `R5`: `(a)` monolingual `MiniLM-384`, `(b)` English-centered `4096->384` ($\mu_{\text{en}}$), `(c)` language-centered `4096->384` ($\mu_{\text{lang}}$), `(d)` native `4096-d Qwen3-8B`; report honestly if `(b)` fails from $W^\top(\mu_{\text{lang}}-\mu_{\text{en}})$. Confirm Phase-0 memorization per model; expand `DEV` to `N_dev >= 40` works **paired with `>= 40` benign siblings (`Q_benign`)** for two-class $\tau_{\text{gov}}$ calibration; hash-assign `DEV/TEST`; cluster-bootstrap CIs **by work** (`60` clusters).

## C4 — `D9` Part B Unified Protocol & `trl 1.18.2` Check: **LOCK ALL FOUR PILLARS**
1. **Structural GRPO Fix + `trl` Guard:** Set `FALSE_ABSTAIN_REWARD = WRONG_DETERMINATE_REWARD = -1.00`, `grad_accum=8`, `r=16`, `lr=5e-6`, `beta=0.05`, `60` steps (`eval_steps=10`). **`trl` is absent locally (Spark #2 runtime unknown):** upstream TRL `GRPOConfig` widened `scale_rewards` from `bool` (`False`) to `Literal["group","batch","none"]`. Via `d9:2140, 2552`'s `inspect.signature(GRPOConfig.__init__)` guard, pass `scale_rewards="none"` (fallback `False`; **hard-fail if absent**—reward functions cannot undo TRL's internal `/std` division). Log `scale_rewards`/`loss_type` in `env_versions`; add `lr/beta/scale_rewards/false_abstain_reward` to the resume key (`d9:2494`).
2. **Parity Fix (F4 resolved):** Per `d9:2090-2106`, cast loaded `bf16` `lora_A/B` weights `.to(torch.float32)` before unmerged CPU forward, log actual dtypes, check multiple prompts/positions, and reject non-finite diffs.
3. **Confirmatory Discipline (Fable `F2` + Carbon):** Retire `seed_42` as aborted DEV-diagnostic; train **3 fresh seeds `(137, 2026, 31415)`** with DEV-only selection (remove `EVAL_VIRGIN_V2` from `d9:2640` loop), hash all 6 adapters into `selection_frozen.json`, commit an amendment ledger pre-launch, then run **one eval-only `EVAL_VIRGIN_V2` pass** (`final_step@60` primary; `best_dev` with `det_em >= sft_det_em - 0.03`, step-0 SFT, and `"none_eligible"` secondary). Report a null as *"K4 stands"*.

## C5 — `/formal` Non-Triviality, Procrustes, Cyclic $\Gamma^2$ WFS, & `M6`
- **Procrustes Optimality Cures `F9` (Orientation Guard):** **YES.** Given thin-SVD witness $Y^\top X = U\Sigma V^\top$ ($\Sigma\ge 0$ diagonal, $U\in\operatorname{St}(d_1,d_0)$, $V\in O(d_0)$), Frobenius optimality of $W^*=UV^\top$ over all $W\in\operatorname{St}(d_1,d_0)$ needs **only trace cyclicity + Cauchy-Schwarz** ($|(U^\top WV)_{ii}|\le \|Ue_i\|_2\|WVe_i\|_2=1$)—**zero Mathlib4 SVD existence**. **Critical guard:** $\|XW-Y\|_F$ preserves $\|XW\|_F^2$ only for square $d_1=d_0$; for rectangular `384->4096` ($W^\top W=I_{d_0}$), state it in Chelated's **lifting orientation** $\|X_{\text{leg}}W^\top - Y_{\text{new}}\|_F \ge \|X_{\text{leg}}(UV^\top)^\top - Y_{\text{new}}\|_F$ so $\|X_{\text{leg}}W^\top\|_F^2=\|X_{\text{leg}}\|_F^2$ cancels.
- **Locked Suite:** `T1a` norm-isometry + `2->3->2` counterexample (+ affine/nonlinear disclaimer); `T1b` rectangular Procrustes; `T2` adjoint + $1/\sqrt{1-\eta(q)}$ distortion + within-shard cosine invariance (fold `T3` into a per-doc $\gamma_d\ge\gamma_{\min}$ lemma with $\delta_\gamma$ and empirical `% queries clearing \Delta_cert`); `T4` **cyclic** antimonotone $\Gamma$ / monotone $\Gamma^2$ WFS + incremental=full **with deletions** + stratum non-interference (audit `ek5:1169`); `T5` **`Anchored-Clipped EMA`** $\|\mu_q^{(t)}-\mu_0\|_2\le\rho$ (+ mTLS authority-tier binding; stateless $\mu_q=\mu_0$ in `PM-6`). Ban `native_decide`; cut `M3` from Lean (keep $n$-sweep vs tuned Ridge); require **per-mutation** `ek5_ref_cli` parity, `nanoda_bin`, axiom allowlist/negative control, and NDA gating.

## C6 — `PM-6` Baselines, `PM-1b`, Spark #3 Offload, & Wave Table
Drop redundant `mem0ai`; run crossed baselines (`Bare LLM` canary ceiling, `Prompt Guardrail`, `Vector DB Delete`, **`Tier-2A N-gram-Only`**, **`White-Box CAA/WHP Steering`**, `Chelated Dual-Gate` + `Gate-1-only`/`Gate-2-only`/`Raw-Cosine`/`Linear-Bridge` ablations), adaptive red-team **`R2b`** (interleaving + unseen obfuscations), and integer-only `D8` `copilot_receipt_v2` (`0/56` Non-NDA). Re-scope `PM-1b` as **Clean Non-Inferiority (`95.00%` vs `95.83%`) + `2.42x` Compaction + `+29.5 pp` `KU` Drift Gate**. Patch `D7` (`arena:397` `_build_response_format`, `arena:371` validation, truncation/retry counters).

| Release | Node | Clean Detached SHA | Exit Gate |
|---|---|---|---|
| **`v6.0`** | Spark #1 | Patched `D7` | `arm_failure=0` |
| **`v6.0`** | Spark #3 (`200GbE`) | Patched `D10-A` | `0` leak, `fill@10=100%` |
| **`v6.0`** | Spark #2 (serial) | `D9-B` (`137,2026,31415`) $\to$ `D10-B` (`70B`) | `45/45` bundles, `0/56` Non-NDA |
| **`v7.0`** | Local (`feat/v7`) $\to$ Fleet | `/formal` (`T1a–T5`, `nanoda`) + `D11` + `<45s` `verify_all_offline.py` | Offline exit `0`; human `PM-4b` |

**Unknowns:** Spark #2 `trl 1.18.2` `scale_rewards` signature; `ek5:1169` non-interference; Phase-0 `7B` yield; live cross-lingual `ROC-AUC`.
**Reviewer:** `gemini-4-neon` (High) | **Phase:** Round 2 of 3 | **Record:** `docs/research/council-logs/wave14-r2-neon4-high.md`.
