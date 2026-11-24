---
panel_type: panel-of-experts-final-consensus
artifact: docs/research/council-logs/wave14-r3-final-consensus-master-plan.md
run_date: "2026-11-24"
run_id: wave14-r3-final-consensus-4model-3round
challenge_depth: round-3-of-3-unanimous-convergence
active_experts:
  - claude-fable-5-2 (effort: high)
  - gpt-6.2-astra (reasoningEffort: high)
  - gemini-4-neon (thought_level: high)
  - gemini-4-carbon (thought_level: high)
persist_version: "1.1.0"
---

# Final 3-Round 4-Model Council Master Plan (`Waves 11–15`: `v6.0` $\to$ `v7.0`)

**Unanimous Round-3 Binding Verdict (`4/4` High-Effort Models):** **`APPROVED WITH FINAL AMENDMENTS — ALL ROUND-3 DELTAS APPLIED BELOW`**
- **Round 1 Memos:** `wave14-r1-fable52-high.md`, `wave14-r1-astra62-high.md`, `wave14-r1-neon4-high.md`, `wave14-r1-carbon4-high.md`
- **Round 2 Cross-Adjudication Memos:** `wave14-r2-fable52-high.md`, `wave14-r2-astra62-high.md`, `wave14-r2-neon4-high.md`, `wave14-r2-carbon4-high.md`
- **Round 3 Final Binding Sign-Offs:** `wave14-r3-fable52-high.md`, `wave14-r3-astra62-high.md`, `wave14-r3-neon4-high.md`, `wave14-r3-carbon4-high.md`

---

## 1. Release Boundary, Provenance, & Fleet Placement (`Waves 11–15`)

1. **Strict Release Separation (`v6.0` Sealed Before `v7.0` Merges):**
   - Complete **`v6.0` (`Waves 11–13`, `45/45` bundles)** before merging any `v7.0` (`/formal` or `Demo D11 / PM-6`) code into `fix/stage0-metric-lineage-and-m1-ladder`.
   - **Bundle-Level SHA Provenance (`Fable A6`):** Retained already-sealed bundles (`D6` `0578d56b`, `D8` `3fea611f`, `D9-A`) record their exact producing SHA/file digests in `part_vii_code_provenance`; newly executed runs (`D7`, `D9-B`, `D10-A`, `D10-B`) execute from a single committed `v6.0-rc1` SHA synced cleanly across Spark #1 and Spark #2 (`0` untracked `scp` overlays).
   - **Release Gate:** `release_status` remains `PRE-RELEASE / HUMAN AUDIT PENDING` until human `PM-4b` (`pm4_human_audit_100_blind_sheet.csv`) is graded by a human (`0` agent self-grading, `0` reads of `*sealed_key*` or `*blind-followup*`).

2. **Measured 4-Node Fleet Placement:**
   - **Spark #1 (`10.0.0.49`, `52 GiB` `MemAvailable`, `383 GiB` free NVMe):** Runs patched **`Demo D7`** $\to$ asserts `MemAvailable >= 48 GiB` $\to$ runs patched **`Demo D10 Part A`** (`8.84M` standalone MSMARCO + `9.47M` 7-BEIR cutover; shards already local). *(Offloading `D10-A` to Spark #3 is rejected because Spark #3 has only `37 GiB` `MemAvailable` against `D10-A`'s `44 GiB` RSS gate.)*
   - **Spark #2 (`10.0.0.194`, `76 GiB` `MemAvailable`, `trl 1.18.2`, `bitsandbytes 0.50.2`):** Runs **`Demo D9 Part B` (`Protocol v2`)** $\to$ `release_native_heap()` + headroom check $\to$ **`Demo D10 Part B`** (`Llama-3.3-70B 4-bit NF4` reader-scale comparison) **strictly sequentially**.
   - **Spark #3 (`10.0.0.24`, `37 GiB` `MemAvailable`):** Untouched during `v6.0`; hosts isolated `Qwen3-Embedding-8B` (`~16 GiB`) during `Wave 14B` (`Demo D11 / PM-6`).
   - **Local Windows Workstation:** Develops `Wave 14A` (`/formal` Lean 4 + `ek5_ref_cli` + `nanoda_bin`) in an isolated `feat/v7-formal-pm6` worktree concurrently with Wave-11 Spark GPU runs, and runs `scripts/verify_all_offline.py`.

3. **Ordered Wave Roadmap:**
   - **Wave 11 (`v6.0-rc1`):** Apply & unit-test `D7`, `D9-B`, `D10-A/B` patches + commit `docs/research/d9b_amendment_v1.json`; run Spark #1 (`D7 -> D10-A`) and Spark #2 (`D9-B -> D10-B`) in parallel; seal `D7`, `D9`, and `D10` locally.
   - **Wave 12 (`v6.0` Validation & Docs):** 4-model peer cross-validation of `D6–D10` receipts; build `verified_registry.json` `v6.0` (`45/45` bundles); update Whitepaper/Decks `v6.0` + Non-NDA Teaser/Deck (`0/56` IP check); 2-judge final `v6.0` sign-off.
   - **Wave 13 (`v6.0` Seal & Fast Verifier):** Ship **`scripts/verify_all_offline.py --fast` (`<45s` CPU target, printing `formal_tier: "SKIPPED_USE_FULL_FORMAL"`)** at the `v6.0` seal (`Fable A6` + `Astra` timing resolution); pass `ruff`, `pytest`, `build_whitepaper_assets.py --check` (`45/45`); push `v6.0`; seal encrypted vault `WAVE10-V6-DEMOS-D6-D10-COMPLETE.7z`.
   - **Wave 14A (`v7.0` NDA-Gated `/formal` Prerequisite Gate):** Merge & compile 5-module `/formal` Lean 4 suite (`0` `sorryAx`, `0` `native_decide`, pre-populated Mathlib `.olean` cache, pinned `lean4export` + `nanoda_bin` external kernel verification) + **`1,000`-sequence per-mutation 3-way `EK5` differential conformance** BEFORE running `Demo D11`.
   - **Wave 14B (`v7.0` `Demo D11 / PM-6`):** Execute Phases `0–3` + post-freeze `R2b` adaptive red-team across Sparks #1–#3; seal Bundle #46 (`wave14_d11_pm6_takedown_carryforward`).
   - **Wave 15 (`v7.0` Closeout):** Extend `scripts/verify_all_offline.py --full-formal`, run `v7.0` cross-model & judge adjudication, seal `verified_registry.json` `v7.0` (`46/46` bundles), push, and seal `v7.0` encrypted `.7z` vault addendum.

---

## 2. Binding `Wave 11` (`v6.0-rc1`) Code & Statistical Amendments

1. **`Demo D7` (`chelated_graphiti_arena_v1.py` — `Astra #1`, `Fable F7/F8`, `Carbon A3`):**
   - Route **every** parsed response—including repaired JSON (`n_rep > 0`)—through `response_model.model_validate(parsed)` before returning (`arena:371` bypass deleted).
   - On `ValidationError` retry, call `self._build_response_format(response_model)` instead of hardcoding `"json_schema"`, **and whenever the retry executes in `"json_object"` mode (including mid-call fallback), inject `response_model.model_json_schema()` into `retry_messages[-1]`** (`Carbon A3`).
   - Count `finish_reason == "length"` (`length_truncation_count`) across both initial and retry calls plus `validation_retry_count` in `token_snapshot()`. Verify offline replay tests before Spark #1 launch.

2. **`Demo D9` (`scripts/run_wave11_d9_fast_extractor_and_u_grpo.py` — `Neon Δ3`, `Fable A3/A6`, `Carbon A4`, `Astra #2`):**
   - **Part A:** Retain sealed run (`96.67%` Suite A, `100.0%` JSON validity); report `2,104 ms/case` honestly as **"latency gate not met (`2,104 ms` vs `1,500 ms` gate; `4.6x` faster than `30B`)"** (`Fable A6`).
   - **Part B (`Protocol v2`, pre-committed in `docs/research/d9b_amendment_v1.json`):**
     - **Bundled Protocol v2 Knobs (zero single-knob causal attribution claimed):** `FALSE_ABSTAIN_REWARD = WRONG_DETERMINATE_REWARD = -1.00`, `scale_rewards="none"`, `loss_type="dr_grpo"`, `grad_accum=8` (`prompt_groups_seen = run_steps * grad_accum` fixing `:2672`), `r=16`, `lr=5e-6`, `beta=0.05`, `grpo_steps=60`, `eval_ckpts = range(10, 61, 10)`.
     - **Fail-Closed `trl` Startup Check:** Assert `"scale_rewards"` in `GRPOConfig` and verify on a synthetic unequal-spread group that advantages are mean-centered without std-division.
     - **Parity Fix:** Cast `peft_fp32 = peft_fp32.float()`, probe `>=8` prompts/positions, assert `math.isfinite(diff) and diff < 1e-4` (`fp32`-vs-`fp32`), and downgrade `:2636` `bf16` reload diff to ungated telemetry (`Neon Δ3`).
     - **`best_dev` & Resume Hygiene:** Delete `:2604–2605` fallback that overwrote missing `best_dev_dir` with `final_step`; include `step=0` SFT baseline (`fell_back_to_step0_sft: bool` when no `step > 0` clears `det_em >= sft_det_em - 0.03`); bind resume keys + `_ckpt_stage1_w11.json` + `merged_sft_dir` SHA-256 + `(seed, steps, lr, beta, r, grad_accum, scale_rewards, loss_type, false_abstain_reward, script_sha256)`.
     - **One-Shot Out-of-Loop Holdout Evaluator:** Retire `seed_42` as DEV-diagnostic; train **3 fresh seeds `(137, 2026, 31415)`** with **DEV-only** scoring inside the loop (moving `EVAL_VIRGIN_V2`, `EVAL_WAVE6_LOOK2`, and `D1_k17/k5` completely out of the seed loop!); freeze `merged_sft_dir` + all distinct GRPO adapter SHA-256s in `selection_frozen.json`; execute **one single digest-locked evaluator pass** (`final_step@60` 3-seed mean PRIMARY, reporting `(137, 2026)` 2-seed mean alongside; `best_dev` descriptive secondary; pre-declared paired bootstrap + Holm test; null reported as *"K4 stands under Protocol v2"*; any subsequent config change requires minting `EVAL_VIRGIN_V3`).

3. **`Demo D10` (`scripts/run_wave11_d10_5m_real_cutover_and_70b_scale.py` — `Astra #4`, `Neon Δ4`, `Fable A1/A6`, `Carbon A5`):**
   - **Part A (`Spark #1`):** Pre-freeze the 50-query Flat-oracle sample, recall floor, and tie convention; assert `MemAvailable >= 48 GiB` pre-run, `unmasked_leaks > 0`, and `10` distinct admissible IDs per query (`fill@10 = 100%`); bind `report_receipt.json` hashes directly to `Path(args.part_a_json)` and `Path(args.part_b_json)` bytes.
   - **Part B (`Spark #2`, serialized after `D9-B`):** Add **`Llama-3.3-70B (4-bit NF4) Reader + Chelated (Nemotron-30B Extractor)`** routed through the exact `chelated_decisions` (`CONTESTED_ABSTAIN -> "ABSTAIN"`) short-circuit AND `qwen_judge_scores`, including `Nemotron-30B` extraction latency in total Chelated latency and reporting gate-decided vs reader-decided cases separately (`Fable A1`, `Neon Δ4`). Frame strictly as **"reader scale varied (`1.5B` / `7B` / `70B 4-bit NF4`) under a fixed `30B`-extractor governance pipeline vs bare `7B` / `30B` / `70B` (`k=5` paired primary, `70B k=17` full-recall sensitivity)"**—never *"fully crossed"* or *"governance beats scale"*. On GB10 `bnb` load failure, emit a schema-valid hashed `status: "NOT_RUN"` JSON so `assemble-manifest` succeeds cleanly and drop `70B` claims from `v6.0` prose.

---

## 3. Binding `Wave 14A` (`v7.0` NDA-Gated): 5-Module `/formal` Lean 4 & Conformance Spec

1. **Toolchain, Cache, & Axiom Floor (`Fable A4`, `Astra #3`, `Neon Δ6`):**
   - NDA-gated `/formal` (`0/56` Non-NDA public files untouched); pinned `lean-toolchain`, `mathlib4` commit (with pre-populated `.olean` cache; missing cache reports `NOT_RUN`, never `PASS`), `lean4export`, and `nanoda_bin`.
   - Hash `Formal/Statement.lean` before writing tactic proofs; ban `sorryAx` and `Lean.ofReduceBool`/`native_decide`; enforce `#print axioms` allowlist (`propext`, `Classical.choice`, `Quot.sound`) + negative-control test; include non-vacuous satisfiability witnesses (`∃ ...`) for every theorem.

2. **Exact 5-Module Specification (`Formal/Statement.lean` + 5 Proof Modules):**
   - **`Theorem1a_MonotoneIsometry.lean`:** For $0 < d_0 \le d_1 \le \dots \le d_k$, $W_t^\top W_t = I_{d_{t-1}}$, $\alpha_t > 0$, $B = (\prod \alpha_t) W_k \cdots W_1$ satisfies $\forall x \in \text{EuclideanSpace }\mathbb{R}\text{ (Fin }d_0\text{)},\ \|B x\|_2 = (\prod \alpha_t)\|x\|_2$ (extremal stretch ratio $\kappa_2(B)=1$ for $d_0 > 0$), plus `2 -> 3 -> 2` rank-collapse counterexample. State explicitly that runtime `normalize((x - \mu) Q)` adds affine centering + spherical projection outside $B$.
   - **`Theorem1b_ProcrustesOptimality.lean` (`Neon Δ1`, `Fable A4`, `Carbon A2`, `Astra #3`):** Let $A \in \mathbb{R}^{n \times d_1}$ (centered new anchors) and $B \in \mathbb{R}^{n \times d_0}$ (paired centered legacy anchors, $d_0 \le d_1$), with thin-SVD witness $A^\top B = U \Sigma V^\top$ ($\Sigma \ge 0$ diagonal, $U \in \operatorname{St}(d_1, d_0)$, $V \in O(d_0)$) and $W^* = U V^\top \in \operatorname{St}(d_1, d_0)$, using explicit entrywise sum-of-squares Frobenius norm:
     1. **Inner-Product Alignment Optimality:** $\forall W \in \operatorname{St}(d_1, d_0),\ \operatorname{tr}(W^\top A^\top B) \le \operatorname{tr}((W^*)^\top A^\top B) = \operatorname{tr}(\Sigma)$.
     2. **Unconditional Upward Lifting Frobenius Optimality:** $\forall W \in \operatorname{St}(d_1, d_0),\ \|B (W^*)^\top - A\|_F^2 \le \|B W^\top - A\|_F^2$.
     3. **Isotropic Compression Corollary & Rational $3$-$4$-$5$ Counterexample:** Downward $\|A W^* - B\|_F^2 \le \|A W - B\|_F^2$ holds under $A^\top A = c I_{d_1}$, and is **refuted** in general at $d_1=2 \to d_0=1$ by exact rational $A = \operatorname{diag}(3, 4)$, $B = (1, 1)^\top$ ($W^* = (3/5, 4/5)^\top$ yields squared residual $137/25 = 5.48 > 5 = \|A e_1 - B\|_F^2$, zero `Real.sqrt` required!). Disclose in-sample plain-branch scope and log empirical Procrustes residual beside `captured_energy_ratio`.
   - **`Theorem2_AdjointZeroMuPerturbation.lean` (`Neon Δ2`, `Astra #3`):** Adjoint identity $\langle q, W d \rangle = \langle W^\top q, d \rangle$; general squared projector identity $\|W^\top q\|_2^2 = \|q\|_2^2 - \|(I - W W^\top) q\|_2^2$ (specializing to $\|W^\top q\|_2 = \sqrt{1 - \eta(q)}$ for unit $\|q\|_2 = 1$, with nonzero projected norm $\|W^\top q\|_2 > 0$ required for normalization + orthogonal-query fixture); exact `8 B/doc` `ZeroMu` centered-cosine identity given unit-norm centered query $\hat{q}_c$ and stored `4 B/doc` centered norm $\|d - \mu_d\|_2$ (unit-norm $d$ not required); and per-doc forward error bound over $\gamma_d \ge \gamma_{\min} > \delta_\gamma$ covering $\varepsilon_{\text{quant}}$, `fp32` norm rounding $\delta_\gamma$, and CSLS terms (+ empirical `% BEIR queries clearing \Delta_cert`).
   - **`Theorem3_EK5_CyclicWFS.lean` (`Astra #3`, `Fable A4`, `Neon Δ6`, `Carbon A6`):** Over `Finset (Fin n)` finite Knaster-Tarski iteration, prove $\Gamma_P$ antimonotone and $\Gamma_P^2$ monotone on **arbitrary finite cyclic graphs** (`T = lfp(\Gamma_P^2)`, `F = V \setminus \Gamma_P(T)`, `U` residual; `lfp` $\ne$ unique fixpoint when cyclic `REFUTE` rings exist), **using `EK5`'s exact target-domain authority predicate `eff_auth_for_domain(u, domain(w)) >= eff_auth(w)` (`ek5:1180`)** for Stratum 1 $\to$ Stratum 2 non-interference. **`ek5_ref_cli` Soundness Binding:** `ek5_ref_cli` must execute a computable function carrying a kernel-checked equality theorem to the `lfp(\Gamma_P^2)` spec (`Statement.lean` delineates Stratum 1–2 deductive proof coverage vs full Stratum 0–3 `ek5_ref_cli` conformance; incremental-with-deletions proof is timeboxed—if missed, omit rather than weaken and label *"conformance-tested across 1,000 deletion/cycle sequences, unproven"*).
   - **`Theorem4_AnchoredClippedEMA.lean`:** Inductive projection bound $\|\mu_q^{(t)} - \mu_0\|_2 \le \rho$ + angular bound under $\|q - \mu_0\|_2 \ge \gamma_q > \rho$, paired with 100-query burst test (`M6`; `PM-6` gates freeze $\mu_q = \mu_0$; mTLS authority-channel binding documented as operational governance outside the EMA theorem).

---

## 4. Binding `Wave 14B` (`v7.0`): `Demo D11` (`PM-6`) — Inference-Time Takedown Suppression & Cross-Gen Carry-Forward

1. **Terminology, Scope, & Legal Citation Contract:**
   - Strictly **"Inference-Time Takedown Suppression & Cross-Generation Governance Carry-Forward"** (hosted-gateway scope; zero *"unlearning"* or *"Parametric Recall Neutering"* claims; `CITES_AUDIT` disclosed as auditable suppression, not physical erasure; public-domain canaries distinguished from legal conclusions about specific copyrighted works; primary-docket verification required for *In re OpenAI* / *GEMA*; `R3–R4` framed as Enterprise Privilege Clawback, Embargo, & Contractual Opt-Out Suppression).

2. **Entity/Series-Disjoint Split, Pre-Committed Salt, & Zero-Leak Calibration (`Astra #2`, `Fable A3`, `Neon Δ5`, `Carbon A5`):**
   - Commit split salt **before** Phase-0 yields are observed; confirm memorization (`ROUGE-L >= 0.80` or block recall `>= 0.75`) separately per generator (`Nemotron-30B` and `Qwen2.5-7B`; pre-declared reduced-yield split if either generator yields `<110` works).
   - Assign **one global entity/author/series-disjoint split** (`50` `DEV` work-families + `50` `Q_benign` siblings; `60` `TEST` work-families + `60` `Q_benign` siblings) across generators, languages, and derivative echoes.
   - **Strict Zero-Test-Leakage Rule:** Exclude **all `TEST` work-families, their `Q_benign` siblings, and all `480` restricted vault nodes** when fitting Stiefel bridge `W`, global/per-language centers `μ_en` / `μ_es` / `μ_de` (`0` `TEST` queries!), `τ_gov` / `τ_arm`, and the frozen `4,096`-vector `r_q` reference pool (hash-locked pre-calibration; for native `4096-d` oracle arm `(d)`, compute an independent `4096-d` reference pool & sidecars and log re-embedding GPU cost separately). Cluster-bootstrap all CIs **by held-out work-family (`60` clusters)**.

3. **Closed-Loop Verdict-Ack Client-Byte Release (`Astra #1`, `Fable A2`, `Carbon A1`, `Neon Δ5`):**
   - **Replace open-loop `L_p99` safety assumption with a deterministic closed-loop release state machine:** Track raw UTF-8 byte offsets through tokenization and canonicalization; advance `release_cursor` **only** after every scheduled window covering those bytes has closed and returned `CLEAR` across all active tiers (buffering incomplete multi-byte UTF-8 codepoints, checking short `< W_dec` final suffixes before EOS flush, discarding pending bytes on error/cancel, and **either holding back or rejecting `logprobs`, `tool_calls`, and `reasoning_content` SSE fields**!). `L_p99` sizes buffer telemetry only; verify via an injected `500 ms > L_p99` Spark #3 delay test across every stride offset (`0` flagged bytes emitted).
   - **Two Separated Transport Profiles:**
     - **`Profile A` (Strict Compliance Mode — Primary Safety Gate):** Unconditional Tier 2A (`s_2A=1`) + Tier 2B (`W=64, s_2B=16`, Spark #3 `Qwen3-8B`) verdict-ack gate before any byte leaves the server. Measure `4,096`-vector `r_q` GEMV, `480`-node restricted GEMV, encoder RPC, and first-content TTFT separately from keepalives. Report `0/60` per-work flagged-window client-byte leak (`<= 4.9%` one-sided 95% Clopper-Pearson) separately from detector recall.
     - **`Profile B` (Armed + Async-Scored Low-Latency SLA Mode):** Synchronous Tier-2A verdict-ack (`s_2A=1`, `0` literal 10-gram byte leak) on all traffic + `DEV`-calibrated soft prompt arming (`τ_arm` escalating to synchronous Tier-2B) + **continuous async Tier-2B scoring (`s_2B=16`) on unarmed streams with mid-stream abort** (`Fable A2` + `Carbon A1`), bounding unarmed `R3` exposure to `max(0, W + s_2B - 2 + L_2B_actual - H_0)` tokens.

4. **Matched-FPR Single-Factor Baseline & Ablation Matrix (`Astra #4`, `Fable A5`, `Carbon A5`):**
   - Calibrate all thresholded arms on `DEV` at **matched empirical benign `FPR`** and report full `R3-vs-R5` ROC curves + paired work-cluster AUC differences within each generator and phase:
     1. `Arm 1`: Bare LLM (`100%` Phase-0 selection ceiling on `R1` by construction).
     2. `Arm 2`: Hash-frozen System-Prompt Guardrail.
     3. `Arm 3`: Tier-2A Canonicalized N-Gram/LCS Filter Only *(pre-registered rule: if Arm 3 matches Full Chelated on `R1/R2`, narrow `Chelated`'s `R1/R2` claim strictly to `R3/R4` + cross-gen carry-forward)*.
     4. `Arm 4`: Standalone **Full Gate-1-Only** (`EK5` admission mask + prefix screen, no Gate 2).
     5. `Arm 5`: Standalone **Full Gate-2-Only** (`ZeroMu + CSLS` streaming gate, no Gate-1 `EK5` RAG/derivative mask — isolating single-factor gate contributions cleanly!).
     6. `Arm 6`: **Gate-1 + Gate-2 Raw Cosine** (zero-LLM-cost offline rescoring disabling `ZeroMu + CSLS`, isolating spherical calibration!).
     7. `Arm 7`: Timeboxed White-Box CAA / Contrastive Logit Suppression on `Qwen2.5-7B` (`DEV`-tuned multiplier, compared strictly against `Qwen2.5-7B` arms as an *"untuned/lightweight literature reference"*).
     8. `Arm 8`: Linear Bridge (`CAR` / `Drift-Adapter`) + Gate 2 (Phase 3 cross-gen comparison).
     9. `Arm 9`: **Full `Chelated` Dual-Gate (`EK5 v3` + `ZeroMu + CSLS`, frozen `\mu_q = \mu_0`)**, emitting integer-only `copilot_receipt_v2` (`0/56` Non-NDA clean).
   - Plus **Phase-3 4-Way Cross-Lingual `R3-ES/DE` Falsifiable Ablation** (`monolingual MiniLM-384`, `4096->384 μ_en`, `4096->384 unlabeled μ_lang`, `native 4096-d oracle`) and **Post-Freeze Budgeted Adaptive Red-Team (`R2b`, reporting per-work any-of-$K$ breach rate)**.

---

## 5. Retained Material Dissent & Irreducible Limitations Ledger
1. **`PM-1b` Status Wording (`Astra` vs `Carbon/Neon/Fable`):** Keep `assignment_gates_pass: false` in `bundle_registry_spec.json` (`Astra`) AND record status text verbatim as *"SCOPED/OPEN — clean-chat difference not detectable (`95.00%` vs `95.83%`, `N=120`, underpowered; non-inferiority not established, no pre-registered margin); `2.42x` token compaction vs flat; `+29.49 pp` on `LongMemEval KU` drift"*.
2. **Verifier Timing (`Fable/Astra` vs `Carbon/Neon`):** Resolved in favor of strictest bar — ship `verify_all_offline.py --fast` (`<45s` target, printing `formal_tier: "SKIPPED_USE_FULL_FORMAL"`) at **Wave 13 (`v6.0` seal)** and add `--full-formal` at **Wave 15 (`v7.0` seal)**.
3. **Irreducible Technical & Statistical Boundaries:**
   - `/formal` certifies exact real-arithmetic properties of the Lean model + sampled/executable discrete `EK5` semantics; IEEE-754 `fp32`/`fp16` rounding, HNSW approximate recall, and upstream LLM claim-extraction errors (`M6`) are bounded empirically, not eliminated deductively.
   - `PM-6` provides hosted-gateway inference-time suppression (not open-weight `.safetensors` erasure), `CITES_AUDIT` is auditable suppression (not physical erasure), `"0 client bytes"` applies strictly to detector-flagged windows (never claiming `0%` semantic miss rate against unbounded allegory), and `60` held-out work-families bound `0/60` false refusals at `<= 4.9%` (`95%` one-sided Clopper-Pearson).
   - `D9-B` `EVAL_VIRGIN_V2` is virgin for GRPO only (Base/SFT previously scored), and `Protocol v2` ships 7 bundled fixes without single-knob attribution.
   - Final exit from `PRE-RELEASE` requires independent **human grading of `PM-4b`** (`pm4_human_audit_100_blind_sheet.csv`).
