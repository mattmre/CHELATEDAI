# Round-2 Converged Master Plan (`wave14-r2-synthesis-draft-plan` — Input to Final Round 3 Binding Audit)

- **Artifact ID:** `wave14-r2-synthesis-draft-plan`
- **Repository:** `D:\GITHUB\CHELATEDAI-STAGE0-NIGHT1` (`HEAD 4fae0b77c6ae59213e24a12e89f5dd7b28405ca8`)
- **Date:** `2026-11-24`
- **Synthesized From:** All 4 Round-1 memos (`wave14-r1-{fable52,astra62,neon4,carbon4}-high.md`), all 4 Round-2 cross-examination memos (`wave14-r2-{fable52,astra62,neon4,carbon4}-high.md`), plus live read-only hardware verification of Spark #1/#2/#3 RAM and `trl 1.18.2` signatures.

---

## 0. Newly Verified Ground-Truth Facts (Resolving Round-2 `[U]` Unknowns)
1. **Spark #2 `trl` & `bitsandbytes` (`mattmre@10.0.0.194`, `.venv-rlvr`):**
   - `trl == 1.18.2`, `bitsandbytes == 0.50.2`.
   - `inspect.signature(GRPOConfig.__init__)` exposes **`scale_rewards: str = 'group'`** (`"none"` disables std-scaling), **`loss_type: str = 'dapo'`** (`"dr_grpo"` supported), and **`beta: float = 0.0`**.
   - Available unified memory on Spark #2 (with `:8890` resident): **`76 GiB` available** (`60 GiB` free + `17 GiB` cache).
2. **Spark #1 vs Spark #3 Available Unified Memory:**
   - **Spark #1 (`10.0.0.49`):** **`52 GiB` available RAM**, **`383 GiB` free NVMe** (`8.84M` MSMARCO shards already local).
   - **Spark #3 (`10.0.0.24`):** **`37 GiB` available RAM** (`84 GiB` used by resident containers).
   - **Binding Fleet Consequence:** Confirming Fable `C6` & Astra `C6`, offloading `D10-A` (`44 GiB` RSS gate) to Spark #3 (`37 GiB` available) is **unsafe and rejected**. `D10-A` runs on **Spark #1 (`52 GiB` available, shards already local)** right after `D7`; Spark #3 (`37 GiB` available) is reserved for `Wave 14B` isolated `Qwen3-Embedding-8B` (`~16 GiB`) serving.
3. **`ek5_v3_stratified.py:1134–1189` Phase C/D Inspection:**
   - Phase C recomputes effective authority (`eff_auth`) over `D_V` (`COPY_OF` / `DERIVES_CURRENT`) *before* Phase D filters `REFUTE` / `SUPERSEDE` edges by `eff_auth[u] >= eff_auth[w]`, confirming unidirectional Stratum 1 $\to$ Stratum 2 causal precedence when `D_V` has no reverse dependence on Stratum 2 truth values.

---

## 1. Locked Release Architecture & Ordered Wave Sequence (`Waves 11–15`)

1. **Hard Release & Provenance Boundary (`v6.0` Before `v7.0`):**
   - Complete and seal **`v6.0` (`Waves 11–13`, `45/45` bundles)** from a single clean committed revision (`v6.0-rc1`) synced across Spark #1 and Spark #2 (`0` untracked `scp` overlays) BEFORE merging any `v7.0` (`/formal` or `D11/PM-6`) files into the main branch.
   - `release_status` in `bundle_registry_spec.json` remains `PRE-RELEASE / HUMAN AUDIT PENDING` until human `PM-4b` (`pm4_human_audit_100_blind_sheet.csv`) is graded by a human (`0` agent self-grading, `0` reads of `*sealed_key*` / `*blind-followup*`).

2. **Numbered Wave Execution Order:**
   - **Wave 11 (`v6.0-rc1` Code Repairs, Pre-Registration Commit, & Live Spark #1/#2 Completion):**
     1. Commit `D7`, `D9-B`, and `D10-A/B` patches + `docs/research/d9b_amendment_v1.json` on branch `fix/stage0-metric-lineage-and-m1-ladder` (`v6.0-rc1` SHA) and sync cleanly to Spark #1 and Spark #2.
     2. **Spark #1 (serial):** Run patched `Demo D7` (`Suite A N=60` + `Suite B N=20`) $\to$ patched `Demo D10 Part A` (`8.84M` standalone + `9.47M` 7-BEIR cutover & `EK5` bitset proof).
     3. **Spark #2 (serial, concurrent with Spark #1):** Run `Demo D9 Part B` (`Protocol v2`, seeds `137, 2026, 31415`, DEV-only training $\to$ `selection_frozen.json` $\to$ single out-of-loop `EVAL_VIRGIN_V2` pass + `assemble-manifest`) $\to$ `Demo D10 Part B` (`Llama-3.3-70B 4-bit NF4` crossed scale-vs-governance sweep).
     4. **Local Windows (concurrent in isolated `feat/v7-formal-pm6` worktree):** Build & verify `Wave 14A` (`/formal` Lean 4 + `ek5_ref_cli` + `nanoda_bin`) while Sparks #1 and #2 execute Wave 11.
   - **Wave 12 (`v6.0` Cross-Model Receipt Validation, Registry `45/45`, & Docs `v6.0`):**
     - Peer cross-validation of all 5 sealed `Part VII` receipts (`D6–D10`), `verified_registry.json` `v6.0` (`45/45`), `v6.0` Whitepaper/Decks + Non-NDA Teaser (`0/56` IP check), `PM-1b` retired as a stated clean-chat negative (`no detectable difference on clean chat [95.00% vs 95.83%, underpowered]; 2.42x compaction; +29.49 pp on LongMemEval KU`), and final judge sign-off.
   - **Wave 13 (`v6.0` Gates, Push, & Vault Seal):**
     - Pass `ruff`, `pytest`, `0/56` Non-NDA check, `build_whitepaper_assets.py --check` (`45/45`), push `v6.0`, and seal encrypted `.7z` vault addendum.
   - **Wave 14A (`v7.0` Track A — NDA-Gated `/formal` Lean 4 + Per-Mutation Conformance Gate):**
     - Land `/formal` (`Statement.lean` + `Theorems 1a, 1b, 2, 3, 4`), pass `lake build` (`0` `sorry`, `0` `native_decide`), `#print axioms` allowlist + negative control, `nanoda_bin` external kernel verification, and **`1,000`-sequence per-mutation 3-way `EK5` conformance (`Lean` vs `Python` vs `full-recompute oracle`)** BEFORE running `Demo D11`.
   - **Wave 14B (`v7.0` Track B — `Demo D11 / PM-6` Inference-Time Takedown Suppression & Cross-Gen Carry-Forward):**
     - Execute Phases `0–3` + `R2b` adaptive red-team across Sparks #1, #2, #3 (`Qwen3-8B` isolated on Spark #3), seal Bundle #46 (`wave14_d11_pm6_takedown_carryforward`).
   - **Wave 15 (`v7.0` Closeout — Two-Tier `verify_all_offline.py`, Peer Adjudication, & `v7.0` Vault Seal):**
     - Ship `<45s` `scripts/verify_all_offline.py --fast` and `--full-formal`, run `v7.0` peer/judge sign-off, seal `verified_registry.json` `v7.0` (`46/46` bundles), push, and seal `v7.0` vault archive (ready for human `PM-4b` promotion).

---

## 2. Exact `Wave 11` (`v6.0-rc1`) Code & Statistical Contracts

1. **`Demo D7` (`chelated_graphiti_arena_v1.py`):**
   - Validate **every** parsed dictionary—including repaired JSON (`n_rep > 0`)—via `response_model.model_validate(parsed)` before returning (`arena:371` bypass removed).
   - Use `self._build_response_format(response_model)` on validation retry (`arena:397`), never re-sending `"json_schema"` after locking `"json_object"`.
   - Record `length_truncation_count` (`finish_reason == "length"`) and `validation_retry_count` in `token_snapshot()`, and verify offline replay tests for both repaired-invalid JSON and `json_object`-locked retry before Spark #1 launch.

2. **`Demo D9` (`scripts/run_wave11_d9_fast_extractor_and_u_grpo.py`):**
   - **Part A:** Retain sealed run (`96.67%` Suite A, `2,104 ms/case` vs `1,500 ms` target gate disclosed transparently as a variance without altering the gate).
   - **Part B (`Protocol v2`, pre-registered in `docs/research/d9b_amendment_v1.json`):**
     - `FALSE_ABSTAIN_REWARD = WRONG_DETERMINATE_REWARD = -1.00`, `scale_rewards="none"`, `loss_type="dr_grpo"`, `grad_accum=8`, `r=16`, `lr=5e-6`, `beta=0.05`, `grpo_steps=60`, `eval_steps=10` (with `step=0` SFT baseline included in `best_dev` candidates; `[0.25, 0.45]` abstain band is an abort tripwire/telemetry diagnostic only, never an acceptance gate).
     - Startup fail-closed check: assert `"scale_rewards"` in `GRPOConfig` and verify on a synthetic unequal-spread group that advantages are NOT divided by group std.
     - Parity fix (`d9:2090-2106`): cast `peft_fp32 = peft_fp32.float()`, evaluate across `>=8` probe prompts/positions, assert `math.isfinite(diff) and diff < 1e-4` (`fp32`-vs-`fp32`), and record ungated `bf16` reload diff separately.
     - Key resume paths by `(seed, steps, lr, beta, scale_rewards, loss_type, false_abstain_reward)`.
     - **Out-of-Loop Holdout Protocol:** Retire `seed_42` as the aborted DEV-diagnostic run; train **3 fresh confirmatory seeds `(137, 2026, 31415)`** with **DEV-only** scoring inside the loop; freeze all 6 adapter SHA-256s in `selection_frozen.json`; execute **one single out-of-loop `EVAL_VIRGIN_V2` pass** (`final_step@60` PRIMARY; `best_dev` with `det_em >= sft_det_em - 0.03` or `"none_eligible"` descriptive secondary; `EVAL_VIRGIN_V2` labeled *"virgin for the GRPO arm"*). Pre-commit: if `GRPO - SFT` is null, report *"K4 stands under Protocol v2"* (significance is a hypothesis test, never a script-completion gate).

3. **`Demo D10` (`scripts/run_wave11_d10_5m_real_cutover_and_70b_scale.py`):**
   - **Part A (`Spark #1`):** Assert `unmasked_leaks > 0`, `all(len(top_ids) == 10)` (`fill@10 = 100%`), and sampled exact-oracle `admitted_recall@10` on the `9.47M` fenced query set; hash `Path(args.part_a_json)` and `Path(args.part_b_json)` bytes directly into `report_receipt.json`.
   - **Part B (`Spark #2`, serialized after `D9-B`):** Add **`Llama-3.3-70B (4-bit NF4) + Chelated`** (~5 lines reusing loaded `70B` + `chelated_ctx`) so the `{Scale x Governance}` design is fully crossed (`Bare 7B/30B/70B k=5,k=17` vs `1.5B/7B/70B + Chelated`), always label `"Llama-3.3-70B 4-bit NF4"`, log post-`release_native_heap()` memory headroom, and if GB10 4-bit load fails on one attempt, record `NOT_RUN` without blocking `v6.0`.

---

## 3. Locked `Wave 14A` (`v7.0` NDA-Gated): `/formal` Lean 4 + Conformance Contract

1. **IP & Build Hygiene:** NDA-gated directory `/formal` (`0/56` Non-NDA public surfaces untouched); pinned `lean-toolchain` + `mathlib4` commit + `nanoda_bin` revision; **`0` `sorryAx`**, **`0` `Lean.ofReduceBool` / `native_decide`**, explicit `#print axioms` allowlist (`propext`, `Classical.choice`, `Quot.sound`) + negative-control test that catches an injected `sorry`/`axiom`, and non-vacuous satisfiability witnesses (`∃ ...`) for every theorem hypothesis.
2. **Exact 4-Module Theorem Specification (`Formal/Statement.lean` ~65 lines):**
   - **`Theorem1a_MonotoneIsometry.lean`:** Over finite-dimensional Euclidean spaces (`EuclideanSpace ℝ (Fin d)`), for $d_0 \le d_1 \le \dots \le d_k$, $W_t^\top W_t = I_{d_{t-1}}$, and $\alpha_t > 0$, $B = (\prod \alpha_t) W_k \cdots W_1$ satisfies $\forall x,\ \|B x\|_2 = (\prod \alpha_t)\|x\|_2$ (extremal stretch ratio $\kappa_2(B) = 1$), plus formal `2 -> 3 -> 2` non-monotone rank-collapse counterexample. Explicitly note in `Statement.lean` that runtime `normalize((x - \mu) Q)` applies affine centering + spherical normalization outside the linear map $B$.
   - **`Theorem1b_ProcrustesOptimality.lean`:** Given thin-SVD witness $X^\top Y = U \Sigma V^\top$ ($\Sigma \ge 0$ diagonal, $U \in \operatorname{St}(d_1, d_0)$, $V \in O(d_0)$), prove via trace cyclicity + Cauchy-Schwarz:
     1. **Bidirectional Inner-Product Alignment Optimality:** $\forall W \in \operatorname{St}(d_1, d_0),\ \operatorname{tr}(W^\top X^\top Y) \le \operatorname{tr}((U V^\top)^\top X^\top Y) = \operatorname{tr}(\Sigma)$.
     2. **Lifting / Isotropic Frobenius Corollary:** $\|X_{\text{leg}} V U^\top - Y_{\text{new}}\|_F \le \|X_{\text{leg}} W^\top - Y_{\text{new}}\|_F$ when $d_0 \le d_1$ (or $X^\top X = I$), paired with a formal $2\times 2$ counterexample (`X = diag(1, 10), y = (1, 1)^T`) proving downward $\|X W - Y\|_F$ minimization is **false** for tall $W \in \operatorname{St}(d_1, d_0)$ ($d_1 > d_0$). Explicitly state in-sample plain-branch scope (`bridge:317-345`).
   - **`Theorem2_AdjointZeroMuPerturbation.lean`:** Adjoint read duality $\langle q, W d \rangle = \langle W^\top q, d \rangle$, within-shard positive-scale cosine rank invariance, cross-shard norm factor $\|W^\top q\|_2 = \sqrt{1 - \eta(q)}$, exact `8 B/doc` `ZeroMu` centered-cosine identity (given unit-norm $d$ or stored centered norm $\|d - \mu_d\|_2$), and per-doc forward error bound over $\gamma_d = \|d - \mu_d\|_2 \ge \gamma_{\min} > \delta_\gamma$ incorporating both $\varepsilon_{\text{quant}}$ and `fp32` norm rounding $\delta_\gamma$ + CSLS penalty terms (paired with empirical Python reporting of `% real BEIR queries clearing \Delta_cert`).
   - **`Theorem3_EK5_CyclicWFS.lean`:** Formalize Van Gelder's **antimonotone** stability operator $\Gamma_P$ and **monotone** $\Gamma_P^2$ on **arbitrary finite cyclic graphs** (`T = lfp(\Gamma_P^2)`, `F = V \setminus \Gamma_P(T)`, `U = V \setminus (T \cup F)`, explicitly distinguishing `lfp(\Gamma_P^2)` from unique fixpoint when cyclic `REFUTE` rings exist), Stratum-1 $D_V$ authority precedence non-interference over Stratum-2 edge qualification, and incremental dirty-cone recomputation equivalence **under both insertions and deletions**.
   - **`Theorem4_AnchoredClippedEMA.lean`:** Projection drift bound $\|\mu_q^{(t)} - \mu_0\|_2 \le \rho$ + angular stability bound $\arcsin(\rho / \gamma_q)$ when $\|q - \mu_0\|_2 \ge \gamma_q > \rho$, paired with a 100-query adversarial burst benchmark (`M6` defense; `PM-6` gates freeze $\mu_q = \mu_0$ statelessly).
   - **Cut from Lean 4:** `M3` (Wedin / Marchenko-Pastur) remains prose/LaTeX + empirical $n$-anchor sweep against **CV-tuned Ridge**.
   - **Per-Mutation 3-Way Differential Conformance (`M4`):** `1,000` multi-step mutation sequences (insertions, deletions, authority changes, `D_V` positive cycles, odd/even `REFUTE` cycles, imported `SUPERSEDE` cycles) verified after **every single mutation** across `(1)` compiled `lake exe ek5_ref_cli`, `(2)` Python `ek5_v3_stratified.py`, and `(3)` independent full-graph WFS oracle (`100%` bit-exact state parity).

---

## 4. Locked `Wave 14B` (`v7.0`): `Demo D11` (`PM-6`) — Inference-Time Takedown Suppression & Cross-Gen Carry-Forward

1. **Claim, Terminology & Citation Hygiene:**
   - Title/claim strictly **"Inference-Time Takedown Suppression & Cross-Generation Governance Carry-Forward"** (zero *"unlearning"* or *"Parametric Recall Neutering"* claims; explicit hosted-runtime scope boundary; explicit disclosure that `CITES_AUDIT` retention is governance suppression with auditability, not physical erasure).
   - Verify all case citations (*In re OpenAI*, *GEMA v. OpenAI*) by exact court, docket, date, and procedural posture from primary sources; frame `R1–R2` + verbatim translation as protected-expression suppression and `R3` plot/synopsis + `R4` echoes as **Enterprise Privilege Clawback, Embargo, & Contractual Derivative Suppression**.

2. **Phase 0 & Two-Class `DEV` Calibration (`0` Test Leakage, Cluster Bootstrap):**
   - Confirm memorization (`ROUGE-L >= 0.80` or block recall `>= 0.75`) **per generator under test** (`Nemotron-30B` primary; `Qwen2.5-7B` secondary, with pre-registered fallback if `7B` yields `<110` works: run `7B` at full confirmed yield with exact Clopper-Pearson CIs instead of loosening `ROUGE-L`).
   - Deterministic SHA-256 split into **`50` `DEV` restricted works + `50` paired same-entity `Q_benign` siblings (`100` `DEV` items)** and **`60` `TEST` restricted works + `60` paired `Q_benign` siblings (`120` core `TEST` works/pairs)**.
   - **Crucial CSLS Reference Fix (`Fable C2` + `Astra C2`):** Compute Tier-2B `ZeroMu + CSLS` against the **`480` restricted vault nodes** (`<0.2 ms` exact GEMV), where `r_d` comes from the frozen `1M`-corpus `8 B/doc` sidecar AND **`r_q` is computed against a frozen `4,096`-vector background reference sample** (NOT online over the `480` restricted nodes!), locking $\tau_{\text{gov}}$ on `DEV` at pre-registered `FPR(DEV Q_benign) <= 0.02`. Cluster-bootstrap all CIs **by work (`60` clusters)**.

3. **Exact Client-Byte Holdback Theorem & Dual-Profile Evaluation (`C2`):**
   - Enforce **`H >= W_dec + s + ceil(L_p99) - 2`** tokens before releasing UTF-8 bytes to the SSE client, holding all undecided bytes server-side through batching, short EOS flushes (`< W_dec`), timeouts, and cancellations.
   - Evaluate and report **two strictly separated profiles** (never blending Profile A safety with Profile B TTFT):
     - **`Profile A` (Strict Compliance Mode — Primary `PM-6` Safety Gate):** Unconditional `H = 64 + 16 + ceil(L_p99)` (`s = 16`, `W = 64`; `Qwen3-Embedding-8B` isolated on Spark #3). Verified on captured client byte streams across all boundary offsets, split encodings, short EOS flushes, and cancellations: **"0 client bytes emitted from any detector-flagged window (`0/N`, with one-sided 95% Clopper-Pearson bound); detector miss rate on `R1–R4` and `H`-token TTFT delay reported separately."**
     - **`Profile B` (Armed Low-Latency Gateway SLA Mode — Secondary Systems Point):** Universal Tier-2A holdback `H_0 = 8 + s + ceil(L_2A)` on all traffic (`0` literal 10-gram byte leak) + soft prompt arming (`entity mention` OR proximity `< \tau_{\text{arm}}` OR translate/summarize cue $\to$ escalates holdback to `H_1` without refusing), reporting benign TTFT, arming recall on `R3`/`R2b`, and bounded unarmed exposure (`<= H_1 - H_0` tokens).

4. **Crossed 8-Arm Matrix, 4-Way Phase-3 Cross-Lingual Ablation, `R2b` Red-Team, & `D8` Receipts:**
   - **Arms (matched generator & decoding budget):** `(1)` Bare LLM (`100%` selection ceiling on `R1`), `(2)` Hardened System-Prompt Guardrail, `(3)` Tier-2A Canonicalized N-Gram/LCS Only, `(4)` Gate-1 + Raw Vector Delete Only, `(5)` Gate-2 Raw Cosine Only (no `ZeroMu+CSLS`, no `EK5`), `(6)` Timeboxed White-Box CAA / Contrastive Logit Suppression on `Qwen2.5-7B` (*"untuned literature reference"*), `(7)` Linear Bridge (`CAR` / `Drift-Adapter`) + Gate 2 (Phase 3), `(8)` **Full `Chelated` Dual-Gate (`EK5 v3` + `ZeroMu + CSLS`, frozen `\mu_q = \mu_0`)**.
   - **Phase 3 Cross-Lingual `R3-ES/DE` vs `R5` 4-Way Ablation:** `(a)` monolingual `MiniLM-384`, `(b)` `4096->384` ($\mu_{\text{en}}$), `(c)` `4096->384` (unlabeled $\mu_{\text{lang}}$), `(d)` native `4096-d` `480`-node oracle (both nulls pre-registered).
   - **Post-Freeze Budgeted Adaptive Red-Team (`R2b`):** Fixed-budget interleaved-token, homoglyph/cipher, and entity-omitted multi-turn prompts run *after* freezing $\tau_{\text{gov}}$ and $\tau_{\text{arm}}$.
   - **Receipts:** Integer-only `copilot_receipt_v2` (`score_ppm`, `threshold_ppm`) passing `Demo D8` `verify_ledger()` and `0/56` `verify_non_nda_ip_clean.py`.

---

## 5. Two-Tier Reproducibility Verifier (`scripts/verify_all_offline.py`)
- **`--fast` (`< 45s` CPU target, cache state & CPU logged):** Verifies all `45/45` (`v6.0`) / `46/46` (`v7.0`) SHA-256 manifests & receipts, Ed25519 hash-chains, `0/56` Non-NDA public files, and recomputes headline CIs/p-values from committed summary/row data.
- **`--full-formal` (~2–4 min target):** Executes `lake build`, `#print axioms` allowlist + negative control, `nanoda_bin` external kernel verification, `1,000`-sequence per-mutation `EK5` differential conformance, and the 100-query `Anchored-Clipped EMA` burst test.
