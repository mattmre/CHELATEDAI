---
panel_type: panel-architecture-planning
artifact: docs/research/council-logs/wave14-r0-master-roadmap-and-pm6-formal-spec.md
run_date: "2026-11-24"
run_id: wave14-r1-carbon4-high
challenge_depth: round-1-of-3
active_experts:
  - gemini-4-carbon (Chief Architecture, Release Governance, & Enterprise/Frontier Product Lead)
persist_version: "1.1.0"
---

# Round 1 Review — Wave Sequencing, Formal Diligence, & `PM-6` Architecture

**Scope & Identity:** Static audit of `wave14-r0-master-roadmap-and-pm6-formal-spec.md` (`HEAD 4fae0b77`) against `bundle_registry_spec.json`, `ek5_v3_stratified.py`, `spherical_migration_bridge.py`, `chelated_graphiti_arena_v1.py`, `demo_workbench/compliance_copilot_engine.py`, `scripts/verify_non_nda_ip_clean.py`, and `D6–D10` runners. Reviewer: `gemini-4-carbon` (`high`), Round 1/3; output: `docs/research/council-logs/wave14-r1-carbon4-high.md` (no live GPU/SSH/Docker runs).

## Actionable Findings (Ordered by Consequence)

### 1. [Q1 — BLOCKING] `D7` Retry Bug, `D9-B` Post-Hoc Seed Bias, & `v6.0` Provenance Split
- **Source & Evidence:**
  - `chelated_graphiti_arena_v1.py:364-404`: After switching `structured_output_mode = "json_object"`, the `ValidationError` retry (`lines 397-403`) hardcodes `"type": "json_schema"`, re-triggering HTTP 400 on `:8891` and Graphiti `ARM_FAILURE` (`run_wave11_d7_live_zep_graphiti_arena.py:620`).
  - `scripts/run_wave11_d9_fast_extractor_and_u_grpo.py:1919, 2080-2109, 2487, 2542-2566, 3103`: `_verify_fp32_merge_and_unload_parity` already runs in CPU `fp32` (indicating Spark #2 ran an unsynced copy), while the repo still hardcodes `lr=2e-5, beta=0.02, grpo_steps=160`, checks `DEV` at only 4 quartiles (`line 2487`), and omits `determinate_em >= sft_det_em - 0.03` in `DevTrajectoryCallback` (`line 2546`). Choosing `60` steps / `lr=5e-6` / `beta=0.06` after inspecting `seed_42` makes `seed_42` a tuning run. Part A (`2,104 ms/case`, `R0:38`) also misses `ms_per_case_below_1500` (`line 1919`).
  - `docs/research/bundle_registry_spec.json:5-42`: `v5.0` required two provenance carve-outs (`scp` overlay and pre-commit runs). With `D6` (`0578d56b`) and `D8` (`3fea611f`) sealed, landing `v7.0` code before `D7/D9/D10` fractures `v6.0` `git_sha` lineage.
- **Impact:** Avoidable `D7` `ARM_FAILURE`s, diligence challenge on `D9-B` stopping selection, and dirty `v6.0` provenance.
- **Correction:** Seal **`v6.0` (`Waves 11–13`, `45/45` bundles) before merging `v7.0`**. Patch `D7` (`self._build_response_format(response_model)` on retry); wire `D9-B` (`lr=5e-6, beta=0.06, eval_steps=10`, non-collapse guard) with **3 fresh confirmatory seeds** `(137, 2026, 31415)` (logging `seed_42` as DEV-calibration and disclosing Part A's `2,104 ms` variance without altering the gate). Sync one clean `v6.0-rc1` SHA (`git checkout --detach`, zero `scp` overlays): run `D7` on **Spark #1** (`:8891` pinned), offload `D10-A` (`9.47M` FAISS cutover) over 200GbE QSFP to idle **Spark #3**, run `D9-B` then `D10-B` (`70B` NF4) **sequentially** on **Spark #2** (`128 GiB` budget with `:8890`), and develop `/formal` + `PM-6` concurrently on **Local Windows** in an isolated `feat/v7-formal-pm6` worktree.
- **Verification:** Clean `git status` across nodes; `D7` `arm_failure == 0`; `D9-B` 3 confirmatory seeds `<1e-4` merge parity; `bundle_registry_spec_v6_0` passes `45/45` bundles and `0/56` Non-NDA scan.

### 2. [Q3 — HIGH] `PM-6` Holdback TTFT Tax, `R3`/`R5` Overlap, Cross-Lingual Baseline, & `D8` Receipt Schema
- **Source & Evidence:**
  - `demo_workbench/compliance_copilot_engine.py:1199-1308` & `scripts/verify_non_nda_ip_clean.py:24-81`: `D8` (`copilot_receipt_v1`) rejects floats (`_reject_floats`) and fails closed (`500`) if `non_nda` payloads hit any of `56` patterns (`ek5`, `zeromu`, `csls`, `stiefel`, `qwen`, `minilm`).
  - `R0:74-77`: Unconditional 32-token SSE holdback adds `~500–950 ms` perceived TTFT (`32 * TPOT`) to all benign streams (inference). Scoring `ZeroMu + CSLS` against *"restricted synopsis nodes"* conflates `R3` (plot paraphrase) with `R5` (`Q_benign` same-entity QA) and overfits at `N_dev=20`.
- **Impact:** Broken `D8`/`D11` receipt compatibility, violated `<300 ms` vLLM TTFT SLOs, and elevated `R5` false refusals.
- **Correction:** (1) Activate 32-token holdback + Tier 2B **conditionally** only when Gate 1 (restricted-entity/ANN proximity within `2\Delta_{\text{cert}}`) or Tier 2A (`<0.3 ms` CPU shingle/canonicalizer) triggers (`0` holdback on benign traffic), and evaluate unseen obfuscations beyond Tier 2A regexes; (2) Split expression-level sequence/window matching (copyright `R1–R3`) from `EK5` `DERIVES_CURRENT` slot suppression (privilege clawback `R4`), expand `DEV` to `>=50` disjoint works, and report `ROC-AUC` + `TPR @ FPR <= {1%, 5%}`; (3) Add a **monolingual `MiniLM-384` direct cross-lingual baseline** in Phase 3 proving `Qwen3-8B (4096-d) -> Stiefel -> MiniLM (384-d)` maps Spanish/German queries into the English `384-d` vault; (4) Use integer-only `copilot_receipt_v2` (`score_ppm`/`threshold_ppm` `x10^6`) with math/model identifiers isolated inside NDA `payload["internal"]`.
- **Verification:** `D8` `canonical_json()`, `verify_ledger()`, and `assert_non_nda_clean()` pass on `D11` receipts; benign TTFT overhead `<2 ms` p95; `R5` false-refusal `<=2%`.

### 3. [Q2 — HIGH] `/formal` DAG-vs-Cycle Mismatch, Mathlib4 Tarpit, & `M6–M8` Credibility Defenses
- **Source & Evidence:** `R0:55-56` scopes `Theorem 4` to *DAGs*, whereas `ek5_v3_stratified.py:111-237, 380-405, 801-850` resolves cyclic `REFUTE` rings (`U`), Tarjan `D_V` positive-SCCs (`pos_cyc`), and `SUPERSEDE` cycles. `M3` (Marchenko-Pastur + Davis-Kahan) is a multi-week Mathlib4 tarpit, and `/formal` identifiers match `verify_non_nda_ip_clean.py:25-51`.
- **Impact:** Reviewers dismiss DAG-only proofs as bypassing cyclic bugs; `M3` stalls `v7.0`; `/formal` leaks IP pre-NDA.
- **Correction:** Prove `Theorems 1–3` plus **cyclic finite-graph alternating-fixpoint WFS (`Theorem 4`)** in zero-`sorry` Lean 4 (`<90s` cached build), backed by `1,000` cyclic/adversarial graphs in `ek5_ref_cli` vs Python (`M4`). **Cut `M3` from Lean 4** (keep in LaTeX/Python). Add operational defenses: **`M6` (Extraction GIGO)** — bind `authority_tier` to authenticated channel/mTLS identity (never LLM-inferred) and pair `Statement.lean` with prose-extraction recall budgets; **`M7` (IP Boundary)** — gate `/formal` strictly under NDA; **`M8` (2-Min UX)** — ship pre-cached `.olean` Docker (`<30s` `nanoda`/`lake`) plus CPU `verify_all_offline.py` (`<45s`).
- **Verification:** `#print axioms` shows `0` `sorryAx`/custom axioms; `1,000/1,000` cyclic parity passes; Non-NDA surfaces stay `0/56`.

### 4. [Q4 — MEDIUM] Portfolio Closure: `verify_all_offline.py` Capstone, `PM-1b` Framing, & `PM-4b` Gate
- **Source & Evidence:** `bundle_registry_spec.json:3, 352-367`: `D3` already proves multi-agent poisoning quarantine (`0/80` ASR); `PM-1b` (`95.00%` vs `95.83%` `NoEK5` on clean `Virgin-120` at `2.42x` compaction) reflects zero authority conflicts in casual chat; `release_status` stays `PRE-RELEASE` pending human `PM-4b`.
- **Impact:** Re-tuning `PM-1b` for clean-chat accuracy burns holdouts; extra poisoning waves duplicate `D3`; pending `PM-4b` blocks release promotion.
- **Correction:** Cut `Wave 15+` sprawl and duplicate multi-agent waves. Frame `PM-1b` as a **Clean Non-Inferiority + `2.42x` Compaction + `+29.5 pp` (`p=0.00025`) `KU` Drift Gate** alongside `D7`. Ship **`scripts/verify_all_offline.py`** (`<45s` CPU check of all `v6.0`/`v7.0` bundle hashes, Ed25519 chains, `0/56` Non-NDA scans, and `ek5_ref_cli` parity) as the `v7.0` capstone alongside human `PM-4b`.
- **Verification:** `python scripts/verify_all_offline.py` exits `0` in `<45s` offline; human `PM-4b` kappa sealed.

## Material Unknowns & Required Independent Acceptance
- **Missing Runtime Proof:** Live `D7`, `D9-B` (`60`-step), `D10-A/B`, `PM-6` gateway p99 latency, and `Qwen3-8B -> MiniLM-384` cross-lingual `ROC-AUC` remain unexecuted.
- **Independent Human Gate:** Human grading of `pm4_human_audit_100_blind_sheet.csv` (`PM-4b`) is required to exit `PRE-RELEASE` (never open `*sealed_key*` or `*blind-followup*`).
