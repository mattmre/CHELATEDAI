# Master Architecture & Diligence Proposal (`R0` Input to 3-Round 4-Model `$panel-of-experts` Council)

- **Artifact ID:** `wave14-r0-master-roadmap-and-pm6-formal-spec`
- **Repository:** `D:\GITHUB\CHELATEDAI-STAGE0-NIGHT1` (`branch: fix/stage0-metric-lineage-and-m1-ladder`, base HEAD `4fae0b77c6ae59213e24a12e89f5dd7b28405ca8`)
- **Date:** `2026-11-24`
- **Council Roster (3 Mandatory Cross-Model Rounds):**
  1. `claude-fable-5-2` (`effort: high`) — Skeptical Diligence, Formal Logic/Proof Rigor, Statistical & Holdout Integrity
  2. `gpt-6.2-astra` (`reasoningEffort: high`) — Principal Systems & Security Architecture, Streaming/ANN Physics, Mathlib/Kernel Feasibility
  3. `gemini-4-neon` (`thought_level: high`) — Post-Training RLVR (`U`-GRPO), Parametric Unlearning/Steering Baselines, Multi-Encoder Geometry
  4. `gemini-4-carbon` (`thought_level: high`) — Holistic Wave Sequencing, Frontier-Lab/Enterprise Use-Case Strategy, Provenance & Release Governance

---

## 1. Ground-Truth Baseline State (`Waves 1–11`)

### 1.1 Sealed Historical & Overnight Registry (`v5.0`: `40/40` Bundles)
- `docs/research/bundle_registry_spec.json` (`v5.0`) locks `40/40` cryptographically verified bundles (`Parts I–VI`), with byte-exact backup sealed at `archive/whitepaper-v5.0-pre-v6-backup/BACKUP_MANIFEST.json`.
- Documented council variances & open gates in `v5.0`:
  - `K1`: Wave-6 `D5` `50%` retraction `Recall@10 = 0.994186` (`< 0.995` floor) — **already closed in Wave-11 `D6` (`1.0000` across `0/10/50/90%` on 3 live DBs)**.
  - `K2`: `PM-1b` conversational assignment gate remains `SCOPED/OPEN` (`assignment_gates_pass=false`), and `34/101` binary contrasts exhibit discrete `1/N` bootstrap percentile ties.
  - `K4`: Wave-6 `D4` (`scripts/run_wave6_d4_trained_epistemic_grpo.py`) gains were carried by SFT (`GRPO - SFT` `p=0.45`, `EVAL false_commit_on_u = 0.875`).
  - `PM-4b`: Blind 100-item human audit sheet (`pm4_human_audit_100_blind_sheet.csv`) awaits human grading (`release_status` remains `PRE-RELEASE` until human `PM-4b` completes; agents NEVER self-grade `PM-4b` or open `*sealed_key*` / `*blind-followup*`).

### 1.2 Hardware Fleet (4-Node Local Cluster)
- **Local Workstation (Windows):** Orchestration, CPU/offline verification, Lean 4 / Rust / Python test suites, `O:\backups\` encrypted `.7z` vault.
- **DGX Spark #1 (`mattmre@10.0.0.49`, NVIDIA GB10 128 GiB unified):** Resident `nemotron-37c3` (`:8891`, untouched).
- **DGX Spark #2 (`mattmre@10.0.0.194`, NVIDIA GB10 128 GiB unified):** Resident `nemotron-6724` (`:8890`) & `glm53-remote-nfs` (untouched).
- **DGX Spark #3 (`mattmre@10.0.0.24`, NVIDIA GB10 128 GiB unified):** Resident `nemotron-eba1` (`:8801`) & user containers (untouched). Connected to Spark #1/#2 via 200GbE QSFP.

### 1.3 Current Status of `Wave 11` (`Part VII` Flagships `Demos D6–D10`)
1. **`Demo D6` (`chelated_multibackend_appliance.py`, `scripts/run_wave11_d6_opensearch_pgvector_qdrant.py`) — 100% COMPLETE & SEALED (`0578d56b715d`):**
   - Live `OpenSearch 3.9.0`, `pgvector 0.8.7` (`PostgreSQL 17.12`), and `Qdrant 1.20.2`. Identical `nDCG@10` across all 3 engines; `admitted_recall_at_10 = 1.0000` and `stale_resurrection_rate = 0.0000` across `0/10/50/90%` retraction densities.
2. **`Demo D8` (`demo_workbench/compliance_copilot_engine.py`, `scripts/run_wave11_d8_compliance_copilot.py`) — 100% COMPLETE & SEALED (`3fea611fe16a`):**
   - Live SEC `8-K` restatement + peer deadlock, GDPR Art. 17 erasure cascade (`REFUTE` -> `COPY_OF` / `DERIVES_CURRENT` while preserving `CITES_AUDIT`), FDA/HIPAA withdrawal, Ed25519-signed hash-chained receipts, and `0/56` Non-NDA IP scan.
3. **`Demo D7` (`chelated_graphiti_arena_v1.py`, `scripts/run_wave11_d7_live_zep_graphiti_arena.py`) — PATCHED & READY FOR SPARK #1 RELAUNCH:**
   - Live `graphiti-core 0.31.2` + `Neo4j 5.26 Community` + `mem0ai 2.2.1` vs `Chelated v4` on `Suite A (N=60)` + `Suite B (N=20 LongMemEval knowledge-update)`. Fixed `neo4j 6.4.1` `Record` key-membership check in `_cypher_scalar`, empty-`properties` `EntityAttributes` schema handling + `max_tokens<=4096` + `model_validate` retry in `MeteredVllmOpenAIGenericClient`, and `.venv-wave5` Python path.
4. **`Demo D9` (`chelated_fast_edge_extractor.py`, `scripts/run_wave11_d9_fast_extractor_and_u_grpo.py`) — PART A COMPLETE; PART B NEEDS PARITY + STABILITY PATCH:**
   - **Part A (Done):** `Qwen2.5-0.5B-Instruct` LoRA achieved `96.67%` (`58/60`) semantic & strict EM on `Suite A` (`100.0%` JSON validity across `1,020/1,020` docs) at `2,104 ms/case` (`4.6x` faster, `60x` smaller than `30B`).
   - **Part B (Paused after `seed_42`):** Untruncated 4-step `SFT 7B` lifted `EVAL_VIRGIN_V2` to `50.00%` (`61.67%` on `EVAL_WAVE6_LOOK2`). On `seed_42`, `DEV` peaked at `step=40` (`dev_exact_match=0.7000`, `det_em=0.6852`, `u_em=0.7308`), then under `lr=2e-5, beta=0.02` over `160` steps drifted into over-abstention (`dev_abstain_rate=0.85` at `step=160`), and failed line 2628's `<1e-4` check comparing `fp32` `merge_and_unload()` against `bf16` base + `fp32` LoRA (`0.002645` bf16 rounding). Proposed fix: compare `fp32` base + `fp32` LoRA vs `fp32` merged (`<1e-4`), and tighten GRPO to `lr=5e-6`, `beta=0.06`, `60` steps across all 3 seeds (`42, 137, 2026`), evaluating both pre-registered `final_step` and `best_dev_checkpoint` (selected strictly on `DEV` with a `determinate_em >= sft_det_em - 0.03` non-collapse guard).
5. **`Demo D10` (`scripts/run_wave11_d10_5m_real_cutover_and_70b_scale.py`) — 4/4 `8.84M` SHARDS STAGED; PARTS A & B QUEUED:**
   - All `8,841,823` unsubsampled `BeIR/msmarco` passages encoded (`384-d` + `20%` `1024-d` `BAAI/bge-m3`) on Spark #1; `"fiqa"` domain-key bug and zero-qrels anchor selector patched; Part A (`8.84M` standalone + `9.47M` 7-BEIR `IndexHNSWSQ` cutover + `EK5 v3` bitset fence) ready on Spark #1, Part B (`unsloth/Llama-3.3-70B-Instruct-bnb-4bit` `k=5`/`k=17` sweep) ready on Spark #2.

---

## 2. Proposed New Track A: `/formal` Machine-Checked Lean 4 + Kernel Verification + Skepticism Counter-Suite

### Motivation
Eliminate mathematical and spec-to-code skepticism from frontier-lab Principal Engineers, researchers, and acquirer diligence reviewers ("Pedigree Bypass") by shipping zero-`sorry` Lean 4 proofs (`lake build`), independent external kernel verification (`nanoda` / `#print axioms`), and a live Lean-to-Python differential conformance harness.

### Proposed `/formal` Deliverables & Skepticism Defenses (`M1–M5`)
1. **`Formal/Statement.lean` (~60 lines):** Self-contained, clean theorem statements readable in 2 minutes without scrolling through tactics.
2. **`Formal/Theorem1_MonotoneStiefel.lean`:** For any monotone dimension chain $d_0 \le d_1 \le \dots \le d_k$, semi-orthogonal $W_t \in \mathbb{R}^{d_t \times d_{t-1}}$ ($W_t^\top W_t = I_{d_{t-1}}$), and $\alpha_t > 0$, composed map $B = (\prod \alpha_t) W_k \cdots W_1$ satisfies $B^\top B = (\prod \alpha_t^2) I_{d_0}$ and spectral condition number $\kappa_2(B) = 1$; plus explicit formal counterexample for non-monotone `2 -> 3 -> 2` rank collapse.
3. **`Formal/Theorem2_AdjointReadDuality.lean` (`M1` Defense):** Prove $\forall q \in \mathbb{R}^{d_{\text{new}}}, d \in \mathbb{R}^{d_{\text{leg}}}$, $\langle q, W d \rangle = \langle W^\top q, d \rangle$ (justifying zero-rewrite downward query projection $q_\downarrow = W^\top q$ in `D6`/`D10`), and prove $\|W^\top q\|_2^2 = \|q\|_2^2 - \|(I - W W^\top) q\|_2^2$, explaining why post-projection unit re-normalization introduces query-dependent scale distortion across shards unless corrected by `ZeroMu + CSLS`.
4. **`Formal/Theorem3_ZeroMuPerturbation.lean` (`M2` Defense):** Prove exact `8 B/doc` sidecar affine-centered cosine recovery $\frac{\langle \hat{q}, d \rangle - \langle \hat{q}, \mu_d \rangle}{\|d - \mu_d\|_2} = \cos(\hat{q}, d - \mu_d)$ (`0 B` vector egress), plus forward quantization error bound $|\hat{s}(q,d) - s(q,d)| \le \varepsilon_{\text{quant}} / \gamma$ whenever $|\widehat{\langle \hat{q}, d \rangle} - \langle \hat{q}, d \rangle| \le \varepsilon_{\text{quant}}$ and $\|d - \mu_d\|_2 \ge \gamma > 0$, defining certified refill margin $\Delta_{\text{cert}} = 2\varepsilon_{\text{quant}}/\gamma$.
5. **`Formal/Theorem4_EK5_WFS_Incremental.lean`:** Formalize the 4-valued epistemic lattice and `K=4` authority-stratified transition operator over finite signed dependency DAGs (`COPY_OF`, `DERIVES_CURRENT`, `CITES_AUDIT`, `SUPERSEDE`, `REFUTE`), prove stratum-by-stratum monotonicity / Tarski least-fixpoint uniqueness, and prove incremental dirty-cone recomputation equals full-graph recomputation (`|D| = 0`).
6. **Finite-Sample $n < d$ Subspace Theory (`M3` Defense):** Wedin / Davis-Kahan perturbation bound explaining why rectangular Stiefel $\operatorname{St}(4096, 384)$ converges at $n = 2,000$ ($n > 5 d_{\min}$, $\|W\|_{\text{op}} = 1$) while unconstrained $4096 \times 4096$ Ridge/OLS collapses under Marchenko-Pastur eigenvalue inversion.
7. **Spec-to-Code Differential Conformance (`M4` Defense):** Compile executable Lean 4 `ek5_ref_cli` and test `1,000` random + adversarial cyclic/multi-stratum signed graphs against Python `ek5_v3_stratified.py` in `pytest` (`1,000 / 1,000` bit-exact parity), plus CI `#print axioms` gate (`0` `sorryAx`, `0` custom axioms) and external Lean kernel checker (`nanoda_bin` where buildable).

---

## 3. Proposed New Track B: `Demo D11` (`PM-6`) — Post-Training Epistemic Takedown, Parametric Recall Neutering, & Cross-Gen Governance Carry-Forward

### Motivation
Address hosted frontier-LLM copyright, privilege-clawback, and opt-out takedown liability (*In re OpenAI*, *GEMA v. OpenAI*, Best-of-$N$ extraction, activation-steering benign over-refusal) where restricted works exist simultaneously in **(1) frozen LLM weights $\theta$**, **(2) RAG chunks**, and **(3) pre-cached derivative agent summaries**, and must remain suppressed across **`384-d -> 4096-d` encoder/model upgrades** with **zero vault re-embedding** and **near-zero false refusals on benign same-topic queries**.

### Proposed Architecture & 6 Redlines Over Raw Draft
1. **Explicit Scope Boundary:** Hosted API / Compound AI Runtime inference-time epistemic unlearning & governance carry-forward (explicitly not claiming raw weight erasure inside downloadable `.safetensors`).
2. **4-Phase "Confirm-Then-Neuter" Protocol on 4-Node DGX Spark Fleet:**
   - **Phase 0 (Empirical Canary Lock):** Scan ~300 public-domain (`PG-19` literary openings, pre-2024 featured Wikipedia leads, classic RFCs/speeches) candidates against bare `Qwen2.5-7B` and `Nemotron-30B` (`temp=0.0`, 50-token prefix -> 200 tokens) to lock **`20` disjoint `DEV` calibration works** + **`60` held-out confirmed-memorized test works** (`ROUGE-L >= 0.80` or block recall `>= 0.75`), paired with **`60` same-entity Benign Sibling QA controls (`Q_benign`)**.
   - **Phase 1 (`1M`-Vector Restricted Vault + `180` Derivative Echoes):** Seed `300` target chunks + `1,000,000` real distractor passages (`384-d` legacy encoder) + `180` LLM-generated derivative summaries/synopses/fact-extractions wired into `ek5_v3_stratified.py` via `COPY_OF` and `DERIVES_CURRENT` edges (plus `CITES_AUDIT` legal-hold log).
   - **Phase 2 (Dual-Gate Runtime Firewall Across `6 Arms x 5 Regimes = 300 Evals/Arm`):**
     - Issue `60` Stratum-4 `REFUTE` takedown orders in `EK5 v3` (Ed25519-signed, hash-chained receipts).
     - **Gate 1 (Pre-Inference):** `EK5 v3` admission bitset on RAG/memory + canonicalized rolling 13-gram/MinHash + `ZeroMu` prompt-continuation screen (never blocking mere entity mentions!).
     - **Gate 2 (2-Tier Streaming Output Chelation with Holdback):** 64-token sliding window with **32-token SSE client holdback** (`0` leaked tokens on mid-stream halt): Tier 2A CPU `<0.3 ms` canonicalizer (stripping inter-char delimiters `"C-a-l-l"`, homoglyphs, Base64/ROT13) + 10-gram/LCS check; Tier 2B `ZeroMu + CSLS` check against both restricted verbatim chunks and restricted synopsis nodes, using $\tau_{\text{gov}}$ locked strictly on the disjoint `20`-work `DEV` split, plus configurable `min_infringing_tokens` policy knob.
     - **6 Arms:** (1) Bare LLM, (2) Hardened System-Prompt Guardrail, (3) Raw Vector DB Delete (leaves 180 derivative echoes + weights unguarded), (4) Live `mem0ai 2.2.1` delete/overwrite + prompt guardrail, (5) Linear Bridge (`CAR` / `Drift-Adapter`) + output filter (Phase 3), (6) `Chelated` Dual-Gate (`EK5 v3` + `CausalEMA ZeroMu + CSLS`).
     - **5 Regimes (`60` each):** `R1` Direct 50-token prefix (`temp=0`), `R2` Best-of-10 adversarial continuation (`temp=0.7`), `R3` Detailed plot-beat paraphrase & Spanish/German translation, `R4` Derivative memory echo attack, `R5` Benign sibling factual QA (`Q_benign`, scoring false refusals as errors).
   - **Phase 3 (`384-d -> 4096-d` Zero-Reembed Governance Carry-Forward):** Upgrade live stack to `Nemotron-30B` + `Qwen3-Embedding-8B` (`4096-d`) without re-embedding the `1,000,480`-vector `384-d` vault; test whether bridging multilingual `Qwen3-8B` (`4096-d`) into legacy `MiniLM` (`384-d`) also unlocks zero-shot cross-lingual takedown enforcement.

---

## 4. Specific Questions for the 4-Model Council (Round 1 Mandate)

1. **Sequencing & Wave Architecture (`Waves 11–13` `v6.0` vs `Wave 14+` `v7.0`):**
   - Should we seal `Waves 11–13` (`D6–D10`, `45/45` bundles, `v6.0`) cleanly first before landing `/formal` and `Demo D11 (PM-6)` in `v7.0`, or integrate any piece earlier? Are our proposed fixes for **`D7`** (`.venv-wave5`, `_cypher_scalar`, empty-object schema + `model_validate` retry) and **`D9` Part B** (`fp32` base+adapter merge parity + `lr=5e-6, beta=0.06, 60 steps` + `DEV` non-collapse guard) completely airtight, or do you spot any subtle statistical/selection bias or runtime flaw in them?
2. **Critique of Track A (`/formal` Lean 4 + Skepticism Defenses `M1–M5`):**
   - Which Lean 4 / Mathlib4 formalizations are realistic, fast to compile (`lake build`), and high-signal vs. where will Mathlib4 matrix/SVD/real-analysis bureaucracy become a multi-day tarpit? How should each theorem be stated so it is **100% honest, non-trivial (not tautological!), and zero-`sorry`**?
   - What *other* mathematical, algorithmic, or systems skepticism will a hostile Anthropic / DeepMind / NVIDIA / Databricks / academic reviewer throw at Chelated Memory & Spherical Migration that `M1–M5` still misses?
3. **Critique of Track B (`Demo D11` / `PM-6` Takedown & Carry-Forward):**
   - Poke holes in Phase 0–3, Gate 1/Gate 2 latency & holdback semantics, `R3` plot-summary vs `R5` benign-QA separability (`ROC-AUC` / false-refusal risk!), `R2` Best-of-10 / obfuscation evasion, cross-lingual `4096-d -> 384-d` geometry, and citation/claim hygiene. What exact baselines, ablations, or gates make `PM-6` bulletproof?
4. **Additional High-Impact Use Cases, Waves, or Ruthless Cuts:**
   - Looking across all existing/planned demos (`D1–D10` + `D11/PM-6` + `/formal` + open `PM-1b`), is there **any critical missing enterprise/frontier-lab use case, failure-closing wave (e.g., closing `PM-1b`?), or 1-command reproducibility artifact** we MUST add—or any low-ROI bloat we should cut—to maximize technical credibility, acquirer/investor diligence readiness, and Staff/Principal hiring impact?
