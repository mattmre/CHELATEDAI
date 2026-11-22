# Phase 2 Engineering & Compute Roadmap: Real-World Use-Case Demonstrations, Scale Validation, and Lab-Grade Industry Proof

**Date:** November 22, 2026  
**Baseline Foundation:** Post-R5b Verified Dual-Flagship Stack (`27/27` SHA-256 Bundles, `registry_content_sha256: b57b05660edf`)  
**Compute Footprint:** `2×` Local NVIDIA DGX Spark Nodes (`gx10-37c3` & `gx10-6724`, GB10 `128 GB` unified memory each = `256 GB` 24/7 local cluster) + **Google AI Ultra Subscription** (`1,000–2,000` Colab Compute Units/month = **`~55–110` hosted NVIDIA H100 80GB hours/month** or **`~80–370` A100 hours/month** with 24-hour browser-closed background execution + **`$40–$100/month` Google Cloud credits**)

---

## 1. Plain-English Mental Model: What This Stack Actually Does to an LLM

Neither Flagship #1 nor Flagship #2 requires pretraining a massive LLM from scratch. Instead, they form a **deterministic, sub-millisecond geometric and epistemic memory substrate** around frozen or serving models (such as **`Nemotron-3.5-30B-A3B`**, `70B–120B+` open-weights models, or frontier APIs):

1. **Flagship #1 (`spherical_migration_bridge.py` + `TRA-1`) — Zero-Reindex Geometric Transport:**
   - **What breaks today:** When an organization upgrades embedding models (`384-d → 768-d → 1024-d → 2560-d → 4096-d`) or routes queries across specialist code/legal/scientific vector indexes, naive linear adapters blow up spectrally ($\kappa \sim 10^{12}$) across multi-hop chains and collapse to near-zero (`0.005–0.020` nDCG@10) out-of-domain because the fit-domain target centroid $+\mu_y$ acts as a false-positive magnet.
   - **What our stack changes:** `EM-2` (`A2-ZeroMu-Query` at **`0` index writes**; `A2-ZeroMu-DualFit` at **`1` float/doc**; `A2-ZeroMu-DualAdap` at **`0` target anchors**), `A2c-Polar` ($\kappa = 1.0000$ across interior dimension peaks), `A2g` geodesic polar alignment, and `B4 + B2-OldSpace` selective backfill let a new model query legacy or cross-domain vector stores immediately (`0.718 ms` cached p95, up to **`51.2×`** OOD gain over Centered Affine Ridge), while `TRA-1` blocks `100%` (`12/12`) of mismatched/forged representation packets fail-closed.
2. **Flagship #2 (`ek5_v3_stratified.py` + `em4_soft_telemetry_routing.py`) — Stratified $K=4$ Epistemic Truth & Echo Control:**
   - **What breaks today:** Scaling an LLM from `8B → 30B → 70B → 400B+` **cannot** fix retrieval drowning when `top_k` context slots are flooded by `12–50` stale paraphrased recaps, nor can prompt rules prevent low-trust rumors laundered through high-tier summaries from overwriting certified facts. Meanwhile, LLM-extracted memory layers (`mem0ai`) burn **`2,988–6,399 ms` of LLM inference per write** and still lose **`50.0 pp`** (`0.475` vs `0.975`) under multi-hop retraction and authority laundering.
   - **What our stack changes:** `EK5 v3` + `EM-4` (`C4_v4`, `M4f`, `M4g`) resolves bitemporal visibility (`as_of(vt, tx)`), capability-scoped anti-laundering (`Stratum 1`), exact Well-Founded Semantics reinstatement (`Stratum 2`, `0/381,497` oracle mismatches), and version lineage (`Stratum 3`) in **`0.48–0.90 ms`** (`6,225×` faster than `mem0ai`), cuts prompt/KV tokens by **`40–47%`**, and preserves **`0.00 pp` clean regression** (`+1.00` to `+1.33 pp` gain with `M4g`) across `800` unmodified conversational queries (`LoCoMo-300` + full `LongMemEval-S N=500`).

---

## 2. Critical Compute Decision: Do We Need to Train Massive Models From Scratch?

**Short answer: NO—and attempting from-scratch pretraining would waste months and weaken the pitch.**

| Compute Strategy | Cost & Timeline | Scientific & Hiring Signal at Top AI Labs | Verdict |
|---|---|---|---|
| **Option A: Pretrain a Massive Base Model from Scratch (`30B–70B`)** | `50,000–500,000+` H100 hrs (`$100k–$1M+`), `3–6` months | **Low / Negative ROI:** Frontier labs (NVIDIA, OpenAI, Anthropic, DeepMind, xAI) already own $100M+ base models. A solo/small-cluster base pretrain cannot beat frontier base weights and obscures the architectural contribution. | **DO NOT DO** |
| **Option B: Inference-Scale Proofs on Massive Existing Models (`Nemotron-3.5-30B`, `70B–120B+`) + `10M+` Live Vector Corpora** | **`0` extra cash** (`2×` DGX Spark 24/7 + `~25–35` Colab H100 hrs from Google AI Ultra), `2–3` weeks | **Very High ROI:** Proves directly on the exact models labs serve (`Nemotron-3.5`, `Llama-3.3-70B`, `Qwen2.5-72B`) that `70B+` parameters still fail flooded temporal/OOD drift without ChelatedAI—and jump `+25` to `+80 pp` at `-45%` KV cost with it. | **PRIMARY TRACK** |
| **Option C: Targeted Post-Training / RLVR (`GRPO` / `LoRA` on `8B–30B` via `EK5 v3` Formal Verifier)** | **`0` extra cash** (`~25–35` Colab H100 80GB hrs using Ultra 24h background mode), `1.5–2` weeks | **Exceptional Differentiator:** Turns `EK5 v3` (`0/381,497` exact WFS oracle) into a **verifiable reward environment (RLVR)** for non-monotonic epistemic reasoning & calibrated abstention—the #1 bottleneck in frontier reasoning post-training today. | **HIGH-LEVERAGE TRACK** |

### Exact Hardware Allocation Matrix (`$0` Incremental Compute Spend)
- **Local Dual NVIDIA DGX Spark (`gx10-37c3` + `gx10-6724`, `2× 128 GB` Unified Memory, 24/7):**
  - Host persistent `Nemotron-3.5-30B-A3B-NVFP4` vLLM services, `10M–25M` vector memory-mapped/resident indexes (leveraging 128 GB unified RAM per node!), continuous multi-day streaming ingestion, and `1M–10M`-node `EK5 v3` soak tests at **$0 marginal cost**.
- **Google AI Ultra Subscription (`~1,000–2,000` Colab Compute Units/mo + `$40–$100/mo` GCP Credits):**
  - **NVIDIA H100 80GB (`~18.05 CCUs/hr` $\implies$ `~55–110 H100 hrs/mo` with 24h browser-closed background execution):** Reserved strictly for high-bandwidth batch jobs where HBM3 (`3.35 TB/s`) gives a `10×` speedup over GB10: (1) fast batch embedding of `5M–10M` external chunks across `384/1024/4096-d` encoders (`~8 H100 hrs`), (2) `70B–72B` (`Llama-3.3-70B` / `Qwen2.5-72B` / `Nemotron-70B`) unquantized/FP8 scale validation (`~15 H100 hrs`), and (3) `GRPO`/`DPO` LoRA RLVR post-training runs (`~30 H100 hrs`).
  - **GCP Monthly Credits (`$40–$100/mo`):** Host the public/interviewer-ready **Cloud Run Interactive Demo + Live Receipt Verifier** (`$5–$15/mo`) and burst Vertex AI / Gemini / Claude frontier judge evaluations.

---

## 3. Six-Week Phased Execution Plan & Timing (Incorporating 4-Model Frontier Panel Directives)

```
Stage R6 (Days 1-4) : PM-4b Human Audit + Label-Free PM-2 / Online-Centroid Rerun + Delta-Tracked EK5 Core
Week 1 - Week 2     : Campaign D1 — Raw-Text 0.6B-7B Epistemic Edge Extractor + Prospective Post-Cutoff SEC/CVE Stream
Week 2 - Week 3     : Campaign D2 — 10M-Chunk Zero-Downtime Vector Migration (Qdrant/pgvector + Online Streaming mu_q)
Week 3 - Week 4     : Campaign D3 — Heterogeneous 70B+ Multi-Agent Anti-Poisoning Fabric (AgentDojo / Rumor Laundering)
Week 4 - Week 5     : Campaign D4 — Epistemic-RLVR Post-Training (EK5-WFS Verifier -> GRPO/LoRA on 7B/30B Reader & Judge)
Week 6              : Campaign D5 — Hosted Interactive Demo, Cost/ROI Simulator & Frontier Lab Technical Portfolio
```

### Stage R6 — Clean Patch & Human Audit Closure (`Days 1–4`, Local Dual Spark + `~2 hrs` human time)
Directly closing the exact items flagged by our 4-Model Frontier Panel (`GPT-6.2 Astra Ultra`, `Claude Opus 5.5 Ultrathink`, `Grok 4.7 XHigh`, `Gemini 4 Neon XHigh`):
1. **Grade `PM-4b` Blind 100-Item Human Sheet:** Score `pm4_human_audit_100_blind_sheet.csv` via hardened `scripts/score_pm4_human_audit.py` (`100` unique expected IDs, off-git sealed key).
2. **Label-Free `PM-2` & Streaming-Centroid Rerun:** Upgrade `resolve_slot()` to filter `blocked_authority_edges` directly (eliminating the harness `c["family"]` check), replace evaluation-batch `mu_q_eval` with an **online streaming EMA centroid** (plus a tiny `2–4 H100 hr` low-rank residual adapter closing `Query0W`'s `0.695 → 0.965` gap), and benchmark end-to-end encode + `resolve_with_tra1()` + BM25+dense + live `EK5 v3` lookup at `10^6+` chunks.
3. **Delta-Tracked / Rust-PyO3 `EK5 v3` Kernel:** Replace `dict(self.eligibility)` full-map copy (`~7.8 ms` 1M floor) and linear `remove_edges` scan with delta-tracked maps and incremental Stratum-1 SCC maintenance (targeting `< 0.15 ms` p95 at `1M` nodes).
4. **Raw-Text `0.6B–7B` Epistemic Edge-Proposer LoRA (`~6–10 H100 hrs`):** Train a fast sub-10ms extractor (`ModernBERT-large` / `Qwen2.5-7B` LoRA) emitting `(REFUTE, SUPERSEDE, AMENDS, COPY_OF, CITES_AUDIT)` edges and `(ECHO_COLLAPSE vs PRESERVE_CONFLICT)` from **raw unstructured text**—completely retiring synthetic regex prefixes and structured-input asymmetry against `mem0ai` and `graphiti-core`!

### Campaign D1 — Zero-Downtime 10M-Vector Fleet Migration (`Weeks 1–2` | Local Spark 128GB + `~8 H100 hrs`)
- **Unseen External Benchmarks:** **BRIGHT** (ICLR 2025 reasoning-intensive retrieval across coding, theorem proving, economics, biology) + **5M–10M Wikipedia / SEC EDGAR chunks**.
- **Setup:** Build a realistic production HNSW/flat cosine index in legacy `384-d`/`1024-d` (`MiniLM` / `BGE-M3`) and simulate an enterprise upgrade to a `4096-d` flagship encoder (`NV-Embed-v2` / `Qwen3-Embedding-8B`) with only `2,000` unlabeled anchor queries/docs (`0.02%` of corpus).
- **Metrics Proven:**
  1. **Zero-Downtime Cutover:** `A2-ZeroMu-Query` (`0` index writes) + `A2-ZeroMu-DualFit` (`1` float/doc = `40 MB` sidecar vs `160 GB` full 4096-d float32 re-index!) vs Centered Affine Ridge across cross-domain slices.
  2. **Progressive Background Backfill Curve:** `B4 SmoothTail + CSLS + B2-OldSpace` at `0% → 5% → 10% → 20%` re-embedded budget, measuring exact **GPU-hours and cloud $ saved** to reach `95%` and `98%` of full native re-index accuracy.

### Campaign D2 — `Nemotron-3.5` & `70B+` on Unseen Real-World Drift Streams (`Weeks 2–3` | Dual Spark + `~15 H100 hrs`)
- **Why External Scenarios Matter:** Moves completely outside `LoCoMo`, `LongMemEval`, and `Zero-Shortcut` onto **public, verifiable real-world non-monotonic streams** where facts genuinely mutate:
  1. **Financial Filing Stream (`SEC 10-K / 10-Q / 8-K Restatements & Earnings Guidance`):** Quarterly KPI revisions, M&A cancellations, and financial restatements drowned under `15+` analyst summary echoes.
  2. **Living Codebase Stream (`Git Commit + Revert + CVE Advisory` on real OSS repos):** Breaking API changes, reverted PRs (`v1 → v2 [reverted] → v1`), security patches (`CVE` refutations), and stale `README`/issue comments echoing deprecated flags.
  3. **Clinical Protocol Updates:** Guideline dosage/contraindication revisions (`EM-1` minimal-pair stress) + retracted preprints cited by secondary review articles (`Stratum 1` anti-laundering stress).
- **Scale Sweep (`8B → 30B Nemotron-3.5 → 70B/72B` on H100):** Demonstrates visually that **model parameter scaling alone (`8B → 72B`) stays flat/broken when `top_k` retrieval is poisoned by echoes or laundered rumors**, whereas `Nemotron-3.5-30B + EK5 v3 + EM-4` beats a bare `72B` model by `+30` to `+60 pp` at `< 1/3` the serving cost!

### Campaign D3 — Heterogeneous Multi-Agent & Edge-to-Cloud Recomposition (`Weeks 3–4` | Dual Spark `37c3 ↔ 6724`)
- **Architecture:** `gx10-37c3` acts as an edge/fast-tier node (`384-d` embeddings + `EK5 v3` bitemporal store); `gx10-6724` hosts `Nemotron-3.5-30B-A3B` orchestrating three specialist indices (`1024-d` Code, `768-d` Science, `4096-d` Legal/Finance) bridged across interior peaks (`A2c-Polar`) and OOD shifts (`EM-2 ZeroMu-DualAdap`) under `TRA-1` cryptographic contracts.
- **Metrics Proven:** End-to-end multi-agent task success, `0%` cross-agent rumor laundering (`CITES_AUDIT` vs `D_V`), `0%` silent wrong-space vector corruption, and `< 2 ms` routing overhead.

### Campaign D4 — Epistemic-RLVR Post-Training (`GRPO` / `LoRA` via `EK5 v3` Formal Verifier) (`Weeks 4–5` | `~30 H100 hrs` Background Colab)
- **Core Insight:** Frontier post-training teams use RLVR (Reinforcement Learning with Verifiable Rewards) for math and code because compilers/CAS verify answers deterministically. **Long-horizon memory, contradiction resolution, and calibrated abstention previously lacked a deterministic formal verifier—`EK5 v3` (`0/381,497` exact WFS parity) IS that verifier!**
- **Execution:**
  1. Generate `25,000` diverse multi-hop epistemic lifecycles (authority laundering, mutual-refutation abstention, `v3→v2→v1` middle-refuted version lineage, `AMENDS` vs `COPY_OF`, minimal-pair updates) with exact ground-truth `ADMIT` / `CONTESTED` / `AUDIT_ONLY` states from `EK5 v3`.
  2. Train a lightweight `LoRA` (`r=32`) via `GRPO` / `DPO` on an open `8B–30B` model (using Colab H100 80GB 24-hour background sessions) to natively interpret `[EM-3 TELEMETRY]` certificates and abstain on `CONTESTED` states.
  3. Evaluate zero-shot on held-out real-world SEC/Code/Conversational benchmarks (Campaign D2), proving measurable gains at **both** the data-curation/training layer and the inference layer.

### Campaign D5 — Executive Live Workbench & Hiring Pitch Package (`Week 6` | GCP Credits `$10–$20`)
- Deploy a self-contained **Interactive Live Demo** (Cloud Run UI tunneling or replaying live receipts + 1-click Colab notebook) + a **2-page Executive Systems Brief** tailored to Inference/RAG Infrastructure, Agent Platform, and Post-Training/RLVR teams.

---

## 4. Concrete Value to a Top AI Company (Inference, Recomposition, Training, & Revenue ROI)

If handed to **NVIDIA (`Nemotron` / `NIM`), OpenAI, Anthropic, Google DeepMind (`Vertex` / `Gemini`), xAI, or Databricks/Snowflake**, this stack delivers four immediate bottom-line impacts:

| Layer | Specific Engineering Use Case | Quantified Net Benefit (Verified in R4/R5b) | Revenue / Cost Impact for a Frontier Lab |
|---|---|---|---|
| **1. Inference & Hosting Economics** | Replace LLM-driven memory extraction/reconciliation (`mem0ai`-style) + prune stale echo bloat before prefill | **`6,225×` faster memory writes** (`0.48 ms` CPU vs `2,988–6,399 ms` GPU LLM calls); **`40–47%` fewer prompt/KV tokens** (`162` vs `305` toks); `0.718 ms` p95 coupled query | Eliminates `1–2` hidden background LLM calls per turn (`>99%` memory-write GPU savings) and increases **KV-cache tenant density by `~1.7×`** per H100/GB200 pod, directly boosting gross API/enterprise margins. |
| **2. Enterprise Customer Switching & Lock-In Removal (`Flagship #1`)** | "Zero-Reindex Embedding Upgrade" inside NIM / Vertex / OpenAI Vector Stores (`384/768/1536-d → 4096-d`) | **`0` index writes (`ZM-Query`)** or **`1` float/doc (`ZM-DualFit`)**; **`16/18` OOD wins (up to `51.2×` CAR)**; **`96.5%` vs `98.5%` native coupled QA** | Unblocks enterprise adoption of higher-tier embedding & reasoning SKUs immediately without forcing customers to re-embed billions of documents or suffer migration downtime. |
| **3. Model Recomposition & Multi-Agent Trust (`TRA-1` + `EK5 v3`)** | Typed representation ABI across specialist MoE/retriever tools + anti-laundering multi-agent memory bus | **`12/12` (`100%`) hostile/mismatched space leaks blocked**; **`100%` vs `0–16%` accuracy under authority-laundered rumors** | Unlocks regulated enterprise contracts (**Finance, Healthcare, Legal, Defense/Gov**) where deterministic bitemporal auditability (`as_of(vt, tx)`) and anti-poisoning guarantees are mandatory deal-breakers. |
| **4. Pretraining Curation & RLVR Post-Training** | (a) Corpus deduplication via `EM-1`/`EM-4` without destroying minimal-pair state updates; (b) `EK5 v3` WFS oracle as an RLVR reward environment | Preserves `100%` minimal-pair updates (`+3.40` to `+8.00 pp` over naive semantic dedup `M4b`); `0/381,497` exact formal WFS verification | Gives post-training teams a **scalable, zero-human-label formal verifier** to train frontier reasoning models on non-monotonic belief revision, source-authority reasoning, and calibrated abstention. |

### Target Team Mapping for Outreach
1. **NVIDIA** — *Nemotron Post-Training / NeMo Retriever & NIM Inference Microservices* (already benchmarked natively on `2× DGX Spark GB10` + `Nemotron-3.5-Lightning-30B-A3B-NVFP4`!).
2. **Anthropic / OpenAI / Google DeepMind / xAI** — *Long-Horizon Agent Memory, Enterprise RAG/Search Infrastructure, & RLVR Reasoning Environments*.
3. **Databricks (Mosaic AI) / Snowflake / Qdrant / Turbopuffer / Pinecone** — *Zero-Downtime Vector Index Lifecycle & Governed Compound AI Systems*.
