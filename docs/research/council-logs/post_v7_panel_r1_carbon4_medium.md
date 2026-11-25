---
panel_type: architecture-planning + feasibility-research
artifact: post_v7_wave_queue_drafts.md (Queued Items 1-6, 100% Additive Multi-Branch Mandate)
run_date: 2026-11-24
run_id: post-v7-panel-r1-carbon4-medium
round: 1_of_2
challenge_depth: critical
active_experts:
  - Chief Multi-Branch Release & Provenance Architect (Carbon 4 Medium)
  - Principal Anti-Overfitting & Pure-Tech Moat Strategist (Carbon 4 Medium)
persist_version: 1
---

# Post-`v7.0` Expansion Panel — Round 1 Critical & Additive Architecture Analysis (`Carbon 4 Medium`)

## 1. Multi-Branch Topology, Wave/Bundle Namespacing, & Trunk Isolation (`Q1`)

Running all **18 sub-tracks across `Items 1–6`** without pruning is a high-leverage empirical strategy **if and only if** branch provenance and claim hierarchy are isolated structurally so exploratory or vertical-specific variance never pollutes the sealed `v7.0` (`46/46`) core trunk:

1. **Hub-and-Spoke Git Topology off Sealed `v7.0` (`tag: v7.0-formal-pm6-sealed`):**
   - **`Wave 16` (`feat/v8.0-core-spine`, Trunk Candidate `Bundles #47–#48`):** Build the **shared foundational primitives (`Item 6A` + `Item 6D` core interfaces + shared conventional challengers)** *once* on a single short-lived spine branch off `v7.0`. Every downstream branch forks off **`v8.0-core-spine`**, never directly off divergent ad-hoc copies.
   - **Six Isolated Parallel Feature Worktrees Forked off `v8.0-core-spine` (`Waves 17A–17F`):**
     - `branch/w17a-generalization-wild` (`Items 6B + 6C`: 12-domain/4-hop zero-retune gauntlet + wild NVD/VEX & SEC `8-K` vs Postgres 17 Recursive-CTE ledger) $\to$ packets `PKT-17A-1`, `PKT-17A-2`.
     - `branch/w17b-partner-field-kits` (`Item 5`: Search-Operator Migration Kit + Records-Owner Natural-Base-Rate Governance Kit) $\to$ packets `PKT-17B-1`, `PKT-17B-2`.
     - `branch/w17c-fabric-dpu-orbital` (`Item 1`: Network DPU/Switch & Multi-Tranche LEO Satellite `/fabric` Control Plane) $\to$ packet `PKT-17C-1`.
     - `branch/w17d-pooled-agent-goodput` (`Item 4`: Pooled-Agent Loops through mid-job encoder change & source withdrawal) $\to$ packet `PKT-17D-1`.
     - `branch/w17e-cpu-storage-cluster` (`Items 2A–2E`: 5 pure-CPU + NVMe/Spinning-HDD demos graduating `computational_storage_poc/`) $\to$ packets `PKT-17E-1..5`.
     - `branch/w17f-gbench-neuron-probes` (`Items 3A–3D`: `LOFT SciFact` + disjoint `PM-2` exact closure, `BEAM 128K` + paired withdrawal, `QSCCI` singleton `k=1`, `PM-6` observational shadow probe) $\to$ packets `PKT-17F-1..4`.
2. **Zero-Collision Registry & Workbench Rule:**
   - Branch-local runs emit self-contained signed manifests (`artifacts/branch_packets/<pkt_id>/packet_manifest.json`) verified via `scripts/verify_packet_offline.py --packet <dir>` **without** mutating `bundle_registry_spec.json`, `demo_workbench/server.py`, or `scripts/verify_all_offline.py` inside the spoke branches.
   - **Wave 18 (`v8.0` / `v9.0` Consolidation & Promotion Gate):** Only *after* all 18 packets complete and are scored against pre-committed branch gates do we assign sequential permanent `verified_registry.json` bundle numbers (`#47..#N`) — **promoting BOTH passing and falsified/retired core-attribution packets into an immutable Complete Evidence Ledger** so no critic can accuse us of hiding negative branches.

---

## 2. Shared Foundational Spine ("Build Once in `Wave 16`, Import Everywhere") (`Q2`)

Without a shared `Wave 16` spine, the 6 branches would re-implement **four** critical subsystems inconsistently:

| Duplicated Subsystem | Appears In | Canonical `Wave 16-Core` Module (Build Once) |
| :--- | :--- | :--- |
| **Conventional Bitemporal + Authority + Declared-Edge Ledger** | `Item 1` (Arm 2), `Item 2D` (`C3`), `Item 4` (Cell 1/2), `Item 5.2` (Arm 2), `Item 6C` (Arm 1) | `chelated_kernel/challengers/bitemporal_ledger.py` providing **both** an in-memory copy-aware versioned ledger AND a real **PostgreSQL 17 `WITH RECURSIVE` Bitemporal + Authority CTE** backend with identical Python protocol (`ingest`, `retract`, `reinstate`, `query_as_of`). |
| **$4 \times 2$ Bridge $\times$ Merge Calibration & Span-Profiled Cutover Harness** | `Item 1`, `Item 2C`, `Item 3A`, `Item 4`, `Item 5.1`, `Item 6B` | `chelated_kernel/migration_matrix.py` implementing `{UncenteredProcrustes, DriftAdapter, Rank1QueryResidual, ZeroMu} × {CalibratedZScore, CSLS}` + `RequestSpanProfiler` + `Item 6A` HNSW pre-filter bitmap pushdown. |
| **Lazy Audit-Reason Canonicalizer on `EK5 v3`** | `Item 1`, `Item 4`, `Item 5.2`, `Item 6A`, `Item 6C` | `ek5_v3_stratified.py::export_canonical_audit_trace()` closing the `153–366` Pitfall-B `reason`/`pos_cyc` diffs to `0` before `Items 5.2` and `6C` grade explanation faithfulness. |
| **3-Line `chelated-kernel` SDK Envelopes (`CloudEvents` / `OSV-VEX` / `Git` / `PROV`)** | `Item 1`, `Item 2E`, `Item 4`, `Item 6C`, `Item 6D` | `packages/chelated_kernel/` (`wrap_index`, `EventLedger`, `wrap_stream`) built in `Wave 16` so `Items 1, 2E, 4, 6C` are **dogfooding tests of the actual 3-line SDK**! |

---

## 3. Item-by-Item Anti-Overfitting & Systems Trap Audit (`Q3`)

1. **`Item 1` (`/fabric` DPU & LEO Mesh):** *Trap:* Synthetic flow vectors where `DNS/NTP` hubs or C2 clusters are generated with toy Gaussian offsets will look cooked to NVIDIA/Cloudflare engineers. *Fix:* Derive encrypted-flow embeddings from real public netflow/TLS-metadata feature schemas (`CIC-IDS` / `CTU-13` / `CESNET-TLS` public statistical distributions embedded via real `MiniLM`/`BGE`/`MPNet`/`Qwen3-8B` text/tabular serializers) and use real CelesTrak TLE orbital contact windows for the LEO DTN partition schedule.
2. **`Item 2A` (`30B-A3B` MoE under `8 GiB` `cgroup`):** *Trap:* `Qwen3-30B-A3B` (`Q4_K_M` ~`17.5 GiB` on disk) + `4.5 GiB` pinned experts + `0.6B` draft (`380 MiB`) + KV cache + `mmap`/allocator overhead can OOM-kill inside a hard `8.0 GiB` `MemoryMax` if glibc arena fragmentation or GGUF metadata pages aren't accounted for; also `Qwen3-0.6B` vs `Qwen3-30B-A3B` must verify exact vocab/special-token ID parity. *Fix:* Use `O_DIRECT` + pre-allocated aligned slab ring (`<= 6.4 GiB` total RSS ceiling leaving `1.6 GiB` kernel/KV headroom) and measure **cold-start + steady-state** `tok/s` separately.
3. **`Item 2B` (`1,000`-Cartridge HDD/SSD Library):** *Trap:* Calling `250` rank-perturbed synthetic clones a "`1,000`-specialist library" risks diligence pushback if accuracy gains come from synthetic routing tricks. *Fix:* Separate the claims cleanly in the manifest: **(a) Systems I/O & `EK5` Revocation Track** (`1,000` physical stripe-aligned `.cspg`/GGUF extents totaling `>= 40 GiB` to defeat OS cache and stress spindle seek + `< 1 ms` `EK5` revocation), and **(b) Real-Adapter Accuracy Track** evaluated strictly on **genuinely trained/public HuggingFace LoRA adapters** (`D9-A` + real domain LoRAs, zero synthetic weight perturbations in accuracy claims).
4. **`Item 2C` (`9.47M` Cold-OSD RAG):** *Trap:* Comparing custom AVX2/NEON `IVF-INT8` sequential extent scans against an untuned `Qdrant on_disk=True` HNSW graph (which does random 4KB pointer-chasing I/Os catastrophic on HDDs) could be dismissed as "IVF beats HNSW on spinning rust, not CHELATEDAI." *Fix:* Add **`IVF-Flat INT8` without `ZeroMu+CSLS` / `EK5` extent-skip** (`Rung b+`) on the **exact same `2–8 MiB` sequential extent layout** so the `ZeroMu+CSLS` quality delta and `EK5` revoked-extent I/O skip speedup are isolated cleanly from the `IVF` vs `HNSW` layout effect!
5. **`Item 2D` (`< 1 GiB` CPU Appliance) & `Item 2E` (SSD KV/Prefix Repo Cache):** *Trap in `2E`:* Serializing full fp16 Transformer KV states across 50 commits can exceed NVMe bandwidth gains if chunk boundaries invalidate cross-chunk causal attention! *Fix in `2E`:* Explicitly distinguish **position-independent/RoPE-reindexed chunk KV (`CacheBlend` / `PromptCache` block-attention regime with measured quality delta)** vs **prefix-ordered topological module packing**, and report exact fidelity vs full re-prefill.
6. **`Items 3A–3D` (`gBench` & Neuron-Basis):**
   - `3A`: Enforce SHA-256 stem/doc-ID audit between `LOFT SciFact` and BEIR `SciFact` (`100%` expected root corpus overlap -> treat `LOFT SciFact` strictly as mechanism pilot and run exact Tango NI confirmation on a **100% disjoint corpus** such as `TREC-COVID` or `BioASQ` with frozen $N$ and margin).
   - `3B`: Pin local deterministic open-weight judge alongside `gBench` Gemini-judge rubric and isolate `BEAM 128K` native score from stateful replay.
   - `3C` (`QSCCI` singleton): Pre-register exact criterion for "chelation-specific separation" (`selection_cos(chelated, ordinary_contrastive) < 0.95` AND out-of-family nuisance invariance win on `REPORT`) so if both selectors again pick the same neuron, it cleanly falsifies selector uniqueness rather than moving goalposts.
   - `3D` (`PM-6` shadow probe): Expand benign holdout to **`>= 160` independent author/topic families** so `0` false positives proves `< 2.0%` exact 95% upper bound.
7. **`Item 4` (Pooled-Agent Loops):** *Trap:* Live LLM worker variance can create noisy retries unrelated to retrieval/memory. *Fix:* Pair seeds/arrival schedules identically across all $2 \times 2$ cells (`Common Random Numbers` + deterministic `temperature=0.0` primary pass + `temperature=0.3` robustness pass) so paired differences isolate `ZeroMu` and `EK5 v3`.
8. **`Items 6B & 6C` (Zero-Retune 12-Domain & Wild Revision Streams):**
   - In `6B`, commit `frozen_kernel_config_v8.json` (SHA-256) **before** downloading/evaluating any of the 12 corpora; report **all 12 domains + pooled random-effects meta-analysis** (zero domain dropping) AND separate **1-hop (`UncenteredProcrustes + CSLS` vs `ZeroMu + CSLS`)** from **4-hop (`v1->v2->v3->v4` without re-indexing)**.
   - In `6C`, give `Postgres_17_Recursive_CTE_Bitemporal_Ledger` proper composite B-tree/GiST `(entity_id, valid_range, tx_range)` indexes and cycle-detection (`CYCLE ... SET is_cycle`) so it is an **industrial-grade SQL challenger**, proving `EK5 v3`'s advantage comes specifically from **stratified WFS unfounded-set collapse on unlinked mirror echoes + 3-valued `Disputed` isolation**, not unindexed SQL.

---

## 4. 4-Node Fleet Scheduling (`Q4`)

- **Phase P0 (`Wave 16-Core`, ~2–3h on Desktop + Spark #1 Docker):** `Item 6A` (HNSW pre-filter pushdown + `EK5` audit-reason canonicalizer) + `Item 6D` (`chelated-kernel` SDK skeleton + shared `bitemporal_ledger.py` & `migration_matrix.py`).
- **Phase P1 (Parallel Execution Wave A):**
  - *Desktop CPU:* `Item 5` (Partner Field Kits Stage 1) + `Item 6C` (Wild NVD/GHSA/OSV/VEX + SEC `8-K` vs Postgres 17 Recursive-CTE).
  - *Spark #1:* `Item 1` (`/fabric` DPU & LEO embeddings + 200-seq fuzz) $\to$ `Item 2C` (`9.47M` Cold-OSD CPU scan reusing local MSMARCO shards, `CUDA_VISIBLE_DEVICES=""`, `MemoryMax=4G`).
  - *Spark #2:* `Item 6B` (12-domain / 4-hop GPU cache generation, batched fast) $\to$ `Item 2A` (`Qwen3-30B-A3B` CPU-only `MemoryMax=8G` Disk-MoE).
  - *Spark #3:* `Item 3C` (`QSCCI` `Qwen3.5-2B-Base` singleton) + `Item 3D` (`PM-6` shadow probe post-`W14B`) $\to$ `Item 2D` (`< 1 GiB` CPU `IQ-04` appliance).
- **Phase P2 (Parallel Execution Wave B):**
  - *Spark #1 + `:8891` Nemotron:* `Item 4` (Pooled-Agent Loops Stages 1–3) + `Item 3B` (`BEAM 128K` + paired withdrawal).
  - *Spark #2 / Desktop CPU:* `Item 2B` (`1,000`-Cartridge Array `MemoryMax=6G`) + `Item 2E` (50-commit SSD KV/Prefix Repo Cache `MemoryMax=8G`) + `Item 3A` (`LOFT SciFact` + disjoint `PM-2` exact Tango closure) + `Item 6D` final 4-vertical `<= 15 LOC` clean-room SDK certification.

---

## 5. Pre-Committed Anti-"Muddy-Waters" Claim Hierarchy & Adversarial Additions (`Q5` & `Q6`)

1. **Three-Tier Pre-Registered Scorecard (Eliminates Post-Hoc Cherry-Picking Across 18 Tracks):**
   - **Tier A — Core Pure-Tech Moat Gates (Must Pass Holm-Bonferroni Family-Wise $\alpha = 0.025$; Governs Headline Valuation):** `Item 6A` (`< 35 ms` p95 @ `90%` retraction + `0` audit-reason diffs), `Item 6B` (12-domain zero-retune `CSLS` + 4-hop `ZeroMu` stability), `Item 6C` (Wild NVD/VEX + SEC `8-K` win over indexed Postgres 17 Recursive-CTE at `0` LLM write tokens), `Item 6D` (`<= 15 LOC` zero-services SDK across 4 verticals), and `Item 3A` (`PM-2` exact Tango NI closure on disjoint corpus).
   - **Tier B — Vertical & Hardware Reference Architectures (Evaluated Per-Vertical with Explicit Component Attribution):** `Item 1` (`/fabric`), `Items 2A–2E` (CPU/SSD/HDD), `Item 4` (Pooled-Agent Goodput), `Item 5` (Partner Kits). Every Tier-B receipt reports **marginal kernel attribution** (`Kernel-On` vs `Strongest Conventional Challenger on Identical Systems Layout`).
   - **Tier C — Mechanistic & Boundary-Falsification Probes (Pre-Declared Exploratory/Diagnostic; Negative Results Published Intact):** `Item 3C` (`QSCCI` singleton), `Item 3D` (`PM-6` shadow probe), and `Item 5` component-retirement checks (`ZeroMu` vs `UncenteredProcrustes + CSLS` in 1-hop static cutover).
2. **High-Leverage Additive Test to Fold Into `Item 6` (`Item 6E` Micro-Gate inside `6B`/`6C`):**
   - **Adversarial Anchor & Echo Robustness:** Test *(a)* **`5%–10%` corrupted/poisoned unlabeled anchor pairs** during `ZeroMu` bridge fitting (`TRA-1` outlier-trimmed Huber/RANSAC anchor guard vs naive SVD collapse), and *(b)* **adversarial synonym-paraphrased echo injection** vs benign independent corroboration in `EK5`'s observable echo-cluster collapse (reporting exact TPR/FPR ROC curve of echo collapse on wild NVD/VEX mirror advisories).
