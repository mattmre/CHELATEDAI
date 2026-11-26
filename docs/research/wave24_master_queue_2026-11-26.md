# Wave 24 Master Queue — Zero-Cost CI, Hygiene, Untested High-Upside Lanes & Human Gates Last

- **Created:** 2026-11-26 (`pv7/w16-core`, base `f824e244`)
- **Machine-readable manifest:** `docs/research/wave24_master_queue_2026-11-26.json`
- **Hard Policies:**
  1. **Zero GitHub Billing Forever:** Never pay GitHub for hosted Actions. Ruleset `13081256` (`MATT ONLY MAIN MERGE`) has no `required_status_checks`; CI runs on `$0/mo` self-hosted Spark runners (`gx10-6724`) or local cryptographic verification.
  2. **Compute Priority:** Offload GPU & heavy CPU work to the DGX Spark fleet (`gx10-6724` `76 GiB` free, `gx10-37c3` `56 GiB` free) and Colab Ultra (`3,031.12 CU` via `gx10-eba1`).
  3. **Human Time Strictly Last:** `W24-15` (`W22-13` `PM-4b`) and `W24-16` (`W22-14` partner trial) remain at the end of the queue after all automated engineering and empirical lanes finish.
  4. **Never Touch:** `*sealed_key*`, `*blind-followup*`, `pm4_human_audit_100_blind_sheet.csv`.

---

## P0 — Zero-Cost CI & Core Hygiene (Immediate)

| ID | Task | Target | Status | Gate / Acceptance |
|---|---|---|---|---|
| **W24-01** | Zero-Cost CI (`gx10-6724` free ARM64 self-hosted runner + remove paid `ubuntu-latest` PR triggers) | `gx10-6724` | QUEUED | `$0` GitHub spend; green self-hosted/local verification |
| **W24-02** | Fix **`CD-W22-11-01`** (`compliance_copilot_engine.py:785,929,1086` canonical reason export) | Local / Spark | **IN_PROGRESS** | Canonical reason on all 3 paths + stale `pos_cyc` regression test |
| **W24-03** | Split **`mlr-v1` (`09184c83`)** out of `#364` into clean signed linear PR to `main` | Local / Spark | QUEUED | Replaces faulty `ndcg_at_k` on `origin/main` cleanly |
| **W24-04** | PR hygiene: `#369` (opt-in `D-06`), `#368` (3 float tie sites), close `#295`, `#376` to BHS 100 | Local / Spark | QUEUED | Signed linear commits, BHS 100 on landing PRs |

## P1 — High-Upside Untested Lanes (Sparks + Colab Ultra)

| ID | Lane | Task | Target | Status | Preregistered Goal |
|---|---|---|---|---|---|
| **W24-05** | Lane 8 (`#49`) | Bundle #49 live 3-family cold-start with **7B–14B open coders** | `colab:A100` (~3–5 CU) | **READY_TO_LAUNCH** | `>= 30/36` overall, `>= 8/12` per family, `0` leaks |
| **W24-06** | Lane 1 (`W22-15`) | `0.5B` extraction on **vLLM / llama.cpp FP8/INT4** + deterministic pre-filter *(keep 3.4 GB weights!)* | `gx10-6724` / `colab:L4` | QUEUED | `G1 <= 1500 ms` (stretch `<= 250 ms`), `G2` pass, real `G3` speedup |
| **W24-07** | Lane 2 (`W22-05`/`#55`) | Fix `classify_high_similarity_pair` acronym bug + cyclic/multi-hop/deletion benchmark vs SQLite & **PostgreSQL 17** | `gx10-6724` / `gx10-37c3` | QUEUED | Restore OSV/SEC echo recall (`>> 0.99%`), resolve `825` vs `962`, test `100–10k` cyclic graphs |
| **W24-08** | Lane 3 (`W22-07`/`W22-10`) | Graded provenance sweep $\lambda \in \{0.0, 0.05, 0.10, 0.25, 0.50\}$ + **`W22-10` `N3` swarm threshold** | `gx10-6724` | QUEUED | Map crossover $\lambda^*$ (extending `18/18` win at $\lambda=0.25$) + unlock `N3` |
| **W24-09** | Lane 4 (`W22-04`/`W22-06`) | 12-domain online query drift (`ZeroMu` EMA vs frozen `CenteredProcrustes`) + **CSLS + margin-triggered adaptive backfill** + **`W22-06(b)`** (3 more encoders) | `colab:L4/A100` (~5–10 CU) | QUEUED | Paired 12-domain drift CI; beat uniform 20% backfill on `#50/#54`; finish `W22-06(b)` |
| **W24-10** | Lane 6 (`W22-18`/`#53`) | Harvest `N >= 150` real public-corpus sibling benigns + Unicode NFKC/confusable defense + **live `#53` gateway** | `colab:L4` + `gx10-6724` | QUEUED | `R2b < 5%`, `live_pm6_gateway_executed=True`, `CP_UB95(FPR) <= 2.0%` |

## P2 — Formal Methods & Exploratory Mechanistic / SSM Lanes

| ID | Lane | Task | Target | Status | Goal |
|---|---|---|---|---|---|
| **W24-11** | Lane 5 (`W22-11`) | Large-graph (`50–1,000` node) deletion stress + **Lean 4 `T-Rs` (`CW-1` + deletion soundness)** | `gx10-6724` / `gx10-37c3` | QUEUED | `>= 100k` large-graph checkpoints `0` mismatches + `nanoda`-verified Lean 4 proof |
| **W24-12** | Lane 7a (`#52`) | Multi-layer (`k=4,8,16`, layers 11–18) causal-gradient `QSCCI` restoration vs `Top-K SAE` & `CAA` | `colab:L4/A100` (~3–5 CU) | QUEUED | Test multi-site causal restoration + IFEval KL collateral |
| **W24-13** | Lane 7b (`W22-09`) | Exploratory `T-Rn` test on frozen **linear-attention / Mamba-2 / DeltaNet** recurrent state | `colab:A100` (~3–5 CU) | QUEUED | True compressed-state retraction vs segment-tree replay |
| **W24-14** | `W22-08` | Independent blind second coding of `F-W8` and `F-R5` defect lists | Spark / local CPU | QUEUED | Settle `0.82/0.85` vs `0.80` overlap fragility |

## P3 — Human Time (Strictly Last)

| ID | Legacy ID | Task | Target | Status |
|---|---|---|---|---|
| **W24-15** | `W22-13` | Independent human grading of `pm4_human_audit_100_blind_sheet.csv` (~6 human hours) | Owner / External Grader | **QUEUED_LAST** |
| **W24-16** | `W22-14` | External workload-owner trial (`#63` Search Operator or `#64` Records Owner) | Owner + External Partner | **QUEUED_LAST** |
