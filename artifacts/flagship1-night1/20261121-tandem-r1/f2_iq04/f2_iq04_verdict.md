# Flagship #2 (`IQ-04` / `EK7` / `EK5`) vs. ScaleMAS Surrogate + Live Nemotron 3.5 Verdict

- **Git SHA:** `f03cc5e53f319609e6273d7f57f951c341d5a8e0`
- **Honest Comparator Label:** `C1_scalemas_unsigned` = `SCALEMAS_ADAPTED_SURROGATE` (BM25 + decaying unsigned citation boost).
- **EK5 Sparse Invalidation Gate:** `exact_parity=True`, `max_work_ratio=0.0497` (`<= 0.25`) → **`PASS`**
- **EK7 Surrogate Gate (800 REPORT lifecycles):** `C4` vs `C3` rel. stale-error reduction = **`100.0%`** (`LCB95=+0.1171`), clean-recall loss = `+0.00 pp`, p95 ratio = `1.43x` → **`PASS`**

## 1. Stage 2A — Deterministic 800-Lifecycle REPORT Audit (All 6 Arms)

| Arm | Post-Correction Stale/False Error ↓ | Post-Correction Accuracy ↑ | Clean Control Recall ↑ | p95 Latency (ms) |
|---|---:|---:|---:|---:|
| `C0_bm25_flat` | 0.8571 | 0.1429 | 1.0000 | 0.053 |
| `C1_scalemas_unsigned` | 0.8571 | 0.1429 | 1.0000 | 0.058 |
| `C2_recency_version_rag` | 0.8571 | 0.1429 | 1.0000 | 0.054 |
| `C3_ordinary_bitemporal_tms` | 0.1429 | 0.8571 | 1.0000 | 0.067 |
| `C4_iq04_candidate` | 0.0000 | 1.0000 | 1.0000 | 0.096 |
| `C5_oracle_upper_bound` | 0.0000 | 1.0000 | 1.0000 | 0.051 |

## 2. Per-Family Post-Correction Stale/False Error Rate (REPORT)

| Case Family | C0 BM25 | C1 ScaleMAS (Unsigned) | C2 Recency | C3 Ordinary Bitemp TMS | **C4 IQ-04 Candidate** | C5 Oracle |
|---|---:|---:|---:|---:|---:|---:|
| `early_popular_error_corrected` | 1.0000 | 1.0000 | 1.0000 | 0.0000 | **0.0000** | 0.0000 |
| `popular_valid_control` | 0.0000 | 0.0000 | 0.0000 | 0.0000 | **0.0000** | 0.0000 |
| `unpopular_valid_minority` | 1.0000 | 1.0000 | 1.0000 | 0.0000 | **0.0000** | 0.0000 |
| `refutation_supersession_chain` | 1.0000 | 1.0000 | 1.0000 | 0.0000 | **0.0000** | 0.0000 |
| `delayed_observation_as_of` | 1.0000 | 1.0000 | 1.0000 | 0.0000 | **0.0000** | 0.0000 |
| `erroneous_retraction_reinstated` | 0.0000 | 0.0000 | 0.0000 | 0.0000 | **0.0000** | 0.0000 |
| `hidden_common_cause_paraphrase_copy` | 1.0000 | 1.0000 | 1.0000 | 1.0000 | **0.0000** | 0.0000 |
| `missing_edge_and_cycle_stress` | 1.0000 | 1.0000 | 1.0000 | 0.0000 | **0.0000** | 0.0000 |

## 3. Stage 2B — Live Delivered-Answer Evaluation on `nemotron-lightning` (`120` Stratified REPORT Queries × 4 Arms)

- **Live `C4` vs `C3` relative stale-error reduction:** **`0.0%`** (`abs=+0.0000`, `LCB95=+0.0000`)
- **Live `C4` vs `C1_scalemas_unsigned` abs stale-error drop:** **`+0.0952`** (`LCB95=+0.0381`)

| Arm | Live Post Stale/False Error ↓ | Live Post Accuracy ↑ | Live Clean Accuracy ↑ | Overall Acc ↑ | API Errors | Mean Prompt Toks | Mean Lat (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|
| `C0_bm25_flat` | 0.2476 | 0.7524 | 1.0000 | 0.7833 | 0 | 355.1 | 840.9 |
| `C1_scalemas_unsigned` | 0.0952 | 0.8095 | 1.0000 | 0.8333 | 0 | 355.1 | 828.4 |
| `C3_ordinary_bitemporal_tms` | 0.0000 | 1.0000 | 1.0000 | 1.0000 | 0 | 243.2 | 766.5 |
| `C4_iq04_candidate` | 0.0000 | 1.0000 | 1.0000 | 1.0000 | 0 | 211.1 | 703.1 |
