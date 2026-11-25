# Flagship #2 Unified Zero-Shortcut (`EK5 v2` WFS) + Fresh Held-Out ACL 2024 LoCoMo Verdict

- **Git SHA:** `18973789615ecab1268a77ebd3c69a283607241c`
- **Model:** `nemotron-lightning` @ `http://127.0.0.1:8890/v1`
- **Exact Call Accounting:** Part A `900` answer calls (`0` judge calls); Part B `1680` answer calls + `576` cached deterministic judge calls
- **Wall time:** `243.99s`

## Part A — Zero-Timestamp-Shortcut Unified `EK5 v2` WFS Live Suite (ZERO Role Tags)

| Arm | Overall Acc ↑ | Stale-Newest After Corr ↑ | Repeated Truth vs Late Rumor ↑ | Mutual Refute Abstain ↑ | 12-Copy Flood ↑ | Unlinked Synth ↑ | Prompt Toks |
|---|---:|---:|---:|---:|---:|---:|---:|
| `C0_bm25_flat` | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 256.2 |
| `C1_scalemas_bounded` | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 256.2 |
| `C2_latest_timestamp` | 0.9933 | 1.0000 | 1.0000 | 0.9667 | 1.0000 | 1.0000 | 261.6 |
| `C2b_bm25_jaccard_dedup` | 0.8000 | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 216.5 |
| `C3_ordinary_bitemporal_tms` | 0.9867 | 1.0000 | 1.0000 | 0.9333 | 1.0000 | 1.0000 | 261.4 |
| **`C4_v2_iq04_ek5_wfs`** | **1.0000** | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 145.9 |

**Part A Paired Bootstrap (`n=150`, two-sided 95% CI lower bound `lcb_95`):**
- `C4v2_vs_C0_bm25_flat`: `Δ = +0.0000` (`LCB95 = +0.0000`)
- `C4v2_vs_C1_scalemas_bounded`: `Δ = +0.0000` (`LCB95 = +0.0000`)
- `C4v2_vs_C2_latest_timestamp`: `Δ = +0.0067` (`LCB95 = +0.0000`)
- `C4v2_vs_C2b_bm25_jaccard_dedup`: `Δ = +0.2000` (`LCB95 = +0.1333`)
- `C4v2_vs_C3_ordinary_bitemporal_tms`: `Δ = +0.0133` (`LCB95 = +0.0000`)

## Part B.1 — Fresh Held-Out LoCoMo Regime A (Unmodified, 80 Answerable + 40 Abstention)

| Arm | Answerable Acc ↑ | Answerable Ev Recall@5 ↑ | Abstention Acc ↑ | Overall Acc ↑ | Prompt Toks | API Errs |
|---|---:|---:|---:|---:|---:|---:|
| `M0_bm25_flat` | 0.2500 | 0.4875 | 0.3000 | 0.2667 | 345.1 | 0 |
| `M1_hybrid_flat` | 0.2500 | 0.6000 | 0.2500 | 0.2500 | 341.4 | 0 |
| `M1b_bm25_jaccard_dedup` | 0.2375 | 0.4875 | 0.3250 | 0.2667 | 345.1 | 0 |
| `M2_scalemas_unsigned` | 0.2125 | 0.5500 | 0.2750 | 0.2333 | 339.3 | 0 |
| `M3_recency_hybrid` | 0.2625 | 0.5375 | 0.3000 | 0.2750 | 343.4 | 0 |
| **`M4c_cross_session_collapse`** | **0.2375** | 0.5375 | 0.3000 | **0.2583** | 335.9 | 0 |
| **`M4d_cross_session_plus_tag`** | **0.2500** | 0.5375 | 0.3000 | **0.2667** | 352.8 | 0 |

**Answerable-Only Paired Bootstrap (`n=80`):**
- `M4c_vs_M0_bm25_flat`: `Δ = -0.0125` (`LCB95 = -0.1125`)
- `M4c_vs_M1_hybrid_flat`: `Δ = -0.0125` (`LCB95 = -0.0875`)
- `M4c_vs_M1b_bm25_jaccard_dedup`: `Δ = +0.0000` (`LCB95 = -0.1000`)
- `M4c_vs_M2_scalemas_unsigned`: `Δ = +0.0250` (`LCB95 = -0.0628`)
- `M4c_vs_M3_recency_hybrid`: `Δ = -0.0250` (`LCB95 = -0.1125`)
- `M4d_vs_M0_bm25_flat`: `Δ = +0.0000` (`LCB95 = -0.1000`)
- `M4d_vs_M1_hybrid_flat`: `Δ = +0.0000` (`LCB95 = -0.0750`)
- `M4d_vs_M1b_bm25_jaccard_dedup`: `Δ = +0.0125` (`LCB95 = -0.0875`)
- `M4d_vs_M2_scalemas_unsigned`: `Δ = +0.0375` (`LCB95 = -0.0625`)
- `M4d_vs_M3_recency_hybrid`: `Δ = -0.0125` (`LCB95 = -0.1000`)

## Part B.2 — Fresh Held-Out LoCoMo Regime B (Cross-Session Echo Stress)

| Arm | Answerable Acc ↑ | Answerable Ev Recall@5 ↑ | Abstention Acc ↑ | Overall Acc ↑ | Prompt Toks | API Errs |
|---|---:|---:|---:|---:|---:|---:|
| `M0_bm25_flat` | 0.1250 | 0.2750 | 0.3500 | 0.2000 | 367.5 | 0 |
| `M1_hybrid_flat` | 0.1750 | 0.3750 | 0.4500 | 0.2667 | 352.7 | 0 |
| `M1b_bm25_jaccard_dedup` | 0.2375 | 0.5000 | 0.3000 | 0.2583 | 353.4 | 0 |
| `M2_scalemas_unsigned` | 0.1500 | 0.2500 | 0.5000 | 0.2667 | 352.3 | 0 |
| `M3_recency_hybrid` | 0.1875 | 0.4125 | 0.4000 | 0.2583 | 355.1 | 0 |
| **`M4c_cross_session_collapse`** | **0.2750** | 0.5375 | 0.2500 | **0.2667** | 342.1 | 0 |
| **`M4d_cross_session_plus_tag`** | **0.2750** | 0.5375 | 0.2250 | **0.2583** | 358.8 | 0 |

**Answerable-Only Paired Bootstrap (`n=80`):**
- `M4c_vs_M0_bm25_flat`: `Δ = +0.1500` (`LCB95 = +0.0750`)
- `M4c_vs_M1_hybrid_flat`: `Δ = +0.1000` (`LCB95 = +0.0250`)
- `M4c_vs_M1b_bm25_jaccard_dedup`: `Δ = +0.0375` (`LCB95 = -0.0500`)
- `M4c_vs_M2_scalemas_unsigned`: `Δ = +0.1250` (`LCB95 = +0.0125`)
- `M4c_vs_M3_recency_hybrid`: `Δ = +0.0875` (`LCB95 = +0.0000`)
- `M4d_vs_M0_bm25_flat`: `Δ = +0.1500` (`LCB95 = +0.0625`)
- `M4d_vs_M1_hybrid_flat`: `Δ = +0.1000` (`LCB95 = +0.0250`)
- `M4d_vs_M1b_bm25_jaccard_dedup`: `Δ = +0.0375` (`LCB95 = -0.0500`)
- `M4d_vs_M2_scalemas_unsigned`: `Δ = +0.1250` (`LCB95 = +0.0125`)
- `M4d_vs_M3_recency_hybrid`: `Δ = +0.0875` (`LCB95 = +0.0000`)
