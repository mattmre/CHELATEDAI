# Flagship #2 — External Real-World Benchmark: ICLR 2025 LongMemEval-S (`nemotron-lightning`)

- **Git SHA:** `0626dd1dcd7d8ede280695e9552a5a5f75f13c37`
- **Dataset:** `xiaowu0162/longmemeval-cleaned` (`longmemeval_s_cleaned.json`: ALL `78` `knowledge-update` + `50` `single-session-user` clean controls = `128` real multi-session benchmarks)
- **Model:** `nemotron-lightning` @ `http://127.0.0.1:8890/v1` (strictly observable retrieval: zero `has_answer` / `answer_session_ids` / gold metadata)
- **Wall time:** `446.35s`

## 1. Regime A — Unmodified LongMemEval-S (Zero Haystack Edits)

| Arm | Knowledge-Update Acc ↑ | KU Latest-Sess Recall@5 ↑ | Clean Control Acc ↑ | Overall Acc ↑ | Prompt Toks | Mean Lat (ms) | API Errs |
|---|---:|---:|---:|---:|---:|---:|---:|
| `M0_bm25_flat` | 0.7436 | 0.9872 | 0.8800 | 0.7969 | 881.7 | 2134.3 | 0 |
| `M1_hybrid_flat` | 0.7179 | 0.9744 | 0.9000 | 0.7891 | 894.5 | 1983.5 | 0 |
| `M2_scalemas_unsigned` | 0.7436 | 0.9744 | 0.8800 | 0.7969 | 897.8 | 1975.0 | 0 |
| `M3_recency_hybrid` | 0.7051 | 0.9744 | 0.8600 | 0.7656 | 894.4 | 1914.8 | 0 |
| **`M4_iq04_temporal_supersession`** | **0.6923** | 0.9872 | 0.8400 | **0.7500** | 892.7 | 2142.5 | 0 |

**Paired Bootstrap (`knowledge-update`, `n=78`):**
- `M4_vs_M0_bm25_flat`: `Δ = -0.0513` (`LCB95 = -0.1538`)
- `M4_vs_M1_hybrid_flat`: `Δ = -0.0256` (`LCB95 = -0.1410`)
- `M4_vs_M2_scalemas_unsigned`: `Δ = -0.0513` (`LCB95 = -0.1667`)
- `M4_vs_M3_recency_hybrid`: `Δ = -0.0128` (`LCB95 = -0.1282`)

## 2. Regime B — LongMemEval-S Multi-Session Echo-Persistence Stress (6 Unlinked Earlier Echoes)

| Arm | Knowledge-Update Acc ↑ | KU Latest-Sess Recall@5 ↑ | Clean Control Acc ↑ | Overall Acc ↑ | Prompt Toks | Mean Lat (ms) | API Errs |
|---|---:|---:|---:|---:|---:|---:|---:|
| `M0_bm25_flat` | 0.4872 | 0.6667 | 0.8800 | 0.6406 | 793.5 | 1914.2 | 0 |
| `M1_hybrid_flat` | 0.4615 | 0.6667 | 0.9200 | 0.6406 | 794.7 | 1799.3 | 0 |
| `M2_scalemas_unsigned` | 0.3205 | 0.4231 | 0.8600 | 0.5312 | 775.7 | 1838.6 | 0 |
| `M3_recency_hybrid` | 0.5000 | 0.7051 | 0.8800 | 0.6484 | 791.4 | 1799.6 | 0 |
| **`M4_iq04_temporal_supersession`** | **0.6923** | 0.9872 | 0.8600 | **0.7578** | 887.5 | 1904.6 | 0 |

**Paired Bootstrap (`knowledge-update`, `n=78`):**
- `M4_vs_M0_bm25_flat`: `Δ = +0.2051` (`LCB95 = +0.0513`)
- `M4_vs_M1_hybrid_flat`: `Δ = +0.2308` (`LCB95 = +0.0769`)
- `M4_vs_M2_scalemas_unsigned`: `Δ = +0.3718` (`LCB95 = +0.2176`)
- `M4_vs_M3_recency_hybrid`: `Δ = +0.1923` (`LCB95 = +0.0385`)

