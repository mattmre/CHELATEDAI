# Flagship #2 — External Real-World Benchmark: ICLR 2025 LongMemEval-S (`nemotron-lightning`)

- **Git SHA:** `d0a2c4b8e745a3680cd6dbedea0123999af7960d`
- **Dataset:** `xiaowu0162/longmemeval-cleaned` (`longmemeval_s_cleaned.json`: ALL `78` `knowledge-update` + `50` `single-session-user` clean controls = `128` real multi-session benchmarks)
- **Model:** `nemotron-lightning` @ `http://127.0.0.1:8890/v1` (strictly observable retrieval: zero `has_answer` / `answer_session_ids` / gold metadata)
- **Wall time:** `362.38s`

## 1. Regime A — Unmodified LongMemEval-S (Zero Haystack Edits)

| Arm | Knowledge-Update Acc ↑ | KU Latest-Sess Recall@5 ↑ | Clean Control Acc ↑ | Overall Acc ↑ | Prompt Toks | Mean Lat (ms) | API Errs |
|---|---:|---:|---:|---:|---:|---:|---:|
| `M0_bm25_flat` | 0.7179 | 0.9872 | 0.8800 | 0.7812 | 881.7 | 2256.4 | 0 |
| `M1_hybrid_flat` | 0.7179 | 0.9744 | 0.9200 | 0.7969 | 894.5 | 2249.8 | 0 |
| `M2_scalemas_unsigned` | 0.7179 | 0.9744 | 0.8600 | 0.7734 | 897.8 | 2198.3 | 0 |
| `M3_recency_hybrid` | 0.7179 | 0.9744 | 0.8600 | 0.7734 | 894.4 | 2180.1 | 0 |
| **`M4_iq04_temporal_supersession`** | **0.6667** | 0.9872 | 0.8000 | **0.7188** | 892.7 | 2243.5 | 0 |
| `M4b_iq04_echo_collapse_ranked` | 0.7051 | 0.9872 | 0.8400 | 0.7578 | 896.2 | 2199.9 | 0 |

**Paired Bootstrap (`knowledge-update`, `n=78`):**
- `M4_vs_M0_bm25_flat`: `Δ = -0.0513` (`LCB95 = -0.1538`)
- `M4_vs_M1_hybrid_flat`: `Δ = -0.0513` (`LCB95 = -0.1538`)
- `M4_vs_M2_scalemas_unsigned`: `Δ = -0.0513` (`LCB95 = -0.1667`)
- `M4_vs_M3_recency_hybrid`: `Δ = -0.0513` (`LCB95 = -0.1538`)
- `M4b_vs_M0_bm25_flat`: `Δ = -0.0128` (`LCB95 = -0.0901`)
- `M4b_vs_M1_hybrid_flat`: `Δ = -0.0128` (`LCB95 = -0.1026`)
- `M4b_vs_M2_scalemas_unsigned`: `Δ = -0.0128` (`LCB95 = -0.1154`)
- `M4b_vs_M3_recency_hybrid`: `Δ = -0.0128` (`LCB95 = -0.1026`)

## 2. Regime B — LongMemEval-S Multi-Session Echo-Persistence Stress (6 Unlinked Earlier Echoes)

| Arm | Knowledge-Update Acc ↑ | KU Latest-Sess Recall@5 ↑ | Clean Control Acc ↑ | Overall Acc ↑ | Prompt Toks | Mean Lat (ms) | API Errs |
|---|---:|---:|---:|---:|---:|---:|---:|
| `M0_bm25_flat` | 0.4872 | 0.6667 | 0.9000 | 0.6484 | 793.5 | 2147.7 | 0 |
| `M1_hybrid_flat` | 0.4615 | 0.6667 | 0.9000 | 0.6328 | 794.7 | 2082.7 | 0 |
| `M2_scalemas_unsigned` | 0.3333 | 0.4231 | 0.8800 | 0.5469 | 775.7 | 2056.4 | 0 |
| `M3_recency_hybrid` | 0.4744 | 0.7051 | 0.8800 | 0.6328 | 791.4 | 2028.0 | 0 |
| **`M4_iq04_temporal_supersession`** | **0.6923** | 0.9872 | 0.8200 | **0.7422** | 887.5 | 2183.2 | 0 |
| `M4b_iq04_echo_collapse_ranked` | 0.7436 | 0.9872 | 0.8400 | 0.7812 | 875.0 | 2067.8 | 0 |

**Paired Bootstrap (`knowledge-update`, `n=78`):**
- `M4_vs_M0_bm25_flat`: `Δ = +0.2051` (`LCB95 = +0.0513`)
- `M4_vs_M1_hybrid_flat`: `Δ = +0.2308` (`LCB95 = +0.0769`)
- `M4_vs_M2_scalemas_unsigned`: `Δ = +0.3590` (`LCB95 = +0.2051`)
- `M4_vs_M3_recency_hybrid`: `Δ = +0.2179` (`LCB95 = +0.0641`)
- `M4b_vs_M0_bm25_flat`: `Δ = +0.2564` (`LCB95 = +0.1410`)
- `M4b_vs_M1_hybrid_flat`: `Δ = +0.2821` (`LCB95 = +0.1538`)
- `M4b_vs_M2_scalemas_unsigned`: `Δ = +0.4103` (`LCB95 = +0.2821`)
- `M4b_vs_M3_recency_hybrid`: `Δ = +0.2692` (`LCB95 = +0.1538`)

