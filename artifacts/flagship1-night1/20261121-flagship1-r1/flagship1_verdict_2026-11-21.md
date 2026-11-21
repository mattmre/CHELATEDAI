# Night-1 Flagship #1 Empirical Verdict (Unquarantined Complete-Qrels `mlr-v1`)

- **Git SHA:** `09184c83c09d55be4ea332bba0b33cc0ebae701d`
- **Old encoder retired premise:** `True`
- **Gate 1 (M3 Single-Hop Residual Science Gate):** **`CLOSED_BY_ORDINARY_LADDER`**
- **Gate 2 (Multi-Hop Lineage Accumulation Gate):** **`MULTIHOP_SIGNAL_SURVIVES`**
- **Gate 3 (TRA-1 Interlock & M4 QPP Router Gate):** **`TRA1_OR_M4_NEEDS_REVIEW`**

## 1. Single-Hop M3 Frontier Summary (REPORT Split, Budget <= 20%)

| Dataset | Swap | C2O nDCG@10 | Best Ordinary (<=20%) | Best Ord nDCG@10 | Residual Gap | Min Seed LCB95 | Survives (>=0.02 & LCB>0)? |
|---|---|---:|---|---:|---:|---:|---|
| SciFact | E0_to_E1_samedim | 0.7258 | `floor_bm25_hybrid` | 0.6833 | +0.0426 | -0.0277 | False |
| SciFact | E0_to_E3_crossdim | 0.6543 | `floor_bm25_hybrid` | 0.6790 | -0.0247 | -0.0903 | False |
| NFCorpus | E0_to_E1_samedim | 0.2946 | `floor_bm25_hybrid` | 0.3067 | -0.0121 | -0.0497 | False |
| NFCorpus | E0_to_E3_crossdim | 0.2847 | `floor_bm25_hybrid` | 0.3048 | -0.0201 | -0.0620 | False |
| FiQA2018 | E0_to_E1_samedim | 0.3596 | `floor_csls_k10` | 0.2988 | +0.0608 | +0.0258 | True |
| FiQA2018 | E0_to_E3_crossdim | 0.4600 | `map_plus_backfill_20` | 0.3771 | +0.0829 | +0.0518 | True |

## 2. Multi-Hop Lineage Accumulation at Hop m=3 (E3 -> E2 -> E1 -> E0)

| Dataset | Composed Procrustes − Composed Ridge | Direct Ridge − Composed Ridge | Cells |
|---|---:|---:|---:|
| SciFact | +0.0652 | +0.1597 | 12 |
| NFCorpus | +0.0300 | +0.0451 | 12 |
| FiQA2018 | +0.0510 | +0.1082 | 12 |

## 3. TRA-1 Interlock & M4 QPP Router Audit

- Hostile wrong-space negative cases: **8**, wrong-space executions: **0**
- Worst family TRA-1 minus fixed LCB97.5: **-0.0457**
- Mean TRA-1 minus authorization-filtered QPP router: **-0.0000**

| Dataset | Mutation Family | TRA-1 | QPP (M4) | Best Fixed | Native Re-embed | TRA-1 − QPP |
|---|---|---:|---:|---:|---:|---:|
| SciFact | query_swap_samedim | 0.6207 | 0.6077 | 0.6207 | 0.7258 | +0.0130 |
| SciFact | query_swap_crossdim | 0.5990 | 0.5969 | 0.5877 | 0.6543 | +0.0021 |
| SciFact | doc_vector_corruption | 0.2437 | 0.2379 | 0.2437 | 0.6618 | +0.0058 |
| SciFact | domain_home_cross | 0.6618 | 0.6618 | 0.6618 | 0.6618 | +0.0000 |
| SciFact | dtype_tier_quant | 0.6741 | 0.6749 | 0.6732 | 0.6618 | -0.0008 |
| SciFact | multihop_lineage_2hop | 0.5803 | 0.5803 | 0.5803 | 0.6825 | +0.0000 |
| NFCorpus | query_swap_samedim | 0.2216 | 0.2162 | 0.2216 | 0.2946 | +0.0054 |
| NFCorpus | query_swap_crossdim | 0.2084 | 0.2223 | 0.2309 | 0.2847 | -0.0139 |
| NFCorpus | doc_vector_corruption | 0.1133 | 0.1133 | 0.1133 | 0.2696 | +0.0000 |
| NFCorpus | domain_home_cross | 0.2696 | 0.2696 | 0.2696 | 0.2696 | +0.0000 |
| NFCorpus | dtype_tier_quant | 0.2682 | 0.2680 | 0.2682 | 0.2696 | +0.0001 |
| NFCorpus | multihop_lineage_2hop | 0.2097 | 0.2042 | 0.2097 | 0.2818 | +0.0055 |
| FiQA2018 | query_swap_samedim | 0.2646 | 0.2646 | 0.2646 | 0.3596 | +0.0000 |
| FiQA2018 | query_swap_crossdim | 0.3083 | 0.3266 | 0.3229 | 0.4600 | -0.0183 |
| FiQA2018 | doc_vector_corruption | 0.0774 | 0.0780 | 0.0774 | 0.3456 | -0.0006 |
| FiQA2018 | domain_home_cross | 0.3456 | 0.3456 | 0.3456 | 0.3456 | +0.0000 |
| FiQA2018 | dtype_tier_quant | 0.3473 | 0.3466 | 0.3473 | 0.3456 | +0.0006 |
| FiQA2018 | multihop_lineage_2hop | 0.2403 | 0.2395 | 0.2403 | 0.3445 | +0.0008 |

## 4. M2 Development Preflight Summary (90 cells)
- ADMIT cells: **15** / 90
- REFUSE cells: **75** / 90
