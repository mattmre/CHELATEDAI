# Spherical Distillation & 3rd-Order Cumulant Strictly Additive Ablation (`mlr-v1`)

- **Git SHA:** `872db009878fb398cc8e377433592c4db8c89659`
- **Wall time:** `274.31s`

## 1. Single-Hop Strictly Additive Zero-Write Ladder (REPORT nDCG@10, Mean Across 3 Seeds)

| Dataset | Swap | A0 Uncent Proc | A1 Ridge | A1b Ridge+CSLS | A2 Centered Proc (+1st) | A3 Whitened (+2nd) | A4 +3rd Cumulant | A5 Sph-KL | A6 Sph-KL+3rd | A7a KL+CSLS | A7b KL+3rd+CSLS | Native E0 | Full C2O |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SciFact | E0_to_E1_samedim | 0.4558 | 0.6219 | 0.6131 | 0.6170 | 0.6364 | 0.6087 | 0.6139 | 0.6139 | 0.6152 | 0.6137 | 0.6618 | 0.7258 |
| SciFact | E0_to_E3_crossdim | 0.5939 | 0.5916 | 0.5918 | 0.5951 | 0.6049 | 0.5794 | 0.5903 | 0.5904 | 0.5853 | 0.5862 | 0.6618 | 0.6543 |
| NFCorpus | E0_to_E1_samedim | 0.1849 | 0.2239 | 0.2206 | 0.2375 | 0.2386 | 0.2292 | 0.2381 | 0.2381 | 0.2315 | 0.2316 | 0.2696 | 0.2946 |
| NFCorpus | E0_to_E3_crossdim | 0.2293 | 0.2104 | 0.2135 | 0.2289 | 0.2341 | 0.2183 | 0.2289 | 0.2294 | 0.2210 | 0.2210 | 0.2696 | 0.2847 |
| FiQA2018 | E0_to_E1_samedim | 0.1142 | 0.2649 | 0.2988 | 0.2675 | 0.2998 | 0.2488 | 0.2747 | 0.2747 | 0.3059 | 0.3050 | 0.3456 | 0.3596 |
| FiQA2018 | E0_to_E3_crossdim | 0.3248 | 0.3085 | 0.3338 | 0.3188 | 0.3378 | 0.2915 | 0.3019 | 0.3023 | 0.3248 | 0.3251 | 0.3456 | 0.4600 |

## 2. Exact 2x2 Factorial: {No-KL, KL} x {No-3rd-Order, 3rd-Order} (Mean Across 3 Seeds)

| Dataset | Swap | Base (A2) | +3rd Only (A4−A2) | +KL Only (A5−A2) | +3rd over KL (A6−A5) | Interaction Synergy |
|---|---|---:|---:|---:|---:|---:|
| SciFact | E0_to_E1_samedim | 0.6170 | -0.0083 | -0.0031 | +0.0000 | +0.0083 |
| SciFact | E0_to_E3_crossdim | 0.5951 | -0.0156 | -0.0048 | +0.0001 | +0.0158 |
| NFCorpus | E0_to_E1_samedim | 0.2375 | -0.0082 | +0.0006 | -0.0000 | +0.0082 |
| NFCorpus | E0_to_E3_crossdim | 0.2289 | -0.0105 | +0.0000 | +0.0006 | +0.0111 |
| FiQA2018 | E0_to_E1_samedim | 0.2675 | -0.0187 | +0.0072 | +0.0000 | +0.0187 |
| FiQA2018 | E0_to_E3_crossdim | 0.3188 | -0.0273 | -0.0170 | +0.0005 | +0.0278 |

## 3. Multi-Hop Composition Chain (E0 -> E1 -> E2 -> E3, Mean Across Anchor Budgets & Seeds)

| Dataset | Hop | Comp Uncent Proc | Comp Ridge | Comp Centered Proc (+1st) | Comp Sph-KL | Comp Sph-KL+3rd |
|---|---:|---:|---:|---:|---:|---:|
| SciFact | 1 | 0.3953 | 0.5646 | 0.5670 | 0.5663 | 0.5646 |
| SciFact | 2 | 0.1037 | 0.4314 | 0.3679 | 0.3795 | 0.3778 |
| SciFact | 3 | 0.4597 | 0.3721 | 0.4400 | 0.3598 | 0.3584 |
| NFCorpus | 1 | 0.1714 | 0.2133 | 0.2281 | 0.2166 | 0.2165 |
| NFCorpus | 2 | 0.0763 | 0.1517 | 0.1495 | 0.1527 | 0.1528 |
| NFCorpus | 3 | 0.1843 | 0.1542 | 0.1844 | 0.1539 | 0.1543 |
| FiQA2018 | 1 | 0.0789 | 0.2174 | 0.2269 | 0.2280 | 0.2276 |
| FiQA2018 | 2 | 0.0022 | 0.1319 | 0.1003 | 0.1071 | 0.1062 |
| FiQA2018 | 3 | 0.1925 | 0.1318 | 0.1742 | 0.1068 | 0.1069 |

## 4. Selective Backfill & Quantile Calibration Additive Ladder (REPORT nDCG@10)

| Dataset | Swap | Budget | B0 Night-1 Z-Score+FastFill | B1 Sph-KL+Quantile+FastFill | B2 +Sph-Distortion Order | B3 +Exposure x Distortion |
|---|---|---:|---:|---:|---:|---:|
| SciFact | E0_to_E1_samedim | 10% | 0.6007 | 0.6397 | 0.6173 | 0.6403 |
| SciFact | E0_to_E1_samedim | 20% | 0.6017 | 0.6434 | 0.6312 | 0.6534 |
| SciFact | E0_to_E3_crossdim | 10% | 0.6066 | 0.6078 | 0.5948 | 0.6055 |
| SciFact | E0_to_E3_crossdim | 20% | 0.6024 | 0.6110 | 0.6045 | 0.6040 |
| NFCorpus | E0_to_E1_samedim | 10% | 0.2265 | 0.2403 | 0.2454 | 0.2430 |
| NFCorpus | E0_to_E1_samedim | 20% | 0.2295 | 0.2418 | 0.2462 | 0.2491 |
| NFCorpus | E0_to_E3_crossdim | 10% | 0.2118 | 0.2397 | 0.2398 | 0.2401 |
| NFCorpus | E0_to_E3_crossdim | 20% | 0.2206 | 0.2432 | 0.2510 | 0.2480 |
| FiQA2018 | E0_to_E1_samedim | 10% | 0.2446 | 0.2864 | 0.2822 | 0.2890 |
| FiQA2018 | E0_to_E1_samedim | 20% | 0.2291 | 0.2958 | 0.2878 | 0.2938 |
| FiQA2018 | E0_to_E3_crossdim | 10% | 0.3402 | 0.3177 | 0.3244 | 0.3235 |
| FiQA2018 | E0_to_E3_crossdim | 20% | 0.3670 | 0.3407 | 0.3312 | 0.3420 |
