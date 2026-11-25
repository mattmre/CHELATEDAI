# Precipitated Random / Non-Distributional vs. Manifold-Precipitated Anchor Sets (`mlr-v1`)

- **Git SHA:** `be717ae5c7faf80c67561a1daafc3aa6de17d90e`
- **Wall time:** `688.97s`

## 1. Single-Hop Mean REPORT nDCG@10 Across Anchor Budgets {100, 250, 500, 1000} (Centered Ridge / Nystrom)

| Dataset | Swap | P0 Uniform i.i.d. | P1 User: Synthetic Random | P2 User: Tail Precip | P3 User: 50/50 Mixed | P4 Rec: k-Center | P5 Rec: Leverage D-Opt | P6 Rec: k-Means++ | P7 Rec: Nystrom Kernel |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| SciFact | E0_to_E1_samedim | 0.4413 | 0.2660 | 0.3820 | 0.4017 | 0.4614 | 0.4405 | 0.4525 | 0.1984 |
| SciFact | E0_to_E3_crossdim | 0.4110 | 0.2337 | 0.2793 | 0.3583 | 0.4487 | 0.4118 | 0.4234 | 0.2317 |
| NFCorpus | E0_to_E1_samedim | 0.1820 | 0.1387 | 0.1591 | 0.1722 | 0.1822 | 0.1809 | 0.1832 | 0.1627 |
| NFCorpus | E0_to_E3_crossdim | 0.1767 | 0.1112 | 0.1317 | 0.1529 | 0.1649 | 0.1749 | 0.1685 | 0.1413 |
| FiQA2018 | E0_to_E1_samedim | 0.1603 | 0.0406 | 0.0598 | 0.1342 | 0.1226 | 0.1536 | 0.1619 | 0.0410 |
| FiQA2018 | E0_to_E3_crossdim | 0.1668 | 0.0331 | 0.0974 | 0.1365 | 0.1653 | 0.1666 | 0.1576 | 0.0325 |

## 2. Single-Hop Mean REPORT nDCG@10 — Centered Procrustes Mapper

| Dataset | Swap | P0 Uniform i.i.d. | P1 User: Synthetic Random | P2 User: Tail Precip | P3 User: 50/50 Mixed | P4 Rec: k-Center | P5 Rec: Leverage D-Opt | P6 Rec: k-Means++ |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| SciFact | E0_to_E1_samedim | 0.4298 | 0.2532 | 0.3676 | 0.4004 | 0.4582 | 0.4297 | 0.4458 |
| SciFact | E0_to_E3_crossdim | 0.4489 | 0.3070 | 0.2979 | 0.4126 | 0.4660 | 0.4351 | 0.4521 |
| NFCorpus | E0_to_E1_samedim | 0.1851 | 0.1225 | 0.1641 | 0.1727 | 0.1852 | 0.1864 | 0.1880 |
| NFCorpus | E0_to_E3_crossdim | 0.1907 | 0.1345 | 0.1459 | 0.1677 | 0.1898 | 0.1897 | 0.1872 |
| FiQA2018 | E0_to_E1_samedim | 0.1613 | 0.0229 | 0.0619 | 0.1268 | 0.1182 | 0.1495 | 0.1618 |
| FiQA2018 | E0_to_E3_crossdim | 0.1849 | 0.0497 | 0.1233 | 0.1545 | 0.1822 | 0.1855 | 0.1863 |

## 3. Multi-Hop Hop-3 (E3 -> E2 -> E1 -> E0) Composed Centered Procrustes (Mean Over Budgets {250, 500} x 3 Seeds)

| Dataset | P0 Uniform i.i.d. | P1 User: Synthetic | P2 User: Tail | P3 User: 50/50 Mix | P4 Rec: k-Center | P5 Rec: Leverage | P6 Rec: k-Means++ |
|---|---:|---:|---:|---:|---:|---:|---:|
| SciFact | 0.2595 | 0.0706 | 0.2015 | 0.1953 | 0.3146 | 0.2660 | 0.2909 |
| NFCorpus | 0.1328 | 0.0561 | 0.1203 | 0.1071 | 0.1350 | 0.1439 | 0.1461 |
| FiQA2018 | 0.0683 | 0.0003 | 0.0060 | 0.0377 | 0.0466 | 0.0648 | 0.0653 |
