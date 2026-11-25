# Flagship #2 `EK5 v2` (3-Valued Well-Founded Semantics) — Million-Node Scaling & Dense-Cascade Phase Transition

- **Git SHA:** `18973789615ecab1268a77ebd3c69a283607241c`
- **1,000-Graph Tangled Parity Fuzzer:** **`1000/1000` (`100.0%`) exact bitwise parity** (`325` graphs with `CONTESTED` mutual-refutation/derivation cycles, vs. `60.7%` pre-fix `b37e6a55`)
- **Wall time:** `14.82s`

## 1. True Persistent Wall-Clock Scaling (Zero Full-Graph Scans Per Update)

| Total Nodes | Total Edges | Mean Dirty Nodes | Dirty Fraction | Median Inc (ms) | p95 Inc (ms) | Median Full WFS (ms) | **Real Wall Speedup** | Exact Parity |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 10,000 | 9,025 | 11.0 | 0.1100% | **0.0159** | 0.0266 | 11.50 | **723.5x** | `3/3` |
| 100,000 | 90,025 | 11.0 | 0.0110% | **0.0231** | 0.0507 | 218.41 | **9454.9x** | `3/3` |
| 1,000,000 | 900,015 | 11.0 | 0.0011% | **0.0275** | 0.0603 | 3325.25 | **120918.1x** | `1/1` |

## 2. Dense-Entanglement Hub Phase Transition (`N = 100,000` Nodes)

| Target Cascade Frac | Actual Dirty Nodes | Actual Dirty % | Inc Wall (ms) | Full WFS Wall (ms) | **Wall Speedup** | Exact Parity |
|---:|---:|---:|---:|---:|---:|---|
| 0.01% | 12 | 0.01% | 0.047 | 216.16 | **4609.02x** | `True` |
| 0.10% | 102 | 0.10% | 0.212 | 222.88 | **1051.34x** | `True` |
| 1.00% | 1,002 | 1.00% | 1.871 | 248.92 | **133.01x** | `True` |
| 5.00% | 5,002 | 5.00% | 33.391 | 224.10 | **6.71x** | `True` |
| 15.00% | 15,002 | 15.00% | 31.627 | 226.15 | **7.15x** | `True` |
| 35.00% | 35,002 | 35.00% | 79.020 | 249.20 | **3.15x** | `True` |
| 60.00% | 60,002 | 60.00% | 150.979 | 215.34 | **1.43x** | `True` |
