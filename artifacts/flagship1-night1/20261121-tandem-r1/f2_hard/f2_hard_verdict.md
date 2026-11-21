# Flagship #2 Stage 2C — Hard Adversarial Live Nemotron 3.5 Stress Test (No Role Tags, Flood > Top-5)

- **Git SHA:** `0fe94a52d5aae466a725dd56a4487f7ddf237e5e`
- **Model:** `nemotron-lightning` @ `http://127.0.0.1:8890/v1` (`600` live calls, `0` role-tag hints)
- **Live `C4` vs `C3_ordinary_bitemporal_tms`:** rel. stale-error reduction = **`100.0%`** (`abs=+0.5500`, `LCB95=+0.4583`)
- **Live `C4` vs `C1_scalemas_unsigned`:** abs stale-error drop = **`+0.6667`** (`LCB95=+0.5833`)

## 1. Overall Live Adversarial Results

| Arm | Post Stale/False Error ↓ | Post Accuracy ↑ | Hard Clean-Trap Acc ↑ | Overall Acc ↑ | Prompt Toks | Mean Lat (ms) |
|---|---:|---:|---:|---:|---:|---:|
| `C0_bm25_flat` | **0.6500** | 0.3500 | 1.0000 | 0.4800 | 312.2 | 908.6 |
| `C1_scalemas_unsigned` | **0.6667** | 0.3333 | 1.0000 | 0.4667 | 312.2 | 816.8 |
| `C3_ordinary_bitemporal_tms` | **0.5500** | 0.4500 | 1.0000 | 0.5600 | 274.6 | 779.1 |
| `C4_iq04_candidate` | **0.0000** | 1.0000 | 1.0000 | 1.0000 | 164.1 | 685.8 |

## 2. Per-Family Live Stale/False Error Rate (Lower Is Better)

| Adversarial Family | C0 BM25 | C1 ScaleMAS Unsigned | C3 Ordinary Bitemp TMS | **C4 IQ-04 Candidate** |
|---|---:|---:|---:|---:|
| `flood_12_unlinked_paraphrase_copies` | 1.0000 | 1.0000 | 1.0000 | **0.0000** |
| `flood_10_linked_popular_error_corrected` | 0.4000 | 0.4333 | 0.0000 | **0.0000** |
| `hard_clean_trap_lexical_overlap_valid_update` | 0.0000 | 0.0000 | 0.0000 | **0.0000** |
| `unlinked_downstream_synthesis_from_retracted` | 1.0000 | 1.0000 | 1.0000 | **0.0000** |
| `erroneous_retraction_then_reinstated_flood` | 0.2000 | 0.2333 | 0.2000 | **0.0000** |
