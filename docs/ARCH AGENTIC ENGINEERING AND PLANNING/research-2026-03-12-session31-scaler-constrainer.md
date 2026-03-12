---
name: Session 31 Research - Scaler-Constrainer / BoundedAdapter
description: Architecture analysis for BoundedAdapter wrapper addressing INT8 quantization noise floor and correction budget
type: research
date: 2026-03-12
session: 31
---

# Scaler-Constrainer / BoundedAdapter Architecture

## Key Findings

### Correction Magnitude Analysis
| Adapter | Init Correction Norm | INT8 Step Size | Ratio | Visible? |
|---------|---------------------|----------------|-------|----------|
| MLP (std=0.001) | ~7e-6 | ~0.0012 | 0.006x | No |
| Low-rank (post-fix) | ~4e-4 | ~0.0012 | 0.33x | Marginal |
| Procrustes | ~0.02 | ~0.0012 | 17x | Yes |

### INT8 Quantization Interaction
- Configured at `antigravity_engine.py:111-119` with `ScalarQuantization(INT8, quantile=0.99)`
- Vectors upserted to Qdrant after adapter training are quantized for search index
- Corrections below half-step (~0.0006) are rounded away and invisible
- MLP corrections are 85x below minimum detectable change

### BoundedAdapter Design (Implemented in PR #100)
- Wrapper class (decorator pattern) around any adapter type
- **Correction budget**: L2 norm clamp to epsilon (prevents divergence)
- **Per-dimension scaling**: `dim_scale` parameter learns which dimensions to correct
- **Minimum correction floor**: ensures corrections exceed INT8 noise floor
- **Factory integration**: `create_adapter(bounded=True, min_correction=0.01, max_correction=0.5)`
- 3 config presets: conservative, balanced, aggressive

### Design Decisions
1. Wrapper not base class — zero risk to existing 986 tests
2. Correction extracted as `inner_out - x` in normalized space
3. Per-dimension scaling via learnable parameter (initialized to 1.0)
4. Floor enforcement preserves correction direction while boosting magnitude
5. Regularization combines inner adapter reg + scale deviation penalty

### Connection to Dual-Hemisphere Vision
- Implements the scaler-constrainer from architecture vision
- `epsilon` = correction budget preventing divergence
- `dim_scale` = per-dimension scaling addressing LIMIT capacity bound
- Adapter-agnostic wrapping = uniform correction interface for all route types

## References
- Drift-Adapter (EMNLP 2025) — DSM improved recall 95-97% to 98-99%
- LIMIT (DeepMind, arXiv:2508.21038) — fixed-dim capacity ceiling
- GAM-RAG (March 2026) — Kalman-gain uncertainty-aware updates
- Session 30 Bug 6: INT8 quantization eating corrections
