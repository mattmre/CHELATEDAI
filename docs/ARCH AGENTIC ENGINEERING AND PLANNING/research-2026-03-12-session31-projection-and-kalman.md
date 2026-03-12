---
name: Session 31 Research - DimensionProjection Training & Kalman-Gain LR
description: Technical research for DimensionProjection training fix and Kalman-gain adaptive sedimentation LR
type: research
date: 2026-03-12
session: 31
---

# DimensionProjection Training & Kalman-Gain Sedimentation LR

## Part A: DimensionProjection Fix

### Where Defined
- `teacher_distillation.py:24-58` — nn.Module with Linear(teacher_dim, student_dim)
- Near-identity init: eye * 0.999 + randn * 0.001

### Three Independent Blockers (Why It Never Trained)
1. `project_numpy()` wraps forward() in `torch.no_grad()` — kills gradient flow
2. Neither optimizer includes projection params — only `self.adapter.parameters()`
3. Projection applied during target preprocessing, not in differentiable forward pass

### Fix (Implemented in PR #99)
- Added `project_tensor()` method preserving gradients
- Included `teacher_helper._projection.parameters()` in optimizer at both training sites
- 5 new tests verifying gradient flow and weight updates

## Part B: Kalman-Gain Adaptive LR

### Current State
- Flat `optim.Adam(lr=learning_rate)` with no scheduling
- ConvergenceMonitor: binary stop/continue only, no LR modulation
- TeacherWeightScheduler: modulates blend ratio, not LR
- No existing adaptive LR mechanism in the codebase

### Design (Implemented in PR #102)
- `KalmanLRScheduler` class with Kalman-gain analogy:
  - Process noise Q: expected variation in corrections
  - Measurement noise R: estimated from loss variance
  - Gain K = Q / (Q + R): how much to trust new corrections
  - Effective LR = base_lr * K
- High variance → low K → conservative LR
- Low variance → high K → aggressive LR
- Complements ConvergenceMonitor (operates between cycles, not within)
- 3 config presets: conservative, balanced, aggressive
- 30 new tests

### Relationship to Existing Adaptive Stack
| Mechanism | What It Adapts | Timescale |
|-----------|---------------|-----------|
| TeacherWeightScheduler | Blend ratio | Per-epoch |
| ConvergenceMonitor | Training duration | Per-epoch |
| AdaptiveMargin | Triplet margin | Per-query |
| **KalmanLRScheduler** | **Learning rate** | **Per-epoch** |

## References
- GAM-RAG (March 2026) — Kalman-gain uncertainty-aware updates
- SEKF (March 2026) — Selective parameter updates
- Session 29: LR=0.01 sweet spot
- Session 30 Bug 3: DimensionProjection untrained
