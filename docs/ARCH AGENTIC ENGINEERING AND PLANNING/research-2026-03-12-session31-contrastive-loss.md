---
name: Session 31 Research - Contrastive Loss for Sedimentation
description: Architecture analysis for switching sedimentation training from MSE to contrastive/InfoNCE loss
type: research
date: 2026-03-12
session: 31
---

# Contrastive Loss for Sedimentation Training

## Key Findings

### Current MSE Loss (two locations)
- `antigravity_engine.py` `run_sedimentation_cycle()` ~line 1048: `criterion = torch.nn.MSELoss()`
- `antigravity_engine.py` `_run_offline_distillation()` ~line 1295: same

### Why MSE Is Wrong for Retrieval
1. No awareness of other vectors — trains each vector independently, can create new collisions
2. Homeostatic targets are heuristic points, not geometric constraints
3. Teacher distillation via MSE collapses information — want to preserve teacher's ranking, not specific vectors
4. Confirmed by EmbedDistill (Thakur 2023) and RankDistil (AISTATS 2021)

### Existing InfoNCE in online_updater.py (lines 137-193)
- `InfoNCEOnlineLoss` — temperature-scaled cosine similarity with log-softmax
- `TripletMarginOnlineLoss`, `CosineSimilarityOnlineLoss` also available
- `create_online_loss()` factory function
- Full pluggable infrastructure with `AdaptiveMargin`, `OnlineLossScheduler`, `OnlineUpdateDiagnostics`

### Hard Negative Mining from chelation_log
- `self.chelation_log = defaultdict(list)` records collision centers per doc_id
- Documents sharing same center-of-mass were in same collapse cluster
- These collision partners are ideal hard negatives for contrastive training

### Proposed Architecture (Implemented in PR #101)
- New `sedimentation_loss.py` module with:
  - `SedimentationInfoNCELoss` — batch contrastive: output[i] must be closest to target[i]
  - `SedimentationHybridLoss` — weighted MSE + InfoNCE for stability
  - `HardNegativeMiner` — mines chelation_log collisions
  - `create_sedimentation_loss()` factory (mse/infonce/hybrid)
- `engine.set_sedimentation_loss()` API — backward compatible, MSE default
- Config presets: mse, contrastive, hybrid

### Design Decisions
1. Reuse online_updater.py loss concepts but new implementation for batch sedimentation
2. MSE remains default for backward compatibility
3. Batch-InfoNCE: each sample in batch acts as anchor, all others are negatives
4. Temperature default 0.07 (standard NT-Xent)
5. Frobenius regularization preserved alongside contrastive loss
6. ConvergenceMonitor compatible (uses relative threshold, robust to loss scale changes)

## References
- EmbedDistill (Thakur 2023, arXiv:2301.12005)
- RankDistil (AISTATS 2021)
- Session 30 strategic research findings
