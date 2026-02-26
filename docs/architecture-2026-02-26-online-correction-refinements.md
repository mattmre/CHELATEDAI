# Architecture: Online Correction Refinements
Date: 2026-02-26 | Session: 21 | Status: Approved

## Summary
Pluggable loss functions, adaptive margins, per-vector computation, loss scheduling, and diagnostics for OnlineUpdater.

## New Components (all in online_updater.py)
- OnlineLossFunction: Abstract base with compute() and get_state()
- TripletMarginOnlineLoss: Current behavior + per_vector aggregation option
- InfoNCEOnlineLoss: NT-Xent contrastive loss with temperature parameter
- CosineSimilarityOnlineLoss: Direct similarity optimization
- create_online_loss(): Factory function
- AdaptiveMargin: Dynamic margin based on retrieval quality signals
- OnlineLossScheduler: Composes TeacherWeightScheduler for loss weight decay
- OnlineUpdateDiagnostics: Per-dimension gradients, running stats, stability tracker bridge

## Config Additions
- ONLINE_LOSS_TYPE="triplet_margin", ONLINE_AGGREGATION="mean"
- ONLINE_ADAPTIVE_MARGIN_ENABLED=False, ONLINE_INFONCE_TEMPERATURE=0.07
- ONLINE_LOSS_SCHEDULE="constant", ONLINE_DIAGNOSTICS_ENABLED=False
- New preset: online_update (conservative, balanced, aggressive)

## Key Design: Adapter processes ALL pos/neg vectors individually
Loss function decides aggregation (mean vs per_vector), not the updater.
This enables per-vector granularity for InfoNCE while preserving mean behavior for triplet.

## Test Plan: ~48 new tests (18 -> 66 total)
- Loss functions (15), adaptive margin (8), scheduler (7), diagnostics (8), extended updater (10)
