# Obj B CPU preflight falsifier

**Verdict: CLOSE**

> Obj B fails constructively — under the residual>=0.05 gate no cell admits. The residuals are small mainly because the absolute recoverable NDCG gaps are small, and the dev-selected gated local ridge frequently does not even beat the no-op floor. This does NOT show local ridge dominates chelation; it shows the synthetic preflight could not construct a discriminating home-turf residual under small-magnitude non-affine warps, so no GPU chelation run is justified.

This run used synthetic Gaussian-cluster vectors and NumPy closed-form fits only. It used no GPU and no embedding model. Routing was inferred from corrupted vectors; clean cluster IDs were retained only for diagnostics and corpus construction.

## Per-family admission

| Family | Verdict | Operating point / smallest reason |
|---|---:|---|
| quadratic | CLOSE | The strongest small, sane cell left only 0.0295 NDCG after the dev-selected gated local ridge (< 0.05). |
| soft_fold | CLOSE | The strongest small, sane cell left only 0.0328 NDCG after the dev-selected gated local ridge (< 0.05). |

## Grid

The three ladders are oracle minus floor, oracle minus the anchor-dev-selected global map, and oracle minus the anchor-dev-selected detector-gated local ridge.

| Family | s | gamma | n | oracle-floor | oracle-global | oracle-local | residual | mean/max disp | mean/max rel disp | purity | admitted |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| quadratic | 0.15 | 0.10 | 4 | 0.0010 | 0.0361 | 0.0031 | 0.0031 | 0.0040/0.1000 | 0.0011/0.0292 | 1.000 | no |
| quadratic | 0.15 | 0.10 | 8 | 0.0010 | 0.0062 | 0.0010 | 0.0010 | 0.0040/0.1000 | 0.0011/0.0292 | 1.000 | no |
| quadratic | 0.15 | 0.10 | 16 | 0.0010 | 0.0062 | 0.0021 | 0.0021 | 0.0040/0.1000 | 0.0011/0.0292 | 1.000 | no |
| quadratic | 0.15 | 0.30 | 4 | 0.0051 | 0.0395 | 0.0082 | 0.0082 | 0.0120/0.3000 | 0.0033/0.0877 | 1.000 | no |
| quadratic | 0.15 | 0.30 | 8 | 0.0051 | 0.0154 | 0.0041 | 0.0041 | 0.0120/0.3000 | 0.0033/0.0877 | 1.000 | no |
| quadratic | 0.15 | 0.30 | 16 | 0.0051 | 0.0092 | 0.0031 | 0.0031 | 0.0120/0.3000 | 0.0033/0.0877 | 1.000 | no |
| quadratic | 0.15 | 0.60 | 4 | 0.0103 | 0.0425 | 0.0113 | 0.0113 | 0.0240/0.6000 | 0.0066/0.1753 | 1.000 | no |
| quadratic | 0.15 | 0.60 | 8 | 0.0103 | 0.0178 | 0.0092 | 0.0092 | 0.0240/0.6000 | 0.0066/0.1753 | 1.000 | no |
| quadratic | 0.15 | 0.60 | 16 | 0.0103 | 0.0188 | 0.0103 | 0.0103 | 0.0240/0.6000 | 0.0066/0.1753 | 1.000 | no |
| quadratic | 0.15 | 1.00 | 4 | 0.0160 | 0.0443 | 0.0165 | 0.0165 | 0.0401/1.0000 | 0.0110/0.2922 | 1.000 | no |
| quadratic | 0.15 | 1.00 | 8 | 0.0160 | 0.0306 | 0.0092 | 0.0092 | 0.0401/1.0000 | 0.0110/0.2922 | 1.000 | no |
| quadratic | 0.15 | 1.00 | 16 | 0.0160 | 0.0258 | 0.0051 | 0.0051 | 0.0401/1.0000 | 0.0110/0.2922 | 1.000 | no |
| quadratic | 0.25 | 0.10 | 4 | 0.0051 | 0.0226 | 0.0092 | 0.0092 | 0.0075/0.1000 | 0.0020/0.0305 | 1.000 | no |
| quadratic | 0.25 | 0.10 | 8 | 0.0051 | 0.0092 | 0.0072 | 0.0072 | 0.0075/0.1000 | 0.0020/0.0305 | 1.000 | no |
| quadratic | 0.25 | 0.10 | 16 | 0.0051 | 0.0092 | 0.0051 | 0.0051 | 0.0075/0.1000 | 0.0020/0.0305 | 1.000 | no |
| quadratic | 0.25 | 0.30 | 4 | 0.0113 | 0.0491 | 0.0154 | 0.0154 | 0.0224/0.3000 | 0.0061/0.0914 | 1.000 | no |
| quadratic | 0.25 | 0.30 | 8 | 0.0113 | 0.0174 | 0.0113 | 0.0113 | 0.0224/0.3000 | 0.0061/0.0914 | 1.000 | no |
| quadratic | 0.25 | 0.30 | 16 | 0.0113 | 0.0154 | 0.0092 | 0.0092 | 0.0224/0.3000 | 0.0061/0.0914 | 1.000 | no |
| quadratic | 0.25 | 0.60 | 4 | 0.0144 | 0.0746 | 0.0195 | 0.0195 | 0.0447/0.6000 | 0.0123/0.1828 | 1.000 | no |
| quadratic | 0.25 | 0.60 | 8 | 0.0144 | 0.0205 | 0.0154 | 0.0154 | 0.0447/0.6000 | 0.0123/0.1828 | 1.000 | no |
| quadratic | 0.25 | 0.60 | 16 | 0.0144 | 0.0174 | 0.0231 | 0.0231 | 0.0447/0.6000 | 0.0123/0.1828 | 1.000 | no |
| quadratic | 0.25 | 1.00 | 4 | 0.0232 | 0.0617 | 0.0267 | 0.0267 | 0.0746/1.0000 | 0.0204/0.3047 | 1.000 | no |
| quadratic | 0.25 | 1.00 | 8 | 0.0232 | 0.0205 | 0.0236 | 0.0236 | 0.0746/1.0000 | 0.0204/0.3047 | 1.000 | no |
| quadratic | 0.25 | 1.00 | 16 | 0.0232 | 0.0275 | 0.0295 | 0.0295 | 0.0746/1.0000 | 0.0204/0.3047 | 1.000 | no |
| soft_fold | 0.15 | 0.10 | 4 | 0.0010 | 0.0365 | 0.0010 | 0.0010 | 0.0031/0.1000 | 0.0008/0.0280 | 1.000 | no |
| soft_fold | 0.15 | 0.10 | 8 | 0.0010 | 0.0010 | 0.0010 | 0.0010 | 0.0031/0.1000 | 0.0008/0.0280 | 1.000 | no |
| soft_fold | 0.15 | 0.10 | 16 | 0.0010 | 0.0041 | 0.0000 | 0.0000 | 0.0031/0.1000 | 0.0008/0.0280 | 1.000 | no |
| soft_fold | 0.15 | 0.30 | 4 | 0.0031 | 0.0441 | 0.0041 | 0.0041 | 0.0093/0.3000 | 0.0025/0.0840 | 1.000 | no |
| soft_fold | 0.15 | 0.30 | 8 | 0.0031 | 0.0092 | 0.0051 | 0.0051 | 0.0093/0.3000 | 0.0025/0.0840 | 1.000 | no |
| soft_fold | 0.15 | 0.30 | 16 | 0.0031 | 0.0082 | 0.0041 | 0.0041 | 0.0093/0.3000 | 0.0025/0.0840 | 1.000 | no |
| soft_fold | 0.15 | 0.60 | 4 | 0.0082 | 0.0445 | 0.0113 | 0.0113 | 0.0186/0.6000 | 0.0051/0.1680 | 1.000 | no |
| soft_fold | 0.15 | 0.60 | 8 | 0.0082 | 0.0144 | 0.0082 | 0.0082 | 0.0186/0.6000 | 0.0051/0.1680 | 1.000 | no |
| soft_fold | 0.15 | 0.60 | 16 | 0.0082 | 0.0154 | 0.0092 | 0.0092 | 0.0186/0.6000 | 0.0051/0.1680 | 1.000 | no |
| soft_fold | 0.15 | 1.00 | 4 | 0.0185 | 0.0466 | 0.0148 | 0.0148 | 0.0309/1.0000 | 0.0085/0.2800 | 1.000 | no |
| soft_fold | 0.15 | 1.00 | 8 | 0.0185 | 0.0220 | 0.0113 | 0.0113 | 0.0309/1.0000 | 0.0085/0.2800 | 1.000 | no |
| soft_fold | 0.15 | 1.00 | 16 | 0.0185 | 0.0214 | 0.0205 | 0.0205 | 0.0309/1.0000 | 0.0085/0.2800 | 1.000 | no |
| soft_fold | 0.25 | 0.10 | 4 | 0.0031 | 0.0174 | 0.0021 | 0.0021 | 0.0054/0.1000 | 0.0015/0.0298 | 1.000 | no |
| soft_fold | 0.25 | 0.10 | 8 | 0.0031 | 0.0031 | 0.0010 | 0.0010 | 0.0054/0.1000 | 0.0015/0.0298 | 1.000 | no |
| soft_fold | 0.25 | 0.10 | 16 | 0.0031 | 0.0041 | 0.0031 | 0.0031 | 0.0054/0.1000 | 0.0015/0.0298 | 1.000 | no |
| soft_fold | 0.25 | 0.30 | 4 | 0.0072 | 0.0246 | 0.0082 | 0.0082 | 0.0162/0.3000 | 0.0045/0.0893 | 1.000 | no |
| soft_fold | 0.25 | 0.30 | 8 | 0.0072 | 0.0123 | 0.0072 | 0.0072 | 0.0162/0.3000 | 0.0045/0.0893 | 1.000 | no |
| soft_fold | 0.25 | 0.30 | 16 | 0.0072 | 0.0072 | 0.0082 | 0.0082 | 0.0162/0.3000 | 0.0045/0.0893 | 1.000 | no |
| soft_fold | 0.25 | 0.60 | 4 | 0.0144 | 0.0545 | 0.0246 | 0.0246 | 0.0324/0.6000 | 0.0089/0.1787 | 1.000 | no |
| soft_fold | 0.25 | 0.60 | 8 | 0.0144 | 0.0205 | 0.0144 | 0.0144 | 0.0324/0.6000 | 0.0089/0.1787 | 1.000 | no |
| soft_fold | 0.25 | 0.60 | 16 | 0.0144 | 0.0123 | 0.0103 | 0.0103 | 0.0324/0.6000 | 0.0089/0.1787 | 1.000 | no |
| soft_fold | 0.25 | 1.00 | 4 | 0.0242 | 0.0646 | 0.0328 | 0.0328 | 0.0540/1.0000 | 0.0149/0.2978 | 1.000 | no |
| soft_fold | 0.25 | 1.00 | 8 | 0.0242 | 0.0383 | 0.0206 | 0.0206 | 0.0540/1.0000 | 0.0149/0.2978 | 1.000 | no |
| soft_fold | 0.25 | 1.00 | 16 | 0.0242 | 0.0219 | 0.0144 | 0.0144 | 0.0540/1.0000 | 0.0149/0.2978 | 1.000 | no |

## Frozen admission and parity rules

A cell requires residual >= 0.05, mean relative displacement <= 0.12, max relative displacement <= 0.35, and oracle-floor in [0.015, 0.350], with corrupted-space partition purity >= 0.90. Lambda and rank use the same anchor train/dev selection boundary; eval queries select no hyperparameter or method.

Dense local ridge is intentionally not handicapped. The low-rank local alternative uses the same lambda candidates and chooses rank on the same anchor dev rows. The reported best-local ladder uses whichever local family had lower anchor-dev MSE.

Protocol audit: 144 seed-runs; 0 clean-ID fit violations; 0 low-rank budget violations; minimum corrupted-space purity 1.000. The dev-selected local method was low-rank in 144 seed-runs and dense in 0; method selection never used eval NDCG.

Detector limitation: anchors are generated only for truly harmed clusters, so the anchor-derived detector labels are equivalent to nonzero paired displacement and tau is 1e-10 in every run. This is an oracle-generous presence gate shared by all methods, not an independently learned or stress-tested detector. It biases the preflight toward finding that local ridge can close the gap.
