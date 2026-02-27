# Parameter Sweep Analysis: Sedimentation Hyperparameter Tuning on SciFact

**Date:** 2026-02-27
**Dataset:** MTEB SciFact (5,183 documents, 300 queries)
**Embedding Model:** `ollama:nomic-embed-text` (768-dim)
**Adapter Type:** MLP (default, `chelation_p=85`)
**Sweep Script:** `run_sweep.py`
**Baseline NDCG@10:** 0.58669

---

## 1. Executive Summary

An 81-configuration parameter sweep was conducted during Session 19 to evaluate the
impact of four sedimentation hyperparameters on retrieval quality (NDCG@10). The sweep
tested all combinations of:

| Parameter        | Values Tested       | Count |
|------------------|---------------------|-------|
| `learning_rate`  | 0.01, 0.1, 0.5     | 3     |
| `threshold`      | 1, 2, 3             | 3     |
| `noise_scale`    | 0.0, 0.05, 0.2     | 3     |
| `epochs`         | 5, 10, 20           | 3     |

**Total:** 3 x 3 x 3 x 3 = 81 configurations.

### Key Findings

1. **No configuration beat the baseline.** The best result retained 99.9% of baseline
   NDCG. Sedimentation, in its current form, is a quality-preserving mechanism rather
   than a quality-improving one.
2. **Learning rate is the dominant parameter**, explaining 61.5% of outcome variance.
   Any `learning_rate >= 0.1` causes catastrophic adapter collapse.
3. **63% of configurations collapsed** to an identical degenerate NDCG of 0.1870,
   indicating a fixed-point attractor in the adapter weight space at high learning rates.
4. **Threshold is the second most important parameter** (5.8% of variance). Values >= 2
   trigger excessive sedimentation that degrades quality by 5--54%.
5. **Noise injection and epoch count have minimal impact** (1.5% and 0.1% of variance
   respectively) within the safe operating range.

---

## 2. Top 10 Best Configurations

All top configurations share `learning_rate=0.01` and `threshold=1`.

| Rank | NDCG@10  | Gain      | Retention | LR   | Thresh | Noise | Epochs |
|------|----------|-----------|-----------|------|--------|-------|--------|
| 1    | 0.58605  | -0.00063  | 99.9%     | 0.01 | 1      | 0.2   | 5      |
| 2    | 0.58370  | -0.00298  | 99.5%     | 0.01 | 1      | 0.05  | 5      |
| 3    | 0.58316  | -0.00353  | 99.4%     | 0.01 | 1      | 0.0   | 5      |
| 4    | 0.58304  | -0.00364  | 99.4%     | 0.01 | 1      | 0.0   | 20     |
| 5    | 0.58257  | -0.00412  | 99.3%     | 0.01 | 1      | 0.05  | 20     |
| 6    | 0.58228  | -0.00441  | 99.2%     | 0.01 | 1      | 0.05  | 10     |
| 7    | 0.57973  | -0.00695  | 98.8%     | 0.01 | 1      | 0.2   | 10     |
| 8    | 0.57709  | -0.00960  | 98.4%     | 0.01 | 1      | 0.0   | 10     |
| 9    | 0.56662  | -0.02007  | 96.6%     | 0.01 | 1      | 0.2   | 20     |
| 10   | 0.55523  | -0.03146  | 94.6%     | 0.01 | 2      | 0.05  | 20     |

**Observation:** The top 9 are all `threshold=1`. The 10th-best config is the first
appearance of `threshold=2`, already showing 5.4% degradation.

---

## 3. Parameter Impact Analysis

### 3.1 Learning Rate (61.5% of variance explained)

This is the single most critical parameter. The sweep reveals a sharp phase transition:

| Learning Rate | Mean NDCG | Best NDCG | Behavior                        |
|---------------|-----------|-----------|----------------------------------|
| 0.01          | 0.4519    | 0.5861    | Viable; gradual degradation      |
| 0.10          | 0.1666    | 0.1870    | Catastrophic collapse            |
| 0.50          | 0.1870    | 0.1870    | Complete collapse (all identical) |

At `LR=0.01`, the adapter makes small corrections that preserve embedding structure.
At `LR >= 0.1`, the adapter weights diverge within the first few gradient steps,
destroying the near-identity initialization. All 27 configurations with `LR=0.5`
produced the identical NDCG of 0.18696, and 24 of 27 `LR=0.1` configs also collapsed
to this value. This 0.18696 value represents a degenerate fixed point where the adapter
maps all embeddings to a low-information subspace.

The three `LR=0.1, noise=0.0, threshold=1` configs are the only `LR=0.1` results that
did NOT hit the 0.18696 fixed point -- they collapsed even further to NDCG < 0.005,
suggesting that without noise regularization, the adapter enters an even worse
degenerate state at `LR=0.1`.

**Recommendation:** Keep `learning_rate` in the range `[0.001, 0.01]`. The current
default of 0.001 in `ChelationConfig` is conservative and safe. Values of 0.01 are
the empirical upper bound.

### 3.2 Threshold (5.8% of variance explained)

The `threshold` parameter controls the minimum collapse frequency required before
sedimentation targets a document pair. Lower thresholds are more selective.

| Threshold | Mean NDCG | Best NDCG | Mean Retention |
|-----------|-----------|-----------|----------------|
| 1         | 0.2978    | 0.5861    | Varies widely  |
| 2         | 0.2957    | 0.5552    | 50.4%          |
| 3         | 0.2120    | 0.4118    | 36.1%          |

Focusing on the viable `LR=0.01` region:

| Threshold | Mean NDCG (LR=0.01) | Best NDCG (LR=0.01) |
|-----------|----------------------|----------------------|
| 1         | 0.5805               | 0.5861               |
| 2         | 0.5133               | 0.5552               |
| 3         | 0.2619               | 0.4118               |

At `threshold=1`, sedimentation only acts on the most frequently collapsed pairs, making
minimal but targeted corrections. At `threshold=3`, the system aggressively targets many
more document pairs, causing destructive interference between correction targets.

**Recommendation:** Use `threshold=1` for quality preservation. The current config
presets use `threshold=3` (medium) and `threshold=1` (large), which should be revisited.

### 3.3 Noise Scale (1.5% of variance explained)

Noise injection acts as a regularizer during sedimentation training.

| Noise Scale | Mean NDCG | Best NDCG | Mean Retention |
|-------------|-----------|-----------|----------------|
| 0.0         | 0.2642    | 0.5832    | Varies         |
| 0.05        | 0.2949    | 0.5837    | Varies         |
| 0.2         | 0.2464    | 0.5861    | Varies         |

Within the safe operating region (`LR=0.01, threshold=1`):

| Noise Scale | Mean NDCG | Std NDCG | Stability |
|-------------|-----------|----------|-----------|
| 0.0         | 0.5811    | 0.0028   | Stable    |
| 0.05        | 0.5829    | 0.0006   | Most stable |
| 0.2         | 0.5775    | 0.0081   | Variable  |

Noise at 0.05 produces the most consistent results (lowest standard deviation across
epoch counts). Noise at 0.2 produces the single best result (NDCG=0.5861 at 5 epochs)
but is much more variable -- at 20 epochs with noise=0.2, NDCG drops to 0.5666.

**Recommendation:** Use `noise_scale=0.05` for stability, or `noise_scale=0.2` with
`epochs=5` for maximum quality retention. The current default of 0.05 in
`ChelationConfig.NOISE_INJECTION_BASE_SCALE` is well-chosen.

### 3.4 Epochs (0.1% of variance explained)

Epoch count has the smallest impact overall, but interacts with other parameters.

| Epochs | Mean NDCG (LR=0.01, Thresh=1) | Std NDCG |
|--------|-------------------------------|----------|
| 5      | 0.5843                        | 0.0013   |
| 10     | 0.5797                        | 0.0021   |
| 20     | 0.5774                        | 0.0076   |

**Fewer epochs are better.** Each additional epoch slightly degrades quality and
increases variance. The adapter's near-identity initialization means early epochs make
the smallest corrections; continued training accumulates drift.

**Recommendation:** Use `epochs=5` for sedimentation. The current default of 10 in
`ChelationConfig.DEFAULT_EPOCHS` should be reduced, or convergence detection should
be enabled with aggressive settings to stop early.

---

## 4. Interaction Effects

### 4.1 Learning Rate x Threshold

```
             Threshold=1    Threshold=2    Threshold=3
LR=0.01      0.5805         0.5133         0.2619
LR=0.10      0.1260         0.1870         0.1870
LR=0.50      0.1870         0.1870         0.1870
```

Only `LR=0.01` produces meaningful variation across thresholds. At higher learning
rates, the adapter has already collapsed before the threshold parameter matters.

### 4.2 Threshold x Epochs (LR=0.01)

```
             Epochs=5       Epochs=10      Epochs=20
Threshold=1  0.5843         0.5797         0.5774
Threshold=2  0.5500         0.5375         0.4523
Threshold=3  0.2817         0.2465         0.2576
```

The epoch effect compounds with threshold severity. At `threshold=2, epochs=20`, NDCG
drops to 0.4523 (23% degradation). The `threshold=3` row shows catastrophic degradation
regardless of epoch count.

### 4.3 Noise x Epochs (LR=0.01, Threshold=1)

```
             Epochs=5       Epochs=10      Epochs=20
Noise=0.0    0.5832         0.5771         0.5830
Noise=0.05   0.5837         0.5823         0.5826
Noise=0.2    0.5861         0.5797         0.5666
```

An interesting pattern: noise=0.2 with epochs=5 produces the best single result but
degrades fastest as epochs increase. Noise=0.05 shows the flattest degradation curve,
confirming its role as a stabilizer.

---

## 5. Collapse Modes

### 5.1 Degenerate Fixed Point (NDCG = 0.18696)

51 of 81 configurations (63%) collapsed to the identical NDCG of 0.18696. This includes:
- All 27 `LR=0.5` configurations
- 24 of 27 `LR=0.1` configurations

This suggests the adapter has a strong attractor state at high learning rates where the
residual connection is overwhelmed by the learned transformation, mapping all embeddings
to a low-rank subspace that retains only coarse semantic information.

### 5.2 Near-Zero Collapse (NDCG < 0.01)

5 configurations achieved NDCG < 0.1 (effectively random retrieval):
- `LR=0.1, thresh=1, noise=0.0, epochs={5,10,20}` -- NDCG 0.003 to 0.005
- `LR=0.01, thresh=3, noise=0.2, epochs={5,10}` -- NDCG 0.013 to 0.078

The first group shows that high LR without noise regularization destroys the adapter
even more severely than with noise (which at least stabilizes at the 0.187 fixed point).
The second group shows that aggressive sedimentation (`threshold=3`) combined with
strong noise injection can also cause near-complete collapse.

### 5.3 Gradual Degradation Zone

The remaining 25 configurations (all `LR=0.01`) show a smooth degradation continuum
from 99.9% to 36% of baseline quality, governed primarily by the threshold parameter.

---

## 6. Variance Decomposition

A crude between-group variance analysis attributes the total variance in post-sedimentation
NDCG to each parameter:

| Parameter       | Between-Group Variance | % Explained |
|-----------------|------------------------|-------------|
| `learning_rate` | 0.016888               | 61.5%       |
| `threshold`     | 0.001599               | 5.8%        |
| `noise_scale`   | 0.000402               | 1.5%        |
| `epochs`        | 0.000035               | 0.1%        |
| Interactions/residual | 0.008546          | 31.1%       |

The large residual (31.1%) reflects interaction effects, particularly between learning
rate and other parameters -- since `LR >= 0.1` causes collapse, the threshold/noise/epoch
parameters only matter within the `LR=0.01` region.

---

## 7. Recommended Presets

Based on the sweep findings, the following preset updates are recommended for
`ChelationConfig` in `config.py`.

### 7.1 Updated Adapter Presets

```python
ADAPTER_PRESETS = {
    "small": {  # <1000 documents
        "learning_rate": 0.01,   # Was 0.01 (unchanged, validated)
        "epochs": 5,             # Was 5 (unchanged, validated)
        "threshold": 1,          # Was 10 (CHANGE: reduce to 1)
        "description": "Few documents, require strong signal"
    },
    "medium": {  # 1000-10000 documents
        "learning_rate": 0.005,  # Was 0.005 (unchanged)
        "epochs": 5,             # Was 10 (CHANGE: reduce from 10 to 5)
        "threshold": 1,          # Was 3 (CHANGE: reduce from 3 to 1)
        "description": "Standard datasets"
    },
    "large": {  # >10000 documents
        "learning_rate": 0.001,  # Was 0.001 (unchanged, validated)
        "epochs": 5,             # Was 15 (CHANGE: reduce from 15 to 5)
        "threshold": 1,          # Was 1 (unchanged, validated)
        "description": "Large datasets, capture all patterns"
    }
}
```

**Rationale:**
- `threshold` reduction to 1 across all presets: the sweep shows `threshold >= 2` always
  degrades quality, and the "small" preset's value of 10 was never tested but is likely
  far too aggressive.
- `epochs` reduction to 5 across all presets: additional epochs provide negligible benefit
  and increase variance. Early stopping via convergence detection is preferred for
  longer training.

### 7.2 Recommended Sedimentation Presets

```python
SEDIMENTATION_PRESETS = {
    "conservative": {
        "collapse_threshold": 1,   # Was 5 (CHANGE)
        "push_magnitude": 0.05,    # Unchanged
        "noise_scale": 0.05,       # NEW field
        "description": "Minimal intervention, maximum quality preservation"
    },
    "balanced": {
        "collapse_threshold": 1,   # Was 3 (CHANGE)
        "push_magnitude": 0.1,     # Unchanged
        "noise_scale": 0.05,       # NEW field
        "description": "Balanced sedimentation with noise regularization"
    },
    "aggressive": {
        "collapse_threshold": 2,   # Was 1 (CHANGE: raise from 1 to 2)
        "push_magnitude": 0.2,     # Unchanged
        "noise_scale": 0.2,        # NEW field
        "description": "Aggressive sedimentation (expect 5-10% quality loss)"
    }
}
```

**Rationale:**
- Even "aggressive" should not exceed `threshold=2` based on sweep evidence.
- Adding `noise_scale` to sedimentation presets aligns with the sweep's finding that
  noise is a meaningful regularizer.
- The `aggressive` preset now honestly warns about expected quality loss.

### 7.3 New "sweep_optimized" Preset

A new preset capturing the single best configuration:

```python
# Could be added to ADAPTER_PRESETS or as a standalone
"sweep_optimized": {
    "learning_rate": 0.01,
    "epochs": 5,
    "threshold": 1,
    "noise_scale": 0.2,
    "description": "Best config from 81-point sweep on SciFact (99.9% retention)"
}
```

---

## 8. Limitations and Future Work

### 8.1 Limitations of This Sweep

1. **Single dataset.** All results are on SciFact (5,183 documents, scientific claims).
   Patterns may differ on larger or non-scientific corpora.
2. **Single adapter type.** Only the default MLP adapter was tested. The sweep did not
   evaluate `procrustes` or `low_rank` adapter types.
3. **Fixed chelation_p=85.** The chelation percentile was held constant. It may interact
   with sedimentation parameters.
4. **No push_magnitude variation.** The `HOMEOSTATIC_PUSH_MAGNITUDE` was held at the
   default 0.1. The planned 7,350-config sweep (`run_large_sweep.py`) would have tested
   5 values of push magnitude.
5. **Narrow learning rate range.** Only 3 values were tested. The transition from viable
   (0.01) to collapsed (0.1) suggests important dynamics in the 0.02--0.08 range.

### 8.2 Recommended Follow-Up Experiments

1. **Fine-grained LR sweep:** Test `learning_rate` in `[0.005, 0.01, 0.02, 0.03, 0.05,
   0.07]` to locate the exact collapse boundary.
2. **Adapter type comparison:** Run the same grid with `procrustes` and `low_rank`
   adapters. Orthogonal (Procrustes) adapters may resist collapse better due to their
   constrained weight space.
3. **Push magnitude sweep:** Test `push_magnitude` in `[0.01, 0.05, 0.1, 0.2, 0.5]`
   as planned in `run_large_sweep.py`.
4. **Multi-dataset validation:** Run the top-5 configs on NFCorpus, FiQA, and TREC-COVID
   to test generalization.
5. **Convergence detection interaction:** Test whether enabling convergence detection
   (`patience=2, aggressive`) with `epochs=20` can match the `epochs=5` results by
   stopping early when quality plateaus.

---

## 9. Raw Data Summary

### 9.1 Distribution Statistics

| Metric                  | Value     |
|-------------------------|-----------|
| Total configurations    | 81        |
| Baseline NDCG@10        | 0.58669   |
| Mean post-NDCG          | 0.26868   |
| Median post-NDCG        | 0.18696   |
| Std post-NDCG           | 0.16577   |
| Min post-NDCG           | 0.00302   |
| Max post-NDCG           | 0.58605   |
| Configs beating baseline| 0 (0%)    |
| Configs > 99% retention | 6 (7.4%)  |
| Configs > 95% retention | 9 (11.1%) |
| Configs > 90% retention | 15 (18.5%)|
| Collapsed (NDCG=0.187)  | 51 (63.0%)|
| Near-zero (NDCG < 0.1)  | 5 (6.2%)  |

### 9.2 All LR=0.01 Results (Sorted by NDCG)

| Rank | NDCG     | Gain      | Retention | Thresh | Noise | Epochs |
|------|----------|-----------|-----------|--------|-------|--------|
| 1    | 0.58605  | -0.00063  | 99.9%     | 1      | 0.2   | 5      |
| 2    | 0.58370  | -0.00298  | 99.5%     | 1      | 0.05  | 5      |
| 3    | 0.58316  | -0.00353  | 99.4%     | 1      | 0.0   | 5      |
| 4    | 0.58304  | -0.00364  | 99.4%     | 1      | 0.0   | 20     |
| 5    | 0.58257  | -0.00412  | 99.3%     | 1      | 0.05  | 20     |
| 6    | 0.58228  | -0.00441  | 99.2%     | 1      | 0.05  | 10     |
| 7    | 0.57973  | -0.00695  | 98.8%     | 1      | 0.2   | 10     |
| 8    | 0.57709  | -0.00960  | 98.4%     | 1      | 0.0   | 10     |
| 9    | 0.56662  | -0.02007  | 96.6%     | 1      | 0.2   | 20     |
| 10   | 0.55523  | -0.03146  | 94.6%     | 2      | 0.05  | 20     |
| 11   | 0.55346  | -0.03322  | 94.3%     | 2      | 0.2   | 5      |
| 12   | 0.54984  | -0.03685  | 93.7%     | 2      | 0.05  | 5      |
| 13   | 0.54680  | -0.03989  | 93.2%     | 2      | 0.0   | 5      |
| 14   | 0.53883  | -0.04786  | 91.8%     | 2      | 0.05  | 10     |
| 15   | 0.53728  | -0.04941  | 91.6%     | 2      | 0.2   | 10     |
| 16   | 0.53651  | -0.05017  | 91.4%     | 2      | 0.0   | 10     |
| 17   | 0.53388  | -0.05280  | 91.0%     | 2      | 0.0   | 20     |
| 18   | 0.41177  | -0.17492  | 70.2%     | 3      | 0.05  | 10     |
| 19   | 0.40999  | -0.17670  | 69.9%     | 3      | 0.05  | 5      |
| 20   | 0.38291  | -0.20377  | 65.3%     | 3      | 0.05  | 20     |
| 21   | 0.35766  | -0.22903  | 61.0%     | 3      | 0.0   | 5      |
| 22   | 0.31439  | -0.27230  | 53.6%     | 3      | 0.0   | 10     |
| 23   | 0.28459  | -0.30210  | 48.5%     | 3      | 0.0   | 20     |
| 24   | 0.26772  | -0.31896  | 45.6%     | 2      | 0.2   | 20     |
| 25   | 0.10540  | -0.48129  | 18.0%     | 3      | 0.2   | 20     |
| 26   | 0.07755  | -0.50914  | 13.2%     | 3      | 0.2   | 5      |
| 27   | 0.01321  | -0.57347  | 2.3%      | 3      | 0.2   | 10     |

---

## 10. Conclusions

1. **Sedimentation is currently a preservation mechanism, not an improvement mechanism.**
   The best configuration retains 99.9% of baseline quality. Future work should
   investigate whether teacher distillation, alternative adapter types, or chelation_p
   tuning can push past the baseline.

2. **The safe operating envelope is narrow.** Only the parameter combination
   `LR=0.01, threshold=1` reliably preserves quality. Any increase in either parameter
   leads to rapid degradation.

3. **The adapter's near-identity initialization is its key safety feature.** The small
   weight std (0.001) means the adapter starts as an approximate identity function.
   High learning rates destroy this initialization within the first epoch.

4. **Noise injection (0.05) acts as a stabilizer**, reducing variance across epoch counts
   and providing a slight regularization benefit. Higher noise (0.2) can help at very
   low epoch counts but becomes destabilizing over longer training.

5. **The current `ChelationConfig` defaults are reasonable** but the adapter presets'
   threshold values should be reduced. The `DEFAULT_EPOCHS` of 10 would benefit from
   reduction to 5, and convergence detection should be enabled by default for any
   training run longer than 5 epochs.
