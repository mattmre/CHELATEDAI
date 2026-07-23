# Obj A v2 validation — powered block-LOO

**Frozen-label verdict: NEGATIVE**. AND-bar met: **FALSE**.

## Achieved scale

- Independent dataset×encoder-family block-n: **12**
- Cell-n: **12** of 12 registered
- Dataset blocks: **4** of 4
- Encoder-family blocks: **3** of 3
- Excluded cells: **0**

## Frozen block-LOO tables

### dataset

| Predictor | Spearman | MAE | Cell-n |
|---|---:|---:|---:|
| `oracle_margin_mean` | -0.7063 | 0.1492 | 12 |
| `oracle_gap_ols` | -0.4965 | 0.1239 | 12 |
| `mean_R` | -0.5182 | 0.0996 | 12 |
- Margin vs `oracle_gap_ols`: **FAIL** (Spearman strict=False; MAE strict=False).
- Margin vs `mean_R`: **FAIL** (Spearman strict=False; MAE strict=False).

### encoder_family

| Predictor | Spearman | MAE | Cell-n |
|---|---:|---:|---:|
| `oracle_margin_mean` | -0.6853 | 0.1055 | 12 |
| `oracle_gap_ols` | -0.4476 | 0.0943 | 12 |
| `mean_R` | -0.7096 | 0.1041 | 12 |
- Margin vs `oracle_gap_ols`: **FAIL** (Spearman strict=False; MAE strict=False).
- Margin vs `mean_R`: **FAIL** (Spearman strict=True; MAE strict=False).

## Partial and raw rank associations

- Partial Spearman(margin, R | gap): **-0.1246**
- Raw Spearman(margin, R): **0.0420**
- Raw Spearman(gap, R): **0.2238**
- Margin-minus-gap raw Spearman: **-0.1818**

## Per-cell R

| Regime | Dataset | Encoder | Margin mean | Floor | Oracle gap | R |
|---|---|---|---:|---:|---:|---:|
| `scifact_minilm_to_mpnet` | SciFact | mpnet | 0.078161 | 0.0000 | 0.8076 | 0.8434 |
| `scifact_minilm_to_bge_large` | SciFact | bge-large | 0.046963 | 0.0167 | 0.8260 | 0.7224 |
| `scifact_minilm_to_e5_base_v2_af040` | SciFact | e5-base-v2 | 0.012167 | 0.0000 | 0.7880 | 0.5967 |
| `nfcorpus_minilm_to_mpnet` | NFCorpus | mpnet | 0.003969 | 0.0239 | 0.5814 | 0.6471 |
| `nfcorpus_minilm_to_bge_large` | NFCorpus | bge-large | 0.007905 | 0.0527 | 0.5909 | 0.6640 |
| `nfcorpus_minilm_to_e5_base_v2_af040` | NFCorpus | e5-base-v2 | 0.002383 | 0.0553 | 0.5617 | 0.6019 |
| `fiqa2018_minilm_to_mpnet_af040` | FiQA2018 | mpnet | 0.079358 | 0.0083 | 0.7781 | 0.8211 |
| `fiqa2018_minilm_to_bge_large_af040` | FiQA2018 | bge-large | 0.027909 | 0.0170 | 0.7484 | 0.7471 |
| `fiqa2018_minilm_to_e5_base_v2_af040` | FiQA2018 | e5-base-v2 | 0.006377 | 0.0000 | 0.7627 | 0.5535 |
| `arguana_minilm_to_mpnet_af040` | ArguAna | mpnet | -0.061244 | 0.0215 | 0.6302 | 0.8569 |
| `arguana_minilm_to_bge_large_af040` | ArguAna | bge-large | -0.018433 | 0.0072 | 0.7667 | 0.7932 |
| `arguana_minilm_to_e5_base_v2_af040` | ArguAna | e5-base-v2 | -0.016409 | 0.0053 | 0.6824 | 0.7466 |

## Novel held-out block checks

- **ArguAna** (3 cells): margin rho=-1.0000, MAE=0.2928; gap rho=-0.5000, MAE=0.1198; rank relation=loses; strict both=FAIL; aggregate agreement=True.
  - `arguana_minilm_to_bge_large_af040` actual=0.7932; margin=0.5464; gap=0.7094; mean-R=0.6886.
  - `arguana_minilm_to_e5_base_v2_af040` actual=0.7466; margin=0.5524; gap=0.6747; mean-R=0.6886.
  - `arguana_minilm_to_mpnet_af040` actual=0.8569; margin=0.4194; gap=0.6533; mean-R=0.6886.
- **e5-base-v2** (4 cells): margin rho=-0.8000, MAE=0.1358; gap rho=-0.6000, MAE=0.1303; rank relation=loses; strict both=FAIL; aggregate agreement=True.
  - `arguana_minilm_to_e5_base_v2_af040` actual=0.7466; margin=0.7592; gap=0.7485; mean-R=0.7619.
  - `fiqa2018_minilm_to_e5_base_v2_af040` actual=0.5535; margin=0.7609; gap=0.7804; mean-R=0.7619.
  - `nfcorpus_minilm_to_e5_base_v2_af040` actual=0.6019; margin=0.7606; gap=0.7006; mean-R=0.7619.
  - `scifact_minilm_to_e5_base_v2_af040` actual=0.5967; margin=0.7613; gap=0.7905; mean-R=0.7619.

## Availability, parity, and leakage audit

- `scifact_minilm_to_mpnet`: reused_existing; parity max=0; safe fit n=600; eval-positive fit docs=0; frozen scores=c3a,floor,mlp,oracle,procrustes,ridge.
- `scifact_minilm_to_bge_large`: reused_existing; parity max=0; safe fit n=600; eval-positive fit docs=0; frozen scores=floor,oracle,ridge.
- `scifact_minilm_to_e5_base_v2_af040`: built_new; parity max=0; safe fit n=600; eval-positive fit docs=0; frozen scores=floor,oracle,ridge.
- `nfcorpus_minilm_to_mpnet`: reused_existing; parity max=0; safe fit n=869; eval-positive fit docs=0; frozen scores=floor,oracle,ridge.
- `nfcorpus_minilm_to_bge_large`: reused_existing; parity max=0; safe fit n=869; eval-positive fit docs=0; frozen scores=floor,oracle,ridge.
- `nfcorpus_minilm_to_e5_base_v2_af040`: built_new; parity max=0; safe fit n=869; eval-positive fit docs=0; frozen scores=floor,oracle,ridge.
- `fiqa2018_minilm_to_mpnet_af040`: reused_existing; parity max=0; safe fit n=600; eval-positive fit docs=0; frozen scores=floor,oracle,ridge.
- `fiqa2018_minilm_to_bge_large_af040`: reused_existing; parity max=0; safe fit n=600; eval-positive fit docs=0; frozen scores=floor,oracle,ridge.
- `fiqa2018_minilm_to_e5_base_v2_af040`: built_new; parity max=0; safe fit n=600; eval-positive fit docs=0; frozen scores=floor,oracle,ridge.
- `arguana_minilm_to_mpnet_af040`: built_new; parity max=0; safe fit n=600; eval-positive fit docs=0; frozen scores=floor,oracle,ridge.
- `arguana_minilm_to_bge_large_af040`: built_new; parity max=0; safe fit n=600; eval-positive fit docs=0; frozen scores=floor,oracle,ridge.
- `arguana_minilm_to_e5_base_v2_af040`: built_new; parity max=0; safe fit n=600; eval-positive fit docs=0; frozen scores=floor,oracle,ridge.

## Freeze ordering

Hash lock verified. Preregistration precedes every new pack file: **TRUE**.

E5 used the same plain harness text path as mpnet and bge-large, without `query:`/`passage:` prefixes. This may understate e5 absolute quality but keeps the encoder-swap protocol identical.
