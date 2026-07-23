# Obj A validation — pre-registered block-LOO

**Verdict: PROMISING-BUT-UNDERPOWERED**. Success bar met: **TRUE**. This result is not labeled validated.

## Achieved scale

- Independent dataset×encoder-family block-n: **6**
- Cell-n: **6**
- Whole-dataset holdout units: **3**
- Whole-encoder-family holdout units: **2**
- Pseudo-replicate cells: **0**

## Frozen block-LOO results

| Scheme | Predictor | Spearman | MAE | Cell-n |
|---|---|---:|---:|---:|
| dataset | `oracle_margin_mean` | 0.8857 | 0.0252 | 6 |
| dataset | `oracle_gap_ols` | -0.0857 | 0.0773 | 6 |
| dataset | `mean_R` | -0.7171 | 0.0853 | 6 |
| encoder_family | `oracle_margin_mean` | 0.8286 | 0.0324 | 6 |
| encoder_family | `oracle_gap_ols` | 0.4286 | 0.0696 | 6 |
| encoder_family | `mean_R` | -0.2928 | 0.0807 | 6 |

Strict comparisons (both greater Spearman and lower MAE required):

- dataset: margin vs `oracle_gap_ols` — **PASS** (Spearman: True; MAE: True).
- dataset: margin vs `mean_R` — **PASS** (Spearman: True; MAE: True).
- encoder_family: margin vs `oracle_gap_ols` — **PASS** (Spearman: True; MAE: True).
- encoder_family: margin vs `mean_R` — **PASS** (Spearman: True; MAE: True).

## Increment over oracle gap

- Partial Spearman(margin, R | gap): **0.7897**
- Raw Spearman(margin, R): **0.8857**
- Raw Spearman(gap, R): **0.6571**
- Raw margin-minus-gap Spearman: **0.2286**

## Per-regime values

| Regime | Dataset | Encoder | Margin mean | Floor | Oracle gap | R |
|---|---|---|---:|---:|---:|---:|
| `scifact_minilm_to_mpnet` | SciFact | mpnet | 0.078161 | 0.0000 | 0.8076 | 0.8434 |
| `scifact_minilm_to_bge_large` | SciFact | bge-large | 0.046963 | 0.0167 | 0.8260 | 0.7224 |
| `nfcorpus_minilm_to_mpnet` | NFCorpus | mpnet | 0.003969 | 0.0239 | 0.5814 | 0.6471 |
| `nfcorpus_minilm_to_bge_large` | NFCorpus | bge-large | 0.007905 | 0.0527 | 0.5909 | 0.6640 |
| `fiqa2018_minilm_to_mpnet_af040` | FiQA2018 | mpnet | 0.079358 | 0.0083 | 0.7781 | 0.8211 |
| `fiqa2018_minilm_to_bge_large_af040` | FiQA2018 | bge-large | 0.027909 | 0.0170 | 0.7484 | 0.7471 |

## FiQA held out

- `fiqa2018_minilm_to_bge_large_af040` actual R=0.7471; margin=0.7036, gap-only=0.7441, mean-R=0.7192.
- `fiqa2018_minilm_to_mpnet_af040` actual R=0.8211; margin=0.8305, gap-only=0.7599, mean-R=0.7192.
- FiQA-only MAE: margin 0.0265, gap-only 0.0321, mean-R 0.0648.

## Availability and audit

- `scifact_minilm_to_mpnet`: available; action=reused_existing; parity max=0; safe fit n=600; eval-positive fit docs=0.
- `scifact_minilm_to_bge_large`: available; action=reused_existing; parity max=0; safe fit n=600; eval-positive fit docs=0.
- `nfcorpus_minilm_to_mpnet`: available; action=reused_existing; parity max=0; safe fit n=869; eval-positive fit docs=0.
- `nfcorpus_minilm_to_bge_large`: available; action=reused_existing; parity max=0; safe fit n=869; eval-positive fit docs=0.
- `fiqa2018_minilm_to_mpnet_af040`: available; action=reused_new; parity max=0; safe fit n=600; eval-positive fit docs=0.
- `fiqa2018_minilm_to_bge_large_af040`: available; action=reused_new; parity max=0; safe fit n=600; eval-positive fit docs=0.
