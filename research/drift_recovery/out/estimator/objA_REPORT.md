# Obj A report — honest scale-up of `oracle_margin_mean`

## Built

Reused the four frozen SciFact/NFCorpus packs and built only the two registered FiQA2018 packs (mpnet and bge-large, anchor 0.40), sequentially and offline. Every available pack passed merged-harness NDCG parity at 1e-12 and the feature fit used a leakage-safe document index.

## Scale and result

Achieved independent pair block-n **6** and cell-n **6**, spanning 3 whole-dataset holdouts and 2 whole-encoder-family holdouts. Verdict: **PROMISING-BUT-UNDERPOWERED**; the frozen AND-bar was met.

Dataset-block LOO: margin Spearman 0.8857, MAE 0.025; gap-only Spearman -0.0857, MAE 0.077. Encoder-block LOO: margin Spearman 0.8286, MAE 0.032; gap-only Spearman 0.4286, MAE 0.070.

Partial Spearman(margin, R | oracle gap) was **0.7897**. With the whole FiQA dataset held out, margin MAE was 0.026 versus 0.032 for gap-only.

## Brutal conclusion

It beat oracle-gap-alone under both schemes on both locked metrics. However, inside the two-cell FiQA holdout alone, margin tied gap-only on Spearman rather than strictly beating it, although its MAE was lower. The n=4 +1.0 ordering is descriptive history, not validation. Even a passing six-pair point estimate would remain underpowered because the offline cache supplies only three dataset and two encoder-family holdout units.
