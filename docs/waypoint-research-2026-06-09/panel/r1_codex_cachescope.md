# Codex offline-cache scope (enumerate only — build NOTHING)

Determine what drift regimes can be built **fully offline** (HF_HUB_OFFLINE=1, HF_DATASETS_OFFLINE=1)
toward a ≥8-regime target for the D3 recoverability-estimator validation. We currently have 4 regimes
(SciFact/NFCorpus × MiniLM→mpnet / MiniLM→bge-large). Do NOT train, embed, or freeze anything. Just
enumerate and report.

## Tasks
1. **Cached sentence-transformer / HF models.** List every model already present in the local HF cache
   (`~/.cache/huggingface`, `HF_HOME`, and any repo-local cache) usable as an encoder — especially
   candidates for NEW encoder-swap targets beyond mpnet/bge-large (e.g. gte, e5, bge-base/small,
   gtr, minilm variants). Report exact model ids and that they load offline.
2. **Cached datasets.** List which BEIR/MTEB retrieval datasets (corpus + queries + qrels) are cached
   locally and loadable offline via the harness (`load_mteb_data` path) beyond SciFact/NFCorpus
   (e.g. fiqa, arguana, scidocs, trec-covid, nfcorpus, scifact). Confirm each has qrels.
3. **Achievable NEW regimes.** Cross datasets × encoder-swaps × (optionally) drift magnitudes ×
   anchor-fractions and list the concrete NEW regimes buildable OFFLINE beyond the existing 4, enough
   to reach ≥8 total. If ≥8 is NOT achievable offline, say so explicitly and report the max achievable
   count + exactly which pieces (models/datasets) are missing from cache.
4. **Feasibility verdict.** State whether the D3 ≥8-regime validation is FEASIBLE offline right now,
   and if not, the smallest set of downloads that would unblock it (for the operator to fetch).

## Deliverable
Write `research/drift_recovery/out/estimator/offline_cache_scope.md` with: the model list, dataset
list, the concrete achievable NEW-regime list toward ≥8, and the FEASIBLE / NOT-FEASIBLE verdict with
the exact missing pieces if any. Be brutally honest — if only 4-6 regimes are achievable offline, say
so; do NOT imply 8 are available if they are not. Build nothing.
