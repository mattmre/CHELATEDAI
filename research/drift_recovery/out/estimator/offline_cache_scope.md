# D3 offline-cache scope

Scope date: 2026-07-10. This is an enumeration only. No model was trained, no text was embedded, and no pack was frozen. Offline checks used `HF_HUB_OFFLINE=1`, `HF_DATASETS_OFFLINE=1`, and `TRANSFORMERS_OFFLINE=1`. Model checks instantiated `SentenceTransformer` on CPU without calling `encode`; dataset checks called the repository's `benchmark_utils.load_mteb_data` and counted corpus, queries, and qrels.

## Cached encoder models

The active Hugging Face cache is `C:\Users\mattm\.cache\huggingface`. `HF_HOME`, `HUGGINGFACE_HUB_CACHE`, and `HF_DATASETS_CACHE` are unset, so their defaults resolve under that directory. No additional repo-local Hugging Face/model cache was found in this worktree.

| Exact model id | Offline constructor result | Encoder-swap status for the current harness |
|---|---:|---|
| `sentence-transformers/all-MiniLM-L6-v2` | PASS, 384 dimensions | Usable offline; this is the old/store encoder, not a new swap target. |
| `sentence-transformers/all-mpnet-base-v2` (also resolvable as `all-mpnet-base-v2`) | PASS, 768 dimensions | Usable offline; already represented in two existing regimes. |
| `BAAI/bge-large-en-v1.5` | PASS, 1024 dimensions | Usable offline; already represented in two existing regimes. |
| `nomic-ai/nomic-embed-text-v1` | PASS, 768 dimensions **only with `trust_remote_code=True`** | **Not usable by the current D3 harness as written.** `SentenceTransformerEmbeddingBackend` calls `SentenceTransformer(model_name, device=device)` without `trust_remote_code=True`; the exact harness-style constructor fails offline with a `ValueError` requesting that flag. Do not count this model toward immediately buildable regimes. |

Other model directories in the hub cache are not valid new sentence-encoder swaps for this harness: `BAAI/bge-reranker-v2-m3` is a reranker/classifier; `colbert-ir/colbertv2.0` is a ColBERT late-interaction model; `microsoft/layoutlmv3-base` is a document-layout model; the Qwen entries are generative models; OCR, speech, GGUF, and multimodal entries are outside the sentence-transformer path. `microsoft/infoxlm-large` and `nomic-ai/nomic-bert-2048` have incomplete cache entries and fail offline construction.

No cached `gte`, `e5`, `gtr`, `bge-base`, `bge-small`, or additional MiniLM sentence-transformer target was found.

## Cached retrieval datasets

These are the only cached MTEB retrieval datasets that load successfully through `load_mteb_data` with both offline flags enabled:

| Harness task name | HF dataset id | Corpus | Queries | Queries with qrels | Qrel entries | Offline result |
|---|---|---:|---:|---:|---:|---|
| `SciFact` | `mteb/scifact` | 5,183 | 300 | 300 | 339 | PASS |
| `NFCorpus` | `mteb/nfcorpus` | 3,633 | 323 | 323 | 12,334 | PASS |
| `FiQA2018` | `mteb/fiqa` | 57,638 | 648 | 648 | 1,706 | PASS |

`FiQA` is not the registered task name in the installed MTEB version; use `FiQA2018`.

`ArguAna`, `SCIDOCS`, and `TRECCOVID` were checked through the same loader and fail in offline mode because `mteb/arguana`, `mteb/scidocs`, and `mteb/trec-covid` are absent. No other BEIR/MTEB retrieval dataset cache was found. SciFact and NFCorpus are listed for completeness but are not new dataset axes.

## Concrete new regimes achievable offline

The four existing regimes are SciFact/NFCorpus crossed with MiniLM -> mpnet and MiniLM -> bge-large, all at anchor fraction 0.4.

The following four **new** regimes require no downloads and reach eight total:

1. `fiqa2018_minilm_to_mpnet_af040`: `FiQA2018`, `sentence-transformers/all-MiniLM-L6-v2` -> `sentence-transformers/all-mpnet-base-v2`, anchor fraction 0.40.
2. `fiqa2018_minilm_to_bge_large_af040`: `FiQA2018`, `sentence-transformers/all-MiniLM-L6-v2` -> `BAAI/bge-large-en-v1.5`, anchor fraction 0.40.
3. `scifact_minilm_to_mpnet_af020`: `SciFact`, MiniLM -> mpnet, anchor fraction 0.20 (distinct from the existing 0.40 anchor setting).
4. `nfcorpus_minilm_to_bge_large_af020`: `NFCorpus`, MiniLM -> bge-large, anchor fraction 0.20 (distinct from the existing 0.40 anchor setting).

The harness and estimator contract explicitly treat anchor setting as part of the held-out regime axis, and `load_retrieval_evalsplit` accepts `anchor_fraction`. No unimplemented drift-magnitude axis is being assumed here. The exact 0.20 selection is proposed, not previously validated; building it would require adding regime specifications, which this enumeration deliberately does not do.

Brutal-honesty boundary: if the scientific target requires eight unique **dataset x encoder-family** cells and does not accept anchor-fraction changes as separate regimes, the answer is **NOT FEASIBLE** from cache. Only six such cells are currently harness-buildable: the existing four plus FiQA2018 -> mpnet and FiQA2018 -> bge-large. Nomic is not counted because the current backend cannot construct it, and changing only an anchor fraction does not add a dataset or encoder family.

## Verdict

**FEASIBLE offline right now for the stated D3 definition, which names dataset / encoder family / anchor setting as the regime axis:** four cached-data/model regimes can be added to the existing four, as listed above, reaching exactly eight. This is a cache-feasibility verdict, not evidence that the regimes have been built or that their oracle gaps will be positive; `run_estimator.py` rejects non-positive oracle gaps, so successful eventual pack construction is not guaranteed by cache presence alone.

For the stricter eight-unique-dataset/encoder-family interpretation, the maximum is **six**, and the smallest download-only unblock is one additional harness-compatible sentence-transformer target (for example `BAAI/bge-base-en-v1.5`, `intfloat/e5-base-v2`, or a GTE model) whose normal `SentenceTransformer(model_name)` constructor works offline after caching. Crossing that one target with SciFact and NFCorpus would add two cells and reach eight. Alternatively, downloading one additional retrieval dataset with corpus, queries, and qrels adds only two cells with the presently usable swap targets and also reaches eight.

Missing requested examples: `mteb/arguana`, `mteb/scidocs`, `mteb/trec-covid`; all GTE/E5/GTR models; BGE base/small models; and extra MiniLM variants. Nomic needs no model download, but it needs a deliberate harness code change to opt into trusted remote code before it can honestly be counted.
