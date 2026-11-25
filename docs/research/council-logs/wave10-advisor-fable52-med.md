# Wave 10 Co-Advisor & Methodology Judge Report (`L1–L5`, Demos `D6–D10`)

- **`panel_type`:** `architecture-planning`
- **`artifact`:** `docs/research/council-logs/wave10-r0-tier1-tier2-master-spec.md`
- **`run_date`:** `2026-11-23`
- **`run_id`:** `wave10-r0-d6-d10-spec`
- **`challenge_depth`:** `1`
- **`active_experts`:** `["claude-fable-5-2 (Advisor/Judge, medium)"]`
- **`persist_version`:** `1.1.0`

**Scope and limits.** Spec read in full; grep-level spot checks of `ek5_v3_stratified.py`, `wave6_common.py`, `demo_workbench/server.py`, `scripts/run_wave6_d4_trained_epistemic_grpo.py`, and the non-NDA teaser. Nothing was executed. One reviewer, one pass; lenses are not independent models. Library and engine facts below are from memory, not checked against live docs, so confirm before freezing.

## Spec defects to fix before Wave 11

1. **`D6` pgvector cannot HNSW-index `4096-d` (spec line 32).** pgvector indexes `vector` up to 2,000 dims and `halfvec` up to 4,000. The native shard needs exact scan or MRL truncation, and the receipt must say which. Labelling a sequential scan "HNSW" is a false claim.
2. **`D9-B` `EVAL` is no longer a holdout (lines 74, 81).** The curriculum is designed from a diagnosis of *`EVAL`* failure topologies (`odd_refutation_cycle`, `contested_successor`). Byte-identical files do not cure adaptive overfitting. Generate and hash-seal a fresh `EVAL-2` (new seed, unseen subspace) before any training; make it PRIMARY and report old `EVAL` as development-exposed.
3. **`D10` size arithmetic is wrong (line 92).** `8 B/doc` vs `1024-d` float32 is exactly `512×` (`4096-d`: `2048×`), not `524×`/`2,097×`. Full rewrites are `19.16 GiB` and `76.66 GiB`, not `20.09`/`80.38`.
4. **`D10` MSMARCO is a subsample.** BEIR MSMARCO has `8.84M` passages; the spec uses exactly `4.40M`. Subsampling while keeping qrels positives inflates `nDCG@10`. "100% real" holds; "MSMARCO dev" comparability does not.
5. **Model identity conflict.** The spec says `Qwen2.5-72B-bnb-4bit`; the review brief says `Llama-3.3-70B`. Pick one. Disclose 4-bit quantization wherever "frontier" appears.
6. **"Cryptographically signed/sealed" (lines 15, 69).** A SHA-256 digest is not a signature. Add a real keyed signature with a published verify key, or say "hash-sealed".

## Acceptance conditions

**`L1` (`D6`)**
- Manifest records each `RepoDigests` value (`qdrant/qdrant` is untagged in the spec), server-reported version, and post-ingest counts from the engine (`_count`, `SELECT count(*)`, `points_count`) equal to the expected shard sizes.
- Score inversion is unit-tested: OpenSearch `dot = s − 1` if `s ≥ 1`, else `1 − 1/s`; pgvector `dot = −(<#>)`. On ≥1,000 returned hits per engine, recovered dot matches client-side numpy within `1e-4` relative. Any mismatch fails the run.
- Recall ground truth is brute force over the *visible* (post-retraction) set. `k_os`/`ef_search` are frozen on DEV queries; a tune-until-`0.995` loop on test queries is not acceptable. Report queries returning `<10` hits, since post-filtered HNSW starves at 50–90% retraction.
- Stale-leak count is computed from raw engine responses and persisted per query, not asserted.

**`L2` (`D7`)**
- Per case: Neo4j node and edge counts after ingest (Cypher `count`), Graphiti LLM-call count, and retrieved-context length. Any case with zero edges or empty context is tagged `ARM_FAILURE` and reported separately, never scored as a wrong answer.
- Zero broad `except` around `add_episode`/`search`; exceptions land in the manifest. The JSON adapter's repair rate is published.
- `Delta=0s` is reported as an architectural latency property, separate from accuracy; also report accuracy after Graphiti ingestion completes.
- Suite B (`N=20`) is descriptive only, with no superiority language.

**`L3` (`D8`)**
- `server.py:282` already serves a `SYNTHETIC_NUMPY_MODE_B_SIMULATOR`; no `/copilot` path may call it.
- Mutation test: a custom retraction with a never-seen ID flips the verdict and changes the ledger hash; removing it restores both. A test asserts answers are not literals in engine source.
- Every response carries the oracle cross-check result; a disagreement surfaces as an error.
- Non-NDA scan runs over rendered DOM, all API JSON (keys, enum values, error strings), the exported certificate, and static JS/CSS, not only the happy-path body.

**`L4` (`D9`)**
- The disjointness gate reuses the fail-closed `assert_four_way_disjoint_splits` pattern (`run_wave6_d4…py:718`), extended to the `N=600` extractor set vs all `1,020` Suite A documents, plus n-gram and embedding near-duplicate checks. Entity-set disjointness alone misses template paraphrase.
- Anti-collapse: with the `−0.20`/`+1.00` shaping, always-`ABSTAIN` earns `+0.40` at 50% `U` sampling. Gate on `T/F` determinate accuracy non-inferior to SFT (pre-registered margin), `abstain_rate` on `T/F` reported, and an always-`ABSTAIN` baseline row in the results table.
- PRIMARY is the fixed final step, 3-seed mean, Holm-corrected; no best-seed or best-checkpoint headline.
- The extractor's `93.3%` (56/60) claim reports a paired CI vs the 30B extractor, not only vs Mem0.

**`L5` (`D10`)**
- Global IDs are `{dataset}:{orig_id}`; assert uniqueness equals `5,023,716` and every qrels ID resolves. BEIR numeric IDs collide across MSMARCO, Quora, FiQA and SciFact. Evaluate each query set with its own qrels against the merged corpus; exclude ArguAna self-matches.
- Manifest states the exact FAISS factory string, `M`, `efSearch`, `nprobe`, PQ settings, measured peak RSS, and recall vs a Flat ground truth on a sampled query set. No "HNSW" label on IVF-PQ numbers.
- Reader sweep: every arm gets the same raw documents, token budget, system prompt and decoding. "Bare" means no governance layer, not no context. The judge is cross-family for every arm.

## Unknowns

- `graphiti-core 0.31.2` existence and its Nemotron compatibility: unverified.
- `bitsandbytes` 4-bit on GB10/aarch64: unverified; a fallback changes the claim.
- The teaser cites `OpenSearch 3.9 gRPC` while `D6` pins `3.0.0`; reconcile before either ships.

Reviewer: `claude-fable-5-2`, medium effort, depth 1, static review only. Runtime proof is owed by Wave 12.
