# Principal Systems Engineering Design & Risk Memo

- **`panel_type`:** `architecture-planning`
- **`artifact`:** `docs/research/council-logs/wave10-factory-sol62-high.md`
- **`run_date`:** `2026-11-23`
- **`run_id`:** `wave10-r0-d6-d10-spec`
- **`challenge_depth`:** `1`
- **`active_experts`:** `["gpt-6.2-sol (High, Factory)"]`
- **`persist_version`:** `1.1.0`

## L1 · D6 backend score contract

Use uncompressed float `knn_vector` HNSW indexes per dimension with `space_type: "innerproduct"`. OpenSearch 3.0 Faiss translates distance `−dot`; Lucene translates `dot`. Both expose `s=1/(1−dot)` for `dot≤0` and `s=1+dot` otherwise. For bare k-NN, invert in float64 as `1−1/s` for `0<s≤1`, else `s−1`. Set query `k` and response `size` to the shortlist depth on each logical shard. Set `_source: {"excludes":["embedding"]}` and omit vector fields. Rounded float32 scores can collapse distinct dots to the same value. Require maximum absolute error `<1e−6` against a float64 oracle using identical indexed operands, tested across both engines, both dimensions, and negative, zero, positive, and near-tie scores. Unit-norm operands limit inversion sensitivity. For bitwise recovery or failed tolerance, return a server-computed raw scalar from original vectors with defined accumulation and rounding. Record REST or verified native gRPC per arm. [OpenSearch 3.0 score definitions](https://docs.opensearch.org/3.0/field-types/supported-field-types/knn-spaces/)

PostgreSQL sorts ascending by bare `embedding <#> $1` and projects `-(embedding <#> $1) AS raw_dot`. HNSW caps `vector_ip_ops` at **2,000** dimensions and `halfvec_ip_ops` at **4,000** since v0.7.0. A `halfvec(4096)` column is valid, but stock HNSW rejects its index. Keep `vector(384)` HNSW and use exact admitted-row scoring over `vector(4096)` for full-dimensional parity. Pin the extension version and record `EXPLAIN` for each route. [pgvector v0.7.0 limits](https://github.com/pgvector/pgvector/blob/v0.7.0/README.md#hnsw)

## L1 · Retraction recall

D5’s 50% retraction cell reached **0.994186** against a **0.995** floor. It used `k_os=64`, scaled ef, and allowed approximate-prefix certification. The gateway never passes its authoritative allowlist to backend shortlist calls. Pin admission and index generation together. Use OpenSearch’s in-query filter, Qdrant DOT with a payload/ID filter and vectors disabled, and PostgreSQL partition/partial-index predicates where applicable. PostgreSQL residual predicates apply after ANN traversal, so enable iterative scans on v0.8+ or use an exact admitted join. Under lag, retrieve a superset of admitted IDs and enforce the epoch fence before rescoring. [D5 runner](../../../scripts/run_wave6_d5_grpc_interceptor_gateway.py#L1813) · [Gateway](../../../chelated_grpc_gateway.py#L1541) · [pgvector filtering](https://github.com/pgvector/pgvector#filtering)

Scale post-filter shortlist depth from `ceil(64/p_admit)` (~128 at 50%) and tune ef independently. Refill geometrically within budget, then use filtered OpenSearch `knn_score`, PostgreSQL exact score scans, or Qdrant `exact=true` with quantized scoring bypassed. Certify only verified admitted exhaustion or an exact dot-prefix upper bound below the kth score, accounting for rounding and exact or bounded CSLS query penalties across shards. A short page can still omit admitted IDs. Replace the gateway’s unconditional exhaustive-fallback success flag with verified coverage. Apply the same proof to single and batch paths. Gate each backend at 0/10/50/90% with matched-epoch exhaustive ZeroMu/CSLS ground truth, deterministic ties, admitted Recall@10 ≥0.995, and zero stale hits. Log fallback rate and p99. [Gateway stop logic](../../../chelated_grpc_gateway.py#L1626)

## L5a · D10 memory

4,018,973 × 384 + 1,004,743 × 1,024 = 2,572,142,464 components. Flat requires 10,288,569,856 B (9.582 GiB). SQ `QT_fp16` requires 5,144,284,928 B (4.791 GiB). Graph bytes are `4 × allocated link slots + 12N + 8` per shard on 64-bit. Expected graph bytes per node are `4 × (2M + M/(M−1)) + 12`, before capacity slack. Expected graph cost is 0.675/1.273 GiB at M=16/32, including upper layers. With the 38.328 MiB sidecar, expect Flat 10.294/10.893 GiB and SQ 5.503/6.102 GiB. [Faiss graph allocation](https://github.com/facebookresearch/faiss/blob/main/faiss/impl/HNSW.cpp) · [Scalar quantizer](https://github.com/facebookresearch/faiss/blob/main/faiss/impl/ScalarQuantizer.cpp)

Start with Flat M=16 and `METRIC_INNER_PRODUCT` and retain restartable disk shards. Retaining only the migrated legacy copies adds ~1.70 GiB at M=32. Two full Flat M=32 generations require ~21.79 GiB before transient allocations, leaving 30.21–53.21 GiB of the stated free RAM. Stream one build at a time in ≤65,536-row float32 batches (256 MiB at 1,024 dimensions). Cap build/search concurrency and measure allocator growth, page cache, IDs, epoch masks, and unified GPU pressure. Admit each phase only with measured peak demand plus 8 GiB reserve below live availability and applicable memory limits. Pause or reduce shard size before exceeding that budget. Measure old/new overlap under live query load, including exact-fallback scratch and disk-backed vectors. Qualify SQ recall against the original-vector oracle before promotion.
