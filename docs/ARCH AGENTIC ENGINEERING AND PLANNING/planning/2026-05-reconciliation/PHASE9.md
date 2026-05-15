# Phase 9: Performance Optimization & Scalability
# Phase 10: Monitoring, Observability & Production Readiness
#
# ChelatedAI - Detailed Improvement Plan
# Generated: 2026-04-22

================================================================================
PHASE 9: PERFORMANCE OPTIMIZATION & SCALABILITY
================================================================================

This phase addresses the 10 most impactful performance bottlenecks identified
across the flat-file repository (180+ Python files at root). The AntigravityEngine
(~1621 lines in antigravity_engine.py) is the primary target, along with
dashboard_server.py (~928 lines) and embedding_backend.py (325 lines).

================================================================================
9.1 Profiling Infrastructure
================================================================================

**Goal:** Establish measurable performance baselines before making changes.

**Action 1: Create `profile_engine.py` — Engine-level profiling harness**

Create a new module at the project root that profiles the full AntigravityEngine
pipeline. This uses `cProfile`, `line_profiler`, and `memory_profiler`.

```
profile_engine.py
├── ProfileEmbedding() — time embed() for single, small (10), and large (1000) batches
├── ProfileIngestion() — time ingest() across corpus sizes: 100, 1000, 10000
├── ProfileSearch() — time search() at k=10, 50, 100 with varying collection sizes
├── ProfileTraining() — time sedimentation training loops (1 epoch, 5 epochs)
├── ProfileAdapter() — time adapter forward pass for varying batch sizes
└── ProfileFullPipeline() — ingest 1000 docs, run 100 queries, train 5 epochs
```

Key code patterns:

```python
# In profile_engine.py
import cProfile
import pstats
import tracemalloc
from antigravity_engine import AntigravityEngine
from config import ChelationConfig

def profile_embed(engine: AntigravityEngine, sizes: list[int]):
    tracemalloc.start()
    for size in sizes:
        texts = [f"Document {i}" for i in range(size)]
        start = time.perf_counter()
        engine.embed(texts)
        elapsed = time.perf_counter() - start
        current, peak = tracemalloc.get_traced_memory()
        print(f"embed({size}): {elapsed:.3f}s, peak_memory={peak / 1024 / 1024:.1f}MB")
        tracemalloc.reset_peak()
```

**Action 2: Create `benchmarks/profile_baseline.py` — Shared benchmark infrastructure**

Currently 6 separate benchmark files have no shared infrastructure:
- benchmark_beir.py
- benchmark_comparative.py
- benchmark_distillation.py
- benchmark_evolution.py
- benchmark_multitask.py
- benchmark_rlm.py

Create a shared `benchmarks/` directory with:

```
benchmarks/
├── __init__.py
├── base_benchmark.py      — BaseBenchmark class with timing, memory tracking
├── profiling.py           — profile_engine integration, output as JSON
├── throughput.py          — docs/sec, queries/sec, embeddings/sec metrics
├── memory_profile.py      — RSS, peak memory, GC stats
└── generate_report.py     — Consolidates all benchmark outputs into report
```

Shared `BaseBenchmark` base class:

```python
# benchmarks/base_benchmark.py
import time
import json
import tracemalloc
import gc
from abc import ABC, abstractmethod
from typing import Dict, Any, List

class BaseBenchmark(ABC):
    def __init__(self, name: str):
        self.name = name
        self.results: List[Dict[str, Any]] = []

    def run(self, iterations: int = 3) -> Dict[str, Any]:
        gc.collect()
        timings = []
        for i in range(iterations):
            tracemalloc.start()
            start = time.perf_counter()
            result = self._benchmark_once()
            elapsed = time.perf_counter() - start
            _, peak_mem = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            timings.append({
                "iteration": i,
                "elapsed_seconds": elapsed,
                "peak_memory_mb": peak_mem / 1024 / 1024,
            })
        self.results.extend(timings)
        return self._format_results(timings)

    @abstractmethod
    def _benchmark_once(self) -> Any:
        pass

    def _format_results(self, timings: List[Dict]) -> Dict:
        avg_time = sum(t["elapsed_seconds"] for t in timings) / len(timings)
        avg_mem = sum(t["peak_memory_mb"] for t in timings) / len(timings)
        return {
            "benchmark": self.name,
            "avg_time_seconds": round(avg_time, 4),
            "min_time_seconds": min(t["elapsed_seconds"] for t in timings),
            "max_time_seconds": max(t["elapsed_seconds"] for t in timings),
            "avg_peak_memory_mb": round(avg_mem, 2),
        }
```

**Action 3: Create `benchmarks/throughput.py` — Throughput benchmarks**

```python
# benchmarks/throughput.py
from benchmarks.base_benchmark import BaseBenchmark
from antigravity_engine import AntigravityEngine
from config import ChelationConfig

class IngestionThroughputBenchmark(BaseBenchmark):
    def _benchmark_once(self):
        engine = AntigravityEngine(
            qdrant_location=":memory:",
            model_name="all-MiniLM-L6-v2"
        )
        docs = [f"Document {i}: Lorem ipsum dolor sit amet." for i in range(1000)]
        start = time.perf_counter()
        engine.ingest(docs)
        elapsed = time.perf_counter() - start
        self.throughput = 1000 / elapsed
        return {"throughput_docs_per_sec": self.throughput}

class QueryThroughputBenchmark(BaseBenchmark):
    def _benchmark_once(self):
        engine = AntigravityEngine(
            qdrant_location=":memory:",
            model_name="all-MiniLM-L6-v2"
        )
        engine.ingest([f"Document {i}" for i in range(1000)])
        start = time.perf_counter()
        for _ in range(100):
            engine.search("test query", top_k=10)
        elapsed = time.perf_counter() - start
        self.qps = 100 / elapsed
        return {"qps": self.qps}
```

**Estimated effort:** 2 days
**Priority:** CRITICAL — no performance improvements possible without baselines
**Files created:** `profile_engine.py`, `benchmarks/__init__.py`, `benchmarks/base_benchmark.py`,
`benchmarks/profiling.py`, `benchmarks/throughput.py`, `benchmarks/memory_profile.py`,
`benchmarks/generate_report.py`
**Files modified:** None initially

---

================================================================================
9.2 Embedding Optimization
================================================================================

**Goal:** Reduce embedding generation latency by 2-5x through batching, caching, and quantization.

**Action 1: Embedding cache in `embedding_backend.py`**

Add an LRU cache to `LocalEmbeddingBackend` and `OllamaEmbeddingBackend`:

```python
# In embedding_backend.py, modify LocalEmbeddingBackend:
import hashlib
from collections import OrderedDict

class EmbeddingCache:
    """Simple in-memory LRU cache for embeddings with TTL support."""

    def __init__(self, max_size: int = 10000, ttl_seconds: int = 3600):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._cache: OrderedDict = OrderedDict()
        self._timestamps: Dict[str, float] = {}
        self._hits = 0
        self._misses = 0

    def _key(self, text: str) -> str:
        return hashlib.sha256(text.encode('utf-8')).hexdigest()

    def get(self, text: str) -> Optional[np.ndarray]:
        key = self._key(text)
        if key in self._cache:
            if time.time() - self._timestamps.get(key, 0) > self.ttl_seconds:
                self._evict(key)
                return None
            self._cache.move_to_end(key)
            self._hits += 1
            return self._cache[key]
        self._misses += 1
        return None

    def put(self, text: str, embedding: np.ndarray):
        key = self._key(text)
        if len(self._cache) >= self.max_size and key not in self._cache:
            evicted_key, _ = self._cache.popitem(last=False)
            self._timestamps.pop(evicted_key, None)
        self._cache[key] = embedding
        self._timestamps[key] = time.time()

    @property
    def hit_rate(self) -> float:
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0
```

Integrate into `LocalEmbeddingBackend.embed_raw()`:

```python
def __init__(self, ...):
    # existing init...
    self._cache = EmbeddingCache(max_size=10000, ttl_seconds=3600)

def embed_raw(self, texts: List[str]) -> np.ndarray:
    if not texts:
        return np.array([])

    # Check cache first
    cached_embeddings = [None] * len(texts)
    uncached_texts = []
    uncached_indices = []

    for i, text in enumerate(texts):
        cached = self._cache.get(text)
        if cached is not None:
            cached_embeddings[i] = cached
        else:
            uncached_texts.append(text)
            uncached_indices.append(i)

    # Embed only uncached texts
    if uncached_texts:
        new_embeddings = self.model.encode(
            uncached_texts,
            convert_to_numpy=True,
            show_progress_bar=False
        ).astype(np.float32)
        for idx, emb in zip(uncached_indices, new_embeddings):
            cached_embeddings[idx] = emb
            self._cache.put(uncached_texts[uncached_texts.index(text)], emb)

    return np.array(cached_embeddings, dtype=np.float32)
```

**Action 2: Batch size optimization in `config.py`**

Add embedding-batch-specific configuration:

```python
# In config.py, add:
EMBEDDING_BATCH_SIZE_LOCAL = 256      # Optimized for GPU batch processing
EMBEDDING_BATCH_SIZE_OLLAMA = 10      # Rate-limited by Ollama API
EMBEDDING_CACHE_ENABLED = True
EMBEDDING_CACHE_MAX_SIZE = 10000
EMBEDDING_CACHE_TTL_SECONDS = 3600
EMBEDDING_CACHE_QUANTIZE = False      # Store FP16 in cache if True (saves ~50% memory)
```

**Action 3: Model quantization support**

Create `quantization.py` with FP16/INT8 quantization utilities:

```python
# quantization.py
import numpy as np
import torch
from typing import Tuple

def quantize_fp16(embeddings: np.ndarray) -> Tuple[np.ndarray, float, float]:
    """Quantize embeddings to FP16, returning (quantized, dtype, scale)."""
    fp32 = embeddings.astype(np.float32)
    fp16 = fp32.astype(np.float16)
    return fp16, np.float16, 1.0

def dequantize_fp16(embeddings: np.ndarray, dtype=np.float16, scale=1.0) -> np.ndarray:
    return (embeddings.astype(dtype) * scale).astype(np.float32)

def quantize_int8(embeddings: np.ndarray, axis: int = -1) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Quantize to INT8 with per-tensor scale/zero-point."""
    flat = embeddings.reshape(-1, embeddings.shape[-1])
    qmin, qmax = -128, 127
    s = (flat.max(axis=1) - flat.min(axis=1)).clip(min=1e-5)
    zp = -flat.min(axis=1) / s
    quantized = ((flat / s[:, None]) + zp[:, None]).clip(qmin, qmax).astype(np.int8)
    return quantized, s, zp
```

**Action 4: Batch embedding in `embedding_backend.py`**

Add configurable batch size to `model.encode()`:

```python
# In LocalEmbeddingBackend.embed_raw():
batch_size = ChelationConfig.EMBEDDING_BATCH_SIZE_LOCAL
all_embeddings = []
for i in range(0, len(texts), batch_size):
    batch = texts[i:i+batch_size]
    batch_emb = self.model.encode(batch, convert_to_numpy=True, show_progress_bar=False)
    all_embeddings.append(batch_emb)
return np.concatenate(all_embeddings, axis=0).astype(np.float32)
```

**Estimated effort:** 3 days
**Priority:** HIGH — embedding is the most frequent operation
**Files created:** `quantization.py`
**Files modified:** `embedding_backend.py`, `config.py`
**Expected improvement:** 2-3x embedding speedup via batching, 50-90% cache hit rate on repeated queries

---

================================================================================
9.3 Vector Search Optimization
================================================================================

**Goal:** Improve search latency and throughput through query batching and index tuning.

**Action 1: Query batching in `antigravity_engine.py`**

Add batched search method:

```python
# In antigravity_engine.py, add method:
def search_batch(self, queries: List[str], top_k: int = 10,
                 use_embedding_cache: bool = True) -> List[List]:
    """Search multiple queries in a single batch to reduce embedding overhead."""
    if use_embedding_cache:
        query_embeddings = self.embed(queries)
    else:
        query_embeddings = self.embedding_backend.embed_raw(queries)

    results = []
    for i, query_emb in enumerate(query_embeddings):
        points = self.qdrant.search(
            collection_name=self.collection_name,
            query_vector=query_emb.tolist(),
            limit=top_k
        )
        results.append([(p.id, p.score, p.payload) for p in points.points])
    return results
```

**Action 2: HNSW index configuration in `config.py`**

Add vector index optimization parameters:

```python
# In config.py, add:
HNSW_M = 16              # Number of edges per node (default: 16)
HNSW_EF_CONSTRUCT = 100  # Construction time budget (default: 100)
HNSW_EF_SEARCH = 64      # Search time budget (higher = more accurate, slower)
MAX_INDEX_SIZE_MB = 2048  # Target max index size
OPTIMIZE_ON_SYNC = True   # Run optimization on every sync
```

**Action 3: Create `index_optimizer.py`**

```python
# index_optimizer.py
from qdrant_client import QdrantClient
from qdrant_client.models import HnswConfigDiff, VectorParams, Distance

class IndexOptimizer:
    """Optimize Qdrant HNSW index for search performance."""

    def __init__(self, client: QdrantClient, collection_name: str):
        self.client = client
        self.collection_name = collection_name

    def optimize_hnsw(self, m: int = 16, ef_construct: int = 100) -> bool:
        """Update HNSW config for the collection."""
        return self.client.update_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(
                size=self._get_vector_size(),
                distance=Distance.COSINE,
                hnsw_config=HnswConfigDiff(m=m, ef_construct=ef_construct)
            )
        )

    def optimize_on_disk(self, max_index_size_mb: int = 2048) -> bool:
        """Optimize on-disk storage for large collections."""
        return self.client.recreate_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(
                size=self._get_vector_size(),
                distance=Distance.COSINE,
                on_disk=True
            )
        )

    def compact_segments(self) -> bool:
        """Request Qdrant to compact segment data."""
        return self.client.reload_collection(self.collection_name)
```

**Action 4: Sparse-dense hybrid search support in `vector_store.py`**

Extend `QdrantVectorStore` with payload-aware search:

```python
# In vector_store.py, add to QdrantVectorStore:
def search_with_payload(self, collection_name, query_vector, limit=10,
                        with_payload=True, with_vectors=False):
    """Search with explicit payload filtering to reduce network transfer."""
    return self._client.search(
        collection_name=collection_name,
        query_vector=query_vector.tolist() if hasattr(query_vector, 'tolist') else query_vector,
        limit=limit,
    )
```

**Estimated effort:** 1.5 days
**Priority:** HIGH — search is the most latency-sensitive operation
**Files created:** `index_optimizer.py`
**Files modified:** `antigravity_engine.py`, `config.py`, `vector_store.py`
**Expected improvement:** 30-50% search latency reduction with HNSW tuning, 2-3x throughput with batching

---

================================================================================
9.4 Ingestion Pipeline Optimization
================================================================================

**Goal:** Parallelize ingestion for large corpus loads.

**Action 1: Parallel ingestion in `antigravity_engine.py`**

Create `ingest_parallel()` method using ThreadPoolExecutor:

```python
# In antigravity_engine.py:
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

def ingest_parallel(self, texts, payloads=None, max_workers: int = 4,
                    batch_size: int = 100):
    """Parallel ingestion using thread pool for embedding + upsert."""
    if isinstance(texts, str):
        texts = [texts]
    if not texts:
        return {"total_docs": 0, "total_batches": 0}

    total_batches = (len(texts) + batch_size - 1) // batch_size
    success_count = 0
    failure_count = 0
    lock = Lock()

    def ingest_batch(batch_start: int) -> tuple:
        batch_texts = texts[batch_start:batch_start + batch_size]
        batch_payloads = (payloads[batch_start:batch_start + batch_size]
                         if payloads else [{}] * len(batch_texts))
        try:
            embeddings = self.embed(batch_texts)
            embeddings = np.asarray(embeddings)
            points = [
                PointStruct(
                    id=batch_start + j,
                    vector=embeddings[j],
                    payload=({**batch_payloads[j], "text": batch_texts[j]}
                            if self.store_full_text_payload else batch_payloads[j])
                )
                for j in range(len(batch_texts))
            ]
            self.qdrant.upsert(collection_name=self.collection_name, points=points)
            return (len(batch_texts), 0)
        except Exception as e:
            self.logger.log_error("parallel_ingestion", f"Batch {batch_start} failed", exception=e)
            return (0, len(batch_texts))

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(ingest_batch, i * batch_size)
                   for i in range(total_batches)]
        for future in as_completed(futures):
            s, f = future.result()
            with lock:
                success_count += s
                failure_count += f

    return {
        "total_docs": len(texts),
        "successful": success_count,
        "failed": failure_count,
        "success_rate": success_count / len(texts) if texts else 0,
    }
```

**Action 2: Async streaming ingestion**

Enhance existing `ingest_streaming()` with asyncio:

```python
async def ingest_streaming_async(self, texts_iterable, payloads_iterable=None,
                                  batch_size: int = 100):
    """Async streaming ingestion with semaphore-based concurrency."""
    import asyncio
    semaphore = asyncio.Semaphore(4)  # max 4 concurrent batches

    async def process_batch(batch):
        async with semaphore:
            embeddings = self.embed(batch["texts"])
            await self._async_upsert(batch["texts"], embeddings, batch["payloads"])

    queue = asyncio.Queue()
    # Feed queue from iterable, process in parallel
    return {"status": "complete"}
```

**Action 3: Create `ingestion_pipeline.py`**

```python
# ingestion_pipeline.py
class IngestionPipeline:
    """Configurable ingestion pipeline with multiple strategies."""
    def __init__(self, engine, strategy="sequential", **kwargs):
        self.engine = engine
        self.strategy = strategy
        self.kwargs = kwargs

    def run(self, texts, payloads=None):
        if self.strategy == "sequential":
            return self.engine.ingest(texts, payloads)
        elif self.strategy == "parallel":
            return self.engine.ingest_parallel(texts, payloads, **self.kwargs)
        elif self.strategy == "streaming":
            return self.engine.ingest_streaming(iter(texts), iter(payloads) if payloads else None, **self.kwargs)
        elif self.strategy == "streaming_async":
            import asyncio
            return asyncio.run(self.engine.ingest_streaming_async(iter(texts), iter(payloads) if payloads else None, **self.kwargs))
```

**Estimated effort:** 2 days
**Priority:** HIGH — ingestion is the main bulk data operation
**Files created:** `ingestion_pipeline.py`
**Files modified:** `antigravity_engine.py`
**Expected improvement:** 3-4x ingestion throughput for local models, 2-3x for Ollama

---

================================================================================
9.5 Training Performance
================================================================================

**Goal:** Accelerate sedimentation training with gradient accumulation and
mixed precision.

**Action 1: Mixed precision training support**

Create `training_acceleration.py`:

```python
# training_acceleration.py
import torch
import torch.cuda.amp as amp

class MixedPrecisionTrainer:
    """Adds mixed precision (FP16) support to training loops."""
    def __init__(self, enabled: bool = True):
        self.enabled = enabled and torch.cuda.is_available()
        self.scaler = amp.GradScaler() if self.enabled else None

    def forward(self, model, inputs, **kwargs):
        if self.enabled:
            with amp.autocast():
                return model(inputs, **kwargs)
        return model(inputs, **kwargs)

    def backward(self, loss):
        if self.enabled:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

    def step(self, optimizer):
        if self.enabled:
            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            optimizer.step()

    def zero_grad(self, optimizer):
        optimizer.zero_grad()
```

**Action 2: Gradient accumulation in `sedimentation_trainer.py`**

Modify training functions to support accumulation:

```python
def train_with_accumulation(engine, corpus, epochs=10, accumulation_steps=4,
                            learning_rate=0.001, mixed_precision=False):
    """Training with gradient accumulation for effective large-batch training."""
    from training_acceleration import MixedPrecisionTrainer
    accelerator = MixedPrecisionTrainer(enabled=mixed_precision)
    optimizer = torch.optim.Adam(engine.adapter.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        for batch_start in range(0, len(corpus), accumulation_steps):
            batch_end = min(batch_start + accumulation_steps, len(corpus))
            for i in range(batch_start, batch_end):
                loss = compute_sedimentation_loss(...)
                accelerator.backward(loss)
            accelerator.step(optimizer)
            accelerator.zero_grad(optimizer)
```

**Action 3: Create `training_profiler.py`**

```python
# training_profiler.py
class TrainingProfiler:
    """Profile sedimentation training: epoch time, GPU utilization, memory."""
    def __init__(self, engine):
        self.engine = engine
        self.metrics = []

    def profile_epoch(self, epoch: int, corpus: list, loss_fn=None):
        start = time.perf_counter()
        gpu_start = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        self.engine.train(corpus, epochs=1)
        elapsed = time.perf_counter() - start
        gpu_end = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        self.metrics.append({
            "epoch": epoch,
            "elapsed_seconds": elapsed,
            "gpu_memory_allocated_mb": (gpu_end - gpu_start) / 1024 / 1024,
        })
        return self.metrics[-1]
```

**Estimated effort:** 2 days
**Priority:** MEDIUM — only relevant when GPU is available
**Files created:** `training_acceleration.py`, `training_profiler.py`
**Files modified:** `sedimentation_trainer.py`, `config.py`
**Expected improvement:** 2-3x training speedup with mixed precision on Volta+ GPUs

---

================================================================================
9.6 Memory Optimization
================================================================================

**Goal:** Reduce memory footprint for large datasets.

**Action 1: Memory-mapped storage for embeddings**

Create `memory_mapped_storage.py`:

```python
# memory_mapped_storage.py
import numpy as np
from pathlib import Path

class MemoryMappedEmbeddingStore:
    """Store embeddings on disk using memory-mapped files."""
    def __init__(self, filepath: str, vector_dim: int, dtype=np.float32):
        self.filepath = Path(filepath)
        self.vector_dim = vector_dim
        self.dtype = dtype
        self._n_vectors = 0

    def reserve(self, n_vectors: int):
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
        self._n_vectors = n_vectors
        self._mmapped = np.memmap(
            self.filepath, dtype=self.dtype, mode='w+',
            shape=(n_vectors, self.vector_dim)
        )

    def add(self, index: int, embedding: np.ndarray):
        self._mmapped[index] = embedding

    def get(self, index: int) -> np.ndarray:
        return self._mmapped[index].copy()

    def get_batch(self, indices: list) -> np.ndarray:
        return self._mmapped[indices].copy()

    def close(self):
        if hasattr(self, '_mmapped'):
            del self._mmapped
```

**Action 2: LRU cache for embeddings (covered in 9.2 Action 1)**

**Action 3: Garbage collection tuning in `config.py`**

```python
# In config.py, add:
GC_ENABLED = True
GC_THRESHOLD = 700
GC_EMBEDDINGS_AFTER_INGEST = True
GC_EMBEDDINGS_AFTER_TRAIN = True
```

**Action 4: Lazy model loading in `embedding_backend.py`**

Add lazy model loading to avoid keeping the full model in memory when not in use:

```python
# In LocalEmbeddingBackend:
def __init__(self, model_name, logger=None, lazy: bool = False):
    self._lazy = lazy
    self._model = None

@property
def model(self):
    if self._model is None:
        from sentence_transformers import SentenceTransformer
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self._model = SentenceTransformer(self.model_name, device=device)
        self._vector_size = self._model.get_sentence_embedding_dimension()
    return self._model

def unload_model(self):
    if self._model is not None:
        del self._model
        self._model = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
```

**Estimated effort:** 2 days
**Priority:** MEDIUM — important for large corpus scenarios (>100k docs)
**Files created:** `memory_mapped_storage.py`
**Files modified:** `embedding_backend.py`, `config.py`
**Expected improvement:** 50-70% memory reduction for large datasets via mmapped storage

---

================================================================================
9.7 Caching Strategy
================================================================================

**Goal:** Multi-layer caching: embedding cache, query result cache, metadata cache.

**Action 1: Query result cache in `antigravity_engine.py`**

Add LRU cache for search results:

```python
# In antigravity_engine.py, add to __init__:
from collections import OrderedDict

class LRUCache:
    """Thread-safe LRU cache for query results."""
    def __init__(self, max_size: int = 1000, ttl_seconds: int = 300):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._cache: OrderedDict = OrderedDict()
        self._timestamps: Dict[str, float] = {}
        self._lock = Lock()

    def get(self, key: str):
        with self._lock:
            if key in self._cache:
                if time.time() - self._timestamps.get(key, 0) > self.ttl_seconds:
                    del self._cache[key]
                    del self._timestamps[key]
                    return None
                self._cache.move_to_end(key)
                return self._cache[key]
            return None

    def set(self, key: str, value):
        with self._lock:
            if len(self._cache) >= self.max_size and key not in self._cache:
                evicted, _ = self._cache.popitem(last=False)
                self._timestamps.pop(evicted, None)
            self._cache[key] = value
            self._timestamps[key] = time.time()

    def invalidate(self, prefix: str = None):
        with self._lock:
            if prefix:
                keys_to_remove = [k for k in self._cache if k.startswith(prefix)]
                for k in keys_to_remove:
                    del self._cache[k]
                    del self._timestamps[k]
            else:
                self._cache.clear()
                self._timestamps.clear()
```

**Action 2: Integrate cache into `search()` method**

```python
# In antigravity_engine.search():
def search(self, query, top_k=10, use_cache: bool = True):
    cache_key = f"search:{hashlib.md5(query.encode()).hexdigest()}:{top_k}"
    if use_cache:
        cached = self._result_cache.get(cache_key)
        if cached is not None:
            return cached
    # original search logic...
    result = [(p.id, p.score, p.payload) for p in points.points]
    if use_cache:
        self._result_cache.set(cache_key, result)
    return result
```

**Action 3: Create `cache_manager.py` — Unified cache layer**

```python
# cache_manager.py
from collections import OrderedDict
import time
import hashlib

class CacheManager:
    """Unified caching layer with multiple cache types."""
    def __init__(self):
        self.embedding_cache = LRUCache(max_size=10000, ttl_seconds=3600)
        self.query_cache = LRUCache(max_size=1000, ttl_seconds=300)
        self.metadata_cache = LRUCache(max_size=5000, ttl_seconds=7200)
        self.adapter_cache = {}

    def get_stats(self) -> dict:
        return {
            "embedding": self.embedding_cache.stats(),
            "query": self.query_cache.stats(),
            "metadata": self.metadata_cache.stats(),
        }
```

**Estimated effort:** 1 day
**Priority:** MEDIUM — benefits interactive/iterative workflows
**Files created:** `cache_manager.py`
**Files modified:** `antigravity_engine.py`
**Expected improvement:** 5-10x query latency reduction for repeated searches

---

================================================================================
9.8 Benchmark-Driven Optimization
================================================================================

**Goal:** Every optimization must be validated with before/after benchmark comparison.

**Action: Create `benchmarks/generate_report.py`**

```python
# benchmarks/generate_report.py
import json
from pathlib import Path
from datetime import datetime

class BenchmarkReport:
    """Generate comparison reports between benchmark runs."""
    def __init__(self, baseline_path: str, optimized_path: str):
        with open(baseline_path) as f:
            self.baseline = json.load(f)
        with open(optimized_path) as f:
            self.optimized = json.load(f)

    def generate(self) -> dict:
        report = {
            "timestamp": datetime.utcnow().isoformat(),
            "improvements": {},
        }
        for key in self.baseline:
            if key in self.optimized:
                old = self.baseline[key].get("avg_time_seconds", 0)
                new = self.optimized[key].get("avg_time_seconds", 0)
                if old > 0:
                    improvement = ((old - new) / old) * 100
                    report["improvements"][key] = {
                        "baseline_sec": round(old, 4),
                        "optimized_sec": round(new, 4),
                        "improvement_pct": round(improvement, 2),
                        "speedup": round(old / new, 2),
                    }
        return report
```

**Estimated effort:** 0.5 days
**Priority:** CRITICAL — validates all other Phase 9 work
**Files created:** `benchmarks/generate_report.py` (extends existing benchmark infrastructure)

---

================================================================================
9.9 GPU Utilization
================================================================================

**Goal:** Maximize GPU throughput for embedding and training operations.

**Action 1: Multi-GPU embedding support in `embedding_backend.py`**

```python
# In embedding_backend.py, add MultiGPULocalEmbeddingBackend:
class MultiGPULocalEmbeddingBackend(EmbeddingBackend):
    """Distribute embedding across multiple GPUs."""
    def __init__(self, model_name: str, logger=None):
        super().__init__(model_name, logger)
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available")
        self.num_gpus = torch.cuda.device_count()
        self._models = []
        for gpu_id in range(self.num_gpus):
            device = f"cuda:{gpu_id}"
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer(model_name, device=device)
            self._models.append((gpu_id, model))

    def embed_raw(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.array([])
        batch_size = max(1, len(texts) // self.num_gpus)
        results = []
        for gpu_id, model in self._models:
            start = gpu_id * batch_size
            end = start + batch_size if gpu_id < self.num_gpus - 1 else len(texts)
            batch = texts[start:end]
            if not batch:
                continue
            device = f"cuda:{gpu_id}"
            with torch.cuda.device(device):
                emb = model.encode(batch, convert_to_numpy=True, show_progress_bar=False)
                results.append(emb)
        return np.concatenate(results, axis=0).astype(np.float32)
```

**Action 2: GPU configuration in `config.py`**

```python
# In config.py, add:
GPU_ENABLED = True
GPU_DEVICE = "cuda:0"
GPU_FP16 = True
GPU_MAX_MEMORY_FRAC = 0.9
```

**Action 3: GPU utilization monitoring**

Create `gpu_monitor.py`:

```python
# gpu_monitor.py
import pynvml

class GPUUtilizationMonitor:
    """Monitor GPU utilization in real-time."""
    def __init__(self):
        pynvml.nvmlInit()

    def get_utilization(self) -> dict:
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
        return {
            "gpu_util_pct": util.gpu,
            "memory_total_mb": mem.total / 1024 / 1024,
            "memory_used_mb": mem.used / 1024 / 1024,
        }

    def __del__(self):
        pynvml.nvmlShutdown()
```

**Estimated effort:** 2 days
**Priority:** MEDIUM — only useful with multi-GPU setups
**Files created:** `gpu_monitor.py`
**Files modified:** `embedding_backend.py`, `config.py`
**Expected improvement:** 2-4x throughput per second with multiple GPUs

---

================================================================================
9.10 I/O Optimization
================================================================================

**Goal:** Reduce disk I/O bottlenecks for checkpoint loading, log reading, and file operations.

**Action 1: Buffered I/O for checkpoint loading in `checkpoint_manager.py`**

Modify `load_checkpoint()` to use mmap for large files:

```python
def load_checkpoint(self, path: str) -> Optional[dict]:
    import mmap
    with open(path, 'rb') as f:
        file_size = os.path.getsize(path)
        if file_size > 10 * 1024 * 1024:
            with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
                data = mm.read()
        else:
            data = f.read()
    sha256 = hashlib.sha256(data).hexdigest()
    if sha256 != self.expected_hash:
        self.logger.log_error("checkpoint", f"SHA256 mismatch: {sha256}")
        return None
    return torch.load(io.BytesIO(data), map_location='cpu')
```

**Action 2: Async file I/O for log writing**

Create `async_io.py`:

```python
# async_io.py
import asyncio
import aiofiles

class AsyncLogWriter:
    """Async log writer for high-throughput logging."""
    def __init__(self, path: str, buffer_size: int = 1000, flush_interval: int = 100):
        self.path = path
        self.buffer = []
        self.buffer_size = buffer_size
        self.flush_interval = flush_interval
        self._lock = asyncio.Lock()
        self._counter = 0

    async def write(self, entry: dict):
        async with self._lock:
            self.buffer.append(json.dumps(entry) + "\n")
            self._counter += 1
            if self._counter >= self.flush_interval:
                await self._flush()

    async def _flush(self):
        if not self.buffer:
            return
        async with aiofiles.open(self.path, 'a') as f:
            await f.writelines(self.buffer)
        self.buffer.clear()
        self._counter = 0
```

**Action 3: Buffered reading for large files**

Modify `dashboard_server.py` `load_events()` to use buffered reads:

```python
def load_events_buffered(log_file: str, chunk_size: int = 8192):
    """Load events using buffered reads."""
    events = []
    buffer = ""
    with open(log_file, 'r', encoding='utf-8') as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                if buffer.strip():
                    try:
                        events.append(json.loads(buffer.strip()))
                    except json.JSONDecodeError:
                        pass
                break
            buffer += chunk
            lines = buffer.split('\n')
            buffer = lines[-1]
            for line in lines[:-1]:
                line = line.strip()
                if line:
                    try:
                        events.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
    return events
```

**Action 4: I/O configuration in `config.py`**

```python
# In config.py, add:
IO_BUFFER_SIZE = 8192
IO_CHECKPOINT_MMAP_THRESHOLD = 10 * 1024 * 1024
IO_LOG_BUFFER_SIZE = 1000
IO_ASYNC_LOGGING = False
```

**Estimated effort:** 1 day
**Priority:** LOW-MEDIUM
**Files created:** `async_io.py`
**Files modified:** `checkpoint_manager.py`, `dashboard_server.py`, `config.py`

---

================================================================================
PHASE 9 SUMMARY
================================================================================

Total estimated effort: ~14 days (can be parallelized across team members)

Priority ordering:
1. CRITICAL: 9.1 Profiling (no changes without baselines)
2. CRITICAL: 9.8 Benchmark-driven optimization (validates everything)
3. HIGH: 9.2 Embedding optimization (caching + batching)
4. HIGH: 9.3 Vector search optimization (HNSW tuning)
5. HIGH: 9.4 Ingestion pipeline optimization (parallel)
6. MEDIUM: 9.5 Training performance (mixed precision, grad accumulation)
7. MEDIUM: 9.6 Memory optimization (mmapped storage)
8. MEDIUM: 9.7 Caching strategy (query results + metadata)
9. MEDIUM: 9.9 GPU utilization (multi-GPU support)
10. LOW-MEDIUM: 9.10 I/O optimization (async, buffered)

Key new files to create:
- profile_engine.py
- benchmarks/__init__.py, base_benchmark.py, profiling.py, throughput.py, memory_profile.py, generate_report.py
- quantization.py
- index_optimizer.py
- ingestion_pipeline.py
- training_acceleration.py, training_profiler.py
- memory_mapped_storage.py
- cache_manager.py
- gpu_monitor.py
- async_io.py

Key files to modify:
- antigravity_engine.py (search batching, parallel ingest, result cache)
- embedding_backend.py (embedding cache, quantization, batch size, multi-GPU)
- config.py (all new performance configuration parameters)
- sedimentation_trainer.py (gradient accumulation)
- checkpoint_manager.py (buffered I/O)
- dashboard_server.py (buffered file reading)
