# Phase 4 Experiment Protocol

## Scope

Phase 4 adds four capabilities for larger-scale testing and analysis:

1. Memory-aware ingestion (`ingest_streaming`)
2. Chelation log capping
3. Adaptive threshold tuning (opt-in)
4. Multi-task benchmark runner + log dashboard

All features are backward compatible by default.

---

## 1) Streaming ingestion for large corpora

`AntigravityEngine.ingest_streaming(...)` processes iterables in batches without requiring a full in-memory corpus list.

```python
from antigravity_engine import AntigravityEngine

engine = AntigravityEngine(
    qdrant_location=":memory:",
    model_name="all-MiniLM-L6-v2",
)

def docs():
    for i in range(10000):
        yield f"Document {i}"

stats = engine.ingest_streaming(
    texts_iterable=docs(),
    batch_size=100,
    start_id=0,
)
print(stats)
# {'total_docs': ..., 'total_batches': ..., 'start_id': ..., 'end_id': ...}
```

### Relevant config

- `ChelationConfig.STREAMING_BATCH_SIZE`
- `ChelationConfig.STREAMING_PROGRESS_INTERVAL`

---

## 2) Chelation log capping

`_spectral_chelation_ranking(...)` now keeps only the most recent N chelation centers per document.

### Relevant config

- `ChelationConfig.CHELATION_LOG_MAX_ENTRIES_PER_DOC` (default: `1000`)

This bounds in-memory growth for long-running query sessions.

---

## 3) Adaptive threshold tuning (opt-in)

Adaptive tuning is disabled by default. Enable it explicitly:

```python
engine.enable_adaptive_threshold(
    percentile=75,
    window=100,
    min_samples=20,
    min_bound=0.0001,
    max_bound=0.01,
)

stats = engine.get_threshold_stats()
print(stats["enabled"], stats["current_threshold"])
```

Disable and reset:

```python
engine.disable_adaptive_threshold()
```

During inference, the engine records observed variance and updates `chelation_threshold` once `min_samples` is reached, clamped to `[min_bound, max_bound]`.

### Relevant config

- `ChelationConfig.ADAPTIVE_THRESHOLD_PERCENTILE`
- `ChelationConfig.ADAPTIVE_THRESHOLD_WINDOW`
- `ChelationConfig.ADAPTIVE_THRESHOLD_MIN_SAMPLES`
- `ChelationConfig.ADAPTIVE_THRESHOLD_MIN`
- `ChelationConfig.ADAPTIVE_THRESHOLD_MAX`

---

## 4) Multi-task benchmark runner

`benchmark_multitask.py` benchmarks multiple MTEB retrieval tasks and reports:

- Retrieval quality (`ndcg_10`)
- Stability (`avg_jaccard`)
- Learning gain (`post_ndcg - pre_ndcg`)

### Example commands

```bash
# Preset suite
python benchmark_multitask.py --tasks mini --epochs 2 --max-queries 50

# Custom task list
python benchmark_multitask.py --tasks SciFact,NFCorpus --epochs 3 --lr 0.001
```

---

## 5) Dashboard server

`dashboard_server.py` provides a stdlib HTTP server + JSON APIs over JSONL logs.

### Start server

```bash
python dashboard_server.py --host localhost --port 8080 --log-file chelation_events.jsonl
```

### Open UI

- `http://localhost:8080/dashboard/`

### APIs

- `GET /api/summary`
- `GET /api/events?limit=25`
- `GET /api/events?event_type=query`

---

## Validation evidence (this session)

### Targeted Phase 4 tests

```bash
python -m pytest test_memory_optimization.py -q
python -m pytest test_adaptive_threshold.py -q
python -m pytest test_benchmark_multitask.py -q
python -m pytest test_dashboard_server.py -q
```

Results:
- `test_memory_optimization.py`: 16 passed
- `test_adaptive_threshold.py`: 17 passed
- `test_benchmark_multitask.py`: 24 passed
- `test_dashboard_server.py`: 24 passed

### Full relevant suite

```bash
python -m pytest test_unit_core.py test_recursive_decomposer.py test_aep_orchestrator.py test_integration_rlm.py test_benchmark_rlm.py test_teacher_distillation.py test_benchmark_distillation.py test_memory_optimization.py test_adaptive_threshold.py test_benchmark_multitask.py test_dashboard_server.py -q
```

Result: **234 passed, 1 warning**.
