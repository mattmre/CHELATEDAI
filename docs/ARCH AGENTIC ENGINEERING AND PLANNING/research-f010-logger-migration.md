# Research F-010: Logger Migration for antigravity_engine.py

**Date:** 2026-02-13 | **Status:** Complete

## Summary
- 27 print() calls + 1 custom `_log_event()` method to replace
- All map to existing ChelationLogger methods (no new logger methods needed)
- Delete `_log_event()` entirely, replace with `log_query()` at call site
- Remove `self.event_log_path` attribute

## Print Statement Catalog

### Initialization (12 calls -> log_event / log_checkpoint)

| Line | Current | Logger Method | Level |
|------|---------|--------------|-------|
| 34 | Ollama Mode init | `log_event("initialization")` | INFO |
| 42 | Connected to Ollama | `log_event("initialization")` | INFO |
| 48 | WARNING: Ollama connection | `log_error("connection")` | WARNING |
| 49 | Vector size fallback | `log_event("initialization")` | WARNING |
| 54 | Local Mode init | `log_event("initialization")` | INFO |
| 59 | Device Selected | `log_event("initialization")` | INFO |
| 64 | Model loaded | `log_event("initialization")` | INFO |
| 67 | Adapter init | `log_event("adapter_init")` | INFO |
| 70 | Loaded adapter weights | `log_checkpoint("load")` | INFO |
| 72 | Created new adapter | `log_event("adapter_init")` | INFO |
| 86 | Quantization enabled | `log_event("initialization")` | INFO |
| 97 | Loaded collection | `log_event("collection_init")` | INFO |

### Embedding Errors (7 calls -> log_error)

| Line | Current | error_type | Level |
|------|---------|-----------|-------|
| 136 | Ollama timeout | "timeout" | WARNING |
| 139 | Connection lost | "connection" | ERROR |
| 142 | Missing embedding key | "api_response" | ERROR |
| 145 | Unexpected error | "embedding" | ERROR |
| 162 | Failed after retries | "embedding_failed" | ERROR |
| 176 | Embedding timeout | "timeout" | WARNING |
| 179 | Embedding failed | "embedding" | ERROR |

### Ingestion (3 calls -> log_event)

| Line | Current | event_type | Level |
|------|---------|-----------|-------|
| 196 | Ingesting N documents | "ingestion_start" | INFO |
| 218 | Batch progress | "ingestion_progress" | DEBUG |
| 220 | Ingestion complete | "ingestion_complete" | INFO |

### Sedimentation (10 calls -> training methods)

| Line | Current | Logger Method | Level |
|------|---------|--------------|-------|
| 357 | Running sedimentation | `log_training_start()` | INFO |
| 361 | Found N candidates | `log_event("training_preparation")` | INFO |
| 364 | Brain is stable | `log_event("training_skipped")` | INFO |
| 380 | Fetching data | `log_event("training_preparation")` | DEBUG |
| 413 | Training adapter | fold into training_start | - |
| 426 | Epoch progress | `log_training_epoch()` | DEBUG |
| 430 | Adapter saved | `log_checkpoint("save")` | INFO |
| 433 | Syncing vectors | `log_event("vector_update_start")` | INFO |
| 473/476 | Batch errors | `log_error()` | ERROR |
| 480/482 | Sedimentation complete | `log_training_complete()` | INFO |

### Custom _log_event() (lines 486-515)
- Replace the single call at L570 with `self.logger.log_query()`
- Delete entire method
- Remove `self.event_log_path` from `__init__` (L23)

## Implementation Steps
1. Add `from chelation_logger import get_logger` at top
2. Add `self.logger = get_logger()` in `__init__`
3. Replace all 27 print() calls per tables above
4. Replace `_log_event()` call with `log_query()`
5. Delete `_log_event()` method
6. Remove `self.event_log_path` from `__init__`
