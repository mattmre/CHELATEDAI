# Research F-006: Config Mapping for antigravity_engine.py

**Date:** 2026-02-13 | **Status:** Complete

## Summary
- 27 hardcoded values in engine, zero imports from config.py
- 1 behavioral change: chelation_p default 80->85
- 3 config gaps need new constants
- All 67 tests safe (no test depends on engine defaults)

## Mapping Table

| Line | Hardcoded | Config Constant | Match? |
|------|-----------|----------------|--------|
| L12 | chelation_p=80 | DEFAULT_CHELATION_P=85 | **NO** |
| L23 | "chelation_events.jsonl" | EVENT_LOG_PATH | Yes (path type change) |
| L25 | 0.0004 | DEFAULT_CHELATION_THRESHOLD | Yes |
| L26 | "adapter_weights.pt" | ADAPTER_WEIGHTS_PATH | Yes (path type change) |
| L33 | ollama_url | OLLAMA_URL | Yes |
| L36 | 768 | DEFAULT_VECTOR_SIZE | Yes |
| L80 | "antigravity_stage8" | **GAP: needs COLLECTION_NAME** | N/A |
| L125 | 4096 (num_ctx) | **GAP: needs OLLAMA_NUM_CTX** | N/A |
| L127 | 30 (timeout) | OLLAMA_TIMEOUT | Yes |
| L149,155,159 | 6000,2000,500 | OLLAMA_TRUNCATION_LIMITS | Yes |
| L169 | 2 (max_workers) | OLLAMA_MAX_WORKERS | Yes |
| L198 | 100 (batch_size) | BATCH_SIZE | Yes |
| L222,270,524 | 50 (scout limit) | SCOUT_K | Yes |
| L375 | 100 (chunk_size) | CHUNK_SIZE | Yes |
| L399 | 0.1 (push magnitude) | **GAP: needs HOMEOSTATIC_PUSH_MAGNITUDE** | N/A |
| L347 | threshold=3 | DEFAULT_COLLAPSE_THRESHOLD | Yes |
| L347 | lr=0.001 | DEFAULT_LEARNING_RATE | Yes |
| L347 | epochs=10 | DEFAULT_EPOCHS | Yes |

## New Constants Needed in config.py
- COLLECTION_NAME = "antigravity_stage8"
- OLLAMA_NUM_CTX = 4096
- HOMEOSTATIC_PUSH_MAGNITUDE = 0.1

## Additional Notes (Session 2 research, 2026-02-13)

### Duplicate References
- `OLLAMA_TIMEOUT` (30) appears at L127 + L173
- `SCOUT_K` (50) appears 4 times: L222, L271, L524 + default arg
- `TOP_K` (10) appears 3 times: L573, L576, L577

### Path Type Handling
- `EVENT_LOG_PATH` and `ADAPTER_WEIGHTS_PATH` are `Path` objects in config
- Engine uses string paths -- convert with `str()` when assigning

### Truncation Refactor
- Lines 149, 155, 159 should iterate over `OLLAMA_TRUNCATION_LIMITS` list
- Replace three hardcoded retry blocks with a loop
