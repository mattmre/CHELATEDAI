# Architecture: Batch-Optimized Teacher Encoding
Date: 2026-02-26 | Session: 21 | Status: Approved

## Summary
Add configurable batch_size, eager loading, chunked encoding, and parallel multi-teacher encoding.

## Changes to teacher_distillation.py
- TeacherDistillationHelper: new params batch_size, eager_load, show_progress, max_corpus_chunk
- Chunked encoding: splits large text lists into max_corpus_chunk-sized chunks
- Per-chunk error handling (resilient to partial failures)
- EnsembleTeacherHelper: parallel encoding via ThreadPoolExecutor (matches Ollama pattern)
- Factory function updated to forward new params

## Config Additions (config.py)
- TEACHER_BATCH_SIZE=64, TEACHER_EAGER_LOAD=False, TEACHER_SHOW_PROGRESS=False
- TEACHER_MAX_CORPUS_CHUNK=10000, ENSEMBLE_PARALLEL_ENCODING=True, ENSEMBLE_MAX_WORKERS=4
- New preset category: teacher_encoding (default, large_corpus, memory_constrained, gpu_optimized)

## Backward Compatibility
All new params have defaults matching current behavior. Existing callers unaffected.

## Test Plan: 24 new tests (45 -> 69 total in test_teacher_distillation.py)
- Batch size forwarding (11), parallel encoding (6), factory/config (7)
