# Architecture: Multi-Dataset BEIR Evaluation Extension
Date: 2026-02-26 | Session: 21 | Status: Approved

## Summary
Extends benchmark evaluation to support multiple BEIR datasets (SciFact, NFCorpus, FiQA2018, TRECCOVID, NQ, HotpotQA).

## Key Components
- **BEIRDatasetRegistry**: Central registry with metadata and tier-based grouping (quick/small/medium/research/full)
- **DatasetLoader**: Wraps load_mteb_data() with corpus sampling for large datasets
- **BEIRBenchmarkRunner**: Orchestrates ComparativeTestbed across multiple datasets
- **BEIRBenchmarkReport**: Full cross-product results (configs x datasets) with aggregation

## New File: benchmark_beir.py
- Composes existing ComparativeTestbed (no modifications to it)
- Dataset tiers: quick (SciFact), small (+NFCorpus), medium (+FiQA), research (+TRECCOVID), full (+NQ, HotpotQA)
- Corpus sampling preserves qrels integrity for large datasets
- JSON output compatible with dashboard

## Dashboard Integration
- New endpoint: GET /api/beir_results
- New tab: "BEIR Benchmarks" with heatmap table + bar chart

## CLI
python benchmark_beir.py --tier medium --output benchmark_beir_results.json

## Test Plan: ~30 tests in test_benchmark_beir.py
- TestBEIRDatasetRegistry (5), TestDatasetLoader (6), TestMultiDatasetResult (3)
- TestBEIRBenchmarkRunner (10), TestBEIRBenchmarkReport (4), TestCorpusSampling (5)
