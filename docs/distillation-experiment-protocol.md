# Distillation Experiment Protocol

## Overview

This document provides reproducible commands and analysis guidelines for benchmarking ChelatedAI's three training modes: **baseline**, **offline**, and **hybrid**.

## Prerequisites

### Environment Setup
```bash
# Install dependencies
pip install numpy torch sentence-transformers qdrant-client mteb requests

# Verify installation
python -c "import torch; import sentence_transformers; import mteb; print('✓ All dependencies installed')"
```

### Optional: GPU Acceleration
```bash
# Check GPU availability
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')"
```

## Quick Test: Unit Tests

### Test Teacher Distillation Module
```bash
# Run unit tests (fast, no downloads)
python -m pytest test_teacher_distillation.py -v

# Expected output: 19 tests passed in < 5 seconds
```

### Test Benchmark Utilities
```bash
# Run benchmark utility tests (fast, mocked)
python -m pytest test_benchmark_distillation.py -v

# Expected output: 22 tests passed in < 3 seconds
```

## Full Benchmark: Comparative Training Modes

### Command Template
```bash
python benchmark_distillation.py \
    --task <MTEB_TASK> \
    --model <LOCAL_MODEL> \
    --teacher <TEACHER_MODEL> \
    --cycles <NUM_CYCLES> \
    --queries-per-cycle <NUM_QUERIES> \
    --epochs <EPOCHS> \
    --lr <LEARNING_RATE> \
    --teacher-weight <WEIGHT> \
    --output <OUTPUT_JSON>
```

### Example 1: Quick Test (SciFact, 3 Cycles)
```bash
python benchmark_distillation.py \
    --task SciFact \
    --model sentence-transformers/all-MiniLM-L6-v2 \
    --teacher sentence-transformers/all-MiniLM-L6-v2 \
    --cycles 3 \
    --queries-per-cycle 50 \
    --epochs 10 \
    --lr 0.001 \
    --teacher-weight 0.5 \
    --output results_scifact_quick.json
```

**Expected Runtime**: ~10-15 minutes on CPU, ~3-5 minutes on GPU

**Expected Output Structure**:
```json
{
  "baseline": [
    {"cycle": 1, "ndcg": 0.6523, "ndcg_std": 0.1234, ...},
    {"cycle": 2, "ndcg": 0.6587, "ndcg_std": 0.1198, ...},
    {"cycle": 3, "ndcg": 0.6612, "ndcg_std": 0.1156, ...}
  ],
  "offline": {
    "pretraining_time": 45.23,
    "cycles": [
      {"cycle": 1, "ndcg": 0.6801, "ndcg_std": 0.1089, ...},
      ...
    ]
  },
  "hybrid": [
    {"cycle": 1, "ndcg": 0.6734, "ndcg_std": 0.1145, ...},
    ...
  ]
}
```

### Example 2: Extended Run (10 Cycles, Production Settings)
```bash
python benchmark_distillation.py \
    --task SciFact \
    --model sentence-transformers/all-MiniLM-L6-v2 \
    --teacher sentence-transformers/all-mpnet-base-v2 \
    --cycles 10 \
    --queries-per-cycle 100 \
    --epochs 15 \
    --lr 0.0005 \
    --teacher-weight 0.7 \
    --output results_scifact_extended.json
```

**Expected Runtime**: ~1-2 hours on CPU, ~20-30 minutes on GPU

### Example 3: Teacher Weight Sweep
```bash
# Test different teacher weights for hybrid mode
for weight in 0.0 0.3 0.5 0.7 1.0; do
    python benchmark_distillation.py \
        --task SciFact \
        --model sentence-transformers/all-MiniLM-L6-v2 \
        --teacher sentence-transformers/all-MiniLM-L6-v2 \
        --cycles 5 \
        --queries-per-cycle 50 \
        --epochs 10 \
        --lr 0.001 \
        --teacher-weight $weight \
        --output results_sweep_weight_${weight}.json
done

# Aggregate results
python -c "
import json, glob
for f in glob.glob('results_sweep_weight_*.json'):
    with open(f) as fp:
        data = json.load(fp)
        final_ndcg = data['hybrid'][-1]['ndcg']
        weight = f.split('_')[-1].replace('.json', '')
        print(f'Weight {weight}: NDCG@10 = {final_ndcg:.4f}')
"
```

## Metrics Interpretation

### Primary Metric: NDCG@10
**Definition**: Normalized Discounted Cumulative Gain at rank 10.
- **Range**: [0.0, 1.0]
- **Interpretation**: 
  - 1.0 = Perfect ranking (all relevant docs in top-10, correctly ordered)
  - 0.0 = No relevant docs in top-10
  - >0.65 = Good retrieval quality on SciFact

**Calculation**:
```python
def ndcg_at_k(relevances, k=10):
    dcg = sum(rel / log2(i+2) for i, rel in enumerate(relevances[:k]))
    ideal_dcg = sum(rel / log2(i+2) for i, rel in enumerate(sorted(relevances, reverse=True)[:k]))
    return dcg / ideal_dcg if ideal_dcg > 0 else 0.0
```

### Secondary Metrics

#### NDCG Standard Deviation
**Purpose**: Measures retrieval consistency across queries.
- Lower std → More consistent performance
- Higher std → Query-dependent quality

#### Training Time
- **baseline**: No extra time (homeostatic only)
- **offline**: Upfront pretraining time (logged separately)
- **hybrid**: Per-cycle teacher encoding time (included in sediment_time)

#### Query Time
**Purpose**: Measure inference overhead.
- Should be stable across modes (embedding is shared)
- Variations indicate logging overhead or memory pressure

#### Sedimentation Time
**Purpose**: Measure adapter training cost per cycle.
- **baseline**: Training only
- **offline**: Training + one-time corpus distillation
- **hybrid**: Training + per-batch teacher encoding

## Analysis Guidelines

### Comparing Modes

#### 1. Final Performance (Last Cycle)
```python
import json

with open('results.json') as f:
    data = json.load(f)

baseline_final = data['baseline'][-1]['ndcg']
offline_final = data['offline']['cycles'][-1]['ndcg']
hybrid_final = data['hybrid'][-1]['ndcg']

print(f"Baseline: {baseline_final:.4f}")
print(f"Offline:  {offline_final:.4f} ({(offline_final - baseline_final)*100:+.2f}%)")
print(f"Hybrid:   {hybrid_final:.4f} ({(hybrid_final - baseline_final)*100:+.2f}%)")
```

**Expected Observations**:
- Offline typically has highest final NDCG (explicit teacher alignment)
- Hybrid balances teacher guidance with adaptive correction
- Baseline may lag initially but can catch up in domain-specific scenarios

#### 2. Learning Trajectory
```python
import matplotlib.pyplot as plt

cycles = [r['cycle'] for r in data['baseline']]
baseline_ndcg = [r['ndcg'] for r in data['baseline']]
offline_ndcg = [r['ndcg'] for r in data['offline']['cycles']]
hybrid_ndcg = [r['ndcg'] for r in data['hybrid']]

plt.plot(cycles, baseline_ndcg, label='Baseline', marker='o')
plt.plot(cycles, offline_ndcg, label='Offline', marker='s')
plt.plot(cycles, hybrid_ndcg, label='Hybrid', marker='^')
plt.xlabel('Cycle')
plt.ylabel('NDCG@10')
plt.title('Training Mode Comparison')
plt.legend()
plt.grid(True)
plt.savefig('training_comparison.png')
```

**Expected Pattern**:
- Offline starts high (pre-training boost)
- Hybrid converges faster than baseline (teacher guidance)
- Baseline may have steeper learning curve (pure adaptation)

#### 3. Consistency Analysis
```python
import numpy as np

baseline_stds = [r['ndcg_std'] for r in data['baseline']]
hybrid_stds = [r['ndcg_std'] for r in data['hybrid']]

print(f"Baseline Avg Std: {np.mean(baseline_stds):.4f}")
print(f"Hybrid Avg Std:   {np.mean(hybrid_stds):.4f}")
```

**Interpretation**:
- Lower std → More robust to query variations
- Teacher guidance typically reduces std (semantic grounding)

#### 4. Computational Cost
```python
# Total time per mode (excluding evaluation)
baseline_time = sum(r['query_time'] + r['sediment_time'] for r in data['baseline'])
offline_time = data['offline']['pretraining_time'] + sum(r['query_time'] + r['sediment_time'] for r in data['offline']['cycles'])
hybrid_time = sum(r['query_time'] + r['sediment_time'] for r in data['hybrid'])

print(f"Baseline Total: {baseline_time:.1f}s")
print(f"Offline Total:  {offline_time:.1f}s (includes {data['offline']['pretraining_time']:.1f}s pretraining)")
print(f"Hybrid Total:   {hybrid_time:.1f}s")
```

**Trade-off Analysis**:
- Cost per NDCG point: `total_time / final_ndcg`
- Amortized offline cost: If serving >1000 queries, upfront cost becomes negligible

## Debugging Failed Runs

### Issue 1: Dimension Mismatch
**Symptom**: Error message about teacher/student dimension incompatibility.
```
ERROR dimension_mismatch_targets: Cannot blend: teacher dim 384 != student dim 768
```

**Solution**: Use same-dimension models.
```bash
# Both 384-dim
--model sentence-transformers/all-MiniLM-L6-v2 \
--teacher sentence-transformers/all-MiniLM-L6-v2

# Both 768-dim
--model sentence-transformers/all-mpnet-base-v2 \
--teacher sentence-transformers/all-mpnet-base-v2
```

### Issue 2: MTEB Task Not Found
**Symptom**: `ERROR: Task 'TaskName' not found in MTEB registry!`

**Solution**: Check available tasks.
```python
import mteb
tasks = mteb.get_tasks(task_types=["Retrieval"])
print([t.description["name"] for t in tasks])
```

### Issue 3: Out of Memory
**Symptom**: `RuntimeError: CUDA out of memory`

**Solution**: Reduce batch size or use CPU.
```bash
# Smaller corpus batches (in benchmark_distillation.py, modify batch_size)
# Or force CPU
CUDA_VISIBLE_DEVICES="" python benchmark_distillation.py ...
```

### Issue 4: Teacher Model Download Failure
**Symptom**: `OSError: Can't load model from 'sentence-transformers/...'`

**Solution**: Pre-download model.
```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
print("Model cached successfully")
```

## Validation Checklist

Before considering results valid, verify:
- [ ] All three modes completed without errors
- [ ] NDCG values are in reasonable range (0.4-0.9 for SciFact)
- [ ] Offline pretraining time is non-zero
- [ ] Hybrid mode shows blending effects (between baseline and offline)
- [ ] JSON output file is well-formed and contains expected keys
- [ ] Cycle count matches `--cycles` argument
- [ ] Teacher weight matches `--teacher-weight` argument (check logs)

## Reproducibility Notes

### Random Seeds
The benchmark currently **does not** set random seeds. For reproducibility:
```python
# Add to top of benchmark_distillation.py
import random, numpy as np, torch
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
```

### MTEB Version
Lock MTEB version for reproducibility:
```bash
pip install mteb==1.1.1  # Or current stable version
```

### Model Caching
Models are cached in `~/.cache/huggingface/`. To ensure identical weights:
```bash
# Clear cache and re-download
rm -rf ~/.cache/huggingface/
python benchmark_distillation.py ...
```

## Advanced: Custom Teacher Models

### Using Domain-Specific Teacher
```bash
# Example: BiomedBERT for SciFact
python benchmark_distillation.py \
    --task SciFact \
    --model sentence-transformers/all-MiniLM-L6-v2 \
    --teacher dmis-lab/biobert-v1.1 \
    --cycles 5 \
    --epochs 10 \
    --output results_biobert_teacher.json
```

**Note**: Teacher must output same dimension as student, or modify code to add projection layer.

### Multi-Task Evaluation
```bash
# Run across multiple MTEB tasks
for task in SciFact NFCorpus TRECCOVID; do
    python benchmark_distillation.py \
        --task $task \
        --model sentence-transformers/all-MiniLM-L6-v2 \
        --teacher sentence-transformers/all-MiniLM-L6-v2 \
        --cycles 5 \
        --queries-per-cycle 50 \
        --epochs 10 \
        --output results_${task}.json
done
```

## Expected Outcomes (SciFact Baseline)

Based on initial testing:
- **Baseline**: NDCG@10 ≈ 0.62-0.66 (varies by initialization)
- **Offline**: NDCG@10 ≈ 0.66-0.70 (+4-6% over baseline)
- **Hybrid (weight=0.5)**: NDCG@10 ≈ 0.64-0.68 (+2-4% over baseline)

**Note**: These are approximate ranges. Actual results depend on model quality, query distribution, and hyperparameters.

## Contact and Support

For issues with benchmark execution:
1. Check logs in `chelation_events.jsonl`
2. Review error messages for fallback chains
3. Validate input parameters with `--help`
4. Consult [hybrid-distillation-research.md](./hybrid-distillation-research.md) for conceptual questions
