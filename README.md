# ChelatedAI - Adaptive Vector Search with Self-Correcting Embeddings

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-passing-green.svg)](test_unit_core.py)

**ChelatedAI** is a research prototype implementing adaptive vector search with automatic embedding correction. It addresses semantic collapse in RAG systems through dynamic dimension masking ("chelation") and neural adaptation.

## Key Features

- **Adaptive Retrieval**: Automatically switches between fast and precision modes based on query entropy
- **Self-Correcting Embeddings**: Learns to fix embeddings that repeatedly collapse into semantic noise
- **Spectral Chelation**: Reranks results by shifting to cluster center-of-mass reference frame
- **Production-Ready**: Error recovery, checkpointing, structured logging, and cross-platform support

## Quick Start

### Prerequisites

```bash
# Python dependencies
pip install numpy torch sentence-transformers qdrant-client mteb requests

# Optional: Ollama for Docker-based embeddings
docker run -d -p 11434:11434 ollama/ollama
docker exec ollama ollama pull nomic-embed-text
```

### Basic Usage

```python
from antigravity_engine import AntigravityEngine

# Initialize engine
engine = AntigravityEngine(
    qdrant_location="./my_database",
    model_name="ollama:nomic-embed-text",  # or any SentenceTransformer model
    chelation_p=85,
    use_quantization=True
)

# Ingest documents
documents = ["Document 1 text...", "Document 2 text..."]
engine.ingest(documents)

# Query with adaptive retrieval
std_ids, chel_ids, mask, jaccard = engine.run_inference("What is machine learning?")
print(f"Top Results: {chel_ids}")

# Optional: Train adapter on accumulated patterns
engine.run_sedimentation_cycle(threshold=3, learning_rate=0.01, epochs=10)
```

### Phase 4 Features (Memory-Efficient)

```python
# Streaming ingestion for large datasets (avoids loading all into memory)
def document_generator():
    for i in range(100000):
        yield f"Document {i} content..."

stats = engine.ingest_streaming(document_generator(), batch_size=100)
print(f"Ingested {stats['total_docs']} documents")

# Enable adaptive threshold tuning (auto-adjusts based on query patterns)
engine.enable_adaptive_threshold(percentile=75, window=100)

# Multi-task benchmarking
# python benchmark_multitask.py --tasks mini --epochs 2 --max-queries 50

# Dashboard visualization
# python dashboard_server.py --port 8080
```

### Run Tests

```bash
# Unit tests (Phase 1-3)
python test_unit_core.py

# Phase 4 tests
python -m pytest test_adaptive_threshold.py test_memory_optimization.py \
    test_benchmark_multitask.py test_dashboard_server.py -v

# Integration tests
python test_dynamic_adaptation.py
python test_longitudinal_adaptation.py

# Benchmarks (requires MTEB)
python benchmark_evolution.py --task SciFact --lr 0.5
python benchmark_multitask.py --tasks mini --epochs 2 --max-queries 50
```

## Architecture

### Core Components

#### 1. AntigravityEngine (antigravity_engine.py)
Main retrieval system with:
- Dual-mode embedding (Ollama HTTP API or local SentenceTransformers)
- Variance-based adaptive path selection
- Spectral chelation reranking
- Neural adapter for persistent corrections

#### 2. ChelationAdapter (chelation_adapter.py)
Lightweight residual neural network that learns to correct embeddings:
- Identity initialization (preserves base model quality)
- Trained on collapse events via MSE loss
- Outputs L2-normalized vectors for cosine similarity

#### 3. Configuration System (config.py)
Centralized hyperparameter management:
- Cross-platform path handling
- Validation and clamping
- Presets for different use cases

#### 4. Checkpoint Manager (checkpoint_manager.py)
Safe training with automatic rollback:
- SHA256 integrity verification
- Automatic checkpoint cleanup
- Context manager for safe operations

#### 5. Structured Logger (chelation_logger.py)
JSON-formatted logging with:
- Performance metrics
- Query analysis
- Training progress
- Error tracking

## Configuration

### Hyperparameters

| Parameter | Default | Range | Impact |
|-----------|---------|-------|--------|
| `chelation_p` | 85 | 0-100 | Higher = fewer dimensions masked |
| `chelation_threshold` | 0.0004 | 0.0001-0.001 | Variance cutoff for triggering chelation |
| `learning_rate` | 0.001-0.5 | 0.0001-1.0 | Adapter training speed |
| `epochs` | 10 | 1-100 | Adapter training iterations |
| `scout_k` | 50 | 10-100 | Neighborhood size for variance calc |

### Presets

```python
from config import ChelationConfig

# Conservative (high-quality embeddings)
config = ChelationConfig.get_preset("conservative", "chelation")

# Balanced (general purpose)
config = ChelationConfig.get_preset("balanced", "chelation")

# Aggressive (noisy embeddings)
config = ChelationConfig.get_preset("aggressive", "chelation")
```

## Performance

### Benchmarks (SciFact Dataset)

| Configuration | NDCG@10 | Delta |
|---------------|---------|-------|
| Baseline (no chelation) | 0.5090 | - |
| Spectral Chelation | 0.5315 | +4.4% |
| Adaptive Brain (P=99.5) | 0.5360 | +5.3% |

### Resource Usage

- **Memory**: ~500MB base + O(batch_size × vector_dim)
- **CPU**: Efficient with quantization enabled
- **GPU**: Optional for local SentenceTransformers
- **Disk**: Persistent Qdrant databases (~100MB per 1K docs)

## Advanced Usage

### With Checkpointing

```python
from checkpoint_manager import CheckpointManager, SafeTrainingContext
from pathlib import Path

checkpoint_mgr = CheckpointManager(Path("./checkpoints"))

# Safe training with automatic rollback on failure
with SafeTrainingContext(checkpoint_mgr, Path("adapter_weights.pt"), "experiment_1") as ctx:
    engine.run_sedimentation_cycle(threshold=1, learning_rate=0.1, epochs=20)
    ctx.mark_success()  # Prevent rollback
```

### With Structured Logging

```python
from chelation_logger import get_logger

logger = get_logger(Path("debug.jsonl"), console_level="INFO")

# All engine operations are automatically logged
# Or log custom events:
logger.log_event("custom", "My custom event", metric=42)
```

### Configuration Files

```python
from config import ChelationConfig

# Save configuration
config = {
    "chelation_p": 90,
    "learning_rate": 0.01,
    "epochs": 15
}
ChelationConfig.save_to_file(config, Path("my_config.json"))

# Load configuration
loaded = ChelationConfig.load_from_file(Path("my_config.json"))
```

## File Structure

```
CHELATEDAI/
├── antigravity_engine.py        # Main retrieval engine
├── chelation_adapter.py          # Neural adapter module
├── homeostatic_engine.py         # Prototype (direct vector updates)
├── config.py                     # Configuration management
├── checkpoint_manager.py         # Checkpoint/recovery system
├── chelation_logger.py           # Structured logging
├── benchmark_evolution.py        # MTEB benchmarking
├── test_unit_core.py            # Unit tests
├── test_dynamic_adaptation.py   # Adapter validation
├── test_longitudinal_adaptation.py # Homeostatic validation
├── TECHNICAL_ANALYSIS.md        # Detailed architecture docs
├── REFACTORING_PLAN.md          # Development roadmap
└── README.md                    # This file
```

## Development

### Phase 1: Stabilization (Completed)
- ✅ Fixed critical bugs (duplicate returns, error handling)
- ✅ Cross-platform path support
- ✅ Standardized ID management
- ✅ Timeout protection for Ollama requests

### Phase 2: Robustness (Completed)
- ✅ Configuration management system
- ✅ Checkpoint/rollback functionality
- ✅ Error recovery with detailed logging

### Phase 3: Observability (Completed)
- ✅ Structured JSON logging
- ✅ Performance metrics tracking
- ✅ Comprehensive unit tests (21 tests passing)

### Phase 4: Memory Optimization & Adaptive Controls (Completed)
- ✅ Streaming ingestion for large datasets (`ingest_streaming()`)
- ✅ Chelation log capping (automatic memory management)
- ✅ Adaptive threshold tuning (runtime optimization)
- ✅ Multi-task benchmarking framework (`benchmark_multitask.py`)
- ✅ Web dashboard for log visualization (`dashboard_server.py`)
- ✅ 234 tests passing, 1 warning (expected)

See `docs/phase4-experiment-protocol.md` for detailed usage instructions.

## Troubleshooting

### Ollama Connection Fails

```bash
# Check if Ollama is running
curl http://localhost:11434/api/version

# Pull required model
docker exec ollama ollama pull nomic-embed-text

# Check container logs
docker logs -f ollama
```

### Adapter Dimension Mismatch

```python
# If you switch models, delete old adapter
import os
os.remove("adapter_weights.pt")

# Or specify different adapter path
engine = AntigravityEngine(..., adapter_path="new_adapter.pt")
```

### Out of Memory

```python
# Reduce batch size
from config import ChelationConfig
ChelationConfig.BATCH_SIZE = 50  # Default is 100

# Or use quantization
engine = AntigravityEngine(..., use_quantization=True)
```

## Contributing

This is a research prototype. Contributions welcome:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass: `python test_unit_core.py`
5. Submit a pull request

## Research Background

### The "Semantic Collapse" Problem

Standard embedding models can encode unrelated concepts in similar high-dimensional spaces due to:
- **Dimensional Noise**: Some dimensions capture dataset artifacts rather than meaning
- **Topic Collapse**: Frequent co-occurrence creates spurious similarities
- **Contextual Drift**: Same words in different contexts create confusion

### Our Approach

1. **Detect**: Measure per-dimension variance in local neighborhoods
2. **Chelate**: Mask high-variance "toxic" dimensions
3. **Adapt**: Train lightweight adapter to permanently fix problematic embeddings
4. **Verify**: Spectral reranking provides additional precision path

## License

MIT License - See LICENSE file for details

## Citation

If you use this code in research, please cite:

```bibtex
@software{chelatedai2024,
  title={ChelatedAI: Adaptive Vector Search with Self-Correcting Embeddings},
  author={ChelatedAI Contributors},
  year={2024},
  url={https://github.com/mattmre/CHELATEDAI}
}
```

## Contact & Support

- **Issues**: GitHub Issues tracker
- **Documentation**: See TECHNICAL_ANALYSIS.md for architecture details
- **Performance**: See manual_results.txt for benchmark data

---

**Warning**: This is a research prototype. While production-hardened with error recovery and testing, it should be thoroughly validated before production deployment.
