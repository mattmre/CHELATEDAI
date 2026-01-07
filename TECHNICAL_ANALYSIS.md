# ChelatedAI Technical Analysis & Architecture Guide

**Generated**: 2026-01-06
**Version**: 1.0

---

## System Overview

ChelatedAI is an adaptive vector search system that addresses semantic collapse in embedding-based retrieval through dynamic dimension masking ("chelation") and neural adaptation.

### Core Innovation
The system detects when embedding dimensions contribute more noise than signal (high variance in local neighborhoods) and adaptively masks or corrects them.

---

## Architecture Components

### 1. AntigravityEngine (antigravity_engine.py)
**Purpose**: Main retrieval engine with adaptive chelation

#### Key Methods

##### `__init__(qdrant_location, chelation_p, model_name, use_centering, use_quantization)`
Initializes the engine with dual-mode embedding support.

**Parameters**:
- `qdrant_location`: Path or ":memory:" for vector DB
- `chelation_p`: Percentile threshold (0-100) for dimension selection
- `model_name`: "ollama:model-name" or HuggingFace model ID
- `use_centering`: Enable spectral chelation reranking
- `use_quantization`: Enable INT8 quantization + adaptive logic

**Embedding Modes**:
1. **Ollama Mode**: HTTP API to Docker container (model_name starts with "ollama:")
2. **Local Mode**: Direct SentenceTransformer inference

##### `embed(texts) -> np.ndarray`
Generates embeddings via Ollama or local model, passing through the adapter.

**Flow**:
```
Text → Base Model → Adapter (if local) → L2 Normalized Vector
```

**Ollama Mode Retry Logic** (antigravity_engine.py:131-148):
- Try 6000 chars (~1500 tokens)
- Retry with 2000 chars (~500 tokens)
- Final attempt with 500 chars
- Return zero vector if all fail

##### `ingest(text_corpus, payloads)`
Batch ingestion with progress tracking (100 docs/batch).

##### `run_inference(query_text) -> (std_top_10, chel_top_10, mask, jaccard)`
Full retrieval pipeline with adaptive decision-making.

**Decision Tree**:
```
if use_quantization:
    if global_variance > threshold OR use_centering:
        ACTION = CHELATE (Spectral Reranking)
    else:
        ACTION = FAST (Trust Quantized Index)
elif use_centering:
    ACTION = CHELATE_ALWAYS
else:
    ACTION = STANDARD
```

**Returns**:
- `std_top_10`: Standard retrieval top 10 IDs
- `chel_top_10`: Chelated/reranked top 10 IDs
- `mask`: Dimension mask (legacy, mostly unused now)
- `jaccard`: Overlap between std and chel results

##### `run_sedimentation_cycle(threshold, learning_rate, epochs)`
Trains the adapter on accumulated collapse events.

**Training Pipeline**:
1. Filter docs with collapse frequency ≥ threshold
2. Fetch current vectors from Qdrant
3. Calculate target vectors (pushed away from noise centers)
4. Train adapter with MSE loss
5. Update Qdrant with corrected vectors
6. Save adapter weights

**Key Variables**:
- `chelation_log`: dict mapping doc_id → list of noise_centers
- `threshold`: Minimum collapse frequency to trigger update
- `learning_rate`: Adam optimizer LR (0.001-0.5 tested)
- `epochs`: Training iterations (10 default)

#### Private Methods

##### `_gravity_sensor(query_vec, top_k) -> np.ndarray`
Retrieves local neighborhood vectors (antigravity_engine.py:200).

##### `_chelate_toxicity(local_cluster) -> np.ndarray`
Calculates dimension mask based on variance (antigravity_engine.py:216).

**Algorithm**:
```python
dim_variance = var(local_cluster, axis=0)
threshold = percentile(dim_variance, chelation_p)
mask = (dim_variance < threshold).astype(float)  # Keep low-variance
```

##### `_spectral_chelation_ranking(query_vec, local_vectors, local_ids)`
Reranks by shifting to center-of-mass reference frame (antigravity_engine.py:287).

**Algorithm**:
```python
center_of_mass = mean(local_vectors)
centered_query = query - center_of_mass
centered_candidates = candidates - center_of_mass
scores = cosine_similarity(centered_query, centered_candidates)
```

**Side Effect**: Logs noise centers to `chelation_log` for learning.

---

### 2. ChelationAdapter (chelation_adapter.py)
**Purpose**: Lightweight neural adapter for embedding correction

#### Architecture
```
Input (D) → Linear(D → D/2) → ReLU → Linear(D/2 → D) → Residual → Normalize
```

**Design Principles**:
1. **Residual Connection**: `output = normalize(input + correction(input))`
2. **Identity Initialization**: Small weights (std=0.001) prevent disruption
3. **Hypersphere Constraint**: L2 normalization maintains cosine compatibility

#### Methods

##### `forward(x) -> torch.Tensor`
Applies correction and normalization.

##### `save(path)`
Saves state_dict to disk.

##### `load(path) -> bool`
Loads weights with dimension mismatch handling (updated with try/except).

---

### 3. HomeostaticVectorStore (homeostatic_engine.py)
**Purpose**: Prototype demonstrating persistent vector updates

**Difference from AntigravityEngine**:
- Updates vectors in DB rather than training an adapter
- Simpler proof-of-concept for "sleep cycle" idea
- Used in test_longitudinal_adaptation.py

#### Update Rule
```python
V_new = V_old - (AvgNoiseVector * LearningRate)
```

This directly modifies stored vectors without a neural adapter.

---

## Data Flow Diagrams

### Query Processing (Adaptive Mode)
```
Query Text
    ↓
embed() → [Base Model] → [Adapter] → Query Vector
    ↓
Qdrant.query_points() → Top 50 Results
    ↓
Calculate global_variance = mean(var(results, axis=0))
    ↓
if variance > threshold:
    Spectral Chelation Reranking
    Log noise centers
else:
    Return standard results
    ↓
Top 10 Results + Metrics
```

### Training Cycle (Sleep Mode)
```
Accumulated chelation_log {doc_id: [noise_centers]}
    ↓
Filter: len(noise_centers) >= threshold
    ↓
Fetch current vectors from Qdrant
    ↓
For each vector:
    avg_noise = mean(noise_centers)
    target = current + normalize(current - avg_noise) * 0.1
    ↓
Train Adapter: MSE(adapter(current), target)
    ↓
Save adapter.pt
    ↓
Update vectors in Qdrant
    ↓
Clear chelation_log
```

---

## Configuration Parameters

### Hyperparameters

| Parameter | Default | Range | Impact |
|-----------|---------|-------|--------|
| `chelation_p` | 85 | 0-100 | Higher = fewer dims masked, more conservative |
| `chelation_threshold` | 0.0004 | 0.0001-0.001 | Variance cutoff for triggering chelation |
| `learning_rate` | 0.001-0.5 | 0.0001-1.0 | Adapter training speed |
| `epochs` | 10 | 1-100 | Adapter training iterations |
| `threshold` (freq) | 1-3 | 1-10 | Min collapse events to trigger update |
| `scout_k` | 50 | 10-100 | Neighborhood size for variance calc |

### Tuning Guidelines

**For High-Quality Embeddings** (e.g., OpenAI, Cohere):
- Higher `chelation_p` (95-99.5)
- Lower `chelation_threshold` (0.0001-0.0003)
- Conservative `learning_rate` (0.001-0.01)

**For Lower-Quality Embeddings** (e.g., small SentenceTransformers):
- Lower `chelation_p` (80-90)
- Higher `chelation_threshold` (0.0005-0.001)
- Moderate `learning_rate` (0.05-0.1)

**Dataset-Specific**:
- **Dense topics** (low variance): Increase threshold to avoid over-chelation
- **Sparse topics** (high variance): Decrease threshold to catch noise
- **Small datasets** (<1000 docs): Use higher frequency threshold (5-10)
- **Large datasets** (>10000 docs): Use lower frequency threshold (1-3)

---

## Performance Benchmarks

### SciFact (Manual Results)
| Configuration | NDCG@10 | Delta |
|---------------|---------|-------|
| Baseline (no chelation) | 0.5090 | - |
| P=80 | 0.4995 | -1.9% |
| P=90 | 0.5161 | +1.4% |
| P=99.5 (optimal) | 0.5360 | +5.3% |
| Adaptive Brain | 0.5323 | +4.6% |

### Event Log Statistics (10,671 queries)
- **FAST path**: ~70% of queries (low variance)
- **CHELATE path**: ~30% of queries (high variance)
- Average FAST variance: ~0.0002
- Average CHELATE variance: ~0.0008

---

## Known Issues & Limitations

### Critical Bugs (Being Fixed)
1. Duplicate return statement (antigravity_engine.py:160)
2. Bare except clauses suppress errors silently
3. Hardcoded Windows paths break cross-platform use
4. String/int ID confusion in MTEB integration

### Design Limitations
1. **Threshold Sensitivity**: Performance highly dependent on tuned thresholds
2. **Feedback Loops**: Adapter updates vectors, which affects future training
3. **No Convergence Guarantee**: Homeostatic updates could theoretically oscillate
4. **Memory Scaling**: Full batch loading limits dataset size
5. **Single Model Lock-in**: Adapter tied to specific embedding dimension

### Compatibility Notes
- Adapter weights are NOT transferable between models
- Database format changes if vector_size changes
- Chelation events assume stable document IDs

---

## Testing & Validation

### Unit Tests (To Be Created)
- `test_chelate_toxicity()`: Verify dimension masking logic
- `test_spectral_ranking()`: Validate reranking algorithm
- `test_adapter_identity()`: Confirm initialization near identity
- `test_id_mapping()`: Validate string/int conversions

### Integration Tests (Existing)
- `test_dynamic_adaptation.py`: Validates adapter learning
- `test_longitudinal_adaptation.py`: Validates homeostatic updates

### Benchmark Scripts
- `benchmark_evolution.py`: MTEB task evaluation with before/after
- `manual_benchmark_scifact.py`: Detailed SciFact analysis

---

## API Reference

### AntigravityEngine

```python
engine = AntigravityEngine(
    qdrant_location="./db",          # Path or ":memory:"
    chelation_p=85,                  # Dimension percentile threshold
    model_name="ollama:nomic-embed-text",  # Model identifier
    use_centering=False,             # Enable spectral chelation
    use_quantization=True            # Enable adaptive mode
)

# Ingest documents
engine.ingest(
    text_corpus=["doc1", "doc2"],
    payloads=[{"meta": "data"}, {"meta": "data2"}]
)

# Query with adaptive retrieval
std_ids, chel_ids, mask, jaccard = engine.run_inference("query text")

# Train on accumulated events
engine.run_sedimentation_cycle(
    threshold=3,        # Min collapse frequency
    learning_rate=0.01, # Adapter LR
    epochs=10          # Training iterations
)
```

### ChelationAdapter

```python
adapter = ChelationAdapter(input_dim=768, hidden_dim=384)
adapter.load("adapter_weights.pt")

# Forward pass
corrected = adapter(embeddings_tensor)

# Training
optimizer = torch.optim.Adam(adapter.parameters(), lr=0.001)
loss = torch.nn.MSELoss()(adapter(inputs), targets)
loss.backward()
optimizer.step()

adapter.save("adapter_weights.pt")
```

---

## Future Research Directions

1. **Adaptive Thresholds**: Learn `chelation_threshold` from data statistics
2. **Multi-Model Support**: Train adapter ensemble for different base models
3. **Continual Learning**: Online updates without full retraining
4. **Theoretical Analysis**: Prove convergence properties
5. **Benchmark Expansion**: Test on more MTEB tasks (FEVER, HotpotQA, etc.)
6. **Compression**: Quantize adapter weights for deployment
7. **Explainability**: Visualize which dimensions are toxic per query

---

## References

- **MTEB**: Massive Text Embedding Benchmark
- **Qdrant**: Vector database used for storage
- **SentenceTransformers**: Embedding model framework
- **Ollama**: Local LLM/embedding serving platform
