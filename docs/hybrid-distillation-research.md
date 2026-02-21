# Hybrid Teacher Distillation for ChelatedAI

## Research Context

This document compares the ChelatedAI approach to the recently published SDFT paper (arXiv 2601.19897) and documents the rationale for ChelatedAI's hybrid training modes.

## Background: Retrieval-Oriented Distillation

### SDFT Approach (arXiv 2601.19897)
The Sequential Distillation and Fine-Tuning (SDFT) paper proposes:
1. **Pre-distillation**: Train student model on teacher embeddings (offline)
2. **Task fine-tuning**: Adapt to retrieval task with contrastive loss
3. **Sequential pipeline**: Two-stage training with distinct objectives

### ChelatedAI Approach
ChelatedAI implements **adaptive homeostatic correction** with three training modes:
1. **Baseline**: Query-driven homeostatic push (no teacher)
2. **Offline**: Pure teacher distillation (pre-training phase)
3. **Hybrid**: Real-time blend of homeostatic + teacher guidance

## Key Differences

### 1. Training Trigger
- **SDFT**: Requires labeled query-document pairs and contrastive loss
- **ChelatedAI**: Self-supervised via semantic collapse detection (chelation log)

### 2. Adaptation Mechanism
- **SDFT**: Global model fine-tuning
- **ChelatedAI**: Lightweight adapter network (preserves base model)

### 3. Target Generation
- **SDFT**: Hard teacher embeddings or contrastive labels
- **ChelatedAI Baseline**: Homeostatic push away from noise centroids
- **ChelatedAI Offline**: Pure teacher embedding alignment
- **ChelatedAI Hybrid**: Weighted blend: `target = (1-α)·homeostatic + α·teacher`

### 4. Deployment
- **SDFT**: Full model retraining required
- **ChelatedAI**: Hot-swappable adapter with checkpoint recovery

## Training Mode Rationale

### Baseline Mode: Pure Homeostatic
**When to use:**
- No external teacher model available
- Privacy-sensitive deployments (no external API calls)
- Domain-specific corpus where general teachers fail

**Mechanism:**
```python
# Homeostatic push formula
avg_noise = mean(collapse_vectors)
diff = current_vec - avg_noise
target = current_vec + normalize(diff) * push_magnitude
```

**Trade-offs:**
- ✅ Zero external dependencies
- ✅ Privacy-preserving
- ❌ May amplify existing biases
- ❌ No explicit semantic grounding

### Offline Mode: Pre-Training with Teacher
**When to use:**
- Cold-start scenarios (new corpus)
- Known-good teacher model available
- Pre-production warm-up phase

**Mechanism:**
```python
# Run once before serving traffic
engine.run_offline_distillation(
    batch_size=100,
    learning_rate=0.005,
    epochs=15
)
```

**Trade-offs:**
- ✅ Fast convergence to teacher quality
- ✅ Explicit semantic alignment
- ❌ Requires teacher model compatibility (same dimension)
- ❌ Upfront compute cost

### Hybrid Mode: Real-Time Blending
**When to use:**
- Production systems with query-time adaptation
- Domains where homeostatic + teacher complement each other
- Continuous learning scenarios

**Mechanism:**
```python
# Per-sedimentation cycle blending
target = (1 - teacher_weight) * homeostatic_target + teacher_weight * teacher_embedding
target = normalize(target)
```

**Configuration:**
- `teacher_weight=0.0`: Pure homeostatic (equivalent to baseline)
- `teacher_weight=0.5`: Balanced blend (default)
- `teacher_weight=1.0`: Pure teacher (equivalent to offline)

**Trade-offs:**
- ✅ Best of both worlds (semantic grounding + adaptive correction)
- ✅ Configurable balance via `teacher_weight`
- ❌ Requires teacher model at runtime
- ❌ Slightly higher inference cost

## Key Research Questions

### 1. Teacher Weight Tuning
- **Question**: What is the optimal `teacher_weight` for different domains?
- **Hypothesis**: Information-sparse domains (SciFact) benefit from higher teacher weight (0.7-0.9), while information-rich domains (Wikipedia) prefer lower weights (0.3-0.5)
- **Experiment**: Sweep `teacher_weight` from 0.0 to 1.0 in 0.1 increments across MTEB tasks

### 2. Dimension Compatibility
- **Limitation**: Current implementation requires teacher_dim == student_dim
- **Future Work**: Implement projection layer for mismatched dimensions
  ```python
  if teacher_dim != student_dim:
      self.projection = nn.Linear(teacher_dim, student_dim)
  ```

### 3. Teacher Model Selection
- **Question**: Which teacher models work best for ChelatedAI?
- **Candidates**: 
  - `all-MiniLM-L6-v2` (384d, fast, general-purpose)
  - `all-mpnet-base-v2` (768d, higher quality, slower)
  - Domain-specific teachers (BiomedBERT for SciFact, etc.)
- **Evaluation**: Compare NDCG@10 degradation over multiple query-sediment cycles

### 4. Collapse Detection vs Teacher Guidance
- **Question**: Does teacher guidance reduce the frequency of detected collapses?
- **Metric**: Track `chelation_log` size over cycles for baseline vs hybrid
- **Expected**: Hybrid mode should exhibit fewer collapses (teacher prevents drift)

### 5. Computational Trade-offs
- **Baseline**: No extra cost (homeostatic only)
- **Offline**: One-time upfront cost (amortized over queries)
- **Hybrid**: Per-sedimentation teacher encoding cost
- **Benchmark**: Measure wall-clock time per mode in `benchmark_distillation.py`

## Known Risks and Mitigations

### Risk 1: Teacher Model Bias Amplification
**Risk**: Teacher embeddings may encode societal biases or factual errors.
**Mitigation**: 
- Blend with homeostatic (hybrid mode reduces over-reliance)
- Monitor alignment metrics (`compute_alignment_metric`)
- Fallback to baseline if teacher embeddings degrade NDCG

### Risk 2: Dimension Mismatch
**Risk**: Teacher and student models may have different embedding dimensions.
**Current State**: Hard failure with error log.
**Mitigation**: 
- Pre-flight dimension check in engine initialization
- Graceful fallback to baseline mode
- Future: Learnable projection layer

### Risk 3: Overfitting to Teacher
**Risk**: Offline mode may overfit to teacher, losing domain-specific signal.
**Mitigation**:
- Limit offline epochs (default: 15)
- Use hybrid mode for production (preserves homeostatic correction)
- Monitor retrieval metrics on held-out queries

### Risk 4: Teacher Availability at Runtime
**Risk**: Hybrid mode requires teacher model loaded in memory.
**Mitigation**:
- Lazy loading (only load when needed)
- Shared teacher instance across engine instances
- Fallback to baseline if teacher loading fails

## Experimental Protocol

See [distillation-experiment-protocol.md](./distillation-experiment-protocol.md) for reproducible benchmarking instructions.

## Implementation Notes

### Teacher Loading Strategy
```python
# Lazy loading in TeacherDistillationHelper
def load_teacher_model(self):
    if self.teacher_model is not None:
        return  # Already loaded
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    self.teacher_model = SentenceTransformer(
        self.teacher_model_name,
        device=device,
        trust_remote_code=True
    )
```

### Fallback Chain
1. Try hybrid/offline mode with teacher
2. If teacher fails → log error and fallback to baseline
3. If baseline fails → return current embeddings unchanged

### Configuration Integration
All distillation parameters are centralized in `config.py`:
```python
DEFAULT_TRAINING_MODE = "baseline"
DEFAULT_TEACHER_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_TEACHER_WEIGHT = 0.5
DEFAULT_OFFLINE_EPOCHS = 15
```

## Future Research Directions

1. **Multi-Teacher Ensemble**: Blend multiple teacher models with learned weights
2. **Dynamic Teacher Selection**: Switch teacher based on query domain detection
3. **Adversarial Distillation**: Add discriminator to distinguish teacher vs student embeddings
4. **Curriculum Learning**: Start with high teacher weight, anneal to homeostatic over time
5. **Cross-Lingual Distillation**: Use multilingual teacher to improve non-English retrieval

## References

- SDFT Paper: arXiv 2601.19897 (Sequential Distillation and Fine-Tuning)
- ChelatedAI Core: [antigravity_engine.py](../antigravity_engine.py)
- Teacher Module: [teacher_distillation.py](../teacher_distillation.py)
- Benchmark: [benchmark_distillation.py](../benchmark_distillation.py)

## Change Log

- 2026-02-16: Initial hybrid distillation implementation
- 2026-02-16: Added dimension compatibility checks
- 2026-02-16: Implemented fallback chain for robust production use
