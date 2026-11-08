# Shim Benchmark Extension Specification
## Agent 4 Deliverable (Benchmark & Evaluation Extension) — Steering-Chelation-RAGDAG-MicroSLM Program

**Status**: Loop 1/2 design artifact (BHS-auditable spec)  
**Date**: 2026-05-26  
**Cross-references**:
- `docs/steering_chelation_rag_dag_research/shim_nodes_mtp_lookahead_nomenclature.md` (primary nomenclature: Shim Vector (SV), Shim Node (SN), Shim Insertion Point (SIP), Shim Cascade (SC), MTP Shim Lookahead (MSL), Shim Registry (SR), SE-RDAG, insert-once semantics, Precomputed Shim (PCS), Usage-Refined Shim (URS), Shim Backdoor, Shim Tier/Order (ST-k))
- `synthetic_collapse_benchmark.py` (full module — see exact functions below)
- `benchmark_utils.py` (metrics + isolated_adapter_state)
- `run_road_course_campaign.py` (RoadCourseProfile, evaluate_rankings, evaluate_profile, control_diagnostics including jaccard/route quality, variance, mask_density)
- `run_live_fire_diagnostics.py` (KNOWN_GOOD_THRESHOLDS incl. structural_health_min, StructuralHealthScore integration)
- `learned_mask_policy.py` (before/after extension pattern over synthetic fixture)
- `research_pathway_analyzer.py` (meta-analysis aggregation of synthetic + learned families)
- `structural_health_score.py` (StructuralHealthResult + evaluate with collapse/isomer/topology components)
- `feature_direction_bank.py` (FeatureDirectionBank: get_direction + update_from_activation overrides — direct analogy for Shim Registry)
- `tts_pipeline.py` (VectorSteerer + SteeringSignal — ephemeral contrast to registered/insert-once shims)
- `antigravity_engine.py` (run_inference SIPs: post-embed TTS intercept ~2452-2458, chelation decision ~2582-2600, _chelate_toxicity, set_static_dimension_mask, get_structural_health_report)
- `STEERING_CHELATION_RAGDAG_MICROSLM_RESEARCH_PLAN.md`, `STEERING_CHELATION_BHS_RESEARCH_RUBRIC.md`, `STEERING_CHELATION_10_LOOP_BHS_PROGRAM.md` (esp. Loop 8 evaluation, route acceptance under noise, Budget-Adjusted Lift, Reroute Acceptance Rate, Route Cohesion Score, Quantization Survival Delta)
- `test_synthetic_collapse_benchmark.py`, `test_road_course_campaign.py`, `test_live_fire_diagnostics.py`
- CLAUDE.md + `docs/conventions/brutal-honesty-rulebook.md` (v3.3): EVIDENCE/SMOKE, L1-L13 disclosures, Tier B independence, no "complete" claims without runtime proof

**Premise (BHS)**: This spec defines measurable, auditable extensions only. No production Shim Node code exists (confirmed via exhaustive grep: zero `ShimNode|ShimVector|ShimRegistry|MTP.*Lookahead|shim_nodes` in *.py). All "wiring" below is harness-only for now. Promotion of any shim benchmark result requires full BHS evidence chain on the exact production code paths once implemented.

---

## 1. Goals of the Extension

Enable the existing evaluation harness (starting with the deterministic synthetic surface, extensible to road-course/live-fire) to test Shim Nodes per the nomenclature without waiting for full SE-RDAG / engine integration.

Concrete required capabilities (per task slice):
- New scenario family: **"shim insertion under controlled semantic collapse"**
- Metrics for **cascade efficiency**: extra (simulated) tokens vs quality lift; cascade depth vs success
- Simple simulation of **MTP Shim Lookahead** (mock predictor sufficient for Loop 1-2)
- **Ability to register temporary Shim Nodes** for an experiment and measure before/after (rollback guarantees, no shared-state pollution — mirror `isolated_adapter_state`)

All extensions must be:
- Deterministic / reproducible (seeded where RNG used)
- Quantization-aware (shim vectors must be testable under same INT8/Bounded floors as embeddings)
- BHS-evidence-ready: every run produces `EVIDENCE:` (command + output), `SMOKE:` lines, before/after deltas, side-effect checks
- Reference exact existing symbols (no vague "the benchmark")

Success for skeleton "passing" (see skeleton file BHS notes): running the new entrypoints on the canonical collapse fixture (topic_count=4, collapse_strength=4.0) produces positive delta_ndcg for a corrective shim, correct cascade depth accounting, lookahead hit metrics, and clean rollback on registry. No mutation of module-level state across calls.

---

## 2. Reference: Existing Evaluation Surfaces (Exact Symbols + Lines)

### 2.1 synthetic_collapse_benchmark.py (core extension target)

**Public API (to be extended or wrapped)**:
- `build_synthetic_collapse_fixture(topic_count: int = 4, collapse_strength: float = 4.0) -> Dict[str, Any]` (lines 48-78): Returns `{"queries": {qid: np.ndarray}, "documents": {did: np.ndarray}, "qrels": {qid: relevant_did}, "collapse_dim": int}`. Creates topic_count topics + 1 collapse_dim; query + distractor both have high value on collapse_dim.
- `evaluate_synthetic_collapse(fixture: Dict[str, Any], *, masked_dims: List[int] | None = None) -> Dict[str, Any]` (lines 81-109): Core before/after harness. Applies optional multiplicative mask (0.0 on listed dims), runs `_cosine_scores` + `_rank`, then `_metric_row`. Returns `{"metrics": {"ndcg_at_3": float, "mrr": float, "recall_at_3": float}, "rankings": {qid: List[doc_id]}, "masked_dims": List[int]}`.
- `run_synthetic_collapse_benchmark(topic_count=4, collapse_strength=4.0) -> Dict[str, Any]` (lines 112-129): Orchestrates baseline + masked=[collapse_dim], computes `delta_ndcg_at_3`, `"recovered": bool`.
- Internal helpers (reusable):
  - `_cosine_scores(query: np.ndarray, documents: Mapping[str, np.ndarray]) -> Dict[str, float]` (14-20)
  - `_rank(scores) -> List[str]` (23-27)
  - `_metric_row(rankings, qrels, k=3) -> Dict` (30-45) — uses `ndcg_at_k`, `mean_reciprocal_rank`, `recall_at_k` from benchmark_utils.

**Current intervention model**: Multiplicative mask (zero-out). Shim insertion is **additive** (or gated-add) vector correction at a modeled SIP. Perfect analogy for extension: replace or augment the mask application with shim vector application.

**Test surface**: `test_synthetic_collapse_benchmark.py` asserts `recovered`, baseline < masked ndcg==1.0, distractor wins without intervention, validation on topic_count<2.

### 2.2 benchmark_utils.py (metrics + isolation primitives)

- `ndcg_at_k(r, k)`, `dcg_at_k(r, k)`, `mean_reciprocal_rank`, `recall_at_k`, `mean_average_precision_at_k` (exact impls lines 144-252)
- `isolated_adapter_state(adapter_path=None)` context manager (105-137): Critical pattern for clean before/after. Temporarily removes/restores adapter checkpoint. **Must be mirrored for any Shim Registry state**.
- MTEB helpers + canonicalize_id (used heavily in road-course).

**Recommendation**: Add `isolated_shim_registry_state()` context manager in the extension (or later in benchmark_utils) for the same reason.

### 2.3 Road-course / Live-fire surfaces (for later wiring)

**run_road_course_campaign.py** (exact):
- `RoadCourseProfile` dataclass (32-49): many controls (chelation_p/threshold, quantization, adapter_type, sedimentation, tts_config, etc.)
- `evaluate_rankings(rankings, qrels, k=10)` (217-240): Computes ndcg_at_10 / map / mrr / recall_at_10 using benchmark_utils metrics + canonicalize_id.
- `evaluate_profile(...)` (311-416): Constructs engine (with `_temporary_adapter_config`), optional sedimentation warmup, runs `engine.run_inference` per query, collects `rankings`, `action_mix`, `control_diagnostics` (variance_min/mean/max, jaccard_mean (primary route quality proxy), mask_density_mean, reformulation stats), `latency_ms_mean`, `telemetry`, optional TTS deltas. Calls `evaluate_rankings`.
- `run_campaign(...)` (468+): Loads MTEB via `load_mteb_data`, `select_road_course_slice`, evaluates profiles, picks best vs baseline, runs `quantization_survival_check` using `QuantizationPromotionGate`.
- SIP relevance: `engine.run_inference` post-embed TTS, chelation path, jaccard as route cohesion.

**run_live_fire_diagnostics.py** (exact):
- `KNOWN_GOOD_THRESHOLDS` (46-61) includes `"structural_health_min": 0.60`
- `StructuralHealthScore` integration (imported), `EventCollector`, deterministic `DeterministicEmbeddingBackend` + `FakeQdrant` mocks (perfect for shim vector injection tests without real models).
- Exercises `AntigravityEngine` + `RetrievalFitnessEvaluator`, `FitnessCompositionOrchestrator`, `QuantizationPromotionGate`, `IntegratedDiagnosticsReport`, `StabilityTracker`, etc.

**structural_health_score.py**:
- `StructuralHealthScore(collapse_weight=0.4, ...).evaluate(persistent_collapse_ratio, isomer_ratio, topology_drift) -> StructuralHealthResult` (score [0,1] + components + `penalty_multiplier`).
- Engine exposes `get_structural_health_report()` (tested in test_structural_health_report.py).

**antigravity_engine.py SIPs (for future real wiring)**:
- Post-embed: TTS `_tts.apply(q_vec)` → `q_vec = after_steering` (~2452-2458)
- Chelation decision + `_spectral_chelation_ranking` + `_chelate_toxicity` (~2582-2600, 528+)
- `set_static_dimension_mask` (826+)
- Telemetry via `get_runtime_telemetry()`, `get_last_runtime_diagnostics()`, `get_last_tts_result()`
- `get_structural_health_report()`

**learned_mask_policy.py** (extension precedent, lines 17-72):
- `learn_pairwise_collapse_mask(...)` → `{"policy", "masked_dims", ...}`
- `run_learned_mask_smoke(...)`: build fixture → learn → evaluate baseline + learned_result → delta + recovered. Directly models the desired "shim insertion" before/after.

**research_pathway_analyzer.py** (aggregation precedent):
- `run_meta_analysis` runs `run_synthetic_collapse_benchmark()` + `run_learned_mask_smoke()`, includes under `"synthetic_collapse"` / `"learned_mask"` keys with ndcg deltas + recovered.

---

## 3. New Test Families (Shim-Specific)

### Family A: Shim Insertion Under Controlled Semantic Collapse
- Fixture: reuse `build_synthetic_collapse_fixture` exactly (same collapse_dim noise).
- Intervention: instead of (or in addition to) `masked_dims`, register 1+ `ShimNode`(s) whose vector has negative component on collapse_dim + positive on semantic topic dim (or learned corrective).
- SIP model in harness (synthetic level): "post_embed" = add (gated) shim vector to query before `_cosine_scores`. Support "multiplicative_gate" or "additive" per nomenclature insert-once.
- Metrics: same as `_metric_row` (ndcg_at_3 primary) + delta vs baseline + "recovered" (ndcg >= 0.95 or ==1.0) + "shim_vector_norm" + "insertion_delta_norm".
- Variants: single corrective shim (Order-0), distractor shim (should not help or regress), tiered (ST-1 meta-shim).

### Family B: Cascade Efficiency (Compounding Shims)
- Define `ShimCascade` = ordered list[ShimNode] (depth = len).
- Execution model (simulated): start with baseline query vec; sequentially apply each shim in cascade (accumulate delta_norm cost); final scoring; optional "verification pass" cost.
- Simulated token accounting (required for BHS Budget-Adjusted Lift):
  - Base retrieval cost: constant (e.g. 100 "tokens" for embedding + top-k)
  - Per-shim insertion cost: `shim.cost_tokens` (default 5-20; configurable; higher for higher ST-k)
  - Cascade overhead: `depth * 3 + fanout_penalty`
  - Verification / rollback cost if cascade fails
- Metrics (new, in addition to NDCG):
  - `extra_tokens`: total cascade cost - baseline
  - `quality_lift`: ndcg_shim_cascade - ndcg_baseline (or vs no-shim retrieval)
  - `cascade_efficiency`: quality_lift / max(1, extra_tokens)   (primary; higher better)
  - `cascade_success`: bool (lift > min_lift_threshold AND depth <= max_depth AND final_structural_health >= threshold)
  - `depth_vs_success`: table or correlation across depths 1..K
  - `token_normalized_ndcg`: ndcg / (base + extra)
- BHS gate (from rubric): Report both raw lift and budget-adjusted. Unbounded cascades = failure (max_depth=3 default, max_fanout=2).

### Family C: MTP Shim Lookahead Simulation
- `MockMTPShimLookahead` (or `SimpleMTPPredictor`): lightweight mock (no real MTP head; dict of historical co-activation or rule-based).
  - `register_cascade_pattern(trigger_shim_id: str, likely_followers: List[str], scores: List[float])`
  - `predict_next(trigger_shim_id: str, context: Optional[Dict]=None, top_k: int=3) -> List[Tuple[str, float]]`
  - Optional: "hit rate" against synthetic "usage traces" (pre-generated successful cascades from Family B).
- Metrics:
  - `lookahead_precision@k`: fraction of predicted followers that appear in ground-truth cascade
  - `lookahead_recall@k`
  - `speculative_hit_rate`: % of cases where predicted shim(s) improve final ndcg when auto-inserted vs non-lookahead
  - Cost of false positives: extra_tokens on misses
- Integration: In cascade run, after first shim activation, consult mock predictor to auto-extend cascade (gated by policy score > threshold).

### Family D: Temporary Registration + Before/After + Rollback
- `TempShimRegistry` (or `ShimRegistry` with temp mode):
  - `register_temp(shim: ShimNode, experiment_id: str) -> token`
  - `apply_shims_to_vector(vec: np.ndarray, active_shims: List[ShimNode], sip: str="post_embed") -> Tuple[ndarray, Dict]`
  - `get_active_shims(experiment_id) -> List`
  - `unregister_temp(token)` or context exit → guaranteed rollback (no persistent mutation)
- Context manager: `with temp_shim_experiment(registry, [shim1, shim2]) as active: ...` (like isolated_adapter_state)
- Before/after protocol (exact):
  1. baseline = evaluate... (no shims)
  2. with temp registration: shimmed = evaluate...(with active shims)
  3. post-exit: re-evaluate baseline2 == baseline (bitwise or within 1e-12)
  4. Report side_effect_delta = |baseline2.ndcg - baseline.ndcg|
- Must work under `isolated_adapter_state` nesting (future engine shims will interact with adapters).

### Cross-Family Integration
- Extend `run_meta_analysis` style in `research_pathway_analyzer.py` to include shim families.
- Structural health under shims: feed shim-induced collapse/isomer deltas into `StructuralHealthScore.evaluate`.
- Quantization survival for shims: same `QuantizationPromotionGate` pattern as road-course `quantization_survival_check`.
- Later: road-course profile extension `RoadCourseProfile(..., shim_profiles: List[TempShimConfig])` and engine-level injection point (once SIPs exist).

---

## 4. Metric Definitions (Precise, Auditable)

Reuse:
- All from `benchmark_utils`: `ndcg_at_k`, `mean_reciprocal_rank`, `recall_at_k`, `mean_average_precision_at_k`

New / shim-specific (implement in extension module, export for tests):
```python
@dataclass
class CascadeMetrics:
    ndcg_at_3: float
    baseline_ndcg_at_3: float
    quality_lift: float
    cascade_depth: int
    simulated_extra_tokens: float
    cascade_efficiency: float  # lift / extra (or 0 if no lift)
    cascade_success: bool
    structural_health_after: float
    insertion_delta_norms: List[float]
    # + BHS fields: evidence_command, smoke_output_hash, etc.
```

- `compute_cascade_efficiency(lift: float, extra_tokens: float, depth: int, max_depth: int = 3) -> float`
- `compute_cascade_success(...) -> bool` (uses thresholds from KNOWN_GOOD or config: min_lift=0.05, max_depth=3, health_min=0.60)
- Lookahead metrics as above.
- Route quality under shims: extend jaccard computation to "shim_jaccard" (rankings with vs without shims at same depth).

All metrics must be float, finite, reported with per-query breakdowns where road-course does (for attribution).

---

## 5. Proposed Wiring / Implementation Surface (Exact References)

**Option A (preferred for minimal diff, Loop 1-2)**: New module `shim_collapse_benchmark_extension.py` (this task's skeleton) that **imports and composes** the existing functions. No edits to `synthetic_collapse_benchmark.py` required for first evidence. Later (Loop 3+): upstream the stable helpers.

**Option B**: Add to `synthetic_collapse_benchmark.py`:
- New dataclasses at top
- `class SyntheticCollapseBenchmark:` (wrapping the functions for stateful registry; task prompt references "SyntheticCollapseBenchmark class" — introduce here)
  - `def __init__(self, ...)` holding optional registry
  - Methods delegating to module funcs + new shim-aware ones
- New public: `build_shim_aware_fixture`, `evaluate_synthetic_collapse_with_shims(fixture, registry, active_shim_ids, mtp_predictor=None, ...)` — inside: copy of mask logic but `q_shimmed = apply_shim_insertion(q, shims)`
- `run_shim_collapse_benchmark(...)` that returns richer dict with all new metrics + "before": {...}, "after": {...}

**Shim data model (exact, nomenclature-aligned)**:
```python
@dataclass(frozen=True)
class ShimNode:
    shim_id: str
    vector: np.ndarray  # unit or bounded norm; stored normalized
    tier: int = 0  # ST-k
    cost_tokens: float = 10.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    # + provenance, version, cascade_partners: List[str]
```

**Registry (analogous to FeatureDirectionBank.update_from_activation + overrides)**:
- `class TempShimRegistry:`
  - `_overrides: Dict[str, ShimNode]`
  - `register_temp(...)`
  - `lookup_by_context(...)` (future: embedding similarity)
  - `get_cascade(shim_id) -> List[ShimNode]`
  - Context manager support for isolation

**Application helper** (new, modeled on `_cosine_scores` + VectorSteerer.steer):
- `apply_shim_vector(base_vec: np.ndarray, shim: ShimNode, strength: float=1.0, insert_once: bool=True) -> np.ndarray`

**MTP mock**:
- `class MockMTPShimLookahead:`
  - `predict_next(...)` returns candidates
  - `hit_rate(ground_truth_cascades: List[List[str]], predictions: ...) -> float`

**Entry points for smoke/EVIDENCE**:
- `python -m shim_collapse_benchmark_extension --family shim_insertion --topic-count 4`
- `run_shim_insertion_under_collapse_benchmark()`
- Integration smoke in `research_pathway_analyzer` style

**Road-course wiring (future)**:
- Add `shim_configs` to `RoadCourseProfile`
- In `evaluate_profile`, after engine creation (or inside a shim-aware engine subclass), apply temp registry to TTS or post-embed hook if present. Until then: vector-level shim injection on the q_vec extracted from diagnostics (or use live-fire deterministic backend).
- Extend `control_diagnostics` with `shim_cascade_depths`, `shim_efficiency_mean`.

**Live-fire wiring**:
- Use `FakeQdrant` + deterministic embed to inject shim vectors directly.
- Assert against `structural_health_min` after shims.

**BHS-mandatory in all outputs**:
- Every result dict must contain: `"bhs_evidence": {"command": "...", "output_snippet": "...", "timestamp": "..."}`
- `"side_effect_free": bool` (post-rollback baseline match)
- Quantization variant runs
- Max depth/fanout assertions (fail loud if violated)

---

## 6. Implementation Phases & Acceptance (BHS-Gated)

1. Skeleton (this task): dataclasses + stubs + one working `run_shim_insertion_under_collapse` using synthetic fixture + additive correction shim that recovers ndcg (analogous to mask). All TODOs marked. Tests pass on new test_ file (future).
2. Full metrics + cascade + MTP mock + TempRegistry context manager with rollback proof.
3. Wire as dependency into `research_pathway_analyzer.run_meta_analysis` (add "shim_families" key).
4. Live-fire / deterministic engine smoke using mocks (no real model).
5. Road-course profile extension + first real SIP hook (requires engine changes; out of this slice).
6. Full artifact cards + Tier B review when any claim promoted.

**"Passing" definition for this deliverable** (see skeleton BHS notes): `python shim_collapse_benchmark_extension.py` (or equiv) on default fixture emits JSON with positive `quality_lift`, `cascade_efficiency > 0`, `recovered: true` for corrective case, `side_effect_free: true`, and MTP mock hit metrics. All without external deps beyond numpy (already used).

---

## 7. Open Questions for Loop 2 (to be resolved with evidence)

1. Exact SIP surface in AntigravityEngine for registered shims (extend VectorSteerer? New chelation action "SHIM_INSERT"?).
2. How Shim Registry coexists with / extends FeatureDirectionBank (same overrides dict? separate?).
3. Token cost model calibration (real micro-SLM inference cost for shim selection vs simulated).
4. Interaction with sedimentation / online updates (do successful shims trigger adapter updates?).

---

## 8. Brutal Honesty on This Spec (Self-Applied)

This is a design for harness-level measurement of a not-yet-implemented primitive. It is **not** evidence that shims work. The synthetic surface is the only place where "shim insertion" can be measured today without new production code. All road-course claims in this spec are aspirational until engine SIPs exist.

References to "wiring into SyntheticCollapseBenchmark class" note that no such class currently exists in `synthetic_collapse_benchmark.py` (only free functions); the spec and skeleton introduce the class wrapper as the clean integration point.

No telemetry for per-shim token costs or cascade provenance exists in any benchmark today. This spec introduces simulated versions only.

All numbers (cost_tokens=10, thresholds) are placeholders for later calibration against real usage traces / OPSD data.

**Next concrete action (BHS)**: Implement the skeleton, run it, capture full stdout + hash of output as first EVIDENCE artifact in `artifacts/`.

---

*This spec converts the nomenclature's "Loop 8" benchmark callout into an executable, reference-exact, BHS-auditable plan while preserving strict separation between harness experiments and production surfaces.*