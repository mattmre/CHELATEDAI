# Shim Nodes + Cascades: SIPs & Smoke Plan (Agent 3 Deliverable)

**Program**: Steering-Chelation-RAGDAG-MicroSLM (10-Loop BHS-Governed)  
**Agent Slice**: Agent 3 — SIPs & Smoke Plan (Loop 1 substrate audit input)  
**Date**: 2026-05-26  
**Status**: Planning artifact only — no production code changes performed.  
**Inputs Consumed**:
- Full `shim_nodes_mtp_lookahead_nomenclature.md` (read end-to-end; canonical terms: Shim Vector (SV), Shim Node (SN), SIP, Shim Cascade (SC), MTP Shim Lookahead (MSL), SE-RDAG, insert-once semantics, usage-refined, Precomputed Shim (PCS), Shim Registry (SR) as FeatureDirectionBank extension).
- Deep audit of `antigravity_engine.py` (run_inference end-to-end, chelation paths, `_spectral_chelation_ranking`, `_chelate_toxicity`, variance/adaptive threshold, TTS intercept, model_scope observation).
- Full `tts_pipeline.py` (VectorSteerer, SteeringSignal, steer() impl, TTSPipeline.apply() chaining + feature_event loading).
- `model_scope_steering.py` (ModelScopeShadowSteerer.evaluate_capture, SteeringActuator.apply + scale/suppress paths, InterventionRecord provenance).
- `steering_policy.py` (ModelScopeSteeringPolicy, SteeringRule, PolicyRegistry, modes SHADOW/SOFT_SCALE/SUPPRESSION, status ACTIVE/DISABLED/BLOCKED).
- Supporting: `feature_direction_bank.py` (get_direction / overrides), `self_healing_chelation.py` (SelfEditDirective + generate_directives), `synthetic_collapse_benchmark.py` (build_synthetic_collapse_fixture + evaluate), engine wiring (enable_tts, run_inference intercept), config.py (chelation_threshold, adaptive params, SCOUT_K), L11 guards, tests (patched vs production paths), research docs (plan, README, BHS rubric extensions).

**Cross-References (file:line where relevant)**:
- Nomenclature integration table: `shim_nodes_mtp_lookahead_nomenclature.md:113-126` (TTS/VectorSteerer, chelation decision logic in antigravity, Model-Scope policies).
- Open questions: `shim_nodes_mtp_lookahead_nomenclature.md:167-174` (minimal VectorSteerer/FeatureDirectionBank changes; failed cascade detection/rollback; synthetic seeding).
- BHS considerations in nomenclature: `shim_nodes_mtp_lookahead_nomenclature.md:160-164` (token accounting, cascade bounding, provenance).
- Primary SIP locations hypothesized in nomenclature: `shim_nodes_mtp_lookahead_nomenclature.md:51-58`.

**Scope Lock (per this slice)**: Only planning document + pseudocode. No edits to `*.py`, no new classes, no registry impl, no test changes. All "proposed" are sketches for Loop 2 architecture consideration.

---

## 1. Prioritized SIP List (Minimal 3 for Highest First-Smoke Signal)

Rationale for selection (evidence-based from audits):
- **Highest leverage / lowest surface for smoke**: Must exercise (a) chelation variance as explicit trigger (nomenclature core), (b) steering node path (ephemeral SteeringSignal vs registered Shim Vector distinction at `tts_pipeline.py:27-31` vs nomenclature `36-41`), (c) synthetic collapse fixture (`synthetic_collapse_benchmark.py:48-78`) which deterministically produces high global_variance on collapse_dim.
- Existing decision surfaces already compute exactly the signals nomenclature wants (global_variance at `antigravity_engine.py:2569`, feature matches at `model_scope_steering.py:72-88`, delta application at `tts_pipeline.py:63-79`).
- Avoids aspirational surfaces (no RerouteDAG/SE-RDAG exists anywhere in code; block_graph dispatch is in computational_storage_poc but not wired to inference hot path).
- Enables "synthetic collapse + one shim + one short cascade" with measurable before/after on public benchmark fixture + engine run_inference path.
- BHS alignment: surfaces already produce runtime diagnostics, jaccard, masks, TTSResult, InterventionRecords — easy extension points for shim provenance without new storage initially.
- Minimality: 2 primary (steering + variance/chelation) + 1 supporting (feature/policy) gives cascade signal path diversity while keeping smoke harness small (numpy fixture + 1-2 engine calls).

**Prioritized SIPs**:

1. **SIP-1: TTS / VectorSteerer Steering Path (Highest priority for smoke)** — Direct contrast of ephemeral vs registered semantics. Post-embedding application point.
2. **SIP-2: AntigravityEngine Variance + Chelation Decision Surface (Core chelation signal path)** — Where nomenclature explicitly says "chelation variance threshold" proposes shim insertion.
3. **SIP-3: ModelScopeShadowSteerer + SteeringActuator Feature-to-Action Path (Supporting policy-driven)** — Feeds feature_events into SIP-1; extends rules for shim proposal.

All three are in hot or near-hot paths exercised by `run_inference` and `evaluate_capture`.

---

## 2. Detailed SIP Specifications

### SIP-1: VectorSteerer / TTSPipeline Steering Insertion (tts_pipeline.py + antigravity_engine.py)

**Exact Location (file:line range)**:
- Primary decision/application: `tts_pipeline.py:47-80` (VectorSteerer.steer full method) and `tts_pipeline.py:212-227` (TTSPipeline.apply steering stage, including feature_event-driven clear_signals + from_sparse_feature_event loading).
- Call site / intercept: `antigravity_engine.py:2452-2479` (post-embed, post-static-mask, pre-retrieval TTS intercept in run_inference: `q_vec = _tts_result.after_steering`).
- Construction site: `antigravity_engine.py:1066` (steerer = VectorSteerer(max_strength=0.3) inside enable_tts).
- Signal construction: `tts_pipeline.py:83-129` (from_sparse_feature_event using FeatureDirectionBank).

**What Decision Currently Happens There**:
- Accumulate zero or more ephemeral `SteeringSignal` (direction unit vec + strength + source string).
- In `steer()`: sum (strength * unit_dir), clamp total_delta_norm <= max_strength (0.3), return `v + total_delta`.
- Metadata: signals_applied count, total_delta_norm, was_steered bool.
- In apply(): optional transient load from feature_event (clears prior, adds), then steer. Stages_applied tracks "steering".
- Result flows into Antigravity q_vec before Qdrant scout (affects all downstream retrieval + variance calc + chelation).
- Distinction per nomenclature: these are **ephemeral per-inference-step additive**; no registration, versioning, cascade metadata, or insert-once guarantee.

**Proposed Minimal Change or Hook to Insert a Shim Node (Pseudocode / Diff Sketch)**:
```python
# tts_pipeline.py (minimal shim hook inside steer or as pre-pass in apply)
# --- DIFF SKETCH (non-executable; for architecture review only) ---
# Add (for smoke only, behind a _shim_smoke_enabled flag):
from typing import Optional, Dict, Any
# Assume stub:
# class ShimRegistryStub:
#     def lookup_by_context(self, ctx: np.ndarray, top_k=1) -> List[ShimCandidate]: ...
#     def get_cascade(self, shim_id: str) -> List[str]: ...
#     def record_insertion(self, shim_id, outcome): ...

def steer(self, v: np.ndarray, shim_context: Optional[Dict] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
    v = np.array(v, dtype=float)
    if not self._enabled or not self._signals:
        # NEW: shim consideration even with no ephemeral signals
        if shim_context and getattr(self, "_shim_registry", None):
            cands = self._shim_registry.lookup_by_context(shim_context.get("embedding") or v, top_k=1)
            if cands:
                shim_vec, shim_meta = cands[0]  # unit-norm SV + {"shim_id", "version", "tier":0}
                # insert-once: check a per-inference set of applied_shim_ids
                if shim_meta["shim_id"] not in self._applied_shims_this_call:
                    delta = shim_meta.get("strength", 0.2) * shim_vec
                    # clamp against existing total logic
                    ...
                    self._applied_shims_this_call.add(shim_meta["shim_id"])
                    meta["shims_inserted"] = [shim_meta]
                    # Trigger short cascade stub (MTP advisory)
                    cascade_ids = self._shim_registry.get_cascade(shim_meta["shim_id"])[:1]  # bound depth=1 for smoke
                    if cascade_ids:
                        meta["cascade_triggered"] = cascade_ids
                        # For smoke: sequentially add next SV (or fail closed)
    # ... existing total_delta accumulation for ephemeral signals ...
    # After existing delta:
    if "shims_inserted" in meta:
        # provenance for BHS
        meta["shim_versions"] = {s["shim_id"]: s.get("version") for s in meta["shims_inserted"]}
    return v + total_delta, meta
```
- In `TTSPipeline.apply()` and engine intercept: pass through `{"embedding": current, "variance": global_variance_from_caller, "collapse_signature": ...}`.
- Registry stub: backed by FeatureDirectionBank overrides + hardcoded smoke shims (seeded gaussians for collapse_dim correction).
- No change to SteeringSignal dataclass; shims are parallel "registered" path.

**How MTP Lookahead or Chelation Signal Would Feed Into It**:
- Chelation signal: `global_variance` (computed `antigravity_engine.py:2569`) + local_cluster or q_vec centroid passed as shim_context key for `lookup_by_context`.
- High variance (above adaptive threshold `antigravity_engine.py:2576`) raises priority of "correction shims" (e.g., negative on collapse_dim direction from synthetic fixture).
- MTP Shim Lookahead (MSL): stub `MTP_LOOKAHEAD_TABLE = {"shim_collapse_fix_v1": ["shim_verify_rerank_v0"]}` (advisory only, gated by policy stub + depth bound). When primary shim inserted, lookup suggests 1 dependent; attempt insert if not already applied. Per nomenclature `79-84` and `135`: "advisory + gated", never unconditional.
- In smoke harness: after successful primary insertion, force MTP suggestion and measure if secondary contributes (cascade_acceptance).

**Success / Failure Criteria for This Insertion Point**:
- **Success**: (a) On collapse fixture query, shim inserted (metadata["shims_inserted"] non-empty, version recorded); (b) jaccard or ranking improves vs identical run with shim hook disabled; (c) short cascade (depth=1) fires and is recorded; (d) insert-once prevents double-application in same call; (e) TTSResult extended with shim provenance without breaking existing stages_applied or total_delta_norm.
- **Failure**: (a) No insertion or regression in existing steering delta (L4 partial); (b) un-bounded cascade or explosion (even in stub); (c) shim delta violates max_strength clamp or produces NaN/inf; (d) breaks L11 TTS error fallback (original q_vec retained on error); (e) no rollback metadata produced.

### SIP-2: Antigravity Variance Threshold + Spectral Chelation Decision (antigravity_engine.py)

**Exact Location (file:line range)**:
- Core decision: `antigravity_engine.py:2566-2601` (after scout: dim_variances / global_variance calc at 2569, `_update_adaptive_threshold`, `with lock: active_threshold`, then `if self.use_quantization: if global_variance > active_threshold or self.use_centering: action="CHELATE" ... _spectral_chelation_ranking` else FAST; similar for `elif self.use_centering`).
- Supporting chelation surfaces: `antigravity_engine.py:528-551` (_chelate_toxicity: percentile variance mask), `1542-1602` (_spectral_chelation_ranking: center_of_mass, centering shift, mask application, temp-scaled rerank, chelation_log append).
- Upstream variance consumers: `2572` (adaptive), `2652` (retrieval_policy), `1213-1232` (_select_retrieval_policy), TTS post-embed `2452`.

**What Decision Currently Happens There**:
- Compute mean dim variance of local scout cluster as "K" / entropy signal.
- Adaptive threshold (lock-protected history, percentile) or static.
- Branch: high var (or centering forced) → "CHELATE" (full spectral centering + toxicity mask rerank via _spectral..., updates chelation_log) vs "FAST" (trust scout).
- Produces final_top_ids, mask (identity on FAST; _last_chelation_mask on CHELATE), jaccard (std vs chel), retrieval_policy dict with variance_above_threshold.
- Directly drives log, stability_tracker, online_updater, diagnostics.
- Per nomenclature: "High local variance or isomer drift can propose 'shim insertion' as an action alongside or instead of classic rerank" (`121`).

**Proposed Minimal Change or Hook (Pseudocode / Diff Sketch)**:
```python
# antigravity_engine.py:2578 (inside run_inference, after active_threshold)
# --- DIFF SKETCH (planning only) ---
action = "FAST"
shim_insertion = None
if global_variance > active_threshold or self.use_centering:
    # NEW minimal hook (behind smoke flag, non-mutating first)
    if getattr(self, "_shim_smoke_mode", False) and hasattr(self, "_shim_registry"):
        ctx = {"embedding": q_vec, "global_variance": global_variance, "local_centroid": np.mean(local_vectors, axis=0)}
        shim_cand = self._shim_registry.lookup_by_context(ctx.get("embedding"), variance=global_variance)
        if shim_cand and shim_cand.confidence > 0.6:  # gate
            shim_insertion = {"shim_id": shim_cand.id, "vec": shim_cand.vector, "version": "smoke-v0", "cascade": self._shim_registry.get_cascade(shim_cand.id)[:1]}
            # Apply shim vector directly to q_vec (insert-once at this surface) BEFORE chelate decision
            q_vec = q_vec + (shim_cand.strength * shim_cand.vector)  # or gated blend
            action = "SHIM_CHELATE_HYBRID"
            # Then proceed to (or short-circuit) spectral? For smoke: still call for comparison.
    if action != "SHIM_CHELATE_HYBRID":
        action = "CHELATE"
        chel_top, center_of_mass = self._spectral_chelation_ranking(q_vec, local_vectors, std_top)
        ...
# Later in _build_runtime_diagnostics / retrieval_policy: record shim_insertion + cascade
```
- Or lighter: post-variance, before if, call advisory `consider_shim(q_vec, variance)` returning optional correction vector (shim) to add to q_vec or to pass into spectral.
- Inside _spectral or _chelate_toxicity: after mask, optional additional shim vector * mask.
- Changes only diagnostic paths + one early q_vec adjustment; existing chelation_log / mask paths untouched initially.

**How MTP Lookahead or Chelation Signal Would Feed Into It**:
- Chelation signal is *native*: global_variance + dim_variances + center_of_mass (from spectral) are first-class keys for registry lookup (nomenclature `109`: "Chelation variance signals are first-class triggers").
- High variance directly elevates shim priority over pure FAST.
- MTP: on high-var pattern match, MTP stub suggests "post-correction verification shim" (e.g., one that biases toward retention of original relevant docs). Advisory: only attempted if primary shim improved local jaccard proxy.
- In engine: variance history window could seed simple frequency-based MTP predictions for smoke.

**Success / Failure Criteria for This Insertion Point**:
- **Success**: (a) On synthetic collapse fixture (high collapse_dim variance), shim hook triggers (action recorded as SHIM_* or shim_insertion present in diagnostics); (b) final_top or jaccard improves vs baseline same-seed run without hook; (c) chelation signal (variance) is the *sole* trigger for this smoke (no feature_event required); (d) mask / centering still run (or short-circuited cleanly) for comparison; (e) rollback path: if post-shim jaccard < pre-shim, restore original q_vec + record rollback in diagnostics.
- **Failure**: (a) Shim never considered despite variance > threshold (L2 escape); (b) corrupts adaptive_threshold lock or _variance_history; (c) double-counts correction (shim + full chelate without accounting); (d) regression on non-collapse queries (FAST path must be identical); (e) no provenance in _build_runtime_diagnostics or retrieval_policy.

### SIP-3: ModelScopeShadowSteerer Feature Rule Matching to Steering Action (model_scope_steering.py)

**Exact Location (file:line range)**:
- `model_scope_steering.py:53-145` (evaluate_capture: observation loop, feature dict build `68-71`, rule matching `72-88` (layer, min_value), matched_features, ActivationEvent + SparseFeatureEvent construction, actuator.apply).
- `model_scope_steering.py:220-313` (SteeringActuator.apply: target extraction, status/BLOCKED/DISABLED/max caps `238-253`, SHADOW vs SOFT_SCALE `277-282` (multiply scale_factor) or SUPPRESSION (zero), InterventionRecord with original/modified, provenance).
- Policy input: `steering_policy.py:13-35` (SteeringRule), `39-62` (ModelScopeSteeringPolicy), `82-100` (SteeringPolicyConfig), registry ACTIVE filter.

**What Decision Currently Happens There**:
- For each layer observation: match active rules on feature_id + min_value → collect recommended_actions (with strength).
- Build SparseFeatureEvent → actuator.apply (policy status/caps guard) → in non-SHADOW: mutate feature values (scale or zero) → return new_event + full InterventionRecord (applied bool, features_modified, decline_reason, rollback via original_values).
- Output drives TTS (via feature_event path in SIP-1) or shadow recording.
- Strong provenance (record_id, timestamps, run/layer/model_id) but limited to scale/suppress; no "shim" action_type yet.

**Proposed Minimal Change or Hook (Pseudocode)**:
```python
# model_scope_steering.py (in rule matching or apply)
# Extend SteeringRule with optional action_type="shim_insert" + shim_id
if rule.action_type == "shim_insert":
    # Instead of (or after) scale:
    shim_cand = shim_registry.lookup_by_feature(rule.feature_id, value)
    if shim_cand:
        record = InterventionRecord(..., features_modified=[f"shim:{shim_cand.id}"], ...)
        # Emit to caller (evaluate_capture return) a new "shim_proposals" list
        # Do not mutate features; shim handled downstream in SIP-1 with registered SV
        return feature_event, record, {"shim_proposals": [shim_cand]}
# In shadow steerer return dict: add "shim_proposals"
```
- For smoke: one rule that on high collapse-related feature value proposes a specific shim_id.

**How MTP / Chelation Would Feed**:
- Feature values can be downstream of chelation variance (via model_scope observation in engine run_inference `2326` + `1267` "steering").
- MTP could predict "next feature → shim" pairs from historical matched_rules + successful shims.

**Success/Failure**:
- Success: rule match on synthetic data emits shim_proposal; actuator records it without breaking scale/suppress on other rules; proposal reaches TTS steerer.
- Failure: BLOCKED/DISABLED paths swallow shim proposals silently; provenance incomplete for rollback.

---

## 3. End-to-End Smoke Test Scenario (Synthetic Collapse + One Shim + Short Cascade)

**Fixture**: `synthetic_collapse_benchmark.py:build_synthetic_collapse_fixture(topic_count=4, collapse_strength=4.0)`. Produces queries with deliberate high-magnitude collapse_dim that should be "toxic" (high variance in local cluster).

**Scenario Steps (smoke harness pseudocode, production path execution required)**:
1. Build fixture + gold qrels.
2. Baseline (no shims, no TTS or minimal): `evaluate_synthetic_collapse(fixture)` → record ndcg_at_3, mrr, recall@3, rankings.
3. Enable minimal shim substrate (stub registry pre-populated with 2-3 PCS shims: one primary "collapse_fix" SV that counters collapse_dim direction (seeded via FeatureDirectionBank style or explicit negative), one dependent "verify" shim).
4. Wire smoke hooks (non-mutating where possible, or behind flag) into:
   - SIP-2 decision (variance > thresh → consider/lookup/insert primary shim on q_vec pre-scout or pre-chelate).
   - SIP-1 (in steer or apply: after primary, MTP stub suggests + inserts short cascade shim if not applied).
5. Run identical fixture through instrumented path (or full AntigravityEngine with enable_tts + populated corpus mirroring fixture vectors, run_inference per query).
6. Capture: extended diagnostics (shim_inserted, versions, cascade_triggered, per-shim delta_norm, rollback_events), jaccard, final rankings, TTSResult or retrieval_policy.
7. Compute deltas vs baseline.
8. Inject "bad shim" variant (wrong direction SV): run, detect failure (jaccard drop or post-insertion variance spike or structural health proxy), exercise rollback (restore prior vector state + record success).
9. Repeat with MTP disabled (single shim only) for cascade delta.
10. Full replay: serialize key diagnostics artifact (query + variance + shims_applied + versions + outcome), fresh checkout + re-execute same path from artifact, match results.

**Expected Smoke Outcome (for "success" declaration gate)**: Measurable lift on collapse recovery (e.g., ndcg delta >0.1 or equivalent to classic mask in fixture) with documented shim + cascade, zero regression on control queries, full provenance.

---

## 4. Required New Metrics (for Smoke + Future BHS)

- **Cascade Acceptance Rate**: (num runs where primary shim triggered AND MTP-suggested dependent was attempted and contributed measurable delta) / (num primary insertions). Target for smoke: >0 (existence) + bounded depth/fan-out=1.
- **Token Delta (proxy at this layer)**: For embedding-only smoke: (a) vector-op count / FLOPs for shim insertion+lookup vs full spectral chelation (center_of_mass + mask + scores); (b) "effective retrieval depth saved" (scout K reduction enabled by shim correction). Note: true LLM token accounting (full RAG + generation) deferred to Loops 8-10 per nomenclature `157-158`. Must never claim "token reduction" without before/after on identical queries + quality gates.
- **Rollback Success Rate**: (successful restores to pre-shim state on injected-bad-shim cases, verified by identical pre/post vector + downstream ranking) / (bad-shim injections). Must include provenance (InterventionRecord-style or new shim ledger entry).
- **Shim Insertion Rate under Trigger**: % of high-variance collapse queries that actually performed >=1 shim insertion (tests SIP-2 trigger fidelity).
- **Jaccard / NDCG Delta with/without Shim (paired)**: On exact same fixture + seeds.
- **Provenance Completeness**: % of shim insertions that produced versioned record + cascade metadata in diagnostics/TTSResult/InterventionRecord (target 100% for smoke pass).
- Existing retained for comparison: global_variance, active_threshold, jaccard, retrieval_policy action, TTS total_delta_norm.

All must be captured in runtime diagnostics (extend _build_runtime_diagnostics, TTSResult, etc.) and emitted in smoke script output.

---

## 5. BHS Evidence Requirements for Declaring Smoke Successful

Per CLAUDE.md brutal honesty + nomenclature `160-164` + program rubric (route acceptance under noise, rollback, quant survival, no "it worked in simulation"):

- **EVIDENCE:** line in any summary: exact command + stdout from *production code path* (unpatched AntigravityEngine.run_inference or synthetic evaluate with hooks exercised on real fixture vectors; not unit test mocks).
- **SMOKE:** reproducible script (or notebook) + artifact (JSON with queries, variances, shim_ids+versions, before/after rankings, cascade events, rollback trace) that survives `git clean -fdx` + fresh checkout + re-run.
- **Independent adversarial run**: Second agent (not author) executes smoke harness end-to-end, confirms deltas + rollback.
- **Full chain**: (a) baseline fixture numbers; (b) shim-enabled numbers on identical inputs; (c) cascade trace; (d) bad-shim + rollback success; (e) no L11 violations (inference never died); (f) quant simulation path exercised if possible.
- **No omission**: Any L1-L13 stub/escape/mock/partial/broad-catch in the smoke harness itself must be disclosed with `file:line` (even if harness is throwaway).
- **Quantitative gate example**: "Shim + cascade recovered ndcg_at_3 within 5% of ideal mask baseline on collapse fixture, with cascade_acceptance=1.0 (n=4 queries), rollback_success=1.0 (n=3 bad injections), zero regression on control paths (measured jaccard delta <0.01 on low-var queries)."
- Promotion to Loop 2 architecture requires Tier B review + BHS_OFFICIAL=100 on the smoke evidence package.
- "Visible means verified": No UI/dashboard surfacing of "shim success" until evidence exists.

---

## 6. Brutal Honesty on Implementation Difficulty + Hidden Dependencies Discovered

**This is a high-quality planning artifact only. No claim is made that shims "work" or are "ready". All integration is hypothesis.**

**Difficulty Assessment (Brutal)**:
- **High (7-8/10 for even minimal smoke)**: The surfaces are clean and high-signal, but the delta between "audit + pseudocode" and "working insert-once registered versioned cascadable shim with MTP advisory + rollback + full provenance surviving quant + BHS replay" is massive. Requires new ShimNode dataclass, ShimRegistry (even stub), version ledger, cascade bounding primitive, extended diagnostics in 3+ files, smoke harness that exercises production paths, and rigorous paired before/after + adversarial review. Synthetic seeding (nomenclature open Q5) is mandatory for first useful data — organic usage does not exist.
- **Cascade semantics are underspecified in practice**: nomenclature gives excellent terms, but "compounding" (vector-to-vector? state machine? DAG edge annotation?) has zero implementation precedent. Smoke must artificially define "one sequential additive dependent" and bound it ruthlessly.
- **Metrics translation risk**: "token delta" at vector layer is a proxy; claiming efficiency requires future full LLM integration. Easy to overclaim.
- **Time to first real EVIDENCE line**: Multiple days of careful non-regression work even for throwaway harness. Risk of L4/L5/L11/L12 violations is real during integration.

**Hidden / Non-Obvious Dependencies & Landmines (file:line cited)**:
- **L11 safety nets everywhere** (`antigravity_engine.py:2465-2469`, `2471-2479`, `1082-1086`, similar in TTS apply): shim code *must* live inside or replicate the "never kill inference" contract. New exception paths are high-risk.
- **State management seams**: VectorSteerer._signals cleared conditionally (`tts_pipeline.py:220`); adaptive threshold lock (`antigravity_engine.py:2575`, `_adaptive_threshold_lock`); model_scope observation flags (`antigravity_engine.py:1239-1242`). Shim state (applied_shims set, registry) risks races or leakage.
- **FeatureDirectionBank limitations** (`feature_direction_bank.py:30-52`): only overrides + gaussian; no versioning, no cascade metadata, no usage ledger. "Extension" is non-trivial refactor surface.
- **SelfEditDirective not yet shim-aware** (`self_healing_chelation.py:287-407` generate_directives): nomenclature wants `shim_directive` variant (Loop 7), but smoke cannot rely on it.
- **Model scope is best-effort/optional** (`antigravity_engine.py:1235-1284`): SIP-3 may have zero observations in minimal runs.
- **Synthetic collapse vs full engine**: Fixture is pure np (`synthetic_collapse_benchmark.py`); full SIP-1/2 smoke needs corpus population, Qdrant, possible teacher, adapter — many init paths (`antigravity_engine.py:23-159`).
- **Dashboard / telemetry** (`antigravity_engine.py:2460-2464`): new shim fields must not break update calls.
- **Quantization simulation** (`antigravity_engine.py:188-211`, `204-211`): shims must be tested under INT8 floors or explicitly declared out-of-scope for smoke.
- **Test vs prod gap**: `test_*.py` heavily patch loggers/dashboard; BHS demands unpatched runtime evidence.
- **No existing rollback primitive at vector/TTS level**: only feature_event rollback (`model_scope_steering.py:385-402`). New shim rollback must be invented.
- **Broad catches + silent degradation** (multiple L11 in engine run_inference): easy to swallow shim failures.
- **Config / preset surface** (`config.py:127-165`): chelation_thresholds, adaptive params are road-course tuned; shim insertion changes effective "K" behavior.
- **Absence of RerouteDAG**: All DAG talk is aspirational (`docs/.../*.md` only). Smoke is strictly vector correction + retrieval ranking improvement.

**Overall BHS Self-Assessment on This Document**: This plan is complete for its narrow slice (SIPs + smoke design). It cites exact lines, distinguishes hypothesis from reality, discloses difficulty and landmines, and provides actionable criteria. It does *not* claim any implementation progress. Any future PR using this must still produce its own independent EVIDENCE/SMOKE + full Brutal Honesty section (no "per Agent 3 plan" shortcuts).

**Recommendation for Loop 1 Synthesis / Loop 2**: Elevate SIP-1 + SIP-2 as the two minimal insertion surfaces for first SE-RDAG prototype. Fund synthetic seeding + stub registry + harness as explicit work item before any micro-SLM policy work. Treat MTP as advisory lookup table until real head exists.

*End of Agent 3 SIPs & Smoke Plan deliverable. All work confined to /home/mattmre/CHELATEDAI. No production files modified.*

---

**Appendix: Quick Line Map for Reviewers**
- Nomenclature SIP candidates: shim_nodes_mtp_lookahead_nomenclature.md:51-58, 113-126
- Antigravity variance/chelation: antigravity_engine.py:2566-2601, 528-551, 1542-1602, 2452-2479
- TTS/VectorSteerer: tts_pipeline.py:47-80, 212-227, 83-129
- ModelScope actuator: model_scope_steering.py:53-145, 220-313
- Synthetic fixture: synthetic_collapse_benchmark.py:48-78, 81-109
- Wiring: antigravity_engine.py:1044-1067 (enable_tts)
- BHS guardrails: CLAUDE.md (via system), nomenclature:160-164, research rubric.