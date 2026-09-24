# 02: VectorSteerer SIP Substrate Audit (BHS 5-Min Shim Loop, Cycle 1 — Agent A)

**Agent**: Agent A (Research & Mapping)  
**Cycle**: Official first BHS 5-Minute Shim Loop (per BHS_5MIN_SHIM_LOOP_GOAL.md:91-104)  
**Date**: 2026-05-26  
**Selected Surface**: `tts_pipeline.py` VectorSteerer + TTS intercept (strongest of the two mandated options; see selection rationale below)  
**Governing Docs**: shim_nodes_mtp_lookahead_nomenclature.md (full), 15_shim_concepts_mapping.md, shim_node.py + shim_node_interface.md (steering artifacts/), brutal-honesty-rulebook.md v3.3, STEERING_CHELATION_BHS_RESEARCH_RUBRIC.md (cross-ref)  
**Output Contract**: Short, file:line-grounded, brutally honest. No L1-L13 in this artifact itself. All claims cite runtime surface or explicit absence. This discharges the "Full substrate audit of one major host surface" slice (goal:99) and "shim substrate readiness" mandate (nomenclature:152).

---

## 1. Selection of Surface (Strongest Signal)

Per nomenclature §3 (113-126) and 15_shim...md:44-51 (Tier-1 hosts ranked):

- **Chosen**: `tts_pipeline.py` (VectorSteerer.steer + from_sparse_feature_event + TTSPipeline.apply + TTS intercept in antigravity_engine.py:2452-2458).  
  This is the **direct execution site for directional additive overrides** (the exact semantic target of "Shim Vector" vs ephemeral SteeringSignal per nomenclature:39). FeatureDirectionBank seeds the directions here (tts:107,120).

- **Rejected (for this slice)**: antigravity_engine.py variance/chelation paths (`_chelate_toxicity:528-551`, `get_chelated_vector:553-593`, global_variance + mask logic:2566-2590, `_spectral_chelation_ranking:1542+`).  
  These are high-signal *decision/trigger* surfaces (variance > threshold → policy) and host the TTS call, but apply **multiplicative per-dimension masks** (q_vec * mask), not additive registered directional vectors. They are Tier-2 per mapping. Strong for future "high-variance → lookup_by_context shim proposal" hook, but weaker for core insert-once/registered mechanics.

Cross-surface observation (honest): The TTS intercept **is** the production usage of VectorSteerer (antigravity:2456-2458: `q_vec = _tts_result.after_steering`). Any real SIP wiring must survive this path.

**BHS Evidence of selection correctness**: Grep across prod *.py for "SteeringSignal|VectorSteerer|steer\(" returns exclusively tts_pipeline.py:27-289 + its callers (antigravity + tests). Zero shim-related symbols in any core file.

---

## 2. Exact Seams vs Shim Nomenclature Requirements (insert-once, registered vs ephemeral, cascade, provenance)

**Core mismatch (L4/L9 risk if ever mis-surfaced)**: The entire surface implements the *counter-example* explicitly called out in nomenclature:39 and 15_shim...md:19-20.

### 2.1 Ephemeral-only, no registration (nomenclature 2.1, §4.1, interface:63-76)
- `tts_pipeline.py:27-30`: 
  ```python
  @dataclass
  class SteeringSignal:
      direction: np.ndarray
      strength: float
      source: str  # only "provenance"
  ```
  No `shim_id`, no `tier`, no `version`, no `cascade_targets`, no `provenance: dict`, no `usage_stats`.
- `tts_pipeline.py:37`: `self._signals: List[SteeringSignal] = []` — transient queue.
- `tts_pipeline.py:43-45`: `clear_signals(self)` — destructive.
- `tts_pipeline.py:216-222` (in TTSPipeline.apply, the hot path):
  ```python
  if feature_event is not None:
      self._steerer.clear_signals()  # "transient per-inference"
      for sig in VectorSteerer.from_sparse_feature_event(...)._signals:
          self._steerer.add_signal(sig)
  ```
  Comment at 218-219 explicitly codifies the ephemeral contract: "Signals added via steerer.add_signal() (external/persistent) only persist when feature_event=None."
- `tts_pipeline.py:83-129` (from_sparse_feature_event): Builds **fresh temporary VectorSteerer** every time from FeatureDirectionBank. `source=f"sparse_feature_{feature_id}"` is the sole identity. No registry lookup, no versioning.

**Registered Shim Vector requirement (nomenclature:36-41, shim_node.py:180-183)**: "A Shim Vector is **registered**, **versioned**, and **cascadable**." "Actual vector application ('insert') happens at a SIP outside this module." Current surface has zero registration surface.

### 2.2 No insert-once semantics (nomenclature §4.1, interface:72)
- `tts_pipeline.py:47-80` (steer):
  ```python
  for sig in self._signals:
      ... total_delta += sig.strength * d
  # global clamp only (72-74), no per-id guard
  return v + total_delta, {"signals_applied": len(self._signals), ...}
  ```
  No `_inserted_this_pass` set, no `if shim_id in ...: skip`. Every call to steer() re-applies whatever is in the list. Clear + rebuild on feature_event path makes "once" impossible without external state the SIP does not own.

### 2.3 No cascade support (nomenclature 61-71, shim_node.py:399-440 get_cascade)
- Zero `cascade_targets`, zero `get_cascade`, zero compounding traversal, zero tier (ST-k).
- `steer()` sums flat list; no ordered escalation Order-0/1/k, no meta-shims.

### 2.4 Provenance / rollback / ledger gaps (nomenclature §4.7, interface:86, shim_node.py:98+265-278)
- Only `source: str` on signal. No `input_hash`, no `created_at`, no stable 16-char provenance (cf. self_healing_chelation _stable_hash), no ledger entry per *insertion* that survives replay.
- `antigravity_engine.py:2638` (log_query) and chelation_log record *variance/action*, not per-correction identity. TTS result metadata (tts:239-241) records aggregate `total_delta_norm` + `stages_applied`, never "shim_id:xxx inserted at this hash".
- No rollback path keyed to a specific override (existing ES rollback_to_elite is population-level, not per-shim).

### 2.5 Related precursor surface (FeatureDirectionBank) — bridge exists but unused for shims
- `feature_direction_bank.py:30-52`: `_overrides: Dict[str, np.ndarray]`, `update_from_activation` (register real SAE row), `get_direction` (seeded gaussian fallback). SHA-256 seeding matches shim_node register_seeded exactly.
- **Seam**: This *could* be the ShimVectorProvider (shim_node.py:54-80 Protocol + 538-554), but no adapter, no shim_id abstraction, zero usage telemetry, zero cascades. Grep confirms zero cross-calls between bank and any shim artifact in prod paths.

### 2.6 TTS intercept seam in host engine (antigravity)
- `antigravity_engine.py:2452-2458`:
  ```python
  _tts = getattr(self, '_tts_pipeline', None)
  if _tts is not None:
      _tts_result = _tts.apply(q_vec)  # NOTE: no feature_event, no shim context passed
      ...
      q_vec = _tts_result.after_steering
  ```
- Variance decisions (2566-2590) feed `_select_retrieval_policy` (1213+) but never a shim registry. Mask application remains pure multiplicative (589, 1574).

**No production ShimNode or ShimRegistry symbols exist anywhere in *.py outside docs/steering.../artifacts/** (exhaustive grep 2026-05-26). The full ShimRegistry (register/get_cascade/record_activation/provenance stamping/insert semantics contract) is L4 research scaffold only (shim_node.py:34-36, 620-632 explicit BHS self-attestation).

---

## 3. How Far From Supporting Real Shim Nodes (Brutal Honesty)

**Distance**: Extremely far — this surface is the *canonical illustration of what shims are not*.

- 0 lines of SIP wiring for registered/insert-once/cascadable shims.
- 0 runtime evidence (no EVIDENCE:/SMOKE: for any shim behavior on prod path; tests exercise the ephemeral signal path only).
- Current behavior is *by design* the ephemeral SteeringSignal model that nomenclature §2.1 and §7 call out as the problem to solve.
- The isolated `shim_node.py` + `TempShimRegistry` + benchmark harnesses (steering/artifacts/) prove the *data structures* are implementable in isolation (and pass their internal BHS evidence predicates), but deliver **zero observable effect** on VectorSteerer.steer(), antigravity inference, or any retrieval metric. Claiming "shims are ready" from these artifacts alone would be L4 + L9 + L13.
- Visible-means-verified (rulebook Rule 2): Nothing is surfaced. Good. But the gap is architectural, not "just a few lines."

**L-taxonomy disclosures in current substrate (for any future wiring PR)**:
- L1 risk if a stub `insert_shim` were added that returns without effect.
- L2 risk if "if not shim_registry: return old_behavior" guards appear in the same diff as "shim support."
- L5/L8: Existing tests (test_tts_pipeline.py, test_antigravity_engine.py) assert on signals_applied / delta_norm; any shim extension must not make those tests "assert the bug."
- L11: The broad except in TTS dashboard (antigravity:2465-2470) and TTS error fallback (2471-2479) are already disclosed in source; shim path must not add new swallows.

No hidden claims. This audit itself is the evidence.

---

## 4. 2-3 Concrete Next-Build Recommendations (Scoped for BHS 100 in Next Slices)

Prioritized for minimal delta that can produce runtime evidence + floor smoke while respecting 100-gate + carried-debt rules. These map directly to goal backlog item 1 ("Wire first real minimal SIP... with rollback") and nomenclature open Q1 (167).

1. **Minimal VectorSteerer SIP extension (highest leverage, Agent B slice)**:  
   Add to `VectorSteerer` (tts_pipeline.py:33 class):
   - Optional `shim_registry: Optional["ShimRegistry"] = None` (import under TYPE_CHECKING or string to avoid cycle).
   - `insert_shim_once(self, shim_id: str, context: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Dict]` (or mutate-in-place + return meta only).
     - Guard: `if self._inserted_shims is None: self._inserted_shims = set()`; if shim_id in set: return no-op meta with "skipped":"insert-once".
     - On hit: `vecs = registry.get_vectors(shim_id)` (or provider), sum (respect existing clamp at 72-74), record `registry.record_activation(shim_id, ...)`, return augmented meta: {"shim_inserted": shim_id, "provenance_hash": node.provenance["input_hash"], "tier": ...}.
   - Keep *all* existing signal paths 100% unchanged and exercised.
   - BHS floor smoke target: import + roundtrip register_seeded shim → insert_shim_once on a steerer → verify delta applied exactly once, registry usage_stats incremented, old signal path still works, no mutation of inputs. Explicit "ceiling gap: no token-accounted end-to-end on held-out queries; no MTP" in CARRY_FORWARD.
   - File:line impact: ~15-25 LOC delta + docstring BHS EVIDENCE block. Matches shim_node.py:624-625 target.

2. **FeatureDirectionBank → ShimVectorProvider bridge (low-risk compatibility slice)**:  
   In `feature_direction_bank.py` (or thin `shim_vector_provider.py` adapter), implement the Protocol from shim_node.py:54-80.  
   - `get_vectors(shim_id)` delegates to `self.get_direction(shim_id)` wrapped as [unit-norm copy].
   - Add one deterministic test: after `bank.update_from_activation(fid, row)`, a registry seeded from the provider returns bitwise-identical vector (per shim_node.py:310-313 evidence predicate).
   - Enables first PCS shims without duplicating gaussian logic. Zero behavior change to existing callers.

3. **Variance → shim lookup hook at decision surface (antigravity trigger path, pairs with 1)**:  
   In `antigravity_engine.py` variance branch (2566-2590) or `_select_retrieval_policy:1213`, when `shim_registry` is wired:
   - `candidates = registry.lookup_by_context(q_vec, top_k=3)` if global_variance > active_threshold.
   - Surface in the returned `retrieval_policy` dict (and diagnostics) as `"shim_candidates": [ {"id": s.shim_id, "tier": s.tier, "sim": ...}, ... ]`.
   - No auto-insertion. This makes chelation variance a first-class SIP *proposal* surface (nomenclature:53,121) and gives MTP/policy something to consume later.
   - Evidence: before/after policy dict on same query; zero change to retrieval results or masks.

These three can be done in parallel slices by different agents, each hitting BHS 100 independently with floor smoke + honest debt disclosure. They produce the first *observable* shim registry interaction on a real SIP without scope explosion.

---

## 5. Cross-References & Evidence Basis (for Tier B / Next Agents)

- All cited seams verified by direct read + grep 2026-05-26 on the exact files.
- Prior Loop 01 artifacts (15_shim...:19-35 pain points; 00_kickoff:18-19 Qs; shim_smoke_plan.md:292-293) independently flag the identical tts:27-222 and antigravity variance lines.
- shim_node.py:620-632 and interface:101-111 already contain the L4 self-disclosure that this audit confirms at the SIP layer.
- No runtime output from any "shim" path in prod code exists (tests, harnesses in artifacts/, and dashboard are the only consumers of the research shims).
- BHS rulebook §0 evidence rule, §1 L4/L5/L9/L13, Rule 2, §4 template fully internalized for this output.

**Next-cycle handoff**: This note + the selected surface file:line map should be input to Agent B (build) for slice #1 above. Update BHS_SHIM_LOOP_DASHBOARD.md with "VectorSteerer SIP audit complete; 0/7 shim requirements met; 3 scoped recs ready."

---

## Brutal Honesty on This Audit Note (Self-Contained)

**What this did NOT do (to avoid L4 overclaim)**:
- No code changes, no stubs added to tts_pipeline.py or antigravity_engine.py.
- No execution of any harness against a "shim" version of VectorSteerer (none exists).
- No claim of "progress toward production shims" beyond "audit performed and gaps named at file:line."
- No new metrics, no dashboard update (that is Agent E).
- This is mapping + gap analysis only. The 3 recs are recommendations, not implemented slices.

**What it *did* deliver (evidence)**: 
- Exhaustive, citation-dense mapping of the exact highest-signal SIP against the canonical requirements (nomenclature + interface + prior loop artifacts).
- Brutal distance assessment grounded in "zero symbols in prod paths."
- Actionable, scoped, BHS-gate-respecting next steps that a fresh agent can execute in <5min cycle budget.

All assertions survive the "try to disprove" test against the actual files. Empty answers are justified (no production shims = nothing to run smoke against).

**BHS_SELF_DRAFT (for this research slice only)**: 88 (strong coverage of mandated surfaces + nomenclature contract; minor: did not re-audit every test file line or every variance subpath in antigravity to same depth — those are secondary for this SIP choice).

---

## Cycle 2 Agent A (BHS 5-Min Shim Loop) Targeted Cross-Ref Note
**Date**: 2026-05-26 (Cycle 2)  
**Action**: Performed fresh cross-reference of the exact seams cited in §2 (tts_pipeline.py:47-80 steer + 216-222 apply path; antigravity_engine.py:2452-2458 TTS intercept) against the *current* `docs/steering_chelation_rag_dag_research/artifacts/shim_node.py` contract (apply_shim_cascade:488-554, get_cascade visited-set insert-once:464-471, provenance stamping + _stable_hash:301-321+725-728, ShimCascadeApplication copy safety).

**Output**: New dedicated artifact created at `loop_01/03_sip_hook_candidates.md` (per task slice for this cycle). Contains:
- BHS EVIDENCE block with actual grep invocations + direct read citations performed 2026-05-26.
- 2-3 concrete minimal pseudocode SIP hook sketches (research-only, explicitly L4-scaffolded, designed to be consumable by Agent B/C for harness-only smoke without mutating any prod source files).
- Brutal honesty on gaps blocking runtime evidence (zero SIPs wired; apply_shim_cascade only smokes its own demo; FeatureDirectionBank seed logic matches register_seeded but zero bridge code).

This note is the *only* change to this Cycle 1 audit file. No other edits. Full analysis + pseudocode live in the 03_ candidate doc.

**BHS evidence for this note**: 
- Grep (via tool): `grep -n "class VectorSteerer|def steer|...` on tts_pipeline.py (21 matches, exact lines 27,33,47,83,216+).
- Grep (via tool): pattern for Shim* symbols restricted to *.py → only 2 files under docs/.../artifacts/ (0 in tts_pipeline.py, 0 in antigravity_engine.py, 0 in feature_direction_bank.py).
- Direct reads: shim_node.py:155-188 (ShimCascadeApplication doc), 488-554 (impl), 442-483 (get_cascade), tts:47-80,216-222, antigravity:2452-2458 (cited above), feature_direction_bank.py:54-70 (exact seed match to shim register_seeded).
- All per BHS_5MIN_SHIM_LOOP_GOAL.md:18-29 (evidence rule) + brutal-honesty-rulebook.md v3.3.

*End of Cycle 2 note. See 03_sip_hook_candidates.md for the hooks and cross-ref.*

---

*End of 02_vectorsteerer_sip_audit.md. Drive the loop. Produce evidence. Be brutally honest.*