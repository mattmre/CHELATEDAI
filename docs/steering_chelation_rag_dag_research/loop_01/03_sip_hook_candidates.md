# 03: SIP Hook Candidates — Minimal Pseudocode for Future Wiring (BHS 5-Min Shim Loop, Cycle 2 — Agent A)

**Agent**: Agent A (Research & Mapping)  
**Cycle**: 2 (per BHS_5MIN_SHIM_LOOP_GOAL.md)  
**Date**: 2026-05-26  
**Mandate**: Targeted cross-ref of Cycle 1 seams (02_vectorsteerer_sip_audit.md §2: tts_pipeline.py:47-80, 216-222; antigravity_engine.py:2452-2458) vs current `docs/steering_chelation_rag_dag_research/artifacts/shim_node.py` contract (apply_shim_cascade, insert-once via visited, provenance/input_hash). Produce 2-3 concrete minimal pseudocode hook examples (research only).  
**Governing**: BHS_5MIN_SHIM_LOOP_GOAL.md (explicitly referenced throughout), brutal-honesty-rulebook.md v3.3, shim_node_interface.md §2 (registration ≠ insertion), shim_nodes_mtp_lookahead_nomenclature.md §2.1/4.1, 02_vectorsteerer_sip_audit.md (this extends it), shim_collapse_benchmark_extension.py (harness target for evidence).  
**Output Contract**: Short, file:line-grounded, brutally honest. 0 production code. 0 stubs in tts/antigravity/feature_direction_bank. All claims backed by cited grep + direct reads. Prioritizes slices that can yield runtime EVIDENCE:/SMOKE: in same or next cycle via harness-only paths.

---

## BHS EVIDENCE — Analysis Performed (Grep Commands + Direct Reads Cited)

All evidence captured 2026-05-26 via allowed tools (grep tool + read_file). No terminal `grep` or `rg` used. Fresh checkout semantics respected for citations.

**Grep 1 (seam confirmation in tts_pipeline.py — VectorSteerer + ephemeral contract)**:  
`grep pattern="class VectorSteerer|def steer|def from_sparse_feature_event|class SteeringSignal|clear_signals|add_signal" path=tts_pipeline.py`  
Result (exact): 10 lines — tts_pipeline.py:27 (SteeringSignal), 33 (VectorSteerer), 39 (add_signal), 43 (clear_signals), 47 (steer), 83 (from_sparse_feature_event), 122 (add_signal in bank path), 218+220+222 (clear + rebuild in apply).  
Direct read citations: tts_pipeline.py:47-80 (full steer: sums _signals, clamp at 72-74, returns signals_applied + total_delta_norm), 216-222 (if feature_event: clear_signals(); rebuild from from_sparse...; explicit comment "transient per-inference"), 83-129 (fresh VectorSteerer + FeatureDirectionBank each call; source= only str provenance).

**Grep 2 (TTS intercept seam in host — antigravity_engine.py)**:  
`grep pattern="_tts_pipeline|_tts_result|after_steering|_tts = getattr|tts\.apply" path=antigravity_engine.py -B 2 -A 10` (head-limited)  
Result (exact): intercept at antigravity_engine.py:2452-2458: `_tts = getattr(self, '_tts_pipeline', None); if _tts is not None: _tts_result = _tts.apply(q_vec); ... q_vec = _tts_result.after_steering`. Note: call site passes NO feature_event, NO shim context. Broad except at 2471-2479 (L11-disclosed safety fallback).  
Direct read: antigravity_engine.py:1047 (import VectorSteerer), 1067 (steerer=VectorSteerer...; _tts_pipeline = TTSPipeline...), 2452-2458 (the post-embed intercept), 1088 (get_last_tts_result).

**Grep 3 (zero production shim symbols — isolation proof)**:  
`grep pattern="ShimNode|ShimRegistry|apply_shim_cascade|shim_id|from .*shim_node import|insert_shim|shim_registry" path=. glob="*.py" output_mode="files_with_matches"`  
Result (exact): ONLY 2 files — `docs/steering_chelation_rag_dag_research/artifacts/shim_node.py` and `shim_collapse_benchmark_extension.py`. Zero matches in tts_pipeline.py, antigravity_engine.py, feature_direction_bank.py, steering_policy.py, self_healing_chelation.py, or any other core *.py.  
**This is the BHS EVIDENCE that the substrate audit gap remains 100% architectural: contract lives in research scaffold only.**

**Grep 4 (contract surface in shim_node.py — apply_shim_cascade + insert-once + provenance)**:  
`grep pattern="apply_shim_cascade|ShimCascadeApplication|get_cascade|insert-once|visited set|provenance\["input_hash"|\.provenance" path=docs/steering_chelation_rag_dag_research/artifacts/shim_node.py`  
Result (exact, 34 lines): apply_shim_cascade defined 488-554; delegates to get_cascade; ShimCascadeApplication docstring 155-188 explicitly calls out "insert-once-respecting" + "visited-set insert-once"; get_cascade 442-483 (visited: set at 464, `if sid in visited... continue`, DFS bounded); register provenance stamping 301-321 (input_hash via _stable_hash at 725-728, 16-char); record_activation 559-593; apply does NOT mutate usage (pure read+copy per 524).

**Direct reads (key ranges, all performed)**:  
- shim_node.py:155-188 (ShimCascadeApplication dataclass + EVIDENCE predicates: no dups, independent copies via from_dict roundtrip at 538, composite mean+norm).  
- shim_node.py:488-554 (full apply_shim_cascade impl; BHS EVIDENCE block 509-524).  
- shim_node.py:442-483 (get_cascade: visited prevents re-entry; returns [] on unknown).  
- shim_node.py:253-339 (register + provenance construction + _stable_hash).  
- shim_node.py:731-747 (BHS SELF-ATTESTATION: L4 scaffold; "zero production-path insertion"; explicit "must supply EVIDENCE... affecting a real SIP in tts_pipeline.VectorSteerer").  
- shim_node.py:762-847 (the if __main__ runtime demo: EVIDENCE: + SMOKE: lines for apply_shim_cascade on 4-shim tree; asserts insert-once no-dups, copy safety, unit-norm, composite).  
- feature_direction_bank.py:54-70 (exact `_gaussian_unit_vector` SHA-256(salt+id) + default_rng; identical to shim_node.py:359-369 register_seeded).  
- shim_node_interface.md:63-77 (registration ≠ insertion; SIPs are the application sites; insert-once is per-SIP decision).  
- Also: 02_vectorsteerer_sip_audit.md:28-91 (original seam analysis), 15_shim_concepts_mapping.md:19-24 (ephemeral contrast), BHS_SHIM_LOOP_DASHBOARD.md:32-39 (Cycle 2 Agent A mandate matches this slice), BHS_5MIN_SHIM_LOOP_GOAL.md:18-29 + 95 (primary objective + backlog item 1: "Wire first real minimal SIP... insert-once shim behavior with rollback").

**Runtime smoke of contract itself (pre-existing, not new)**:  
`python docs/steering_chelation_rag_dag_research/artifacts/shim_node.py` (executes the demo at bottom; produces EVIDENCE:/SMOKE: asserting apply_shim_cascade + insert-once on real registry path). This was re-validated in analysis (no change to file).

**Fresh runtime capture performed during this Agent A slice (2026-05-26)**:  
Command: `python /home/mattmre/CHELATEDAI/docs/steering_chelation_rag_dag_research/artifacts/shim_node.py 2>&1 | tail -30`  
Output (last 30 lines, exit 0):  
```
...
--- Executing BHS EVIDENCE assertions (will raise on violation) ---
ALL ASSERTIONS PASSED.

*** RUNTIME EVIDENCE CAPTURED ***
EVIDENCE: apply_shim_cascade (defined in this file) executed on 4-shim
  test case (s0→s1,s2 ; s1→s3). insert-once (no dups), bounded depth,
  independent copies, unit-norm, and composite all verified by direct
  execution of the production code path inside ShimRegistry.
SMOKE: python /home/mattmre/CHELATEDAI/docs/steering_chelation_rag_dag_research/artifacts/shim_node.py
  (floor-tier for research artifact: import/exec of new helper + assertions)
This is the first concrete runtime evidence for the BHS 5-Min Shim Loop.
=== END OF AGENT B (BUILD) DELIVERABLE ===
Limitations (see final writeup): still L4 research-only; no SIP wired;
  no persistence in this slice (chose a); demo vectors are seeded not
  'precomputed' from real data; 5-min scope respected.
```
This output (plus full run) constitutes additional direct runtime evidence cited in this doc. The demo exercised exactly the contract surface (apply_shim_cascade, get_cascade visited-set, copies, provenance indirectly via construction) that the pseudocode hooks target for future SIPs. No new code was added; this is re-execution of existing research artifact for citation freshness.

**Zero runtime evidence from any SIP on prod paths**: Confirmed by all greps + reads. No before/after on real inference, no usage_stats increments from tts/antigravity, no ledger linkage.

---

## Cross-Reference Matrix: Seams vs Contract (Brutally Honest Gaps)

| Seam Location | Contract Element (shim_node.py) | Current Reality in Seam | Gap / Risk (L-tax) | Potential Hook Surface |
|---------------|---------------------------------|-------------------------|--------------------|------------------------|
| tts_pipeline.py:47-80 (VectorSteerer.steer) | apply_shim_cascade (returns deduped nodes + composite); insert-once via visited in get_cascade; provenance["input_hash"] + usage_stats on nodes | Pure ephemeral sum of _signals (no shim_id, no registry, no visited, no record_activation). Clamp logic (72-74) matches contract "caller clamps". | 0 shim awareness. Every inference rebuilds (216-222 clear). No provenance per insertion. L4 (if ever claimed integrated) + L9 (rollback impossible). | Extend steer() or add parallel path: `if registry: cascade = registry.apply_shim_cascade(...)` then sum selected node vectors (research-only subclass). |
| tts_pipeline.py:216-222 (TTSPipeline.apply, feature_event path) + 83-129 (from_sparse) | ShimVectorProvider + register_seeded (exact seed match to FeatureDirectionBank) | Clears + rebuilds fresh VectorSteerer from bank every feature_event. source=str only. Bank is perfect provider candidate but zero wiring. | Transient contract codified in comment. No insert-once possible without external state owned by SIP. L2 escape conditional risk if "if shim_registry" guard added naively. | Hook before/after clear: registry-backed signals via provider; local per-inference _inserted_shims set (SIP owns the "once"). |
| antigravity_engine.py:2452-2458 (TTS intercept in inference) + variance paths (~2566) | lookup_by_context + get_cascade + ShimCascadeApplication for proposal surfaces | Hardcoded _tts.apply(q_vec) with zero context passed; result.after_steering blindly accepted. Variance does multiplicative masks only. | No shim proposal surface. Broad except (2471) swallows. No variance→shim_candidates. L11 + L5 (tests may assert old meta shape). | Post-embed: `if variance > thresh: cands = reg.lookup_by_context(q_vec); policy["shim_candidates"] = ...` (no auto-insert). |
| feature_direction_bank.py:32-70 (get_direction + _gaussian) | ShimVectorProvider.get_vectors + register_seeded (identical SHA-256 + rng) | Standalone; used only by tts from_sparse. No Protocol impl, no set_vector_provider. | Bridge exists in math only. Grep: zero calls between bank and any shim artifact. L4 (scaffold). | Thin adapter: class BankShimProvider: def get_vectors(self, sid): return [bank.get_direction(sid)] |

**Key architectural mismatch (from interface.md:63-77 + shim_node.py:223-226)**: Registration/apply_shim_cascade is *advisory* to SIPs. The registry never auto-inserts. Current seams implement the exact "ephemeral" counter-example called out in nomenclature:39 and 02_audit:30. Contract's insert-once is *per-cascade-traversal only*; a real per-inference "once" requires SIP-local state (cleared at same cadence as existing _signals).

**Provenance gap**: shim nodes carry stable 16-char input_hash + created_at. tts steering_meta and antigravity logs carry aggregate delta_norm only. No linkage possible today.

---

## 2-3 Concrete Minimal Pseudocode Hook Examples (Research Only — L4 Scaffolds)

These are **pseudocode sketches only**. They are NOT code to paste into prod. They are designed as "minimal" so a future Agent B (Build) can turn the chosen one into a harness-only extension of `shim_collapse_benchmark_extension.py` (which already runs, emits bhs_evidence dicts, and does rollback_proof) — producing the first *new* runtime EVIDENCE:/SMOKE: from a shim registry path without mutating tts_pipeline.py or antigravity_engine.py source. This directly targets goal:95 (backlog #1) and Cycle 2 dashboard mandate.

**Hook 1: VectorSteerer SIP (core insert-once + cascade apply — highest leverage for later evidence)**  
```python
# RESEARCH PSEUDOCODE — harness-only sketch (extend shim_collapse...py simulate path)
# NEVER import into tts_pipeline.py until BHS promotion + Tier B + EVIDENCE chain.

from typing import Optional, Tuple, Dict, Any
import numpy as np
# from artifacts.shim_node import ShimRegistry, ShimCascadeApplication  # research path only

class ShimAwareVectorSteerer:  # research subclass / mixin, not patch
    def __init__(self, registry: Optional["ShimRegistry"] = None, ...):
        self._registry = registry
        self._inserted_this_pass: set[str] = set()  # SIP-local insert-once (cleared per inference, like _signals)
        ...

    def steer(self, v: np.ndarray, context: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        v = np.asarray(v, dtype=float).copy()
        meta = {"signals_applied": 0, "shims_applied": 0, "total_delta_norm": 0.0, "was_steered": False, "shims": []}

        # Existing ephemeral path unchanged (preserve 100% backward compat in harness tests)
        # ... original _signals sum + clamp ...

        if self._registry is not None:
            # Minimal SIP: use the contract's clean primitive
            # Start from a seed id or context-driven lookup (future MTP would pick)
            start_id = "some_context_seed"  # or from lookup_by_context(context) [0].shim_id
            if start_id and start_id not in self._inserted_this_pass:
                cascade: "ShimCascadeApplication" = self._registry.apply_shim_cascade(
                    start_id, max_depth=2, max_fanout=2, include_composite=True
                )
                if cascade.nodes:
                    # Apply composite (or per-node for tiered) with same clamp discipline as 72-74
                    if cascade.composite_vector is not None:
                        delta = 0.1 * cascade.composite_vector  # strength from policy
                        # clamp logic identical to contract + tts:72-74
                        dnorm = np.linalg.norm(delta)
                        if dnorm > self._max_strength:
                            delta *= (self._max_strength / dnorm)
                        v = v + delta
                        meta["shims_applied"] = len(cascade.cascade_ids)
                        meta["shims"] = [{"id": sid, "provenance_hash": self._registry.get(sid).provenance.get("input_hash") if self._registry.get(sid) else None} for sid in cascade.cascade_ids]
                        for sid in cascade.cascade_ids:
                            self._inserted_this_pass.add(sid)
                            self._registry.record_activation(sid, was_success=True, token_cost_delta=0.0, compounding_used=len(cascade.cascade_ids)>1)
                    # Provenance survives for ledger/rollback in harness output
        return v, meta

    def clear_for_new_inference(self):
        self._inserted_this_pass.clear()
        # ... existing clear_signals ...
```
**Why minimal + evidence-friendly**: Calls the exact public contract (apply_shim_cascade + record_activation + provenance). SIP-local set gives real insert-once across multiple steer calls in one "inference". Harness can assert: before/after delta, usage_stats incremented, input_hash present in meta, rollback by re-running without registry yields original, no mutation of registry nodes.

**Hook 2: Antigravity variance → shim proposal surface (no auto-insert; feeds MTP/policy)**  
```python
# RESEARCH PSEUDOCODE — in a harness wrapper around AntigravityEngine inference simulation only
# (never patch the real 2452 block until full promotion)

def _maybe_propose_shims(q_vec: np.ndarray, registry: Optional["ShimRegistry"], variance: float) -> Dict[str, Any]:
    if registry is None or variance <= ACTIVE_THRESH:
        return {"shim_candidates": []}
    # Direct use of contract lookup (already deterministic, bounded)
    candidates = registry.lookup_by_context(q_vec, top_k=3, min_similarity=0.1)
    return {
        "shim_candidates": [
            {
                "shim_id": c.shim_id,
                "tier": c.tier,
                "sim": float(...),  # cosine
                "provenance_hash": c.provenance.get("input_hash"),
                "cascade_preview": registry.get_cascade(c.shim_id, max_depth=1)[:3],
            }
            for c in candidates
        ],
        "proposal_source": "variance_trigger"
    }
# Later SIP (Hook 1) could consume the proposal and decide insert.
# In harness: assert "shim_candidates" in policy_dict; zero change to q_vec or masks.
```
**Evidence path**: Extend benchmark_extension to emit this in retrieval_policy and compare before/after on synthetic fixture. Zero risk to prod paths.

**Hook 3: FeatureDirectionBank → ShimVectorProvider bridge (enables PCS/precomputed shims immediately)**  
```python
# RESEARCH PSEUDOCODE — thin adapter (can live in harness or new research shim_vector_provider.py)
from typing import List
import numpy as np
# from artifacts.shim_node import ShimVectorProvider
# from feature_direction_bank import FeatureDirectionBank

class FeatureBankShimProvider:  # implements the Protocol exactly
    def __init__(self, bank: "FeatureDirectionBank"):
        self.bank = bank

    def get_vectors(self, shim_id: str) -> List[np.ndarray]:
        vec = self.bank.get_direction(shim_id)  # reuses exact seeded gaussian or override
        return [np.asarray(vec, dtype=float)]  # list-of-arrays contract

# Usage in harness registry setup:
# reg.set_vector_provider(FeatureBankShimProvider(existing_bank))
# Then reg.get_vectors("some_fid") or register_seeded will be compatible.
# BHS EVIDENCE: after update_from_activation on bank, reg.get_vectors(id) matches bitwise.
```
**Why critical**: Exact math identity (confirmed by read of both _gaussian impls). Zero duplication. Enables first "precomputed shims" from real SAE rows without touching VectorSteerer.

---

## Brutal Honesty on This Artifact + Remaining Gaps (Full §4 Disclosure)

**What this slice did NOT do (to avoid L1/L4/L13)**:
- Added 0 lines of executable code anywhere (no search_replace on any .py).
- Produced 0 new runtime EVIDENCE or SMOKE output from any SIP (the hooks are prose pseudocode).
- Did not execute or extend any harness (shim_collapse_benchmark_extension.py remains at its Cycle 1 state; its demo was only read, not re-run for new numbers).
- Did not touch prod files (tts, antigravity, bank) even with comments/flags. No L2 escape conditionals introduced.
- No claim that "shims are closer to production" — the cross-ref proves the gap is unchanged (0 production symbols).
- Scheduler / 5-agent dispatch / 5-min wall not exercised (meta debt from Cycle 1 persists; see dashboard).
- Tier B independence: self-performed (Agent A only); future D auditor must treat this as input.

**What it DID deliver (evidence-backed)**:
- Targeted file:line update to 02_vectorsteerer_sip_audit.md (the note at end with exact pointers).
- New 03_ artifact with exhaustive cited EVIDENCE of the cross-ref.
- 3 pseudocode hooks explicitly scoped for harness-only consumption by later agents this cycle (prioritizing goal §95 item 1 + paths to token-accounted before/after + rollback_proof in synthetic fixture).
- Quantification of exact contract vs seam mismatch at the level of methods and lines.

**Carried Debt surfaced / bounded (for D auditor)**:
- SHIM-CD-08 (new, from this slice): Pseudocode hooks exist only in docs; zero implementation even in research harnesses → risk of L4 if any future cycle presents "SIP design complete" without harness smoke. Severity: important. Mitigation: Agent B/C must convert ≥1 hook to running code in shim_collapse_benchmark_extension.py + emit fresh EVIDENCE:/SMOKE: in same cycle or mark deferred.
- All prior Cycle 1 debt (0 SIPs, 0 production evidence, 5-agent model gaps) unchanged.
- No new L1-L13 introduced in prod (because 0 prod changes).

**Gaps blocking runtime evidence in this/near cycle (honest ceiling)**:
- To get real EVIDENCE this cycle, Agent B must implement one hook *inside the benchmark_extension harness only* (e.g. a `simulate_registered_shim_insertion(registry, fixture)` that calls apply_shim_cascade on real registry, measures delta on synthetic vectors, calls record_activation, emits bhs_evidence with command + before/after + rollback by re-instantiating registry from to_dict). This can produce new SMOKE: `python .../shim_collapse_benchmark_extension.py --family shim-cascade --bhs-evidence` without ever importing into tts/antigravity.
- Ceiling (token-accounted end-to-end on held-out, real engine path): impossible this cycle (and disclosed). Requires future promotion gate.
- MTP lookahead / SelfEditDirective shim_directive: 0 surface in any hook (future).
- Persistence/artifact-card roundtrip for shims in evidence chains: partially present in contract (to_dict) but unexercised against real logs.

**Prioritization for later agents this cycle (to maximize evidence strength)**: Start with Hook 1 or 3 in the benchmark harness. It is the shortest path to "EVIDENCE: registry.apply_shim_cascade + record_activation exercised on fixture with rollback_proof" + quantifiable self-improvement delta (new bhs_evidence entries). This satisfies goal:18-29 success def for a partial cycle.

**BHS_SELF_DRAFT (this research slice)**: 82 (strong file:line discipline + EVIDENCE citations + scoped hooks that avoid scope creep; docked for: no new harness execution numbers produced, self-review only, no scheduler verification).

---

*This document is research mapping + pseudocode only. It survives the "try to disprove" test: every seam citation, contract element, and gap was re-verified against live files via the tools on 2026-05-26. No overclaim.*

**References** (all read/grepped in this slice): BHS_5MIN_SHIM_LOOP_GOAL.md (full), 02_vectorsteerer_sip_audit.md (updated), shim_node.py + interface.md (primary contract), nomenclature.md, 15_shim_concepts_mapping.md, BHS_SHIM_LOOP_DASHBOARD.md (Cycle 2 plan), brutal-honesty-rulebook.md v3.3, feature_direction_bank.py, tts_pipeline.py, antigravity_engine.py, shim_collapse_benchmark_extension.py.

*End of 03_sip_hook_candidates.md. Drive the loop. Produce evidence (harness-first). Be brutally honest.*