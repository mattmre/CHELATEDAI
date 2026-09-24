# ShimNode + ShimRegistry Interface Contract (Research Artifact v1)

**Status**: Loop 1/2 substrate definition. Research-only.  
**Source**: `shim_node.py` (same directory)  
**Cross-refs**: `shim_nodes_mtp_lookahead_nomenclature.md` (nomenclature), `feature_direction_bank.py`, `tts_pipeline.py` (SteeringSignal/VectorSteerer), `steering_policy.py` (PolicyRegistry), `self_healing_chelation.py` (SelfEditDirective + stable hash), BHS rulebook v3.3.

---

## 1. Public API Surface (Minimal Complete)

### ShimNode (dataclass)
```python
@dataclass
class ShimNode:
    shim_id: str
    vectors: List[np.ndarray]          # unit-norm (or bounded) after registry
    tier: int = 0                      # ST-k (0=direct, >=2=meta/compounding)
    cascade_targets: List[str] = ...
    metadata: Dict[str, Any] = ...
    usage_stats: Dict[str, Any] = ...  # activation_count, success_count, cumulative_token_cost_delta, last_activated_at, compounding_frequency
    provenance: Dict[str, Any] = ...   # created_at, source, input_hash (16-char stable), schema_version, ...
```

- `to_dict()` / `from_dict()` roundtrippable (vectors as lists; numeric tolerance 1e-10).
- Instances are only considered well-formed when constructed by `ShimRegistry.register` (or `from_dict` of a prior valid registry snapshot).

### ShimVectorProvider (Protocol)
```python
@runtime_checkable
class ShimVectorProvider(Protocol):
    def get_vectors(self, shim_id: str) -> List[np.ndarray]: ...
```
- Explicit bridge for `FeatureDirectionBank` wrappers, SAE rows, or learned heads.
- Contract: returns copies; unknown → `[]`; repeated calls stable unless upgrade occurred.

### ShimRegistry (core class)
Constructor:
```python
ShimRegistry(dim: Optional[int] = None, seed_salt: str = "chelated_shim_registry_v1")
```

**Required methods** (per task + nomenclature §2.3):
- `register(shim_id, vectors, tier=0, cascade_targets=None, metadata=None, provenance=None) -> str`
- `get(shim_id) -> Optional[ShimNode]`
- `lookup_by_context(context_embedding: np.ndarray, top_k=5, min_similarity=0.0) -> List[ShimNode]`
- `get_cascade(shim_id, max_depth=3, max_fanout=4) -> List[str]`
- `record_activation(shim_id, was_success=True, token_cost_delta=0.0, compounding_used=False) -> bool`
- `update_from_feedback(shim_id, feedback: Dict[str, Any]) -> bool`

**Additional hygiene** (matching `PolicyRegistry` / `CandidateProvenanceLedger` patterns):
- `register_seeded(shim_id, dim=None, ...)` — deterministic Gaussian identical to `FeatureDirectionBank` logic
- `set_vector_provider(provider: Optional[ShimVectorProvider])`
- `get_vectors(shim_id) -> List[np.ndarray]` — registry-first, provider fallback
- `list_all() -> List[str]`, `count() -> int`
- `to_dict() / from_dict(cls, data)` — full ledger/artifact-card serializable form

All vector operations are deterministic given the salt. All mutating methods return success bool or id; none swallow errors broadly.

---

## 2. Insertion Semantics (Critical Distinction)

**Registration ≠ Insertion**.

- `register(...)` / `ShimRegistry` only makes the node addressable and versioned. It performs normalization, provenance stamping, and usage-ledger initialization.
- Actual **vector application** ("shim insertion") occurs at a **Shim Insertion Point (SIP)** in a different layer:
  - Post-embedding in `AntigravityEngine` chelation path
  - Inside (extended) `VectorSteerer.steer()`
  - RerouteDAG node expansion
  - Micro-SLM route policy forward pass
  - Block-graph dispatch points
- **Insert-once**: A given `ShimNode` affects downstream state only at the moment its SIP decides to apply it (or its cascade). Subsequent inference steps see the adjusted representation **unless** the policy explicitly re-inserts or a compounding cascade re-triggers.
- Cascades (`get_cascade`) are **advisory** data for the policy/MTP lookahead head. The registry does not auto-execute them.
- A `SelfEditDirective` (future `shim_directive` variant) can propose `register`, `update_from_feedback`, or deprecation; the directive is still advisory until gated.

This separation preserves the existing ephemeral `SteeringSignal` model while adding the registered, versioned, usage-refined layer demanded by the nomenclature.

---

## 3. Quantization / Boundedness Contract (Non-Negotiable)

1. **Storage invariant**: Every vector stored in a `ShimNode` (after `register` or `update_from_feedback` vector replacement) satisfies `||v||_2 ∈ [1-ε, 1+ε]` with ε=1e-9 (or an explicitly documented bounded-norm alternative declared in `metadata["norm_contract"]`).
2. **Delta application** (at any SIP): the caller is responsible for the same clamping discipline used by `VectorSteerer` (`tts_pipeline.py:72-74`): total steering delta norm is clamped to `max_strength` (default 0.3 in existing surfaces).
3. **INT8 / BoundedAdapter survival**: Shim vectors must be usable under the same `QuantizationPromotionGate` (used in `self_healing_chelation.py`) and `BoundedAdapter` floors that protect existing correction surfaces. No vector may be promoted whose quantized version produces > tolerance regression on retention/structural-health probes.
4. **Cascade boundedness**: `get_cascade` enforces hard `max_depth` + `max_fanout` at lookup time. Any policy that consumes cascades must additionally apply token-budget / structural-health gates before execution (see program rubric route-cohesion + budget-adjusted-lift metrics).
5. **Rollback provenance**: Every activation that affects a result must be traceable via `usage_stats` + `provenance["input_hash"]` + ledger entries (mirrors `CandidateLedgerEntry` in self_healing_chelation.py). A full evidence chain requires before/after state replay from a serialized registry snapshot.

Violation of any of the above is a promotion blocker under the BHS Research Rubric.

---

## 4. Determinism & Reproducibility Requirements

- All seeded vectors use `hashlib.sha256(salt + shim_id)` exactly as `FeatureDirectionBank._gaussian_unit_vector`.
- `lookup_by_context` + `get_cascade` are pure functions of registry contents + inputs (secondary sort by shim_id for stability).
- `to_dict()` snapshots are sufficient to reconstruct identical behavior via `from_dict` (modulo fresh timestamps on new activations).
- Stable hashing for provenance uses the identical `_stable_hash` + `_json_safe` construction from `self_healing_chelation.py:698-710`.

---

## 5. Current Limitations (Explicit — see Brutal Honesty in shim_node.py)

- No MTP lookahead head.
- No SE-RDAG expansion logic.
- No production SIP wiring.
- `update_from_feedback` vector replacement is present for the upgrade path but not yet exercised by any OPSD/EGGROLL loop in this artifact.
- Cascade traversal is simple DFS; richer priority / learned ordering is future work.
- No built-in persistence (file, vector store, block-graph); callers must use `to_dict` / `from_dict`.
- Quantization survival of cascades is a contract, not an implemented gate inside this module.

These are disclosed L4 items by design. They become lies only if later agents present this artifact as "integrated shims working end-to-end."

---

*This interface document + the accompanying `shim_node.py` constitute the minimal viable first concrete artifact for the shim primitive. Promotion to any runtime surface requires the full evidence chain defined in the 10-loop BHS program.*