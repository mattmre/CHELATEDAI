# Shim Nodes, Compounding Cascades, and MTP Lookahead
## Formal Nomenclature and Integration Specification for the Steering-Chelation-RAGDAG-MicroSLM Program

**Status**: Program primitive definition (Loop 1 input / Loop 2 architecture driver)  
**Date**: 2026-05 (program kickoff)  
**Cross-references**: 
- `STEERING_CHELATION_RAGDAG_MICROSLM_RESEARCH_PLAN.md`
- `feature_direction_bank.py`, `tts_pipeline.py` (VectorSteerer, SteeringSignal)
- `model_scope_features.py`, `steering_policy.py`
- `self_healing_chelation.py` (SelfEditDirective)
- `computational_storage_poc/` (block_graph, repo_graph_memory, mock_array speculative racing)
- `docs/llm-architecture-ai-engineering-adaptation-review-2026-04-27.md` (MTP section, arXiv:2404.19737)
- OPSD Loop 01 artifacts (asymmetric privileged distillation, KL control for self-correction)

---

## 1. Core Thesis Extension

The existing steering node and chelation machinery provides **point corrections** and **reroutes** in embedding / feature / DAG space. 

**Shim Nodes** introduce **structured, composable, usage-refined directional overrides** that function as:
- Low-compute "leveling" adjustments (analogous to a physical shim under a cabinet leg).
- Blank-space activation points that unlock higher-order computation or knowledge access without dense weight mutation.
- Backdoors that the RAG-DAG + usage telemetry progressively optimizes for minimal token cost and maximal relevance.

When combined with **MTP Shim Lookahead**, hitting one shim can automatically surface and pre-activate a small set of high-utility compounding shims, turning isolated corrections into efficient, learned reasoning cascades.

This is not another adapter or reranker. It is a new **node type and activation discipline** inside the SE-RDAG (Shim-Enabled RerouteDAG).

---

## 2. Primary Nomenclature (Canonical Terms)

### 2.1 Foundational Primitives

**Shim Vector (SV)**  
A unit-norm (or bounded-norm) directional vector in a chosen space (embedding, residual stream, sparse feature, or block-graph coordinate).  
- **Insertion effect**: Added (or multiplicatively gated) exactly once at the activation point. Subsequent passes see the adjusted state unless explicitly reset.
- **Distinction from SteeringSignal**: A SteeringSignal is ephemeral and additive per inference step. A Shim Vector is **registered**, **versioned**, and **cascadable**.
- Stored in the **Shim Registry** (extension of `FeatureDirectionBank`).

**Shim Node (SN)**  
A first-class, addressable node in the RerouteDAG (or its Model-Scope / computational-storage projection).  
- Acts as a "blank space" placeholder.
- Carries one or more Shim Vectors + metadata: activation conditions, tier/order, known cascades, usage statistics, provenance.
- When a decision surface (chelation variance threshold, MTP prediction, explicit steering policy, or SelfEditDirective) selects the node, the shim is **inserted** and its effect applied.
- Types:
  - **Static Shim Node (SSN)**: Precomputed, immutable in a given release.
  - **Dynamic Shim Node (DSN)**: Weights / vectors updated via online refinement or OPSD-style distillation.

**Shim Insertion Point (SIP)**  
A hook or decision surface where Shim Nodes may be activated. Primary locations:
- Post-embedding in `AntigravityEngine` (chelation decision path).
- Inside `VectorSteerer.steer()` or extended `ModelScopeShadowSteerer`.
- At RerouteDAG node expansion time (before or after retrieval).
- Inside micro-SLM route policy forward pass.
- Block-graph payload dispatch points (for drive-node shims).

### 2.2 Composition and Escalation

**Shim Cascade (SC) / Compounding Shim**  
A directed activation chain or tree: SN₀ → SN₁ → SN₂ … where activation of one triggers (via policy or MTP lookahead) one or more dependents.  
- Enables **tiered escalation**:
  - Order-0: Direct vector adjustment (classic chelation-like correction).
  - Order-1: Simple shim insertion for focus reroute or knowledge backdoor.
  - Order-k (k≥2): Meta-shims that operate on other shims, propose new DAG topology, or trigger higher-order reasoning subgraphs.
- **Compounding** occurs when the output state of SNᵢ becomes the input context for SNᵢ₊₁ (vector-to-vector or shim-to-shim).

**Shim Tier / Order (ST-k)**  
The escalation level of a Shim Node. Higher k implies greater abstraction or computational extension (more expensive but higher potential reasoning power). The micro-SLM or steering policy learns to select the minimal sufficient tier.

**Precomputed Shim (PCS)**  
A Shim Vector or small sub-cascade that has been materialized offline (via EGGROLL population search, distillation from larger teacher, or successful historical cascades) and stored for O(1) or near-O(1) lookup + insertion.  
- Primary value: Provides compact, high-fidelity regression in representation space **without** forced quantization, dimension halving, or edge-case distortion of the base model.
- Can live in the Shim Registry, block-graph payloads (computational storage), or a dedicated shim cache.

### 2.3 Learning, Lookahead, and Refinement

**MTP Shim Lookahead (MSL)**  
Application of Multi-Token Prediction (MTP) principles (arXiv:2404.19737 and related speculative decoding work already referenced in the repo) to the shim layer.  
- When a Shim Node is activated (or strongly predicted), the MTP-style head (in the micro-SLM route policy or a dedicated lightweight lookahead head) predicts the most likely next 1–N Shim Nodes that should be pre-fetched or pre-inserted.
- "If this shim is engaged for this class of query / DAG state, these related shims have high historical utility."
- Enables **speculative shim activation** analogous to speculative token decoding, but at the level of reasoning primitives.

**Usage-Refined Shim (URS)**  
A Dynamic Shim Node whose parameters, cascade partners, and activation priority are continuously updated by the RAG-DAG telemetry:
- Activation count
- Success rate (downstream fitness / route cohesion lift)
- Token cost delta (including cascade cost)
- Compounding frequency with other shims
After sufficient usage, high-utility URS become **Shim Backdoors**.

**Shim Backdoor**  
An emergent, low-token-cost, high-precision pathway (via one or a short cascade of shims) from a common activation context to a semantically distant but relevant region of the knowledge manifold or stored corpus.  
- The RAG-DAG "learns" these over months of heavy use.
- Goal: Minimize total token usage for recurring reasoning patterns by turning expensive retrieval + reasoning into "shim + minimal verification" operations.
- Tracked in the **Shim Utility Ledger** (extension of existing provenance / fitness ledgers).

**Shim Registry (SR)**  
The canonical store and lookup service for all Shim Nodes / Vectors (static + dynamic).  
- API: `register(shim_id, vectors, metadata)`, `lookup_by_context(context_embedding, top_k)`, `get_cascade(shim_id)`, `update_usage(shim_id, outcome)`.
- Can be backed by the vector store, block-graph storage, or a hybrid.
- Supports versioning and safe rollback (critical for BHS gates).

**Shim-Enabled RerouteDAG (SE-RDAG)**  
The evolution of the RerouteDAG in which nodes may be:
- Standard retrieval / reasoning nodes, or
- Shim Nodes (with insertion semantics).
Edges may be annotated with "shim-augmented" or "cascade" labels. Chelation variance signals are first-class triggers for shim consideration.

---

## 3. Integration with Existing Surfaces (Concrete Mapping)

| Existing Component              | How Shims Extend It                                                                 | File / Surface                          |
|--------------------------------|-------------------------------------------------------------------------------------|-----------------------------------------|
| `FeatureDirectionBank`         | Becomes the low-level vector provider for Shim Vectors (Gaussian seeds + SAE overrides) | `feature_direction_bank.py`            |
| `VectorSteerer` / `SteeringSignal` | Extended to support registered Shim Nodes with insertion-once semantics and cascade metadata | `tts_pipeline.py`                      |
| `SelfEditDirective`            | New `shim_directive` variant that proposes insertion, promotion, or deprecation of Shim Nodes | `self_healing_chelation.py`            |
| Model-Scope Steering Policy    | Policies can now select/score Shim Nodes in addition to feature scaling/suppression | `steering_policy.py`, `model_scope_*`  |
| Chelation decision logic       | High local variance or isomer drift can propose "shim insertion" as an action alongside or instead of classic rerank | `antigravity_engine.py`                |
| Block graph / drive nodes      | Shim Vectors and small cascades can be compiled into block-graph payloads for fast speculative dispatch and lookup | `computational_storage_poc/block_graph.py`, `mock_array.py` |
| OPSD / EGGROLL training        | Successful shim cascades become privileged traces for asymmetric distillation; low-rank population search over shim combinations | OPSD Loop 01 patterns + `evolution_strategies_optimizer.py` |
| Synthetic collapse + road-course fixtures | Extended with "shim insertion under noise" tasks and cascade acceptance metrics | Existing benchmark surfaces            |
| Micro-SLM (2-4 GB route policy)| Primary learner of shim selection policy + MTP Shim Lookahead heads                 | New training surface (Loops 6–9)       |

---

## 4. Key Behavioral Properties (Requirements)

1. **Insert-once semantics**: A given Shim Node affects the state only at the moment of insertion unless the policy explicitly re-inserts or chains it.
2. **Compounding without explosion**: Cascades must be bounded (max depth, max fan-out) by policy + budget-aware collection (reuse existing adaptive overlay / verifier card discipline).
3. **Precomputed preference**: When a high-utility Precomputed Shim exists for a context, the system prefers it over on-the-fly generation or heavy distillation.
4. **Usage-driven refinement**: After N activations (configurable), a Dynamic Shim Node must have its utility ledger entry; low-utility shims can be demoted or pruned.
5. **MTP Lookahead is advisory + gated**: Predictions from MTP Shim Lookahead are treated as high-priority candidates for the steering policy / micro-SLM, never as unconditional execution.
6. **Quantization and boundedness**: All Shim Vectors must be compatible with BoundedAdapter / INT8 floors (same discipline as existing correction surfaces).
7. **Rollback and provenance**: Every insertion that affects a result must be recorded with sufficient metadata for replay, rollback, and BHS evidence chains.

---

## 5. Research Implications for the 10-Loop Program

This primitive is large enough to warrant its own workstream inside the existing program (not a separate program).

**Recommended elevation in architecture (Loop 2 target)**:
- Define the `ShimNode` dataclass / protocol + `ShimRegistry` interface.
- Extend RerouteDAG to SE-RDAG with shim node expansion rules.
- Specify the MTP Shim Lookahead head interface (can be a tiny auxiliary head on the micro-SLM or a standalone lightweight model).
- Design the Shim Utility Ledger schema (integrates with existing artifact cards).

**Placement in loops (proposed refinement of the plan)**:
- **Loop 1 (current)**: Treat this nomenclature doc as required reading. Audit how FeatureDirectionBank + VectorSteerer + existing steering policies can host the first Shim Node implementation. Add "shim substrate readiness" to the substrate audit.
- **Loop 2**: Make Shim Nodes + SE-RDAG + Shim Registry a core deliverable of the architecture phase alongside the general reroute policy.
- **Loop 3–4**: Include shim cascade losses and MTP lookahead objectives in the loss family + stability work.
- **Loop 5–6**: Special focus on precomputed vs dynamic shims and quantization survival for cascades.
- **Loop 7**: SelfEditDirective integration now explicitly includes shim proposal / deprecation / cascade editing.
- **Loop 8–9**: New benchmark families: "shim insertion under controlled semantic collapse", "cascade token efficiency vs baseline retrieval depth", "MTP lookahead hit rate on held-out usage traces".
- **Loop 10**: Explicit evaluation of whether shim backdoors + MTP lookahead delivered measurable reduction in total tokens for recurring complex queries while preserving or improving quality and stability.

**BHS Considerations (add to rubric)**:
- Any claim of "token reduction via shim backdoors" requires before/after token accounting on the exact same query set with identical quality gates.
- Cascade depth and fan-out must be reported; unbounded or high-variance cascades are failures.
- Precomputed shims must show they were derived from evidence (not hand-crafted) or carry a "human-authored with audit" flag.

---

## 6. Open Questions (for Loop 1 agents to attack)

1. What is the minimal interface change to `VectorSteerer` and `FeatureDirectionBank` to support registered, versioned, cascadable Shim Nodes with insert-once semantics?
2. How does MTP Shim Lookahead differ in training objective and inference cost from standard next-token or next-feature prediction?
3. Can small cascades of Precomputed Shims serve as a practical alternative (or complement) to aggressive quantization or Matryoshka-style dimension slicing for compact representation?
4. What does a "failed shim cascade" look like in structural health / route cohesion metrics, and how quickly can the system detect and rollback?
5. How do we seed the first useful Shim Nodes without waiting for months of organic usage (synthetic cascade generation via teacher models + EGGROLL search)?

---

## 7. Brutal Honesty on This Document

This is a formalization of a promising direction, not a proven mechanism. No code yet implements Shim Nodes or MTP Shim Lookahead under this program. All integration claims are hypotheses grounded in the existing high-quality surfaces (FeatureDirectionBank, TTS steering, OPSD patterns, block graphs). Promotion of any shim-related pattern will require the full BHS evidence chain defined in the program rubric, including token-accounted quality-preserving efficiency gains on held-out workloads.

**Next action recommendation**: Incorporate this nomenclature as required context for all remaining Loop 1 agents. Update the master synthesis (`10_master_synthesis_and_prioritization.md`) to treat Shim Nodes + MTP Lookahead as one of the highest-leverage new primitives for the SE-RDAG architecture in Loop 2.

---

*This document converts the raw directional-shim + MTP-backdoor intuition into executable research nomenclature while preserving strict compatibility with the program's BHS standards and existing substrate.*