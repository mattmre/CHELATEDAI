# SEAL + EGGROLL Multi-Panel Architecture Review

## Verdict

The best ChelatedAI outcome is not autonomous base-model mutation. It is a gated, adapter-only self-healing loop:

```text
runtime diagnostics + new context
  -> SEAL-style self-edit directives and eval probes
  -> sandboxed adapter-only candidate execution
  -> retrieval-native reward, retention replay, quantization survival, structural health, latency gates
  -> ReSTEM-style positive filtering
  -> candidate ledger / route-limited shadow replay
  -> road-course promotion only after repeatability and transfer
```

SEAL contributes the self-edit control loop. EGGROLL contributes low-rank, scalar-fitness, quantization-aware candidate search. ChelatedAI contributes retrieval fitness, structural diagnostics, quantization gates, adapter routing, safety testbeds, road-course evidence, and storage-scope boundaries.

## Panel consensus

| Panel question | Consensus |
|---|---|
| Is the first `self_healing_chelation.py` implementation enough? | It was safe scaffolding, but shallow proof. It generated advisory directives without a deeper execution loop. |
| What should be built now? | Candidate ledger, self-generated eval probes, sandbox execution, iterative validation loops, real retrieval-based live-fire rewards, and explicit claim boundaries. |
| Should base model weights mutate? | No. Keep base embedding/LLM weights frozen. Promote only adapter checkpoints/routes after gates. |
| Where does EGGROLL fit? | Adapter-only low-rank ES, retrieval-native scalar fitness, deterministic seed replay, quantization-aware candidate survival. |
| Where does storage fit? | Sharded ANN, candidate metadata, deterministic seed/config replay, scalar local scoring, and latency-aware routing. Not full SSD-hosted LLM execution. |

## Implementation changes from the deeper review

- `SelfGeneratedEvalProbe`: deterministic SEAL-style knowledge and retention probes with negative/confuser terms.
- `CandidateProvenanceLedger`: append-only candidate records with context hash, diagnostics hash, directive hash, metrics, gate reasons, and safety metadata.
- `SelfEditSandboxExecutor`: isolated directive execution wrapper that records adapter-only mutation scope and artifacts.
- `SelfHealingChelationPlanner.execute_shadow_round()`: runs directives through the sandbox and reuses the same reward gates.
- `SelfHealingChelationPlanner.run_adaptive_validation_loop()`: repeats sandbox/gate rounds and records debug actions when candidates fail.
- `run_live_fire_diagnostics.py`: no longer assigns fixed self-edit rewards. It builds self-generated probes, evaluates them with `RetrievalFitnessEvaluator.evaluate_engine()`, and logs a two-round adaptive validation loop.

## Cross-correlation matrix

| SEAL / EGGROLL concept | Existing repo capability | Deeper wiring now added or still needed |
|---|---|---|
| SEAL self-edit directives | `SelfEditDirective` | Added sandbox execution and iterative validation loop; model-generated policy remains future work. |
| Synthetic implications / QA | deterministic synthetic examples | Added self-generated eval probes; future work is full qrel/hard-negative generation from arbitrary documents. |
| Inner TTT/SFT update | `online_updater.py`, sedimentation trainer, adapter factory | Sandbox executor now provides the execution seam; real cloned-adapter training remains the next implementation layer. |
| Outer reward | retrieval fitness, composition orchestrator, adaptive gates | Live-fire now uses actual retrieval fitness over generated probes instead of fixed reward deltas. |
| ReSTEM filtering | positive reward filter | Extended with ledger, retention gate, quantization gate, and round-level debug actions. |
| Catastrophic forgetting | retention metric key and checkpoint rollback | Retention probes and ledger are added; broad replay benchmark remains required before persistence. |
| EGGROLL low-rank ES | `evolution_strategies_optimizer.py` | Existing optimizer remains the candidate executor target; future work should run retrieval-native ES in road-course campaigns. |
| EGGROLL quantized viability | quantization simulation/gate | Self-healing candidates preserve quantization-gate details in ledger entries. |
| Storage-node thesis | computational-storage POC and mock evaluators | Boundary remains scoped to near-data ANN/scoring and evidence transport, not GPU replacement. |

## Remaining implementation sequence

1. Implement cloned-adapter candidate execution for `adapter_sft`, `online_contrastive_update`, and `eggroll_es`.
2. Feed accepted shadow candidates into a route-limited adapter bank instead of global defaults.
3. Run multi-seed BEIR/multitask road-course campaigns comparing baseline, Adam, ES, and SEAL-guided ES.
4. Add real retention replay qrels and hard-negative probes before enabling persistent adapter updates.
5. Keep storage work focused on sharded ANN/scoring and calibrated device profiles.

## Claim boundary

Implemented now:

- advisory SEAL/EGGROLL planning
- candidate provenance
- self-generated eval probes
- sandbox execution seam
- adaptive validation loops
- real retrieval-fitness live-fire scoring for self-healing probes

Not claimed yet:

- learned SEAL self-edit policy
- broad retrieval lift
- safe persistent self-adaptation
- storage-resident LLM training/inference
- default promotion beyond the already road-course-supported `0.01` chelation threshold guardrail
