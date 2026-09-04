# BHS Research Rubric — Steering Node + Chelation + RAG-DAG + MicroSLM Program

**Baseline**: Inherits and extends `docs/chelation_opsd_research/CHELATION_OPSD_BHS_RESEARCH_RUBRIC.md` (v from 2026-05-15 OPSD swarm). All BHS v3.3 rules from `docs/conventions/brutal-honesty-rulebook.md` and CLAUDE.md apply with no exceptions.

**Program-Specific Severity and Evidence Rules** (additions only):

## Route-Specific Metrics (must appear in every candidate report / artifact card)
- **Reroute Acceptance Rate under Controlled Noise**: % of synthetic or real noisy neighborhoods where the system elects to cast ≥1 reroute and at least one improves the downstream metric vs baseline.
- **Route Cohesion Score**: Topology / isomer-style metric over the *proposed route set* (not just final top-k). Penalizes semantically divergent or high-variance route families.
- **Rollback Success Rate**: After accepting a reroute or micro-SLM update, ability to revert to pre-reroute baseline behavior on replay with zero or bounded regression.
- **Quantization Survival Delta**: NDCG / route acceptance delta when all actuators + micro-SLM head run under the same INT8/Bounded constraints as production.
- **Budget-Adjusted Lift**: Primary metric must be reported both raw and normalized by extra retrieval / steering / ES-population tokens or drive-node dispatches used.

## Micro-SLM Specific Gates (before any training run is considered "evidence")
- Base model compatibility: frozen weights + adapter/steering head only. Any run that mutates the 2-4 GB core without explicit exception + rollback evidence is L4 (partial as complete).
- Legacy case retention: on a fixed set of "old base cases" (SciFact clean, NFCorpus, plus 3-5 curated retrieval facts from prior road-courses), the micro-SLM configuration must not regress below pre-registered tolerance without the actuator being disabled.
- Training data provenance card: every trace used for OPSD-style privileged vs student must carry (source, collapse-severity or route-failure label, success/failure outcome, checksum of prompt+context that produced it).

## Drive-Node / Graph Substrate Evidence
- Any claim involving `computational_storage_poc/` dispatch or block-graph candidate evaluation must include parity evidence between software replay and the dispatch path (existing `test_computational_storage_*` discipline).
- Speculative multi-path claims must report both latency model and correctness parity; "faster in simulation" without parity is not evidence.
- RP2040 or real hardware only for transport contract verification per existing retention policy. Emulation + mock_array is the ceiling for this program unless new hardware evidence is captured.

## Promotion / Rejection Language (mandatory phrasing)
- "Promoted to Tier S candidate": only after full artifact card + replay + holdout + quant gate + BHS_TIER_B = 100.
- "Shows directional promise on X but failed Y gate — retained as guarded research": the honest default for most early loops.
- "Rejected for this tranche — fundamental instability under Z condition": when a pattern repeatedly produces unrecoverable regressions or KL-shock analogs.

## Carried Debt from Prior Sessions (must be re-audited in Loop 1)
- No default chelation profile has survived multi-task confirmation (Sessions 32-34).
- Adaptive overlay / learned gate work is strong on instrumentation but weak on promotion.
- Computational-storage drive-node claims remain scope-locked to transport + software parity.
- Model-Scope steering is in shadow/advisory mode only; no production steering actuator has a full evidence chain yet.

Every Loop N synthesis must contain an explicit "Carried Debt Re-audit" section that says for each prior debt item: "Still open / Partially addressed by <new surface> / Closed by evidence in <artifact>".

**BHS Research Score for this Program**:
- Starts at program kickoff with the honest score of the *connected prior surfaces* (OPSD Loop 01 synthesis was strong; Model-Scope and TTS are instrumented but not yet promoted; EGGROLL mapping is analysis only).
- Each loop must publish an updated program-level BHS Research Score (0-100) with the same 5-iteration / Tier B discipline as PRs where relevant.
- A score < 70 at the end of Loop 5 triggers mandatory scope reduction or pivot review before continuing.

**Trigger Phrases for This Program** (use in every agent dispatch and review):
- "Be brutally honest about the reroute."
- "If I disable the steering node / micro-SLM head, what visible behavior on the DAG changes?"
- "Show me the route cohesion and rollback evidence, not the test."
- "What fraction of the claimed lift survives when we force the same quantization floor and token budget as the baseline?"
- "Is this a new route or just a prettier way to describe an old chelation adapter?"

*This rubric is living. Update it in every loop closeout.*