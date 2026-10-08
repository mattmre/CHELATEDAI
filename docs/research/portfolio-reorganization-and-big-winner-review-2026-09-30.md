# Portfolio reorganization and big-winner review — 2026-09-30

**Status:** `LOCAL PLANNING UPDATE / NOT AN EXECUTION AUTHORIZATION`.

**Scope:** This document integrates the operator-endorsed 2026-09-30 portfolio review into one ordered plan. It also re-examines earlier candidate ideas, focusing on those that could still be large wins.

**Relationship to other documents:**
- It amends the *order* of work in the [portfolio schedule](research-priorities-and-testing-schedule-2026-09-12.md) and the [innovation queue](innovation-test-queue-2026-09-12/README.md).
- It does not freeze or alter any protocol.
- It does not change a result classification retrospectively.
- It does not open SELECT or REPORT data.
- It does not reserve compute.
- It does not override any block/debt surface.

Nothing was run, downloaded, published, pushed or committed to produce it.

## 1. Summary

### 2026-10-08 integration addendum

The [October 7 mathematical-source review](openai-math-applicability-review-2026-10-07.md) now has [MA0–MA7 tasks and 24 specified cases](innovation-test-queue-2026-09-12/math-findings-integration-2026-10-07.md) attached to existing IQ lanes. Stage 0/1 receive theorem/definition admission and complete numerical receipts; IQ-01 receives prospective amplification descriptors; IQ-02/issue #106 receive local coefficient/state certificates; IQ-03 receives finite-code diagnostics and a bit ledger; IQ-04/05 receive only explicitly selected scoped controls. This supports the current ranking and does not promote holographic composition or reopen closed mechanisms. Cases are not implemented or run. The source/status observations dated September 30 below retain that date and are not new remote-state checks.

- **The mechanism-level ideas are not where this repository leads.** Every closed-loop, annealing, stacking or geometric mechanism tested so far was matched or beaten by an ordinary baseline. The documented record of *how* claims failed is the strongest asset, and this plan treats it as infrastructure.
- **Organizing thesis: pre-execution compatibility.** The surviving leads share one question. Before comparing, composing or promoting a representation, can we decide whether the operation is authorized, using only information available before outcomes? And does acting on that decision beat the best fixed ordinary action? See §3.
- **Four candidates can still be big:**
  1. typed lineage authorization in the encoder-upgrade arena;
  2. ordered composition-harm prediction (IQ-01);
  3. temporal evidence correction (IQ-04);
  4. the executable audit discipline itself.

  Everything else is parked, folded into one of these, or kept as a bounded side item (§5).
- **The validity reconciliation is larger than previously stated.** The metric-lineage repair never reached `main`; details are in §2. Stage 0 (§6) is therefore a reconciliation with `main`, not only a local debt sprint.
- **Operator decision recorded 2026-09-30:**
  - The prime-ring float-FFT tie contract will be **fixed**, not dropped.
  - Work waits for the operator's go.
  - The "retain abstract theorem scope only" fallback in `task_plan.md` is no longer the intended closeout.

## 2. Verified state corrections (checked 2026-09-30)

These were checked against GitHub and `origin/main` (`2bac5d3f`) from this checkout:
- branch `codex/prime-ring-onion-method-dev`;
- HEAD `65c9085c`;
- 39 commits ahead of `origin/main` and 105 behind.

The checkout's planning surfaces predate several of these changes.

| Item | What this branch's planning surfaces say | Verified now | Consequence |
| --- | --- | --- | --- |
| Metric lineage on `main` | `CD-MLR-01` open and expired. The quarantine index, validator and banners exist. | `main` still uses the retrieved-list ideal ranking: `benchmark_utils.ndcg_at_k` sorts only the retrieved relevance vector. `drift_recovery_metrics.ndcg_at_k` passes retrieved relevance only. `main`'s swap results document has **no** `LEGACY_METRIC_LINEAGE_BLOCKED` banner. The repair commits `4d49d9ce`, `835f6199` and `f2e41d42` exist on remote branch `codex/pr292-metric-lineage-recondition` but are not ancestors of `main`. | The defect is live on `main`, and affected numbers are presented there without the quarantine notice. This is the first item of Stage 0. |
| PR [#292](https://github.com/mattmre/CHELATEDAI/pull/292) | "Public PR #292 remains at `eb750958`" | **MERGED** 2026-09-22 at head `f6d52580`, a rebase that did not carry the three quarantine commits | The H2/H4/H5 verdicts are on `main` without their metric-lineage quarantine. |
| Block flag | `BLOCKED` (`CD-MLR-01`, `CD-R13-01`, `CD-R16-01`) | `main`'s `docs/next-session.md` says `CLEAR`. Its table has none of the three rows. | The two surfaces disagree. The operator must decide how `main` records the expired debt; see §7. |
| PR [#293](https://github.com/mattmre/CHELATEDAI/pull/293) / `CD-R13-01` | Rejected at iteration six; withdraw, do not repair again; preserve `454e4a32` | **MERGED** 2026-09-23 (merge `782ab62d`, head `56896e13`). The head includes later commits "make disintegration callbacks transactional", "close disintegration state aliases" and "reject protection laundered across prune callbacks". `454e4a32` is not an ancestor of `main`. | Whether the merged code survives the iteration-six probes is **unknown**: BaseException partial mutation, concurrent lost updates, and forged or aliased provenance. It needs a re-probe, not an assumption either way. |
| PR [#295](https://github.com/mattmre/CHELATEDAI/pull/295) / `CD-R16-01` | Withdraw. Preserve the four consumed lock/REPORT artifacts. | **OPEN** at `d2393942`, title "FAIL-CLOSED on both arenas" | It still needs the operator's close action under `DS-R16-01`. |
| PR [#308](https://github.com/mattmre/CHELATEDAI/pull/308) (EGV recommissioning preregistration) | Open draft at `8ce7feb9` ([portfolio schedule](research-priorities-and-testing-schedule-2026-09-12.md), EGV section) | **MERGED** 2026-09-23 at head `25be7944` | Whether the merged text matches the amended local design is **unverified**. Re-check before any EGV-v2 admission. |
| Flagship foundations | `representation_space.py`, `semantic_cache_resolver.py`, the [latent option-value audit](latent-option-value-audit-2026-07.md) and the BCC-1 pack builder (corrected metric and `_fit_ridge`) are tracked on this branch | None of these four is on `main` | The leading candidate's code foundations exist only on a divergent branch. Porting them is a Stage-0 decision. |
| Encoder-upgrade linear-map comparison | Cited in the 2026-09-30 review as a simple linear map recovering most of the gap | It came from local, uncommitted research code that could not be located in any current worktree today. A ridge fit exists in the repo at `scripts/build_bcc1_method_dev_pack.py:408`. | This is not repository evidence. It also falls under the same metric quarantine. Rebuild it inside the baseline ladder (§4.3, M1) before using it. |

## 3. Organizing thesis: pre-execution compatibility

The latent option-value audit established the following ([§5](latent-option-value-audit-2026-07.md)):
- ex-post headroom is not deployability;
- a metric label is not metric identity;
- an oracle choice is not a policy.

The remaining ideas worth funding are all the same type of question. Each asks about an observable fact that is fixed before any outcome is seen.

| Compatibility question | Pre-outcome observable | Existing lane or evidence |
| --- | --- | --- |
| May a query from encoder B be scored against a vector from encoder A? | Encoder lineage, dimension, normalization | Encoder-upgrade (A4) arena; [TRA-1](latent-option-value-audit-2026-07.md) §4.9 |
| May a domain-specialized route serve this query? | Home versus cross-domain correspondence | Rung-16 Arena B, post hoc: home-domain routes +0.0257, cross-domain −0.0309 (audit §4.8; metric-quarantined, fail-closed) |
| May a lower-precision tier serve this vector? | dtype / quantization tier | Rung 16 name; issue #106 "quantization-tier calibration" |
| May adapter B be applied after adapter A? | Declared operator order, singleton effects | IQ-01 / LIR / SRS |
| May this evidence item still support a claim at query time? | Valid-time and transaction-time history, retraction state | IQ-04 / EK7 |

Shared machinery:
- the typed descriptor `S = (role, lineage, compatibility domain, metric, dimension, normalization, dtype)`;
- the corrected metric contract;
- the executable baseline ladder;
- the gap preflight.

This thesis is an organizing hypothesis, not a result. Every row has substantial prior art (§5 and Sources). A contribution would be a typed, fail-closed authorization contract that measurably beats the best fixed action under held-out mutation families. It would not be the compatibility idea itself.

## 4. Integrated recommendations

### 4.1 Ranked differentiated work

1. **Executable audit discipline.**
   - The record of the IDCG lineage defect, the estimator that inverted with sample size, the gate no method could win and the degenerate oracle is unusually complete.
   - Its in-repository form is M1–M2 below plus the metric contract.
   - Any publication planning is tracked outside this repository.
2. **Typed lineage authorization in the encoder-upgrade arena (TRA-1).**
   - This is the only regime with a large measured gap: frozen `0.818737 → 0.006394`, re-embed oracle `0.813164` on SciFact.
   - Those values are legacy metric lineage; the qualitative collapse is large enough that it *may* survive regeneration, but that is a prediction, not evidence.
   - Learned compatibility itself is prior art. The open part is the authorization contract plus a measured advantage over the best fixed action.
3. **IQ-01 ordered composition harm.**
   - Best science per unit of time in the queue. Its novelty is narrowed by SimMerge to ordered sequential-chain harm.
   - The protocol's LIR 10% and SRS 5% gates are preserved.
4. **IQ-04 temporal evidence correction.**
   - The practical flagship, against ordinary bitemporal and copy-aware baselines.
   - The EK7 gate is unchanged.
5. **Contrastive SAE steering with the ISI1 paired evaluator.**
   - A real but prior-art positive: contrast 0.604, KL 0.00047.
   - Use it only as a cheap real-model testbed and as a mandatory "stable because dead" gate.
   - Claims about feature identity need multi-seed SAEs.

### 4.2 Stop list (no further investment without a new mechanism and protocol)

**Closed by this repository's own evidence:**
- the chelation corrector, on both the home-turf and the encoder-upgrade side;
- the `oracle_margin_mean` recoverability predictor;
- the H5 living bank;
- H4 compound cycles;
- the RB-15 factorial;
- RB-13 Wave 0A / BIL1 / SPU0;
- VAR1;
- the current BCC-1 family;
- the EGV-v1 affected tasks (diagnostic only);
- JO1 stacking;
- the D2 β = 0.10 arm.

**Kept as design vocabulary, retired as claims:**
- issue #106's "novel contributions";
- the dual-hemisphere framing.

**Reduction memos only:** IQ-05 (wave/diffusion) and IQ-06 (fixed-graph congruence).

**Salvage only the narrow pieces named in its packet:** IQ-07.

### 4.3 New methods, each with a kill test

| ID | Method | Kill test |
| --- | --- | --- |
| M1 | **Executable baseline ladder.** The harness computes the ordinary rungs on the same split and metric hash, and refuses to emit a candidate-method row without them. Default A4 ladder, cheapest first: (a) identity / frozen (C0); (b) keep serving the old query encoder, *if the arena permits*; (c) orthogonal Procrustes map; (d) ridge map (reuse the repository ridge implementation); (e) budgeted partial re-embedding ("backfill") at matched encode budget, ordered as in FastFill; (f) map plus partial backfill; (g) full re-embed (C2O), a costed reference. | Remove any rung row from a fixture: the candidate row must fail closed. The shared contract's ordinary-control rule ([shared-test-contract.md:46](innovation-test-queue-2026-09-12/shared-test-contract.md)) is prose today. M1 closes that soft-prose-claimed-as-mechanical (L13) gap. |
| M2 | **Gap preflight before admission.** On DEVELOPMENT units only, compute (best costed reference) − (best ladder rung). Admit a campaign only if the point estimate meets the declared practical-effect threshold δ and the paired-bootstrap 95% lower bound is above zero. The function refuses SELECT/REPORT inputs. | Replay H5 (C5 == C5s) and the home-turf corrector preflight on their development data: M2 must return `REFUSE` for both. |
| M3 | **Encoder-upgrade residual gap.** A candidate must improve the (recovery, re-encoding cost) Pareto frontier formed by rungs (c)–(g), not merely beat C0. FastFill and Drift-Adapter are mandatory comparators. | If the frontier of (c)–(f) reaches the full-re-embed reference within δ at every budget, close the science question. Keep TRA-1 as safety engineering only. |
| M4 | **Lineage-metadata authorization versus query-feature routing.** The candidate decides from typed descriptors, not from query features. | A query-performance-prediction (QPP) router given the same action set must not match or beat it on held-out mutation families. If it does, close M4. |

The arena must state explicitly whether the old encoder remains servable. If it does, rung (b) restores the original space at query-encoding cost, and most of the arena's motivation disappears. Encoder-retirement literature (backward-compatible training, Hot-Refresh) assumes the old model is retired. The assumption has to be declared, not inherited.

### 4.4 Process adjustments

- **One admitted compute campaign at a time.** Read-only analysis and documentation can proceed alongside it.
- **Validity before claims.** No threshold may be chosen from quarantined numbers.
- **Mark novelty statements honestly.** Where novelty rests on a targeted (not exhaustive) search, mark it `UNVERIFIED`, and say "I don't know" instead of "not covered".
- **Inventory before cleanup.** First list the 20 linked worktrees, the untracked root `.codex-*` files and the `build/lib/` copies. Any removal follows the full-path enumeration rule, with the count and total size shown and owner approval.

## 5. Big-winner review of past possibles

"Disposition" is a planning recommendation. It does not create or close any debt row.

| # | Candidate (origin) | If it works | Evidence today | Main risk / prior art | Cheapest decisive test | Disposition |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | **Typed lineage authorization for mixed-lineage indexes** (audit §4.9, §7 item 2) | A fail-closed serving contract: no cross-space scoring without an authorized bridge, and per-lineage choice of map, backfill or re-embed | Descriptor, resolver and abstention foundations exist on this branch only. The arena gap is large but metric-quarantined. | BCT/Hot-Refresh, FastFill, Drift-Adapter, vec2vec. Cross-space score fusion resembles federated-search normalization. | After A4 regeneration: M2 on the full M1 ladder; then the audit's bar of zero wrong-space executions and a one-sided 97.5% lower bound > −0.02 versus the best safe fixed action, with whole mutation families held out | **Flagship.** Rows 2–5 are cells inside it. |
| 2 | Encoder-upgrade residual gap / budgeted backfill (2026-09-30 review) | Shows whether anything remains beyond map plus partial backfill | Legacy C3a loop at 0.160864 versus oracle 0.813164 (quarantined). The linear-map comparison is not repository evidence (§2). | FastFill and Drift-Adapter likely cover most of it | M3 | First cell of #1. It decides whether #1 is science or engineering. |
| 3 | Correction-geometry persistence across encoder swaps (2026-06-09 novelty item 3: "most novel, least developed") | Learned corrections survive model upgrades | The original premise is gone: there is no surviving correction worth transporting, because the corrector is dead. The live variant is multi-hop lineage: composed A→B→C maps versus a direct A→C refit versus re-embedding. | Single-hop is prior art. Multi-hop error accumulation: **I don't know** whether it is covered. | Three-encoder chain on the regenerated arena; compare composed and refit maps at matched anchor budget | Mutation family inside #1, not a standalone lane |
| 4 | Domain home/cross authorization (rung-16 Arena B) | A pre-outcome feature that reverses a routing aggregate | Post hoc only; REPORT consumed; fail-closed; metric-quarantined | Mixture-of-Retrievers / per-query routing | New IDs and paths, home/cross authorization frozen before outcomes (audit §7 item 3) | Compatibility-domain field of #1 |
| 5 | Quantization-tier calibration (issue #106; rung 16) | Serve cheaper tiers where predicted loss is bounded | No admissible evidence | AdaWidth, adaptive re-ranking | A dtype-mutation family in #1, with end-to-end latency and bytes measured | dtype field of #1. Otherwise the engineering lane. |
| 6 | **IQ-01 ordered composition harm** | Predict harmful adapter orderings before deployment | Protocol and packet only | SimMerge, PermDoRA, AlphaEdit | C0 timing audit, then C1–C2 bounded METHOD_DEV | **Science track #2** |
| 7 | **IQ-04 temporal evidence correction** | Fewer false or stale promotions under retraction lifecycles | Predecessor mechanism tests only | Bitemporal stores, copy-aware truth discovery | T0 predecessor map, then EK7 gate (≥25% reduction, ≤2 pp clean-recall loss, ≤1.5× p95, ≤2× storage) | **Practical flagship #3** |
| 8 | **Executable audit discipline** | Every future claim is mechanically guarded against the failure classes already observed | Prose rules, the BHS validators, the quarantine validator (branch only) | Verification-gap survey, RECLAIM, DCP, falsifiable publication records | M1 and M2 fail closed on the H5 and corrector replays | **Infrastructure, Stage 1** |
| 9 | PRW-T1R exact-Hamming theorem plus tie contract | A proven decoder with production parity | Theorem retained; direct scorer 336/336; float-FFT 117/336 | A collision with the 1992 Legendre-inner construction is recorded, and the specialist equivalence review is still pending | A declared tie contract (exact/integer rescoring at near-ties, or a tolerance band with a canonical tie-break), plus 336/336 exact/direct/FFT parity | **Bounded fix; operator chose FIX; awaits go.** Not a big winner until the novelty review resolves. |
| 10 | Contrastive SAE steering plus the ISI1 gate | Cheap real-model intervention testbed | Positive on real Qwen-Scope; QSCCI v4 shows no chelation-specific advantage | Contrastive SAE steering, CE-Bench, seed dependence | Multi-seed SAE replication before any feature-identity claim | Testbed and gate only |
| 11 | EGV-v2 | Practical agent-evaluation flagship | v1 unlearnable; amended design local; #308 merged | Not assessed here | Learnable corpus plus blind solvability proof | Alternative to #7; selected by the operator, not both |
| 12 | Rung 15 GNN | Graph structure beats flat pools | JO1 stack identical to flat; G2 static advantage false | Standard graph neural networks | It must beat the flat pool on held-out evidence first | **Park** |
| 13 | Rung 17 disk pool / computational storage | Disk-first serving | Engineering PoC; RP2040 hardware evidence still blocked (no device) | Not scientific | End-to-end bytes, latency and quality | Engineering lane only |
| 14 | Issue #106: event-triggered adaptation, failure-fingerprint routing | Self-correcting retrieval | Event-triggered adaptation is the closed loop that lost; fingerprint routing reduces to QPP routing | QPP, MoR | None proposed | Retire as claims |
| 15 | QSCCI follow-up | Distinct selector behavior | v4 identical features, zero advantage | Contrastive steering | Its own new-fixture gate | Small discriminator only; not a big winner |
| 16 | RHPC / IQ-03 / IQ-07 kernels | Representation benefit | Constructed METHOD_DEV; IQ-07 battery invalid | Resonator networks, linear hyperdimensional codes | Same-information ordinary controls | Contained discriminators (`P4`) |

**Focus set:** #1 (with #2–#5 as cells), #6, #7 and #8. Of these, #1 has the largest possible upside and the cheapest decisive test, because the arena, the descriptors and a repository ridge fit already exist. It also has the highest prior-art risk, so M3 is ordered before any authorization campaign.

## 6. Execution order

These are readiness stages, not dates. Each stage lands as reviewed PRs under the repository rules before the next begins.

### Stage 0 — validity reconciliation (no compute)

| Step | Work | Exit criterion |
| --- | --- | --- |
| 0.1 | Port the metric-lineage repair to `main` from a fresh branch: quarantine banners, the fail-closed validator, the v2 index and the CI gate. Then the shared metric contract and caller migration (repair protocol steps 3–5). | `main` cannot compute a retrieved-list ideal ranking for a multi-positive query without failing closed. Quarantine banners are present on every affected `main` document. The validator runs in CI. |
| 0.2 | Re-probe the merged rung-13 code (`782ab62d`) with the iteration-six probe set | Probe receipts are recorded. Then either `CD-R13-01` closes with evidence, or a revert/fix PR plus a debt row follows. |
| 0.3 | Close PR #295 under `DS-R16-01`, preserving the four consumed artifacts byte-for-byte | Operator action recorded |
| 0.4 | Decide whether to port the flagship foundations (`representation_space.py`, `semantic_cache_resolver.py`, the audit, the BCC-1 pack) to `main` | Operator decision recorded; if yes, reviewed port PRs |
| 0.5 | Truth-sync planning surfaces: #292, #293 and #308 merged; #295 open; the flag on `main` | No planning document contradicts the live PR state |
| 0.6 | Inventory only: worktrees, root `.codex-*` files, `build/lib/` | Full list with sizes delivered; nothing removed |

### Stage 1 — executable discipline (CPU, code and tests)

- **M1** baseline ladder.
- **M2** gap preflight.
- Both must fail closed on the replay fixtures named in §4.3.
- Reuse the repository ridge implementation rather than the lost local code.

### Stage 2 — flagship #1

1. Regenerate the SciFact and NFCorpus encoder-upgrade campaigns under the corrected contract (repair protocol step 6, "swap"). Record `REPRODUCED`, `QUANTITATIVE_CHANGE_NO_DECISION_CHANGE` or `DECISION_CHANGED_REQUIRES_NEW_PROTOCOL`.
2. Declare whether the old encoder is servable. Run M2 and M3 on DEVELOPMENT units.
3. **If M3 closes the gap:**
   - stop the science question;
   - TRA-1 continues only as safety engineering.

   **If it does not close the gap:**
   - freeze the TRA-1 protocol with these held-out mutation families: query swap, document corruption, domain home, domain cross, dtype tier, multi-hop lineage, and wrong-space negatives;
   - pass bar: zero wrong-space executions, and a one-sided 97.5% lower bound > −0.02 versus the best safe fixed action.

### Stage 3 — IQ-01

C0 timing/operator audit, then C1–C2 bounded METHOD_DEV, under the preserved CRSV → LIR → SRS hierarchy.

### Stage 4 — practical flagship

- The operator selects IQ-04 (recommended) or EGV-v2, not both.
- Entry requires the chosen lane's predecessor and solvability gates.

### Bounded side item — prime-ring tie contract

- The operator chose **fix** on 2026-09-30.
- Scope:
  - declare the numerical-tie contract;
  - implement it in the production decoder path;
  - add the 336-case boundary parity test.
- It does not reopen the `DS-PRW-001` campaign or any novelty claim. Starts on the operator's go.
- It can run alongside Stage 0 or Stage 1 as a correctness fix. Whether it is admissible while any block flag stands is the operator's call.

## 7. Operator decisions needed

1. **How `main` records the metric-lineage debt.** Options:
   - a new `CD-MLR-01` row with a fresh TTL;
   - carrying over the expired state, which would set `main` to `BLOCKED`;
   - an explicit rulebook-compliant disposition.
2. **The rung-13 merge.** Re-probe first, or revert first.
3. **Closing PR #295.**
4. **Porting the flagship foundations to `main`**, or keeping the flagship on a research branch.
5. **The encoder-upgrade arena premise.** Is the old encoder retired, or still servable?
6. **The go for the prime-ring tie-contract fix.**
7. **IQ-04 or EGV-v2** as the practical flagship, when Stage 4 is reached.

## 8. What this document does not establish

- It establishes no new experimental result.
- Every number quoted from the drift-recovery, rung-16 or swap lines is legacy metric lineage, and none supports a claim.
- Novelty statements come from targeted searches, not exhaustive clearance.
- The "pre-execution compatibility" thesis is an organizing choice. It is not evidence that any row in §3 will pass.
- The §2 PR states are as of 2026-09-30 and must be rechecked before acting.

## Sources

Local:
- [latent-option-value-audit-2026-07.md](latent-option-value-audit-2026-07.md): §3.1, §4.4, §4.8, §4.9, §5, §7.
- [metric-lineage-repair-protocol-2026-07.md](metric-lineage-repair-protocol-2026-07.md): §4 steps 3–9.
- [research-priorities-and-testing-schedule-2026-09-12.md](research-priorities-and-testing-schedule-2026-09-12.md).
- [innovation-test-queue-2026-09-12/](innovation-test-queue-2026-09-12/README.md).
- [docs/next-session.md](../next-session.md): this branch and `origin/main`.
- `findings.md`: RB-11 at lines 640–657; QSCCI v4.
- `docs/drift-recovery-swap-results-2026-06.md`.
- `docs/ROADMAP_EXECUTION.md`.

External, read at abstract level unless noted elsewhere:
- [FastFill (arXiv 2303.04766)](https://arxiv.org/abs/2303.04766)
- [Drift-Adapter (arXiv 2509.23471)](https://arxiv.org/abs/2509.23471)
- [Hot-Refresh (arXiv 2201.09724)](https://arxiv.org/abs/2201.09724)
- [Bidirectional compatible training (arXiv 2204.13919)](https://arxiv.org/abs/2204.13919)
- [SimMerge (arXiv 2601.09473)](https://arxiv.org/abs/2601.09473v2)
- [Contrastive SAE steering (arXiv 2601.03595)](https://arxiv.org/abs/2601.03595)
- [CE-Bench (arXiv 2509.00691)](https://arxiv.org/abs/2509.00691)
- [SAE seed dependence (arXiv 2606.12138)](https://arxiv.org/abs/2606.12138)
- [QPP for retriever selection (arXiv 2109.10739)](https://arxiv.org/abs/2109.10739)
- [Mixture of Retrievers (arXiv 2506.15862)](https://arxiv.org/abs/2506.15862)
- [AdaWidth (arXiv 2608.23862)](https://arxiv.org/abs/2608.23862)
- [Adaptive re-ranking (arXiv 2606.25249)](https://arxiv.org/abs/2606.25249)
- [Verification-gap survey (arXiv 2608.05179)](https://arxiv.org/abs/2608.05179)
- [RECLAIM (arXiv 2609.28850)](https://arxiv.org/abs/2609.28850)
- [Discovery Certification Protocol (arXiv 2609.09219)](https://arxiv.org/abs/2609.09219)
- [Falsifiable publication records (arXiv 2609.17631)](https://arxiv.org/abs/2609.17631)
