# W22-15 Preregistration: selective-trigger 0.5B extraction (Suite A + #62/D13)

- **Queue item:** W22-15, "Selective / Fast-Path 0.5B Extraction Unlock for Flagship #2 (D9-A / D10-B / #62 Latency Bottleneck)".
- **Source:** `docs/research/wave22_queue_additions_2026-11-25.json`.
- **Branch and base:** branch `pv7/w23b-w22-15-selective-extractor`, base `2c408925`.
- **Committed:** this file is committed alone, before any code for this item and before any result.
- **Results seen so far:** none. No model has been trained, loaded or run for this item.
- **What has been viewed:** the Suite A corpus text (`prospective_realworld_drift_corpus.py` output) and the recorded D9-A, D10-B, D7 and W22B artifacts. See "Known contamination risk" below.

## 1. Queue criterion (verbatim)

> Advance only if mean e2e write+read latency drops below Bare 70B (<=1.5 s/case target, stretch <500 ms) while preserving Suite A >=95% accuracy (<=1.7 pp drop) and non-zero on-time throughput on a sham-controlled #62 replay.

## 2. Preregistered departures from the queue spec

Each departure below is a reduction or a substitution. Each one is also carried into `report.md`.

| # | Queue spec says | This run does | Why |
|---|---|---|---|
| D1 | "already-trained D9-A 0.5B LoRA extractor" | Retrains a 0.5B LoRA with the same code: `chelated_fast_edge_extractor.train_fast_claim_extractor_lora`, `Qwen/Qwen2.5-0.5B-Instruct`, r=16, alpha=32, 3 epochs, seed 42, splits `build_disjoint_extractor_splits(600, 120, seed=20261123)`. | The D9-A weights are not on this host. The result is labelled **"retrained, not byte-identical to D9-A"**. Recorded D9-A merged sha prefix `4cef79a7`. The retrain's sha is recorded and is expected to differ. Only 0.5B is trained; there is no 0.5B-vs-1.5B DEV selection. |
| D2 | "vLLM/FP8 batched" | bf16 HuggingFace `transformers` on one local RTX 3090 (24 GB). | No vLLM on this host, and no FP8 on Ampere. |
| D3 | Suite A accuracy as recorded for D9-A (Nemotron-30B reader) | A **deterministic rule reader** (§4.3) in both local arms. | The Nemotron reader is not available. The local FULL-vs-SELECTIVE comparison is like-for-like. The comparison with recorded D9-A 0.9667 is **NOT like-for-like** and is reported as context only. |
| D4 | Compare with Bare 70B | Bare 70B is **not run**. Comparison is against recorded means only: `mean_query_ms_k5 = 3257.82` and `mean_query_ms_k17 = 5361.28` (`artifacts/wave11_d10_5m_real_cutover_and_70b_scale/part_b_70b_sweep_summary.json`). | A 70B model does not fit on 24 GB. Recorded Bare 70B latency is read-only query time on different hardware, so the comparison crosses hardware and is not like-for-like. |
| D5 | Swap onto D10-B, D7 and D13 | **D13 (#62)** is executed (CPU stage 1, then live on GPU). **D10-B** (5M-store cutover) and **D7** (Zep/Graphiti arena) are **not re-executed**. They appear only as recorded e2e means on the same Suite A: D10-B chelated 26596.47 ms/case; D7 Chelated_v4 10373.31 ms/case. | Those stacks are not runnable here within budget. This is declared `DEFERRED_SCOPE`. |
| D6 | #62 worker = "D9-A 0.5B LoRA extractor" | The #62 live worker is base `Qwen/Qwen2.5-0.5B-Instruct`, the same model as the prior W22B colab live run. | The D9-A LoRA is trained on the claim-JSON schema (`entity`/`slot`/`value`/`stance`). #62's `LiveVLLMWorkerBackend` prompt asks for `field_value`/`cited_source_id`, so a schema-mismatched adapter would be a confound. |
| D7 | #62 full grid (4 pool sizes x 4 schedules x 2 branches x 5 regimes x 5 system cells) | **Live:** `pool_size=4`, `S1_steady`, branches A and B, regimes `{no_change, sham_subtask_reopen_interrupt, withdrawal_only}`, all 5 system cells = 30 cells x 24 jobs = 720 jobs per arm. **CPU stage 1:** the full default grid. | The reduced live grid is the minimum set from which the runtime computes `sham_reopen_isolation` (it needs `no_change`, `sham`, `withdrawal_only`). |
| D8 | (n/a) | In the selective #62 arm the router acts on **every** reopen call, including the `full_restart_rebuild` comparator cell. That comparator is therefore no longer a forced full re-extraction in the selective arm. | Routing is a backend property. This is disclosed, and the comparator is not used in any gate. |
| D9 | (n/a) | #62 job latency uses the runtime's existing semantics: simulated arrival timeline plus **measured** live service time per job (`chelated_pooled_agent_runtime.PooledAgentCoordinator.run_cell`). It is unchanged here. | The existing harness is reused as is. |
| D10 | Hardware for comparison | Recorded D9-A ran on unrecorded hardware (no hardware field in `part_a_fast_extractor_results.json`). The local FULL arm is the like-for-like latency baseline. | Recorded latencies are not comparable across unknown hardware. |

## 3. What "selective trigger" means here (implemented in full, not stubbed)

Module: `selective_trigger_router.py` (repo root, flat layout). The router never reads gold, regime, case labels, `stale_or_false_values`, or #62 structured `field_value`.

### 3.1 Suite A doc-level router (write path)

A document is **triggered** (sent to the 0.5B extractor and written to the claim store) iff either condition holds:

- **(a) Authority.** Its `source_channel` authority in the pre-existing `chelated_raw_text_exocortex.DEFAULT_CHANNEL_AUTHORITY` table is `>= 3`. The table was not written for this item.
- **(b) Amendment/conflict cue.** Its lower-cased text matches any stem in the frozen cue list `AMENDMENT_CONFLICT_STEMS`. The stems are taken from the stance definitions in `chelated_epistemic_proposer_v2.EXTRACTION_SYSTEM_PROMPT_V2` (SUPERSEDE / DEBUNK_RUMOR / REFUTE):
  `supersed, replac, revision, revise, restat, updated guidance, debunk, refut, denial, denied, deny, counter-filing, counter filing, disput, audit, bulletin, retract, rescind, withdraw, amend, correction`.

Non-triggered documents are **not extracted and not indexed**. They go to an append-only raw archive (an in-memory list in this experiment) that the read path does not query. This is a real functional cost: in the selective arm, non-triggered text is not retrievable. It is disclosed.

**Thread level.** A case (thread) is "amended/conflicting" iff at least one of its docs is triggered. Every Suite A case is constructed to contain an amendment or conflict, so thread-level gating alone is expected to save nothing on Suite A. That expectation is stated now; the measured thread-trigger rate is reported.

If a case has zero triggered docs, the selective arm answers `""` (a fail-closed miss, scored as incorrect).

### 3.2 #62 / D13 reopen router (selective re-extraction)

The `SelectiveTrigger*WorkerBackend` classes wrap the existing recorded-replay or live backend:

- **Initial (non-reopen) subtask:** always executed by the inner backend. The live inner backend makes the LLM call. The draft and the visible candidate doc ids are cached per `(matter_id, field_name)`.
- **Reopen subtask:** re-extracted only if the thread was amended. That is, **re-extract iff** either:
  1. the cached `cited_source_id` is missing or not among the reopen's candidate docs (it was withdrawn or became inadmissible); or
  2. the reopen candidates include a doc absent from the cached candidate set whose text matches a cue stem or whose `authority_tier >= 3`.
  
  Otherwise the cached draft is returned with `live_call=False` and `exec_ms` equal to the measured wall time of the routing decision.
- **Cache scope.** The cache is scoped to a job. `run_cell` processes jobs sequentially, and each job's initial pass overwrites the `(matter, field)` entry before any reopen of that job. A unit test asserts this invariant.
- **What the router reads.** It reads `doc_id`, `text` and `authority_tier`. It never reads `field_value`.

## 4. Measurement

### 4.1 Arms (Suite A, N=60 cases x 17 docs, corpus sha256 `cb6bc0082b5850f4edfea668d1ecb2a73fe6f898b7476e34ed11036cb378765b`)

- **FULL (local control).** Same retrained extractor; all 17 docs per case in one `batch_size=17` generate; all 17 indexed. Same as the D9-A Part A pipeline except for the reader.
- **SELECTIVE.** Router of §3.1; only triggered docs extracted (one batched generate per case) and indexed.

Both arms use:
- `FastEdgeClaimExtractor`, `max_new_tokens=64`, greedy decoding;
- `BAAI/bge-m3` encoding (`max_seq_length=384`, normalized);
- `AuthorityDominatedExocortexV2`, `FastBM25Index`, `_rrf_combine`, `select_context_for_query(top_k=5)`.

Arm order is FULL, then SELECTIVE, sequentially in one process after one untimed warmup generate.

### 4.2 Latency definition (per case, wall clock, `time.perf_counter`, CUDA-synchronized around generate)

`e2e_ms = route_ms + extract_ms + write_ack_ms + read_ms + reader_ms`

- `route_ms`: router decision over the case's 17 docs (0 in FULL).
- `extract_ms`: batched generate wall time for the docs to extract (0 if none).
- `write_ack_ms`: bge-m3 encoding of the indexed doc texts and the `"{entity} {slot}"` strings, plus exocortex build and BM25 build.
- `read_ms`: query encoding, hybrid score and `select_context_for_query`.
- `reader_ms`: rule reader wall time.

The **reported statistic is the arithmetic mean over 60 cases**; p50 and p95 are also reported. `route_ms` and `reader_ms` are expected to be under 1 ms, whereas Bare 70B's recorded time is dominated by 70B generation. This asymmetry is a known part of the comparison and is disclosed (D3, D4).

### 4.3 Deterministic rule reader (identical in both arms)

- If `sel["status"] == "CONTESTED_ABSTAIN"`, then `pred = "ABSTAIN"`.
- Else, among `sel["selected_ids"]`, restrict to events whose `(tenant, entity, slot)` equals `sel["inferred_slot"]` (fall back to all selected if none match). `pred` is the `claim_value` of the event with the maximum `(effective_authority, valid_time, tx_time)`.
- If nothing is selected, `pred = ""`.

Scoring uses `scripts/run_wave6_d4_trained_epistemic_grpo.py::score_d1_prose_prediction`. **Primary accuracy = mean `frozen_d1_match`**, the same field as recorded D9-A `deterministic_frozen_d1_accuracy`. `strict_exact_match` is also reported.

### 4.4 #62 sham-controlled replay

- **Stage 1 (CPU).** `run_pooled_agent_experiment(mode="stage1-controlled-replay")` on the full default grid, run twice: with `RecordedReplayWorkerBackend` (baseline) and with `SelectiveTriggerReplayWorkerBackend`. The recorded backend reads structured `field_value` by existing design. That is the runtime's own deterministic control, not a live result, and it is **not used for the gate**.
- **Live (GPU).** A local OpenAI-compatible micro-batching server built from `MicroBatchedCausalServer` (`scripts/run_colab_l4_w22b_b62_live_vllm_workers.py`, bf16, `max_batch_size=8`). The `hardware_cohort` label is **parametrized** from the local GPU name (`local:<gpu>`), never "colab". Two arms run on the reduced grid of D7:
  - `LiveVLLMWorkerBackend`, the plain live baseline;
  - `SelectiveTriggerLiveWorkerBackend`.
  
  The model is `Qwen/Qwen2.5-0.5B-Instruct`, with `timeout_s=15` and `JOB_SLA_DEADLINE_MS=560` untouched.
- The verifier `scripts/verifiers.d/62_verifier.py` is run on the outputs where applicable, and its result is reported as is.

## 5. Decision rule (frozen)

**ADVANCE iff all of G1, G2 and G3 hold. KILL if any measured gate fails. INCONCLUSIVE if any gate could not be measured** (for example, the GPU never became free); the reason is stated.

- **G1 latency.** SELECTIVE mean `e2e_ms` over 60 cases `<= 1500 ms`. This also implies it is below recorded Bare 70B `mean_query_ms_k5 = 3257.82`. The stretch (`< 500 ms`) is reported and does not gate.
- **G2 accuracy.** SELECTIVE `frozen_d1_match` `>= 57/60` (0.95; this also satisfies `>= 0.9667 - 0.017 = 0.9497`). **AND** SELECTIVE correct count `>=` FULL correct count `- 1` (a 1.7 pp drop of 60 cases is 1.02 cases).
- **G3 #62.** In the **live** selective arm, the sum over the two `sham_subtask_reopen_interrupt` cells of system `ZeroMu+EK5_v3` (branches A and B) of `on_time_cited_correct_jobs` is `> 0`, **AND** `sham_stale_commits == 0` for those cells.
  - The runtime's stricter `sham_reopen_control_valid` gate (it requires every sham job correct) is reported but does not gate.
  - The plain live baseline arm and the conventional system cells are reported for context, since W22-03 found that #62 ties the conventional arms.

Reported but non-gating:
- FULL arm latency and accuracy;
- per-component latency split;
- doc and thread trigger rates;
- router recall on the D9 synthetic DEV split. The cue list was frozen in this file before that number was computed; that recall is "share of DEV docs with `target_stance != ASSERT` that are triggered".

## 6. Known contamination risk (disclosed before results)

The author of the router read Suite A document text before writing this file (`artifacts/pv7_w23b_w22_15/scratch/suite_a_dump.txt`, not committed). The cue stems are taken from the existing extractor prompt's stance definitions. Even so, the overlap with Suite A's templated cue words ("supersedes", "refutes", "revision", "counter-filing") is not independent evidence of generalization. **Any G1 or G2 pass is a pass on a corpus the router was designed with visibility of.** The D9 DEV-split router recall (§5) is the only out-of-corpus check, and it shares a generator family with Suite A.

## 7. Bulk data and evidence

- **Bulk** goes to `artifacts/pv7_w23b_w22_15/scratch/`, which is gitignored via `.gitignore:87 artifacts/**/scratch/` (verified with `git check-ignore`). Bulk means: the seed cache copy, the LoRA checkpoint and merged model, server logs, per-job replays and full manifests.
- **Evidence** is committed outside `scratch/`: `w22_15_results.json` and `report.md`.

No push, no PR, no deletions.
