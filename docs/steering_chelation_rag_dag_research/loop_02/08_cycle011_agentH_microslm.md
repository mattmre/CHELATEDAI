# 08_cycle011_agentH_microslm.md — Agent H (Micro-SLM Policy Sketch) — Cycle-011

**Re-read performed 2026-05-27 12:45 PT (protocol §1 + goal:109 #9 + H role + cycle0400 MicroSLM cites; all verbatim; no drift)**: 
1. BHS_5MIN_SHIM_LOOP_GOAL.md (full: H role verbatim at :56 "Agent H — Micro-SLM Policy Sketch: Draft objectives + synthetic data format for a 2-4GB route-policy head that learns reroutes from chelation + shim activations"; backlog #9 at :108-109 "Incorporate min-max style lightweight block/index scoring as a cheap relevance signal for shim activation and SE-RDAG rerouting (Agent 7 draft...)" + expanded 115-168 with MinMaxBlockRelevanceScorer success criteria :136-141 + risks :150-157 "adding this slice while backlog #1 remains 0% risks further L9/L4"; #10 at :109-110; 10-agent roles :48-58; success def :18-29; 4Qs :108-114; Model Change Log :213-230 "L4/L9/L13 on post-hoc 10-agent narrative vs scheduler 019e669bf1bb reality" + "10-agent model begins with Cycle 009"; §128 :191-195 + recs; SHIM risk :157).
2. 10_AGENT_SAFE_MERGE_COORDINATION_PROTOCOL.md (full + launch record :100-116; 10-agent fidelity :10; re-read mandate :74 "Re-read #3 at HH:MM: goal:109 backlog #9 still highest + 0 SIPs; cycle0400:64 'Human intervention mandatory'"; H launch :110 "H: 019e66f9-9179-7631-865d-3ddd3d308431 (Micro-SLM doc sketch)"; safe order :38-41 "distinct per-agent loop_02/ NN_cycle011_agentX_*.md"; L9/L13 guards :8-9).
3. artifacts/cycle_20260527_0400.md (Cycle-010 reality: "0/10 independent artifacts" :5/23; "goal:109-227 (backlog #9/10 + Model Change Log)" :25/67; "0 substrate" repeated :32/42/64/71; "§128 active. Human intervention mandatory" :65/73; MicroSLM program framing + 010 20/100 + 5-vs-10 L4/L9/L13 :38/64).
4. artifacts/BHS_SHIM_LOOP_DASHBOARD.md (Cycle-010 row :956-993 "25/100... 0 substrate/SIP advance... 10th failure pattern... §128 rec"; header NARRATIVE MODEL CHANGE + program 10/100 flat; prior rows confirm 0 deltas).
5. docs/next-session.md (:22 "BLOCKED — Carried Debt..."; SHIM-CD-01-08 at :61-68 "0 SIPs remain per exhaustive non-docs grep" + "L9 remediation failure" + "Blocking YES" for criticals; count:2 via script).
6. loop_02/ latest (08_cycle010_agent8_bhs_process_gap_audit.md full + 09_cycle009_agent9...; style: absolute paths + "0 SIPs" + L citations + "doc-only" + "does not satisfy").
7. docs/steering_chelation_rag_dag_research/artifacts/shim_node.py (:2 MicroSLM program; L4 guards :34-36 "zero production-path insertion"; usage_stats :163-170 "activation_count, success_count, cumulative_token_cost_delta, compounding_frequency"; ShimNode dataclass).
8. docs/steering_chelation_rag_dag_research/artifacts/shim_collapse_benchmark_extension.py (MinMaxBlockRelevanceScorer class :593- (compute :651+, partition :623+, filter : ; guarded --minmax-blocks demo :1993+; "research/artifacts/ ONLY" :21-26).
9. scripts/check_block_flag.py (:108-109 BLOCKED detection; :275-280 "RESULT: FAIL"; :231 "Carried Debt row count"; semantics per :224).
10. 0-prod verification (grep -r patterns from cycle010 json + synthesis/apply-instructions:10 "0 hits (prod isolation; only research shims with L4 guards at shim_node.py:34-36)"; confirmed exactly 2 research files + comments only in prod seams (antigravity:2585-2601, tts:47-80); no imports).
11. scheduler context (0 tasks per all cycle mds; 5-agent dispatch per goal Model Change Log).
12. STEERING_CHELATION_RAGDAG_MICROSLM_RESEARCH_PLAN.md (:54 "Micro SLM (2-4 GB class) as learned 'route policy head' — inputs now include active shim context + chelation signals + current DAG state..."; :219 context_variance chelation; comparative :204+).
13. shim_nodes_mtp_lookahead_nomenclature.md (:79-84 "MTP Shim Lookahead (MSL)" + usage-refined :85-91; ShimNode tier/cascade).
14. antigravity_engine.py (chelation gate + dim_variances/global_variance :2606-2607 "dim_variances = np.var...; global_variance = np.mean(dim_variances)"; draft SIP comment :2585-2601 "variance decision / chelation gate" + "MinMax as cheap pre-filter signal mirroring existing dim_variances"; :2582+).
**All absolute paths + exact lines re-read via tools before synthesis. Protocol §1 + §5 VR-drift prevention followed. BHS v3.3 + goal contract. No edits to any prior file (pure new doc in loop_02/ only).**

**Agent Role**: H (Micro-SLM Policy Sketch) per goal:56 + protocol:110.  
**Cycle**: 011 (10-agent per goal update; BLOCKED/research-only).  
**Output Constraint**: 1-2 page sketch only. **doc-only / 0 implementation**. L4 bounded (research design note, sketch not capability). Does not satisfy goal success def #1 (no runtime prod/harness evidence; 0 SIPs; 0 substrate deltas). §128 active. Human intervention mandatory.

---

## Micro-SLM Route-Policy Head Sketch (2-4GB Class) — Research Design Note Only

**Motivation (tied to backlog #9 + SE-RDAG + MTP + chelation signals)**:  
Per goal:109 #9 (MinMaxBlockRelevanceScorer cheap signals for shim activation/SE-RDAG rerouting) + nomenclature:79 (MTP Shim Lookahead) + plan:54 (Micro SLM as learned route policy head consuming chelation + shim context) + antigravity:2606 (dim_variances/global_variance as chelation variance trigger at SIP seam :2582) + shim_node:142 (usage_stats for URS refinement). A lightweight 2-4GB policy head (quant-survivable, e.g. via llama.cpp/models/ or distilled) can learn to propose reroutes / shim cascades / precomputed shims using only cheap signals already available or cheap to compute in harness. This compounds with MinMax (harness:593+) as pre-filter + existing StructuralHealthScore / _cosine_scores without replacing them. Bounded to research/artifacts/ + future guarded harness only until #1 SIP + BHS promotion.

**Input Feature Vector (cheap + stats + predictions; all O(1) or O(blocks) after pre-agg)**:  
- Min-max scores + range (relevance variance proxy): per-block min_proj, max_proj, range from MinMaxBlockRelevanceScorer.compute (harness:656- ; floor=0.0078 BoundedAdapter compat; filter_candidates output). EVIDENCE: harness:593-620 class doc + :651 compute impl (pure numpy dot-projections, copy-safe).
- dim_variances + global_variance (chelation variance signal): from antigravity_engine.py:2606-2607 in local_cluster_np (post-retrieval variance/chelation gate :2582+). Scalar + top-k dim stats. EVIDENCE: antigravity:2603-2610 + draft SIP comment :2585-2601 "mirroring existing dim_variances".
- Cascade depth (current + MTP-predicted horizon): integer current depth + vector of predicted next 1-N shim probs (from MTP head per nomenclature:81-83).
- Shim activation/usage stats: activation_count, success_count, cumulative_token_cost_delta, last_activated_at, compounding_frequency (shim_node.py:163-170 in ShimNode.usage_stats; updated via record_shim_activation paths in harness). Per-shim or aggregate top-K.
- MTP predictions: next-shim selection logits/probs + speculative cascade utility (I role / nomenclature:79 "speculative shim activation").
- Optional: query embedding summary stats (norm, entropy) + block count / partition metadata (from partition_blocks :623+).

Vector size target: ~128-512 dims (min-max per block + variance vec truncated + usage embed + MTP top-K). Quant to INT8/4-bit friendly. Pre-agg hooks for block_graph payloads encouraged (plan:56).

**Objective (token reduction + collapse improvement vs baseline)**:  
Primary: maximize expected (baseline_tokens - policy_tokens) + gamma * (post_cascade_collapse_metric - baseline_collapse) while penalizing over-cascade depth and false-positive reroutes (missed utility).  
- Token term: measured via harness synthetic (sip_effect family ~0.7886 noise baseline per 0400 json + prior cycles) + future real engine token acct.
- Collapse improvement: ndcg@3 / recovered / noise_reduction delta vs no-policy baseline on synthetic collapse fixtures (with explicit block partitions per Agent2 Cycle-010 work).
- Reg: KL on usage (prevent forgetting high-utility URS); bounded divergence from MinMax conservative scores (plan:236 min_max_shim_adapt pseudocode clips).
- Training: offline on labeled traces (see below) + on-policy refinement from usage_stats feedback. Asymmetric privileged loss (successful G traces upweighted). Quant-aware + low-rank adapter friendly (BoundedAdapter min/max floors).
- Success proxy (research harness only): >=15-25% relative token reduction + no regression on ndcg/recovered vs ungated baseline on sip/sip_effect families (mirrors goal:136 MinMax target).

**Synthetic Data Format from G Traces (Agent G OPSD / EGGROLL privileged successful cascades)**:  
JSONL records (one per query + decision point; generated from harness simulate_sip_effect / record_shim_activation extensions + future OPSD population traces).  

Example record (synthetic; schema only):
```json
{
  "trace_id": "G-EGGROLL-20260527-abc123",
  "ts": "2026-05-27T...",
  "query_vec": [0.01, -0.03, ...],  # or hash summary
  "chelation_variance": 0.142,       # global_variance from antigravity seam
  "minmax_features": {
    "block_0": {"min_proj": 0.008, "max_proj": 0.47, "range": 0.462},
    "block_1": {"min_proj": 0.12, "max_proj": 0.31, "range": 0.19}
  },
  "dim_variances": [0.003, 0.012, ..., 0.007],  # truncated or stats
  "global_variance": 0.142,
  "current_cascade_depth": 1,
  "usage_stats": {
    "shim_alpha": {"activation_count": 47, "success_count": 39, "cumulative_token_cost_delta": -1850.0, "compounding_frequency": 12},
    "shim_beta": {"activation_count": 9, ...}
  },
  "mtp_pred": {"next_shim_probs": {"shim_gamma": 0.82, "shim_delta": 0.11}, "horizon_utility": 0.67},
  "context": {"block_partition": "round_robin_2", "scout_limit": 8},
  "label_reroute_action": "apply_cascade: [shim_alpha, shim_gamma]",  # or "no_reroute" | "insert_precomputed:PCS-042"
  "outcome": {
    "token_delta_vs_baseline": -142,
    "collapse_improvement": 0.18,   # e.g. noise_reduction delta or ndcg lift
    "success": true,
    "rollback_safe": true,
    "post_shim_ndcg": 0.94
  },
  "cycle_tag": "G-trace-synth-011"
}
```
Generation: Extend harness (Agent G role) to emit labeled successful cascades (positive from OPSD/EGGROLL privileged + synthetic collapse with injected "optimal" shims). Negative samples: random or high-variance failed paths. Held-out split by trace_id hash or temporal. EVIDENCE: harness sip_effect families + record_shim_activation (Cycle-007+ hygiene :2139+); nomenclature:85 URS + plan:55 "mixes OPSD (privileged successful reroute traces and successful shim cascades)".

**Eval (hit-rate on held-out + secondary metrics)**:  
- Primary: Reroute decision hit-rate = |{held-out traces where policy top-1 action == label_reroute_action}| / N_heldout. Thresholded acceptance (policy confidence > tau). Target: >0.65-0.75 on G-trace held-out (research harness only).
- Secondary (on accepted decisions): mean token reduction (harness-accounted), mean collapse improvement vs baseline (no-policy / MinMax-only / random), cascade depth distribution, false-positive rate (accepted reroute with negative outcome).
- Robustness: stratified by chelation_variance buckets + cascade_depth; ablation (remove minmax features / usage_stats / mtp_pred).
- Harness smoke: re-run extended sip/sip_effect families with policy head stub (dict lookup or tiny MLP) emitting decisions; compare gated vs baseline on ndcg/recovered/noise (bitwise match on core except new deltas); persist Cycle-011 bhs_evidence with "microslm_hit_rate", "token_reduction", "policy_vs_minmax" fields + rollback proof. EVIDENCE: 0400 json fields + prior Cycle-00x emission patterns.
- Held-out protocol: 80/20 trace split; no leakage from training usage_stats.

**BHS / L-Taxonomy Disclosures (mandatory per rulebook §4 + goal §18-29 + protocol §6)**:  
- L4 (Partial-with-claim-of-complete): "route-policy head that learns reroutes" language while this is prose sketch only (0 model, 0 training loop, 0 integration into harness/MTP/I role, 0 SIP). "2-4GB" aspirational per plan:54; no actual weights or hosting demonstrated. Matches 10-cycle pattern (cycle0400:5 0/10 fidelity; goal Model Change Log :220).
- L9 (Hygiene / doc-as-impl): Additional doc volume (this loop_02/ md) while backlog #1 0% SIPs + 8+ OPEN SHIM-CDs (next-session:61) + BLOCKED (check_block_flag.py FAIL) + 10 cycles 0 substrate (dashboard:973 + cycle0400:32). "sketch for future" does not advance #9 MinMax promotion or close debt.
- L1 (Scaffold-as-feature): All described vectors/stats exist in research scaffolds (shim_node usage_stats, harness MinMax, antigravity variance) but un-wired to any policy.
- L13 (Soft-prose-claimed-as-mechanical): Any future claim this "enables self-improving reroutes" without runtime EVIDENCE from prod-path + Tier B pass is L13.
- L3 (Mock-ate-the-real): Relies on MockMTP + synthetic harness (extension:52+).
- Additional: Over-pruning risk (mirrors goal:156 for MinMax); interaction with existing global_variance decision untested; pre-agg cost in dynamic indexes (new debt if not bounded).
- Process: Adding H sketch (per goal:56) while core #1 0% + §128 exceeded 7x+ (cycle0400:65) follows the exact risk flagged at goal:157.

**EVIDENCE (tool-grounded, absolute paths + lines; survives fresh checkout)**: All re-reads above + this file's creation via write (no prior existence per list_dir loop_02/ pre-write). 0-prod grep (0 executable imports outside 2 research files; comments only at antigravity:2585 + tts seams). Cycle-010 artifacts (0400.json, bhs_10agent_...json, loop_02/08+09 mds) + harness MinMax demo + shim_node usage_stats. No new py/md except this. Pre/post list_dir + grep for "08_cycle011_agentH" confirms unique.

**SMOKE (rejection tests on fresh checkout; run these)**: 
1. `python -B scripts/check_block_flag.py` → BLOCKED + "Carried Debt row count: 2" + "RESULT: FAIL".
2. `grep -r --include="*.py" "ShimNode\|MinMaxBlockRelevanceScorer\|apply_shim_cascade" --glob='!**/docs/**' --glob='!**/artifacts/bhs_*.json' | cat` → 0 hits (or only L4-guarded research).
3. `ls docs/steering_chelation_rag_dag_research/loop_02/ | grep 011_agentH` → exactly this file.
4. Re-read this file header re-read log + "doc-only / 0 implementation" + "L4 bounded" present; no claims of "implemented" or "wired".
5. Harness smoke (extended): PYTHONPATH=. python -B .../shim_collapse...py --family sip_effect --research-shim (metrics bitwise match prior baselines except any new gated fields; no MicroSLM code path executed).
Any claim "Micro-SLM policy advanced substrate / reduced debt / closed #9" fails these + goal:18-29.

**Brutal Honesty (per rulebook v3.3 §4 + CLAUDE.md + goal §128 + protocol §6 + cycle0400:65 trajectory)**:  
This is a narrow, bounded research design sketch drafted under explicit H role (goal:56). It compiles cheap signals already surfaced in Cycle-010 MinMax work (harness:593+) + existing variance/chelation (antigravity:2606) + usage (shim_node:163) + MTP nomenclature into a coherent input/objective/data/eval proposal for a future 2-4GB head. **0 code, 0 weights, 0 training, 0 integration, 0 runtime evidence.** Does not satisfy any success criterion. Adds to doc volume (L9 risk) while 0 SIPs + BLOCKED + 10-cycle 0 substrate (cycle0400:71 "10 cycles of unambiguous failure"; dashboard:990 "Human intervention required"). Trajectory unchanged. Per §128 + every prior E/D/Agent9/010 output: **PAUSE or TERMINATE scheduler 019e669bf1bb (or scope-reduce to pure audit collection)** until first real SIP + prod EVIDENCE + BHS >=60 + deltas. This sketch is L4 scaffolding only. Independent reviewer disproving via above SMOKE + re-reads of goal:109/56 + cycle0400 succeeds. Carried debt +1 (process).

**References (absolute, key)**: goal:56/108-110/157/213-230; protocol:10/74/110/38-41; cycle_20260527_0400.md:5/38/67/71; dashboard:956-993/973; next-session:22/61-69; check_block_flag.py:108-280; shim_node.py:2/34-36/163-170; harness:593-620/651+/1993+; antigravity_engine.py:2585-2601/2606-2607; plan:54/204-272; nomenclature:79-91; loop_02/08_cycle010_agent8...md + 09_... + 0400.json + bhs_10agent_integrator...json; rulebook v3.3 §0-4/6.2-6.3/§128; 0-prod greps + list_dir pre/post.

*Cycle-011 Agent H complete. Pure doc sketch. 0 implementation / 0 substrate. L4 bounded. References goal:109 #9 + H role:56 + cycle0400. §128 active. Evidence or stop. Human intervention required.*

---

**Post-creation verification (this agent)**: list_dir loop_02/ (new file present, unique name per protocol:41); 0 code changes anywhere (confirmed via no search_replace on *.py + 0-prod grep); re-read this file itself for consistency. All per constraints: Pure doc. BHS "sketch not capability". Long-running bounded (todo phases + re-reads). No broadening. End of H output.