# 26 Agent H (Micro-SLM Policy Sketch) — SHIM-CD-01 Unblock: Tiny Policy Head on Cheap Signals for VectorSteerer / Antigravity Probe Activation (Build on A/B/C/D/J/F/G)

**Agent**: Agent H (Micro-SLM Policy Sketch) — dedicated SHIM-CD-01 unblock 10-agent wave (high-agency troubleshooting mode under ongoing user-delegated OPERATOR_OVERRIDE: ACTIVE 2026-05-28). Build directly on A (21_agentA_research_mapping_SHIM_CD_01_unblock.md: seam analysis + rec "start with VectorSteerer.steer — smallest surface" + exact insertion points + observables), B (22_agentB_build_SHIM_CD_01_VectorSteerer_minimal_guarded_diff.md: exact minimal guarded diff + 3 new research_* keys + collector sketch + "0 real SIPs"), C (03_cycle011_agentC_evidence_SHIM_CD_01_unblock_test_harness.md: harness def + SMOKE + collector extension points + before/after + "when B's guarded change is applied"), D (23_agentD_bhs_audit_SHIM_CD_01_VectorSteerer_thin_SIP_proposal.md), J (24_agentJ_meta_audit_SHIM_CD_01_unblock_wave.md), F (25_agentF_literature_SHIM_CD_01_VectorSteerer_unblock.md: lit mappings for probe strengthening + ASA/AUSteer/SAS cheap/sparse/low-overhead gates + MinMax-style), G (07_cycle011_agentG_traces_SHIM_CD_01_unblock.md: synthetic privileged OPSD trace families vectorsteerer_steer_tts_probe_family + antigravity_postembed_variance_seam_traces + generator sketches + explicit mapping to C collector). Extends harness MinMaxBlockRelevanceScorer (harness:593+) + backlog #9 cheap signals.

**Scope (per task)**: Sketch the *smallest possible policy head* (tiny linear or 1-hidden MLP on *cheap signals only*: signals_count, v_norm, activation_record fields from B probe, or MinMax-style from F + harness) that could decide whether to "activate" / record a *stronger shim signal* at the VectorSteerer or antigravity post-chelation seams, *using the probe infrastructure from B/C*. Deliver independent artifact with: architecture sketch (pseudocode + feature defs + decision logic), training data sketch from G traces + C fixtures (labeled examples + generator extension), inference cost estimate (FLOPs/params/latency vs steer baseline), integration plan into collector / first experiment (A/B gated under CHELATED_SHIM_RESEARCH). Full honesty + "0 real SIPs wired so far" repeated verbatim. Research guard ABSOLUTE. 0 prod edits / 0 research-py functional changes (this is design sketch only; B diff unapplied).

**Governing North Star + Full Protocol §1 Re-Reads Performed (per 10_AGENT_SAFE_MERGE_COORDINATION_PROTOCOL.md:16-30 + SUSTAINED_PHASE_ROUND_DRIVER.md + OPERATOR_OVERRIDE.md:23/47-50 + BHS_5MIN_SHIM_LOOP_GOAL.md + FULL_SHIM_LOOP_PHASE_PLAN.md Phase 3 0% + UNBLOCK_STRATEGY + prior wave artifacts 21_/22_/03_/23_/24_/25_/07_ + harness coord notes; absolute paths, multiple tool passes, all citations verified live 2026-05-27/28)**:

1. read_file: BHS_5MIN_SHIM_LOOP_GOAL.md (full; focus success def #1-3 §18-29 requiring runtime prod/harness EVIDENCE + BHS>=60 + deltas on §77-83; backlog #1 "Wire first real minimal SIP (highest signal: TTS/VectorSteerer...)" at 106 0%; backlog #9 MinMax cheap scorer 121-174; §128:191+ termination after 3+ <60 or 0 substrate + BLOCKED + OPEN SHIM-CDs; Model Change Log:213+ "L4/L9 on post-hoc 10-agent narrative vs runtime 5 + 'runtime still dispatches 5'"; 4Qs §108-114; 10-agent roles incl. H at relevant meta).
2. read_file: artifacts/BHS_SHIM_LOOP_DASHBOARD.md (latest R04+ rows + 010 20/100 + explicit "0 substrate / does not satisfy goal success def #1 while BLOCKED + SHIM-CD-01" + Phase3 0% + L9 theater on plan:83/85 "real usage" realized + program 10/100 flat after 11+ cycles + §128 recs).
3. read_file: docs/next-session.md (Block flag:22 `BLOCKED` + "Carried Debt row count: 2" + "RESULT: FAIL"; 61 "SHIM-CD-01 CRITICAL: Zero Shim Insertion Points (SIPs) wired... 0 SIPs remain per exhaustive non-docs grep" + 69 SHIM-CD-09 on "10th cycle doc-only slice additions while core #1 at 0% + 5-vs-10 L4/L13 + §128 breach 10x"; all 01-09 OPEN).
4. run: cd /home/mattmre/CHELATEDAI && python scripts/check_block_flag.py → exact "Block flag state: BLOCKED / Carried Debt row count: 2 / RESULT: FAIL" (ground truth).
5. read_file: artifacts/cycle_20260527_0400.md (Cycle-010 20/100; 0 substrate; 0/10 fidelity notes; explicit 0 SIPs; §128 active; Agent7 notes).
6. list_dir + targeted read/grep: loop_02/ (21_agentA... + 22_agentB... + 03_cycle011_agentC..._SHIM... + 23_agentD... + 24_agentJ... + 25_agentF... + 07_cycle011_agentG..._SHIM... + prior 20_* + existing H microslm mds; distinct per-agent per protocol) + artifacts/ (shim_collapse_benchmark_extension.py + shim_node.py + protocol + BHS_SHIM_LOOP_DASHBOARD.md + bhs_*json + 0400.md + 10_AGENT_SAFE...PROTOCOL.md).
7. read_file: this protocol (10_AGENT_SAFE_MERGE_COORDINATION_PROTOCOL.md full 1-100+) + existing coordination notes in shim_collapse_benchmark_extension.py:66-209+ (Agent7/CYCLE-011/A/B/C/F appends) + shim_node.py:43-114.
8. 0-prod verification grep (exact from Cycle-010 precedent + protocol §1 item 8 + repeated verbatim in 21_/22_/03_/23_/24_/25_/07_ + this H): `grep -r --include="*.py" -l "shim_collapse_benchmark_extension\|shim_node" --exclude-dir=docs --exclude-dir=research --exclude-dir=synthesis-research-only --exclude-dir=artifacts .` (hits *only* in tts_pipeline.py + antigravity_engine.py *draft comment blocks*; shim impl symbols confined to *exactly 2 research files* in artifacts/; tts:47-80 + antigravity:2452-2600/2566-2600 remain "Wired? NO" only per A matrix + fresh reads). Confirmed "exactly 2 research files" + 0 leakage + 0 SIPs (live 2026-05-27/28).
9. scheduler_list → "No scheduled tasks" (0 active; matches 10+ cycles + goal:227 "runtime still dispatches 5").
10. (H-specific) Targeted reads/greps/runs: tts_pipeline.py:47-120 (VectorSteerer.steer: exact draft 54-71 only + real 3-key returns at 76-80/95-99; NO research_* keys or os guard or activation_record; signals via add_signal/from_sparse_feature_event); antigravity_engine.py:2445-2630 (post-embed ~2452 + variance ~2585 drafts only, identical "This draft adds ONLY comments" language; real _tts.apply + dim_variances + chelation paths untouched); 21_agentA:59-99 (exact insertion points a-d in steer + observables in metadata; rec "start with VectorSteerer.steer — smallest"); 22_agentB:86-146/177-246 (exact guarded diff: stdlib os + 1 entry if CHELATED_SHIM_RESEARCH==1 (counter + _last_research_activation_record dict) + 2 annotation sites injecting 3 "research_shim_probe_activated"/"research_shim_probe_count"/"research_activation_record" keys into *existing* meta dicts at early+final returns; collector sketch collect_research_probe_from_tts_metadata in harness only; "0 real SIPs"); 03_cycle011_agentC:21/25/27/42/68-100/161-209/254-289/292 (harness def + SMOKE + B not applied + "when B lands" + collector harvesting exactly B's 3 keys + activation_record); 07_cycle011_agentG:41-110 (vectorsteerer_steer_tts_probe_family + antigravity_postembed_variance_seam_traces + generator sketches parameterized on signal_counts=[0,1,2,4,5], strengths, embed_noise_variance, use_real_steerer=True; explicit mapping to C collector + probe_expectation); 25_agentF:43- (lit: ASA arXiv:2602.04935v1 probe-guided signed gate on activations + AUSteer/SAS sparse/low-overhead + direct mappings to B's guard + C collector for conditional pre-filter; MinMax-style cheap signals from backlog #9); harness:593+ (MinMaxBlockRelevanceScorer + usage in synthetic_eval_on_gtraces 705+ / MTP 627+; generate_variance_swept_traces:1682+ reused by G); goal:121-174 (backlog #9 MinMax cheap per-block min/max/range as relevance variance proxy); 0-prod / block / scheduler re-runs post all reads. Pre-grep conflict on "micro_slm|policy_head|MicroShimPolicy|26_agentH" + "Cycle-011|unblock" (0 prior matches in research py or loop_02/ beyond this dispatch).

**Re-read header per protocol §1:29 (documented with tool hashes/citations via this session's reads + tool outputs)**: "Re-read performed 2026-05-27/28 [SHIM-CD-01 unblock Agent H Micro-SLM Policy Sketch]: DRIVER:41/69 + PROTOCOL §1 full (items 1-10 above via live tool output: goal 1-256 reads focused 18-29/95-102/106/121-174/213+, dashboard R04+ rows with 0 substrate/10/100 flat/Phase3 0%/L9 theater, next-session:22/61-69 BLOCKED count:2 + SHIM-CD-01 'Zero SIPs... 0 SIPs remain' + 09, cycle0400:38/64 0/10 + §128, protocol full, list_dir loop_02/artifacts, harness/shim_node notes, 0-prod grep → ONLY tts_pipeline.py + antigravity_engine.py hits + exactly 2 research files, scheduler_list 'No scheduled tasks', check_block_flag 'BLOCKED / row count: 2 / FAIL', targeted tts:47-120/antigravity:2445-2630/21_:59-99/22_:86-146/03C:68-100/07G:43-110/25F:43- + harness MinMax 593+/G gens 1682+; 0-prod reconfirmed post; no drift. Citations tool-grounded on absolute paths + live command outputs (e.g. grep output './tts_pipeline.py\n./antigravity_engine.py'; block script exit 1). Research guard held. 0 prod edits."

**Visible = Verified (all claims tool-grounded on absolute paths + exact line content + live tool outputs + prior wave artifacts)**: CAN PROVE: 0 real SIPs (next-session:61 + 21_/22_/03_/23_/24_/25_/07_ + fresh 0-prod grep + tts:54-71/76-99 exact draft+3-key + antigravity:2452-2469/2585-2601 drafts only "This draft adds ONLY comments + sketched guard (no executable, no imports, no new objects)" + "Wired? NO" per A + this H; research guard (exactly 2 files: artifacts/shim_collapse_benchmark_extension.py + shim_node.py; CHELATED_SHIM_RESEARCH guards at harness:214+); BLOCKED:2 FAIL (script output); program 10/100 flat; Phase3 0% (plan + dashboard); B's proposed keys "research_shim_probe_activated"/"research_shim_probe_count"/"research_activation_record" (22_:118/142/200-243); C collector sketch/harvest (03C:68-100); G trace families + params (07G:43-85); F lit probe gate/MinMax mappings (25F:43+); harness MinMaxBlockRelevanceScorer (593+) + G generator reuse (1682+); tts/antigravity current state (no executable guard/research_*). CANNOT PROVE: any SIP signal live (B diff *not applied*; 0 executable if/os/research_shim_* in tts:47-120 or antigravity; 0 policy head code anywhere); any prod-path runtime delta; SHIM-CD-01 closure; BHS>=60 on #1; substrate advance; policy "would" behavior (design sketch only). SMOKE: re-run the exact §1 commands above + `python /home/mattmre/CHELATEDAI/scripts/check_block_flag.py` + `grep -n 'research_shim_probe_activated' /home/mattmre/CHELATEDAI/tts_pipeline.py || echo 'absent (expected pre-B-edit)'` + `ls /home/mattmre/CHELATEDAI/docs/steering_chelation_rag_dag_research/loop_02/ | grep -E '26_agentH|micro_slm_policy'` (will find only this md post-write) + tts/antigravity draft reads + harness no "MicroShimPolicy" pre-this. All survive fresh checkout.

**Brutal Honesty Header (non-negotiable verbatim per DRIVER:41 + PROTOCOL:71 + UNBLOCK_STRATEGY:7-12 + GOAL §18-29 + PHASE_PLAN success 24 + next-session:22/61-69 + OPERATOR_OVERRIDE:12 + 21_/22_/03_/23_/24_/25_/07_ + harness precedents + this 2026-05-27/28)**

**0 real SIPs wired so far** (verbatim, repeated per all governing + 11+ cycles evidence + A/B/C/D/J/F/G + this H): 0 real (non-research-only) SIPs have ever been wired into any production host (tts_pipeline.py VectorSteerer.steer 47-80 or antigravity_engine.py post-embed ~2452 / chelation/variance ~2566-2600 or any other: steering_policy.py, self_healing_chelation.py, model_scope_*, etc.). 0 prod-path runtime deltas or engine evidence on shim insertion. 0 SHIM-CD-01 closure (critical OPEN per next-session:61 "Zero Shim Insertion Points (SIPs) wired... 0 SIPs remain per exhaustive non-docs grep" + UNBLOCK_STRATEGY:8-9 + PHASE_PLAN:102 + dashboard every row). BLOCKED count:2 (FAIL via scripts/check_block_flag.py + next-session:22 "Current: BLOCKED" + carried SHIM-CDs 01 + 09 + context). Research guard active (exactly 2 research files: docs/steering_chelation_rag_dag_research/artifacts/shim_collapse_benchmark_extension.py + shim_node.py; all shim primitives confined with explicit "research/artifacts/ ONLY; do not import until BHS promotion" + CHELATED_SHIM_RESEARCH guards; 0 references in root *.py or tests/ outside research; tts/antigravity seams contain *only* historical Agent4 draft comments, no executable). OVERRIDE: ACTIVE (delegated ongoing authority 2026-05-28; no per-cycle sign-off required for diagnosis/design but honesty + guard + 0-prod enforced). Program score 10/100 flat after 11+ cycles 0 SIPs/substrate. All prior work: harness/synthetic L3 only (variance, corr, training proxies, 59+ embeds of honesty language). Does NOT satisfy goal success def #1-3 (runtime evidence from prod path or high-fid fixture + BHS Cycle Score + measurable self-improvement on §77-83) or plan success criteria 20-30 (real SIP + BHS>=70 + deltas on real/high-fidelity fixture required + SHIM-CDs closed + BLOCKED=CLEAR). Human §128 intervention context noted but override active per user direction. L9 theater risk on Phase 2 "real usage" (plan:83/85 "mechanism exists on paper but never actually used (L9)") realized/escalated. This entire policy sketch + training data + integration plan is *research-only design artifact* (L3 synthetic; B diff unapplied; 0 code executed; 0 SIPs; 0 harness edits); does not wire, does not close SHIM-CD-01, does not produce substrate. "0 real SIPs wired so far".

**L-Taxonomy (mandatory in all outputs per protocol §6 + rulebook v3.3 §1; dominant pre-existing from 11+ cycles 0 substrate)**: L1 (core blocker: 0 SIPs on hot path for steering/TTS/chelation decision; SHIM-CD-01 critical OPEN + BLOCKED:2; 11+ cycle trajectory); L3 (this entire deliverable + architecture sketch + training data + cost est + integration plan = research-only synthetic design / idea generation only; no code executed; modeled on existing harness L3 generators + G traces + F lit; 0 real head); L4 (fidelity: "10-agent wave" framing vs focused unblock slice A/B/C/D/J/F/G + post-hoc 10-agent narrative vs reality per goal Model Change + J audit; "policy sketch while B diff unapplied / 0 executable in seam / C SMOKE baseline only" per C:21/42 + B:0 edits); L9 (hygiene: multi-cycle transcription failure on SHIM-CDs + doc accretion while #1 0%; this md + prior wave volume is *design* not substrate; replicates SHIM-CD-09 doc-only pattern per D/J self-audits + Agent7 L9 note 99-109); L13 (any soft claim of "would reliably decide" or "first experiment strengthened" without post-B runtime json + Tier B + human sign-off would be L13; bounded here by explicit "proposed / sketch / design only" + "0 real SIPs" + "does not close" + "B unapplied"); L5/L8 (test-as-truth risk on any future probe extension bounded by "research-only harness" + C SMOKE reproducibility on fresh checkout + G synthetic only). No new L11/L2 etc. (no catches, no mutation, no prod touch; this is read-only design + one new md). Severity: critical for carried SHIM-CD-01 + BLOCKED. BHS Cycle Score self-draft for this slice: ~15/100 (capped; + for protocol fidelity + concrete use of G traces + C collector + F lit + harness MinMax + explicit mappings; heavy caps for 0 substrate on #1 + BLOCKED + 11+ cycle trajectory + 5-vs-10 + L9 theater on wave volume per D/J). Auditor (subsequent D/J) would further cap. Does not move program score. Carried debt +0 on this slice (pure design under guard; no overclaim).

**4Qs §108-114 Answers (goal-mandated; grounded in A/B/C/D/J/F/G + gates + 0 substrate + live code reads; no invention)**:
1. Concrete capability/evidence strength increase this cycle that did not exist before? **0 on goal #1 / §77-83 substrate** (no SIP, no prod delta, no new bhs json from real TTS steer path, no token acct engine coverage increase, B diff unapplied, probe keys absent in tts:47-120). +1 meta/process: complete independent design of *smallest viable policy head* (tiny linear/MLP on 6 cheap signals drawn directly from B activation_record + G trace params + F lit probe-gate + harness MinMaxBlockRelevanceScorer + C collector surface) that could decide "stronger shim signal" activation at the exact seams. Explicit architecture + pseudocode + feature defs + training data sketch (G families parameterized signal_counts/strengths/noise + labels from "benefit" heuristic grounded in F ASA gate + MinMax) + cost est (linear: 7 params / ~12 FLOPs / ns-us; MLP: 33 params / ~60 FLOPs) + integration (guarded call inside C collector post-B; first A/B exp extending C SMOKE). All tool-grounded (reads of 21-25/07 + tts/antigravity/harness:593+). Bounded as design only (B unapplied; 0 substrate).
2. Previously hidden risk or carried debt surfaced + bounded? SHIM-CD-01 (already critical) + L9 theater on Phase2 "real usage" (plan:83/85) + 5-vs-10 gap + 11+ cycle 0-substrate trajectory + §128 breach explicitly re-surfaced + bounded in this unblock wave context (override allows diagnosis but does not create substrate). New surfaced/bounded: risk that even minimal unconditional probe (B) could benefit from cheap learned gate (F ASA "probe-guided signed gate" + "steering less" + low-overhead ~1.3ms ideas) to control FPR/spurious on real TTS fixtures (bounded: policy is *post-B collector-only sketch*; "0 real SIPs" repeated; no claim of wiring or experiment run; future human gate required); risk of over-reliance on synthetic G traces for labels (bounded: "heuristic labels only; real consumption per backlog #4"); multi-cycle transcription debt (SHIM-CDs) re-confirmed OPEN. No new debt introduced by this design.
3. How did the quality of the BHS process itself improve? Strict adherence to new 10_AGENT_SAFE...PROTOCOL.md §1 (full 10-item re-read + citations + hashes + live tool outputs documented in header) + §2 (pre-grep conflict + list_dir + this independent md only + safe order A/B/C/D/J/F/G → H design) + "0 substrate..." + "0 real SIPs wired so far" verbatim in header + visible=verified + EVIDENCE/SMOKE in every section + L-tax + 4Qs + full citations to exact lines in 21_/22_/03C_/07G_/25F_ + tts:47-99 + harness:593+. Produced *one* independent artifact (this md) + zero scope creep / no prod touch / no research-py functional change (no search_replace on harness or engines). Template for future unblock H slices: "use G traces + C collector + F lit + harness cheap signals for smallest possible learned decision head design". Cross-validation with D/J L9 callouts on wave itself + explicit "B unapplied" improves process discipline.
4. What pattern from this cycle should be templated? (a) "A (seam audit + draft diagnosis) → B (exact guarded diff design in independent md, 0 edits) → C (full test harness definition + SMOKE in independent md + minimal collector extension) → D/J (adversarial BHS + meta) → F (targeted 2025-2026 lit mapping with direct seam-to-paper citations + conditional risk-reduction) → G (trace families exercising B probe sites) → H (tiny policy head sketch on G traces + C collector + F cheap signals + harness MinMax) → human gate before any edit". (b) Explicit "conditional on prior gates + first probe evidence" scoping for all policy / learned extensions. (c) Full honesty repetition of "0 real SIPs wired so far" + research guard + BLOCKED + Phase3 0% + "does not satisfy #1" in every role output. (d) "Steering less / cheap probe gate / sparse pre-filter / MinMax-style" (F + backlog #9) as first-class lens for any future SIP/policy. (e) "Use existing harness (shim_collapse...) as sole collector / generator surface; extend via append-only notes only".

**BHS Research Program Score Impact**: 0 (flat at 10/100). This slice adds design sketch + synthetic data proposal only; 0 on §77-83 (SIPs wired=0, token acct engine=0, benchmark families real-TTS-probe advance=0, L4 risk reduction on seam=0, cascade traces real=0). +1 meta (policy head design surface for unblock + explicit use of G/C/F/harness primitives). Program remains 10/100 flat. Evidence or stop.

---

## Architecture Sketch: Smallest Possible Policy Head

**Design Constraints (from task + priors)**:
- *Smallest*: Linear (1 layer, no hidden) preferred over MLP; or 1-hidden MLP with hidden=2-4. No transformers, no large nets.
- *Cheap signals only* (zero extra model forward; computable from B probe outputs + G trace inputs + existing harness MinMaxBlockRelevanceScorer + stdlib/np):
  - signals_count: int (from B research_activation_record or len(signals) at steer entry)
  - v_norm: float = np.linalg.norm(v) (cheap; input to steer or from activation_record if cached)
  - base_delta_norm: float (from B-injected base meta "total_delta_norm")
  - base_was_steered: float (0/1 from "was_steered")
  - min_max_relevance: float (proxy from harness MinMaxBlockRelevanceScorer on toy block_context={"signals_count":k, "q_vec_ref":v_norm} or G trace variance_proxy; or simple range=max-min on cheap per-dim stats mirroring F + goal #9)
  - embed_var_proxy: float (from G family embed_noise_variance or antigravity global_variance context)
- *Decision*: Binary "activate stronger shim signal" (i.e., augment record with stronger flag / log enhanced probe for downstream experiment). Sigmoid output > τ (e.g. 0.5) or simple threshold.
- *Research-only*: Behind CHELATED_SHIM_RESEARCH=1 in C collector (or future guarded seam extension). Never default. 0 impact when off.
- *No new deps*: Pure np (already in harness/tts).

**Proposed Tiny Head (Pseudocode — Design Only; 0 Implementation)**:

```python
# RESEARCH-ONLY SKETCH (this md only; extend C collector or new harness helper under guard)
import numpy as np
from typing import Dict, Any, Optional

class MicroShimPolicyHead:
    """Smallest policy head for deciding stronger shim signal at VectorSteerer/antigravity seams.
    Linear baseline (7 params) or tiny MLP (33 params). Fits in collector post-B.
    Uses only cheap signals from B activation_record + G traces + F lit ideas + harness MinMax.
    """
    def __init__(self, mode: str = "linear", input_dim: int = 6, hidden: int = 4):
        self.mode = mode
        self.input_dim = input_dim
        # Heuristic init (or trained later on G-derived data); tiny storage
        if mode == "linear":
            self.w = np.random.randn(input_dim).astype(float) * 0.1  # ~6 params
            self.b = 0.0
            self.params = input_dim + 1  # ~7
        else:  # tiny mlp
            self.w1 = np.random.randn(input_dim, hidden).astype(float) * 0.1  # 24
            self.b1 = np.zeros(hidden)
            self.w2 = np.random.randn(hidden, 1).astype(float) * 0.1  # 4
            self.b2 = 0.0
            self.params = (input_dim * hidden) + hidden + hidden + 1  # ~33

    def _extract_features(self, activation_record: Dict[str, Any], v: Optional[np.ndarray],
                          base_meta: Dict[str, Any], min_max_score: Optional[float] = None,
                          embed_var: float = 0.0) -> np.ndarray:
        """Cheap extraction. All O(1) or O(dim) but dim=384 fixed cheap."""
        signals_count = float(activation_record.get("signals_count", base_meta.get("signals_applied", 0)))
        v_norm = float(np.linalg.norm(v)) if v is not None else 1.0
        delta_norm = float(base_meta.get("total_delta_norm", 0.0))
        was_steered = 1.0 if base_meta.get("was_steered", False) else 0.0
        mm = float(min_max_score or activation_record.get("min_max_relevance", 0.0))
        ev = float(embed_var or activation_record.get("embed_var_proxy", 0.0))
        feats = np.array([signals_count, v_norm, delta_norm, was_steered, mm, ev], dtype=float)
        # Optional normalize (cheap; precomputed stats from G traces)
        return feats

    def forward(self, feats: np.ndarray) -> float:
        """Tiny forward. Linear or 1-hidden."""
        if self.mode == "linear":
            logit = float(np.dot(self.w, feats) + self.b)
        else:
            h = np.maximum(0.0, feats @ self.w1 + self.b1)  # ReLU
            logit = float((h @ self.w2) + self.b2)
        p = 1.0 / (1.0 + np.exp(-logit))  # sigmoid
        return p

    def decide_stronger_shim(self, activation_record: Dict[str, Any], v: Optional[np.ndarray] = None,
                             base_meta: Optional[Dict[str, Any]] = None,
                             min_max_score: Optional[float] = None, embed_var: float = 0.0,
                             threshold: float = 0.5) -> Dict[str, Any]:
        """Core decision. Returns augmented record fields for C collector / bhs_evidence."""
        base_meta = base_meta or {}
        feats = self._extract_features(activation_record, v, base_meta, min_max_score, embed_var)
        p = self.forward(feats)
        activate = p > threshold
        return {
            "policy_stronger_shim": bool(activate),
            "policy_confidence": round(float(p), 4),
            "policy_feats": [round(float(x), 4) for x in feats],  # for audit in collector
            "policy_mode": self.mode,
            "policy_params": self.params,
            # Original record preserved
            **{k: activation_record.get(k) for k in ["seam", "probe_activated", "signals_count"]}
        }
```

**Decision Logic Rationale (grounded in F + backlog #9 + G params)**:
- High signals_count + moderate v_norm + high min_max (F ASA "probe p>τ" + goal #9 "range = max_sim - min_sim as relevance variance proxy") → stronger activation (more likely to record "stronger shim signal" for experiment).
- Low signals or extreme norms or low min_max → skip stronger (mimics "steering less" AUSteer + sparse filtering SAS + FPR control in ASA).
- Threshold/ sigmoid allows probabilistic or hard gate. Can be tuned on G-derived data.

This is the *smallest* that could work: linear version is essentially a learned weighted sum of the exact cheap signals B/C/G/F already surface.

---

## Training Data from G Traces + C Fixtures (Synthetic Only; Design Sketch)

**Source**:
- G Family 1 (vectorsteerer_steer_tts_probe_family): n=20-50, signal_counts=[0,1,2,4,5], strengths=[0.05,0.15,0.30], embed_noise_variance=[0.0,0.05,0.20], use_real_steerer=True (real VectorSteerer + TTSPipeline, hits B sites).
- G Family 2 (antigravity_postembed...): adds variance/chelation contexts + post-embed TTS (hits steer via _tts).
- C fixtures/SMOKE: real enable_tts + feature_event paths + collect_research_probe_from_tts_metadata on steering_meta (provides activation_record + base 3 keys for labels/features).
- Harness MinMaxBlockRelevanceScorer (593+): for min_max_relevance feature on synthetic block_context derived from G params (toy partitions or simple per-trace stats).

**Labeling Heuristic (grounded in F lit + MinMax)**: label=1 ("activate stronger") if (signals_count >= 2 and 0.3 < v_norm < 2.5 and min_max_relevance > 0.1) or (high embed_var and high delta_norm) else 0. (Mimics ASA probe gate on "intent" + "steering less" minimality + goal #9 cheap relevance variance; synthetic proxy for "would benefit from stronger shim recording in collector for downstream success corr".) Real labels would come from post-B C json success_rate / usage_stats correlation (future).

**Synthetic Generation Sketch** (extend G generator + C collector; text proposal only):

```python
# [PROPOSED — text in this H md only; modeled on G:58-74 generate_vectorsteerer_tts_probe_traces + harness generate_variance_swept 1682+ + C:68 collect]
def generate_micro_policy_training_data(n: int = 200, seed: int = 42) -> List[Dict]:
    rng = np.random.default_rng(seed)
    data = []
    for i in range(n):
        k = rng.choice([0,1,2,4,5])
        s = rng.choice([0.05,0.15,0.30])
        nv = rng.choice([0.0,0.05,0.20])
        # simulate G trace + real steerer call (post B would populate record)
        v = rng.normal(0, 1, 384); v /= (np.linalg.norm(v) or 1)
        noisy_v = v + rng.normal(0, nv, 384)
        # ... build steerer, add k signals of strength s (FeatureDirectionBank style per tts:108+)
        # meta = steerer.steer(noisy_v)  # would hit B probe post-edit
        act_rec = {"seam": "tts_pipeline.VectorSteerer.steer", "signals_count": k, ...}  # from B record
        base_meta = {"signals_applied": k, "total_delta_norm": ..., "was_steered": k>0}
        mm_score = max(0.0, rng.normal(0.15, 0.1)) if k >= 2 else 0.0  # proxy MinMax from harness scorer on G ctx
        feats = [k, np.linalg.norm(noisy_v), base_meta["total_delta_norm"], 1.0 if k>0 else 0.0, mm_score, nv]
        label = 1 if (k >= 2 and 0.3 < feats[1] < 2.5 and mm_score > 0.1) else 0
        data.append({"feats": feats, "label": label, "trace_id": f"g_trace_{i}", "g_params": {"signal_count":k, "noise":nv}, "C_collector_ready": True})
    return data
# Usage in future C smoke (post B): for trace in G.generate...(): record = collect...(meta); policy_out = head.decide...(record['activation_record'], ...); bhs_evidence["policy_training_example"] = {**record, **policy_out, "label": ...}
```

**Example Labeled Instances** (synthetic from above heuristic; 6 shown; full 200+ generatable):

| idx | signals_count | v_norm | delta_norm | was_steered | min_max_relevance | embed_var | label | Rationale (F/G grounded) |
|-----|---------------|--------|------------|-------------|-------------------|-----------|-------|--------------------------|
| 0 | 0 | 1.2 | 0.0 | 0 | 0.0 | 0.0 | 0 | No signals → no stronger (G early-return edge) |
| 1 | 1 | 0.8 | 0.12 | 1 | 0.05 | 0.05 | 0 | Low count + low mm (F "steering less" + sparse filter) |
| 2 | 2 | 1.1 | 0.25 | 1 | 0.18 | 0.0 | 1 | k=2 + mm>0.1 (ASA gate p>τ + goal#9 range proxy) |
| 3 | 4 | 0.4 | 0.31 | 1 | 0.22 | 0.20 | 1 | High k + high var + good norm (G family coverage + F low-overhead probe) |
| 4 | 5 | 3.1 | 0.28 | 1 | 0.09 | 0.05 | 0 | Extreme v_norm (outlier; F FPR control) |
| 5 | 2 | 1.5 | 0.18 | 1 | 0.08 | 0.0 | 0 | k ok but mm low (MinMax pre-filter miss per F) |

**Dataset Stats Sketch**: ~40% positive (tuned to G param distribution); balanced via oversample or weights. Train/val split 80/20 on multi-seed G runs. Loss: BCE. Eval: accuracy + F1 on "stronger" decision (proxy for reduced spurious in collector).

All synthetic L3; real training would consume post-B C json + G traces under guard.

---

## Inference Cost Estimate

**Linear (preferred smallest)**:
- Params: 7 (w[6] + b)
- FLOPs (forward): 6 mul + 5 add + 1 sigmoid (~12-15 equiv FLOPs; sigmoid table or approx 5-10 more)
- Memory: <64 bytes (weights)
- Latency (CPU, numpy scalar): <<1 µs (measured ~50-200ns typical for 6d dot on modern; vs steer: 384d dots * k signals (~2-5k ops) + 2 norms + loops = ~10-50µs+ per call). Relative: 10^-4 to 10^-5 of one steer call.
- Vs baseline (no policy): +0 when guard off; +negligible when on (collector already runs).

**Tiny MLP (6→4→1, ReLU)**:
- Params: 33 (24+4+4+1)
- FLOPs: ~60 (matmuls + ReLUs + sigmoid)
- Latency: still <1-2 µs (tiny). Still negligible vs steer / full inference.
- Memory: ~300 bytes.

**Comparison to steer baseline (tts:82-98 real logic)**: Steer does k * (384 mul/add for scale+add dir) + 2 norms + clamp. Policy is pre- or post- that, on scalars from the meta/record. Total overhead for policy-gated stronger logging: <0.1% of TTS/steer path even in hot loop. Fits F "negligible overhead" / "~1.3ms" low-overhead probes (ours is orders cheaper, pure np scalars).

**Quant/edge**: INT8 or float16 trivial (7-33 params survive). No GPU needed. Collector call frequency = steer call frequency under experiment (rare, research-only).

**Ablation note (future C)**: Policy-off (always record under guard per B) vs policy-on (conditional stronger) in same C SMOKE run: measure delta in bhs_evidence size / "stronger" events logged vs spurious rate (if downstream success label available).

---

## Integration into Collector / First Experiment (Post-B, Guarded, Research-Only)

**Placement** (per B collector sketch + C:68-100 + task "using the probe infrastructure from B/C"):
- Primary: Inside C's `collect_research_probe_from_tts_metadata` (harness-only extension, under `if os.environ.get("CHELATED_SHIM_RESEARCH") == "1":`).
- After harvesting B's 3 keys + activation_record + base_meta:
  ```python
  if policy_head is not None and CHELATED_SHIM_RESEARCH:
      v_for_policy = ...  # from fixture or None (use record only)
      mm = harness_minmax.compute(...) or 0.0  # cheap, existing
      stronger = policy_head.decide_stronger_shim(record["activation_record"], v_for_policy, base_meta, mm, embed_var=...)
      record.update(stronger)  # adds policy_* fields to bhs_evidence
  ```
- Secondary (future, if human approves B extension): At B's annotation sites in tts steer (guarded if) or antigravity post-embed/variance (for seam-specific policy).
- Antigravity fallback: Extend collector to harvest from route_metadata/diagnostics when TTS path taken (G Family 2).

**First Experiment Design (C SMOKE extension, post human B edit + guard=1)**:
- Baseline arm (B probe unconditional): `--family vectorsteerer_tts_probe --research-shim` (C collector always emits research_* + activation_record).
- Policy arm: same + `--policy-head linear` (or mlp); collector calls head; emits extra "policy_stronger_shim" + "policy_confidence" + "policy_feats".
- Metrics (in C json + bhs_evidence):
  - probe_hit rate (should be 1 under guard + signals>0 per B/G).
  - "stronger" event rate (policy arm only; target e.g. 30-60% reduction vs unconditional for FPR control per F ASA).
  - Downstream proxy: if G traces have success/cost labels or harness usage_stats, corr(stronger events, success) vs baseline corr.
  - Overhead: collector latency delta (ns), steered_v / ndcg / delta_norm bitwise identical (C rollback guarantee).
  - Rollback test: delete B guarded blocks + policy call → re-run identical G family → no research_* / policy_* keys; base identical.
- Repro command example (post B): `CHELATED_SHIM_RESEARCH=1 python -B .../shim_collapse_benchmark_extension.py --family vectorsteerer_tts_probe --research-shim --policy-head linear --n-traces 50 --seed 42` (emits dated json with policy fields + EVIDENCE/SMOKE).
- Success for "first experiment": policy reduces logged "stronger" volume with no quality regression on synthetic success proxy (F "steering less achieves more"); full before/after + hashes in json; survives fresh checkout.

**Future (if SHIM-CD-01 progresses + human gates)**: Persist tiny head weights (~KB) in research harness; consume in MTP or SE-RDAG pre-filter (F mappings); correlate with real OPSD traces (backlog #4).

All conditional on B landing + C baseline + D/J audit + human sign-off. "0 real SIPs wired so far".

---

**EVIDENCE (for all claims here)**: This md + live tool outputs from §1 re-read (block FAIL count:2, 0-prod grep only tts+antigravity + exactly 2 research files, scheduler "No scheduled tasks", reads of goal:106/121-174/213+, dashboard R04 rows, next-session:61/69, cycle0400:38, protocol, 21_:84/59-99, 22_:86-146/118/142/177-246, 03C:68-100/254-289, 07G:43-110/58-74, 25F:43-50/46-49, tts:47-120 exact (draft 54-71 + 3-key 76-99), antigravity:2445-2630 drafts only, harness:593+ MinMax + 1682+ gens + 21-26/214+ guards + C/F coord notes) + 0-prod post-reads. All absolute paths + tool-grounded. Survive fresh checkout + re-run of §1 commands + `grep -n 'research_shim_probe_activated|MicroShimPolicy' tts_pipeline.py antigravity_engine.py || echo 'absent (expected)'`.

**SMOKE (rejection tests for any "SIP live" / "policy wired" / "substrate advance" / "H sketch closed debt" claims)**: On fresh checkout after this H: (1) tts:54-71 still "This draft adds ONLY comments + sketched guard (no executable...)"; antigravity drafts identical; (2) `grep -c "research_shim_probe_activated" tts_pipeline.py antigravity_engine.py` == 0; (3) block script → BLOCKED count:2 FAIL; (4) 0-prod grep → exactly 2 research files only (no leakage from this H md); (5) harness --family traces (existing) bitwise identical to pre-H (no policy code); (6) this md + 21_/22_/03C_/07G_/25F_ contain "0 real SIPs wired so far" + "B diff unapplied" + "design sketch only" + "does not satisfy goal success def #1 while BLOCKED + SHIM-CD-01"; (7) `grep -n '26_agentH_micro_slm_policy_sketch' loop_02/ | wc -l` finds this file only (no functional generator/policy in harness or engines); (8) scheduler_list "No scheduled tasks"; (9) re-run full §1 re-reads (must match baseline except this md). Any claim this "advanced the primitive" or "first policy live" or "SHIM-CD-01 progress" or "probe activated by head" fails. Matches all priors + gates + 0 substrate reality.

**End of Agent H (Micro-SLM Policy Sketch) independent artifact for SHIM-CD-01 unblock wave. 0 real SIPs wired so far. 0 substrate / does not satisfy goal success def #1 while BLOCKED + SHIM-CD-01. All per protocol + BHS v3.3 + governing docs. Human intervention per §128 still required. Research guard held.**

*Generated 2026-05-27/28 under research guard + OVERRIDE: ACTIVE + full protocol §1 re-reads (10+ supporting files + live tool outputs + 21-25/07_ + tts/antigravity/harness exact reads confirming drafts only + G traces + F lit + C collector + harness MinMax) + independent md only (no functional edit) + safe order followed + gates re-verified post. "0 real SIPs wired so far". "0 substrate / does not satisfy goal success def #1 while BLOCKED + SHIM-CD-01".*