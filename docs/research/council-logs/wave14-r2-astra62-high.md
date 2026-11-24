---
panel_type: panel-of-experts
artifact: docs/research/council-logs/wave14-r0-master-roadmap-and-pm6-formal-spec.md
run_date: "2026-11-24"
run_id: wave14-r2-astra62-high
challenge_depth: round-2-of-3
active_experts: ["gpt-6.2-astra (high)"]
persist_version: 1.1.0
---

# Wave 14 cross-model decisions

Reviewed R0 and all four R1 memos in full at checkout HEAD `4fae0b77`. Findings below distinguish source evidence, supplied Spark history and proposed acceptance measurements.

## C1 · Keep the 70B sweep

Carbon, Neon and Astra prevail over Fable’s cut. The cached Llama-3.3-70B-bnb-4bit makes the paired k=5/17 comparison affordable. The D10 test at `tests/test_wave11_d10_scale_proof.py:185` exercises CLI and grading contracts. Live accuracy, memory headroom and completion remain open. Bind model, prompts and actual inputs to hashes, publish both k values and serialize behind D9-B on Spark #2.

## C2 · Decide before sending client bytes

Reject Carbon’s conditional holdback. Indirect R3 can miss both early triggers. At token 63, a 32-token holdback can already have emitted 31 tokens of a match first detectable at 64. For a validated 64-token onset-to-verdict bound, score every stride-one window before releasing its oldest token. Keep all undecided bytes server-side during batching, queuing and Unicode decoding. Check short final windows before flush and discard pending bytes on failure or cancellation. For unbounded semantic delay, buffer the whole response through verdict. Claim “zero restricted bytes released for detected spans within the tested delay bound.” Report missed spans separately using captured client bytes.

Exact restricted scoring touches 480 rows, about 0.74/7.86 MB FP32 at 384/4096 dimensions. Gather their precomputed corpus-sidecar `r_d` values and specify the query-side `r_q` reference population before calibrating thresholds (D10:415–419). A full-corpus query neighborhood has separate retrieval cost. Treat <0.2 ms as a kernel hypothesis. Measure embedding, queue, scoring, holdback and client TTFT p95/p99 independently. Even instantaneous scoring incurs roughly 63 token intervals before first release on a full window.

## C3 · Retain Phase 3 as a falsifiable ablation

Fable’s legal objection stands. Score translation of protected expression separately from plot-fact summaries. Label summary blocking as enforcement of an explicit contractual or privilege policy where applicable. For each tested generator, hash-split confirmed works into 50 DEV and 60 held-out works, each with benign same-entity siblings. Lock thresholds on DEV. Compare direct MiniLM-384, English-centered and language-centered 4096→384 bridges, and a separately re-embedded native-4096 reference. Fit centers without test works. Report paired work-cluster bootstrap intervals for R3-versus-R5 AUC and per-language recall at fixed benign FPR. Publish negative transfer if observed.

## C4 · One amended D9-B protocol

Adopt Neon’s −1.00 false-abstain reward, accumulation 8, rank 16, learning rate 5e−6, β=0.05 and 60 steps. False abstentions now tie wrong determinate answers, removing their advantage in all-wrong, equal-reward groups. Partial-credit completions still matter (D9:1132). `artifacts/wave6_d4_trained_epistemic_grpo/wave6_d4_epistemic_grpo_manifest.json:516` records TRL 1.18.2. No installed trainer source is present here. D9:2140 inspects the signature but never records scaling support. Before training, inspect the installed field and instantiate its accepted disable value (`False` or `"none"`). Verify unequal-spread groups produce raw mean-centered advantages. If unavailable, change trainer-side normalization or pin a verified trainer. Reward rescaling alone cancels under standardization. Verify `loss_type="dr_grpo"` separately for constant length normalization.

Commit configuration, DEV history, data/script hashes and selection rules before launch. Bind resume keys to the amendment. Retire 42 as diagnostic. Use 137, 2026, 31415. The supplied dtype trace resolves F4. Cast saved bf16 adapters and base to fp32 and require finite parity <1e−4. Every ten steps assess DEV. Step 60 is primary, best eligible DEV secondary (determinate EM ≥ SFT−0.03, record none eligible). Freeze all selections and hashes before one out-of-loop VIRGIN_V2 evaluation. Prior baseline scores exist.

## C5 · Prove trace optimality and cyclic semantics

For compression, the proposed Frobenius theorem is false. Take X=diag(3,1), Y=(1,0)ᵀ. The thin SVD selects e₁, with residual norm 2. Feasible e₂ gives √2. Given SVD factors, prove trace maximization by bounding each diagonal entry of UᵀWV by one and weighting by nonnegative singular values. This distinguishes the fitted map from arbitrary isometries without proving SVD existence. Frobenius optimality follows when ‖XW‖ is constant in Frobenius norm, e.g. for row-isometric expansion or isotropic source Gram (bridge:317–348, untruncated branch).

Prove antimonotone Γ, monotone Γ² and T=lfp(Γ²), F=V∖Γ(T) on cyclic reducts (EK5:189–201). Prove deletion equivalence and specific stratum non-interference. Compare Lean, Python and independent recomputation after each mutation, including mutual REFUTE. Ban `native_decide` and `sorry`. Check exported dependencies with pinned `nanoda_bin` and an axiom allowlist. Projection proves anchored EMA drift ≤ρ, while directional stability needs a centered-norm floor above ρ. Freeze gate centroids. Test retrieval under bursts separately.

## C6 · Wave assignments and closure

| Wave | Assignment and acceptance |
|---|---|
| 11 / v6 | Retain sealed D6/D8. Repair D7 retry/validation on #1. Transfer D10-A shards over 200GbE to #3 after verifying hashes and resident headroom. On #2 run D9-B then D10-B. Use one clean candidate revision across hosts. |
| 12 / v6 | Cross-model adjudication of sealed Part VII run receipts, including failures and variances. |
| 13 / v6 | Reconcile registry, verify 45/45 bundles and public scan, then seal the evidence bundle. |
| 14 / v7 | Local NDA formal work can proceed in isolation. Integrate after v6. Run cyclic conformance before D11 Phases 0–3. Compare bare, prompt, deletion, n-gram-only, Gate-1-only, Gate-2-only, raw cosine without ZeroMu/CSLS, white-box steering/logit suppression and full gates. Match generator, decoding budget and operating point. Freeze thresholds before budget-matched adaptive R2b and report per-work success. |

PM-1b stays SCOPED/OPEN. Registry:367 reports 95.00% versus 95.83% NoEK5, 2.42× compaction versus flat, and KU +29.49 points versus MiniLM. Non-inferiority needs a margin and interval. Target <45 s measured offline for `verify_all_offline.py` to rehash bundles, verify signatures/chains, recompute headline statistics, replay conformance fixtures and scan public outputs. Time on a named CPU with cache state recorded. Expose any omitted checks. Human PM-4b remains the release gate.
