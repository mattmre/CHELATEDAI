# Wave-4 R2 judgment — Grok 4.7 (xhigh)

Four R1 memos only. **MODIFY** on J1–J7. Direction is right; the literal patch can still fit the reported split.

| ID | Vote | Accept | Guardrail |
|---|---|---|---|
| J1 | MODIFY | Drop idx overlap and substring pay. Format ≤0.05 on frozen WFS exact-match. | Bonus only if the span or closed abstain token already scores 1 (formatted error ≤0.05; margin is 0.25). Reject negation. No `compute_alternating_wfs_oracle` in the policy. Hold out entity, value, and template family, not merely the seed. Report reward and exact-match apart. No GRPO delta without a checkpoint. |
| J2 | MODIFY | Remove `c["entity"]` and `c["slot"]` from `run_r6_pm2_labelfree_closure.py`. Flags = gold-key reads. Rename to `DualFit_1Float_FixedMuX`. | One proposer-v2 text read for every arm. Empty or tied top-k → UNKNOWN and abstain. No metadata fallback. Do not retune k at Native=1.000. Keep Tango LCB −0.0572. |
| J3 | MODIFY | ID join; DEV z-score and RRF with CSLS and raw merge; per-k_os curve; two-sided 2.5% CIs. | Cosine error 0 per BEIR id, plus an id digest. DEV-only scalers. Report all four. No winner pick on n=86. Recall@10 vs exact top-10, not nDCG. QueryAffine 0.0 is a bug suspect. No FiQA or NFCorpus imputation. Keep the old 5th-percentile LCB. |
| J4 | MODIFY | Clean LME +7.8 pp is DualGran. EK5 Δ=0. One Opus R1–R4 test. | One (τ, δ, ⌈k/2⌉) on LoCoMo 0–4 and a frozen LME DEV hash. Claim convs 5–9 and the LME complement only. Full N includes DEV. Cast names, not qrels. No forced abstention or grader cue. DEV-frozen budgets. Do not refit 0.80/0.55/0.62/0.42. Print `assignment_gates_pass=false` with +25.6 pp. |
| J5 | MODIFY | Receipt labels; `item_id`+`content_hash`; rubric parity; cluster and stratum CIs; κ=0.8400 on n=200 only. | 4096-d; 99.77% mean and 90% min at ef=32; 100% at 64. Not per-depth nDCG 0.5252. Hide dataset and qid. No placeholder items. Preference rule on. YES+NO is a bad parse. Cluster bootstrap keeps judge pairs. Do not pool 800 structured rows. CI crosses 0.80. Disclose the Nemotron self-judge. |
| J6 | MODIFY | {attack, correct, abstain, other} and Clopper–Pearson per family. Lead with co-design and trusted tier. | Complete linkage is pre-declared. Single linkage is not chosen by ASR. Fit τ off eval templates, FP cap frozen, one look. Allow cross-value copies. Abstain=1.0 is not success. Drop pooled [0, 0]. |
| J7 | MODIFY | Eight comparators, PRE-RELEASE banner, and ~−1 pp vs uncentered Procrustes at std 0.038, beside 16/18 OOD, 12/12 multi-hop, and 4-byte HNSW-DOT. | CSLS +29 pp is vs raw mix until J3. Also print failed gates, EK5 Δ=0, Cat-5 0.39→0.13, PM-2 ceiling, D1 n=60 vs Mem0, 99% synthetic 1M, one domain, no GRPO weights, D3 co-design, κ not shown above 0.80, TRA-1 is not a vec2vec defense, 1023× sidecar bytes, and both OOD losses. |

Neon’s bare CSLS and HNSW line stays qualified by Opus and Sol. J1–J6 are code under these freezes. J7 prints only frozen claims.
