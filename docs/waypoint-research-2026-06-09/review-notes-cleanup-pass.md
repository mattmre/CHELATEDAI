# Cleanup / Review Pass Notes (running log)

## 🔬 2026-06-30 — CHELATION REFRAME ("we mis-benchmarked it") + expert panel convened  [NEXT-SESSION REFERENCE]

**Operator asked: don't give up on the chelation idea — did we miss something?** Claude's analysis (logged
here for next session): **YES — a scoping error, not a falsification.**

**The miss:** the query-encoder-UPGRADE drift arena is *adversarial to chelation by construction.*
- Encoder-upgrade drift between two frozen transformers ≈ a **global linear/affine transform** (Procrustes
  bounds; our own "MLP doesn't beat ridge" result confirms the recoverable structure is linear). Recovering a
  large GLOBAL rotation is exactly what a global linear map is optimal for.
- Chelation is by design **bounded / near-identity / local / unsupervised-triggered** — built for SMALL,
  HETEROGENEOUS, within-model touch-ups with NO paired target. That is the *opposite* regime. The bound alone
  forbids the large rotation the drift needs. So chelation loses *by construction* — this does NOT falsify the
  chelation hypothesis; it says drift-recovery was the wrong benchmark.
- **Chelation's true home turf (never tested):** within-model **semantic collapse** (anisotropy / hubness /
  isomer-collapse that grows in long-lived stores), where (a) the pathology is heterogeneous, (b) there is no
  new encoder / no paired target, (c) the baseline is GLOBAL whitening/ZCA — which a *local* corrector can
  plausibly beat *because* the collapse is heterogeneous. Uses the isomer/topology machinery for its LEGIT role.

**Candidate untested paths (rank + odds TBD by panel):**
- **P1** clean **local (per-cluster) paired maps vs one global paired map** — the fair "local > global" test
  never actually run (post-bank was crippled by weak anchor-InfoNCE supervision). Cheapest, decisive. ~1 hr.
- **P2** **relational/neighborhood correction** — feed the surviving kNN-graph, not just each point vector, to
  try to exceed the point-wise 84% linear ceiling (all prior art = point maps). Higher risk/reward.
- **P3** **detection-gated hybrid** — unsupervised topology/isomer signals for DETECTION+ROUTING + minimal
  paired anchors only where they fire.
- **P4** pivot the arena to within-model anisotropy/hubness/multilingual collapse (whitening = baseline).

**PANEL (2026-06-30):** dispatched a self-contained briefing (`panel/briefing-2026-06-30.md`) to a 3-model
adversarial roundtable to answer: (1) revival potential, (2) is the reframe technically correct or a
rationalization, (3) enough evidence to kill the drift-corrector hypothesis while keeping within-model-collapse
alive? **Grok 4.5** (novelty lens) + **Codex gpt-5.6-sol xhigh** (technical-rigor lens) ran; **Gemini
tier-blocked** (Code-Assist-for-individuals deprecated → needs Antigravity migration or an API key). Claude =
3rd panelist + chair. Outputs → `panel/out_*.txt`; synthesis → `panel/roundtable-synthesis.md` (pending).

---


## ✍️ 2026-06-30 — 4-AGENT REVIEW PANEL (wgi5t1j0k) applied to the paper

Panel (narrative / technical / adversarial / positioning) → consolidated edit list. APPLIED:
- **#1 NUMBER RECONCILIATION (critical):** re-ran the FULL ladder + generality on ONE consistent
  eval-split (`scratchpad/ladder_evalsplit.py`, `rank1_generality.py`). Canonical SciFact set:
  C3a 20%, C4a 25%, lstsq 67%, ridge 84.4%, Procrustes 82.6%, MLP 81.2%, low-rank 49%, α{0.9,2.1,11.8,58.7,84.4};
  generality eval-split: mpnet SciFact 84.4/2.1, NFCorpus 66.4/2.2, bge SciFact 78.9/0.0, NFCorpus 74.3/2.1.
  → **ridge/C3a = 84.4/20 = exactly 4.2×.** Fixed §5.5 table (removed the 0.131 "living post-bank"
  CONFLATION — that was C5, not C3a), the decomposition (+59 supervision / −5 bound), generality prose,
  ALL 3 figures, propagated 4.2× everywhere, fixed §5.3 0.159→0.161 + 25×→19%-gap.
- **Scaffolding removed (critical C3):** working-draft banner, §5.2 ⚠️ regenerating banner, the
  finalization note (→ clean reader-facing pipeline note), the Drafting-notes section.
- **§1 three-hazards count fix** (harness = lead-in, hazards numbered 1–3). **Abstract restructured**
  (leads with the punchline). **Title** retitled to foreground Hazard 3. **Proposition 1 hedged**
  (no-descent/stationary, not "global optimum"). **GloVe + text-residual overclaims hedged.**
  **§2 "What is new vs Drift-Adapter" enumeration box added** (adversarial reviewer's #1 defense).

**REMAINING (flagged, not yet done):**
- **§3 (System Under Test) is still a STUB** (M2) — needs the adapter forward maps, the exact bound
  formula + its floor, and the controller drift→temperature schedule. Biggest remaining writing task.
- §5.1–5.3 still report the harness/campaign pipeline (different from §5.5's self-contained eval-split);
  the pipeline note explains it, but full reconciliation = re-run campaign conditions on the matched eval.
- minors: m6 oracle naming (C2 vs C2O), m10 relocate S1 from §7 to its own §5.6 + 2-seed, m11 move the
  oracle-supplies-X0 unification to §4.3, m15 rename heading "cross-encoder"→"cross-model-version",
  m16 strengthen Reproducibility (commit/SHAs/repro command) + soften self-deprecating Limitations.
- DO-NOT-DO (panel): don't soften S1's "dominated" verdict; don't invent a two-splits story for 85.5
  vs 84.4 (use 84.4); don't manufacture a 30% upper bound.

---


## ✅ 2026-06-30 — CITATION VERIFICATION (workflow wz6wva8wd, 18 items checked live) + figures cleaned

**All 18 citations verified against live arXiv/source records.** Corrections APPLIED to the draft:
- **BC-family framing was WRONG (fixed §2):** "each retrains the new encoder" is false for FCT/XBT/OCA.
  Correct taxonomy now in §2 — (i) retrain-new: BCT (2003.11942) + BT² (2211.03989); (ii) transform-old:
  FCT (2112.02805, side-info + old→new map, new encoder trained normally); (iii) map/align on frozen:
  XBT (2405.14715, new→old projection) + OCA (2408.08793, orthogonal alignment layer) — (iii) is CLOSEST
  to us, NOT a contrast pole. Added Iscen feature-adaptation (2004.00713, old→new MLP = our direction).
- **SPFresh arXiv id added: 2410.14452** (was "SOSP 2023" only).
- **Procrustes-Bounds:** full title "When Embedding Models Meet: Procrustes Bounds and Applications";
  §5.5 Fig-5 "orthogonal-beats-linear" claim HEDGED (flagged for line-confirmation — not line-verified).
- **QDC mean-translation correction CONFIRMED** from its equations + source code (Eq4 mean of f_t−f_{t−1},
  Eq6 subtraction; "simple vector subtraction without re-indexing"; re-index is the underperforming baseline).
- doc2query original id = **1904.08375** (correct, used in §7). vec2vec (Gao) = 2306.12689 (NOT the 2025
  Jha/Morris "Universal Geometry" vec2vec — don't conflate). query2doc 2303.07678, Promptagator 2209.11755,
  Doc2Query-- 2301.03266 all ✓.

**⚠️ MANUAL CHECK BEFORE CAMERA-READY (verifier could not fully confirm):**
- **docTTTTTquery: claimed id 1910.14424 is WRONG** (= "Multi-Stage Document Ranking with BERT"). The
  docTTTTTquery tech report was never on arXiv; if cited, use doc2query 1904.08375 or the Waterloo PDF/GitHub.
  (We currently cite only doc2query in §7, so no live error — just don't add 1910.14424.)
- (c1) Iscen/Xing(IALP IEEE 10661153)/Schönemann(Psychometrika 1966) — non-arXiv records not page-level re-fetched.
- (c2) Procrustes-Bounds Figure 5 exact result — line-confirm against the PDF.
- (c3) QDC exact numbers (49.4→53.5 etc.) + drift magnitudes — re-pull from PDF tables if cited specifically.

**Figures (paper-draft/figures/) regenerated PRISTINE** — constrained_layout, figure-centered titles that
fit, no clipping/overlap, notes moved to markdown captions: fig1 recovery ladder, fig2 bound curve, fig3 generality.

---


## 📚 2026-06-30 — DRIFT-ADAPTER LIT-MAP (workflow w11qc8usw; paper fetched live) — §2 to-do + corrections

Fetched the ACTUAL Drift-Adapter paper (arXiv:2509.23471 = EMNLP'25 main #805, sole author H. Vejendla).
Key facts to USE: same MiniLM-L6→MPNet swap as our arena; 3 correctors = OP / low-rank-affine / residual-MLP
(+ Diagonal Scaling) on ~20k random paired embeddings; headline **95–99% recall recovery, MLP BEST**;
**their own GloVe→MPNet severe-drift run = only 71.5%** (their evidence of our linear-ceiling thesis).
Direction is the DUAL of ours (they map query→OLD to keep old index; we move queries to the NEW encoder).
**Discrepancy to flag:** they find MLP best; we find regularized-LINEAR best (4.5× over elaborate corrector)
— attribute to drift severity (catastrophic vs their mild) + ridge regularization, not a universal claim.

**The 4-paper core our §2 must engage:** Drift-Adapter (2509.23471) + Procrustes Bounds (2510.13406) +
QDC (2506.00037) + **ERA "Align then Train" (2604.03403)** — the one CONFIRMED forward-citation of
Drift-Adapter (query-side 2-stage align+adapt, no re-index, 126 tasks; MUST ADD + differentiate: we show
trivial ridge recovers most + supervision-TYPE dominates, an ablation they don't isolate).

**ADD (lineage we were missing):** the backward/forward-compatible-representation family — BCT (2003.11942),
FCT (2112.02805), BT² (2211.03989), XBT (2405.14715), OCA (2408.08793), CoReS/Stationary, RACT (2201.09724)
[they RETRAIN the new encoder = contrast pole to our frozen-both post-hoc]; vec2vec (Gao, 2306.12689, the
embedding-translator ancestor); the foundations Drift-Adapter cites (Iscen ECCV'20 feature-adaptation =
OUR doc→new direction, Xing IALP'24 Procrustes-Hungarian, Schönemann 1966); supervision-source cluster for
S1 (doc2query, query2doc 2303.07678, Promptagator 2209.11755, Doc2Query-- 2301.03266); SPFresh (systems, cite once).

**⚠️ CITATION CORRECTIONS (load-bearing — fix in §2/§5):**
1. **QDC (2506.00037)** is a single constant mean-drift TRANSLATION (rank-0 affine offset) on MILD
   continual-FT drift (nDCG 49→54), NOT a ridge/Procrustes map and NOT a catastrophic swap. Don't lump it
   into our "trivial linear" family; frame as "simplest possible, ours is strictly more expressive." Its
   re-index-underperforms-QDC result IS valid ammo for our re-embed hazard.
2. **Procrustes Bounds (2510.13406):** their Fig 5 finds orthogonal Procrustes BEATS unconstrained linear
   when upgrading to a stronger query model — OPPOSITE to our "bound costs ~+9." Reconcile explicitly
   (regime + our ridge regularization); do NOT assert linear>orthogonal as universal (reviewers will catch).
3. **Corrector Networks (2409.01890):** setting MISMATCH — it corrects document/target embeddings DURING
   training, not inference-time query-upgrade recovery. NOT a baseline to beat. Cite as the method-family
   ancestor of our elaborate corrector (name-collision to distinguish), not same-setting prior art.

**VERIFICATION STATUS:** only Drift-Adapter + ERA (and ERA's citation of Drift-Adapter) verified LIVE this
session. QDC/Procrustes-Bounds/Corrector-Networks/vec2vec/BC-family = second-hand (prior task). Iscen/Xing/
Schönemann/SPFresh/doc2query-id = UNVERIFIED (from bibliography, never fetched). **Re-pull all before camera-ready.**

**§2 positioning sentence (agent's, endorsed):** "Drift-Adapter and its concurrent neighbors (Procrustes
Bounds, QDC, ERA) all show a cheap post-hoc map realigns upgraded encoders without re-indexing and report it
as near-solved success; we instead expose the linear post-hoc ceiling under *catastrophic* query-encoder
upgrades, show a trivial regularized linear/Procrustes map already captures most recoverable drift
(supervision-type ≫ orthogonality-bound ≫ residual), and surface three hazards the neighborhood leaves uncorrected."

---


## 🏁 2026-06-30 — S1 CLEAN FORM TESTED (true doc2query) → WORKS but is DOMINATED. Novelty question CLOSED.

`scratchpad/s1_doc2query.py` — used cached **Qwen2.5-0.5B-Instruct** (GPU free, internet available) to
generate a TRUE search question per fit-doc (genuine questions, distinct from doc text, e.g. doc on
endotoxemia → "What is the role of the GLUT1 isoform in endotoxin-induced glucose influx?"), embedded
with the new encoder, paired with cached old vectors, fit ridge. SciFact, 600 gen pairs:
- floor 0.009 · anchor-InfoNCE corrector ~0.161 (20%) · **S1 true doc2query 0.643 (81.0%)** ·
  oracle-pair ridge ceiling 0.678 (85.5%).

**So the clean form WORKS (81%) — it is NOT just partial re-embedding (the questions are genuinely
different text). BUT it is STRICTLY DOMINATED, which kills it as a useful method:**
- The simpler **oracle-pair** (fit ridge: cached_old_doc → new_model(doc_text)) recovers **85.5%**, is
  **also old-model-free** (cached old vectors; never needs f_old) and **also label-free**, and needs
  **no LLM**. It = Drift-Adapter with cached old vectors.
- S1's claimed novel advantages collapse on the cost/result: (a) "drops the legacy encoder" — the
  oracle-pair already does. (b) "drops the corpus re-embedding pass" — FALSE: generating a question
  per doc is an LLM pass over the corpus, *more* expensive than embedding, not a drop. (c) "aligns to
  query→doc geometry" — empirically recovers LESS (81% < 85.5%). **No operational scenario favors S1:**
  generating a question requires reading the doc; if you can read the doc you can re-embed it with the
  new encoder (cheaper + higher recovery). S1 is a more-expensive, lower-recovery variant of the
  oracle-pair/Drift-Adapter.

**STEELMAN VERDICT (workflow wdckvfrki, ran before the S1 result):** all 4 steelmans landed
"survives-narrowed"; every one names Drift-Adapter (2509.23471) as the dominant threat. No new
MECHANISM (anticipated). It flagged S1 as "the one surviving genuinely-novel claim, untested" — but
that was conditional on S1 dropping the corpus pass + winning on query-geometry, **both now falsified
by the run + the cost analysis above.** What genuinely survives per the verdict: a **methodology /
negative-result paper built on Drift-Adapter** — the *catastrophic-regime* ~4.5× corrector-vs-lstsq
gap (Drift-Adapter stays in the 95–99% mild regime), the orthogonal two-factor decomposition
(supervision-type ~+60 ⟂ bound ~+9 ⟂ residual ~0), and the three named hazards. Must cite Drift-Adapter
as the baseline it refines, not precedes.

**DEFINITIVE ANSWER to "any new novel approach, or just reconfirmation?":** No new *useful* method —
S1, the last candidate, is now tested and dominated. The genuine (narrow) novelty is the methodology +
hazards + the catastrophic-regime decomposition: a real but derivative methods/negative-result paper,
NOT a new mechanism. Novelty question is CLOSED.

---


## 🔧 2026-06-29 — NFCorpus eval diagnosis: campaign numbers are harness-internally inflated

Chasing the airtight NFCorpus row, found: my self-contained eval gives NFCorpus base **0.333** on the
1200-doc slice; the harness reports **0.598** on the SAME slice. Binarizing my NDCG to match the
harness (it scores binary `score>0`, not graded) barely moved it (0.329→0.333) — so multi-grade qrels
were NOT the cause. My 0.333 **matches published MTEB MiniLM-L6/NFCorpus (~0.33–0.40)**, so the harness
0.598 is inflated by an internal step (centering / quantization / id-mapping). **Implication: the
campaign's NFCorpus absolute numbers (base 0.598, C2O 0.612, C3a 0.052…) are on an inflated scale —
cite with care; do not mix with my-eval numbers.** SciFact is unaffected (my 0.789 ≈ harness 0.819).
Qualitative findings unchanged in every eval. Clean fix for the paper's NFCorpus row: score the
ridge-corrected store through the harness's own `evaluate_engine_with_query_vectors` (both corrector +
fair-baseline on the identical pipeline) — deferred (deep store-mutation work); SciFact carries the
airtight headline meanwhile.

---


## ✅ 2026-06-29 — S1 (HyDE self-pairs) WORKS — the night's POSITIVE novel result

`scratchpad/s1_hyde.py`, exact SciFact arena. Pseudo-query = doc-opening text, embedded with the NEW
encoder, paired with the doc's cached OLD vector → fit ridge old→new. **No old model. No full-doc
re-embedding.** Recovery of oracle gap:
- anchor-InfoNCE (our corrector): ~20%
- **S1 first-12-words: 68%** · first-25w: 79.5% · first-50w: 84% (≈ oracle-pair ceiling 85.5%)

**This is a real, positive, backfill-free, old-model-free drift-recovery method** that plugs
Drift-Adapter's explicitly-named "f_old unavailable" gap — and it nearly reaches the linear ceiling.
Even a 12-word pseudo-query gives 68% (~3.4× our corrector).

**HONEST CAVEATS (must state in the paper):**
1. The pseudo-query here is the doc's OPENING TEXT (a degenerate generator — no doc2query model was
   cached offline). As K→full-doc this converges to *partial re-embedding with the new model*, so the
   high-K numbers (50w→84%) blur into "re-embed part of the doc." The GENUINE cheap signal is the
   LOW-K result (12w→68%): a short doc-derived text suffices.
2. **⚠️ Novelty is THINNER than the synthesis assumed — caught on implementation.** The synthesis
   claimed S1 plugs Drift-Adapter's "f_old unavailable" gap. But the oracle-pair itself (cached old
   vec + re-embed doc with the NEW model) is *already* old-model-free given cached vectors — you never
   need f_old, you re-embed only with f_new. So "old-model-free" is NOT S1's novel advantage; the
   obvious cached-vec approach already has it. S1's ONLY genuine sliver is using a SHORT doc-derived
   text (doc2query question) INSTEAD of re-embedding the full doc — a cost/cleanliness optimization.
   And my first-K-words proxy blurs into "re-embed part of the doc" as K grows (50w→84% ≈ the
   oracle-pair). The cleanest signal is first-12w→68% (a genuinely short, claim-like prefix on SciFact
   abstracts). A TRUE doc2query (generated *questions* ≠ doc text) is the only clean test — and it's
   **BLOCKED offline (no generation model cached)**. So S1 is "promising signal, clean novelty UNTESTED."
3. Single dataset (SciFact), 1 seed so far. Generalize to NFCorpus + bge + seeds before a headline.

**Paper status:** S1 is the genuine novelty extension — a positive method section. Frame honestly as
"backfill-free old-model-free recovery via pseudo-query self-pairs; doc-opening floor recovers 68–84%;
true doc2query is future work." This is the one thing the night produced that is a *method*, not a hazard.

---


## 🔭 2026-06-29 — NOVELTY-EXTENSION HUNT (workflow wra0bbcoa, 26 agents) — S1 is the one live method

Full output `tasks/wra0bbcoa.output`. **Brutal bottom line: the core phenomenon (supervision-bound,
~85% linear ceiling) is LARGELY ANTICIPATED** — Drift-Adapter (2509.23471) already publishes the
paired-sample saturation curve in the IDENTICAL setting and concludes supervision (not capacity) is
the bottleneck; Procrustes Bounds (2510.13406) gives the theorem; our Hazard 2 (unsupervised →
identity-optimal) is a corollary of domain-translation identifiability no-go theorems (2401.09671,
2605.17918) and Drift-Adapter's stated paired-data necessity. **So: methodology paper, cite heavily.**

**The ONE genuinely-novel positive method — S1 (HyDE/doc2query self-pairs):** for each cached doc,
generate a pseudo-query from its text, embed it with the NEW encoder, pair with the doc's cached OLD
vector → real (old-space, new-space) pairs to fit the ridge/Procrustes map — **without the old model
and without re-embedding docs.** Plugs Drift-Adapter's explicitly-named "f_old unavailable" gap; no
prior art uses corpus-generated pseudo-queries as cross-encoder-version alignment supervision. Result
is publishable either sign (a positive backfill-free method, OR an honest quantification of the
query-doc supervision gap). **Decision: run S1 tonight** (the one load-bearing experiment).

**Operator sketches re-examined (honest):** σ-signaling / slot-warp / stacked-gates / replaceable-
subspace = anticipated correctors (NoisyNet/MoE-LoRA/Drift-Adapter re-skins), DEAD (architecture is
provably not the bottleneck). The ONLY live sketch sliver: "localized invariance / relative-
representations" as an UNSUPERVISED-ceiling MEASUREMENT (how close to 85% can label-free alignment
get on encoder-upgrade drift?) — a one-run panel, not a method. Secondary side-panels: anisotropic
spectrum-derived bound (S2, the +9pt lever); mutual-kNN self-pairs (S3, likely flat-negative in the
catastrophic regime).

---


## 🌐 2026-06-29 — GENERALITY: the finding holds across a SECOND encoder family (paper-hardening)

`scratchpad/rank1_generality.py`. Ran the fair-map + bound-ablation across different swap (new)
encoders. **The pattern is robust — not specific to MiniLM→mpnet:**

| new encoder | dataset | floor | oracle | ridge (unbounded) | bounded α=0.1 |
|---|---|---|---|---|---|
| mpnet (control) | SciFact | 0.009 | 0.792 | **85.5%** | 4.4% |
| mpnet | NFCorpus | 0.013 | 0.410 | 75.0% | 2.6% |
| **bge-large-en-v1.5** (diff family, 1024-d) | SciFact | 0.010 | 0.860 | **81.4%** | 0.7% |
| **bge-large** | NFCorpus | 0.017 | 0.448 | **76.3%** | 1.7% |

Across BOTH new-encoder families and BOTH datasets: (1) drift is catastrophic (floor ~1–2% of base);
(2) a trivial unbounded ridge map recovers ~75–85% of the oracle gap; (3) the near-identity bound
(α=0.1) collapses recovery to <5%. **The trivial-baseline + bound findings generalize.** (nomic-embed
skipped — needs `trust_remote_code`; bge is sufficient for the generality claim. A 2nd new-encoder
family is enough to harden the paper.)

---


## 🎯 2026-06-29 — CAPSTONE: THE BOUND IS THE SMOKING GUN (fair-map + α-ablation sweep, both datasets)

`scratchpad/rank1_sweep.py`, exact campaign arena, BOTH datasets, real runs. Two decisive results:

**(A) A properly-regularized fair map nearly closes the gap:**
| map (SciFact, oracle gap) | recovery | (NFCorpus) | recovery |
|---|---|---|---|
| ridge λ=1.0 | **85.5%** | ridge λ=0.1 | **80.3%** |
| Procrustes (orthogonal) | 85.2% | lstsq-full | 80.6% |
| lstsq-full | 71.6% | Procrustes | 76.9% |
| lowrank-64 | 54.7% | lowrank-64 | 57.8% |

→ The drift is ~80–85% recoverable with a one-shot regularized LINEAR map. The residual ~15–20% is
the linear ceiling (oracle re-embeds → captures nonlinearity). **Drift was never hard.**

**(B) THE α-SHRINK ABLATION IS THE SMOKING GUN — the near-identity BOUND is the specific cause of
our corrector's failure:**
| shrink α (map = I + α·(W−I)) | SciFact recovery | NFCorpus recovery |
|---|---|---|
| 0.05 (≈ our near-identity bound) | **2.8%** | **1.6%** |
| 0.1 | 7.4% | 3.3% |
| 0.25 | 34.7% | 26.8% |
| 0.5 | 70.7% | 63.2% |
| 1.0 (unbounded) | 71.6% | 80.6% |
| **campaign C3a/C5 (bounded)** | **~16%** | **~30%** |

**Our corrector sits EXACTLY on the bound-crippling curve.** The near-identity bound — *the core
ChelatedAI premise* ("small, safe, bounded corrections near identity to preserve base quality", the
`BoundedAdapter`, the whole chelation metaphor) — is the QUANTIFIED, PROVEN root cause of failure.
Catastrophic encoder-upgrade drift requires a LARGE full-rank realignment; the bound that guarantees
base-quality safety **guarantees recovery failure**. This single fact explains: H5 (bounded post-bank
→ ~16%), C5==C5s (the lifecycle is irrelevant once the bound caps recovery), and why σ/slot/living
variants were all doomed (every one is a bounded-correction variant).

**MLP confirms the ceiling (`scratchpad/rank1_mlp.py`):** a residual MLP recovers 81.9% — does NOT beat
ridge's 85.5%. So **~85% is the hard POST-HOC ceiling**; nonlinearity/capacity beyond a regularized
full-rank linear map is wasted. The irreducible last ~15% is the oracle's advantage (re-embedding raw
text — info not present in cached old vectors). Complete recoverability picture: **floor 0% → bounded
near-identity (ours) ~16% → regularized linear ~85% → oracle 100%; the 16%→85% jump is purely the bound.**

**⚠️ SELF-CORRECTION (do not let the above overstate the bound).** "The bound is THE smoking gun" is
TOO STRONG. The α-ablation shrank a GOOD (lstsq) map toward identity → so it shows the bound is
*sufficient* to wreck a good map, NOT that the bound is our corrector's dominant deficit. The real
decomposition, from REAL harness numbers (C3a/C4a are our bounded/unbounded anchor-InfoNCE corrector;
SciFact recovery of oracle gap):
| | bounded | unbounded |
|---|---|---|
| **our supervision** (anchor relevance + InfoNCE) | C3a **16%** | C4a **25%** |
| **paired-embedding least-squares** (Drift-Adapter) | (tight α=0.05) 2.8% | lstsq/ridge **72–85%** |

- **Supervision effect (DOMINANT, ~+60pts):** holding *unbounded*, switching our anchor-InfoNCE → paired
  least-squares = C4a 25% → 85%. Our corrector uses a much weaker/indirect signal (make relevant q–d
  pairs close) than the direct "map each doc to its new-space position" regression.
- **Bound effect (SECONDARY, ~−9pts for our corrector):** C4a 25% → C3a 16%.
- The α-curve's dramatic 2.8% is what a tight bound does to a GOOD map — it is NOT where our corrector
  actually sits causally (it lands at 16% via weak-supervision-25%-then-mild-bound, a coincidence of
  magnitude, not pure bound).

**Corrected precise claim:** our corrector underperforms a proper Drift-Adapter by ~70pts on TWO axes —
PRIMARILY weak supervision (anchor-InfoNCE vs paired-embedding least-squares), SECONDARILY the bound.
A tight bound alone can wreck even a good map (α-ablation), and the chelation "small bounded near-identity
correction" premise is genuinely ill-suited to catastrophic drift — but the honest headline is
"wrong supervision signal + bounded, vs the trivial paired-least-squares baseline," not "the bound did it."
This is still a strong, publishable trivial-baseline / methodology finding. (Also: campaign NFCorpus baseline 0.598 was a 1200-subset
inflation — real MTEB MiniLM/NFCorpus ≈ 0.397, which this run reproduces.)

---


## 💥 2026-06-29 — RANK 1 IN-ARENA (real, EXACT campaign arena) — corrector beaten 4.5× by one line of numpy

`scratchpad/rank1_inarena.py` — reuses the harness's EXACT `QueryEncoderDrift` (same seeded
projection C2O uses) on the same 1200-doc/100-query SciFact slice. **Sanity REPRODUCES the campaign
arena exactly:** baseline(old/old)=**0.8144** (campaign 0.819), drift_floor=**0.0089** (campaign C0
0.006), oracle C2O=**0.7920** (campaign 0.813). Same arena, confirmed.

**RESULT:**
| corrector | NDCG@10 | % of oracle gap recovered |
|---|---|---|
| drift floor (none) | 0.0089 | 0% |
| **TRIVIAL `np.linalg.lstsq`** (fit old-doc→oracle-target on HALF the docs, apply to all) | **0.5694** | **72%** |
| campaign C3a / C5 / C5s (bounded adapter, living bank, all of it) | 0.131 | **~16%** |
| oracle C2O (re-embed) | 0.7920 | 100% |

**FINDING (definitive, real, sanity-matched): our entire corrector line is beaten 4.5× by a single
least-squares call.** The drift is NOT hard to recover — a plain 384×384 linear map fit on a sample
of paired embeddings (the standard Drift-Adapter setup) recovers 72% of the oracle gap. Our corrector
got 16% because it is a *bad linear map*: bounded near-identity + few relevance-anchors + InfoNCE,
instead of a direct least-squares fit on paired embeddings. The "catastrophic barely-recoverable
drift" narrative was an artifact of (a) the frozen-C0 straw-man baseline and (b) our underpowered
corrector — NOT a property of the drift.

**This confirms the opportunity scan's Rank 1 prediction with real numbers, and it is the sharpest,
most honest result of the whole effort.** It also kills any residual "corrector" contribution: there
is none — `lstsq` wins. The surviving + now SHARPENED paper claims:
1. The query-encoder-upgrade ARENA (methodology).
2. C2-oracle degenerate-baseline hazard.
3. Supervision-signal / identity-collapse hazard.
4. **NEW — the trivial-baseline hazard:** elaborate correctors (bounded adapters, living/annealed
   banks, MoE post-banks) underperform a one-line least-squares map 4.5× in the exact same arena;
   the frozen-C0 straw man hid it. *Always benchmark the trivial linear map.* This is itself a clean
   brutal-honesty case study and arguably the most useful thing the project produced.

---


## 🔬 2026-06-29 — RANK 4 DIRECTION TEST (real run) — asymmetry DEAD + a bigger discovery

`scratchpad/rank4_direction.py`, full SciFact corpus (5183 docs, 300 test queries), real
MiniLM/mpnet, least-squares maps fit on a held-out half-corpus. **Sanity PASSED**: old-vs-old
NDCG@10 = **0.648** = the published MTEB MiniLM-L6/SciFact value (~0.645) → eval is correct, and the
**campaign's 0.819 "baseline" was inflated by its 1200-doc subset** (a separate methodology note).

Results:
- baseline old-vs-old: **0.648**
- **NEW→OLD (Drift-Adapter easy dir): 0.543 (84%)**  ·  **OLD→NEW (our "hard" dir): 0.557 (86%)**
- **ASYMMETRY = −0.0136 (negligible; if anything old→new is slightly EASIER).** Geometry residuals
  near-symmetric (o→n 0.617, n→o 0.646), linear CKA 0.83.

**FINDING 1 — the asymmetry is an ARTIFACT (FALSIFIED).** Both directions recover ~85% of baseline
with a TRIVIAL least-squares map; the "old→new fundamentally harder" upside (the scan's one
conditional novelty) is **dead**, exactly as the opportunity scan predicted. The query-vs-doc
asymmetry was a fixed-index framing, not a geometric law.

**FINDING 2 (bigger) — a trivial linear map recovers ~85% of the realistic encoder swap, vs our
campaign corrector's 16–21%.** A plain `np.linalg.lstsq` old→new map ≈ Drift-Adapter's low-rank
affine, and it ≈ matches Drift-Adapter's 95–99% ballpark. This means our campaign's "catastrophic,
barely-recoverable drift" (0.006 floor, 16% recovery) is NOT a fundamental ceiling — it is some
combination of (a) the campaign's **seeded random projection** inflating difficulty and/or (b) our
**bounded / few-anchor / InfoNCE corrector being a badly underpowered linear map.** A linear map
absorbs a linear projection, so the strong prior is (b): **our corrector massively underperforms a
trivial baseline.** This is the scan's Rank 1 prediction, now strongly indicated and needing the
in-arena confirmation (fit lstsq IN the campaign's exact drift, compare to C3a/C5).

**Net:** the one remaining novelty (asymmetry) is dead, BUT Rank 4 surfaced a load-bearing honest
result — our whole corrector line is likely beaten by a one-line least-squares baseline. Running
Rank 1 in-arena next to confirm. This is the real story (and a sharper hazard than we had).

---


## 🧪 2026-06-29 — H3 C3b (teacher distillation) FULL-SCALE (3 seeds) + H2 crash

| cond | SciFact (base 0.819) | NFCorpus (base 0.598) |
|---|---|---|
| C0 frozen | 0.0064 | 0.0407 |
| C2O oracle | 0.8132 | 0.6119 |
| C3a bounded-supervised | 0.1609 | 0.0524 |
| **C3b teacher-distill** | **0.1550** (≈C3a, slightly worse) | **0.1269** (~2.4× C3a; best post-hoc on NFCorpus, ~21% of oracle gap) |

- **C3b helps on the HARD dataset (NFCorpus 0.127 vs C3a 0.052), ties/loses on SciFact.** It's the
  strongest non-oracle-application corrector on NFCorpus. **CAVEAT (must disclose):** C3b's recovery
  depends on TEACHER PROVENANCE — if the teacher is new-model-derived it's oracle-adjacent (privileged),
  not a fair corrector. Verify before any headline. [[novelty-map]] still holds: mechanism anticipated.
- **H2 NFCorpus swap-campaign re-run CRASHED** (exit 1, numpy MemoryError — RAM exhaustion in the heavy
  budget-sweep, partial budget/main dirs written, no manifest). LOW VALUE (paper §5 numbers, superseded
  by H5/H3; scan deprioritized the budget sweep as anticipated). Re-run only if needed, with smaller grid
  / per-cell subprocess. H5 + H3 (the load-bearing results) are complete and clean.

## 🧮 2026-06-29 — PIPELINE OPPORTUNITY SCAN (workflow wo8cwmipv, 7 skeptical agents + rank)

Question: anything left to flush out / additively explore for a real result? Full output
`tasks/wo8cwmipv.output`. **Answer: no hidden MECHANISM win — every additive idea is anticipated or
re-confirms our in-house negative. One genuinely-unexplored measured asymmetry has upside (conditional).
The live asset is a methodology/hazards paper buildable from ~4 cheap (mostly-CPU, cached-embedding) runs.**

Ranked, test-ready:
- **RANK 1 — Drift-Adapter head-to-head → PURSUE-NOW.** The load-bearing fair baseline (replaces the
  frozen-C0 straw man). Fit Procrustes/low-rank/MLP/QDC on cached pairs in the HARD old→new direction;
  score beside C0/C2O/C5/C5r. CPU-hours. Likely shows we LOSE to Drift-Adapter — but "easy-direction
  adapters recover only ~20% of catastrophic old→new query-upgrade drift; bounding+cycling add nothing"
  IS the honest citable result. Until it's in the table no other number is interpretable.
- **RANK 2 — capacity curve → PURSUE-IF-CHEAP** (1-seed SciFact). Sweep rank/budget/adapter-type;
  the ONE non-anticipated payload = a measured forward-realignment CEILING tied to the analytic
  Procrustes bound (arXiv 2510.13406). Gate: plateau <0.35 of gap → structural-ceiling headline;
  climbs >0.6 → DROP (overlaps Drift-Adapter recovery).
- **RANK 3 — supervision-ceiling (anchor-count pilot only) → PURSUE-IF-CHEAP.** 2 extra runs
  (have 30→0.159, 1000→0.181) nail the saturation figure. Drop the multi-teacher sweep (anticipated).
- **RANK 4 — direction-theory (falsification check only) → PURSUE-IF-CHEAP/first.** The "old→new is
  fundamentally harder" premise is probably a fixed-index ARTIFACT (unbounded>bounded contradicts an
  irreversibility law). Compute CKA / mutual-kNN / Procrustes-residual BOTH directions; best use is to
  HONESTLY KILL the asymmetry if it's an artifact. **Rank 2's upside is conditional on this not gutting it.**
- **DROPPED:** dimension-masking (diagonal ⊂ affine we already use → can't rotate old→new, must
  underperform; anticipated by Learning-to-Select 2602.03306); σ-ablation (NoisyNet; no exploration
  objective → inference noise = pure variance on a deterministic optimum); invariance/slot/6D
  (fully-explored-dead: relative-representations 2209.15430 / Product-of-Invariances 2310.01211 /
  IsoCLIP / OPQ / MoE own every reading; H5 shows structured = net-negative).

**Bottom line (agent, verbatim spirit):** "The methodology/hazards paper is the only live asset — but
a genuine, publishable one, not a consolation prize." No surviving mechanism contribution; our own data
falsified the bounded/structured/lifecycle thesis. Do Ranks 1→4→2/3 (a day or two of mostly-CPU) → the
paper's three claims: (a) the query-upgrade ARENA, (b) the fair-baseline ~20% benchmark, (c) the
forward/backward asymmetry + Procrustes bound IFF Rank 4 confirms it's not a fixed-index artifact.

---


## 🎯 2026-06-29 — H5 FULL-SCALE VERDICT (real, 3 seeds, both datasets) — THESIS FALSIFIED

GPU campaign (1200 docs, 100 queries, 12 cycles, seeds 42/1337/7). Manifests in
`gpu-campaigns/experiment_runs/fullscale-2026-06-29/{h5-scifact,h5-nfcorpus}`.

| cond | SciFact final (base 0.819) | NFCorpus final (base 0.598) |
|---|---|---|
| C0 frozen | 0.0064 | 0.0407 |
| C2O oracle | 0.8132 | 0.6119 |
| **C5 living** | **0.1311** | **0.0464** |
| **C5s static** | **0.1311** (≡ C5, 15 s.f.) | **0.0464** (≡ C5) |
| C5r one-shot | 0.1809 | 0.0458 |
| `living_bank_wins` | **False** | **False** |

**Findings (real numbers, not a smoke):**
1. **C5 == C5s to 15 s.f. on BOTH datasets** → the prune/re-anneal LIFECYCLE is a confirmed
   NO-OP at full scale + 3 seeds. **The "living/annealed bank beats static" thesis is FALSIFIED.**
   (Confirms smoke + the scale-independent re-anneal-identical mechanism.)
2. **Corrections DO help vs frozen at full scale** — SciFact 0.006→0.131 (~20×), NFCorpus
   0.041→0.046 (~14%). So the smoke's "corrections hurt" was a tiny-scale (~2-anchor) overfit
   artifact; with ~full anchors the post-hoc corrector has real signal. BUT only ~16% (SciFact) /
   ~7.6% (NFCorpus) of the oracle's recovery.
3. **C5r (UNBOUNDED one-shot) ≥ C5/C5s (BOUNDED, multi-cycle)** — SciFact 0.181 > 0.131; NFCorpus
   ~tie. The bound limits recovery and the cycling adds nothing; one-shot is as good or better.
4. `living_bank_wins=False` on both → **the load-bearing H5 gate is NOT met, with real numbers.**

**Combined with the novelty map:** the corrector is BOTH anticipated (Drift-Adapter / NoisyNet /
MoE-LoRA) AND empirically fails to beat the baselines. **The contribution is the hazards + the
query-upgrade arena, full stop.** Even "corrections help vs frozen" is weak — C0 is a straw man;
the fair baseline is Drift-Adapter (TODO: add it). H3 C3b + H2 re-running (subprocess-isolated,
b9m24acdn) after the v1 run crashed ~30 cells in from GPU-memory accumulation (now per-campaign).

---


## 🗺️ 2026-06-29 — NOVELTY MAP (multi-agent adversarial sweep, workflow w459qcw2l) — STRATEGIC

Full output: `tasks/w459qcw2l.output`. Brutal verdict: **2 of 4 contributions die on Drift-Adapter
(arXiv:2509.23471, EMNLP 2025); σ-signaling dies on NoisyNet (2017). What SURVIVES = the hazard
pair + the query-upgrade ARENA. Plan the paper around the methodology, NOT a new mechanism.**

| Contribution | Verdict | Killer prior art |
|---|---|---|
| **1. C2 degenerate-oracle hazard** | **NOVEL** (methodology, small) — SURVIVES | Astute RAG (2410.07176, construction-leakage in QA) only adjacent; nothing names the re-embed-inverts-synthetic-drift oracle. The **query-upgrade arena** (re-embed oracle DEFEATED) is the citable asset. |
| **2. Supervision-signal hazard** | PARTIALLY → **anticipated as mechanism**; survives only as *expository* ("we NAME the implicit constraint") | Drift-Adapter + QDC (2506.00037) use the paired pre-drift target BY CONSTRUCTION; identity-collapse of anchor-free objectives is established (SAR/Tent/VICReg/T-REGS). |
| **3. σ-signaling (annealed per-direction stochastic correction)** | **ANTICIPATED** — survives only if a σ=0 ablation measurably loses | **NoisyNet (Fortunato et al., arXiv:1706.10295, 2017)** = learned per-element σ that SELF-ANNEALS — σ-signaling verbatim, just in an RL head. + Parameter-Space Noise (1706.01905). |
| **4. slot-warp codebook / post-bank / MoE-LoRA** | **ANTICIPATED** as architecture — survives only as empirical arena finding | **Drift-Adapter** owns swap-for-drift + Procrustes/low-rank/MLP warp (our exact adapter family), 95–99% recall. Query-aware dims = "Learning to Select" (2602.03306); MoE-for-drift = DriftMoE/DyMoE; prune lifecycle = REAP/MoRA; cluster-routed LoRA banks = LoraRetriever. |

**LEAD WITH:** Contribution 1's arena finding + Contribution 2 as the theoretical spine → a
**methodology/harness paper** ("a fair evaluation arena for embedding-drift recovery, and why the
obvious baselines cheat"). It does NOT collide with Drift-Adapter (which never does drift detection,
never names the oracle hazard, never separates doc-side vs query-side arenas).
**DROP as a headline:** Contribution 4 (slot-warp-codebook) — "novelty theater"; demote to one
ablation row.
**⚠️ CRITICAL METHODOLOGY FIX (changes the campaign baselines):** the FAIR baseline is **Drift-Adapter
(its 3 parameterizations) + QDC**, NOT a frozen-C0 straw man. Our C3a *is* Drift-Adapter's residual-MLP
in the old→new direction; we must benchmark head-to-head, not against frozen. The honest empirical
headline is "~25× above frozen / budget-monotonic / short of oracle" framed against those real baselines.
**For the operator (σ-signaling gut):** it's a real, sound *engineering* transfer (NoisyNet→drift
actuator) but NOT novel as a mechanism — only as a benchmarked empirical result vs σ=0 + vs Drift-Adapter.

---


## 🔎 2026-06-29 — HAZARDS novelty (the surviving contribution) + new TRAINING-TIME direction

**Hazards appear UNCLAIMED as specific findings (search confirms the surviving novelty):**
- Hazard 1 (C2 degenerate-oracle: a maintenance baseline that *inverts synthetic drift by
  construction* → not a fair baseline) — not found named. Adjacent KNOWN themes exist and must be
  cited as related, NOT as our claim: "A Synthetic Benchmark to Explore Limitations of Localized
  Drift Detections" (arXiv 2408.14687) and "The Window Dilemma: Why Concept Drift Detection is
  Ill-Posed" (2602.06456) both argue synthetic-drift evals / drift *detection* are unreliable. Our
  precise contribution = the degenerate-*correction-baseline* mechanism, sharper than the general
  skepticism. Defensible but must be framed against that adjacent work.
- Hazard 2 (unsupervised/homeostatic correction has no pre-drift target → loss-optimal = do
  nothing) — not found named. Continual-learning drift-compensation papers assume a supervision/
  distillation signal; none frame the no-signal degeneracy as the hazard. Defensible.
- **Net: the methodology hazards are the strongest novel core** (vs the corrector, which is
  anticipated by Drift-Adapter + MoE-of-LoRAs). Frame the paper around the hazards + framework.

## 🧪 2026-06-29 — Operator direction #3: TRAINING-TIME "stacked gates / perturbation-manifold" idea

Operator: "annealing is only 1-D thinking; consider directional/overlapping stacked gates DURING
MODEL TRAINING that populate denser subspace neighbors with replaceable value-position stacking —
fold/warp the next-token possibilities in a perturbation-manifold schema." Honest mapping:

- This is a **pivot from POST-HOC frozen-base correction → TRAINING-TIME architecture** (aligns
  with the earlier "make our own model"). Much larger scope than the shim/post-bank line.
- Component-by-component, it overlaps known work (so the *combination* would have to be the claim):
  - "annealing is 1-D → directional/overlapping" = **anisotropic / per-subspace** noise schedule
    (scalar σ → a σ-field). Real generalization; anisotropic noise exists in diffusion variants.
  - "stacked gates during training" = training-time **gated mixtures / MoE** (well-trodden).
  - "denser subspace neighbors + replaceable value-position stacking" = a **codebook / slot**
    representation (VQ-VAE codebooks, slot attention) with swappable entries.
  - "fold/warp next tokens in a perturbation manifold" = **perturbation-invariant / manifold-
    regularized** representation learning (e.g. Vaccine 2402.01109 "invariant hidden embeddings").
- **Honest status:** an evocative SKETCH, not yet a design. Not novelty-checkable or buildable
  until ONE concrete version is formalized: *what* are the gates (params/inputs), what does
  "replaceable value-position stacking" mean as an operation, what is the manifold and what's
  invariant on it. Without that it reduces to "MoE + codebook + robust training," all prior art.
- **Recommendation:** do NOT commit to training a custom model on a sketch. Two cheap gates first:
  (1) the full-scale H5 campaign — establish whether the *post-hoc* path even has a ceiling worth
  abandoning; (2) formalize ONE crisp version of this (a half-page math spec) so it can be
  novelty-checked + prototyped at toy scale before any training run. Custom-model training is the
  most expensive path in the program; it needs a precise hypothesis + a cheap toy proof first.

---


## 🔎 2026-06-29 — NOVELTY CHECK (σ-signaling + post-bank + dimensional-warp) — real lit search

Operator asked: is σ-signaling novel, or already discovered (and can we extend)? Searched HF
papers + web. Honest verdict: **the MECHANISMS are largely prior art; the strongest remaining
novelty is our two methodology HAZARDS, not the correction mechanism.** Key hits:

- **🚩 Drift-Adapter (arXiv 2509.23471, EMNLP 2025) — the critical near-prior-art.** A learnable
  transform (Orthogonal Procrustes / Low-Rank Affine / Residual MLP — *exactly our adapter
  family*) for embedding-MODEL-UPGRADE drift in vector DBs, trained on a small paired-embedding
  sample, **recovers 95–99% of Recall@10/MRR**, <10µs latency. This anticipates our entire
  adapter-based drift-correction core AND outperforms our fractional recovery. **IMPORTANT
  NUANCE / our only daylight:** Drift-Adapter maps NEW queries → OLD/legacy space (reuse the old
  index — the easy direction); our arena realigns OLD docs → NEW query space (the hard direction
  we struggled with). Related but inverse problem. We must cite + differentiate against this.
- **σ-signaling mechanism = prior art.** Annealed noise-scale σ is simulated annealing + diffusion
  noise schedules + SGLD; stochastic experts/adapters already exist — **S2MoE** (stochastic MoE,
  2503.23007), **MoSA** "stochastic activation" adapters (2312.02923), and a **stochastic-embedding
  → diffusion** framework (2603.20423). σ-as-mechanism is NOT novel; at best a narrow
  drift-recovery *application* of it could be, and that niche is crowded.
- **The post-bank = Mixture-of-LoRAs + routing.** ReMix (RL routing for mixtures of LoRAs,
  2603.10160), HyperRouter, Self-Routing, "Rewiring Experts on the Fly" (online test-time MoE
  rerouting, 2510.14853 — anticipates the "living/adaptive" angle). **Operator's RAG instinct is
  correct:** bare post-bank + similarity routing IS MoE/RAG-flavored; the lookup is not the novelty.
- **"Dimensional tuning/warping module" = query-aware dimension selection.** "Learning to Select:
  Query-Aware Adaptive Dimension Selection for Dense Retrieval" (2602.03306) predicts per-dimension
  importance from the query — close to the operator's dimensional-warp idea. Partly prior art.
- Other drift prior art: DeDrift (2308.02752, quantizer update under content drift), Query Drift
  Compensation (2506.00037, continual-learning retrieval).

**Where genuine novelty plausibly survives (honest, not forced):**
1. **The two evaluation HAZARDS** (C2 degenerate-oracle baseline; the supervision-signal/no-target
   failure) — these are *methodology findings*, not mechanisms, and I did not find them named in the
   search. This is the strongest novel contribution and is exactly what the Zenodo bundle banks.
2. The **old-doc → new-space** direction (harder than Drift-Adapter's new→old) — a real
   differentiator, but our results there are a *negative* (corrections overfit / tie static).
3. A precise **annealed-σ + held-out-selected living lifecycle for drift** — possibly a thin novel
   combination, but every component is published; would need a strong empirical win to claim it.

**6D-stacked-matrix / localized-invariance idea (operator):** under-specified to novelty-check, but
"two stacked transforms → locally-invariant region per post, looked up by similarity" reduces to a
per-region MoE/RAG transform unless the *invariance construction itself* is the formal claim. Needs a
precise math statement before it's checkable; flagged as a design hypothesis, not yet a contribution.

**Takeaway for the operator:** σ-signaling is a sound *engineering* direction but not a novelty
flag on its own. The defensible novel core is the **hazards + the methodology framework**, not the
corrector. Reframe accordingly (this also de-risks the Zenodo/paper framing).

---


## 🧭 2026-06-29 — H5 REDESIGN DIRECTION: σ-signaling shims (operator insight + analysis)

Operator's framing after the three-layer negative: "annealing not possible with these models →
make our own model? subspace variation/warping is novel with inserted shims, but introduce sigma
signaling variations instead of the blank-shims idea." Distilled into the H5 redesign direction:

**Reframe of the negative — two SEPARABLE problems (not one):**
- **P1 — annealing was VACUOUS, not impossible.** The shims are DETERMINISTIC (one fixed adapter
  per cluster, fixed seed). Annealing is a temperature schedule over an *exploration distribution*;
  a deterministic shim has no distribution, so explore/stabilize/prune has nothing to act on
  (re-anneal rebuilt the identical post). This is a property of the SHIM, not of MiniLM/mpnet.
- **P2 — the corrections overfit sparse anchors** and hurt held-out NDCG (separate failure).

**The σ-signaling fix (targets P1, plausibly helps P2):** make each shim a STOCHASTIC family
parameterized by σ (the correction is sampled, not fixed); bind the annealing temperature → σ
(high σ = wide exploration of warps early, low σ = commit to the best warp late). Now annealing
has real work — sample + select over a σ-indexed family — and high-σ-early also regularizes
against P2's overfitting.

**Honest caveats (do NOT over-rotate):**
- **No new ENCODER needed.** Frozen MiniLM/mpnet is a fine substrate; the novel subspace-warp-shim
  idea is intact. What changes is the CORRECTION PARAMETERIZATION (stochastic σ-warp) + the
  SELECTION SIGNAL — both live in our code, not the base weights.
- **σ gives a family to select FROM; you still need an honest signal to select ON.** Even with
  σ-variation, keep/prune/commit must score against HELD-OUT queries, not the training anchors
  (anchor fitness overfits — proven). σ fixes "nothing to anneal"; it does NOT fix "grading on the
  training set."
- **Untested.** This is design reasoning on a 15-doc smoke. Right shape; not validated at scale.

**Existing pieces to build on:** the repo already has dynamically-scaled NOISE INJECTION (the σ
mechanism) + the ANNEALING SCHEDULE (#280). New work = bind temperature→σ→a stochastic shim family
+ a held-out keep/commit signal.

**Test ORDER (cheapest-first; full-scale needs the 3090):**
1. Full-scale CURRENT deterministic mechanism first — it may just need ~480 anchors not ~2 to stop
   overfitting. If it helps at scale → annealing is the only gap → σ-signaling is a clean refinement.
   If it STILL hurts at scale → the correction family is wrong → σ-signaling is REQUIRED, not optional.
2. Then σ-parameterized shims + held-out selection.

**OPEN OPERATOR FORK (awaiting):** (a) free the 3090 for the full-scale run (test #1), or
(b) stage the σ-shim design / build it. Recommendation: (a) first — it tells us whether σ-signaling
is a refinement or a requirement before we spend build effort.

---


## ⚖️ 2026-06-29 — DECISIVE: Option A would be a HOLLOW gate-pass at smoke scale; the real test is full-scale GPU

Deeper trace of Option A (held-out per-post fitness + drop): because the post-bank corrections
HURT held-out NDCG at smoke scale (C5 0.555 < C0 0.566), a held-out leave-one-out fitness marks
EVERY post harmful → drops them all → C5 reverts to C0 (0.566) > C5s (0.555). That is technically
`living_bank_wins=True`, but the living bank wins by **declining all corrections (reverting to
frozen)**, not by keeping good posts. Per the guardrail — "the gate is retrieval changed and beats
the static baseline, NOT the actuator firing" — that is a HOLLOW pass: building it just to flip the
boolean would be gaming the gate. **Do not do that.**

**What this means:**
- At smoke scale (15 docs / 2 anchors / default knobs) the post-bank corrections do not help; the
  only way a living bank "beats" static is degenerate (apply nothing).
- BUT smoke scale is tiny and may not generalize: the corrections overfit ~2 anchors. At FULL
  scale (1200 docs, ~480 anchors, swept budget/cluster knobs, 12 cycles) the per-cluster adapters
  may actually help — which would make the living-bank advantage REAL (keep helpful posts, drop the
  few harmful ones) rather than degenerate. **This cannot be known without the full GPU campaign.**
- So the honest priority is NOT "build Option A to pass the gate" — it's **run the full-scale H5
  campaign on the 3090**. If corrections help at scale → real living-bank test (build the held-out
  fitness then). If they still hurt at scale → the H5 thesis is genuinely negative (reframe, Option C:
  publish "post-bank corrections overfit and don't beat frozen; the living advantage is degenerate").

**Bottom line for the operator:** the single highest-value next action is freeing the 3090 for the
full H5/H3/H2 campaigns. The code is ready (#290 driver), every cheap fitness has been validated as
insufficient, and the held-out-fitness build is designed + de-risked but should only be built AFTER
full-scale shows corrections help (else it's a hollow pass). No fabricated numbers; no gamed gate.

---


## 🛠️ 2026-06-29 — H5 fix design notes (Option A: held-out per-post fitness) — de-risking the build

The path to the H5 gate (deeper design pass; capture so the build doesn't hit these traps):

- **Fitness must be HELD-OUT, not training-anchor.** Per-post score = NDCG contribution on the
  held-out eval queries. Two forms: (i) leave-one-out `NDCG(full) − NDCG(full∖p)`; (ii) single-post
  `NDCG(only p) − NDCG(C0)`. (ii) is simpler/cheaper (O(posts+1) evals). Both need the engine +
  drifted_eval_vectors + eval_qrels (run_post_bank_cycle has all three).
- **TRAP 1 — store state.** Each per-post eval mutates the store; you must apply each probe bank to
  the ORIGINAL snapshot (`apply_post_bank_to_store(..., original_doc_points)`) and RESTORE originals
  between probes and before the cycle's real apply. Budget an explicit `_restore_store(engine,
  original_doc_points)`.
- **TRAP 2 — routing shifts on drop.** `apply_post_bank_to_store` routes EVERY doc to its nearest
  *surviving* post. Dropping a harmful post does not cleanly revert that post's docs to originals —
  they re-route to the next-nearest kept post. So "drop harmful → revert toward C0" is NOT exact;
  the kept posts still touch the orphaned docs. To get a clean C0 revert when ALL posts are harmful,
  you need `min_posts = 0` (empty bank) AND the empty-bank apply must explicitly restore originals
  (it currently no-ops, leaving the prior cycle's mutation). Both need handling.
- **TRAP 3 — min_posts.** Default 1 keeps the least-harmful post → C5 is a PARTIAL win at best. For
  the clean C5==C0>C5s win when every post hurts, `min_posts=0` + empty-bank-restores-originals.
- **Why it likely wins:** the smoke shows corrections HURT held-out NDCG (C5 0.555 < C0 0.566), so a
  held-out signal would mark posts harmful and drop them → C5 reverts toward C0 (0.566) > C5s (0.555).
  But it is NOT a one-liner — it's a careful store-state + routing + min_posts=0 slice (~the reason it
  warrants fresh context / operator sign-off rather than a tail-of-session rush).
- **WIP branch `feat/h5-alignment-fitness`** already has the drop-harmful plumbing
  (`reanneal_pruned=False`, `prune_below` in fitness units) — the held-out build extends it by
  swapping `alignment_fitness_by_post` → `held_out_fitness_by_post` + the TRAP-1/2/3 handling.

---


## 🔬 2026-06-29 — H5 LIVING-BANK THESIS: three-layer NEGATIVE finding (validated on real models)

Operator-allowed validation smokes (1 seed / 15 docs / 2 cycles, real MiniLM+mpnet, CUDA,
nvidia-smi-gated) drove three successive fixes; **C5 (living) == C5s (static) to 15 s.f.
(0.5550465351419688) in ALL THREE.** The living bank cannot beat the static bank with any cheap
fitness/lifecycle tried. This is the real blocker behind the H5 gate — found by RUNNING, not
asserting.

1. **Hits-fitness is non-discriminative.** Per-post hit counts are ~uniform → no post is "low
   fitness" → nothing prunes → C5 == C5s. (PR #283's labeled proxy.)
2. **Prune+re-anneal is a no-op even with a discriminative signal.** I built alignment fitness
   (mean doc→query cosine improvement) — but `prune_and_reanneal` rebuilds the pruned post from
   the SAME anchors+seed → restores it IDENTICALLY → bank unchanged → C5 == C5s.
3. **Drop-harmful + anchor-alignment STILL ties.** I added `reanneal_pruned=False` (drop, don't
   restore) + alignment fitness + prune_below 0.0. But the alignment fitness is measured on the
   TRAINING ANCHORS, where the corrections OVERFIT (positive anchor alignment) → no post scores
   below 0 → nothing drops → C5 == C5s. Meanwhile eval NDCG is HURT (C5 0.555 < C0 0.566): the
   corrections help the anchors but hurt held-out retrieval.

**CONCLUSION (honest, load-bearing):** the H5 living-bank advantage requires a **held-out
per-post retrieval-contribution fitness** (each post scored by its NDCG delta on held-out eval
queries, not training-anchor alignment or hit counts). Every cheap proxy fails — and the smoke
also shows the post-bank corrections themselves *hurt* held-out NDCG at this scale (overfitting
the sparse anchors). This is the same class as the original supervision-signal hazard: **the
mechanism needs a held-out signal it doesn't have.** The H5 gate ("living beats static") is NOT
reachable as currently designed; the next research step is a held-out per-post eval-contribution
fitness (expensive: O(posts × eval) re-eval) — and it is an OPEN question whether even that makes
the living bank win, given the corrections hurt held-out NDCG.

**WIP not merged:** `feat/h5-alignment-fitness` (alignment fitness + drop-harmful, stub-tested,
ruff clean) is pushed but **deliberately NOT merged** — the smoke proves it does not achieve the
gate (C5 == C5s still), so merging it would imply progress that isn't there. It's a stepping
stone toward the held-out-fitness design, kept as a branch for reference.

---


Findings from higher-reasoning review passes over the implementing agent's slices.
The cleanup pass should resolve every open item below, run genuine Tier B reviews,
and open the PRs. Add new sections per slice as they land.

## PR-1: drift_injector (branch `feat/drift-injector-pr1`, commit 7dd2d82) — reviewed 2026-06-11

**Independently verified (do not re-do):** all 4 tests pass (0.186s), `ruff check .`
clean, branch based on main `bf23a47`, rotation is disjoint Givens pairs (proper
orthogonal, norm-preserving), determinism test asserts byte-identical vectors across
two fresh engines, `engine.qdrant` is the QdrantVectorStore abstraction alias
(antigravity_engine.py:110) so the plan's abstraction requirement is satisfied.

**Open items for the cleanup pass (ordered):**

1. **L13 disclosure required before PR opens.** The degradation-evidence test
   (`test_rotation_drift_measurably_degrades_real_retrieval_ndcg`) patches
   `create_embedding_backend` with a synthetic one-hot backend and patches the adapter
   to identity. Production retrieval path is real; embedding geometry is synthetic
   (hence baseline NDCG exactly 1.0). PR body's Brutal Honesty section must state:
   floor-tier evidence through production retrieval path on synthetic embedding
   geometry; real-model drift behavior arrives in PR-4/5.
2. **RNG reproducibility policy.** `_select_affected` and `_rotation_pairs` share one
   `default_rng`; second injection on the same instance differs from a fresh injector,
   and the manifest records seed but not call index. Pick one: (a) document
   "one fresh DriftInjector per injection event" as the contract, or (b) add an
   `injection_index` counter to the manifest. PR-4 harness must follow whichever is
   chosen.
3. **Run genuine Tier B** (fresh adversarial agent, diff + BHS section + rulebook,
   instructed to DISPROVE), then push branch and open PR with full BHS body
   (BHS_SELF_DRAFT/BHS_TIER_B/etc. lines per rulebook §4). Branch is currently
   local-only and unpushed — Session Rule 2 debt.
4. **Minor, PR-body-only:** (a) confirm engine-stored vectors are unit-norm so
   `inject_noise_drift`'s re-normalization is semantics-preserving; (b) add a comment
   on the test NDCG helper noting DCG≡NDCG only because exactly one relevant doc per
   query (IDCG=1).

## Subsequent slices (PR-2..PR-5)

### Status update 2026-06-12 (review pass)

- **PR-1 (#258), PR-2 (#259), PR-3 (#260) all MERGED to main at BHS_OFFICIAL 100** with
  genuine fresh Tier B reviewers (Descartes / Galileo / Nash→Meitner→Tesla). PR-3's Tier B
  loop caught 3 real bugs across 2 iterations before passing — the loop works.
- Independently re-verified on main `f0c643a`: `python -m unittest test_drift_injector
  test_drift_recovery_metrics test_annealing_controller -v` → 22 tests, all pass. CI green
  across py3.9–3.12 + BHS workflow.
- PR-1 review items: L13 synthetic-backend disclosure adequately covered by the PR body's
  L3 line; named-vector schema preservation bug found AND fixed during its review
  (regression test added). PR-2 correctly reused `benchmark_utils.ndcg_at_k` via wrapper.
- **STILL OPEN → must be enforced in PR-4:** the RNG call-index policy (item 2 above) was
  NOT addressed. No `injection_index` in manifest, no documented fresh-injector-per-event
  contract. PR-4 harness MUST instantiate a fresh `DriftInjector` per injection event and
  record seed + event order in its run config, or extend the manifest first.
- `enable_annealing_controller` exists at antigravity_engine.py:824 (lazy import pattern).
- **Scope watch:** PRs #256 (SHIM-CD-01 SIP probe / VectorSteerer) and #257 (progress
  branch) are OPEN and predate the June plan. Both are OUT of June scope. Do not merge as
  part of this campaign; operator decision whether to park or close.
- Next: PR-4 harness (`run_drift_recovery_experiment.py`) then PR-5 campaign. No
  experiment results exist yet — `experiment_runs/drift-recovery/` not created.

### Cleanup pass 2026-06-12 (evening, Claude)

- **RNG item FIXED** on branch `feat/drift-injector-injection-index`: manifests now
  record 0-based `injection_index`; determinism contract documented in class docstring.
  Tier B iteration 1 (fresh reviewer "Hopper", 88/100 important) found a REAL additional
  hole: rotation drift consumed RNG in `_select_affected` before `_validate_dims` could
  raise, desynchronizing RNG state from injection_index. Fixed in commit 526b2ef: all
  validation (dims, finite angle, non-negative finite sigma, fraction) now precedes any
  RNG consumption in both modes; regression test proves a post-failure injection equals
  a fresh same-seed replay. 7/7 tests green, ruff clean. Tier B iteration 2: 100/100,
  severity none (reviewer confirmed regression test discriminates old-vs-new ordering
  and happy-path RNG order is unchanged). **MERGED as #261 / a445b2b** (admin merge per
  repo branch-policy precedent, all checks green, BHS_OFFICIAL 100). Re-verified 7/7
  green on merged main. PR-1 review items 1–4: ALL CLOSED.
- **PRs #256 and #257 parked**: converted to draft with scope-freeze comments. Do not
  merge during the campaign.
- Overnight brief for codex written:
  [overnight-brief-2026-06-12-codex.md](overnight-brief-2026-06-12-codex.md) — PR-4
  harness then PR-5 campaign matrix; morning pass verifies both + starts paper figures.

## Day 2026-06-14 (goal execution; survived a mid-session crash)

State recovered after a crash. Progress so far:

- **Track 0 hygiene MERGED** as PR #267 (`705298b`). Closed CD-247-01/02 honestly (only
  1 of 6 cited except-sites was a genuine swallow; rest documented stale). Block flag now
  legitimately CLEAR / 0 debt rows. Removed stale `serene-aryabhata-3ed9a9` worktree; merged
  drift branches already auto-deleted. Tier B (Curie) 100 on substance + orchestrator ran
  the runtime gates. CI flaked once on an unrelated 3.12 test, re-run green, admin-merged.
- **#257 NOT closed (deviation from plan, flagged for operator):** the branch carries ~20
  unmerged `test_shim_*.py` + tts/vector_store changes — substantial SHIM work, NOT cleanly
  superseded. Closing would discard it. Kept parked as a draft alongside #256. Operator call.
- **PR-A1 — corrected arena (in progress, branch `feat/query-encoder-drift`):** while
  implementing the planned document-side model-swap, CAUGHT that it does NOT defeat the C2
  oracle (C2 re-embeds unchanged text with the original encoder → restores baseline exactly).
  Verified + documented in `finding-oracle-survives-document-model-swap.md`. Operator chose
  the **query/encoder-upgrade arena** (defeats the oracle, instantiates Concept 4). Built
  `query_encoder_drift.py` + tests: drift = embed eval queries with a 2nd encoder
  (all-mpnet-base-v2, 768→384 frozen seeded projection) while cached docs stay in MiniLM
  space; C2 re-embed-with-original is a demonstrated no-op and cannot recover; re-embed into
  the new space recovers. Real mpnet model loads at 768-dim on cuda. Tier B (Noether) iter 1
  found the oracle-breaker assertion was circular (`x<=x`); fixed by routing C2 through a real
  re-embed backend that OBSERVES the no-op (commit `ad8afeb`). **Tier B iter 2: 100/none**
  (Noether ran a divergent-encoder counterfactual to prove the C2 assertion is falsifiable).
  **PR #268 OPEN with full BHS body, CI pending.** Paper §4.3/§7 corrected to the
  query-encoder-upgrade arena (oracle survives doc-side model-swap; only encoder upgrade
  defeats it).
- **Flaky test fix — PR #269 OPEN at Tier B 100, CI pending.** Worktree-isolated workflow
  diagnosed it: exact-float golden equality vs a torch-matmul trajectory, perturbed by torch
  GLOBAL state (not the seed — production reseeds correctly). Two-layer fix: (1) tolerance-based
  `_assert_trajectory_close` (rtol 1e-9 on float leaves, structure exact) — load-bearing,
  global-agnostic for the low-bit FAIL class; (2) setUp pins default dtype=float32 + device=cpu
  (+threads=1) for the CRASH class tolerance can't help, addCleanup restores. Tier B (Lovelace)
  iter 1 (82) caught an uncovered cuda-device crash vector + argued for tolerance; iter 2 (100)
  stress-tested 24+ FP-global vectors, all neutralized; confirmed regression power preserved
  (catches 0.213→0.30 drift + ranking/bool/int/zero-leaf changes). Test-only, no prod change.
- **Paper draft (Track 3) DONE through §§1–4, 6–8** in `paper-draft/main.md` (§5 Results
  blocked until a real data point). Lead assets: C2-oracle hazard (§4.3, now even stronger —
  oracle survives doc-side model-swap) + supervision-signal mechanism (§6).
- Execution mode chosen by operator: implement + checkpoint each PR.
- Remaining: finish PR-A1 → flaky fix → PR-A2 (anchor-InfoNCE supervision + P0/P1 mechanism
  fixes: use_quantization=True, NDCG-drop trigger) → A3 (teacher) → A4 (campaign) → B1/B2.

### Merge progress (2026-06-14, continued)
- **MERGED: #267 (Track 0), #268 (PR-A1 query-encoder drift), #269 (flaky-test fix).** main @ `bd3e628`.
- PR-A2 split into **A2a** (arena in harness: query_encoder_swap drift + search-by-vector eval +
  C0/C2/C2-oracle, harness-level oracle-breaker proof — NO supervised loop) and **A2b**
  (supervised C3a + P0/P1 mechanism fixes that make the actuator fire). Each independently shippable.
- **A2a — PR #270 OPEN, CI pending.** Built via worktree-isolated workflow + in-workflow
  adversarial reviewer (SOUND/none: store-checksum-verified C2 no-op, sabotage-tested C2O for
  circularity, regression-checked). Applied to feat/query-encoder-arena-harness, independently
  re-verified (12 + 24 tests pass, lint clean). HARNESS-LEVEL ORACLE-BREAKER PROVEN on a real
  tiny run: baseline 1.0 → C0 0.4981 → C2 no-op 0.4981 → C2O recovers 1.0. Drift mode
  query_encoder_swap + evaluate_engine_with_query_vectors + C2O condition added; rotation/noise
  + C1/C3/C4 + #269 flaky isolation all intact.
- **A2b NEXT (the crux):** supervised C3a (anchor-pair InfoNCE so the bounded adapter realigns
  cached doc vectors to the drifted query space) + NDCG-drop trigger. Makes the actuator FIRE.

### 2026-06-14/21 (crash recovery + continued; merges)
- **MERGED: #270 (A2a arena), #271 (real-model smoke opt-in).** main @ `230197f`. #270 and #271
  each needed a CI re-run for UNRELATED transient HF-network flakes (real-model hang; SciFact
  dataset fetch in test_learned_mask_gate via load_mteb_data). KNOWN RISK: `load_mteb_data`
  docstring promises (None,None,None) on error but doesn't wrap `task.load_data()` (benchmark_utils.py:331)
  — HF hiccups crash CI; transient so far, harden if it recurs. Pruned 3 stale workflow worktrees.
- **A2b design simplified (confirmed):** C3a uses a DEDICATED supervised step (anchor InfoNCE via
  SedimentationInfoNCELoss), NOT run_sedimentation_cycle — bypasses the use_quantization P0 issue
  entirely. Spec: pr-a2b-implementation-spec.md. Anchor/eval split (anchor_fraction config,
  default 0 = A2a behavior preserved; >0 = split, all conditions measure on eval subset for fair
  comparison, anchors reserved for supervision).
- **A2b — PR #272 OPEN, CI pending (the crux).** Built via workflow; workflow reviewer found 2
  defects (needs-changes); I fixed both + a fresh Tier B (Shannon) verified at 100/none.
  **RESULT: the actuator FIRES** (should_correct from real ndcg_drop, correction_applied=True via
  store checksum, mean norm 0.136 C3a / 0.462 C4a — Tier B sabotage-confirmed not a bound-floor
  artifact). **HONEST NEGATIVE on the stub geometry:** supervised adapter does not beat fair-C0;
  oracle C2O recovers fully (arena IS recoverable). Fixes: (1) fair same-eval-subset C0 baseline
  (was L13 mislabel); (2) seed-before-reinit per cycle → genuine idempotency (cycles=3 test +
  sabotage). DISCLOSED scope caveat (Shannon): 30-step/lr-0.01 budget under-fits even in-sample
  (~0.515 w/ all anchors vs 1.0 at 2000 steps) → the stub negative is budget+generalization, not
  pure generalization → **PR-A4 MUST sweep training budget before concluding no-recovery on real data.**
- **GPU/machine note (operator, 2026-06-14):** full 3090 access for tonight's testing; other work
  runs through the night. Be careful, don't OOM. GPU checked: 22/24GB free, ~1% util. A4 (real-model
  campaign) uses the GPU — run sequentially, modest batches, nvidia-smi check before/during, abort if tight.
- **Next:** A3 (teacher C3b), then **A4 — the real-data SciFact/mpnet campaign** (headline numbers;
  MUST sweep C3a training budget per Shannon; runs the real swap path closing CD-A2-01), then B1/B2,
  then paper results. A4 is the one that answers "does the loop actually recover on real data."

### A4 — REAL-DATA HEADLINE RESULT (2026-06-21, offline GPU run)
- Env feasibility confirmed: mpnet (768d), MiniLM (384d), SciFact (5183/300/300) all load offline
  from cache via HF_HUB_OFFLINE=1 + HF_DATASETS_OFFLINE=1. GPU ~5GB/24GB used during campaign — other
  night-work untouched, no OOM.
- **A4 harness PR #273 OPEN** (budget knobs + run_drift_recovery_swap_campaign.py + tests). Tier B
  Hamilton 98→100 (added report-content assertion). CI pending.
- **REAL-DATA NUMBERS (3 seeds, SciFact, MiniLM base / mpnet swap, query_encoder_swap arena):**
  - C0 frozen: baseline 0.819 → final **0.006** (encoder upgrade is catastrophic)
  - C2 re-embed-original: **0.006** == C0 → THE ORACLE IS GENUINELY DEFEATED (no-op confirmed on real data)
  - C2O oracle (re-embed into new space): **0.813** (recovery possible / upper bound)
  - C3a bounded supervised: **0.159** (FIRES + writes; ~25x above C0; partial recovery)
  - C4a unbounded supervised: **0.206** (unbounded > bounded, as expected)
  - C3a budget sweep (seed42): 30→0.159, 200→0.173, 1000→0.181 — MORE BUDGET HELPS (monotonic).
- **HONEST READING:** NOT a flat negative. The supervised closed loop recovers a MEASURABLE FRACTION
  of catastrophic encoder-upgrade drift (C0 0.006 → C3a 0.16 / C4a 0.21), recovery scales with budget,
  but does not close the gap to the oracle (0.81). Ordering: C0=C2 < C3a < C4a << C2O. This is the
  publishable mixed/positive result the paper needs — bounded correction partially recovers; full
  recovery needs either the oracle re-embed or (likely) more anchors/capacity/budget. Campaign finishing
  last 4 runs (2000-step budget cell + 3-seed confirm); results doc auto-generates. Commit as PR-A4-results.

## Standing checks for every slice review

- Independently run the slice's SMOKE command — do not trust the agent's report.
- `ruff check .` and confirm branch parent is current main.
- Grep for `X | None` annotations (py3.9), pytest imports, stubs/TODOs.
- Compare against the acceptance criteria in
  [implementation-plan-june-2026-drift-paper.md](implementation-plan-june-2026-drift-paper.md).
- Check mock usage in evidence tests: mocks are fine for isolation, but any claim of
  "real path" evidence must be qualified in the BHS section (L13).
- Confirm artifacts (JSON/plots) come from actual runs, never hand-edited (Rule 3).

## Overnight 2026-06-13

- PR-4 #262 merged via admin after all checks green: `0438f9b` (`Add drift recovery experiment harness`). BHS_OFFICIAL 100 after Tier B loop: Chandrasekhar found critical C3 detection/correction honesty gaps (70/critical); fixes removed forced drift signal, split `sedimentation_attempted` from `correction_applied`, tightened C2 affected-only refresh; Boole final pass 100/none.
- PR-5 #263 merged via admin after all checks green: `d2e5559` (`Add drift recovery campaign results`). BHS_OFFICIAL 100, Tier B reviewer Planck 100/none.
- Runs completed vs failed: SciFact target matrix completed 30/30, failed 0/30. Campaign log: `experiment_runs/drift-recovery/campaign_driver.log`; JSON artifacts and plots committed under `experiment_runs/drift-recovery/`.
- Host/device: initial online HF model-load sanity check failed with the known SSL certificate issue; retry with `HF_HUB_OFFLINE=1` used local cache and succeeded. `torch.cuda.is_available()` true; device recorded as `cuda` / NVIDIA GeForce RTX 3090.
- Headline numbers from `docs/drift-recovery-results-2026-06.md`: rotation final NDCG C3 0.814820 vs C0 0.814730, C1 0.814730, C2 0.829212; recovery@12 C3 3/3, C0 3/3, C1 3/3, C2 3/3. Noise final NDCG C3 0.685332 vs C0 0.684292, C1 0.684292, C2 0.829212; recovery@12 C3 0/3, C0 0/3, C1 0/3, C2 3/3.
- Honest result: C3 slightly beats C0/C1 on final NDCG but loses to C2 on final NDCG in both drift modes; report as mixed/negative against the maintenance-only baseline, not as a win.
- Open blockers: none for PR-4/PR-5 target scope. Stretch goals not run: NFCorpus transfer and C5 masking ablation.

## Morning review pass 2026-06-13 (Claude)

- Independently verified: #262/#263 merged on main (`0438f9b`, `d2e5559`); harness tests
  3/3 green locally; 73 artifacts + 2 plots present; results doc numbers match the
  overnight report. Overnight agent behavior was honest (mixed result reported plainly).
- **Three diagnoses from the results table (hypotheses for PR-6 to confirm/refute):**
  - H1: rotation(0.5,25°) too weak — even frozen C0 "recovers" at cycle 1.00 because the
    ~1.7% drop never left the 0.95 recovery threshold. Setting cannot discriminate.
  - H2: correction mechanism barely moves — C3 mean correction norm exactly 0.010000
    (suspected BoundedAdapter cap saturation); C4 unbounded 0.000189 (near-zero learning
    signal). Need per-cycle detection/correction traces.
  - H3: C2 is an oracle, not a fair baseline — 0.829212 identical across both modes,
    presumably equals pre-drift baseline; re-embedding unchanged text with the frozen
    model undoes synthetic vector drift by construction.
- Day goal issued: [goal-day-2026-06-13.md](goal-day-2026-06-13.md) — PR-6 diagnostics,
  PR-7 severity calibration (8–20% drop zone), PR-8 pre-registered C3 knob sweep with
  full-cell reporting + 3-seed confirmation, stretch PR-9 model-swap drift (the arena
  where the C2 oracle fails by construction).

## Day 2026-06-13

- PRs merged:
  - PR-6 #264 `a5a3e11` — diagnostics over existing drift-recovery artifacts.
  - PR-7 #265 `0612004` — calibrated severity campaign.
  - PR-8 #266 `7ececa1` — calibrated C3 knob sweep.
- H1 verdict: confirmed for the original rotation setting. Rotation fraction 0.5 / 25
  degrees was too weak: C0 rotation dropped only 1.739% and all C0 rotation runs were
  already above the 95% recovery threshold at cycle 1. Noise dropped 17.471%.
- H2 verdict: confirmed. Detection fired and sedimentation was attempted in the
  original C3/C4 cycles, but `correction_applied` was false; C3 norms saturated near
  0.010000 and C4 stayed near 0.000189.
- H3 verdict: confirmed. C2 returned exactly to each run's pre-drift baseline
  (`max baseline-final diff = 0`), so it is reported as an oracle re-embed upper bound.
- Chosen calibrated settings:
  - Noise: fraction 0.5 / sigma 0.035, scout drop 9.317%, in the 8-20% zone.
  - Rotation: fraction 0.5 / 35 degrees, scout drop 7.732%; no rotation scout landed in
    the 8-20% zone, so this was the closest to the 12% target.
- Calibrated full-matrix result: C3 still did not materially beat C0/C1 and remained
  below C2. Rotation C3 final 0.764609 +/- 0.023323 vs C0 0.764666 +/- 0.023299 and
  C2 0.829212 +/- 0.010032. Noise C3 final 0.763342 +/- 0.017833 vs C0
  0.763149 +/- 0.017778 and C2 0.829212 +/- 0.010032.
- PR-8 knob-sweep grid summary, calibrated rotation only, seed 42:
  - bound 0.01 / trigger 0.05 / default: final 0.754416, recovery none, 0/12
    should_correct, 0/12 attempted, 0/12 applied.
  - bound 0.01 / trigger 0.05 / hotter: final 0.754416, recovery none, 0/12
    should_correct, 0/12 attempted, 0/12 applied.
  - bound 0.01 / trigger 0.15 / default: final 0.754416, recovery none, 0/12
    should_correct, 0/12 attempted, 0/12 applied.
  - bound 0.01 / trigger 0.15 / hotter: final 0.754416, recovery none, 0/12
    should_correct, 0/12 attempted, 0/12 applied.
  - bound 0.05 / trigger 0.05 / default: final 0.753156, recovery none, 0/12
    should_correct, 0/12 attempted, 0/12 applied.
  - bound 0.05 / trigger 0.05 / hotter: final 0.753156, recovery none, 0/12
    should_correct, 0/12 attempted, 0/12 applied.
  - bound 0.05 / trigger 0.15 / default: final 0.753156, recovery none, 0/12
    should_correct, 0/12 attempted, 0/12 applied.
  - bound 0.05 / trigger 0.15 / hotter: final 0.753156, recovery none, 0/12
    should_correct, 0/12 attempted, 0/12 applied.
  - bound 0.10 / trigger 0.05 / default: final 0.754530, recovery none, 0/12
    should_correct, 0/12 attempted, 0/12 applied.
  - bound 0.10 / trigger 0.05 / hotter: final 0.754530, recovery none, 0/12
    should_correct, 0/12 attempted, 0/12 applied.
  - bound 0.10 / trigger 0.15 / default: final 0.754530, recovery none, 0/12
    should_correct, 0/12 attempted, 0/12 applied.
  - bound 0.10 / trigger 0.15 / hotter: final 0.754530, recovery none, 0/12
    should_correct, 0/12 attempted, 0/12 applied.
- PR-8 selection policy: top two seed-42 cells by final NDCG; ties broken by lower
  trigger threshold, then profile name, then lower bound epsilon. The bound 0.10 /
  trigger 0.15 cells tied but were not confirmed because the lower-trigger tie-break
  selected trigger 0.05.
- PR-8 three-seed confirmation:
  - bound 0.10 / trigger 0.05 / default: mean final 0.760189 +/- 0.024486,
    recovery@12 1/3, versus C0 -0.004476, versus C2 -0.069022.
  - bound 0.10 / trigger 0.05 / hotter: mean final 0.760189 +/- 0.024486,
    recovery@12 1/3, versus C0 -0.004476, versus C2 -0.069022.
- PR-8 honest conclusion: negative sensitivity result. No grid or confirmation cell
  triggered correction; C3 loses to frozen C0 and to the C2 oracle.
- Verification: PR-8 local checks passed (`python -m unittest ...` 16 OK,
  `python -m ruff check .`, `git diff --check`, offline `scripts/smoke_pipeline.py`).
  GitHub checks all passed after converting the PR body to literal `FIELD:` lines.
- BHS: adversarial sub-agent review found and forced fixes for default-preservation
  evidence, tie-policy disclosure, calibrated-reference scoping, hotter-profile wording,
  and artifact durability. Operator waived the exact Tier-B-100 rule for this session;
  final focused review found no merge-blocking issue.
- Stretch PR-9: not started. The model-swap drift prototype remains the next useful
  arena because C2 cannot trivially undo that drift by re-embedding with the same model.
- Blockers: none for PR-6 through PR-8. Known host caveats remain: online Hugging Face
  SSL fetch fails on this machine; offline cache path works. Windows console Unicode
  logging errors appear on Greek SciFact query text but experiment artifacts exit 0 with
  `cycle_errors: []`.

## Deep review pass 2026-06-13 (Claude, ultracode workflow: 7 agents + 2 adversarial verifications)

- **Verified load-bearing finding:** the drift-recovery closed loop NEVER engaged.
  `correction_applied = 0/36` main matrix, `0/12` knob sweep (store byte-identical before/after).
  Three confirmed causes: (1) DEEP — no supervision target encodes pre-drift geometry, so
  ~zero correction is the correct optimum (C4 norm 0.000189 proves it; C3's 0.010000 is the
  BoundedAdapter floor manufacturing a delta); (2) shallow — `use_quantization=False` in the
  harness leaves `chelation_log` empty so sedimentation finds no candidates and early-returns;
  (3) wrong trigger signal — `global_variance` is drift-insensitive. Correction to earlier
  note: detection fires 36/36 at trigger 0.0; the 0/12 is only the knob sweep at nonzero
  thresholds.
- **Verified publish decision: do NOT publish to arXiv today.** Never-engaged actuator +
  tautological C2 oracle + self-admittedly under-severe rotation = under-powered null that
  risks first-author credibility. Path A: one sprint (supervised variant + model-swap arena),
  then methodology paper. Path B: Zenodo artifact release now to bank timestamp.
- **Full remaining-work analysis:** [remaining-work-2026-06-13.md](remaining-work-2026-06-13.md).
- **Next goal issued:** [goal-2026-06-14.md](goal-2026-06-14.md) — Track 0 hygiene (close
  block-flag debt CD-247-01/02, prune 9 merged branches, close PR #257, remove stale
  worktree), Track 1 model-swap arena (kills the C2 oracle + gives the loop a real target),
  Track 2 supervised synthetic variant, Track 3 paper drafting in parallel.
- **Phase II status:** steps 1–9 done; 11 minimal/in-progress; 14 apparatus-complete but
  exit criterion (documented recovery) unmet; 10/12/13/15/16/17 not started. Steps 12/13/15
  (DAG/disintegration/GNN) correctly blocked until the loop actually changes retrieval.

## GOAL ARC COMPLETE — 2026-06-21

8 PRs merged this session, each at a genuine fresh-adversarial Tier B 100:
#267 (Track 0 hygiene/debt-clear) · #268 (query-encoder drift defeats C2 oracle) ·
#269 (flaky-test fix) · #270 (arena harness C0/C2/C2O) · #271 (real-model smoke opt-in) ·
#272 (supervised loop C3a/C4a — actuator FIRES) · #273 (A4 campaign harness + budget knobs) ·
#274 (A4 real-data results). main @ 380eff4.

**Central question answered on REAL data (SciFact, MiniLM->mpnet encoder upgrade, 3 seeds):**
detection-triggered bounded correction PARTIALLY recovers catastrophic encoder-upgrade drift
(C0 frozen 0.006 -> C3a 0.159, ~25x; C4a unbounded 0.206), but with a CHARACTERIZED CAPACITY
CEILING (budget sweep plateaus 0.159->0.182 over 30->2000 steps; oracle C2O = 0.813). C2
re-embed-with-original is a proven no-op on real data (oracle defeated). Honest partial-recovery
result — the publishable headline.

Paper §5 written with the real result (paper-draft/main.md); header + §8 updated.

CD-A2-01 now satisfiable (A4 ran the real all-mpnet swap path) — close in next hygiene pass.

**Remaining secondary / follow-up (operator to direct):**
- A3: teacher-supervised C3b variant.
- B1/B2: supervised variant on synthetic drift (isolates "supervised recovers known drift").
- NFCorpus: 2nd-dataset breadth for the paper (cheap; confirms the partial-recovery finding generalizes).
- LaTeX conversion + arXiv/Zenodo submission decision.
- Phase II steps 12/13/15 (evidence DAG / disintegration / GNN) remain gated; the loop now DOES change
  retrieval (partially), so the gate is closer to satisfied than at 2026-06-13.

## 2026-06-28 (overnight Phase II execution — Claude)

**H1 (eval-split fix) — MERGED as PR #276 (`d5fc7df`).** Root-caused the NFCorpus C3a
baseline divergence: `_build_engine` attached a build-time *bounded* adapter for C3a/C4a,
and `embed()` applies engine.adapter to every ingested doc vector + the baseline query
path, so C3a's ~bound_epsilon floor contaminated its pre-correction baseline (C4a's
near-identity unbounded adapter did NOT diverge — which pinned the cause). Fix: restrict
the build-time override to synthetic {C3,C4}; C3a/C4a ingest+baseline on the default
near-identity adapter (identical to C0) and recreate the bounded/unbounded adapter per
correction cycle. Runtime evidence (NFCorpus seed42, offline GPU): all 5 conditions share
baseline 0.5550044219616403; C3a−C0=0.0 (was +0.0013). 22/22 harness tests pass (new H1
regression guard + a live-adapter bounded guard replacing a tautological `meta['bounded']`
check). Tier-B 82/important (stale C3a result docs + tautological test) → remediated →
100/none. CI all green; admin-merged.
- **CD-H1-01 opened**: committed swap result docs carry stale C3a numbers (C3a ingest/
  snapshot shifts → final NDCG moves; C0/C2/C2O/C4a unaffected). Disclosed inline in both
  result docs; closed by the H2 re-run.
- **H2 (re-run + paper numbers) IN PROGRESS**: full swap campaign re-run (SciFact +
  NFCorpus) with the H1 fix to regenerate both result docs, then H2 paper honesty fixes.
  Two earlier re-run attempts hit a spurious numpy 1.76 MiB ArrayMemoryError despite 28 GB
  RAM free — likely a concurrent in-process run plus a transient external spike;
  relaunching subprocess-per-cell with retry-on-failure + skip-if-exists.

## Overnight 2026-06-28 — Phase II execution (goal narrative: complete the lattice, then the paper)

Plan: `goal-narrative-phase-II-completion-2026-06-28.md`. Working slices H1–H6 (harden rung 14) then A1/B1.

- **H1 — MERGED as PR #276 (`d5fc7df`), Tier-B 100/none.** Root-caused the NFCorpus C3a
  baseline divergence flagged in the paper-readiness review: `_build_engine` attached a
  build-time bounded adapter for C3a/C4a, and `embed()` applies engine.adapter to every
  ingested doc vector + the baseline query path, so the bounded C3a floor (`bound_epsilon`)
  contaminated C3a's ingest/baseline (C4a's near-identity unbounded adapter did not — pinning
  the cause). Fix: restrict the build-time override to the synthetic conditions {C3,C4}; C3a/C4a
  ingest+baseline on the default near-identity adapter (identical to C0) and install the
  bounded/unbounded adapter per-cycle in `_supervised_anchor_cycle`. RUNTIME EVIDENCE (NFCorpus
  seed 42, offline GPU): all 5 conditions now share baseline 0.5550044219616403; C3a−C0=0.0
  (was +0.0013). Tests: replaced a build-time `isinstance(BoundedAdapter)` assertion (it
  asserted the bug) with an H1 regression guard + a live-adapter spy on
  `_apply_adapter_to_all_docs` (non-tautological). 22/22 harness tests pass; ruff clean.
  Tier-B loop: iter1 82/important (tautological test + undisclosed stale C3a result docs) →
  remediated (real spy + inline "superseded — pending re-run" disclosure in both result docs +
  CD-H1-01) → iter2 100/none.
- **CD-H1-01 OPEN** (next-session.md): H1 changes C3a's ingest/baseline/snapshot, so the
  committed swap result-doc C3a rows + budget sweeps are stale. **H2 re-run IN PROGRESS**
  (background, branch `feat/drift-h2-swap-rerun`): both SciFact + NFCorpus swap campaigns on the
  H1-fixed harness, offline GPU, regenerating both auto-gen result docs. On completion: commit
  regenerated docs/manifests, refresh paper §5 (local), clear CD-H1-01.
- Tooling note for this session: the Bash tool intermittently fails to parse commands
  ("line 105 unexpected EOF matching '") — even simple ones; PowerShell is reliable. Run python
  via script files (sys.path.insert + os.chdir to the repo) rather than `python -c`. Background
  jobs: launch via PowerShell `run_in_background`, not Bash.

### H5 design recon (2026-06-28) — post-bank proving experiment

Building blocks that already exist (reuse, don't reinvent):
- **Harness seam EXISTS** (a recon agent wrongly called it a "gap"): `run_drift_recovery_experiment.py`
  has `CONDITIONS`, `_run_condition_cycle(engine, condition, ...)` dispatch, and `_supervised_anchor_cycle`
  as the per-cycle template. A new post-bank condition (e.g. `C5`) plugs in by mirroring
  `_supervised_anchor_cycle` + adding to `CONDITIONS` + a branch in `_run_condition_cycle`.
- **`adapter_router.py` `AdapterRouter`** already does centroid-keyed routing over a BANK of adapters
  (`register(key, centroid, adapter)`, `select(query_vec) -> route`, `record_outcome(...)`), on
  retrieval-store geometry. This is the ready-made basis for the two baselines the proving experiment
  needs to beat: the **frozen SVF-style static bank** (register posts once, never re-anneal) and the
  **one-shot LD-MoLE-style router** (route once, no lifecycle).
- **`create_adapter`** supports mlp/procrustes/low_rank/quant_low_rank/attnres + `BoundedAdapter`
  (min/max_correction). Bank members = per-cluster bounded adapters trained on anchor pairs.
- **Vector store** snapshot/writeback via `scroll`/`retrieve`/`upsert` (already used by
  `_snapshot_doc_points` + `_apply_adapter_to_all_docs`) — the lifecycle prune/re-anneal mutates the
  store the same way.
- **Tests** use stub backends (TinyEmbeddingBackend/StubSwapBackend) — a new condition is testable
  without real models.
- **Gap for H6:** `set_temperature` is only a scalar SCORE-scaler (monotonic, doesn't reorder), NOT an
  annealing scheduler. H6 must add a real temperature schedule driving the bank's explore/stabilize/prune
  phases — it is genuinely new, not a wrapper over `set_temperature`.

Planned H5 conditions (mirror `_supervised_anchor_cycle`, all scored on the same eval subset):
`C5` living annealed-route post-bank · `C5s` frozen SVF-static bank baseline · `C5r` one-shot
LD-MoLE-style router baseline. DoD: C5 beats C5s AND C5r on final NDCG with the prune/re-anneal
lifecycle exercised (store mutation logged), 3 seeds, both datasets. This is the load-bearing slice.

### Progress addendum (2026-06-28, later)

- **Active H2 re-run branch is `feat/drift-h2-rerun-c3a`** (supersedes the earlier
  `feat/drift-h2-swap-rerun` mention). Re-run = subprocess-per-cell, retry(3)+timeout(1800s),
  skip-if-exists; fresh (all swap cells cleared first). Device journey: GPU hit a transient
  `0xC0000005` access violation → tried CPU via `CUDA_VISIBLE_DEVICES=""` which is flaky
  (empty-string ambiguously interpreted) and exposed a real telemetry bug → reverted to GPU,
  which is now running healthy (no crashes; retry untriggered). NFCorpus per-cell JSONs are
  TRACKED (committed in #275); SciFact per-cell JSONs are gitignored — only manifests + docs
  are the tracked H2 deliverables to commit.
- **Telemetry CUDA-guard fix MERGE-PENDING on the H2 branch (`6c3e184`):**
  `AntigravityEngine.get_runtime_telemetry` called `torch.cuda.get_device_name(0)` guarded only
  by `is_available()`; with CUDA hidden (`CUDA_VISIBLE_DEVICES=""`/`-1` / CI) `is_available()` can
  be True while `device_count()==0` → "Invalid device id" crash on every CPU-only run. Added a
  `device_count() > 0` guard + `test_engine_telemetry_cuda_guard.py` (3 tests, stub engine, no GPU).
  Ships in the H2 PR.
- **B1 (Evidence DAG schema, rung 12) IMPLEMENTED (untracked, ready to commit on its own branch):**
  `evidence_dag.py` (typed QUERY/CLUSTER/ACTUATOR nodes; RETRIEVED_IN/CORRECTED_BY/OPERATES_ON edges
  with endpoint-type contract; JSON (de)serialize; `validate_evidence_dag` = unique ids + known types
  + edge endpoints exist + endpoint-type match + acyclicity; `EVIDENCE_DAG_JSON_SCHEMA`;
  `from_attribution_pool` deterministic builder) + `test_evidence_dag.py` (12 tests, stdlib-only).
  12/12 pass, ruff clean. Kept untracked to avoid git contention with the live re-run; commit on
  `feat/evidence-dag-schema` once the tree is clear, then PR (Tier-B).
- **H4 (one-shot question) RESOLVED by characterization** —
  `h4-one-shot-characterization-2026-06-28.md`: flat trajectories are intentional idempotency
  (per-cycle re-init from a fixed pre-drift snapshot; asserted by `test_c3a_multi_cycle_is_idempotent`).
  Compounding rejected (breaks idempotency + drifts off the supervised target). Paper frames it as a
  one-shot supervised realignment (§5.4/§8). No code change.
- **Zenodo priority-insurance bundle STAGED (not uploaded):** `zenodo-staging/` (README + .zenodo.json
  + FILES.md), flagged for operator (license/author/final-numbers are explicit TODOs). Local-only.

### B1 MERGED + campaigns resumed (2026-06-29 morning)

- **B1 — Evidence DAG schema (rung 12) MERGED, PR #277 (`81e3614`), Tier-B 100/none.** Typed
  QUERY/CLUSTER/ACTUATOR graph + endpoint-type-contracted edges + acyclicity validator + draft-07
  JSON schema + deterministic `from_attribution_pool` builder; schema-first, stdlib-only, no GNN.
  Tier-B loop: 72/critical → 100/none. The critical finding was a REAL builder bug: it gated
  actuators on a fictional `REFORM` token (an `action_mix` count key, not a per-row action), masked
  by a fictional-token test fixture → would have silently dropped every REFORMULATE/CHELATE_ALWAYS
  correction. Fixed to the canonical `{CHELATE, CHELATE_ALWAYS, REFORMULATE}` vocabulary
  (`adaptive_overlay.infer_aggression_level`) with a discriminating test. Done via an isolated
  worktree so the live H2 re-run was never disturbed. Goal-narrative rung-12 row → partial.
- **Machine freed (~23 GB RAM, GPU ~23 GB free); NFCorpus re-run RESUMED on GPU** (`b24ehsq01`) —
  both datasets now uniformly GPU. SciFact already regenerated (C3a 0.161; headline unchanged by
  H1, confirming it was a cleanliness fix). On NFCorpus completion: commit regenerated
  docs/manifests + telemetry fix → H2 PR → clear CD-H1-01 → finish paper §5.2/§5.3.
- **Morning status:** `morning-status-2026-06-29.md`.
- **Next:** finish H2, then H3 (C3b) and H5 (post-bank — load-bearing) now that the GPU is free.

### 2026-06-29 — operator re-plan: GPU-deferred code-first session

Operator reserved the 3090 for a concurrent Grok documentation workstream (branch
`feat/brain-file-map-b0-b1` — it switches the main working tree's branch in waves, which is
what corrupted the earlier in-tree H2 NFCorpus cells; ALL my work now runs in isolated
worktrees). New plan: `goal-narrative-gpu-deferred-codefirst-2026-06-29.md` — build the
lattice machinery code-first, stub-tested, defer every GPU campaign until the card frees.

- **H2 (numbers) DEFERRED to the 3090.** SciFact regenerated + paper §5.1 updated; NFCorpus
  re-run paused (worktree `.claude/worktrees/h2-rerun` holds partial valid cells; resume +
  verify `C3a baseline == C0` before committing). CD-H1-01 stays open.
- **S1 / H5a — steering-post bank MERGED, PR #279 (`06f4f5c`), Tier-B 100/none.**
  `steering_post_bank.py`: bank of correction posts (centroid-routed), `apply` with
  store-mutation logging, fitness→prune→re-anneal lifecycle with temperature modulation +
  min_posts floor + drift-gated `prune_and_reanneal`. numpy-only, torch-decoupled, 18 tests.
  Tier-B loop 88/important (temperature-floor × min_posts coverage gap) → discriminating
  regression test → 100. The novel-core substrate; the C5/C5s/C5r head-to-head campaign is
  GPU-deferred.
- **H6 / rung 11 — annealing temperature schedule MERGED, PR #280 (`9db098e`), Tier-B
  100/none.** `annealing_schedule.py`: the one temperature schedule driving the bank's
  explore→stabilize (constant/linear/cosine/step/adaptive), stdlib-only, 15 tests. Distinct
  from the engine scalar `set_temperature`. Tier-B loop 97/cosmetic (3 non-discriminating
  tests) → tightened (cosine≠linear, step transition index, init-clamp) → 100. CI: 3.9
  flaked on a transient HF 429 (unrelated to this stdlib-only code; 3.10–3.12 green); re-ran
  3.9 green, admin-merged.
- **S2a / H5b — post-bank-from-anchors builder MERGED, PR #281 (`65d3bbf`), Tier-B 100/none.**
  `post_bank_correction.py`: `cluster_vectors` (seeded k-means) + `build_post_bank` (one post
  per cluster, keyed by centroid, correction from a caller-supplied `make_post` factory).
  Decoupled from engine/torch, 11 tests. Tier-B loop 78/important (the empty-cluster re-seed
  was non-reentrant → left a cluster empty on duplicate/degenerate input) → reentrant fix
  (steal worst-fit member of the currently-largest cluster, counts recomputed per empty) +
  degenerate regression tests → 100. The adversarial loop caught a real bug here AND in B1
  (the fictional-`REFORM` builder bug) — 2 of the last 3 slices.
- **All three post-bank building blocks now on main:** the bank (S1 #279), the schedule
  (H6 #280), the builder (S2a #281). **Next: S2b** — wire C5 (living bank) / C5s (frozen
  static) / C5r (one-shot router) conditions into `run_drift_recovery_experiment.py`,
  consuming all three against the engine + real bounded adapters; stub-tested. That is the
  last code slice before the experiment; the C5-vs-C5s-vs-C5r **campaign stays GPU-deferred**.
  Then S3 (H3 C3b code). All in isolated worktrees.
- **S2b-runtime / H5b — post-bank build+apply glue MERGED, PR #282 (`cc58030`), Tier-B
  100/none (code).** `post_bank_runtime.py`: `build_post_bank_from_anchor_pairs` +
  `apply_post_bank_to_store` (route each doc → apply post correction → upsert back; faithful
  corrected−original norm delta; idempotent on a fixed snapshot). numpy-only, fake-store
  tested, 6 tests.
- **⚠️ Tooling note: the SUBAGENT Bash tool is broken session-wide** (`/usr/bin/bash: line 108:
  unexpected EOF` on bare `true`). Tier-B reviewer agents cannot capture test/ruff runs; they
  verify code-clean by full-source trace, and the IMPLEMENTER captures the green run
  (main-session PowerShell works). For S2b-runtime two fresh reviewers found the code 100-clean
  on all axes; raw numerics (96, 90) were withheld solely for the un-captured run — fully
  disclosed in the PR. Workaround stands for remaining slices.
- **Post-bank plumbing COMPLETE on main:** bank (S1 #279) + schedule (H6 #280) + builder
  (S2a #281) + build/apply glue (S2b-runtime #282). **Final code slice: S2b-conditions** —
  wire C5 (living bank) / C5s (frozen static) / C5r (one-shot router) into
  `run_drift_recovery_experiment.py` with a real per-cluster bounded-adapter `make_post` + the
  anneal/prune/re-anneal lifecycle (for C5), stub-tested. Then the GPU campaign (deferred).
- **S2b-conditions / H5b — C5/C5s/C5r + lifecycle MERGED, PR #283 (`7e8443f`), Tier-B
  100/none.** `post_bank_conditions.py`: `evolve_or_build_bank` (pure build-once-vs-evolve +
  prune/re-anneal), `make_adapter_post_factory` (per-cluster bounded adapter via InfoNCE),
  `run_post_bank_cycle` (trigger → build/evolve → apply). Harness: CONDITIONS += C5/C5s/C5r,
  threads `cycle_index`, dispatches; config gains post_bank_clusters/post_prune_below/
  post_min_posts (campaign sweep knobs). 11 unit + 1 end-to-end swap-arena test (all 3 fire +
  mutate the store; C5 prune/re-anneal FORCED + asserted end-to-end). Tier-B loop 92/important
  (the end-to-end C5 lifecycle assertion was conditional + temperature-only) → forced-prune +
  unconditional assertion → 100. 34/34 pass.
- **🎯 THE POST-BANK EXPERIMENT IS CODE-COMPLETE ON MAIN:** bank (S1 #279) + schedule
  (H6 #280) + builder (S2a #281) + glue (S2b-runtime #282) + conditions/lifecycle
  (S2b-conditions #283). The H5 experiment is fully implemented + stub/integration-tested at
  BHS 100. **Only remaining: the GPU head-to-head campaign** (C5 vs C5s vs C5r — the thesis
  verdict) + H2's NFCorpus re-run, both parked for the operator's 3090.
- **A1a / rung 10 — executable steering rollback plan MERGED, PR #284 (`efe6d1b`), Tier-B
  100/none.** Additive to `model_scope_steering.py`: `RollbackPlan` + `build_rollback_plan` +
  `execute_rollback` (reverse-order replay of `rollback_feature_event` → restores the EARLIEST
  pre-steering value; the hardcoded rollback descriptor made executable). 6 tests + 56 existing
  steering tests still green (no regression). First piece of the rung-10 SHIM decomposition
  (plan: `a1-rung10-shim-implementation-plan-2026-06-29.md`).
- **Next: A1b** — wire `quantization_promotion_gate` to a promoted steering route
  (quantization-survival gate) + strengthen `promotion_contract.require_rollback_path` from
  "string present" to "rollback plan actionable". Then **A1c** — production-seam un-guarding +
  the rung-10 DoD doc.
- **Session tally (code-first, GPU-deferred): 8 PRs merged** — H1 #276, B1 #277, S1 #279,
  H6 #280, S2a #281, S2b-runtime #282, S2b-conditions #283, A1a #284. All campaigns parked
  for the 3090.
- **A1b / rung 10 — steering-route promotion gate MERGED, PR #285 (`49b3604`), Tier-B
  100/none.** `steering_route_promotion.py`: `evaluate_steering_route_promotion` composes the
  quantization-survival gate + A1a's executable RollbackPlan into a FAIL-CLOSED route-promotion
  decision (promotable only if quant survives AND rollback is actionable; strict `isinstance`
  guard rejects string-path impostors). 8 tests. The rung-10 "quantization-survival passes for
  a promoted shim route + rollback-backed" gate.
- **Next: A1c** — production-seam un-guarding (live-fire flags → production-wired control plane,
  default-safe, USING the A1a rollback + A1b promotion gate) + the rung-10 DoD doc. Final
  rung-10 piece.
- **A1c / rung 10 — production steering control plane + DoD MERGED, PR #286 (`d8181f6`),
  Tier-B 100 (one minor doc-prose note, softened).** `production_steering_control.py`:
  `resolve_production_mode` — live steering (SOFT_SCALE/SUPPRESSION) permitted only for an
  A1b-promoted route, else default-safe SHADOW; fail-closed on None/attr-missing. Plus
  committable `docs/rung10-shim-substrate-dod.md`. 6 tests.
- **🎯 RUNG 10 (SHIM substrate) SUBSTANTIALLY COMPLETE:** A1a (#284 executable rollback) + A1b
  (#285 quantization+rollback promotion gate) + A1c (#286 default-safe production control plane
  + DoD). The control-plane AUTHORITY is built, tested, default-safe. **One operational step
  honestly deferred** (per the DoD): wire `run_live_fire_diagnostics --enable-model-scope` to
  consult `resolve_production_mode` — a thin run-script change, not rushed into a campaign path.
- **Session tally: 10 PRs merged** — H1 #276, B1 #277, S1 #279, H6 #280, S2a #281,
  S2b-runtime #282, S2b-conditions #283, A1a #284, A1b #285, A1c #286. **Phase II Part B (A1
  rung-10 + B1 rung-12) + the full H5 post-bank apparatus (rungs 11+14) are now CODE-COMPLETE.**
  All GPU campaigns (H5 verdict, H2, H3) parked for the 3090.
- **H3a / H3 (C3b) — teacher-supervised distillation module MERGED, PR #287 (`44ded13`),
  Tier-B 100/none.** `teacher_supervised_correction.py`: `build_teacher_pairs` (aligns old-doc
  vectors with the swap-encoder re-embedding of each doc's text = the teacher) +
  `train_distillation_adapter` (MSE distillation adapter(doc)→teacher, the new signal vs C3a's
  InfoNCE; returns initial/final MSE to PROVE it learns). 4 tests (real CPU adapter: MSE drops
  <half on unit embeddings). DISCLOSED CAVEAT: C3b's teacher is oracle-derived (the C2O swap
  re-embedding); H3b/campaign must frame it honestly. (CI: 3.10 runner stuck; green on
  3.9/3.11/3.12 + all gates; admin-merged.)
- **Next: H3b** — wire C3b as a harness condition in `run_drift_recovery_experiment.py` (build
  teacher pairs from store-payload doc text + `query_drift.embed_queries`, train via H3a's
  distillation, apply to all docs), stub + swap-arena tested. Then the C3b campaign (both
  datasets × 3 seeds, GPU-deferred).
- **Session tally: 11 PRs merged** — H1 #276, B1 #277, S1 #279, H6 #280, S2a #281,
  S2b-runtime #282, S2b-conditions #283, A1a #284, A1b #285, A1c #286, H3a #287. All GPU
  campaigns parked for the 3090.
- **H3b / H3 (C3b) — teacher-supervised condition WIRED, PR #288 (`2ee1325`), Tier-B 100/none.**
  `_supervised_anchor_cycle` C3b branch: builds teacher pairs (anchor doc vec ↔ swap re-embed of
  doc text via new `_doc_texts_by_id` + `query_drift.embed_queries`), trains `adapter(doc)→teacher`
  via MSE distillation (H3a), applies to all docs. `query_drift` threaded; C3b in CONDITIONS +
  SUPERVISED_CONDITIONS; action=`supervised_teacher_distillation_correction`, bounded. InfoNCE
  path (C3a/C4a) byte-equivalent in `else` (no regression). End-to-end swap-arena test: C3b FIRES
  + MUTATES the store (norm > bounded floor 0.01). 24/24 harness tests. **⚠️ DISCLOSED:** C3b's
  teacher is the C2O ORACLE signal — not oracle-free recovery; the campaign must frame it so.
  **H3 is now CODE-COMPLETE** (H3a module #287 + H3b wiring #288); the recovery campaign is GPU-deferred.
- **🎯 EVERY CODE SLICE IN THE GOAL IS DONE:** H1 #276 · B1 (rung 12) #277 · H5 post-bank apparatus
  rungs 11+14 (#279-283) · rung-10 SHIM #284-286 · H3 C3b #287-288. Remaining = GPU campaigns only
  (H5 verdict, H2 re-run, C3b/H3 campaign, H4 trajectory) + the deferred A1c run-script adoption.
- **Session tally: 12 PRs merged** — H1 #276, B1 #277, S1 #279, H6 #280, S2a #281, S2b-runtime #282,
  S2b-conditions #283, A1a #284, A1b #285, A1c #286, H3a #287, H3b #288. All GPU work parked for the 3090.
- **H2 LOCAL paper-honesty fixes — COMPLETE (GPU-free part of H2).** Verified/added all 5 in
  `paper-draft/main.md`: (1) "~25× not 26×" — no 26× remains; (2) **2.25% NFCorpus oracle-gap now
  stated explicitly** in §5.3 (was qualitative "smaller fraction") + the honest-framing note ("read
  the oracle-gap fraction, not the 25×"; 19% SciFact / 2.25% NFCorpus); (3) one-shot disclosed (§5.2
  "recovery saturates at cycle 1"); (4) 60-query eval subset clarified (§4.4); (5) budget inversion
  stated (§5.3). REMAINING H2 = the NFCorpus §5-number re-run (GPU-gated) — the single-digit oracle-gap
  framing holds regardless.
- **Zenodo priority-insurance bundle — STAGED + FLAGGED.** `zenodo-staging/` (README w/ both hazard
  findings + manifest + .zenodo.json) verified against main (only `test_engine_telemetry_cuda_guard.py`
  is H2-branch-pending; manifest flags it). Operator actions noted in morning status (license/author
  TBD, pull regenerated numbers, zip). Not uploaded (guardrail).
- **A1c run-script adoption — INVESTIGATED + RESOLVED (not deferred), PR #289 (`d1bfc09`),
  Tier-B 100/none.** Verified `run_live_fire_diagnostics.py --enable-model-scope` is
  OBSERVATION-ONLY (`enable_model_scope_observation()`; reports `observation_count`; no
  `SteeringActuator`/`SOFT_SCALE`/`SUPPRESSION`/`.apply` in the run path). So there is NO live
  steering seam to retrofit — the path is already default-safe by construction. The DoD's "one
  remaining wiring step" framing was wrong; corrected it: live-route introduction is a FUTURE
  feature (overlaps rung 16), and `resolve_production_mode` already gates it for if/when a live
  route exists. **Rung 10 is now definitively complete with no hanging step.**
- **🎯 EVERY GPU-FREE GOAL ITEM IS DONE — 13 PRs merged.** H1 #276, B1 #277, S1 #279, H6 #280,
  S2a #281, S2b-runtime #282, S2b-conditions #283, A1a #284, A1b #285, A1c #286, H3a #287,
  H3b #288, rung10-DoD-fix #289. Plus H2 local paper-honesty fixes (all 5) + Zenodo bundle
  staged/flagged. The ONLY remaining work is GPU campaigns (H5 verdict, H2 NFCorpus re-run,
  C3b/H3 campaign, H4 trajectory) — all blocked solely on the operator's 3090 reservation.
- **H4 — "characterize why one-shot is the fixed point" branch RESOLVED (GPU-free).** Added the
  characterization to `paper-draft/main.md` §5.2: one-shot is a *designed* fixed point (each cycle
  re-inits the adapter from a fixed seed, trains on the same anchors, applies to the same frozen
  pre-drift snapshot → reproduces the identical correction). The idempotent re-init was chosen
  over compounding because compounding OVERSHOT (recorded norm escalation 0.136→0.497→0.443 — a
  documented finding in the `_supervised_anchor_cycle` code comment, not fabricated). The H4
  "make cycles compound" trajectory re-measurement (a `compound_cycles` ablation) is the GPU
  branch, noted as a small future experiment that doesn't change the partial-recovery headline.
- **DEFINITIVELY: zero GPU-free goal items remain.** H1✓ H2-local✓ H3-code✓ H4-characterize✓
  H5-code✓ H6✓ A1✓(+DoD finding) B1✓ + Zenodo✓. Every remaining item (H2 NFCorpus re-run, H3
  campaign, H5 verdict, H4 compound-trajectory) is GPU-gated. The single lever is the 3090.
- **⚠️ REAL GAP FOUND + FIXED — the H5 campaign was NOT runnable. PR #290 (`6d3af30`), Tier-B
  100/none.** The swap campaign driver only ran `MAIN_CONDITIONS=(C0,C2,C2O,C3a,C4a)` — C5/C5s/C5r
  (and C3b) had **no campaign runner**, so "free the GPU → run H5" was impossible. Added
  `run_condition_head_to_head(config, conditions, runner)` + `_post_bank_verdict` (the H5 gate:
  living_bank_wins iff C5 > C5s AND C5 > C5r) + `render_head_to_head_report`. Runner-injectable
  (5 stub tests; existing campaign 5/5, no regression). Same driver serves the H3 C3b head-to-head.
  **This corrects the earlier inaccurate "one command from running" claim — now it is TRUE.**
- **CAMPAIGNS ARE NOW ACTUALLY RUNNABLE (driver-wise) the moment the 3090 is free:**
  - H5: `run_condition_head_to_head(SwapCampaignConfig(task=..., output_dir=..., report_md=...))`
    (default C0/C2O/C5/C5s/C5r) per dataset.
  - H3 C3b: `run_condition_head_to_head(cfg, conditions=("C0","C2O","C3a","C3b"))` per dataset.
  - H2 NFCorpus re-run: existing `run_swap_campaign(SwapCampaignConfig(task="NFCorpus", ...))`.
  All offline (HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1), sequential, nvidia-smi first.
- **Session tally: 14 PRs merged** — H1 #276, B1 #277, S1 #279, H6 #280, S2a #281, S2b-runtime #282,
  S2b-conditions #283, A1a #284, A1b #285, A1c #286, H3a #287, H3b #288, rung10-DoD #289,
  H5-campaign-driver #290. Plus H2-local + H4-characterize + Zenodo. Only GPU execution remains.
- **H4 'make cycles compound' branch — capability ADDED, PR #291 (`34ce4b5`), Tier-B 100/none.**
  `DriftRecoveryConfig.compound_cycles` (default False): when True the supervised cycle applies
  the adapter to the LIVE store each cycle (corrections stack) instead of the fixed snapshot.
  Discriminating test (correction-norm flat for one-shot vs changing for compound; fails if the
  flag is ignored). Default-False byte-equivalent (no regression), C5/C5s/C5r isolated. 25/25
  harness tests. **BOTH H4 branches now done** (characterize in the paper + compound capability).
  GPU ablation (overshoot magnitude) deferred.
- **🎯 EVERY CODE BRANCH OF EVERY GOAL ITEM IS DONE — 15 PRs.** The two real gaps found by
  continuing to dig (H5 campaign DRIVER #290 was missing → campaigns weren't runnable; H4
  compound CAPABILITY #291 had no code) are both closed. Remaining = GPU EXECUTION only
  (H5 verdict, H2 NFCorpus re-run, C3b/H3 campaign, H4 compound ablation) — all one command via
  the merged drivers, blocked solely on the 3090.
- **Session tally: 15 PRs merged** — #276 #277 #279 #280 #281 #282 #283 #284 #285 #286 #287
  #288 #289 #290 #291. Plus H2-local fixes + H4-characterize + Zenodo staged/flagged.
- **Minor foot-gun (documented-around, not PR'd — marginal):** `SwapCampaignConfig.report_md`
  defaults to the A4 report path (`docs/drift-recovery-swap-results-2026-06.md`), shared by both
  `run_swap_campaign` and the new `run_condition_head_to_head`. The head-to-head MANIFEST path is
  distinct (`post-bank-headtohead-manifest`), but a bare default config would write the REPORT to
  the A4 path. The morning-status fire-commands use EXPLICIT `output_dir`/`report_md` per campaign,
  so this never triggers in documented use. A future tidy-up could give the head-to-head its own
  default report path; not worth a PR + Tier-B against the documented-explicit-path usage.
- **VERIFIED-COMPLETE on the GPU-free side.** Drivers + conditions + knobs all exist and are
  composition-verified runnable; the only untestable-without-GPU piece is the full real-model
  end-to-end run (the campaign numbers). Not fabricating them; not running real models on the
  operator-reserved shared machine without an OK. Blocked solely on the 3090.

### 2026-06-29 — operator-allowed VALIDATION SMOKE (real models, tiny; NOT the campaign)

Operator's earlier carve-out ("very small testing using GPU just to make sure the code is
working") — ran `run_condition_head_to_head` end-to-end on the REAL swap models (CUDA, mpnet
768-dim), 1 seed / 15 docs / 6 queries / 2 cycles / conditions C0,C5,C5s,C5r. nvidia-smi gated
(GPU 2GB/24GB used, RAM 12GB free — not tight). Output discarded (not committed).

- **✅ DRIVER VALIDATED END-TO-END.** It produced a valid manifest + report + verdict on real
  models. The H5 pipeline genuinely runs (closes "I can't test the real run"). The C5/C5s/C5r
  conditions fire + mutate the store (applied 1/1 each) on real data.
- **⚠️ TWO HONEST PRE-CAMPAIGN SIGNALS (NOT the verdict — far too small):**
  1. **The post-bank corrections did NOT help at this scale:** C0 (frozen) 0.5660; C5 & C5s
     0.5550 (BELOW frozen); C5r 0.5660. Corrections slightly hurt / no-op on 15 docs / 2 anchors.
  2. **C5 == C5s EXACTLY → the living-bank LIFECYCLE DID NOT ENGAGE.** With default knobs +
     2 cycles, no post fell below the prune threshold, so no prune/re-anneal fired and the living
     bank never diverged from the static bank. **The goal's gate is "lifecycle EXERCISED, living
     bank beats both" — and the default-knob run exercises NOTHING.**
- **ACTION FOR THE REAL CAMPAIGN (important):** the head-to-head MUST be configured so the
  lifecycle actually fires — e.g. raise `post_prune_below` and/or use enough cycles + a temperature
  schedule that drives prunes, and enough anchors→clusters that fitness varies. Otherwise C5 ≡ C5s
  and there is no "living" advantage to measure (the verdict would be a trivial tie, not a real
  test). This is the single most important thing to get right before burning GPU on the full run.
  The current `run_condition_head_to_head` uses _run_config's DEFAULT post-bank knobs (clusters 3,
  prune_below 0.5, min_posts 1) — a follow-up should expose/raise these for the H5 campaign.
