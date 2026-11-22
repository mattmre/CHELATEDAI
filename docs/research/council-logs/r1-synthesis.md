# Tri-Model Council — Refinement Round 1 Synthesis & Mandate (2026-11-21)

**Council Roster:** `gemini-4-carbon` (`thought_level=high`, max supported), `gemini-4-neon` (`thought_level=xhigh`), `gpt-6.2-astra` (`reasoningEffort=xhigh`)  
**Stage:** **Refinement Round 1 of 3** (Post-Cross-Check Enrichment, New User Architectural Extensions & Exact Convergence)  
**Read-Only Rule:** Do **NOT** edit repository files during R1–R3. Run only read-only in-memory Python/inspection probes. All code, tests, benchmark reruns, Doctoral Whitepaper v2, and Presentation HTML v2 edits occur in the consolidated Implementation Pass after R3 locks.

---

## 1. Cross-Check Reports Archived (3/3 Complete)
- **Carbon Cross-Check:** `D:\GITHUB\CHELATEDAI-STAGE0-NIGHT1\docs\research\council-logs\xc-carbon.md`
- **Neon Cross-Check:** `D:\GITHUB\CHELATEDAI-STAGE0-NIGHT1\docs\research\council-logs\xc-neon.md`
- **Astra 6.2 Cross-Check:** `D:\GITHUB\CHELATEDAI-STAGE0-NIGHT1\docs\research\council-logs\xc-astra.md`

## 2. Unanimously Confirmed Cross-Check Foundations (Locked — Do Not Re-Litigate)
All 3 models independently reproduced and confirmed:
1. **Pristine Preservation:** `archive/pristine-b37e6a55/` (`15/15` BOM/LF-damaged, preserved as forensic evidence) vs. `archive/pristine-b37e6a55-exact/` (`15/15` byte-exact SHA-256 match to tag `baseline/pre-fix-pristine-2026-11-21` @ `b37e6a55`, `ast.parse` OK, read-only).
2. **14 Verified Receipt Bundles:** `12` historical (`0` CSV hashes; `f2_longmemeval` v1 superseded by v2; `s1` wall time `1,226.40s`; `f2_iq04` full SHA `148d80bf3faadd77f9ad0f9eeb1f1a089d8e87b89025c9f645d4a685ad7e619b`) + `2` `20261121-rigor-r3` bundles (`6/6` listed artifact hashes verified).
3. **`EK5 v2` (`HEAD`) Critical Defects:** (a) computes Fitting's Kripke–Kleene fixed point (`200/256` 2-node, `2,352/4,096` 3-node disagreements vs. true alternating-fixpoint WFS); (b) **Zombie Reinstatement** under 2-step `SUPERSEDE` ($e_3 \to e_2 \to e_1$ resurrects $e_1$ and its derived copies); (c) unconditional forward BFS causes `~10k`-node out-hub blowup on zero-delta writes + `10,001`-edge in-hub scan; (d) lacks bitemporal `(valid_time, tx_time)` cutoffs, drops out-of-order edges, coalesces edge multiplicities, and returns `{}` on initial node insertion.
4. **Naive Counter-Gating Unsound on Cycles:** Unranked DRed counters / dead-attacker skips fail cyclic negation (`4/256`, `364/4,096`, `57/2,000`). Fixed by **Astra's linear-time positive-SCC-collapse WFS theorem** (`262,144/262,144` exhaustive 3-node exactness in $O(|D|+|E_{\text{incident}}(D)|)$) + **Carbon's Fitting-stratum-rank shield (`rank(u) < rank(w)`) / Neon's backward-independence witness** (`0/25,000` random ops, `0–3` nodes on 10k in/out hubs).
5. **Flagship #1 Exact Math & Bugs:**
   - **Theorem 1 (Three-Regime Stiefel Trichotomy):** Universal full-rank $\kappa = 1.00$ **iff monotone**; universal equal nonzero singular values ($\kappa_+ = 1.00$) **iff valley-shaped (quasiconvex, no interior peak $d_{k-1} < d_k > d_{k+1}$)**; interior peaks yield Jordan principal cosines $\cos\theta_i$ (`384→768→384` $\kappa \gg 1$).
   - **Repaired Proposition 1b:** For fixed Stiefel $Q$, $s_{\text{LS}}(Q) = \frac{\langle X_c Q, Y_c \rangle_F}{\|X_c Q\|_F^2} = \cos_F(X_c Q, Y_c) \cdot s_{\text{rad,proj}}$ (Frobenius cosine, NOT canonical correlation).
   - **Out-of-Sample Projected Radius Overfitting (`Carbon XC §2.2`):** When $n < d_{\text{in}}$ (`n=500, 4096→384`), in-sample `||X_c Q||_F` still suffers a **65.5% OOS radius collapse** (`0.3446` vs `0.9983` with held-out unlabeled validation split $X_{\text{val}} Q$)! Also reject degenerate constant inputs with `ValueError` (eliminating `scale = 2e12`).
   - **Backfill & Ties:** `BF20` beats `A3+CSLS` on only `9/18` cells; `score_quantile_calibrated_backfill` ignores `csls_k` and clamps right tails causing alphabetical top-1/top-2 inversion; two `argpartition` tie bugs survive (`m1_baseline_ladder.py:309`, `night1_common.py:399`); `scripts/night1_common.py:10` breaks Windows with unguarded `import resource`.
6. **Flagship #2 Runner Bugs:** Part A is `150/150` solvable by a 4-line observable rule, mixes `valid_time <= retracted_tx`, rewards unlinked `val_sup_cnt`, and lets unrelated sibling `ADMIT` suppress `CONTESTED_ABSTAIN`; Part B `LoCoMo` took `117 conv-26 + 3 conv-30`, had `15/56` distractor-gold overlaps, compound ID `'D8:6; D9:17'`, omitted `M1c_hybrid_jaccard_dedup`, suffered a `3-key / 25-row` concurrent judge-cache race, and had `M4d` order-dependent tag overwrite; `LongMemEval-S` v1→v2 shifted `2.56 pp` from generator text changes (`judge_diffs_same_pred = 0`), and `cc5ded98` flipped with byte-equivalent reconstructed dated prompts.
7. **NEWLY LANDED VERIFIED `f1_4096_ood` RECEIPT (`15th` Bundle! `artifacts/flagship1-night1/20261121-rigor-r3/f1_4096_ood/f1_4096_ood_manifest.json`, SHA-256 `ce81b93031501542ffcd2dd1291b1e2014eea732e28a44add58b34e47e06f1f0`, `3,479.66s` on `NVIDIA GB10`):**
   - **In-Domain Extreme Dimension Jumps (`384→1024/2560/4096`, `n_anchors=2000`, uncapped `SciFact` 5,183 docs & `CoSQA_Code` 20,604 docs; linear & `2^r-1` identical on binary qrels):**
     - **6/6 wins over Centered Affine Ridge on query down-projection (`D384→D1024/2560/4096`):** `SciFact D384→D4096 (10.7×)`: **`A3 = 0.6599`** (96.1% of native `0.6866`) vs `Centered Affine Ridge = 0.6193` (`+0.0406`) vs `Uncentered Ridge = 0.6316` vs `A2c = 0.6080`; `CoSQA_Code D384→D4096`: **`A2c = 0.2549`**, `A3 = 0.2493` vs `Centered Affine Ridge = 0.2343` (`+0.0206`).
     - **True Query Up-Projection (`D4096→D384`, bridge fits `384→4096`):** **`A2_unscaled` (`0.5859` SciFact, `0.2713` CoSQA)** & **`A2c` (`0.5815`, `0.2611`)** crush both `Centered Affine Ridge` (`0.5416`, `0.2435`) and `A3_full` (`0.5423`, `0.2426`), while `A3_subspace` recovers `+0.0160` (`0.5583`) and `+0.0093` (`0.2519`) over `A3_full`!
   - **3-Hop Monotone Chain (`4096→2560→1024→384`, `n=2000`, `cond=1.00`):**
     - `CoSQA_Code` (20,604 docs): **`A2c` sweeps `3/3` hops vs Centered Affine Ridge** (`0.2223` vs `0.2008`; `0.2120` vs `0.1709`; `0.1911` vs `0.1688`).
     - `SciFact`: `A2c` wins Hop 1 (`0.5358` vs `0.5255`) and Hop 3 (`0.4597` vs `0.4531`), **but LOSES Hop 2 (`2560→1024→384`) to Centered Affine Ridge (`0.4768` vs `0.5023`, `-0.0255`)!** Directly validates Carbon's `F1-NEW-1` (`n=2000 < d=2560` nullspace noise!) + `Carbon XC §2.2` (in-sample down-projection radius collapse)!
   - **Zero-Shot Cross-Domain OOD (`CoSQA_Code ↔ Legal_Contracts / Legal_QuAD / SciFact`, 0 target anchors):**
     - Proves bridges are **manifold-local** (zero-shot degrades sharply vs native: `0.4418` vs `0.6458` on `Code→Legal_Contracts`; `0.1126` vs `0.6484` on `Code→SciFact`), **yet `Centered Affine Ridge` collapses catastrophically off-manifold (`0.0000–0.1704`)**, losing by **3×–27×** to `A2c`/`A3` (`Code→Legal_Contracts D384→D4096`: `A3=0.4418, A2c=0.4091` vs `Centered Affine Ridge=0.1324`; `SciFact→Code`: `A3=0.0817, A2c=0.0793` vs `Centered Affine Ridge=0.0030`), and `A3_subspace` beats `A3_full` on `5/6` OOD up-projections (`D4096→D384`).
   - **Action for R1:** Incorporate these exact `f1_4096_ood` findings into Topic 3 (Flagship #1 architecture & falsification matrix) and Topic 5 (Whitepaper v2 + HTML v2 — now **15 verified receipt bundles**!).

---

## 3. Open Disagreements & Brand-New User Research Directives to Resolve in Refinement Round 1

Each council member MUST deeply analyze, stress-test (with in-memory Python proofs/counterexamples where applicable), and refine ALL FIVE topics below:

### Topic 1 — Resolve the `SUPERSEDE` Policy & Incremental Complexity Tension (`Astra` vs. `Neon` vs. `Carbon`)
Three crisp sub-tensions emerged in Cross-Check:
1. **`C REFUTE+SUPERSEDE B`, `B SUPERSEDE A`:**
   - **Astra's Direct-$B_i$ Normal Program** ($B_i \leftarrow V_j$ for direct `j SUPERSEDE i`): `B` is `REFUTED`, so `B` loses replacement authority and BOTH `A` and `C` remain current (if same single-valued slot, abstain with both witnesses unless explicit `C→A` or slot-lineage exists).
   - **Neon's Transitive `SUPERSEDE+` Rule:** `C --SUP+--> A` keeps `A` `SUPERSEDED` through refuted `B`, while pure `C REFUTE B` reinstates `A`.
2. **Mutual `SUPERSEDE` Cycle ($A \leftrightarrow_{\text{SUP}} B$):**
   - **Astra:** Both remain valid and lose currency (`AUDIT_ONLY`; reject malformed cycles on ordinary writes, quarantine imported cycles).
   - **Neon & Carbon:** Mutual/self `SUPERSEDE` enters `CONTESTED(TEMPORAL)` so readers abstain instead of silently returning empty.
3. **Self-Amending Patches ($e_2 \xrightarrow{\text{DERIVES\_FROM} + \text{SUPERSEDE}} e_1$):**
   - Under Astra's $C_i \leftarrow V_i \wedge \neg B_i \wedge \bigwedge_{j \in D_i} C_j$, if $e_2$ both `DERIVES_FROM` and `SUPERSEDE`s $e_1$, does $C_{e_1} = \mathbf{F}$ kill $C_{e_2}$ (making every incremental patch self-obsoleting!)? How do we cleanly separate `COPY_OF` / `DERIVES_CURRENT` (inherits currency) from `DERIVES_SNAPSHOT` / `AMENDS` (inherits epistemic validity $V$, NOT currency $C$)?
4. **Verify Unification of Astra's Linear-Time Positive-SCC Collapse + Carbon's Fitting-Stratum-Rank Shielding:** Prove (or find a counterexample to!) whether combining Astra's linear-time positive-SCC-collapse local WFS solver with Carbon's `strict_t_shield_cnt` ($\text{rank}(u) < \text{rank}(w)$) handles **both** arbitrary insertions AND arbitrary deletions/rewinds in $O(|D|+|E_{\text{incident}}(D)|)$ with `0` mismatches against an alternating-fixpoint WFS oracle!

### Topic 2 — NEW USER DIRECTIVE: Stratified $K$-Axis Ordinal Stabilization (`K = 2, 4, 6, 8` Axes)
The user explicitly asked: *"what about 4 and 6 and 8 axis as well for ordinal stabilization if that makes sense?"*
Rigorously evaluate and design:
- **Axis Taxonomy & Algebra:**
  - **$K = 2$ (Minimal Core):** Axis 1 Epistemic Validity (`REFUTE` + positive deps, WFS) $\to$ Axis 2 Temporal Currency (`SUPERSEDE` + current-inheriting deps).
  - **$K = 4$ (Production Sweet Spot):** Add Axis 3 **Authority / Trust Stratum** (Preference-Based Argumentation: e.g., primary signed tool receipt > user correction > agent summary > unlinked echo, preventing weak-source epistemic DoS!) and Axis 4 **Bitemporal Visibility Scope (`valid_time × tx_time`)**.
  - **$K = 6$ (Enterprise Scope):** Add Axis 5 **Contextual / Jurisdictional Scope** (tenant/region/entity slot isolation) and Axis 6 **Defeasible Specificity** (specific exception overrides general default).
  - **$K = 8$ (Full Unified Chelated Lattice):** Add Axis 7 **Representation / Cryptographic Integrity (`TRA-1` ABI)** and Axis 8 **Bounded Continuous Ordinal Tie-Breaker** (calibrated retrieval/freshness score strictly inside `ADMIT`).
- **Mathematical Rigor Check:** State and prove the exact **Stratified $K$-Axis Ordinal Stabilization Theorem** (exact conditions — acyclic stratum DAG, lexicographic product order, monotonicity/WFS within each stratum — under which $K \in \{2,4,6,8\}$ guarantees unique fixed-point ordinal stabilization and incremental $O(\|\Delta\|)$ maintenance), AND be brutally honest about **which axes belong in the executable Proof Engine v1 ($K=2$ vs $K=4$ vs $K=8$)** vs. theoretical extension so we don't bloat the core or overclaim! Specifically check: **Where in the stratum order MUST `Authority` sit relative to `REFUTE` WFS and `Bitemporal Visibility`?** (Can a higher-authority node be refuted by a lower-authority node, or does Authority filter the `REFUTE` relation *before* Axis 1 WFS? What happens if an authoritative node is itself refuted by an *equal-or-higher* authority node?)

### Topic 3 — NEW USER DIRECTIVE: "Gauss Polar Manifold" → Riemannian Geodesic Polar Alignment (`A2g`) & Stiefel Polar Re-Projection (`A2c-Polar`)
The user explicitly asked to test a **"Gauss polar manifold"** idea (and approved keeping and testing all low-cost geometric ablations on cached embeddings!). Evaluate cleanly:
1. **`A2g` (Riemannian Geodesic Polar Tangent Alignment via Gauss's Lemma $\operatorname{Log}_{\bar{\mu}_x} \to s_\theta Q \to \operatorname{Exp}_{\bar{\mu}_y}$):**
   - Formulate the exact equations (pole $\bar{\mu} = \mu / \|\mu\|$, geodesic angle $\theta = \arccos(\operatorname{clip}(\langle x, \bar{\mu}\rangle, -1, 1))$, tangent vector $v = \frac{\theta}{\sin\theta}(x - \langle x, \bar{\mu}\rangle \bar{\mu})$, tangent-space orthogonal Procrustes / shrinkage-whitened Gaussian tangent alignment with geodesic radius ratio $s_\theta = \text{RMS}(\theta_y) / \text{RMS}(\theta_x \text{ proj})$, and $\operatorname{Exp}_{\bar{\mu}_y}(w) = \cos(\|w\|)\bar{\mu}_y + \sin(\|w\|)\frac{w}{\|w\|}$).
   - Identify **exact failure modes & guards** (isotropic pole variance when $\|\mu\| \approx 0$, antipodal cut locus $\theta \to \pi$, wrap-around when $\|w\| > \pi$, non-collapsibility across hops vs $0.05\text{ ms}$ sequential relay) and derive when `A2g` should theoretically beat or lose to extrinsic `A2c` (e.g., narrow-cone `E2` $r=0.44$ $\leftrightarrow$ isotropic `E0` $r=0.93$, and `384→4096`!).
2. **`A2c-Polar` (Stiefel Polar-Factor Re-Projection $Q_{\text{comp}} \leftarrow U_+ V_+^\top$ for non-monotone peak chains)** + complete Flagship #1 ablation matrix on cached embeddings (`A2`, `A2c`, `A2c-Subspace`, `A2c-OOS-Radius`, `A2c-Polar`, `A2g`, `A2c-Q`, `A2c-IW`, `A3-Subspace`, `Centered Affine Ridge`, `B4`). Confirm the exact falsification criteria so every variant is either promoted on real OOS/multi-hop wins or logged cleanly in the falsification ledger!

### Topic 4 — Lock Harmonized `PM-1..PM-5` Pre-Release Gates (Zero Ambiguity)
Resolve the exact remaining margin & protocol divergences between Astra, Carbon, and Neon:
- **`PM-1` Clean Non-Inferiority Margin vs. Statistical Power:** Reconcile Astra's `LCB95 > -1.0 pp` (requires $N \ge 400$–$800$ clean items!) vs. Carbon/Neon's `point >= -1.0 pp, LCB95 >= -3.0 pp` at $N=500$ + deterministic prompt-SHA caching + multi-pass/batch-controlled generation.
- **`PM-2` Native Non-Inferiority Margin:** Absolute `-3.0 pp` (`LCB95 >= -5.0 pp`) vs. relative `95% of native`.
- **`PM-3` Upstream Baselines:** Exact handling if `ScaleMAS` has no public repo (`UNAVAILABLE_UPSTREAM_ARCHIVE` + `mem0ai` + `graphiti-core` + `C1_bounded`) and whether `+10 pp` vs upstream is an existing or newly proposed sub-gate.
- **`PM-5` Fallback vs. Zero-Full-Scan Wording:** Reconcile Astra's catch that exact services on dense cascades or deliberate dense fallbacks cannot promise "zero full-graph scans ever" — instead require **zero spurious full-graph scans on sparse/no-op/shielded updates** + explicit measured crossover curve.

### Topic 5 — Lock Single-Source Doctoral Monograph v2 & Presentation HTML v2 Architecture
Specify the exact chapter-by-chapter structure of **Doctoral Whitepaper v2**, the exact figure/panel/accessibility/print/schema specification of **Presentation HTML v2**, and the **single-source Python verification-registry builder** (`scripts/build_whitepaper_assets.py` or equivalent) that reads all verified `report_receipt.json` + CSV artifacts and emits synchronized Markdown tables, full 64-char SHA-256 ledgers (with explicit inclusion/supersession rules across all 14+ bundles), and HTML data/SVG/static-table blocks with **zero manual transcription drift**.

---

## 4. Required Output Format for Refinement Round 1
Structure your report under the 5 headings matching Topics 1–5 above, plus **Section 6: Concrete Actionable Implementation Diff Checklist** (exact file paths, functions, tests, and theorems to lock for Round 2 cross-check). Be maximally rigorous, cite exact lines/math/code probes, and challenge any weak spot remaining in the consensus!
