import Mathlib.LinearAlgebra.Matrix.Trace
import Mathlib.Analysis.InnerProductSpace.PiL2
import Mathlib.Data.Finset.Basic
import Mathlib.Analysis.Real.Sqrt

/-!
# Pre-Registered Formal Specification (`ChelatedFormal.Spec`)

Scope & Irreducible Boundaries (`wave14-r3-final-consensus-master-plan.md` §3):
1. `Theorem1a` & `Theorem1b` certify exact real-arithmetic linear/Stiefel properties (`Wᵀ W = I`).
   Runtime `normalize((x - μ) Q)` applies affine centering and spherical projection outside `W`;
   `Theorem1b` characterizes the in-sample plain SVD branch (`spherical_migration_bridge.py:315-348`).
2. `Theorem2` certifies adjoint query-read duality, exact `ZeroMu` centered-cosine recovery from stored
   centered norms `‖d - μ_d‖₂`, and per-document forward perturbation bounds (including CSLS hubness terms)
   under quantization/rounding.
3. `Theorem3` certifies deductive Stratum 1–2 cyclic alternating-fixpoint Well-Founded Semantics (`Γ_P`
   antimonotone, `Γ_P²` monotone over arbitrary finite cyclic graphs, target-domain authority qualification).
   Full Stratum 0–3 runtime behavior (`ek5_v3_stratified.py`) is additionally verified via `1,000`
   multi-mutation differential conformance sequences against `ek5_ref_cli`.
4. `Theorem4` certifies inductive `ρ`-ball drift bounds for `Anchored-Clipped EMA` (`M6` query-burst defense);
   `PM-6` gates freeze `μ_q = μ_0`, and channel/mTLS authority authentication lies outside the EMA theorem.
-/

namespace ChelatedFormal

open scoped BigOperators Matrix

abbrev Euc (d : ℕ) := EuclideanSpace ℝ (Fin d)

/-- Column-orthonormal rectangular Stiefel matrix `W ∈ ℝ^{d₁ × d₀}` (`Wᵀ W = I_{d₀}`). -/
def IsStiefel {d₁ d₀ : ℕ} (W : Matrix (Fin d₁) (Fin d₀) ℝ) : Prop :=
  Wᵀ * W = 1

/-- Explicit entrywise sum-of-squares Frobenius norm squared `‖M‖_F² = ∑_{i,j} M_{i,j}²`. -/
def frobSq {m n : ℕ} (M : Matrix (Fin m) (Fin n) ℝ) : ℝ :=
  ∑ i : Fin m, ∑ j : Fin n, (M i j) ^ 2

/-- Euclidean vector norm squared on `Fin d → ℝ`. -/
def vecNormSq {d : ℕ} (v : Fin d → ℝ) : ℝ :=
  ∑ i : Fin d, (v i) ^ 2

/-- Standard dot product on `Fin d → ℝ`. -/
def vecDot {d : ℕ} (u v : Fin d → ℝ) : ℝ :=
  ∑ i : Fin d, u i * v i

/-- Theorem 1a: Semi-orthogonal Stiefel embeddings enforce dimension monotonicity `d₀ ≤ d₁` and compose
    conformally across chains `d₀ ≤ d₁ ≤ ⋯` with uniform Euclidean stretch `∏ α_t` (`κ₂ = 1`). -/
def Theorem1a_MonotoneChainIsometry_Stmt : Prop :=
  (∀ {d₀ d₁ : ℕ} (W : Matrix (Fin d₁) (Fin d₀) ℝ), IsStiefel W → d₀ ≤ d₁) ∧
  (∀ {d₀ d₁ d₂ : ℕ} (α₁ α₂ : ℝ)
    (W₁ : Matrix (Fin d₁) (Fin d₀) ℝ) (W₂ : Matrix (Fin d₂) (Fin d₁) ℝ),
    0 < α₁ → 0 < α₂ → IsStiefel W₁ → IsStiefel W₂ →
    IsStiefel (W₂ * W₁) ∧
    ∀ x : Fin d₀ → ℝ,
      vecNormSq (((α₂ * α₁) • (W₂ * W₁)).mulVec x) = (α₂ * α₁) ^ 2 * vecNormSq x)

/-- Theorem 1a Non-Monotone `2 → 3 → 2` Rank-Collapse Counterexample (`κ₂` blows up / unit vector maps to `0`). -/
def Theorem1a_NonMonotoneCollapse_Stmt : Prop :=
  ∃ (W₁ : Matrix (Fin 3) (Fin 2) ℝ) (W₂ : Matrix (Fin 2) (Fin 3) ℝ) (x : Fin 2 → ℝ),
    IsStiefel W₁ ∧ IsStiefel W₂ᵀ ∧ vecNormSq x = 1 ∧ vecNormSq ((W₂ * W₁).mulVec x) = 0

/-- Thin-SVD witness `Aᵀ B = U Σ Vᵀ` (`U ∈ St(d₁, d₀)`, `V ∈ O(d₀)`, `Σ ≥ 0` diagonal). -/
structure ThinSvdWitness {n d₁ d₀ : ℕ}
    (A : Matrix (Fin n) (Fin d₁) ℝ) (B : Matrix (Fin n) (Fin d₀) ℝ) where
  U : Matrix (Fin d₁) (Fin d₀) ℝ
  V : Matrix (Fin d₀) (Fin d₀) ℝ
  σ : Fin d₀ → ℝ
  hU : IsStiefel U
  hV_left : IsStiefel V
  hV_right : IsStiefel Vᵀ
  hσ : ∀ i, 0 ≤ σ i
  hDecomp : Aᵀ * B = U * Matrix.diagonal σ * Vᵀ

/-- Theorem 1b (i): Bidirectional Inner-Product Alignment Optimality of `W* = U Vᵀ` over `St(d₁, d₀)`. -/
def Theorem1b_TraceOptimality_Stmt : Prop :=
  ∀ {n d₁ d₀ : ℕ} (A : Matrix (Fin n) (Fin d₁) ℝ) (B : Matrix (Fin n) (Fin d₀) ℝ)
    (w : ThinSvdWitness A B) (W : Matrix (Fin d₁) (Fin d₀) ℝ),
    IsStiefel W →
    Matrix.trace (Wᵀ * (Aᵀ * B)) ≤ Matrix.trace ((w.U * w.Vᵀ)ᵀ * (Aᵀ * B)) ∧
    Matrix.trace ((w.U * w.Vᵀ)ᵀ * (Aᵀ * B)) = ∑ i : Fin d₀, w.σ i

/-- Theorem 1b (ii): Unconditional Upward-Lifting Frobenius Optimality `‖B (W*)ᵀ - A‖_F² ≤ ‖B Wᵀ - A‖_F²`. -/
def Theorem1b_UpwardLiftingFrobenius_Stmt : Prop :=
  ∀ {n d₁ d₀ : ℕ} (A : Matrix (Fin n) (Fin d₁) ℝ) (B : Matrix (Fin n) (Fin d₀) ℝ)
    (w : ThinSvdWitness A B) (W : Matrix (Fin d₁) (Fin d₀) ℝ),
    IsStiefel W →
    frobSq (B * (w.U * w.Vᵀ)ᵀ - A) ≤ frobSq (B * Wᵀ - A)

/-- Theorem 1b (iii-a): Downward Frobenius Optimality holds under isotropic source Gram `Aᵀ A = c I`. -/
def Theorem1b_IsotropicDownwardFrobenius_Stmt : Prop :=
  ∀ {n d₁ d₀ : ℕ} (A : Matrix (Fin n) (Fin d₁) ℝ) (B : Matrix (Fin n) (Fin d₀) ℝ)
    (c : ℝ) (w : ThinSvdWitness A B) (W : Matrix (Fin d₁) (Fin d₀) ℝ),
    Aᵀ * A = c • (1 : Matrix (Fin d₁) (Fin d₁) ℝ) → IsStiefel W →
    frobSq (A * (w.U * w.Vᵀ) - B) ≤ frobSq (A * W - B)

/-- Theorem 1b (iii-b): Exact Rational `3-4-5` Counterexample falsifying general downward Frobenius
    minimization (`d₁ = 2 → d₀ = 1`, `A = diag(3, 4)`, `B = (1, 1)ᵀ`: `W* = (3/5, 4/5)ᵀ` gives `137/25 > 5`). -/
def Theorem1b_DownwardFrobeniusCounterexample_Stmt : Prop :=
  ∃ (A : Matrix (Fin 2) (Fin 2) ℝ) (B : Matrix (Fin 2) (Fin 1) ℝ)
    (w : ThinSvdWitness A B) (W_alt : Matrix (Fin 2) (Fin 1) ℝ),
    IsStiefel W_alt ∧
    frobSq (A * W_alt - B) = 5 ∧
    frobSq (A * (w.U * w.Vᵀ) - B) = (137 : ℝ) / 25 ∧
    frobSq (A * W_alt - B) < frobSq (A * (w.U * w.Vᵀ) - B)

/-- Theorem 2 (a): Adjoint Query-Read Bilinear Identity `⟨q, W d⟩ = ⟨Wᵀ q, d⟩` + Projector Norm Decomposition. -/
def Theorem2_AdjointAndProjector_Stmt : Prop :=
  ∀ {d₁ d₀ : ℕ} (W : Matrix (Fin d₁) (Fin d₀) ℝ) (q : Fin d₁ → ℝ) (d : Fin d₀ → ℝ),
    IsStiefel W →
    vecDot q (W.mulVec d) = vecDot (Wᵀ.mulVec q) d ∧
    vecNormSq (Wᵀ.mulVec q) + vecNormSq (q - (W * Wᵀ).mulVec q) = vecNormSq q

/-- Theorem 2 (b): Exact `ZeroMu` Sidecar Centered-Cosine Identity & Forward Quantization + CSLS Perturbation Bound. -/
def Theorem2_ZeroMuAndPerturbation_Stmt : Prop :=
  ∀ {d : ℕ} (q_hat d_vec μ_d : Fin d → ℝ)
    (γ_d γ_hat dot_hat rq rq_hat rd rd_hat ε_quant δ_γ ε_rq ε_rd γ_min : ℝ),
    γ_d ≠ 0 →
    (vecDot q_hat d_vec - vecDot q_hat μ_d) / γ_d = vecDot q_hat (fun i => d_vec i - μ_d i) / γ_d ∧
    (0 < δ_γ → δ_γ < γ_min → γ_min ≤ γ_d →
      |dot_hat - vecDot q_hat d_vec| ≤ ε_quant →
      |γ_hat - γ_d| ≤ δ_γ →
      |rq_hat - rq| ≤ ε_rq →
      |rd_hat - rd| ≤ ε_rd →
      0 ≤ ε_quant →
      |((dot_hat - vecDot q_hat μ_d) / γ_hat - (rq_hat + rd_hat) / 2) -
        ((vecDot q_hat d_vec - vecDot q_hat μ_d) / γ_d - (rq + rd) / 2)| ≤
        (ε_quant + δ_γ * |vecDot q_hat d_vec - vecDot q_hat μ_d| / γ_min) / (γ_min - δ_γ) +
          (ε_rq + ε_rd) / 2)

/-- Finite multi-domain authority-qualified cyclic epistemic graph (`Stratum 1 → Stratum 2` WFS). -/
structure CyclicEpistemicGraph (n doms : ℕ) where
  domainOf : Fin n → Fin doms
  effAuthForDomain : Fin n → Fin doms → ℕ
  effAuthSelf : Fin n → ℕ
  rawRefutes : Fin n → Fin n → Bool
  rawDeps : Fin n → Fin n → Bool

/-- Qualified refutation edge `u → w` active iff `rawRefutes u w` and `effAuthForDomain u (domainOf w) ≥ effAuthSelf w`. -/
def activeRefute {n doms : ℕ} (G : CyclicEpistemicGraph n doms) (u w : Fin n) : Prop :=
  G.rawRefutes u w = true ∧ G.effAuthSelf w ≤ G.effAuthForDomain u (G.domainOf w)

/-- Computable boolean predicate for active authority-qualified refutation `u → w`. -/
def isActiveRefute {n doms : ℕ} (G : CyclicEpistemicGraph n doms) (u w : Fin n) : Bool :=
  G.rawRefutes u w && decide (G.effAuthSelf w ≤ G.effAuthForDomain u (G.domainOf w))

/-- Single monotone positive-support closure step under `assumedOut`:
    node `w` stays supported if no active attacker `u` lies outside `assumedOut` AND every
    positive dependency `d` (`G.rawDeps w d = true`) already lies in `supported`. -/
def posSupportStep {n doms : ℕ} (G : CyclicEpistemicGraph n doms)
    (assumedOut : Finset (Fin n)) (supported : Finset (Fin n)) : Finset (Fin n) :=
  Finset.univ.filter (fun w =>
    (∀ u : Fin n, isActiveRefute G u w = true → u ∈ assumedOut) ∧
    (∀ d : Fin n, G.rawDeps w d = true → d ∈ supported))

/-- Monotone positive closure `posClosure G assumedOut` (`n`-step least fixpoint of `posSupportStep` from `∅`). -/
def posClosure {n doms : ℕ} (G : CyclicEpistemicGraph n doms) (assumedOut : Finset (Fin n)) : Finset (Fin n) :=
  Nat.iterate (posSupportStep G assumedOut) n ∅

/-- Van Gelder's **antimonotone** stability operator `Γ_G(I) := posClosure G (univ \ I)`
    (`I` = set of atoms assumed potentially true, so `univ \ I` are assumed out). -/
def gammaOp {n doms : ℕ} (G : CyclicEpistemicGraph n doms) (I : Finset (Fin n)) : Finset (Fin n) :=
  posClosure G (Finset.univ \ I)

/-- Van Gelder's **monotone** two-step alternating operator `Γ_G²(I) = Γ_G(Γ_G(I))`. -/
def gammaSqOp {n doms : ℕ} (G : CyclicEpistemicGraph n doms) (I : Finset (Fin n)) : Finset (Fin n) :=
  gammaOp G (gammaOp G I)

/-- Computable WFS well-founded true set `T(G) = (Γ_G²)^n(∅)` and false set `F(G) = univ \ Γ_G(T(G))`. -/
def wfsTrueSet {n doms : ℕ} (G : CyclicEpistemicGraph n doms) : Finset (Fin n) :=
  Nat.iterate (gammaSqOp G) n ∅

def wfsFalseSet {n doms : ℕ} (G : CyclicEpistemicGraph n doms) : Finset (Fin n) :=
  Finset.univ \ gammaOp G (wfsTrueSet G)

/-- Theorem 3: On arbitrary finite cyclic multi-domain graphs:
    (1) `Γ_G` is **antimonotone**: `I₁ ⊆ I₂ → gammaOp G I₂ ⊆ gammaOp G I₁`,
    (2) `Γ_G²` is **monotone**: `I₁ ⊆ I₂ → gammaSqOp G I₁ ⊆ gammaSqOp G I₂`,
    (3) `wfsTrueSet G` is the **least fixpoint** of `gammaSqOp G` (`gammaSqOp G (wfsTrueSet G) = wfsTrueSet G`
        and `∀ S, gammaSqOp G S = S → wfsTrueSet G ⊆ S`), with `Disjoint (wfsTrueSet G) (wfsFalseSet G)`,
    (4) Cyclic `REFUTE` witness `G_cyc` where `wfsTrueSet G_cyc = ∅ ∧ wfsFalseSet G_cyc = ∅`
        (undecided ring `U = univ`) while distinct non-empty two-valued fixpoints of `gammaOp` exist. -/
def Theorem3_EK5_CyclicWFS_Stmt : Prop :=
  (∀ {n doms : ℕ} (G : CyclicEpistemicGraph n doms) (I₁ I₂ : Finset (Fin n)),
    I₁ ⊆ I₂ → gammaOp G I₂ ⊆ gammaOp G I₁) ∧
  (∀ {n doms : ℕ} (G : CyclicEpistemicGraph n doms) (I₁ I₂ : Finset (Fin n)),
    I₁ ⊆ I₂ → gammaSqOp G I₁ ⊆ gammaSqOp G I₂) ∧
  (∀ {n doms : ℕ} (G : CyclicEpistemicGraph n doms),
    gammaSqOp G (wfsTrueSet G) = wfsTrueSet G ∧
    (∀ S : Finset (Fin n), gammaSqOp G S = S → wfsTrueSet G ⊆ S) ∧
    Disjoint (wfsTrueSet G) (wfsFalseSet G)) ∧
  (∃ (G_cyc : CyclicEpistemicGraph 2 1) (S_alt : Finset (Fin 2)),
    wfsTrueSet G_cyc = ∅ ∧ wfsFalseSet G_cyc = ∅ ∧ S_alt.Nonempty ∧ gammaOp G_cyc S_alt = S_alt)

/-- Theorem 4 (`Anchored-Clipped EMA` Drift Bound): Projecting onto the `ρ`-ball around `μ₀` guarantees
    inductive invariance `‖μ_t - μ₀‖₂ ≤ ρ` for arbitrary adversarial query streams `{q_t}`. -/
noncomputable def radialClip {d : ℕ} (μ₀ x : Euc d) (ρ : ℝ) : Euc d :=
  if ‖x - μ₀‖ ≤ ρ then x else μ₀ + (ρ / ‖x - μ₀‖) • (x - μ₀)

def Theorem4_AnchoredClippedEMA_Stmt : Prop :=
  ∀ {d : ℕ} (μ₀ : Euc d) (ρ α lam : ℝ),
    0 ≤ ρ →
    ∀ (k : ℕ) (q : ℕ → Euc d),
      let μ_seq : ℕ → Euc d := Nat.rec μ₀ (fun t μ_prev =>
        radialClip μ₀ ((1 - α - lam) • μ_prev + α • q t + lam • μ₀) ρ)
      ‖μ_seq k - μ₀‖ ≤ ρ

end ChelatedFormal
