import ChelatedFormal.Spec
import Mathlib.LinearAlgebra.Matrix.Rank

/-!
The conformal estimate is for the linear Stiefel chain. Runtime
`normalize((x - μ) Q)` also applies affine centering and spherical projection.
-/

namespace ChelatedFormal

open scoped BigOperators Matrix

theorem vecDot_mulVec_adjoint {m n : ℕ} (W : Matrix (Fin m) (Fin n) ℝ)
    (q : Fin m → ℝ) (x : Fin n → ℝ) :
    vecDot q (W.mulVec x) = vecDot (Wᵀ.mulVec q) x := by
  change dotProduct q (W.mulVec x) = dotProduct (Wᵀ.mulVec q) x
  rw [Matrix.dotProduct_mulVec, ← Matrix.vecMul_transpose, Matrix.transpose_transpose]

theorem vecDot_self_eq_vecNormSq {n : ℕ} (x : Fin n → ℝ) :
    vecDot x x = vecNormSq x := by
  simp [vecDot, vecNormSq, pow_two]

theorem IsStiefel.transpose_mulVec_leftInverse {m n : ℕ}
    {W : Matrix (Fin m) (Fin n) ℝ} (hW : IsStiefel W) (x : Fin n → ℝ) :
    Wᵀ.mulVec (W.mulVec x) = x := by
  rw [Matrix.mulVec_mulVec, hW, Matrix.one_mulVec]

theorem IsStiefel.vecNormSq_mulVec {m n : ℕ}
    {W : Matrix (Fin m) (Fin n) ℝ} (hW : IsStiefel W) (x : Fin n → ℝ) :
    vecNormSq (W.mulVec x) = vecNormSq x := by
  rw [← vecDot_self_eq_vecNormSq, vecDot_mulVec_adjoint,
    hW.transpose_mulVec_leftInverse, vecDot_self_eq_vecNormSq]

theorem IsStiefel.mul {m n p : ℕ} {U : Matrix (Fin m) (Fin n) ℝ}
    {V : Matrix (Fin n) (Fin p) ℝ} (hU : IsStiefel U) (hV : IsStiefel V) :
    IsStiefel (U * V) := by
  unfold IsStiefel at *
  rw [Matrix.transpose_mul, Matrix.mul_assoc, ← Matrix.mul_assoc Uᵀ U V, hU,
    Matrix.one_mul, hV]

theorem IsStiefel.inputDim_le_outputDim {m n : ℕ}
    {W : Matrix (Fin m) (Fin n) ℝ} (hW : IsStiefel W) : n ≤ m := by
  have hRank := Matrix.rank_mul_le_right Wᵀ W
  rw [hW, Matrix.rank_one, Fintype.card_fin] at hRank
  exact hRank.trans W.rank_le_height

theorem IsStiefel.vecNormSq_scaled_mulVec {m n : ℕ}
    {W : Matrix (Fin m) (Fin n) ℝ} (hW : IsStiefel W) (α : ℝ) (x : Fin n → ℝ) :
    vecNormSq ((α • W).mulVec x) = α ^ 2 * vecNormSq x := by
  rw [Matrix.smul_mulVec, ← hW.vecNormSq_mulVec x]
  simp [vecNormSq, mul_pow, Finset.mul_sum]

theorem theorem1a_monotone_chain_isometry : Theorem1a_MonotoneChainIsometry_Stmt := by
  constructor
  · intro _ _ W hW
    exact hW.inputDim_le_outputDim
  · intro _ _ _ α₁ α₂ W₁ W₂ _ _ hW₁ hW₂
    exact ⟨hW₂.mul hW₁, (hW₂.mul hW₁).vecNormSq_scaled_mulVec (α₂ * α₁)⟩

/-- A chain indexed by its input and output dimensions, including the empty chain. -/
inductive ConformalChain : ℕ → ℕ → Type
  | nil (d : ℕ) : ConformalChain d d
  | snoc {m n p : ℕ} (chain : ConformalChain m n) (α : ℝ) (hα : 0 < α)
      (W : Matrix (Fin p) (Fin n) ℝ) (hW : IsStiefel W) : ConformalChain m p

namespace ConformalChain

noncomputable def matrix {m n : ℕ} : ConformalChain m n → Matrix (Fin n) (Fin m) ℝ
  | .nil _ => 1
  | .snoc chain _ _ W _ => W * chain.matrix

noncomputable def scale {m n : ℕ} : ConformalChain m n → ℝ
  | .nil _ => 1
  | .snoc chain α _ _ _ => α * chain.scale

theorem isStiefel {m n : ℕ} (chain : ConformalChain m n) : IsStiefel chain.matrix := by
  induction chain with
  | nil => simp [matrix, IsStiefel]
  | snoc _ _ _ _ hW ih => exact hW.mul ih

theorem scale_pos {m n : ℕ} (chain : ConformalChain m n) : 0 < chain.scale := by
  induction chain with
  | nil => exact zero_lt_one
  | snoc _ _ hα _ _ ih => exact mul_pos hα ih

end ConformalChain

theorem theorem1a_conformal_chain {m n : ℕ} (chain : ConformalChain m n) :
    m ≤ n ∧ ∀ x : Fin m → ℝ,
      vecNormSq ((chain.scale • chain.matrix).mulVec x) = chain.scale ^ 2 * vecNormSq x :=
  ⟨chain.isStiefel.inputDim_le_outputDim, chain.isStiefel.vecNormSq_scaled_mulVec chain.scale⟩

theorem euc_norm_eq_sqrt_vecNormSq {n : ℕ} (x : Fin n → ℝ) :
    ‖(WithLp.toLp 2 x : Euc n)‖ = √(vecNormSq x) := by
  rw [← Real.sqrt_sq (norm_nonneg (WithLp.toLp 2 x)), EuclideanSpace.real_norm_sq_eq]
  rfl

/-- Every unit input has the same positive stretch, giving condition number one
when the input dimension is positive. -/
theorem ConformalChain.norm_mulVec {m n : ℕ} (chain : ConformalChain m n)
    (x : Fin m → ℝ) :
    ‖(WithLp.toLp 2 ((chain.scale • chain.matrix).mulVec x) : Euc n)‖ =
      chain.scale * ‖(WithLp.toLp 2 x : Euc m)‖ := by
  rw [euc_norm_eq_sqrt_vecNormSq, euc_norm_eq_sqrt_vecNormSq,
    (theorem1a_conformal_chain chain).2 x, Real.sqrt_mul (sq_nonneg _),
    Real.sqrt_sq (le_of_lt chain.scale_pos)]

theorem theorem1a_non_monotone_collapse : Theorem1a_NonMonotoneCollapse_Stmt := by
  refine ⟨!![1, 0; 0, 0; 0, 1], !![1, 0, 0; 0, 1, 0], ![0, 1], ?_, ?_, ?_, ?_⟩
  · unfold IsStiefel
    ext i j
    fin_cases i <;> fin_cases j <;>
      norm_num [Matrix.mul_apply, Matrix.one_apply, Fin.sum_univ_succ]
  · unfold IsStiefel
    ext i j
    fin_cases i <;> fin_cases j <;>
      norm_num [Matrix.mul_apply, Matrix.one_apply, Fin.sum_univ_succ]
  · norm_num [vecNormSq, Fin.sum_univ_succ]
  · norm_num [vecNormSq, Matrix.mulVec, dotProduct, Matrix.mul_apply, Fin.sum_univ_succ]

theorem theorem1a_witness_nonvacuous :
    ∃ (α₁ α₂ : ℝ) (W₁ W₂ : Matrix (Fin 1) (Fin 1) ℝ) (x : Fin 1 → ℝ),
      0 < α₁ ∧ 0 < α₂ ∧ IsStiefel W₁ ∧ IsStiefel W₂ ∧ vecNormSq x = 1 ∧
        vecNormSq (((α₂ * α₁) • (W₂ * W₁)).mulVec x) = 36 := by
  have hI : IsStiefel (1 : Matrix (Fin 1) (Fin 1) ℝ) := by simp [IsStiefel]
  have hx : vecNormSq (fun _ : Fin 1 => 1) = 1 := by simp [vecNormSq]
  refine ⟨2, 3, 1, 1, fun _ => 1, by norm_num, by norm_num, hI, hI, hx, ?_⟩
  rw [(hI.mul hI).vecNormSq_scaled_mulVec, hx]
  norm_num

end ChelatedFormal
