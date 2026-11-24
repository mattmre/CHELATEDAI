import ChelatedFormal.Spec
import ChelatedFormal.Theorem1a_MonotoneIsometry

namespace ChelatedFormal

open scoped BigOperators Matrix

private theorem vecDot_comm {n : ℕ} (u v : Fin n → ℝ) : vecDot u v = vecDot v u := by
  simp [vecDot, mul_comm]

private theorem vecNormSq_sub {n : ℕ} (u v : Fin n → ℝ) :
    vecNormSq (u - v) = vecNormSq u + vecNormSq v - 2 * vecDot u v := by
  unfold vecNormSq vecDot
  rw [Finset.mul_sum, ← Finset.sum_add_distrib, ← Finset.sum_sub_distrib]
  exact Finset.sum_congr rfl (fun i _ => by change (u i - v i) ^ 2 = _; ring)

theorem theorem2_adjoint_and_projector : Theorem2_AdjointAndProjector_Stmt := by
  intro d₁ d₀ W q d hW
  refine ⟨vecDot_mulVec_adjoint W q d, ?_⟩
  rw [← Matrix.mulVec_mulVec, vecNormSq_sub, hW.vecNormSq_mulVec,
    vecDot_mulVec_adjoint, vecDot_self_eq_vecNormSq]
  ring

theorem theorem2_unit_query_projected_norm {d₁ d₀ : ℕ} (W : Matrix (Fin d₁) (Fin d₀) ℝ)
    (q : Fin d₁ → ℝ) (hW : IsStiefel W) (hq : vecNormSq q = 1) :
    ‖(WithLp.toLp 2 (Wᵀ.mulVec q) : Euc d₀)‖ =
      √(1 - vecNormSq (q - (W * Wᵀ).mulVec q)) := by
  have hSq := (theorem2_adjoint_and_projector W q (0 : Fin d₀ → ℝ) hW).2
  rw [hq] at hSq
  rw [← eq_sub_of_add_eq hSq, euc_norm_eq_sqrt_vecNormSq]

/-- An orthogonal unit query has zero projected norm, so projected normalization requires
a nonzero projected vector. -/
theorem theorem2_orthogonal_query_fixture :
    ∃ (W : Matrix (Fin 2) (Fin 1) ℝ) (q : Fin 2 → ℝ),
      IsStiefel W ∧ vecNormSq q = 1 ∧ Wᵀ.mulVec q = 0 ∧
        vecNormSq (q - (W * Wᵀ).mulVec q) = 1 := by
  let W : Matrix (Fin 2) (Fin 1) ℝ := !![1; 0]
  let q : Fin 2 → ℝ := ![0, 1]
  have hW : IsStiefel W := by
    unfold IsStiefel
    ext i j
    fin_cases i
    fin_cases j
    norm_num [W, Matrix.mul_apply, Matrix.one_apply, Fin.sum_univ_succ]
  have hq : vecNormSq q = 1 := by norm_num [q, vecNormSq, Fin.sum_univ_succ]
  have hOrth : Wᵀ.mulVec q = 0 := by
    funext i
    fin_cases i
    norm_num [W, q, Matrix.mulVec, dotProduct, Fin.sum_univ_succ]
  refine ⟨W, q, hW, hq, hOrth, ?_⟩
  have hProjector := (theorem2_adjoint_and_projector W q (0 : Fin 1 → ℝ) hW).2
  rw [hOrth, hq] at hProjector
  simpa only [show vecNormSq (0 : Fin 1 → ℝ) = 0 by simp [vecNormSq], zero_add] using hProjector

private theorem abs_ratio_sub_le (a b γ γhat ε δ γmin : ℝ)
    (hδ : 0 < δ) (hmin : δ < γmin) (hγ : γmin ≤ γ)
    (hab : |b - a| ≤ ε) (hγhat : |γhat - γ| ≤ δ) :
    |b / γhat - a / γ| ≤ (ε + δ * |a| / γmin) / (γmin - δ) := by
  have hMinPos := hδ.trans hmin
  have hγPos := hMinPos.trans_le hγ
  have hLower : γmin - δ ≤ γhat := by
    have hError := (abs_le.mp hγhat).1
    linarith
  have hLowerPos := sub_pos.mpr hmin
  have hHatPos := hLowerPos.trans_le hLower
  have hDecomp : b / γhat - a / γ = ((b - a) - (γhat - γ) * (a / γ)) / γhat := by
    field_simp [hγPos.ne', hHatPos.ne']
    ring
  have hTerm : |γhat - γ| * (|a| / γ) ≤ δ * (|a| / γmin) := by
    exact mul_le_mul hγhat
      (div_le_div_of_nonneg_left (abs_nonneg _) hMinPos hγ)
      (div_nonneg (abs_nonneg _) hγPos.le) hδ.le
  have hNum : |(b - a) - (γhat - γ) * (a / γ)| ≤ ε + δ * |a| / γmin := calc
    _ ≤ |b - a| + |(γhat - γ) * (a / γ)| := by
      simpa only [sub_eq_add_neg, abs_neg] using abs_add_le (b - a) (-((γhat - γ) * (a / γ)))
    _ ≤ _ := by
      rw [abs_mul, abs_div, abs_of_pos hγPos, mul_div_assoc]
      exact add_le_add hab hTerm
  have hNumNonneg := (abs_nonneg _).trans hNum
  rw [hDecomp, abs_div, abs_of_pos hHatPos]
  exact (div_le_div_of_nonneg_right hNum hHatPos.le).trans
    (div_le_div_of_nonneg_left hNumNonneg hLowerPos hLower)

theorem theorem2_zeromu_and_perturbation : Theorem2_ZeroMuAndPerturbation_Stmt := by
  intro d qhat doc μd γd γhat dotHat rq rqHat rd rdHat εquant δ εrq εrd γmin _
  constructor
  · simp only [vecDot, ← Finset.sum_sub_distrib, mul_sub]
  · intro hδ hmin hγ hdot hγhat hrq hrd _
    let a := vecDot qhat doc - vecDot qhat μd
    let b := dotHat - vecDot qhat μd
    have hab : |b - a| ≤ εquant := by
      dsimp [a, b]
      rw [show dotHat - vecDot qhat μd - (vecDot qhat doc - vecDot qhat μd) =
        dotHat - vecDot qhat doc by ring]
      exact hdot
    have hRatio := abs_ratio_sub_le a b γd γhat εquant δ γmin hδ hmin hγ hab hγhat
    have hHub : |(rqHat - rq + (rdHat - rd)) / 2| ≤ (εrq + εrd) / 2 := by
      rw [abs_div, abs_of_nonneg (by norm_num : (0 : ℝ) ≤ 2)]
      exact div_le_div_of_nonneg_right
        ((abs_add_le _ _).trans (add_le_add hrq hrd)) (by norm_num)
    change |(b / γhat - (rqHat + rdHat) / 2) - (a / γd - (rq + rd) / 2)| ≤ _
    rw [show (b / γhat - (rqHat + rdHat) / 2) - (a / γd - (rq + rd) / 2) =
      (b / γhat - a / γd) - (rqHat - rq + (rdHat - rd)) / 2 by ring]
    refine le_trans ?_ (add_le_add hRatio hHub)
    simpa only [sub_eq_add_neg, abs_neg] using
      abs_add_le (b / γhat - a / γd) (-((rqHat - rq + (rdHat - rd)) / 2))

/-- A unit query with nonzero document centering, norm error, dot-product error, and
both CSLS errors satisfies the hypotheses and yields an exact centered cosine of one. -/
theorem theorem2_witness_nonvacuous :
    ∃ (q doc μ : Fin 1 → ℝ) (γ γhat dotHat rqHat rdHat : ℝ),
      vecNormSq q = 1 ∧ γ ^ 2 = vecNormSq (doc - μ) ∧
      (0 : ℝ) < 1 / 4 ∧ (1 : ℝ) / 4 < 2 ∧ 2 ≤ γ ∧
      |dotHat - vecDot q doc| = 1 / 2 ∧ |γhat - γ| = 1 / 4 ∧
      |rqHat - 0| = 1 / 4 ∧ |rdHat - 0| = 1 / 2 ∧
      (vecDot q doc - vecDot q μ) / γ = 1 ∧
      |((dotHat - vecDot q μ) / γhat - (rqHat + rdHat) / 2) - 1| ≤ (45 : ℝ) / 56 := by
  refine ⟨![1], ![3], ![1], 2, 9 / 4, 7 / 2, 1 / 4, 1 / 2, ?_⟩
  norm_num [vecDot, vecNormSq, Fin.sum_univ_succ, Pi.sub_apply, abs_of_nonneg, abs_of_nonpos]

end ChelatedFormal
