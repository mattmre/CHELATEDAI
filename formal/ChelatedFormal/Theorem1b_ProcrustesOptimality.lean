import ChelatedFormal.Spec
import ChelatedFormal.Theorem1a_MonotoneIsometry

/-!
The Frobenius results concern the in-sample plain SVD branch with the specified
anchor matrices and thin-SVD witness.
-/

namespace ChelatedFormal

open scoped BigOperators Matrix

private theorem stiefel_column_norm {m n : ℕ}
    {W : Matrix (Fin m) (Fin n) ℝ} (hW : IsStiefel W) (i : Fin n) :
    ∑ k : Fin m, W k i ^ 2 = 1 := by
  have hEntry := congrArg (fun M : Matrix (Fin n) (Fin n) ℝ => M i i) hW
  simpa [Matrix.mul_apply, dotProduct, pow_two] using hEntry

private theorem stiefel_cross_diag_le_one {m n : ℕ}
    {X Y : Matrix (Fin m) (Fin n) ℝ} (hX : IsStiefel X) (hY : IsStiefel Y) (i : Fin n) :
    (Xᵀ * Y) i i ≤ 1 := by
  have hSum : (∑ k : Fin m, 2 * (X k i * Y k i)) ≤
      ∑ k : Fin m, (X k i ^ 2 + Y k i ^ 2) := by
    exact Finset.sum_le_sum (fun k _ => by nlinarith [sq_nonneg (X k i - Y k i)])
  rw [← Finset.mul_sum, Finset.sum_add_distrib, stiefel_column_norm hX i,
    stiefel_column_norm hY i] at hSum
  simp only [Matrix.mul_apply, Matrix.transpose_apply]
  linarith

private theorem procrustes_trace_sum {n d₁ d₀ : ℕ}
    {A : Matrix (Fin n) (Fin d₁) ℝ} {B : Matrix (Fin n) (Fin d₀) ℝ}
    (w : ThinSvdWitness A B) (W : Matrix (Fin d₁) (Fin d₀) ℝ) :
    Matrix.trace (Wᵀ * (Aᵀ * B)) = ∑ i : Fin d₀, ((W * w.V)ᵀ * w.U) i i * w.σ i := by
  calc
    _ = Matrix.trace ((Wᵀ * (w.U * Matrix.diagonal w.σ)) * w.Vᵀ) := by
      rw [w.hDecomp]
      simp only [Matrix.mul_assoc]
    _ = Matrix.trace (((W * w.V)ᵀ * w.U) * Matrix.diagonal w.σ) := by
      rw [Matrix.trace_mul_comm]
      simp only [Matrix.transpose_mul, Matrix.mul_assoc]
    _ = _ := by simp [Matrix.trace, Matrix.mul_diagonal]

theorem theorem1b_trace_optimality : Theorem1b_TraceOptimality_Stmt := by
  intro n d₁ d₀ A B w W hW
  have hCandidate : w.U * w.Vᵀ * w.V = w.U := by
    rw [Matrix.mul_assoc, w.hV_left, Matrix.mul_one]
  have hValue : Matrix.trace ((w.U * w.Vᵀ)ᵀ * (Aᵀ * B)) = ∑ i : Fin d₀, w.σ i := by
    rw [procrustes_trace_sum, hCandidate, w.hU]
    simp
  refine ⟨?_, hValue⟩
  rw [hValue, procrustes_trace_sum]
  apply Finset.sum_le_sum
  intro i _
  simpa only [one_mul] using
    mul_le_mul_of_nonneg_right
      (stiefel_cross_diag_le_one (hW.mul w.hV_left) w.hU i) (w.hσ i)

private theorem trace_transpose_mul_entries {m n : ℕ} (X Y : Matrix (Fin m) (Fin n) ℝ) :
    Matrix.trace (Xᵀ * Y) = ∑ i : Fin m, ∑ j : Fin n, X i j * Y i j := by
  simp only [Matrix.trace, Matrix.diag_apply, Matrix.mul_apply, Matrix.transpose_apply]
  exact Finset.sum_comm

private theorem frobSq_eq_trace {m n : ℕ} (X : Matrix (Fin m) (Fin n) ℝ) :
    frobSq X = Matrix.trace (Xᵀ * X) := by
  simp [frobSq, trace_transpose_mul_entries, pow_two]

private theorem frobSq_sub {m n : ℕ} (X Y : Matrix (Fin m) (Fin n) ℝ) :
    frobSq (X - Y) = frobSq X + frobSq Y - 2 * Matrix.trace (Xᵀ * Y) := by
  rw [trace_transpose_mul_entries]
  simp only [frobSq, Matrix.sub_apply]
  rw [Finset.mul_sum, ← Finset.sum_add_distrib, ← Finset.sum_sub_distrib]
  apply Finset.sum_congr rfl
  intro i _
  rw [Finset.mul_sum, ← Finset.sum_add_distrib, ← Finset.sum_sub_distrib]
  exact Finset.sum_congr rfl (fun j _ => by ring)

private theorem upward_expansion {n d₁ d₀ : ℕ}
    (A : Matrix (Fin n) (Fin d₁) ℝ) (B : Matrix (Fin n) (Fin d₀) ℝ)
    (W : Matrix (Fin d₁) (Fin d₀) ℝ) (hW : IsStiefel W) :
    frobSq (B * Wᵀ - A) = frobSq B + frobSq A - 2 * Matrix.trace (Wᵀ * (Aᵀ * B)) := by
  have hNorm : frobSq (B * Wᵀ) = frobSq B := by
    rw [frobSq_eq_trace (B * Wᵀ), frobSq_eq_trace B, Matrix.transpose_mul, Matrix.transpose_transpose]
    calc
      _ = Matrix.trace (W * (Bᵀ * B * Wᵀ)) := by simp only [Matrix.mul_assoc]
      _ = Matrix.trace ((Bᵀ * B) * (Wᵀ * W)) := by
        rw [Matrix.trace_mul_comm]
        simp only [Matrix.mul_assoc]
      _ = _ := by rw [hW, Matrix.mul_one]
  have hCross : Matrix.trace ((B * Wᵀ)ᵀ * A) = Matrix.trace (Wᵀ * (Aᵀ * B)) := by
    rw [← Matrix.trace_transpose ((B * Wᵀ)ᵀ * A), Matrix.transpose_mul,
      Matrix.transpose_transpose, Matrix.trace_mul_cycle']
  rw [frobSq_sub, hNorm, hCross]

theorem theorem1b_upward_lifting_frobenius : Theorem1b_UpwardLiftingFrobenius_Stmt := by
  intro n d₁ d₀ A B w W hW
  have hTrace := (theorem1b_trace_optimality A B w W hW).1
  rw [upward_expansion A B _ (w.hU.mul w.hV_right), upward_expansion A B W hW]
  linarith

private theorem isotropic_downward_expansion {n d₁ d₀ : ℕ}
    (A : Matrix (Fin n) (Fin d₁) ℝ) (B : Matrix (Fin n) (Fin d₀) ℝ)
    (W : Matrix (Fin d₁) (Fin d₀) ℝ) (c : ℝ)
    (hGram : Aᵀ * A = c • (1 : Matrix (Fin d₁) (Fin d₁) ℝ)) (hW : IsStiefel W) :
    frobSq (A * W - B) = c * d₀ + frobSq B - 2 * Matrix.trace (Wᵀ * (Aᵀ * B)) := by
  have hNorm : frobSq (A * W) = c * d₀ := by
    rw [frobSq_eq_trace, Matrix.transpose_mul]
    calc
      _ = Matrix.trace (Wᵀ * ((Aᵀ * A) * W)) := by simp only [Matrix.mul_assoc]
      _ = c * d₀ := by
        rw [hGram, Matrix.smul_mul, Matrix.one_mul, Matrix.mul_smul, hW, Matrix.trace_smul]
        simp
  rw [frobSq_sub, hNorm, Matrix.transpose_mul, Matrix.mul_assoc]

theorem theorem1b_isotropic_downward_frobenius : Theorem1b_IsotropicDownwardFrobenius_Stmt := by
  intro n d₁ d₀ A B c w W hGram hW
  have hTrace := (theorem1b_trace_optimality A B w W hW).1
  rw [isotropic_downward_expansion A B _ c hGram (w.hU.mul w.hV_right),
    isotropic_downward_expansion A B W c hGram hW]
  linarith

theorem theorem1b_downward_frobenius_counterexample : Theorem1b_DownwardFrobeniusCounterexample_Stmt := by
  let A : Matrix (Fin 2) (Fin 2) ℝ := Matrix.diagonal ![3, 4]
  let B : Matrix (Fin 2) (Fin 1) ℝ := !![1; 1]
  let U : Matrix (Fin 2) (Fin 1) ℝ := !![3 / 5; 4 / 5]
  let W : Matrix (Fin 2) (Fin 1) ℝ := !![1; 0]
  have hU : IsStiefel U := by
    unfold IsStiefel
    ext i j
    fin_cases i
    fin_cases j
    norm_num [U, Matrix.mul_apply, Matrix.one_apply, Fin.sum_univ_succ]
  let w : ThinSvdWitness A B := {
    U := U
    V := 1
    σ := ![5]
    hU := hU
    hV_left := by simp [IsStiefel]
    hV_right := by simp [IsStiefel]
    hσ := by intro i; fin_cases i; norm_num
    hDecomp := by
      ext i j
      fin_cases i <;> fin_cases j <;>
        norm_num [A, B, U, Matrix.mul_apply, Matrix.vecMul, dotProduct, Matrix.diagonal_apply,
          Fin.sum_univ_succ]
  }
  have hW : IsStiefel W := by
    unfold IsStiefel
    ext i j
    fin_cases i
    fin_cases j
    norm_num [W, Matrix.mul_apply, Matrix.one_apply, Fin.sum_univ_succ]
  have hAlt : frobSq (A * W - B) = 5 := by
    norm_num [frobSq, A, B, W, Matrix.mul_apply, Matrix.diagonal_apply, Fin.sum_univ_succ]
  have hOpt : frobSq (A * (w.U * w.Vᵀ) - B) = (137 : ℝ) / 25 := by
    norm_num [frobSq, w, A, B, U, Matrix.mul_apply, Matrix.diagonal_apply, Fin.sum_univ_succ]
  exact ⟨A, B, w, W, hW, hAlt, hOpt, by rw [hAlt, hOpt]; norm_num⟩

theorem theorem1b_witness_nonvacuous :
    ∃ (A : Matrix (Fin 2) (Fin 2) ℝ) (B : Matrix (Fin 2) (Fin 1) ℝ)
      (w : ThinSvdWitness A B) (W : Matrix (Fin 2) (Fin 1) ℝ),
      Aᵀ * A = 1 ∧ IsStiefel W ∧ w.σ 0 = 1 ∧
      Matrix.trace (Wᵀ * (Aᵀ * B)) < Matrix.trace ((w.U * w.Vᵀ)ᵀ * (Aᵀ * B)) := by
  let B : Matrix (Fin 2) (Fin 1) ℝ := !![1; 0]
  let W : Matrix (Fin 2) (Fin 1) ℝ := !![0; 1]
  have hB : IsStiefel B := by
    unfold IsStiefel
    ext i j
    fin_cases i
    fin_cases j
    norm_num [B, Matrix.mul_apply, Matrix.one_apply, Fin.sum_univ_succ]
  have hW : IsStiefel W := by
    unfold IsStiefel
    ext i j
    fin_cases i
    fin_cases j
    norm_num [W, Matrix.mul_apply, Matrix.one_apply, Fin.sum_univ_succ]
  let w : ThinSvdWitness (1 : Matrix (Fin 2) (Fin 2) ℝ) B := {
    U := B
    V := 1
    σ := fun _ => 1
    hU := hB
    hV_left := by simp [IsStiefel]
    hV_right := by simp [IsStiefel]
    hσ := fun _ => zero_le_one
    hDecomp := by simp [Matrix.diagonal_one]
  }
  refine ⟨1, B, w, W, by simp, hW, rfl, ?_⟩
  rw [(theorem1b_trace_optimality _ _ w W hW).2]
  norm_num [w, Matrix.trace, Matrix.mul_apply, W, B, Fin.sum_univ_succ]

end ChelatedFormal
