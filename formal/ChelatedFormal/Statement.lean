import ChelatedFormal.Spec
import ChelatedFormal.Theorem1a_MonotoneIsometry
import ChelatedFormal.Theorem1b_ProcrustesOptimality
import ChelatedFormal.Theorem2_AdjointZeroMuPerturbation
import ChelatedFormal.Theorem3_EK5_CyclicWFS
import ChelatedFormal.Theorem4_AnchoredClippedEMA

/-!
# Master Wave-14A Formal Verification Bundle (`ChelatedFormal.Statement`)

Aggregates all five kernel-checked proof modules against the SHA-256-locked specification
`ChelatedFormal.Spec` (`formal/spec_sha256.json`) with **0 `sorry`, 0 `native_decide`, and 0 custom axioms**:

1. `Theorem1a_MonotoneIsometry`: Stiefel dimension monotonicity (`d₀ ≤ d₁`), conformal chain isometry,
   and `2 → 3 → 2` non-monotone rank-collapse counterexample.
2. `Theorem1b_ProcrustesOptimality`: Thin-SVD trace alignment optimality, unconditional upward-lifting
   Frobenius optimality, isotropic downward Frobenius corollary, and exact rational `3-4-5` (`137/25 > 5`)
   downward counterexample (in-sample plain SVD branch).
3. `Theorem2_AdjointZeroMuPerturbation`: Adjoint query-read duality, projector norm identity, unit-query
   `√(1 - η)` specialization, orthogonal-query fixture, and `ZeroMu + CSLS` forward perturbation bound.
4. `Theorem3_EK5_CyclicWFS`: Stratum 1–2 cyclic alternating-fixpoint Well-Founded Semantics (`Γ_P`
   antimonotone, `Γ_P²` monotone, `T = lfp(Γ_P²)`, `T ∩ F = ∅`, mutual `REFUTE` undecided ring witness,
   Stratum 1 → Stratum 2 non-interference, and `evalStratum2Wfs_sound` binding `ek5_ref_cli`).
   Full Stratum 0–3 behavior is conformance-tested across `1,000` multi-mutation sequences (`7,722` checkpoints).
5. `Theorem4_AnchoredClippedEMA`: Inductive `ρ`-ball drift invariance for `Anchored-Clipped EMA` (`M6`)
   plus `2ρ/γ` chordal and `1 - 2(ρ/γ)²` cosine stability bounds.
-/

namespace ChelatedFormal

theorem all_wave14a_formal_theorems_verified :
    Theorem1a_MonotoneChainIsometry_Stmt ∧
    Theorem1a_NonMonotoneCollapse_Stmt ∧
    Theorem1b_TraceOptimality_Stmt ∧
    Theorem1b_UpwardLiftingFrobenius_Stmt ∧
    Theorem1b_IsotropicDownwardFrobenius_Stmt ∧
    Theorem1b_DownwardFrobeniusCounterexample_Stmt ∧
    Theorem2_AdjointAndProjector_Stmt ∧
    Theorem2_ZeroMuAndPerturbation_Stmt ∧
    Theorem3_EK5_CyclicWFS_Stmt ∧
    Theorem4_AnchoredClippedEMA_Stmt :=
  ⟨theorem1a_monotone_chain_isometry,
   theorem1a_non_monotone_collapse,
   theorem1b_trace_optimality,
   theorem1b_upward_lifting_frobenius,
   theorem1b_isotropic_downward_frobenius,
   theorem1b_downward_frobenius_counterexample,
   theorem2_adjoint_and_projector,
   theorem2_zeromu_and_perturbation,
   theorem3_ek5_cyclic_wfs,
   theorem4_anchored_clipped_ema⟩

end ChelatedFormal
