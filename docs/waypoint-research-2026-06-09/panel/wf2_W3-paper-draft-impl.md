# Draft the paper-updates section (chair will apply; do NOT touch the paper itself)

Write a chair-ready draft to `D:\GITHUB\CHELATEDAI\docs\waypoint-research-2026-06-09\panel\paper-updates-2026-07-10-draft.md`. Ground EVERY number in the
frozen artifacts — read these first:
- research/drift_recovery/out/estimator/objA_validation.md (+ objA_REPORT.md, objA_validation.json)
- research/drift_recovery/out/d2b_preflight/preflight_report.md
- research/drift_recovery/out/d2/D2_REPORT.md
The draft must contain three blocks of proposed paper text (markdown, ready to paste):
1. A new subsection for §7 ("A margin-based recoverability estimator — promising but underpowered"):
   preregistered univariate oracle_margin_mean; block-LOO Spearman 0.886 (dataset) / 0.829 (encoder) vs
   gap-only null; partial Spearman 0.79 controlling for oracle gap; the FiQA-only-holdout tie disclosed;
   only 3 dataset-blocks -> NOT validated; what a powered test requires.
2. A §7 paragraph closing the "home-turf" question: the sparse-local non-affine CPU preflight (48 cells,
   144 seed-runs, two warp families) admitted no cell at residual>=0.05 after a fair CV-lambda gated local
   ridge (max 0.033). MUST use the corrected framing: residuals are small because absolute gaps are small
   and local ridge often loses to floor; this does NOT show local ridge dominates chelation; it shows no
   discriminating home-turf residual was constructible, so the bounded/annealed corrector gets no second arena.
3. One-to-two-sentence proposed additions to §8 Limitations (underpowered estimator; synthetic-only preflight).
HARD BANS (from the C1 chair rules): no "ceiling" claims, no "irreducible", no "ridge~=MLP proves linearity",
no upgrading PROMISING-BUT-UNDERPOWERED to validated. Label each block with WHERE it goes in main.md.
