# Adversarial review (objA-beats-gap)

Read the result artifact at `research/drift_recovery/out/estimator/objA_validation.md` (relative to the agent-build worktree root) and any
sibling JSON/report it references. Try to DISPROVE its verdict on this specific angle:

Does oracle_margin_mean actually beat the oracle-gap-only null under block-LOO once FiQA is added, or did the n=4 +1.0 collapse toward gap-level? Verify the block-LOO is not circular and that the verdict label (POSITIVE vs UNDERPOWERED/NEGATIVE) matches the numbers. Recompute the block-LOO Spearman/MAE for margin vs gap-only yourself from the artifacts.

Recompute from the frozen artifacts where possible. Return: PASS / PASS-WITH-FIXES / FAIL, the
specific defect(s) with file:line, and a one-sentence bottom line on whether the verdict is honest.
