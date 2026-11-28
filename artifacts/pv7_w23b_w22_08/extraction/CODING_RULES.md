# Shared coding rules for W22-08 extraction (all families)

1. COUNT a statement as a defect only if the auditor marks it as wrong/missing/needing change:
   it demands an edit/disclosure, or uses defect language (defect, omits, missing, misattributed, fails,
   contradicts, unsupported, misprint, stale, must, should, not X). A neutral restatement of a number,
   even an unflattering one, with no demand or defect framing, is NOT counted. Log borderline exclusions.
2. "Keep X SCOPED/OPEN" alone confirms an existing status; it is not a defect.
3. Defects in derived deliverables (thesis, deck, workbench, whitepaper, registry, brief) that misstate a
   specific bundle's evidence map to THAT bundle's ordinal. Deliverable-only content that names no
   bundle (TCO coefficients, margins, UI defaults, untracked code across all bundles) maps to "XC".
4. A misprint in the round brief of a bundle's digest or number is a defect of that bundle (C07 for
   digests/provenance, C10 for numbers).
5. Class tie-break order for the same underlying defect (use the SAME class across auditors):
   - failed/missed gate, adverse or null result, or loss that is undisclosed -> C05
   - headline/label/ratio/denominator overstates, or omits a caveat that would shrink the claim -> C03
   - flag, metric name, threshold or prose disagrees with code/artifact -> C10
   - numbers presented as measured that are assumptions/unsourced -> C13
   - multiplicity/family/CI method problems -> C14
   - unpinned image, untracked code, missing hash/commit -> C07
   - unfair/budget-unmatched/format-confounded comparator -> C08
6. One capture per (list, canonical ID). Several statements by one auditor mapping to the same ID are
   recorded as separate records but collapse to one capture; that is fine.
