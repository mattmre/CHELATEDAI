export const meta = {
  name: 'gap-close-swarm',
  description: 'Grok swarm: implement Obj B preflight test debt + paper-updates draft, each adversarially cross-reviewed',
  phases: [
    { title: 'Implement', detail: 'Grok implements each gap item' },
    { title: 'Review', detail: 'fresh Grok adversarially reviews each implementation' },
  ],
}

const WT = 'D:\\GITHUB\\CHELATEDAI\\.claude\\worktrees\\agent-build'
const PANEL = 'D:\\GITHUB\\CHELATEDAI\\docs\\waypoint-research-2026-06-09\\panel'

function driveGrok(fileKey, promptBody) {
  const promptPath = `${PANEL}\\wf2_${fileKey}.md`
  const outPath = `${PANEL}\\out_wf2_${fileKey}.txt`
  return [
    `You are driving an external Grok CLI agent. Do exactly this and nothing else:`,
    `1. Write the following text verbatim to \`${promptPath}\` with your Write tool:`,
    `-----BEGIN PROMPT-----`,
    promptBody,
    `-----END PROMPT-----`,
    `2. Run this PowerShell command with tool timeout 570000 ms and wait:`,
    `   grok -m grok-4.5 --effort low --cwd "${WT}" --prompt-file "${promptPath}" *> "${outPath}"`,
    `3. Read \`${outPath}\` and return its full contents as your final message, prefixed "GROK ${fileKey}:".`,
    `   If the run errored/timed out, return "GROK ${fileKey}: CLI_FAILED" plus whatever partial output exists.`,
  ].join('\n')
}

const W1_IMPL = [
  `# Implement the missing Obj B preflight test file (test debt)`,
  ``,
  `The build of research/drift_recovery/d2b_preflight/ skipped its test deliverable. Write`,
  `\`research/drift_recovery/tests/test_sparse_local_preflight.py\` (plain unittest, no pytest; offline;`,
  `CPU-only; keep runtime under ~60s by using small N/d/K). Import from`,
  `research.drift_recovery.regimes.sparse_local_nonaffine and research.drift_recovery.d2b_preflight.preflight.`,
  `Read those modules first to match real APIs. Required tests:`,
  `1. Warp properties: harmed-cluster fraction matches s (sparse); displacement is zero on clean clusters;`,
  `   the within-cluster map is NON-AFFINE (fit the best least-squares affine map on a harmed cluster's`,
  `   clean->corrupted pairs and assert residual > tolerance) and NON-RADIAL (displacement not parallel to`,
  `   the radial direction from the cluster centroid) for BOTH families (quadratic, soft_fold).`,
  `2. No clean-ID leakage: fit_local_ridges training_audit has clean_cluster_ids_accepted == False, and`,
  `   run_preflight_cell's detector block reports clean_ids_used_by_fit == False and that value is derived`,
  `   from the fit audits (check it flips if you monkeypatch an audit to True — behavioral, not just a flag read).`,
  `3. Sanity ladder: on a synthetic cell, the best GLOBAL ridge leaves a residual while a dense local fit`,
  `   using oracle clean cluster IDs recovers strictly more (diagnostic upper bound ordering).`,
  `4. Hyperparameter hygiene: lambda/rank selection uses only anchor train/dev rows (selection_source ==`,
  `   'anchor_dev_mse_only'; eval indices disjoint from fit+dev indices).`,
  `5. New audit fields present: training_audit contains selected_lambda, selected_rank (None allowed for`,
  `   dense), selected_lambdas_per_cluster.`,
  `Then RUN: python -m unittest research.drift_recovery.tests.test_sparse_local_preflight -v (with`,
  `HF_HUB_OFFLINE=1) and paste the full pass/fail output. If any test fails, fix the TEST unless it reveals`,
  `a real bug in preflight.py — in that case report the bug with file:line and do NOT paper over it.`,
].join('\n')

const W1_REVIEW = [
  `# Adversarially review the new test file tests/test_sparse_local_preflight.py`,
  ``,
  `A fresh implementer just wrote it. Try to DISPROVE that it provides real coverage:`,
  `- Are the non-affine / non-radial assertions real (fit-an-affine-and-check-residual) or tautological?`,
  `- Would the clean-ID-leakage test actually CATCH a leak (behavioral check), or does it only read a flag?`,
  `- Does the sanity ladder genuinely order global < oracle-ID local, with meaningful margins?`,
  `- RUN the suite yourself (python -m unittest research.drift_recovery.tests.test_sparse_local_preflight -v)`,
  `  and confirm it passes; also run the two pre-existing D2 suites (test_d2_decisions, test_synthetic_collapse)`,
  `  to confirm no regression.`,
  `Return PASS / PASS-WITH-FIXES / FAIL with file:line specifics and a one-line bottom line.`,
].join('\n')

const W3_IMPL = [
  `# Draft the paper-updates section (chair will apply; do NOT touch the paper itself)`,
  ``,
  `Write a chair-ready draft to \`${PANEL}\\paper-updates-2026-07-10-draft.md\`. Ground EVERY number in the`,
  `frozen artifacts — read these first:`,
  `- research/drift_recovery/out/estimator/objA_validation.md (+ objA_REPORT.md, objA_validation.json)`,
  `- research/drift_recovery/out/d2b_preflight/preflight_report.md`,
  `- research/drift_recovery/out/d2/D2_REPORT.md`,
  `The draft must contain three blocks of proposed paper text (markdown, ready to paste):`,
  `1. A new subsection for §7 ("A margin-based recoverability estimator — promising but underpowered"):`,
  `   preregistered univariate oracle_margin_mean; block-LOO Spearman 0.886 (dataset) / 0.829 (encoder) vs`,
  `   gap-only null; partial Spearman 0.79 controlling for oracle gap; the FiQA-only-holdout tie disclosed;`,
  `   only 3 dataset-blocks -> NOT validated; what a powered test requires.`,
  `2. A §7 paragraph closing the "home-turf" question: the sparse-local non-affine CPU preflight (48 cells,`,
  `   144 seed-runs, two warp families) admitted no cell at residual>=0.05 after a fair CV-lambda gated local`,
  `   ridge (max 0.033). MUST use the corrected framing: residuals are small because absolute gaps are small`,
  `   and local ridge often loses to floor; this does NOT show local ridge dominates chelation; it shows no`,
  `   discriminating home-turf residual was constructible, so the bounded/annealed corrector gets no second arena.`,
  `3. One-to-two-sentence proposed additions to §8 Limitations (underpowered estimator; synthetic-only preflight).`,
  `HARD BANS (from the C1 chair rules): no "ceiling" claims, no "irreducible", no "ridge~=MLP proves linearity",`,
  `no upgrading PROMISING-BUT-UNDERPOWERED to validated. Label each block with WHERE it goes in main.md.`,
].join('\n')

const W3_REVIEW = [
  `# Adversarially review the paper-updates draft`,
  ``,
  `Read \`${PANEL}\\paper-updates-2026-07-10-draft.md\` and try to DISPROVE its fitness:`,
  `- Verify EVERY number against research/drift_recovery/out/estimator/objA_validation.md and`,
  `  research/drift_recovery/out/d2b_preflight/preflight_report.md. Any number not in an artifact = FAIL.`,
  `- Hunt L13 overclaims: does any sentence upgrade the estimator toward "validated", call anything a`,
  `  "ceiling", claim local-ridge-dominates-chelation, or soften the FiQA-holdout tie?`,
  `- Check the home-turf paragraph uses the corrected causal framing (small absolute gaps; local often < floor).`,
  `- Is the §8 limitations text honest about 3 dataset-blocks and synthetic-only preflight?`,
  `Return PASS / PASS-WITH-FIXES / FAIL with quoted offending sentences and exact replacement text.`,
].join('\n')

const ITEMS = [
  { key: 'W1-test-debt', impl: W1_IMPL, review: W1_REVIEW },
  { key: 'W3-paper-draft', impl: W3_IMPL, review: W3_REVIEW },
]

phase('Implement')
const results = await pipeline(
  ITEMS,
  item => agent(driveGrok(`${item.key}-impl`, item.impl), {
    label: `grok-impl:${item.key}`, phase: 'Implement', agentType: 'general-purpose',
  }),
  (implReport, item) => agent(driveGrok(`${item.key}-review`, item.review), {
    label: `grok-review:${item.key}`, phase: 'Review', agentType: 'general-purpose',
  }).then(review => ({ key: item.key, impl: implReport, review })),
)

return {
  items: results.filter(Boolean).map(r => ({
    key: r.key,
    implTail: String(r.impl).slice(-2000),
    reviewTail: String(r.review).slice(-3000),
  })),
}
