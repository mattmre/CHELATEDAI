export const meta = {
  name: 'objAB-review-swarm',
  description: 'Grok adversarial cross-review swarm over the Obj A + Obj B build results, then synthesize a go/no-go',
  phases: [
    { title: 'Cross-review', detail: 'parallel Grok reviewers, distinct attack angles per result' },
    { title: 'Synthesize', detail: 'aggregate verdicts into final Obj A/B decision + remaining work' },
  ],
}

const WT = 'D:\\GITHUB\\CHELATEDAI\\.claude\\worktrees\\agent-build'
const PANEL = 'D:\\GITHUB\\CHELATEDAI\\docs\\waypoint-research-2026-06-09\\panel'

// Each review target: a result file + a specific adversarial angle. Grok is driven via CLI
// (fast, within the sub-agent tool-call ceiling), so these run reliably inside workflow agents.
const REVIEWS = [
  {
    key: 'objA-beats-gap',
    result: 'research/drift_recovery/out/estimator/objA_validation.md',
    angle: 'Does oracle_margin_mean actually beat the oracle-gap-only null under block-LOO once FiQA is added, or did the n=4 +1.0 collapse toward gap-level? Verify the block-LOO is not circular and that the verdict label (POSITIVE vs UNDERPOWERED/NEGATIVE) matches the numbers. Recompute the block-LOO Spearman/MAE for margin vs gap-only yourself from the artifacts.',
  },
  {
    key: 'objA-power-honesty',
    result: 'research/drift_recovery/out/estimator/objA_validation.md',
    angle: 'Is the power/pseudo-replication disclosure honest? Confirm block-n counts independent dataset x encoder-family blocks (not hyperparam cells), that anchor-fraction variants were not used to pad, and that any POSITIVE is correctly downgraded to PROMISING-BUT-UNDERPOWERED at 3 dataset-blocks.',
  },
  {
    key: 'objB-close-or-proceed',
    result: 'research/drift_recovery/out/d2b_preflight/preflight_report.md',
    angle: 'Is the CLOSE/PROCEED verdict honest? The admission gate is residual-after-GATED-LOCAL-ridge >= 0.05 (NOT after global ridge). Confirm the local ridge got a fair CV-lambda (tuned like chelation alpha would be), that clusters were inferred on corrupted vectors only (no clean-ID leakage), that both warp families were tested, and that a PROCEED is not just measuring residual-after-GLOBAL-ridge. If CLOSE, is the one-liner warranted; if PROCEED, is the operating point real and small-magnitude?',
  },
]

phase('Cross-review')
const findings = await parallel(REVIEWS.map(r => async () => {
  const promptPath = `${PANEL}\\wf_${r.key}.md`
  const outPath = `${PANEL}\\out_wf_${r.key}.txt`
  const prompt = [
    `# Adversarial review (${r.key})`,
    ``,
    `Read the result artifact at \`${r.result}\` (relative to the agent-build worktree root) and any`,
    `sibling JSON/report it references. Try to DISPROVE its verdict on this specific angle:`,
    ``,
    r.angle,
    ``,
    `Recompute from the frozen artifacts where possible. Return: PASS / PASS-WITH-FIXES / FAIL, the`,
    `specific defect(s) with file:line, and a one-sentence bottom line on whether the verdict is honest.`,
  ].join('\n')

  const agentReport = await agent(
    [
      `You are driving an external Grok adversarial reviewer via CLI. Do exactly this and nothing else:`,
      `1. Write the following text to the file \`${promptPath}\` (use your Write tool, verbatim):`,
      `-----BEGIN PROMPT-----`,
      prompt,
      `-----END PROMPT-----`,
      `2. Run this PowerShell command (set the tool timeout to 540000 ms) and wait for it:`,
      `   grok -m grok-4.5 --effort low --cwd "${WT}" --prompt-file "${promptPath}" *> "${outPath}"`,
      `3. Read \`${outPath}\` and return its full contents as your final message, prefixed with`,
      `   "REVIEW ${r.key}:". If the grok run errored or timed out, return "REVIEW ${r.key}: CLI_FAILED" plus the error.`,
    ].join('\n'),
    { label: `grok-review:${r.key}`, phase: 'Cross-review', agentType: 'general-purpose' }
  )
  return { key: r.key, report: agentReport }
}))

phase('Synthesize')
const synthesis = await agent(
  [
    `You are the synthesis brain. Below are ${findings.filter(Boolean).length} independent Grok adversarial reviews of the`,
    `Obj A (oracle_margin_mean scale-validation) and Obj B (chelation home-turf preflight) build results.`,
    ``,
    ...findings.filter(Boolean).map(f => `## ${f.key}\n${f.report}\n`),
    ``,
    `Produce a tight final decision memo: (1) Obj A verdict — is the margin predictor a real (if underpowered)`,
    `lead or did it collapse; ship-label. (2) Obj B verdict — CLOSE (chelation has no unique home turf) or`,
    `PROCEED-TO-GPU with the operating point. (3) Any review FAIL/PASS-WITH-FIXES that must be applied before`,
    `these results are trustworthy, with file:line. (4) The remaining work, if any. Be brutally honest; a CLOSE`,
    `or UNDERPOWERED-NEGATIVE is an acceptable, valuable outcome — do not manufacture a positive.`,
  ].join('\n'),
  { label: 'synthesis', phase: 'Synthesize' }
)

return { reviews: findings.filter(Boolean).map(f => f.key), synthesis }
