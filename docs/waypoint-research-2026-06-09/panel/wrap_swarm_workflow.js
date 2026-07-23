export const meta = {
  name: 'lattice-wrap-swarm',
  description: 'Close out the session: rung-15 CUT, stale-PR triage, within-space wedge prereg (power math FIRST), and session wrap — all adversarially verified',
  phases: [
    { title: 'Investigate', detail: '11 parallel agents: PR triage, wedge-test design, rung-15 CUT, wrap drafts' },
    { title: 'Verify', detail: '8 adversarial reviewers incl. 3-vote on the load-bearing power math' },
    { title: 'Synthesize', detail: 'merge into chair-ready deliverables' },
  ],
}

const WT = 'D:\\GITHUB\\CHELATEDAI\\.claude\\worktrees\\relaxed-wozniak-271e04'
const PANEL = 'D:\\GITHUB\\CHELATEDAI\\docs\\waypoint-research-2026-06-09\\panel'
const MEM = 'C:\\Users\\mattm\\.claude\\projects\\D--GITHUB-CHELATEDAI\\memory'

const CONTEXT = `
## Session context you MUST respect (it is load-bearing)

This program has produced SIX consecutive fail-closeds where elaborate structure lost to a trivial
global LINEAR baseline:
1. drift-corrector chelation beaten ~4.2x by a one-line ridge map (Holm p<0.001);
2. recoverability estimator (oracle_margin_mean) INVERTED at powered scale (block-LOO Spearman
   +0.886 at 6 cells -> -0.706 at 12 cells; loses to a gap-only null) = inventory overfit;
3. sparse-local non-affine home-turf preflight admitted NO cell (48 cells/144 runs, max residual
   0.033 < 0.05 gate);
4. H5 living/annealed post-bank tied a FROZEN STATIC bank bit-identically and lost to a one-shot router;
5. H4 compounding collapsed recovery ~45x (0.236 -> 0.005);
6. rung-16 quant-aware routing plane FAIL-CLOSED on both arenas (plane LOSES to single-global by
   -0.0033 / -0.0074).
   IMPORTANT NUANCE (Tier B caught this): in Arena B, domain specialists actually HELPED when routed
   to their own domain (+0.0257, n=21) but centroid routing under encoder-swap misrouted 65% of
   queries and those (-0.0309, n=39) dominated. So the binding constraint was ROUTE ASSIGNMENT, not
   specialist capacity. Arena B falsified the preregistered centroid-margin plane, NOT domain routing.

A 106-agent deep-research pass (23 sources, 25 adversarially-verified claims, 6 refuted) established:
- The superposition/compressed-sensing WEDGE IS REAL AND PROVEN: a linear readout is provably NOT
  span-exhausting. Garg-Kleinberg-Peng 2026 (arXiv:2602.11246) prove a QUADRATIC gap — nonlinear
  (l1/compressed-sensing) decoding of k-sparse features needs d=O(k log(m/k)) while linear
  accessibility needs d=Otilde(k^2 log m), with a matching lower bound. Corroborated by Anthropic
  superposition and Engels et al. ICLR 2025 (arXiv:2405.14860) causally-verified 2D circular features.
- BUT every demonstrated exploiter of that wedge is GRADIENT-TRAINED (SAE, MP-SAE), and sparse
  dictionary learning is provably NON-IDENTIFIABLE (arXiv:2512.05534: zero reconstruction loss while
  recovering ZERO ground-truth features, empirically 3/3200).
- At CROSS-space alignment, a gradient-free LINEAR method (mini-vec2vec, arXiv:2510.02348) MATCHES OR
  EXCEEDS the nonlinear adversarial vec2vec; vec2vec's OOD-robustness claim was REFUTED 0-3.

An earlier design (HI-1: gradient-free harmonic operator vs ridge in OOD regions of a frozen cloud)
was CUT before any GPU because a red-team proved it was:
 (a) UNDERPOWERED BY DESIGN — realistic n was ~9-27 queries; paired 95% half-width ~0.08-0.14; the
     written bar (CI lower bound >= +0.01) needed a true effect of ~0.09-0.15 absolute NDCG, larger
     than rung-16's entire effect. A guaranteed fail-closed, not a coin flip.
 (b) OPERATOR-STARVED — the "unknown" stratum was DEFINED as low-density, but diffusion maps need
     density and Nystrom extension into sparse regions is the textbook spectral failure mode.
 (c) BASELINE TOO WEAK — "best of ridge/Procrustes" is weaker than the repo's leakage-safe full-fit
     AFFINE ridge (Wx+b), while the treatment got 6 knobs to the baseline's one lambda.
 (d) UNFALSIFIABLE — both outcomes were pre-narrated, so decision value was ~0.

THE LESSON, which governs every design you produce here: **do the power math FIRST and derive the win
bar from it. If the bar is not clearable at the achievable n, say CUT.** Never propose a bar first and
hope n cooperates.

Repo: ${WT}. Panel (drafts go here): ${PANEL}. Memory: ${MEM}.
Be brutally honest. A CUT recommendation is a WIN, not a failure. Never fabricate numbers or citations.
`

phase('Investigate')

// ---- Stale PR triage (3 parallel) ----
const prTriage = [278, 257, 256].map(n => () => agent(
  [
    `You are triaging stale PR #${n} in the CHELATEDAI repo (cwd = ${WT}).`,
    CONTEXT,
    `Use the gh CLI (via PowerShell) to inspect PR #${n}: title, body, age, files changed, additions/`,
    `deletions, CI/check status, mergeStateStatus, review state, and whether its branch still exists.`,
    `Then inspect whether its content is SUPERSEDED by work merged since it was opened (the lattice`,
    `rungs, drift-recovery series #258-#291, track-0 hygiene #267, etc.).`,
    ``,
    `Deliver a recommendation of exactly one of: MERGE (still valid + green), CLOSE (superseded or`,
    `abandoned — say precisely what superseded it), or REWORK (valuable but stale — say what must`,
    `change). Justify with concrete evidence (dates, file overlap, superseding PR numbers, check`,
    `status). Flag if it contains uncommitted-elsewhere value that would be LOST by closing.`,
    `Be decisive; "leave it open" is not an option unless you can justify why it is still actionable.`,
  ].join('\n'),
  { label: `pr-triage:#${n}`, phase: 'Investigate', agentType: 'general-purpose' }
).then(r => ({ kind: 'pr', n, report: r })))

// ---- Within-space wedge test: POWER MATH FIRST (load-bearing) ----
const powerAgent = () => agent(
  [
    `You are doing the LOAD-BEARING POWER ANALYSIS for a proposed "within-space wedge test". Nothing`,
    `else in this design matters if the bar is not clearable — HI-1 died exactly here.`,
    CONTEXT,
    `## The proposed test`,
    `In ONE frozen embedding space containing KNOWN superposed / multi-dimensional features (either`,
    `synthetically constructed with known ground truth, or Engels-style circular features), can a`,
    `GRADIENT-FREE nonlinear readout (harmonic / compressed-sensing style, e.g. l1 / basis-pursuit`,
    `decoding, closed-form spectral) beat a LINEAR PROBE at recovering the feature?`,
    ``,
    `## Your job — compute, do not hand-wave`,
    `1. Identify candidate METRICS (feature-recovery accuracy/AUC/R^2, or a downstream retrieval label`,
    `   that depends on the nonlinear feature). For each, state its variance behavior and how many`,
    `   samples are needed for a given detectable effect.`,
    `2. Compute the ACHIEVABLE n. Synthetic construction means n is essentially FREE (we can generate`,
    `   thousands of points) — this is the key structural advantage over HI-1's ~9-27 queries. Verify`,
    `   that claim and quantify: what n is cheap on CPU? what n on the cached 3090?`,
    `3. Derive the MINIMUM DETECTABLE EFFECT at 80% and 95% power for the leading metric at several n`,
    `   (e.g. 200 / 1k / 10k / 100k). Show the arithmetic (paired bootstrap half-width or standard`,
    `   power formula — state which and why).`,
    `4. Compare against the THEORETICALLY PREDICTED effect: Garg-Kleinberg-Peng predict a quadratic`,
    `   dimensional gap (nonlinear d=O(k log(m/k)) vs linear d=Otilde(k^2 log m)). Translate that into`,
    `   an expected accuracy/recovery gap in a concrete (d, m, k) regime we could actually construct.`,
    `   Name (d, m, k) settings where the gap should be LARGE and unambiguous.`,
    `5. VERDICT: is there a concrete (metric, n, d, m, k) configuration where the win bar is clearly`,
    `   clearable AND the predicted effect is well above the minimum detectable effect? If YES, name`,
    `   the exact operating point and the bar. If NO, say CUT and explain.`,
    ``,
    `Show all arithmetic. Do NOT propose a bar you have not shown is clearable. Write your analysis to`,
    `${PANEL}\\wedge-power-analysis.md and return its key numbers + verdict.`,
  ].join('\n'),
  { label: 'wedge:power-math', phase: 'Investigate', agentType: 'general-purpose', effort: 'high' }
)

const featureAgent = () => agent(
  [
    `Design the FEATURE CONSTRUCTION for the within-space wedge test.`,
    CONTEXT,
    `We need a frozen embedding space containing features that are provably IN-SPAN but LINEARLY`,
    `INACCESSIBLE — the regime where the compressed-sensing wedge predicts nonlinear decoding wins.`,
    `Two candidate routes:`,
    `  (A) SYNTHETIC: construct x = sum_i a_i f_i where f_i are m >> d almost-orthogonal feature`,
    `      directions in R^d, activations a are k-sparse. Ground truth is known exactly. This gives`,
    `      free n and exact labels, and directly instantiates the Garg-Kleinberg-Peng setting.`,
    `  (B) REAL: use a real frozen encoder and a known multi-dimensional feature (Engels-style circular`,
    `      day-of-week / month features), where a 1-D linear readout provably cannot represent the`,
    `      modular structure.`,
    `For each: give the exact construction, the ground-truth label, why a linear probe should fail,`,
    `why a nonlinear/CS readout should succeed, what could confound it, and how expensive it is.`,
    `Recommend which route (or both, A as the clean instrument and B as the reality check).`,
    `Write to ${PANEL}\\wedge-feature-construction.md and return a summary.`,
  ].join('\n'),
  { label: 'wedge:features', phase: 'Investigate', agentType: 'general-purpose' }
)

const operatorAgent = () => agent(
  [
    `Design the GRADIENT-FREE NONLINEAR READOUT for the within-space wedge test.`,
    CONTEXT,
    `Constraint that makes this novel: it must be GRADIENT-FREE — no backprop, no SGD, no learned`,
    `dictionary. That rules out SAEs/MP-SAE (which are the only demonstrated exploiters of the wedge,`,
    `and are provably non-identifiable). Candidates to evaluate:`,
    `  - l1 / basis-pursuit / LASSO decoding against a KNOWN dictionary (convex, closed-form-ish,`,
    `    no gradient training of the dictionary) — the literal compressed-sensing decoder;`,
    `  - Orthogonal Matching Pursuit (greedy, deterministic);`,
    `  - kernel / spectral readout (closed-form eigendecomposition);`,
    `  - anchor-relative + diffusion-harmonic coordinates.`,
    `For each: is it truly gradient-free? does it exploit the CS wedge or just add capacity? what does`,
    `it need to know a priori (the dictionary? the sparsity k?) and is assuming that a fair test or a`,
    `cheat? Recommend ONE primary operator + a fallback, and state exactly what it is allowed to know.`,
    `CRITICAL FAIRNESS QUESTION you must answer: if the operator is given the true dictionary, is the`,
    `comparison against a linear probe still meaningful, or is it rigged? Propose the fair information`,
    `parity condition.`,
    `Write to ${PANEL}\\wedge-operator-design.md and return a summary.`,
  ].join('\n'),
  { label: 'wedge:operator', phase: 'Investigate', agentType: 'general-purpose' }
)

const baselineAgent = () => agent(
  [
    `Design the STRONGEST LINEAR BASELINE the within-space wedge test must beat.`,
    CONTEXT,
    `HI-1 was faulted for a weak baseline. Do not repeat that. Enumerate and rank the strongest linear`,
    `readouts: plain linear probe (least squares / logistic), ridge with lambda tuned on a dev split,`,
    `reduced-rank / PLS / CCA, LDA, and any linear method that could plausibly recover a superposed`,
    `feature. State which is strongest for THIS task and why, and specify the tuning protocol (dev`,
    `split only, never test).`,
    `Then specify CAPACITY / INFORMATION PARITY between the linear baseline and the nonlinear operator:`,
    `same training samples, same dev budget, same prior knowledge (dictionary? sparsity?). Write the`,
    `parity contract explicitly so a reviewer can check it.`,
    `Also: state the condition under which the linear baseline SHOULD win (i.e. what result would`,
    `falsify the wedge in this setting) — a good test must be able to come out either way.`,
    `Write to ${PANEL}\\wedge-baseline-design.md and return a summary.`,
  ].join('\n'),
  { label: 'wedge:baseline', phase: 'Investigate', agentType: 'general-purpose' }
)

// ---- Rung 15 CUT ----
const rung15Agent = () => agent(
  [
    `Draft the RUNG 15 (GNN prototype) CUT decision and the exact ROADMAP edit.`,
    CONTEXT,
    `The operator has decided to CUT rung 15 and close Phase II. Your job is to make that decision`,
    `HONEST and defensible, not a rationalization for avoiding a negative result.`,
    `Read ${WT}\\docs\\ROADMAP_EXECUTION.md (Phase II table, step 15 row + status snapshot + the`,
    `"What we are not doing" bullet) and ${WT}\\evidence_dag.py.`,
    `Rung 15's exit criteria: "Lightweight GNN over evidence DAG (PyG or DGL); only after steps 12-14`,
    `green; must beat flat-pool baseline on drift fixture or fail closed."`,
    ``,
    `Produce:`,
    `1. An honest CUT justification. The strongest honest grounds are: (i) the same power problem that`,
    `   killed HI-1 — estimate the achievable n on the drift fixture and whether "beat flat-pool" is`,
    `   clearable there (do the arithmetic if you can from repo artifacts); (ii) the six-fail-closed`,
    `   pattern; (iii) a GNN over the evidence DAG is another within-span reparameterization, which the`,
    `   deep-research verdict says is dominated. BE HONEST: if you think CUT is actually unjustified and`,
    `   rung 15 should be RUN, say so — the operator asked for honesty, not agreement.`,
    `2. The EXACT old->new text for the step-15 ROADMAP row (status CUT with reasoning), the status`,
    `   snapshot line, and the "What we are not doing" bullet, so Phase II reads as`,
    `   complete-with-one-documented-cut rather than abandoned.`,
    `3. One sentence for CHANGELOG.`,
    `Write to ${PANEL}\\rung15-cut-decision.md and return the exact edit text.`,
  ].join('\n'),
  { label: 'rung15:cut', phase: 'Investigate', agentType: 'general-purpose' }
)

// ---- Phase II honest state ----
const phase2Agent = () => agent(
  [
    `Establish the GIT-VERIFIED honest state of Phase II so "closing Phase II" is a true statement.`,
    CONTEXT,
    `In ${WT}: for EACH Phase II step 9-17, verify from git (git log, merged PRs, actual code presence)`,
    `what is truly DONE, PARTIAL, CUT, or OPEN — do not trust the ROADMAP prose, verify it.`,
    `Note that rungs 13, 16, 17 were completed in THIS session and are on branches/PRs not yet merged`,
    `to main (#293 rung13, #294 rung17, rung16 committed locally on lattice/rung16-routing-20260714).`,
    `Deliver a table: step | claimed status | git-verified status | evidence (PR/commit/file) | any`,
    `discrepancy. Flag anything the ROADMAP currently overstates or understates. This is the docs-truth`,
    `check that must hold before we declare Phase II closed.`,
    `Write to ${PANEL}\\phase2-final-state-audit.md and return the table.`,
  ].join('\n'),
  { label: 'phase2:audit', phase: 'Investigate', agentType: 'general-purpose' }
)

// ---- Session wrap drafts ----
const memoryAgent = () => agent(
  [
    `Draft the MEMORY update for this session's endgame round.`,
    CONTEXT,
    `Read the existing memory at ${MEM}\\MEMORY.md and`,
    `${MEM}\\drift-orchestration-outcome-2026-07-09.md to match style and avoid duplication.`,
    `Draft an update capturing ONLY what is new and non-obvious from the endgame round:`,
    `- rung 13 DONE (detector-driven Evidence-DAG prune; caught a chelation-mode signedness bug,`,
    `  fail-closed on non-sedimentation mode), PR #293, Tier B 100;`,
    `- rung 17 DONE (block-graph pool-shard read, bit-exact via FP16 byte-lane encoding, verified on`,
    `  100k random float32; byte-equality parity), PR #294, Tier B 100;`,
    `- rung 16 FAIL-CLOSED both arenas + the Arena B purity nuance (specialists +0.0257 when home-routed,`,
    `  misroutes -0.0309 dominate, purity 30/3.3/36.7% -> binding constraint is ROUTE ASSIGNMENT not`,
    `  capacity); Tier B 90 -> fix applied;`,
    `- the deep-research verdict (wedge proven real; gradient-free nonlinear unpursued but dominated);`,
    `- HI-1 CUT before GPU (4 reasons incl. unfalsifiable framing = chair design error);`,
    `- rung 15 CUT, Phase II closed.`,
    `Follow the memory format rules exactly (frontmatter name/description/metadata type, [[links]]).`,
    `Decide: update the existing outcome file vs create a new one — justify. Write your DRAFT to`,
    `${PANEL}\\memory-update-draft.md (do NOT write to the memory dir; the chair applies it).`,
  ].join('\n'),
  { label: 'wrap:memory', phase: 'Investigate', agentType: 'general-purpose' }
)

const nextSessionAgent = () => agent(
  [
    `Draft the next-session.md update for this session's endgame round.`,
    CONTEXT,
    `Read ${WT}\\docs\\next-session.md (Tier C state surface; fixed schema — do NOT rename headings or`,
    `collapse tables). Draft:`,
    `- any Carried Debt rows this round genuinely opens (be strict: a closed research question is NOT`,
    `  debt; an unlanded PR or an untested path IS). Consider: rung-16 committed-but-unpushed at time`,
    `  of writing, the 3 stale PRs, and anything the Tier B reviews deferred;`,
    `- Deferred Scope rows (honestly-bounded not-in-scope items): rung 15 CUT, the within-space wedge`,
    `  test if not run, rung-16's oracle/domain-label routing ablation;`,
    `- a short disposition note for the endgame round.`,
    `Verify the block flag stays CLEAR (run scripts/check_block_flag.py) and say so.`,
    `Write your DRAFT to ${PANEL}\\next-session-update-draft.md and return it.`,
  ].join('\n'),
  { label: 'wrap:next-session', phase: 'Investigate', agentType: 'general-purpose' }
)

const investigations = await parallel([
  ...prTriage,
  powerAgent, featureAgent, operatorAgent, baselineAgent,
  rung15Agent, phase2Agent, memoryAgent, nextSessionAgent,
])

const prReports = investigations.filter(Boolean).filter(r => r && r.kind === 'pr')
const [power, features, operator, baseline, rung15, phase2, memoryDraft, nextSessionDraft] =
  investigations.slice(prTriage.length)

phase('Verify')

// 3-vote adversarial verification of the POWER MATH (the load-bearing claim)
const powerVotes = await parallel([1, 2, 3].map(i => () => agent(
  [
    `You are adversarial verifier #${i} of 3 on a POWER ANALYSIS. Your job is to REFUTE it.`,
    CONTEXT,
    `Read ${PANEL}\\wedge-power-analysis.md. The analysis claims a concrete operating point where the`,
    `within-space wedge test's win bar is clearable at achievable n.`,
    `ATTACK: (a) is the arithmetic right — recompute the minimum detectable effect yourself; (b) is the`,
    `claimed achievable n real, or does something (compute, construction cost, label scarcity) cap it`,
    `far lower; (c) is the predicted effect size from the compressed-sensing theorem correctly`,
    `translated into the chosen metric, or is it an unjustified leap; (d) would this test, like HI-1,`,
    `be underpowered-by-design or unwinnable-by-construction; (e) is the proposed bar derived FROM the`,
    `power math, or retrofitted to look clearable?`,
    `Default to REFUTED if you cannot independently confirm the arithmetic. Return a clear verdict:`,
    `REFUTED or CONFIRMED, with your own recomputation and the single strongest objection.`,
  ].join('\n'),
  { label: `verify:power-vote-${i}`, phase: 'Verify', agentType: 'general-purpose', effort: 'high' }
)))

// Independent Grok review of the whole wedge design
const grokWedge = agent(
  [
    `Drive an external Grok CLI agent to adversarially review the whole within-space wedge test design.`,
    `1. Write a prompt file to ${PANEL}\\wf5_wedge_review.md instructing Grok to read`,
    `   ${PANEL}\\wedge-power-analysis.md, wedge-feature-construction.md, wedge-operator-design.md, and`,
    `   wedge-baseline-design.md, and to answer: is this test winnable-in-principle AND falsifiable AND`,
    `   fairly baselined? Is the operator genuinely gradient-free? Is giving it the dictionary a cheat?`,
    `   Is the power math sound? Verdict PROCEED / PROCEED-WITH-CHANGES / CUT with the single`,
    `   highest-value change. Tell Grok the HI-1 cut reasons so it applies the same standard.`,
    `2. Run (PowerShell, tool timeout 570000 ms):`,
    `   & "C:\\Users\\mattm\\.grok\\bin\\grok.exe" -m grok-4.5 --effort high --cwd "${WT}" --prompt-file "${PANEL}\\wf5_wedge_review.md" *> "${PANEL}\\out_wf5_wedge_review.txt"`,
    `3. Read the output and return it verbatim, prefixed "GROK WEDGE REVIEW:".`,
  ].join('\n'),
  { label: 'verify:grok-wedge', phase: 'Verify', agentType: 'general-purpose' }
)

const verifyRung15 = agent(
  [
    `Adversarially review the rung-15 CUT decision at ${PANEL}\\rung15-cut-decision.md.`,
    CONTEXT,
    `The danger: cutting rung 15 to AVOID a predictable negative would be intellectually dishonest`,
    `(result-avoidance dressed as discipline). Attack it: is the CUT justified by power/dominance`,
    `arithmetic, or is it convenience? Would running rung 15 actually teach us something the six prior`,
    `fail-closeds did not? Is the proposed ROADMAP text honest about WHY it was cut (and does it avoid`,
    `implying rung 15 was completed)? Verify the exit criteria are quoted accurately.`,
    `Return PASS / PASS-WITH-FIXES / FAIL + the exact corrected text if needed.`,
  ].join('\n'),
  { label: 'verify:rung15-cut', phase: 'Verify', agentType: 'general-purpose' }
)

const verifyPRs = agent(
  [
    `Adversarially review the three stale-PR triage recommendations below. For each, check the`,
    `recommendation is justified by evidence (not just age), that CLOSE recommendations correctly`,
    `identify what superseded them, and that no unique un-merged value would be destroyed.`,
    CONTEXT,
    ...prReports.map(p => `## PR #${p.n}\n${p.report}\n`),
    `Return, per PR: AGREE or DISAGREE (with the corrected call) and a one-line justification, plus a`,
    `recommended action order for the operator.`,
  ].join('\n'),
  { label: 'verify:pr-triage', phase: 'Verify', agentType: 'general-purpose' }
)

const verifyWrap = agent(
  [
    `Adversarially fact-check the session-wrap drafts against reality.`,
    CONTEXT,
    `Read ${PANEL}\\memory-update-draft.md and ${PANEL}\\next-session-update-draft.md, plus the`,
    `phase-2 audit at ${PANEL}\\phase2-final-state-audit.md.`,
    `Verify EVERY factual claim (PR numbers, Tier B scores, NDCG figures, purity percentages, rung`,
    `statuses) against the repo artifacts and git. Any number or claim you cannot verify = FLAG IT.`,
    `Specifically check: rung-16 Arena B purity (30.0/3.3/36.7%) and attribution (+0.0257 home n=21,`,
    `-0.0309 cross n=39, total -0.007383); rung 13/17 Tier B scores; that unmerged work is described`,
    `as unmerged. Return a defect list with corrections.`,
  ].join('\n'),
  { label: 'verify:wrap-facts', phase: 'Verify', agentType: 'general-purpose' }
)

const verifyPhase2 = agent(
  [
    `Adversarially verify the Phase II final-state audit at ${PANEL}\\phase2-final-state-audit.md.`,
    CONTEXT,
    `For every step marked DONE, confirm from git that the delivering code/PR really exists; for every`,
    `step marked OPEN/CUT/PARTIAL, confirm it is genuinely not delivered. Any status the audit gets`,
    `wrong in EITHER direction is a defect. Also judge: is it honest to declare "Phase II closed" given`,
    `rungs 13/16/17 are on unmerged branches and rung 15 is cut? Propose the precise honest wording.`,
    `Return PASS / PASS-WITH-FIXES / FAIL with per-step corrections.`,
  ].join('\n'),
  { label: 'verify:phase2-audit', phase: 'Verify', agentType: 'general-purpose' }
)

const [wedgeGrok, rung15Review, prReview, wrapReview, phase2Review] =
  await parallel([() => grokWedge, () => verifyRung15, () => verifyPRs, () => verifyWrap, () => verifyPhase2])

phase('Synthesize')

const synthesis = await agent(
  [
    `You are the synthesis brain. Produce ONE chair-ready close-out memo from the material below.`,
    CONTEXT,
    `## Power analysis\n${power}`,
    `## Power verification votes (3-vote adversarial; need 2/3 CONFIRMED to survive)`,
    ...powerVotes.filter(Boolean).map((v, i) => `### vote ${i + 1}\n${v}`),
    `## Wedge design — features\n${features}`,
    `## Wedge design — operator\n${operator}`,
    `## Wedge design — baseline\n${baseline}`,
    `## Grok independent wedge review\n${wedgeGrok}`,
    `## Rung 15 CUT draft\n${rung15}`,
    `## Rung 15 CUT review\n${rung15Review}`,
    `## Phase II audit\n${phase2}`,
    `## Phase II audit review\n${phase2Review}`,
    `## PR triage review\n${prReview}`,
    `## Wrap fact-check\n${wrapReview}`,
    ``,
    `Deliver, tightly:`,
    `1. WEDGE TEST VERDICT: does the power math survive 2/3 adversarial votes AND Grok? If yes, state`,
    `   the exact operating point (metric, n, d/m/k), the derived win bar, and PROCEED. If no, state`,
    `   CUT and why. Do not hedge.`,
    `2. RUNG 15: final CUT text (or RUN if the review overturned it), with exact ROADMAP edits.`,
    `3. PHASE II: the honest closing statement, accounting for unmerged branches.`,
    `4. STALE PRs: final per-PR action (#278/#257/#256) in operator order.`,
    `5. WRAP: the corrected memory + next-session text, with any fact-check defects fixed.`,
    `6. A short "what the chair must do by hand" list.`,
    `Flag anything that failed verification. Be brutally honest; a CUT is a fine outcome.`,
  ].join('\n'),
  { label: 'synthesis', phase: 'Synthesize', effort: 'high' }
)

return {
  prTriage: prReports.map(p => p.n),
  powerVerdicts: powerVotes.filter(Boolean).length,
  synthesis,
}
