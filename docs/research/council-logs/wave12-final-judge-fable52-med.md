# Wave 12/13 Final Judge #2 — 6.0 Seal Adjudication

- **Judge:** claude-fable-5-2 (medium), `/panel-of-experts` v1.1.0 + Brutal-Honesty Rulebook v3.3
- **Date:** 2026-11-25
- **Repo / HEAD:** `D:\GITHUB\CHELATEDAI-STAGE0-NIGHT1` @ `9e4921a9` (working tree dirty, see F7)
- **Registry under review:** `docs/research/verified_registry.json`, `registry_content_sha256 = c23d9801adb2528567fe767576fa76f2f718e54afb3ef68d5512fcd16b40d6a2`

## VERDICT: BLOCK

The block is narrow and text-only. The registry, receipts, manifests and both mechanical gates are sound, and I found no inflated gate status in the whitepaper, the presentation HTML or the spec. The seal is blocked by publication-text defects: the two Non-NDA investor files carry no pre-release or pending-human-audit statement and describe model-panel audits as "third-party" (F1), and the "all nuances disclosed on all three surfaces" criterion is not met (F2–F4). No experiment needs to be re-run. Correcting F1–F4 and re-running the two gates would move this to APPROVED_FOR_V6_SEAL; F5–F9 are non-blocking.

## What I ran (read-only, `CUDA_VISIBLE_DEVICES=""`)

| Check | Result |
|---|---|
| `python scripts/verify_non_nda_ip_clean.py` | `ZERO_IP_LEAK_AUDIT: PASS (0/56 forbidden method patterns matched across 2 file(s))`, exit 0 |
| `python scripts/build_whitepaper_assets.py --check` | `WHITEPAPER_ASSETS_CHECK: PASS (bundles=45/45, side_hashes=59, unhashed_hist_raw=20, pristine=15/15, registry_sha=c23d9801adb2)`, exit 0 |
| `git status` before and after the two runs | unchanged |

Registry facts confirmed by direct read: 45 bundles, 59 verified side artifacts, 20 unhashed historical raw files, pristine mirror 15/15 byte-exact, `release_status = "PRE-RELEASE / HUMAN AUDIT & CLEAN-RERUN GATES PENDING"`, PM-1b recorded as SCOPED/OPEN with `assignment_gates_pass=false`.

## What I did not do

- I did not run `scripts/verify_all_offline.py` (not in my scope), any GPU job, or any benchmark. Every D6–D10 number below is taken from the peer memos and publication text, not re-measured.
- I did not open any `*sealed_key*` or `*blind-followup*` file.
- I did not recompute manifest or side-artifact hashes by hand; I rely on the `--check` output for that.
- The "panel" is one model applying several lenses (statistics, release governance, investor-disclosure, skeptic). It is not independent multi-reviewer evidence.
- The four peer memos audited code and receipts at `6d4b41c9` / `9e4921a9`. None of them reviewed the 6.0 publication prose, so the disclosure cross-check in this memo is the first pass over that text.

## Gate-preservation check

| Surface | Banner | PM-1b `assignment_gates_pass=false` | PM-4b human audit pending | Inflation |
|---|---|---|---|---|
| `verified_registry.json` | exact | yes | yes | none found |
| `bundle_registry_spec.json` | exact | yes | yes | none found |
| Whitepaper 6.0 (line 3–4, Ch. 8) | exact; release stated BLOCKED | yes | yes, "sole human gate" | none found |
| Presentation HTML 6.0 | shortened, see F5 | yes (5 mentions) | yes (5 mentions) | none found |
| `external-investor-teaser-non-nda.md` | **absent** | **absent** | **absent** | see F1 |
| `external-investor-teaser-deck-non-nda.html` | **absent** | **absent** | **absent** | see F1 |

## Blocking findings

### F1 — Investor files state no release status and call the audits "third-party"

- **Location:** `external-investor-teaser-non-nda.md` header and §2 intro; `external-investor-teaser-deck-non-nda.html` whole deck.
- **Evidence:** zero matches for `PRE-RELEASE`, `pending`, `human`, `PM-` in either file (also zero at `HEAD`, so nothing was removed; the qualifier was never there). The brief says "Validated 4-Node Hardware Prototype", "Audited Competitive Advantage", "Black-Box Verified Result", "Cleared for External Non-NDA Sharing", and "third-party audit trails available under NDA". The audits on record are LLM model panels run inside the project; the only human audit (PM-4b) is pending and the monograph itself says release is BLOCKED.
- **Consequence:** the two documents intended for outside readers present a stronger verification status than the registry holds. Rulebook L4 (partial as complete) and Rule 2 (visible means verified). The IP-leak script passing does not cover this; it checks method patterns only, and `build_whitepaper_assets.py` has no reference to either investor file.
- **Further evidence of selective framing in the same files:**
  - "`0.00%` observed leakage across 3 live search engines" sits in the `43,040`-query gRPC row. The 43,040-query run (D5) was against Qdrant only; the 3-engine result (D6) is a simulated stale replica on SciFact only.
  - "`0.5B` extractor matches `30B` accuracy `4.6×` faster" omits that the 1,500 ms latency gate was missed (2,104 ms).
  - "`7B` + Chelated matches `70B` … `6.05×` faster reads" is reader-only; the Chelated pipeline's end-to-end time (24.75–26.60 s under the fixed 30B extractor) is slower than bare 70B at 3.26 s, and the parity is a non-rejection on N=60, not a superiority result.
  - "`7B` local models to outperform bare `30B` `99.44%` vs. `38.33%`" uses the retrieval-conditioned D1 k=17 secondary figure.
  - The brief does disclose the Quora −6.92 pt loss and the ~2.1 s / ~10.4 s ingestion times. The deck discloses neither.
- **Correction:** add a visible pre-release line to both files stating that human audit (PM-4b) and one scoped gate (PM-1b) are open; replace "third-party audit trails" with wording that matches what exists (internal multi-model audit logs); fix the four bullets above.
- **Confirming check:** grep both files for the pre-release line and for `third-party` (expect present / absent); re-run `verify_non_nda_ip_clean.py` and expect 0/56.

### F2 — Whitepaper abstract still says 40 bundles

- **Location:** whitepaper line 18: "backed by **40 SHA-256 hash-consistent receipt bundles**".
- **Evidence:** line 4, Appendix A.1 and the registry all say 45/45.
- **Consequence:** an internal contradiction in the first paragraph a reader sees. It understates rather than inflates, but a sealed 6.0 text should not disagree with its own registry, and the `--check` validator did not catch it.
- **Correction:** change to 45. **Check:** `grep -c "40 SHA-256"` returns 0.

### F3 — Presentation HTML omits disclosures the peer audits marked mandatory

Present in HTML: D6 63/70 exact fallback and no-ANN-latency, SciFact-only stale replica, FiQA 0% only; D7 10,373 ms slowest write, 144.6 ms ack scope, Suite B descriptive-only, `p<1e-4`, redacted seal; D8 floor-tier list; D9 latency gate missed, CP95 −0.74 pp, K4 stands, 20k vs 10k; D10 one significant loss, six Recall@10 point losses, NF4, κ=0 ceiling.

Absent from HTML (each is in a peer memo's mandatory list):

| Demo | Missing |
|---|---|
| D6 | nDCG under >0% retraction not recorded |
| D9 | Look #2 non-inferiority gate missed (81.67 vs 85.00); gains are SFT-carried; JSON 100% includes salvage/coercion |
| D10 A | 12 n.s. positive / 2 n.s. negative split of the 24 contrasts (HTML shows only "9/24 wins"); sampled-oracle scopes (0.984 on 50 queries, 0.944 on n=25, 200 retractions) |
| D10 B | Small-1.5B result −5.00 pp with Holm p=0.1929 n.s.; end-to-end latency 24.75–26.60 s vs 3.26 s bare 70B (HTML shows only "6.05× faster reader") |

- **Consequence:** the HTML D10 Part B cell reads as a speed win when the full pipeline is the slow one. That is the single most misleading omission in the three technical surfaces.
- **Correction:** add the missing items to the five Part VII rows. **Check:** grep HTML for `0.1929`, `26.60`, `not recorded`, `85.00`.

### F4 — Secondary scope limits missing from the whitepaper and spec

The whitepaper §4.9 carries every headline negative, null, fallback and latency item. The following peer-mandated scope limits appear in none of whitepaper, HTML or spec (grep count 0):

- **D6:** halfvec 2e-3 parity tolerance; single-host, single-run latency with no replicates; raw-engine 0% stale is zero by construction under the allowlist.
- **D7:** N=60 with one seed, one reader, one judge; the "CP95" figure is a Bonferroni difference of two binomial intervals; the committed manifest carries no implementation hashes.
- **D9:** `eval_virgin_v2_scope = virgin_for_grpo_arm_only`; seed 42 retired; a retune needs a new virgin set; the one-shot rule is procedural, not mechanically enforced; fp32 merge variance exceeds the amendment's stated tolerance.
- **D10 A:** the 1024-d shard is 1,891,457 documents (19.98%), not the whole corpus; the "standalone 8.84M" arm is an admission mask on the shared 9.47M graph evaluated on the same 500 queries; the six Recall@10 point losses were not multiplicity-tested.
- **Also:** whitepaper sentence on D6 reads as if 1.0000 admitted Recall@10 covers FiQA; FiQA has no oracle recall.
- **Spec only:** missing `0.0553` (merged MSMARCO vs RRF does not survive Holm), D10 B end-to-end latency, D7 McNemar values.
- **Consequence:** the task criterion is "all" nuances on all three surfaces. These are scope limits rather than hidden negatives, so taken alone I would rate them required-before-seal but not severe.
- **Correction:** one added sentence per demo in §4.9 and the spec notes. **Check:** grep for `virgin_for_grpo_arm_only`, `1,891,457`, `2e-3`, `not multiplicity`.

## Non-blocking findings

- **F5 — HTML banner shortened.** HTML reads "PRE-RELEASE / HUMAN AUDIT & GATES PENDING"; the registry string includes "CLEAN-RERUN". Meaning is preserved; make it exact.
- **F6 — Holm p shown as a point value at the bootstrap floor.** "Holm p = 0.0012" for the D10 Quora loss and the nine wins is the 20,000-resample floor (5e-5 × 24), not a measured value. Write `≤ 0.0012 (floor)`. D7 already uses `p < 1e-4`, which satisfies the Sonnet 5.5 escalation condition. "Beating Bare-30B / Bare-7B" in D10 B is descriptive and outside the Holm family of two; label it so.
- **F7 — Seal artifacts are uncommitted.** Registry, spec, whitepaper, HTML, both investor files and `build_whitepaper_assets.py` are modified in the working tree at `9e4921a9`, and `wave12-xval-grok47-xhigh.md` is untracked. The sha `c23d9801adb2` describes working-tree content that no commit yet reproduces. Commit before sealing, then re-run `--check` on a fresh checkout.
- **F8 — Sign-off cited before it existed, and the Sol 6.2 BLOCK is not narrated.** Whitepaper line 485 lists `wave12-final-judge-carbon4-med.md` and this memo as the sign-off record. Neither file existed when I started; the carbon4 memo still does not. The same line lists the four peer audits without saying one of them (Sol 6.2) returned BLOCK on D8 for a non-reproducible seal, later fixed and re-sealed at `4b1520cd`. The fix is real per the Grok 4.7 re-audit (85 tests passed) but the history should be stated. After this memo, line 485 must also not describe this judge's outcome as a sign-off.
- **F9 — Smaller items.** `bundle_registry_spec.json` still has `schema_version = bundle_registry_spec_v5_0`. Whitepaper and HTML cite `verify_all_offline.py --fast` as a gate without saying fast mode skips the formal tier (`formal_tier=SKIPPED_USE_FULL_FORMAL`). Peer-reported code defects (stale `stubs_L1` sentence, stale bge-large comment, untested CLI mutual exclusion) remain open as Wave-13 items; I did not re-verify them.

## Challenge round (skeptic lens against my own verdict)

- *"The investor files are marketing, outside the seal."* They are named in the adjudication scope and the brief cites the 45/45 registry and monograph v6.0 as its backing. A seal that covers them must cover their status claims.
- *"F3/F4 are quota-filling."* F3's D10 B latency omission changes what a reader concludes. F4 I have rated lower and said so. I did not list items that are disclosed adequately.
- *"Block is too strong when both gates pass."* The gates test hash integrity and method-pattern leakage. Neither tests whether prose matches evidence, which is what was asked. Passing them is necessary and was confirmed; it is not sufficient.
- *What would change my mind:* evidence that the investor files are explicitly excluded from the 6.0 seal by an owner decision. In that case F1 becomes a separate release item and the verdict on the three technical surfaces would be approve-after-F2–F4.

## Hard-safety compliance

- `D:\GITHUB\CHELATEDAI` not modified.
- No code, test, receipt, manifest, registry or publication file edited in the Stage-0 repo.
- No `*sealed_key*` or `*blind-followup*` file opened; no GPU job run.
- This memo is the only file written.

## Required to clear the block

1. F1: pre-release line and corrected audit wording in both investor files; fix the four framing bullets.
2. F2: abstract bundle count 40 → 45.
3. F3: add the missing D6/D9/D10 items to the HTML Part VII rows.
4. F4: add the scope-limit sentences to whitepaper §4.9 and the spec notes.
5. Re-run both gates, commit, and have a judge re-read the changed text.
